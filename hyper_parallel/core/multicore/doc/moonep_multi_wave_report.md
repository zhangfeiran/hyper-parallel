# MegaMoe MoonEP 风格有界副本与 multi-wave 技术报告

## 1. 文档定位

本文维护 Torch MegaMoe 的有界热专家副本、lossless multi-wave、执行生命周期与正确性约束。
通用 scheduler/任务开发见[Multicore 设计](README.md)，模型运行见
[Qwen benchmark](../../../../examples/torch/multicore/mega_moe/README.md)。

| 项目 | 内容 |
| --- | --- |
| 整理基线 | megamoe-sun，合并提交 ee5a6965；整理前文档快照 5e26dfdd |
| 实现 | 副本默认关闭；串行多层可显式共享 B 池；有界梯度接收与 receive-only 重叠并存 |
| 规模边界 | EP16/32/64 的 CPU 模拟和 meta 构图已实现；实际多机端到端另行验收 |
| 验证入口 | [当前待办与已完成基线](megamoe_optimization_backlog.md) |

历史实验集中在 history/ 的[运行时验证](history/runtime_validation.md)、
[通信实验](history/communication_experiments.md)与[Qwen 验证](history/qwen_validation.md)。
其中性能数字对应旧源码、native 和具体 workload；此次文档整理没有重跑，也不把不同轮次百分比累加。

### 1.1 实现摘要

- 有界 placement 在每个 rank 的 B-slot 预算内分流热专家，剩余 overflow 由 kernel 内 multi-wave 无损处理。
- home-only 使用原 E runtime；真实发生复制时才使用 physical runtime，并维护精确 source-to-physical metadata。
- shared-B 只共享 BF16 副本权重、FP32 副本梯度和受 lease/event 保护的串行执行资源；optimizer 仍属于各层 home。
- owner 接收容量由独立的梯度窗口限制，全部 replica contribution 在 FP32 中累加后才 cast。
- one-wave 且双向 route-fused 的 backward 返回有效 dX 的 owned clone，避免共享 workspace 覆盖输入梯度。
- 当前 runtime task/event 容量按图分配；容量、wire format 和 decoder 计费见
  [descriptor v2](runtime_descriptor_v2.md)。
- CPU planner、copy runs、有界接收的独立测量见[planner 与梯度窗口](hotcopy_planner_and_gradient_window.md)；
  离线预测、HBM ledger 和两类 profiler 的分工见[成本模型](theoretical_cost_model.md)。

## 2. 目标、边界与术语

### 2.1 设计目标

1. 路由过载时不截断 token，也不切到 host MoE fallback。
2. 用显式预算 `B` 限制每个 rank 的热专家副本数量，避免按最坏负载无限扩容。
3. `B` 不足时仍能通过 multi-wave 完成所有 route，正确性不依赖 placement 达到精确均衡。
4. 保持 home expert 的参数、`state_dict` 和 optimizer ownership 不变。
5. 支持完整训练闭环：forward、hidden/router/weight backward、replica gradient 返回 owner、optimizer step。
6. 保留当前方向独立的 route fusion、route-slot、workspace 共享和 retained/deferred backward 语义。

### 2.2 非目标

- 当前实现不是上游 MoonEP GPU 性能的代理，也没有复刻其 CUDA/NVLink zero-copy kernel。
- 当前主分支不包含逐行贴合上游 planner 的 `moonep_native` 后端；该基线单独保存在第 13 节所述备份
  分支。
- 当前 planner 仍是 host reference planner，不是最终的 device-only control plane。
- Qwen 随机权重 benchmark 验证执行和性能边界，不证明模型收敛质量。

### 2.3 符号

| 符号 | 含义 |
| --- | --- |
| `R` | EP rank 数，即 `ep_size` |
| `E` | router 可见的全局 logical expert 数 |
| `H = E / R` | 每个 rank 的 optimizer-owned home expert 数 |
| `B` | 每个 rank 的 replica slot 预算，即 `replica_slots_per_rank` |
| `P = E + R * B` | kernel 使用的全局 physical expert ID 数 |
| `S` | 每 rank 的本地 token 数 |
| `K` | TopK |
| `T = S * K` | 每个 source rank 产生的 route row 数 |
| `C` | 每 wave 的 receive capacity |
| `L[r]` | placement 后目的 rank `r` 的 route row 数 |
| `W` | wave 数，`max(1, ceil(max(L) / C))` |

公开的 `expert_capacity_factor=f` 先解析为：

```text
C = align_up(ceil(T * f), 128)
```

`C` 是每个目的 rank 在一轮中可复用的固定接收窗口，不是 token 丢弃阈值。`L[r] > C` 时，超出的
行进入后续 wave。

## 3. 总体架构

当前实现将控制面、数据面和参数所有权分离：

```text
Qwen router
  │
  ├─ topk_ids / topk_weights
  └─ local logical histogram [E]
          │
          ▼
  async all-gather -> global counts [R, E]
          │
          ├─ max(home load) <= C
          │      └─ home runtime: P=E, no placement, no weight P2P
          │
          └─ max(home load) > C
                 ├─ bounded host planner: at most B replicas/rank
                 ├─ shared-B: prefetch B suffix only; home weights stay as op inputs
                 │     └─ legacy: materialize per-layer [H+B] physical view
                 ├─ logical route -> physical route remap
                 └─ physical runtime: P=E+R*B
                              │
                              ▼
                  exact offsets + W fixed windows
                              │
              fused forward / fused multi-wave backward
                              │
                              ▼
                   physical dW for H+B slots
                              │
                    replica dW P2P to owner
                              │
                              ▼
                  home dW for H optimizer parameters
```

| 层次 | 主要职责 | 核心实现 |
| --- | --- | --- |
| 模型接口 | 固定 shape、home 参数、显式副本/存储开关 | `modules/mega_moe/module.py` |
| 路由控制面 | count gather、快路径选择、placement、physical remap、offset | `platform/torch/mega_moe/route.py` |
| 放置算法 | 有界副本、负载搬移、计数守恒、transfer 描述 | `modules/mega_moe/placement.py` |
| 权重控制面 | shared-B lease/cache、legacy physical pool、P2P、owner reduction | `platform/torch/mega_moe/replica.py` |
| 执行与资源 | runtime DAG、workspace lease、forward/backward bridge | `runtime.py`、`workspace.py`、`execution.py` |
| 设备数据面 | wave replay、窗口求交、任务依赖和 SHMEM 信号 | `ops/runtime/worker_kernel.h` 及两个 worker |

## 4. 路由控制面

### 4.1 计数与全局视图

`prepare_topk_route()` 先得到 rank-local INT32 logical histogram。调用方可以传入可信的
`tokens_per_expert`；否则 backend 使用 `torch.bincount(..., minlength=E)`。Qwen learned-TopK 示例已使用
固定尺寸的 `zeros(E) + scatter_add_` 生成 histogram，避免 `bincount` 的数据依赖输出尺寸，并保留
零负载 expert。

随后用异步 `all_gather_into_tensor` 收集 `[R, E]` count matrix：

- `B=0` 时，轻量 route endpoint 的准备可以与 count gather 重叠，消费 count 前显式等待 handle；
- `B>0` 且发生 overflow 时，host planner 必须读取全局 matrix，因此当前会在 `.tolist()` 前等待并产生
  D2H 同步；
- 共享 workspace 若上一 lease 尚未完成，会先在当前 NPU stream 上排入 reuse event wait，再开始下一轮
  count exchange，避免跨层或连续 forward 覆盖相同 SHMEM 地址。

### 4.2 自适应双 runtime

配置 `B>0` 时 backend 同时构造：

- physical runtime：每 rank `H+B` 个 GMM group，全局 `P=E+R*B` 个 physical ID；
- home runtime：每 rank `H` 个 GMM group，全局 `E` 个 logical/home ID。

全局 count 完成后先计算静态 home destination load。如果 `max(load) <= C`，直接使用 home runtime。
这一分支不会创建 placement 或启动 replica transfer；snapshot 和 lazy compact 也不会分配 physical-weight
pool。默认 eager compact 的 suffix backing 已在该层第一次 forward 开始时准备好，因此仍支付 `B` 个
suffix 的持久 HBM，但不支付 physical descriptor 执行、remap 和 P2P 税。

### 4.3 有界 placement 算法

发生 overflow 时，`build_constrained_expert_placement()` 接收不可变的 `[R, E]` logical counts：

1. 每个 expert 的全部负载先放在其 home rank，计算各 rank 初始负载。
2. 每个 rank 预留恰好 `B` 个 suffix slot；未绑定槽位用 `-1` 表示，保持 GMM group shape 一致。
3. 对 overflow route，目标 load ceiling 取精确平均 `T`。这比只压到 `C=1.5T` 更积极，目标是同时消除
   额外 wave 和单 wave 内的长 GMM 尾部。
4. 迭代选择“高负载 copy -> 低负载 rank”的最佳搬移。排序优先考虑可搬行数、复用已有 copy、source
   surplus、target room，并用 rank/expert ID 保证确定性。
5. 新目的 rank 首次承载该 expert 时消耗一个 replica slot；已有 copy 的再次分流不重复占槽。
6. 达到 ceiling、所有候选 slot 用尽，或不存在可改善的 move 时停止。
7. 按 logical expert 和 source rank 顺序物化精确 source-to-physical count matrix。

算法允许链式 transfer：一个已经承载某 expert 的 rank 可以继续拆分其负载，但实际权重 transfer 始终由
home owner 发往每个 replica。核心不变量是：对任意 source rank 和 logical expert，把所有对应 physical
copy 的 count 相加，必须严格等于输入 histogram。planner 无法完全均衡时也不会丢行；残余 `max(L)`
直接决定 `W`。

相同 `(counts, B, planning ceiling)` 的 immutable plan 使用 128 项 host LRU 缓存，固定路由无需每层
每步重复运行 Python heuristic。候选预筛选保持评分与平局次序，详见
[planner 优化](hotcopy_planner_and_gradient_window.md)。

### 4.4 logical-to-physical remap

placement 产生两类映射：

- `slot_to_logical_expert[R][H+B]`：每个本地 physical slot 当前代表哪个 logical expert；
- `dispatch_counts[R][P]`：每个 source rank 应向每个 physical ID 发送多少行。

设备 remap 先按 logical expert 构造 expert-major route-slot 顺序，再按当前 source 的 physical quota
生成等长 physical-ID 序列，最后 scatter 回原始 TopK 槽位顺序。这样重复 expert ID、同一 token 的多个
TopK 槽位以及 route weight 都保持逐槽语义。

当前 route-slot 用 FP32 保存整数索引，因此 route-fused 或 `B>0` 配置要求 `T <= 2^24`。超过该范围会
显式拒绝，而不是静默产生不可精确表示的索引。

### 4.5 精确 metadata

`_compute_route_metadata()` 从 physical count matrix 构造：

- source-major dispatch offset/size；
- destination-major combine offset/size；
- 每个 local physical expert 的 cumulative `group_list`；
- `expert_capacity`、`retained_capacity`、`receive_capacity` 和 `wave_count`。

单 wave 时，`expert_capacity` 是 `max(L)` 按通信 split 对齐后的精确容量。multi-wave 时滚动容量固定为
`C`，而 `retained_capacity=max(L)`，用于保存跨 wave 的必要前向状态。所有 offset 都描述真实行，不靠
padding 或清空 capacity tail 保证正确性。

## 5. physical weight、shared-B 与 optimizer ownership

### 5.1 固定本地布局

每个 rank 的 kernel 权重布局为：

```text
[ home_0 ... home_(H-1) | replica_0 ... replica_(B-1) ]
```

router 和 scheduler 使用展平的全局 physical ID：`rank * (H+B) + local_slot`。home 参数对象的 shape
仍是 `[H, ...]`，只有 kernel view 是 `[H+B, ...]`；因此 checkpoint、`state_dict` 和 optimizer 继续只
拥有 home rows。

placement callback 在 plan 产生后立刻启动 prefetch。P2P batch 在 physical count H2D、route remap 和
token permutation 之前启动，尽量掩盖 remote weight 搬运。每个 send tensor 和 receive target 都保活到
handle 完成；kernel 读取前调用 `wait()`，异常路径调用 `drain()`，避免提前释放或消费未完成 buffer。

### 5.2 跨层 shared-B split-weight 路径

`shared_physical_weights=True` 是当前串行多层降低副本 HBM 的主路径，并与
`compact_physical_weights` 互斥。相同 resource group 的 layer 共享：

- `[B, ...]` BF16 gate/up 与 down replica weight；
- `[B, ...]` FP32 replica gradient；
- split-weight runtime descriptor 和串行 lease/event。

home Parameter 不复制、不重绑，forward/backward ACLNN schema 仍收到 `[H, ...]` home tensor。扩展的
runtime descriptor 只携带稳定的 B-slot weight/gradient address，kernel 用 `local_experts` 划分 prefix 与
suffix：slot `< H` 从普通算子输入取地址，slot `>= H` 从 shared descriptor 取地址。这样 optimizer、
`state_dict`、saved tensor version 与 storage alias 都只观察原始 home Parameter，也彻底避开跨层共享时
对每个 Parameter 执行 `set_()`。

shared pool 是 exclusive scratch，不允许两个 MoE call 并发读取不同 binding。lease 在当前 stream 的最后
一个 consumer 后记录 event，下一次 claim 先等待 event；forward graph 只保存 manager、immutable plan
和 home Parameter，backward 会重新 claim pool。FP32 suffix dW 先通过 owner-directed P2P 聚合进当前层
FP32 home dW，autograd 最终只返回 `[H, ...]`，因此 shared pool 不会成为 Parameter.grad 或 optimizer
state 的所有者。

初版 shared-B 每次 claim 都重传 replica，包括 checkpoint recompute forward 后立即发生的同层 backward。
当前 cache key 包含完整 slot map、两个 Parameter identity/data pointer/version；所有 transfer handle 完成后
才发布 key。相同 binding 的下一次 claim 跳过 P2P，以下任一变化会强制刷新：

- 切换 layer 或 Parameter 对象；
- optimizer/in-place update 推进 version；
- storage address 变化；
- placement 的任一 physical slot 改绑。

cache hit 前仍在 EP group 上执行 INT32 `MIN` all-reduce。只要一个 rank miss，所有 rank 就一起走 P2P，
避免 sender 和 receiver 选择不同协议。16 层 reentrant checkpoint 的一次 step 有 48 次 shared claim：16 次
初始 forward、16 次 recompute forward、16 次 backward。固定同一路由时，首个 reverse layer 的 recompute
以及每层 recompute 后的 backward 共 17 次可命中，transfer phase 从 48 降到 31；跨 layer 的 pool overwrite
仍然 miss。

当前 shared-B 把该 agreement 放到 resource-group 独立 NPU control stream，并用 `torch.full((), ...)`
直接生成 INT32 flag，避免 host tensor H2D copy。它只读取 Python 侧的 Parameter identity、version、
placement 等元数据，不读取 weight 或 B pool 内容，所以 control stream 不必等待 compute event。
`all_reduce` 的结果仍在 control stream 上通过 `.item()` 等待后才决定 hit/miss；退出上下文立即恢复
调用者的 compute stream。原有 lease event、P2P transfer、handle wait 和 kernel consumer 顺序不变，
控制流提前完成不能使 B pool 提前改写。

这不是跳过 collective 或假设各 rank 的 version 一致：单 rank 参数更新仍会使全组刷新。只为 shared-B
启用独立流，legacy compact/snapshot agreement 仍在调用者流上；两条路径都使用 device-side flag fill。
当前仍只有一个串行 B pool，没有增加 weight/gradient 双 buffer。

### 5.3 compact layout 时机与首次交叠保护

`compact_layout_strategy` 支持两种显式策略：

| 策略 | 行为 | 主要取舍 |
| --- | --- | --- |
| `eager_on_first_forward`（默认） | 每对 layer 参数第一次进入 backend 时、route 和 autograd graph 建立前执行一次 `set_` | 地址在首次使用后稳定；无条件支付 `B` 个 suffix HBM |
| `lazy_on_first_replica` | 直到首次真正需要 replica 才尝试 compact；纯 home workload 不分配 suffix | 保留最低 HBM；首次安全转换可能延后并临时使用 snapshot |

旧 lazy 实现只跟踪已经物化的 physical graph，遗漏了尚未使用 compact 的 home-only graph：

```text
forward A (home, graph live)
  -> forward B (first replica, Parameter.set_)
  -> backward A (saved version mismatch)
```

修复后，compact 尚未准备时的 home forward 会通过 `_HomeWeightLeaseFunction` 返回共享原 storage/version
的轻量 wrapper。manager 只保存 wrapper 的 weak reference，因此 graph 正常 backward 或释放后 lease
自动消失，不延长 activation 生命周期。首次 replica placement 检查到 live home lease 时，不修改 Parameter，
而为该 physical generation 分配隔离 snapshot：

```text
forward A (home lease live)
  -> forward B (snapshot physical pool, no set_)
  -> backward A / B
  -> next replica forward (leases released, compact set_ is now safe)
```

reentrant checkpoint 的第一次 no-grad forward 不会建立 lease；backward recompute 使用当时有效的 home
prefix。non-reentrant checkpoint 若仍保留 home tensor，则由同一 lease 机制阻止中途重绑。

### 5.4 generation-aware legacy pool cache

非 shared-B 的 snapshot/compact 路径中，每对 layer 参数拥有一个 cache entry。共享串行 decoder layer 时，
cache 容量等于共享成员数，所以 8 层可以各自保留一个 pool，同时只共享一套 runtime/workspace。

cache key 包含：

- 全局 physical slot map；
- 两个 home 参数对象 identity；
- storage address；
- PyTorch version counter。

slot binding、参数 storage 和版本均未变化时，pool 直接复用，跳过分配、home copy 和 remote P2P。
optimizer step 推进参数版本后：

- 旧 materialization 已释放：原地刷新同一 pool；
- 旧 graph 仍可能 backward：为新 graph 分配隔离 storage，不覆盖旧图需要的权重；
- compact pool 仍被 live graph 占用且需要不同 placement：新 graph 临时使用 snapshot pool，不再次
  `set_` home 参数。

缓存命中使用 EP group 上的 INT32 `MIN` all-reduce 达成一致决策。即使只有 owner rank 观察到版本变化，
所有 sender/receiver 也会一起进入刷新协议，避免一侧发送、另一侧误判 cache hit。

### 5.5 owner-directed autograd

生产执行中，外层 `_MegaMoeFunction` 的 autograd 输入始终是两个 logical home Parameter。legacy replica
prefetch 的 `PendingExpertWeights` 只作为 route payload 传入，在该 custom Function 的 forward 内等待并
换成 physical kernel view；因此 physical view 与 home Parameter 之间不会建立一个提前把梯度舍入到
BF16 的中间 autograd edge。legacy backward 产生 `[H+B, ...]` 的 FP32 physical `dW1/dW2`；shared-B
backward 则分别写 `[H, ...]` call-local home dW 和 `[B, ...]` shared suffix dW。两者随后都执行：

1. 本地 home prefix 作为 owner gradient 基底；
2. replica rank 以 FP32 把 suffix gradient P2P 发回 logical expert 的 home owner；
3. owner 等待 receive 完成，同一 expert 的所有 contribution 在 FP32 中聚合后再统一舍入；
4. autograd 只向原 Parameter 返回 `[H, ...]` gradient。

`_PhysicalExpertWeightFunction` 仍保留给 replica manager 的直接调用接口，但 backend 生产路径的归并在
外层 MegaMoe backward 完成。因此 replica 不产生新的 optimizer parameter，也不会被 optimizer 独立更新。

owner 接收按全组一致的批次进入有界窗口。receive-only 的 NPU shared-B rank 可在 control stream
提前接收，consumer join 后累加，再让 control stream 等待消费结束才复用窗口；含发送 rank 保留
native dW producer 依赖。具体窗口上限、FP32 存储和异常失效合同见
[梯度接收窗口](hotcopy_planner_and_gradient_window.md)。

## 6. lossless multi-wave 数据面

### 6.1 固定窗口切片

设备 worker 从 tiling 读取 `C` 和 `W`。`W=1` 保持原静态执行；`W>1` 时，同一份 task graph 在一个
kernel launch 内循环执行 `W` 次。

第 `w` 个 wave 的全局接收区间是 `[w*C, (w+1)*C)`。每个 peer segment 和 expert group 都与该区间
求交：

```text
segment_wave = segment_full ∩ [w*C, (w+1)*C)
local_offset = segment_wave.begin - w*C
```

空交集变为 size 0；非空部分修正本地和 peer offset。由此每一行恰好属于一个 wave，所有 wave 复用
相同的 `[C, hidden]` symmetric expert buffer。

### 6.2 forward

forward 仍执行融合链：

```text
Dispatch -> GMM1 -> SwiGLU -> GMM2 -> Combine/weighted unpermute
```

multi-wave 下：

- dispatch、SwiGLU、GMM2 及通信中间量使用容量为 `C` 的 rolling buffer；
- GMM1 输出按 `retained_capacity=max(L)` 保留，因为 backward 需要完整的 up-projection 状态；
- combine 把各 wave 输出写回 route 对应的全局位置；
- terminal weighted-unpermute 和只应执行一次的任务由 wave 条件限制在首轮或末轮。

### 6.3 backward

multi-wave backward 使用专用 DAG，而不是把单 wave backward 在 Python 中 launch 多次。每轮包含：

```text
hidden redispatch / copy
  -> activation recompute
  -> dY dispatch
  -> act_grad (dX branch) || w2_grad
  -> SwiGLU grad
  -> w1_grad || gate_grad
  -> Combine
  -> dW accumulation
```

关键处理包括：

- hidden input 可按 route 再 dispatch，避免为每个 wave长期保存完整 expert-major hidden；
- dW 输出先初始化一次，后续 wave 通过 local accumulate 或可选 atomic GMM 累加；
- dTopK 只执行一次，并复用 ABI 中已经死亡的 route-index slot；需要 retained/deferred backward 时重建
  compact inverse mapping；
- 最终 dX 在所有 wave 完成后才恢复 token 顺序；
- retained dW scratch 被复用后推进 saved tensor version，使非法的第二次 backward 失败关闭。

### 6.4 wave 与 route generation 同步

每个 wave 后，所有活动 Cube/AIV worker 通过独立 `WAVE_BARRIER_EVENT` 完成本地阶段同步，再通过
SHMEM ready generation 协调各 rank 进入下一轮。route-fused backward 中，dY 与 hidden redispatch
可能在同一个 signal arena 中使用两个 phase，因此 arrival threshold 编码为：

```text
generation = wave_index * route_generation_count + route_generation_phase + 1
```

这避免上一 wave 或上一 phase 的旧 signal 被下一轮误消费。`TaskDesc` 的兼容编码为：

- `extra_value_3` 低 8 位：A2A type；中间位：route phase/count；高位：copy/compute ACK 等 flag；
- `extra_value_4` 低位：offset 是否按 token row 计；高位：`top_k`。

新增的 `LOCAL_COPY`、`LOCAL_ACCUMULATE`、`PHASE_BARRIER` 和 route ACK task 都走统一 scheduler/runtime
descriptor，不在 Python 侧逐 wave 调度。

### 6.5 默认与实验同步路径

以下环境变量是内部 A/B 开关，默认均为 `0`，不是稳定公共 API：

| 环境变量 | 作用 | 约束 |
| --- | --- | --- |
| `HP_MEGA_MOE_INPLACE_WAVE_BACKWARD` | dY、hidden、dX 复用 expert buffer | 需要 backward fused permute |
| `HP_MEGA_MOE_TILE_PHASE_ACK` | 以 tile/expert ACK 替代全设备 phase barrier | 需要 in-place |
| `HP_MEGA_MOE_DESTINATION_WAVE_FENCE` | 下一 wave 只等待将被覆盖的 destination | 需要 tile ACK |
| `HP_MEGA_MOE_REUSE_RETAINED_DW_SCRATCH` | 用已死亡的 retained up-proj prefix 放整矩阵 dW scratch | 与 atomic 互斥 |
| `HP_MEGA_MOE_ATOMIC_DW_ACCUMULATION` | 后续 wave 直接 atomic add 到最终 dW | 与 retained scratch 互斥 |
| `HP_MEGA_MOE_KERNEL_CYCLE_PROFILE` | 写入设备侧分类 cycle 计数 | 仅诊断 |

## 7. route fusion 兼容性

forward 和 backward fusion mode 可独立选择 `legacy`、`permute`、`unpermute` 或 `both`。当前默认：

```text
forward:  permute
backward: both
```

可以用 `HYPER_PARALLEL_MEGA_MOE_FUSION_MODE` 统一覆盖，或用
`HYPER_PARALLEL_MEGA_MOE_FORWARD_FUSION_MODE` / `...BACKWARD_FUSION_MODE` 分方向覆盖。

组合实现专门覆盖了以下交叉情形：

- forward fused permute，但 backward dY 使用 legacy endpoint；
- backward fused permute，而 forward 保存的是外部 expert-major input；
- multi-wave W1 hidden redispatch 与 dY dispatch 共享 route signal arena；
- repeated expert ID 和非 2 次幂 `top_k=3` 的逐 route-slot 解码。

route-fused kernel 当前支持 `top_k <= 16`；不支持的 shape 默认回退到 legacy mode。显式强制不支持的
fusion mode 会报错。

## 8. workspace 与 HBM 模型

### 8.1 共享 workspace

`MegaMoeExperts.share_execution_resources()` 必须在第一次 forward 前调用，并要求模块 shape、拓扑、
副本预算和存储模式完全兼容。共享后，串行 layer 复用：

- 一个 symmetric expert buffer：`C * hidden * dtype_size`；
- 一个 routed buffer：`T * hidden * dtype_size`；
- forward/backward event arenas；
- 32 MiB GMM workspace；
- 16 MiB SwiGLU-grad workspace。

共享 workspace 是串行 lease，不支持并发调用。释放时在当前 NPU stream 记录 event；下一 owner 用
`wait_event` 排序，无需每层 host synchronize。symmetric heap 在首次初始化前按所有 active resource group
求和，并向 64 MiB 对齐。

GMM dW scratch 采用 high-watermark：只有需求增大时才同步并扩容，较小的后续调用不会 shrink 掉可能
仍被 deferred backward 使用的 tail。代价是达到过的最大 scratch 会保留到 workspace 释放。

### 8.2 shared-B 的层数无关存储

设一个 BF16 expert 的两组权重合计为 `M_expert`。shared-B 每个 resource group、每个 rank 的 persistent
replica storage 为：

```text
BF16 replica weight = B * M_expert
FP32 replica dW     = 2 * B * M_expert
shared-B total      = 3 * B * M_expert
```

这个公式不乘 layer 数。home 参数、Parameter.grad 和 AdamW state 仍按层存在，但它们是训练本身的状态，
不是 replica storage。shared-B backward 仍需要一个当前层的 FP32 `[H, ...]` home dW；在 owner reduction
完成并交给 autograd 后即可释放，不会为所有层同时保留 FP32 physical dW。

历史 shared-B 样本的 `B=1`、H5120/I1792 下，每 rank 实测 shared weight `52.5 MiB`、shared gradient
`105.0 MiB`，合计 `157.5 MiB`。16 层 per-layer compact 仅 BF16 suffix 理论上就需要 `840 MiB`；shared-B
连同 FP32 suffix dW 后仍只需 `157.5 MiB`，且完全不修改 Parameter storage。

### 8.3 snapshot 与 compact physical weights

设一个 BF16 expert 的两组权重大小为：

```text
M_expert = (hidden * 2*intermediate + intermediate * hidden) * 2 bytes
         = 6 * hidden * intermediate bytes
```

| 模式 | 持久 physical pool | 语义 |
| --- | --- | --- |
| snapshot（默认） | 每个激活 layer 额外 `(H+B) * M_expert` | home prefix 与 Parameter 隔离，graph isolation 最强 |
| compact eager（compact 默认） | 每个首次 forward 的 layer 净增 `B * M_expert` | graph 前重绑，首次使用后地址稳定 |
| compact lazy（显式） | 每个安全激活 replica 的 layer 净增 `B * M_expert` | 未使用 replica 时零 suffix；首次交叠暂用 snapshot |

compact 分配 `[H+B, ...]` backing storage，把 home rows 复制到 prefix，再用 `Parameter.set_()` 保持
Parameter identity 和逻辑 shape。eager 在该层第一次 forward 的 route/autograd 之前执行；lazy 在首次
需要 replica 且没有 live home lease 时执行。optimizer 之后直接更新 prefix，不再每步复制 home rows；
suffix 接收 replica。home Parameter 与 full physical view 共享 storage 和 version counter。

标准约束是 forward 到对应 backward 之间不能原地修改 Parameter。违反时 version counter 会使 backward
报错，而不是静默使用错误权重。默认 eager 消除了正常 route 切换造成的中途 `set_`；lazy 使用 home
lease + transient snapshot 保护首次切换。当前 CPU 回归覆盖先创建 AdamW moment、再 eager 重绑和继续
更新，以及 `state_dict` save/load shape 与 backing 地址。持有 raw storage alias 的调用方仍必须把 alias
创建在 eager 首次 forward 之后；lazy 转换前创建的 raw alias 不会自动跟随新的 storage。

### 8.4 Qwen shape 的 legacy 理论值

legacy Qwen shape 中，`hidden=5120`、`intermediate=1792`，所以一个 BF16 expert 的 W1+W2 是
`52.5 MiB`。`E=48`、`R=8`、`H=6`、8 层、`B=1` 时：

- snapshot pool：`8 * (6+1) * 52.5 MiB = 2.871 GiB`；
- compact 净增：`8 * 1 * 52.5 MiB = 0.410 GiB`。

实际 peak 还包含 runtime descriptor、route arena、activation、allocator 对齐以及 overlapping graph 的
临时保护，因此不会精确等于上式。

精度修复后，replica route 的 physical dW 使用 FP32。单个活跃 layer 的 W1+W2 dW 总量为
`2 * (H+B) * M_expert`，相对旧 BF16 dW 多 `(H+B) * M_expert`；multi-wave retained dW scratch 也按
4-byte element 规划。上述 shape 下单层额外理论量为 `7 * 52.5 MiB = 367.5 MiB`。这是临时 backward
buffer，不改变 compact physical weight 的持久 suffix 公式，但会改变 allocator/HBM peak，必须纳入新基线。

## 9. API 与执行状态

串行多层推荐显式启用 shared-B，并在第一次 forward 前绑定同一 resource group：

```python
layers = [
    MegaMoeExperts(
        local_num_tokens=4096,
        hidden_size=5120,
        intermediate_size=1792,
        num_experts=48,
        top_k=8,
        expert_capacity_factor=1.5,
        replica_slots_per_rank=1,
        shared_physical_weights=True,
        ep_size=8,
        ep_group=ep_group,
    )
    for _ in range(num_layers)
]

MegaMoeExperts.share_execution_resources(layers)
```

`shared_physical_weights=True` 要求 `B > 0`，并排斥 `compact_physical_weights=True`。如果需要独立
per-layer physical view，仍可选择 legacy compact：

```python
experts = MegaMoeExperts(
    # Same shape/topology arguments as above.
    replica_slots_per_rank=1,
    compact_physical_weights=True,
    compact_layout_strategy="eager_on_first_forward",  # default
)
```

状态选择如下：

| 配置/路由 | runtime | physical pool | wave |
| --- | --- | --- | --- |
| `B=0`，home load `<= C` | home | 无 | 1 |
| `B=0`，home load `> C` | home | 无 | `ceil(max(home load)/C)` |
| `B>0`，home load `<= C` | home fast path | shared/snapshot/lazy 无 transfer；eager compact 有 backing | 1 |
| `B>0`，副本后 load `<= C` | physical | shared-B suffix 或 legacy snapshot/compact | 1 |
| `B>0`，副本后仍 `> C` | physical | shared-B suffix 或 legacy snapshot/compact | residual multi-wave |

完整 Qwen 命令由 `examples/torch/multicore/benchmark_qwen_moe_mega_moe.sh` 封装。例如：

```bash
bash examples/torch/multicore/benchmark_qwen_moe_mega_moe.sh \
  --backend mega_moe \
  --routing dirichlet \
  --router-load-alpha 0.1 \
  --router-load-seed 42 \
  --replica-slots-per-rank 1 \
  --compact-physical-weights \
  --compact-layout-strategy eager_on_first_forward \
  --warmup-steps 5 \
  --measured-steps 10 \
  --tag dirichlet_a01_b1_compact
```

需要保留“从未使用 replica 就不增加权重 HBM”的策略时，改为
`--compact-layout-strategy lazy_on_first_replica`。该选择会写入 benchmark JSON，避免把两种 layout
时机的 first-step/HBM 结果混合比较。

## 10. 性能证据入口

| 要判断的问题 | 记录 | 解释边界 |
| --- | --- | --- |
| 重基和 dX ownership 修复是否退化 | [运行时验证](history/runtime_validation.md) | 同权重独立层 ABBA；诊断构建与正式构建分开 |
| P2P 字节、等待、rank readiness 与 receive-only 收益 | [通信实验](history/communication_experiments.md) | 历史单机 EP8；局部等待不相加，合并后的有界窗口尚未重测 ABBA |
| 合包、收尾和前向 peer 轮转是否保留 | [实验取舍](history/communication_experiments.md#早期-shared-b-与前向调度记录) | 否决结论和补丁恢复入口保留 |
| connected Qwen 的精度、计时与 HBM | [Qwen 验证](history/qwen_validation.md) | 首次精度通过；原宽度多步重复性与竞争窗口未完成验收 |
| EP64/E384 如何选择 B 与接收窗口 | [planner 与梯度窗口](hotcopy_planner_and_gradient_window.md) | 区分单机归并微测量、CPU 模拟和实际多机 |
| 不实际运行如何估算时间和 HBM | [理论成本模型](theoretical_cost_model.md) | 公式/校准预测，当前不是 tile FIFO 关键路径或物理显存上界 |

早期 shared-B cache、legacy compact、四态 kernel 矩阵等完整历史表格，可由
[通信实验的固定 Git 入口](history/communication_experiments.md#早期-shared-b-与前向调度记录)恢复。
评估副本收益时同时观察 home/physical wave 数、transfer 数、权重更新频率及 pool hit；
B pool 字节不能直接当成总峰值增量，减少 B 也可能增加残余 wave 和 retained activation。

## 11. 正确性与验证

验收按作用域分别记录，不能把 CPU 模拟、多进程协议检查和真实 fused kernel 比较视为同一种证据：

- planner/remap 验证 count 守恒、确定性、unused slot 与 source-to-physical 映射。
- 通信验证 peer/shape/dtype/bytes 匹配、FP32 owner 聚合、窗口消费后复用和异常失效。
- kernel 验证单 wave、residual multi-wave、延迟 backward、checkpoint、资源共享及跨流生命周期。
- optimizer 验证输出、dX、route/expert gradients、更新后参数与所要求的 optimizer state。
- connected Qwen 的首次精度、后续多步重复性、计时和 HBM 各自验收。

历史重基失败、正式修复与逐阶段 native 身份见[运行时验证](history/runtime_validation.md)；
合并后的实际已完成项及剩余缺口见[当前待办](megamoe_optimization_backlog.md)。
独立层速度不能代替整网收敛；source/native、配置和进程时间窗口必须随新测量一起记录。

## 12. 已落地机制与实验取舍

当前保留 fixed-size INT32 histogram、独立 source/destination route scan、grow-only GMM dW scratch、
versioned shared-B cache 与全 EP agreement、remap 显式长度、planner 候选预筛选、线性 copy runs、
有界梯度窗口和 receive-only 接收解耦。task-cycle/host-scope 成本 profiler 与 P2P/all-rank trace
各有入口和时钟语义，详见[成本模型](theoretical_cost_model.md)及[通信诊断](history/communication_experiments.md)。

整专家梯度合包、独立收尾重叠和两种前向 peer 轮转均未保留；证据及原补丁恢复方式集中在
[通信实验](history/communication_experiments.md)。其他早期候选完整取舍表仍可从整理前快照恢复。
后置方案及重新进入条件仅维护在[当前待办](megamoe_optimization_backlog.md)。

## 13. 与原版 MoonEP + native MoE 基线的关系

为区分“当前生产组合”和“原版算法语义”，另做过一个 example-only 基线：

| 项目 | 内容 |
| --- | --- |
| 本地备份分支 | `backup/moonep-native-baseline` |
| 提交 | `14e4e4bc feat(megamoe): add MoonEP native baseline` |
| 上游参考 | MoonEP `2bd860b4dd083df62b79d5e916fca71ec5742228` |
| 当前主分支状态 | 不包含 `moonep_native` 代码或测试 |

该备份基线逐项复刻上游 Torch reference planner 的两阶段 allocation，训练固定使用 `B=E/R`，并在
`R=1/2/3/4/8` 的 500 组随机 histogram 上与上游 `alloc[expert,destination]` 逐元素一致。数据面使用
Ascend native permute、HCCL all-to-all、GMM/SwiGLU 和 reverse all-to-all，反向也把 replica dW 返回
owner。

它的用途是语义基线，不是当前 MegaMoe fused 数据面的性能基线：

| 路由 | native | `moonep_native` | 结论 |
| --- | ---: | ---: | --- |
| balanced p50 | 581.332 ms | 863.263 ms | 无需副本时 adapter 有明显额外成本 |
| Dirichlet `alpha=0.1` p50 | 1066.091 ms | 1109.414 ms | 2--3 wave 被压到 1 wave，但仍慢 4.06% |
| balanced sampled HBM | 32.289 GiB | 33.410 GiB | +1.122 GiB |
| skew sampled HBM | 37.878 GiB | 35.171 GiB | 热点临时区下降抵消副本，-2.707 GiB |

差距来源包括 host planner、Python/HCCL split、每次 backward 的 prefetch 和 replica dW P2P、缺少
MoonEP 的 symmetric VMM、fused zero-copy dispatch/combine 及 rank-level deduplicated transport。
因此不能拿该分支的 `moonep_native` 数字代表上游 MoonEP，也不能与其他历史 fused MegaMoe 数字
混为同一 backend 对比。

## 14. 已知限制与风险

1. planner 的 `.tolist()` 以及 P2P peer table 仍在 host；overflow step 存在 D2H 和 Python 控制开销。
2. split-weight ABI 已允许 home 与 B suffix 分离，但 replica 仍通过显式 P2P 填充 rank-local pool；尚无
   上游 MoonEP 式全局 `E+B` symmetric VMM 或 rank-level transport deduplication。
3. optimizer 每步更新会使 shared/legacy pool version 失效；下一 step 的初始 forward 仍需刷新 replica。
   当前新增命中主要覆盖 checkpoint recompute 后的同层 backward，以及同版本 micro-batch/梯度累积。
4. shared cache agreement 每次 claim 都有一个 device scalar `MIN` all-reduce 和 `.item()`；当前已经移到
   独立 control stream，避免排在 compute 尾部，但固定 collective/host 决策成本尚未与 count exchange 合并。
5. eager compact 在每层第一次 forward 建图前重绑 Parameter storage；lazy 在首次无 live home lease 的
   replica route 重绑。跨该边界持有的 raw storage alias 不兼容；lazy 的首次交叠和后续 overlapping
   placement 可能临时退回 snapshot，瞬时 HBM 高于 steady state；shared-B 不走该 `set_` 路径。
6. shared workspace/shared-B pool 只支持串行 layer/call；真正并发的 MoE 调用必须使用独立 resource
   group，当前没有双 buffer 跨层预取。
7. 当前 Torch backend 要求 NPU BF16、固定 token shape、`E % R == 0`，route-fused `top_k <= 16`。
8. 独立层与 connected Qwen 已有不同范围的历史测量，但原宽度 Qwen 多步数值重复性和干净计时窗口
   仍待验收。B1 的历史 55-route sweep 不替代 native/B0/B1 全曲线、逐 step 动态 skew、
   residual waves 性能或长时收敛；具体缺口见当前待办。

## 15. 下一步优先级

统一维护在 [MegaMoe 当前待办与验收清单](megamoe_optimization_backlog.md)。
本设计文档和历史记录不再维护重复的阶段待办。

## 16. 代码与测试索引

| 内容 | 路径 |
| --- | --- |
| 公共模块 | `hyper_parallel/core/multicore/modules/mega_moe/module.py` |
| placement contract/heuristic | `hyper_parallel/core/multicore/modules/mega_moe/placement.py` |
| Torch route control plane | `hyper_parallel/core/multicore/modules/mega_moe/route.py` |
| replica pool/autograd | `hyper_parallel/core/multicore/modules/mega_moe/replica.py` |
| runtime/schedule | `hyper_parallel/core/multicore/modules/mega_moe/plan.py` |
| RATR 队列合约回归 | `tests/ut/core/multicore/mega_moe/test_scheduler.py` |
| workspace lifecycle | `hyper_parallel/core/multicore/modules/mega_moe/workspace.py` |
| differentiable execution | `hyper_parallel/core/multicore/modules/mega_moe/function.py` |
| multi-wave backward DAG | `hyper_parallel/core/multicore/modules/mega_moe/backward/graph.py` |
| shared device worker | `hyper_parallel/core/multicore/ops/runtime/worker_kernel.h` |
| Qwen model/optimizer benchmark | `examples/torch/multicore/mega_moe/qwen_moe_model.py`、`qwen_moe_benchmark.py` |
| Qwen 单后端驻留 / CPU reference | `examples/torch/multicore/mega_moe/qwen_moe_isolated.py`、`qwen_moe_snapshots.py` |
| checkpointed multi-layer benchmark | `examples/torch/multicore/benchmark_mega_moe_layers.py` |
| opt-in rank-zero trace / all-rank forward/backward cycles | `examples/torch/multicore/mega_moe_layer_profile.py` |
| opt-in all-rank P2P payload / readiness | `examples/torch/multicore/mega_moe_p2p_profile.py` |
| planner/route/pool UT | `tests/ut/core/multicore/mega_moe/` |
| real fused-kernel ST | `tests/torch/expert_parallel/test_mega_moe.py`、`_test_mega_moe_shared_weights.py` |
