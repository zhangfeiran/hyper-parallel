# MegaMoe MoonEP 风格有界副本与 multi-wave 技术报告

## 1. 文档定位

本文说明 HyperParallel 当前 Torch MegaMoe 路径中，MoonEP 风格热专家副本与 lossless multi-wave
的组合设计、执行流程、内存模型、正确性约束和实测结果。它面向后续开发、评审和性能回归，不替代
[Multicore MoE 通用设计](README.md)或
[Qwen optimizer-step 运行说明](../../../../examples/torch/multicore/README.md)。

| 项目 | 内容 |
| --- | --- |
| 整理日期 | 2026-09-07 |
| 代码分支 | `megamoe-perf-sun` |
| 代码基线 | `megamoe-perf-sun`；以本文同提交代码为准 |
| 主功能提交 | `0157ed63`、`eadec971`、`d51ec884`、`43aa7d6c`、`b255b821`、`40b6963b`、`9f75fd20`、`86f4b433`、`364af596` |
| 状态 | 已实现；multi-wave 保持 lossless；副本默认关闭；串行多层可显式启用 shared-B |

代码是最终事实来源；本文中的性能数字只对应第 10 节写明的硬件、模型、路由和测量方法。

### 1.1 结论摘要

- 当前组合不是把 MoonEP CUDA 实现直接移植到 Ascend，而是把其“按路由热度动态复制专家”的思想，
  适配成有界的 rank-local physical expert pool，再用 MegaMoe multi-wave 无损承接副本预算不足后的残余
  倾斜。
- 当前串行多层主路径使用 `shared_physical_weights=True`：每个 resource group 只保留一份 B-slot BF16
  replica weight 和一份 B-slot FP32 replica gradient，home Parameter 继续作为普通算子输入。它不再为每层
  建 `[H+B]` physical weight，也不执行 `Parameter.set_()`；persistent replica storage 与层数无关。
- `replica_slots_per_rank=0` 时保持原有 home-only 路径。`B > 0` 也不会强制复制：如果 home placement
  已能在一个 receive window 内完成，会自动选择较小的 home runtime，跳过 placement、物理 ID remap
  和权重 P2P。非 shared-B 的 compact 路径仍默认在每层第一次 forward 前准备 suffix；显式 lazy 策略
  才会同时跳过未使用的 suffix backing。
- 在 8 卡 16 层 checkpointed Qwen MoE、严重固定偏斜、每步 AdamW 的当前复测中，`B=1 + shared-B`
  相比 `B=0`，F+B 从 `1789.069 ms` 降到两次运行均值 `1247.859 ms`，降低 `30.251%`；full step 从
  `1920.229 ms` 降到 `1379.185 ms`，降低 `28.176%`。conservative peak 只增加 `77.888 MiB`，而
  allocator peak reserved 反而减少 `724 MiB`。
- 本轮首先落地 shared-B 的同版本 cache hit：checkpoint recompute 紧接同层 backward 时复用 B-slot，
  只有 transfer 全部完成后才发布 cache key；层、placement、storage 或参数版本变化都会失效，且 EP
  group 用 `MIN` all-reduce 保证一致决策。两轮 A/B 的 F+B 均值降低 `1.321%`，full step 降低
  `1.182%`，HBM 逐字节不变。
- 旧的 8 层 compact 路径复测中，`B=1 + compact` 相比
  `B=0`，optimizer-step rank-max p50 从 `1156.840 ms` 降到 `671.199 ms`，降低 `41.980%`，
  等价吞吐提升 `1.724x`；该组数据用于比较 legacy per-layer storage，不是当前 shared-B 多层主结果。
- FP32 physical dW 下，该 compact 配置相对 `B=0` 的 sampled HBM peak 增加 `3.783 GiB`，不再是旧数据的
  `0.120 GiB`。它仍把 snapshot 的 `5.755 GiB` 增量压到 `3.783 GiB`，回收 `34.25%`；若只看 live
  allocator allocation，则回收 `95.07%`。剩余 sampled HBM 主要体现瞬时 dW/workspace 推高并留在
  allocator reserve 中。shared-B 已进一步消除了这类随层数增长的 replica weight persistence。
- 轻微偏斜、home-only 单 wave 时，`B=1 + compact` 与 `B=0` 的差异落在运行噪声内。因此当前实现的
  收益条件是“副本减少了足够多的 wave 或热点 GMM 尾部”，而不是任何路由都必然加速。
- compact 同时支持默认 `eager_on_first_forward` 和可选 `lazy_on_first_replica`。lazy 的 home graph
  通过 autograd lease 纳入生命周期；首次 replica 与未完成 home forward 交叠时临时使用 snapshot，
  避免 `Parameter.set_()` 触发 saved-tensor version 错误。
- 精度修复后，replica route 的 physical dW 保持 FP32，所有 copy 在 owner 侧聚合后才回到 logical
  Parameter；外层 MegaMoe autograd 同时负责等待 physical weight 和返回 logical gradient，避免中间
  BF16 autograd edge。第 10.2 节是当前 shared-B 主结果；第 10.4--10.5 节保留 legacy compact 对照。

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
每步重复运行 Python heuristic。

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

第 10.2 节的 `B=1`、H5120/I1792 下，每 rank 实测 shared weight `52.5 MiB`、shared gradient
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

第 10.3 节 legacy Qwen shape 中，`hidden=5120`、`intermediate=1792`，所以一个 BF16 expert 的 W1+W2 是
`52.5 MiB`。`E=48`、`R=8`、`H=6`、8 层、`B=1` 时：

- snapshot pool：`8 * (6+1) * 52.5 MiB = 2.871 GiB`；
- compact 净增：`8 * 1 * 52.5 MiB = 0.410 GiB`。

实际 peak 还包含 runtime descriptor、route arena、activation、allocator 对齐以及 overlapping graph 的
临时保护，因此不会精确等于上式。

精度修复后，replica route 的 physical dW 使用 FP32。单个活跃 layer 的 W1+W2 dW 总量为
`2 * (H+B) * M_expert`，相对旧 BF16 dW 多 `(H+B) * M_expert`；multi-wave retained dW scratch 也按
4-byte element 规划。第 10 节 shape 下单层额外理论量为 `7 * 52.5 MiB = 367.5 MiB`。这是临时 backward
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

## 10. 性能结果

### 10.1 当前 shared-B 测量口径

本轮首先固定一个能稳定触发副本的 production-shape 路由，用于判断优化方向，而不是用单点代替完整
55-route curve：

| 项目 | 值 |
| --- | --- |
| 硬件/软件 | 8 张空闲 Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1 <!-- codespell:ignore cann --> |
| 模型范围 | 16 个独立 MoE layer；全部 forward 后 reverse backward；无 attention/router network |
| checkpoint | reentrant；每层 backward 前重算 forward |
| shape | 每 rank 4096 token，H5120，I1792，E48，TopK8，BF16，EP8 |
| 路由 | Dirichlet `alpha=0.1, seed=42`；home 2 wave，`B=1` placement 后 1 wave |
| optimizer | AdamW；每个 sample 更新参数；`foreach=False`、`fused=False` |
| 测量 | 每进程 3 warmup + 7 measured；先取 rank-max，再求进程内 median |
| 顺序 | cache-off、cache-on、cache-on、cache-off；每组 fresh 8-rank process |

cache-off 使用 detached `364af596` worktree；cache-on 使用同一提交加本报告所述 working-tree patch。两者
共用同一份由 `364af596` 源码构建的 Torch/Ascend910B native payload；本轮优化只改 Python replica
control plane，不改 native ABI 或 kernel。四组 route、workspace、optimizer state、shared pool 字节数完全
一致。原始 JSON 位于被 `.gitignore` 排除的
`output/qwen_moe_optimizer/shared_cache_ab_{baseline,optimized}*_20260907.json`。

### 10.2 当前结果：shared-B 整体收益与同版本 cache

| 配置 | F+B median | full-step median |
| --- | ---: | ---: |
| `B=0`，multi-wave-only（1 run） | 1789.069 ms | 1920.229 ms |
| shared-B，cache-off run 1 | 1262.698 ms | 1394.836 ms |
| shared-B，cache-off run 2 | 1266.435 ms | 1396.517 ms |
| shared-B，cache-off 两次均值 | 1264.567 ms | 1395.677 ms |
| shared-B，cache-on run 1 | 1249.571 ms | 1380.989 ms |
| shared-B，cache-on run 2 | 1246.147 ms | 1377.382 ms |
| **shared-B，cache-on 两次均值** | **1247.859 ms** | **1379.185 ms** |

当前 shared-B + cache 相对 `B=0`：

- F+B 降低 `541.210 ms / 30.251%`，等价吞吐 `1.434x`；
- full step 降低 `541.044 ms / 28.176%`，等价吞吐 `1.392x`；
- optimizer 部分约 131 ms，`B=0/B=1` 基本一致，收益来自 MoE execution，不是减少 AdamW 工作。

cache-on 相对 cache-off 的两次运行均值：

- F+B 降低 `16.708 ms / 1.321%`；14 个 sample pooled median 为 `1264.139 -> 1247.539 ms`；
- full step 降低 `16.492 ms / 1.182%`；pooled median 为 `1395.397 -> 1378.900 ms`；
- 四组 measured sample 的 allocator/SHMEM 字段逐字节一致。

内存是 allocator peak 加完整 SHMEM heap 的 conservative 口径，不是 `npu-smi` 高频采样：

| 配置 | stable allocated | peak allocated | peak reserved | conservative peak | shared-B pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=0` | 15.840 GiB | 22.740 GiB | 24.795 GiB | 23.552 GiB | 0 |
| `B=1`, shared-B | 15.707 GiB | 22.816 GiB | 24.088 GiB | 23.628 GiB | 157.5 MiB |
| shared-B 相对 `B=0` | -136.367 MiB | +77.888 MiB | -724 MiB | **+77.888 MiB** | +157.5 MiB |

shared-B 的 weight + gradient pool 虽是 157.5 MiB，但 `B=1` 把 2-wave workspace high-watermark 降低了
314.938 MiB，所以整体 peak 没有按 pool 大小等量增加，reserved 反而下降。结论是：在该 16 层路由上，
旧 per-layer compact 的 GiB 级 HBM 问题已经基本消失；后续不应继续以压 BF16 weight storage 为最高优先级。

同版本 cache 理论上消掉 48 次 shared claim 中的 17 次 transfer phase，但端到端只收益约 1.2%。这说明
weight P2P 已能部分 overlap 且不是主瓶颈；继续做复杂 weight cache/double-buffer 前，应先量化 FP32 dW、
GMM 尾部、replica-gradient return 和 host control plane。

### 10.3 Legacy per-layer compact 测量口径

以下历史对照使用固定 workload：

| 项目 | 值 |
| --- | --- |
| 硬件/软件 | 8 张 Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1 <!-- codespell:ignore cann --> |
| 并行 | TP1 / EP8 |
| 模型 | 8 层 Qwen-compatible MoE 随机权重模型 |
| token shape | batch 1，每 rank sequence/token 4096 |
| hidden / intermediate | 5120 / 1792 |
| logical expert / TopK | E48 / K8 |
| dtype | BF16 |
| capacity | factor 1.5，`T=32768`，`C=49152` |
| optimizer | fused AdamW，每个 sample 都更新权重 |
| 测量 | fresh 8-rank process，5 warmup + 10 measured |
| timed scope | zero-grad、forward/loss、backward、dense grad sync、optimizer step |

每个 step 先跨 rank 取 elapsed maximum，再对 10 个 rank-max sample 求 p50/p90。显存表来自独立的
untimed stable optimizer step；sampled HBM 通过 `total-free` 每 10 ms 采样，单独列出以覆盖可能不计入
PyTorch allocator 的 SHMEM。

主表采集于提交 `40b6963b`，已经包含 `43aa7d6c` 的 FP32 physical dW 精度修复和 `b255b821` 的
graph-safe compact layout。三种配置分别使用 fresh 8-rank process；compact 使用默认
`eager_on_first_forward`。三个 aggregate JSON 的 tag 分别为 `current_fp32dw_20260907_b0`、
`current_fp32dw_20260907_b1_snapshot` 和 `current_fp32dw_20260907_b1_compact_eager`。原始 JSON 被
`.gitignore` 排除，不作为仓库发布物。

### 10.4 Legacy 严重偏斜：B=0 与 compact

固定 Dirichlet `alpha=0.1, seed=42`，三组运行的 logical histogram 完全相同：

| 配置 | 8 层执行状态 | rank-max p50 | rank-max p90 |
| --- | --- | ---: | ---: |
| `B=0` | 7 层 2 wave，1 层 3 wave | 1156.840 ms | 1158.258 ms |
| `B=1`, snapshot | 每层 7 个 replica transfer，全部 1 wave | 671.685 ms | 675.701 ms |
| `B=1`, compact eager | placement 同上，只移除 home snapshot | 671.199 ms | 672.549 ms |

最终 compact 组合相对 `B=0`：

- p50 降低 `41.980%`，等价吞吐为 `1.724x`；
- p90 降低 `41.934%`；
- 相对 snapshot p50 降低 `0.072%`、p90 降低 `0.467%`，差异很小，没有观察到 compact 造成性能回退。

一次 untimed stable step 的 rank-max 分相如下：

| 配置 | forward | backward | dense grad sync | AdamW | zero-grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=0` | 298.148 ms | 796.337 ms | 35.053 ms | 28.089 ms | 0.475 ms |
| `B=1`, snapshot | 218.226 ms | 392.777 ms | 41.264 ms | 28.085 ms | 0.468 ms |
| `B=1`, compact eager | 216.320 ms | 393.528 ms | 40.995 ms | 28.093 ms | 0.453 ms |

AdamW 时间基本不变，主要收益仍来自减少 forward/backward wave，而不是省略 optimizer 工作。与精度修复前
相比，`B=1` 的 backward 增加约 37--39 ms；这与 FP32 physical dW 及 owner 聚合路径变化方向一致，
但由于 CANN 也从 9.1 切到 9.2，不能把差值完全归因于单个提交。

### 10.5 Legacy compact HBM 结果

stable-step rank-max：

| 配置 | allocated | peak allocated | peak reserved | sampled HBM peak |
| --- | ---: | ---: | ---: | ---: |
| `B=0` | 16.661 GiB | 26.149 GiB | 27.338 GiB | 29.176 GiB |
| `B=1`, snapshot | 19.257 GiB | 28.700 GiB | 33.027 GiB | 34.930 GiB |
| `B=1`, compact eager | 16.789 GiB | 26.232 GiB | 31.053 GiB | 32.959 GiB |
| compact 相对 `B=0` | +0.128 GiB | +0.083 GiB | +3.715 GiB | **+3.783 GiB** |
| compact 相对 snapshot | -2.468 GiB | -2.467 GiB | -1.975 GiB | **-1.971 GiB** |

snapshot 相对 `B=0` 的 sampled HBM 增量为 `5.755 GiB`，compact 后为 `3.783 GiB`，即消除 `34.25%`。
这个比例不能代替 live allocation：compact 把 steady allocated 增量从 snapshot 的 `2.596 GiB` 压到
`0.128 GiB`，消除 `95.07%`；peak allocated 也消除 `96.75%`。但 FP32 physical dW/workspace 的瞬时
分配会扩大 allocator cache，stable-step 结束时 peak reserved 仍比 `B=0` 高 `3.715 GiB`，所以
`total-free` 看到的 sampled HBM 不能随 live tensor 一起降到旧水平。若目标是继续消除 HBM 代价，下一步
应直接复用/预分配 FP32 dW scratch 或主动控制 allocator reserve，而不是继续压缩 BF16 weight prefix。

精度修复前的同口径历史值与当前复测对比如下；两轮同时跨越 CANN 9.1 到 9.2，因此只用于整体回归，
不是单提交因果实验：

| 配置 | 历史 p50 | 当前 p50 | 变化 | 历史 sampled HBM | 当前 sampled HBM | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B=0` | 1155.271 ms | 1156.840 ms | +0.136% | 29.173 GiB | 29.176 GiB | +0.003 GiB |
| `B=1`, snapshot | 635.781 ms | 671.685 ms | +5.647% | 31.571 GiB | 34.930 GiB | +3.359 GiB |
| `B=1`, compact | 631.915 ms | 671.199 ms | +6.217% | 29.293 GiB | 32.959 GiB | +3.666 GiB |

### 10.6 历史补充：轻微动态偏斜的成本

learned TopK 使用 `learning_rate=1e-6`，3 warmup + 5 measured：

| 配置 | 行为 | rank-max p50 | 相对 `B=0` |
| --- | --- | ---: | ---: |
| `B=0` | home 1 wave | 490.429 ms | 基线 |
| `B=1`，双 runtime | 0 transfer，自动选 home runtime | 493.027 ms | +0.530% |
| `B=1`, compact lazy | 0 transfer，不分配 pool | 489.330 ms | -0.224% |
| `B=1`, compact + fixed histogram | 与上一行相同 route counts | 487.663 ms | -0.564% |

这些小差异跨独立运行不稳定，应视为噪声；可靠结论是 home fast path 已把 `B>0` 的轻负载税压到不可见。
作为设计对照，曾经“无条件精确均衡”和“虽不复制但仍使用 E56 physical runtime”的中间实现分别为
`562.396 ms` 和 `515.740 ms`，因此双 runtime 是必要优化。

### 10.7 历史补充：独立四态 fused-kernel 矩阵

另一组 8 卡测试使用 hidden 7168、intermediate 2048，只测 fused forward+backward；后三行共享相同
clustered logical route：

| 状态 | physical destination load | wave | median F+B | conservative peak |
| --- | --- | ---: | ---: | ---: |
| home-only（不同的近均衡健康基线） | 近均衡 | 1 | 44.846 ms | 4.504 GiB |
| multi-wave-only, `B=0` | `[131088,131056,0,0,0,0,0,0]` | 5 | 294.279 ms | 5.105 GiB |
| replica + residual, `B=1` | `[65520,65496,21856,21856,21856,21856,21856,21848]` | 2 | 167.021 ms | 5.346 GiB |
| replica-only, `B=2` | 每 rank 32768 | 1 | 76.652 ms | 5.126 GiB |

同路由下，`B=1` 相对 multi-wave-only 降低 `43.2%`，`B=2` 降低 `74.0%`。该矩阵未在每轮间执行
optimizer step，代表相同权重版本、pool 可直接命中的 micro-batch/梯度累积最佳情形；不能替代 10.4 节
的完整 optimizer-step 结果。

### 10.8 如何理解“是否提升”

当前证据支持一个条件化结论：

- 严重 skew 使 home placement 需要 2--5 waves 时，有界副本只要减少 wave，就有显著净收益；
- `B` 足以精确均衡时收益最大，但 `B` 不足也可通过“副本 + residual multi-wave”连续退化；
- home 已是单 wave 时，系统自动绕过副本，性能与 `B=0` 基本持平；
- 不能只用配置了 `B` 就推断加速，必须同时观察 `baseline_wave_count`、placement 后 `wave_count`、
  transfer 数、权重更新频率和 pool hit。

## 11. 正确性与验证

### 11.1 当前工作树重新验证

执行：

```bash
pytest -q \
  tests/ut/core/multicore/mega_moe \
  tests/ut/core/multicore/test_mega_moe_module.py \
  tests/ut/examples/torch/multicore
```

结果：`141 passed, 105 subtests passed in 11.73s`。新增回归覆盖 cache key 只在 transfer wait 后发布、
相同 layer/version/placement 命中、layer/版本/placement 失效，以及本 rank 命中但任一远端 rank miss 时
全组重新进入 P2P。benchmark UT 同时验证 Python implementation digest 能发现 dirty-tree 变化。

目标文件 `pylint`、`git diff --check` 和本文 `markdownlint-cli2` 均通过。当前环境没有安装 Ruff 命令，
因此不把 Ruff 计为已执行检查。

完整 `pytest -q tests/ut` 也已尝试，但当前 Python 环境缺少 `transformers`、`toml`、`matplotlib` 和
`pulp`，在 16 个无关测试模块的 collection 阶段终止；没有进入测试执行，因此不把它计作代码回归。

当前 shared-B + reentrant checkpoint 两卡 fused-kernel ST 在系统 CANN 9.2.0 与新构建的
Torch/Ascend910B payload 上结果为 `1 passed in 103.17s`。它使用 native MoE oracle 覆盖跨层 forward、
reverse backward、checkpoint recompute、optimizer update、pool storage 稳定和 finite output/gradient。
构建中的 multicore vendor、Torch adapter 与 Torch SHMEM 均成功；总 `build.sh` 最后只在无关的 MindSpore
custom-ops 阶段失败，因为本机 MindSpore 2.9 不支持 CANN 9.2 headers，不影响上述 Torch payload/ST。

此前两卡 fused-kernel eager/lazy 切换 ST 结果为 `2 passed in 126.20s`。两条用例分别验证 eager 首次
forward 前重绑及后续地址稳定，以及 lazy 首次交叠 snapshot 和 graph 释放后 compact。第一次尝试曾在
`symm.initialize()` 报告
`undefined symbol: aclrtGetFuncBySymbol`；检查发现继承 shell 的 `LD_LIBRARY_PATH` 把用户目录中的
CANN 9.0 放在系统 9.2 之前。重新 source `/usr/local/Ascend/cann/set_env.sh` 后，同一 native payload 和
同一用例通过，因此该报错属于 runtime 搜索路径污染，不是 layout 逻辑或 native 构建失败。

### 11.2 实施周期内已经完成的验证

- placement/remap conservation fuzz：覆盖 `R=1/2/4/8`、多种 home/replica budget。
- 两进程 Gloo：remote weight prefetch 与 replica gradient owner accumulation。
- 两卡 Ascend HCCL BF16：physical remap、metadata、prefetch overlap、owner gradient return。
- 两卡真实 fused kernel：capacity overflow、checkpoint、`retain_graph`、deferred backward、one/multi-wave
  混合重放、`B=1` residual multi-wave、`top_k=3`。
- 两卡真实 fused kernel：in-place、tile ACK、destination fence、retained dW scratch、atomic dW 及合法组合。
- 8 卡 Ascend 910B3：E48/K8 近均衡、四态 placement 性能矩阵、Qwen 每步 optimizer。
- CANN 9.1 Ascend 910B Torch multicore + SHMEM 严格 native build。<!-- codespell:ignore cann -->
- compact storage：eager/lazy 选择、home/physical alias、Parameter shape、shared version、已有 AdamW
  moment 后更新、`state_dict` round trip、activation checkpoint、中途非法修改失败关闭和 overlapping
  graph 临时 snapshot。
- eager/lazy 首次切换：CPU 复现覆盖 `home forward A -> replica forward B -> backward A/B`；两卡
  fused-kernel ST 同时覆盖 eager 地址稳定，以及 lazy snapshot 过渡和 graph 释放后的 compact 激活。
- shared-B：两卡 native oracle 覆盖多层、reentrant checkpoint、retained/deferred graph、参数更新和
  B-slot weight/gradient storage；当前 cache 增量继续复用这套真实 kernel 验证。
- peer-buffer reuse fence：CPU source harness 覆盖首个 Combine tile 选举 publisher；双卡小 shape 验证
  publisher 不依赖固定 worker 获得 Combine 任务，避免所有实际 worker 等待未发布 event。

本轮重新执行 11.1 节 UT、重建并验证 Torch/Ascend910B payload，再完成第 10.2 节五组 8 卡进程：一组
`B=0`、cache-off/on 各两组。所有 finite 检查和 AdamW state 检查通过，logical histogram 与 placement
完全相同；`B=0` 为 2 wave，`B=1` 为 7 个 replica binding 后 1 wave。第 10.4--10.5 节的三组 8 层
legacy compact 结果来自前一轮 `40b6963b` 复测，不能与当前 16 层数字混成同一 aggregate。

## 12. 已落地优化与被否决方案

### 12.1 从 profiler-driven 分支选择性吸收

- `0157ed63` 已以 replica-compatible 方式吸收 kernel-internal multi-wave、固定 workspace 和 scheduler
  主体；没有整段 cherry-pick 覆盖当前 physical routing 语义。
- `d51ec884` 吸收 fixed-size INT32 histogram，并保留零负载 expert。
- `d51ec884` 吸收 GMM dW scratch high-watermark，消除 W2 大需求后 W1 小需求导致的 shrink/reallocate
  同步和 deferred backward 风险。
- 当前保留独立 source/destination route scan。hidden-dispatch copy、dispatch-service、route polling 和
  大诊断 descriptor 等无稳定收益实验没有进入默认路径。

### 12.2 当前 shared-B 优化周期

- `9f75fd20` 把 B-slot BF16 weights 与 FP32 gradients 提升到 resource-group 级共享，并以 split-weight
  descriptor 让 native kernel 同时读取 layer-local home inputs 与 shared suffix pointers。
- `86f4b433`、`364af596` 加入 16 层 reentrant-checkpoint + AdamW 的 route-matched benchmark，避免用
  单层同版本 pool hit 外推完整训练。
- 本轮为 shared lease 增加 versioned binding cache，只在 transfer 完成后发布，并保留 EP-wide
  cache/P2P agreement。第 10.2 节 A/B 证明性能小幅但稳定提升，HBM 不变。
- benchmark provenance 现在额外散列 Torch MegaMoe Python implementation tree，能区分相同 Git HEAD
  上的 dirty-tree runtime patch；原来的 worker/adapter/native-artifact fingerprint 保留。

### 12.3 实测后回退

| 候选 | 结果 | 决策 |
| --- | --- | --- |
| 只搬到 `1.5T` capacity，不追求均值 | p50 678.617 ms，仍有长 GMM 尾部 | overflow 后 target 改为 `T` |
| 合并 route permutation | p50 721.499 ms | 回退，保留独立 scan |
| fixed histogram 改显式 shape `ones` | 每 rank reserved +500 MiB，HBM 约 +0.486 GiB | 回退 |
| 标量广播 `index_put_(accumulate=True)` | 1382.974 us，对比 vector scatter 949.365 us | 保留 `scatter_add_` |

learned TopK 在随机权重、重复 batch、`learning_rate=1e-3` 下出现 gate backward 数值异常，因此动态
边界测试使用 `1e-6`。这属于 synthetic workload 稳定性约束，不是 physical remap 的正确性结论。

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
因此不能拿该分支的 `moonep_native` 数字代表上游 MoonEP，也不能与第 10 节当前 fused MegaMoe 数字
混为同一 backend 对比。

## 14. 已知限制与风险

1. planner 的 `.tolist()` 以及 P2P peer table 仍在 host；overflow step 存在 D2H 和 Python 控制开销。
2. split-weight ABI 已允许 home 与 B suffix 分离，但 replica 仍通过显式 P2P 填充 rank-local pool；尚无
   上游 MoonEP 式全局 `E+B` symmetric VMM 或 rank-level transport deduplication。
3. optimizer 每步更新会使 shared/legacy pool version 失效；下一 step 的初始 forward 仍需刷新 replica。
   当前新增命中主要覆盖 checkpoint recompute 后的同层 backward，以及同版本 micro-batch/梯度累积。
4. shared cache agreement 每次 claim 都有一个 device scalar `MIN` all-reduce 和 `.item()`；它保证协议安全，
   但也是 cache hit 的固定控制面税，尚未与已有 count exchange 合并。
5. eager compact 在每层第一次 forward 建图前重绑 Parameter storage；lazy 在首次无 live home lease 的
   replica route 重绑。跨该边界持有的 raw storage alias 不兼容；lazy 的首次交叠和后续 overlapping
   placement 可能临时退回 snapshot，瞬时 HBM 高于 steady state；shared-B 不走该 `set_` 路径。
6. shared workspace/shared-B pool 只支持串行 layer/call；真正并发的 MoE 调用必须使用独立 resource
   group，当前没有双 buffer 跨层预取。
7. 当前 Torch backend 要求 NPU BF16、固定 token shape、`E % R == 0`，route-fused `top_k <= 16`。
8. 当前性能单点是随机权重、独立 MoE layers，不是 connected transformer，也不做 native 数值 oracle；
   已有 shared-weight ST 负责 native-equivalent 输出/梯度，仍需完整 route sweep、动态 skew 与长时收敛。

## 15. 下一步优先级

1. **先补可归因 profile。** 在当前 16 层 shared-B benchmark 中分别记录 count gather、planner、remap、
   cache agreement、weight P2P、forward GMM、backward dW 和 replica-gradient return；同时跑 55-route
   sweep。cache 消掉 35.4% transfer phase 却只改善约 1.2%，不应继续凭 transfer 次数猜主瓶颈。
2. **缩短 FP32 dW 与 owner receive 生命周期。** 用 allocator snapshot/trace 验证 call-local `[H, ...]`
   FP32 dW、owner receive tensors 和 retained scratch 的真实峰值；只在确认 churn 后实现 event-ordered
   high-watermark/ring buffer，或在所有 replica contribution FP32 聚合完成后尽早 cast/release。不能提前
   BF16 累加，也不能为省临时 allocation 无条件增加更大的 persistent pool。
3. **优化 replica-gradient return。** 评估复用 owner receive storage、按 peer 合批以及把通信/FP32
   accumulation 与下一层 backward overlap；当前 weight prefetch 已不是首要瓶颈，gradient return 更接近
   backward critical path。
4. **合并 control-plane collective。** 把 cache agreement/version generation 与现有 histogram exchange
   合并或异步 overlap，再把 placement、physical IDs/counts/group-list 和 transfer table 消费整体下沉到
   NPU。只移动 planner 而保留 Python `P2POp(peer_rank=...)` 不能消除同步。
5. **最后评估双 buffer 与预测 prefetch。** 若分段 profile 证明剩余 31 次 weight refresh 仍显著，再做
   shared-B ping-pong pool和跨层 backward prefetch；上一 histogram 预测必须带 generation 校验，不能静默
   使用错误 plan。
6. **扩展边界。** 覆盖 `B=0/1/2/4/6`、capacity、active expert、更新频率、动态 skew、频繁换绑和连续
   generation。重点确认本轮 `B=1` 的低 HBM 结论在更大 B、3+ residual waves 和 connected model 中成立。

## 16. 代码与测试索引

| 内容 | 路径 |
| --- | --- |
| 公共模块 | `hyper_parallel/core/multicore/modules/mega_moe/module.py` |
| placement contract/heuristic | `hyper_parallel/core/multicore/modules/mega_moe/placement.py` |
| Torch route control plane | `hyper_parallel/core/multicore/platform/torch/mega_moe/route.py` |
| replica pool/autograd | `hyper_parallel/core/multicore/platform/torch/mega_moe/replica.py` |
| runtime/schedule | `hyper_parallel/core/multicore/platform/torch/mega_moe/runtime.py` |
| workspace lifecycle | `hyper_parallel/core/multicore/platform/torch/mega_moe/workspace.py` |
| differentiable execution | `hyper_parallel/core/multicore/platform/torch/mega_moe/execution.py` |
| multi-wave backward DAG | `hyper_parallel/core/multicore/modules/mega_moe/backward/graph.py` |
| shared device worker | `hyper_parallel/core/multicore/ops/runtime/worker_kernel.h` |
| Qwen model/profile | `examples/torch/multicore/qwen_moe_mega_moe_model.py`、`qwen_moe_mega_moe_profile.py` |
| checkpointed multi-layer benchmark | `examples/torch/multicore/benchmark_mega_moe_layers.py` |
| planner/route/pool UT | `tests/ut/core/multicore/mega_moe/` |
| real fused-kernel ST | `tests/torch/expert_parallel/test_mega_moe.py`、`_test_mega_moe_shared_weights.py` |
