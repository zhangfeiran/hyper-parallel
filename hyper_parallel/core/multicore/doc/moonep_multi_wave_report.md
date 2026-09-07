# MegaMoe MoonEP 风格有界副本与 multi-wave 技术报告

## 1. 文档定位

本文说明 HyperParallel 当前 Torch MegaMoe 路径中，MoonEP 风格热专家副本与 lossless multi-wave
的组合设计、执行流程、内存模型、正确性约束和实测结果。它面向后续开发、评审和性能回归，不替代
[Multicore MoE 通用设计](README.md)或
[Qwen optimizer-step 运行说明](../../../../examples/torch/multicore/README.md)。

| 项目 | 内容 |
| --- | --- |
| 整理日期 | 2026-09-08 |
| 代码分支 | `megamoe-sun`；历史性能数据来自重基前的 `megamoe-perf-sun` |
| 代码基线 | 重基于 `megamoe-minimal-api@c24cfe7a`；以本文同提交代码为准 |
| 主功能提交 | `0157ed63`、`eadec971`、`d51ec884`、`43aa7d6c`、`b255b821`、`40b6963b`、`9f75fd20`、`86f4b433`、`364af596`、`50a4e68c`、`5d2a3e38`、`ce089b10` |
| 状态 | 已实现；multi-wave 保持 lossless；副本默认关闭；串行多层可显式启用 shared-B |

代码是最终事实来源；本文中的性能数字只对应第 10 节写明的硬件、模型、路由和测量方法。

重基说明：下文实测数据及主功能提交号保留原分支的历史身份，不代表重基后的精度或性能验证。
当前实现已迁到 `modules/mega_moe` 的 spec/plan/function/module 结构，保留 minimal API 的
owned-SHMEM 生命周期、`expert_capacity_factor=None` 默认值及按绑定容量分配的 SHMEM workspace；
显式设置有限 capacity factor 后才使用有界接收窗口。标准 EP reference 已替换旧私有 native 实现。
重新比较 latency/HBM 前必须从当前源码重建 native payload，并重新执行精度与性能验证。

2026-09-07 已完成针对 `02082125 → fa1a1859` 的同配置复核，见
[重基精度与性能复核](megamoe_sun_rebase_validation_20260907.md)。shared-B 主比较无明显性能
或新增数值损失，HBM 不变；但发现 Conda 构建探测、测试 reference 适配问题，并确认重基前已存在
的串联 workspace 梯度别名缺陷。不能把历史通过项或本轮局部 A/B 表述为整个新分支验收通过。

后续 P0 已修复 fused backward 的 dX ownership，并通过串联图 / checkpoint / optimizer 验收；
同配置 ABBA 的偏斜 / 平衡 step 差为 +0.068% / +0.173%，偏斜 conservative peak 增加 39.001 MiB。
Conda 构建探测与 reference 适配随后也已修复，常规完整构建及 12 项 NPU 回归通过。
详见第 10.2.8--10.2.9 节及 [修复报告](megamoe_sun_fix_report_20260907.md)。

随后完成当前正式产物的 all-rank P2P 归因：两轮偏斜诊断中，16 层累计梯度回传暴露等待
约 85--192 ms/rank，权重刷新约 34--55 ms/rank。下一项优先研究梯度回传，权重双缓冲仍后置；
这不是已实现的加速或可直接兑现的等待节省，详见第 10.2.10 节及
[P2P 归因报告](megamoe_p2p_attribution_20260907.md)。

2026-09-08 随后验证并否决 B1 梯度整专家合包：虽保持 312 个张量逐位一致，偏斜训练 ABBA
仍回退 1.806%，候选已完整撤回。下一项验证不新增池的梯度通信/收尾重叠，见第 10.2.11 节及
[梯度 P2P 实验报告](megamoe_gradient_p2p_optimization_20260908.md)。

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
- 在上一阶段 8 卡 16 层 checkpointed Qwen MoE、严重固定偏斜、每步 AdamW 的复测中，`B=1 + shared-B`
  相比 `B=0`，F+B 从 `1789.069 ms` 降到两次运行均值 `1247.859 ms`，降低 `30.251%`；full step 从
  `1920.229 ms` 降到 `1379.185 ms`，降低 `28.176%`。conservative peak 只增加 `77.888 MiB`，而
  allocator peak reserved 反而减少 `724 MiB`。
- 上一阶段落地 shared-B 的同版本 cache hit：checkpoint recompute 紧接同层 backward 时复用 B-slot，
  只有 transfer 全部完成后才发布 cache key；层、placement、storage 或参数版本变化都会失效，且 EP
  group 用 `MIN` all-reduce 保证一致决策。两轮 A/B 的 F+B 均值降低 `1.321%`，full step 降低
  `1.182%`，HBM 逐字节不变。
- 在 `50a4e68c` 上继续按 trace 优化 shared-B cache agreement：移到独立 control stream 后，两轮配对
  运行的 F+B median 均值再降低 `1.688%`，含 AdamW 的 step 降低 `1.466%`。allocated/conservative peak
  不变；启用副本后 reserved 增加 `2 MiB`。这些是第 10.2.1 节的后续增量，不应与前一轮噪声混算。
- `5d2a3e38` 的 B1 完整 55-route 扫测通过，覆盖 33 次 home/replica 切换。31 个原始 W2--W4 路由
  均变为 W1；conservative peak 最高 `23.933 GiB`，共享 B 池首次启用后固定为 `157.5 MiB/rank`。
  这不是 native/B0/B1 全曲线对照，也不覆盖 B1 放置后仍需多 wave 的边界；详见第 10.2.2 节。
- `ce089b10` 之后补齐默认关闭的 forward cycle 诊断，默认路径不增加持久 HBM。最后一次 checkpoint
  recomputation 中，偏斜 up/down GMM 计数与平衡接近，而 Cube dependency wait 约为 `4.68x`。
  下一步优先定位 dispatch/combine 就绪与跨 rank 等待；这不是新的端到端加速结论，见第 10.2.5 节。
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

该 cache 阶段的 shared-B 相对 `B=0`：

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

#### 10.2.1 下一阶段：独立 control-stream agreement

本阶段先用可选 `--profile-output` 采集 rank-zero CPU/NPU trace，再修改 `replica.py`；不改 planner
算法、weight/gradient payload、native ABI、kernel 或 AdamW。CPU 标注只在 example 的诊断上下文安装，
正式 timing/HBM 完成后才启用，并在异常路径恢复原函数。最终诊断入口先完成一个 route warmup 与
AdamW update，再捕获一个完整 step，确保包含 update 后的 replica refresh。

探索 trace 给出的关键证据如下。CPU 数字为 inclusive launch/wait scope，device 数字为 kernel duration
之和；不能相加，也不能把 CPU 等待缩短量直接解释为端到端收益。

| 分段，16 层 | 调用数 | 原 agreement | 独立 control stream |
| --- | ---: | ---: | ---: |
| cache agreement，CPU | 48 | 268.761 ms | 23.029 ms |
| physical ID remap，CPU | 32 | 33.894 ms | 30.779 ms |
| replica-gradient return，CPU | 16 | 17.612 ms | 14.746 ms |
| fused forward，NPU kernels | 32 | 478.325 ms | 476.924 ms |
| fused backward，NPU kernels | 16 | 325.198 ms | 332.804 ms |

原 agreement 内 `aten::copy_` 占 `240.951 ms`，而 49 次 device all-reduce（含诊断 barrier）合计
`5.977 ms`。主要问题是 flag 的阻塞 H2D copy 排在 recompute/compute 尾部，不是标量传输量大。
独立流上的 flag fill + collective 只依赖 host cache metadata，使 CPU 可以继续准备和提交 backward；
compute 工作本身没有消失。这里没有证据支持立即增加 FP32 owner receive pool 或 B 双 buffer，暂不扩大
这两类 persistent HBM。

两份探索 trace 为 `shared_phase_trace_baseline_20260907.json` 和
`shared_control_trace_optimized_20260907.json`，均在 `output/qwen_moe_optimizer/`。它们来自首版诊断
入口，紧接未做 optimizer step 的 finite validation，因此有额外一次跨 step warm hit：30 次 weight
refresh，而不是正常 post-update 的 31 次。上表用于定位 stream 等待，不作为更新后 transfer 次数或正式
性能数据；修正后的入口单独采集 post-update smoke trace。

`shared_control_trace_post_update_20260907.json` 已确认 48 次 agreement（CPU scope 共 `23.202 ms`）、
31 次 weight refresh；同一 smoke 的 balanced 摘要为 `home-only / 0 replicas`。其 1 warmup + 1 measured
只用于验证诊断入口，不加入下面的性能 aggregate。

正式性能按 A/B/B/A 顺序运行，A 为已提交的 `50a4e68c`，B 只改变上述 Python control plane。每点仍为
3 warmup + 7 measured，全部参数每步更新。A1/B1 是单点；B2/A2 使用完全相同的三点顺序
`10000:42 -> 0.005:943 -> 0.1:42`，保留 route history 对 allocator 的影响。

| `alpha=0.1, seed=42` | 原 agreement F+B | 独立流 F+B | 原 agreement full step | 独立流 full step |
| --- | ---: | ---: | ---: | ---: |
| 配对 1 | 1253.688 ms | 1233.034 ms | 1384.570 ms | 1364.339 ms |
| 配对 2 | 1247.644 ms | 1226.085 ms | 1377.584 ms | 1357.314 ms |
| 两个进程 median 的均值 | 1250.666 ms | 1229.560 ms | 1381.077 ms | 1360.826 ms |
| 降幅 | — | **21.106 ms / 1.688%** | — | **20.251 ms / 1.466%** |

三点配对中的其他两个点：

| 路由 | 实际路径 | 原 full step | 独立流 full step | 解释 |
| --- | --- | ---: | ---: | --- |
| `10000:42` | home-only，W1 | 804.076 ms | 799.981 ms | 0.509% 差异视为噪声；未创建 B pool/control stream |
| `0.005:943` | home W4，B1 后 W1 | 1419.168 ms | 1395.930 ms | 降低 1.637%，仍只是一个配对 |

这三点未覆盖 B1 placement 后仍 residual multi-wave 的性能边界，也不是完整 55-route 曲线。
原三点 JSON 的 balanced `replica_slot_bindings` 来自旧 ST helper 的 exact-mean proposal；该点实际
`shared_pool_per_rank` 全为 0，backend 走 home bypass。当前 multi-layer adapter 已把 home 单 wave 摘要
修正为 0 active replica、原始 destination loads 和 `home-only`，并补 CPU 回归；不改动历史原始 JSON。

各配对的 stable allocated、peak allocated、conservative peak、SHMEM heap 和 B pool 字节数完全相同。
只有使用 control stream 的点 peak reserved 增加 `2 MiB`；balanced 点连 reserved 也不变。没有增加
FP32 dW、owner receive 或 weight 双 buffer，不能把 reserved 的小增量说成绝对零 HBM 代价。

原始数据：`shared_phase_baseline_20260907.json`、`shared_control_optimized_20260907.json`、
`shared_control_optimized_repeat_20260907.json`、`shared_control_baseline_repeat_20260907.json`。四份
native artifact 和 ST worker hash 相同。A2 使用隔离 editable 环境的 detached worktree；共同的 51 个
Python runtime source 中只有 `replica.py` 与 B 不同。旧 digest 还把主树生成目录下的 232 个 Python
build/prebuild 文件算入，因此主树 file count 为 283、detached 为 51；不能把该总 digest 差异误判成
232 处 runtime 改动。
当前 source digest 已以显式 v2 scope 排除这些 generated build/prebuild 目录，同时继续记录 dirty
runtime source 与实际加载的 native artifacts；对应 CPU 回归验证 clean/built checkout 的一致性。

#### 10.2.2 按优先级推进：完整 B1 路由切换扫测

先冻结 `5d2a3e38` 的 runtime、benchmark 与 native artifacts，完成 16 层、reentrant checkpoint、
每步 AdamW、EP8、每点 3 次 warmup + 7 次 measured 的默认 55-route 扫测。仍是独立 MoE layers，
不是 connected transformer；路由按 `shuffle_seed=20260903` 打乱，每个路由内部固定。

| 执行路径 | 点数 | 原始 home waves | B1 waves | full-step 点中位数范围 | conservative peak 范围 |
| --- | ---: | --- | ---: | ---: | ---: |
| home-only | 24 | 1 | 1 | 810.586--976.876 ms | 23.152503--23.572610 GiB |
| replica-only | 31 | W2: 21，W3: 6，W4: 4 | 1 | 1095.011--1463.370 ms | 23.563092--23.933213 GiB |

序列有 33 次 home-only/replica-only 切换；55 点的前后输出、输入/router/参数梯度、更新后的参数和
AdamW state 有限值检查全部通过。每点 7 个样本中，最慢值相对该点 median 的最大偏离为 `0.763%`；
这是样本内尾部指标，不是跨路由 p99 或长时稳定性证明。

所有 rank 的 B pool 在首次 replica 前为零，此后权重 `52.5 MiB` + FP32 gradient `105 MiB` 不增长。
保留 allocator high-watermark 的整个序列里，`peak_reserved` 最高 `24.957031 GiB`；它与
`peak_allocated + SHMEM heap` 是不同口径，不能相加。该结果支持本次路由切换范围内的稳定池大小，
不证明所有临时 allocation 已消除。

原始数据在 `output/qwen_moe_optimizer/shared_priority_b1_sweep55_20260907.json`；runtime v2 fingerprint
为 `f800e74c4bb73a4c3df10b44a8beb0048df2c1300511bd86a499d7bc6f6ff2e6`（51 个 Python 文件）。
单独本轮没有 native/B0 对照，不能把不同路由或旧运行的差值当成本轮优化收益。

**完成后的优先级复审：** 本轮未发现 B-pool 随路由持续增长或数值有限性失败，先不增加 persistent
buffer。虽然所有 B1 点都是 W1，偏斜路径仍显著慢于平衡路径，因此下一步优先补各 rank 的 kernel
cycle 证据，再决定计算、通信等待与 control-plane 同步的相对优先级。本组数据没有残余 multi-wave，
也没有每一步随机换路由，不能据此取消 multi-wave 或宣称动态边界已全覆盖。

#### 10.2.3 第二步：各 rank 的 backward cycle 诊断

多层 benchmark 新增默认关闭的 `--profile-kernel-cycles`，复用 ST 的 native counter 解码与 EP gather。
它在全部 timing/HBM 采样之后，先完成选中路由的 warm step + AdamW，再临时启用计数执行一个额外
完整 step，最后恢复环境（异常也恢复）。普通 benchmark 拒绝继承
`HP_MEGA_MOE_KERNEL_CYCLE_PROFILE=1`，避免把 instrumented kernel 当成默认路径计时。

本小节采集时的固定 workspace tail 只保留**最后一次 backward launch** 的计数，即本 benchmark 的 layer 0；
不是全部 16 层，不包含 forward，也未把 GMM 进一步拆为 dX/dW。单位是 `system_cycle`，不假设设备
频率换算毫秒。先完成 smoke，第三步复测后又补采了匹配协议的平衡路由：下表均使用显式长度版本、
3 warmup + 7 measured、相同的 post-update trace/cycle 顺序，runtime/helper/adapter/native fingerprint
完全一致，仅路由不同。它们是两个独立诊断 step 的各类别 rank/core maximum，不能相加当 wall time：

| backward 类别 | 平衡 `10000:42` | 偏斜 `0.1:42` |
| --- | ---: | ---: |
| Cube GMM | 730396 | 993265 |
| Cube dependency wait | 65689 | 291649 |
| Vector dependency wait | 357223 | 840560 |
| Vector dispatch dY | 177237 | 307942 |
| Vector combine | 158599 | 255066 |
| Vector synchronization | 38082 | 264651 |

如果改看单个 worker 的顺序类别之和，平衡路由最大 Cube row 为 rank 3/core 15 的 `811689` cycles，
偏斜为 rank 5/core 8 的 `1307167` cycles；该偏斜 row 中 GMM 为 `981295`、dependency wait 为
`290617`。这避免了把不同 core 的最大值拼成一个不存在的关键路径，但仍只是 instrumented backward
中已覆盖代码段的诊断，不等于完整 step。

对应记录为 `shared_priority_balanced_matched_20260907.json` 与
`shared_priority_remap_b1_20260907.json`，均在 `output/qwen_moe_optimizer/`，另有同名 `_trace_` 文件。
匹配的 rank-zero device trace 给出更明确的后续方向：

| device trace 项目 | 平衡 `10000:42` | 偏斜 `0.1:42` |
| --- | ---: | ---: |
| 32 次 forward kernel duration 之和 | 308.830 ms | 479.281 ms |
| 16 次 backward kernel duration 之和 | 256.756 ms | 331.704 ms |
| 32 次 histogram all-gather duration 之和 | 8.764 ms | 74.444 ms |

前向 kernel 的路由间差距约 `170 ms`，大于 backward 的约 `75 ms`，不能把偏斜代价都归到 FP32 dW。
all-gather duration 包含跨 rank 到达时间差，count payload 很小，不能把其 duration 全算成搬运成本。
早期 A1 trace 的 CPU `_finish_static_count_gather` scope 为 `921.375 ms`，还包含等待先前已排队计算，
更不能直接作为可删除的同步开销。

偏斜 trace 另外有 47 次 `batch_send_recvAicpuKernel`（31 次 weight refresh、16 次 gradient return），
以及 692 个 `SDMA_SQE` 事件。后者 duration 总和为 `222.969 ms`，在同一 device 时间轴合并重叠区间
后为 `169.464 ms`；这仍不是去掉 P2P 后可节省的 wall time。profiler 的 `Communication` 汇总只有
`80.578 ms`，恰为 all-gather + all-reduce 之和，不能用它代表包含 AICPU/SDMA P2P 的完整通信成本。

**完成后的优先级复审：** 偏斜路径同时出现更高 GMM 计数和显著依赖等待，因此优先细分 forward
等待、dX/dW 与通信交叠，而不是仅压 transfer 次数或立即增加 B 双 buffer。先以极小改动验证已有
trace 明确定位的 remap sizing 同步，然后将后续主要投入转回设备关键路径；这些单次诊断不构成
kernel 方案的性能收益证明。补采匹配 trace 后，前向流水与 P2P 数据面归因被提升为前两项。

#### 10.2.4 第三步：消除 physical remap 的长度推断同步

`_remap_topk_ids_to_physical` 现在给 `repeat_interleave` 显式传入 `output_size=topk_ids.numel()`，
并在创建 device tensor 前用已有 host quota 的总数检查长度一致性。该检查不读取 device 数据，也不
重新计算 logical histogram；调用者仍须提供与 TopK 一致的 expert counts。没有新增缓存、流、buffer
或 native ABI 变更。

相同 `0.1:42` 路由、16 层、每步 AdamW、3 warmup + 7 measured，按 A/B/B/A 运行。A 为原 runtime，
B 为显式长度版本；每个计时窗口内均关闭 profiler/cycle。A1/B1 在计时结束后另采 trace/cycle，
B2/A2 不做额外诊断。下面聚合的是各次运行的 rank-max median，不是拼接样本后的新 median：

| 运行 | F+B median | full-step median |
| --- | ---: | ---: |
| A1 原实现 | 1229.525 ms | 1360.876 ms |
| B1 显式长度 | 1231.492 ms | 1362.093 ms |
| B2 显式长度 | 1229.615 ms | 1360.653 ms |
| A2 原实现 | 1230.463 ms | 1361.821 ms |
| A 两次均值 | 1229.994 ms | 1361.348 ms |
| B 两次均值 | 1230.554 ms | 1361.373 ms |

full step 差异为 `+0.002%`，F+B 为 `+0.045%`；**没有确认端到端加速或明显退化**。各次 full-step
样本范围重叠，不能把微小正负差异当收益。trace 则确证 remap 内的 32 次 `aten::sum` 与 32 次
`aten::item` 均消失，`repeat_interleave` CPU inclusive scope 从 `4.372 ms` 降为 `1.089 ms`；整个
remap scope 为 `31.106 → 30.594 ms`。这些嵌套 CPU scope 不能相加或直接外推 wall-time 收益。

四次 allocator baseline、peak allocated、peak reserved、heap 和 shared B pool 逐字节相同：
`peak_allocated=24498462720`、`peak_reserved=25866272768`、
`conservative_total_peak=25370877952` bytes。该改动没有新增 HBM 代价。

原始文件为 `output/qwen_moe_optimizer/shared_priority_remap_{a1,b1,b2,a2}_20260907.json`，另有
`a1_trace`/`b1_trace`。A 的 runtime SHA 为第 10.2.2 节的 `f800e74c...`，B 为
`c87f4c94426b59675cd831c3eeec5b5bedb54324e020c910e80fc6fe77017dd9`；native artifacts、route 与
optimizer 参数四次完全一致。第二轮 A 后已恢复 B 代码，不保留运行时实验开关。

**完成后的优先级复审：** 保留消除多余同步且不增加内存的固定形状路径，但不将其列为性能收益。
本轮证据不支持继续围绕这几毫秒展开缓存改造；下一项转向 fused kernel 的前向流水、backward
dX/dW 与跨 rank 等待。新增持久 buffer 或双 buffer 仍须先证明 wall-time 收益和峰值生命周期。

#### 10.2.5 第四步：补齐前向流水计数

先将上一阶段直接本地提交为 `ce089b10`，不使用 autogit、不 push，再按优先级补 forward 诊断。
本项不修改 placement、GMM 数值运算、producer/consumer dependency 或 shared-B pool 协议。

forward worker 复用已有 compile-time profiling 框架，新增 `CYCLE_PROFILE=0/1` template key。默认关闭
时编译掉计数读写；开启后按 worker 记录 descriptor decode、dependency wait、up projection、down
projection、hidden dispatch、combine、local compute 和 synchronization。local compute 包含 SwiGLU
及适用的 epilogue，不是纯 SwiGLU 内核计时；dispatch/combine 也不是无等待的纯链路带宽指标。

前向计数使用独立 `forward_cycle_profile`，每个 shared resource group 只在首次诊断时分配 `4 KiB`，
之后清零复用。48 个 Cube/Vector worker 的 8 类 int64 计数使用其中 `3072 bytes`。backward 仍使用
SwiGLU-grad workspace 原有固定 tail，互不覆盖。关闭前向诊断时传已有 GMM scratch 作为未使用的
输入，不新增 allocation，也不清零诊断区；曾开启后关闭会保留这 `4 KiB`，纯默认执行不分配它。

私有 forward operator ABI 新增 tensor input 27 和末尾 `cycle_profile` attribute，因此源码更新后必须
重建 multicore payload。kernel runtime descriptor 的旧位置保持不变：tiling 为 28，events 为 29，
诊断地址追加到 30。public `MegaMoeExperts` 签名不变。MindSpore C++ bridge 对新输入传已有 scratch、
新 flag 传 0，维持其既有默认接口；本机 MindSpore 2.9 不满足当前 adapter 构建版本要求，这部分只做
参数链路审查，未宣称 MindSpore 编译或运行验证。

benchmark 的 `diagnostic_kernel_cycles.counters` 继续表示 backward，新增 `forward_counters`。
它们分别保留最后一次 launch，即独立 16 层/reentrant 流程中 layer 0 的 forward recomputation 与
backward，不汇总全部 layers。采集仍在**所有** timing/HBM 测量之后，先 warm F+B + AdamW，再开
计数执行一个额外完整 step；环境在异常时也恢复。默认性能与 HBM 对照不能使用这些诊断计数当 wall time。

验证已完成：

- CPU 的 workspace/execution/route/replica/module/example/native-loader 回归共 `167 passed`、
  `122 subtests passed`，覆盖默认零分配、双向隔离、4 KiB 复用与参数/开关一致性。
- Torch/Ascend910B native payload 重建通过；5 项双卡 ST 通过，耗时 `161.75 s`。新增 ST 在开关切换、
  reentrant checkpoint、单/双 wave 下与 native 比较全部输出/梯度，且检查双向计数非零和关闭后不写
  诊断 buffer；另外回归 balanced、hot replica、residual multi-wave 与 shared-B lifecycle。
- 另在全程 `HP_MEGA_MOE_KERNEL_CYCLE_PROFILE=1` 下重复 residual replica multi-wave 和 shared-B
  lifecycle，`2 passed`、`62.39 s`，验证 instrumented forward 与 FP32 replica backward 组合的 native 精度。
- shared-B lifecycle 沿用 `HP_MEGA_MOE_SHARED_REENTRANT_ONLY=1` 的已知验证范围，本项没有修复或
  宣称覆盖其 non-reentrant checkpoint 限制。
- changed Python pylint、ST launcher 无 framework import、AGENTS catalog 与 `git diff --check` 通过。

默认关闭诊断的开销按 A/B/B/A 测量。A 是独立 editable worktree 中的 `ce089b10` 和保存的旧 payload，
B 是新增前向计数后的源码与重建 payload；两边均为 16 层、EP8、每步 AdamW、3 warmup + 7 measured，
每次先测 `10000:42` 再测 `0.1:42`。B1 在全部计时后采 cycle，另外三次不采诊断。

| 运行 | 平衡 F+B | 平衡 full step | 偏斜 F+B | 偏斜 full step |
| --- | ---: | ---: | ---: | ---: |
| A1 | 670.698 ms | 801.962 ms | 1226.766 ms | 1357.810 ms |
| B1 | 674.325 ms | 805.897 ms | 1235.207 ms | 1366.021 ms |
| B2 | 671.292 ms | 802.669 ms | 1238.352 ms | 1369.396 ms |
| A2 | 668.821 ms | 799.818 ms | 1235.397 ms | 1366.635 ms |
| A 两次均值 | 669.760 ms | 800.890 ms | 1231.082 ms | 1362.222 ms |
| B 两次均值 | 672.808 ms | 804.283 ms | 1236.779 ms | 1367.709 ms |

这一组 full-step 均值为 `+0.424%` / `+0.403%`，不能称作加速或已证明零回退。四次各 rank 的
baseline allocated、peak allocated、peak reserved、conservative peak、SHMEM heap 逐字节相同，
路由、B pool 和 optimizer state ledger 也一致；runtime free/used 不在这一逐字节声明内。两条路线的
conservative rank-max 分别为 `24755807232` / `25368836608 bytes`。纯默认路径没有新增持久 buffer。

额外的首次单路线偏斜诊断出现 `1627--3794 ms` 的 full-step 波动；随后观察到其他用户的测试作业，
但没有该异常窗口的连续进程记录，不能断言全部差异的原因。保留该原始文件，不用于下述性能或 cycle
结论。重新采集时每约 10 秒记录 NPU process census；采样点只观察到本轮 8 个 worker。这不是硬件
独占锁，也不能排除采样间隙内的短任务。

两个单路线窗口又分别按 B/A 顺序做了新旧对照，仍为 3 warmup + 7 measured；新版本只在计时之后
采 cycle，旧版本不采。结果单列，不与不同路由历史的 A/B/B/A 混合平均：

| 单路线重采 | A 原实现 F+B / step | B 新实现 F+B / step | full-step 差异 |
| --- | ---: | ---: | ---: |
| 平衡 `10000:42` | 668.840 / 799.753 ms | 670.353 / 801.364 ms | +0.201% |
| 偏斜 `0.1:42` | 1231.256 / 1362.608 ms | 1230.269 / 1361.760 ms | -0.062% |

两组成对 HBM 指标仍逐字节一致；偏斜单路线的 conservative peak 为 `25370877952 bytes`，与先平衡
后偏斜的过程略有不同，不能跨不同 allocator 历史比较“省了多少 HBM”。这些结果没有确认新增端到端
加速，也不足以宣称严格零延迟开销；默认路径的亚百分比差异继续保留为后续约束，不把诊断功能算作
性能优化收益。

下面的 forward 计数来自重采的两个单路线过程：两边 runtime、adapter、native、helper、worker
fingerprint 与 profiling 协议一致，仅路由不同；都在正常 timing 之后先 warm/update 再采一次额外 step。
单位仍为 system cycles，表格每格是各 rank/core 的类别 maximum，**不能相加当 wall time**。

| forward 类别 | 平衡 `10000:42` | 偏斜 `0.1:42` |
| --- | ---: | ---: |
| Cube up projection | 231936 | 228101 |
| Cube down projection | 120301 | 118953 |
| Cube dependency wait | 55562 | 260231 |
| Vector dispatch hidden | 169434 | 297111 |
| Vector combine | 177247 | 256922 |
| Vector dependency wait | 221118 | 538596 |
| Vector local compute | 23501 | 23760 |
| Vector synchronization | 22789 | 37913 |

若只比较实际存在的单个 Cube worker 顺序类别之和，平衡最大 row 是 rank 3/core 6 的 `413253`
cycles，其中 up/down 共 `337024`、dependency wait `54760`；偏斜最大 row 是 rank 5/core 14 的
`623612`，其中 up/down 共 `329149`、dependency wait `260122`。这支持优先检查就绪/依赖等待，
不支持把前向差距主要归因于 GMM 执行段变慢。GMM 段本身也含内部 DMA/流水等待，不是纯算力计数；
它只覆盖最后一次重算，不能外推全部 32 个前向 kernel。

匹配 backward 的 Cube GMM maximum 仍从 `728129` 增至 `997736`，与 forward 的结论不同。因此
backward dX/dW 细分继续保留，但不要混淆前后向瓶颈。本文没有用 instrumentation 的额外 step 声称
端到端速度收益，也没有删掉任何 dependency、扩大 SHMEM 或新增 B 双 buffer。

原始文件都在 `output/qwen_moe_optimizer/`：`forward_cycles_{baseline_a1,candidate_b1,candidate_b2,baseline_a2}_20260907.json`，
以及 `forward_cycles_{balanced_recheck,skew_recheck}_20260907.json`；相邻 `*_monitor_*.log` 保留进程采样。
单路线旧版本对应 `forward_cycles_baseline_balanced_recheck_20260907.json` 与
`forward_cycles_baseline_recheck_20260907.json`。
排除的异常文件为 `forward_cycles_skew_diagnostic_20260907.json`。A runtime SHA 是
`c87f4c94426b59675cd831c3eeec5b5bedb54324e020c910e80fc6fe77017dd9`，B 是
`22da77a42a1c8da5b9f538ad78a6191e5ea1e7255b7c43287f08cdeb3fd6478c`。各自重复运行的 source identity
相同；原有 backward 与 SHMEM 二进制的 SHA256 在 A/B 间也不变。

**完成后的优先级复审：** 前向 GMM 段没有随偏斜明显增大，先定位 dispatch/combine 的 producer
就绪、rank 到达与依赖事件，把前向等待拆到具体 transfer/task，再决定排队顺序或 overlap 改动。
P2P payload 分解紧随其后；FP32 dW 细分仍保留第三位，直接调前向 GMM tiling 和增加双 buffer 暂不优先。

#### 10.2.6 第五步：验证并否决前向 peer 轮转

本轮先将前向诊断直接本地提交为 `fdea4970`，后续每轮验证完成后也直接本地 commit，不使用 autogit、
不 push。本项检查上一节排在首位的通信排队问题，不改 placement 或增加 shared-B buffer。

原 RATR 的队列次序是 `local expert → tile → destination rank`。目标 rank 按本 rank 循环偏移，但
`ProcessStatic` 只有奇数 Vector lane 工作，实际有 `num_workers / 2` 个 worker；910B 是 24 个，
而不是 48 个。每个 worker 按 24 的 stride 取下一任务，因此 EP8 时它长期固定服务同一个 peer。
planner 平衡的是各目标 rank 的接收总量，不能保证每个源 rank 到各 peer 的发送量也均匀。

第一种候选只在每个 tile 的 peer 排列中加入轮转。令 `q = local_expert * tiles_per_expert + tile`、
`W = num_workers / 2`，第 `lane` 个 peer 为 `(rank_id + lane + floor(q * EP / W)) % EP`。
当 `EP` 整除 `W` 时，同一 worker 每个完整取任务 round 换一个目标，避免固定 peer。每个 segment
独立计算偏移；combine 的起始队列位置只改变 worker 编号的排列，不改变这一负载分散性质。

保持不变的协议边界：

- 仅重排前向 dispatch/combine 的 `vector_task_indices`；不改任何 `TaskDesc`、tile index、目标地址、
  route arrival slot、trigger 次数、ready handshake 或 Cube/SwiGLU/epilogue 队列。
- 不跨 local-expert/tile 的优先级边界；空 payload 的 tile 仍发布到达信号，不能删除。
- backward 普通与 multi-wave 的队列均不改，copy/compute ack 与 wave reuse fence 也不变。
- 构造时选择顺序，不依赖运行时路由重建描述符；序列化长度、buffer 数量、通信字节数不变。

CPU 使用 `0.1:42` 已记录的相同逻辑 histogram 和真实 planner 重建 EP8/B1 quotas：各 rank 都接收
32768 tokens，但旧次序最忙 Vector worker 的 dispatch/combine 都承担 3189 tokens；轮转后分别为
1872 / 1664 tokens，总量保持不变。这是任务负载模型，不是 wall-time 加速比例；还没有包含
MTE 争用、远端就绪、信号成本及流水 overlap。平衡 `10000:42` 的最大负载均为 1379 tokens。

第一种候选按 A/B/B/A 测量，A 为原顺序，B 为轮转。四次源码与 native payload 完全相同，只切换
实验用 `HP_MEGA_MOE_FORWARD_PEER_ROTATION=0/1`，每个过程结束时再次校验 source identity。
仍是 EP8、16 独立层、reentrant checkpoint、每步 AdamW、3 warmup + 7 measured；本轮统一先测
偏斜 `0.1:42`，再测平衡 `10000:42`，不与上一节相反路由历史的 HBM 混算。

| 运行 | 偏斜 F+B / step | 平衡 F+B / step |
| --- | ---: | ---: |
| A1 | 1231.460 / 1362.433 ms | 670.406 / 801.879 ms |
| B1 | 1264.568 / 1395.723 ms | 669.964 / 801.100 ms |
| B2 | 1266.980 / 1398.548 ms | 674.330 / 805.534 ms |
| A2 | 1229.503 / 1360.795 ms | 671.894 / 803.149 ms |
| A 两次中位数的均值 | 1230.482 / 1361.614 ms | 671.150 / 802.514 ms |
| B 两次中位数的均值 | 1265.774 / 1397.136 ms | 672.147 / 803.317 ms |

偏斜 full step 回退 `2.609%`、F+B 回退 `2.868%`；平衡 full step 为 `+0.100%`，不称为收益。
因而不采用这一更均匀的静态负载排列。

随后验证第二种更保守的候选：只在切换 local expert 时轮转，即 peer 为
`(rank_id + lane + local_expert) % EP`，保留同一 expert 内原有的 tile/peer 节奏。
CPU 最忙 worker dispatch/combine 为 `3160 / 2595` tokens。重新固定该候选源码做 B/A 对照，
其余配置和路由顺序不变：

| 按 expert 轮转 | 原顺序 A：F+B / step | 候选 B：F+B / step | full-step 差异 |
| --- | ---: | ---: | ---: |
| 偏斜 `0.1:42` | 1233.534 / 1364.572 ms | 1285.700 / 1415.691 ms | +3.746% |
| 平衡 `10000:42` | 671.463 / 802.991 ms | 671.334 / 802.045 ms | -0.118% |

这一组不是重复四次的统计证据，也不把平衡的亚百分比差异称为加速；偏斜回退已足以否决默认启用。

两种候选都在计时结束后才开启 cycle 诊断，采样口径仍是 layer 0 最后一次 forward recomputation。
下表分别取各自配对 A/B 的 rank/core category maximum，不能相加当 wall time：

| 偏斜 forward 类别 | 按 worker round：A → B | 按 expert：A → B |
| --- | ---: | ---: |
| Cube dependency wait | 267926 → 303208 | 260082 → 381221 |
| Vector dispatch hidden | 297701 → 314080 | 297169 → 416470 |
| Vector combine | 269006 → 269694 | 263836 → 266555 |
| Vector dependency wait | 531340 → 455548 | 530986 → 652085 |

尤其第一种候选的 Vector wait 降低，但 Cube wait 和总 step 增加，说明不能把单类等待降低或
worker 字节均衡直接当作流水改善。信号就绪、peer/MTE 争用及跨 worker 的进度相互影响是后续需
分离的因素；现有聚合计数尚不能断言是哪一个 expert/peer 的首次到达导致回退。

六次对应路线、shared pool、optimizer 配置一致；各 rank 的 baseline allocated、peak allocated、
peak reserved、conservative total peak 和 SHMEM heap 均逐字节一致。偏斜/后续平衡的 rank-max
`peak_allocated` 分别为 `24498462720 / 24057401344 bytes`，加 heap 后分别为
`25370877952 / 24929816576 bytes`，heap 均为 `872415232 bytes`。runtime free/used 不在逐字节
声明内。本轮没有增加 HBM。约每 10 秒的 NPU process census 只观察到每次本轮的 8 个 worker；
CPU 上同时有其他用户的 gzip 作业，且没有硬件独占锁，不能宣称整个机器无竞争或排除采样间隙。

验证与保留范围：

- 第一种候选 CPU 为 `173 passed / 174 subtests passed`；balanced、hot replica、residual multi-wave
  与 shared-weight lifecycle 四项双卡 ST 通过，`122.82 s`。
- shared-weight lifecycle 使用 `HP_MEGA_MOE_SHARED_TOKENS=2048`，确保队列真正跨多个 worker round，
  覆盖混合 home/replica 路由、延迟反向、保留图、reentrant checkpoint、跨层复用及权重更新。
  仍使用 `HP_MEGA_MOE_SHARED_REENTRANT_ONLY=1`，不扩展已有 non-reentrant 限制的验证声明。
- 第二种候选的六项调度 UT 与上述扩大 shape 的 lifecycle ST 通过，共 `7 passed / 52 subtests passed`。
- 性能回退后，**两种候选和实验 selector 均撤掉**，不是保留为默认关闭的运行时功能。Torch runtime、
  scheduler、forward descriptor builder 与 `fdea4970` 逐字节一致；native 和 MindSpore 路径未改。
- 按 `add-unit-test` 保留五项针对真实现有 RATR 的回归：EP/worker 组合、非对齐段边界、空 tile 保留、
  正反向两个重排入口一致、非法 segment count，以及三种 forward endpoint 图只改 Vector queue。
  另保留真实 planner quotas 的 payload 守恒与接收均衡不等于 worker 均衡的案例。最终全套 CPU 回归
  为 `172 passed / 151 subtests passed`；不把负收益候选的期望输出固化为运行时要求。
- 撤回后再次运行 2048-token shared-weight lifecycle，`1 passed / 33.43 s`，与 native 比较通过。
  最终 Python pylint、Markdown lint、AGENTS catalog 和 `git diff --check` 通过。

原始数据：`output/qwen_moe_optimizer/peer_rotation_{a1,b1,b2,a2}_20260907.json`，及
`peer_expert_rotation_{b1,a1}_20260907.json`；相邻 `*_monitor_20260907.log` 保存进程采样。
第一种实验 runtime SHA 为 `6b7941f8aa3b52e7607bbceec33bf36e20af740364d1586ac0e278350bf020cc`，
第二种为 `bac8e174592e8393154c4720c68e95001782d8ceec8404cf25a705e0086e8ac1`；每一组成对源码、
helper、worker 和 native fingerprint 相同。撤回后恢复上一节的 `22da77a4...` runtime SHA。

**完成后的优先级复审：** 不继续泛化两种静态轮转，也不直接推广到 backward。下一项把 P2P weight
refresh / gradient return 的实际 payload、等待与 overlap 分解提升为第一位；前向保持第二位，但先
补到具体 expert 的就绪与非空 transfer 完成事件，再考虑调度。FP32 dW 第三位，双 buffer 仍最后。
本轮未取得可保留的端到端加速，当前性能路径保持不变。

#### 10.2.7 重基后的独立复核

`megamoe-perf-sun@02082125` 与 `megamoe-sun@fa1a1859` 做新的 A/B/B/A，固定八个 rank 的
全部初始参数、路由、capacity=1.5、16 独立层、B1、checkpoint 和每步 AdamW。
偏斜 full-step 两次中位数均值为 `1361.275 → 1359.927 ms`（-0.099%），平衡为
`802.988 → 804.218 ms`（+0.153%）。对应各 rank allocator / SHMEM 指标逐字节相同。
本轮未观察到有意义的性能回退；亚百分比差异不作为收益。

四步 optimizer 和路由切换的 312 个张量跨分支逐位一致，但正式 NPU 用例仍有两个失败。
其中 connected shared-workspace 的梯度覆盖在旧分支同样复现，需先处理 dX storage ownership，
再继续性能优化。新 adapter 使用当前 C++ 源码的 Conda-aware 诊断构建，常规 CMake 构建未通过。
完整方法、失败归因、默认 None 的 HBM 差异及原始证据见
[本轮复核报告](megamoe_sun_rebase_validation_20260907.md)。

#### 10.2.8 P0 修复：fused dX 不再别名共享 workspace

`_finish_backward` 在双向 fused 的 one-wave 路径，只对有效 `[T, hidden]` dX 返回 owned clone，
复制位于 workspace release event 之前。其他已有私有输出的路径不增加复制。新增 CPU ownership
回归和真实 NPU 串联训练矩阵，覆盖两份 outstanding graph、home/shared-B 切换、W1/W2、
reentrant checkpoint、每步 AdamW 与 resource close；186 项 CPU 和本轮 10 项 NPU launcher 通过。

从 `7fe7db65` 的修复前基线做新的 matched-weight ABBA，16 独立 Qwen MoE 层、B1、capacity=1.5，
偏斜 full-step 均值 `1360.145 → 1361.070 ms`（+0.068%），平衡 `801.770 → 803.155 ms`（+0.173%）。
偏斜 conservative peak 增加 `39.001 MiB/rank`，平衡不变；reserved 增加 `36 MiB/rank`，
final allocated、SHMEM heap 和 shared-B pool 均不变。这是 correctness 修复的小幅内存成本，
不是零开销，也没有观察到明显的 step 回退。测试边界、原始 hashes 及构建限制见
[修复报告](megamoe_sun_fix_report_20260907.md)。

#### 10.2.9 恢复常规构建和 reference 生命周期验收

multicore / SHMEM 的两个 Torch CMake 改用公开 include discovery，支持 Conda prefix 头文件，
过滤缺失 fallback 并在 compile 前检查必要 header。补齐 wheel/Conda/缺失 header 的真实 CMake
configure UT；CPU 回归为 189 passed / 156 subtests。control-stream 测试同步更新标准 reference
的 `w1/w3`，不再访问旧 packed 参数名，也未放宽容差或删掉 rank-local cache 失效验证。

常规 SHMEM 和 multicore 构建全部通过；六个重新生成的 kernels 与前一阶段逐位一致。正式
adapters 恢复相对 RUNPATH，不再使用诊断构建的绝对路径和额外 `torch_python` 依赖。使用这些
产物的 12 项 NPU 回归全部通过，另测 T=128 的 shared lifecycle 通过，T=2048 和 T=128 均实际
覆盖两种 checkpoint 模式。还补验了四卡 None/1.5 × experts/router 的默认接口矩阵。
正式产物再测同权重 16 层 B1，每步 AdamW：偏斜 / 平衡 full step 为 `1360.044 / 801.564 ms`，
各 rank allocator / SHMEM ledger 与第 10.2.8 节修复版逐字节相同，未观察到额外回退；这是单次
补充运行，不替代 ABBA 或作为新的加速结论。
这不代表整网长期收敛、所有 shape、其他平台或完整 wheel 分发已验证。复现命令及证据见
[修复报告](megamoe_sun_fix_report_20260907.md)。

#### 10.2.10 完成 P2P payload / wait 归因

example 新增默认关闭的 `--profile-p2p`，在所有性能/HBM 测量结束后采一轮额外的已更新权重
F+B。八卡分别记录真实 transfer bytes、CPU builder/launch/wait 和三个 caller-stream 时间戳，
逐 phase-call、peer、shape、dtype 验证发送/接收匹配；不改原通信，不在 F+B 内加全设备同步。
observer 不保留 tensor，异常时恢复原函数，默认路径不创建 observer/events 或额外持久池。

当前正式 native payload 上的两轮 Qwen-shape 16 层 B1、每步 AdamW 偏斜诊断一致确认：31 次
weight refresh 总发送 `11392.5 MiB`，16 次 FP32 gradient return 总发送 `11760 MiB`。
前者累计暴露等待 `33.947--54.704 ms/rank`，后者 `85.038--191.673 ms/rank`；梯度累计 pre-wait
只有 `0.003--0.005 ms/rank`，与立即 wait 结构一致。平衡 home-only 路由八卡均为零 batch/payload。

新采 rank-zero trace 通过 host/device flow 唯一归因 47/47 个 AICPU batch；`SDMA_SQE` 区间并集
为 `169.333 ms`，但无有效 connection ID 可逐条归入 phase。事件包含 enqueue/诊断空隙，pre-wait
不是有效计算重叠；局部等待或 SDMA 并集均不是可预测的 step 收益。本轮只落地诊断和优先级依据，
没有推广新的 runtime 调度。详细 per-rank 结果、计时/HBM 隔离验收、复现命令和局限见
[P2P 归因报告](megamoe_p2p_attribution_20260907.md)。

#### 10.2.11 否决 B1 梯度整专家合包

在 `5a2ae4b0` 上尝试将 gate/up + down 的两个 FP32 梯度存成一块 flat pool 的不重叠 view，
B1 每个 expert 以一个 105 MiB payload 发回 owner。真实 NPU 确认发送 view 零拷贝；owner 仍先
wait 再按原次序 FP32 聚合。207 项 CPU、五项 NPU launcher、八卡 B1/B2 生命周期及跨版本
312/312 张量逐位对照通过，未放宽精度。

隔离 16-batch 传输的 ABBA 降低 3.663%，但同配置完整 16 层 Qwen-shape 训练的偏斜 full step
均值 `1365.100 → 1389.760 ms`，回退 **1.806%**；平衡 `804.293 → 803.246 ms`，基本不变。
梯度字节数相同、描述符减半，但 rank 4/5/7 的暴露等待反而增大。池的逻辑字节不变，实际偏斜
allocated 增加约 1 MiB，后续平衡 reserved 增加 240 MiB，不能用逻辑池账本代替完整 HBM 验收。

候选生产代码及测试改动已撤回，仅保留实验补丁与反证。下一轮先验证独立 backward 收尾与梯度
通信的重叠；若窗口不足，再补高等待 rank 的共同时间线，不继续盲目放大消息或重排 peer。
完整方法、补丁及原始 hashes 见 [实验报告](megamoe_gradient_p2p_optimization_20260908.md)。

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

### 11.1 分阶段验证记录

2026-09-07 重基复核、修复和新诊断证据分别见第 10.2.7--10.2.10 节、
[修复报告](megamoe_sun_fix_report_20260907.md)及 [P2P 归因报告](megamoe_p2p_attribution_20260907.md)。
下列“本轮 / 本阶段”数字保留重基前各优化周期的历史记录，不替代当前分支的验收结果。

执行：

```bash
pytest -q \
  tests/ut/core/multicore/mega_moe \
  tests/ut/core/multicore/test_mega_moe_module.py \
  tests/ut/examples/torch/multicore
```

本轮结果：`155 passed, 117 subtests passed in 4.58s`。新增 CPU 回归覆盖显式 `output_size`、空 source、
unused experts/slots、split source 的 host oracle 对照、错误 quota 总数在 device launch 前失败，以及
cycle 诊断默认隔离、所有测量结束后才采集、非零 rank 参与和异常后恢复环境。

本轮两卡 native-oracle ST：shared weight pool lifecycle、legacy hot-expert replication、replication with
residual multi-wave，合计 `3 passed in 98.41s`。shared lifecycle 使用
`HP_MEGA_MOE_SHARED_REENTRANT_ONLY=1`，不把已有非 reentrant checkpoint 限制算作已验证通过。
本轮仍仅改 Python，未重建 native；第 10.2.4 节 A/B/B/A 及匹配 trace 的 runtime/native fingerprint
已检查，benchmark 结束后没有残留 worker。

上一阶段结果：`149 passed, 105 subtests passed in 11.82s`。回归覆盖 cache key 只在 transfer wait 后发布、
相同 layer/version/placement 命中、layer/版本/placement 失效，以及本 rank 命中但任一远端 rank miss 时
全组重新进入 P2P。本阶段新增独立 agreement stream 复用和异常后上下文恢复、device flag fill/MIN
语义、profiling wrapper 异常恢复、非记录 rank 不创建 profiler、home bypass 摘要，以及排除 generated
build/prebuild 后仍能发现 dirty runtime source 的 digest。

目标文件 `pylint`、`git diff --check` 和本文 `markdownlint-cli2` 均通过。当前环境没有安装 Ruff 命令，
因此不把 Ruff 计为已执行检查。

前一阶段尝试过完整 `pytest -q tests/ut`，但当时 Python 环境缺少 `transformers`、`toml`、`matplotlib` 和
`pulp`，在 16 个无关测试模块的 collection 阶段终止；没有进入测试执行，因此不把它计作代码回归。

本阶段 shared-B + reentrant checkpoint 两卡 fused-kernel ST 结果为 `1 passed in 31.18s`。它使用
native MoE oracle 覆盖跨层 forward、reverse backward、checkpoint recompute、参数 update、pool storage
稳定和 finite output/gradient；新增两个非默认 compute stream 上的 checkpoint replay，并只更新 rank 0
参数，验证远端局部 hit 时仍全组刷新且调用者 stream 不变。legacy hot-expert replication 两卡 ST 也通过
（`1 passed in 33.72s`），覆盖共享 helper 的 device flag fill 不破坏 compact/snapshot 路径。

本阶段仅改 Python，没有重建 native。沿用上一阶段系统 CANN 9.2.0 与 Torch/Ascend910B payload；当时
shared-B ST 为 `1 passed in 103.17s`，multicore vendor、Torch adapter 与 Torch SHMEM 构建均成功。
总 `build.sh` 最后只在无关的 MindSpore
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

前一阶段重新执行 UT、重建并验证 Torch/Ascend910B payload，再完成第 10.2 节五组 8 卡进程：一组
`B=0`、cache-off/on 各两组。所有 finite 检查和 AdamW state 检查通过，logical histogram 与 placement
完全相同；`B=0` 为 2 wave，`B=1` 为 7 个 replica binding 后 1 wave。第 10.4--10.5 节的三组 8 层
legacy compact 结果来自前一轮 `40b6963b` 复测，不能与当前 16 层数字混成同一 aggregate。
前一阶段另外完成第 10.2.1 节四组 A/B/B/A 及一组诊断 smoke。本轮已完成第 10.2.2 节的 B1 完整
55-route 扫测；native/B0/B1 全曲线配对仍未完成。

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
- 本轮按第 10.2.2--10.2.4 节完成 B1 55-route、各 rank backward cycle 入口和 remap 显式长度；同步点
  消失、HBM 不变，但 A/B/B/A 没有确认 step 加速。因此后续优先级转向前向流水与 P2P 设备数据面。

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
4. shared cache agreement 每次 claim 都有一个 device scalar `MIN` all-reduce 和 `.item()`；当前已经移到
   独立 control stream，避免排在 compute 尾部，但固定 collective/host 决策成本尚未与 count exchange 合并。
5. eager compact 在每层第一次 forward 建图前重绑 Parameter storage；lazy 在首次无 live home lease 的
   replica route 重绑。跨该边界持有的 raw storage alias 不兼容；lazy 的首次交叠和后续 overlapping
   placement 可能临时退回 snapshot，瞬时 HBM 高于 steady state；shared-B 不走该 `set_` 路径。
6. shared workspace/shared-B pool 只支持串行 layer/call；真正并发的 MoE 调用必须使用独立 resource
   group，当前没有双 buffer 跨层预取。
7. 当前 Torch backend 要求 NPU BF16、固定 token shape、`E % R == 0`，route-fused `top_k <= 16`。
8. 当前性能数据是随机权重、独立 MoE layers，不是 connected transformer，也不做 native 数值 oracle；
   shared-weight ST 负责 native-equivalent 输出/梯度。B1 已完成 55-route sweep，但 native/B0/B1 全曲线
   配对、逐 step 动态 skew、B1 residual waves 性能和长时收敛仍未覆盖。

## 15. 下一步优先级

**重基复核后的前置项：** 串联 dX / shared-workspace alias 已修复并完成 ABBA，见第 10.2.8 节。
Conda include 探测及新 reference 测试适配随后也已完成，常规完整构建和 12 项 NPU 回归通过，
见第 10.2.9 节。随后完成第 10.2.10 节的 all-rank P2P payload / wait 归因，梯度回传累计暴露等待
明显大于权重刷新，故下一项优先收敛 gradient return。dX 小幅复制暂不优先于它，也不能为省去
该复制重新暴露已修复的 alias。

本轮按“扫测 → backward 诊断 → sizing 小改动 → A/B/B/A → forward 诊断 → 否决两种 peer 轮转”
逐步复审，结果见第 10.2.2--10.2.6 节。当前顺序如下；每项落地后先验证正确性、wall time 和 HBM，
直接本地 commit，再决定下一项。

1. **优先验证 FP32 gradient return 与独立收尾的重叠。** 两轮初始诊断的累计暴露等待为
   `85--192 ms/rank`，权重仅 `34--55 ms/rank`；梯度 pre-wait 几乎为零。整专家合包虽让隔离传输
   更快，完整训练却回退 1.806%，已按第 10.2.11 节否决。下一项先将安全的独立收尾工作放在 wait
   前；若窗口不足，再关联高等待 rank 4/5/7 的共同时间线、peer 到达与批次依赖，不凭 payload
   大小直接轮转或继续放大消息。owner receive contribution 必须全部完成 FP32
   聚合后才能 cast；shared-B dW 发送完成前不能复用 storage，不能只移走 wait 破坏消费/复用依赖。
   默认 HBM 不增加优先；若合批需要额外复制或池，必须单独核算并经匹配 ABBA 才保留。局部事件
   和 rank-zero 约 `169 ms` 的 SDMA 并集均不是预计可节省的 wall time。
2. **前向调度先补 expert 就绪证据，不继续盲目轮转。** 两种 peer 重排虽保持通信字节数与 HBM，
   偏斜 full step 分别回退 `2.609%` / `3.746%`，均已撤回。接收 rank 或 worker 的 payload 均衡不能
   替代 GMM readiness 和跨 rank 进度；下一步关联具体 expert 的 dependency/event、非空 transfer 完成
   和 rank 到达时间，分离 sender 排队、MTE/peer 争用、reader readiness 与等待远端进入 kernel。
   先有上述证据再改前向队列，不直接推广失败策略到 backward，也不优先调 GMM tiling。
   不删除真实 dependency；后续 schedule 改动仍须同时验证单 wave、residual multi-wave、匹配且记录
   竞争情况的 wall time，以及默认 HBM。
3. **按内存证据缩短 FP32 dW 与 owner receive 生命周期。** 用 allocator snapshot/trace 验证 `[H, ...]`
   FP32 dW、owner receive tensors 和 retained scratch 的真实峰值；只在确认 churn 后实现 event-ordered
   high-watermark/ring buffer，或在所有 replica contribution FP32 聚合完成后尽早 cast/release。不能提前
   BF16 累加，也不能为省临时 allocation 无条件增加更大的 persistent pool。计算侧先把现有 GMM 计数
   拆成 dX/dW；目前尚不能确定新增 GMM 时间中各自的比例。
4. **有选择地继续压 control plane。** sizing 的 32 次 device `sum/item` 已移除，但 A/B/B/A 无确认的
   step 收益，不继续围绕它增加缓存。剩余 count D2H、metadata H2D 和约 `23 ms/step` 的独立流
   agreement 需分别量化。合并 agreement 与 histogram exchange 必须保留任一 rank miss 后全组一致的
   协议；只移 planner 而保留 Python `P2POp(peer_rank=...)` 也不能消除全部同步。
5. **扩展真实性和边界证据。** 本轮 B1 55-route 已完成，但它们全是 physical W1，且不是逐步换路由。
   后续补 native/B0/B1 全曲线配对、逐 step 动态 skew、`B=0/1/2/4/6`、capacity、active expert、更新频率、
   频繁换绑和连续 generation。重点验证 B1 residual 3+ waves 及 connected model；不得用当前扫测作为
   取消 multi-wave 的依据。
6. **最后评估双 buffer 与预测 prefetch。** 新诊断已显示 31 次 weight refresh 的暴露等待小于梯度回传，
   暂不增加权重双缓冲。只有其剩余等待在后续成为主要限制，且更低内存代价的调度仍不足，再做
   shared-B ping-pong pool 和跨层 backward prefetch；上一 histogram 预测必须带 generation 校验，不能静默
   使用错误 plan。保持“性能收益足以覆盖新增 HBM 与生命周期复杂度”的进入条件。

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
| checkpointed multi-layer benchmark | `examples/torch/multicore/benchmark_mega_moe_layers.py` |
| opt-in rank-zero trace / all-rank forward/backward cycles | `examples/torch/multicore/mega_moe_layer_profile.py` |
| opt-in all-rank P2P payload / readiness | `examples/torch/multicore/mega_moe_p2p_profile.py` |
| planner/route/pool UT | `tests/ut/core/multicore/mega_moe/` |
| real fused-kernel ST | `tests/torch/expert_parallel/test_mega_moe.py`、`_test_mega_moe_shared_weights.py` |
