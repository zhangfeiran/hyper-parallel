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
| 主功能提交 | `0157ed63`、`eadec971`、`d51ec884`、`43aa7d6c`、本提交（compact layout 与集成修复） |
| 状态 | 已实现；Torch managed MegaMoe 默认关闭副本，compact 默认 eager，multi-wave 保持 lossless |

代码是最终事实来源；本文中的性能数字只对应第 10 节写明的硬件、模型、路由和测量方法。

### 1.1 结论摘要

- 当前组合不是把 MoonEP CUDA 实现直接移植到 Ascend，而是把其“按路由热度动态复制专家”的思想，
  适配成有界的 rank-local physical expert pool，再用 MegaMoe multi-wave 无损承接副本预算不足后的残余
  倾斜。
- `replica_slots_per_rank=0` 时保持原有 home-only 路径。`B > 0` 也不会强制复制：如果 home placement
  已能在一个 receive window 内完成，会自动选择较小的 home runtime，跳过 placement、物理 ID remap
  和权重 P2P。compact 默认在每层第一次 forward 前准备 suffix；显式 lazy 策略才会同时跳过未使用的
  suffix backing。
- 在 8 卡 Qwen E48/K8、严重固定偏斜、每步 fused AdamW 的正式测试中，`B=1 + compact` 相比
  `B=0`，optimizer-step rank-max p50 从 `1155.271 ms` 降到 `631.915 ms`，降低 `45.302%`，
  等价吞吐提升 `1.828x`。
- 同一测试中，compact 模式相对 `B=0` 的 sampled HBM peak 只增加 `0.120 GiB`。相较 snapshot
  模式的额外 HBM，已消除 `94.99%`；性能没有因省显存而回退。
- 轻微偏斜、home-only 单 wave 时，`B=1 + compact` 与 `B=0` 的差异落在运行噪声内。因此当前实现的
  收益条件是“副本减少了足够多的 wave 或热点 GMM 尾部”，而不是任何路由都必然加速。
- compact 同时支持默认 `eager_on_first_forward` 和可选 `lazy_on_first_replica`。lazy 的 home graph
  通过 autograd lease 纳入生命周期；首次 replica 与未完成 home forward 交叠时临时使用 snapshot，
  避免 `Parameter.set_()` 触发 saved-tensor version 错误。
- 精度修复后，replica route 的 physical dW 保持 FP32，所有 copy 在 owner 侧聚合后才回到 logical
  Parameter；外层 MegaMoe autograd 同时负责等待 physical weight 和返回 logical gradient，避免中间
  BF16 autograd edge。第 10 节历史性能/HBM 数据早于该修复，不能当作当前提交的复测结果。

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
                 ├─ start weight prefetch
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
| 权重控制面 | physical pool、P2P prefetch、owner-directed autograd | `platform/torch/mega_moe/replica.py` |
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

## 5. physical weight 与 optimizer ownership

### 5.1 固定本地布局

每个 rank 的 kernel 权重布局为：

```text
[ home_0 ... home_(H-1) | replica_0 ... replica_(B-1) ]
```

router 和 scheduler 使用展平的全局 physical ID：`rank * (H+B) + local_slot`。home 参数对象的 shape
仍是 `[H, ...]`，只有 kernel view 是 `[H+B, ...]`；因此 checkpoint、`state_dict` 和 optimizer 继续只
拥有 home rows。

placement callback 在 plan 产生后立刻调用 `ExpertReplicaManager.start_prefetch()`。P2P batch 在
physical count H2D、route remap 和 token permutation 之前启动，尽量掩盖 remote weight 搬运。每个
send 的 contiguous 临时 tensor 和每个 receive target 都保活到 handle 完成；kernel 读取前调用
`wait()`，异常路径调用 `drain()`，避免提前释放或消费未完成 buffer。

### 5.2 compact layout 时机与首次交叠保护

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

### 5.3 generation-aware pool cache

每对 layer 参数拥有一个 cache entry。共享串行 decoder layer 时，cache 容量等于共享成员数，所以 8 层
可以各自保留一个 pool，同时只共享一套 runtime/workspace。

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

### 5.4 owner-directed autograd

生产执行中，外层 `_MegaMoeFunction` 的 autograd 输入始终是两个 logical home Parameter。replica
prefetch 的 `PendingExpertWeights` 只作为 route payload 传入，在该 custom Function 的 forward 内等待并
换成 physical kernel view；因此 physical view 与 home Parameter 之间不会建立一个提前把梯度舍入到
BF16 的中间 autograd edge。MegaMoe backward 先产生 `[H+B, ...]` 的 FP32 physical `dW1/dW2`，随后：

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

### 8.2 snapshot 与 compact physical weights

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

### 8.3 Qwen shape 的理论值

第 10 节 Qwen shape 中，`hidden=5120`、`intermediate=1792`，所以一个 BF16 expert 的 W1+W2 是
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

模型侧显式启用方式：

```python
experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=48,
    top_k=8,
    expert_capacity_factor=1.5,
    replica_slots_per_rank=1,
    compact_physical_weights=True,
    compact_layout_strategy="eager_on_first_forward",  # default
    ep_size=8,
    ep_group=ep_group,
)
```

串行 decoder layer 构造完成后、第一次 forward 前调用：

```python
MegaMoeExperts.share_execution_resources(layer.experts for layer in layers)
```

状态选择如下：

| 配置/路由 | runtime | physical pool | wave |
| --- | --- | --- | --- |
| `B=0`，home load `<= C` | home | 无 | 1 |
| `B=0`，home load `> C` | home | 无 | `ceil(max(home load)/C)` |
| `B>0`，home load `<= C` | home fast path | snapshot/lazy compact 无；eager compact 有 suffix backing | 1 |
| `B>0`，副本后 load `<= C` | physical | snapshot 或 compact | 1 |
| `B>0`，副本后仍 `> C` | physical | snapshot 或 compact | residual multi-wave |

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

## 10. 历史性能结果（精度修复前）

### 10.1 测量口径

主结果使用以下固定 workload：

| 项目 | 值 |
| --- | --- |
| 硬件/软件 | 8 张 Ascend 910B3，CANN 9.1 <!-- codespell:ignore cann --> |
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

以下所有数字采集于 FP32 replica dW 精度修复 `43aa7d6c` 之前，只用于说明有界副本减少 wave 时的历史
收益和 compact 对持久权重 pool 的影响，不能代表当前提交的端到端性能或 HBM。精度修复增加了 physical
dW/scratch 字节数并改变 owner reduction，当前版本必须用相同命令重新测量后才能更新结论。

这些数据也采集于 layout selector 引入之前，当时 compact 行为等价于现在显式的
`lazy_on_first_replica`。严重偏斜 workload 在第一次 warmup 就激活 replica，故 stable-step 的 compact
backing 与 eager 相同；home-only 的“零 pool”结论只适用于 lazy。默认 eager 的 first-step latency 和
home-only HBM 需要作为新的独立数据组重测，不能从下表外推。

### 10.2 严重偏斜：B=0 与组合实现

固定 Dirichlet `alpha=0.1, seed=42`，三组运行的 logical histogram 完全相同：

| 配置 | 8 层执行状态 | rank-max p50 | rank-max p90 |
| --- | --- | ---: | ---: |
| `B=0` | 7 层 2 wave，1 层 3 wave | 1155.271 ms | 1157.133 ms |
| `B=1`, snapshot | 每层 7 个 replica transfer，全部 1 wave | 635.781 ms | 636.746 ms |
| `B=1`, compact | placement 同上，只移除 home snapshot | 631.915 ms | 633.549 ms |

最终 compact 组合相对 `B=0`：

- p50 降低 `45.302%`，等价吞吐为 `1.828x`；
- p90 降低 `45.248%`；
- 相对 snapshot p50 再降低 `0.608%`，没有观察到以性能换显存。

一次 untimed stable step 的 rank-max 分相如下：

| 配置 | forward | backward | dense grad sync | AdamW | zero-grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B=0` | 295.667 ms | 795.111 ms | 34.956 ms | 28.066 ms | 0.939 ms |
| `B=1`, snapshot | 218.691 ms | 356.307 ms | 39.478 ms | 28.069 ms | 0.541 ms |
| `B=1`, compact | 219.534 ms | 354.658 ms | 38.053 ms | 28.094 ms | 0.620 ms |

AdamW 时间基本不变，主要收益来自减少 forward/backward wave，而不是省略 optimizer 工作。

### 10.3 HBM 结果

stable-step rank-max：

| 配置 | allocated | peak allocated | peak reserved | sampled HBM peak |
| --- | ---: | ---: | ---: | ---: |
| `B=0` | 16.661 GiB | 26.149 GiB | 27.338 GiB | 29.173 GiB |
| `B=1`, snapshot | 19.666 GiB | 28.410 GiB | 29.674 GiB | 31.571 GiB |
| `B=1`, compact | 17.203 GiB | 25.945 GiB | 27.393 GiB | 29.293 GiB |
| compact 相对 `B=0` | +0.541 GiB | -0.204 GiB | +0.055 GiB | **+0.120 GiB** |
| compact 相对 snapshot | -2.463 GiB | -2.465 GiB | -2.281 GiB | **-2.278 GiB** |

snapshot 相对 `B=0` 的 sampled HBM 增量为 `2.399 GiB`，compact 后只剩 `0.120 GiB`，即消除
`94.99%`。compact 的 steady allocated 增量 `0.541 GiB` 接近理论 suffix `0.410 GiB` 加 runtime/
对齐成本；peak allocated 反而略低，是因为 `B=1` 把多 wave activation 和 scratch 峰值压成单 wave。

加入 overlapping-placement snapshot 保护后的补充运行（3 warmup + 5 measured）为 p50 `635.264 ms`、
sampled HBM peak `29.270 GiB`，说明保护 retained graph 的分支没有破坏逐步 optimizer 的低显存路径。

### 10.4 轻微动态偏斜的成本

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

### 10.5 独立四态 fused-kernel 矩阵

另一组 8 卡测试使用 hidden 7168、intermediate 2048，只测 fused forward+backward；后三行共享相同
clustered logical route：

| 状态 | physical destination load | wave | median F+B | conservative peak |
| --- | --- | ---: | ---: | ---: |
| home-only（不同的近均衡健康基线） | 近均衡 | 1 | 44.846 ms | 4.504 GiB |
| multi-wave-only, `B=0` | `[131088,131056,0,0,0,0,0,0]` | 5 | 294.279 ms | 5.105 GiB |
| replica + residual, `B=1` | `[65520,65496,21856,21856,21856,21856,21856,21848]` | 2 | 167.021 ms | 5.346 GiB |
| replica-only, `B=2` | 每 rank 32768 | 1 | 76.652 ms | 5.126 GiB |

同路由下，`B=1` 相对 multi-wave-only 降低 `43.2%`，`B=2` 降低 `74.0%`。该矩阵未在每轮间执行
optimizer step，代表相同权重版本、pool 可直接命中的 micro-batch/梯度累积最佳情形；不能替代 10.2 节
的完整 optimizer-step 结果。

### 10.6 如何理解“是否提升”

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
pytest -q tests/ut/native tests/ut/core/multicore tests/ut/examples/torch/multicore
```

结果：`124 passed, 75 subtests passed in 3.81s`。其中包括两种 compact layout 策略及首次 home-to-replica
交叠的 CPU 回归。备份分支因为额外包含 `moonep_native` 代码和测试，其计数不能与当前分支混写。

完整 `pytest -q tests/ut` 也已尝试，但当前 Python 环境缺少 `transformers`、`toml`、`matplotlib` 和
`pulp`，在 16 个无关测试模块的 collection 阶段终止；没有进入测试执行，因此不把它计作代码回归。

新增的两卡 fused-kernel eager/lazy 切换 ST 在显式激活系统 CANN 9.2.0 和本提交重新构建的
Ascend910B native payload 后结果为 `2 passed in 126.20s`。两条用例分别验证 eager 首次 forward 前重绑
及后续地址稳定，以及 lazy 首次
交叠 snapshot 和 graph 释放后 compact。第一次尝试曾在 `symm.initialize()` 报告
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
- peer-buffer reuse fence：CPU source harness 覆盖首个 Combine tile 选举 publisher；双卡小 shape 验证
  publisher 不依赖固定 worker 获得 Combine 任务，避免所有实际 worker 等待未发布 event。

本次 rebase 集成重新执行了 11.1 节 UT，重建 Torch/Ascend910B native payload 并运行新增的两卡 ST。
第 10 节仍保留精度修复前同一实现周期内采集的历史结果。原始 aggregate JSON 位于被 `.gitignore` 排除的
`output/qwen_moe_optimizer/optimizer_step_*.json`，因此不作为仓库发布物。

## 12. 已落地优化与被否决方案

### 12.1 从 profiler-driven 分支选择性吸收

- `0157ed63` 已以 replica-compatible 方式吸收 kernel-internal multi-wave、固定 workspace 和 scheduler
  主体；没有整段 cherry-pick 覆盖当前 physical routing 语义。
- `d51ec884` 吸收 fixed-size INT32 histogram，并保留零负载 expert。
- `d51ec884` 吸收 GMM dW scratch high-watermark，消除 W2 大需求后 W1 小需求导致的 shrink/reallocate
  同步和 deferred backward 风险。
- 当前保留独立 source/destination route scan。hidden-dispatch copy、dispatch-service、route polling 和
  大诊断 descriptor 等无稳定收益实验没有进入默认路径。

### 12.2 实测后回退

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
2. Ascend rank-local GMM 需要连续 `[H+B, ...]` 权重，尚无上游 MoonEP 式全局 `E+B` symmetric VMM。
3. optimizer 每步更新会使 physical pool version 失效；固定 placement 仍需刷新 replica。pool hit 的最佳
   收益主要出现在同版本 micro-batch 或梯度累积。
4. eager compact 在每层第一次 forward 建图前重绑 Parameter storage；lazy 在首次无 live home lease 的
   replica route 重绑。跨该边界持有的 raw storage alias 不兼容；lazy 的首次交叠和后续 overlapping
   placement 可能临时退回 snapshot，瞬时 HBM 高于 steady state。
5. 共享 workspace 只支持串行 layer/call；真正并发的 MoE 调用必须使用独立 resource group。
6. 当前 Torch backend 要求 NPU BF16、固定 token shape、`E % R == 0`，route-fused `top_k <= 16`。
7. benchmark 是随机权重集成负载；虽已覆盖 activation/state-dict checkpoint 契约，尚未完成真实
   checkpoint 的长时收敛、FP32 replica dW 后的性能/HBM 基线、默认 eager 的新 HBM 基线和动态 skew
   压力测试。

## 15. 下一步优先级

1. 把 histogram、placement、physical IDs/counts/group-list 生成和 transfer table 消费一起下沉到 NPU；
   只移动 planner 而保留 Python `P2POp(peer_rank=...)` 不能消除 control-plane 同步。
2. 评估 VMM/映射适配层，使 snapshot 语义也不复制 home prefix，并最终消除 eager/lazy 首次 `set_`
   的瞬时 storage 切换。
3. 降低 remote-weight prefetch 和 replica-gradient return 成本，加入 planner/P2P/remap/GMM/grad-return
   分段计时。
4. 评估上一 histogram 预测与 count gather overlap，但必须用 generation 校验避免错误 plan 被静默复用。
5. 扩展 `B`、capacity、active expert、更新频率和动态 skew sweep，并补频繁换绑、连续 generation 和
   更长时长的 8 卡压力测试。

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
| planner/route/pool UT | `tests/ut/core/multicore/mega_moe/` |
| real fused-kernel ST | `tests/torch/expert_parallel/test_mega_moe.py`、`_test_mega_moe.py` |
