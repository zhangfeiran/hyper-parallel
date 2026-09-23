# MegaMoe 动态 token 数设计

状态：设计草案，尚未实现或完成设备验证。日期：2026-09-23。

基线：本轮 fetch 后的 `upstream/master`，提交
`051d821fbb8c873983a640ba5db356f8c52c4540`。
本文只设计 multicore 本身的能力，不移植 DSV4.1 Trainer 或实验分支代码。

## 1. 目标和选定方案

同一个 `MegaMoeExperts` 实例应能连续处理不同的真实 token 数，且同一次调用中
不同 EP rank 可以有不同长度，包括某个 rank 没有源 token。
调用方传入真实 hidden states、Top-K ID 和权重；后端不得为了形状一致构造虚假路由。

选定方案是：**声明最大容量、按真实 token 构建路由、按组内最大长度选择任务图、复用固定 SHMEM。**

- 资源上限在第一次执行前确定，SHMEM 对称地址和分配顺序保持稳定。
- 每次调用从输入形状取得真实长度；permute、通信有效数据、专家计算和返回值只包含真实路由。
- 各 rank 通过已有路由计数交换得到一致的任务图长度。它向上对齐到 128，但不补齐输入 Tensor。
- 任务图和 tiling 做有界缓存；各次前向的长度、路由和计划由各自 autograd context 保存。
- 超出声明上限或接收容量时，全组在启动 MegaKernel 前报错；不丢 token，不截断，不隐式扩容。

这里的“动态”是有限容量内的任意实际长度，而不是每次输入都必须是某个固定长度档位。
后端任务图的对齐与缓存是内部执行机制，不要求模型或数据管线选择档位。

| 方案 | 判断 |
| --- | --- |
| 调用方补到最大长度并加入零权重路由 | 只作回归对照；仍产生无效 permute、通信和专家工作 |
| 每种实际长度新建完整 executor 和 SHMEM | 不采用；资源随长度种类增长，难以保证跨 rank 对称分配及延迟反向 |
| 固定最大任务图，运行时读取真实 counts | 作为第一阶段正确性实现；数据量动态，但短序列仍遍历大量空任务 |
| 固定容量资源 + 按本次长度缓存较小任务图 | **目标实现**；复用当前 counts 驱动的 kernel，同时缩小任务数量 |
| 完全由设备生成变长任务队列 | 后续优化选项；需要新的调度及事件协议，不作为本轮依赖 |

第一版完整交付包括不等长 rank、非 128 倍数、零源 token、前后向及资源复用。
仅支持所有 rank 等长，或仅删除入口校验，不视为完成。

## 2. 当前 master 的约束和可复用部分

| 位置 | 已核对的行为 | 动态支持的改动方向 |
| --- | --- | --- |
| [module.py](../modules/mega_moe/module.py) | 构造时要求正的 `local_num_tokens` 且能被 128 整除；每次输入行数必须完全相等；共享 key 包含此值 | 保留静态模式，增加独立的最大容量模式 |
| [spec.py](../modules/mega_moe/spec.py) | `routed_slots=T*K`；接收容量由固定 T、K、EP 和 factor 计算 | 区分资源上限与调用长度，避免一个字段同时表示两者 |
| [route.py](../modules/mega_moe/route.py) | 已交换 `[EP,E]` counts；偏移和 `group_list` 来自真实计数；本地计算容量为 `max(1,R_rank)` | 复用路由公式，交换实际 T 和错误状态，显式返回本次形状 |
| [plan.py](../modules/mega_moe/plan.py) | 每个资源对象持有一个计划，`TaskSplitValue.seq_size` 来自固定 T | 将计划与 SHMEM 解耦，按调度长度缓存计划 |
| [workspace.py](../modules/mega_moe/workspace.py) | SHMEM source/receive 容量固定；前后向共享数据区，具有 completion event 和 ready generation | 固定按上限分配，允许每次只使用有效前缀 |
| [function.py](../modules/mega_moe/function.py) | 本地中间张量已按实际接收量分配；返回时 clone 整个 source buffer；permute-grad 从 `plan.spec` 读取 T | 只复制真实 source 前缀；反向读取本次前向保存的 T |
| [modules/module.py](../modules/module.py) | 串行层在首次使用前共享资源，禁止并发 lease | 保留所有权与串行语义，不引入按长度创建的资源池 |
| [forward/graph.py](../modules/mega_moe/forward/graph.py)、[backward/graph.py](../modules/mega_moe/backward/graph.py) | dispatch/combine/SwiGLU 任务数按 T 推导；GMM 任务数主要由专家数和核数决定 | 明确 T 在图中是调度覆盖上限，非实际 Tensor 长度 |
| [alltoall.py](../tasks/alltoall.py)、[swiglu.py](../tasks/swiglu.py) | 事件阈值包含固定 tile 数 | 各 rank 必须使用一致的调度长度，不能独立缩减任务 |
| [前向 worker](../ops/hyper_mega_moe/op_kernel/worker_kernel.cpp)、[反向 worker](../ops/hyper_mega_moe_grad/op_kernel/worker_kernel.cpp) | 通信读取实际 size；SwiGLU 根据 `group_list` 跳过无效 tile；GMM 修改实际 M，dW 修改实际 K | 验证任意尾部长度、零长度及计划反复复用 |

`route.py::_expert_capacity()` 当前通过一次 `.tolist()` 将各目标 rank 的接收量交给 Host，
用于协调溢出检查和本地内存分配。因此本方案第一版不宣称零 D2H：新增控制摘要应合并到
这一次现有传输中，避免再加一轮 Host 同步或 collective。

该 master 尚不包含 `swiglu_limit`、`create_parameters=False`、外部 `expert_weights`
以及 push/pull 切换接口。本方案不将这些能力当作既有依赖；它们可在后续模型集成时独立组合。
固定通信 split 继续使用 128，不引入 128/1024 的动态切换。

## 3. 对外接口和兼容性

拟议接口，尚不可在当前 master 直接调用：

```python
# 旧接口保持严格固定长度及原有默认值。
fixed = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=2304,
    num_experts=48,
    top_k=6,
    ep_size=8,
    ep_group=ep_group,
)

# 新接口允许每次、每个 rank 的真实长度分别变化。
dynamic = MegaMoeExperts(
    max_local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=2304,
    num_experts=48,
    top_k=6,
    ep_size=8,
    ep_group=ep_group,
)

# x: [T_rank, H] 或其他可展平前导维度；ids/weights: [T_rank, K]。
# T_rank 从 x.shape 得到，不是 4096，也不要求是 128 的倍数。
y = dynamic(x, ids, weights)
```

构造参数 `local_num_tokens` 和 `max_local_num_tokens` 均可默认 `None`，必须恰好提供一个。
不能把旧 `local_num_tokens` 静默改解释为上限；旧的错误检查、默认容量和 checkpoint 参数名保持兼容。
动态模式下 `local_num_tokens` 不记录“最近一次 T”，避免模块可变状态污染其他前向的反向。

动态模式约定：

1. `max_local_num_tokens` 是正整数上限，各 EP rank 的该值及其余资源配置一致。
2. 每次 `0 <= T_rank <= max_local_num_tokens`，真实输入不要求 128 对齐。
3. `topk_ids`、`topk_weights` 严格为 `[T_rank,K]`，输出形状与输入一致。
4. 每个 token 的 K 个 expert ID 在合法范围内且互不重复。这也是当前每专家任务覆盖上限
   所依赖的 Top-K 语义；对外文档应明确，debug 路由校验覆盖重复 ID。
5. `tokens_per_expert` 若提供，必须是本次真实路由的精确本地 histogram，不能包含补位。
6. 保留 BF16 NPU、完整 WORLD EP、现有本地专家数上限等约束，不扩大拓扑支持范围。
7. 同一 EP 组以相同顺序调用相同层的前向和反向；不同 rank 可有不同 T，不能因此跳过该层。
8. 第一版要求组内 grad-enabled 状态及 hidden/专家参数的 `requires_grad` 策略一致。
   冻结输入、冻结专家等组合可支持，但不支持单 rank 独立 detach 导致反向参与集合不同。

首次资源绑定时，在 SHMEM 分配前进行一次组内配置一致性检查，覆盖容量、E/K/H/I、dtype、
模式、factor 和计划版本；共享组注册与分配顺序须一致。输入形状从 Host 元数据获取，
不为读取 T 新增 `.item()`。符号形状编译和 graph capture 不在第一版支持范围内。

## 4. 区分实际长度、调度长度与容量

令 `A(x)=ceil(x/128)*128`，`K=top_k`，`P=EP`，`E` 为全局专家数。

| 符号 | 含义 | 是否随调用变化 |
| --- | --- | --- |
| `T_cap` | 声明的最大本地 token 数 | 否 |
| `T_reserve=A(T_cap)` | 内部容量对齐后的 token 上限 | 否 |
| `T_r` | rank r 本次真实 token 数 | 是 |
| `S_r=T_r*K` | rank r 的真实源端 routed rows | 是 |
| `S_cap=T_reserve*K` | 每卡对称 source buffer 的保留行数 | 否 |
| `C` | 每卡对称 receive buffer 的保留行数 | 否 |
| `R_r` | rank r 的专家本次实际接收行数 | 是 |
| `T_plan=max(128,A(max_r T_r))` | 本次任务图覆盖长度 | 是 |

接收容量沿用既有 factor 含义，但动态模式明确以声明上限为基数：

```text
expert_capacity_factor=None: C = A(P * S_cap)
expert_capacity_factor=f:    C = A(ceil(f * S_cap))
```

有限 factor 仍要求有限且不小于 1。它限制预留接收行数，不是按本次平均负载重新计算的阈值。
因此短 batch 即使负载偏斜超过其“本次平均值 × f”，只要 `max_r R_r <= C` 就可以无损执行。
默认 `None` 覆盖当前路由约束下的最坏总接收量；上限不会在 batch 间自动增长。

| 张量或资源 | 分配/处理依据 |
| --- | --- |
| 输入、router 输出、最终输出、输入梯度 | `T_r` |
| permute payload、inverse mapping、expert-major 返回值 | `S_r` |
| SHMEM routed/source buffer | 分配 `S_cap`，每次有效区间 `[0,S_r)` |
| SHMEM expert/receive buffer | 分配 `C`，每次有效区间 `[0,R_r)` |
| up-proj、activation、saved dispatch、反向本地中间量 | 逻辑长度 `R_r`；仅 ABI 空指针兼容可使用 `max(1,R_r)` storage |
| 图描述符 | `T_plan`，不参与构造 token 或 histogram |
| 专家参数与 dW | 原有 `[E_local,H,2I]`、`[E_local,I,H]` |

例如 `T_cap=4096`、两卡分别 `T=[224,240]` 时，`T_plan=256`，
真实路由仍为 `[224*K,240*K]`，不会变成 `[256*K,256*K]`，更不会变成 `[4096*K,4096*K]`。

显存收益的主要来源是更小的 source 临时张量和 forward 保存量，以及避免虚假路由放大 R。
SHMEM heap 仍按上限预留，不能把计算量减少直接表述为等比例峰值 HBM 下降。

## 5. 路由、组内协商和溢出处理

复用已有的一次 count all-gather，动态模式将其 payload 扩展为固定大小的
`[status, T_r, autograd_mask, counts_r[0:E]]`，以 INT64 做控制摘要及前缀计算，
转换到 native 字段前检查范围。`autograd_mask` 编码本次 grad-enabled 与 hidden/专家参数
的梯度需求；不一致时在同一检查点拒绝，避免零源 rank 脱离必要的反向图。
不增加单独的 token 数 all-reduce，也不交换完整 token 数据到 Host。
静态模式的原计数交换可保留，以避免引入无关性能变化。

`counts[s,e]` 表示源 rank s 发给全局专家 e 的真实行数。所有 rank 得到同一计数矩阵，计算：

```text
source_start[s,e] = sum(counts[s,j] for j < e)
expert_rows[e]   = sum(counts[s,e] for all source s)
R_d              = sum(expert_rows[e] for e owned by destination d)
dest_start[d,e,s] = sum(expert_rows[j] for local j < e)
                   + sum(counts[q,e] for source q < s)
```

dispatch 从 `source_start[s,e]` 复制到 `dest_start[d,e,s]`；combine 逆向使用相同映射。
保留当前 expert-major、source-major 的稳定布局和 `group_list`，不能在 source 偏移中
插入 `rank * T_plan` 或 `rank * T_cap` 之类的等长假设。
数据量为 `sum_r T_r*K` 条路由；没有补位专家 ID，也没有为补位生成零权重。

一次调用的顺序：

1. 检查输入元数据、资源状态；等待 workspace 上一 lease 的 completion event。
2. 计算本地 counts，发起控制/count all-gather；有效的非空输入可并行执行 permute。
3. 等待 gather，生成 `T_vector`、`R_vector`、错误状态和 `T_plan`。
4. 将长度/接收量/状态摘要合并为一次 Host 读取，复用原 `_expert_capacity()` 的同步位置。
5. 所有 rank 依据同一摘要检查 `T_cap`、C、计数总量和 native 整数范围，再准备计划并启动 kernel。

对 `T_r>T_cap` 这样的可识别动态错误，该 rank 发送错误状态和零 counts，且不执行非法 permute；
其他 rank 也在同一检查点退出，避免一部分启动 kernel、一部分提前报错。
这不意味着能够恢复任意 Python 异常、设备故障或未参加 collective 的 rank。
非法 dtype/device/调用顺序仍是 API 前置条件，分布式测试须用超时定位非对称失败。

计数摘要检查至少包括 `sum_e counts[r,e] == T_r*K`、非负负载、`max R_r <= C`
以及组内一致的 autograd 策略。
外部 supplied counts 与每个 ID 的逐元素一致性仍是可信输入契约，debug 模式单独验证。
native 通信 size 是 INT32 元素数，而 offset 是 INT64；需在乘 H 和转 INT32 前检查
每个 segment 的上界，同时检查总计数、图任务数和序列化字节偏移的范围，防止先溢出再检查。

## 6. 任务图选择、覆盖证明和缓存

### 6.1 从资源配置中分离计划配置

内部职责拟划分为：

- 资源配置：T_cap、S_cap、C、E/K/H/I、EP 拓扑、device/dtype，生命周期内不变。
- 调用状态：本次 T_vector、T_r、S_r、R_r、路由及 inverse mapping，前后向独立持有。
- 计划：以 T_plan 生成的 forward/backward RuntimeConfig、tiling 和 profiling 元数据。

`_MegaMoeExecutionResources` 拥有一个 workspace 和计划缓存；创建它不再立即绑定唯一 T 的计划。
`workspace.ensure()` 只接受资源容量配置，不从所选 plan 的长度重新推导 C 或 source allocation。
不能简单对原 `MegaMoeSpec` 执行 `replace(local_num_tokens=T_r)`，这会同时改变容量和共享约束。

### 6.2 为什么所有 rank 选择同一个 T_plan

当前 `AllToAllFillConfig` 的 dispatch 事件阈值包含 `tile_count * EP`；combine 完成阈值也由
计划任务数生成。若 rank 各自用本地 T 建图，即使通信 size 正确，也可能信号不足而死锁。
因此第一版由同一次 count gather 得到组内最大 T，所有 rank 使用相同 T_plan，保留现有事件语义。

在 Top-K ID 每行互异的契约下：

```text
counts[s,e] <= T_s <= T_plan
expert_rows[e] <= sum_s T_s <= EP * T_plan
```

当前 dispatch/combine 每条 source/expert segment 有 `T_plan/128` 个 tile，足以覆盖前一上界；
每个 expert 的 SwiGLU 有 `EP*T_plan/128` 个 tile，足以覆盖后一上界。
GMM 和 dW 继续使用 `group_list` 中的实际 M/K，而不是 T_plan。
复制和计算的尾部仍由实际 counts 决定，图的最后一个 128 tile 不等于执行 128 个虚假 token。

仅在存在真实工作时访问数据；无工作描述符仍完成原定的事件协议。
第一阶段可固定 `T_plan=T_reserve` 验证动态数据语义，但最终应增加按调用长度选择计划，
避免短输入一直遍历最大任务图。RATR 等已有排序继续通过原生成器生成，不改通信 split。

### 6.3 缓存和未完成反向的寿命

缓存位于共享资源组内，key 至少包含 T_plan、E/K/H/I、rank/EP、device/dtype、核数、split
及图/tiling 版本。未来若组合 clipped SwiGLU 等执行变体，变体也必须进入兼容性 key。
T_r、counts、R_r 和参数地址不能作为 plan 中的可变全局状态。

第一版使用小型有界 LRU，例如最多保留 4 个空闲计划；此数字是初始工程配置，需根据
计划字节数和实际长度分布评估，不能当成最优性能结论。

- 冷 miss 用原图生成器构建完整前后向计划，不为它新分配 SHMEM。
- 命中不重建、不重新上传描述符；可在训练前按已知长度做预热。
- autograd context 持有 plan 的强引用。LRU 淘汰只移除缓存所有权，不能释放未完成前后向
  或设备 stream 仍在使用的 runtime/tiling。
- 真正释放 device storage 前必须满足 tensor stream lifetime 或显式 completion event，
  不能仅凭 Python 引用数提前 `resize_(0)`。
- 缓存限额约束的是闲置计划；在途调用所需计划与保存量随在途前向数增长，单独计入 HBM。
- tiling 的设备私有区域会被 GMM/SwiGLU 修改，并非只读对象；同一资源组的 lease 继续串行化。
  复用计划时必须完整刷新本次动态 M/K、rowLen/baseRowLen，不能继承上次尾部值。

热路径仍有一次已有位置的 Host 摘要读取；冷 miss 的 Python 构图和 H2D 时间必须单独记录。
若 miss 频繁，可进一步调整内部计划缓存策略，不能以补位到 T_cap 隐藏成本。

## 7. Native ABI、尾部和零 token

第一版优先保留现有 Tensor 参数列表及 RuntimeConfig 格式。Python 调用传入真实 source rows、
真实 metadata 和固定容量的对称目标；现有 `seq_size` 标量在该路径明确表示正的 T_plan。
真实 T 不再从此标量或 `plan.spec` 反推，而在本次调用/反向状态中显式保存。
当前 host tiling 会拒绝 `seq_size<=0`，因此全空调用也使用 `T_plan=128`。

现有 worker 的动态数据访问主要由 offsets、size 和 `group_list` 决定，这为复用 ABI 提供基础，
但不构成设备验证。实现时必须同时审计 Torch wrapper、ACLNN contiguous/transpose 处理、
host tiling、前后向 worker 和 Meta 路径，排除任何按 seq_size 读写实际 source 的隐含假设。
若最终需要增加字段，Python serializer、C++ reader、schema 和版本检查必须一起更新。

### 7.1 零 size 通信必须保留 signal

当前 `ExecuteShmemPutMem()` 对无效 tile 设置 `send_data_size=0`，仍调用 put+signal。
[PutMemSignalKernel](../ops/hyper_mega_moe/op_kernel/put_mem_signal/put_mem_signal_kernel.cpp)
在复制循环前仍形成数据指针并调用 `remote_ptr`。缩短真实 source buffer 后，空 tile 的
派生偏移可能超过实际 storage，不能依赖“最后没有复制”作为完整的越界安全证明。

方案要求在前后向的 put helper 中建立明确的零 size 分支：不形成/解析数据地址、不读写数据，
但仍执行该描述符应有的远端 signal 及必要的顺序保证。
不得直接从 `ExecuteShmemPutMem()` 返回而漏 signal，因为共享 worker 对 SHMEM task 不另补事件。
正 size 尾部继续精确裁剪并保证目的偏移不超过 C/S_cap。

### 7.2 零源 token 不代表没有专家工作

| 场景 | 必须执行的行为 |
| --- | --- |
| `T_r=0, R_r>0` | 收取其他 rank 的 token，执行本地 experts 和 dW，向来源返回输出/梯度 |
| `T_r>0, R_r=0` | 发出本地 token，并接收它们在远端专家的结果；本地空专家 dW 为零 |
| `T_r=0, R_r=0`，其他 rank 非空 | 参加 count、ready、signal 及匹配的反向，不能单卡提前退出 |
| 全组 T 均为零 | 第一版仍走一致的控制流程，返回空输出和零参数梯度；不新增单独 fast path |

空输入的 permute、unpermute 和 permute-grad 是否接受零维度必须设备核验。
实现显式的空映射分支，不依赖这些算子对零长度的偶然行为。native 若要求非空 Tensor，
可仅在私有 ABI 边界使用一行 dummy storage：counts 保持零，不增加一个 routed row，不送入 GEMM。

空输出仍须连接 `_MegaMoeFunction` 的 autograd 路径；不能用脱离计算图的 `new_empty()`
绕开 backward。所有参与 rank 必须调用反向，即使本地 loss 贡献为零；测试使用可微空输出的
`sum()`，并验证远端 token 在该 rank 的专家权重梯度。冻结参数及输入的组合也需要验证
参与顺序，不能只依赖“本地有没有 token”决定是否启动反向。需要专家反向时，空 unpermute
分支返回原 expert-major 输出的空 view 并保留路由权重的空梯度连接；不另造 detached 输出。
只有路由权重需要梯度、hidden 和专家均冻结时，应验证 unpermute 的 dTopK 路径且无需虚构
专家反向；组内所有梯度均关闭时则为一致的 inference 路径。

## 8. 前后向状态和资源生命周期

前向必须保存本次 T_r、S_r、R_r、所选 plan、原始输入形状、路由 metadata 及 permutation mapping。
输入梯度恢复改为 `permute_grad(..., num_out_tokens=T_r, ...)`，只读取 source 梯度的 `[0,S_r)`。
前向 expert-major 输出也只 clone `combine[:S_r]`，不能复制整个 S_cap 后交给 unpermute。

权重梯度仍是目标专家 owner 上对真实收到的 token 求和；不因为长度不同添加额外 EP 平均，
也不改变 router 权重梯度的缩放。loss 的跨 rank token 归一化属于 Trainer，不由 MegaMoe 推断。

必须覆盖以下时间关系：

```text
forward A(T=224) -> forward B(T=4096) -> backward B -> backward A
forward A(T=224) -> forward B(T=4096) -> backward A -> backward B
```

A 的反向使用 A 的状态；不能读取 module 最近一次长度、cache 当前 plan 或 B 的可变 route。
保存的 dispatch、up-proj、activation 和 mapping 必须独立于可复用 SHMEM。
非重入 checkpoint 重算同样根据该 microbatch 输入重建等价路由和计划选择；不额外引入重复
router 调用，不把资源缓存命中当作允许省略必要 collective 的条件。

资源组沿用已有串行 lease：

1. count exchange 与 native 执行都遵守上一 completion event 的 stream 顺序。
2. kernel、输出有效前缀复制、saved dispatch 保存和输入梯度恢复完成入队后，才能释放 lease。
3. 每次清普通事件计数；ready generation 按方向持久化，切换计划不重新初始化或复位 generation。
4. 同一容量组中各层共享 workspace 和 cache，参数、梯度与 autograd 保存量各自独立。
5. `close()` 必须在所有前后向及设备使用结束后释放资源，再退出 SHMEM；有在途反向时显式拒绝。
6. 首次初始化和最终关闭可保留基线必要 barrier；稳态不新增 Host SHMEM barrier。

动态资源兼容 key 包含模式、T_cap/T_reserve、C 策略、拓扑、dtype 及执行变体，不包含 T_r。
第一版不允许不同容量的模块自动合并资源，不支持同一 workspace 上并发 kernel。
SHMEM 配置、引用计数和关闭规则参见 [SHMEM 文档](shmem.md)。

## 9. 实现顺序和涉及文件

以下为后续代码落地的逻辑阶段，不表示本设计分支已包含这些修改。

| 阶段 | 主要范围 | 完成条件 |
| --- | --- | --- |
| A：容量与调用状态分离 | `module.py`、`spec.py`、`route.py`、`workspace.py`、`function.py` | 先复用最大计划，任意实际 T、不等长 rank、0 token 的前后向通过；旧静态接口回归通过 |
| B：按实际长度选择计划 | `plan.py`、资源组、前后向 graph/generator 的参数语义及测试 | 相同容量下计划可变化；有界 cache；延迟反向和淘汰安全；不分配多套 SHMEM |
| C：native 边界核验和必要修正 | 前后向 worker、put helper、Torch/ACLNN/host tiling | 零 size 只 signal、非对齐尾部正确、空 Tensor 边界明确、native 构建及设备矩阵通过 |
| D：完整验收及使用文档 | 现有 multicore UT/ST、benchmark、README | 精度、资源寿命、峰值 HBM、冷/热性能均有对应源码和 payload 证据 |

C 的零 size 和尾部安全是 A/B 设备测试及功能交付的前置条件，不能推迟到上线后。
推荐先完成 CPU 状态/路由/图覆盖检查，再做必要 native 修改，随后跑 A 的设备精度，最后验证 B。

优先扩展现有 `test_module.py`、`test_route.py`、`test_function.py`、`test_workspace.py`，
复用现有 `_test_mega_moe_resources.py`、`_test_mega_moe_ready.py` 和精度 helper。
必要时增加一个专门的动态 token ST worker；launcher 遵守仓库“不导入框架”的规则。
不为设计草案添加镜像实现的假 kernel 测试，不改 Trainer、optimizer 或其他模型行为。

## 10. 验证与验收

### 10.1 CPU 契约与图检查

- 静态旧接口保持原值、原错误条件；动态模式参数互斥、输入形状和上限检查。
- 不等长 counts 的 dispatch/combine 区间无重叠、无空洞；每个真实 route 恰好来回一次。
- 对照数学前缀和验证偏移、group_list、R_r 与全组 T_plan；覆盖非对齐和空 rank。
- max-T 上限、receive overflow 和整数上界的协调错误；不让任一 rank 进入 native。
- 小/大计划切换时的事件阈值、RATR 任务映射、实际 tile 覆盖及序列化范围。
- 多前向延迟反向、计划 LRU 淘汰、共享关闭、构建失败、不同 stream 的使用顺序。
- spy 原生调用边界：permute 输入 T_r、路由总数 S_r、clone/permute-grad 有效前缀均为真实值。

### 10.2 NPU 精度和生命周期矩阵

| 维度 | 最小覆盖 |
| --- | --- |
| 拓扑 | EP1、EP2、EP8；EP8 全局 E48，每卡 6 专家 |
| 长度 | 0、1、127、128、129、224、240、255、256、257、4095、4096；上限 4096 时 4097 协调报错 |
| 不等长 EP8 | `[224,240,1,127,128,129,0,257]`；`[4096,1,0,129,0,240,4095,128]`；全空 |
| 连续变化 | 小→大→小、非空→空→非空、同一 plan 内反复改变尾部长度、跨 plan 后再次命中 |
| 路由 | 均匀、集中到少数 rank、空专家、零源但有接收、非零源但无接收 |
| 容量 | 默认 lossless、有限 factor 成功、真实接收 overflow；无丢弃/截断 |
| Autograd | 输出、dX、dTopKWeight、dW1/dW2、一轮 optimizer update；冻结输入/参数；梯度累积 |
| 生命周期 | 共享层、两个未反向前向、两种反向顺序、非重入 checkpoint、跨 stream、缓存淘汰后反向 |
| 残留检测 | 对称 buffer 无效区和普通中间量填 poison，切换长短后结果不受旧值影响 |
| 故障 | 单 rank 上限违规、组内配置不一致；必须明确失败，不依赖超时才发现死锁 |

精度主对照使用相同真实输入、权重和路由的独立 FP32 oracle，检查输出及全部相关梯度。
同时与 native EP 参考、旧固定长度路径对照；旧路径为比较而构造的零权重补位不得作为唯一 oracle。
阈值使用已有同 dtype/shape 验收标准并在运行前冻结；若发生 BF16 舍入争议，用相同算子输入定位，
不通过放宽阈值掩盖 route 丢失、重复或梯度缩放错误。CPU 数学验证不代替 NPU kernel 精度。

通用 master 验收使用当前支持的 SwiGLU 语义；与 clipped SwiGLU 独立改动组合后，再验证
limit=1/10 且实际触发 clamp。DSV4.1 整网验收还需要独立的模型适配集成，不能把本分支的
通用算子测试称为已通过 VLM 整网。

### 10.3 性能和 HBM

比较三条路径，均固定实际输入、routes、权重和计算精度：

1. 原 master 固定 T_cap，测试 harness 补零权重路由：测移除补位的直接收益。
2. 新动态路径：测不补位、计划选择和缓存成本。
3. native EP 真实长度参考：衡量实际 MoE 加速比，不以无效 token/s 制造优势。

用独立进程 ABBA 成对测试短、长、混合、不等长与偏斜输入，分别报告冷 miss 和预热稳态。
满长度 `T=T_cap` 对照旧路径检查额外控制开销；这是回归验收，不能只报告短序列收益。
稳态需无新增 Host SHMEM barrier、无按 batch 重分配对称空间、无按层新增 token collective。

每份证据记录：实际 T_vector、S_r/R_r、T_plan、T_cap、factor、cache hit/miss、计划字节、
route/Host 摘要/构图/前向/反向时间、有效 token/s、allocated/reserved 峰值、整卡 HBM 采样峰值、
SHMEM heap 和在途保存量。heap 与其中 buffer 字节不重复相加；理论字节账本不是峰值实测。
进程归属或设备干扰不明的性能样本不进入结论；记录源码、native payload、CANN/Torch 身份。

通用算子通过后，DSV4.1 适配层再删除 dummy token/id/weight 拼接和输出裁剪，改传真实 T，
将配置推导值作为最大容量。保留原 router、image_mask、shared experts、外部权重和 FSDP hooks，
重新验文本/VLM 整网数值、MoE 时间占比与整网吞吐，不能直接外推单层加速比。

## 11. 本轮交付边界和待设备确认项

本轮仅交付设计文档。已基于上述 master 核对 Python 路由、容量、任务图、资源、autograd，
以及 native 通信、动态 GMM/SwiGLU、事件触发和 host seq_size 检查位置。
未修改 kernel、未构建 payload、未运行 NPU，也未取得新的精度或性能结果。

设计公式以独立 Python 整数模型检查：固定随机种子 `20260923`，覆盖 EP1/2/4/8、边界长度、
上表的不等长组合及随机 Top-K 计数，共 172 组、1,378,350 条真实路由。
验证源/目标区间、dispatch/combine 往返、计划覆盖及无损容量上界；其中包含 36 个
“源长度为零但接收非零”的 rank。该检查没有调用框架或 native，只验证公式和索引关系。

实现前需明确验证的关键点：

- ACLNN 对空 source 的实际行为，以及 dummy storage 是否足以维持零数据的控制调用。
- 空输出的 autograd 在各类冻结参数组合下是否保证所有 rank 都进入匹配反向。
- SwiGLU 小尾部→大尾部、dW 动态 K 反复变化时 tiling 刷新与缓存可见性是否完整。
- 计划淘汰后设备 stream 与延迟反向的 storage 生命周期。
- 实际长度分布下的构图/cache 成本及最大 rank 对任务数的影响。

上述项目纳入实现验收；不能以保留输入补位、跳过零 token、只测均匀等长或放宽数值阈值规避。
