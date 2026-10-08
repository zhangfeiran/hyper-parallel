# 容量受限的专家热拷贝

该特性需要显式启用。在不改变逻辑专家归属的前提下，它将数量受限的临时专家副本分配到其他 EP rank。
共享规划器、路由契约和稀疏传输接口均位于本目录，不依赖 multicore 或 multi-wave 调度。

下文用 home 表示原始 owner rank 上的专家，用 guest 表示其他 rank 上的临时副本。

## 使用入口

原生（native）分组专家使用现有 EP 策略：

```python
from hyper_parallel.core.expert_parallel import ExpertParallel

ExpertParallel(replica_slots_per_rank=1).apply(experts, ep_mesh)
```

`experts` 为 `components.modules.moe.GroupedExperts`，其 `w1`、`w2`、`w3` 参数及 state-dict 布局不变。
当前适配器支持 BF16 NPU SwiGLU、专家 TP=1 和同步 `all_to_all`。
不支持的异步 combine 或 deredundency 组合会在构造时被拒绝。

Multicore 在 `MegaMoeExperts` 上提供相同的 B 参数：

```python
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=24,
    top_k=8,
    ep_size=4,
    ep_group=ep_group,
    replica_slots_per_rank=1,
    dispatch_mode="push",
    initial_capacity_factor=1.25,
    capacity_growth_factor=1.25,
)
```

Push 和 pull 都执行单个物理专家调度，B=0 保留原有执行路径。
热拷贝要求每个 token 的 TopK 专家 ID 均在合法范围内且互不重复。
Multicore 在稀疏权重通信前以集合通信方式校验这些条件；native 沿用现有按专家组织的计数契约。

Multicore 对 ID 做 clamp 后，使用 `scatter_add_` 构建固定长度的 int64 直方图，
并将越界及重复 ID 标志保留在同一份集合通信载荷中。
这样可以避免 `bincount` 内部的设备 min/max 标量回读，同时保持集合校验契约和计数精度。

## 容量与专家放置

设 R 为 EP 并行度，H 为每个 rank 的 home 专家数，S 为每个源 rank 的 token 数，
K 为每个 token 选取的不同专家数，b=min(B,H)。定义 A=S*K，U=R*S*min(K,H)。

整数规划器首先平衡工作量，使每个接收 rank 最多从一个原始 owner 接收迁移工作。
对于平衡过程中迁移的 m 行，它选择最大的 b 个专家片段，保留其中 floor(b*m/H) 行。
随后利用剩余 guest 槽位和已有副本继续迁移，以降低过载。
每个源 rank、每个逻辑专家的计数都精确守恒，不丢弃任何行。

对于固定的专家放置，规划器优先满足源 rank 的本地配额。
在分配剩余远端流量之前，先达到该放置下能够实现的最大本地行数。

当 0 < b < H 时，保守容量上界为：

```text
Cmax = align128(min(U, ceil(((H-b)*U + b*A) / H) + R - 1))
```

当 B>=H 时，Cmax=align128(A)。
MegaMoe 的 B=0 路径保留原有基于全局 token 数的上界及路由契约。

Push 仍按 `initial_capacity_factor` 初始分配，并按 `capacity_growth_factor` 动态增长。
初始分配和几何增长的预留空间都不能超过 Cmax，非法容量需求会在容量快速路径之前被拒绝。
显式堆预算仍使用现有的最小增长重试机制。
规划器根据当前容量设置目标负载，避免为已经能容纳的路由创建不必要的副本。
Pull 继续使用按源 rank 大小分配的对称存储和 rank 本地接收临时空间。

## 训练与传输

HCCL P2P 仅传输选中的 owner 到 guest 权重切片。
传输顺序确定，每个异步 handle 都会在结果使用前完成等待。
同一进程组按设备、dtype 和矩阵布局共享持久化的 B 槽位池。
池借用 home 矩阵，仅存储 guest 权重和 FP32 guest dW。

独占租约与完成事件保证不同 stream 之间的复用顺序。
不同层可以共享存储，池不会持有这些层的参数。
Native 分别执行 home 和 guest GMM 段；multicore 使用版本化地址扩展选择 home/guest 矩阵及梯度输出。
B=0 保留稠密内核 ABI。使用分离权重前，需要基于当前版本重新构建 multicore 原生载荷。
Native 仍需要现有的 W1/W3 打包操作。

普通 guest 分配不受 push 堆替换影响。
设置 `MegaMoeExperts(..., replica_transport="shmem")` 可以启用单边权重 put 和 owner 侧 FP32 梯度 get。
对称 B 槽位接收区纳入堆占用统计、集体分配与释放，以及增长前置检查。
堆重建后，每次调用都会绑定当前接收区，不使用远端 FP32 原子汇聚。
该实现依靠发布与消费 barrier，并包含额外的暂存拷贝。
P2P 仍为默认传输方式，单边通信需要显式启用。

`replica_transport="shmem_signal"` 直接使用对称 B 槽位中的执行权重和 FP32 guest dW。
内核直接读写这些视图，省去中间接收区拷贝和普通 guest 池。
持久化 provider 管理 stream 租约和协议 epoch。

接收方仅在上一轮槽位消费者完成后授予 credit；owner put 权重后发布 ready，接收方确认权重已到达。
梯度阶段，每个 guest 发布已完成的 FP32 输出，owner get 并累加，再确认远端读取完成。
所有接收方向的 credit 都先入队，再执行发送方向的等待，以避免等待环。
不同 channel/peer/slot 组合的信号使用独立的 64 字节缓存行。

初始化和少见的 int32 epoch 回绕使用 host barrier。
稳态预取与梯度返回使用按 stream 排序的信号，不经过 host barrier。
该模式尚不将通信与 home 专家计算重叠。

`replica_transport="shmem_signal_sdma"` 沿用相同的直接槽位和信号协议，
但通过 ACL 异步拷贝访问 SHMEM 映射的 peer 地址。
该模式要求直接 peer 映射，未映射的 peer 会被拒绝。

直接使用底层 provider 时，通过 `SignalReplicaTransport(..., use_sdma=True)` 选择该能力，
并要求 runtime 的 put/get 接受该关键字。
SHMEM runtime 也提供 `put/get(..., use_sdma=True)`。
该选项保留 stream 顺序及分配校验，不增加暂存缓冲区，也不改变规划器和容量策略。
P2P 仍为默认方式。

`replica_transport="shmem_signal_sdma_parallel"` 进一步按目标 peer 延迟创建独立 stream，
用于预取发送方向的权重。
直接使用底层 provider 时，对应选项为
`SignalReplicaTransport(..., use_sdma=True, parallel_prefetch=True)`。

每个拷贝 stream 等待源数据准备和所有接收方向的 credit。
调用方 stream 先确认接收权重，再等待发送方向的拷贝完成。
因此，即使 owner 或调用方 stream 发生变化，池租约仍覆盖所有拷贝 stream。
源 tensor 会记录在其拷贝 stream 上，以保证 allocator 管理的存储生命周期。

梯度读取与 FP32 累加保留原有确定顺序。
该模式增加 stream/event 资源，但不增加专家暂存缓冲区或对称堆字节数，
也不将预取与 home 专家计算重叠。
所有 rank 必须选择相同模式，runtime 必须支持在不同 stream 上调用。

`replica_transport="shmem_signal_sdma_bidir"` 还会为每个投影矩阵使用独立 stream，
读取并累加 guest 梯度。
直接使用底层 provider 时，在上述选项中加入 `parallel_gradients=True`。

每个投影仍按原有目标 rank 顺序进行 FP32 累加，并使用独立输出，
因此不会有两个 stream 更新同一个梯度矩阵。
所有投影都等待完成并汇合到调用方 stream 后，才确认远端读取或释放槽位租约。

该模式按 provider 延迟缓存一个专家的全部投影 FP32 梯度临时空间，与 B 和 EP 大小无关：
占用 `4 * sum(prod(matrix_shape))` 字节；对于打包的 W13/W2，占用 `12 * D * I` 字节。
只有读取远端梯度的 rank 才会分配，并在不同 peer、槽位和调用之间复用。
公开属性 `gradient_scratch_bytes` 报告已分配 tensor 的字节数。

这是本地 allocator 缓存，与对称 B 槽位及 SHMEM 堆预算分开管理，
在所有 stream 完成后随 provider 释放。
相对于顺序实现的临时缓冲区，这个分配量不能直接视为实测的峰值 HBM 净增量。
原有 P2P、串行 SDMA 和仅并行权重预取的 SDMA 模式仍可使用。

`replica_transport="shmem_signal_sdma_overlap"` 保留双向并行拷贝，
并允许 MegaMoe 在 guest 权重到达之前开始 home 计算。
这种重叠是 MegaMoe 特有能力；native 使用在返回前完成的 HCCL P2P 权重预取。

共享的 `prefetch_weights(..., overlap=True)` 上下文返回处于租约中的 tensor，
并提供 `wait_weights()` 以及可选的 `weight_ready=(base_address, epoch)` 元数据。
普通即时预取 provider 保留原有行为，MegaMoe 融合消费者遵守每个槽位的 ready 元数据。
初始实现保持原有任务顺序，在前向中重叠 home GMM1，在反向中重叠 home 激活梯度矩阵乘。

Credit 和必要的源 tensor 连续化转换在拷贝 stream 分叉前完成。
随后各拷贝 stream 只下发 SDMA 权重和 ready 字拷贝，
发布 ready 不需要 AIV 辅助任务与融合内核并行执行。
每个 guest 槽位使用已有缓存行表示 ready，不增加对称堆字节数。

共享租约在释放存储前等待所有拷贝完成并确认接收方，即使调用方没有 guest 行也必须参与。
所有 rank 必须使用相同模式。

Multicore 的延迟消费者需要分离权重 runtime ABI v3，
该版本在 v2 布局后追加 ready 基址和 epoch。
必须重新构建匹配的前向与反向内核；旧载荷不支持这一可选模式。
新内核仍接受用于即时预取的 v2。

`replica_transport="shmem_signal_sdma_projection"` 为每个矩阵独立发布 ready。
MegaMoe provider 选择 `projection_ready=True`，并将该选项纳入 `signal_storage_bytes(...)`。
共享视图提供 `projection_ready=((matrix0_base, matrix1_base, ...), epoch)`：
`wait_weights(matrix_index)` 只等待指定投影，`wait_weights()` 则等待全部投影。

每个槽位、每个矩阵的发布使用独立的 64 字节缓存行，
增加 `64 * B * matrix_count` 个对称字节。
这些缓存行不与 peer credit、整个槽位的 ready 或 ACK 通道共享存储。

前向按 W13、W2 的顺序拷贝，反向按 W2、W13 的顺序拷贝。
每个矩阵在自身 SDMA 拷贝完成后立即发布 ready，
使第一个 guest 矩阵乘可以在另一矩阵仍处于拷贝过程中时开始。
Multicore 使用包含两个 ready 基址及 epoch 的分离 runtime ABI v4；
匹配内核也接受 v2 和 v3。

槽位释放仍须等待所有矩阵消费者与全部拷贝 stream。
默认方式仍为 P2P；为指定 shape 和路由选择传输模式时，需要测量完整训练步。

信号存储和持久化 provider 由堆管理器一起释放、重建。
Multicore 的 autograd 不保存对称地址或旧 provider。
兼容的 MegaMoe 层必须共享执行资源，才能共享对称槽位；
native 层共享普通 HCCL P2P guest 池。

单边传输及权重拷贝与 home 计算的重叠均属于 MegaMoe 能力。
Native 通过 `ExpertParallel(..., replica_slots_per_rank=B, replica_min_rows=...)`
使用共享规划器、即时 HCCL P2P 权重预取和 FP32 梯度返回。
它不接受专家副本传输 provider，也不会初始化 SHMEM runtime。
共享规划器始终位于 multicore 之外，不依赖其 runtime。

每次 autograd 调用保存自身不可变的路由和原始 owner 权重。
反向重新预取该路由，因此允许其他前向先执行。
FP32 guest 梯度按目标 rank 分轮返回，使 owner 的接收临时空间每个投影最多需要 B 个专家槽位。
这些梯度先加到 owner 的 FP32 partial 中，再进入原始参数边界及其 hook。
不会注册副本参数或优化器状态。

Multicore 的分组内核直接产生 FP32 dW。
在已验证后端上，native CANN 非量化 grouped-matmul API 不支持 BF16 输入、FP32 输出的 dW。
因此 native 适配器的前向和 dX 使用分组计算，dW 使用普通 FP32 矩阵乘；
分段边界来自 plan，而不是设备回读。

当专家 ID 范围可以用 24 位表示时，排序键使用精确的 FP32 值，排列索引始终为整数。
更大的专家 ID 范围使用整数排序。
CPU 规划在计数交换后执行；可选设备规划器将配额与源 rank 的连续路由段保留在 NPU 上，
只回读一份控制摘要。

## 当前实现范围与限制

| 能力 | 已实现的执行方式 | 验证范围 |
| --- | --- | --- |
| 共享 CPU 规划器（默认） | Native 和 MegaMoe push/pull | CPU 策略、容量测试及 NPU 训练 ST |
| 共享设备规划器 | 独立 Ascend AIV；native 和 MegaMoe | 与 CPU 精确一致、保留旧 plan、双 stream NPU ST；仍有一次控制摘要回读 |
| Native 专家热拷贝 | 普通 HCCL P2P、FP32 guest 梯度合并 | 前向、反向、优化器、跨层和逆序反向 ST；不支持单边 provider |
| MegaMoe 按投影发布 ready | SDMA 权重拷贝与 home 计算重叠 | 投影及 kernel-gradient ST，包括融合消费者下发后才发布 ready |
| MegaMoe 提前返回梯度 | W2/W13 ready 与 FP32 owner 累加 | Kernel-gradient ST；完整步收益需要单独进行一致配置的基准对照 |
| Push 接收容量 | 动态增长，受理论最大值约束 | 保留旧前向时发生增长，随后逆序反向 |
| 离线校准的配额优化 | 仅 CPU 规划器 | 设备规划拒绝该选项；成本估计为近似值 |
| 多节点、专家 TP、图捕获、FSDP/DP 组合 | 当前测试不构成支持声明 | 需要专门的组合 ST |

设备规划器仍受后文所述拓扑、临时空间和 int32 配额限制。
能力实现与正确性测试不代表已经取得性能收益。
槽位池的租约会拒绝重叠的 host 提交，而不是额外分配 B 槽位。
Native 参数打包及逐专家 FP32 dW 矩阵乘的成本仍需测量。

基于 barrier 的 RMA 还预留 B 槽位 FP32 对称接收区。
基于信号的 RMA 将 guest 权重和梯度直接放在对称内存中，
另有 `5 * ep_size * B * 64` 字节信号存储和矩阵对齐填充。
分配大小变化不能视为实测的完整执行峰值 HBM 变化。

精度验证不构成完整执行加速或峰值 HBM 降低的证据。

## 测试

CPU 规划器与容量测试位于
[`test_hot_replica.py`](../../../../tests/ut/core/expert_parallel/test_hot_replica.py)。
分布式 native/push/pull 启动入口位于
[`test_hot_replica.py`](../../../../tests/torch/expert_parallel/test_hot_replica.py)。

Worker 还接受 `--backend`、`--budget`、`--replica-transport`、
`--replica-planner={cpu,device}`、`--default-group` 和 `--result-dir`，
用于受控的新进程验证。
测试比较输出、输入/路由/权重梯度、SGD 更新和 momentum，
并检查不同活动 plan 的逆序反向。
小 shape 还覆盖独立层共享 guest 池、不同权重及逆序反向。

Worker 可通过 `--tokens`、`--hidden`、`--intermediate` 和 `--top-k` 选择实际模型维度。
`--same-backend-reference` 将 multicore 热拷贝与相同计算后端的 B=0 比较；
默认参考实现为 native B=0。
`--fp32-reference` 选择显式的 native B=0 FP32-dW 参考实现，不修改生产 hook；
它与 `--same-backend-reference` 互斥。
FP32 参考用于正确性对照，不用于生产基线速度对照。

具名启动入口覆盖默认 WORLD 和不连续 subgroup 的 P2P、native 设备规划、
使用 projection 和 kernel-gradient 传输的 push/pull 设备规划，
以及 K=1/2/3/4/5/6/8 的延迟反向。

延迟反向检查合法且不同的专家，以及两个确实保存了副本的实际 plan。
完整的活动副本验收使用 B>0 和 `--replica-min-rows=0`。
Push 的 K>=5/B1 用例检查旧前向仍存活时的真实增长。
规划器一致性测试在两个 stream 上保留十二个 plan。
结果记录请求与实际配置、源码身份、EP 成员、CANN/Torch 版本，
以及已导出接口可提供的已加载载荷 hash 和 ABI。

例如，按后文说明构建并激活载荷后，可运行：

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_test_hot_replica.py \
  --backend push --budget 1 --top-k 8 --replica-planner device \
  --replica-transport shmem_signal_kernel_gradient --fp32-reference \
  --result-dir ./logs/hot_replica/push-device-kernel-gradient
```

接收梯度 partial 保留逐元素精度门槛。
延迟反向还检查所捕获 partial 的精确 BF16 累加，因为相消会放大最终 BF16 梯度的相对误差。
比较失败会在清理前以集合方式报告，避免其他 rank 被滞留。

信号测试在检查结果前，还会在两个 NPU stream 上轮换 owner 十二次，每轮都延迟一个 rank。
CPU 协议测试覆盖任意 rank 的推进顺序和提前入队的大量调用。
Worker 接受 `--benchmark-iterations`，用于在固定热点路由上测量预热后的前向、反向与 SGD。
性能对照需要新进程配对运行及进程归属审计，与精度结果分开记录。

## 过滤小专家副本

`ExpertParallel(..., replica_min_rows=N)` 和
`MegaMoeExperts(..., replica_min_rows=N)` 使用相同的 host 规划策略。
默认 `N=0` 保留按 token 数平衡的专家放置。
当阈值为正时，规划器优先将较小副本的工作返还给 home owner。

如果容量约束要求部分工作仍留在远端，规划器先尝试将这些行集中到已选择的副本中，
不创建额外副本；必须保留的副本即使行数低于阈值也不会移除。

过滤后，保留的副本可以吸收其原始 owner 的更多行。
每次迁移都在两端负载平衡时停止，因此不会增加全局接收峰值。
副本位置及完整专家的权重/梯度字节数保持不变，但 token 流量与计算耗时可能变化。
这一步优化使用行数，而不是校准后的后端成本模型。

阈值是针对工作负载的暴露拷贝成本近似值，需要通过完整训练步对照校准。
固定行数阈值不是完整的计算/通信成本模型。
源直方图和所有逻辑 TopK 选择不变，原始 owner 仍接收完整的 FP32 梯度和。

Multicore 根据精确的 `S/K/B` shape 提供已证明的接收上界。
Native 只有按专家组织的直方图，不包含原始 `S/K` 元数据。
当各源 rank 大小相同时，规划器使用 `T=S*K` 和 `max(count)<=S`，
选择能够整除 `T` 的最大可行整数 `K`。
对于固定 `T`，接收上界随 `K` 增大不增加，
因此得到的上界对所有兼容的、不重复 TopK shape 都是安全的。

源 rank 大小不相等时，使用已观测 home 负载代入构造式上界。
仅当新 plan 满足该上界，且不超过原始纯 home 执行的峰值负载时，才移除副本。

将工作返还 home 可能增加所需接收容量，同时减少远端权重/梯度字节数。
Push 仍在同一理论最大值以内动态增长，策略不改变驻留 guest 预算 `B`，也不额外分配权重缓存。
规划策略按模块设置，不改变共享执行存储布局。
同一次调用的全部 EP rank 必须使用相同策略。

## 消费调用独占的梯度缓冲区

Native 和 multicore 适配器在每次反向时分配新的 FP32 home 梯度。
它们向共享 `return_gradients(...)` 辅助函数传递 `consume=True`，
允许直接在这些缓冲区中累加，而不是克隆全部 home 专家梯度。

只有 detached、FP32 且由当前调用独占的缓冲区才能被消费。
它们不能与参数、保存的激活、guest 槽位、其他投影或此前返回的梯度共享存储。
调用方必须使用返回的 tensor，此后不能复用原始梯度值。

Provider 的普通 `return_gradients` 方法保留不消费输入的契约。
Provider 只有显式实现 `return_gradients_owned(gradients, guests, route)`，
才表示支持所有权转移。
P2P 和内置单边 provider 均支持该接口；未实现该方法的 provider 仍接收原有三个参数。
所有路径都保留相同的 FP32 peer 累加顺序。

在并行信号传输中，生产 stream 仍在梯度生成与发布后记录分叉事件。
拷贝 stream 等待该事件，在各自 stream 上记录输出 tensor 的生命周期，
并在 ACK 与租约释放前完成汇合。
返回的 home 梯度是普通的调用内分配，不属于可复用 guest 池或对称堆。

## 在 multicore 反向内核中返回 W2

`shmem_signal_kernel_gradient` 保留按投影的 SDMA 权重预取，
并在融合反向内核中使用单边 MTE 读取 W2 梯度。

适配器首先校验每个 Cube 都在 ActGrad 之前执行 W2Grad，
并确认该专家的 ActGrad 事件等待其所有 Cube worker。
未知调度保留较晚的梯度返回路径。共享规划器和接收容量策略不变。

原本空闲的偶数编号 AIV worker 累加各 home W2 矩阵中互不相交的块，
每个 FP32 元素都保持目标 rank 的累加顺序。
0 号 worker 在等待远端生产方之前，先发布本 rank 全部 guest 的 ready。

全部 worker 完成后，内核确认远端读取，并等待本 rank guest 的读取方完成，随后才能结束。
奇数编号 AIV 和 Cube 队列不依赖这些 worker。
之后 Python 通过普通传输路径返回 W13。

V5 runtime 扩展指向当前调用独占的元数据及缓存行完成标志。
它使用独立 epoch 复用 provider 的 ready/ACK 通道，不增加完整专家接收区。
该模式仅适用于 MegaMoe；native 在本地反向计算完成后通过 HCCL P2P 返回两个投影。

## 复用 plan 中的接收负载

热拷贝路由已经包含精确的 CPU `destination_loads`。
MegaMoe 同时将其用于 rank 本地中间缓冲区分配（`max(1, local_load)`）和全局 push 增长检查，
省去对已上传 dispatch counts 的第二次设备求和及 host 回读。

没有副本 plan 的路由保留计数回读。
Push 仍在理论最大容量以内动态增长；pull 保留其预分配上界。
计数交换的等待和路由 offset 不变。

## 基于实测成本优化配额

`ExpertReplicaCostModel` 可以在容量规划后，进一步优化已有副本的配额。
通过 `ExpertParallel(replica_cost_model=model)` 或
`MegaMoeExperts(replica_cost_model=model)` 向每个 rank 传入同一个不可变模型。

校准数据必须匹配 hidden/intermediate 维度、EP 大小、后端和传输模式。
Native 校准使用普通 HCCL P2P。
版本 2 要求显式设置 `calibration_version=2`、使用 FP32 权重 partial，
并提供兼容的调度身份：
`native_grouped_v1`、`fixed_queue_v1` 或 `fixed_queue_w13_first_v1`。
调度身份默认由后端和传输模式推导。

旧校准数据需要重新生成；仅修改版本号，不代表已经验证其适用于新模型。

在没有实测调度前缀重叠窗口的情况下，所有模式都计入完整的暴露权重和梯度传输成本。
固定队列中，后续 home 工作不能抵消第一个 guest 依赖处的阻塞。
接收和发送传输分别计数后相加，不推断全双工带宽收益或多个 peer 的竞争关系。

评分为 `max(rank forward) + max(rank backward)`，
报告和配额优化使用同一目标。
这是保守的阶段启发式，不表示真实执行中间存在全局 barrier，也不保证延迟上界。
当前不支持按阶段前缀折扣计算重叠收益。

模型需要从 `(0, 0)` 开始、成本单调不减的前向与反向 `(rows, milliseconds)` 表，
以及每个副本的完整权重拷贝成本、暴露梯度返回成本和可选的远端 token 成本。
在 `minimum_gain_ms` 中计入测量不确定性及额外 host 规划时间。

这些都是近似成本。训练中启用模型前，还需要单独测量完整前向、反向和优化器步。
没有校准数据，或专家行数超出测量范围时，保留原配额策略。

优化仅搜索已有复制边，可以移除副本，
并只接受相对当前阶段评分、严格超过 `minimum_gain_ms` 的改进。
接收行数可以在现有容量上限内增加。
B 槽位预算、无损路由，以及受理论上界约束的 push 动态增长均保持不变。

校准后的 CPU 优化在单次调用内保存 rank 评分和源行数合计。
每个 quota trial 只重新计算原始 owner 和 guest target，
保留原有 rank 内 home/guest 累加顺序及前向、反向分别取最大值的方式。
移除 guest 边后，在处理下一条边之前更新两个端点的传输计数。
评分与插值缓存在调用结束后丢弃，不在不同调用之间共享路由或权重状态。

路由在同一个对齐缓冲区中上传调用独占的连续路由段和 dispatch counts。
当全局没有副本时，逻辑 ID 直接映射到 home 物理槽位，包括 B 预留的空槽。
CPU plan 的派生数据仅缓存在不可变 plan 上；设备元数据和权重不跨调用缓存。

## 共享融合设备规划器

在 `ExpertParallel` 或 `MegaMoeExperts` 上设置 `replica_planner="device"` 可选择设备规划器，
默认仍为 `"cpu"`。

设备求解器实现相同的整数放置策略，包括 `target_load`、`replica_min_rows`、
满足容量约束的小副本集中，以及保留复制边的再平衡。
离线校准成本优化要求 CPU 规划器，设备后端会拒绝该选项。

可选 Ascend 910B 内核独立于 multicore 和 SHMEM 构建。
激活 CANN 环境后，在仓库根目录运行：

```bash
cmake -S hyper_parallel/core/expert_parallel/hot_replica/_device_kernel \
  -B build/replica-planner
cmake --build build/replica-planner -j 4
cmake --install build/replica-planner \
  --prefix "$PWD/hyper_parallel/core/expert_parallel/hot_replica/_device_kernel"
```

打包可选库时，将其安装到
`build/replica-planner-payload/core/expert_parallel/hot_replica/_device_kernel`，
并使用现有 wheel 构建选项
`HYPER_PARALLEL_NATIVE_OUTPUT_ROOT="$PWD/build/replica-planner-payload"`。

同时打包 multicore 时，两者放到同一个载荷根目录。
尚未构建的源码安装需要运行上述 CMake 命令，
并将 prefix 指向所安装的 `hot_replica/_device_kernel` 目录。

库采用延迟加载，CPU 规划和普通 native EP 不加载该库。
首版融合实现使用 180 KiB AIV 临时空间，接受满足以下条件的拓扑：
`5*EP*E + 3*E + 4*EP <= 23040`，包括 EP16/E256。
更大的拓扑在 launch 前被拒绝，计数值及算术溢出在设备端检查。

单个设备任务读取 gather 后的直方图，
写出物理专家放置、目标计数、rank splits、int32 dispatch quotas 及预排序的源连续路由段。
连续路由段和 int32 counts 直接供后续路由使用，
不需要额外排序、quota gather 或计数转换内核。
只有紧凑控制前缀被回读。

每次调用独占输出存储，后续调用和不同 stream 不会覆盖仍保留的路由。
生产方必须建立正常的当前 stream 依赖，launch 在该 stream 上记录 tensor 生命周期。
不需要图缓存或可变 replay workspace。

重映射与计数消费者直接使用设备 tensor。
一次紧凑控制拷贝为 host P2P、本地 native 分配及 push 容量增长提供物理归属、
目标边界和 rank splits，因此执行边界仍由 host 控制。
Native 不导入 multicore 或单边传输实现，push 继续增长到理论上界以内。

当生成的 Cube 调度能够证明逐专家 ready 时，
MegaMoe 的 `shmem_signal_kernel_gradient` 模式还会在反向内核中返回 W13。
每个 Cube 上 W13Grad 都先于 dX；完整的 dX 事件保证 W13 输出完成。

W2 和 W13 使用不同 epoch 和完成标志存储。
内核在释放 guest 租约前等待所有远端读取方完成，未识别的调度保留较晚的返回路径。

## 受控性能测试与诊断

使用与正确性测试相同的四卡环境和原生载荷，运行新进程 worker：

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_benchmark_hot_replica.py \
  --backend push --budget 1 --replica-transport shmem_signal_sdma_projection \
  --replica-planner cpu --result-dir /tmp/replica-benchmark
```

默认 EP4/E24，每个 rank 有六个 home 专家，S=512、D=5120、I=1792、K=8。
Worker 直接使用 MegaMoe 拥有的参数和固定路由概率，
测量完整前向、反向与 SGD 步；rank 最大耗时的归约位于计时区间之外。

它分别记录延迟初始化、首次热点路由与增长，以及平衡、稳态热点、轮换热点分布。
预热后的测量区间必须保持容量和 SHMEM epoch 不变。
比较实现前先审计记录的 plan 和容量；计时不证明数值正确性。

在独立新进程中按 ABBA 顺序比较相邻模式：
B=0、B=1 CPU/P2P、双向 SDMA、槽位重叠、按投影 ready、内核梯度返回，
最后在选定传输方式下比较 CPU/device。

分析单个机制时，应保持 shape、路由、plan、精度和预热后的容量相同。
B=0 计算 BF16 权重 partial，B=1 使用 FP32 partial，
因此第一组是实际生产基线对照，而不是仅隔离通信的实验。
该 worker 内优化器和参数直接归属一致，不同 driver 的绝对耗时不能互换使用。

添加 `--diagnose` 可额外采集不计时的 TorchNPU trace、内部 cycle trace，
以及包含排队和等待的 host/stream 时间区间。
记录涵盖 count gather、solver/AIV launch、控制 D2H、重映射、元数据上传、
调用元数据和传输下发。

这些区间可能嵌套、重叠并包含队列空隙，不能直接相加得到完整步延迟，
也不能把 stream 区间视为纯设备内核时间。
Host 观测器排除 profiler 清理和 plan 审计回读；内核耗时应查看框架 trace。

内部 `ReplicaWeightReadyWait` 记录标识 rank、投影、槽位、epoch、消费者阶段、
逻辑专家和 owner peer。`epoch` 不是任务编号。
只有 profiling 路径记录 cycle 对，普通执行不增加同步或轮询日志。
Trace 检查记录丢失，以及活动 guest rank 上缺少等待记录的情况。

内存快照区分 Torch 已分配/预留峰值和外部 SHMEM 堆，
并列出 guest 权重/FP32 梯度载荷字节数、它们共享的底层存储、梯度临时空间、
预期 home FP32 partial 和保存的非参数存储。

这些子项可能重叠，不能重复相加。
保存存储只是下界，不代表全部激活内存。
只有在不增长区间内恒定的 SHMEM 预留，才可以加到该区间的 Torch 峰值上；
不同训练步的峰值不能相加。

如果要单独实验候选 plan，先生成校准数据：

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_benchmark_replica_calibration.py \
  --replica-transport shmem_signal_sdma_projection --result-dir /tmp/replica-calibration
```

然后为 CPU benchmark 传入 `--cost-model /tmp/replica-calibration/cost-model.json`，
在新进程中按 ABBA 顺序与未校准的 CPU 运行比较。

校准测量单专家前向/反向、完整权重拷贝、两个梯度投影和 host 配额优化开销。
逐专家相加的成本表包含 launch/control 成本，并近似估计链路竞争，
但不是完整调度的精确参考。

实际候选 plan、完整步耗时分布和容量变化应与单项传输机制对照分开报告。
模型评分降低本身不足以支持在训练中启用校准。

精度 worker 接受相同的 `--cost-model` 文件。
使用变化后的 plan 之前，先用 `--fp32-reference` 验证输出、dX、路由概率梯度和 owner dW。
参考实现不会接收候选校准数据。

对于全局 plan 没有传输的 MegaMoe B>0 调用，
前向和反向跳过 guest 池租约、guest 梯度清零及梯度返回。
调用仍保存自身不可变路由，并计算 FP32 home 权重 partial。

Runtime 保留现有 v2 home 专家寻址标志，guest 指针为空。
该标志还用于选择逐专家 GMM 寻址，不能省略。
是否使用这一快速路径由全局传输列表决定：
有发送方向传输的 owner 即使没有本地 guest 工作，也必须参与。

Profiling 关闭时，MegaMoe 还会在其串行 workspace 租约内复用完全相同的静态 v2 runtime 镜像。
缓存同时比较基础 tensor 身份与完整后缀字节，每个基础 tensor 只保留最新镜像，
并在当前 stream 上记录每次使用。

Workspace 完成事件保证跨 stream 访问顺序；
close 和堆重建在现有同步清理边界释放镜像存储。
这些镜像包含可写 worker 临时空间，不能在租约外复用，也不是全局不可变调度 tensor。

动态 v3/v4/v5 的 ready、epoch 和梯度返回描述符仍由每次调用独占。
Profiling 独立构造镜像；B=0 返回原有基础 runtime。
该优化保留现有原生 ABI。

为 benchmark 添加 `--host-breakdown`，可额外执行五个不计时的平衡/热点步，
仅记录 host 时间区间。
该模式不启用 Torch/内部 profiling，不插入逐阶段设备 event，也不单独同步各区间。
它报告 runtime 基础/镜像大小及观测到的缓存命中。

Host 区间仍可能包含已排队设备工作的提交等待，也可能嵌套，
性能判断仍以完整步 ABBA 为准。
`replica_runtime_image_bytes` 是与其他 Torch 内存统计重叠的子项，不能再次加到已记录峰值中。

`tests/torch/expert_parallel/test_hot_replica.py` 中的四 rank 启动入口
`test_push_static_runtime_streams` 和 `test_pull_static_runtime_streams`
检查两个不同的保留 plan、逆序反向，
以及前向和反向各自在两个真实 NPU stream 上复用同一个 v2 镜像。

它们将输出、dX、路由概率梯度和 owner 权重梯度与 FP32 native 参考比较。
如果要覆盖纯 home v2 路径的 profiler，给 benchmark 添加
`--diagnose --diagnose-pattern balanced`；
默认诊断 pattern 仍为 `home_hot`。

副本计数交换使用 `all_gather_into_tensor`，
填充由当前调用独占、按 rank 排列的连续缓冲区。
MegaMoe 在每行中包含集合路由校验标志；native gather 现有逻辑计数。
两者均在使用结果前等待异步集合操作完成，
EP=1 则拷贝本地载荷，不创建集合通信。

CPU 和设备规划器消费相同布局，无需 stack 多个独立 rank tensor。
专家放置、配额、错误校验和现有设备控制摘要回读不变。
Native 继续通过 HCCL P2P 传输专家权重和梯度。

轻量 benchmark 观测器还报告计数交换前路由载荷的 `validation_and_counts`。
该 host 区间覆盖 ID 校验、重复检测和逻辑直方图，也可能包含队列等待，
不能与嵌套区间相加推算完整步延迟。

启动入口 `test_push_projection_runtime_lifetime` 和
`test_pull_projection_runtime_lifetime`
在两个真实 NPU stream 上保留 v4 镜像，并逆序执行反向。
每个方向都必须使用 epoch 不同的独立镜像。

每个消费者执行后取得的设备尾部快照必须与该调用的打包字节一致，
输出与梯度检查使用 FP32 native 参考。
动态 v4 镜像仍独立构造。
