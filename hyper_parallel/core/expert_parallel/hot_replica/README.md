# 容量受限的专家热副本

热点专家可以临时复制到其他 EP rank 的 B 个执行槽位。原 rank 继续拥有参数和优化器状态，
副本仅借用权重，反向将梯度以 FP32 归还 owner。每次调用使用当前路由生成计划，
不缓存跨调用的专家权重。B=0 使用原执行路径。

## 使用入口

Native GroupedExperts 通过现有 EP hooks 接入，使用 HCCL P2P，不依赖 multicore 或 SHMEM：

```python
from hyper_parallel.core.expert_parallel import ExpertParallel

style = ExpertParallel(replica_slots_per_rank=1, replica_planner="cpu")
style.apply(experts, ep_mesh)
```

MegaMoe 的 push/pull 共享规划器，支持两种副本传输：

| `replica_transport` | 权重与梯度执行 |
| --- | --- |
| `p2p`（默认） | HCCL P2P 权重复制，按 peer 顺序归还 FP32 梯度 |
| `shmem_signal_kernel_gradient` | 按 peer 并行 SDMA 预取，分投影 ready，反向内核归还 FP32 梯度 |

```python
from hyper_parallel.core.multicore import MegaMoeExperts

module = MegaMoeExperts(
    local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
    num_experts=24, top_k=8, ep_size=4, ep_group=ep_group,
    dispatch_mode="push", replica_slots_per_rank=1,
    replica_transport="shmem_signal_kernel_gradient",
    replica_planner="cpu",
)
output = module(values, topk_ids, topk_weights)
output.backward(gradient)
module.close()
```

由调用者提供参数时，设置 `create_parameters=False`，每次 forward 传入
`expert_weights=(gate_up_weight, down_weight)`。本地矩阵布局分别为 `[H, hidden, 2*intermediate]`
和 `[H, intermediate, hidden]`，H 是每个 rank 拥有的逻辑专家数。副本不注册为参数。
Multicore 的构建、载荷激活和 SHMEM 生命周期见
[构建说明](../../multicore/docs/build.md)和 [MegaMoe 入口](../../multicore/README.md)。

## 规划、容量与收益准入

逻辑专家数 E 必须能被 EP rank 数 P 整除，H=E/P。每个 rank 固定 H 个 home 槽位和 B 个 guest
槽位；物理专家数为 P*(H+B)，空槽位标记为 -1。合法路由要求每个 token 的 TopK 专家互不重复，
且满足 1<=K<=E。路由错误在稀疏传输之前由所有 rank 一致拒绝。

规划器汇总各 source 的整数计数，生成确定性的放置和配额。稳定重映射保留每个 TopK 位置、
源顺序和总行数，既不丢 token，也不执行 multi-wave。`replica_target_load` 是期望接收负载，
独立于缓冲区容量；push 未设置时使用当前容量，pull 未设置时使用平均负载。

`ExpertReplicaConfig.maximum_receive_rows(S, K)` 给出有界接收容量。B=0 的未对齐上界为
P*S*min(K,H)，B>=H 时为 S*K；其余预算按 home/guest 混合界和整数截断余量计算。
push 按实际路由需求增长并受此界约束，pull 继续使用路由 source 布局。
`initial_capacity_factor`、`capacity_growth_factor` 仅用于 push，不能作为丢弃阈值。

常用选项：

| 参数 | 约束与行为 |
| --- | --- |
| `replica_slots_per_rank` | 非负整数 B；每个 EP rank 使用相同预算 |
| `replica_planner` | `cpu` 或 `device`；默认 `cpu` |
| `replica_min_rows` | 小副本软阈值；Native 默认 0，MegaMoe 默认 1024；容量必需副本继续保留 |
| `replica_target_load` | 可选正整数；所有 EP rank 一致 |
| `replica_minimum_load_reduction` | `None` 要求当前路由峰值下降至少 40%；`0.0` 关闭可选副本收益门槛 |
| `replica_cost_model` | 可选离线成本模型；要求 CPU 规划器且匹配执行配置 |

收益门槛只控制可选副本。若理想平均负载也不能达到门槛，规划器直接使用 home 计划；
求解后再次检查最终峰值，容量必需的复制继续保留。低层规划 API 的
`minimum_load_reduction` 和 `minimum_saved_rows` 默认均为零，分别表示相对与绝对收益门槛。
空副本计划使用紧凑 home 拓扑，下一次调用可以重新启用副本。

## 传输与调用生命周期

`p2p` 在使用 guest 权重之前等待发送/接收完成。guest 梯度按 peer 顺序加到 FP32 owner
缓冲区，通信存储受 B 槽位限制；没有 remote 工作的 rank 仍参与一致的调用顺序。

`shmem_signal_kernel_gradient` 在一个独占 provider lease 内执行以下协议：

1. receiver 在上一轮消费者完成后发放槽位 credit；owner 完成 credit 等待后启动 SDMA。
2. forward 先复制 W13 再复制 W2，backward 反序复制；每个投影分别发布 ready。
3. GMM 仅等待所消费投影的 ready，允许 home 计算与 guest 复制重叠。
4. 反向等待完整投影生产 fence，归还 W2/W13 FP32 梯度，并等待所有 remote reader 的 ACK。
5. 所有复制流 join 到调用流后才释放 lease。epoch 回卷前由所有 rank 同步清空信号。

缺少完整投影 fence 的调度使用内部 SDMA 梯度回退；它不额外暴露传输模式。
同一调用的 backward 可以复用 forward 的权重代际，但必须先通过 autograd 参数版本校验，
且该代际仍在原槽位中。其他 forward、provider/heap 重建和 epoch 回卷都会使复用失效。
延迟 backward 持有自己的路由、参数版本和中间张量，不能读取已被下一层覆盖的 workspace。

共享 execution resources 需要相同拓扑、传输、shape、dtype 和 tile splits。串行 lease 与事件
约束跨流重用；同一个 provider 不允许并发 host lease。关闭模块之前应完成或释放所有保留的图。
底层 `return_gradients(..., consume=True)` 仅接受调用独占的 detached FP32 owner 缓冲区，
并要求调用者停止使用原值；普通借用输入通过复制保留。

## 设备规划器与融合计数

设置 `replica_planner="device"` 可使用可选 Ascend 910B 求解器。它保留与 CPU 相同的整数策略、
准入条件、容量约束和配额守恒；路由配额留在设备，host 仅复制一次紧凑控制摘要。
因此不应将该入口解释为全路径零 readback。设备规划器不支持离线成本微调。

激活匹配的 CANN 环境后，在仓库根目录构建可选库：

```bash
cmake -S hyper_parallel/core/expert_parallel/hot_replica/_device_kernel -B build/replica-planner
cmake --build build/replica-planner -j 4
cmake --install build/replica-planner \
  --prefix "$PWD/hyper_parallel/core/expert_parallel/hot_replica/_device_kernel"
```

规划器 ABI 为 4；ABI 不匹配明确报错。未安装规划器时，显式 device 入口报错。
融合计数库为可选加速：不支持的 shape/device 或未安装时使用 Torch 整数计数，
重复专家和越界标记随计数一起参加 collective。Native 接收已有专家计数，不重建 TopK。

打包时将可选库安装至 native payload 的 `core/expert_parallel/hot_replica/_device_kernel`，
通过 `HYPER_PARALLEL_NATIVE_OUTPUT_ROOT` 指向该载荷根目录；与 multicore 组合时使用同一个根目录。

## 离线成本模型

`ExpertReplicaCostModel` 在容量规划之后微调已有副本配额。模型记录 shape、EP、backend、
transport、schedule 和 FP32 partial dtype，拒绝配置不匹配或旧 schema。校准范围之外保留
原策略；没有实测依据时不扣除假设的 home overlap。两轮确定性搜索保留 B 预算和 source 守恒。
模型的 rank score 缓存只存在于一次求解内，不代表权重驻留缓存。

离线实测数据由调用方提供；所有 EP rank 应加载同一个文件，再传入
`ExpertReplicaCostModel(**payload["model"])`。模型的预测需要用完整步骤基准验证。

## 验证与基准

CPU 回归验证计划守恒、准入、路由、transport 协议、生命周期及调度/ABI：

```bash
python3 -m pytest -q tests/ut/core/expert_parallel tests/ut/core/multicore
```

NPU 系统测试通过轻量 launcher 启动四个 rank，覆盖两种最终传输、两种规划器、push/pull、
数值与 SGD 状态、逆序 backward、跨流、heap 增长及 collective 路由校验。
实际 case 矩阵与命令见 [专家并行测试](../../../../tests/torch/expert_parallel/README.md)。

完整步骤基准只保留 `tests/torch/expert_parallel/_benchmark_hot_replica.py`：

```bash
torchrun --nproc-per-node=4 tests/torch/expert_parallel/_benchmark_hot_replica.py \
  --backend push --replica-transport shmem_signal_kernel_gradient --replica-planner cpu \
  --budget 1 --route hot --checkpoint --result-dir /tmp/replica-benchmark
```

该入口测量 warmed F+B+AdamW，支持 balanced/hot/rotating_hot，cold 和 plan audit 独立于计时样本。
checkpoint 使用 reentrant 完整重算，取证记录有梯度的 forward 所用计划，允许 push 容量增长后重新规划。
每个 rank 输出配置、source/payload identity、实际计划、local/rank-max 样本和峰值内存。
对照应保持 route、seed、shape、checkpoint 和资源所有权一致，使用新进程 A/B/B/A；
诊断或单算子耗时不能替代完整步骤收益，CPU 回归和 build/load 也不能代替 NPU 数值验收。

## 按行带重叠梯度返回

`shmem_signal_kernel_gradient_bands` 使用 Cube 直接发布行带就绪信号，
使 W2/W13 梯度可以在整专家计算结束之前返回。启用此模式前需要重新构建 multicore 原生载荷。
行带按 FP32 矩阵形状和 4 MiB 目标大小划分，不根据路由 skew、seed 或副本预算选择策略。
每个 Cube 只在该行带全部 K 维累加与最终 GM 写回完成后发布信号；
没有分配到输出 tile 的 Cube 和零行专家也必须发布完成信号。

返回侧按实际 worker 数和行带数划分最多四个独立 AIV 组。
一组等待某个 peer 时，其他组可以推进自己的行带。
每个元素仍按原目标 rank 顺序累加 FP32 梯度，不改变路由、副本配额和数值精度。
两个投影均完成所有读取后，在最终 ACK 汇合处释放 guest 租约。

对称的 guest 信号表直接放入已有 replica inbox，纳入构造阶段的 SHMEM 堆预算，
随 inbox 一起释放，在 epoch 回绕时同步清零；普通 owner 信号表保持调用内独占并记录 stream 生命周期。
无法证明两个投影的生产调度、行带数超过 64 或 Cube 数超过预留的 32 时，保留完整专家返回。
B=0 和没有实际副本搬运的路由沿用原有执行路径。
