# 专家并行组合测试

本目录包含分布式测试，用于验证基础专家并行（EP）与其他并行策略（DP、TP、CP）的组合行为。

## 测试目标

验证 ExpertParallel 和 ExpertTensorParallel 与数据并行（FSDP）、张量并行和上下文并行组合时的正确性。

## 验证方法

所有测试遵循相同流程：

1. 构建不使用并行策略的独立 MoE 模型作为参考。
2. 构建应用目标并行策略的 MoE 模型。
3. 使用相同输入，分别执行前向和反向。
4. 在容差范围内比较输出与梯度。

## 测试覆盖

| 模板名称 | dp | ep | tp | cp | 最少卡数 | 验证范围 |
| --- | --- | --- | --- | --- | --- | --- |
| ep-only | 1 | 2 | 1 | 1 | 2 | 基础 EP 功能 |
| tp-only | 1 | 1 | 2 | 1 | 2 | 专家内部 TP |
| dp-ep | 2 | 2 | 1 | 1 | 4 | DP 与 EP 组合 |
| ep-tp | 1 | 2 | 2 | 1 | 4 | EP 与 TP 组合 |
| dp-ep-tp | 2 | 2 | 2 | 1 | 8 | DP、EP、TP 三维组合 |
| dp-ep-cp | 2 | 2 | 1 | 2 | 8 | EP 与 CP 维度兼容性；仅验证 EP 能在包含 CP 维度的 mesh 上工作，不验证 CP 通信 |
| dp-ep-cp-with-attention | 2 | 2 | 1 | 2 | 8 | 使用 Self-Attention 与 MoE 模块验证实际 CP 通信 |

### 两卡可运行用例（共 32 个）

| 测试组 | 数量 | 参数变化 |
| --- | --- | --- |
| 基础 EP | 6 | num_experts：2/4/8；top_k：1/2 |
| EP + grouped_mm | 6 | num_experts：2/4/8；top_k：1/2 |
| EP + 共享专家 | 6 | num_experts：2/4/8；top_k：1/2 |
| 仅 TP | 8 | num_experts：2/4；top_k：1/2；hidden_dim：64/128 |
| 配置校验 | 6 | 合法及非法 mesh 配置 |

## 运行测试

### 两卡环境

```bash
pytest tests/torch/expert_parallel/test_combinations.py -v -k "test_2card_group"
```

## 约束

| 约束 | 说明 |
| --- | --- |
| `num_experts % ep == 0` | 专家数必须能被 EP rank 数整除 |
| `hidden_dim % tp == 0` | 隐藏维度必须能被 TP 并行度整除 |
| `dp % ep == 0`（`dp > 1` 时） | 数据并行度必须能被 EP 并行度整除 |
| `dp * ep * tp * cp == world_size` | 各维度乘积必须等于总设备数 |
| `sequence_length % cp == 0` | CP attention 测试要求序列长度能被 CP 并行度整除 |
| CP 支持范围 | 普通 CP 组合只验证维度兼容性；完整 CP 通信由包含 attention 的用例验证 |

## Dirichlet 热副本扫描

`dirichlet_routes.py` 保留 `megamoe-sun` 中带固定 seed 的 Dirichlet 采样、注水分配
及互不重复的 TopK 构造方法。所有对照系列按相同的打乱顺序重放原有 55 组 alpha/seed。
较小的 alpha 通常产生更大的不均衡，但应比较实际 home 目标负载的 skew，
不能假设 alpha 唯一决定 rank 负载。Skew 在副本分配前统计。

Worker `_benchmark_dirichlet_replica.py` 对比 native 分组专家和 MegaMoe，
扫描默认选择每个 rank 的副本槽位数 B=0 和 B=1。
默认配置为 EP16/E96、16 个独立 MoE 层、每 rank 4096 tokens、H5120/I1792、TopK8 和 BF16。
每步执行所有 checkpoint 前向、逆序反向及完整 AdamW 更新，范围不包含 attention 或 router 网络。
两个 B=1 系列默认使用当前 CPU 规划器和 HCCL P2P 传输。
MegaMoe 使用 push，初始容量因子为 1.5，并保留当前自动增长策略。
各层共享执行存储和有界副本池，路由变化时保留内存高水位。
Native 在现有分组专家 EP 入口前后保留原始 NPU token-permute 和带权重的 token-unpermute 算子。

启动 Python 前，激活 CANN 和当前 checkout 的原生载荷，
并确认 editable 安装指向当前 checkout。
选择全部 16 个设备，在现有设备空闲等待机制下运行：

```bash
python scripts/run_dirichlet_replica_sweep.py --output /path/to/smoke --phase smoke
python scripts/run_dirichlet_replica_sweep.py --output /path/to/accept --phase accept
python scripts/run_dirichlet_replica_sweep.py --output /path/to/sweep --phase sweep
python scripts/plot_dirichlet_replica_sweep.py \
    --input /path/to/sweep --output /path/to/dirichlet_curves
```

Smoke 在两个负载分布下检查两个小型 MoE 层。
验收在生产尺寸上，将均衡、中度不均衡和高度不均衡路由的输出、输入及路由梯度、
三个专家权重梯度与 native B=0 比较，rank-max relative L2 门槛为 1%。
验收与计时运行分别进行。
扫描对每个系列启动两个新进程，第二轮按相反系列顺序运行；
每条路由预热 3 步、测量 7 步。
源码、原生载荷、初始权重、输入及路由 hash 与所有 rank 的结果一同记录。
`--phase full` 依次执行验收、扫描和绘图。
`--resume` 仅保留物理监控干净、源码及原生载荷 hash 仍匹配的完整运行，
重试前会归档无效的运行记录。

仅选择 MegaMoe B=1 及其 kernel-gradient 副本传输模式：

```bash
python scripts/run_dirichlet_replica_sweep.py --output /path/to/kernel_gradient \
    --phase full --variants megamoe-b1 --replica-transport shmem_signal_kernel_gradient
```

单独选择一个系列时，仍执行验收和两次独立扫描。
只有选择全部四个系列时，才自动生成四系列图。
复用旧基线需要核对模型及 runtime 源码、原生载荷、初始权重、输入、路由和测量边界。
Worker 分别记录请求的传输模式、实际 provider 及 W2/W13 ready 状态。

针对少量点的对照，worker 支持 `--pairs` 和 `--replica-target-load`。
后者以每个 rank 接收的行数设置 MegaMoe 规划器目标，独立于 push 缓冲区的初始容量。
例如，`--replica-target-load 32768` 将 EP16/E96 的路由目标对齐到平均负载，
同时保留初始容量因子 1.5。默认 push 规划仍根据当前容量设置目标。
比较的系列应使用相同的点及顺序重新运行，以保持权重更新历史一致。

Worker 还支持任意非负 `--budget`，以及 MegaMoe 的 `--replica-min-rows`（默认 1024）。
比较新的小副本策略时，可设为零，恢复此前按 token 数平衡的放置策略。
Native 保留原有默认阈值零。
传输证据记录实际执行专家数、活动副本预算和权重梯度 dtype；
全局 guest plan 为空时可执行普通图，同时保留配置的传输能力供后续热点使用。

`--diagnose` 在计时结束后，单独采集一个 checkpoint MoE 层的 Torch/NPU 和内部 MegaKernel trace。
记录原始前向、重计算、反向、实际放置、provider 标志及包含等待的 host 时间区间。
这些启用 profiler 的 trace 用于诊断重叠和 ready 等待，
不能替代关闭 profiler 时的 16 层训练步样本。

延迟取 rank-max 训练步耗时的中位数。
Accounted HBM 为 rank-max allocator 峰值加完整的外部 SHMEM 堆，包含优化器状态；
allocator 预留内存与采样物理卡/进程 HBM 单独记录。
物理监控记录 PID 及其 launcher 祖先关系。
绘图要求所有 rank 的运行完整且配置匹配，并拒绝存在外部进程、设备异常或缺少物理采样的记录。
最终生成 CSV、JSON、PNG 和 SVG 产物。
