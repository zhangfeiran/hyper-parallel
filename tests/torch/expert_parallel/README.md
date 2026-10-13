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

## 热副本验收

实现与约束见 [热副本说明](../../../hyper_parallel/core/expert_parallel/hot_replica/README.md)。
`test_hot_replica.py` 保持无框架导入，通过 `torchrun_case` 启动四个 rank：

| 参数矩阵 | 验收内容 |
| --- | --- |
| Native × CPU/device × P2P | 普通 EP hooks、FP32 参考、TopK 边界和逆序 backward |
| push/pull × CPU/device × P2P/kernel_gradient | 输出、dx、router dW、专家 dW、SGD 参数及 momentum |
| 同一训练 case 内 | balanced/hot 切换、保留两个 forward、跨层和跨流重用、push heap 增长 |
| count/route/device boundaries | 精确 int64 计数、无效路由 collective 拒绝、设备求解与 CPU 对齐 |

```bash
pytest -v tests/torch/expert_parallel/test_hot_replica.py
# 单独运行 CPU 规划、push 和内核梯度返回
pytest -v tests/torch/expert_parallel/test_hot_replica.py \
  -k 'push_cpu_shmem_signal_kernel_gradient'
```

worker 在 `HP_HOT_REPLICA_RESULTS`（默认 `./logs/hot_replica`）下按 case、TopK 和 rank 保存
实际 plan、数值误差、执行/载荷 identity 和生命周期证据。主动复制夹具显式关闭收益门槛，
避免 balanced 的默认 home 计划绕过待验证协议。

性能入口各保留一个：`_benchmark_hot_replica.py` 测量 warmed F+B+AdamW，
`_benchmark_replica_calibration.py` 生成离线模型。参数、范围与完整步骤比较方法见热副本说明。
