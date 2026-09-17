# DeepSeek-V4.1 MegaMoe block 精度实施记录

## 范围

已实现显式 `swiglu_limit=0`、routed/shared 无 clamp 执行、独立 MegaMoe block adapter、
CPU FP32 oracle 和 NPU 精度入口。当前是 H=512、I=128、E=8 的随机权重 block 验证，
不代表完整 DSV4.1 Trainer、FSDP、checkpoint 或训练收敛已经验收。

HF 5.13.0 的零值仍会 clamp 到零。因此只修改 config 不足够；本实现对零值实例绑定
无 clamp 方法，同时保留 limit=10 的 HF 方法和参数身份。MegaMoe 对非零 limit 明确报错。
模型构造入口记录源 limit，并保留原配置文件。

## 环境及复现

- 分支：`feat/deepseek-v4.1-megamoe`；验证基于 `80fb24f4` 加本次工作区改动。
  每个结果 JSON 的 `source_files_sha256` 记录当次源码，包括未提交的新文件。
- Torch `2.9.1+cpu` + torch-npu `2.9.1`、Transformers `5.13.0`、torchdata `0.11.0`。
  Transformers 使用隔离环境，HyperParallel editable import 指向当前 checkout。
- CANN `9.1.0`、Ascend910B3；native payload 从当前分支重新编译。
  编译阶段避开本机不可读的其他 vendor 配置；运行阶段使用原始 CANN OPP 路径。
- Torch adapter SHA256：`35d6923f5cbe280eb60e00d44a37248e7042bda7b439d09874c9f3be63ad416a`。
  vendor opapi SHA256：`a49b453c8c7e25323acdc71294aed33ae27545c4ff85ee8ec33af89dfbc9be3e`。
- 每 rank 128 tokens；H=512、I=128、E=8、默认 TopK=2；BF16、limit=0、lossless capacity。
  `sqrtsoftplus`、routed scaling=1.7；权重种子 2026，输入种子 `4000 + global_rank`。
  连续三步 SGD，lr=0.01；每 rank 输入及外部 loss derivative 不同。
- 按用户要求允许 busy NPU，保留进程占用记录。本报告没有性能结论。

运行说明见 [precision example](../examples/mega_moe/deepseek_v41_precision.md)。
矩阵包括 EP1/EP2/EP4 的 learned/hotspot × push/pull（12 组），
WORLD4 下两个 strided EP2 subgroup 的视觉路由 × push/pull（2 组），
以及 EP2 pull 的 TopK=1/8（2 组）。

## 独立 FP32 oracle

Oracle 使用 CPU FP32 matmul、SiLU、score function 和 autograd，既不调用 HF expert，
也不调用 native kernel。它读取每条路径当步的权重和输入，固定实际离散 expert IDs，
重新计算可微路由权重。Hotspot 使用独立 route-weight leaves。

每 rank 的 FP32 expert dW 在真实 EP group 内以 FP32 归约，然后按 group-local rank
映射到 native 本地布局。检查 output、dX、route dW、全部参数 dW，以及同状态下一次 SGD 更新。
shared/router 梯度保持 rank-local，与当前 block baseline 的合同一致。
这是逐步数值 oracle，不是一条额外的、连续三步 FP32 master-weight 训练轨迹。

## 独立更新轨迹的 16 组结果

每个张量分别计算相对 L2 和峰值归一化误差，表中取所有 rank/step/tensor 的最大值。
峰值归一化误差为 `max(abs(actual-reference)) / max(abs(reference))`。

| 路径 | 对 FP32 的最坏相对 L2 | 对 FP32 的最坏峰值归一化误差 |
| --- | --- | --- |
| HF BF16 | 0.5482% | 1.1716% |
| MegaMoe | 0.5177% | 0.9225% |

原 `rtol=0.02, atol=0.002` HF BF16 逐元素硬门槛仅通过 2/16 组（EP1 learned push/pull）。
FP32 复核也能在**未改动的 shared experts** 中复现少量同阈值失败，说明不能把所有
逐元素超界直接归为 MegaMoe 集成错误；同样不能仅凭整体 L2 较小就隐藏局部误差。
完整 JSON 保留超界元素数、最坏位置的 actual/reference、相对容差比和每个张量结果。

两条路径独立更新后，EP4 learned 的第 3 步出现 TopK 分歧：push 在 rank2，
pull 在 rank2/rank3。pull rank3 的跨路径输出相对 L2 为 6.23%，dX 为 5.50%。
这组结果不能宣称多步轨迹一致。各自相对当前权重对应的 FP32 oracle 仍接近，
需要与同状态比较区分；当前报告保留这项失败，不将其归入 kernel 舍入误差后忽略。

## 同状态比较

`--synchronize-step-weights` 在第 2/3 步开始前，把 HF 当前权重按本地 expert slice/layout
拷入候选路径。第 1 步首先精确检查原始构造时的映射，不用同步掩盖转换问题。
每次同步前，上一轮 native optimizer 更新已经完成并逐参数检查。
每步再次要求初始参数精确相等，保留真实 learned router 和独立 forward/backward。
该模式用于算子及参数更新验证，不宣称两条独立训练轨迹相同。

16/16 组完成，所有 rank/step 的初始权重、路由 IDs 和梯度存在性精确匹配。
原 BF16 逐元素硬门槛通过 5/16 组，其余仍如实报告失败。
同状态下 MegaMoe 对 FP32 的最坏相对 L2 为 **0.5177%**，峰值归一化误差为 **1.0054%**；
HF BF16 对 FP32 分别为 **0.5482%** 和 **1.1716%**。

| 配置 | 原 BF16 门槛 | HF / FP32 L2 | MegaMoe / FP32 L2 | HF 峰值误差 | MegaMoe 峰值误差 |
| --- | --- | --- | --- | --- | --- |
| aligned-ep2-pull-topk1 | 失败 | 0.5176% | 0.4938% | 1.1716% | 0.7592% |
| aligned-ep2-pull-topk8 | 通过 | 0.5482% | 0.5147% | 0.9741% | 0.8279% |
| aligned-subgroup-pull-vision | 失败 | 0.5315% | 0.5152% | 0.9712% | 1.0054% |
| aligned-subgroup-push-vision | 失败 | 0.5315% | 0.5152% | 0.9712% | 1.0054% |
| ep1-pull-hotspot-aligned | 通过 | 0.5039% | 0.4852% | 0.7009% | 0.6580% |
| ep1-pull-learned-aligned | 通过 | 0.5294% | 0.5177% | 0.9334% | 0.9234% |
| ep1-push-hotspot-aligned | 通过 | 0.5039% | 0.4852% | 0.7009% | 0.6580% |
| ep1-push-learned-aligned | 通过 | 0.5294% | 0.5177% | 0.9334% | 0.9234% |
| ep2-pull-hotspot-aligned | 失败 | 0.5067% | 0.4863% | 0.7584% | 0.6865% |
| ep2-pull-learned-aligned | 失败 | 0.5309% | 0.5177% | 0.9238% | 0.8309% |
| ep2-push-hotspot-aligned | 失败 | 0.5067% | 0.4863% | 0.7584% | 0.6865% |
| ep2-push-learned-aligned | 失败 | 0.5309% | 0.5177% | 0.9238% | 0.8309% |
| ep4-pull-hotspot-aligned | 失败 | 0.5212% | 0.4896% | 0.9907% | 0.7894% |
| ep4-pull-learned-aligned | 失败 | 0.5309% | 0.5177% | 0.9113% | 0.8752% |
| ep4-push-hotspot-aligned | 失败 | 0.5221% | 0.4896% | 0.8029% | 0.7616% |
| ep4-push-learned-aligned | 失败 | 0.5309% | 0.5177% | 0.9724% | 0.9724% |

## 已确认的工程验收口径（2026-09-17）

经用户确认，将**同状态** FP32 oracle 用于数值验收，同时保留 HF BF16 逐元素对比及独立轨迹诊断。
对每个 rank、step、tensor 单独检查，不能跨张量平均后掩盖失败：

1. 所有值有限，参数初始映射、同状态路由 IDs、梯度存在性及张量形状精确匹配。
2. 对 FP32 的相对 L2 不超过 1%。
3. 对 FP32 的峰值归一化最大绝对误差不超过 2%；全零 reference 要求 actual 也全零。
4. 对 HF BF16 baseline 施加相同 FP32 标准；保留原逐元素超界明细。

这是针对 BF16 block 的工程回归判据，不是理论误差上界，也不是完整模型收敛标准。
默认 `--acceptance fp32` 自动开启 oracle 和每步权重同步；`--acceptance hf_bf16` 保留原硬门槛。
JSON 分别记录新判据、旧逐元素门槛和精确匹配项，历史日志不回写。
完整四层 crop 的 logits/loss、多步训练和 checkpoint 属于下一阶段。

## CPU 与静态检查

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/auto_models/models/deepseek_v41 tests/ut/core/multicore
```

结果：`156 passed, 610 subtests passed`。覆盖 limit=0 的大幅值输出/梯度、limit=10 保留、
真实四层模型构造、源配置不变、HF text / V4.1 text / visual router、EP slice/layout、
三步更新、独立 FP32 oracle，以及既有 Multicore 回归。

静态检查沿用现有规则。两个已有函数仍超过 lizard 阈值：配置构造函数从 CCN20/NLOC150
变为 CCN21/NLOC156；模型 forward 保持 CCN19/NLOC93。此轮不重构无关的模型控制流。
代码未修改通用 optimizer，也未恢复已删除的 `test_parameter_copy.py`。

后续工作见 [适配计划](deepseek_v41_adaptation_plan.md)。

## 算子级追踪

后续以 `c126ee4c` 加诊断改动为源码基线，native 二进制保持上述 SHA256。
实际 scratch、相同输入的单算子 FP32 对照、融合激活替换及 EP 梯度归约对照见
[算子精度定位报告](deepseek_v41_operator_precision_report.md)。
