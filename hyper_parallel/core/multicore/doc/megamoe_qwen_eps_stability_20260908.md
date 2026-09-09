# Qwen AdamW eps 控制与其他 trainer 用法对照

接续 [expert 参数干预](megamoe_qwen_parameter_intervention_20260908.md)，基于
`megamoe-sun@0681a29b`。生产 Python / native 相对 `ee5a6965` 未变。本轮只做诊断，
不改变默认 eps、LR、梯度归约、容差或 replica / FP32 main 合同。

结论：**增大 eps 不能据本轮结果作为精度修复。** 四档真实梯度复放中，FP32 main 最大差
下降，但 BF16 model 的最大差、不同元素数、RMSE 没有同时改善；`eps=1e-7` 的完整模型
仍在第二步 logits、第六步 loss 超原阈值。其他 trainer 使用相同的 core fused AdamW，
主要差异是优化器选择、LR、分组、调度、裁剪和梯度归约，不是另一套特殊 AdamW 公式。

## 1. 保持不变的实验合同

硬件/运行时沿用前几轮：EP8、Ascend 910B3、Torch/TorchNPU 2.9.1、CANN 9.2.0，
editable import 指向当前 checkout，native payload 仍为当前源码构建产物。初始权重、
batch、配置及 source/native identity 均链接到此前确定性控制的指纹。

原 example 为八层 connected Qwen model，H5120 / I1792、E48 / K8、每 rank 4096 tokens，
Dirichlet 0.1 固定路由、shared-B1、capacity factor 1.5。BF16 model、FP32 main/moments，
AdamW 的 LR=1e-3、weight decay=0.01、betas=(0.9, 0.999)。确定性模式仅是诊断控制。
原 `rtol=0.02 / atol=0.002` 保留，归一化误差大于 1 表示超过此阈值。

通过临时、局部 optimizer factory 覆盖只改变 eps，并检查实际 leaf defaults 和每个
param group；不是修改报告标签而继续运行原 eps，也不是用 CPU optimizer 替代设备更新。
所有临时脚本、JSON、张量只保存于 `/tmp/qwen-eps-stability.iUEwDO/`。

## 2. 四档 eps：固定真实梯度、独立首步更新

使用此前实际第八层的初始权重、native 梯度和 B1 梯度，每档分别运行：native layout +
native gradient、MegaMoe layout + 同一 native gradient、MegaMoe layout + 实际 B1 gradient。
每组重新创建 model/optimizer，仅做一次原 fused AdamW 更新，不运行新的 F+B。
四档 × 三组 × 八 rank，共 96 个独立 rank-local optimizer 实例/更新。

全量比较覆盖该层 1321205760 个逻辑权重。相同梯度的两种布局，在全部 eps 下的 main、
两个 moments、BF16 model 均逐位一致；eps=1e-8 的全部状态指纹复现上一轮记录。
梯度转 FP32、group step=1、BF16 直接 cast 和梯度消费均符合原合同。同一 eps 下 B1 对
native 的状态差异始终限定在输入梯度不同的 40566 个坐标内；其他位置没有新增状态差异。
三组各自固定梯度的完整 moments 指纹均不随 eps 改变。

### 2.1 跨 backend 残差

下表均为 B1 相对同 eps 的 native，元素数跨八 rank 求和，最大差取全量最大值；
RMSE 的分母为全层权重元素数，不是仅取差异位置。

| eps | FP32 main 不同元素 | main 最大差 | main RMSE | BF16 model 不同元素 | model 最大差 | model RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-8 | 40245 | 6.493181e-6 | 4.102406e-9 | 738 | 1.220703e-4 | 3.228289e-8 |
| 1e-7 | 40566 | 1.931563e-6 | 3.723977e-9 | 826 | 1.220703e-4 | 3.307562e-8 |
| 1e-6 | 40513 | 1.944602e-6 | 3.709006e-9 | 815 | 1.220703e-4 | 3.501729e-8 |
| 1e-5 | 39098 | 1.912005e-6 | 3.688080e-9 | 763 | 1.220703e-4 | 3.074808e-8 |

例如 1e-7 把 main 最大差降低约 70.3%，但 BF16 不同元素从 738 增到 826，RMSE 也略增。
1e-5 的 BF16 RMSE 下降约 4.8%，但最大差未变、不同元素数仍高于原配置。
不能只根据 main 最大差选择所谓“精度更好”的 eps：BF16 舍入边界的位置也随正常更新改变。

独立 CPU FP64 复核全部 40566 个梯度差异坐标，实际 main 相对首步理想公式
`w * (1 - lr * wd) - lr * g / (abs(g) + eps)` 的最大误差均不超过 2.291e-9。
不同 eps 下没有出现新的更新公式、布局或写回错误证据。

### 2.2 增大 eps 同时改变正常更新

以下比较同一 native 梯度下，新 eps 的首步更新与原 eps=1e-8 更新；它不是 backend 残差。

| 新 eps | native main 不同元素 | main 最大差 | main RMSE | native BF16 model 不同元素 | model 最大差 | model RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-7 | 770703350 | 5.194941e-4 | 9.291866e-5 | 323327282 | 6.103516e-4 | 9.374245e-5 |
| 1e-6 | 770703350 | 8.181827e-4 | 2.798714e-4 | 645727012 | 8.544922e-4 | 2.779793e-4 |
| 1e-5 | 770703350 | 9.386940e-4 | 5.238589e-4 | 770600102 | 9.765625e-4 | 5.191292e-4 |

改变正常更新的幅度远大于本次 backend 残差。因此在完整训练前明确选择新档中改动最小的
1e-7 做动态控制，不根据后续 loss 事后挑档，也不推荐把更大 eps 直接设为默认值。

## 3. 完整模型十步：eps=1e-7 未消除分歧

使用原 `_optimizer_step`，所有参数组统一设为 eps=1e-7，不只修改末层 expert；每步都执行
真实 F+B、原 BF16 dense all-reduce、FP32-main AdamW 和 BF16 写回。其余合同不变。
按 native A1 / MegaMoe B1 / MegaMoe B2 / native A2 顺序独立驻留，避免双套设备训练状态。

A1 导出十步完整 loss/logits，以及首两步全量 83 个 canonical 梯度/更新后参数。后三组
按原阈值比较；每组另外记录全部阶段指纹和末步完整 model/main/moments 指纹。
首步 forward 与历史 eps=1e-8 逐位相同，native 首步全部梯度也不变，符合 eps 尚未参与更新。

首两步全量梯度与更新后参数仍通过原 tensor 阈值，但这不阻止下一次 logits 失败：

| 步数/阶段 | 不同张量数（跨 rank 名称并集 / 83） | 最大绝对差 | 最大归一化误差 | 原 gate |
| --- | ---: | ---: | ---: | --- |
| 1 / gradients | 16 | 9.536743e-7 | 0.000476256 | 通过 |
| 1 / updated parameters | 16 | 1.220703e-4 | 0.0527872 | 通过 |
| 2 / gradients | 83 | 3.051758e-5 | 0.0149530 | 通过 |
| 2 / updated parameters | 67 | 3.906250e-3 | 0.708363 | 通过 |

首组跨 backend 对照如下。loss 列为八 rank 均值，但精度判定逐 rank 执行，不能以均值抵消差异。

| 步数 | native mean loss | MegaMoe mean loss | loss 最大归一化误差 | loss 超限 rank | forward 最大归一化误差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 10.539054 | 10.539054 | 0 | 无 | 0 |
| 2 | 0.980820 | 0.980826 | 0.010356 | 无 | 10.664919 |
| 3 | 0.072665 | 0.072634 | 0.043739 | 无 | 12.898761 |
| 4 | 0.080321 | 0.080216 | 0.216489 | 无 | 34.934174 |
| 5 | 2.049541 | 2.045538 | 0.384933 | 无 | 98.762787 |
| 6 | 1.920564 | 1.913746 | 1.304379 | 1 / 2 / 6 / 7 | 65.732635 |
| 7 | 7.506670 | 7.521816 | 0.330558 | 无 | 95.236740 |
| 8 | 7.314979 | 7.475851 | 4.819952 | 2 / 5 | 834.811951 |
| 9 | 8.189187 | 8.025925 | 1.505480 | 0 / 2 / 5 | 1428.254639 |
| 10 | 8.594246 | 8.602003 | 0.927607 | 无 | 1384.625366 |

第二步 forward 失败来自 logits，不是 loss；后续 loss 并非一旦超限就每步持续失败。
与同源码/初始状态的历史 eps=1e-8 确定性对照相比，首次 logits / loss 失败仍为第 2 / 6 步。
两 backend 的 loss 仍显著回升。动态轨迹已被 eps 改变，不能把某一步差值下降当作原配置已修复。

独立重读第二步的 1048576000 个 logits，eps=1e-7 有 111865143 个超过原阈值，RMSE 为
0.004180811；历史 eps=1e-8 分别为 111605047 / 0.004172864，最大绝对差均为 0.0625。
该步不仅最大归一化误差没有改善，超限数量和 RMSE 也略增。末层独立 eps 复放的首步
BF16 写回与本次完整模型对应参数在两侧八 rank 全部逐位一致，连接了两个实验的实际状态。

## 4. 其他模块 trainer 的实际 AdamW 用法

### 4.1 共享实现，不同训练合同

`BaseTrainer._build_optimizer` 通过 YAML target 的 builder 构造 optimizer；AdamW builder
及 Muon 的 fallback AdamW 最终都调用 `get_hyper_optimizer`，进入同一个
`core.optimizer.adamw.AdamW` → `torch._fused_adamw_`。配置中的 `adamw_` 前缀在 factory
剥离，未填写的 beta/eps 采用构造默认值。开启 `fp32_main_params` 时，trainer 使用的
`Float16OptimizerWithFloat16Params` 与 benchmark 是同一个 wrapper。

| 项目 | 当前 Qwen benchmark | Qwen3-MoE trainer recipe | Qwen2-MoE FSDP 精度专用用例 |
| --- | --- | --- | --- |
| optimizer | 所有参数纯 AdamW | 矩阵走 Muon，其余 fallback AdamW | 纯 AdamW |
| AdamW 基础 LR | 1e-3 | 1e-5；Muon 为 3e-5 | 1e-4 |
| AdamW betas / eps | (0.9, 0.999) / 1e-8 | 相同 | 相同，由 core 默认补齐 |
| weight decay | 所有参数一组 0.01 | AdamW 分 0.01 / 0 两组；Muon 为 0 | 全部为 0 |
| no-decay 规则 | 无，norm 也有 decay | 名称包含 bias / norm / ln_ | builder 同样分组，但两组 decay 均为 0 |
| FP32 main wrapper | 始终开启 | YAML 未开启；配置默认 False | 显式 True |
| 梯度归约/保留 | dense BF16 SUM/8，再转 FP32 | 由 trainer/FSDP policy 配置 | FP32 reduce，并保留 FP32 main_grad |
| 裁剪 | 无 | max_grad_norm=1 | 测试循环 max_norm=inf，仅比较 norm |
| LR scheduler | 无 | cosine + 1 步 warmup | 专用测试循环无 scheduler |

这里的 Qwen3 recipe 是预训练 Qwen3-30B-A3B、真实数据、默认 FSDP8/EP1、全层 checkpoint
的三步示例，不是本次随机八层、固定 batch/路由的 EP8 benchmark。
`split_muon_adamw_params` 把 ndim>=2 且名称不包含 embed/lm_head/out_head/out_proj.bias
等保留关键词的参数分给 Muon，包含普通 expert 矩阵；所以不能称该配方已经证明“expert
使用另一种 AdamW 设置即可稳定”。三个 `examples/training_demo/train_parallel_*.yaml`
也采用相同的 Muon/fallback AdamW 配置模式。

用一个包含 embedding、norm、gate 和 expert 矩阵的小 CPU 模型，实际解析 YAML 并构造
AdamW，验证前缀、参数分组、默认 eps/betas，以及 scheduler 顺序。该 dry-run 不产生梯度、
不启动 NPU、不执行 fused 更新，不冒充完整 trainer 训练。

Qwen3 recipe 的 `lr_config.lr=3e-5` 是调度比例的基准，不会把 AdamW 基础 LR 变成 3e-5。
当前 LambdaLR 初始化及 trainer 的“optimizer.step 后 scheduler.step”顺序，使三次更新
所用的 AdamW LR 实测为 **0 → 1e-5 → 5e-6**，第三次调度后为 0。第一步零 LR 不等于
跳过 backward 或 moments 更新；因此不能把这份三步 recipe 与恒定 LR=1e-3 的十步轨迹等同。

另查 `compile.GraphTrainer`：它在 `_init_optimizer` 中构造的是 `torch.optim.Adam`，
默认 LR=1e-4，可选 `grad_clip`，并不是上述 AdamW 路径，不能混作 AdamW 稳定性证据。

### 4.2 同一个 FP32-main wrapper，不代表同一个梯度路径

trainer 配置在 `fp32_main_params=True` 时强制要求 `reduce_dtype='float32'`；FSDP adapter
进一步设置 `apply_grad_on_fp32_main_grad=True`。专用精度用例明确检查归约后 model.grad
为空、main_grad 为 FP32，随后 `prepare_grads`、`step_with_ready_grads` 并检查直接 cast 写回。
当前 benchmark 没有 FSDP；dense BF16 grad 先归约/平均，wrapper 才转 FP32，不能恢复已发生
的 BF16 归约舍入。这个区别值得单独验证，但不推翻此前确定性首步 dense 梯度逐位一致的证据。

TextTrainer/VLMTrainer 都先做 micro-batch 累积与 FSDP 同步，再裁剪、optimizer.step、
zero_grad、scheduler.step。FP32-main wrapper 本身没有加入动态 loss scaler 或自动裁剪；
`mixed_precision.enabled` 也是与 `fp32_main_params` 分开的配置项。

梯度归一化也必须连同 loss 合同看：trainer 用有效 token 计数加权 backward loss，配合
FSDP 的 DP/CP/TP reduction/scaling；benchmark 用每 rank 本地平均 CE，再手工平均 dense
梯度。不能把 trainer 的缩放因子脱离 mesh/loss 语义移植，也不能据两个 backend 都通过
首步比较，就声称已验证其与真实 trainer 的全局目标完全一致。

### 4.3 不可直接照搬全局裁剪

项目 `clip_grad_norm_` 优先读 main_grad，通信来自梯度/参数的 DTensor placements：
Shard 统计跨 rank 求和，plain tensor 被当作 replicated，不发起该分片归约。
NativeMoeExperts 是 EP-sharded DTensor，而当前 MegaMoe expert 是 packed 本地 Parameter。
如果对两侧直接调用同一 helper，MegaMoe 的其他 rank expert 统计不会自动进入 norm。

后续控制需要显式区分 dense 副本和 home expert 分片，统一原有梯度归一化口径后，
使 dense 只计一次、各 rank home expert 统计合并一次；不能把 physical replica 再计为
独立参数，也不能因统计域不同而给两 backend 施加不同的 clip coefficient。应先用 max_norm=inf 验证
norm 与聚合定义，再尝试实际裁剪。以上是移植约束，本轮没有实现裁剪或宣称其已改善 loss。

### 4.4 源码入口

- [benchmark optimizer/step](../../../../examples/torch/multicore/mega_moe/qwen_moe_benchmark.py)
- [BaseTrainer](../../../trainer/base.py)、[TextTrainer](../../../trainer/text_trainer.py)、
  [VLMTrainer](../../../trainer/vlm_trainer.py)
- [optimizer builders](../../../components/optim/builders.py)、
  [parameter groups](../../../components/optim/parameter_groups.py)、
  [FP32-main wrapper](../../../components/optim/mixed_precision_optimizer.py)
- [optimizer factory](../../optimizer/__init__.py)、[fused AdamW](../../optimizer/adamw.py)、
  [scheduler](../../optimizer/lr_scheduler.py)
- [Qwen3 recipe](../../../models/qwen3_moe/recipes/train.yaml)、
  [Qwen2 精度配置](../../../../tests/torch/dualmode_trainer/fsdp_integaration/test_yamls/qwen2_tp_cp_ep_fsdp_global_accuracy.yaml)、
  [精度测试循环](../../../../tests/torch/dualmode_trainer/fsdp_integaration/_test_fsdp_accuracy.py)
- [trainer config 校验](../../../trainer/config/trainer.py)、
  [FSDP adapter](../../../distributed/_builder/fsdp_adapter.py)、
  [loss aggregation](../../../trainer/runtime/metrics.py)、
  [Torch clip 实现](../../../platform/torch/clip_grad.py)、
  [GraphTrainer](../../../compile/trainer.py)

## 5. 完成本步后的优先级

1. **不把增大 eps 纳入默认优化。** 保留原 baseline 和失败记录，不再根据较小的 main
   最大差推断训练稳定。当前证据没有支持重写 AdamW 或高代价 native-exact dW。
2. **下一项优先单变量 LR=1e-4 控制。** 这是现有纯 AdamW 精度用例的量级，也是此前
   native-only 短轨迹曾表现稳定的档位；应补齐两个 backend 的真实多步 loss/logits、
   梯度/参数与自重复。先保持 eps=1e-8、WD=0.01、无裁剪/调度，不能把同时换 WD、
   Muon、warmup 的结果误认作 LR 效果；新合同通过也不替代原合同验收。
3. **随后分别核验 trainer 对齐项。** 梯度全局归一化及 norm 定义先于实际裁剪；FP32
   dense 归约需独立记录精度、通信和 HBM 成本；no-decay 分组、WD、warmup 各自单列。
   不把整套 trainer recipe 一次搬入 benchmark 后再倒推是哪一项有效。
4. **性能 gate 独立。** 本轮有完整 CPU snapshot、全量 hash/oracle，不能发布 latency、
   HBM 增量或新加速比例；原有竞争/确定性模式边界不变。

后续已执行其中的单变量 LR=1e-4 控制，见
[低 LR 多步对照](megamoe_qwen_lr_stability_20260908.md)：eps/WD 与原梯度路径不变，
两侧十步 loss 持续下降且跨 backend loss 全部通过，但 logits 仍从第二步失败。
这更新下一阶段优先级，不改变本报告 eps 实验的原始合同或失败结论。

## 6. 验证与本地证据

四个独立八卡训练进程组均正常退出，完成 320 次 rank-local 真实更新。native A1/A2 与
MegaMoe B1/B2 在八 rank 的以下自重复检查全部通过：

- 十步 loss 和全量 logits 指纹逐位相同。
- 首两步全部 canonical 梯度/更新后参数指纹逐位相同。
- 第十次更新后的完整 model、FP32 main、一阶/二阶 moment 指纹逐位相同；group step=10。
- source/native、初始权重、batch、配置、实际 optimizer defaults 和 worker 脚本身份一致。

自重复的最终状态比较是同 backend 的全量 hash，不是跨 backend 最终 main/moments 的
数值 oracle；第 3--10 步没有保存全量梯度/更新后参数，不声称十步所有训练状态均通过
跨 backend 精度或有限值验收。每步 loss/logits 则都做了原全量有限值/精度检查。

独立审计重新读取 112 份 native references、80 份 MegaMoe forward 文件，核对 2976 个
张量的 shape/dtype/实际字节 SHA256 及 reference contract。由这些完整值重新计算的十步
forward 最大差、归一化误差及 pass/fail 与运行时 oracle 一致；loss 与 logits 分开判断。
原参考值没有被候选覆盖。四档复放的全部差异坐标、实际 optimizer 设置及 FP64 首步公式
另行复核通过；空差异 rank 的审计空集处理修正后重跑，不影响设备复放的原始记录。

进程采样情况如下。外部 PID 为不属于本组八个 worker 的设备进程，观测到的部分命令为
外部 pytest；不干预这些任务。无未知采样帧不代表连续严格独占。

| 进程组 | 采样帧数 | 有外部 PID 的帧数 | 未知帧数 |
| --- | ---: | ---: | ---: |
| 四档独立层复放 | 341 | 0 | 0 |
| native A1 | 283 | 58 | 0 |
| MegaMoe B1 | 494 | 70 | 0 |
| MegaMoe B2 | 452 | 0 | 0 |
| native A2 | 279 | 0 | 0 |

因此四组只用于本次数值控制，不称为严格独占的性能 ABBA。两 backend 在不同采样条件下
各自逐位复现支持本轮数值结果，但不构成竞争不影响一般精度/性能的保证。

临时脚本语法检查、CPU trainer 配置/scheduler dry-run、上述设备诊断及独立数据审计通过。
报告表格逐项与本地汇总核对，Markdown 链接与 diff whitespace 检查通过。生产源码未改，
不新增 UT，不把诊断正常退出冒充原跨 backend logits/loss gate 通过。

本地证据布局如下；JSON、张量、临时脚本均不入 git：

```text
/tmp/qwen-eps-stability.iUEwDO/
  replay_probe/             training_probe/
  run_training.py          trainer_optimizer_probe.py
  summarize_replay.py      audit_replay.py
  summarize_training.py    audit_training.py    verify_report.py
  replay/
    launch.json            worker.log           census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}/eps{1e-08,1e-07,1e-06,1e-05}/
      native.json          same_grad.json       b1.json
      changed_gradient_points.pt
  train_{a1,b1,b2,a2}/
    launch.json            worker.log           census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}_step{1..10}.json
  train_b1/result/rank{0..7}/step{1..10}_forward.pt
  training_reference/rank{0..7}/
    step{1..10}_forward.{pt,json}
    step{1,2}_{gradients,updated_parameters}.{pt,json}
  replay_summary.json      replay_audit.json
  training_summary.json    training_audit.json
  trainer_optimizer_probe.json                 manifest.json
```

输入沿用 `/tmp/qwen-expert-dw.9pLDND/`、`/tmp/qwen-adamw-diagnosis.DI8gFa/` 和
`/tmp/qwen-multistep-diagnosis.kI6Fza/` 的实际 capture/reference，未重新生成替代原梯度。
复放/训练 launcher 及原始命令、环境见各组 `launch.json`；脚本、汇总及报告 SHA256
汇集在本地 `manifest.json`。关键汇总身份为：

| 文件 | SHA256 |
| --- | --- |
| replay_summary.json | `327deb3d760ecdcd4b69aceda3f8a3ad3e7de245ebf8fbf3090cf86588b5492e` |
| training_summary.json | `6c938de2885e475bc4d13740b66c9d64376b2b219711daea31d10f01a60be9eb` |
| training_audit.json | `bdec8d78ce5b6a90aa0220d26cb6106e026eb2a29671313ea1d78a326e0b0d2c` |
