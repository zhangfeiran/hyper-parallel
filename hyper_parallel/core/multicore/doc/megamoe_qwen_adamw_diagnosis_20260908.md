# Qwen AdamW 与 BF16 参数写回的放大诊断

本轮接续 [expert dW 切分诊断](megamoe_qwen_expert_dw_diagnosis_20260908.md)，基于
`megamoe-sun@7891f94d`。按当前选择保留 replica / FP32 partial 路径，不实施高代价的
native-exact dW 重算或重新切分。生产 Python / native 相对 `ee5a6965` 未变。

结论：在真实第 8 层权重/梯度的独立 optimizer 复放中，**没有发现 AdamW、参数布局或
FP32 main 生命周期错误**。同梯度时两种布局的 main/moments/model 全部逐位一致；换入
B1 梯度后，差异经小梯度区域的 AdamW 更新敏感度和 BF16 写回边界两阶段作用，首步出现
738 个不同的 BF16 参数元素。两侧首步写回均与之前各自完整模型的该层结果逐位一致。
这定位了本层首步参数差异，尚未完成下一次 forward 的因果干预或整网多步精度验收。

## 1. 实验合同：真实梯度、真实 optimizer，不运行模型反馈

使用上一轮保存的原配置确定性首步数据：第 8 层（索引 7）、EP8、H5120 / I1792、E48 / K8，
每 rank 六个 home experts。初始权重相同，native / B1 梯度来自固定实际 hidden、route、dY
的复放，全部数据指纹与保留的 capture 对上。

三组各构造独立 optimizer，按顺序在八个 rank 执行：

| 对照 | 实际参数布局 | 输入梯度 | 目的 |
| --- | --- | --- | --- |
| native_ref | NativeMoeExperts，三个 EP-sharded DTensor 参数 | native | 复现原始更新 |
| mega_same_grad | MegaMoeExperts，两个 packed 本地参数 | 同一份 native 梯度 | 排除布局/optimizer 差异 |
| mega_b1_grad | MegaMoeExperts，两个 packed 本地参数 | B1 实际梯度 | 只改变 dW |

复用 `qwen_moe_benchmark.py::_build_workload` 的原 optimizer factory：
`Float16OptimizerWithFloat16Params` → `ChainedOptimizer` → 项目 `core.optimizer.adamw.AdamW`
→ `torch._fused_adamw_`。不是用 CPU AdamW 或另一个优化器替代设备执行。

LR=1e-3、weight decay=0.01、betas=(0.9, 0.999)、eps=1e-8、amsgrad/maximize=False，
BF16 model / FP32 main / FP32 moments。沿用确定性模式和原 `rtol=0.02 / atol=0.002`。

每个 optimizer 连续更新三次，**每次重复喂同一份捕获梯度**，不做新的 forward、backward、
MoE 或 dense gradient synchronization。共 24 个独立 optimizer 实例、72 次 rank-local 更新。
它用于隔离更新/写回及状态延续，绝不是完整模型三步训练、收敛或性能测试。

## 2. 已排除的实现问题

1. 三组每步 BF16 model grad 转 FP32 main grad 都与输入梯度的直接 FP32 转换逐位一致；
   model.grad 随后被消费，main_grad 没有意外残留，下一步 zero-grad 正常清理。
2. fused AdamW 的 group step 依次为 1 / 2 / 3。第二、三步的 FP32 main 在更新前与上一步
   更新后的全量指纹相同，没有重新从 BF16 model 初始化 main。
3. 每步 BF16 model 写回与实际 FP32 main 的直接 BF16 cast 全量逐位一致。
4. native_ref / mega_same_grad 的 main、一阶/二阶 moment、BF16 model，在八 rank、三次
   更新中全量逐位一致，排除本次 packed / transpose / DTensor 布局造成不同 optimizer 结果。
5. native_ref / mega_b1_grad 首步写回的 gate-up / down，各自在八 rank 上与历史完整模型
   `first_deterministic_common` / `first_deterministic_mega` 的第 8 层最终参数指纹完全相同。
   原权重及 source/native identity 同样匹配，不是只对齐了某一个最大误差。

全量比较覆盖该层 1321205760 个逻辑权重元素，每个状态阶段均比较完整 gate-up / down。
B1 与 native 的任何状态差异，都只出现在输入梯度不同的 40566 个坐标内；输入梯度相同处
没有出现新的 optimizer 数值差异。以上是本层、固定梯度三次更新的结论，不外推到任意缺失
梯度、checkpoint、参数 offload 或其他模型的 optimizer 生命周期。

## 3. 全量状态差异

以下为 mega_b1_grad 相对 native_ref，元素数跨八 rank 求和，最大绝对差取全量最大值。
三步重复的是首步梯度，因此输入 dW 最大差始终为 4.7683716e-7。

| 固定梯度更新次数 | FP32 main 不同元素 | FP32 main 最大差 | BF16 model 不同元素 | BF16 model 最大差 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40245 | 6.4931810e-6 | 738 | 1.2207031e-4 |
| 2 | 40462 | 1.2986362e-5 | 1338 | 1.2207031e-4 |
| 3 | 40552 | 1.9479543e-5 | 1940 | 1.2207031e-4 |

每一步的一阶、二阶 moment 都有 40566 个不同元素，与输入梯度差异坐标数一致：

| 更新次数 | 一阶 moment 最大差 | 二阶 moment 最大差 |
| --- | ---: | ---: |
| 1 | 4.7683898e-8 | 1.0618242e-13 |
| 2 | 9.0600224e-8 | 2.1225903e-13 |
| 3 | 1.2922465e-7 | 3.1823155e-13 |

所有比较状态有限，仍通过原 tensor 阈值。三步 main 最大归一化误差为 0.00878502，model
为 0.0527872，均小于 1。这个宽容差通过不代表 BF16 参数相同，也不能撤销此前完整模型
第二步 logits、后续 loss 的失败记录。

## 4. 第一阶段：AdamW 对小梯度敏感

零初始化 moments 的首步理想更新为：

```text
u(g) = lr * g / (abs(g) + eps)
w_main_new = (1 - lr * weight_decay) * w_initial - u(g)
du/dg = lr * eps / (abs(g) + eps)^2
```

小梯度区域的最大斜率为 `lr / eps = 100000`。因此是否符号翻转并不是唯一条件；
同号的小梯度也可能把约 1e-10 的梯度差变成约 1e-6 的更新差。

首步 main 最大差的真实位置是 rank 5、packed gate-up 展平索引 40626214：

| 项目 | native | B1 |
| --- | ---: | ---: |
| 初始权重 | 0.01025390625 | 0.01025390625 |
| 梯度 | -2.5756889954e-9 | -2.4738255888e-9 |
| 更新后 FP32 main | 0.0104586184025 | 0.0104521252215 |
| BF16 model | 0.01043701171875 | 0.01043701171875 |

此处梯度差只有 1.0186341e-10，main 差为 6.4931810e-6，但 BF16 写回仍相同。
说明“最大 dW 差的位置”“最大 main 差的位置”“最大 BF16 参数差的位置”不是一回事；
不能把三个全局最大值直接相除来解释逐元素放大。

六个符号翻转位置的梯度绝对值最大仅 5.5252e-11，远小于 eps；首步 main 最大差为
5.7875e-6。这组 expert 残差**没有实测出接近 2*LR=0.002 的更新差**，不能把此前默认
dense collective 非确定性控制中的结果照搬到本组。

## 5. 第二阶段：跨 BF16 舍入边界

BF16 写回经常掩盖小的 main 差异，但当两个 main 值位于同一舍入边界的两侧，就会产生
不同的模型参数。一个首步最大 BF16 差的实际例子：rank 5、down 展平索引 19032926。

| 项目 | native | B1 |
| --- | ---: | ---: |
| 初始权重 | -0.026611328125 | -0.026611328125 |
| 梯度 | -2.0489096642e-8 | -2.0372681320e-8 |
| 更新后 FP32 main | -0.0259390473366 | -0.0259403046221 |
| BF16 model | -0.02587890625 | -0.0260009765625 |

两个 BF16 值的中点是 **-0.02593994140625**，两个 main 值恰在两侧：
梯度差 1.1641532e-10 → main 差 1.2572855e-6 → BF16 差 1.2207031e-4。
这里的 BF16 cast 与 CPU 直接 cast 完全一致，不是拷贝精度错误。

首步 738 个 BF16 差异中，671 个相差一个相邻 BF16 间隔，均核对到 main 跨越中点；
另外 67 个相差多个间隔，不能笼统称作全部“一个 ULP”。仅在 BF16 不同的坐标上计算
`BF16 参数差 / main 差`，中位数约 7.29，最大约 4681.14，也有小于 1 的位置。
量化既可能掩盖，也可能扩大差异，不是每个元素都被统一放大。

FP32 main 并未被 BF16 写回覆盖。后续真实训练的反馈路径应区分为：

```text
dW 残差 → FP32 AdamW 更新差 → BF16 模型权重差 → 下一次 forward / backward → 新的梯度差
```

本轮直接验证到 BF16 写回。固定梯度三次更新的 main 差近似线性增加，不能冒充上面最后
两条模型反馈边的实测，也不能据此解释此前所有层、多步 loss 回升的幅度。

## 6. FP64 公式复核

保存全部 40566 个输入梯度不同坐标的初始权重、两组梯度及每步实际 main/moments/model。
这不是抽样：全量状态比较已经确认差异没有出现在这些坐标之外。

在 CPU 上使用 FP64 按固定梯度的理想 AdamW 公式计算：

```text
d = 1 - lr * weight_decay
m_t = (1 - beta1^t) * g
v_t = (1 - beta2^t) * g^2
w_t = d^t * w_0 - lr * g / (abs(g) + eps) * sum(d^i, i=0..t-1)
```

两组取较大值，实际设备 main 到 FP64 理想值的最大距离分别为
2.291e-9 / 6.445e-9 / 5.962e-9；三步一阶 moment 最大距离不超过 6.345e-12，二阶不超过
4.780e-16。该有限精度实现误差远小于本组跨梯度的最大 main 差，支持正常 AdamW 数值
敏感度的解释，没有发现足以解释当前差异的错误更新公式。该公式复核不要求 fused FP32
运算与 FP64 理想公式逐位一致，也未将新阈值代替原全量 tensor 比较。

## 7. 证据、复现和下一步

全部运行正常退出，源码/native identity 前后、跨 rank 与之前捕获一致。生产代码和 native
未改动，不需要重构建或新增结构性 UT；本轮不提供性能/HBM 测量，也不改变 LR、eps、
确定性默认值或原精度阈值。原始 JSON 和张量不入 git，保留在：

```text
/tmp/qwen-adamw-diagnosis.DI8gFa/
  worker.py          run_case.py       probe_util.py
  analyze.py         audit.py
  summary.json       audit.json        analysis.log
  replay/launch.json
  replay/worker.log
  replay/census.jsonl
  replay/result/rank{0..7}.json
  replay/result/rank{0..7}/native_ref/step{1..3}.pt
  replay/result/rank{0..7}/{native_ref,mega_same_grad,mega_b1_grad}/step{1..3}.json
  replay/result/rank{0..7}/mega_b1_grad/step{1..3}_changed_gradient_points.pt
```

native_ref 保留各步全量 main/moments/model 张量；候选保留全量比较、指纹，以及 B1 的
全部梯度差异坐标状态，不额外保存三套完整候选状态。`audit.py` 验证 24 个 optimizer 实例
的输入与原 capture、首步写回与两组完整模型 reference 的一致性，另记录脚本和摘要 SHA256。
`summary.json` SHA256 为 `710bff3de26c64aa97ff50f0fbfc9d0cd2d4dec460e6cd9a3d986786d50c5f4a`。

本机 venv Python 为 `/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python`，启动
`run_case.py --directory <新诊断根目录>/replay --python <venv Python>`；
先 source CANN 和当前 payload 的 `set_env.bash`，设置 `HYPER_PARALLEL_PLATFORM=torch`、
`OMP_NUM_THREADS=1`。入口无额外 worker 参数，会顺序运行三组固定梯度更新。
脚本读取上述上一轮 capture 的绝对路径；`analyze.py` / `audit.py` 汇总同目录下的 `replay`。
重跑须用 `mktemp -d` 建立新目录、复制脚本并保持对应目录布局；脚本拒绝覆盖已有结果。

后续已完成 [首步 expert 参数干预](megamoe_qwen_parameter_intervention_20260908.md)：只恢复
不同的 expert BF16 参数，MegaMoe 第二次 logits/loss 与历史 native 逐位一致；在 native
反向注入相同差异，也与 MegaMoe 对应组的各层输出逐位一致。末层的 738 个差异值已链接到
下一次 forward，但完整模型后续多步 loss gate 仍未通过。以下保留本轮结束时的原始优先级；
当前下一项转为显式 eps 稳定性控制，不改默认参数或原阈值。

完成此步时的优先级：

1. **下一次 forward 的参数干预。** 固定同一更新后 reference 状态，只替换已确认的 expert
   BF16 参数差异，比较 logits/loss；与恢复这些参数的对照一起量化因果贡献，再扩展到其余层。
   当前只直接链接了第 8 层首步，尚不能声称它单独解释整模型第二步 logits 失败。
2. **保留当前 AdamW 和 FP32 main 生命周期。** 不把正常舍入敏感度当作已发现的 optimizer
   bug，不重置 moments、不把 main 从 BF16 重载，也不为逐位对齐引入高代价 dW 重算。
   如后续评估 eps/LR 稳定性，只能作为显式控制，不能静默修改 baseline 或冒充精度修复。
3. **性能工作继续独立验收。** 原默认 attention / dense collective 非确定性、旧/新代码
   配对计时窗口等边界仍在；不从本轮 optimizer 复放推导性能、收敛或完整多步精度已通过。

本轮只将报告与索引更新直接本地 commit，不使用 autogit，不 push。
