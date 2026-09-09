# Qwen 首步 expert 参数差异对下一次 forward 的因果干预

本轮接续 [AdamW 与 BF16 写回诊断](megamoe_qwen_adamw_diagnosis_20260908.md)，基于
`megamoe-sun@f70d3fdf`。生产 Python / native 相对 `ee5a6965` 未变；保留当前 replica、FP32
partial 聚合和 optimizer 实现，不做 native-exact dW 重算，不改变默认超参数或精度阈值。

关键结果：在 MegaMoe 上，把首步更新后不同的 expert BF16 参数恢复为 native 值，第二次
logits/loss 就与历史 native 全量逐位一致；重新注入差异则重新出现分歧。末层已经定位的
738 个参数差异单独就能使 32 个 logits 元素超过原阈值，但不能代表其余七层的贡献。
反向在 native 注入同一组差异，也与 MegaMoe 对应组的全部层输出、norm、logits/loss
逐位一致。本轮验证的是首步参数到第二次 forward 的因果关系，不是完整模型多步精度修复。

## 1. 实验合同与隔离方式

沿用原始八层 Qwen example、8 × Ascend910B3、Torch/TorchNPU 2.9.1、CANN 9.2.0、
Transformers 5.5.3：EP8，H5120 / I1792 / T4096 / V32000，E48 / K8，shared-B1，
capacity=1.5，Dirichlet alpha=0.1、route/model seed=42，固定每个 rank 的原 batch。
BF16 model / FP32 main / FP32 moments，AdamW LR=1e-3、weight decay=0.01、
betas=(0.9, 0.999)、eps=1e-8。确定性模式仍只是诊断控制，不改生产默认值。

每个 backend 用独立进程组和自己的模型、optimizer；两个 EP8 进程组顺序运行，不同时驻留：

1. 用当前 benchmark 的 `_build_backend_model`、`_build_workload`、`_optimizer_step`
   执行一次真实完整训练 step，包括 backward、dense 同步和实际 fused AdamW 更新。
   核对输入、初始参数、首步 loss/logits、更新后全部 canonical 参数和 source/native identity
   与之前确定性首步记录一致。
2. 从 MegaMoe 首步更新后参数中提取相对历史 native reference 的**全部差异坐标和值**。
   dense 参数必须全量相同；只有 expert 的实际不同元素用于注入或恢复，不改其他值。
   末层坐标、native 值、B1 值逐位核对上一轮实际 AdamW 的 BF16 写回结果。
3. 固定首步状态，切换 BF16 expert 参数后，再调用原训练 step，但仅在临时脚本中把
   `optimizer.step()` 替换为空操作。保留正常的 F+B、dense 同步、zero-grad 以及计算图释放，
   避免改用另一条 inference 路径；本轮只分析 forward，不把这些重复 F+B 计作训练更新。
4. 每组前后核对**全模型**参数指纹：选中层恰好等于 B1 首步值，其余恰好等于 native 首步值，
   执行探针前后参数不变。全部探针前后的 FP32 main、两个 moments 全量 hash 和 group step
   也比较；不从 BF16 参数重载 main，不推进 optimizer，不使用下一轮训练反馈。

这种只替换 BF16 model 而保留原 main/moments 的状态是刻意的因果干预，不能作为一致的训练
checkpoint 继续 optimizer step。实验结束销毁模型；不把它当成恢复 optimizer 的做法。

MegaMoe 共 14 个探针：原始未插桩状态、全层注入、全量恢复、八个逐层单独注入、
仅前七层注入、再次全量恢复、再次全层注入。native 共五个：reference 未插桩、reference
插桩、仅末层注入、全层注入、再次恢复。插桩保存八个 decoder layer 输出、最终 norm 输出、
loss 和全量 logits；每个 backend 都有无插桩的 forward 对照。

原比较仍是 FP32 `abs(actual-reference) / (0.002 + 0.02 * abs(reference))`，大于 1 超限。
另报逐位不同元素数与 RMSE，用于解释误差分布，**不替换原 gate**。logits 覆盖八 rank 的
1048576000 个元素，计数求和、最大误差取全局最大，loss 按对应 rank 比较，不只看均值。

## 2. 实际干预规模

全部 dense 首步参数相同。八层 routed experts 合计 10569646080 个逻辑参数中，有
**11227 个 BF16 元素不同**；每层的最大绝对差都为 1.220703125e-4。

| 模型层（从 1 开始） | BF16 参数差异元素数 |
| --- | ---: |
| 1 | 2046 |
| 2 | 1380 |
| 3 | 1635 |
| 4 | 1133 |
| 5 | 1535 |
| 6 | 1518 |
| 7 | 1242 |
| 8 | 738 |
| 合计 | 11227 |

这不是随机扰动、对最大差位置的抽样或给整层统一加一个偏移；每个注入值都是原 B1 首步
实际参数值。gate/up packed 坐标显式映射到 native 的 w1/w3，down 映射到 native w2；
每组参数 hash 验证该映射的全量结果，不能假设 canonical 的临时拼接张量是可写参数别名。

## 3. 第二次 logits/loss 的变化

以下均为 MegaMoe 执行路径，与原 native 首步更新后的第二次 forward 比较。只注入第 N 层
表示其余七层及全部 dense 参数都是 native reference。表中的 RMSE 覆盖全部 rank/logits。

| 注入范围 | logits 最大差 | 最大归一化误差 | logits 超限元素 | logits RMSE | 对应 rank loss 最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无，全部恢复 native | 0 | 0 | 0 | 0 | 0 |
| 仅第 1 层 | 0.125 | 10.0202 | 111669098 | 4.17404e-3 | 2.13206e-4 |
| 仅第 2 层 | 0.0625 | 9.42333 | 91498210 | 3.78021e-3 | 1.44958e-4 |
| 仅第 3 层 | 0.0625 | 8.59579 | 70015900 | 3.35633e-3 | 2.11895e-4 |
| 仅第 4 层 | 0.0625 | 6.65221 | 48527531 | 2.91790e-3 | 1.59323e-4 |
| 仅第 5 层 | 0.0625 | 6.24185 | 28363224 | 2.47088e-3 | 1.52826e-4 |
| 仅第 6 层 | 0.125 | 4.17603 | 11511821 | 2.01447e-3 | 1.58131e-4 |
| 仅第 7 层 | 0.0625 | 2.91826 | 1195403 | 1.49118e-3 | 7.81417e-5 |
| 仅第 8 层 | 0.0625 | 1.54400 | 32 | 4.01474e-4 | 3.85046e-5 |
| 前七层，恢复末层 | 0.0625 | 10.2398 | 111605854 | 4.17287e-3 | 1.08659e-4 |
| 全八层 | 0.0625 | 10.2398 | 111605047 | 4.17286e-3 | 1.15275e-4 |

所有注入组的本次 loss 差仍通过原阈值，不能称为第二步 loss gate 失败；logits 则都已失败。
末层单独注入产生 71411298 个不同 logits，但只有 32 个超限；全层注入产生 801878157 个
不同 logits，111605047 个超限。不能只凭最大绝对差相同（0.0625）说它们影响相当。

各层的因果影响**不可相加**：只注入第 1 或第 6 层的最大绝对差反而大于全层一起注入。
本组浅层单独注入的 logits RMSE 更大，但各层扰动坐标、数量和数值不同，不能由此估计
统一的“每多经过一层放大多少倍”。恢复末层也不能修复已经由其余层引起的第二步分歧。

## 4. 隐藏状态与恢复对照

八个单层干预、每个八 rank，首个不同 decoder 输出都恰好出现在被注入层；它之前的层
输出逐位一致。特别是末层 738 个参数干预，前七层输出仍相同，差异从第 8 层输出出现，
再传播到 norm、logits/loss。这补齐了上一轮仅验证到 BF16 写回、尚未直接验证的下一条边。

只注入第 1 层时，该层输出 RMSE 为 2.73177e-4，到第 8 层输出变为 9.42436e-3；后七层
参数均保持 native 值，也没有新的 optimizer 更新。这直接观察到同一次 forward 中扰动沿层
传播，不能把这段变化误写为多次 AdamW 更新累积，或由不同层张量尺度推导统一的放大倍率。

MegaMoe 全量恢复后，logits/loss 与之前 native 第二步完整 CPU reference 逐位一致；
全层注入、全量恢复各自再次执行时，八层输出、norm、logits/loss 均复现对应的首次结果。
全层注入的有/无插桩 logits/loss 也一致。原参数到全层注入之间虽然经过额外固定状态 F+B，
仍通过这个自重复控制，不能把恢复一致归因于恰好走了一次不同的运行序列。

随后在实际 NativeMoeExperts backend 上执行相同的 BF16 参数干预。两 backend 的三个
同权重状态，八 rank 结果如下：

| 同权重状态 | 全模型参数 | 八层输出与最终 norm | logits/loss |
| --- | --- | --- | --- |
| 全部 native 首步参数 | 全量相同 | 全量逐位一致 | 全量逐位一致 |
| 仅注入末层 738 个 B1 参数差异 | 全量相同 | 全量逐位一致 | 全量逐位一致 |
| 注入全部 11227 个 B1 参数差异 | 全量相同 | 全量逐位一致 | 全量逐位一致 |

native 自身的无插桩、插桩 reference 及注入后的恢复也完全一致；其 reference 的 logits/loss
复现历史第二步。因而本组第二步分歧可以由首步权重差异重现并通过权重恢复消除，没有发现
额外的、同权重条件下仍存在的 forward backend 差异。该结论限定于本组配置和这些干预状态，
不是任意路由/权重下两种 kernel 都逐位等价的证明。

## 5. 验证边界与后续优先级

已经直接连接的证据是：

```text
真实 expert dW 残差 → 实际 FP32 AdamW 更新差 → BF16 model 写回差 → 第二次 forward 差异
```

末层的前三段在前两轮分别验证，本轮对其 738 个不同写回值完成下一次 forward 干预；
其余层使用真实完整模型的首步参数差异，证明这些参数差异对第二次 forward 的影响，不冒充
其每个 dW 元素都已完成 native/replica/FP64 分解。

固定状态 F+B 产生的新梯度没有做全量精度 oracle；本轮没有执行干预后的新 optimizer 更新，
也未推进第三步及以后的反馈，没有解释完历史第 6 步开始超 loss 阈值及后续 loss 回升的幅度。
仍是固定 batch、随机权重、
合成固定路由的八层 example，不能外推为真实预训练长期收敛结论。原配置跨 backend 的
logits gate 仍失败；解释了失败的数值路径不等于修复或放宽这个 gate。

完成此步后的优先级调整：

1. **降低没有新证据支持的 AdamW 实现错误排查优先级。** 保留 FP32 main、moments、
   replica contribution 合同，不为了逐位对齐重写正确的更新/写回，不引入高代价 dW 重算。
2. **下一项改为显式数值稳定性控制。** 优先单列 eps 敏感性实验：用真实梯度和原 fused
   AdamW 验证参数差异是否降低，再检查真实多步 loss/logits；必要时加入已表现敏感的 LR
   控制。改变 eps/LR 是超参数实验，不是修复现有实现，不能静默改默认、放宽阈值，或用
   新超参数的通过替代原配置的失败。不能只优化最大参数差而忽略训练轨迹。
3. **性能 gate 独立推进。** 本轮有大量 CPU snapshot、逐层同步与固定状态 F+B，不发布
   wall time、HBM 或新 commit 加速比例。默认 attention/dense collective 非确定性和旧/新
   配对计时窗口仍是原边界，不混入本次确定性因果对照。

后续已执行四档 eps 首步复放和 eps=1e-7 的完整模型十步控制，并对照其他 trainer 的
AdamW 用法；较小的 FP32 main 最大差没有消除 BF16/训练分歧。当前优先级已转向单变量
LR 控制及 trainer 梯度合同核验，详见 [eps 与 trainer 对照](megamoe_qwen_eps_stability_20260908.md)。

## 6. 本地证据与复现

两个八卡进程组正常退出，完成两次真实完整首步训练和 19 组固定状态 F+B（152 个 rank-local
探针）。八 rank 的身份、初始状态、首步历史链接、干预范围、参数不变、main/moments 不变、
插桩和恢复控制均通过；三个同权重跨 backend 状态的全部保存输出逐位一致。临时脚本通过
`py_compile`；生产源码未改，不将本轮诊断正常退出冒充原 logits gate 已通过。

独立 `audit.py` 重新读取全部 152 份 CPU trace，核对每个张量的 shape/dtype/SHA256；
再从 native 完整 reference 和稀疏注入值重建全部 128 个 rank-local canonical expert
张量，其 hash 与历史 B1 首步完整参数一致。八 rank 全部通过，报告表格也与汇总逐行核对。
MegaMoe / native 各保留 373 / 273 个进程采样帧，均无未知帧或观测到的外部 PID；这只是
离散采样结果，不作为严格独占或性能测量的声明。

原始 JSON、稀疏参数差异、全量 CPU forward traces 和临时脚本只保留本地，不入 git：

```text
/tmp/qwen-parameter-intervention.oUmFcD/
  worker.py       probe.py       run_case.py       run_all.py
  analyze.py      audit.py
  mega/           native/
    launch.json   worker.log     census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}/result.json
    result/rank{0..7}/<probe>/forward.pt
    result/rank{0..7}/<probe>/result.json
  mega/result/rank{0..7}/deltas.pt
  mega/result/rank{0..7}/delta_metrics.json
  analysis_rank{0..7}.json
  summary.json    manifest.json  audit.json
  analysis.log    analysis_parallel.log    analysis_rank{2..7}.log
  analyze_serial.py                       audit.log
```

读取的历史 references 为 `/tmp/qwen-multistep-diagnosis.kI6Fza/` 的
`first_deterministic_reference`、`first_deterministic_{common,mega}` 和
`train_deterministic_reference`；末层写回坐标来自
`/tmp/qwen-adamw-diagnosis.DI8gFa/replay/result/rank{0..7}/mega_b1_grad/`。
source/native 指纹须与这些捕获一致；脚本检测到不一致会停止，不能用新源码混跑旧 reference。

CPU 汇总最初串行，完成 rank 0–2 后只停止本轮自有分析进程，改为按 rank 并行并复用这三份
完整结果；原脚本与串行日志保留在 `analyze_serial.py` / `analysis.log`，最终汇总日志为
`analysis_parallel.log`。没有中断、重跑或改变已经完成的 NPU 探针。`manifest.json` 记录
脚本和逐 rank JSON 的 SHA256。`summary.json` SHA256 为：

```text
03e2cf3d658dc643ff1655094b63607cbe6fb3d826931aada7574d4445bc1b47
```

本机复现时用 `mktemp -d` 建新目录，将本轮六个 `.py` 脚本复制过去，然后在仓库根目录：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HYPER_PARALLEL_PLATFORM=torch
export OMP_NUM_THREADS=1
/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python <新诊断目录>/run_all.py
```

入口顺序启动 MegaMoe 和 native，不复用进程组；父脚本选择独立 MASTER/SHMEM 端口，
HCCL socket range 为 `62000-62063`，拒绝覆盖已有 case。只终止自己启动且失败/超时的
进程组，不操作外部作业。逐帧保存进程采样，采样无命中不等于硬件严格独占。

本轮不改生产代码，不需要 native 重构建或新增结构性 UT。仅将报告和索引直接本地 commit，
不用 autogit，不 push；诊断 JSON、张量和脚本不入仓。
