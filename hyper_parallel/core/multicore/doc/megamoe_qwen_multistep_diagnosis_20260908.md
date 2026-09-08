# Qwen 多步数值分歧与计时竞争诊断

本轮基于拉取后的 `megamoe-sun@ee5a696554133d99c80aa9029e5cdd9ab039c1ac`，继续
[单后端驻留报告](megamoe_qwen_isolated_benchmark_20260908.md) 第 4.4 节留下的诊断。
本轮不修改模型、optimizer、MegaMoe runtime 或原精度阈值；确定性模式、FP32 归约、低学习率
均是单列的诊断对照，不替换默认 baseline。

结论：默认模式有两 backend 共有的非确定性，确定性控制可恢复各自重复性，但跨 backend
仍有从 expert dW 开始的可重复差异，多步 logits/loss 尚未通过原门槛。低 LR 只改善本组
十步轨迹；默认计时的竞争记录仍不对称，不能发布新合并 commit 的性能增量。

## 1. 版本与实验合同

拉取版本合并了 compact runtime descriptor、planner cost model、replica pruning、receive window
等工作；Qwen model、benchmark、config、isolated runner 相对 `a3e1b51b` 未变。
先用当前源码重新构建 Torch/Ascend910B native payload，避免用旧二进制解释新的 descriptor ABI。
旧 payload 已备份，构建成功后实验期间不再改动源码或 native 文件。

- 硬件：8 × Ascend910B3，单卡 64 GiB；CANN 9.2.0，Torch / TorchNPU 2.9.1，Transformers 5.5.3。
- 原始配置：EP8，8 层，H5120 / I1792 / T4096 / V32000，E48 / K8，shared-B1，capacity=1.5，
  Dirichlet alpha=0.1、route seed=42，model seed=42；每 rank 固定自己的 batch，逐步重复。
- Attention：16 个 query heads、2 个 KV heads、head dimension 320；无 dropout。
- 原 optimizer：BF16 model + FP32 main parameters / AdamW，LR=1e-3，weight decay=0.01，
  betas=(0.9, 0.999)，eps=1e-8。每个训练 step 均执行 zero-grad、F+B、dense grad sync、optimizer。
- 沿用 `rtol=0.02 / atol=0.002` 的全量 FP32 比较；归一化误差为
  `abs(actual-reference) / (0.002 + 0.02 * abs(reference))`，大于 1 才超原阈值。
- 一次只启动一个后端的八卡进程组。所有 rank 都记录输入、初始/最终 canonical 参数 SHA256、
  source/native identity、逐步 loss 和绝对时间，不只看 rank 0。

共同 identity map 摘要，算法为 `sha256(json.dumps(map, sort_keys=True).encode()).hexdigest()`：

```text
source_sha256: 4ef0fc3fc3ffc695e03d8491e4242524b6aeb3140395eb74460e83a0bd9fb0c4
native_sha256: 5c43d54081781262bb5903d369c9a6ad8e612374085c60cd5520a487ad841ff9
```

## 2. 已隔离的数值差异来源

### 2.1 不更新参数，common 自重复仍有梯度差异

对完整八层 common 连续做三次固定权重 F+B，optimizer 对象仍按原配置构造，但诊断中显式
将 `step()` 替换为空操作；检查初始/最终全量参数 hash 相同。这个对照不属于训练性能测量。

- 三次全量 logits 在所有 rank 上逐位一致；标量 loss 的归约差异不超过 2.8610e-6。
- dense 同步前已有梯度差异：跨 rank 并集 77/83 个 canonical 张量非逐位一致，最大绝对差
  3.8147e-6。最末层 Q/K/V projection 等开始出现差异，而 lm_head 本地梯度在所有 rank
  的三次结果中逐位一致。
- BF16 dense 同步后，lm_head 梯度不再逐位一致。全部同步后梯度仍通过原阈值，不能把
  “通过 atol”理解为没有可被 optimizer 放大的差异。
- 当前 MegaMoe 两次固定权重 F+B 与 common reference 对照：初始/最终权重、输入完全相同，
  logits 逐位一致；所有 rank 的全量梯度均通过原阈值，两步最大绝对差分别为
  2.3842e-6 / 2.8610e-6。两后端 source/native 指纹一致。

这排除了“必须先更新参数才产生差异”，也不支持把所有早期差异都归因于 MegaMoe expert 路径。

### 2.2 固定真实梯度输入，单独复放 dense collective

从上述 common 首步 CPU reference 读取每个 rank 的同步前梯度，按原顺序选取 67 个 dense
张量，四次各自复制同一组输入，执行原 `all_reduce_coalesced(SUM)` 和除以 8。
不运行模型、attention、MoE 或 optimizer，比较完整 lm_head 梯度的 163,840,000 个元素。
下表范围取第 2–4 次相对第 1 次复放；八个 rank 的统计相同。

| 独立复放模式 | 非逐位相同元素数 | 最大梯度绝对差 | 梯度符号翻转数 | 预测首步 AdamW 更新最大差 |
| --- | ---: | ---: | ---: | ---: |
| 原 BF16 归约 | 56,303,540–57,523,144 | 3.8147e-6 | 17,746–18,047 | 1.9589e-3 |
| BF16 + `torch.use_deterministic_algorithms(True)` | 0 | 0 | 0 | 0 |
| FP32 归约后 cast 回 BF16 | 5–7 | 2.9802e-8 | 0 | 1.2751e-6 |

因此 BF16 dense collective 本身就能产生足以改变微小梯度符号的差异；不需要 attention 输入
变化或 MegaMoe 参与。FP32 对照显著降低差异，但不是严格逐位确定，且没有把生产归约改为 FP32。
这里只比较了完整 lm_head 输出，不声称其余 66 个 dense 输出也全部通过逐位验证。

### 2.3 独立 attention backward 也有非确定性

使用模型实际调用的 `run_qwen3_moe_flash_attention`，固定合成 BF16 Q/K/V 和输出梯度，
配置 `[batch=1, query_heads=16, kv_heads=2, sequence=4096, head_dim=320]`，
causal / BNSD / sparse_mode=2 / dropout=0，每种模式连续四次 F+B。

| 对照 | forward | dQ 最大绝对差 | dK 最大绝对差 | dV 最大绝对差 |
| --- | --- | ---: | ---: | ---: |
| 默认 | 逐位一致 | 2.4414e-4 | 3.90625e-3 | 3.90625e-3 |
| 确定性模式 | 逐位一致 | 0 | 0 | 0 |

这确认实际 attention adapter 所调用的反向路径可以独立产生差异，与完整模型最末层梯度的
观察一致。输入是**同形状合成张量**，不是捕获的真实 Q/K/V；上表绝对差不能直接换算成
整模型 loss 差，也没有完成每一条模型反向边的穷尽归因。

### 2.4 为什么微小梯度差会变成约 0.002 的参数差

零动量状态的首步 AdamW（忽略实现舍入）为：

```text
w_new = w * (1 - lr * weight_decay) - lr * g / (abs(g) + eps)
```

当两次归约使接近零的 `g` 符号相反、且 `abs(g)` 相对 eps 足够大，更新差可接近 `2 * lr`，
本配置就是 0.002。把已经 BF16 归约的结果再转成 FP32 main grad，不能追回归约阶段丢失的信息。

历史小形状 common/common reference 提供了实际 optimizer 的交叉检查：初始 lm_head 权重
完全一致，最大梯度差 1.5259e-5、472 个符号翻转，更新后最大参数差 0.0019989014；CPU
首步公式与实际 BF16 更新的最大差仅 1.5259e-5。expert 0 的 gate-up 对照中，梯度最大差
9.5367e-7、4 个符号翻转，更新后最大差 9.765625e-4，CPU 首步公式与实际结果一致。
这组历史证据来自重构前后的 common，不是当前八层新训练；支持 AdamW 的数值放大机制，
没有发现 optimizer 实现错误。

## 3. 完整模型训练对照

### 3.1 默认模式：自重复同样不能通过多步比较

按 common A1、MegaMoe B1、MegaMoe B2、common A2 顺序，各做 10 个真实 optimizer step。
第 2 步保存或比较全量 loss/logits、83 个梯度和 83 个更新后 canonical 模型参数；第 1、3–10
步记录全部 rank 的 loss，最终再记录全部模型参数 hash。FP32 master / moments 没有做逐步
全量 oracle，因此本轮不是所有 10 步、所有 optimizer 状态的完整精度验收。

四组配置、seed、对应 rank 的输入/初始参数、source/native map 完全相同；各组运行前后指纹
未变。各组自身的 replicated dense 初始/最终参数在八个 rank 间均完全一致，但跨独立运行
的最终参数不同。所有记录的 loss 均有限。

第 2 步对 A1 的全量比较如下；数值为跨八个 rank 的最大归一化误差：

| 候选 | loss/logits | 梯度 | 更新后模型参数 |
| --- | ---: | ---: | ---: |
| common A2 | **16.0054，失败** | 0.029711，通过 | **1.876729，失败** |
| MegaMoe B1 | **15.9694，失败** | 0.022477，通过 | **1.856529，失败** |
| MegaMoe B2 | **15.4316，失败** | 0.029848，通过 | **1.846680，失败** |

forward 的失败来自 logits，三组最大绝对差均为 0.125；第 2 步 loss 本身仍在原阈值内。
更新后参数最大绝对差均为 0.00390625。common/common 已出现同级别失败，因此不能把
这些独立轨迹间的失败单独归因于新合并的 MegaMoe 改动，也不能继续称默认多步精度已通过。
本轮第 2 步以 common A1 为统一 anchor，B1/B2 未另存全量候选梯度做直接互比；
MegaMoe 自重复结论来自全 10 步 loss 和最终全部模型参数 hash，不冒充其第 2 步完整互比。

下面是记录后在 CPU 上计算的八 rank loss 均值，不是额外施加给训练的 loss reduction：

| optimizer step | common A1 | MegaMoe B1 | MegaMoe B2 | common A2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10.539054 | 10.539055 | 10.539054 | 10.539054 |
| 2 | 0.989696 | 0.989683 | 0.989798 | 0.989789 |
| 3 | 0.083473 | 0.083493 | 0.083478 | 0.083446 |
| 4 | 0.072547 | 0.072488 | 0.072601 | 0.072690 |
| 5 | 1.501870 | 1.497062 | 1.450180 | 1.484002 |
| 6 | 1.264459 | 1.238857 | 1.224548 | 1.253639 |
| 7 | 5.115752 | 5.085871 | 5.020524 | 5.035926 |
| 8 | 7.681335 | 7.767803 | 7.860365 | 7.721008 |
| 9 | 8.563452 | 8.310156 | 8.119903 | 8.752827 |
| 10 | 8.604605 | 8.206761 | 7.998721 | 8.582307 |

对每个对应 rank 按原阈值比较 loss，A1/A2、B1/B2、A1/B1、A1/B2 都从第 5 步首次失败。
仅看最终 rank-zero loss 或跨 rank 平均值会漏掉部分更早的分歧；原始 JSON 保留全部 rank。

### 3.2 确定性模式对照

唯一执行模式变化是启动时调用 `torch.use_deterministic_algorithms(True)`，仍为原 LR=1e-3，
同样完成 A1/B1/B2/A2 四组各 10 步，使用同一第 2 步 snapshot 合同。
它不是基于警告忽略或放宽阈值的“确定性”；验证结果如下：

| 比较 | 八 rank × 十步 loss | 最终全部 canonical 模型参数 | 第 2 步全量对照 |
| --- | --- | --- | --- |
| common A1/A2 | 逐位一致 | hash 全部一致 | logits、梯度、更新后参数全部逐位一致 |
| MegaMoe B1/B2 | 逐位一致 | hash 全部一致 | 两组分别与 common A1 比较；未另做候选全量互比 |
| common A1 / MegaMoe B1 或 B2 | 第 6 步首次有 rank 超原 loss 阈值 | 不同 | logits 失败；梯度和更新后参数通过 |

跨 backend 第 2 步最大 logits 绝对差为 0.0625、归一化误差 10.239767；梯度最大绝对差
3.0518e-5、归一化误差 0.014966；更新后参数最大绝对差 0.00390625，但这次发生在更宽
reference-relative 容差的元素上，最大归一化误差 0.785867，故通过原阈值。
绝对最大差相同，不意味着是否通过也相同。

确定性模式恢复了本组实测的 backend 内部重复性，**没有消除可重复的跨 backend 差异**，
也没有消除 loss 回升。八 rank 的均值从第 4 步 common/MegaMoe 的 0.072696/0.072610，
升到第 10 步的 8.620320/8.424216。不能把确定性模式当成跨 backend 精度修复或收敛修复。

### 3.3 确定性首步定位与低学习率控制

补做 common / MegaMoe 各一次确定性首步训练，保存首次 optimizer 更新之前的完整梯度，
再比较更新后全部模型参数。输入、初始参数与 source/native 指纹完全相同：

| 首步阶段 | 八 rank 的结果 |
| --- | --- |
| loss / logits | 全量逐位一致 |
| 67 个 dense 梯度 | 全量逐位一致 |
| expert 梯度 | 跨 rank 非逐位张量并集为全部 16 个 expert 张量；最大绝对差 9.5367e-7 |
| 更新后 dense 模型参数 | 全量逐位一致 |
| 更新后 expert 模型参数 | 最大绝对差 1.2207e-4，最大归一化误差 0.052787，通过原阈值 |

首步梯度最大归一化误差为 0.000476，仍通过原阈值；但“通过”不代表首步 expert 权重相同。
结合第 3.2 节可重复的第二步 logits 失败，下一步定位应从 **expert dW 的累加/舍入路径**
开始，而不是继续泛化怀疑 attention 或 dense 同步。这里没有直接检查每条隐藏状态的 dX，
不能由 dense 参数梯度一致推导任意中间张量都逐位相同。

代码候选范围是 common 的 `components/functional/npu_grouped_swiglu.py` 中
`_NPUGroupedMatmul.backward`（grouped dW），与 MegaMoe 自定义 dW、physical copy / replica
贡献的 FP32 聚合及最终 cast。上述证据尚不能唯一归因到分块累加顺序、copy 切分或某一侧
舍入，也没有证明哪一侧数学结果错误；不能为贴齐 reference 而提前把 replica contribution
降为 BF16，破坏已有的 FP32 聚合合同。

再做两次默认执行模式的 common 控制，仅将 LR 改为 1e-4，不开启确定性。它们只记录
10 步 loss 和输入/初始/最终参数指纹，不做全量梯度 oracle。两次在所有 rank 上均逐步
下降，八 rank 均值如下：

| step | common A1，LR=1e-4 | common A2，LR=1e-4 |
| --- | ---: | ---: |
| 1 | 10.539054 | 10.539054 |
| 4 | 6.884703 | 6.884747 |
| 7 | 3.079109 | 3.079263 |
| 10 | 0.683563 | 0.683609 |

十步对应 rank 的 loss 全部通过原阈值，最大绝对差 5.0521e-4、最大归一化误差 0.012971；
loss 和最终模型参数仍非逐位一致。该控制支持学习率敏感性，不证明长期收敛，也没有验证
低 LR 的跨 backend 全量精度。原 LR=1e-3、原 rtol/atol 和默认执行模式均未修改。

## 4. 计时与竞争的可解释边界

诊断 worker 在每步 barrier / device synchronize 后记录 `time.time_ns()` 和
`time.perf_counter_ns()`，执行与原 benchmark 相同的训练 step，再同步并记录结束时间。
父进程不断保存带采样起止绝对时间的 `npu-smi info` 原始输出，并尽可能记录 PID 命令行。
每步用八个 rank 最早开始到最晚结束的区间，保守匹配与该区间重叠的采样帧，排除自己的
八个 worker PID，留下外部进程。

第 2 步全量 snapshot 用于精度诊断，不参与稳态计时；第 4–10 步不做 snapshot。
统计口径为每步取 rank-max，再取七步 median。快照落盘与比较在稳态之外，但这仍是带
额外 warmup 诊断的运行，不冒充原 shell 的无改动重跑。固定权重测试计时也不作为性能结果。

当前默认组观测值如下。四组稳态窗口均没有采样失败帧，但竞争记录不对称：

| 默认组 | 七步 median（ms） | 与稳态窗口相交的外部 PID 采样 |
| --- | ---: | --- |
| common A1 | 1289.887 | 七步全部命中，每步 45–46 个外部 PID |
| MegaMoe B1 | 714.911 | 七步全部命中，NPU 4 上的 PID 54264，约 136 MiB 驻留 |
| MegaMoe B2 | 715.593 | 七步均未命中 |
| common A2 | 1206.259 | 七步均未命中 |

同 backend 的 common A1/A2 相差约 83.6 ms，与采样到的环境差异共现。B1 的小额进程驻留
不证明它在执行计算。B2/A2 可作为较干净的相邻观测，但默认组没有两轮共同干净的 ABBA，也没有
匹配的旧版本独占重跑，因此这里不发布新合并 commit 的加速/回退比例。

单列的确定性组四次 median 为 common A1 **1220.890 ms**、MegaMoe B1 **727.858 ms**、
MegaMoe B2 **727.702 ms**、common A2 **1219.385 ms**；其 28 个稳态窗口均未采到外部
PID，且没有采样失败帧。它提供了当前版本较稳定且可审计的对照，但执行模式不同，不能
混入默认组统计，也不能与历史默认结果相减来估算本轮合并的性能增量。

采样未命中外部 PID 不等于硬件独占；约 0.5 秒的采样空隙、命令自身采样区间及短命进程都
有盲区。外部 PID 与慢步重叠也只是竞争证据，不能仅凭共现量化其因果耗时。未终止或修改
任何外部 Jenkins / NPU 作业。

首次 `train_default_a1` 尝试在模型初始化后因 `npu-smi` 采样超过 8 秒而被临时父脚本终止，
未完成训练 step，不计入结果。保留该尝试到 `train_default_a1_census_timeout/` 和
`training_default.log`；修正临时采样脚本，超时返回 `returncode=124 / coverage=unknown`，
继续训练并统计缺测帧，不能把缺测当作无竞争。重跑日志为 `training_default_retry.log`。

## 5. 验证与原始证据

本轮已有 CPU 回归：MegaMoe UT、module UT 和 Qwen model/config/snapshots/isolated UT，
`193 passed / 202 subtests`，26.95 秒。当前 native 完整构建成功；固定权重 common 和
MegaMoe、默认/确定性训练矩阵、确定性首步和低 LR 控制共 14 组模型实验均正常退出。
112 份 rank 结果的输入、初始参数、模型配置和运行前后 source/native identity 一致；
optimizer LR 的控制差异单列。正常退出不代表数值对照通过，失败项见第 3 节。

本地保留的 `megamoe_qwen_multistep_diagnosis_20260908.json` 包含全部 rank 的
逐步 loss、原阈值比较结果、时间/竞争摘要、首步定位、独立复放，以及 318 份保留原始文件的
SHA256。该摘要与原始 JSON、全量模型/梯度张量均不入 git，存放在下述本地诊断目录；
reference 的逐张量 hash 元数据也列入本地校验清单。

本轮原始日志、全量 CPU references、逐 rank / step JSON、采样原文与临时诊断脚本均在：

```text
/tmp/qwen-multistep-diagnosis.kI6Fza/
  megamoe_qwen_multistep_diagnosis_20260908.json
  native_build.log
  previous_native_lib/
  pulled_cpu_ut.log
  historical_adamw.json
  fixed_common/                  fixed_common_reference/
  fixed_mega/
  train_default_a1/              train_default_b1/               train_default_b2/       train_default_a2/
  train_deterministic_a1/        train_deterministic_b1/         train_deterministic_b2/ train_deterministic_a2/
  first_deterministic_common/    first_deterministic_mega/
  train_low_lr_a1/               train_low_lr_a2/
  collective_bf16/               collective_deterministic/       collective_fp32/
  attention_model_shape_default.json
  attention_model_shape_deterministic.json
  worker.py                     run_case.py                     summarize.py
  collective_replay.py           attention_replay.py             run_training.py
  run_followup_controls.py       training_overview.py
```

临时脚本复用当前模型、训练 step、snapshot oracle，不改生产模块；大体积 CPU references 不入 git。
`attention_default.json` / `attention_deterministic.json` 是早先 40/8 heads、D128 的探索性复放，
不作为本报告实际模型形状的证据。早期 fixed-common census 的结构化 PID 解析遗漏了表格竖线，
最终汇总已从保留的完整 raw table 重新解析，不能用旧帧的空 `processes` 字段声称无竞争。

本机复现实验入口如下。脚本拒绝覆盖已有 case/reference，因此重跑时先通过 `mktemp -d`
建立新的诊断目录并放入同版本脚本，不直接复用已完成的结果目录：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HYPER_PARALLEL_PLATFORM=torch
export OMP_NUM_THREADS=1
QWEN_DIAG_PY=/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python
QWEN_DIAG_DIR=/tmp/qwen-multistep-diagnosis.kI6Fza

"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_training.py" --group default
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_training.py" --group deterministic
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_followup_controls.py"
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/training_overview.py"
```

以上入口顺序执行，不同时启动多个本轮八卡进程组。parent 为每组选择独立 MASTER / SHMEM
端口，HCCL NPU socket range 为 `62000-62063`。跟随控制中的低 LR 两组不选择任何 tensor
snapshot step；`loss_only_no_tensor_references` 仅满足临时 worker 复用原 parser 的单后端参数
要求，不会生成或读取不存在的张量 reference。对应 loss 比较、初始/最终 hash 和运行身份
由逐 rank JSON 完成，不宣称该控制验证了所有梯度或 optimizer 状态。

## 6. 本轮后的优先级

1. **先拆 expert dW 的可重复残差。** 固定相同权重、路由和实际层输入/输出梯度，逐段比较
   原生 grouped dW、自定义 dW、physical copy 部分和 owner 聚合结果；用 B0/B1 对照区分
   kernel 累加顺序与 replica 切分/最终 cast。必要时用受控的小块高精度计算判定舍入误差，
   不提前削减 FP32 contribution 精度，也不将本轮 tensor-level 定位冒充具体 kernel 的定罪。
2. **精度与性能分开验收。** 确定性模式可用于复现数值问题，但当前跨 backend 多步检查仍失败。
   修复后重新验证原配置的 logits、梯度、更新后模型参数，并扩展 FP32 main/moments 与更多
   更新步的状态对照。低 LR 或放宽 rtol/atol 都不能替代这项验收。
3. **再评估默认重复性的低成本改善。** FP32 dense 归约是有独立证据支持的候选，不是已经
   验证的整网修复。若实现，应核对现有 main-grad 生命周期和有界 bucket，避免多一套长驻
   全量梯度池；同步评估通信量、峰值 HBM、snapshot/optimizer 的数值合同，不能静默改 baseline。
4. **稳定测量之后再选调度优化。** 若要判断拉取 commit 的性能增减，需要旧/新代码分别匹配
   native payload，在相同执行模式、相同配置下取得可审计的八卡配对窗口，再采全 rank profile。
   本轮既不终止其他作业，也不从进程驻留或离散采样推断严格独占和具体竞争耗时。

本轮仅将诊断报告入仓，证据摘要保留在本地；不实施上述候选修复，不新增结构性 UT；按约定直接本地
commit，不使用 autogit，不 push。
