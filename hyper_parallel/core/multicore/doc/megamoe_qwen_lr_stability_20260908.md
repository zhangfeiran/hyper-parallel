# Qwen 单变量 LR=1e-4 多步训练对照

本轮接续 [eps 稳定性与 trainer 对照](megamoe_qwen_eps_stability_20260908.md)，基于
`megamoe-sun@8870070c`。生产 Python/native 相对 `ee5a6965` 未变。只通过 benchmark 已有的
`--learning-rate 1e-4` 改变实验 LR，不修改默认值、optimizer 实现或梯度路径。

结论：**LR=1e-4 控制中，两侧十步 loss 均持续下降，跨 backend loss 全部通过，但
logits 仍未通过。** 第 1、2、10 步的全量梯度/参数仍通过原 tensor 阈值。这支持高 LR 是
此前短轨迹不稳定的敏感因素，不等于 expert dW 舍入、原 LR 配置或默认模式精度已修复。

## 1. 合同与覆盖范围

保留原八层 connected Qwen example：EP8、H5120 / I1792、E48 / K8，每 rank 4096 tokens，
vocab 32000、shared-B1、capacity factor 1.5、Dirichlet 0.1 固定路由及相同初始权重/batch。
AdamW 仍为 eps=1e-8、WD=0.01、betas=(0.9, 0.999)，BF16 model / FP32 main/moments。
不加 scheduler、warmup、裁剪、no-decay 分组，也不切换 Muon。

环境仍为 8 张 64 GiB Ascend 910B3，CANN 9.2.0、Torch/Torch-NPU 2.9.1、
Transformers 5.5.3；editable import 已确认指向本 checkout。

本轮使用确定性模式，比较对象为此前同模式 LR=1e-3 的控制，而不是把确定性开关也当作
LR 的效果。此前 `train_low_lr_a1/a2` 是默认非确定性模式下的 native-only 数据：可作为
选择此 LR 档位的线索，但不是本轮逐位参考，也未验证 MegaMoe。

八卡按 native A1 / MegaMoe B1 / MegaMoe B2 / native A2 顺序独立驻留，每组执行十次原
`_optimizer_step`：真实 F+B → 原 BF16 dense SUM/8 → FP32-main fused AdamW → BF16 写回。
没有 optimizer monkey-patch、空更新或只改报告标签。实际 defaults/param group 的
LR、eps、WD、betas 和逐步递增的 group step 均检查。

采集/比较范围：

- 每步完整 loss/logits，原 `rtol=0.02 / atol=0.002`，逐 rank 判定，不用平均 loss 抵消误差。
- 第 1、2、10 步全部 83 个 canonical 梯度与更新后参数，比上一轮增加末步全量比较。
- 末步完整 model/main/moments 指纹；额外全量检查 FP32 main/moments 的 dtype 和有限值。
- source/native、配置、初始参数/batch 指纹，以及两 backend 的首步 forward/梯度历史链接。

第 3--9 步没有保存全量梯度/更新后参数；末步 main/moments 只做同 backend 自重复 hash
及有限值检查，没有保存全量值做跨 backend 的数值 oracle。不能把这些未覆盖项写成已验收。

`/tmp` 只余约 201 GiB，不足以增加末步快照。本轮原始数据保存到仓库外的
`/home/weizheng/work/zfr/qwen-lr-stability.LGYINg/`，旧证据不删除，JSON/张量/临时脚本不入 git。

## 2. 十步 loss 与 forward

以下为 A1/B1 的八 rank 均值；精度判定逐 rank 执行。归一化误差为
`abs(actual - native) / (0.002 + 0.02 * abs(native))`，大于 1 才超原阈值。

| 步数 | native mean loss | MegaMoe mean loss | loss 最大归一化误差 | loss 超限 rank | forward 最大归一化误差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 10.539054 | 10.539054 | 0.000000 | 无 | 0.000000 |
| 2 | 9.093883 | 9.093871 | 0.000368 | 无 | 10.825780 |
| 3 | 7.857574 | 7.857608 | 0.000941 | 无 | 12.620228 |
| 4 | 6.884751 | 6.884725 | 0.001972 | 无 | 11.838282 |
| 5 | 5.545509 | 5.545509 | 0.002370 | 无 | 12.522596 |
| 6 | 4.390320 | 4.390457 | 0.005130 | 无 | 11.882232 |
| 7 | 3.079215 | 3.079110 | 0.005403 | 无 | 12.586909 |
| 8 | 2.047648 | 2.047724 | 0.007032 | 无 | 12.060124 |
| 9 | 1.233722 | 1.233623 | 0.012705 | 无 | 13.001662 |
| 10 | 0.683569 | 0.683580 | 0.009494 | 无 | 12.221617 |

两 backend 的每个 rank 均连续十步 loss 下降。八 rank × 十步的 loss 最大绝对差为
4.620552e-4，最大归一化误差为 0.0127054；不是只在平均后通过。第 2--10 步的 forward
失败仍来自 logits，不能用 loss 的通过覆盖；末步 logits 最大绝对差为 0.125。

独立读取完整 logits 后重算如下，两步各覆盖八 rank 合计 1048576000 个元素，且每个 rank
都存在 logits 超限。不同元素数与超限元素数不是同一指标：

| logits 步数 | 不同元素数 | 超原阈值元素数 | 最大绝对差 | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 2 | 799743683 | 110224179 | 0.03125 | 0.004159040 |
| 10 | 833466173 | 148913179 | 0.125 | 0.005027591 |

同源码、同初始状态、同确定性模式的历史 LR=1e-3 对照，native/MegaMoe 平均 loss 在
第四步为 0.072696 / 0.072610，之后明显回升，第十步为 8.620320 / 8.424216；跨 backend
loss 从第六步有 rank 超阈值。本轮新 LR 的初期 loss 下降较慢，但十步内没有重现回升或
loss 超限。比较的是不同优化轨迹，不能把低 LR 的末步参数直接当作原 LR 的精度参考。

首步两侧全量 forward 与历史逐位一致，全部梯度也分别复现各自历史指纹。LR 首次参与
optimizer 更新之后，轨迹才变化。eps/betas/WD 配置未动；AdamW 的梯度项和 decoupled
decay 项都会随 LR 缩放，本轮没有进一步分解两者的贡献。

## 3. 梯度、参数与自重复

不同张量数是跨 rank 的 canonical 名称并集，不是不同元素数：

| 步数/阶段 | 不同张量数（并集 / 83） | 最大绝对差 | 最大归一化误差 | 原 gate |
| --- | ---: | ---: | ---: | --- |
| 1 / gradients | 16 | 9.536743e-7 | 0.000476256 | 通过 |
| 1 / updated parameters | 16 | 1.220703e-4 | 0.0527872 | 通过 |
| 2 / gradients | 83 | 3.814697e-6 | 0.001905586 | 通过 |
| 2 / updated parameters | 66 | 2.441406e-4 | 0.1055743 | 通过 |
| 10 / gradients | 83 | 7.629395e-6 | 0.003784381 | 通过 |
| 10 / updated parameters | 66 | 1.953125e-3 | 0.4040196 | 通过 |

首步梯度不依赖 LR；首步 BF16 参数最大差也仍为 1.220703e-4，与历史 LR=1e-3 相同。
这不证明不同元素数或整体误差相同，但说明降低 LR 不保证降低 BF16 最大差。继续保留
原 replica FP32 partial 聚合、FP32 main/moments 和 BF16 写回，不为逐位对齐引入高代价重算。

native A1/A2 和 MegaMoe B1/B2 在全部八 rank 各自通过以下自重复检查：

- 十步 loss 和完整 logits 指纹逐位一致。
- 第 1、2、10 步全部 canonical 梯度/更新后参数指纹逐位一致。
- 第十次更新后完整 model、FP32 main、一阶/二阶 moment 指纹逐位一致。

四组末步 main/moments 均通过全量 FP32 dtype/有限值检查，实际 optimizer group step=10。
这是同 backend 的逐位自重复，不是两 backend 的末步 main/moments 数值相同。

## 4. 解释边界与下一步

低 LR 下固定 batch 的十步下降曲线，不等于真实数据长期收敛；降低 LR 是改变优化轨迹，
不是消除 expert dW 舍入，也不能替代原 LR=1e-3 合同的精度验收。loss、logits 和训练状态
各自按原阈值报告，不把其中一项的通过扩展到其他项。

本轮有完整 CPU snapshot、hash/oracle 和额外状态检查，不发布新的 latency/HBM 或内核
加速比例；确定性控制与默认模式、性能测量继续分开。

完成本步后的优先级调整为：

1. **先验证全局梯度归一化和 norm，不继续盲目扫 LR/eps。** 保留 LR=1e-4 为显式的
   稳定短轨迹控制，不改变生产默认值。用可容纳的全局 batch/unsharded 小模型参考核对
   本地平均 CE、dense SUM/EP 和 home expert 梯度对应的全局目标；两个 EP backend
   互相通过，不足以证明这一合同正确。本轮尚未运行该参考，不能先断言存在缩放 bug。
2. **归一化明确后，先测 norm，再裁剪。** dense 副本只计一次、home expert 分片合并
   一次，不重复统计 physical replica；先用 max_norm=inf 验证两 backend 与全局参考
   的 norm，再尝试有限阈值。不能直接给 plain packed expert 套用依赖 DTensor
   placements 的 helper。trainer 的有效 token 加权与 mesh scaling 也须作为整体核对。
3. **FP32 dense 归约独立控制。** 它与 BF16 归约后转 FP32 并非相同数值路径，须单列
   完整训练状态精度、通信量和 HBM 成本。WD/no-decay、warmup/调度继续分项，不一次
   搬入整套 recipe，也不把降低 LR 的效果算到通信或 optimizer 实现优化上。
4. **保留未验收项。** 本轮未覆盖低 LR 的默认非确定性 MegaMoe 多步、自重复及更长
   真实数据轨迹；原 LR 多步 logits/loss 失败记录不撤销。干净性能窗口与无诊断探针的
   完整模型 profile 仍独立验收，不用本轮训练诊断耗时推断性能。

## 5. 本地证据与验证

四个独立八卡进程组均正常退出，完成 320 次 rank-local 真实更新。全部 32 份最终 rank
记录通过 source/native、配置、初始参数/batch、实际 optimizer 设置、worker 身份及
逐步 group step 核验。生产源码/native 未改，旧参考值未被新控制覆盖。

独立 CPU 审计重新读取 128 份 native reference 和 80 份 MegaMoe forward 文件，核对
4304 个张量的 shape/dtype/实际字节 SHA256，reference schema/contract 与运行时指纹
均一致。完整 logits/loss 的有限值、最大差、归一化误差及原阈值 pass/fail 独立重算，
与运行时 forward oracle 一致；loss 和 logits 分开汇总，不用前者覆盖后者的失败。

进程采样如下：a1/a2 为 native，b1/b2 为 MegaMoe。外部 PID 指不属于本组八个 worker
的设备进程；覆盖检查要求每张设备有解析出的进程行或明确的无进程行，否则记为未知。

| 进程组 | 采样帧数 | 有外部 PID 的帧数 | 未知帧数 |
| --- | ---: | ---: | ---: |
| a1 | 350 | 0 | 0 |
| b1 | 661 | 0 | 0 |
| b2 | 703 | 0 | 0 |
| a2 | 349 | 0 | 0 |

采样未见外部设备进程，不代表连续严格独占。即使本轮没有观测到竞争，CPU 快照/hash/
oracle 与额外状态检查仍使其不适合作为性能 ABBA；不据此更新旧的 latency/HBM 结论。

临时脚本语法、四组设备诊断、独立完整快照审计、报告数值表格/链接和 diff whitespace
检查通过。只修改 Markdown 报告与索引，不新增 UT；诊断正常退出不等于 logits gate 通过。

本地证据布局如下；以下均在仓库外，git 只记录报告与索引：

```text
/home/weizheng/work/zfr/qwen-lr-stability.LGYINg/
  training_probe/
    worker.py             run_case.py           probe_util.py
  run_training.py         summarize_training.py
  audit_training.py       verify_report.py
  train_{a1,b1,b2,a2}/
    launch.json           worker.log            census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}_step{1..10}.json
  train_b1/result/rank{0..7}/step{1..10}_forward.pt
  training_reference/rank{0..7}/
    step{1..10}_forward.{pt,json}
    step{1,2,10}_{gradients,updated_parameters}.{pt,json}
  training_summary.json   training_audit.json   manifest.json
```

模型、输入、forward/梯度的历史指纹链接到
`/tmp/qwen-multistep-diagnosis.kI6Fza/` 和 `/tmp/qwen-eps-stability.iUEwDO/`；
本轮参考由新 native A1 生成，未覆盖旧 LR/eps 数据。每组的完整启动命令及通信环境见
`launch.json`，诊断 worker 直接调用原 benchmark 的 workload/optimizer-step，脚本身份
随结果保存。生产入口为
[Qwen benchmark](../../../../examples/torch/multicore/mega_moe/qwen_moe_benchmark.py)及
[现有 LR 参数](../../../../examples/torch/multicore/mega_moe/qwen_moe_config.py)。

脚本、汇总及本报告 SHA256 汇集在本地 `manifest.json`。关键汇总身份为：

| 文件 | SHA256 |
| --- | --- |
| training_summary.json | `a55bc488474bf90f33149e69efb875f92986124e4811a98918f39ca8efb9b9a2` |
| training_audit.json | `c58ee24b6c288d4046200f11f73db691ccf62ce5915743963d71df4c6552bec2` |
