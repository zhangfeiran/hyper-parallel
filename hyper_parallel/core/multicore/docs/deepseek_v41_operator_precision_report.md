# DeepSeek-V4.1 MegaMoe 算子精度定位（2026-09-17）

## 结论

本轮在实际 NPU 中间张量和受控替换中确认了两个主要差异来源：

1. **SwiGLU 前后向的 BF16 舍入边界。** HF 的 SiLU 和乘法分别产生 BF16 中间结果，
   native 使用融合 SwiGLU。仅在诊断副本中替换 HF 激活，EP1 hotspot 的两组 expert dW
   在 push/pull、全部三步中均与 MegaMoe **逐元素一致**。
2. **EP 的 dW 聚合边界。** HF 各 rank 先产生 BF16 局部 dW，再 all-reduce；
   MegaMoe 在 expert owner 上对收到的全部 token 计算 dW。融合 HF 激活后，
   再将诊断参考改为 FP32 局部 dW 聚合、最后舍入一次，EP2/EP4 残差大幅下降。

这些实验定位的是两条 BF16 实现之间的数值差异，没有发现需要据此修正 native kernel 的证据。
没有修改生产激活、native kernel 或 optimizer；融合与归约替换仅存在于独立诊断入口。
完整模型训练轨迹及收敛尚未由这些实验覆盖。

## 实验身份和方法

- 源码基线 `c126ee4c` 加本轮诊断代码；每个 JSON 记录执行时源码文件 SHA256。
- native 二进制保持 [block 报告](deepseek_v41_block_precision_report.md) 中的 SHA256。
  CANN 9.1.0、Ascend910B3、Torch/torch-npu 2.9.1、Transformers 5.13.0。
- 每 rank T=128、H=512、I=128、E=8、TopK=2、limit=0；三步 SGD。
  每步 native 和 HF 权重同步，避免历史更新差异污染算子定位。
- EP1：push/pull × hotspot/learned，共四组、12 次 forward/backward。
  在 native launch 返回后、scratch 被复用前拷贝实际 X、gate/up、activation、down、
  weighted dY、dActivation、dGate/up、dX、dW1、dW2；不改 kernel 执行。
- EP2/EP4：learned × push/pull，共四组，每个 rank 三步。
- 本机证据目录：`/home/feiran/doc/dsv41-megamoe-implementation-20260917.sORKhN`。
  `operators-{push,pull}-{hotspot,learned}.json`、对应 `.stepN.pt`，以及
  `reduction-ep{2,4}-{push,pull}.json`；原始日志均保留。
  使用设备 3 和 4–7，按用户许可直接运行；ownership 日志未发现 foreign/unresolved PID。
  拷贝和同步会改变执行时序，**这些结果不用于性能判断**。

区分两种参考：

- **端到端 FP32 oracle**：从同一状态重新执行完整 block FP32 数学，用于正式验收。
- **单算子 FP32 对照**：使用该算子的实际 BF16 输入转 FP32，仅计算这个算子。
  用来分离本算子误差和上游传播误差，不替代端到端验收。

## EP1：偏差首先集中在 SwiGLU

以下均为相对 L2；范围覆盖 push/pull 和三步，不跨 tensor 平均。

| 对照 | hotspot | learned |
| --- | --- | --- |
| 相同 gate/up 输入，native SwiGLU vs HF SiLU×up | 0.2798–0.2864% | 0.2758–0.2864% |
| 相同 gate/up 和 dActivation，native SwiGLU backward vs HF | 0.2763–0.2866% | 0.2764–0.2837% |
| 原 HF vs native，expert dW1/dW2 | 0.3519–0.3614% | 0.3438–0.3554% |
| **只替换诊断 HF 激活为融合 SwiGLU** 后，expert dW1/dW2 | **全部精确相等** | **0.000109–0.001204%** |

进一步交叉检查：

- 相同实际输入下，native SwiGLU forward/backward 与独立 `torch_npu.npu_swiglu`
  在四组全部三步中逐元素一致。
- hotspot 使用 native 实际 X、权重和 weighted dY 重放整条专家链：普通 NPU matmul
  加融合 SwiGLU 的八组中间结果/梯度全部与 native 逐元素一致。
  普通 matmul 加 HF 分步激活则在 SwiGLU 开始分歧，并传播至 down、dX、dW。
- learned 存在少量 matmul 累加/舍入残差，不能宣称全部 GMM bitwise 相等。
  native GMM 对“实际输入的 CPU FP32 matmul 最后转 BF16”最坏相对 L2 为 0.004835%。
- 所有捕获的单算子对自身输入 FP32 参考的相对 L2 均不超过 0.168268%。
  SwiGLU forward 对 FP32 后转 BF16 全部精确一致；backward 仅 learned 第三步有一个元素
  不同，push/pull 均如此，不能把 CPU 与 NPU 数学实现宣称为无条件 bitwise 等价。
- native dispatch 的 X 精确匹配原 token；路由加权 dY 和 combine 输出均等于相应
  FP32 加权结果最后转 BF16。这排除了这些样例中 transport 搬运或倍率错误。
- shared expert 的 gate/up/down 三个梯度在 native/HF 路径间全部逐元素一致。
  它们对端到端 FP32 的舍入偏差是双方共用路径的误差，不是 routed expert 替换引入的差异。

### 源码对应

HF zero-limit 的 [激活适配](../../../models/deepseek_v41/adapter/activation.py)
使用 `act_fn(gate) * up`。当输入是 BF16 时，SiLU 输出已经舍入，再做乘法还会舍入。
[融合 kernel 适配](../ops/swi_glu_fusion.patch) 则实例化
`SwigluVectorBF16<bfloat16_t, float, bfloat16_t, ...>` 及对应的 FP32 中间计算 backward。

用 `R` 表示转 BF16，前向舍入边界可表示为：

```text
HF:       R(R(SiLU(g)) * u)
MegaMoe:  R(SiLU(float(g)) * float(u))
```

反向同样不同：HF 乘法先产生 BF16 的 SiLU 上游梯度，再调用 SiLU backward；
融合 backward 在内部完成乘法和导数计算后写回 BF16。
这也解释了为何 dW 上可见偏差，却不意味着 dW matmul 本身是主要偏差源。

## EP2/EP4：剩余差异主要来自局部 dW 的提前舍入

使用相同实际路由和权重，先在 HF 诊断副本启用融合激活，再从其实际 BF16
X、dGate/up、activation、weighted dY 计算 CPU FP32 局部 dW。
各 rank 在同一 EP group 用 FP32 all-reduce，最后转 BF16，与 native owner 的 dW 比较。
表内取该配置所有 rank/step、dW1/dW2 的最大相对 L2。

| 配置 | 原 HF BF16 dW 归约 | 融合 HF + BF16 dW 归约 | 融合 HF + FP32 局部 dW 归约后转 BF16 |
| --- | --- | --- | --- |
| EP2 push | 0.415291% | 0.285989% | 0.001897% |
| EP2 pull | 0.415291% | 0.285989% | 0.001897% |
| EP4 push | 0.460146% | 0.355637% | 0.002936% |
| EP4 pull | 0.462780% | 0.355897% | 0.002701% |

相对于未转 BF16 的 FP32 聚合结果，native dW 最坏相对 L2 为 0.166868%，
与最终 BF16 存储舍入量级一致。最后转 BF16 后的残差每 tensor 最多 31 个元素，
仍保留在报告中；不能声称 CPU 分块 matmul、HCCL 树归约和 owner GMM 的累加顺序相同。

[原 block harness](../examples/mega_moe/deepseek_v41_precision.py) 中 HF 的
`dist.all_reduce(parameter.grad)` 输入为 BF16。
[native backward](../modules/mega_moe/function.py) 的权重梯度 GMM 则使用 owner 收到的
完整 X / dActivation。两者舍入时机不同，实验中的逐级替换分离出了这个影响。

## 验收与复现

已确认的默认判据：每 rank/step/tensor 相对 FP32 的 L2 <= 1%、峰值归一化误差 <= 2%；
有限性、形状、初始权重、路由 IDs、梯度存在性仍严格检查，HF BF16 使用相同判据。
旧 `rtol=0.02, atol=0.002` 明细和硬门槛模式仍保留。

CPU 回归通过 `158 passed, 627 subtests passed`，包括形状/梯度缺失、非有限值、
全零 reference、低 L2 但峰值超界和比例不变性等新增验收边界用例。

正式 NPU ST 已通过 **6/6**：WORLD2 的 EP2 learned/hotspot 和两个 EP1 visual subgroup，
每种配置覆盖 push/pull，全部三步；双方 FP32 门槛与严格结构项均通过。
设备 0/1 的 ownership 记录无 foreign/unresolved PID，耗时 238.82 秒。
对应证据为 `accepted-fp32-st.log` 与 `accepted-st-results/**/deepseek_v41_megamoe.json`。
验收聚合代码整理后，EP2 hotspot push 的三步 ST 再次通过，见 `accepted-fp32-smoke.log`。
本轮变更的 pylint、lizard、文档/测试标记检查通过；检查器的测试路径提示属于 advisory，
实际 CPU/ST 覆盖和诊断 NPU 执行以上述日志为准。

诊断命令、ST 和默认验收模式见 [运行说明](../examples/mega_moe/deepseek_v41_precision.md)。
算子入口只支持 EP1，归约入口要求 WORLD=EP；暂不宣称这些逐算子追踪覆盖视觉或非连续 subgroup。
这些组合的端到端证据仍由既有 16 组同状态矩阵及正式 ST 独立提供。
