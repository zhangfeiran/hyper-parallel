# LongCat Flash BF16 精度诊断

本说明针对客户报告的 x86_64 / 910B2C、BF16 相对 L2 约 0.5% 的问题。
已在本地原始宽度实验中复现这个量级的**参考语义差异**，尚未证明它就是客户现场的原因。
默认 `swiglu_precision="fused"` 保持不变；主验收仍采用同语义的原生 NPU GMM。

## SwiGLU 选项控制什么

`swiglu_precision` 最初为对齐原始 HF 的 BF16 中间舍入而加入，随后按交付要求将默认值设为 `fused`。
该选项同时影响前向和反向的数值语义。

| 选项 | 原生 NPU 参考 | MegaMoE |
| --- | --- | --- |
| `fused`，默认 | `torch_npu.npu_swiglu(projected)`，独立算子启动 | 在 MegaMoE 内核中调用从 CANN SwiGLU/SwiGLUGrad 源码适配的设备函数 |
| `hf` | `gate, up = projected.chunk(2, -1); F.silu(gate) * up` | 仍在融合内核中执行，显式补入与分开执行 SiLU、乘法对应的 BF16 舍入及反向计算次序 |

用 `R` 表示舍入为 BF16，前向差异可概括为：

```text
HF 分开执行：R(R(silu_fp32(gate)) * up)
融合执行：  R(silu_fp32(gate) * up)
```

这是舍入位置的示意；实际 SiLU 指令实现也由所选算子决定。反向还有乘法顺序和梯度中间值的差异。
数学公式相同不意味着 BF16 输出相同。`hf` 不是“关闭 MegaMoE 融合”，也不是权重轴序或 EP packing 选项。

两条 `fused` 路径不是同一个 Python 调用链。MegaMoE 构建锁定的 `ops-nn` 源码是 `v9.1.0.pre`，
并应用内嵌执行、clamp 和可选 HF BF16 补丁；原生接口使用客户已安装的 PTA/CANN 算子。
因此还需要核对客户的软件版本、芯片和实际加载产物，不能据算子名称承诺所有环境逐位相同。
源码入口为 [原生 NPU](../hyper_parallel/models/longcat_flash/adapter/npu.py)、
[MegaMoE 前向设备函数](../hyper_parallel/core/multicore/ops/hyper_mega_moe/op_kernel/worker_kernel.cpp)、
[HF 舍入补丁](../hyper_parallel/core/multicore/ops/swi_glu_hf_bf16.patch) 和
[构建依赖](../hyper_parallel/core/multicore/_build/dependencies.lock.json)。

## 本地诊断结果与边界

本地环境为 aarch64、Ascend910_9382（910C）、Python 3.11.15、Torch/PTA 2.9.1、CANN 8.5.0。
逐项数值、校验脚本和 native 哈希、所有 rank 的终态汇总保留在[精简结果记录](longcat-flash-precision-diagnosis-results.json)。
诊断保留 H6144/I2048、TopK12，每 rank 8 个真实专家、4 个 identity 份额；不通过缩小宽度规避差异。
EP1 语义实验使用 E8/Z4、T2048；EP4 边界实验使用 E32/Z16。

同一份输入、参数、路由和上游梯度的 EP1 对照如下。相对 L2 的百分数是报告原值乘 100：

| 比较 | 输出相对 L2 | dX 相对 L2 | 专家梯度及严格逐元素结果 |
| --- | --- | --- | --- |
| 原始 HF 重复执行 | 0 | 0 | 全值一致 |
| 原生 GMM `fused` 对原始 HF | 约 0.4508% | 约 0.4963% | Gate/Up 约 0.3568%，Down 约 0.3576%；严格门限失败 |
| MegaMoE `fused` 对原生 GMM `fused` | 0 | 0 | 全值一致 |
| MegaMoE `hf` 对原生 GMM `hf` | 0 | 0 | 全值一致 |
| 原生 GMM `hf` 对原始 HF | 0 | 0 | Gate/Up 相对 L2 约 0.003751%，仍有 1 个元素未过严格门限 |

最后一行说明兼容激活舍入后，逐专家 `F.linear`、GMM 和专家内 token 的累加顺序仍可能造成权重梯度差异。
不能将 `hf` 描述为整个 MoE 与 HF 逐位相同的保证。
额外的原始宽度 identity 投影实验隔离融合 SwiGLU 的固定操作数，MegaMoE 与原生融合路径输出及梯度一致。

EP4 的 13 组路由边界实验覆盖正常路由、不同真实 slot 数、全真实、全 identity、热 owner、空源 rank、
全空输入、变长 token 及不同激活尺度；4 个 rank 均全值一致。
同实例多次 forward 后统一 backward、两层不同权重共享资源，以及非重入 checkpoint 的 4 组组件实验也在
4 个 rank 上全值一致。它们不构成客户芯片或 FSDP/完整 Trainer 的验收。

## 客户现场区分实验

推荐使用[自动诊断入口](../examples/longcat_flash_repro/diagnose.py)，无需逐条复制以下底层命令。
客户在已有 Python3.11/Torch/PTA 环境中预留四个逻辑 NPU，设置 `CANN_ENV` 为本机 CANN 环境脚本、
`LONGCAT_DIAG_OUTPUT` 为仓库外尚不存在的目录，然后执行：

```bash
python examples/longcat_flash_repro/diagnose.py \
  --devices 0,1,2,3 \
  --cann-env "$CANN_ENV" \
  --output "$LONGCAT_DIAG_OUTPUT"
```

设备列表必须显式传入并替换为现场已预留的逻辑设备。入口会切换到所属仓库根目录，分别激活客户指定的 CANN 和
当前仓库的 native 环境；检查 editable 来源、主机 ELF 架构、算子接口和 Git 历史，记录实际依赖与芯片信息。
首次更新到当前功能分支、尚无匹配 native 产物时加 `--build`，显式编译 `ascend910b,ascend910_93` 并以
`--no-deps --no-build-isolation` 安装本仓库 editable 包。客户 Torch/PTA 和其他依赖沿用原环境。

默认顺序是 EP1 → EP4 fused。EP1 出现完整的数值失败报告时仍继续 EP4；若报告不完整或运行异常，则停止并保留日志。
EP4 fused 完成但数值失败时自动补跑 EP4 hf，hf 通过也不会掩盖 fused 的失败。运行错误不自动重试。
可用 `--ep4-hf always` 总是补跑 hf，或 `--ep4-hf never` 关闭补跑。
只有一个已预留设备时使用 `--stage ep1 --devices 0`；单独运行四卡用 `--stage ep4`。
H6144/I2048、T2048、TopK12 和每卡 8/4 的规模由入口固定。

输出目录包含：

- `summary.txt`：中文结论、逐项相对 L2、失败位置和后续定位方向。
- `summary.json`：各阶段命令、真实退出码、完整 rank 判定、源码标识和机器可读结论。
- `environment.json`、`preflight.log`、`prepare.log`：实际环境、native 选择与准备过程；`--build` 另有 `build.log`。
- `ep1.log`、`ep1/rank0.json`：单卡六项比较。
- `ep4-fused.log`、`ep4-fused/rank0.json` 至 `rank3.json`：四卡 push/pull 比较。
- 补跑时的 `ep4-hf.log` 与 `ep4-hf/rank*.json`。

最终退出码为 0（所有所选主验收通过）、1（数值失败）、2（准备/运行异常或报告不完整）、130（中断）。
中途汇总标记 `RUNNING`，不会将只完成单卡的整个诊断误报为 PASS。主验收需要完整 case、全部五类张量、
全部 rank、当前源码库选择和已关闭的 SHMEM；原始 HF 参考差异单独保留。
客户回传上述日志、JSON 和汇总，即可区分参考语义、单卡计算、分布式路径或项目接入问题。
这些是固定种子的组件实验；现场原始失败批次、checkpoint 与完整模型需要继续在客户项目中对齐。

以下为已有匹配环境时的底层命令，便于单独重跑或在客户原启动器中调用。

先按[当前分支指南](../examples/longcat_flash_repro/README.zh-CN.md)在客户环境重编译、核对 editable 安装来源及加载产物。
不要安装或替换客户自建 Torch/PTA。正常组件主验收继续运行指南里的 EP4 独立历史前端命令；
两侧选择相同的 `swiglu_precision`。

另外预留一个逻辑 NPU，用新的输出目录运行以下语义诊断。变量由客户按现场设备和输出目录设置：

```bash
export ASCEND_RT_VISIBLE_DEVICES="$LONGCAT_DIAGNOSTIC_DEVICE"
export HCCL_DETERMINISTIC=true OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tests/torch/models/longcat_flash/_diagnose_longcat_swiglu.py \
  --backend push --ep 1 --tokens 2048 --real 8 --zero 4 \
  --hidden 6144 --intermediate 2048 --top-k 12 \
  --routing-backend triton --swiglu-precision fused \
  --initial-token-capacity 24576 --output "$LONGCAT_DIAGNOSTIC_OUTPUT"
```

此入口需要已可用的 Triton-Ascend 和仓库中的历史提交，按 EP 比例裁剪专家数量，不缩小 H/I。
它明确切换并记录 `hf` 和 `fused` 两种语义，生成 `rank0.json`，记录芯片、核数、CPU 架构、Torch/PTA、
CANN 版本及 native 产物哈希。退出码要求 HF 重复执行、MegaMoE 对同语义原生 GMM、固定操作数融合 SwiGLU 均通过。
原生 GMM 对原始 HF 的失败明细单独保留，不影响上述同语义主验收。

- 若 `Mega_fused_vs_native_GMM_fused` 通过，而 `native_GMM_fused_vs_original_HF` 约 0.5%，优先检查客户参考是否仍为分开的 SiLU/乘法。
- 若同语义的 MegaMoE/GMM 也失败，参考语义不足以解释差异，应按失败张量、首批失败位置和版本信息继续定位芯片/算子/路由路径。
- 若现场目标必须对齐原始 HF 激活语义，可显式选择 `hf`；性能比较两侧同步选择它，并单独保留整个 MoE 的 HF 失败明细。

## 严格门限与失败报告

组件验收修正为：**每个 rank、每个 transport 的所有张量均通过严格门限，进程才成功退出**。
此前 worker 记录 `strict_pass`，却仅据 `component_pass` 决定退出码，可能错误放行；该问题不等同于计算内核错误，
此前已逐项确认全值零误差的报告仍然有效。

严格门限保持 `abs(actual-reference) <= 2e-5 + 0.05*abs(reference)` 且 `relative_l2 <= 0.05`，没有放宽。
`component_pass` 保留为另一套诊断门限，它与严格门限在所有数值范围内并不互相包含。
即使相对 L2 为 0.5%，也必须检查近零元素是否过逐元素门限，不能直接宣称通过。
新报告还检查 shape、dtype 和非有限值，并记录首批失败 flat index 与两侧数值；范数在 CPU 真正以 FP64 累加，
避免 NPU 不支持 FP64 时静默转成 FP32。
