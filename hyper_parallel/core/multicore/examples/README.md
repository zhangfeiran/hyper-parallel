# MoE Transformer 的三个 MegaKernel 接入示例

本示例使用 **MoE Transformer 最小网络**，在同一条训练路径中展示
`MegaGate`、`MegaMoeExperts` 和 `HyperMegaMhc`。网络使用随机权重、简化的 GQA attention、
`sqrtsoftplus` TopK 路由、路由专家与共享专家，以及四流 mHC。
它是用于理解接口替换和比较完整 optimizer step 的自定义随机网络，不对应已发布模型或完整生产训练方案。
真实模型的 MLA/DSA、VLM、Engram 和并行配置请参考
[DSV4.1 模型实现](../../../models/deepseek_v41/modeling_deepseek_v41.py)及其
[训练配置](../../../../examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml)。

网络包含跨 attention、MoE 和 decoder 层的四流 mHC 状态传递，
common 与 Mega 使用相同的网络结构、初始权重、输入和优化器配置。

| 文件 | 内容 |
| --- | --- |
| [model.py](model.py) | 配置、attention、decoder 与整网组装，串联三个组件和跨层状态 |
| [moe.py](moe.py) | 普通 EP 的 `NativeExperts`、`MegaMoeExperts` 构造及等权重切换，保留共享专家 |
| [gate.py](gate.py) | 同一个 `Gate` 中并列实现 `native_forward` 和 `mega_forward`，共用参数与返回接口 |
| [mhc.py](mhc.py) | 同一个 `MhcBoundary` 中并列实现 `native_forward` 和 `mega_forward`，包含状态初始化与末层合并 |
| [benchmark.py](benchmark.py) | common/Mega 第一步精度对齐和完整优化器步计时 |
| [run_benchmark.sh](run_benchmark.sh) | 激活环境、检查源码导入路径并启动固定 EP8 任务 |

阅读时先看 `model.py` 的组件组装，再到 `moe.py`、`gate.py`、`mhc.py` 分别对照替换范围。
本示例将普通实现称为 native，对照结果 JSON 继续使用 `common`；Mega 路径调用对应的公共受管模块。
其中 native mHC 在 NPU 上使用 Omni Post＋Sinkhorn 及其反向，CPU Torch 公式用于语义检查。
这与底层“native C++/Ascend kernel”的命名不同，后者属于 Mega 实现的内部制品。

## 1. 安装和启动

需要配套的 Torch/torch_npu、CANN toolkit/ops **9.1.0 或更高版本**以及 Ascend NPU。
NPU common 还依赖与这些版本和目标芯片匹配的 `omni_training_custom_ops` 注册包及 vendor OPP，
用于 MhcPost、Sinkhorn 及其反向。缺少注册包或任一前反向接口会直接报错，不回退到 Torch 性能基线。
先在仓库根目录安装源码，再构建和激活与当前 Python、Torch、CANN、SoC 匹配的 native 制品：

```bash
source /path/to/cann/set_env.sh
python -m pip install -e .
bash build.sh --multicore on --strict on --jobs 24
```

也可仅构建组件，具体 SoC 参数和依赖见[构建说明](../docs/build.md)。启动器会检查解释器导入的
`hyper_parallel` 是否属于当前 worktree，避免误用另一个 checkout 的源码。

单机 8 个 NPU rank，默认同时启用三个 kernel：

```bash
CANN_ENV_FILE=/path/to/cann/set_env.sh \
OMNI_ENV_FILE=/path/to/omni/vendor/bin/set_env.bash \
bash hyper_parallel/core/multicore/examples/run_benchmark.sh \
  --mega-kernels moe gate mhc --dispatch-mode push \
  --output output/moe_transformer_mega_push.json
```

改用 pull 时不要传 push 容量因子：

```bash
CANN_ENV_FILE=/path/to/cann/set_env.sh \
bash hyper_parallel/core/multicore/examples/run_benchmark.sh \
  --mega-kernels moe gate mhc --dispatch-mode pull \
  --output output/moe_transformer_mega_pull.json
```

`--mega-kernels` 只改变被比较的 Mega 路径；common 使用普通 EP 专家、Torch 路由和
Omni Post＋Sinkhorn mHC，其 predictor 和 block RMSNorm 算术与 DSV4.1 common 一致。
可传 `--mega-kernels moe`、`--mega-kernels gate`、`--mega-kernels mhc` 或任意组合，
逐项观察收益。两条路径始终保留四流 mHC，确保对照网络的残差结构一致。

默认工作负载为 TP1/EP8、batch size 1、每 rank 4096 token、H5120、两层 decoder、
48 个专家、TopK8、专家中间维度 1792、共享专家中间维度 1792、词表 32000。
激活和普通权重使用 BF16，mHC 预测器参数与映射保持 FP32。默认预热三步、测量五步。
可用 `--warmup-steps` 和 `--measured-steps` 调整步数，形状修改集中在 `MoeTransformerConfig`。

启动器也支持多机，但总 rank 数必须为 8。以两节点、每节点 4 rank 为例，在两端分别设置
`NODE_RANK=0` 和 `NODE_RANK=1` 后运行相同命令：

```bash
NNODES=2 NPROC_PER_NODE=4 NODE_RANK=0 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
CANN_ENV_FILE=/path/to/cann/set_env.sh \
bash hyper_parallel/core/multicore/examples/run_benchmark.sh \
  --mega-kernels moe gate mhc --output output/moe_transformer_mega_ep8.json
```

使用可被所有节点访问的 rank-zero IP。默认 SHMEM 端口为 `MASTER_PORT + 1000`，可用
`SHMEM_HOST`/`SHMEM_PORT` 单独指定。`PYTHON_BIN`、`NATIVE_ENV_FILE`、
`SHMEM_LIB_DIR` 可选择其他已构建环境，不能默认混用不同 Torch/CANN 构建的制品。
若已在当前 shell 激活正确的 Omni vendor OPP，可省略 `OMNI_ENV_FILE`；否则显式指定它。
启动器通过选定的 `PYTHON_BIN -m torch.distributed.run` 启动 worker，保持解释器与源码路径检查一致。

## 2. 先把已有 MoE 的路由与专家执行分开

典型实现中，路由得到全局专家 ID 和分数，再执行 dispatch、两次 grouped matmul 及 combine：

```python
# Before: keep the router outside the expert execution path.
topk_weights, topk_ids = router(hidden_states)
routed, mapping = token_permute(hidden_states, topk_ids)
routed = all_to_all_dispatch(routed)
routed = grouped_swiglu_experts(routed)
routed = all_to_all_combine(routed)
output = token_unpermute(routed, mapping, probs=topk_weights)
```

接入 `MegaMoeExperts` 时，替换的是上述专家执行段，Router 和共享专家仍由模型负责。
构造一次模块，重复 forward；不要每步重新构造。选定本 rank NPU 并初始化 HCCL 后：

```python
import torch
import torch.distributed as dist
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    max_local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=48,
    top_k=8,
    ep_size=dist.get_world_size(),
    ep_group=dist.group.WORLD,
    dispatch_mode="push",
).to(device="npu", dtype=torch.bfloat16)

# After: ids and scores are supplied by the existing router.
hidden_flat = hidden_states.reshape(-1, 5120)
counts = torch.bincount(topk_ids.reshape(-1), minlength=48).to(torch.int32)
routed_output = experts(
    hidden_flat, topk_ids, topk_weights, tokens_per_expert=counts,
)
output = routed_output + shared_expert(hidden_flat)
```

接口变化对应如下：

| 原实现 | MegaMoe 接口约定 |
| --- | --- |
| token 数 `T`、隐藏维度 `H` | 输入展平为 `[T, H]`，BF16、同一 NPU |
| Router TopK | `topk_ids: INT64[T, K]`，使用全局编号，每行 ID 不重复 |
| Router 分数 | `topk_weights: FP32[T, K]`，保留计算图，由反向更新 gate |
| 各专家 token 计数 | 可选 `tokens_per_expert: INT32[E]`，同设备、连续、精确的本 rank histogram |
| dispatch/GMM/SwiGLU/combine | 一次 `experts(hidden, ids, weights, ...)`，返回原 token 顺序的 `[T, H]` |
| 共享专家 | 在 `MegaMoeExperts` 外单独计算并相加 |
| EP 分组 | `ep_group` 及其大小 `ep_size`；专家按组内 rank 顺序连续分配 |

已有精确计数可直接复用；否则省略 `tokens_per_expert`，让模块内部统计。
所有 EP rank 必须按相同顺序进入正反向。不要根据本地是否接收 token 来跳过调用。
原模型若有 SwiGLU clamp，需把相同的 `swiglu_limit` 传给 MegaMoe，并在 common 路径保留同样的计算。
本最小示例使用不带 clamp 的 SwiGLU，不改变已有业务模型的激活定义。

### 迁移已有 gate/up/down 权重

本地专家数为 `L=E/EP`。以常见的 PyTorch Linear 布局
`gate/up: [L, I, H]`、`down: [L, H, I]` 为例，MegaMoe 需要
`gate_up: [L, H, 2I]`、`down: [L, I, H]`，前半段为 gate、后半段为 up：

```python
# Run once when replacing an owned expert module, before constructing the optimizer.
with torch.no_grad():
    experts.gate_up_weight.copy_(torch.cat(
        (old_gate_weight.transpose(-1, -2), old_up_weight.transpose(-1, -2)), dim=-1,
    ))
    experts.down_weight.copy_(old_down_weight.transpose(-1, -2))
```

传入的是当前 EP rank 对应的本地专家分片；若原权重包含全部专家，应先按 EP 组内 rank 切片。
替换完成后，优化器只管理新参数；不要把旧专家参数也继续注册到优化器。这里的基准从同一份
MegaMoe 初始权重反向转换成 common 布局，保证两条路径逐元素对齐。
具体实现见 [moe.py](moe.py)：`NativeExperts.forward` 显式执行 permute、普通 EP 专家和 unpermute，
`MoeTransformerBlock` 构造 `MegaMoeExperts` 并通过相同的 `experts(...)` 接口调用。
`use_native_experts` 在构建优化器之前将同一份初始权重转换为普通 EP 布局，
释放原 MegaMoe 实例后替换 `self.experts`；benchmark 只选择路径，不再持有组件实现。

### 外部 FSDP/参数管理器持有权重

需要由外层模块持有参数时，构造 `create_parameters=False`，每次 forward 显式传入当前已恢复的完整权重：

```python
external_experts = MegaMoeExperts(
    max_local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
    num_experts=48, top_k=8, ep_size=ep_size, ep_group=ep_group,
    create_parameters=False, dispatch_mode="pull",
)

# Build from the currently unsharded parameters inside the owning module's forward.
gate_up = torch.cat((gate_weight.transpose(-1, -2), up_weight.transpose(-1, -2)), dim=-1)
down = down_weight.transpose(-1, -2)
output = external_experts(
    hidden_flat, topk_ids, topk_weights, expert_weights=(gate_up, down),
)
```

`gate_weight`、`up_weight`、`down_weight` 必须保留 autograd 关系，不要 `.detach()`、搬到 CPU，
或缓存某次 all-gather 的结果。非连续权重会在接口内部转为连续布局，梯度仍回到原参数，但每步可能产生复制。
FSDP 的 all-gather/reshard 边界须包住拥有这些参数的 forward；上述接口本身不会替调用方恢复分片参数。
本例仅验证 TP1/EP8，不代表任意 FSDP/TP/CP 组合都可直接套用；真实并行接入见
[DSV4.1 适配说明](../README.md#dsv41-的-tpsp-与-cp-集成)。

## 3. 用 MegaGate 替换路由计算

对应源码集中在 [gate.py](gate.py)：`Gate.native_forward` 展示普通算子序列，
`Gate.mega_forward` 调用 `MegaGate.forward`；二者共用同一组 weight、bias 和返回顺序，
由 `use_mega_gate` 选择。`moe.py` 只消费路由结果，不再实现另一份 Router。

只接 MegaMoe 时可以继续使用自己的 Router。进一步接入 MegaGate，需要先确认评分函数、
correction bias、TopK 归一化和 scaling 与原网络一致。
本例两条路径都采用 `sqrt(softplus(logits))`；bias 仅参与选择，分数从不加 bias 的 scores 中取出。
`K>1` 时对选中分数归一化，最后乘 `routed_scaling_factor`。

```python
from hyper_parallel.core.multicore import MegaGate

gate = MegaGate(
    hidden_size=5120, num_experts=48, top_k=8,
    scoring_func="sqrtsoftplus", routed_scaling_factor=1.0,
).to(device="npu")

# Preserve the source projection and correction bias when replacing a trained router.
gate.weight = old_gate.weight
# gate.bias = old_correction_bias  # Assign an FP32 Parameter if the source has one.
gate.bias.requires_grad_(False)

# Before: linear -> scores -> correction -> TopK -> gather -> normalization.
# After: the return order is logits, weights, ids.
logits, topk_weights, topk_ids = gate(hidden_states)
```

输入是 `[batch, sequence, H]`，后两个输出展平为 `[T, K]`。
权重和 bias 的设备应随模型一起迁移；correction bias 保持 FP32。
本例没有实现负载均衡 bias 更新器，因此冻结它；生产训练应保留原模型的 bias 更新策略。
如需 auxiliary routing loss，可继续使用返回的 `logits`。

当前 native Route/RouteGrad 只覆盖 NPU 上的 `sqrtsoftplus`；`softmax`、`sigmoid` 和 CPU 输入使用
Torch 实现。不能仅为启用 native 而把已有 Qwen/其他模型的 softmax Router 改成 sqrtsoftplus，
否则会改变网络语义。VLM 可构造 `vision_enabled=True`，再传入同设备连续 BOOL
`image_mask: [batch, sequence]`；本例的纯文本基准不使用视觉 bias。

## 4. 用 HyperMegaMhc 替换已有四流 mHC

mHC 改变残差结构，不能对普通 `x + block(norm(x))` 只换一个函数名。
本例将每个 attention 和 MoE 之间的边界表示为五个状态量：

| 状态 | 形状与类型 |
| --- | --- |
| 前一个 block 输出 | `[..., H]`，BF16 |
| residual streams | `[..., 4, H]`，BF16 |
| previous pre / post | 各 `[..., 4]`，FP32 |
| previous residual mix | `[..., 4, 4]`，FP32 |

对于已经具有同样 shifted 协议的 Torch mHC，替换前后为：

```python
# Before: an ordinary Torch implementation computes all five outputs.
residual, pre, post, mix, block_input = torch_mhc_boundary(
    previous_output, residual, pre, post, mix,
)

# After: construct the boundary once; preserve/convert the source parameters first.
from hyper_parallel.core.multicore import HyperMegaMhc
mhc = HyperMegaMhc(hidden_size=5120).to(device="npu")
residual, pre, post, mix, block_input = mhc(
    previous_output, residual, pre, post, mix,
)
```

`block_input` 使用 **previous pre** 和刚更新的 residual；本次返回的 pre/post/mix 供下一个边界使用。
不能在当前边界错误地使用 next pre，也不能每层重新创建初始状态。真实 decoder 中的状态传递如下：

```python
# The helpers make bootstrap and the final pending update explicit.
from hyper_parallel.core.multicore.examples.mhc import (
    initial_mhc_state, finish_mhc_state,
)

state = initial_mhc_state(embedding_output)
for layer in layers:
    residual, pre, post, mix, x = layer.attention_mhc(*state)
    attention_output = layer.self_attn(x, position_ids)
    residual, pre, post, mix, x = layer.ffn_mhc(attention_output, residual, pre, post, mix)
    state = (layer.mlp(x), residual, pre, post, mix)
hidden_states = finish_mhc_state(state)
```

首层以复制的四份 embedding、单位 residual mix、零 post 和 one-hot pre 启动；
末层必须将最后一个 MoE 输出按 post/mix 写回 residual，再用最后的 pre 合并，之后才执行最终 RMSNorm 和 LM head。
[参考实现](mhc.py)保留三条明确的路径：`native_forward` 在 NPU 上使用 DSV4.1 同款 Omni
Post/PostGrad、Sinkhorn/SinkhornGrad，预测器使用归一化后再投影的 Torch 算术；
`mega_forward` 使用 HyperMegaMhc；`reference_forward` 保留可阅读的 Torch 公式，只用于精度参考。
最终 pending output 合并也由两条 NPU 路径共享 Omni Post。
Torch 精度参考在 NPU 上遵循 CANN 的 post 累加顺序、先投影再做 RMS 缩放，
以及 block input 先写入 BF16 再做 RMSNorm 的中间精度。参考投影仅在该次 matmul 期间开启 HF32，
随后恢复原设置。CPU 路径保留数学公式，便于检查算法语义。
两种实现仍有有限精度的差异，可能改变后续 TopK；因此默认按组件验证 mHC 的前反向，
并单独检查同时开启三个 kernel 的整网训练步。严格的整网逐元素对齐模式继续保留，见结果说明。

参数布局为 `phi: FP32[24, 4H]`、`alpha: FP32[3]`、`bias: FP32[24]`、
`norm_weight: BF16[H]`。从已有模型接入时应复用或转换这些参数，并保留它们的梯度。
不要对整个模型执行 `.to(dtype=torch.bfloat16)`，它会把 FP32 predictor 也转换掉；
本例在 BF16 默认 dtype 下构造普通权重，由 HyperMegaMhc 显式保留 FP32 参数，再只迁移 device。

当前训练接口固定四流、二十轮 Sinkhorn，H 为 128 的倍数且不超过 5760；
反向 token 数须不小于 `max(40, AIV 核数)`。这与 MegaMoe 动态 token 支持空 rank 的范围不同，
不能据此假设整个三 kernel 网络支持空输入训练。

## 5. 动态 token、容量和关闭

示例以 `max_local_num_tokens=4096` 接入 MegaMoe 动态行数接口，基准输入仍固定为 4096。
实际各 rank 行数可以不同，且不超过上限；固定行数可改用 `local_num_tokens=4096`，二者互斥。
容量按上限对齐预留，不会因为当前输入变短而自动缩小。

push 默认 `initial_capacity_factor=1.25`、`capacity_growth_factor=1.25`，路由超过接收容量时集体扩容，
不丢 token。可在启动命令增加 `--initial-capacity-factor 2 --capacity-growth-factor 1.25`。
pull 不接受这两个数值参数。两种模式都需要为热点实际接收量保留普通 HBM；
容量因子不是整个模型显存占用的倍数，也不等价于 OOM 保证。

相同配置的串行 decoder 层在首次 forward 前共享执行资源，参数、梯度及待反向激活仍各自独立：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts for layer in model.layers)

# After all backwards and queued device work have completed, close in the same rank order.
model.close()
dist.destroy_process_group()
```

不要在并发线程中提交同一资源组，不要在仍有待反向图时关闭。
集成代码直接使用受管模块即可，无需自己初始化/销毁 SHMEM。
更完整的动态行数、EP 子组、容量扩展和自动清理边界见[组件使用指南](../README.md)。

## 6. 如何读取结果

默认 `--accuracy-mode components` 分三部分验收，精度阈值保持 `rtol=0.02, atol=0.002`：

1. 用完全相同的输入、参数和 Torch 精度参考，分别检查 Omni common 与 HyperMegaMhc
   的五个输出、五组输入梯度和四组参数梯度。
2. 两条整网路径都使用相同的 Omni common mHC，Mega 路径保留所选 MegaMoe/MegaGate，严格比较初始参数、
   logits、loss、全部可训练参数梯度和更新后参数。这样可排除 mHC 舍入误差触发不同 TopK 的干扰。
3. 恢复所选全部 Mega kernel，再执行完整训练步，检查 loss/梯度有限、dense/专家参数确实更新。
   此后预热与测量也保持所有选中的 kernel 开启。

第二项不等价于三个 kernel 同时开启时的整网逐元素精度验收。
JSON 的 `comparison.accuracy.scope`、`mhc_component` 和 `selected_kernels_training_step` 明确记录验证范围。
如果没有选择 mHC，则直接严格比较所选 kernel 的整网训练步。
冻结的 correction bias 不参与梯度检查，但仍参与初始/更新后参数对齐。
基准按 dtype 分桶同步 BF16/FP32 dense 梯度，并为 common EP 的本地专家梯度补齐 DTensor
`main_grad` 布局，交给混合精度优化器更新 FP32 主参数和保留布局的 Adam 状态。

需要严格的自由路由整网对齐时，可传 `--accuracy-mode full_step`。
该模式直接让所选全部 kernel 同时参与 logits、梯度和更新后参数比较，超阈值即失败，不进入性能计时。
随机 mHC 网络中，合法的舍入差异可能改变专家选择，所以这项检查比组件精度检查更严格，不保证通过。
默认模式不会把该项标为通过。

随后分别预热和计时 common、Mega，计时包含 forward、backward、dense 梯度同步和 AdamW update，
每步取 rank 最大耗时。JSON 的 `common_mhc_backend="omni_post_sinkhorn"` 和
`mhc_precision_reference="torch"` 区分计时基线与精度参考，另记录 `architecture`、`mega_kernels`、形状、通信模式、精度结果、
验证步和 steady-state 分布：

- `comparison.performance.common_median_ms` / `mega_median_ms`：各路径完整 optimizer step 中位数。
- `mega_over_common`：Mega/common，越小越好；加速比可按其倒数计算。
- `mega_reduction_percent`：相对 common 的耗时下降百分比。
- `backends`：预热、测量明细和各路径检查结果；`validation_optimizer_step_index` 标记验证步编号，
  `validation_optimizer_step_ms` 为该次验证步耗时，默认组件模式包含额外的严格对照步。

这是两层随机网络的串行 A/B 测试。首次更新后的参数轨迹和后续路由可能逐渐分离，
测量顺序固定为 common 后 Mega，未锁定每步路由，也未采样整个进程的物理 HBM。
结果不能直接外推为真实模型整网加速比。需要稳定性能结论时，应进一步固定路由/环境、
交替运行顺序、多次重复，并分别记录 kernel 时间与完整 optimizer step 时间。
