# 稠密 MegaFFN 训练接入

目标是在注意力之后执行可训练的 BF16 稠密 SwiGLU FFN，通过独立模型和优化器的对齐验收，
并取得相对现有打包实现 `SwiGLUMLP` 的实测收益。稠密 FFN 不涉及路由器、专家归属或
dispatch/combine 通信。

## 语义与训练接口

`frontend/examples/dense_ffn.py` 描述打包的 gate/up Matmul、稠密 SwiGLU 和 down Matmul。
`DenseSpec` 绑定符号维度，不依赖 EP 拓扑。降级过程从 SSA 推导依赖、绑定和保守的中间值生命周期。
必须校验规范原语的身份，并由可信的执行提供方完成准入。

```python
from hyper_parallel.core.multicore import MegaFFN

ffn = MegaFFN(hidden_size=1024, intermediate_size=4096).to("npu")
output = ffn(tokens)  # 连续存储的 BF16 张量，形状为 [..., 1024]
output.float().square().mean().backward()
```

参数布局为按列打包的 gate/up `[H, 2I]` 和 down `[I, H]`。可使用常规的 `nn.Module`、
autograd、优化器、state dict，以及 checkpoint/重计算接口。归一化和残差组合保留在解码器边界。
初始稠密契约不包含偏置和 TP 集合通信。

调用方持有的本地权重通过 `MegaFFN(..., create_parameters=False)` 和
`ffn(tokens, weights=(local_gate_up, local_down))` 接入。借用权重不被缓存或复制。
autograd 状态归每次调用所有；关闭后拒绝新的前向调用，但已有调用的待执行反向仍然有效。

## 现有 Trainer 替换机制与检查点

`modules/mega_ffn/adapter.py` 使用现有的 `@module_replacement` 工厂。
在分片和创建优化器之前执行替换，并使用常规的权重映射列表：

```python
import torch

from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)

rules = [ModuleReplacementSpec(("model.layers.*.mlp",), MegaFFNAdapter, SourceMLP)]
plan = compile_module_replacements(model, rules)
model, mapping = apply_module_replacements(model, plan, weights_mapping=[])
model.to(dtype=torch.bfloat16, device="npu")
```

工厂仅打包一次无偏置 SiLU 的 gate/up/down 权重，在迁移到目标设备之前保留源 dtype、设备和全部训练标记。
检查点变换精确逆转拼接和转置。共享投影参数，以及 gate/up 不同的 `requires_grad` 策略会被拒绝。
双卡 BF16 FSDP/SGD 组件门禁已通过权重更新、重计算和精确检查点恢复检查。

## 原生执行与证据范围

执行提供方为原生 Torch Matmul 和 NPU 打包 SwiGLU，具有精确的前向、反向 dispatcher schema。
生成的 C++ 前向/VJP 适配器遵循 SSA，包括转置处理和反向梯度累加。
构建器对生成源码、编译器身份、Torch 版本/ABI 和库哈希进行封存；加载器在绑定算子之前校验这些信息。

原生适配器在前向、反向各使用一次 Python/native 调用，将执行提供方的内核加入当前流。
模块默认使用此主机流后端。`Program.compile()` 生成源码产物，不构建二进制。

显式选择的 `resident_tiles` 候选后端将可按行拆分的 BF16 SSA 降级为配对的 AIC/AIV 私有队列。
每个 Cube 持有自己的 token tile，两条 Vector 通道都参与激活阶段的汇合，包括单行尾部的空通道。
事件发布显式同步标量和 DMA 流水线。消费者使事件缓存行失效，并在每次轮询时通过 volatile 指针
重新读取 GM 值，即使另一个核仍在生成该值也如此。CPU 调度检查器与原生内核中的有界生产者预取
遵循相同的队列顺序。单个混合内核执行前向 Matmul、以 FP32 计算并存为 BF16 的 SwiGLU，以及后续 Matmul。
通用的行内分叉和转置右权重可直接准入，不依赖 FFN 函数体匹配器。
跨行混合、左转置和不支持的存储布局会在执行之前被拒绝。

```python
from pathlib import Path

from hyper_parallel.core.multicore import MegaFFN
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig

execution = DenseExecutionConfig(
    backend="resident_tiles", cann_root=Path("/path/to/cann"), soc="Ascend910B3"
)
ffn = MegaFFN(1024, 4096, execution=execution).to("npu")
```

现有替换 API 接受 `context={"mega_ffn_execution": execution}`。
核数、workspace 字节数和原始 `TCubeTiling` 大小由选定的 SDK 提供，不沿用其他 SDK 或设备的假设。
构建器封存源码、SDK 头文件/脚本/库、设备编译器、TorchNPU ABI，以及打包二进制和主机库。
缓存由调用方持有，构建器不修改原生 vendor 安装。启动器延迟注册精确的二进制，检查注册失败，
并通过 TorchNPU 任务队列按流顺序提交。

当前反向通过独立生成的原生执行提供方 VJP，读取本次 resident 调用的确切打包值和激活缓存。
权重梯度保留完整的 token 维度收缩，不对已经舍入为 BF16 的 tile 局部结果求和。
中间值归每次调用所有，在待执行反向结束之前保持完整。
指针表、事件计数器、workspace 和溢出标记存储使用有界的逐流临时缓存，在同一设备的不同层和
token 形状间共享。指针和事件容量只增长到高水位，各视图保留对应描述符的精确长度。
每次启动都更新当前张量地址，并在该流上重置汇合计数。不同流使用独立槽位；重叠的主机调用和
超出两个缓存槽位的流使用私有临时存储。可执行对象以弱引用登记所有权；最后一个对象关闭时
释放临时缓存，已入队的原生启动仍持有其存储。
checkpoint 和关闭操作使用与主机后端相同的模块生命周期。缓存元数据具有初始化事件，
所有设备存储都在启动流上登记。

原生构建、打包、加载，真实主机 SDK tiling、CPU 队列回放和保存状态协议测试均已通过。
单卡 resident 组件矩阵也已通过 BF16 输出与梯度比较，覆盖空输入、尾部输入、checkpoint、
关闭后待执行反向和两条非默认流。双卡 BF16 FSDP/SGD 组件门禁同样通过。
独立的全模型数值与性能验收仍需单独完成。内核临时存储复用已通过 CPU 协议测试和设备用例，
覆盖借用权重更新、重复使用流、私有存储回退，以及反向之前关闭模块。
扩展组件矩阵和双卡 FSDP/SGD 回归均已在共享临时存储的实现上通过。
resident 反向、中间缓冲区复用和设备调优仍待完成；反向仍需读取的前向缓冲区不会发生别名复用。
CPU 原生执行仅作为参考。

已安装的 `npu_ffn` 接口仅允许门控激活用于 FP16 推理，因此不能作为 BF16 训练的快捷路径。
主机编译成功和 CPU 梯度正确不代表 NPU 数值正确或具有性能收益。

## 最终验收要求

`examples/mega_ffn/qwen_dense_model.py` 在完整稠密语言模型中，将 FFN 接在现有 Qwen GQA 注意力和
归一化之后。它通过同一替换机制支持常规三投影模型（`common`）、现有打包模型（`packed`）
和 AST 模型（`mega_ffn`）。

1. NPU 前向、dX 和两组权重梯度正确，覆盖尾部、空 token 和重计算。
2. 在真实设备上验证 FSDP/本地权重生命周期，以及检查点保存与恢复。
3. 验证独立训练轨迹、FP32 主参数和优化器矩。仅对齐一次初始状态，后续运行之间不再重新同步。
4. 在全新进程中执行原生/原生对照，以及相对两种基线的 ABBA 比较；测量完整优化器步和峰值内存，
   精确记录源码、框架和设备进程归属。
5. 实现常驻 worker/tile 流水线、缓冲区复用，以及 RAW/WAR/WAW 依赖，并取得实测收益。
   仅减少 AST 代码量或编译主机适配器不能满足此项要求。

长时间设备任务使用用户的 `npu_wait_and_run.sh`，保留其锁、健康检查和启动前最后一次空闲复查。

## 可复现的验收入口

基准分别推进独立的模型和使用 FP32 主参数的 AdamW 优化器，记录 logits、loss、梯度、BF16 参数、
FP32 主参数、一阶矩、二阶矩和精确步数。所有初始状态必须逐值一致，后续不再同步。
失败步骤的记录保留在 JSON 中，并终止运行。

监督进程为每个原生对照和 ABBA 测量槽位启动全新进程，每个任务都通过现有空闲等待脚本调度。
它采样记录 NPU 进程归属、Linux 进程启动身份和主机 CPU 时钟计数。
未知或外部设备进程会使本轮结果失效。计时轮次必须具有进程归属观测，且主机各采样区间的忙碌比例
不得超过 20%。如果短测量窗口内没有观测样本，应增加迭代次数。

```bash
python hyper_parallel/core/multicore/examples/mega_ffn/acceptance.py run \
  --helper "$HOME/doc/npu_wait_and_run.sh" \
  --root /tmp/megaffn-dense-qwen-acceptance --device 0 --blocks 5 -- \
  --steps 100 --warmup 100 --iterations 100 --dense-backend resident_tiles
```

`common` 和 `packed` 两种基线都具有独立的原生/原生数值与性能对照。
Bootstrap 置信区间按进程块重采样，不对单个进程内互相关联的步骤重采样。
性能通过要求为：100 步独立数值验收通过；原生对照漂移不超过 2%，且其置信区间包含零；
候选收益的置信区间为正，并同时超过 1% 和原生噪声范围。
保留逐步原始样本、分配器峰值内存、首步耗时、源码/vendor/库哈希，以及优化器、数据和配置身份。
验收分析器还要求候选模型的每个解码器层都具有 resident 前向产物。
`--dense-backend host_stream` 仅用于诊断，不能满足此 resident 要求。

此验收评估固定数据、随机初始化的完整训练图，不代表真实数据上的长期收敛或 FSDP 验收通过。
设备启动和执行失败仍判定为门禁失败。

`tests/torch/multicore/test_dense_ffn.py` 分别提供单卡 resident 和双卡 FSDP 组件用例。
单卡用例还在两条非默认流上使用独立的借用权重和共享可执行计划，检查输出、dX、两组 dW，
以及输入版本计数器保持不变。它在两次前向之后、对应反向之前关闭模块，读取梯度之前等待各流完成。
两个组件矩阵均已通过设备门禁。双卡用例以现有 `SwiGLUMLP` 为对照，检查 BF16 梯度、
跨 unshard/reshard 的多次权重更新、重计算和打包检查点的精确恢复。
它使用独立 SGD 组件，因此 FSDP 下的 FP32 主参数 AdamW 和全模型分布式训练仍需另行验收。
通过相同的空闲等待脚本启动该用例，并精确选择两张设备卡。
