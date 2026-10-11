# 稠密 MegaFFN 训练接入

目标是在注意力之后执行可训练的 BF16 稠密 SwiGLU FFN，通过独立模型和优化器的对齐验收，
并取得相对现有打包实现 `SwiGLUMLP` 的实测收益。稠密 FFN 不涉及路由器、专家归属或
分发/合并通信。

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
自动求导、优化器、状态字典，以及检查点/重计算接口。归一化和残差组合保留在解码器边界。
初始稠密契约不包含偏置和 TP 集合通信。

需要沿用原始 Linear 数值契约时，可显式选择
[`linear_ffn`](examples/linear_ffn.py)。它通过独立的 `ml.linear` 原语保留原始 NK 权重，
gate/up 沿输出维拼接为 `[2I, H]`，down 为 `[H, I]`。
模块从已绑定的输入类型推导权重形状；不通过 FFN 函数体或原语名称匹配来授权执行。

```python
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn

ffn = MegaFFN(1024, 4096, program=linear_ffn).to("npu")
```

这个程序描述现有打包 SwiGLU 的数值契约。常规三投影 FFN 的独立 SiLU 与乘法舍入边界
仍需单独描述和验收；打包程序不能替代该基线。

调用方持有的本地权重通过 `MegaFFN(..., create_parameters=False)` 和
`ffn(tokens, weights=(local_gate_up, local_down))` 接入。借用权重不被缓存或复制。
自动求导状态归每次调用所有；关闭后拒绝新的前向调用，但已有调用的待执行反向仍然有效。

## 现有训练器替换机制与检查点

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

工厂仅打包一次无偏置 SiLU 的 gate/up/down 权重，在迁移到目标设备之前保留源数据类型、设备和全部训练标记。
检查点变换精确逆转拼接和转置。共享投影参数，以及 gate/up 不同的 `requires_grad` 策略会被拒绝。
双卡 BF16 FSDP/SGD 组件门禁已通过权重更新、重计算和精确检查点恢复检查。

替换时可通过 `context={"mega_ffn_program": linear_ffn, "mega_ffn_execution": execution}`
选择显式 Linear 程序。NK 模式只拼接 gate/up；down 检查点通过现有改名变换恢复，不创建空操作转换器。
上述已通过的双卡门禁属于原有 KN 程序，不能直接作为新 NK 程序的设备验收证据。

## 原生执行与证据范围

执行提供方为原生 Torch Matmul 和 NPU 打包 SwiGLU，具有精确的前向、反向调度器规范。
生成的 C++ 前向/VJP 适配器遵循 SSA，包括转置处理和反向梯度累加。
构建器对生成源码、编译器身份、Torch 版本/ABI 和库哈希进行封存；加载器在绑定算子之前校验这些信息。

原生适配器在前向、反向各使用一次 Python 与原生代码之间的调用，将执行提供方的内核加入当前流。
模块默认使用此主机流后端。`Program.compile()` 生成源码产物，不构建二进制。

显式选择的 `resident_tiles` 候选后端将可按行拆分的 BF16 SSA 降级为配对的 AIC/AIV 私有队列。
任务同时按词元行和输出列分区；每个 Cube 持有矩形片段，对应的两条 Vector 通道各处理一半行。
每个原语输出的整行汇合计数覆盖全部列片段。消费者等待所有生产片段完成后才读取本行输入，
两条 Vector 通道都发布完成事件，包括单行尾部的空通道。列分区只划分输出，不切开 Matmul 的
收缩维度，也不对已经舍入为 BF16 的局部结果求和。非对齐列尾部保留实际宽度。
事件发布显式同步标量和 DMA 流水线。消费者使事件缓存行失效，并在每次轮询时通过 volatile 指针
重新读取 GM 值，即使另一个核仍在生成该值也如此。CPU 调度检查器与原生内核中的有界生产者预取
遵循相同的队列顺序。单个混合内核执行前向 Matmul、以 FP32 计算并存为 BF16 的 SwiGLU，以及后续 Matmul。
SwiGLU 每批处理最多 16 行、每行最多 512 列，使用独立的输入和输出 DMA 步长批量搬运。
UB 中每行按 32 字节对齐，尾部补零；写回仅包含实际列，不覆盖相邻行或列片段。
批量计算保留逐元素 FP32 运算和 BF16 舍入顺序；空 Vector 通道仍由外层发布完成事件。
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
核数、工作空间字节数和原始 `TCubeTiling` 大小由选定的 SDK 提供，不沿用其他 SDK 或设备的假设。
未指定 `tile_policy` 时，运行时使用 SDK 的整矩阵分区建议，并对受支持的 `MatMulV2` 保留原生
行分区边界，再根据可用 Cube 数选择输出列分区数。显式 `DenseTilePolicy` 仍可控制行分块高度和列分区；
`column_workers=1` 保留仅按行分区的方式。分区建议和实际策略均进入执行清单。
构建器封存源码、SDK 头文件/脚本/库、设备编译器、TorchNPU ABI，以及打包二进制和主机库。
缓存由调用方持有，构建器不修改原生自定义算子包安装。启动器延迟注册精确的二进制，检查注册失败，
并通过 TorchNPU 任务队列按流顺序提交。

版本 3 的描述符保留 32 字节头部，将每个操作扩展为 44 字节，包含输出列片段宽度、片段数及
整行生产者汇合数量。分块参数表仍为每个原始 `TCubeTiling` 追加 24 字节访问参数，
激活槽位保持相同步长并置零。单核分块规划使用当前矩形片段的形状，同时保留原始整矩阵步长。
参数包含原生 M/N 分区跨度、分区数量、K 块大小和禁用标志。运行时在独立 CPU 子进程中查询
选定 SDK 的内置 `MatMulV2` 主机分块规划和原生内核族选择接口，不导入 TBE Python，
也不改变训练进程的全局编译平台。
查询保留当前确定性配置，结果和准入原因进入执行清单；查询不会进入前向热路径。
构建器同时封存查询桥接库、SDK 注册库、编译元数据和访问规则来源。

实际选择 `MatMulV3` 的矩阵保留顺序访问，不套用 `MatMulV2` 的 K 块重排。
选择接口在两种 split-K 条件下返回不同内核族时，清单保留未确定的原因。
右侧不转置且两侧 L1 均非满载的受支持 `MatMulV2` BF16 ND 模板，可从查询字段推导 K 块逆序和旋转。
内核在原生分区边界切分行任务或输出列，并通过同一组 Matmul 加载回调读取两侧操作数。
尾部重叠区域及显式禁用分区保留顺序访问；分段仍归原来的行任务和事件所有，
每个片段独立持有一个 TPipe。形状或字段 ABI 损坏会被拒绝；尚未证明的模板保留顺序访问，
并在清单中记录原因。主机查询是重建证据，不能替代设备运行时分块参数捕获或全模型验收。

当前自动求导根据真实输入梯度需求及余切存在性选择一张完整梯度图，读取本次常驻内核调用的
确切打包值和激活缓存。全部设备能力已绑定时使用一次常驻内核反向启动；存在未绑定能力时，
整次调用继续使用独立生成的原生执行提供方 VJP。两种方式不会在同一次反向中拼接。
`compiler/dense_backward.py` 将规范执行提供方的显式计算规则转成带类型的梯度 SSA，记录余切种子、
完整 Matmul 收缩维度、原始转置链、贡献依赖和逐次舍入后的累加。生成适配器使用此计划，
并保留原始 C++ 数值调用顺序；绑定产物包含 `bindings/backward.json`。
`compiler/dense_backward_tiles.py` 已提供独立的反向工作队列计划。每个工作单元完成全部行梯度窗口
之后再进入权重梯度阶段，后者等待所有必要的词元生产片段，每个输出矩形独立完成完整 K 收缩。
SwiGLUGrad 按成对通道分区；空词元的权重梯度规划为零填充。可选余切只用主机存在性控制，
单个贡献不再次相加。CPU 回放已覆盖队头阻塞、尾部、分叉、转置输出和独立完整图自动求导对照。
离线反向工作队列继续使用 `native_status=unbound`，设备绑定通过独立入口逐项检查。
`backends/dense_backward.py` 已将规范计划序列化为独立反向 ABI，包含逐操作 M/N/K、左右转置、
输出转置、阶段、行事件和跨轴事件范围。每个工作单元先完成全部行窗口，再进入完整收缩阶段。
输入梯度需求和余切存在性从主机元数据确定；只删除真实无需求的分支，保留所有需要的祖先节点。
同一逻辑权重域内的梯度累加只等待对应行的完整贡献；完整词元收缩仍等待全部输入行事件。
不能把这两种依赖都改为全域等待，否则多次累加的 Vector 消费者会阻塞未来窗口的生产者。
空词元零填充沿逻辑输出域分区，包括原始计算规则有输出转置的情况。实际 SDK 分块规划在调用之前
再次比较规范工作队列和绑定字节，拒绝形状、布局或描述符偏离规范的绑定。

`runtime/dense_backward.py` 的候选入口借用实际输入、保存值和余切，每次调用独立分配中间梯度及
返回梯度。原始张量必须连续，禁止隐式复制输入或权重；带步长余切按现有 VJP 契约转成连续存储。
当前设备工作单元包括 NN/TN/NT Matmul、梯度有序相加、空词元的权重零填充，以及直接复用选定
SDK BF16 计算规则的 SwiGLUGrad。TN 权重梯度仅切输出矩形，每个矩形仍完成完整词元的 K 收缩。
NT 通过独立的可转置 B 类型及物理 N/K 跨度实现 dA 和 dX。支持的 MatMulV2 模板由实际 SDK
查询生成 K 块访问规则，A/B 加载回调使用同一逆序或旋转；矩形只在原生访问规则改变的位置分段，
所有片段继承原来的行事件。MatMulV3、满载、split-K 和其他未证明的模板继续顺序访问，
清单保留对应的 SDK 查询、原生内核族和分区字段。
参考查询只在反向变体构建时执行，不读取设备控制数据，也不进入后续反向热路径。
构建闭包同时封存反向二进制、描述符、绑定代码和实际使用的 SDK 计算规则；启动器保留实际张量存储，
并在当前流登记全部输入、输出和临时存储。

标准非空 MegaFFN 的五个反向操作已经具备设备绑定，实际 dX 和两组权重梯度在同次常驻内核
启动中执行。通用反向的双侧转置和非空转置输出尚未绑定；候选入口在构建或分配设备状态之前
拒绝未绑定能力。相对原始两种训练基线的正式数值验收、全模型分布式训练和性能收益仍需完成。
逐操作候选的顺序 SDK 矩阵访问不能替代各原生模板的完整数值顺序证明。
完整梯度图选择缓存只保存有界的需求组合和原生元数据。关闭或淘汰缓存时，活跃调用保留所有权，
最后一次使用入队之后才释放临时存储。关闭后的待执行反向以实际保存状态重新绑定，不缓存调用张量。
每次查找缓存之前校验精确布尔需求，避免整数或浮点数与 True 的缓存键相等而跳过准入检查。
执行清单记录实际设备和主机调用次数、未绑定能力及设备产物；不把选择机制本身视为设备反向放行。
`test_dense_backward_components` 提供独立设备核对，覆盖完整 K 权重梯度、SwiGLUGrad、重复返回、
分叉累加、可选余切和三条流；此组件入口没有拼接主机 VJP 以绕过完整 FFN 的能力检查。
该设备组件矩阵已在 Ascend910B3 上通过 38 个用例、53 次调用，比较 78 组实际梯度；其中 72 组
逐值一致，6 组最大绝对差不超过 0.00048828125，均通过既有 rtol=0.02、atol=0.002。
覆盖完整 FFN 的三个输入梯度、1024 词元大矩阵、非对齐尾部、空词元、共享权重累加、转置零填充、
SDK 实际工作单元数，以及三条流中的共享临时存储和私有回退。
`test_dense_backward_autograd` 的 11 个真实自动求导用例也已通过，共执行 14 次设备反向、零次
主机反向，覆盖可选余切、retain_graph、保存钩子、非默认流、关闭后反向，以及完整 FFN 的空输入、
尾部和大矩阵。现有 24 个前向组件及保存状态生命周期也通过。
完整 Qwen 稠密图已将冻结的 AST 前向和原始完整主机 VJP，与实际常驻内核前向和完整设备反向
分别推进 100 步。初始状态逐值一致，后续不重新同步；logits、loss、梯度、BF16 参数、FP32 主参数、
AdamW 一阶矩、二阶矩及计数器的最大差异均为零。候选每层实际执行 100 次完整设备反向，无主机补算。
该内部对照仍不替代相对原始常规（`common`）和打包（`packed`）基线的正式验收，也不证明性能收益。
权重梯度保留完整的词元维度收缩，不对已经舍入为 BF16 的分块局部结果求和。
中间值归每次调用所有，在待执行反向结束之前保持完整。内部保存值使用设备共享的有界缓存，
返回值、输入和借用权重不进入此缓存。逻辑值通过 `set_` 创建不暴露私有基准分配的独立叶张量；
同次调用的存储区间互不重叠，并按 64 字节对齐。原生适配器将独占检查和叶张量构造合并到一次
主机调用，只查询主机存储元数据；只有最后一个存储别名释放后才允许再次借用。
retain_graph、外部 detach 和保存钩子持有的别名
都会阻止复用；同时存活超过两个调用时使用私有分配。槽位支持高水位增长和显式跨流事件交接。
CPU 保存钩子可以提早释放设备别名；旧调用反向解包独立存储时，不修改已经借给新调用的槽位代次。
关闭缓存只释放根所有权，保存张量继续持有其存储，待执行反向仍可完成。
指针表、事件计数器、工作空间和溢出标记存储使用有界的逐流临时缓存，在同一设备的不同层和
词元形状间共享。指针和事件容量只增长到高水位，各视图保留对应描述符的精确长度。
每次启动都更新当前张量地址，并在该流上重置汇合计数。不同流使用独立槽位；重叠的主机调用和
超出两个缓存槽位的流使用私有临时存储。可执行对象以弱引用登记所有权；最后一个对象关闭时
释放临时缓存，已入队的原生启动仍持有其存储。
检查点和关闭操作使用与主机后端相同的模块生命周期。缓存元数据具有初始化事件，
所有设备存储都在启动流上登记。

原生构建、打包、加载，真实主机 SDK 分块规划、CPU 队列回放和保存状态协议测试均已通过。
单卡常驻内核组件矩阵也已通过 BF16 输出与梯度比较，覆盖空输入、尾部输入、检查点、
关闭后待执行反向和两条非默认流。完整设备反向已重新通过双卡 BF16 FSDP/SGD 组件门禁，
并通过双卡 FP32 主参数 AdamW、磁盘检查点恢复及后续独立更新的生命周期门禁。
独立的全模型数值与性能验收仍需单独完成。内核临时存储复用已通过 CPU 协议测试和设备用例，
覆盖借用权重更新、重复使用流、私有存储回退，以及反向之前关闭模块。
扩展组件矩阵和双卡 FSDP/SGD 回归均已在共享临时存储的实现上通过。
二维前向的矩阵和单卡训练组件回归也已通过，覆盖不同列分区、非对齐尾部及共享临时存储。
行分区与二维分区已完成完整 Qwen 稠密训练图的 100 步独立回归，logits、loss、梯度、BF16 参数、
FP32 主参数和 AdamW 状态均逐值一致；此内部调度回归不替代相对原始两种基线的最终验收。
批量 Vector 实现也已与冻结的逐行内核完成相同的 100 步独立回归，各项状态逐值一致。
保存缓存已与冻结的逐值私有分配前向完成相同的独立 100 步回归，各项状态逐值一致。
真实 NPU 保存缓存系统测试覆盖重叠调用、实际地址复用、两条非默认流、retain_graph、外部别名、
保存钩子、CPU 卸载后的旧代次，以及关闭后待执行反向。内部训练回归和保存缓存系统测试均具有
干净的进程归属记录。此证据不替代相对原始两种模型基线的正式数值与收益验收。
原始训练基线与设备调优仍待完成；反向仍需读取的前向缓冲区不会发生别名复用。
CPU 原生执行仅作为参考。

## 显式 Linear 的配套数值来源

`ir/dense_matrix.py` 将原始数值算子的来源、M/N/K、两侧转置和坐标映射纳入计划与绑定身份。
NK Linear 的前向使用 NT；输入梯度保留原始 NN。权重梯度在设备上以物理 TN
`dY.T @ X` 直接写入 NK 存储，其数值参考保留原始 TorchNPU `matmul_backward` 的
`(X.T @ dY).T`。参考查询交换 M/N，访问跨度与核分区通过显式转置坐标映射对应物理输出。
主机 VJP 同样保留该原始计算方向；两种等价公式不能自行被认为具有相同 BF16 累加顺序。
这一来源对应 TorchNPU 所固定的
[MatmulBackwardKernelNpuOpApi.cpp](https://github.com/Ascend/op-plugin/blob/8b9c8534fa41eff367a41c155843daa530ab3a08/op_plugin/ops/opapi/MatmulBackwardKernelNpuOpApi.cpp)。
原始借用权重必须连续；不允许通过热路径权重复制改变物理布局。

`backends/dense_linear.py` 为真实前向队列提供独立绑定，复用已经具有正确 NT 类型的共享矩阵核心，
并保留打包 SwiGLU 的 FP32 计算和 BF16 舍入。操作主序的原生事件编号通过显式映射与逻辑队列对应。
旧 Matmul 前向入口拒绝显式 Linear，避免该原语落入另一套数值契约。

SDK 查询保留实际 NN/NT/TN 布局，逐操作访问原因、原始查询字节和参数表哈希均进入清单。
受支持的多 K 块模板使用相同的 K 块逆序或旋转；MatMulV3 保留原有顺序。
单 L0 K 块模板要求实际字段证明完整 K 覆盖、两侧 L1 覆盖以及没有跨核 K 收缩，
因为此时 K 外层只有一次迭代，不能形成块间重排。
其他尚未证明的 Linear 模板会在建立设备状态之前被拒绝，不以另一种数值顺序继续运行。

新路径已通过独立 CPU Linear 输出与全部梯度、真实 ABI 队列解码、检查点恢复和 SDK 字段检查。
新增设备入口为 `test_dense_linear_ffn_training`，覆盖空输入、尾部、非对齐通道、重计算、
关闭后待执行反向，以及大矩阵不同列分区。当前打包基线（`packed`）七项和常规基线（`common`）八项设备组件
已经逐值一致，覆盖大矩阵、尾部、重计算、冻结上投影（`up`）和非默认列分区。
原始完整模型独立轨迹、原始 NK 权重布局（Source-NK）的 FSDP 生命周期与完整训练步收益
仍需分别验收；组件一致性不能替代这些门禁。

已安装的 `npu_ffn` 接口仅允许门控激活用于 FP16 推理，因此不能作为 BF16 训练的快捷路径。
主机编译成功和 CPU 梯度正确不代表 NPU 数值正确或具有性能收益。

## 最终验收要求

`examples/mega_ffn/qwen_dense_model.py` 在完整稠密语言模型中，将 FFN 接在现有 Qwen GQA 注意力和
归一化之后。它通过同一替换机制支持常规三投影模型（`common`）、现有打包模型（`packed`）
和 AST 模型（`mega_ffn`）。

1. NPU 前向、dX 和两组权重梯度正确，覆盖尾部、空词元和重计算。
2. 在真实设备上验证 FSDP/本地权重生命周期，以及检查点保存与恢复。
3. 验证独立训练轨迹、FP32 主参数和优化器矩。仅对齐一次初始状态，后续运行之间不再重新同步。
4. 在全新进程中执行原生/原生对照，以及相对两种基线的 ABBA 比较；测量完整优化器步和峰值内存，
   精确记录源码、框架和设备进程归属。
5. 实现常驻工作单元/分块流水线、缓冲区复用，以及 RAW/WAR/WAW 依赖，并取得实测收益。
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
自助法（Bootstrap）置信区间按进程块重采样，不对单个进程内互相关联的步骤重采样。
性能通过要求为：100 步独立数值验收通过；原生对照漂移不超过 2%，且其置信区间包含零；
候选收益的置信区间为正，并同时超过 1% 和原生噪声范围。
保留逐步原始样本、分配器峰值内存、首步耗时、源码/自定义算子包/库哈希，以及优化器、数据和配置身份。
验收分析器还要求候选模型的每个解码器层都具有常驻内核前向产物和实际完整设备反向记录。
反向必须覆盖三个输入的全部梯度、实际输出余切，以及至少一次设备调用；主机补算、未绑定能力、
缺失反向产物或部分输入需求均不能放行完整训练收益。
`--dense-backend host_stream` 仅用于诊断，不能满足此常驻内核要求。

此验收评估固定数据、随机初始化的完整训练图，不代表真实数据上的长期收敛或全模型 FSDP 验收通过。
设备启动和执行失败仍判定为门禁失败。

`tests/torch/multicore/test_dense_ffn.py` 提供单卡常驻内核、单卡保存缓存和双卡 FSDP 组件用例。
单卡用例还在两条非默认流上使用独立的借用权重和共享可执行计划，检查输出、dX、两组 dW，
以及输入版本计数器保持不变。它在两次前向之后、对应反向之前关闭模块，读取梯度之前等待各流完成。
两个组件矩阵均已通过设备门禁。双卡用例以现有 `SwiGLUMLP` 为对照，检查 BF16 梯度、
跨 unshard/reshard 的多次权重更新、重计算和打包检查点的精确恢复。
`test_dense_ffn_fully_shard` 保留独立 SGD 的四步组件回归。
`test_dense_ffn_fully_shard_adamw` 使用正式 `Float16OptimizerWithFloat16Params` 包装的
HyperParallel AdamW，比较输出、dX、权重梯度、BF16 参数、FP32 主参数、两阶矩和实际参数组计数。
两个进程独立运行八步，覆盖 1/7/129 词元、尾部和两次重计算；第 4 步保存磁盘检查点，
恢复到新的分片模块与优化器，后续四步不再同步状态，恢复与未中断轨迹逐值一致。
此组件矩阵的全部比较差异为零，每个进程的候选与恢复模块分别执行八次和四次完整设备反向，
主机补算次数为零。检查点使用完整模型参数及各进程的本地优化器状态，验证同一双卡拓扑的恢复；
不同分片拓扑的迁移及全模型分布式训练仍需另行验收。

优化器更新及加载状态时使用 `SkipDTensorDispatch`，使恢复过程中触发的初始化也作用于本地分片。
加载模型与优化器状态之后，精确检查 BF16 回写与 FP32 主参数一致，防止恢复时丢失低精度未能表示的主参数位。
通过相同的空闲等待脚本启动双卡用例，并精确选择两张设备卡；可通过 `HP_FFN_CACHE_ROOT`
显式选择已封存的原生构建缓存，通过 `HP_FFN_EVIDENCE_DIR` 保存各进程的逐步比较和磁盘检查点。

## Source-NK 完整训练入口

`examples/mega_ffn/source_training.py` 分别使用原始 `common` 和 `packed` 完整模型作为对照，
候选通过现有模块替换接口选择对应的显式程序。常规基线（`common`）保留三个独立 NK 投影、BF16 SiLU
和 BF16 乘法；打包基线（`packed`）保留打包激活的单次 BF16 舍入。注意力、残差、归一化、嵌入及
语言模型头均包含在同一训练步骤内。

模型和优化器只在初始化时对齐。入口检查 logits、损失、全部更新前梯度、BF16 参数、
FP32 主参数、AdamW 一阶矩、二阶矩，以及真实参数组更新次数。更新后缺失优化器矩会
直接拒绝；两条轨迹即使具有相同但错误的步数，也不能通过应有更新次数检查。

```bash
python -m hyper_parallel.core.multicore.examples.mega_ffn.source_training \
  --mode validate --baseline common --backend mega_ffn --steps 100 \
  --dense-backend resident_tiles --dense-soc Ascend910B3 \
  --cache-root build/native/dense/source-cache --output build/source-common-trajectory.json
```

`--mode measure` 在新进程中记录包含完整反向和 FP32 主参数 AdamW 的训练步样本、
测量时间窗口与峰值设备内存。入口显式启用相同确定性配置，并封存原始数值契约及
新增训练入口的源码身份。运行仍须通过既有设备空闲门禁。

单个入口的成功不放行完整目标。正式数值轨迹还需核对每层三/四输入的真实常驻反向、
零主机补算、源码与设备进程归属；性能还需原生/原生噪声对照、至少三个新进程 ABBA 块、
主机测量窗口、完整训练步置信区间与设备内存证据。原始 NK 权重布局的 FSDP 和磁盘恢复另行验收。

完整监督入口为 `examples/mega_ffn/source_acceptance.py`。它为两种基线各执行原生与
候选的独立训练、原生/原生噪声对照及至少三块完整优化器步 ABBA，共 52 个新进程任务。
常规（`common`）和打包（`packed`）基线的数值契约身份分别保存，其余源码、模型、数据、设备和优化器配置必须一致。
真实更新计数、八项固定阈值、全部层与梯度、零主机补算、峰值 HBM 和测量归属缺一不可。

```bash
python hyper_parallel/core/multicore/examples/mega_ffn/source_acceptance.py run \
  --root build/source-acceptance --helper ~/doc/npu_wait_and_run.sh --device 0 --blocks 3 -- \
  --steps 100 --warmup 100 --iterations 100 --dense-backend resident_tiles \
  --dense-soc Ascend910B3 --cache-root build/native/dense/source-cache
```

监督进程通过脚本路径导入既有统计与归属工具，不加载框架；只有经过原有空闲门禁的
子进程执行训练。失败步骤和污染轮次保留原始文件，不拼接不同轮次来放行。
