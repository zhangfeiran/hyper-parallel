# Python AST 前端与兼容性执行计划

稠密 FFN 降级、生成的原生前向/VJP 适配器和标准训练替换接口见
[MegaFFN 训练接入](MEGAFFN.md)。

这里实现了 2026-10-07 AST 设计中的语义前端基础，覆盖带类型的源码捕获、基于身份的原语 schema、
ProgramIR、CPU 参考解释执行、Gate WorkerPipeline 计划，以及 MoE/MHC TaskDAG 计划。
设备绑定保留各算子族现有的数值内核和显式反向执行规则。

## 使用前端

```python
import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml


@mc.program
def activate(value: ml.Tensor[ml.fp32, ("T", "E")]):
    return ml.sqrt(ml.softplus(value))


ir = activate.lower()
print(ir.dump())
```

可运行的 [Gate Route 示例](examples/gate_route.py) 遵循设计规定的顺序：softplus、sqrt、
不参与梯度计算的选择偏置、top-k、gather、归一化和缩放。在可编辑安装的工作区中运行：

```bash
pip install --no-deps -e .
python -m hyper_parallel.core.multicore.frontend.examples.gate_route
python -m pytest -q tests/ut/core/multicore/frontend
```

`Program.lower(k=3, scale=2.5)` 生成不可变的语义 IR。
`Program.explain(...)` 展示调度意图和确定性的 JSON IR，包含存储 dtype、符号形状、逻辑访问和原始源码位置。
`Program.interpret(...)` 在连续存储的 CPU Torch 张量上执行已注册的参考实现，并统一输入和输出中的符号维度。

解释执行通过参考函数保留 Torch autograd。它可验证 Gate 示例的 logits 梯度和偏置脱离梯度的行为，
但不提供原生反向执行规则。

## 语言与注册契约

- Tensor 注解保留 bf16、fp16、fp32、int32、int64 和布尔类型身份、静态/符号形状，以及连续逻辑布局。
- Constexpr 参数接受精确的标量类型，包括可选标量联合类型。运行时张量不能作为 constexpr 值。
  静态整数、字符串、元组和浮点数都有界限并接受校验。
- 支持普通命名参数、赋值、带注解赋值、元组解包、元组返回、标量算术/比较和 constexpr 分支。
- `@mc.helper` 注册带类型的辅助函数源码，用于内联。错误保留辅助函数定义位置和调用链。
  拒绝递归调用和超过 32 层的辅助函数嵌套。
- `mc.static_range` 支持一至三个整数边界；一次编译中总计最多展开 1024 次迭代，包含辅助函数和嵌套循环。
- 符号按局部词法作用域和精确的已注册原语身份解析。别名可用；同名可调用对象不会自动成为原语。
  只有 DSL 模块允许属性查找。被捕获的全局变量和闭包值用作常量时必须满足静态契约。
- 编译器不执行被捕获的函数体，不使用 eval/exec，也不调用任意函数或属性。
  已注册的推导和参考钩子属于可信扩展代码；参考钩子仅在解释器中执行。
- 不支持的语法会被拒绝，并报告文件、行号和列号，包括特化时被删除的分支中的语法。
  拒绝依赖数据的控制流、while 循环、任意下标访问、可变容器和张量原地写入。

当 inspect 无法取得源码时，`from_source(source, signature, constants, symbols=...)` 捕获且仅捕获一个函数。
类型可通过显式签名提供；原语和辅助函数符号必须显式提供。字符串注解使用相同的受限解析器解析，不求值执行。

`PrimitiveRegistry` 按逻辑命名空间和版本索引 schema，与原生任务编号无关。
每个 schema 提供 inspect 签名、类型/形状推导、逻辑访问推导和可选的参考实现。
调用在推导之前规范化关键字参数和默认参数；重复注册 schema 会失败。
默认访问契约读取张量参数并写入新分配的结果。
自定义访问钩子可声明对逻辑张量参数的 read/write/reduce/atomic 访问。
物理缓冲区别名和通信效果仍需后续的缓冲区与运行时接入。

## Gate WorkerPipeline 主机执行计划

使用 `schedule=mc.WorkerPipeline()` 的受支持文本 Route 程序现在提供
`Program.plan(signature, topology, **constants)`：

```python
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route

plan = _route.plan({"T": 33, "E": 4}, mc.HardwareSpec(48), k=3, scale=2.5)
print(plan.explain())
print(plan.export_manifest())
normal_runtime_bytes = plan.forward.normal
profiled_runtime_bytes = plan.forward.profiled
```

签名精确提供符号输入维度；输入形状全为静态的程序不需要形状签名。
硬件拓扑独立于计算描述可用的 AIV worker。生成计划不探测或读取设备。
`from_source(..., schedule=mc.WorkerPipeline())` 对显式源码支持相同的计划 API。

编译器证明图使用规范的 Route schema 和完整数据流：FP32 logits 和脱离梯度的偏置；softplus 后接 sqrt；
使用分数加偏置进行选择；最后一维的非排序 top-k；收集原始分数；k 大于 1 时使用行求和、
`1.0e-20` epsilon 和除法；最后缩放并输出 int64 索引。
改变这些契约、增加额外操作或声明修改性效果都会被明确拒绝。
仅匹配熟悉的原语名称不足以通过检查。后端在检查后输出固定版本的前向和反向模板。

k 大于 1 时 Gate 语义 IR 有 11 个操作，k 等于 1 时有 8 个操作；两者都降级为现有的十阶段前向描述符序列。
k 等于 1 时仍保留 ReduceSum/AddEpsilon/Div 描述符，并明确记录 `retained_legacy_k1_stage` 原因。
反向选择原始的十一阶段或两阶段描述符模板；独立的 CANN 后处理和可选的直接 logits 梯度累加
作为外部调用元数据保留。

`KernelPlan` 包含普通/带性能记录的镜像、逻辑/原生阶段身份、源码范围、原生绑定顺序和五个保存状态的名称。
调度保留每个已启动 worker 的有序阶段和行区间，包括原生的空尾部 worker。
镜像保留固定的 48 个 worker 槽位容量，其字节内容不依赖 token 数或可用 worker 数。
`schedule.simulate()` 枚举每个已启动 worker 的有序描述符访问，不模拟设备指令、流或 CANN 后处理。

各算子族契约见[兼容性基线](../runtime/baselines/README.md)。
CPU 测试以原始构建器的快照为依据，逐字节复现全部六个 Gate 序列化镜像，
并编译提取出的原始 C++ 声明，检查 MoE、MHC 和 Gate 的 Python 结构大小及字段偏移。

## 当前实现边界

`TaskDAG` 仍记录选择元数据；MoE/MHC 调度降级尚待完成。
默认 MegaMoE 调用路径、worker、运行时 ABI 和反向代码均保留。
组件包在首次访问时加载 MegaMoE/profiler 业务导出，使此前端可在没有 torch_npu 或原生产物时导入。

`Program.plan()` 仍只使用 CPU，在显式实例化之前报告 `native_status=unbound`。
隔离的 Gate 构建器导出固定 Gate 版本并验证源码哈希，同时验证固定的 ops-nn LinearIndex 源码。
它只在独立 vendor 中构建 Route/RouteGrad，不替换当前 MoE 运行时，也不构建 SHMEM。
固定的 tiling API（`TensorShape` 和 `TensorDataType`）要求 CANN 9.2 或更高版本；CANN 9.1 不提供这些类型。
支持的设备目标为 ascend910b 和 ascend910_93。

在可编辑安装的工作区中构建；Git 对象数据库中必须具有固定 Gate 提交，ops-nn 工作区必须干净且符合锁定版本：

```bash
source /path/to/cann/set_env.sh
python -m hyper_parallel.core.multicore._build.build_gate \
    --ops-nn-source /path/to/ops-nn --soc ascend910b --jobs 16
source build/native/gate/payload/set_env.bash
# 激活后启动新的 Python 进程。
```

构建器检查必需的 ACLNN 符号、主机 ELF 加固和设备二进制是否存在，然后输出 `manifest.json`。
构建指纹覆盖构建输入、CANN/编译器/框架身份，以及所有产物的哈希。
绑定在加载适配器之前检查算子族、源码和 schema 身份、指纹及全部产物哈希。
CANN、Torch 和 torch_npu 身份必须与构建时一致。
兼容性产物使用隔离的自定义 OPP 进程：不要在同一进程中激活另一个自定义 vendor，
因为框架 ACLNN 和 tiling 缓存是进程级的。

```python
import torch
import torch_npu
import hyper_parallel.core.multicore.frontend as mc
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route

torch.npu.set_device(0)
topology = mc.HardwareSpec(torch.npu.get_device_properties(0).vector_core_num)
plan = _route.plan({"T": 49, "E": 64}, topology, k=3, scale=2.5)
route = plan.materialize("npu:0")
logits = torch.randn(49, 64, device="npu:0", requires_grad=True)
bias = torch.zeros(64, device="npu:0", requires_grad=True)
weights, indices = route(logits, bias, profile=True)
(weights.square().sum() + logits.square().sum()).backward()
records = [call.records() for call in route.take_profiles()]
```

原生主机代码计算 UB/workspace tiling。实例化检查计划启动的行 worker 数是否与实际设备拓扑相符。
48 个描述符槽位表示容量，不代表硬件核数：Ascend910B3 有 40 个 AIV worker。
原始反向 tiling 只启动非空行 worker，前向则启动完整行 worker 数，包括空尾部。

前向状态归每次 autograd 调用所有；修正偏置脱离梯度，索引不可微，不支持二阶反向。
RouteGrad 使用原始的 11/2 阶段模板和 CANN LinearIndex/scatter/sqrt/softplus 后处理。
直接作用于 logits 的 loss 通过 Torch autograd 组合。
普通与带性能记录的描述符保持独立；性能记录缓冲区归每次调用私有，读取记录仅同步该调用的完成事件。
实例化描述符使用初始化事件和当前流的分配器登记，以便跨流复用。
调用方必须遵循普通 Torch 流规则，建立输入的生产者/消费者流依赖。

激活后运行单卡验收：

```bash
python -m pytest -s tests/torch/multicore/test_ast_gate.py
```

验收精确比较被选专家集合，并以 `rtol=2e-4`、`atol=2e-5` 将 FP32 权重和 logits 梯度与独立的
CPU、NPU Torch 参考实现比较。覆盖 k=1/k>1、小行数/尾部/批量行、直接 logits 梯度、投影 autograd、
重叠前向、多条流、非连续传入梯度，以及实际阶段记录覆盖率。
结果写入 `build/native/gate/acceptance.json`，可通过 `HP_AST_GATE_RESULT` 覆盖路径。

## MoE TaskDAG 接入

[MoE 区域](examples/moe_region.py) 在 `mc.TaskDAG(policy="moe_ratr_v1")` 下声明 dispatch、
分组 matmul、打包 gate/up SwiGLU、分组 matmul 和 combine。
编译器验证规范原语身份、BF16 存储、完整的操作数/分组/元数据流、打包布局、clamp 特化及数值顺序。
它从 ProgramIR 推导前向 DAG 边，再使用原始原生任务节点和完整的 `build_config_for_rank` 路径。
终止任务、队列版本、动态分组临时存储、就绪/完成协议和性能记录布局保留原始契约。
原始 autograd 和副本梯度返回规则提供反向，包含 W13 重叠执行及无副本回退路径。

模型通过现有模块构造函数选择区域：

```python
from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region

experts = MegaMoeExperts(
    local_num_tokens=128, hidden_size=512, intermediate_size=128,
    num_experts=4, top_k=2, ep_size=2, ep_group=ep_group,
    dispatch_mode="push", swiglu_limit=10.0, program=moe_region,
).to(device="npu", dtype=torch.bfloat16)
output = experts(hidden_states, topk_ids, topk_weights,
                 tokens_per_expert=tokens_per_expert)
output.backward(grad_output)
experts.close()
```

路由器/排列、直方图处理、调用方持有的权重、容量增长、热副本规划和 SHMEM/workspace 所有权继续由该模块管理。
Program 身份参与静态 rank 一致性和共享资源兼容性检查。模块必须使用同一个 program 才能共享执行资源。
改变运行时计数或路由偏斜不会特化语义程序。省略 `program` 时保留原始模块入口。

CPU 规划调用 `moe_region.plan(spec, limit=spec.swiglu_limit)`，传入现有 `MegaMoeSpec`，
描述 rank 本地形状、物理核和 guest 槽位。
它输出完整的普通/带性能记录前向与反向镜像、固定步长 worker 队列、阶段源码范围、物理绑定和来源清单，
不要求导入 NPU 依赖或访问设备。`plan.materialize(device)` 返回现有 `MegaMoePlan` 资源对象。
模型执行应使用 `MegaMoeExperts`，使模块同时负责路由和分布式生命周期。

`ml.RaggedTensor[dtype, (capacity, width)]` 将存储容量与有效行数分开。
`ml.TensorList[dtype, matrix_shape]` 描述同类型、同形状的专家矩阵；仅指定 dtype 的形式通过原生规格绑定矩阵形状。
`ml.RouteMetadata` 标识运行时累计的 INT64 专家计数。
CPU 解释器对已经位于本地、按专家分组的行接受 `ml.RaggedTensor(storage, valid_rows)` 和
`ml.RouteMetadata(group_list)`。
dispatch 和 combine 的参考实现验证本地分组并保持行顺序，不模拟分布式通信或 token 排列。
原生执行绑定模块的实际路由缓冲区和动态分组列表。

常规 MoE 构建现在输出 `lib/frontend_manifest.json`，封存固定的 MoE ABI/原生源码基线、构建输入、
工具链/框架身份，以及所有 MoE 和私有 SHMEM 产物文件。
AST 实例化在加载资源之前检查封存信息、产物集合和 CANN/Torch/C++ ABI 身份。
在全新进程中构建并激活 MoE 产物：

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 16
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python -m pytest -s tests/torch/multicore/test_ast_moe.py
```

封存的 MoE vendor 必须位于 `ASCEND_CUSTOM_OPP_PATH` 首位。
Gate 和 MoE 自定义 OPP 激活应使用不同进程，因为 CANN/框架算子缓存是进程级的。
旧 MoE 构建需要重新构建以生成封存信息，之后才能选择 AST program。

视觉掩码、运行时/形状标量、通用缓冲区规划、编译缓存和生成的 Ascend worker 仍属于后续阶段。
设备验收仅确认已测试的拓扑和形状，不代表性能或全模型训练等价性。
Ascend910_93 仍需设备验证。

## MegaMHC shifted TaskDAG 接入

[MHC 边界](examples/mhc_boundary.py) 在 `mc.TaskDAG(policy="shifted_mhc_v1")` 下声明 post、
mapping、输入混合和 RMSNorm。
它返回更新后的残差、下一层的 pre/post/residual 混合系数，以及当前 block 输入。
InputMix 使用**上一层**的 pre 系数；mapping 分支预测下一层系数。
编译器检查规范原语身份、全部九个带类型操作数、五个有序输出、固定四路流、二十次 Sinkhorn 迭代，
以及 NormCast/RMSNorm 相等的 epsilon，然后从 SSA 推导语义分叉。

Mapping 展开为 NormCast、Projection 和 Mapping。
六个原生阶段保留原始 AIV 填补空隙的执行顺序，以及 AIC 投影的重叠执行。
NormCast 在复用 X-cast 环形槽位之前等待 Projection。
反向规则保留输出初始化、RMSNormGrad、previous-A/mapping 准备、AIC Phi/RMS 和 previous-X/post 梯度。
每个 AIC 宏都等待全部对应的 AIV 生产者。
`plan.explain()` 展示阶段源码位置、worker 队列、环形存储所有权、宏汇合，以及七个不同的保存缓存缓冲区。
旧 TensorSpec 占位符不表示物理缓冲区存在别名。

```python
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec

plan = mhc_boundary.plan(MhcSpec(2593, 128, num_cube_cores=20))
print(plan.explain())
# 编译与完整的普通/带性能记录序列化仅需要 CPU Torch。
```

原生前向/反向 worker 和适配器来自设计指定的精确 MHC 版本
`979e2a9ac913413e361f4fc2dd9987766af8ddb4`。
MHC 任务 ID 保持在该算子族内部；不扩展 MoE 任务枚举和原生头文件。
构建器将已验证的源码导出到隔离目录，检查并应用锁定补丁，记录全部实际构建输入和产物。
CANN 9.2 包装层从导出的公共头文件中删除一个未使用的过时 SDK include，不改变固定 worker 的数值实现。

```bash
source /path/to/cann-9.2/set_env.sh
# 先通过现有构建准备常规 MoE 依赖和私有 SDK。
python -m hyper_parallel.core.multicore._build.build_mhc \
    --ops-nn-source build/native/deps/ops_nn/src \
    --ops-mhc-source /path/to/clean/ops-transformer-mhc
unset ASCEND_CUSTOM_OPP_PATH
source build/native/mhc/payload/set_env.bash
python -m pytest -s tests/torch/multicore/test_ast_mhc.py
```

MHC 依赖工作区必须对应 `58b4a6bdeb29feeb0070dd266106bd4e130bb72b`，并符合锁定的目录树和归档哈希，
包括原始锁中允许的兼容哈希。
Git LFS smudge 可改变归档字节：实例化此源码工作区时应保留已提交的指针字节。
构建器当前使用 `build/native/work/multicore/shmem/sdk.json` 中已有的私有 SDK 记录，
满足通用 worker 头文件依赖；它不初始化 SHMEM，也不将 SHMEM 运行时链接到 MHC。
使用全新进程，并确保 `ASCEND_CUSTOM_OPP_PATH` 中仅有已激活的 MHC vendor；
Gate/MoE/MHC 算子缓存不能共享一个进程。

模型入口保留原始参数名称、形状和 dtype：

```python
from hyper_parallel.core.multicore import HyperMegaMhc

layer = HyperMegaMhc(128, device="npu:0")
updated, next_pre, next_post, next_matrix, block_input = layer(
    previous_output, residual, previous_pre, previous_post, previous_matrix,
    profile=True,
)
loss = block_input.float().square().mean() + next_pre.square().mean()
loss.backward()
records = [record for call in layer.take_profiles() for record in call.records()]
layer.close()
```

Phi/alpha/bias 保持 FP32，归一化权重保持 BF16。兼容的自定义区域可通过 `program=` 传入。
模型描述符按展平后的形状、设备和反向契约缓存；每次调用重新绑定输入和参数。
每次调用持有新的事件计数器、保存缓存和可选性能记录存储，支持多个待执行前向和 checkpoint 重计算。
关闭绑定后拒绝新的前向；待执行的 autograd 上下文仍持有其描述符。
调用方建立输入生产者顺序之后，初始化事件和分配器流登记保证跨流复用安全。
原生执行不支持二阶反向。
性能记录报告实际任务周期、源码位置和执行方向，并拒绝丢失或损坏的记录。

原生反向要求 T 不小于物理 AIV 数、H <=5760，且 H 必须能被 128 整除。
较小的推理输入应使用 `need_backward=False`。epsilon 转为 FP32 后仍须有限且为正。
规划拒绝前向和反向中未为原始 8 元素原子计数器写入预留填充的事件边界，不扩大或静默修改该 ABI。
例如，T=10817、行 tile 为 32 时会被拒绝，但可以显式选择更大的安全 tile。
通用自动缓冲区规划、生成的 worker 和统一的 schema 派生绑定属于 P5 工作。

已提交的 CPU 固定样例包含十六份完整原始前向/反向镜像，覆盖尾部、环形回绕和 32/64/96 行 tile。
复现方式：

```bash
python -m tests.ut.core.multicore.backends.fixtures.capture_mhc_snapshots
python -m pytest -q tests/ut/core/multicore/backends/test_mhc.py
```

单卡设备套件使用原始精度门禁（相对 L2 <=0.02、余弦相似度 >=0.999），
将全部五个输出和九个梯度与独立的固定版本 Torch oracle 比较。
它还检查流、多个待执行前向、缺失输出梯度、反向前关闭、checkpoint/SGD 和性能记录覆盖率。
CPU 原语参考实现显式体现原生 BF16 混合输入缓存边界；独立验收 oracle 保留原始数学参考。
设备准入仅适用于已验证的形状和拓扑，不声明性能或全模型收敛。

## P5 共享 schema 与源码生成

`Program.compile(...)` 对 Gate、MoE 和 MHC 使用同一个后端生成入口。
现有 `plan(...)`、实例化和模型调用签名保留各算子族行为。
当前编译生成源码和完整调度产物；清单明确报告 `status="source_only"`，
不将生成的源码包称为已编译的设备二进制。

```python
from pathlib import Path
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.runtime.cache import EmissionCache

emission = _route.compile({"T": 41, "E": 16}, k=3, scale=2.5)
path = EmissionCache(Path("build/native/frontend-cache")).store(emission)
print(emission.export_manifest())
```

源码包包含 Python/C++ 序列化声明、Python/C++ 带类型调用包装器、所选原生绑定 schema、
普通/带性能记录的前向和反向镜像、可选 MoE 无副本反向、静态计划和语义源码映射。
以下身份具有不同的生命周期：

- `definition_key` 覆盖规范化语义 IR、constexpr、算子族 ABI/原生源码闭包和实际生成器输入。
  运行时行数和张量指针不进入此键。
- `plan_key` 额外包含静态形状/容量、拓扑/rank 和序列化调度镜像。
  动态路由计数和进程组对象仍不进入此键。
- `artifact_key` 还覆盖精确的生成文件和源码位置，防止原生语义相同时错误复用其他源码映射。

`EmissionCache` 原子存储仅含源码的包，复用之前验证元数据和全部文件哈希，
拒绝缺失、越界、损坏或额外文件。
调用私有的指针、epoch、缓存、计数器和 workspace 租约继续通过现有算子族资源管理，磁盘缓存不存储它们。

序列化声明派生自 `runtime/baselines/families.json`。
公共枚举、六个原生调用签名和保留的原生字段表示位于 `runtime/native_calls.json`。
生成的 Python ctypes 结构和原生 C++ 类型断言这些共享契约中的每个大小和偏移。
MoE 保留完成/协议字段和动态临时存储；Gate/MHC 保留原始预留头字段和算子族本地任务 ID。
`DynamicData.dynamic_max_seq_len` 的原始原生无符号表示被显式记录，
在保留原始有符号 ctypes 视图的同时维持现有 C++ 契约。

重新生成或检查已提交的产物：

```bash
python -m hyper_parallel.core.multicore.backends.schema
python -m hyper_parallel.core.multicore.backends.schema --check
python -m pytest -q tests/ut/core/multicore/backends/test_codegen.py
```

实际调度器使用生成的 MoE Python 类型。MoE worker 包含生成的 C++ 声明；
Gate/MHC 构建器将对应声明安装到已验证的隔离源码导出中。
MoE 清单仍检查全部固定原生源码。
唯一的头文件适配必须等于对已验证原始 Git 对象的确定性变换，拒绝其他原生源码漂移。
该变换替换序列化声明、格式化 C++、缩短描述符读取器局部名称，并保留访问器行为。

六个运行时前向/反向调用全部使用生成的 Python 启动包装器。
它们一次性验证实际加载的 dispatcher schema，包括参数名称、类型、返回顺序和可变别名注解。
现有 autograd、流、SHMEM、tiling 和分配器所有权仍由各算子族实现负责。
生成的 C++ 转发包装器在 CPU 测试中实例化，验证参数顺序及只读/可变所有权；
现有原生 Torch 适配器仍将各算子族算子加入执行队列。

这交付了 P5 的第一个增量。下文描述 worker 分发和地址表生成，后续章节涵盖数值启动器与 MHC 上下文生成。
通用上下文工厂、更底层 CANN 主机代码生成、通用缓冲区规划，以及编译后二进制/实例化设备缓存仍属于后续工作。
新的任意原语组合仍需具备受支持的数值实现和降级规则。

## P5 原生 worker 胶水代码生成

`runtime/worker_calls.json` 声明每个前向/反向 worker 的原生任务回调、有序输入地址表、槽位常量和自有成员状态。
数值 ID 和逻辑任务名称来自共享算子族 ABI。
schema 拒绝未知或重复任务，以及超出地址表范围的槽位索引。

公共生成器包含 `workers/manifest.json` 和两个方向的生成 `.inc` 片段。
这些片段替换精确的原始分发 switch、TaskDAG 输入/缓存/workspace 地址数组和槽位常量。
Gate 还生成 worker 自有流水线成员声明。
阶段校验、top-k=1 反向路径、算子族策略和数值方法保留原始顺序。

三个构建器都将片段安装到已验证的隔离源码导出中。
适配器在替换任何代码块之前要求其 token 与固定的分发和绑定契约一致。
MoE 跟踪的数值源码保持固定哈希，隔离组装器执行 worker 变换。
构建清单和 MoE 原生缓存键包含 worker 生成器与契约输入。

worker 清单记录上下文边界：Gate 流水线对象归 worker 所有，每个阶段调用内部使用 TPipe；
MHC/MoE 保留回调持有的局部上下文。
现有 Init、流/pipe 同步、重置和销毁仍位于这些数值执行规则中。
此增量生成分发与绑定胶水代码，不生成新的逐原语初始化/销毁工厂，也不替换各算子族的 Torch/ACLNN 主机适配器。

CPU 测试将生成的 switch 与独立的固定版本原始实现一起编译，覆盖合法/未知任务和阶段 ID，
并以不同地址执行全部四个 TaskDAG 地址表。
漂移测试拒绝变化的原生回调、槽位常量、上下文成员和输入映射。
设备验收使用重新构建的 Gate/MHC/MoE 产物及现有数值与生命周期门禁。

## P5 启动器与 MHC 上下文生成

共享的 `native_calls.json` 还定义每个数值入口的原生符号、桥接方式、自有输出、
返回的参数别名和完整 ACLNN 参数顺序。
`backends.launchers` 生成全部六个 NPU/Meta 定义和数值 dispatcher schema。
独立快照在替换之前校验原始 C++ 参数类型/修改行为、桥接参数顺序、返回顺序和注册 schema。
其他算子注册和公共模型接口保持原样。

Gate 保留原始输入校验辅助函数，按已声明的形状/dtype 规则分配五个前向输出或单个梯度输出。
MHC/MoE 返回原始调用方持有的 Tensor 引用。
原生入队和 workspace 分配保留在所选现有扩展或缓存 op-api 桥接层中。

构建器从隔离适配器源码副本编译生成的定义。
Gate/MHC 封存这些实际复制/适配的源码；MoE 还封存暂存的 Torch 适配器和加固源码。
`Program.compile()` 在源码包中包含启动器源码、共享返回类型头文件和启动器所有权清单。

MHC 使用十二条固定上下文规则，生成局部构造/Init 代码块和显式清理片段，
包括借用的反向 pre 梯度 Init 辅助函数。
它们的原始作用域、条件保护、数值 Process 调用、同步、pipe Reset 和 Destroy 顺序均被保留，
并对照独立源码 token 校验。Gate/MoE 保留现有原语局部初始化规则。

CPU 验收以真实 Torch 头文件编译生成入口，在全新真实 dispatcher 中注册并执行全部六个 Meta 算子。
该测试仅将设备入队桥接层替换为桩，且此桥接层不得收到任何调用。
测试检查实际返回别名、Gate 形状/dtype 和非法输入拒绝行为。
设备验收使用重新构建的原生适配器及现有组件门禁。

至此完成三个受支持算子族的数值 Torch 启动器定义共享生成。下文描述 Gate 初始化生成增量。
通用可选原语上下文工厂、MoE 初始化生成、更底层 CANN 主机/tiling 生成、通用缓冲区规划，
以及二进制/实例化设备缓存仍属于后续工作。

## P5 Gate 阶段上下文生成

Gate 的 worker 自有前向/反向流水线绑定状态，以及阶段局部的 TPipe、队列和 UB 缓冲区，
也使用共享上下文规则生成器。
二十五条固定规则在原始 worker 和流水线头文件作用域中生成四十三个初始化/清理片段。
覆盖全部十八个阶段初始化方法、worker/pipeline Init 绑定、反向行分区，
以及两个共享的 MTE3 写完成辅助函数。

上下文契约显式标识相关源码文件和所有权。
安装器在写入任何上下文适配源码之前验证每条选定的 worker/头文件规则。
它拒绝非法源码路径、所有权和阶段，以及改变的 UB 分配表达式或同步/清理行为。
数值阶段计算、复制/队列操作、条件保护和 top-k 分支保留原始顺序。
每个阶段仍构造自己的 pipe；清理在 Destroy 之前等待 MTE3 完成。
生成代码不引入整个 worker 共享的 pipe 复用。

CPU 测试将完整展开后的源码 token 与独立的固定 Gate worker/头文件快照比较。
它们还基于记录调用的 Ascend 接口编译全部十八条初始化/结束规则，检查三种批量/对齐形状下的
精确队列与缓冲区字节数，以及构造、函数体、完成和显式销毁顺序。
原生验收编译实际生成的上下文，使用现有 Gate 数值与生命周期门禁，并执行 MHC/MoE 回归。

这些是固定且受支持的 Gate/MHC 上下文规则。下文描述 MoE 原语上下文生成；
通用可选原语工厂仍属于后续工作。

## P5 MoE 原语上下文生成

MoE 锁定并适配后的 SwiGLU/SwiGLUGrad 入口和实际使用的 GMM Cube 工厂也使用共享上下文规则。
十四条固定规则生成原始实例/Init 片段、GMM 入口局部 pipe/workspace 初始化，以及实际使用的 GMM_CUBE_IMP 宏。
规则标识规范化相对源码路径、重复出现次数，以及函数或宏作用域。
生成的嵌套 include 解析到算子自身的运行时片段。

即使选择单缓冲分支，前向 BF16 仍保留两个相同的双缓冲实例声明。
反向和 half 实例保留原始缓冲区数量。clamp 参数、dtype 分支、Process 调用和作用域退出顺序保持原样。
GMM 保留 AIV 提前返回、转置/FP32 输出选择，以及原始 matmul/compute/process 对象。
入口局部 TPipe 被这些对象借用，并存活到它们按逆序完成析构。
数值原语类头文件及其内部 pipe/队列/UB 行为仍由锁定实现负责。

隔离组装器导出并应用锁定的依赖补丁、复制全部必需源码，再安装上下文胶水代码。
固定的上游仓库和融合/clamp 补丁均保留。
MoE 原生清单封存实际组装的逐 SoC 内核源码闭包，以及暂存的 Torch 适配器、生成器/契约输入和编译库。

CPU 测试在展开 include 并规范化续行之后，比较完整依赖源码 token。
独立的原始/生成 C++ 探针覆盖 dtype、缓冲和 clamp 组合、GMM 转置/FP32 选择、AIV 提前返回、
Init 参数，以及实例/pipe 生命周期。
设备验收使用重新构建的产物和现有数值、生命周期及热副本门禁。

固定且受支持的 worker 和原语上下文规则现在覆盖 Gate、MHC 和 MoE。
通用可选原语组合、更底层 CANN 主机/tiling 生成、通用缓冲区规划，以及二进制/实例化设备缓存仍属于后续工作。
源码包继续保持仅含源码的状态。
