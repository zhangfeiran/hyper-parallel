# 任务域与事件张量 IR

MoE、MHC 和 Gate 的 `Program.plan()` 现在包含 `event_graphs`。
它把既有兼容性计划中的任务域、生产者通知、消费者等待和执行队列转换为同一组不可变 IR，
并校验这些事件规则能够降级回各算子族的原始运行时镜像。

设计参考 [Event Tensor 论文](https://arxiv.org/html/2604.13327v2)。
当前实现从已有算子族计划提取依赖，支持 MHC 受限静态队列变换和 MoE 前缀计数驱动的有效工作域。
尚未从普通 SSA 自动生成完整事件任务图，也没有新增设备动态队列或任意运行时索引函数。

## 查看事件图

```python
from hyper_parallel.core.multicore.frontend import HardwareSpec
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route

plan = _route.plan({"T": 49, "E": 16}, HardwareSpec(40), k=3, scale=2.5)
for graph in plan.event_graphs:
    print(graph.export_manifest())
    # Gate 的顺序关系来自同一 worker 的 FIFO，不需要全局事件计数。
    print(graph.simulate())

emission = _route.compile({"T": 49, "E": 16}, HardwareSpec(40), k=3, scale=2.5)
print([item.path for item in emission.files if item.path.startswith("events/")])
```

`Program.compile()` 将各方向 IR 导出为 `events/forward.json`、`events/backward.json`，
MoE 有副本回退时还会生成 `events/backward_no_replica.json`。
对应的普通和 profiling schedule 镜像通过同一事件 lowering 生成；数值描述符、通信协议和 profiling 字段保留。
生成器身份涵盖事件 IR、编译适配器和绑定代码；事件图进入计划键，缓存会校验导出文件的内容身份。

## IR 契约

| 类型 | 表示内容 |
| --- | --- |
| `TaskGridIR` | 静态多维任务域、原生描述符映射，以及等待与通知映射 |
| `EventTensorIR` | 事件域和每个元素的等待阈值；采用当前从零递增的计数语义 |
| `EventIndexMap` | 覆盖整个任务域的分段仿射索引映射，`None` 表示无事件 |
| `IndexRun` | 区间内的 `base + floor((index-begin)/repeat)*step` 映射 |
| `WorkerQueueIR` | 固定 Cube、Vector 或 Mix worker 的任务 FIFO |
| `EventBoundaryIR` | 原生通信、握手、副本协议或外部调用边界 |
| `EventGraphIR` | 一个算子族执行方向的任务、事件、队列与边界集合 |
| `MoeCountSourceIR` | 原生 `group_list` 的输入别名、专家数和专家/总接收行数上界 |
| `MoeTaskDomainIR` | 专家 × Cube lane 或专家 × Vector tile 的动态有效工作范围 |

任务引用由 grid 编号与域内线性坐标组成。多维任务坐标可通过 `EventIndexMap.at()` 查询事件。
图校验拒绝映射空洞、域外事件、重复或遗漏任务、不足的本地生产者计数，以及越界 worker。
计数阈值、队列放置和广播描述符的表达能力在绑定阶段再次校验。

## 三类执行边界

### Gate

每个前向 worker 都执行同一阶段序列，包含原始空尾部 worker。
反向只包含非空行 worker；后处理的原生外部调用单独保留。
事件图没有全局 `EventTensorIR`，顺序依赖由 worker FIFO 表达。

当前广播描述符无法表达不同 worker 的事件字段，也不能通过修改描述符让 Gate 获得全局事件调度能力。
这类映射和队列顺序变更需要新的设备 lowering。

### MHC

图保留前向 NormCast 的 ring 复用等待、Projection 完成事件和反向宏块汇合阈值。
`simulate()` 验证固定 FIFO 的本地进展；它不模拟指令、缓存可见性、设备流或实际数值。
即使全部通知数足够，等待任务排在其本核生产者之前仍可能死锁，因此必须检查队列进展。

MHC 反向现在支持显式实验策略 `mhc_backward_overlap`：

```python
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec

spec = MhcSpec(12000, 128, num_cube_cores=20, token_tile=64, grad_token_tile=64)
plan = mhc_boundary.plan(spec, event_schedule="mhc_backward_overlap")
emission = mhc_boundary.compile(spec, event_schedule="mhc_backward_overlap")
print(plan.export_manifest()["event_schedule"])
# 设备执行和生成文件使用相同的已变换队列。
```

参数持有模块同样支持 `HyperMegaMhc(128, event_schedule="mhc_backward_overlap")`。
模块策略在构造时固定；无反向的调用使用原始前向队列，训练调用和 checkpoint replay 使用选中的反向队列。

默认 `event_schedule="compatibility"` 保留原始队列。
实验策略在每个 Vector worker 内保留初始化和全部 RMSNormGrad，然后交错其 Prepare 和 Final 任务。
原始固定步长布局可能把同一 tile 的各个阶段分配到不同 worker；
交错过程保留每个任务的原始归属，并通过原有事件等待处理跨 worker 依赖。

受限队列 lowering 拒绝跨 worker 任务迁移、Cube/Mix 重排、事件或阈值变化，以及不能推进的固定 FIFO。
它只改写反向 Vector 队列索引，原生数值描述符、profiling 布局、正向 ring 和各事件均保持原契约。
RMSNormGrad 的 task 模式不使用确定性 reduction workspace；Prepare 和 PhiRms 的数据按 token 范围划分，
局部 TPipe 在任务结束时释放。共享参数梯度仍由原有初始化事件和原子归约保护。

这一策略允许已完成宏块的 Final 与后续 Prepare 重叠。
依赖仿真的首个 Final 进展只证明重叠机会；采用策略前应检查实际形状的数值、完整 step 和 HBM。
`tests/torch/multicore/_ast_mhc_schedule_impl.py` 提供独立训练轨迹、完整前向/反向/SGD step 和 HBM 的测量入口。
当前设备测量中，部分长 token 形状出现完整 step 退化，因此该策略保留为显式实验选择。
更早的局部消费者或更好的仿真进展不能作为默认采用条件。

第二种实验策略 `mhc_backward_wavefront` 按完整 Vector worker 波次推进 Prepare，
然后安排依赖已覆盖宏块的 Final。它通过原有事件映射计算每个 Final 需要的 Prepare 前缀，
保留跨 worker 等待和 Cube 的固定 FIFO，不逐项交错 Prepare/Final。
这一策略旨在减少提前等待的消费者阻塞后续生产者的机会。
与 `mhc_backward_overlap` 一样，它需要显式指定，并遵守相同的所有权、scratch 和进展检查。

```python
plan = mhc_boundary.plan(spec, event_schedule="mhc_backward_wavefront")
# 参数模块和完整 step 测量入口也接受同一策略名称。
```

`mhc_backward_wavefront_fast` 在同一波次队列上，只把 Final 描述符的 `extra_value_2` 设为已有短轮询值 `5`。
它不修改原生 kernel；事件、通知阈值、数值描述符的其余字段和原始前向队列均保留。
当前封存 worker 的单位是 `50` 个 system cycle，值 `5` 对应约 `250` 个 system cycle 的刷新间隔；
默认 Vector 值 `150` 对应 `7500` 个 system cycle。
已有常量名 `FAST_DEPENDENCY_POLL_INTERVAL_US` 不表示当前 worker 按微秒进行换算。
降低轮询间隔可能增加计数器读取流量，因此同样需要通过数值、完整 step 和 HBM 验收。

### MoE

图保留 push/pull 通信发布、入场 ready、pull 退场 completion 和副本协议边界。
原生 transport 的通知不会被仿真计作普通本地通知。
`simulate()` 不自动补齐这些边界；调用方必须显式提供所需外部计数假设，返回结果也会记录这些假设。
这只能证明在给定假设下的本地 FIFO 进展，不能代替跨 rank 通信完成或设备数值验证。

#### 受限动态事件映射

MoE 的 GMM、SwiGLU 和 SwiGLUGrad grid 使用二维任务坐标 `(expert, lane_or_tile)`。
对应事件以 `base + floor(flat_index / tasks_per_expert) * step` 表达，
每个专家的核组共享事件，权重梯度阶段仍可通知原有全局汇合事件。
前向计数输入槽 `6`、`10` 是同一 `group_list` 的别名；反向和副本回退使用槽 `19`。

`group_list[e] - group_list[e-1]` 决定专家的有效行数，首个专家从零开始。
Vector tile 的区间从专家前缀加 `tile * rows_per_task` 开始，并由专家末尾截断。
GMM 的区间表示整个专家数据域；Cube lane 在该区间内的分工仍归原生 tiling。
有效行数为零时，该预留任务不执行数值工作，但仍保留原来的等待、通知和汇合计数。
IR 不把无效数值工作转换为删除队列任务，也不按有效 tile 数缩小事件阈值。

```python
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region

# spec 是已绑定的 MegaMoeSpec；这里示例假定当前 rank 有两个专家。
plan = moe_region.plan(spec)
graph = plan.event_graphs[0]
resolved = graph.resolve_dynamic_events((1, 130))
print(resolved["grids"]["swiglu"]["row_intervals"])
print(resolved["grids"]["swiglu"]["active_work"])
print(graph.grids[1].waits.at((1, 0)))
# 主机快照仅用于诊断；设备执行不调用此解析接口，不拷回计数。
```

绑定时校验原生 `DYNAMIC_DSV3_MOE` 头、输入位置、专家数、连续任务索引、GMM Cube lane 数、
Vector tile 覆盖能力，以及每个专家共享的原生等待/通知映射。
主机快照解析还拒绝非单调计数、专家行数越界和总接收容量越界。
事件 lowering 保留通信、握手、原生 scratch、profile 字段和普通/profiling 两份镜像的契约。

AST 模块资源路径同样构造这些事件图。
`MoeKernelPlan.materialize()` 在设备拷贝前校验既有桥接路径重建的配置与编译镜像一致，
因此动态 IR 不会只留在导出 JSON 中。
前向和反向 launch 为借用的 `group_list` 检查 int64、连续性、长度和执行设备；
热路径不调用 `.item()`、`.cpu()` 或 `.tolist()`，计数内容仍由原生路由和 worker 消费。
这一元数据检查不构成对设备计数内容的独立运行时验证；设备内容的容量约束继续由既有路由协议保证。

这次扩展支持固定容量内的动态有效工作范围和原生专家事件映射。
运行时 `topk` 任意索引、按有效任务数改变通知阈值、重排通信任务、设备 push/pop 队列仍需新的 lowering。

## 验证与后续扩展

CPU 回归验证事件域、汇合计数、队列进展、原生镜像一致性和产物缓存。
原有 Gate、MHC、MoE 设备用例分别验证前向、梯度、流及资源生命周期。
CPU 进展检查、原生代码构建和设备数值放行是独立证据，事件 IR 不声明性能收益。

后续扩展应继续在相同数值任务上比较静态队列变换与同步粒度，
MoE 下一步可在现有前缀差和 tile 截断之外增加经过原生绑定证明的索引操作及任务触发范围，
再实现适合目标 NPU 的队列协议。
新的任务放置需要证明原生任务能力、scratch 所有权和生产者进展条件。
