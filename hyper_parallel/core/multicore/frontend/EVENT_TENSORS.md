# 静态任务域与事件张量 IR

MoE、MHC 和 Gate 的 `Program.plan()` 现在包含 `event_graphs`。
它把既有兼容性计划中的任务域、生产者通知、消费者等待和执行队列转换为同一组不可变 IR，
并校验这些事件规则能够降级回各算子族的原始运行时镜像。

设计参考 [Event Tensor 论文](https://arxiv.org/html/2604.13327v2)。
当前实现是静态兼容性接入：从已有算子族计划提取依赖，尚未从普通 SSA 自动生成完整事件任务图，
也没有新增设备动态队列或运行时符号形状支持。

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

### MoE

图保留 push/pull 通信发布、入场 ready、pull 退场 completion 和副本协议边界。
原生 transport 的通知不会被仿真计作普通本地通知。
`simulate()` 不自动补齐这些边界；调用方必须显式提供所需外部计数假设，返回结果也会记录这些假设。
这只能证明在给定假设下的本地 FIFO 进展，不能代替跨 rank 通信完成或设备数值验证。

动态 `group_list` 仍由现有原生组件消费。
当前 IR 尚不表达运行时 `topk` 索引、动态消费者任务范围或设备 push/pop 队列。

## 验证与后续扩展

CPU 回归验证事件域、汇合计数、队列进展、原生镜像一致性和产物缓存。
原有 Gate、MHC、MoE 设备用例分别验证前向、梯度、流及资源生命周期。
CPU 进展检查、原生代码构建和设备数值放行是独立证据，静态事件 IR 不声明性能收益。

后续扩展应继续在相同数值任务上比较静态队列变换与同步粒度，
再为 MoE 增加受限运行时索引、任务触发范围和适合目标 NPU 的队列协议。
新的任务放置需要证明原生任务能力、scratch 所有权和生产者进展条件。
