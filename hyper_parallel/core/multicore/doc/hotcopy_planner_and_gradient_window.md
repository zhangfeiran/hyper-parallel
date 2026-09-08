# 热专家 planner 与梯度接收窗口

`replica_slots_per_rank`（B）限制每个计算 rank 保存多少份专家副本。
一个 home owner 可以同时拥有来自很多 rank 的副本，因此 B 本身不能限制 owner 的梯度接收显存。
`replica_grad_receive_slots` 单独限制这个接收窗口，默认上限为 8，必须在所有 EP rank 上配置一致。

```python
experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=384,
    top_k=8,
    ep_size=64,
    ep_group=ep_group,
    expert_capacity_factor=1.5,
    replica_slots_per_rank=1,
    shared_physical_weights=True,
    replica_grad_receive_slots=8,
)
```

这是容量配置示例；实际运行还需要底层通信和 kernel descriptor 支持对应的 EP 规模。
当前 descriptor 的任务上限仍是 25600，本轮没有解除此前 EP64 配置触发的任务容量限制。
修改 B 与接收窗口是两个独立的调优动作：前者影响负载均衡、权重复制量和剩余 wave 数，
后者在既定 placement 上平衡梯度接收内存与通信并发。

## Planner 保持等价的加速

贪心选择每次先构造非空、来自过载 rank 的候选副本列表，再遍历可接收的 rank。
原来的评分、target/expert/source 遍历中的平局处理、精确均衡时的链式搬移、
带容量余量时仅搬移溢出部分的规则保持一致。缓存策略和 per-source count matrix 也保持原有语义。

2026-09-08 的单进程 CPU 对照覆盖 EP64/E384 的 228 个配置，以及 EP16/32/E384 的 30 个配置。
全部物理 slot map 和 source-to-physical dispatch matrix 一致。三组 EP64 A/B/B/A 中位耗时：

| 路由 | B | 原 planner | 当前 planner |
| --- | ---: | ---: | ---: |
| 集中的 8 个热点专家 | 1 | 359.16 ms | 89.43 ms |
| Dirichlet α=0.1 | 1 | 307.61 ms | 132.06 ms |
| 192 个热点专家 | 3 | 916.52 ms | 394.60 ms |

这些是 Python planner 的计算耗时；NPU 到 CPU 的计数回读、缓存查找及多进程竞争需要另行计算。
当前实现仍需在 CPU 上规划，高频 cache miss 下仍可能成为瓶颈。

将 CPU int32 count tensor 经由实际 `route._build_replica_plan()` 调用时，三组冷缓存耗时分别从
361.75/310.65/916.79 ms 降至 94.59/136.53/399.14 ms，缓存命中约 5 ms。
这项检查包含 `tolist()`、缓存键生成与规划，未包含 NPU 计数回读。

## 有界接收与 peer 并发

梯度归并从完整 placement 导出所有 rank 一致的收发批次。每个 owner 只分配一次 W1/W2 接收张量，
每批最多使用配置的槽数；`Work.wait()` 将接收完成依赖接入当前计算 stream，
累加消费后下一批才能覆盖这些槽。每次 reduction 独立持有接收张量，避免跨层和并发 backward 互相覆盖。

批次先按 home expert 排列其副本，使不同 peer 能同时供数；稳定排序保留同一专家原有的梯度累加顺序。
活跃 peer 数小于窗口上限时，批次宽度取不超过上限的完整 peer 轮次。例如上限 8、活跃 peer 为 7 时，
实际宽度为 7，避免第八份数据落到一个已在发送的 peer 上形成尾部瓶颈。
多个 owner 可以在同一批通信；没有收发任务的 rank 仍遵循相同的全局批次顺序。

FP32 梯度直接累加到 home 梯度。BF16/FP16 输入按 home slot 分组，使用一个专家的 FP32 累加张量，
全部副本到齐后再写回低精度输出。接收分批不引入跨批次的低精度舍入。

单个专家包含 `P = 3 × H × I` 个参数。通常的 FP32 dW 路径，接收张量的有效数据容量至多是
`replica_grad_receive_slots × P × 4` 字节。H=5120、I=1792 时，每份完整 dW 为 105 MiB：

| 接收上限 | 最大有效数据容量 |
| ---: | ---: |
| 2 | 210 MiB |
| 4 | 420 MiB |
| 8（默认） | 840 MiB |

实际分配随本 rank 的接收数减小，allocator 的块对齐可能增加少量容量。
BF16/FP16 兼容路径还需一个专家的 FP32 累加存储。
这项上限不包含 home/replica 梯度本体、optimizer、激活、SHMEM、其他 kernel scratch 或通信库内存。

## 单机 NPU 实测与 EP64 模型

Ascend 910B3 ×8、Torch 2.9.1、CANN 9.0.0 的独立梯度归并测试，固定 H=5120、I=1792、FP32。
测试 placement 将 rank 0 的 6 个专家复制到其他 7 个 rank，owner 共接收 42 份 dW。
这是用于压力验证的合法 placement。每个 A/B/B/A 点使用新进程，预热 2 次、测量 5 次。
表内耗时为两次同组运行的中位耗时均值。

| 实现 | 归并耗时 | owner 额外 allocated 峰值 |
| --- | ---: | ---: |
| 原实现 | 45.03 ms | 4452 MiB |
| 有界接收，上限 8、实际宽度 7 | 45.69 ms | 736 MiB |

接收阶段额外 allocated 峰值降低约 83.5%，该场景耗时增加约 1.5%。这些数值不包含专家 GEMM 或完整训练步。
按 sender 顺序直接切 4 槽的实验曾达到约 211 ms；调整 peer 分布后约 72 ms。
因此不能只减小窗口而忽略 peer 并发，紧张的显存预算下可显式选择 4 或 2 并测量完整 step。

EP64/E384 的 fluid network 模型使用 8 节点 ×8 rank、每节点每方向 25 GB/s、10 μs 延迟、
100 有效 TFLOPS、800 GB/s 梯度累加带宽、两层 reentrant checkpoint。
模型在两种实现上都计入梯度累加流量，每种通信 phase 的总 payload 保持一致：

| 路由 | B | 原归并的模拟 step | 默认窗口的模拟 step |
| --- | ---: | ---: | ---: |
| 集中的 8 个热点专家 | 1 | 2242.41 ms | 2262.46 ms |
| 192 个热点专家 | 3 | 1769.27 ms | 1769.27 ms |

以上模型不计 planner CPU、HCCL host launch 或实际 tile/ACK 协议。
集中热点中 owner 的 47 份 dW 有效接收容量由 4935 MiB 限制为不超过 840 MiB；这是源代码容量推导。
全模型 OOM 仍需按 forward、backward 和 optimizer 的真实存活阶段统计，且需要预留 allocator 与通信库空间。

## EP64/E384 的使用建议

沿用本次模拟分布时，可从 B=1、capacity factor=1.5、接收上限 8 开始。
若热点覆盖面扩大，按既有 sweep 中的 96/192 热专家分布，分别测试 B=2/3。
等价 planner 加速不会改变所需的副本数或 wave 数，B 的选择仍取决于实际路由分布。

CPU 规划成本下降会改变 B 的收益判断。例如按两层每步 2 次冷缓存规划、2 次命中的预算，
192 热专家、B=3 的数据路径与 CPU 控制开销之和由约 3613 ms 降到 2577 ms；
此前同一分布的 B=0 数据路径模拟约为 2923 ms。
这里把代表性单进程 planner 耗时加到模拟关键路径上，适合筛选实验，仍需完整训练步验证。

显存紧张时先测接收上限 4，再根据吞吐与峰值取舍。不要把缩减 B 作为固定的 OOM 回退：
减少副本可能增加最大接收 token 数和保留激活，抵消权重池的节省。
后续优先减少冷缓存 planner 的关键路径，例如分开缓存全局负载平衡结果和 per-source 分配，
以及在 NPU 上构建规划与 remap 元数据；完整 per-source counts 仍是正确路由所需的数据。

## 验证与参考

- 142 个相关单元测试以及 128 个 subtest 通过。
- 8 个真实 NPU rank 验证多对一、链式与空 placement，FP32/BF16/FP16，窗口 1/2/4/8，
  dense 和独立 B 梯度池，以及非默认 stream 上的接收复用。
- 16/32/64 个真实 Gloo CPU 进程验证 B=1/6、窗口 4/8、FP32/BF16。
  此项固定每个 rank 的 home 专家数为 6，EP64 对应 E384；它验证收发与归并协议。
- 两个 NPU rank 上的热副本单 wave、剩余 multi-wave 和延迟 backward 与 native 对照通过。
  共享池生命周期另通过 21 组对照，包含两层、重算、保留计算图、参数更新和跨 stream 使用。
  这些完整 kernel 检查使用同一基线工作树提供的既有 payload，运行环境为 CANN 9.1.0。

[MoonEP README](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/README.md)
描述了跨层共享 B 池与 owner 读取远端副本梯度的契约。
[MoonEP GradReduceKernel](https://github.com/MoonshotAI/MoonEP/blob/2bd860b4dd083df62b79d5e916fca71ec5742228/moonep/grad_reduce.py)
采用固定 tile 的 FP32 累加。在当前 Ascend 路径中，该思路转化为 HCCL 的有界接收与原位累加。
MoonEP 训练的 B=E/R 要求对应其完整均衡与权重访问契约；MegaMoe 的剩余溢出由 multi-wave 执行，
本次等价 planner 优化不会将 B 强制提高到 E/R。

[ProcessGroupHCCL](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp)
中的 stream 同步和 Work 等待是复用接收存储的必要依赖。
实际多机 token SHMEM、EP64 kernel descriptor 容量及多机端到端训练需要各自验证。
