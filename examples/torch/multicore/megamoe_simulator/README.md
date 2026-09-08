# MegaMoe 多节点与热专家拷贝模拟

在普通 CPU 上模拟 8/16/32/64 个逻辑 rank，运行当前 checkout 的真实
`build_constrained_expert_placement()`，研究副本预算、跨节点带宽、路由热点、
参数更新和多层共享副本池对训练时间的影响。不需要 Docker、NPU 或启动同等数量的进程。

**输出是通信与计算模型的估算，不是多机 NPU 性能或数值正确性验证。**
本工具不会执行 MegaMoe 的 C++ kernel，不会验证 RDMA 协议，也不会对现有 NPU 通信限速。
适合先筛选热专家拷贝方案；目前不适合用来排序 AIV tile 调度、ACK 或指令级优化。

## 运行

从仓库根目录执行。脚本使用 NumPy；planner 直接从本 checkout 的纯 Python 源码加载，
不会通过 `hyper_parallel` 顶层导入初始化 Torch/MindSpore。

```bash
python -m examples.torch.multicore.simulate_mega_moe \
  --ranks 16 32 64 --ranks-per-node 8 \
  --experts 384 --tokens-per-rank 4096 --top-k 8 \
  --hidden 5120 --intermediate 1792 \
  --slots 0 1 2 --capacity-factor 1.0 1.5 \
  --inter-gbps 12.5 25 50 --inter-latency-us 2 10 30 \
  --layers 2 --steps 2 --output output/megamoe-sweep
```

上述是完整参数网格，场景较多。先跑默认的 6 个场景即可确认环境：

```bash
python -m examples.torch.multicore.simulate_mega_moe \
  --ranks 16 32 64 --slots 0 1 --layers 1 --steps 1 \
  --tokens-per-rank 256 --planner-us 0 --output output/megamoe-smoke
```

`--inter-gbps` 的单位是 **十进制 GB/s、每逻辑节点、每个方向的总有效带宽**。
例如 8 个 rank 同时向其他节点发送时共同分享 25 GB/s。它不是每对 rank 的额度，
也不是 Gb/s。若目标机器有多个独立网口，先用总有效带宽作为节点上限；当前版本不区分网口/rail。

固定全局 `experts=384` 可被 16/32/64 整除，分别对应 24/12/6 个 home experts/rank。
当前 48 专家配置不能直接用于 EP32/EP64。默认固定全局专家数及每 rank token 数，
所以增加 rank 时全局 token 数也增加。比较固定全局 batch 时应另行降低 `--tokens-per-rank`。
容量按当前 `spec.py` 的规则向上对齐到 128 行。

## 复用当前实现的范围

| 内容 | 来源/处理 |
| --- | --- |
| 选择哪个专家复制、放在哪个 rank、每副本分配多少 token | 直接运行 `placement.py`，没有另写负载均衡算法 |
| token 流量与多 wave 切片 | 使用真实 `dispatch_counts`，按目标 expert-major/source-rank 顺序切片 |
| 权重预取 | 每条 `plan.replica_transfers` 分别发送 W1 `[H,2I]` 与 W2 `[I,H]`，BF16 |
| 副本梯度 | 按同一 transfer 反向传回 owner，W1/W2 均使用 FP32 |
| shared B-slot 池 | 以层参数身份、参数版本和完整 slot map 为缓存键；跨层覆盖后重载 |
| checkpoint | all forwards → reverse backwards；reentrant 重算完整 forward，随后 backward 再取得租约 |
| 参数更新 | 默认每步更新，下一步缓存失效；`--update-every 0` 保持版本不变 |
| 路由变化 | `--rotate-experts N` 每次初始 forward 旋转 N 列；重算和 backward 使用保存的同一 plan |

`test_megamoe_simulator_transfers.py` 在 16/32/64 ranks 上调用真实
`ExpertReplicaManager._prefetch_transfers()` / `_gradient_transfers()`，使用 meta tensor
逐条核对 source、target、矩阵字节数。它只验证传输描述契约，不执行张量数学或通信。

默认 `--pool shared` 对应当前多层共享 B-slot pool。`--pool per-layer` 模拟各层独立池，
用来评估额外副本显存换取更少反向重载的取舍；不是新增的生产接口。
没有实现 compact/dense physical weight cache 的其他生命周期。

对于两层共享池、固定 plan、每步更新参数的情形，单步典型顺序是：

```text
L0 forward: 复制 L0
L1 forward: 覆盖成 L1
L1 recompute / backward: 复用 L1
L0 recompute: 重载 L0
L0 backward: 复用 L0
optimizer: 参数版本变化
```

因此不能把“同一层重算命中缓存”等同于“所有层反向都不需要重拷贝”。
反向 FP32 梯度数据量还需要单独计入，不能沿用 BF16 权重字节数。

## 网络和计算模型

所有通信在同一个虚拟时钟和 fabric 中调度：副本权重、FP32 梯度、token、count gather、
cache agreement 和 wave 同步共同竞争相应资源。发送端总出口和接收端总入口都限速，
相反方向独立。机内传输受每 rank 发送/接收带宽和节点本地 fabric 总带宽限制。

传输采用流体 max-min 公平共享。新流加入或旧流结束时重新分配带宽。
序列化结束后释放带宽；传播延迟结束后才释放依赖，所以不会把传播延迟错误地串行叠加。
测试覆盖共享发送/接收瓶颈、双向同时发送、传输中途加入新流、带宽再分配和接收可见性。

模型粒度为 rank/wave：收齐该 rank 当前 wave 的数据后计算，再发送 Combine。
不同 rank 的计算和通信可重叠；没有重现当前 kernel 的逐 expert/tile 流水。
每个 rank pair/wave 合并为一个 token 传输，保留 payload 总字节数，
**不保留实际小消息数量、tile ACK 数量及其调度顺序**。
多 wave backward 额外计入 hidden dispatch，所有 rank 都参与 wave 同步。
小集合通信使用 recursive doubling 作为明确的算法假设；实际 HCCL 可能选用其他算法。

默认计算时间使用 GEMM FLOPs / 假设的有效 TFLOP/s 加每 wave 固定开销。
它不是 910B3 峰值，也没有自动以其他 worktree 的历史性能作为校准。
向量算子、本地内存操作等可通过固定开销近似；当前不单独建模 HBM 与通信争用。

默认对当前 planner 实际测量三次，取中位数作为每次 placement 的 CPU 开销，
并将测量值单独保存。每个虚拟 rank 使用该开销；未模拟多个进程竞争同一主机 CPU。
做严格网络敏感性比较时可用 `--planner-us 0` 或固定值，避免当次 CPU 负载改变对比。
`route_us` 用于容纳其他路由准备和 D2H 开销，不包含在纯 planner 测量中。

可以传入校准文件：

```json
{
  "costs": {
    "effective_tflops": 100.0,
    "forward_fixed_us": 5.0,
    "backward_fixed_us": 10.0,
    "route_us": 5.0,
    "optimizer_us": 0.0
  },
  "provenance": {
    "kind": "assumed_example",
    "description": "Replace these values with matching shape/toolchain measurements"
  }
}
```

```bash
python -m examples.torch.multicore.simulate_mega_moe --costs costs.json --planner-us 100
```

`optimizer_us=0` 表示不计 optimizer 执行时间；参数版本失效仍发生。
有效算力、带宽和固定开销均未校准时，适合比较敏感性趋势，不应宣称真实多机加速比。

## 路由和拓扑重放

`--routing balanced|hotspot|dirichlet` 分别生成均衡、固定热点、Dirichlet 分布。
hotspot 将 `--hot-fraction` 比例的 token 送到相同的 Top-K 专家，其余均匀分配。
Dirichlet 使用 Gumbel Top-K 无放回抽样，不允许同一 token 重复选择同一专家。

每次运行保存 `rankN.counts.json`。可以替换为真实路由计数：

```bash
python -m examples.torch.multicore.simulate_mega_moe \
  --ranks 32 --counts output/megamoe-sweep/rank32.counts.json \
  --experts 384 --tokens-per-rank 4096 --top-k 8 --slots 0 1 2
```

输入是二维 `[source_rank, logical_expert]` 矩阵，或含 `counts` 键的对象。
每行必须合计 `tokens_per_rank * top_k`，每个专家计数不得超过该 rank 的 token 数。
不把 8-rank 计数静默复制成 64 ranks；不同规模需要自己的输入。

`--node-map mapping.json` 接受 rank→node ID 数组，仅限一次指定一个 rank 数。
例如 EP8 下 `[0,0,0,0,1,1,1,1]` 模拟两个 4 卡节点；
EP64 默认按连续 8 ranks 分成 8 个节点。自定义映射可研究同一 placement 的拓扑敏感性。

## 输出

- `summary.json`：源码 SHA256、Git SHA、完整参数、校准来源和局限说明。
- `summary.csv`：每个场景的模拟步耗时、跨节点权重/梯度字节数，方便画曲线。
- 每场景 JSON：每步耗时、所有 plan/slot bindings、缓存命中/重载、分阶段流量、资源流量。
- `--trace`：每场景 Chrome/Perfetto trace，可观察权重、token、计算和梯度的相互等待。

`replica_pool_bytes_per_rank` 只包含 B-slot 权重和梯度，不能当作完整训练峰值 HBM。
`compute_busy_fraction_by_rank` 只代表模型里的 GEMM 阶段，不能当作真实 NPU 利用率。

## 单卡多 rank 功能探测

PyTorch 2.9.1 的 `torch.distributed.run` 中没有 `allow_busy_npu` 参数。
`--nproc-per-node` 可以创建更多进程，但需要 worker 自己将 `LOCAL_RANK` 映射到设备，
且通信库必须支持这些 rank 使用相同物理设备。它不保证 HCCL/SHMEM 支持同一通信域内的设备复用。
即使小集合通信成功，持久化 MegaMoe kernel 的并发驻留和等待也需要另外验证。

提供带 deadline 的探针，逐步记录 process-group 初始化、all-reduce 和 P2P：

```bash
# CPU 上验证探针自身，不占 NPU。
python -m examples.torch.multicore.probe_mega_moe_ranks \
  --backend gloo --ranks 2 --output output/probe-gloo

# 按正常设备空闲门控执行；先跑不同设备的控制组。
python -m examples.torch.multicore.probe_mega_moe_ranks \
  --backend hccl --ranks 2 --devices 0 1 --output output/probe-distinct
python -m examples.torch.multicore.probe_mega_moe_ranks \
  --backend hccl --ranks 2 --devices 0 --output output/probe-shared
```

HCCL 内部连接 timeout 至少为 120 秒；探针自己的外部 deadline 默认 60 秒，
超时清理属于该 launcher 的进程组。每次使用新的输出目录，避免混入旧结果。
可选 `--shmem-init` 再测试当前 adapter 的 SHMEM 分配/barrier；要求已构建原生库。
这也不等于验证 token Dispatch/Combine 或 MegaMoe 的前向/梯度。

## RC2 与当前分支的 IP 配置

核查起点是 `megamoe-sun@fa1a1859`，本地 `megamoe-api-rc2@c871f609`。
当前分支已在 `platform/torch/symmetric_memory/lifecycle.py` 读取 `SHMEM_IP_PORT`，
不需要为了模拟重新移植 RC2 的旧生命周期。RC2 的 subgroup endpoint 隔离逻辑
不代表当前 whole-world 生命周期已有同样的 subgroup 契约。

已有 Qwen 启动脚本支持 `MASTER_ADDR`、`SHMEM_HOST`、`SHMEM_PORT`、`NODE_RANK`、`NNODES`，
但该示例固定 EP8。SHMEM 初始化仍选择 MTE，MegaMoe kernel 仍使用直接可访问的远端地址。
这些 IP 参数解决的是启动/会合；普通跨机 RoCE 数据通路仍需要单独适配并验证。
Docker/network namespace 不会自动把 MTE 流量转换成可由主机 `tc` 限速的 TCP 流量。

## 验证

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/examples/torch/multicore/test_megamoe_simulator.py \
  tests/ut/examples/torch/multicore/test_megamoe_simulator_transfers.py \
  tests/ut/core/multicore/mega_moe/test_placement.py
```

新模拟器没有修改 production planner、replica manager 或 kernel。
后续改动 planner 时，重新运行相同保存的 counts、参数和网络模型，并比较源码 digest、
副本流量、重载次数、wave 数与模拟时间。筛选出的方案再进行真实设备数值与性能验证。
