# 有上限的专家副本自适应

For the optional exact-signature device dependency scorer, see
[Replica device critical-path model](replica_critical_path.md).

`replica_slots_per_rank` 表示每个 rank 的物理 guest 槽位上限 `B_max`。
大于零时默认根据实际直方图自适应选择副本；零表示 owner-only。
业务代码不传入成本模型，也不需要设置 adaptive 开关。

## 常规配置

进程先选定 NPU、初始化 HCCL 并激活 native payload，然后构造执行器：

```python
import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    hidden_size=5120,
    intermediate_size=2304,
    num_experts=384,
    top_k=6,
    ep_size=dist.get_world_size(),
    ep_group=dist.group.WORLD,
    dispatch_mode="push",
    replica_transport="shmem_signal_kernel_gradient_bands",
    replica_slots_per_rank=6,
).to(device="npu", dtype=torch.bfloat16)
```

Native EP 同样只需 `ExpertParallel(replica_slots_per_rank=6)`。
底层计数规划器使用 `build_expert_replica_plan(counts, replica_slots_per_rank=6)`。
原来的 `replica_cost_model`、`replica_adaptive` 参数已经移除。

| 参数 | 含义 |
|---|---|
| `replica_slots_per_rank` | 物理 guest 槽位上限；实际使用量可以更少 |
| `replica_transport` | MegaMoe replica transport; defaults to `shmem_signal_kernel_gradient_bands` |
| `replica_min_rows` | 软最小副本行数；容量要求的副本可以保留 |
| `replica_target_load` | MegaMoe 期望接收行数，不是丢弃阈值或容量硬上限 |
| `replica_minimum_load_reduction` | 可选的负载削峰要求；显式值覆盖默认策略 |
| `replica_planner` | 默认 CPU；device 规划器保留原有基于负载的实现，不使用主机成本表 |

每次调用自动读取实际 token 数，支持不等长和空 rank，不再传入 `local_num_tokens` 或
`max_local_num_tokens`。Source 和接收存储在全组计数达成一致后自动增长。
Push 规划的接收预留按本次全组最大实际 token 数和 `initial_capacity_factor` 计算，
不受构造时 128 行起始存储的限制；槽位上限不代表整步 HBM 硬上限。

## 内部成本选择

运行时缓存并精确匹配包内校准表，匹配项包括设备型号、H/I、EP、专家数、TopK、后端和传输协议。
当前内置的是 Ascend910_9392、H5120/I2304、E384、TopK6、EP64 的 Push/Pull gradient-band 表。
forward 重复次数根据框架 checkpoint 上下文确定；非 checkpoint 场景保守计入反向权重复制。
没有匹配表时继续基于负载和容量自适应，不外推其他硬件或形状的毫秒数。

仅供调试消融的隐藏环境开关是 `HYPER_PARALLEL_REPLICA_COST_MODEL`，默认 `1`。
设为 `0` 只关闭成本表，负载自适应、容量检查和 checkpoint 计划缓存仍然启用。
该值在创建执行对象时固定，同一 EP 组必须一致。常规训练无需设置此变量。

### Uncalibrated CPU planning

Unmatched workloads retain the relative receive-peak admission policy. If the
owner layout fits the current receive allowance and even an ideally balanced
layout cannot meet the required reduction, the compact count summary selects
owner-only execution before full CPU planning. Integer rounding, absolute
saved-row floors, capacity-required copies, and ordered checkpoint replay remain
the same as in the full planner. An existing native count agreement needs no
additional histogram readback or collective for this decision.

With positive admission and fragment thresholds, a slot cap of at least two,
and an owner peak at least three times the average, the default CPU policy may
compare one additional completed single-guest seed. It reuses the balancing
skeleton and can rescue a useful layout rejected after thin-fragment pruning.
The proposal must satisfy the original admission and receive allowance, keep
each expert at or below the average rank load, and keep every receiver below
twice that average. Replacing an already active layout additionally requires
halving both matrix edges and the maximum owner fanout. This can trade some
receive rows for fewer fixed matrix transfers; it does not estimate latency.
Light routes retain the existing compact rejection, and no public option,
calibration table, or extra collective is required.

The final uncalibrated layout also consolidates redundant guest matrices above
the small-copy threshold. Existing copies of the same owner may take over work;
no new matrix edge is created and the receive peak cannot increase. Cross-node
edges and owners with larger fanout are considered first.

After final locality placement and consolidation, a bounded pass can refill
already retained guest matrices from their own logical expert's remaining home
rows. Existing guest rows and uncopied home rows establish a lower bound on the
expert peak, so the pass skips home fragments that cannot reduce it. It creates
no matrix edges or active slots. The proposal must reduce the largest expert
fragment while keeping the rank receive peak and every source/node traffic
objective nonincreasing. This pass uses no
additional budget candidates, public parameter, calibration table or collective.
Zero admission/fragment compatibility policies, explicit placement experiments,
and matched cost models retain their existing paths. Structural improvement is
not a latency guarantee.

The optional ABI3 host extension can emit the same ordered whole-guest locality
moves as Python. Python applies each move in sequence to preserve dictionary
insertion order and remains the fallback for unsupported integer ranges or
older extension builds. This CPU primitive does not change device kernels.

For training, MegaMoe derives an internal activation-row credit from its own
expert shape and gradient flags. With intermediate dimension `I`, it credits
`floor(I * (2 * grad_gate_up + grad_down) / 6)` extra remote rows per deleted
matrix edge. The credit is zero when autograd is disabled. A routed row uses
at most `12 * H` BF16 payload bytes for dispatch and combine across forward,
one checkpoint replay, and backward. Thus the credit spends at most half the
deleted projection's mandatory FP32 gradient payload, reserving the rest as
margin. No weight-copy savings or residency hits are assumed.

The planner checks exact source and node deficits before accepting a deletion.
Only a deleted cross-node matrix edge can pay for additional cross-node token
rows. Frozen projections contribute no credit. Shape-free planner callers use
zero credit and cannot increase either remote-row count. Matched cost models
keep their existing scoring, and the device planner is unchanged.

This is a conservative payload heuristic, not a latency calibration or an
end-to-end speed guarantee. Hot-route CPU search costs and device critical paths
still require matched measurement. No public tuning parameter or native ABI
change is needed for these rules.
不支持运行中改变该变量来修改已经创建的执行对象。

有匹配成本时，默认准入按预测收益和不确定性余量决定；没有匹配成本时，
沿用默认的负载削峰要求。显式 `replica_minimum_load_reduction` 始终有效。

## 容量和生命周期

主预算候选共用一次均衡骨架和成本插值缓存，最终只为选中布局生成一次 dispatch。
没有校准表时，以负载和容量为依据选择预算，并在需要时利用剩余 guest 位置降低接收峰值。
候选集合有限，不保证全局最优或实际整步一定提速。

物理编号、权重池、梯度池和通信拓扑一直按 `B_max`，减少实际副本不会释放已预留的槽位内存。
CPU 计划的 `replica_slots_used` 返回各 rank 的实际使用量。
没有副本且 event 偏移兼容时使用 home-only 图。

Push 根据已绑定的对称堆预算、其他 workspace 和当前 source 容量计算接收硬上限。
初始接收预留仍是布局偏好，实际选择后才分配或扩容，B0 和 B>0 使用相同规则。
Pull/Native 的接收张量按所选布局分配。底层规划器的 `capacity_limit` 可表达接收硬约束。
checkpoint 重算检查计数、配置和调用顺序后复用原计划；接收预留扩大不会改变原调用的重算布局。
backward 保留本次调用自己的不可变计划和权重 generation。

实验开关默认为关闭，须在作业启动前对所有参与 rank 一致设置：

| 环境变量 | 启用时的行为 |
| --- | --- |
| `HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH=1` | 无表 B2 强热点下比较完整配额，只接受仍装入两个物理槽且改善接收峰的候选。 |
| `HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH=1` | 用拓扑感知的准入规则比较可投影到 B2 的更宽配额。 |
| `HYPER_PARALLEL_SHMEM_FIXED_HEAP_REPACK=1` | 堆字节数不变时保留 SHMEM runtime，仍同步并重建 workspace 分配。 |
| `HYPER_PARALLEL_REPLICA_RANK_LOCAL_COMM=1` | 按 rank 裁剪空 combine/Pull dispatch，按实际发端同步修正接收信号计数。 |

校准表按运行签名匹配，包含上述开关和通信图身份。

本版不自动探测可用 HBM、不在线采集校准表，也不做副本池热扩缩容。

## 节点内优先与链路成本

CPU 规划器在运行时自动读取 EP rank 的节点归属，不假设每节点固定 16 rank，
也不使用 `rank // 16` 推断同机关系。初始化优先采集内核 boot ID，以识别同一机器内的多个容器；
不可用时回退到 hostname。MegaMoe 复用已有的资源 manifest 交换，Native 每个并行策略与
process group 首次配对时交换一次并缓存；训练每层的路由计数 AllGather 不增加。

选出原始有界自适应布局后，最多追加一次同机候选。已有副本按完整行数移动、合并或交换，
不继续拆成小片段，也不重复执行完整预算搜索及成本配额细化。原布局没有副本时直接跳过这一步，
避免为已经采用 B0 的层增加额外探测成本。槽位上限、接收硬约束和准入余量仍然有效。

有校准值时比较最终布局的预测 forward/recompute/backward 成本，原方案始终作为回退；
缺少校准值时先比较接收峰值，相同时优先跨节点副本更少的方案。
同一节点整体过载时仍允许跨节点放置。优化不增长原接收预留或原方案所需的接收峰值，
也不绕过模型的校准行数覆盖检查。重算缓存包含不可变节点映射。

内部校准表支持可选的 `intra_node_weight_ms`、`inter_node_weight_ms`、
`intra_node_gradient_ms`、`inter_node_gradient_ms` 及相应 backward weight 项。
缺失项沿用原统一系数；没有写入假定的本机/跨机带宽比。
已验证的 resident backward 权重仍按零次复制计费。
这些字段不进入 `MegaMoeExperts` 或 `ExpertParallel` 的业务构造参数，隐藏成本开关保持原样。

离线采集入口为 `tests.torch.expert_parallel._benchmark_replica_links`。
它在实际 EP 节点映射上选择节点内与跨节点 peer，用生产 P2P/SHMEM provider 传输完整 BF16 专家矩阵，
并验证完整矩阵内容和 FP32 梯度返回。设备事件耗时与 CPU wall time 分开保存；
完整 eager 梯度返回时长不自动代替融合 gradient bands 的暴露残差。
输出包含原始样本、权重链路系数，以及与现有匹配表同名的候选文件；
采集不会修改已安装的成本表或已冻结训练包。
候选需要在独立源码/包中替换原匹配条目，再用 profiler-off 对照确认整网收益。
