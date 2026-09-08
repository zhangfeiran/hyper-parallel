# MegaMoe 离线成本模型与任务 profiler

本实现把路由、专家放置、六个 GEMM、跨 rank 数据流和显存生命周期放在同一个可复算模型中。普通预测只需要 Python 和 NumPy，不导入 Torch、不创建 NPU context、不执行 kernel。硬件配置可以使用假设值，也可以导入一次采集得到的校准数据；后续更换 B、容量、rank 数、节点网络和层数时，不需要逐点上卡。

这是用于筛选优化候选的第一版模型。它精确保留选定协议下的路由行数、有效 FLOPs 和传输字节，使用近似的调度和有效速率预测时间。它尚不能代替所有真实测量，尤其不能把单机校准的误差范围直接用于多机。

## 实现入口与责任

| 入口 | 内容 |
| --- | --- |
| `examples/torch/multicore/predict_mega_moe.py` | 无设备预测、B/C 扫描、显存和构图可行性筛选 |
| `examples/torch/multicore/megamoe_simulator/cost.py` | GEMM/vector 工作量、有效速率、按形状和精度区分的服务曲线 |
| `megamoe_simulator/estimate.py` | 前向、重计算、反向、热拷贝和优化器的依赖图 |
| `megamoe_simulator/network.py`、`accelerated.py` | 共享节点出口/入口带宽、竞争和传播延迟 |
| `megamoe_simulator/memory.py` | 唯一 storage 的分配/释放区间、别名、SHMEM 和历史池容量 |
| `megamoe_simulator/profile.py` | 任务记录、Chrome trace、训练集校准和预留集误差 |
| `examples/torch/multicore/audit_mega_moe_plan.py` | 可选的 Torch meta 构图审计，不启动通信和 kernel |
| `examples/torch/multicore/profile_mega_moe_costs.py` | 单独采集真实耗时、显存、精度及诊断数据的分布式 worker |

模型直接加载生产代码 `placement.py`，使用真实放置算法；梯度窗口分组也与 `ExpertReplicaManager` 共用一个纯函数。模型结果保存成本公式、生产源文件的 SHA256，以及使用的硬件、拓扑、训练契约和 counts 摘要。

## 时间公式

记每 rank 输入 token 数为 T，Top-K 为 K，隐藏维度 H，中间维度 I，EP rank 数 R，总专家数 E，额外物理槽位为 B，接收容量为 C。每个专家的参数元素数为 `P = 3HI`。

对某一 wave 中接收 m 行的专家：

| 阶段 | GEMM 的 M/N/K | 有效 FLOPs |
| --- | --- | --- |
| up projection | m / 2I / H | 4mHI |
| down projection | m / H / I | 2mHI |
| activation gradient | m / I / H | 2mHI |
| gate input gradient | m / H / 2I | 4mHI |
| W1 gradient | H / 2I / m | 4mHI |
| W2 gradient | I / H / m | 2mHI |

因此前向为 `6mHI`，反向为 `12mHI`；checkpoint 的完整重计算额外增加一次前向。模型还计算 16 元素 tile 对齐后的算术量，但这只是对齐下界，不代表完整 L0/L1 tiling。

每个 GEMM 的基础服务时间为：

```text
fixed_us + max(padded_flops / effective_flops_per_us,
               logical_matrix_bytes / effective_hbm_bytes_per_us)
```

逻辑矩阵字节包含两个输入矩阵和一个输出矩阵，不包含所有 cache 重读。因此它是校准前的工作量基准，不是完整 HBM 总线流量。最终服务时间乘以对应阶段、H/I、输入与输出精度、专家行数的校准倍率。校准按 2 的幂次聚合行数，取桶内中位数，在 log(row) 上插值；边界外使用端点值，并在报告中明确标记超出校准范围。

按行数区分尤其影响 dW：m 在 dW 中是归约维度 K，不能用均衡专家得到的一个倍率覆盖大专家。本轮任务记录中，长 K 的 dW 服务时间可比原固定倍率模型高约 60%。

SwiGLU、反向逐元素计算、端点归约和 owner 梯度相加按显式读写字节计算带宽服务。非 atomic 的跨 wave dW 累加只发生在后续 wave；inplace backward 的 hidden replay、全局 wave 栅栏和重计算均有独立阶段。

### 通信与关键路径

rank 内的阶段通过依赖连接，跨 rank 的传输共用 fabric。`inter_gbps=25` 表示**每节点、每方向合计 25 GB/s**，不是每个 rank pair 都得到 25 GB/s。传播延迟在传输完成后影响可见时间，不继续占用带宽。节点内有 rank 收发带宽和节点交换容量约束。

热拷贝包含 BF16 权重预取、版本一致性确认、checkpoint/backward 重新绑定，以及 FP32 副本梯度返回和 owner 相加。共享池被其他层占用或优化器修改版本后，不能假定仍然命中。预取允许与 CPU 路由准备重叠。接收窗口保留生产实现的 peer 对齐和批次顺序。

`critical_path.phase_ms` 沿最终完成任务的依赖路径追踪，可以相加得到模型的总时间。`phase_rank_max_ms_nonadditive` 是每阶段各 rank 的最大区间，可能包含重叠和等待，**不能相加**。`step_ms` 是各 step 的独立耗时，`step_end_ms` 是累计边界。

当前调度粒度仍是 rank 的阶段，尚未重放每个专家 tile 的 AIV/Cube FIFO。它可能高估可重叠工作，也可能漏掉队首阻塞和协议等待，不构成耗时上界。

## HBM 生命周期

模型给每块独立 storage 一个 `[allocate, release)` 区间，遍历同时存活的集合求峰值。同一底层 storage 的参数视图、SHMEM 视图和梯度别名只计一次。

主要契约包括：

- 固定 SHMEM heap 独立列账，按运行时粒度取整；它不是每层独立的一组普通 tensor。
- overflow 的 retained up projection 仍按全局最大接收行数分配；wave 内临时张量按 C 分配。减少 C 不会同比缩小 retained up projection。
- 参数、主权重、优化器状态、参数梯度、kernel FP32 梯度和临时接收区按各自生命周期计账。
- B=0 的 BF16 kernel dW 可以与参数 `.grad` 共用 storage；使用副本 runtime 时 FP32 dW 到 BF16 `.grad` 的转换存在重叠峰值。
- 每个独立层保留自己的参数和 checkpoint 输入；共享 execution workspace 和共享副本池只计一份。
- `previous_scratch_bytes`、`previous_replica_slots` 描述历史路由留下的高水位。当前路由转为均衡不等于已释放这些 storage。
- 配置 B 后，如果 home placement 已满足 C，生产模块选择无副本 runtime。模型同步选择该路径，不强行计算 FP32 dW。
- `plan_storage_bytes` 可以接收真实 meta 构图统计，补齐静态任务/tiling tensor；未提供时报告会列出缺项。

H=5120、I=1792 时，一个专家的 BF16 权重为 52.5 MiB，FP32 梯度为 105 MiB。窗口 8 的 FP32 接收区最多 840 MiB，实际宽度还受 peer 对齐影响；这不包括副本池、owner 梯度和其他活跃张量。

`optimizer="fp32-main-adamw"` 表示 BF16 参数、FP32 主权重及两个 FP32 moments，优化器阶段还计入 FP32 主梯度；`bf16-adamw` 按参数精度保存 moments；`none` 不包含优化器状态。这里不推断 FSDP 分片、offload、混合训练框架的额外副本。

同一次预测中，除 owner 接收窗口外各 rank 使用相同全局容量。显存模型按相同接收宽度复用生命周期扫描，保留逐 rank 的结果；384 个完整 ledger 的所有字段与逐 rank 扫描相同。EP64/24 层的 CPU 对照中，单次显存预测从约 0.89–1.15 s 降至 0.022–0.175 s。

`fits_budget_with_reserve` 只是命名张量与显式预算检查。完整模型应通过 `external_gib` 加入 attention、router、shared experts、embedding 等已知占用，并为 allocator reserved/碎片、HCCL/driver 和未列出的 native workspace 配置 `reserve_gib`。不能把该字段当作真实运行绝不会 OOM 的保证。

## 细粒度 profiler

原有 8 类 cycle 累计器保留。新增 `HP_MEGA_MOE_TASK_PROFILE=1`，只在 `HP_MEGA_MOE_KERNEL_CYCLE_PROFILE=1` 时分配/写入任务诊断区。

每个方向最多 48 个 worker、每 worker 256 条记录，记录宽度 128 字节，额外 arena 为 1,579,008 字节。原 4 KiB 尾部的 marker 为 `0x4D545001`，解码器检查 ABI、header、记录长度和时间单调性。步长根据队列总任务数和 wave 数计算，覆盖整个队列，而非只记录前 256 个任务。

记录内容包含 task id/type/index、wave、tiling position、依赖和触发事件、专家行数，以及：

1. descriptor 解码前；
2. 依赖满足后、执行前；
3. 任务执行后；
4. ACK/事件发布后。

由此分离同 worker 的 pre-compute、compute、publication 区间。pre-compute 包含解码和依赖等待；通信任务的 compute 内仍可能包含协议等待。每个设备使用自己的时钟，不能直接比较不同 rank 的绝对时间戳。

计时使用 `GetSystemCycle`。文档规定其系统计数器为 50 MHz，转换为微秒应除以 50，不能使用 Cube 主频。该 API 本身不替代其他流水线的完成同步：[Ascend C API](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0282.html)。GMM 校准取同一专家所有 Cube worker 的最大 compute 区间，并要求完整、stride=1 的 Cube 队列；不把依赖等待纳入算力倍率。

另提供 CPU scope 和当前 stream NPU event 的额外诊断步骤，以及 Chrome trace 导入。scope 保留嵌套关系，trace 汇总按时间区间并集计算。所有诊断均在 profile-off 性能测量之后执行，不能使用诊断步骤的耗时宣称性能收益。诊断 arena 扩容后可能继续保留到资源释放，严谨显存 A/B 应使用新进程。

## 复现与使用

普通预测无需 NPU。先从 editable 安装的 checkout 执行：

```bash
python -m examples.torch.multicore.predict_mega_moe \
  --ranks 64 --experts 384 --tokens 4096 --hidden 5120 --intermediate 1792 \
  --slots 0,1,2,3 --capacity-factors 1.25,1.5 --layers 2 \
  --routing dirichlet --ranks-per-node 8 --inter-gbps 25 \
  --hardware output/cost_model/hardware-calibrated.json \
  --output output/new-cost-prediction.json
```

省略 `--hardware` 会使用明确标注的假设速率。可使用 `--counts` 导入 `{"counts": [[...], ...]}`，替代生成的路由分布。没有实际 counts 或概率分布假设，就无法从 EP/E 两个数字唯一确定耗时。

`--options` 示例：

```json
{
  "inplace_backward": true,
  "atomic_dw": true,
  "linear_copy_runs": true,
  "gradient_receive_slots": 8,
  "optimizer": "fp32-main-adamw",
  "device_gib": 64,
  "reserve_gib": 8,
  "external_gib": 0
}
```

`--training` 示例：

```json
{"layers": 16, "steps": 1, "checkpoint": true, "pool": "shared", "update_every": 1}
```

分析不更新权重的 warm 单层 microbenchmark 时，应设 `layers=1, steps=2, checkpoint=false, update_every=0, optimizer=none`，比较第二个 `step_ms`；第一个 step 可能包含首次权重预取。生产训练不能据此省略版本失效成本。

构图审计可在没有空闲 NPU 时运行；该命令会导入 Torch adapter，但只创建 meta tensors：

```bash
python -m examples.torch.multicore.audit_mega_moe_plan \
  --ranks 16,32,64 --slots 0,1,2,3 \
  --options output/cost_model/sweep/options.json \
  --output output/new-plan-audit.json
```

将审计 JSON 传给预测器的 `--schedule-audit`，要求 shape、rank、B 和相关开关逐项匹配。输出分别提供通过显存筛选的排序和通过真实构图审计的排序，避免把“有成本估计”误认为“当前 kernel 可启动”。

真实校准需要先加载匹配的 CANN 与本 checkout 新构建的 native payload，再通过资源队列运行：

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  -m examples.torch.multicore.profile_mega_moe_costs \
  --output output/fresh-calibration --slots 1 --profile-scopes
```

worker 在 profile-off 测量之前检查 native 输出、输入/路由权重/专家梯度，并在额外 checkpoint 步骤中校验任务 profiler。校准 JSON 必须显式区分 `split=train` 与 `split=holdout`；`calibrate()` 会拒绝混入 holdout 的数据。独立梯度传输可通过 `calibrate_transfer_windows()` 拟合；CLI 的校准 JSON 可同时携带 `transfer_samples` 和 `transfer_identity`。

## 本轮验证与优化边界

实测环境为 8 张 Ascend 910B3、CANN 9.1.0、Torch/torch_npu 2.9.1，匹配本 worktree 构建的 native payload。生产宽度校准形状为 EP8/E48、T4096、H5120、I1792、K8，覆盖 B0/B1 的五种 Dirichlet 路由。三种路由用于拟合，另外两种用于检验；源码身份、原始每 rank 数据和脚本保存在 `output/cost_model/`。

当前预留集端到端误差：B0 约 +8.4% / -2.0%，B1 约 +11.2% / -11.8%。均衡训练点约高估 23%，B1 强热点仍有明显低估。这些结果表明曲线改善了 GMM 建模，但尚未覆盖细粒度队列和 HCCL 的全部时间，不能宣称统一 ±10% 精度。命名 HBM 相对“allocator 峰值 + SHMEM − 常驻 native 参考参数”的预留集误差约在 ±2.5% 内；测量口径仍包含 harness 临时量，不是完整模型显存精度承诺。

独立的 EP8 FP32 梯度归并 microbenchmark 在排除 kernel 到达偏斜后测得约 8–10 ms。用其中两种训练路由的“实测 − 网络与 owner 相加模型”校准出约 5.01 ms/窗口的有效批次开销，另外两种路由保留作验证。该项与 GEMM 校准分离，是给定 peer/形状下的协议和启动成本近似；多机及不同 batch 宽度仍属外推。

模型指导的首个生产改动是 `_physical_copy_runs`：把每 source 的 `E × Ephysical` 重复扫描变为按专家分桶的一次扫描，再按原来的逻辑专家/物理槽位顺序展开，复杂度降为 `O(E + Ephysical)`。原始顺序、分配份额及设备映射保持一致。

18 个 EP16/32/64 配置的所有 672 个 rank 配额逐项相同。新进程、profile-off 的 8 NPU A/B/B/A 验证中，EP8/E192、T512、H512、I128、B1 的强热点单层 F+B 从 9.761 ms 降到 8.127 ms，降低 16.7%，两组方向一致；全程通过 native 精度检查，峰值 HBM 相同。无副本路由为 6.027 → 6.043 ms，方向不稳定，视为无可确认变化。该收益不能外推成 EP64 实测收益。

首轮 E384/T4096 的 EP16/32/64 构图审计发现，原来的 25600 个 task、1024 个 event 固定数组无法容纳这些图。后续 [runtime descriptor v2](runtime_descriptor_v2.md) 已改成按图分配容量，并加入 128 字节 task + 共享 tensor 模板格式。EP16/32/64、B0–B3 的默认与优化配置现已通过 meta 构图审计。实际多机内核仍需独立验证，不能把构图成功或 EP64 模型结果标成多机实测。

当前计划审计还计入 B 配置的 home fallback 计划，以及 `torch.cat(base_config, pointer_metadata)` 缓存的三种扩展 runtime 暖态上界。预测器使用这个上界补齐常驻 HBM；事件区与生产代码共用容量公式。压缩后的 descriptor 服务耗时通过 `--descriptor-profiles` 单独校准，不能按字节缩减比例直接宣称加速。拟合使用全量 seen 计数而非采样条数，并隔离格式、native payload、worker 数和留出路由。`Hardware.vector_workers` 已替代固定的 24；本机 910B3 校准值为 20，meta 审计也应传入 `--cube-cores 20`。

后续迭代应分别处理：按 tile/专家事件重放队列、独立校准 HCCL 权重/梯度传输、缓存 tensor 模板解码并保留 ACK/generation 语义、再评估长 K 的 dW tiling。每个候选先看模型的关键路径与显存峰值，再用独立精度及 profile-off A/B/B/A 验证；不能通过删除必要等待来“降低模型成本”。
