# 当前 MegaMoE 运行时的共享专家融合实验

## 实验范围

实现基于 Hyper `2632ba92`（`megamoe-push-pull`）和 PanGu `f91f5f874`（`feat/megamoe`）。
旧 Hyper 工作树 `8f14cc16` 仅用于审视 ABI 和调度经验，没有直接 cherry-pick。

验收负载采用 **505B 系列、序列长度 4096、MBS2、TP1/EP16**：每个 rank 有 8192 个本地 token，
隐藏维度 5120，路由专家和共享专家的中间维度均为 1792，384 个路由专家，TopK8，BF16，
关闭梯度累加融合。这与 PanGu 的 `validate_megamoe_perf.sh` 和 M-D-M runner 配置一致。

计时覆盖一个完整 MoE 层的前向和反向，包括路由准备、排列、共享专家分支、输出相加，以及输入、
参数和路由权重的梯度计算；不包含 router 投影、Attention、优化器和 DP 梯度归约。
**本报告中的收益属于单层 MoE 前反向，不能作为整模型训练加速比。**

## 实现方案

`MegaMoeExperts(shared_intermediate_size=...)` 接收调用方持有的稠密共享专家权重：
`shared_gate_up_weight` 形状为 `[2I, H]`，`shared_down_weight` 形状为 `[H, I]`。
两者必须连续，并与 BF16 输入的设备和数据类型一致。共享维度为零时保持原有接口行为，
不会重复注册或初始化共享专家权重。

共享专家的 up、SwiGLU、down 及其反向矩阵计算进入已有前向、反向 megakernel。
共享张量使用普通本地 NPU 存储，不新增共享专家通信。路由输出与共享输出的相加、两路 dX 的相加
仍由普通 Torch 算子完成。各 microbatch 的共享激活通过 saved tensors 保留；
workspace lease 和当前 stream 继续保护通信存储的生命周期。

两个 native op 都在 profiler 输入之后追加指针表和打包的共享运行时缓冲区，
原有 tiling、event、profile 槽位不变。指针表以调度标志开头，后接张量地址；共享 TensorDesc
使用 `tensor_type=2`，`dynamic_dim` 表示地址索引，`base_ptr_offset` 表示元素偏移。
加载器要求共享专家 ABI 版本为 2，必须同时重新构建并激活 vendor 和 Torch adapter。

运行时在路由任务的 ready/completion 事件之外增加独立的共享事件。默认调度位置为 `idle`：

- 前向：共享 up/down 的 Cube 轮次放在路由 Cube 计算之后，与路由 Combine 尾部重叠。
- 反向：共享 dActivation 放在 Cube 入口，利用初始 Dispatch 等待；
  共享 dW2/dX/dW1 放在路由 W2Grad/ActGrad 阶段边界。
- 共享 SwiGLU 及其梯度使用此前空闲的偶数 AIV worker。奇数 AIV 继续执行路由队列，
  其循环跳过共享向量任务后缀，但仍由原来的奇数 worker 执行终止任务。

内部实验 runner 还保留 `tail`、`overlap`、`dispatch` 调度位置。
`tail`、`overlap`、`idle` 已做真机检查；`dispatch` 只有队列和事件模拟覆盖，没有性能结论。

PanGu 通过 `--mega-moe-fuse-shared-expert` 或
`accelerate_config.moe.fuse_shared_expert: true` 开启，默认关闭。
要求 `--moe-backend=mega_moe`、TP1、已配置共享专家、`--no-gradient-accumulation-fusion`，
且不使用仅 MLP 重计算。已有 BF16、SwiGLU、bias、dropout、量化约束继续生效。
参数归属、checkpoint 名称和默认 RNG 行为仍由 PanGu 管理。

## Profiler 分析与调度选择

原先每核最多 256 条记录，导致目标负载 trace 丢失 332550 条记录。现将主机和设备端上限提高到
16384，实际分配仍由队列需求决定，并向上对齐到 16 条。实验 runner 会拒绝截断的 trace。
最终 `idle` 目标负载在 rank 0 和 rank 15 各采集 355182 条记录，丢失数均为零。

下表来自 rank 0 的诊断 trace，采集时开启 profiler。阶段时间跨度包含跨 worker 的间隙，
不能当作关闭 profiler 后的耗时，也不能直接相加为共享计算量。时间原点为本次前向 kernel 的起点。

| 前向调度方式 | 共享 up 完成 | 共享 SwiGLU 起止 | 共享 down 完成 | shared down 前 Cube 平均等待 |
| --- | ---: | ---: | ---: | ---: |
| 路由 AIV 队尾（`tail`） | 13.855 ms | 14.913–17.864 ms | 18.343 ms | 4.05 ms |
| 路由 AIV 的 Combine 后段插入（`overlap`） | 13.944 ms | 13.950–16.849 ms | 17.339 ms | 见原始 trace |
| 专用偶数 AIV（`idle`） | 14.084 ms | 14.196–14.335 ms | 14.825 ms | 0.28 ms |

采用 `idle` 后，反向共享 SwiGLUGrad 从路由 AIV 插入方案的 5.334–7.458 ms 提前到
0.907–1.178 ms，且不再占用路由向量队列。共享 Cube 计算仍会消耗 Cube 时间，
计算图分支独立并不意味着计算能全部被掩盖。偶数 AIV 入口的长等待来自等待共享 Cube 生产者。

最早的 `tail` 实验还使用了同步指针表上传；后续改为 pinned 主机表和当前 stream 上的非阻塞复制。
因此，早期 `tail` 到 `overlap` 的整体耗时差不能全部归因于插入位置。
更早的一轮探索性 ABBA 与主机编译并发，未计入最终性能验收。

## 验证与复现方法

本地环境为 Ascend910（`ascend910_93`），16 个逻辑 NPU，每个 64 GiB；
CANN 9.1.0、Torch 2.9.1、torch_npu 2.9.1。
正式计时通过 `npu_wait_and_run.sh` 的设备锁启动，要求两次空闲采样和最后 30 秒空闲窗口。
每轮保存命令、源码补丁、native 二进制哈希、运行时版本、NPU 健康状态、进程归属、
七组输出/梯度签名，以及 HBM 峰值。

仓库入口为 `examples/torch/multicore/shared_expert_profile.py`。
完成 editable 安装后，先构建并激活 native 组件，再导入 Torch：

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910_93
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export LD_LIBRARY_PATH="$PWD/build/native/payload/hyper_parallel/core/multicore/shmem/lib:$PWD/build/native/payload/hyper_parallel/core/multicore/shmem/lib/shmem:$LD_LIBRARY_PATH"
export HYPER_PARALLEL_PLATFORM=torch
export SHMEM_IP_PORT=tcp://127.0.0.1:62828
# Launch through the site's NPU idle gate. Omit --fused for the paired baseline.
python -m torch.distributed.run --nproc-per-node=16 --master-port=62818 \
  examples/torch/multicore/shared_expert_profile.py \
  --output /path/to/result --fused --warmup 5 --steps 50
```

数值和内部 trace 单独使用 `--check --profile` 采集。
`--route dirichlet` 使用固定的全局 Dirichlet(alpha=10) 专家负载，各 rank 独立采样本地 token。
正式性能使用关闭 profiler 的 ABBA：每轮独立重启进程，预热 5 步、计时 50 步，
固定路由、权重和梯度输入，每步取所有 rank 中的最大耗时。p95 使用线性插值。
正式计时不加 `--check`，因为参考实现的额外分配会改变内存池。

Hyper multicore UT 通过 127 项及 634 个 subtest；PanGu adapter UT 通过 31 项；
MoE 参数和配置相关 UT 通过 73 项。队列模拟检查任务覆盖和前进性，包括独立的偶数 AIV 后缀。
小规模及目标负载的 `idle` 真机对比中，所有 rank 的输出、dX、router 梯度、
两组路由权重梯度和两组共享权重梯度的最大绝对误差均为零。

PanGu worker `tests/group3/integrated_tests/model/_test_megamoe_shared_expert.py`
验证严格 checkpoint 加载、三次 SGD 更新、每步两个尚未反传的前向、逆序反传，
以及 DP 归约之前的真实 FP32 DDP 梯度缓冲区。该验证不覆盖完整 trainer 或分布式优化器。
在添加参数保护之前，另用 PanGu `fcd023334` 测试了直接 FP32 梯度累加；
其限制及失败证据见下文。

本机原始结果、trace、构建日志、脚本和阶段时间图统一位于：
`/home/ma-user/zfr/shared-profiler-results-20260917/`。

## 最终性能：关闭 profiler 的 ABBA

每种模式合并两次独立重启的结果，共 100 个“每步 rank 最大耗时”样本。
计时均包含共享专家的完整前向和反向。表中收益统一采用**耗时降幅**。

| 路由分布 | 独立共享专家中位数 | 融合后中位数 | 耗时降幅 | 独立共享专家 p95 | 融合后 p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 均衡 | 61.707 ms | 60.168 ms | 2.49% | 63.229 ms | 61.676 ms |
| Dirichlet(alpha=10) | 64.707 ms | 62.865 ms | 2.85% | 66.611 ms | 63.760 ms |

两组配对的中位数降幅分别为：均衡路由 2.79% / 2.25%，Dirichlet 路由 3.02% / 2.44%。
八轮均使用相同源码补丁、runner、运行时及 15 个 native 二进制哈希。
同一路由负载下，16 个 rank 的七组签名均完全一致。
各轮开始和结束时均有 16 个健康 NPU，外部 worker 数均为零。

| 路由分布 | allocated 峰值，独立/融合 | reserved 峰值，独立/融合 |
| --- | ---: | ---: |
| 均衡 | 5.7770 / 5.7857 GiB | 6.9668 / 6.7383 GiB |
| Dirichlet(alpha=10) | 5.9826 / 5.9913 GiB | 7.1504 / 6.9453 GiB |

以上为 Torch NPU allocator 峰值，不含外部 SHMEM 分配的完整设备占用。
两种负载的 allocated 均增加 8.93 MiB；reserved 反映分配器保留内存，不能视为存活张量大小。
原始设备 HBM 快照与分配器统计一并保留。

最终均衡负载的 PanGu 集成验证中，16 个 rank 在三次更新、每步两个尚未反传的前向条件下，
输出、dX、router 梯度及 FP32 main-gradient 均完全一致。

## 共享计算全部被掩盖时的理论收益上限

### 数据来源与统计口径

MegaMoE 内部 cycle profiler 只记录 megakernel 内的任务，无法直接测量基线中独立执行的
`F.linear → SwiGLU → F.linear` 及其反向。因此另补采了两种路由的 **16 卡独立共享专家基线**，
使用 CANN profiler Level1，记录 shape 和调用栈，保持原负载、权重、路由和运行时。
每次运行先做 5 步预热和 3 步普通执行，再做 1 步 profiler 预热、5 步正式采集；
正式 ABBA 的中位数和 p95 不被这些补采数据替换。

补采源码为 Hyper `a3e3eca6`，与正式实验的 15 个 multicore native 二进制哈希一致，
Torch、torch_npu 和 Python 版本一致；所有 rank 的七组签名与对应 A1 基线完全一致。
两次补采的起止健康检查均为 16 个健康 NPU、零外部 worker。

从 `kernel_details.csv` 的 `Step Id`、`Device_id`、`Stream ID`、`Type`、
`Input Shapes`、`Start Time(us)` 和 `Duration(us)` 提取数据，
结合基线源码和 `operator_details.csv` 的调用栈确认共享分支归属。
每个 rank、每个采集步都严格匹配以下 8 个计算 kernel：前向 2 次矩阵乘和 1 次 SwiGLU，
反向 4 次矩阵乘和 1 次 SwiGLUGrad。共核对 2 种路由 × 16 个 rank × 5 步 = 160 组样本。
这 8 个 kernel 在同一 stream 上串行且计算区间不重叠，所以其 Duration 之和等于计算区间并集长度。
没有累计并行 core 的时间，也没有使用首尾跨度、Wait Time 或 host 耗时。

本节主口径只掩盖矩阵乘和激活计算，保留共享权重梯度的两次物理转置、输出相加、dX 相加、
路由和其他调度开销。若连转置也全部掩盖，另在下文给出扩展口径。

令 `S[r,j]` 为 rank `r`、采集步 `j` 的上述 8 个计算 kernel 耗时之和，取：

```text
S = median_j(max_r(S[r,j]))
```

即每个采集步先取 16 个 rank 的最大共享计算时间，再对 5 步取中位数。
使用 rank 最大值是乐观估计：共享计算最慢的 rank 不一定是完整 MoE 层最慢的 rank。
固定逐 rank 耗时后，移除这些计算使 rank 最大耗时减少的量不会超过 `max_r(S[r,j])`。

下表展开决定中位数 S 的实际样本，单位为 ms；均衡为 rank 11 / step 1，
Dirichlet 为 rank 5 / step 5。选择规则是先按上式得到 5 个最大值，再取中间的样本，
没有按收益挑选步骤。

| 共享计算阶段 | 均衡 | Dirichlet(alpha=10) |
| --- | ---: | ---: |
| 前向 gate/up 矩阵乘 | 0.94868 | 0.94028 |
| 前向 SwiGLU | 0.05418 | 0.05586 |
| 前向 down 矩阵乘 | 0.45220 | 0.45672 |
| 反向 dW2 矩阵乘 | 0.60914 | 0.62340 |
| 反向 dActivation 矩阵乘 | 0.47010 | 0.46780 |
| 反向 SwiGLUGrad | 0.11756 | 0.11998 |
| 反向 dW1 矩阵乘 | 1.02442 | 1.02810 |
| 反向 dX 矩阵乘 | 0.93354 | 0.92872 |
| **前向小计** | **1.45506** | **1.45286** |
| **反向小计** | **3.15476** | **3.16800** |
| **共享计算合计 S** | **4.60982** | **4.62086** |

5 个采集步的 rank 最大共享计算时间范围分别为 4.60690–4.61310 ms 和
4.61318–4.62672 ms。该范围用于展示本次采样波动，不是统计置信区间。

### 理论上限与当前达成率

假设以上共享计算可以全部放入其他工作已有的等待或空闲区间，既不拖慢路由任务，
也不增加同步开销，并保持其余工作耗时不变。用 `T_A`、`T_B` 表示前文正式、
关闭 profiler 的 ABBA 基线与融合中位数，则：

```text
理想耗时       T_ideal = T_A - S
理论耗时降幅   R_max   = S / T_A
当前耗时降幅   R_now   = (T_A - T_B) / T_A
上限达成率     A       = R_now / R_max = (T_A - T_B) / S
```

| 路由分布 | 可掩盖计算 S | 理想耗时 T_ideal | 理论耗时降幅 | 当前节省时间 / 耗时降幅 | 上限达成率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 均衡 | 4.60982 ms | 57.097 ms | **7.47%** | 1.53865 ms / 2.49% | **33.38%** |
| Dirichlet(alpha=10) | 4.62086 ms | 60.086 ms | **7.14%** | 1.84110 ms / 2.85% | **39.84%** |

计算使用原始精度；例如均衡达成率为 `1.5386482701 / 4.60982 = 33.3776%`，
不是用已经舍入的 `2.49 / 7.47` 计算。若以吞吐加速口径 `T_A / T_ideal - 1` 表达，
理论值分别为 8.07% 和 7.69%；本报告的主表统一使用耗时降幅，避免混用两种百分比。

若进一步假设共享 dW1/dW2 的两次物理转置也能全部被掩盖，按同一聚合规则得到：

| 扩展口径 | 可掩盖计算与转置时间 | 理论耗时降幅 | 当前达成率 |
| --- | ---: | ---: | ---: |
| 均衡 | 4.74294 ms | 7.69% | 32.44% |
| Dirichlet(alpha=10) | 4.75500 ms | 7.35% | 38.72% |

**这些数值是 profiler 支撑的理想重叠模型估计，不是已实测的最优耗时，也不是实测重叠率。**
模型将开启 profiler 时的设备计算时间用于估计关闭 profiler 时的可移除工作量，
还假设有足够的空闲计算资源；它不是带统计保证的绝对上界。
当前 33.38% / 39.84% 衡量整体净收益相对于该预算的达成程度，
包含融合引入或减少的其他开销影响，不能据此认定恰有相同比例的共享指令被掩盖。

内部 trace 已表明前向共享计算能利用 Combine 尾部、反向共享 dActivation 能利用 Dispatch 等待，
但共享 dW2/dX/dW1 仍占用 Cube，可能推迟路由计算。达到上述理想上限需要把这部分计算也完全掩盖，
且不把收益转移为路由任务或同步的额外等待。继续调整插入位置的实际收益仍需关闭 profiler 的 ABBA 验证。

补采与计算产物位于结果根目录的 `shared-bound/`，其中 `upper_bound.json` 保存全部 160 组样本、
原始 CSV 路径和行号、CSV SHA256、逐阶段耗时、聚合输入、上限及达成率。
`facts-<route>-rank<N>/` 是 `pangu-perf` 技能生成的原始表格事实包；本节的归属规则和理想模型
由独立分析脚本 `analyze_shared_bound.py` 定义，不属于事实提取技能的结论。
使用结果根目录下的脚本可重新计算：

```bash
python3 /home/ma-user/zfr/shared-profiler-results-20260917/analyze_shared_bound.py
```

## 尚不支持：共享权重梯度直接 FP32 累加

探索性的 FP32 累加检查最初因独立 PanGu 基线无法加载 `libatb.so` 而中止。
激活已安装的 ATB 8.5.2 C++ ABI 1 库并配合 CANN 9.1 后，补充数值实验得以执行。
三次更新的逐元素比较失败：最大绝对误差为 0.0625，最大相对 L2 为 0.003849。
第一步就出现了共享 dW 的舍入差异：独立基线直接累加 FP32 矩阵乘结果，
融合 kernel 则返回 BF16 权重梯度，再由 DDP 加到 FP32 缓冲区。没有放宽容差。

参数校验和 PanGu adapter 现已拒绝共享融合与 `gradient_accumulation_fusion=True` 的组合。
本次验收负载仍有效：现有 505B/M-D-M runner 显式传入 `--no-gradient-accumulation-fusion`。
若要支持共享梯度直接 FP32 累加，需要新增 native 输出或累加路径，超出本次实测配置。
本报告的性能结果均未使用上述补充 ATB 环境。

原始批处理仅在这项额外检查处非零退出，此前八轮 ABBA 和验收配置的 PanGu 集成测试均已完成。
`validate-final.py` 逐轮独立核查，输出 `final/acceptance.json` 和
`final/ACCEPTED_SCOPE_PASSED`；不支持组合的数值失败仍在报告中明确保留。

## 实验产物索引

以下路径均相对于结果根目录：

- `final/balanced/abba_summary.json`、`final/dirichlet/abba_summary.json`：正式 ABBA 汇总。
- `final/<route>-<A1|B1|B2|A2>/`：命令、补丁、二进制哈希、各 rank 样本和签名、起止 NPU 健康状态。
- `final/pull-small/`、`final/dirichlet-check/`：数值比较。
- `final/pangu-integration/`：验收配置的训练状态比较；
  `final/pangu-grad-accumulation/`：保留的直接 FP32 累加失败结果。
- `final-validation-v2.sh`、`run_env.sh`、`validate-final.py`：实际编排和审计脚本。
- `atb-validation-environment.json`、`complete-validation-atb.sh`、`run_atb.sh`：额外检查的环境与命令。
- `placement-spans-final.png` / `.svg` 及各实验的完整内部 cycle trace：调度分析依据。
- `shared-bound/<route>/rank<N>_cann/`：独立共享分支的 CANN profiler 原始产物及解析 CSV。
- `shared-bound/facts-<route>-rank<N>/`、`shared-bound/upper_bound.json`：逐 rank 事实包及上限计算明细。
- `shared_bound_profile.py`、`run-shared-bound.sh`、`analyze_shared_bound.py`：补采入口、命令及计算脚本。

融合开关保持默认关闭。Pull 仅完成小规模真机正确性检查；其生产形状性能、其他 TP 规模、
其他硬件或工具链，以及整模型吞吐，均未在本实验中测量。
