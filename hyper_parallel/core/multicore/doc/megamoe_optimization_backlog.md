# MegaMoe 当前待办与验收清单

更新日期：2026-09-09。合并基线：`ee5a696554133d99c80aa9029e5cdd9ab039c1ac`；
本轮已 rebase 到远端 `ba9a83a2943ffdf8f27691f3e3a07738752e7971`，接收 `caf71330`、`ba9a83a2`
两条 Qwen 诊断文档提交，保留本地八条独立提交。运行代码和测试内容与 rebase 前一致；
本轮只核对文档与 Git 差异，没有重跑设备验收。各历史结果仍使用当时的源码/native/环境身份。
本文件统一记录合并 cost-model 系列与 `megamoe-sun-260908` 后的待办；历史实验见 history/，当前优先级仅在本文件维护。
以下状态随实施更新，未勾选项仍待完成；每轮保留提交和证据入口。

目标是让一次匹配源码、硬件与训练契约的校准支撑后续离线预测，预测耗时与峰值 HBM，
再据此筛选 EP16/32/64、E384 的热专家拷贝和整体 MegaMoe 优化。
普通预测应保持无需 Torch/NPU；构图审计、真实数值验证和硬件校准分别记录。

## 已完成基线

这些能力已经存在，不再作为“从头实现”的待办：

| 项目 | 当前状态与证据边界 |
| --- | --- |
| 单机虚拟多节点 | CPU 模拟 16/32/64 rank，复用真实 planner，并模拟节点带宽竞争、传播延迟；没有执行多机 kernel 或 RDMA |
| 热拷贝规划 | 候选剪枝、线性 physical copy runs 已实现；高频 cache miss、D2H 与控制面仍需进一步优化 |
| 梯度接收 | 默认 8 槽的 peer-balanced 有界窗口与 receive-only control-stream 重叠已结合；消费、复用依赖及 FP32 累加保留 |
| Runtime descriptor | 按图分配 task/event 容量、128 字节 compact task、tensor templates 已实现；EP16/32/64 meta 构图通过不等于设备执行通过 |
| Cost-model | 六类 GEMM 服务、rank 阶段依赖、网络竞争、storage 生命周期、train/holdout 校准和构图筛选已实现；仍是近似模型 |
| Profiler | 两组采集器均保留，T01 已统一关联与多窗口归因；重校准仍在 T03/T05 待办 |
| Qwen 集成 | shared-B、单 backend 驻留、CPU 快照及模型分层已完成；生产形状舍入与 AdamW/层间放大已有因果证据，全局梯度契约及长期训练仍待验收 |
| 工程收束 | 六个主题提交与 260908 九个提交均保留，通过双亲 merge 收束；本轮临时 worktree 及数据已归档，阶段快照不重复合入 |

`ee5a6965` 合并验收：299 个 UT、302 个子测试；EP8 HCCL 梯度窗口测试；
EP8/E384、H512/I128、B1/B2 各 21 组精度及生命周期检查；5 项 connected/Qwen 集成测试通过。
Qwen 回归形状为两层 H1024/I256/T1024，不能替代生产宽度、多步或收敛验收。
三种 EP64/E384 路由、各 B=0/1/2/3，共 12 个预测与归档一致，除当前源码指纹更新。
合并后未重跑性能 ABBA，也未重新标定 receive-only 重叠；不能直接相加两侧历史收益。

## 执行顺序

T01 已完成。T02 下一项是全局目标的梯度归一化与 norm 参考；已有舍入、AdamW 与参数干预诊断
不再从头重做。T03 补齐 Qwen 的共用逐步审计入口，等待可审计的设备窗口建立配对基线。
独立 MoE 按自身已通过的精度契约推进，connected Qwen 的训练/性能结论仍逐配置检查 T02 的门槛。
等待设备时可推进 T04/T05 的 CPU 账本与依赖模型；硬件系数重校准仍依赖 T03。
随后 T06 用模型筛选参数，再依据瓶颈选择 T07/T08 的最小实现。
T09 依赖实际通信条件，不阻塞 CPU 模型工作。

| 优先级 | ID | 待办 | 当前状态 | 主要依赖 |
| --- | --- | --- | --- | --- |
| P0 | T01 | 统一 profiler 语义并支持多窗口归因 | 已实现并完成工具验收 | 两组原有 profiler 均保留 |
| P0 | T02 | Qwen 数值放大与全局训练契约 | 舍入/放大已定位；待归一化、norm 与更长训练验收 | 原合同与显式控制组分别保留 |
| P0 | T03 | 合并后的性能、HBM 和计时窗口基线 | 逐步审计已验收；配对基线受设备占用阻塞 | T01；Qwen 部分依赖 T02 |
| P1 | T04 | 补齐并校验峰值 HBM 生命周期 | 已有账本待核对扩展 | T01/T03 的内存证据 |
| P1 | T05 | 增强关键路径模型并重新校准 | rank 阶段模型待细化 | T01/T03；HBM 结合 T04 |
| P1 | T06 | EP16/32/64、E384 联合参数扫描 | 可重放，待更新排序和边界 | T04/T05 |
| P1 | T07 | 优化 planner 与控制面 | 已有优化，下一项待证据选择 | T01/T05/T06 |
| P1 | T08 | 优化梯度回传和 kernel 调度 | receive-only 已有，下一项待证据选择 | T01/T03/T04/T05 |
| P2 | T09 | 实际多 rank / 多机通信验收 | 条件待办 | 硬件、HCCL/SHMEM 支持与构图审计 |

## P0：先补可解释的测量与数值依据

### T01 — Profiler 整合与多窗口归因

- [x] 为 host scope、kernel task、P2P batch 和内存事件建立可关联的 step/layer/rank/wave/专家及
  placement generation 标识；扩展现有入口，不再建立一套重复采集器。
- [x] 扩展时间线分析器，解除 EP8、16 层、每层单个非空梯度 batch 的硬编码。
  覆盖空 batch、多接收窗口、receive-only、mixed send/receive 和多个 resource group。
- [x] 分别报告 control-stream 等待、compute-stream join、owner 累加及窗口复用依赖；
  将已有六类 GEMM/task 标记接入 dX、dW、recompute 汇总，关联具体 expert 的 readiness。
- [x] 为 cost-model 提供统一导入结果，明确时间单位、时钟域、采样 stride、覆盖范围和源码/native 身份。
  task 的本地 system cycle 不直接作为跨 rank 时钟；不同诊断步骤不假装是同一次执行。

验收：旧 trace 的已有字段与历史分析一致；多窗口的 peer/shape/dtype/字节数与真实 transfer 描述匹配；
缺失事件、错配身份或不可比时钟应明确拒绝。额外诊断不混入 profile-off 性能/HBM 测量，
采样不完整时不宣称已取得全量任务轨迹。GEMM 标定继续要求完整、stride=1 的 Cube 队列。

入口：[layer profiler](../../../../examples/torch/multicore/mega_moe_layer_profile.py)、
[P2P profiler](../../../../examples/torch/multicore/mega_moe_p2p_profile.py)、
[时间线分析器](../../../../examples/torch/multicore/analyze_mega_moe_gradient_timeline.py)、
[task ABI](../modules/mega_moe/task_profile.py)、
[统一导入](../../../../examples/torch/multicore/megamoe_simulator/profile_import.py)。
使用方式与时钟/覆盖边界见 [成本模型文档](theoretical_cost_model.md#关联采集与多窗口归因)。

验收记录（2026-09-08）：multicore 示例目录 139 项 UT、164 个子测试通过；覆盖 EP16/32/64、多窗口、空 batch、
mixed send/receive、多个资源组和坏数据拒绝。旧 EP8×16 层 fixture 与 `7dfd4ee0` reader 的全部输出字段一致；
旧真实 trace 原件未在当前工作区找到，这项是格式回归测试，不是重新采集历史性能。
EP8/E48、T512/H1024/I1024、3 层、2 个资源组、接收窗口 1 的真机采集通过：每层 5 个全局梯度窗口，
包含 receive-only 和本地空窗口；30 个非空窗口与 90 个空窗口均完成归因与 payload 核对。
32 份最后一次 launch 的 task 记录通过完整 Cube/stride=1 检查，共提取 144 条专家 GEMM 服务记录。
9,978 条 allocator 事件中 7,284 条关联到 scope，其余显式保留为未归因。
另一个 EP8/E48、T128/H512/I128、B1 的 cost worker 在均匀与强热点两条路由上
通过 native 与 checkpoint 精度检查和统一导入。两条路由均先完成 profile-off 测量，再采集诊断，
防止已扩容的 task arena 污染后续 HBM；相应回归测试先复现污染再验证修复。
最终两轮的进程占用检查通过，均是小尺寸工具验收，没有重做生产性能 ABBA 或重新拟合模型。

原始结果、trace、脚本及检查日志在
`/home/feiran/doc/megamoe-profiler-integration-20260908/`，有效设备结果分别为 `smoke-final-2/` 与
`cost-smoke-2/`。早期 CANN 导出权限失败、缺少关键 flow 的小尺寸 trace，以及外部作业污染的运行均保留，
不能拿来替代有效验收或做性能标定。CANN 工具只在该目录复制并收紧权限，共享安装未修改。

### T02 — Qwen 舍入放大与全局训练契约

- [x] 在 `ee5a6965` 生产 Python 与重建的 native 上复现原八层 H5120/I1792/T4096、E48/K8、EP8、
  shared-B1、Dirichlet 0.1、LR=1e-3 配置。默认模式的 attention backward 和 BF16 dense
  collective 均有独立非确定性证据；确定性模式恢复同 backend 自重复，但跨 backend 仍有数值差异。
- [x] 固定真实第 8 层权重、hidden、路由及 dY，比较 native、B0 单波、B0 两波和 B1 单波。
  B0 单波的输出/dX/dW 逐位一致；多波与复制仅在被切分/复制 expert 的 dW 有差异，最大为
  4.7683716e-7，均通过原阈值。实际 FP32 owner 聚合和最终 BF16 cast 与 CPU 重算逐位一致。
  有界 FP64 点积诊断支持舍入路径解释，未发现本组 replica 聚合错误，不外推为所有形状已验证。
- [x] 使用原 fused AdamW、真实捕获梯度做三次固定梯度更新。同梯度的两种参数布局，其
  main/moments/model 全部逐位一致；真实残差经过小梯度敏感区及 BF16 写回边界产生参数差异。
  八层首步共有 11227 个 expert BF16 参数元素不同；恢复/注入这些实际差异能消除/复现第二次
  forward 分歧，native 反向注入也复现 MegaMoe 的全部层输出。同权重下未发现额外 forward 差异。
- [x] 单列四档 eps 和 LR=1e-4 控制。增大 eps 未使完整训练通过；低 LR 确定性组的十步 loss
  持续下降且跨 backend 全部通过，第 1/2/10 步全量梯度/更新后参数通过，但 logits 第 2–10 步
  仍超原阈值。默认 LR/eps、replica FP32 partial、FP32 main/moments 和原精度门槛均保留。
- [ ] **下一项：核验全局梯度归一化。** 用可容纳的全局 batch、未分片小模型作为独立参考，
  对齐有效 token 加权的 loss、本地平均 CE、dense SUM/EP 和 home expert 梯度的缩放。
  明确每项梯度对应的全局目标；两个 EP backend 相互通过不足以关闭此项，现有证据未证明有缩放 bug。
- [ ] 归一化明确后核对全局 norm：dense 副本只计一次，各 rank 的 home expert 分片合并一次，
  physical replica 不重复计数。先以 `max_norm=inf` 核验两 backend 与全局参考，再单列有限阈值
  裁剪；plain packed expert 不直接套用依赖 DTensor placements 的裁剪 helper。
- [ ] 将 FP32 dense 归约作为独立控制，核验 main-grad 生命周期、完整训练状态和有界 bucket，
  同时记录通信字节与峰值 HBM。WD/no-decay、warmup/调度逐项对齐，不一次替换整套 trainer recipe。
- [ ] 在匹配实现/产物下扩展 balanced、Dirichlet、learned TopK、真实 shared-B 与多步覆盖。
  默认/确定性模式、原 LR/低 LR 分开；保存逐 rank loss/logits、输入与梯度、更新后参数及
  FP32 main/moments 的跨 backend 数值对照，并增加变化 batch/真实数据与更长轨迹。
  低 LR 的第 3–9 步全量梯度/参数及末步跨 backend main/moments oracle 尚未覆盖。

结论是本组 expert 差异符合切分造成的舍入，经 AdamW/BF16 写回和多层传播放大；没有证据支持
为此重写聚合或 optimizer。原 LR 确定性组第二步 logits、第六步 loss 开始有 rank 超阈值；
低 LR 的 loss 通过也没有关闭 logits gate。层间干预验证到第二次 forward，不代表解释完后续多次更新。

验收：分别报告算子同输入精度、梯度全局目标、训练状态、loss/logits 与长期训练边界。
原 `rtol=0.02 / atol=0.002` 保持；数值归因完成不能改记为完整训练通过，十步固定 batch
下降也不能代表长期收敛。对应合同的精度 gate 失败时，不发布该合同已验收的训练性能结论。

证据：[六轮数值诊断汇总及完整原文](history/qwen_validation.md#数值诊断汇总)、
[全局归一化与 norm 的后续依据](history/qwen_validation.md#低-lr-多步对照)。
入口：[Qwen benchmark](../../../../examples/torch/multicore/mega_moe/qwen_moe_benchmark.py)、
[CPU snapshots](../../../../examples/torch/multicore/mega_moe/qwen_moe_snapshots.py)。

### T03 — 合并后匹配的性能与 HBM 基线

- [x] 给每个 measured step 添加可审计的绝对起止时间，与全部设备 PID census 对齐。
  独立 MoE worker 的各 rank 使用同一组整数 monotonic 时间戳生成耗时与审计窗口，
  保留 PID/start ticks、host/boot/PID namespace 和 realtime anchor。
  审计报告前后采样、最大采样间隔及干扰；采样缺口和未知进程均不能当作干净测量。
- [ ] 为 isolated Qwen 接入上述共用逐步审计合同。远端报告的绝对时间与 census 来自临时诊断
  worker；仓内 `_timed_step` 仍只返回相对耗时，尚未接入 `mega_moe_measurement`。
  保留原同步和 rank-max 计时边界，精度 snapshot/hash 与 profile-off 基线分开采集。
- [ ] 在空闲门控、匹配源码/native/输入下，用 fresh-process A/B/B/A 分别建立独立 MoE 层和
  connected Qwen 基线；明确 common EP、B0、B1 的比较范围，保持优化器、checkpoint 和容量设置一致。
- [ ] 补齐动态路由、参数更新、连续 generation、B1 residual 3+ waves，以及 native/B0/B1
  匹配曲线。已有 B1 55-route 历史曲线不能替代这些场景。
- [ ] 在计时完成后，分别采集全 rank trace、P2P、task/cycle 和内存诊断，用于 T04/T05。

验收：精度先通过；两组配对方向、全步 wall time、分阶段耗时、allocated/reserved 和 external SHMEM
均可追溯。没有稳定收益时如实保留中性结论；旧分支的收益不作为合并后结果。

入口：[多层 benchmark](../../../../examples/torch/multicore/benchmark_mega_moe_layers.py)、
[逐步占用审计](../../../../examples/torch/multicore/mega_moe_measurement.py)、
[isolated Qwen](../../../../examples/torch/multicore/mega_moe/qwen_moe_isolated.py)。

本轮同时修复独立 MoE native benchmark 的 AdamW 适配：优化器绑定已完成 EP 梯度同步的
原有本地 leaf，避开 DTensor 未注册的 `lerp_` 布局推导，保持参数存储及梯度别名。
CPU 回归逐步核对三次 AdamW 更新、moment 和 zero-grad；不改变全局 DTensor 算子注册。

验收记录（2026-09-08，rebase 前代码 `8ded4835`、`48e736d2`，现对应 `7f8fd732`、`5cf6f9e0`）：
150 项 UT、174 个子测试通过，
py_compile、Ruff、Pylint 和文档检查通过。EP8/E48、单层 T4096/H5120/I1792 的 B0/B1
在三种路由上各连续更新三步，输出、输入/路由/专家梯度及更新后权重均通过 native 对照。
当前 55 条 E48 路由经原始 CPU 路由生成逻辑和真实 planner 核对，B0 为 W1–W4，B1 均为 W1。
另以 EP8/E384、T512/H512/I128、12 个集中热点验证 B1 残余 W3 的三步精度；
这是独立的小形状补验，不代表 E48 生产宽度覆盖了 B1 残余 W3。
该补验期间也有外部进程进入，仅保留数值结果。

16 层、生产宽度的首轮 native 完成三条路由、每条 3 warmup + 7 measured steps。
168 个 rank-step 窗口中 144 个被判为外部进程干扰，24 个在采样范围内无干扰；
整轮审计拒绝通过，后续五轮未启动。没有有效的配对性能/HBM 基线，也未拟合模型。
原始结果、census、逐步审计、精度文件和可复测脚本在
`/home/feiran/doc/megamoe-measurement-baseline-20260908/`；
`validation.json` 汇总身份和结果，`baseline-1/native_a/` 仅可作为工具与干扰诊断证据。
继续配对测量需要每轮约六分钟的空闲八卡时间，不能沿用本轮被污染的耗时。

## P1：用模型驱动参数和实现迭代

### T04 — 峰值 HBM 与 OOM 边界

- [ ] 对照 allocator snapshot/trace 核实独立 storage 的生存区间和别名，重点包括 owned dX clone、
  FP32 dW、owner receive 窗口、低精度 cast 重叠、retained scratch、共享池和暖态 runtime caches。
- [ ] 覆盖换路由后的高水位、checkpoint、延迟 backward、跨流 record_stream、多个 resource group；
  shared execution workspace、跨层 B pool、进程 SHMEM heap 分别计账，不重复累加视图。
- [ ] 将 connected Qwen 的 attention/router/dense 参数、激活、优化器状态与临时量纳入独立训练契约。
  不用独立 MoE 层的账本或去除参考模型后的历史误差承诺整模型 HBM 精度。
- [ ] 将 allocator reserve、通信库内存、外部占用与安全余量显式列出，输出峰值阶段及预算不满足的原因。

验收：已建模 storage 的有效字节与生产分配核对一致；holdout 同时报告 allocated、reserved 和
external heap 的口径及误差。预算筛选显式暴露缺项和不确定性，不把 named storage 总和包装成 OOM 保证。

入口：[memory ledger](../../../../examples/torch/multicore/megamoe_simulator/memory.py)、
[workspace](../modules/mega_moe/workspace.py)、[replica manager](../modules/mega_moe/replica.py)。

### T05 — 关键路径模型与重新校准

- [ ] 先把 receive-only 提前接收、caller join、窗口消费/复用顺序纳入模型，再逐步细化为
  expert/tile 的 AIV/Cube 队列重放，表达 FIFO 阻塞、readiness、ACK/generation 和跨 wave 依赖。
- [ ] 分别校准 BF16 权重刷新、FP32 梯度回传及 owner 累加，区分消息数、消息大小、peer 并发、
  接收宽度、发送端就绪偏斜和实际网络服务。历史约 5 ms/窗口系数不能无条件跨配置复用。
- [ ] 更新 compact descriptor 解码、六类 GEMM、host planner/control、vector/HBM 和 optimizer 成本。
  保留当前 worker 拓扑与格式校验；不能按 descriptor 字节缩小比例直接推导耗时收益。
- [ ] 用当前源码重新生成训练/holdout 数据，按形状、B、路由和窗口分层报告误差及外推范围。
  接入 T04 的实际生存时间，逐步覆盖独立 MoE 和 connected Qwen 两种契约。

验收：给定 counts、形状、训练契约和匹配校准，无需实际运行即可输出耗时、关键路径、峰值 HBM
及适用范围；holdout 不参与拟合。公布相对误差分布及对旧模型的改善，保留残留系统性偏差；
单机误差范围不冒充多机精度。模型筛出的候选须通过独立测量核实方向。

入口：[estimate](../../../../examples/torch/multicore/megamoe_simulator/estimate.py)、
[calibration](../../../../examples/torch/multicore/megamoe_simulator/profile.py)、
[采集 worker](../../../../examples/torch/multicore/profile_mega_moe_costs.py)。
公式及历史误差见[理论成本模型](theoretical_cost_model.md)。

### T06 — EP16/32/64、E384 的 B/容量/窗口联合选择

- [ ] 先扫描 B=0/1/2/3、接收窗口 2/4/8 及容量因子；仅在预算允许时扩展 B=4/6。
  同时覆盖 balanced、Dirichlet、集中/分散热点、逐步漂移、更新频率与层数。
- [ ] 对节点出口带宽、延迟和 rank→node 映射做敏感性分析，区分固定全局 batch 与固定每 rank token 数。
- [ ] 先使用真实 meta 构图和 T04 的高水位/安全余量筛选，再按 step 时间排序，输出每配置的
  活跃副本数、残余 waves、跨节点权重/梯度字节、planner 开销和峰值阶段。

验收：生成可重放的配置表与候选集合，明确在哪种路由/预算下选 B=1、2 或 3；
不提供脱离 counts 的单一“最佳 B”。EP16/32/64 构图与模拟身份和真实 NPU 验证分别标注。

入口：[predictor](../../../../examples/torch/multicore/predict_mega_moe.py)、
[plan audit](../../../../examples/torch/multicore/audit_mega_moe_plan.py)、
[模拟器说明](../../../../examples/torch/multicore/megamoe_simulator/README.md)。

### T07 — Planner 与控制面

- [ ] 分开量化 planner 冷/热缓存、counts D2H、`tolist()`/缓存键、metadata H2D、cache agreement
  和 Python P2P peer-table 构造，结合 EP64 的多进程 CPU 竞争评估瓶颈。
- [ ] 按证据选择缓存键/计数增量、拓扑与复制成本评分或设备化路径；保留一个可复算的 CPU oracle。
  评估 agreement 与 count exchange 合并时，必须保持任一 rank miss 后全组一致的协议。

验收：等价优化逐项核对 slot map、dispatch counts、tie-break、窗口与 generation；
改变评分的优化明确记录语义差异，以完整 step/HBM 而非 planner 局部耗时验收。
仅把 planner 移到设备而保留 host peer 决策，不能宣称已消除 D2H。

入口：[placement](../modules/mega_moe/placement.py)、[route](../modules/mega_moe/route.py)、
[热拷贝与窗口报告](hotcopy_planner_and_gradient_window.md)。

### T08 — 梯度回传与 kernel 调度

- [ ] 优先解释合并后剩余的梯度回传、mixed-rank 发送就绪和 owner 累加停顿，
  再选择最小的 stream/queue 或存储生命周期改动。
- [ ] 前向先关联具体 expert 的 readiness、peer/MTE 竞争和 rank 到达，再调整调度；
  只有解码或长 K dW 被当前证据识别为瓶颈时，才继续 tensor-template 缓存或 dW tiling。

验收：模型说明目标关键路径及预期 HBM 变化；实际验证单 wave、residual multi-wave、
shared-B/checkpoint/跨流生命周期及完整梯度/更新后参数，然后做匹配 ABBA。
保留真实发送 producer wait、owner 全量 FP32 累加、窗口复用和 SHMEM ACK/generation 约束。
无稳定收益或峰值超预算的候选撤回到实验归档，不累积在默认路径。

证据：[receive-only 报告](history/communication_experiments.md#已保留的-receive-only-接收解耦)、
[梯度就绪报告](history/communication_experiments.md#跨-rank-就绪时间线)。

## P2：环境与证据满足后再推进

### T09 — 实际多 rank / 多机验收

- [ ] 按当前 Torch/CANN/HCCL/SHMEM 版本分级验证：不同设备控制组、同卡多进程探针、
  SHMEM 初始化/barrier、最后才是 MegaMoe 持久 kernel 和完整前反向；逐层记录是否支持。
- [ ] 有真实多机后先验证 IP/bootstrap、rank/device 映射、P2P/SHMEM 可见性和 generation，
  再扩展到 EP16/32/64、E384 的数值、连续训练和带宽/延迟校准。

验收：每种规模都有明确的源码/native 身份和实际通信证据。Docker、进程数增加或 meta 构图成功
不能单独关闭多机验收；若通信库不支持同卡多 rank，继续使用 CPU 模型并明确外推边界。

入口：[rank probe](../../../../examples/torch/multicore/probe_mega_moe_ranks.py)、
[模拟与 IP 配置说明](../../../../examples/torch/multicore/megamoe_simulator/README.md)。

## 后置或不再重复推进的方向

| 方向 | 当前决定 | 重新评估的条件 |
| --- | --- | --- |
| local ACLNN bridge | 按用户要求未合入当前功能线；旧实验保留 | 用户重新要求且有新收益依据 |
| 为逐位贴齐 native 重算 dW / 重写 AdamW | 本组舍入与放大已有定位，保留 FP32 replica 聚合及原 optimizer | 出现新的同输入聚合、更新公式或生命周期错误证据 |
| 继续盲扫 LR/eps 或静默修改默认值 | eps 与低 LR 控制已完成；优先全局梯度目标及 norm | 明确的新稳定性假设和独立训练验收合同 |
| 全专家梯度 packing、独立 gradient epilogue | 负收益或无确认收益，只有文档/patch 归档 | 新瓶颈证据和独立对照，不能直接重启旧方案 |
| 两种 peer 轮转 | 历史回退，已撤回 | 有逐 expert readiness 证据的新调度假设 |
| 权重双 buffer / 预测 prefetch | 后置，不增加默认常驻池 | 权重刷新成为关键限制且收益覆盖额外 HBM；计划预测保留 generation 校验 |
| rank-level transport dedup / 全局 E+B VMM | 设计候选，当前不是已实现能力 | 先量化可减少的复制/内存，再单独确定协议与生命周期范围 |
| 历史 worktree 快照再次合并 | 不需要，功能已收束 | 发现独立且尚未包含的实际增量时再核对 |

## 每轮关闭待办时的记录

1. 记录完成的 ID、实现提交、配置/输入、源码/native/硬件身份和原始产物位置。
2. 对实现变更运行相称的静态检查、UT 与所需数值回归；性能候选使用空闲门控与新进程，
   精度失败先处理精度。仅整理文档时检查内容、链接和 diff，不启动设备测试。
3. 分开填写“实现”“正确性”“性能/HBM”“模型误差”；缺少相应证据的部分继续保持未完成。
4. 每轮完成后用普通 `git commit` 本地提交，不用 `autogit`；不自动 push、rebase 或 squash。

## 证据索引

仓内设计与历史实验以链接文件为准；本清单中的合并验收摘要已进入 Git，
以下本机目录提供原始日志、hash 和重放材料，离开本机时需另行携带：

- `/home/feiran/doc/megamoe-merge-260908-20260908-162833/`：`VERIFICATION.json`、
  `REPORT.md`、设备日志、源码/native 前后指纹与 12 个 EP64 重放结果。
- `/home/feiran/doc/megamoe-commit-split-20260908-160900/`：六个主题提交的拆分与逐阶段验证。
- `/home/feiran/doc/megamoe-consolidation-20260908/`：归档 manifest、退役记录、独立数据根与重放脚本。
- [Qwen 单后端驻留报告](history/qwen_validation.md#单后端驻留与首次精度)：原始生产形状首次精度及历史计时边界。
- [Qwen 数值诊断](history/qwen_validation.md#数值诊断汇总)：六轮完整报告、原始数据位置和固定 Git 恢复入口；
  本次只核对入库报告，未重新审计仓外大规模张量快照。
- `/home/feiran/doc/megamoe-qwen-rebase-20260909/`：重基提交映射、文档归档完整性与源码一致性检查；
  重基前分支为 `backup/megamoe-sun-before-qwen-rebase-20260909`，可恢复本地原八条提交。
- [Runtime descriptor v2](runtime_descriptor_v2.md)：动态容量、ABI、worker 拓扑与校准边界。
- [MoonEP + multi-wave 报告](moonep_multi_wave_report.md)：实现合同、已撤回实验与历史性能。
