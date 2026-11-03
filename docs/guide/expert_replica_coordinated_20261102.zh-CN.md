# 专家副本四线收口与集成记录（2026-11-02）

四个实现 session 均已完成当前任务并暂停；D1 窄域独立复审也已完成并暂停。
本次将当前实现按逻辑提交汇集到独立工作树，补充组合回归并本地提交，不提交新的集群或 NPU 任务。
下面将已实现功能、历史实验、当前集成验证和未完成事项分别标明。

## 1. 集成位置与来源

| 项目 | 值 |
| --- | --- |
| 工作树 | `/cache/zfr/hyper-expert-replica-coordinated-20261102` |
| 分支 | `integration/expert-replica-coordinated-20261102` |
| 稳定源分支 | `expert-hot-replica`，`c89a26abf8b2e85821c9d0734f5ba4c145fc6bca` |
| 集成起点 | A 的 `3736c8cfc`，已包含 `45d9a61fa` 的关键路径模型与采集入口 |
| 提交方式 | 精确 cherry-pick，记录原提交；冲突仅在集成工作树处理 |
| 发布状态 | 仅本地提交；没有 push，稳定分支及既有集群包没有改动 |

| 方向 | 原提交 | 集成提交 | 内容 |
| --- | --- | --- | --- |
| A：原关键路径与采集 | `45d9a61fa`、`3736c8cfc` | 原提交保留 | 可选关键路径评分、受限 EP64 采集 |
| A：输入与校准审计 | `5dd6498eb` | `983b210ca` | 来源绑定、缺项拒绝、训练组与留出组分离 |
| A：静态执行参考 | `cc57476e5` | `913e562e0` | 按实际物理 worker 重建 band 队列与 ACK 依赖 |
| A：候选审计工具 | `3920143ff` | `340522da1` | 可移植 CLI 与回归，复现容量预剪枝问题；没有修 planner |
| B：因果观测 | `abfc7ce09` | `c4896b851` | 普通静态队列保持不变，增加 producer、return、ACK 与权重等待观测 |
| C：权重驻留 | `d1880be91` | `9cf909db7` | 按值版本与物化代次安全复用共享槽位权重 |
| D1：CPU 原语 | `4816012b3` | `da1ac3ac0` | 编译有序插值、计数校验与 owner 求和 |
| D1：语义修复 | `90fe76d38` | `7ba4a5a5a` | 限制原生输入域，保留 Python 遍历及整数运算语义，ABI 升至 3 |
| D2：重算元数据 | `ca086b5cc` | `fe361d79c` | 同一 checkpoint invocation 内复用只读整数描述符，默认关闭 |
| 集成组合回归 | 本次新增 | `2a3d31577` | 观测开启时，缓存命中不再被计作矩阵发送，ready 仍按新 epoch 发布 |

没有纳入 B 的旧 ready-cursor、owner-first、双缓冲或增加通信 worker 的实验分支。
部分候选只有微小或负向收益，且不属于当前冻结的静态执行方案。
没有将生成的集群包、权重、二进制或一次性性能结果混入框架源码。

## 2. 当前实现状态

### A：关键路径、实际队列与严格校准

已实现可选关键路径评分与输入校准审计。物理链路服务的拟合和完整融合训练关键路径的
拟合分开处理：仅有矩阵服务时间，不能证明完整 critical profile 可安装。
没有用合成系数替代生产采集，也没有把缺失时间补成零。

新的 `static_band_reference.py` 采用实际物理 worker 的 projection/band/edge 顺序。
W2、W13、owner producer-ready、guest-ready 与 ACK 保持各自语义，避免把真实静态队列
替换为过于自由的堆调度，或增加并不存在的全局 barrier。
已核对 192 条实际 worker 队列、1890 个 return 事件和 18 个 ACK。
这是离线依赖参考；尚未接通完整生产关键路径校准，也没有新的整网性能验收。

当前完整 critical profile 仍拒绝安装：准确 readiness、host exposed、原始 baseline 直方图绑定
与 profiler-off 服务绑定等材料不完整。D1 编译不会自动解决这些模型语义和测量缺项。

### B：静态执行下的因果观测

保留当前静态 band 工作分配，增加按 producer、worker、projection、band、return、ACK
关联的观测。权重侧记录提交等待及实际矩阵复制；不把 CPU 提交时间解释为设备服务时间。
设备事件中的 projection 0 为 W2、1 为 W13；权重预取矩阵索引 0 为 W13、1 为 W2，
分析时按各自契约解码。

独立源码版本已完成 EP4 Push B1/B6 观测关闭/开启对照：8/8 rank 结果、梯度和更新后权重
逐字节一致，Host/Device 均无记录丢失。该结果验证 B 自身的观测实现，不能冒充本集成树
的设备验收。B 的 static-v1 辅助负载包与 static-v2 实际计划包均已冻结，后者精确复用 A 的
四份实际计划，覆盖 256 份 source-rank 布局；尚未提交新的集群采集。

### C：有界权重驻留与 FSDP 安全失效

缓存按物理 guest 槽位保存 owner 端证据，校验 tensor/storage 弱身份、值版本、布局与
物化代次。只有 payload 不变时跳过 BF16 矩阵 put；credit、当前 ready、ACK 和 FP32 梯度
归并协议继续执行。换 owner、原地更新、换 storage、异常与 epoch rollover 会失效。

FSDP 会在保留 autograd version 的条件下重新物化参数，因此 tensor version 和地址不能
独立证明值未变。本次一并纳入 `hsdp_param.py` 的 materialization token；不能单独挑选跳过
复制的分支而丢弃这项保护。当前没有跨 rank 的逻辑值 epoch 证书协议。

### D1：可选 CPU 编译原语

仅编译可证明保持语义的输入准备与按序累加；候选、评分顺序、浮点累加顺序、设备队列
和 ACK 不因编译而改变。自定义模型与不支持的输入域继续使用 Python reference。
可选模块不存在时回退 Python；发现旧 ABI 时明确要求重建。

独立复审发现并修复了两类差异：OrderedDict 的 Python items 顺序不等于底层 dict 顺序；
非零整数时间结点提前转 double 不等于 Python 的精确整数中间运算。
修复后两个原始反例、其他控制与实际调用别名检查全部通过，复审结论为通过当前窄域修复。

普通浮点曲线下，旧 D1 二进制的历史 CPU 首次规划下降约 15.8%–18.8%；重算 replay
约 0.13 ms，没有稳定改善。这些数据绑定旧 ABI2 和合成系数，**不能直接写成当前 ABI3
二进制、实际 v2 模型、NPU 或整网的收益**。

### D2：默认关闭的元数据复用原型

隐藏开关 `HYPER_PARALLEL_REPLICA_METADATA_REUSE=1` 在 fresh process 中启用，默认关闭。
仍执行已有的输入检查、counts gather、完整 CPU readback 和精确 plan replay guard。
命中后只复用同一 checkpoint invocation 内的 dispatch、source-run 和 compact-home counts
整数描述符。相同 counts 但不同逐 token IDs 仍重新 remap。

首次上传记录 ready event；跨 stream 消费执行 wait_event 和 record_stream。
这些生命周期仅完成 CPU/mock 验证，没有 NPU 或真实跨 stream 验收。
新的 device guard/header 只存在 CPU oracle 与接口设计，尚未修改共享 wire schema，
不能宣称消除了 D2H、counts collective 或逐 token permutation。

## 3. A 的已完成集群任务：主要结果与新发现

任务 `replica-critical-path-ep64-20261102-001` 共 8 节点、128 die、两个 EP64 组，五阶段均完成，
8/8 节点退出 0；768 份 rank/config 结果、43008 份 rank-step 与 128 rank 终态已核验。
8/8 份去重原计划按冻结模型与容量精确重建。

这是 stateful 单 MoE 组件筛选，使用固定原始窗口；每步先取组内 rank-max，再取中位数。
统一排除预声明的 profiler step23，其他慢样本及冷启动尾部原样保留。
它不构成 fresh-process ABBA 或完整 DSV decoder 的端到端验收。

以下为第二次热点窗口 `v1_hot`，单位 ms；加速比为对应 B0 中位数除以该列中位数。

| 后端 / EP 组 | B0 wall 中位数 | B1 | B6 | B1 加速比 | B6 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Push / 0–63 | 152.063 | 101.901 | 249.908 | 1.4923x | 0.6085x |
| Push / 64–127 | 152.521 | 101.019 | 251.065 | 1.5098x | 0.6075x |
| Pull / 0–63 | 155.491 | 171.919 | 176.408 | 0.9044x | 0.8814x |
| Pull / 64–127 | 173.126 | 187.750 | 192.267 | 0.9221x | 0.9004x |

Push B1 实际有 10 条副本边，B6 有 63 条；Pull 上述 B1/B6 没有复制边，仍存在回退。
不能把这些结果混成一个总平均，也不能把 wall 减设备时间直接算作暴露 CPU 成本。
Pull 第二组 backward 设备时间本身有变化，尚不能把全部回退归因到 host。

### 已定位、尚未修复：容量预剪枝漏掉更少边的可行计划

同一实际热点直方图、原模型与物理 Bmax6 配置下，将原 B1 的 10 边计划扩展到 B6 布局后：

- 峰值为 38400 行，满足原 B6 接收硬容量；完整计划校验通过。
- 原模型预测约 118.180 ms，优于原 63 边计划的 250.216 ms；这是 CPU 模型预测。
- 原搜索预算 1–5 的中间 thin 配置峰值依次为 102016、86528、71040、55552、40064，
  在完成 quota 分配前就被容量检查 continue，合法的完成候选没有进入评分。
- 仅离线放宽中间搜索边界、保持最终容量 38400，可选回合法 10 边方案。

这证明候选可达性存在问题，单纯继续调通信惩罚系数不能解决候选没有参与比较的问题。
本次按用户暂停要求，没有修改生产 planner。审计 CLI、回归及反事实数据已保留。
中度不均案例原 3 边计划仍优于扩展的 5 边计划，不能把热点结论普遍化。
物理 B6 执行 10 边候选没有 NPU 性能验收，不能将上述预测当作训练加速比。

## 4. C 的实际 FSDP 验证：收益与适用边界

独立 d1880be91 版本完成 8 层、EP2 × DP2、H512/I256/T512、完整 checkpoint、两次 AdamW
更新的矩阵，共 9 case、36/36 rank 退出 0，9120 对 tensor SHA 完全一致。
1450 源文件、54 native 文件和 9 脚本共 1513 份 SHA 未变化，layer 归属没有遗漏。

| 相同 FSDP 策略内 OFF/ON 比较 | 普通预取驻留命中 | 权重矩阵 payload 减少 |
| --- | --- | ---: |
| reshard，GAS1/4/8 | 全部 0 | 0% |
| retain，hot，GAS1 | 有命中 | 6.25% |
| retain，hot，GAS4 | 有命中 | 10.9375% |
| retain，hot，GAS8 | 有命中 | 11.71875% |
| retain，rotate，GAS4 | 有命中 | 9.375% |
| retain，ragged，GAS8 | 有命中 | 11.71875% |
| Pull retain，hot，GAS4 | 有命中 | 10.9375% |

已有两层非 FSDP 小组件曾减少约 43.75% 权重复制字节；八层共享槽位的窗口收益更小，
当前受测生产 reshard 则没有普通预取收益。不能只引用最理想的旧数字。
各 case 的 backward 仍沿用已有保存代次复用，矩阵复制为 0；不能将它全部记作新缓存的收益。
上述比例只针对权重矩阵 payload，不是所有通信或 step-time 改善，也没有物理 HBM 峰值结论。

## 5. 本次统一集成验证

1. 独立 editable 环境实际导入本工作树，未借用其他 session 的 editable 安装。
2. 合并后的 EP、multicore 和相关 FSDP 参数 UT：**793 passed、3815 subtests passed**。
   15 条既有 pytest marker / TypedStorage warning，退出 0。
3. 重新编译并安装本树 CPU host ABI3；使用本树二进制，开启 D2 隐藏开关后，对 host、cost、
   critical-path、routing、metadata 和 transport 进行组合回归：**83 passed、89 subtests passed**。
4. B+C 冲突处新增一项回归：第一次记录实际矩阵字节；命中后不产生 WeightPrefetchSubmit，
   记录零字节 WeightPayloadReuse，两个 projection 仍发布新 epoch 的 INT32 ready。
5. 原始 D1 修复已完成独立窄复审；不把旧普通域性能记录转作当前二进制性能验收。
6. `git diff --check` 与 AGENTS catalog 检查通过。全变更 errors-only Pylint 有两条既有
   `test_param.py` 动态 hook 的 E1101；原 c89 基线同样可复现，未修改无关测试以规避检查。
   汇总文档的 AutoGit check 通过，codespell 因未安装而跳过；不宣称完整源码风格门禁全绿。
7. 本次没有新的 NPU、profiler、集群或整网验证，D2 继续默认关闭。

本地新编译 ABI3 SHA256：
`e9d25d3ec1c73537ae6f7eafb7d3affdd20b864fa78a7457a06ade2c1ae3693c`。
它和 D 原工作树二进制身份不同；当前记录只证明编译、加载与上述 CPU 回归。

重现统一 CPU 测试：

```bash
bash /home/ma-user/work/z00961864/expert_replica_coordination_20261102/integration/env.sh \
  python -m pytest -q tests/ut/core/expert_parallel tests/ut/core/multicore \
  tests/ut/core/fully_shard/test_param.py
```

## 6. 暂停后的优先项与证据保留

后续恢复时，优先修复并验证“最终合法但中间容量超限而提前丢弃”的候选问题；随后用 B 的
实际计划观测包确认 owner/guest-ready、实际队列和暴露 tail，再绑定校准。以上均尚未启动。
当前不能承诺 B>0 在任何负载下都不慢于 B0；零边路径、规划成本、保留 workspace 和设备
差异仍需匹配负载的 profiler-off 对照。

各 session 原工作树均已提交且干净，交接状态保存在协调目录。C002 功能矩阵的 grant 已按
9-case 终态关闭，当前没有协调方发出的活动设备 grant，也没有新增采集或提交。

原始 profiler 的细粒度分析尚未完成，因此保留未分析 raw。分析完成后按用户要求清理 raw，
保留报告、指标、源代码/配置/二进制指纹、输入计划和复现方法；不删除活动或其他 session 的数据。

本机证据目录：

- 协调与各线交接：`/home/ma-user/work/z00961864/expert_replica_coordination_20261102/`。
- A 组件、容量反事实与实际计划：`/home/ma-user/work/z00961864/expert_replica_critical_calibration_20261102/`。
- B 静态因果观测：`/home/ma-user/work/z00961864/replica_causal_static_3736_20261102/`。
- C FSDP 矩阵：`/home/ma-user/work/z00961864/replica_fsdp8_validation_20261102/matrix-v2/`。
- D1 原结果：`/home/ma-user/work/z00961864/replica_host_critical_20261102/`。
- D1 修复：`/home/ma-user/work/z00961864/replica_host_domain_fix_20261102/`。
- D2 原型与 oracle：`/home/ma-user/work/z00961864/replica_metadata_oracle_20261102/`。
- 本次组合测试和 CPU build 身份：协调目录下 `integration/`。

独立复审结论：协调目录 `receipts/R-D1-review-fixed.json` 与 `.md`；原失败复现和旧结论均保留。
集群包原始 SHA 与 v1/v2 实际负载定义保持冻结，今后采集需要生成与集成源码匹配的新版本。
