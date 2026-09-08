# MegaMoe P2P 归因与调度取舍：历史实验

本文汇总 2026-09-07 至 2026-09-08 的历史实验，保留各阶段的测量配置、失败与修复边界、原始文件及哈希。
结果对应各节列出的旧源码和 native 产物；本次整理没有重跑性能或数值验收，也不将它们改记为合并后结果。
旧报告里的“本轮/当前”均指该历史阶段。后续实现与验收统一由
[当前待办](../megamoe_optimization_backlog.md)跟踪，本文不维护另一份优先级。

以下 output/ 与 /tmp/ 路径是当时的本机证据位置，原始大文件未入 Git；本次未核验其仍然可用。
文末固定 Git 版本可恢复完整旧报告，不保证恢复未入库的原始数据。

| 实验 | 基线 / 提交 | 取舍与适用范围 |
| --- | --- | --- |
| P2P payload / wait 归因 | e69350fa / 5a2ae4b0 | 保留诊断；局部等待不是可直接兑现的 step 收益 |
| 整专家梯度合包 | 5a2ae4b0 / 59d4736f | 候选撤回；偏斜完整 step 回退 1.806% |
| 独立 backward 收尾重叠 | 59d4736f / 7e31c518 | 候选撤回；偏斜完整 step −0.037%，无确认收益 |
| 八卡梯度就绪时间线 | 7e31c518 / b5631bda | 保留诊断；receive-only rank 5 持续晚就绪 |
| receive-only 解耦 | b5631bda / 7dba0a70 | 保留；独立 16 层偏斜 step −4.086%，不能外推整模型 |

合并后的实现将 receive-only 接收放在有界窗口上，消费后通过反向 stream 依赖保护下一批复用。
当前存储与同步合同见[planner 与梯度窗口](../hotcopy_planner_and_gradient_window.md)；
本页 receive-only 的旧 ABBA 没有重新测量这一合并增量。
task-cycle/host-scope 成本 profiler 与本页 P2P/all-rank trace 均保留，前者见
[成本模型](../theoretical_cost_model.md)。时间线分析器仍限 EP8、16 层和每层一个非空梯度 batch，
有界窗口的多个 batch 尚需泛化；control-stream wait 不能当成 caller 暴露停顿。

## P2P payload 与等待归因

诊断默认关闭，以 --profile-p2p 开启，在所有性能/HBM 点之后执行额外步骤。
观察器只保留 host metadata/stream/events，正常与异常退出均恢复；空 batch 不分配事件。
该轮不修改生产 runtime、native、FP32 聚合或 optimizer ownership。

### 测量协议与计时语义

沿用前一轮 [修复验收](runtime_validation.md#dx-ownership-与构建修复) 的 matched-initialization helper：
8 × Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1；T=4096、H=5120、I=1792、E=48、K=8，
16 个独立 Qwen-shape MoE 层，EP8、B1/shared-B、显式 capacity=1.5、reentrant checkpoint，
每步 AdamW。所有 forward 后再倒序 backward；**不是 connected transformer 的整网计时**。

每路由 3 warmup + 7 measured，每个样本取 rank max，再取运行中位数。偏斜点为 `0.1:42`，
平衡点为 `10000:42`。诊断放在两个点的全部时间、内存、finite 验证之后。第二轮另采 rank-zero
trace，trace capture/export 完成并退出后，再做独立的 event 诊断；两种诊断不同时启用 observer。

每个非空 batch 在同一 caller stream 记录三个时间戳：

| 指标 | 起止位置 | 能说明什么 |
| --- | --- | --- |
| `launch_to_ready_ms` | 原 batch submit 前 → 原 completion wait 后 | 该 caller-stream 区间；不是纯传输时长 |
| `pre_wait_ms` | submit 前 → wait 入口 | 等待前可容纳其他工作的一段窗口；不是实测 useful-compute overlap |
| `exposed_wait_ms` | wait 入口 → wait 后就绪 | 当前消费者前的剩余等待，仍可能含主机提交/诊断空隙 |
| 三个 `*_cpu_ms` | builder / submit / wait 的独立主机计时 | 描述符准备和 API 调用；不是对应 device 操作耗时 |

F+B 内不加 device-wide sync，只在额外步骤结束后读取 event；跨 stream 的 wait 会拒绝归因，
因为这个串行 example 的计时契约只支持同一 caller stream。该限制不改变生产库的跨流接口。

所有 rank 对每个 phase-call、source/target、shape、dtype 核对 send/receive bytes。全局 payload
**只计发送侧一次**，不能把全局 send + receive 相加再称作有效传输字节。

### 偏斜路由：payload 和暴露等待

两次 capture 均通过逐 peer/round 匹配。此路由实际使用 7 个 B-slot，physical destination load
全部为 32768，原 W2 变为 W1。每个 replica expert 的 gate/up 为 `[5120,3584]`，down 为
`[1792,5120]`；合计 BF16 权重 52.5 MiB，FP32 梯度 105 MiB。

| phase | 每 rank 非空 batches | 全组单次发送量 | 16 层总发送量 |
| --- | ---: | ---: | ---: |
| weight refresh | 31 | 7 × 52.5 MiB | 11945902080 bytes = 11392.5 MiB |
| gradient return | 16 | 7 × 105 MiB | 12331253760 bytes = 11760 MiB |

31 次权重刷新对应 16 次 forward + 15 次倒序 replay；最后一个 forward 的权重可由紧随其后的
同层 checkpoint/backward 复用。两个 phase 的总字节接近，并不表示等待相同。

下表均为单次额外 F+B 的各 rank **累计** caller-stream 暴露等待，单位 ms；不是 wall time 之和。

| rank | 权重 S1 | 权重 S2 | 梯度 S1 | 梯度 S2 | 每次梯度 send / receive MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 47.415 | 38.217 | 124.657 | 126.375 | 105 / 105 |
| 1 | 38.825 | 45.015 | 142.336 | 142.841 | 105 / 105 |
| 2 | 47.387 | 54.704 | 119.138 | 121.828 | 105 / 0 |
| 3 | 39.856 | 37.953 | 85.039 | 85.038 | 105 / 0 |
| 4 | 46.777 | 49.801 | 189.050 | 191.673 | 105 / 210 |
| 5 | 35.826 | 34.684 | 102.531 | 104.441 | 0 / 210 |
| 6 | 38.169 | 35.893 | 85.181 | 85.223 | 105 / 0 |
| 7 | 36.908 | 33.947 | 183.644 | 185.262 | 105 / 105 |

梯度的累计 pre-wait 只有约 `0.003--0.005 ms/rank`，与当前
`_wait_handles(_launch_transfers(...))` 的立即等待结构一致。权重的 pre-wait 为
`60.084--77.096 ms/rank`，存在更大的调度窗口，但不能说这些时间已全部有效重叠计算。
各 rank 累计 CPU wait 仅约 `0.115--0.404 ms`，远小于设备事件跨度：只优化 Python wait 调用
或只看 CPU scope 会误判主要成本。

较大 payload 不完全决定尾部：rank 4 同时向 rank 5 发送、从 rank 0/2 接收，确有更重的总量；
但 rank 7 的总量少于 rank 5，暴露等待反而更高。该事实支持继续检查 peer 依赖、通信争用和
到达次序，**不能仅凭字节表把根因定为带宽或直接推出一种 peer 轮转顺序**。

### 当前时间线交叉检查

`p2p_skew_trace_20260907.json` 有 53177 个 Chrome events。用 `async_npu` 的 start/finish flow
关联 device `batch_send_recvAicpuKernel`，再将 host flow start 精确归入同一 pid/tid 的
`start_shared_prefetch` 或 `reduce_replica_gradients` CPU scope，**47/47 个 batch 唯一匹配**：
31 weight refresh、16 gradient return，与独立的八卡 event capture 一致。

时间戳采用十进制处理，避免约 `1.8e15` 微秒时间戳转 float 后损失精度。此次 rank-zero 结果为：

| 项目 | count | 区间总和 | 区间并集 |
| --- | ---: | ---: | ---: |
| weight AICPU batch control | 31 | 4.215 ms | 不作为 payload 指标 |
| gradient AICPU batch control | 16 | 2.121 ms | 不作为 payload 指标 |
| `SDMA_SQE` | 692 | 224.439 ms | 169.333 ms |

全部 692 个 `SDMA_SQE` 的 `connection_id` 都是无效的 UINT64_MAX，不能像 AICPU batch 一样逐条可靠地
归入 host phase。这轮没有把仅按邻近时间切片的 SQE 当作因果关联，也不把 rank-zero 的
169.333 ms 区间并集说成端到端可节省时间。它与旧 trace 数量/量级接近，但本报告只使用本轮
当前实现重新采集的 trace；旧 trace 不是当前结果的替代。

相关 CPU 回归为 203 passed / 165 subtests；四次真实 EP8 benchmark 检查 payload、
finite 和诊断隔离，没有替代上一阶段 native 数值 oracle。完整默认路径 A/B 和 HBM 表可从文末旧报告恢复。
该轮生产 Python fingerprint 为 17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f；
新增诊断 helper SHA256 为 dabc7152e4ca3c6e366354c2ee4a563739c9a70a9e22a65495d0e98dd666cb52。

### P2P 复现与数据

仓内普通入口可用以下命令采样，输出路径必须不存在：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
HYPER_PARALLEL_PLATFORM=torch OMP_NUM_THREADS=1 \
  SHMEM_IP_PORT=tcp://127.0.0.1:21100 \
  HCCL_NPU_SOCKET_PORT_RANGE=62000-62063 \
  python -m torch.distributed.run --nproc-per-node=8 --master-port=29705 \
  examples/torch/multicore/benchmark_mega_moe_layers.py \
  --variant b1 --layers 16 --warmup 3 --measured 7 \
  --pairs 0.1:42,10000:42 --profile-p2p \
  --output output/qwen_moe_optimizer/p2p_new_run.json
```

本报告数值为与前一轮精确配对的初始化，不是上述普通入口初始化：实际调用仍使用
`/tmp/megamoe-sun-audit.y8Gp0T/benchmark_matched.py`，只替换 rank-local expert 初始化并记录所有
参数 hashes；helper SHA256 为 `0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`。
本轮脚本 `/tmp/megamoe-p2p.Xu7EO6/run_profile.bash` 中保存了环境与 census，四次调用为：

```bash
bash /tmp/megamoe-p2p.Xu7EO6/run_profile.bash skew 0.1:42,10000:42 29700 21095 --profile-p2p
bash /tmp/megamoe-p2p.Xu7EO6/run_profile.bash skew_repeat 0.1:42,10000:42 29701 21096 \
  --profile-p2p --profile-output output/qwen_moe_optimizer/p2p_skew_trace_20260907.json
bash /tmp/megamoe-p2p.Xu7EO6/run_profile.bash balanced 10000:42,0.1:42 29702 21097 --profile-p2p
bash /tmp/megamoe-p2p.Xu7EO6/run_profile.bash default 0.1:42,10000:42 29703 21098
```

原始结果留在 `output/qwen_moe_optimizer/`；重新执行须更换 label/输出路径，不覆盖旧结果：

| 文件 | SHA256 |
| --- | --- |
| `p2p_skew_20260907.json` | `cb06cec90c5308558efd21a5b0987fea5328d2b50b54162f55d13a8cc0525cdd` |
| `p2p_skew_repeat_20260907.json` | `622886b90e9be473da42731c910954b91c6781d9e04fed96df38b642987a6e3c` |
| `p2p_balanced_20260907.json` | `8bf4b395eabb2e8d26f541f4b6cf6a3386b69dcc9b7c703dac58aee755ab812e` |
| `p2p_default_20260907.json` | `f0a0525f11a5a4fe8e5edac4aeb2721ad750f7eea0a0e51a721bd6184e378c72` |
| `p2p_skew_trace_20260907.json` | `90dcdbf3d47bec532e9c59a76a697901b12ad89ad7b40fea844eb643d6a37aa3` |

同名前缀 `.log` / `_monitor_20260907.log` 保存 worker 输出和进程 census。
`p2p_diagnostics_red_20260907.log` 为最初预期失败；`p2p_diagnostics_final_ut_20260907.log` 和
`p2p_diagnostics_final_lint_20260907.log` 保存最终 CPU / Python lint 结果。原始大文件不加入 git。

## 跨 rank 就绪时间线

- `benchmark_mega_moe_layers.py` 新增默认关闭的 `--profile-all-ranks`，必须与 `--profile-output`
  一起使用。默认仍只采 rank 0；打开后导出 `stem.rank0.json` 至 `stem.rank7.json`。
- 每个 rank 使用独立 profiler 和 raw-artifact prefix；所有文件与 benchmark JSON 均需互不冲突。
  在共享输出文件系统上，每个 worker 都预检所有 rank 的目的文件，已有文件不会被覆盖。
- `diagnostic_trace.per_rank` 记录完整 manifest；原 `path/rank` 仍指向 rank 0，保留默认格式。
- capture 仍为所有性能/HBM 窗口结束后的额外训练步骤，独立 event 诊断又在 trace export 后执行。
  默认路径不创建 profiler、event 或新 HBM pool；临时 annotation 在正常/异常退出时恢复。
- 新增固定 16 层 EP8 的 `analyze_mega_moe_gradient_timeline.py`，从精确 host/device flow 关联
  native backward、P2P input-ready record、caller wait 和 owner kernel；不按邻近时间猜测关联。
  输入缺 rank、乱序、非 16 层、flow 不唯一或消费顺序不合法时拒绝输出。

### 方法、时钟与证据边界

硬件为同一主机的 8 × Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1。
仍使用固定 rank-local Xavier 初始化的 16 个独立 Qwen-shape MoE 层：
T4096/H5120/I1792/E48/K8，EP8、B1/shared-B、capacity=1.5、reentrant checkpoint，每步 AdamW。
它不是 connected transformer 整网；profile 也不是精度 oracle 或性能收益证明。

两轮均先测偏斜 `0.1:42`、后测平衡 `10000:42`，3 warmup + 7 measured；然后返回偏斜 route，
完成一次 update，再采一整步的八卡 trace。capture/export 后再跑独立的 P2P event/payload 诊断。
两轮 source、全部初始参数 hashes、route/placement、shared pool、optimizer ledger 和 17 个正式
native artifacts 相同，finite/health 检查通过。生产 Python fingerprint 始终为：

```text
17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f
```

时间戳以 `Decimal` 从微秒转为整数纳秒。使用每个 capture 的 CANN `host/start_info`，按
`collectionTimeBegin × 1000 − clockMonotonicRaw` 去除 wall offset，以同机 monotonic raw 坐标
比较各 rank。已核对安装版本的 `torch_npu.profiler.analysis._profiler_config` 和 CANN parser：
framework 时间戳使用这一 CANN offset 转为 local time；不能把文件创建时间当作设备开始时间。

两轮各 rank 的 offset spread 为 `4.410 / 4.370 μs`，远小于观察到的毫秒级模式，但这个 spread
**不是时钟标定误差的上界**。不外推到多机墙钟同步。分析器还验证每次 wait 不早于 native/data
ready，owner kernel 不早于 wait 结束，并检查 wait 结束不早于所有邻居 ready（10 μs 检查余量）。

两轮共有 `406692 / 407843` 个 Chrome events，各自 **128/128** 个 gradient-return 调用均可唯一
关联 native kernel、P2P batch control 及对应 CPU launch/wait scope。关键区别是：

- `batch_send_recvAicpuKernel` 是控制任务，不是 payload 完成；本例它常在 native backward 真正
  开始之前就已执行。仅把 Python launch 再前移，不能证明去掉了输入依赖。
- 在 `_launch_transfers` 中关联到的 caller-stream `EVENT_RECORD` 才给出本次输入 ready；
  `_wait_handles` 中同一 caller stream 的 `EVENT_WAIT` 给出实际暴露等待。
- rank 5 的接收 batch 同样有等待 caller-ready 的通信流事件。下一候选必须改变这一依赖，
  而不是把 AICPU control 的提前启动解释为梯度数据已经传完。

默认采集中的 HCCL 明细 `local_rank/remote_rank/size` 没有有效 payload 数据，通用 `SDMA_SQE`
也没有可用的逐 batch connection ID。本轮**没有**将它们逐条归入 peer 或当作纯 wire time。
邻居关系来自相同固定 route 的独立 P2P payload capture；两次诊断不是同一次执行，但 phase、
call 数、shape/dtype 和 route 合同一致。未对每条链路的实际带宽或 HCCL 内部争用作唯一根因结论。

### 梯度依赖与关键数据

本路由的 rank-level 发送关系如下；每条边是不同专家的 replica contribution，**不是同一份
梯度逐跳转发**，也不意味着中间 rank 必须先完成 receive 才能发送其独立 partial：

```text
6 → 0 → 4 → 5 ← 7 ← 1 ← 3
        ↑
        2
```

31 次 weight refresh、16 次 gradient return；全组总发送分别为
`11945902080 / 12331253760 bytes/step`，与前两轮优化实验相同。

下表均为 16 层累计设备事件时间，单位 ms。`晚邻居窗口` 定义为每层
`max(0, max(本 rank 所有 P2P 邻居的 ready) − 本 rank wait 开始)`，不是可直接节省的时长：

| rank | native backward S1/S2 | 暴露等待 S1/S2 | 晚邻居窗口 S1/S2 | owner kernels S1/S2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 330.835 / 332.424 | 125.168 / 127.377 | 29.320 / 27.884 | 4.120 / 4.137 |
| 1 | 344.410 / 342.555 | 140.977 / 142.951 | 55.901 / 57.888 | 4.144 / 4.115 |
| 2 | 332.528 / 331.477 | 120.944 / 126.245 | 0 / 0 | 0 / 0 |
| 3 | 388.687 / 388.859 | 85.078 / 85.050 | 0 / 0 | 0 / 0 |
| 4 | 331.873 / 332.334 | 189.630 / 192.530 | 88.116 / 87.883 | 8.920 / 8.917 |
| 5 | 417.567 / 417.429 | 102.447 / 105.269 | 0 / 0 | 5.745 / 5.748 |
| 6 | 355.829 / 354.569 | 85.199 / 85.288 | 0 / 0 | 0 / 0 |
| 7 | 333.739 / 333.687 | 183.945 / 186.840 | 81.604 / 81.713 | 4.133 / 4.128 |

owner kernels 栏为该调用中关联到的 owner kernel 首尾跨度，含其中间隙；不是各 kernel duration
的严格求和，也不是整步 FP32 所有代价。只有 send 的 rank 2/3/6 没有 owner 累加。

两轮全部 32 次逐层比较中，rank 5 都是最晚 ready；各 rank 的 native-end → ready 整步累计仅
`0.55--0.72 ms`，与前一轮独立收尾可用窗口很小的结论一致。ready spread 的均值/最大值为：

| capture | 均值 | 最大值 |
| --- | ---: | ---: |
| S1 | 5.541 ms | 5.716 ms |
| S2 | 5.522 ms | 5.690 ms |

示例 S1 的首个 backward（层 15），相对八卡最早 native backward 开始的时间，单位 ms：

| rank | native 结束 | 输入 ready | wait 结束 | 接收来源 / 发送去向 |
| --- | ---: | ---: | ---: | --- |
| 4 | 21.361 | 21.398 | 35.186 | 0、2 / 5 |
| 5 | 26.838 | 26.879 | 35.186 | 4、7 / 无 |
| 7 | 21.675 | 21.710 | 35.166 | 1 / 5 |

这支持尝试移除 receive-only rank 的本地计算输入依赖。仍需匹配 ABBA 验证：接收可能与 GMM
争用 HBM/HCCS；独立流 allocation 也可能改变 allocator reserve，不能用就绪差直接预测收益。

### 就绪时间线原始文件

从仓库根目录激活正式 payload 和 editable Python；沿用 matched initialization helper 后：

```bash
bash /tmp/megamoe-allrank.uLE5Bv/run_profile.bash skew2 0.1:42,10000:42 29752 21502 \
  --profile-output output/qwen_moe_optimizer/allrank_skew_trace2_20260908.json \
  --profile-all-ranks --profile-p2p
python examples/torch/multicore/analyze_mega_moe_gradient_timeline.py \
  output/qwen_moe_optimizer/allrank_skew2_20260908.json
```

已存在的输出会被拒绝；复跑应换 label、trace base 和独立端口。runner 及最初分析脚本保留在
`/tmp/megamoe-allrank.uLE5Bv/`。matched helper 仍为 `/tmp/megamoe-sun-audit.y8Gp0T/benchmark_matched.py`，
hash `0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`。
仓内分析器 SHA256 为 `2dc6d52b9264a11b7b62a541426598c544696c5b9d5e11c41bed83fac114e344`。

原始数据在 `output/qwen_moe_optimizer/`，未把大文件加入 git：

| 文件 | SHA256 |
| --- | --- |
| `allrank_skew1_20260908.json` | `ed7b4951c6bad2ca5022e21bff0665a43de4f8a6b52e134f6f72b5ff2ddf45c8` |
| `allrank_skew2_20260908.json` | `10e72f1c3a83c5bdcaf0082ed9893334c9410d3193fcd4bfbd5d4276294d61bb` |
| `allrank_timeline1_20260908.json` | `fb15324768131951df9fa4ccfddc30c94fca9a3af5901811ae07a9082dc3fe97` |
| `allrank_timeline2_20260908.json` | `cd5cb1b3287e0bf946a050297ce879e0cb3db8fcebcf000ee3335798560a373a` |

`allrank_skew_trace{1,2}_20260908.rank{0..7}.json` 及各自 `_ascend_pt` raw tree 保留；完整路径由
benchmark manifest 指定。S2 关键 rank 的 trace hashes：

```text
rank4 eb246a5842417a1d8eabbc41b4771d7d5c6283b14a69e93caa9098941cabb3dd
rank5 949b2b92dd43379f03c3beb3b21d70a2bfc46250655cd980b2a7ae9ae4025616
rank7 fc7b4de0697199cee6493753817de3aa9f155a0c31540888bd62be49c11c53ac
```

测试和 lint 日志为 `allrank_{trace,analysis,final}_*20260908.log`；原始 `.log` 和 monitor 也保留。

## 已否决的整专家梯度合包

B=1 候选把 gate/up 与 down 的 FP32 梯度放到同一 flat allocation，发送一个 105 MiB view。
外部非邻接输入必须物化以保持共同 wire protocol；home 参数、owner 和 FP32 累加顺序不变。
真实 NPU storage 检查确认 manager 路径不发生 packing copy；312/312 张量逐位一致。
原始报告另含隔离传输 microbenchmark，局部变快未兑现为完整训练收益。

### 完整训练 A/B/B/A

8 × Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1；EP8、16 个独立 Qwen-shape MoE 层，
T=4096、H=5120、I=1792、E=48、K=8，B1/shared-B，显式 capacity=1.5，reentrant checkpoint，
每步 AdamW。仍是 all forwards 后倒序 backwards，不是 connected transformer 整网计时。

每路由 3 warmup + 7 measured，每样本 rank max，再取运行中位数。顺序 A1/B1/B2/A2，均先
偏斜 `0.1:42`、后平衡 `10000:42`；沿用固定 rank-local 初始化及全部初始权重 hashes。
A 为 detached baseline / 独立 editable venv，B 为当时的 candidate working tree。
四轮期间冻结 runtime、worker、example、helper 和正式 native payload，不与本轮其他 NPU
工作、CPU 回归或编译并行。B2/A2 的额外 P2P 诊断只在全部性能/HBM 窗口结束后执行。

| 运行 | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| A1 | 1230.893 | 1362.091 | 672.413 | 803.577 |
| B1 | 1257.047 | 1388.052 | 672.356 | 803.724 |
| B2 | 1260.426 | 1391.467 | 671.630 | 802.768 |
| A2 | 1233.871 | 1368.109 | 673.726 | 805.009 |
| A 均值 | 1232.382 | 1365.100 | 673.069 | 804.293 |
| B 均值 | 1258.737 | 1389.760 | 671.993 | 803.246 |
| B/A − 1 | +2.139% | +1.806% | −0.160% | −0.130% |

两次 B 的偏斜时间均比两次 A 慢；平衡点的亚百分比变化不作为收益。四轮 configuration、
初始参数、route/placement、shared pool、optimizer ledger 及 17 个 native artifact hashes
相同，各自首尾 source freeze 和 finite 检查通过。9/9/10/10 份设备 census 仅观察到每轮八个
worker；这不是排除所有瞬时竞争的硬件独占保证。

### 等待与 HBM：为什么不保留

B2/A2 的 P2P 核对仍为 31 次 weight refresh、16 次 gradient return。全组总发送量仍为
`11945902080 / 12331253760 bytes`；gradient send descriptors 从 224 减为 112，receive 同样
减半，逐 phase-call、peer、dtype、shape 的匹配均通过。

部分 rank 的累计梯度暴露等待如下，单位 ms：

| rank | 原版 A2 | 合包 B2 |
| --- | ---: | ---: |
| 0 | 127.163 | 127.730 |
| 2 | 124.630 | 133.872 |
| 3 | 85.045 | 85.055 |
| 4 | 199.370 | 217.403 |
| 5 | 112.502 | 139.774 |
| 6 | 85.261 | 85.275 |
| 7 | 192.931 | 220.408 |

纯发送 rank 3/6 几乎不变，rank 4/5/7 明显变差。CPU launch 减少但准备时间增加；主要变化
不是少调几个 Python API 就能抵消的。这个现象支持继续调查真实到达时序、消息粒度与 peer
依赖，但**尚未证明具体 HCCL 分块、链路或 allocator 机制是唯一根因**。
不能把事件累计值解释为纯 wire time，或将它们相加当作端到端损失。

两次同版本运行的逐 rank allocator ledger 一致。下表为 rank-max bytes：

| 指标 | 原版 | 合包候选 | 差值 |
| --- | ---: | ---: | ---: |
| 偏斜 peak allocated | 24539358208 | 24540404736 | +1046528 |
| 偏斜 conservative peak | 25411773440 | 25412819968 | +1046528 |
| 偏斜 peak reserved | 25904021504 | 25713180672 | −182 MiB |
| 平衡 peak allocated | 24057401344 | 24059496960 | +2095616 |
| 平衡 conservative peak | 24929816576 | 24931912192 | +2095616 |
| 平衡 peak reserved | 25904021504 | 26155679744 | +240 MiB |
| 两点 final allocated | 22150163968 | 22152259584 | +2095616 |

configured SHMEM heap 保持 `872415232 bytes/rank`，shared weight/gradient pool 的逻辑字节数
保持 `55050240 / 110100480`。更改 allocation 形状仍改变了实际 allocator 状态，不能用“池的
元素数量相同”代替整步 HBM 验收。平衡点保留之前启用副本的 allocator 历史。

候选和专属 runtime/test 改动均已撤回，没有留下默认关闭的开关。
补丁恢复方式集中在[补丁恢复](#已撤回补丁的恢复)。

### 合包实验原始证据

补丁 SHA256：`cc1103651100a7ce4f398edbfc4d8084f6c951c1427f1201ef5240ba20f36c51`。
生产 Python fingerprints：

```text
A 17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f
B e860452dbf002b6319edd395e51e4e64089072e815659ef4753c343c15aa4071
```

本轮目录 `/tmp/megamoe-gradpack.VUlkAi/` 保存 baseline、baseline-venv、transport probe 和三个
运行脚本 `run_probe.bash`、`run_precision.bash`、`run_abba.bash`。matched benchmark / precision
helper 位于 `/tmp/megamoe-sun-audit.y8Gp0T/`，前者 hash 与前一归因报告相同。

原始结果在 `output/qwen_moe_optimizer/`，未将大体积运行数据加入 git：

| JSON 文件 | SHA256 |
| --- | --- |
| `gradpack_old_a1_20260907.json` | `046ab1b2e6cc86338704d95e34fbb112a25e66ba1304021bf3a02cb53bdccf1e` |
| `gradpack_packed_b1_20260907.json` | `fb7a8ce31dc024a3c846c4653680d8d691bf5904835634e6741fbe220cd6073f` |
| `gradpack_packed_b2_20260907.json` | `694dafe0c0f81168b44b9540f6968b80cb8269983e359309956a154ce7c064f3` |
| `gradpack_old_a2_20260907.json` | `36d85f805b4d10e0692fe55682f9899499fe0e8dd44f4fc958f210191bd5e8c1` |
| `gradpack_transport_20260907.json` | `b5629e4d145ba5f4ae2567d8b43da96aaa482eed8229378e234eef2e9b08f20c` |
| `gradpack_precision_comparison_20260907.json` | `676207637f0cd6947f26691da8513f1bba0614804fa0ded58af9050307c9d86a` |

对应 `.log` / monitor 日志保留。`gradpack_{red,full_ut,lint,regression_st}_20260907.log` 为候选验证，
`gradpack_shared_b{1,2}_ep8_20260907.log` 为八卡 native oracle，`gradpack_evidence_20260907/`
为 connected/resource 证据，`gradpack_reverted_ut_20260907.log` 为恢复后的回归。

## 已否决的独立 backward 收尾重叠

候选把独立 dX/route 收尾放到梯度 P2P launch 与 wait 之间。
含发送路径保留 native dW 就绪，owner 在 wait 后按原顺序 FP32 聚合；callback 异常先等待再释放资源。
312/312 张量逐位一致，候选、callback 与专属测试最终全部撤回，恢复 59d4736f 的生产路径。

### 匹配 A/B/B/A

8 × Ascend 910B3，CANN 9.2.0，Torch/Torch-NPU 2.9.1。EP8、16 个独立 Qwen-shape MoE 层，
T4096/H5120/I1792/E48/K8，B1/shared-B，capacity=1.5，reentrant checkpoint，每步 AdamW。
仍为 all forwards 后倒序 backwards，不包含 attention/router 网络，不是 connected transformer 整网。

各运行先偏斜 `0.1:42`、后平衡 `10000:42`，3 warmup + 7 measured。每样本取 rank max，然后取
运行中位数；A/B 各两次中位数再求均值。独立 editable baseline 为 `5a2ae4b0`，其生产代码与
本轮起点 `59d4736f` 相同。B 为当时的候选 working tree。

四轮期间冻结所有 runtime、example、worker、helper 和正式 native payload，不并行运行回归或编译。
B2/A2 的 P2P 诊断均在所有性能/HBM 窗口结束后另跑；不把 profiler 时间混入计时。

| 运行 | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| A1 | 1228.152 | 1359.502 | 669.863 | 800.971 |
| B1 | 1230.206 | 1360.817 | 673.000 | 803.947 |
| B2 | 1230.335 | 1361.201 | 676.532 | 807.571 |
| A2 | 1232.411 | 1363.524 | 669.391 | 800.264 |
| A 均值 | 1230.282 | 1361.513 | 669.627 | 800.617 |
| B 均值 | 1230.270 | 1361.009 | 674.766 | 805.759 |
| B/A − 1 | −0.001% | −0.037% | +0.767% | +0.642% |

偏斜 F+B 只差 `0.012 ms`，full step 差 `0.504 ms`，小于本轮运行间波动，不能宣称加速。
平衡点两次候选均慢于两次 baseline，但样本有限且受先前偏斜 allocator 历史影响，不将此表外推
为所有平衡训练的固定回退，也没有证明它的唯一机制。

四轮 configuration、全部初始权重 hashes、route/placement、shared pool、optimizer ledger、17 个
native artifacts 及 benchmark helpers 均匹配；各自 source 首尾冻结检查、finite 和 device health
检查通过。同版本两次的逐 rank 非 runtime allocator ledger 一致。10/9/10/9 份设备 census 的 PID
并集分别恰好等于运行中另行核对的八个 worker；这不是硬件独占或排除瞬时竞争的保证。

### 等待窗口与显存

消息协议逐 batch、phase-call、peer、shape 和 dtype 完全相同：31 次 weight refresh、16 次 gradient
return。全组总发送仍为 weight `11945902080`、gradient `12331253760 bytes/step`。

下表为一次额外诊断中，16 个梯度 batch 的各 rank 累计 event 时间，单位 ms：

| rank | A2 等待前窗口 | B2 等待前窗口 | A2 暴露等待 | B2 暴露等待 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.004 | 1.133 | 124.800 | 128.358 |
| 1 | 0.004 | 1.136 | 142.020 | 140.817 |
| 2 | 0.003 | 1.127 | 119.082 | 124.897 |
| 3 | 0.003 | 1.314 | 85.036 | 83.995 |
| 4 | 0.005 | 0.874 | 189.448 | 195.157 |
| 5 | 0.004 | 0.685 | 102.519 | 108.304 |
| 6 | 0.003 | 1.334 | 85.229 | 84.211 |
| 7 | 0.004 | 0.932 | 183.681 | 189.793 |

等待前窗口确实出现，但量级只有约一毫秒/整步/每 rank。该事件区间仍含 enqueue/调度空隙，
不能全部算作与 wire 重叠的有效计算，更不能相加各 rank 的等待当作可节省的 step。
纯发送 rank 3/6 的等待减少约一毫秒，owner/fan-in 路径并未普遍改善；具体到达与排队机制仍需
共同时间线验证。这里没有新采 all-rank profiler trace，不能声称已证明 HCCL 的内部因果关系。

下表为 rank-max bytes；两次同版本运行的各项数值一致：

| 指标 | 原版 | 候选 | 差值 |
| --- | ---: | ---: | ---: |
| 偏斜 peak allocated | 24539358208 | 24540406272 | +1048064 |
| 偏斜 conservative peak | 25411773440 | 25412821504 | +1048064 |
| 平衡 peak allocated | 24057401344 | 24057401344 | 0 |
| 平衡 conservative peak | 24929816576 | 24929816576 | 0 |
| 两点 peak reserved | 25904021504 | 25910312960 | +6291456 |
| 两点 final allocated | 22150163968 | 22150163968 | 0 |

SHMEM heap 固定 `872415232 bytes/rank`，shared weight/gradient pool 固定 `55050240 / 110100480`。
本轮没有新池，但改变 allocation 顺序仍改变了实际 allocator 状态；“逻辑池不增加”不足以代替
HBM 验收。小幅增加并非唯一否决原因，核心仍是没有确认的端到端收益。

### 收尾实验原始证据

补丁 SHA256：`630589a7ade165add348bde3a03ed0d61ca3c54e16ff2e399a474bcb0a2794a0`。
生产 Python fingerprints：

```text
A 17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f
B 932c298e3a2e0e70d251f0541d73826ec3979449b32f706089daba222a193c40
```

本轮 `/tmp/megamoe-epilogue.Lnfegc/` 保存三个运行脚本、`compare_abba.py` 和两版 precision tensors。
沿用 `/tmp/megamoe-gradpack.VUlkAi/{baseline,baseline-venv}`，并重新运行精度 baseline；matched
benchmark/precision helpers 仍位于 `/tmp/megamoe-sun-audit.y8Gp0T/`。benchmark helper SHA256 为
`0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`，与上一轮一致。

原始数据保存在 `output/qwen_moe_optimizer/`，未把大体积数据加入 git：

| 文件 | SHA256 |
| --- | --- |
| `epilogue_old_a1_20260908.json` | `de27d2683143dc6085d05abc15dceaf35e53ff55222e98e094d0bf92a7743337` |
| `epilogue_overlap_b1_20260908.json` | `7d6cee544b6e89a14766a91868251ac99960166fe6397d39485a69fc4f5695a3` |
| `epilogue_overlap_b2_20260908.json` | `853e56718b1058ce5e7ab2b5f6b7c6abeca3f2a342d1040a51d8f1494ed4136d` |
| `epilogue_old_a2_20260908.json` | `b338908d338250496c526a8da6b0452dac17588a90a9cc45b8d7b62c8187ca19` |
| `epilogue_comparison_20260908.json` | `fa552f619752a399d42d40cc370312f05c788994cb3d4118969a24b55e6cadae` |

对应运行/monitor 日志保留。`epilogue_{red,execution_red,full_ut,lint,regression_st}_20260908.log`
为候选验证，`epilogue_shared_b{1,2}_ep8_20260908.log` 为八卡 native 对照，
`epilogue_precision_{old,overlap}_20260908.log` 为跨版本检查，`epilogue_evidence_20260908/`
为 connected/resource 证据，`epilogue_reverted_ut_20260908.log` 为恢复后的回归。

## 已保留的 receive-only 接收解耦

### 候选与依赖边界

本轮接续 [八卡梯度就绪诊断](communication_experiments.md#跨-rank-就绪时间线)：固定偏斜路由中，rank 5
只接收 replica 梯度，却在 32 次逐层比较中始终最晚开放 P2P 输入。候选只移除这条不必要的
本地 native backward → receive 输入依赖，不把此前约 88/82 ms 的邻居重合窗口当作预计收益。

改动限定在 `ExpertReplicaManager`：

- 从本次 immutable placement 确认本 rank 有接收且没有发送，并要求 NPU shared-B 梯度路径。
  CPU、legacy physical gradient、空 batch、send-only 和 mixed send/receive 均保持原 caller stream。
- 复用已有 cache-agreement stream；不创建第二份权重或梯度 B pool，不改变通信 tensor、peer、
  dtype、shape、顺序或 payload。此处的接收 buffer 与尚未完成的 home dW 没有 storage 别名。
- receive buffer 在 control stream 内分配，避免无输入 event 的跨流使用碰到 caller allocator
  尚未完成消费的旧 block。不能简单把 caller 上分配的 buffer 移到另一条流进行 DMA。
- 每个接收 tensor 都 `record_stream(consumer)`；P2P launch 和 `Work.wait()` 在 control stream
  内完成 enqueue，随后 caller 执行 `wait_stream(control)`，再按原顺序累加 owner FP32 partial。
  这既保留读取依赖，也保护异步 owner kernel 读取期间的 allocator 生命周期。
- 构建、record、launch、wait 或 join 失败后恢复流上下文、尝试 join，并使 shared pool 失效，
  阻止后续 lease 继续复用。它不是通用 HCCL 故障恢复或部分 DMA 已取消的保证。

native、planner、optimizer ownership、dX owned-storage clone 和 FP32 聚合后再 cast 的规则均不变。
这里没有删除含发送 rank 的 native dW producer event，也没有改变跨层 shared-B lease 的释放顺序。

### 正确性验收

按 `add-unit-test` 先补回归，旧代码因缺少新路径按预期失败。异常测试覆盖 record 和最终 join；
分析器另按 TDD 补齐双流语义、发送依赖及 owner 消费顺序。最终全范围 CPU 回归为
**217 passed / 196 subtests，23.59 s**；四个改动 Python 文件的 pylint 均通过。

两版本均重新以 editable 安装，并从 checkout 外核对 import 路径：A 为 `b5631bda` detached
worktree，B 为当前工作区。EP2、三层、四次 AdamW 更新的 matched precision 对照覆盖：

- balanced / 两种 hot-expert 路由切换；
- 多个 forward 后逆序 backward；
- 普通与 reentrant checkpoint 交替；
- output、dX、droute、参数梯度、更新后参数和完整 AdamW state。

**312/312 张量逐位一致**，两 rank 的保存文件也分别具有相同 SHA256：

```text
rank0 bf2910a27612bca0cd1d775d084bf8db42af9902eed4cb00614265ee3004eb59
rank1 9f26d2d46f7b94ad413ff5d9a3ebcd442e69481ed4236230fac01573e51a7f0a
```

5 项原生/共享资源 ST 全部通过（193.96 s，18 deselected，6 条既有 marker warning）。实际 EP8
B1/B2 各通过 21 组原生 MoE 对照，包含普通和两种 checkpoint 模式、延迟反向、换路由、参数更新、
跨 compute stream 和 storage 稳定性。B1 覆盖 W1/W2/W3，B2 覆盖 W1/W2。
小形状验证中的 shared weight/gradient bytes 分别为 B1 `393216/786432`、B2 `786432/1572864`；
这仅证明逻辑池未扩大，不能替代正式 workload 的 allocator HBM 测量。

### 性能比较方法

沿用固定 rank-local Xavier 初始化的 matched helper：16 个独立 Qwen-shape MoE 层，
T4096/H5120/I1792/E48/K8，EP8，B1/shared-B，capacity=1.5，reentrant checkpoint，每步 AdamW。
它不是 connected transformer 整网。每轮 3 warmup + 7 measured，先测偏斜 `0.1:42`，再测平衡
`10000:42`；每步先取 rank maximum，再取中位数。A1/B1/B2/A2 各使用 fresh 八卡进程。

测试、lint 和精度进程全部退出后才启动性能窗口。正式 17 个 native artifacts、输入、路由、
optimizer 和 benchmark/worker/helper 在四轮间保持固定；只允许候选 production Python 指纹不同。
每 10 s 保留一次设备进程 census，不把离散采样宣称为排除了所有瞬时竞争的硬件独占保证。

B2 的全 rank trace 在所有性能/HBM 窗口后采集，额外 P2P event/payload capture 又在 trace export
之后执行；A2 同样在计时后采 P2P。**receive-only rank 的 P2P event wait 现在位于 control stream，
不能再把它当作计算流的暴露等待。** 必须单独关联 caller join，才可比较真正的计算流停顿。

### 接收提前与剩余瓶颈

同一次 B2 trace 中，128/128 次梯度调用均通过精确 host/device flow 关联，只有 rank 5 的
16 次 receive-only 调用使用独立流（stream 44），并在 native caller stream 47 上完成 join。
其他 rank 的发送输入依赖保持不变。CANN wall-offset spread 为 3.890 μs，不作为误差上界。

分析器现在同时报告 `p2p_wait_*` 与真正的 caller `wait_*`，拒绝没有 caller join 的独立接收，
也拒绝把含发送 batch 放到无 native producer 依赖的流上。其余 EP8/16 层及 exact-flow 合同不变。
重新解析旧 S2 时，全部 128 次调用的原有字段均与旧分析结果相同；仓内分析器解析新 trace 的
输出与初步分析逐字节一致，不通过改变历史公式制造收益。

以下为 16 层累计时间，单位 ms。旧 S2 来自上一轮已排查进程竞争的八卡 capture；两列是
诊断窗口，不是性能 ABBA，不能把各 rank 时间相加推算训练 step。

| rank | 旧 S2 caller wait | 新 caller wait | 新 P2P-stream wait | 新 native backward |
| --- | ---: | ---: | ---: | ---: |
| 0 | 127.377 | 122.435 | 122.435 | 331.727 |
| 1 | 142.951 | 142.717 | 142.717 | 342.656 |
| 2 | 126.245 | 115.768 | 115.768 | 330.662 |
| 3 | 85.050 | 85.070 | 85.070 | 389.301 |
| 4 | 192.530 | 128.828 | 128.828 | 332.308 |
| 5 | 105.269 | 39.812 | 650.287 | 418.437 |
| 6 | 85.288 | 85.247 | 85.247 | 355.301 |
| 7 | 186.840 | 123.061 | 123.061 | 332.891 |

rank 5 的 control-stream wait 覆盖了较早发起接收到远端 dW 完成的时间，其增加并非计算流回退。
首个 backward 中，相对本层八卡最早 native 起点：rank 5 的接收 ready 为 `-11.622 ms`，
native 结束为 `26.394 ms`，caller join 为 `26.439 → 28.282 ms`，owner 首个 Add 在 `28.282 ms`
之后。即接收可以早于本地 backward，但 owner 仍等待本地 dW 与接收两侧完成。

rank 4/7 的晚邻居重合窗口降到累计 `5.163/0.727 ms`；rank 1 仍有 `57.626 ms`，且它的 batch
含发送，不能照搬 receive-only 路径。就绪后的通信/排队等待仍明显，不宣布达到优化上限。

通信描述符逐 batch 对比完全一致，weight refresh / gradient return 的总发送量仍为
`11945902080 / 12331253760 bytes/step`，没有以减少梯度精度或丢弃贡献换取重叠。

### 进程竞争与首轮数据边界

首轮 A1/B1/B2/A2 分别保留 9/9/10/9 次 census。B1 在 `01:40:19/01:40:29` 看到了额外的
`akg_benchmark` 进程，其中一个占用约 43 GiB，无法精确排除与计时窗口重合；因此首轮完整
ABBA **不作为干净的最终性能验收**。其余三轮 census 未出现额外 PID；B2 trace 的八个 PID 与
benchmark worker 一致。原始数据及这一差异全部保留，未用删除异常样本的方式美化结果。

首轮偏斜 full-step 均值为 `1361.471 → 1307.190 ms`（-3.987%），F+B 为
`1230.860 → 1176.484 ms`（-4.418%）；平衡 full step 为 `803.945 → 803.934 ms`。
这些值是带上述环境限定的初步证据；随后另起 fresh processes 重跑完整 A/B/B/A。

### 重跑的最终性能与 HBM 验收

在测试/分析进程退出且八卡均无进程后，使用新端口和新输出重新执行 C1/D1/D2/C2，分别是
旧/新/新/旧。四轮的 9/8/10/16 次 census 均只出现各自八个固定 Python PID，未见额外进程。
这是离散进程采样，不是硬件独占保证。配置、初始化、route、native、helper、P2P descriptors、
pool 和 optimizer ledger 全部匹配；同版本两轮的所有非 runtime-free allocator 字段一致。

| 指标（ms） | C1 旧 | D1 新 | D2 新 | C2 旧 | 旧均值 → 新均值 | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 偏斜 F+B | 1231.662 | 1179.911 | 1179.184 | 1238.330 | 1234.996 → 1179.547 | -4.490% |
| 偏斜 full step | 1362.789 | 1311.066 | 1309.745 | 1369.674 | 1366.232 → 1310.406 | -4.086% |
| 平衡 F+B | 674.189 | 672.166 | 672.845 | 672.026 | 673.108 → 672.506 | -0.089% |
| 平衡 full step | 805.383 | 803.352 | 803.987 | 803.755 | 804.569 → 803.669 | -0.112% |

HBM 的 rank maximum，单位 bytes：

| 指标 | 旧 | 新 | 增量 |
| --- | ---: | ---: | ---: |
| 偏斜 peak allocated | 24539358208 | 24540406272 | 1048064（约 1 MiB） |
| 偏斜 conservative peak | 25411773440 | 25412821504 | 1048064 |
| 平衡 peak allocated | 24057401344 | 24057401344 | 0 |
| 平衡 conservative peak | 24929816576 | 24929816576 | 0 |
| 两点 peak reserved | 25904021504 | 25941770240 | 37748736（36 MiB） |
| final allocated | 22150163968 | 22150163968 | 0 |

逐 rank 检查确认只有 rank 5 有上述差异。平衡点后测，reserve 沿用偏斜时的 allocator cache；
不能将其解释为纯平衡运行本身必然多占 36 MiB。共享 weight/gradient pool 仍为
`55050240/110100480 bytes/rank`，heap 为 `872415232 bytes/rank`。没有把 allocator reserve 变化
冒充已经定位的碎片根因，也不为消除这 36 MiB 而引入更大的常驻接收池。

最终四轮原始 JSON SHA256：

| 文件 | SHA256 |
| --- | --- |
| `recvonly_old_c1_20260908.json` | `08a88fae03a0ff149864947f3e71a93749da68f57606583c545d7bd9423a43d6` |
| `recvonly_receive_d1_20260908.json` | `5faa27f17019ec51d043c7f8b8a982217a45fed4c20f02a3056c22fd6ff5d773` |
| `recvonly_receive_d2_20260908.json` | `90cc8149be84152de7948c9fdb546f57aa843ea3b0d32f28af43a505c411e795` |
| `recvonly_old_c2_20260908.json` | `2c313466f65124873215ee6d18ea87be1c254a044535ec3f9421918201002215` |

### Receive-only 原始证据

runner、detached baseline 和对照脚本保留在 `/tmp/megamoe-recvonly.4xwSPp/`：

```bash
bash /tmp/megamoe-recvonly.4xwSPp/run_precision.bash
python /tmp/megamoe-recvonly.4xwSPp/compare_precision.py
bash /tmp/megamoe-recvonly.4xwSPp/run_validation.bash
bash /tmp/megamoe-recvonly.4xwSPp/run_abba.bash
python /tmp/megamoe-recvonly.4xwSPp/compare_abba.py
bash /tmp/megamoe-recvonly.4xwSPp/run_clean_abba.bash
python /tmp/megamoe-recvonly.4xwSPp/compare_clean_abba.py
```

复跑应更换 label、输出目录和独立端口，不覆盖既有原始证据。日志与 JSON 均位于
`output/qwen_moe_optimizer/recvonly_*20260908.*`，大型 trace/raw tree 不加入 git。
matched benchmark helper 为 `/tmp/megamoe-sun-audit.y8Gp0T/benchmark_matched.py`，SHA256：
`0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`。

生产 Python 指纹 A/B 分别为：

```text
A 17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f
B 88dd2cd0d40036f8be1754aa2b64e68389116b0f1ee3106ad9fe25c57afdf685
```

仓内分析器 SHA256 为 `d2c18eb28c12fec69799301c116e23363c21e503805cc079792a681913c3a206`。
候选时间线 JSON SHA256 为 `bf770fb37fe589b9e85dfb94438bd95cecb0f3fe03cafc4bce565d76a8cd2d45`。
初始实验脚本也保留，不把其统计结果冒充完整模型训练或多机标定证据。

## 早期 shared-B 与前向调度记录

以下实验原先只记录在总报告中，保留主要取舍，详细配置、逐次数据和原始路径可从文末固定提交恢复
moonep_multi_wave_report.md 的旧第 10 至 13 节。它们各用独立基线，不能累加百分比，
也不能把八层 legacy compact、16 独立层 shared-B 和无 optimizer 的四态 kernel 矩阵混算。

| 实验 | 历史结果 | 保留结论 |
| --- | --- | --- |
| 同版本 shared-B cache | 相对 cache-off 完整 step −1.182%，allocator/SHMEM 字段不变 | 保留 versioned cache 与全 EP agreement |
| 独立 control-stream agreement | 避免依赖 compute 尾部 | 保留同步合同；详细 ABBA 见旧第 10.2.1 节 |
| B1 55-route sweep | 含副本点放置后均为 physical W1 | 未覆盖 native/B0/B1 全曲线、动态切换或 residual waves 性能 |
| remap 显式 output_size | 移除 32 次 sum/item；ABBA 无确认 step 收益 | 保留等价 sizing 简化 |
| 按 worker round 轮转前向 peer | 匹配 ABBA 偏斜 step +2.609% | 候选及 selector 撤回 |
| 按 expert 轮转前向 peer | 单次 B/A 偏斜 step +3.746% | 候选撤回；未做重复四次的统计验证 |
| legacy compact | 八层 sampled HBM 比 snapshot 低 1.971 GiB，仍比 B0 高 3.783 GiB | live/reserved/设备采样必须分开 |

两种 peer 轮转虽减少最忙 worker payload，Cube readiness 与总 step 未改善；各 rank HBM 字段不变。
原始文件为 output/qwen_moe_optimizer/peer_rotation_{a1,b1,b2,a2}_20260907.json 和
peer_expert_rotation_{b1,a1}_20260907.json；实验 runtime SHA 分别为：

```text
worker round: 6b7941f8aa3b52e7607bbceec33bf36e20af740364d1586ac0e278350bf020cc
expert:       bac8e174592e8393154c4720c68e95001782d8ceec8404cf25a705e0086e8ac1
```

## 已撤回补丁的恢复

补丁从当前树移除，以下提交仍保留原始内容。恢复文件不改变当前运行实现；
仅在表中基线的隔离 checkout 使用 git apply --check --unidiff-zero，通过后再应用。

| 实验 | 补丁保存提交 | 应用基线 | 补丁文件 |
| --- | --- | --- | --- |
| 整专家合包 | 59d4736ff60dab5da88b89404de0d0ef1507c399 | 5a2ae4b0 | gradient_pair_packing_20260907.patch |
| 独立收尾重叠 | 7e31c5188ffd9389eeb90e4c3ad4f1e0bf58a424 | 59d4736f，或生产代码相同的 5a2ae4b0 | gradient_epilogue_overlap_20260908.patch |

从仓库根目录导出到新的临时目录，并与对应实验节的 SHA256 比较：

```bash
MEGAMOE_PATCH_DIR=$(mktemp -d)
git show 59d4736ff60dab5da88b89404de0d0ef1507c399:hyper_parallel/core/multicore/doc/experiments/gradient_pair_packing_20260907.patch > "$MEGAMOE_PATCH_DIR/gradient_pair_packing_20260907.patch"
git show 7e31c5188ffd9389eeb90e4c3ad4f1e0bf58a424:hyper_parallel/core/multicore/doc/experiments/gradient_epilogue_overlap_20260908.patch > "$MEGAMOE_PATCH_DIR/gradient_epilogue_overlap_20260908.patch"
sha256sum "$MEGAMOE_PATCH_DIR/"*.patch
```

## 原报告与恢复入口

整理前的完整文档固定保存在提交 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec，路径前缀为 hyper_parallel/core/multicore/doc/。
下表文件已由本文替代；旧阶段待办及完整过程叙述仍可从该提交读取。

| 原文件 | 本文锚点 |
| --- | --- |
| megamoe_p2p_attribution_20260907.md | [P2P payload 与等待归因](#p2p-payload-与等待归因) |
| megamoe_gradient_readiness_20260908.md | [跨 rank 就绪时间线](#跨-rank-就绪时间线) |
| megamoe_receive_only_overlap_20260908.md | [已保留的 receive-only 接收解耦](#已保留的-receive-only-接收解耦) |
| megamoe_gradient_p2p_optimization_20260908.md | [已否决的整专家梯度合包](#已否决的整专家梯度合包) |
| megamoe_gradient_epilogue_overlap_20260908.md | [已否决的独立 backward 收尾重叠](#已否决的独立-backward-收尾重叠) |

从仓库根目录查看原文：

```bash
git show 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec:hyper_parallel/core/multicore/doc/megamoe_p2p_attribution_20260907.md
```

早期实验的原总报告也保存在同一提交、同一路径前缀下。
