# MegaMoe shared-B P2P 归因与优先级复审

## 1. 本轮结论与改动边界

本轮完成 [主报告第 15 节](moonep_multi_wave_report.md#15-下一步优先级) 的首项：在当前
`megamoe-sun@e69350fa` 正式 native payload 上，分开统计 weight refresh / gradient return 的真实
tensor payload、CPU 提交与 caller-stream 暴露等待，并复核当前时间线。

两次独立偏斜诊断中，16 层累计梯度回传暴露等待为 **85.038--191.673 ms/rank**，权重刷新为
**33.947--54.704 ms/rank**。梯度回传的 pre-wait 几乎为零，权重刷新有约 60--77 ms/rank 的
pre-wait 窗口。因此下一项优先研究 **FP32 梯度回传的合批、依赖和等待位置**；暂不扩大 shared-B
或增加 weight ping-pong pool。

这不是已经实现的加速，也不是可直接兑现的 192 ms 收益。事件跨度含 enqueue / instrumentation
开销，pre-wait 不等于有效计算重叠，跨 rank 的局部等待不能相加作为 step 节省量。

修改仅在 example、UT 和文档中：

- `benchmark_mega_moe_layers.py` 新增默认关闭的 `--profile-p2p`；native variant 明确拒绝该选项。
- `mega_moe_p2p_profile.py` 在所有性能/HBM 点测完后，额外执行 warm F+B + AdamW、barrier/sync、
  一步被观察的 F+B、sync/read events、AdamW，然后由八个 rank 汇总 metadata。
- 临时观察 `_prefetch_transfers`、`_gradient_transfers`、`_launch_transfers` 和 `_wait_handles`，
  保留原通信对象、调用与返回值；正常完成、调用失败和部分安装失败均恢复原函数。
- 只保留 peer/shape/dtype/bytes 等 host metadata、stream 和 events，不保留通信 tensor 引用。
  空 batch/cache-hit wait 不建 event，未完成或无法分类的 batch 拒绝生成看似完整的结果。
- 默认路径不创建 observer / events，不改生产 runtime、native kernels、FP32 聚合、参数或
  optimizer ownership，也不新增持久 device pool。开诊断的额外步骤本身不是零开销。

## 2. 测量协议与计时语义

沿用前一轮 [修复验收](megamoe_sun_fix_report_20260907.md) 的 matched-initialization helper：
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

## 3. 偏斜路由：payload 和暴露等待

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

## 4. 当前时间线交叉检查

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

## 5. 默认路径、性能与 HBM 隔离验收

平衡 capture 在八卡上均为零 builder call、零非空 batch、零 payload，且逐 rank/peer 校验通过。
该进程先测平衡、再测偏斜，最后返回平衡做诊断，故覆盖了“已激活过 B 池之后的 home-only”。
默认关闭运行的 JSON 不含 `diagnostic_p2p`。

以下均为 observer/profiler 启用**之前**的普通计时，单位 ms。S1/S2/default 先偏斜后平衡；
balanced 行的顺序相反，仅用于平衡诊断验收，不作为相同 allocator 历史的 A/B。

| 运行 | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| S1：之后另做偏斜诊断 | 1236.694 | 1367.805 | 672.989 | 805.355 |
| S2：之后另采 trace 和偏斜诊断 | 1232.126 | 1363.321 | 676.619 | 807.448 |
| balanced：之后另做平衡诊断 | 1234.182 | 1365.089 | 672.611 | 803.803 |
| default：无诊断步骤 | 1231.338 | 1363.402 | 671.863 | 802.985 |
| 上一轮正式构建 C1 | 1229.056 | 1360.044 | 670.697 | 801.564 |

default 相对 C1 的 full step 为偏斜 +0.247%、平衡 +0.177%。这是单次隔离验收，不是 ABBA，
不足以证明统计等价或亚百分比变化的原因。本轮没有 runtime 优化，不从这些波动声称新的加速。

S1/S2/default 及 C1 的逐 rank allocator/SHMEM ledger、shared pool、optimizer-state ledger
逐字节一致；不要求外部 runtime free/used 观测量一致。以下 rank-max 内存值均不变：

| 指标 | 偏斜 | 平衡 |
| --- | ---: | ---: |
| allocator peak allocated | 24539358208 | 24057401344 |
| conservative peak | 25411773440 | 24929816576 |
| allocator peak reserved | 25904021504 | 25904021504 |
| final allocated | 22150163968 | 22150163968 |
| configured SHMEM heap | 872415232 | 872415232 |

每 rank shared weight/FP32 gradient pool 保持 `55050240 / 110100480 bytes`。这些数据只验证普通
路径和诊断前的 HBM，没有把额外 event capture 的开销隐藏到“诊断也不占内存”的结论中。

四轮完整 configuration、八卡初始权重 hashes、路由/placement、生产 Python fingerprint 和
17 个 native artifacts 都匹配。生产 Python fingerprint 仍为
`17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f`；与 C1 相同。
新增诊断 helper 的 SHA256 为 `dabc7152e4ca3c6e366354c2ee4a563739c9a70a9e22a65495d0e98dd666cb52`。
测量进程退出前再次校验 source/native 未变；四轮全部通过 finite / optimizer-state 检查。

四轮串行执行，不与本轮 CPU 回归、native 编译或其他本轮 NPU 工作并行。每约十秒采设备进程表，
分别留存 10/13/10/11 份 census，仅观察到各轮的八个 worker；该采样不是硬件独占保证。

## 6. 测试与复现

按 `add-unit-test` 流程先写 CPU 合同测试，缺失 helper 时得到预期导入失败，再实现诊断。
最终相关回归为 **203 passed / 165 subtests，30.60 s**：

```bash
HYPER_PARALLEL_PLATFORM=torch OMP_NUM_THREADS=1 python -m pytest -q \
  tests/ut/core/multicore/mega_moe \
  tests/ut/core/multicore/test_mega_moe_module.py \
  tests/ut/examples/torch/multicore \
  tests/ut/native tests/ut/platform/torch/symmetric_memory
```

新增/补强覆盖 BF16/FP32 真字节数、phase/peer/dtype/shape/round 不匹配、空 batch 不分配 event、
weakref tensor 生命周期、默认不开诊断、全部测量结束后才采样、非零 rank 参与、sync 后才读取
event、部分安装失败恢复、launch/wait 失败和未完成 batch、非法 event 时长、跨 stream 拒绝误归因。
四个 Python 文件 pylint、三份 Markdown lint、catalog 与 `git diff --check` 通过。

本轮没有生产库改动，没有重跑上一阶段的全部 12 项 NPU 数值 ST；本轮四次真实八卡 benchmark
负责 finite、payload 协议和性能/HBM 隔离验收，**不是新增 native 数值 oracle 或长期收敛结论**。

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

## 7. 下一步的进入条件

1. 先定位 rank 4/7 梯度尾部的 peer 到达和批次依赖，再评估同 peer gate/up + down 合批，或将
   确认独立的 backward 收尾工作放到 completion wait 前。这些是实验方向，不是已证明的收益。
2. 必须保留 owner FP32 聚合及最后 cast 的顺序；shared-B dW 在发送完成前不能复用，owner receive
   在消费前必须完成。不能仅移动 Python wait 或放松 event 依赖制造表面重叠。
3. 优先不增加默认持久 HBM；若合批需要 packing copy/额外 storage，单独核算其时间与 live/reserved
   峰值。任何调度候选都须通过精度、共享生命周期和匹配 ABBA 后才能保留，再直接本地 commit。
4. 权重双缓冲、扩大 B、前向 peer 轮转仍后置；本轮证据没有推翻 multi-wave 的残余容量保障作用，
   也没有证明任意 routing、connected graph 或其他 topology 都有同样的梯度瓶颈。
