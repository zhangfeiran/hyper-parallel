# MegaMoe 梯度回传：全 rank 就绪时间线

## 1. 结论与下一实验

在 [独立收尾重叠未产生确认收益](megamoe_gradient_epilogue_overlap_20260908.md) 后，本轮补齐八卡
共同时间线。生产 runtime 仍为 `megamoe-sun@7e31c518`，未修改 kernel、planner 或梯度协议。

两轮 capture 均显示：**只接收梯度的 rank 5 是全部 16 次 backward 中最晚开放通信输入的 rank**。
每层所有 rank 的 ready spread 均值为 `5.541 / 5.522 ms`。rank 4/7 的累计等待中，分别有约
`88 / 82 ms` 的窗口与邻居较晚就绪重合。rank 5 的本地 native backward 本身较长，receive-only
batch 仍把 caller stream 的完成事件当作输入依赖，因而把独立接收排在本地计算之后。

这不是已经实现的加速，也不证明全部重合窗口都能节省。它把下一候选收敛为：**先验证
receive-only 梯度批次的独立接收**，保留 owner 计算流上的完成等待及原 FP32 累加次序。
不泛化到含发送的 batch，不直接提前使用未完成的 replica dW，也不增加权重双 buffer。

rank 2 在邻居已经就绪后仍有约 `121 / 126 ms` 的累计等待，说明晚到达不是全部瓶颈。
这轮不能据此宣布优化上限，也不能把 `88 + 82 ms` 相加当作预计 step 收益。

## 2. 本轮落地的诊断能力

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

遵循 `add-unit-test` 的 TDD 流程，先验证默认行为、输出碰撞及异常恢复，再进入八卡采集。
分析器另覆盖 epoch 大时间戳的 20 ns 差值、跨 pid/tid 排除及不完整 workload 拒绝。

## 3. 方法、时钟与证据边界

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

## 4. 梯度依赖与关键数据

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

## 5. 默认路径与采集验收

采集前记录的性能/HBM 数值如下，仅用于复现与稳定性核对，不是新的优化 A/B：

| capture | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| S1 | 1233.970 | 1363.866 | 672.604 | 803.312 |
| S2 | 1233.539 | 1364.479 | 670.770 | 801.767 |

S1 启动时与本轮 CPU UT 收尾短暂重合，首份 census 额外见 PID `2444669` 的 116 MiB NPU context，
随后该进程正常退出；因此不把 S1 当作严格独占性能样本。S2 在确认设备无进程后重采，11 份
census 的 PID 并集恰好是八个 worker，无额外 PID；这仍不是排除所有瞬时竞争的硬件独占保证。
核心时间线模式在 S2 复现。两轮均保留原始 monitor 日志，不删除这项环境差异。

两轮相同的 rank-max HBM：偏斜 peak allocated `24539358208`、conservative peak `25411773440`；
平衡分别为 `24057401344 / 24929816576` bytes。两点 peak reserved 均 `25904021504`。
shared weight/gradient pool 仍为 `55050240 / 110100480`、heap 为 `872415232 bytes/rank`。
这些值与前一轮恢复后的生产代码一致，profile 窗口的临时资源不计入性能 HBM 验收。

新增采集合同 TDD 在旧代码上按预期失败；最终 CPU 回归为 **212 passed / 179 subtests，30.29 s**，
pylint 和两次实际八卡导出均通过。
最终分析器重新处理两份 capture，JSON 与最初分析结果逐字节相同；没有通过改公式改变结论。

## 6. 复现与原始文件

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

## 7. 完成后的优先级

1. **先验证 receive-only 梯度接收解耦。** 从 plan 确认本 rank 无发送，再把独立 receive buffer 的
   分配、通信和完成事件放到独立流。优先复用已有 control stream，不复制 B pool。owner 计算流
   必须在读取前等待，跨流消费须 `record_stream` 或等价生命周期保护，异常路径不得提前复用。
   若 allocation stream 改变实际 HBM，必须单独计量；不凭约 88 ms 的重合窗口提前接受候选。
2. 若该受限候选有效，再按新时间线复审含发送 batch；后者必须保留 native dW 的真实就绪依赖。
   先匹配精度、单/多 wave、换路由、checkpoint、AdamW 和 ABBA，不泛化删除 caller wait。
3. 就绪后的等待、rank 5 native backward 的具体 dX/dW 成本、allocator churn 和 control-plane
   合并仍有待验证；前向 readiness 和 connected 负载也未关闭。权重双 buffer 继续后置。
