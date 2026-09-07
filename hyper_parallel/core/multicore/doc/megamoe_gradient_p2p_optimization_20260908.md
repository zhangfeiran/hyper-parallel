# MegaMoe 梯度 P2P 优化：否决整专家合包

## 1. 结论

本轮从 [P2P 归因报告](megamoe_p2p_attribution_20260907.md) 的第一优先级出发，在
`megamoe-sun@5a2ae4b0` 上实现并验证 B=1 shared-gradient 的零拷贝整专家合包。
实验于 2026-09-07 开始，2026-09-08 完成取舍；原始文件保留开始日期。

**候选已撤回，不进入生产路径。** 虽然精度、存储别名和通信字节均通过验收，隔离通信实验也
小幅变快，但完整 16 层训练 ABBA 的偏斜 full step 平均回退 **1.806%**。平衡点基本不变，
并留下更高的 allocator reserve。因此这轮提交只保存实验补丁、证据和后续优先级。

当前代码恢复原来的两段梯度协议；不是增加一个默认关闭却未经收益验证的 runtime 开关。
本轮没有修改 native kernel、planner、默认 capacity、FP32 聚合精度或 dX ownership 修复。
仍有后续实验方向，不能将这一候选失败解释为已达到优化上限。

## 2. 候选设计与精度边界

原版每个 replica expert 向 owner 发送两个 FP32 tensor：gate/up 为 `[5120,3584]`，down 为
`[1792,5120]`。候选将 shared-B 的两组梯度放入一块等大 flat allocation，以两个不重叠、连续
view 提供原 native 指针。B=1 时，发送端用 view 将它们作为一个 105 MiB payload 发送；owner
接收到一个 flat buffer 后，通过两个 view 按原先顺序完成 FP32 累加。

关键约束及验证结果：

- 不重绑 home Parameter，也不改变 optimizer ownership；shared pool 的逻辑字节数不增加。
- 合包协议只取决于 shared/B=1/FP32 合同，不依赖各 rank 是否恰好命中某种存储别名。
- manager 自己准备的梯度无需 packing copy；外部非邻接/非连续输入需要物化后发送，以保持
  相同 wire protocol。B>1、低精度和非 shared physical gradients 保持原协议。
- 原 `handle.wait()` 位置和 owner 的 FP32 累加次序均未改变；没有提前转为 BF16，也不丢贡献。
- 两个 gradient view 共用 storage 后，ST 的内存账本改为按 device/storage 去重，不能把一块池
  算两次。这项测试改动也随候选撤回；生产 benchmark 原本已按不重叠 tensor 元素字节统计池。

额外的真实 NPU view 检查确认：105 MiB flat、W1、W2 和最终发送 view 的 base storage 一致；
发送 view 的 data pointer 等于 flat 起点，W2 offset 为 18350080 个 FP32 元素。
`_pack_gradient_pair` 调用前后 allocated delta 为 0。故没有将一个实际发生 packing copy 的
方案误称为零拷贝；但这不证明更大的消息会改善训练中的通信进度。

## 3. 隔离传输实验：只作为进入条件

使用前一轮真实偏斜路由的七条 replica/owner 关系，比较原两段描述符与一段整专家描述符。
两个 variant 都复用相同的 flat buffers，传输内容逐 peer 验证，显存分配相同；这一步只隔离
描述符粒度，不模拟原版的两块 storage，也不包含 expert 计算或真实的跨 rank 到达差异。

每样本连续执行 16 个 batch，3 warmup + 7 measured，取 rank-max event 时间的中位数，单位 ms：

| 运行 | split A1 | packed B1 | packed B2 | split A2 |
| --- | ---: | ---: | ---: | ---: |
| 16 batch median | 118.101 | 110.891 | 112.326 | 113.603 |

A/B 各两次中位数的均值为 `115.852 → 111.608 ms`，降低 `3.663%`。
它仅支持继续进行完整训练实验，不能当作模型的加速结论。

## 4. 完整训练 A/B/B/A

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

## 5. 等待与 HBM：为什么不保留

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

## 6. 正确性与撤回验收

- CPU TDD：候选实现前为 3 个预期失败；实现后 **207 passed / 165 subtests，32.04 s**。
  覆盖零拷贝别名、不重叠矩阵 view、外部 layout、owner wait/FP32 累加、B>1 和低精度兼容。
- 五项关键 NPU launcher **5 passed / 18 deselected，197.24 s**：shared lifecycle、level0、
  shared resources、connected shared gradients、replica residual multi-wave；六个已有 marker warning。
- 另跑八卡 shared lifecycle 的 B1/B2，均为 21 组对照，实际覆盖 `checkpoint_modes=[false,true]`，
  包含 W1、残余 W2/W3、路由切换、参数更新和 control stream。两者 FP32 gradient pool 分别为
  `786432 / 1572864 bytes/rank`，没有放宽原容差。
- EP2、三层、T=2048/H=512/I=128/E8/K4、capacity=1.0、四步 AdamW 的跨版本对照：
  **312/312 张量逐位一致、全部 finite**，包含 output/dX/route dW/expert dW、参数和 optimizer state。
  两个 rank 的 `.pt` 文件 hashes 也分别一致。

这些检查证明候选在覆盖范围内保持正确性，不证明长期收敛，也不改变性能不合格的取舍。
撤回后，生产 Python 和两个测试文件均逐字节恢复 `5a2ae4b0`；保留的补丁只是一份实验数据，
不会被 import 或影响默认路径。恢复后的同组 CPU 回归为 **203 passed / 165 subtests，30.60 s**。

## 7. 复现与原始证据

可复现补丁：[gradient_pair_packing_20260907.patch](experiments/gradient_pair_packing_20260907.patch)。
基线为 `5a2ae4b0`，仅应在隔离 worktree 应用；零上下文格式避免存储补丁时引入空白格式问题：

```bash
git apply --check --unidiff-zero gradient_pair_packing_20260907.patch
git apply --unidiff-zero gradient_pair_packing_20260907.patch
```

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

## 8. 优先级复审

1. 不继续扩大整专家消息，也不将隔离通信的更快直接外推到训练。下一轮先验证较小改动：在保留
   完成等待与 storage 复用约束的前提下，提前提交梯度通信，将独立 dX/route 收尾放到 wait 前。
2. 若独立收尾的重叠窗口不足，再补 rank 4/5/7 的共同时间线，确定 peer 到达、排队与传输粒度的
   关系；不直接用 payload 表轮转 peer，不假定“大消息更省 API 就更快”。
3. FP32 dW 生命周期、控制面合并和前向 expert readiness 仍有待验证空间。跨层权重双缓冲继续后置，
   不因本次失败无条件增加池或降低精度；每轮仍须验证、直接本地 commit、再复审优先级。
