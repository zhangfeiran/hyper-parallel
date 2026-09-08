# MegaMoe 梯度回传：独立收尾重叠的收益边界

## 1. 取舍

在 [整专家合包被否决](megamoe_gradient_p2p_optimization_20260908.md) 后，本轮按优先级验证了更小的
改动：保留两段 FP32 梯度消息，提前提交 replica-gradient P2P，将独立 dX/route 收尾移至 wait 前。

**候选未保留，生产代码和测试已恢复本轮起点 `59d4736f`。** 完整训练 ABBA 的偏斜 F+B 几乎
不变，full step 仅变化 −0.037%，不足以确认收益；平衡 full step 增加 0.642%。等待前窗口确实
扩大，但每 rank 整步只有约 0.68--1.33 ms，不能兑现原诊断中约 85--190 ms 的大量暴露等待。
偏斜峰值 allocated 还增加约 1 MiB、reserved 增加 6 MiB。因此仅保存实验补丁与反证。

这限制了“只重排 backward 独立收尾”的收益预期，不代表所有 gradient-return 优化已到上限。
下一优先项是高等待 rank 的共同时间线；不继续堆叠未经端到端验证的 callback、池或 runtime 开关。

## 2. 候选实现与安全边界

原 shared-B backward 顺序：native backward → dX/route 收尾 → 梯度 P2P → wait → owner FP32 累加。
候选顺序：native backward → 梯度 P2P → dX/route 收尾 → wait → owner FP32 累加。

- `ExpertReplicaManager.reduce_replica_gradients` 临时增加内部可选 `before_wait` callback；先 launch，
  再执行独立工作，最后在 `finally` 中等待 handles。空批次也执行 callback，输入校验仍先于 launch。
- shared autograd 调用直接将 native 的 FP32 home partials 和 B-slot partials 传给 manager；callback
  执行原 `_finish_backward`，保留 dX 的 owned clone、route cast 和非 fused 路径布局恢复。
- callback 不读 receive buffer，不修改 dW；owner 聚合在 wait 后执行，FP32 次序和最终 cast 不变。
- callback 报错时仍先等待通信，再由外层释放 shared-B lease/workspace，避免 unwind 提前复用发送池。
  这里不新增通信本身失败后的容错协议，仍沿用原 handle 错误传播。
- 非 shared 路径不改变顺序。未修改 planner、native ABI、home Parameter、optimizer、通信 payload
  或 shared-B allocation；也没有删掉重基后修复的 dX clone。

这是 Torch MegaMoe 内部实现实验，不是新的公共 Platform API；平台抽象和 MindSpore backend 未改动。
依照 `platform-dev` 的 stream/storage 约束和 `add-unit-test` 的 TDD 流程，先补等待、异常和别名
边界测试，再进入真机检查；没有为了重叠而返回未完成的梯度。

## 3. 正确性验收

- 新增顺序、空批次、参数校验、异常清理及 autograd ownership 测试在原实现上按预期失败。
  候选相关 CPU 回归 **207 passed / 171 subtests，23.67 s**；四个修改文件的 pylint 通过。
- 五项 NPU launcher **5 passed / 18 deselected，196.30 s**，保留六个已有 marker warning：
  shared lifecycle、shared resources、connected shared gradients、level0 和 replica residual multi-wave。
  这些 launcher 的主要 world size 为 2，不能替代八卡验证。
- 另跑真正八卡 shared lifecycle：B1/B2 均 **21 组通过**，`checkpoint_modes=[false,true]`，
  没有拒绝 repeated backward。B1 覆盖 W1/W2/W3，B2 覆盖 W1/W2，包含路由换绑和更新；native
  oracle 容差未放宽。小 shape 的 B1/B2 梯度池仍分别为 `786432 / 1572864 bytes/rank`。
- 重新运行跨版本精度 helper：EP2、三层、T2048/H512/I128/E8/K4、B1、capacity=1.0、四步 AdamW，
  混合 deferred graphs、路由切换和 reentrant checkpoint。两版 **312/312 张量逐位一致且 finite**，
  覆盖 output/dX/route dW/expert dW、参数和 optimizer state；两 rank 的 `.pt` hashes 分别相同：

```text
rank0 bf2910a27612bca0cd1d775d084bf8db42af9902eed4cb00614265ee3004eb59
rank1 9f26d2d46f7b94ad413ff5d9a3ebcd442e69481ed4236230fac01573e51a7f0a
```

撤回后，同组 CPU 回归为 **203 passed / 165 subtests，30.66 s**，四个 runtime/test 文件逐字节
恢复 `59d4736f`。以上是覆盖范围内的正确性证据，不是长期训练收敛证明。

## 4. 匹配 A/B/B/A

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

## 5. 等待窗口与显存

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

## 6. 复现证据

隔离实验补丁：[gradient_epilogue_overlap_20260908.patch](experiments/gradient_epilogue_overlap_20260908.patch)。
在隔离 worktree 的 `59d4736f` 或生产代码相同的 `5a2ae4b0` 上应用零上下文补丁：

```bash
git apply --check --unidiff-zero gradient_epilogue_overlap_20260908.patch
git apply --unidiff-zero gradient_epilogue_overlap_20260908.patch
```

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

## 7. 完成后的优先级复审

1. **优先补高等待 rank 的共同时间线。** 整专家合包已失败，独立收尾窗口也已量化；不继续在这
   两个局部策略上叠加开关。关联 rank 4/5/7 与其 peer 的 native backward 完成、P2P 到达、排队、
   SDMA 和 owner 累加，先区分可调度等待与必要传输，再决定消息粒度或批次调度实验。
2. **FP32 dW/receive 生命周期与 control plane 仍是候选。** 先用 allocator/trace 分别量化，不凭
   Python API 数量推断收益，不提前 BF16 累加，也不为缓存无条件增加 persistent HBM。
3. **前向 expert readiness、动态路由和真实 connected 负载继续保留。** 双 buffer 跨层权重预取仍
   后置；梯度等待尚未解决，不能把单个小窗口实验的失败解释为整个实现已达到优化上限。
