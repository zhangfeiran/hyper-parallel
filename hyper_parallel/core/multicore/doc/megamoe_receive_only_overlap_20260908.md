# MegaMoe receive-only 梯度接收解耦实验

**结论：保留候选。** 排查进程竞争后重新完成 A/B/B/A，偏斜 full step 减少 **4.086%**，
F+B 减少 **4.490%**；平衡 full step 变化 -0.112%，视为基本不变。312 个张量逐位一致，
原生 MoE 和实际 EP8 B1/B2 回归通过。峰值 allocated 增加约 1 MiB、reserved 增加 36 MiB，
均只发生在 rank 5；没有新增 persistent B pool。测量范围仍是 16 个独立 Qwen-shape MoE 层。

## 1. 候选与依赖边界

本轮接续 [八卡梯度就绪诊断](megamoe_gradient_readiness_20260908.md)：固定偏斜路由中，rank 5
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

## 2. 正确性验收

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

## 3. 性能比较方法

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

## 4. 接收提前与剩余瓶颈

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

## 5. 进程竞争与首轮数据边界

首轮 A1/B1/B2/A2 分别保留 9/9/10/9 次 census。B1 在 `01:40:19/01:40:29` 看到了额外的
`akg_benchmark` 进程，其中一个占用约 43 GiB，无法精确排除与计时窗口重合；因此首轮完整
ABBA **不作为干净的最终性能验收**。其余三轮 census 未出现额外 PID；B2 trace 的八个 PID 与
benchmark worker 一致。原始数据及这一差异全部保留，未用删除异常样本的方式美化结果。

首轮偏斜 full-step 均值为 `1361.471 → 1307.190 ms`（-3.987%），F+B 为
`1230.860 → 1176.484 ms`（-4.418%）；平衡 full step 为 `803.945 → 803.934 ms`。
这些值是带上述环境限定的初步证据；随后另起 fresh processes 重跑完整 A/B/B/A。

## 6. 重跑的最终性能与 HBM 验收

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

## 7. 下一步优先级

1. 先补 Qwen **整模型** shared-B 接入与 matched 训练验收。当前收益证据是独立 MoE 层序列，
   原有整模型 example 尚未显式传入 `shared_physical_weights`，不能宣称完整 Qwen 训练也快 4%。
2. 在真实连接负载上重查剩余梯度等待，尤其 rank 1 的发送依赖及 rank 3 的 native 计算。
   不把 mixed batch 拆成可能产生 HCCL 顺序环的独立收发，不提前读取 native 尚未完成的 dW。
3. 再按 allocator snapshot 解释 36 MiB reserve 和更大的 FP32 dW 生命周期；保留当前正确的
   `record_stream`，不能以移除跨流保护或新增更大的常驻池换取表面峰值改善。
4. 前向 expert readiness、dX/dW 分项计数、control-plane 合并仍有空间；权重双 buffer 继续后置。
   本轮尚未达到优化上限。

## 8. 复现文件

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
