# MegaMoe 路由融合实验结果

本报告记录 2026-10-28 至 2026-10-29 本机的 permute/unpermute 融合实验，
包括真实 DSV MoE 组件、裁剪 DSV4.1 训练和 8 层纯 MoE 训练网络。
接口及存储约束见[路由融合使用说明](mega_moe_route_fusion.md)。

## 结论与配置选择

8 层纯 MoE 网络中，**pull + 前向部分融合 + 反向 legacy** 通过预设数值门槛，
在两组新进程重复中取得超过 1% 门槛的完整训练 step 收益。
补充同批对照里，前向 `both` 在关闭/开启 checkpoint 时分别降低耗时 **2.40%/3.01%**；
前向仅 `permute` 分别降低 **1.96%/2.38%**。
主批中，pull `both/legacy` 相对当前默认 push `permute/legacy` 分别降低 **1.71%/2.78%**。
这些都是延迟下降比例，下面保留各批独立的控制，避免混用不同批次的基线。

吞吐优先可用以下配置；当前验证形状要求静态 `T=4096`：

```yaml
dispatch_mode: pull
dynamic_tokens: false
forward_fusion_mode: both
backward_fusion_mode: legacy
```

不开 checkpoint 且更关注跨层保存张量的占用时，将 `forward_fusion_mode` 改为 `permute`。
相对前向 `both`，它少占 **286.52 MiB Torch allocated 峰值**，完整 step 增加约 1.10 ms，
仍比同批未融合控制快 1.96%。开启 checkpoint 后，两者 allocated 接近；
`both` 更快，`permute` 的 reserved 和物理 HBM 采样较低。

前向发送侧 `[T*K,H]` 激活在上述配置中均不物化。
反向融合配置在纯 MoE 多层梯度检查中超过预设误差门槛，未准入正式性能比较。
裁剪 DSV 的完整训练数值验收也尚未通过，不能把组件或纯 MoE 收益写成完整 DSV 训练收益。

## 源码与运行环境

| 项目 | 本次验证值 |
| --- | --- |
| master 移植基线 | `71881263897ebc553275d4168e4555d37d9661d9` |
| 最终实现及纯 MoE 受测提交 | `9ab534a2a9fa516d7880dce59884ae22e6894391` |
| DSV 验证分支提交 | `47a31145547976e1e12b0d173810657a348ee5d7` |
| 提供的 DSV source 原始提交 | `b6346859ccf0025461b174b7ae4d6b38ae455529` |
| NPU / EP | Ascend910_9382，本机 4 rank / EP4 |
| Torch / torch_npu | `2.9.1+cpu` / `2.9.1`，实际在 NPU 执行 |
| CANN | 归档运行路径为 `Ascend/cann-9.1.0` |
| Native | 冻结的 `native-pull-v6`，transport ABI **5**，70 个 payload 文件有哈希核对 |
| Profiler | 全部正式计时关闭 |

已检视 `megamoe-sun-261027` 和 `megamoe-perf-261027`，相关 61 个 native/task/scheduler 文件
属于共同的融合实现，没有整体合并旧分支。移植后增加 indexed pull，并修复快照及生命周期问题。
v7/v8 是 Python 和验证迭代编号，未把它们当作新的 native ABI。
纯 MoE 实验没有修改上述受测生产实现，G 是已有配置的补充比较。

## 纯 MoE 网络与验收方法

| 项目 | 设置 |
| --- | --- |
| 网络 | 8 层实际串联、独立权重的 MoE，每层 RMSNorm → router → expert FFN → residual |
| 形状 | 每 rank T4096、H5120、I2304；全局 E16、K6、EP4 |
| 数据类型 | BF16 专家权重和激活，FP32 可训练线性 router / Top-K softmax 概率 |
| RMSNorm / residual | 无可学习 scale，eps=1e-6；`x + FFN(rms_norm(x))/sqrt(8)` |
| 初始化 | 规范 CPU FP32 随机初始化再转 BF16；专家按层/rank 独立，router 初值跨 rank 一致 |
| 输入 / 目标 | 每 rank 捕获的真实 DSV MoE 输入；固定种子的合成回归目标 |
| 路由 | 每层每步根据实际 hidden states 重新选择；非固定路由重放 |
| 路由初始先验 | `0.5*log(捕获的全局 expert 频率/均值)` 初始化 bias，此后可训练 |
| Loss / 通信 | 全局 MSE；local loss 除 EP world size；专家反向通信，router 梯度 allreduce 求和 |
| 优化器 | SGD lr=0.01，momentum=0，foreach=False，无 FP32 master 或优化器状态 |
| 测量 | 每组独立新进程，8 次训练预热 + 60 次完整训练计时，每配置重复两组 |

这里包含最终 loss、全网络 backward、router 梯度通信和实际 SGD 更新，
不是把单层时间乘 8。它不含 attention、Engram、输出 head 或 FSDP，
优化器也不同于 DSV 的 Muon/AdamW，因此不代替 DSV 训练验收。
共享 workspace 先用无梯度整网前向预热容量，再执行训练；实际容量增长记录保留。
各模式使用相同 CPU NUMA 分区，框架导入前绑核，没有将它声明为 NPU 最近 NUMA 节点或显式 membind。

主指标逐步取 4 rank 的完整 step 耗时最大值，对 60 步作算术平均，再等权平均两组。
保留全部慢样本，未事后改用中位数。计时包含 zero_grad、整网 forward+loss、
backward+router allreduce、SGD 和阶段同步；loss、grad norm、路由签名的诊断读回在计时及峰值读取后进行。
阶段分解取同一步最慢 rank，以保证阶段和等于完整 step。

预先固定的门槛如下，失败后没有放宽：

- 输出 relative-L2 ≤ 1e-4；**每个**参数梯度和输入梯度 relative-L2 ≤ 1e-3。
- 全部层 router IDs 完全相同，且基线独立重复必须通过。
- 训练轨迹前 10 步 loss rtol=1e-4、grad norm rtol=1e-3；全部 68 步路由签名一致。
- 正式性能要求同批平均 step 耗时下降至少 1%。

### 补充批：前向 both 与仅 permute

顺序为 **A D G G D A**，checkpoint 关/开各独立 6 组。
模式写作“传输 / 前向 / 反向”。allocated/reserved 取全部 68 步的跨 rank/重复组最大值，
包括预热；JSON 另外保存仅 60 步计时窗口的 allocated 峰值。
所有显存单位为 MiB（2^20 bytes）。

| Checkpoint | 模式 | step 均值 ms | 相对本批 A 耗时下降 | Torch allocated 峰值 | Torch reserved 峰值 | SHMEM heap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 关 | A push / legacy / legacy | 244.963 | 0.000% | 11274.323 | 12754 | 616 |
| 关 | D pull / both / legacy | 239.072 | 2.405% | 11665.771 | 14138 | 482 |
| 关 | G pull / permute / legacy | 240.169 | 1.957% | 11379.250 | 12958 | 482 |
| 开 | A push / legacy / legacy | 338.945 | 0.000% | 5938.327 | 7808 | 616 |
| 开 | D pull / both / legacy | 328.759 | 3.005% | 6016.758 | 7804 | 482 |
| 开 | G pull / permute / legacy | 330.895 | 2.375% | 6017.303 | 7588 | 482 |

两次重复的 step 均值范围：checkpoint 关时 A 244.508–245.418、D 238.903–239.242、
G 239.740–240.599 ms；checkpoint 开时 A 338.897–338.993、D 327.613–329.905、
G 330.698–331.091 ms。
两批共 12 行的未舍入指标保存在[结果 CSV](benchmarks/mega_moe_route_fusion_20261029.csv)，
`batch` 区分控制来源。CSV 中 A 的 `speed_gate_passed=false` 表示它是参照，未声称自身达到加速门槛。

### 主批：与当前默认 push 比较

顺序为 **A B D D B A**，checkpoint 关/开各独立 6 组。
B 为当前默认静态 push 的前向 `permute`、反向 `legacy`；不是未融合 A。

| Checkpoint | A push legacy/legacy ms | B push permute/legacy ms | D pull both/legacy ms | D 相对 B 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 关 | 244.407 | 243.793 | 239.618 | 1.712% |
| 开 | 337.807 | 338.675 | 329.261 | 2.780% |

B 相对 A 的变化为 +0.251% / −0.257%，未过 1% 门槛；
D 相对 A 为 +1.959% / +2.530%。补充批没有 B，不能跨批计算 G 相对 B 的加速。

### 显存归因与物理 HBM 边界

不开 checkpoint 时，D 的多层前向状态同时存活。
前向 unpermute 保存 `narrow(T*K)` 专家输出时，视图仍钉住更大的 down-projection 底层接收分配。
峰值对应 rank/step 的前 7 层额外保存约 **285.32 MiB**；G 的独立输出快照精确为 T*K 行，
额外行归零，D→G 同点 allocated 差 **286.52 MiB**。
`snapshot-cause.json` 保存每层行数、底层 storage 和峰值对应关系。
已有 v8 压缩逻辑针对“前向 unpermute + 反向融合 permute”；D 的反向为 legacy，
该条件未触发。G 用已有模式避开了多层累积，未增加生产代码。

G 相对未融合 A 的 allocated 仍多 104.93 MiB，但 SHMEM heap 固定少 134 MiB。
本批每 rank/每步 heap 均恒定，可以列出“allocated 峰值 + 固定 heap”的已跟踪分账：

| Checkpoint | A MiB | D MiB | G MiB | G 相对 A |
| --- | ---: | ---: | ---: | ---: |
| 关 | 11890.323 | 12147.771 | 11861.250 | 少 29.073 MiB |
| 开 | 6554.327 | 6498.758 | 6499.303 | 少 55.024 MiB |

这两项之和没有包含 Torch cache、CANN/HCCL 等全部物理占用，不能称作物理 HBM 峰值。
开启 checkpoint 后，多层快照由重算取代，D/G 的 allocated 接近，reserved 排序仍不同。

约每秒一次的物理 HBM 采样如下，每格保留两组最大样本值，单位 MiB：

| Checkpoint | A push legacy/legacy | D pull both/legacy | G pull permute/legacy |
| --- | --- | --- | --- |
| 关 | 17183, 17327 | 18138, 18138 | 17253, 17254 |
| 开 | 11942, 11940 | 11921, 11922 | 11813, 11813 |

关闭 checkpoint 时，G 相对 D 的采样水位降低，但与 A 区间重叠，
**未验收 G 相对 A 的物理峰值下降**。开启 checkpoint 时，G 的采样峰值比 A 低约 127–129 MiB。
采样可能漏掉瞬时峰值，此处只报告样本指标。

### 数值准入与未通过配置

首次数值矩阵每 rank 14 案例：两种 checkpoint 各比较 A、A 重复、B、C、D、E、F。
其中 C=push both/both，E=pull both/permute，F=pull both/both。
A 重复、B、D 的输出、输入梯度和所有层参数梯度均与 A **完全一致**。
C/E/F 的输出和所有 router IDs 一致，但输入梯度 relative-L2 为 **1.61e-3–1.67e-3**，
超过预设 1e-3，因此该矩阵明确 exit=1；监控正常结束，不将其记作数值通过。

最后一层专家 DW1/DW2 仍完全一致，最后一层 router 梯度差约 7e-8 relative-L2，
前面层出现逐渐放大的 BF16 梯度差异。
独立诊断中，外置加权 dY 与 FP32 乘法后舍入 BF16 逐位一致，排除了概率预先降为 BF16 的解释；
外置 dprob 与 FP64 点积参考的 relative-L2 约 7.77e-8。
这是多层数值边界和误差传播观察，未据此声称已完成全部误差根因定位。

`formal-admission.json` 在正式计时前只准入 A/B/D，保留 C/E/F 失败数据。
G 另做每 rank 6 案例的 A、A 重复、G 数值检查，输出及所有梯度均为 0 差异。
两批获准训练配置的全部 68 步路由签名相同，前 10 步 loss/grad norm 相对差为 0。

## 发送侧 Top-K 向量不进入 HBM

获准的 B/D/G 每次前向均只排序标量 route-slot，`PreparedTopKRoute.routed_tokens=None`，
没有发送侧 `[T*K,H]` permute 激活。当前 T4096/H5120/K6/BF16 对应每个张量 **240 MiB**；
未融合 A 会实际物化该张量，作为对照。pull 的两列 int32 路由元数据约 192 KiB。
Push 从原始 token 读入 UB 后发送；pull 的 indexed GET 从原始 token 读入 UB 后写专家接收区。

这些方案的反向仍为 legacy，加权 dY 仍有 T*K 行，pull 的对称 source 也为反向保留 T*K 行。
专家接收区、FFN 激活及反向快照仍占 HBM，不能把前向发送向量消除推广成所有 Top-K 存储消失。
两方向均融合 permute 才能把 source 降为 T 行；本形状为 240→40 MiB，
但相应反向融合配置未通过上述多层梯度门槛，当前不推荐用于该训练网络。

## DSV 组件与裁剪整网：收益边界

裁剪 DSV 使用提供分支的 8 层、相同 E16/EP4/T4096/H5120/I2304/K6、BF16，
full checkpoint、attention recompute=all、host Engram、其他 Mega/fused attention 开启、
原有 Muon/AdamW，规范 FP32 seed42 初始化。
整网每组 30 步，前 10 步预热，统计后 20 步 rank-max 算术均值；
每配置两组新进程，顺序 A B C D D C B A。全部慢样本保留。

真实 DSV 输入的 MoE 前后向组件：每组 12 次预热、60 次计时，
取 rank-max 中位数，再等权平均两组。以下不同批次分别使用自己的 push 控制：

| 批次 / 模式 | 前后向 ms | 本批相对 push 耗时下降 | Torch allocated 峰值 MiB | SHMEM 请求峰值 MiB |
| --- | ---: | ---: | ---: | ---: |
| v7 push permute/legacy | 25.101686 | — | 1913.410 | 540.009 |
| v7 pull both/legacy | 24.262408 | 3.344% | 1942.210 | 480.197 |
| v7 pull both/permute，快照压缩前 | 24.286476 | 3.248% | 2210.229 | 280.197 |
| v8 push permute/legacy | 25.062388 | — | 1913.410 | 540.009 |
| v8 pull both/permute，快照压缩后 | 24.485201 | 2.303% | 2182.211 | 280.197 |

v7 顺序 ABCCBA，v8 顺序 ABBA。独立组件网格检查输出及全部梯度，
最大 relative-L2 1.852e-7，通过组件预设 1e-4 门槛。
新进程单案例的 JSON 梯度误差字段比较自身，不能作为独立跨模式梯度证据。
纯 MoE 的多层检查更严格地揭示了反向融合边界。

裁剪 DSV v7 整网结果只作为诊断：

| 配置 | 平均 step s | 相对默认 push 耗时下降 | Torch allocated 峰值 MiB | SHMEM heap MiB |
| --- | ---: | ---: | ---: | ---: |
| push permute/legacy | 2.881417 | 0.000% | 30344.430 | 616 |
| push permute/legacy，预留 35328 行 | 2.886508 | −0.177% | 30343.473 | 586 |
| pull both/legacy | 2.870297 | 0.386% | 30438.667 | 482 |
| pull both/permute，快照压缩前 | 2.885535 | −0.143% | 30784.935 | 282 |

默认配置重复运行自身未通过原定前 10 步 loss rtol=1e-4 / grad norm rtol=1e-3，
且上述完整 step 差异均未达 1% 性能门槛。没有完成 DSV 完整训练数值或稳定性能验收。

容量和快照调整有以下可复核的收益及代价：

- push initial/growth 都设为 1.0 时，两次第 22 步均扩容 34944→35328 行，
  step 达 6.84/7.08 s，均值慢 7.04%。预留 `initial_capacity_factor=1.4375`、
  `capacity_growth_factor=1.0` 后，两组各 30 步没有扩容，heap 616→586 MiB；
  这是已观测路由负载的结果，新的容量峰值仍会触发 lossless 扩容。
- v8 在“前向 unpermute + 反向融合 permute + 接收行数大于 T*K + 需要反向”时
  保存精确大小的快照。同模式整网前序 D1/D2 与后续 E1/E2 的最大 allocated 峰值
  从 30784.935 降到 30683.981 MiB，少 **100.954 MiB**；heap 仍为 282 MiB。
  顺序不是 ABBA，不据此验收速度。reserved 波动，物理采样也不支持“精确少 101 MiB”结论。

融合反向虽然少 200 MiB source，却延长了 240 MiB 专家输出快照的寿命，
加上跨层底层 storage 存活、dprob 归约、同步和精确快照拷贝成本，不能仅按少几个算子预测峰值或耗时。
此外 DSV 组件约 25 ms、整步约 2.88 s，其他模块稀释组件收益。
checkpoint 改变张量寿命；该 DSV 的首次前向及重算均启用梯度，不能把 no-grad 快照省略计作它的训练收益。

## 证据归档与复算

仓内 CSV 保留两批纯 MoE 的 12 行未舍入汇总、两次重复范围、allocated/reserved、
SHMEM 请求量/heap 和门槛结果。逐 rank 记录、输入、运行脚本、完整 manifest 及日志仍位于本机归档，
未将大体积原始数据或设备专用二进制纳入 Git；本文和 CSV 不构成跨机器独立复现包。

| 本机归档根目录（`/home/ma-user/zfr/benchmark-results/` 下） | 关键文件 |
| --- | --- |
| `moe-only-route-fusion-20261029` | `plan.json`、`formal-admission.json`、`plan-g.json`、`formal-summary.json`、`formal-g-summary.json`、`results.csv` |
| 同上 | `snapshot-cause.json`、`weighted-grad-diagnostic.json`、`worker.py`、`worker-g.py`、`analyze.py`、`run_case_g.py` |
| 同上 | `source-manifest*.json`、`harness-manifest*.json`、`CHECKSUMS.sha256`、`final-audit.json`、各组 `ranks/` 和 `exit.json` |
| `dsv41-route-push-pull-20261028` | `REPORT.md`、`analysis/formal-v7-summary.json`、`analysis/memory-v8-summary.json`、`analysis/component-fresh-v7-summary.json`、`analysis/component-fresh-v8-summary.json` |

纯 MoE 两批共 24 组训练、96 个 rank worker、6528 条 rank-step，其中 5760 条计时记录。
训练及 monitor 全部 exit=0；首次数值矩阵因 C/E/F 超阈值 exit=1，另有 G 数值矩阵通过。
终态审计确认两组数值和全部训练的 104 个 worker PID 已退出，最终 8 张逻辑 NPU 无进程。
所提供 DSV source 的 HEAD、index tree 及 58 项暂存变更保持原样。
这些是实验归档时的终态记录，后续工作区或设备状态需要重新查询。

在保留了归档的本机重算两批数据：

```bash
cd /home/ma-user/zfr/benchmark-results/moe-only-route-fusion-20261029
python analyze.py formal
python analyze.py formal-g
```

重跑单个 G 配置的原始入口如下；需要保留的输入、配套 native、环境及队列脚本。
运行前使用未占用的 rendezvous 端口，并将 `repro-unique` 改成新组名；已有目录拒绝覆盖。
完整镜像对照仍须按 `plan-g.json` 分别执行关/开 checkpoint 的 A D G G D A，不能用单组重跑替代。

```bash
TASK_MOE_ROOT=/home/ma-user/zfr/benchmark-results/moe-only-route-fusion-20261029
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
NPU_WAIT_VISIBLE_DEVICES=0,1,2,3 \
NPU_WAIT_LOCK_WAIT_SECONDS=-1 NPU_WAIT_FINAL_DELAY_SECONDS=20 \
flock /tmp/npu-wait-and-run.lock \
  bash /home/ma-user/zfr/hyper-tmp/scripts/npu_wait_and_run.sh \
  python "$TASK_MOE_ROOT/run_case_g.py" \
  --group repro-unique --name cp0-G --case G --checkpoint 0 \
  --steps 60 --warmup 8 --port 21000
```
