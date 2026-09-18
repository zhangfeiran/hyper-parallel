# DSV4.1 EP8 / 全局 E48 整网内 MoE 计时

## 配置与测量范围

按用户指定统一为 EP8、全局 48 experts、每卡 6 experts。整网保持四层、
H=5120、expert I=2304、TopK=6、每卡 T4096、shared expert=1，
BF16 参数 + FP32 主参数、Muon/AdamW、routed/shared SwiGLU limit=0。
attention/Sinkhorn/mHC-post 为生产融合路径，mHC pre 沿用分阶段 Torch 实现。
TP/CP/PP 均为 1，没有 activation checkpoint、compile 或容量丢 token。
使用同一组 canonical 初始权重和确定性合成 token，不涉及预训练模型收敛。

对照仍为当前 DSV4.1 owner A2A EP 的逐 expert Torch matmul，未启用 grouped GEMM。
本轮直接在整网内测四层 MoE，接收真实上游激活并执行真实 backward。
按用户后续要求收敛范围，取消额外的独立单层性能测试，不再展开其他模块的算子归因。

原来的 15.24× 来自 EP4/E192，即每卡 48 experts；这轮每卡只有 6 experts。
总 assignment 数仍约 T×TopK/卡，平均每个 expert 的 token 数增加八倍。
因此原基线的逐 expert launch、切片梯度和矩阵乘形状均发生变化。

## 正常运行的完整训练 ABBA

每次新进程读取同一份权重，8 步预热、10 步计时，按各步最慢 rank 统计。
四轮占用记录均无外来/无法归属进程；以下结果没有开启诊断或 profiler。
全部 measured samples 保留，两个后端首步各 rank 的 BF16 loss 完全一致。

全部计时样本均值为 **5584.18 → 5391.46 ms/step**，
吞吐增加 **3.57%**，step 延迟下降 **3.45%**，净节省约 **192.72 ms/step**。
两轮中位数平均 5581.81 → 5387.09 ms，对应吞吐增加 3.61%，与均值结论接近。

## 整网内四层 MoE 的计时口径

`--diagnostics moe` 在预热后对实际模型的四个 `*.mlp` 注册前向和完整反向 hooks，
仅在各模块边界同步设备并记录 wall time。覆盖 router、routed/shared experts、
EP dispatch/combine；反向以模块输入梯度就绪为结束边界，不包含 optimizer 更新。
它测量真实整网中的模块，不将独立 block 的时间乘四。

每个 rank 先累加本步四层的前向与反向时间，然后取 MoE 合计最慢的 rank，
保留该 rank 的前向/反向拆分；三个诊断步取平均。不会累加不同 rank 的局部最大值。
边界同步可能改变原有重叠，因此该加速比属于模块诊断，整网吞吐仍采用上面的正常 ABBA。

## MoE 结果

诊断运行复用 canonical 权重、输入、8 步预热和 18 步学习率计划。
owner EP 和 MegaMoe 分别取无外来/无法归属进程的三个诊断步。

| 范围 | owner EP ms | MegaMoe ms | 加速 |
| --- | ---: | ---: | ---: |
| 四层 MoE forward | 313.46 | 228.68 | **1.37x** |
| 四层 MoE backward | 195.97 | 87.18 | **2.25x** |
| 四层 MoE forward + backward | 509.43 | 315.86 | **1.61x** |

MoE 模块节省约 **193.57 ms/step**，与无诊断的完整训练 ABBA 节省
**192.72 ms/step** 接近。当前 EP8/E48 crop 的端到端收益基本由 MoE 前反向加速解释；
剩余模块没有获得 MegaMoe 加速。该结论限于每卡 6 experts，不能套用此前每卡
48 experts 的 15.24x 单层结果。

## 复现与证据

沿用 [单机裁剪性能报告](deepseek_v41_performance.md) 的环境、官方配置资产和 native payload。
正常训练入口使用 `--experts 48 --tokens 4096 --warmup 8 --steps 10`，
依次以 owner_ep / megamoe / megamoe / owner_ep 启动四个独立进程组。
整网内 MoE 计时追加 `--schedule-steps 18 --diagnostics moe --steps 3`。

本地证据：`/home/feiran/doc/dsv41-megamoe-ep8-e48-20260918`。

- `trainer-summary.json`：无 profiler 的完整训练 ABBA 与全部样本。
- `moe-clean-combined-summary.json`：两侧干净样本的整网四层 MoE 前向/反向汇总。
- `moe-owner_ep/`、`moe-busy3-megamoe/`：逐 rank 原始模块计时。
- `*-ownership.json`：各进程组从启动到退出的设备进程归属记录。
- `source-environment.json`：源码、native 哈希和学习率计划选项的版本边界；
  `benchmark-before-schedule-option.py` 保留增加该可选参数前的 benchmark 源码。

本轮只增加 opt-in benchmark 诊断，没有修改模型、kernel、通用 optimizer 或 allocator。
