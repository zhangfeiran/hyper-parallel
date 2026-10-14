# DeepSeek-V4.1 decoder 重计算与 Host Engram

当前集群比较入口为 `scripts/run_deepseek_v41_cluster.sh`。一次作业按六个 case 顺序执行：
未 packing 的 native → push → pull，再执行 pack 到接近 4096 的 native → push → pull。
同组使用同一随机初始化、数据顺序、模型规模和专家副本预算。

## 启用方式

集群启动器默认：

```bash
ACTIVATION_CHECKPOINT=full
ENGRAM_STORAGE_BACKEND=host
```

`full` 在 benchmark recipe 中设置
`activation_checkpoint.selection.source=model_adapter_safe_regions`，并根据实际语言层数
设置 `layer_count`。每个 decoder layer 是一个重计算 region；模型策略选择
`ATTENTION_RECOMPUTE_TIER="all"`，包括 Full、Reindex 和 Reuse attention。
原 forward 发布跨层状态，backward replay 复用离散选择，避免重复发布。
视觉塔不包含在 decoder 重计算范围内。

对照时可用 `ENGRAM_STORAGE_BACKEND=device` 或 `ACTIVATION_CHECKPOINT=off`；每次使用新的
`RUN_ID`，以免覆盖已有结果。直接调用 Python benchmark 时，Host 由
`--engram-storage-backend=host` 选择；其默认仍为 device。

## Host 参数与优化器

Host Engram 保留原哈希行数、地址、FQN 及 EP owner 分区，只替换表的存储和查询路径。
表在 CPU 上保持 FP32，不参与普通 FSDP 参数聚合或 dense optimizer。查询仅将命中行传到设备；
生产构建把查询输出转换为 BF16，以匹配关闭输入自动转换时的 BF16 WKV 投影。

反向将 CPU 稀疏行梯度去重归并，并在同 owner 的副本之间同步。梯度裁剪同时计入 dense 和
Host 梯度，每个逻辑分片只计一次。Host 参数由 CPU SparseAdam 更新；其 lr、betas、eps
在集群 benchmark 中取自当前 dense AdamW 配置，默认分别为 `1e-5`、`[0.9, 0.999]`、`1e-8`。
SparseAdam 不对未命中行执行 dense AdamW 的完整更新或 weight decay，因此 Host/device
是两种训练方案，不能假定它们多步更新数值完全相同。native/push/pull 在同一方案内比较。

Host 随机初始化按完整表的全局行/元素坐标生成，沿用 benchmark 的 SplitMix64/Box–Muller
算法。CPU owner 初始化使用分块向量生成，避免逐行重置 RNG；benchmark 再按同一 seed42
初始化所有参数并刷新 dense FP32 masters。整个过程不生成权重文件。

通用 checkpoint callback 通过模型 external-state 接口保存和恢复 Host sidecar，包括 owner
权重和 SparseAdam 状态；普通 DCP 只管理 dense 部分。当前集群比较关闭 checkpoint 保存。
Host 路径要求 meta 构建、FP32 初始化/masters、BF16 forward、FP32 reduce，且不支持
torch.compile、异步 Host checkpoint 或 placement-validation 构建。

## 当前 128-die 配置

保留语言 4 层、视觉 1 层、hidden5120、MoE intermediate2304、E384、EP64、denseFSDP128、
expertFSDP2、TP/CP/PP1、MBS1、GBS128、B6，仍使用原始保留层 Engram。
Python benchmark 通过 `--packing-mode=dynamic|packed` 选择数据口径：

- `dynamic`：`token_budget=1`、`min_buffered_samples=1`，每次恰好一条完整源样本。
  budget 只用于选择，不截断样本，也不把模型最大长度改成 1。
- `packed`：`token_budget=4096`、`min_buffered_samples=16`，使用源图像与文本 first-fit packing。
  实际 token 数由样本组合决定，不保证每 rank 恰好 4096；记录实际 MoE 输入行数。

模型最大序列长度和 MegaMoE 最大 local token 上限在两组中都保持 4096。

对单个 `384006168 × 256` 表，FP32 weight 和两份 FP32 moment 在 EP64 下约为
17.17 GiB/rank，即每个 16-die 节点约 274.66 GiB CPU 内存。它没有按 denseFSDP128 再减半；
同 owner 的 EDP 副本各有一份 Host 状态。此估算不包括其他参数、梯度及缓冲。

## 验证范围

相关 CPU 回归覆盖跨层状态与离散选择重放、Host/device 查询和梯度、稀疏更新及恢复、
全局随机行坐标、dense master 刷新，以及两张 packed 图像的 4+1 小尺寸 VLM。
多进程 Gloo fixture 验证 EP owner、同 owner 副本归并、联合梯度裁剪和 checkpoint 恢复。

这些检查不等价于 128-die NPU 训练验收。首次集群运行建议每套至少 8 步，越过此前第 5 步
失败点，并检查全部 rank 完成、loss/gradnorm、CPU 内存和连续物理 HBM，再进行正式测量。


## 一次作业的执行与结果

集群脚本默认每个 case 独立跑 40 步，前 10 步 warmup，后 30 步计入测量。
每个 case 重新启动 torchrun，重新创建模型、Host 表、优化器及数据迭代器；seed 固定为 42。
六组 rendezvous、HCCL 和 SHMEM 端口各自隔离。每节点先发布退出状态，再清理本轮独立
process group；其他节点发现失败后停止自己本轮 worker，等全部节点清理完成才继续。
不会停止其他任务的进程。节点失联或清理失败时停止推进，避免不同 case 的 rank 混用。
如果任一 case 失败，跑完可继续的 case 后作业仍返回非零，不能把部分成功误报为全通过。

node0 每个 case 后原子更新 `summary.json`、`summary.tsv`、`moe_layers.tsv`，并更新
`dynamic/summary.json` 与 `packed/summary.json`。某个 backend 失败时，已完成的
native/push 或 native/pull 仍可形成单独对照；packed 失败也不会删除 dynamic 的结果。
缺失或损坏的 HBM monitor 记录仅使 HBM/性能证据不合格，不覆盖训练退出状态。
全部 rank 的 `result.json`、`steps.jsonl`、初始化指纹和节点退出状态都参与完整性检查。

`MOE_TIMING=1` 默认开启，使用 NPU current-stream Event，记录每个 MoE 层的：

- `forward`：首次前向，包含 gate/router、routed/shared experts 和副本路径。
- `recompute_forward`：checkpoint 实际重放的前向区间，保留 early-stop 的调用标记。
- `backward_to_input_grad`：输出梯度就绪到输入梯度就绪的区间，可能包含嵌套重算。

没有在每层同步设备；在原有整步同步后解析 events，不保留激活或计算图。
反向区间不保证包含所有异步通信或 weight-gradient 尾部，三个阶段不能相加解释成整网 wall time。
MoE 表先在同一 rank 内按层求和，再取该步 rank-max，最后对测量步计算 mean/median；
逐层表也保留各层独立的 rank-max。它们都是诊断口径，整网加速比始终使用同步训练步 wall time：
`mean(native step rank-max) / mean(push or pull step rank-max)`。
MoE 各阶段分别提供 current-stream span 加速比，不能替代整网加速比。
同组检查初始化指纹、实际 token 行数、loss/gradnorm 差异与进程/HBM 采样质量；
有限 loss/gradnorm 和差异报告不等于逐张量精度验收。

结果目录中，`dynamic/native/rank0/steps.jsonl` 保存 rank0 逐步明细，
`dynamic/nodes/node_0/native/` 保存节点日志、进程/HBM 采样和退出状态；其他 case 结构相同。
`nodes/node_0/launcher.log` 保存整个六组启动顺序。
