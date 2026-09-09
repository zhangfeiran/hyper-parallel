# Qwen expert dW：固定实际层输入的切分与舍入诊断

本轮接续 [多步数值诊断](megamoe_qwen_multistep_diagnosis_20260908.md) 的最高优先级，
拆分 native grouped dW、MegaMoe 单波/多波，以及 B1 physical-copy 聚合的影响。
基于 `caf71330`，生产 Python / native 相对 `ee5a6965` 未变，沿用上一轮重新构建的 payload。
本轮不修改 runtime、optimizer、学习率或精度阈值，也不发布性能数据。

结论：本次真实层复放中，B0 单波与 native 的输出、dX、dW 全部逐位一致；B0 多波差异仅在
wave 切开的 expert，B1 差异仅在复制过的 expert。B1 的实际 FP32 owner 聚合和最终 cast 均与
CPU 重算逐位一致。证据支持切分改变累加/舍入路径，未发现本次 replica 聚合错误；
**这不是完整模型多步精度已修复的结论**。

## 1. 固定的不是合成随机梯度，而是真实首步的层边界

先运行原配置 common 完整八层模型的一次确定性 AdamW step：EP8、H5120 / I1792 / T4096 /
V32000、E48 / K8、Dirichlet alpha=0.1、route/model seed=42，LR=1e-3、BF16 model / FP32 main。
捕获第 8 层（索引 7）的 routed experts；固定路由由该层 seed=49 生成。

在首次 optimizer 更新之前保留：

- 八 rank 各自的真实 hidden、expert IDs、scores、counts，以及 expert 输出的上游梯度 dY。
- 本地 gate-up / down 权重及其梯度、expert 输出。
- native 两个 grouped GEMM 的实际输入、输出梯度、group list 和 grouped dW。

捕获运行的全量初始参数、输入及源码/native 指纹与上一轮 `first_deterministic_common`
完全相同；全量 logits 和目标层 expert 梯度的 SHA256 也一致。hook 不修改反向返回值。
目标 hidden 同时供 shared expert 使用，其输入 hook 得到的是两个分支之和，因此不将这个
捕获值冒充 routed-only dX；dX 对照另以独立 common 层复放为 reference。

随后每组只构建一个独立 expert 层，将上述权重、hidden、路由、dY 原样复制到设备，执行
一次 F+B。**层复放没有 optimizer，也不是完整训练计时**；它的目的是阻断不同训练轨迹
改变输入和上游梯度。四组顺序执行，各为独立八卡进程组：

| 层复放 | B | capacity factor | 实际全局 wave count | dW 持久输出 |
| --- | ---: | --- | ---: | --- |
| native common | 不使用 | 不使用 | 不适用 | BF16 |
| MegaMoe B0 单波 | 0 | None，最大 lossless capacity | 1 | BF16 |
| MegaMoe B0 多波 | 0 | 1.5，49152 routed rows | 2 | BF16 |
| MegaMoe shared-B1 | 1 | 1.5，49152 routed rows | 1 | FP32，owner 聚合后 cast |

四组对应 rank 的七项复放输入（hidden / IDs / scores / counts / dY / W1 / W2）指纹一致，
source/native 前后、跨 rank、跨组一致。B 和 capacity 是显式控制变量，未将不同配置伪装成
原 full-model snapshot 合同相同。原 `rtol=0.02 / atol=0.002` 不变。

## 2. 全量 expert 梯度对照

native 独立复放与完整模型捕获的输出、两个 expert 梯度，以及两个 grouped GEMM 的输入和
输出梯度逐位一致。下表比较 MegaMoe 与 native；元素数跨八个 rank 求和，最大误差取全量最大值。
gate-up 共 880803840 个元素，down 共 440401920 个元素，不是抽样梯度比较。

| 对照 | gate-up 非逐位相同元素 | down 非逐位相同元素 | dW 最大绝对差 |
| --- | ---: | ---: | ---: |
| B0 单波 | 0 | 0 | 0 |
| B0 多波 | 13726123 | 6859924 | 4.7683716e-7 |
| shared-B1 | 26979 | 13587 | 4.7683716e-7 |

三组的 expert 输出及 routed-only dX 全量逐位一致。所有 dW 仍通过原阈值；B0 多波最大归一化误差为
0.000238321，B1 为 0.000238273。B1 的 gate-up / down 各有 3 个符号翻转，B0 多波没有；
不能用非逐位元素数量或单个最大误差替代实际 optimizer 敏感度分析。

本形状、真实层输入下，**不分割累加域时，自定义 GMM dW 可以和 native 完全一致**。
这不证明任意 shape / 路由都逐位一致，但排除了本次 residual 必须由两个 GMM 实现的固有
差异解释。B1 同时改变 copy 切分和 dW 输出精度，不能仅凭此表把差异减少全部归功于某一项。

## 3. 差异精确落在哪些 expert

B0 多波只在 expert **35、39** 的两个权重梯度中出现差异，它们分别有 13824 / 16272 个
全局 routed rows，位于 rank 5 / 6 的 receive 序列中，被 49152-row wave 边界切开。
不是所有有第二波的 rank/expert 都有差异，也不是所有热 expert 都有差异。

B1 的有效副本为 7 份，分别复制 expert **10、29、32、36**。只有这四个逻辑 expert 的 dW
非逐位一致，其余 44 个 expert 与 native 完全一致。具体布局如下，home 六个 slot 不变：

| replica rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-slot logical expert | 29 | 未用 | 32 | 36 | 10 | 10 | 29 | 32 |

B0 单波的 group list、SwiGLU 输出及反向传给 gate-up GEMM 的输出梯度，与 native 捕获全部逐位
一致。对 B1，按实际 `dispatch_counts`、source-rank 顺序及每个 logical expert 的 copy 配额，
将 native 相应行重组为 physical-slot 顺序；55 个有效 slot（含零 token 的 home slot）的
SwiGLU 输出及反向传给 gate-up GEMM 的输出梯度，共 110 项全量比较全部逐位一致。
这里直接捕获的是这两种中间张量，不宣称读取了内核内部每一条 dX/dW 输入边。

进一步截取 B1 native 返回后的 home FP32 dW 和 B-slot FP32 dW，在 CPU 上按生产
`_gradient_transfer_groups(plan, 8)` 的真实贡献顺序重算：

1. home partial 加上所有属于该 owner 的 replica partial；与设备 post-reduce FP32 输出比较。
2. 将该 FP32 和转为 BF16；与最终 Parameter.grad 比较。

八 rank × 两个权重的上述两阶段，均为全量逐位一致。这排除了本次实测中 P2P contribution
丢失、owner 槽位错误、聚合顺序意外变化或额外提前 BF16 cast 的解释；不能外推成所有并发、
路由切换或多步生命周期都已重新验证。采样 hook 带同步和复制，不作为 overlap 性能验收。

## 4. FP64 点积与代码路径交叉检查

用捕获的实际 BF16 grouped 输入和输出梯度，在 CPU 上逐元素转 FP64 后执行乘积求和。
每个 expert / 权重选 16 个固定均匀位置，另从各候选与 native 不同的位置中最多均匀选
32 个，去重后共 **1917 个位置**。这是带差异位置偏置的有界诊断，不是随机抽样误差率，
也不是全量 FP64 GEMM oracle。

只统计其中候选与 native 不同的位置，比较两者到 FP64 点积的绝对距离：

| 候选 | 样本中候选不同于 native 的位置 | native 更近 | 候选更近 | 距离相同 |
| --- | ---: | ---: | ---: | ---: |
| B0 多波 | 152 | 152 | 0 | 0 |
| B1 | 256 | 122 | 134 | 0 |

B1 并不固定比 native 更差；其 256 个差异样本中，native / B1 到 FP64 点积的最大绝对距离
分别约 2.38443e-7 / 2.38394e-7。它们是非常接近的不同舍入结果，不能把 native 的每个 BF16
输出当成数学精确值，再为贴齐它而降低 replica partial 精度。

代码与上述定位一致：

- `modules/mega_moe/function.py::_allocate_backward_intermediates`：B0 分配 BF16 dW；B>0
  分配 FP32 dW。shared-B gradients 同样由 `replica.py::prepare_shared_gradients` 分配为 FP32。
- `ops/hyper_mega_moe_grad/op_kernel/worker_kernel.cpp::ExecuteSingleGroupMatmul`：每个物理
  expert / wave 用其实际 token 范围计算 dW；非 atomic 的后续 wave 写 scratch。
- 同文件的 dW accumulator：BF16 分支将两个已舍入的 partial 转 FP32 相加，再 cast 回
  BF16；FP32 分支直接相加。默认 B0 多波因而不能保留一个 expert 完整 token 归约的舍入路径。
- `replica.py::reduce_replica_gradients`：当前 B1 保留 FP32 home/copy contribution，完成
  owner 聚合后才由 autograd 返回到 BF16 参数梯度。

忽略各矩阵乘内部 FP32 分块细节，三者的主要区别可写为：

```text
单波 native / B0：BF16(完整 expert 的 FP32 点积)
B0 多波：         BF16(BF16(wave 0 点积) + BF16(wave 1 点积))
B1：              BF16(FP32(home 点积) + FP32(copy 1 点积) + ...)
```

即使开启确定性，不同 token 切分仍可能有不同的浮点累加顺序。以上验证将本层 B1 residual
定位到 copy 分割的点积 / 舍入路径，不能单凭此推导其余七层、更新后的输入或任意路由都只有
同一种误差来源。

## 5. 验证范围、原始数据与复现

完成一组原八层模型的一步真实 AdamW 捕获，以及四组独立层的 F+B，均为 EP8，40 份 rank
结果正常完成；所有被比较的输出、dX、dW 都有限。源码/native identity 沿用前一份报告，
未改生产代码，不新增结构性 UT，也没有为文档修改重复运行整套已有 CPU UT。

全部原始张量、JSON、日志、设备 PID 采样、临时脚本只保留在本地：

```text
/tmp/qwen-expert-dw.9pLDND/
  capture/result/rank{0..7}.{pt,json}
  common/result/rank{0..7}.{pt,json}
  b0_single/result/rank{0..7}.{pt,json}
  b0_multi/result/rank{0..7}.{pt,json}
  b1/result/rank{0..7}.{pt,json}
  capture_worker.py   worker.py   probe_util.py
  run_case.py         run_replays.py
  analyze.py          analysis.json          analysis_retry.log
```

每个 case 还保留 `launch.json`、`worker.log` 和 `census.jsonl`。`capture_worker.py` 是实际
捕获使用的版本；`worker.py` 是随后增加 MegaMoe 中间张量观察的复放版本，两者 hash 在
各自 rank JSON 中记录。没有通过修改生产源码插入 hook。

初次 CPU 汇总因临时脚本使用 FP32 `linspace` 选择大型张量末尾索引而越界，未生成有效
`analysis.json`；修正为 FP64 索引生成后重新完整汇总，失败日志另留 `analysis.log`。
它没有影响已完成的 NPU 捕获/复放，未修改模型或精度阈值。

本机已有 capture 的复放入口是 `run_replays.py`，分析入口为 `analyze.py`，均使用
`/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python`、`HYPER_PARALLEL_PLATFORM=torch` 和
`OMP_NUM_THREADS=1`。NPU 复放先 source CANN 和当前 payload 的 `set_env.bash`，如前一份报告。
脚本拒绝覆盖已有 case / 结果；重新实验须通过 `mktemp -d` 建新目录，复制同版本脚本及
所需 capture，不能直接覆盖本轮证据。只将本报告与索引更新入 git，JSON 不入仓。

## 6. 完成此步后的优先级

后续已完成实际 fused AdamW 的 main/moments/BF16 写回复放，见
[AdamW 放大诊断](megamoe_qwen_adamw_diagnosis_20260908.md)：等梯度的两种参数布局逐位一致，
B1 首步 738 个 BF16 参数差异与原完整模型该层写回一致；下一项转向下一次 forward 的
参数干预。以下保留本轮结束时的优先级，不把后续固定梯度复放当成完整多步训练验收。

1. **先拆 AdamW 的 FP32 更新与 BF16 参数写回。** 固定同一初始权重和本轮捕获的 expert
   梯度，比较真实首步 FP32 main、moments、更新量和 BF16 model 写回，特别是微小梯度及
   BF16 舍入边界。再扩展到其余层/后续 step，区分正常切分舍入被训练放大与实现错误。
   本轮不因单层 dW 通过而撤销原完整模型多步 logits/loss 的失败记录。
2. **保留 B1 的 FP32 partial / owner 聚合，不为逐位对齐 native 降精度。** 本层未发现
   聚合错误，没有理由重写已验证的 P2P 数值路径或重新引入提前 BF16 cast。B0 多波的
   FP32 累加可作为独立精度改善候选，但不直接解决 B1 residual，且需单列 HBM/性能代价。
3. **数值 gate 和性能 gate 分开。** 此轮没有稳态时间、HBM 峰值或旧/新代码配对测量，
   不能发布性能增减。默认模式的 attention / dense collective 非确定性仍沿用上轮结论；
   低 LR、确定性模式或 FP32 dense 通信不能静默替换原 baseline。

本轮只提交诊断报告，按约定直接本地 commit，不使用 autogit，不 push。
