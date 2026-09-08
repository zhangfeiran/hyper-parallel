# Qwen 单后端驻留基准：精度快照、完整训练与 HBM

本轮从 `megamoe-sun@3e6b903c` 继续，处理上一轮最高优先级：完整 Qwen benchmark 的 common
和 MegaMoe 不再同时常驻设备。这里只改 example、测试和文档，没有修改 MoE native、planner、
FP32 梯度聚合或 dX ownership。common 仍为项目 standard EP/GroupedExperts，**不是原版
MoonEP + native MoE baseline**。

**结果：原始宽度八层 Qwen 已在单后端驻留模式完成 ABBA 和每步 AdamW，首次全量精度通过。**
训练状态为每 rank 一套 36.608 GiB；MegaMoe 稳态 peak allocated + SHMEM 最高 42.059 GiB。
约 40.91% 的 common/MegaMoe 耗时差只作为带竞争记录的整模型观测值，不是本轮内核改动收益。
后续多步 loss 分歧尚需诊断，不能据首次通过宣称多步数值或收敛已验收。

## 1. 单后端执行与原精度合同

默认 shell 入口现在运行不导入框架的 `qwen_moe_isolated.py`，顺序启动并等待四组独立 EP8
进程退出：common A1 → MegaMoe B1 → MegaMoe B2 → common A2。每个 worker 只构建一个完整
模型和一套 FP32-master AdamW；每步仍执行 zero-grad、forward、backward、dense gradient
synchronization、optimizer update。默认 8 层、H5120/I1792/T4096/V32000/E48/K8 未变。

A1 将以下 rank-local first-step references 保存到 CPU 文件；B1/B2/A2 均与同一份 reference
比较，全部阶段通过后才生成最终汇总：

| 阶段 | 内容 | 容差 |
| --- | --- | --- |
| inputs | input IDs、labels | 逐元素精确相同，整数不经 FP32 舍入 |
| initial_parameters | 全部 canonical 本地参数 | 逐元素精确相同 |
| forward | 全量 loss、logits | 原 `rtol=0.02 / atol=0.002` |
| gradients | dense 同步后、optimizer 前的全部参数梯度 | 同上 |
| updated_parameters | 一步 AdamW 后全部 canonical 模型参数 | 同上 |

保留原来相对于 common 的 FP32 `isclose` 和归一化误差定义，不改成对称容差，不抽样代替全量
比较。CPU snapshot 使用 owned copy；common gate/up 的 canonical packing 按层生成，不保留
所有层的设备临时拼接。对照采用 mmap，每次只额外暂存一个 candidate tensor，FP32 比较按
1048576 个元素分块。导出阶段仍持有整阶段 CPU tensor map；first-step 梯度捕获仍持有全部
CPU gradients，**不是总 host memory 单 tensor 有界**。

每个 reference 绑定完整配置、seed、rank/world、AdamW LR/weight decay/betas/eps/dtype。
禁止覆盖已有快照，拒绝缺失/重复 tensor、shape/dtype 不同、非有限数值及不完整 reference。
写入或比较的 rank-local 错误经 all-gather 传播，不能让一个 rank 失败后其他 rank 继续训练。
五阶段完成且训练窗口结束后才写 completion marker；这些文件是精度证据，不是可恢复的
训练 checkpoint，也没有比较 AdamW moment/master state 或验证长期收敛。

`BENCHMARK_MODE=paired` 保留原同进程双模型路径和 oracle，供回归使用；不能把它的 HBM
解释成单模型峰值。parent 拒绝 nested torchrun；超时/取消仅回收其新建的 child process group。
每个节点保存自己的 rank-local references 和 child JSON，node zero 写最终汇总。单机 EP8 已
实测，多节点编排没有硬件验收。

## 2. 计时、HBM 和软件身份

CPU reference I/O、完整精度比较、source/native hash 均在稳态计时之外。first-step CPU 梯度
快照在剩余 warmup 前释放。两侧执行相同 warmup/measurement 数量，warmup 后重置 allocator
peak；计时包括每步 AdamW，以各 rank 最大值为该步耗时，各进程取 median，再对同 backend
的进程 median 取均值。不把不同进程样本混成一个 median。

每 rank 记录以下不同口径，不重复相加：

- `peak_allocated_bytes` / `peak_reserved_bytes`：稳态窗口 Torch allocator 峰值。
- `symmetric_heap_bytes`：进程实际 external SHMEM heap，common 未初始化时为 0。
- `conservative_total_peak_bytes`：peak allocated + SHMEM heap。B pool 已在 allocator 中，
  workspace view 已在 heap 中，不能再加一次。该项不包含 opaque driver/HCCL allocation，
  不能当成完整进程物理峰值上界。
- `final_device_used_bytes`：计时结束后设备 runtime 的 used sample，非峰值，也非进程独占值。
- `resident_training_per_rank`：实际本地参数数、BF16 model、FP32 master/main grad/AdamW state
  张量字节；`shared_pool_per_rank` 单列实际 B weight/gradient bytes，检查跨层一份资源组。

worker 验证 editable import 指向当前 checkout。软件指纹覆盖本 example 的 Python 文件及
`hyper_parallel/**/*.py`，正式 payload 的全部 `.so`、`.o`；本次共 949 个 Python 文件和
17 个 native 文件。每个进程前后、各 rank 之间及四个进程组之间必须匹配，同时记录
Torch/Torch-NPU/Transformers 版本。指纹范围不代表所有系统库或设备环境完全相同。

环境沿用 8×Ascend 910B3 64 GiB、CANN 9.2.0、Torch/Torch-NPU 2.9.1。可选 Transformers 5.5.3
仍只装在上一轮隔离 venv `/tmp/megamoe-qwen-shared.0MYlWz/venv`，主环境依赖不变。

## 3. 回归验收

按照 add-unit-test 流程补齐 CPU snapshot、精确整数比较、原非对称容差、坏 reference、跨 rank
错误传播、单模型构建、CPU gradient capture、warmup/peak reset 顺序、SHMEM 非重叠计数、
ABBA 配置透传、失败/超时清理及 parent 不导入框架的测试。新增 HBM 合同用例先观察到缺失
`symmetric_heap_bytes` 的失败，再补实现通过。

EP8 两层 H1024/I256/T1024/V2048、B1、capacity=1.5、每步 AdamW：

- 新进程隔离模式的 balanced / Dirichlet 0.1:42 / learned TopK 全部通过。
- 原 paired 模式三种路由全部通过，原设备比较逻辑没有被替换或绕过。
- 补充 SHMEM 与 `.o` 指纹后，独立重跑新模式 Dirichlet：1 passed / 2 deselected，120.66 s。
  梯度最大绝对误差 `1.52587890625e-5`；实际 B1 跨层权重/梯度为 `1572864/3145728 bytes`。

小形状短测只用于正确性；部分早期 ST 与 CPU 验证重合，不从其耗时推导性能收益。

最终 CPU 回归：**281 passed / 259 subtests，38.36 s**，5 条既有 marker warning。主环境无
Transformers：19 passed / 25 subtests，模型集成文件按缺少可选依赖 skip 1 项；隔离 venv 的
模型集成用例真实执行。八个改动 Python 文件 pylint、shell `bash -n`、`git diff --check` 通过。

## 4. 原始八层测量

正式运行使用默认宽度和层数、B1 shared-B、capacity=1.5、Dirichlet 0.1:42、seed 42，
每组 3 warmup + 7 measured 完整 optimizer steps。通过实际 shell 默认入口启动 ABBA：

```bash
PYTHON_BIN=/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python \
MASTER_PORT=29794 SHMEM_PORT=21654 HCCL_NPU_SOCKET_PORT_RANGE=62000-62063 \
bash examples/torch/multicore/mega_moe/run_qwen_moe_benchmark.sh \
  --shared-physical-weights --replica-slots-per-rank 1 \
  --expert-capacity-factor 1.5 --routing dirichlet --router-load-alpha 0.1 \
  --warmup-steps 3 --measured-steps 7 --timeout-seconds 900 \
  --artifacts-dir /tmp/megamoe-qwen-isolated.Ud8iJ4/full8 \
  --output output/qwen_moe_optimizer/qwen_isolated_full8_abba_20260908.json
```

未在正式计时期间运行 CPU tests/lint/profile；每约 10 秒记录设备占用与进程归属。早期 Jenkins
作业曾使用八卡，因此先等空闲，没有终止任何不属于本轮的进程。

### 4.1 单模型训练状态与空间代价

原始八层已实际完成 common 和 MegaMoe 的完整 optimizer steps，没有降低模型形状或取消
optimizer。每个 backend、每个 rank 的实际账本均为：

| 项目 | 数值 |
| --- | ---: |
| 本地参数数（不是 bytes） | 2183746560 |
| BF16 model | 4367493120 |
| FP32 master | 8734986240 |
| FP32 main grad | 8734986240 |
| FP32 AdamW moments | 17469972480 |

以上训练状态合计 36.608 GiB/rank，而原 paired 模式两份状态的解析下界为 73.216 GiB。
这里消除的是重复常驻的逻辑张量，**不是实测八层旧/新 allocator 峰值相减**；旧模式八层没有
实际运行过，不能虚构一个 before peak。

MegaMoe 每 rank 的八层共用 BF16 B weight `55050240 bytes`（52.5 MiB）及 FP32 B gradient
`110100480 bytes`（105 MiB），没有成为八份。external SHMEM 为 `872415232 bytes`（832 MiB）。
host references 实际合计 99.575 GiB，保留在本轮 `/tmp` artifact 目录，不进 git；原始 JSON/
日志和下方 hashes 可在本机追溯。CPU/存储成本只用于精度对照，没有将训练 optimizer offload
到 CPU。

### 4.2 竞争记录与可解释范围

完整 census 共 122 帧，回查发现当前 worker 以外的设备 PID，采样时间（UTC）为
`03:01:31--03:01:53`、`03:02:46--03:03:08`、`03:04:01--03:04:12`、`03:15:16`。
第一段在 A1 结果文件写出之后，第二、三段在 B1 生存期，最后一帧在 A2 生存期。部分明确
对应 Jenkins，host launcher 还出现于 `03:03:29--03:04:44`，其他 PID 未完整保留命令行，
不强行全部归属给 Jenkins。B1/B2/A2 结果文件的 mtime 分别为 `03:06:28.404`、
`03:12:00.385`、`03:18:24.798 UTC`。B2 生存期未采到额外设备 PID，但本版没有逐步绝对
计时窗口标记，结果 mtime 不足以精确排除重叠，约 10 秒采样也不能排除短时竞争；**不得把
本轮称作严格独占环境 ABBA，也不据此确认细小优化增量**。没有清理或终止任何额外进程。

两次 MegaMoe 运行的 step median 为 `714.241882/713.191162 ms`；稳态 peak allocated 为
`41.245522/41.246495 GiB`，计入 SHMEM 后为 `42.058022/42.058995 GiB`。这是当前 common /
MegaMoe 对照的观测值，不是本轮改代码的内核加速收益。阶段内 warmup 后才重置峰值，HBM
账本与 device-wide sample 的解释仍严格区分。

所有 B1/B2 inputs、83 个 canonical 初始参数均精确相同；最大梯度绝对误差分别为
`3.814697265625e-6 / 2.384185791015625e-6`，更新后最大归一化误差为
`0.979276 / 0.979855`。更新后最大绝对误差均为 `0.00201416015625`，大于单独的 atol
并不代表失败，原门槛是 `atol + rtol * abs(reference)`。这里没有放宽门槛，也不是逐位一致。

### 4.3 最终汇总与逐 rank 成本

四个进程组均正常退出，source/native identity 前后、跨 rank、跨运行全部匹配。A2 也通过
全部五阶段，最大梯度绝对误差 `3.814697265625e-6`，更新后最大归一化误差 `0.982151`。
以下 HBM 列取该运行的所有 rank 最大值，GiB 均按 2^30 bytes：

| 运行 | full-step median（ms） | peak allocated（GiB） | peak reserved（GiB） | allocated peak + heap（GiB） |
| --- | ---: | ---: | ---: | ---: |
| common A1 | 1208.826 | 43.925621 | 47.488281 | 43.925621 |
| MegaMoe B1 | 714.242 | 41.245522 | 44.222656 | 42.058022 |
| MegaMoe B2 | 713.191 | 41.246495 | 44.222656 | 42.058995 |
| common A2 | 1206.715 | 43.925621 | 47.488281 | 43.925621 |

各 backend 的进程 median 均值为 common `1207.770386 ms`、MegaMoe `713.716522 ms`，
ratio `0.590937`，观测耗时降低 `40.906%`。不与上一轮独立 16 层 receive-only 优化的
`4.086%` 叠加；也不表示这一轮单驻留 runner 修改使内核快了 40.906%。本轮没有测原始八层
balanced 或 learned TopK 的匹配性能曲线。

全组最大峰值降低不等于每个 rank 都降低。下表 common 两轮逐字节相同，MegaMoe 取 B1/B2
中该 rank 的较大值；口径统一为 peak allocated + heap：

| rank | common（GiB） | MegaMoe（GiB） | 差值（GiB） |
| --- | ---: | ---: | ---: |
| 0 | 43.925621 | 41.904706 | -2.020916 |
| 1 | 42.405314 | 42.056083 | -0.349232 |
| 2 | 40.968607 | 41.904215 | +0.935608 |
| 3 | 43.106899 | 41.904706 | -1.202193 |
| 4 | 41.571598 | 42.058022 | +0.486424 |
| 5 | 41.580493 | 42.058995 | +0.478501 |
| 6 | 42.531358 | 41.954512 | -0.576846 |
| 7 | 42.698669 | 41.904704 | -0.793965 |

因此实际减少的是全组最大值约 1.867 GiB，并非所有 rank 的增量完全消失。冷 rank 的额外
heap/pool/计算临时张量仍可能超过负载均衡节省的激活；逐项归因需要另做生命周期 profile。

### 4.4 多步数值边界

本组固定随机权重、重复同一 batch、默认 LR=1e-3。后续 loss 只记录，没有逐步梯度/参数
oracle。rank-zero 第 10 个 optimizer step 的 loss 为 A1 `8.457677`、A2 `9.013068`、
B1 `9.719075`、B2 `8.630766`，两种 backend 内部自重复也存在分歧，不能据此单独归因给
MegaMoe，亦不能称作已经验证了多步精度。所有实测 loss 有限，但这不是收敛实验。

## 5. 本轮后优先级

1. **先补多步数值重复性与可审计测量窗口。** 在原配置下做 common/common、MegaMoe/MegaMoe
   及交叉 backend 的逐步 loss、梯度、更新后参数对照，先区分既有非确定性、学习率敏感性和
   backend 特有差异；不能放宽原阈值掩盖分歧。计时增加绝对窗口标记，与所有设备 PID 采样
   对齐，并在可用的稳定八卡窗口重测。原始配置保留；任何稳定性辅助配置必须单列。
2. 数值和计时边界明确后，用已可运行的单模型 HBM 基准采完整 connected Qwen 全 rank trace，
   重新定位 receive-only / mixed batch、dense 同步和 optimizer 的占比，再选下一项调度修改。
   不从独立层的 rank 1 晚邻居窗口直接推导整模型收益，不泛化删除 mixed send 的 dW producer wait。
3. 沿逐 rank 账本再评估 FP32 dW/owner receive 生命周期。保持全部 replica contribution FP32
   聚合后才 cast；不为减小临时 allocation 无条件增加 persistent pool。shared-B 双 buffer、
   更大 B 和预测 prefetch 仍后置，residual multi-wave 保留。

本轮完成单后端驻留这个前置项，不宣称性能优化已到上限。按约定直接本地 commit，不使用
autogit，不 push。

## 6. 原始证据

完整四次结果及每步 latency、loss、rank ledger、source/native map 都在以下汇总 JSON；
其余文件在同一 `output/qwen_moe_optimizer/` 目录，未入 git。SHA256：

```text
qwen_isolated_full8_abba_20260908.json
  7303a47bff7feb9bb6b609031cc6c73583cafb50b96b093d32e70f738ed4afb5
qwen_isolated_full8_abba_20260908.log
  d928d6eafdebb8ef1490211f97ef6d29ebc518019e42bec41f51fa18297a0b97
qwen_isolated_full8_census_20260908.log
  40878982492c0a02e33d725b176d8052494c2b2b3535cd3132473357803ea002
qwen_isolated_full_ut_final_20260908.log
  116ef4d5bf04a6876ac6118cd5e6375472c6cd334af05e633ceffe2cd2b20456
qwen_isolated_st_final_20260908.log
  a19cccc6775bd443bdd5b34eacd4c1c08e48442170dd6df82edcf772af12c0a0
qwen_isolated_st_remaining_20260908.log
  81bc1a7d3ed2fe8a2774bf541f8203ba295751a6c1dd08b8a49b41eaf88798bc
qwen_isolated_paired_regression_20260908.log
  6a1a785238dec3d9486ac659bcb81d69e9d21f54b655afd030543ad774fa5729
qwen_isolated_optional_ut_20260908.log
  6b163803ab09daba5dac8c3f2af8fe72844567e006121a5a1fbe4aa67ece7c55
```

四组 worker 的共同 identity map 摘要，算法为
`sha256(json.dumps(map, sort_keys=True).encode()).hexdigest()`：

```text
source_sha256 map
  865b2de21520d05ce74e444f2ad54f6b718b8687543ea564e88f8e4ce78eb5ee
native_sha256 map
  1c0c5f4b076280c416c4690d0f85a37961c161ba73f721aa823740c831559452
```

99.575 GiB 原始 CPU references 和四组 child JSON/log 位于：
`/tmp/megamoe-qwen-isolated.Ud8iJ4/full8/qwen_isolated_uzsa2kya/`。
小形状证据位于同一临时根目录的 `st_initial/`、`st_remaining/`、`st_final/`、`paired/`。

## 7. 后续模型定义分层重构

本节记录 `ed03e671` 之后按用户要求做的职责迁移，不替换上述历史测量和源码指纹：

- `NativeMoeExperts` 定义在 `qwen_moe_model.py`，native token permutation、standard EP /
  GroupedExperts、unpermutation 全部属于模型定义。benchmark 不再导入或检查这些具体实现类型。
- 新增模型侧 `build_qwen_moe_model(config, device, backend=..., ep_group=...)`，负责 BF16
  初始化、保留 rotary buffer dtype，以及 common 的 EP 分片和本地权重复制。seed、optimizer、
  精度门槛、snapshot、计时和 HBM 诊断仍由 benchmark 负责。
- 保持先构造全部 decoder、再逐层转换 common experts 的顺序，不把 GroupedExperts 初始化
  插到后续 dense 参数之前；没有改变随机数消耗顺序或参数/optimizer ownership。
- 模型提供 `named_canonical_tensors(gradients=False)`，统一暴露 dense 参数及 packed gate/up、
  down 布局；benchmark 仅消费该接口。canonical 名称、shape、dtype 和按层惰性拼接保持不变。
- AST 对照确认 adapter 的构造/forward/close 与原实现一致（仅 dtype 常量名迁移），canonical
  转换函数体仅将 `model` 换成 `self`。17 个 native artifact hashes 与第 6 节记录一致。

初始重构提交 `15f707a6` 按 add-unit-test 先补 6 项回归并观察其在旧实现上失败，再搬迁代码。完整 CPU 验证为
287 passed / 307 subtests，51.39 s；三个修改 Python 文件 pylint 通过。无 Transformers 主环境
19 passed / 25 subtests，模型集成文件因缺少可选包 skip 1 项；有依赖环境真实执行全部模型测试。
当时新增用例覆盖独立 model import、两 backend 精确初始参数、RoPE dtype、非法 backend、异常后
default dtype 恢复、原路由 histogram/score 透传、canonical gradient 和 benchmark 薄委托。

后续按覆盖收益将该轮新增 UT 从 6 项精简为 2 项，只保留两 backend 初始化参数/dtype 一致及
构造异常后 default dtype 恢复的检查。删除独立导入、薄委托、mock forward 和全 1 梯度布局
检查；真实训练行为继续由已有八卡回归覆盖。模型、benchmark 和 ST 用例均未修改。
精简后 Qwen model/config/snapshots/isolated 四个文件的 CPU 回归为 33 passed / 50 subtests，
2.72 s，修改后的测试文件 pylint 通过；日志为 `output/qwen_moe_optimizer/qwen_model_ut_trim_20260908.log`。
本轮未重跑八卡，以下 ST 和快照数字仍是初始重构提交的验证记录。

此前 EP8 完整 connected 模型小形状回归为 6 passed，519.31 s：paired 与 isolated 各覆盖 balanced、
Dirichlet、learned TopK。配置为两层 H1024/I256/T1024/V2048、E48/K8、shared-B1，包含逐步
AdamW；isolated 使用 fresh-process A/B。所有首次输入/初始化、forward、gradient、更新后
参数检查通过（paired 不单独导出输入快照），继续使用原 `rtol=0.02 / atol=0.002`。

额外逐 rank、逐阶段对比本次 common A1 与重构前保留的 common CPU references，三种路由的
inputs 和 initial parameters 都精确一致；其余阶段通过原阈值，但不是 bitwise 一致：

| 路由 | 每 rank 参数/梯度张量数 | 最大 forward 绝对误差 | 最大梯度绝对误差 | 更新后最大归一化误差 |
| --- | --- | --- | --- | --- |
| balanced | 23 | 1.43051e-6 | 1.52588e-5 | 0.982961 |
| Dirichlet | 23 | 9.53674e-7 | 1.52588e-5 | 0.975253 |
| learned TopK | 25 | 9.53674e-7 | 1.52588e-5 | 0.980974 |

新六组结果、逐 rank CPU snapshots 和跨版本比较脚本保留在
`/tmp/megamoe-qwen-model-refactor.BkjW7d/`（`paired/`、`st/`、`compare_references.py`）；
旧小形状 references 仍在第 6 节的临时根目录。以下日志/汇总位于 `output/qwen_moe_optimizer/`，
没有加入 git；SHA256：

```text
qwen_model_refactor_full_ut_20260908.log
  5d680be5730d8294e34e8eff2696db9c844ae636cdc716ae44e6a8c43db203f5
qwen_model_refactor_st_20260908.log
  8a202a9f1e276cba84be00bf54c0f7ff243d9cabff778e440e7aabe55dec4d5c
qwen_model_refactor_previous_balanced_20260908.json
  8d57c7585f334d22c9f4062ea02058954a46840b6fc53bf3aaf3e4ef87917a87
qwen_model_refactor_previous_dirichlet_20260908.json
  61eba4a622a2980275de4be58b557305a70bd5de32cf7bb7d3b126551d607cd7
qwen_model_refactor_previous_topk_20260908.json
  8e2fab091a8704fa10f0e5bb22c229c7fcacc95670816f7467fb6ec537fa07f6
```

临时比较脚本 SHA256 为
`0971da43bb74ecee28fa29e82a2fcaa432df168c608db7f45b51bcd775816dd3`。

本次是结构性重构，不宣称性能/HBM 优化，不用上述历史八层计时充当本次复测；多步 loss
重复性和计时窗口校准仍是后续待处理项。CPU UT/lint/参考比较与部分短 ST 同时执行，短测
latency 仅作为测试产物，不作为性能证据。
