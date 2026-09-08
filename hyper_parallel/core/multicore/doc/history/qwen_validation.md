# Qwen shared-B 接入与单后端驻留：历史验证

本文汇总 2026-09-07 至 2026-09-08 的历史实验，保留各阶段的测量配置、失败与修复边界、原始文件及哈希。
结果对应各节列出的旧源码和 native 产物；本次整理没有重跑性能或数值验收，也不将它们改记为合并后结果。
旧报告里的“本轮/当前”均指该历史阶段。后续实现与验收统一由
[当前待办](../megamoe_optimization_backlog.md)跟踪，本文不维护另一份优先级。

以下 output/ 与 /tmp/ 路径是当时的本机证据位置，原始大文件未入 Git；本次未核验其仍然可用。
文末固定 Git 版本可恢复完整旧报告，不保证恢复未入库的原始数据。

| 阶段 | 提交 | 已验证范围 |
| --- | --- | --- |
| shared-B 接入 | 51d87203 | 两层小形状三种路由、first-step 全量对照；倾斜组真实使用 B 池 |
| 原宽度检查 | 3e6b903c | 两层/四层首次精度通过；八层 paired 的两套状态下界 73.216 GiB |
| 单后端驻留 | ed03e671 | 原始八层运行及首次精度完成；计时有竞争记录，多步数值未验收 |
| 模型职责迁移 | a3e1b51b | common adapter/canonical tensor 回到模型侧；旧八层结果不是重构后的复测 |

common 指项目 standard EP/GroupedExperts。操作入口见
[Qwen benchmark 说明](../../../../../examples/torch/multicore/mega_moe/README.md)。
多步数值与可审计计时窗口仍由[当前待办 T02/T03](../megamoe_optimization_backlog.md)跟踪。

## Shared-B 接入与原宽度复核

接入阶段保持原默认八层 H5120/I1792/T4096/V32000/E48/K8/EP8，增加显式 shared-B 配置，
跨层复用 execution resource group，home 参数和 optimizer state 仍独立。
修复 RoPE batch 轴、可选组件懒导出，以及只探测第一个冷专家的 first-step 更新检查。
更新探针仍是 smoke check；全量 forward/gradient/updated-parameter oracle 没有被替换。
synthetic route 与 learned TopK 的参数数和实际副本激活情况分开记录。

### 验证环境和范围

8×Ascend 910B3、CANN 9.2.0、Torch/Torch-NPU 2.9.1，正式 native artifacts 未改变。
主环境没有 Transformers，因此在 `/tmp/megamoe-qwen-shared.0MYlWz/venv` 创建继承主环境的隔离
venv，安装 Transformers 5.5.3；editable 仍指向当前 checkout，主环境依赖未改，未安装 DeepSeek
专用 custom operators。Qwen 配置构造使用 `from_dict`，兼容该版本的 strict config 定义。

按照 platform-dev / add-unit-test 流程分别覆盖入口配置、真实 CPU 模型构造、RoPE batch 合同、
懒导出/异常合同与更新探针，再运行真实 NPU 系统测试。ST launcher 不导入框架；独立导入检查
确认 `torch/torch_npu/mindspore/hyper_parallel` 均不在 launcher 的 `sys.modules` 中。

八卡 ST 每组为两层 connected Qwen：H1024/I256/T1024/V2048/E48/K8，B1，capacity=1.5，
BF16 模型、FP32 master AdamW；每个 backend 执行 1 个 first-step warmup 和 2 个后续 optimizer
step。common 是当前项目 standard EP/GroupedExperts 对照，不是原版 MoonEP native baseline。
learned TopK 包含 router 参数训练；平衡和倾斜为固定 synthetic route。这里没有模型 checkpoint。

最终三组全量比较均通过；下表为跨 rank 最大绝对误差：

| 路由 | 参数/梯度张量数 | forward（loss/logits） | 梯度 | 更新后参数 | 更新后最大归一化误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 23 | 4.7684e-7 | 1.5259e-5 | 0.00199890 | 0.979706 |
| Dirichlet 0.1:42 | 23 | 1.4305e-6 | 1.5259e-5 | 0.00199890 | 0.979855 |
| learned TopK | 25 | 9.5367e-7 | 1.5259e-5 | 0.00198364 | 0.988323 |

所有初始 canonical 参数逐位相同。归一化误差按原 `abs(diff)/(atol+rtol*abs(reference))`，
更新后最大值接近 1，不能把“通过”夸大成逐位一致，也没有放宽阈值。这里仅全量比较第一次
optimizer step；后续 loss 有保存，不代表逐步精度配对或收敛验收。

倾斜路由每个 rank 的 BF16 B 权重均为 `1572864 bytes`，FP32 B 梯度 `3145728 bytes`，
两层只持有一份池。balanced 和本次 learned TopK 未激活 replica，池为 0，符合 lazy 分配；
不能把 learned-router 通过算作实际 B 通信覆盖，实际 replica 由倾斜组覆盖。

最终验证：

- CPU：**259 passed / 242 subtests，39.10 s**；5 条既有 marker warning。
- 八卡完整模型：**3 passed，172.09 s**；4 条既有 marker warning。
- 不含 Transformers 的主环境：7 passed / 8 subtests，模型集成文件按缺少可选包 skip 1 项；
  config 和模块导出测试不 skip。隔离环境中的模型集成测试真实执行。
- 九个改动 Python 文件 pylint、`git diff --check` 通过。

首轮 RoPE 失败、随后 cold-expert 探针导致的 1 failed/2 passed 结果均保留。修正探针后先单独
重跑倾斜通过，再 fresh-process 重跑完整三组。CPU/lint 与部分 ST 有重合，所以短测 JSON 中
的 common/MegaMoe A/B 耗时只作 smoke 数据，不用其百分比声称优化收益；没有做整模型 ABBA。

### 复现与证据

完整示例说明见 [README](../../../../../examples/torch/multicore/mega_moe/README.md)。
已激活 CANN/native 并安装 example 可选依赖后：

```bash
HYPER_PARALLEL_PLATFORM=torch OMP_NUM_THREADS=1 \
python -m pytest -q tests/torch/expert_parallel/test_qwen_moe_shared.py
```

Launcher 为每项测试隔离输出目录；可用 `HP_QWEN_MOE_EVIDENCE_DIR` 指定统一 evidence 目录。
最终原始数据位于 `output/qwen_moe_optimizer/`（未入 git）：

```text
qwen_shared_evidence_final_20260908/qwen_shared_balanced.json
  0d682ba63d0f0299d8be9ba09a6ff68f583a4f667db61e40ed26c00fcceba211
qwen_shared_evidence_final_20260908/qwen_shared_dirichlet.json
  910a9e1d0ce0f4a2a8dcc9033556474dbc1a7e7a0aa3370e67c3b6de1f2c368b
qwen_shared_evidence_final_20260908/qwen_shared_topk.json
  d478cd2caa001d4449508d3e57f73b2ed471cf7182c3b7b15f89eefeb424fc82
qwen_shared_st_final_20260908.log
  a8b52a0fb480e44e754ebc226ddec1f35e08c1959dc9c497e87cf6b471a9b483
qwen_shared_full_ut_final_20260908.log
  9250152d6e8ac07478f0f279c3487b71188979b3960741b45819ffa0e9f3e729
qwen_shared_optional_ut_20260908.log
  957faa9af0a32fbd0e3569dbeee98bda417ba87c530978934cfa841faa40559b
```

生产/示例文件 SHA256（对应本轮验收）：

```text
qwen_moe_config.py   b528a5e53ca97289c2ec6c4f03abb6c9fe16ce60300841d16ab5e0ecfaee42b9
qwen_moe_model.py    962b2ee55c72c74dc4451860616f91801b3c82aac3db86dbeab30486e435f83f
qwen_moe_benchmark.py f6ddef180917da4fd757012d5be82f4033d0ef9114af646d5405cc6399660874
modules/__init__.py  f9e4eb9637a000c9fda28553972797ce0d1e1b4912a7dce3d22539439f510166
```

### 原宽度两层/四层与双驻留下界

接入轮已直接本地提交为 `51d87203`。随后保持生产代码不变，另外启动 fresh-process 八卡
两层、四层 Qwen：H5120/I1792/T4096/V32000，E48/K8、B1、capacity=1.5、Dirichlet 0.1:42，
仍为每 backend 1 warmup + 2 measured optimizer steps。两个进程组均正常退出，原精度阈值下
全部 first-step 比较通过：

| 层数 | 参数/梯度张量数 | forward 最大绝对误差 | 梯度最大绝对误差 | 更新后最大归一化误差 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 23 | 2.8610e-6 | 1.9073e-6 | 0.957214 |
| 4 | 43 | 2.8610e-6 | 3.8147e-6 | 0.967874 |

所有初始 canonical 参数逐位相同，更新后参数最大绝对误差均为 0.001953125。每个 rank 的
B weight/gradient 均为 `55050240/110100480 bytes`，即 52.5/105 MiB，四层没有变成四份池。
这些仍是短程正确性复验，不是旧/新 receive-only 版本的 matched ABBA，也没有验证收敛。

进一步审查发现，现有 benchmark 同时构造 common 和 MegaMoe，并一直持有两套模型和
FP32-master AdamW。`mixed_precision_optimizer.py` 为 BF16 模型建立 FP32 master，并保留
FP32 `.grad` 到下一次 `zero_grad`；`core/optimizer/adamw.py` 的两个 moment 按 FP32 grad 分配。
因此不能将该进程的总 HBM 当成单个 shared-B 模型的训练峰值。

固定 synthetic route 没有 learned gate 参数。按当前模型结构，每 rank 的本地参数数量为：

```text
P_layer = 3 * (E / EP) * H * I
          + 2 * H^2 + 2 * H * (num_key_value_heads * head_dim)
          + 3 * H * shared_intermediate + 2 * H
        = 232007680
P_other = 2 * V * H + H = 327685120
P_total = L * P_layer + P_other
```

每个 backend 的 BF16 模型、FP32 master、两个 FP32 moment 和 FP32 main grad 合计 `18 * P`
bytes；两个 backend 为 `36 * P`。以下是逻辑张量下界推导，**不是实测 allocator peak**，
不含激活、首次梯度快照、临时 comparison/GMM tensor、SHMEM、B pool 和 allocator reserve：

| 层数 | 每 backend 本地参数数 | 双模型+master+moments | 再含两套 main grad |
| --- | ---: | ---: | ---: |
| 2 | 791700480 | 20.645 GiB | 26.544 GiB |
| 4 | 1255715840 | 32.745 GiB | 42.101 GiB |
| 8 | 2183746560 | 56.946 GiB | 73.216 GiB |

在相同两层配置上另起 fresh 八卡进程，只在全部计时窗口结束后的 `_write_result` 前读取实际
tensor 元数据。没有改模型、optimizer 或通信，也没有在计时窗口采样。两个 backend、全部八个
rank 的账本完全一致，逐项验证了上述两层参数计数和 dtype 下界：

| 每 backend 项目 | 实际持有 bytes |
| --- | ---: |
| BF16 model | 1583400960 |
| FP32 master | 3166801920 |
| FP32 main grad | 3166801920 |
| FP32 AdamW moments | 6333603840 |

该进程的跨 rank 最大 allocator peak allocated/reserved 为 `36853763584/40489713664 bytes`
（34.323/37.709 GiB）。这是两套 backend 加首次准确性比较及整个训练过程的混合峰值，不是
MegaMoe 单独的 HBM，不能与独立层 benchmark 的峰值直接相减。本次诊断仍通过全部 accuracy
比较，更新后最大归一化误差为 0.957854。临时审计脚本仅保留 workload 引用、统计 local tensor
字节并在结果输出前 gather；原 benchmark 本来就持有这些 workload 到退出，没有延长计时窗口
内的 storage 生命周期。

所以不再直接启动 64 GiB 卡上的默认八层双模型进程来反复触发 OOM。本轮未实跑八层；此处的
不可容纳判断来自当前持有关系和 dtype 的下界，不是将两层实测峰值按比例外推。

该阶段据存储下界停止启动原宽度八层 paired；随后单后端驻留解决了重复训练状态问题，
仍保留相同 optimizer、模型形状与原精度合同。下列是该阶段的独立证据。

原始证据在 `output/qwen_moe_optimizer/`，SHA256：

```text
qwen_shared_original_width_2layers_20260908.json
  846bc2050c5aa456cc47ecac14f75f4ef5917bf1d89c34ada899214f4f3d0799
qwen_shared_original_width_4layers_20260908.json
  aa2c99773d508b269b569215d978b67ebcd62860ca6357174874aff133f1a4e1
qwen_shared_original_width_2layers_20260908.log
  5667f31dbea87cff4ae0d2d7b6dad5dc887c3e7126383c5d20ee3ed87c2afaa0
qwen_shared_original_width_4layers_20260908.log
  53de6637313ae808cc451c4abe113fa292a21ef8bb5b94ee7a07652a757b253b
qwen_shared_resident_ledger_20260908.json
  c0336b543e3267f5639d3ccd01fe4d1f7b6fa2a1de508a5371689d4b0b4f1aff
qwen_shared_resident_ledger_20260908.log
  9e67243f1ef487ded81cf91be9c492a4e7d518a8e0b6754b45c482e5d0d1926d
/tmp/megamoe-qwen-shared.0MYlWz/inspect_ledger.py
  73b8b3453badbaefbe4236867e34a64e8061cd2a55000cbf7b57fdd7fe6f3e38
```

## 单后端驻留与首次精度

### 单后端执行与原精度合同

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

### 计时、HBM 和软件身份

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

### 回归边界

两层 H1024/I256/T1024/V2048、EP8/E48/K8/B1、capacity=1.5，paired 与 isolated 各三种路由通过。
CPU 回归为 281 passed / 259 subtests；缺少可选 Transformers 的环境单列 skip。
这些小形状短测不是性能证据；isolated 的首次精度比较不等于多步 oracle。

### 原始八层测量

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

### 单模型训练状态与空间代价

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

### 竞争记录与可解释范围

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

### 最终汇总与逐 rank 成本

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

### 多步数值边界

本组固定随机权重、重复同一 batch、默认 LR=1e-3。后续 loss 只记录，没有逐步梯度/参数
oracle。rank-zero 第 10 个 optimizer step 的 loss 为 A1 `8.457677`、A2 `9.013068`、
B1 `9.719075`、B2 `8.630766`，两种 backend 内部自重复也存在分歧，不能据此单独归因给
MegaMoe，亦不能称作已经验证了多步精度。所有实测 loss 有限，但这不是收敛实验。

后续基于 `ee5a6965` 的多步数值与计时诊断见
[多步诊断](../megamoe_qwen_multistep_diagnosis_20260908.md)；
该后续结果不替换上述历史测量。

## 单后端驻留原始证据

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

## 后续模型职责迁移与验证

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
  转换函数体仅将 `model` 换成 `self`。17 个 native artifact hashes 与上节原始证据一致。

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
旧小形状 references 仍在上节记录的临时根目录。以下日志/汇总位于 `output/qwen_moe_optimizer/`，
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

## 原报告与恢复入口

整理前的完整文档固定保存在提交 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec，路径前缀为 hyper_parallel/core/multicore/doc/。
下表文件已由本文替代；旧阶段待办及完整过程叙述仍可从该提交读取。

| 原文件 | 本文锚点 |
| --- | --- |
| megamoe_qwen_shared_integration_20260908.md | [Shared-B 接入与原宽度复核](#shared-b-接入与原宽度复核) |
| megamoe_qwen_isolated_benchmark_20260908.md | [单后端驻留与首次精度](#单后端驻留与首次精度) |

从仓库根目录查看原文：

```bash
git show 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec:hyper_parallel/core/multicore/doc/megamoe_qwen_shared_integration_20260908.md
```
