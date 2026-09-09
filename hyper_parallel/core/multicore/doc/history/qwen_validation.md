# Qwen 接入、数值诊断与训练边界：历史验证

本文汇总 2026-09-07 至 2026-09-09 入库的历史实验，保留各阶段的测量配置、失败与修复边界、原始文件及哈希。
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
| 多步与数值放大诊断 | caf71330、ba9a83a2 | 原八层复现、实际 dW/optimizer 复放、参数因果干预及 eps/LR 控制；完整训练门槛仍分项记录 |

common 指项目 standard EP/GroupedExperts。操作入口见
[Qwen benchmark 说明](../../../../../examples/torch/multicore/mega_moe/README.md)。
新增结果先读[数值诊断汇总](#数值诊断汇总)；剩余全局梯度契约与干净计时窗口由
[当前待办 T02/T03](../megamoe_optimization_backlog.md)跟踪。

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
[多步诊断](#多步数值与计时诊断)；
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

## 数值诊断汇总

以下六轮完整报告来自远端 `caf71330`、`ba9a83a2`，实验的生产 Python 基线均为 `ee5a6965`，
环境为 CANN 9.2.0，具体 native/输入及原始数据身份见各节。2026-09-09 rebase 只接收文档，
未把这些结果改记为本地 profiler/计时审计版本的重新验收，也没有重新运行八卡实验。
正文、数据表、失败记录和复现命令均保留，仅调整标题层级及相对链接；各轮“下一步”描述其历史时点，
后续已完成的项目以本表和[当前待办 T02/T03](../megamoe_optimization_backlog.md)为准。

| 已有证据 | 结论及边界 |
| --- | --- |
| [默认与确定性控制](#多步数值与计时诊断) | attention backward、BF16 dense collective 均可产生共有非确定性；确定性模式恢复同 backend 自重复，跨 backend 多步仍有差异 |
| [实际第 8 层 dW 复放](#expert-dw-切分与舍入) | B0 单波逐位一致；B0 多波/B1 差异局限于被切分/复制的 expert，dW 最大差 4.7683716e-7；FP32 owner 聚合和最终 cast 与 CPU 重算一致 |
| [原 fused AdamW 复放](#adamw-与-bf16-写回) | 同梯度、不同布局的 main/moments/model 三次更新逐位一致；真实梯度舍入差异经过小梯度敏感区和 BF16 写回边界形成参数差异 |
| [八层参数因果干预](#参数干预与层间传播) | 首步 11227 个 expert BF16 参数差异的恢复/注入消除/复现第二次 forward 分歧；同权重下两 backend 全层输出逐位一致，尚未覆盖后续多次更新反馈 |
| [eps 控制](#eps-稳定性与-trainer-契约) | 增大 eps 降低 main 最大差但未使完整训练通过；保留原默认值 |
| [LR=1e-4 控制](#低-lr-多步对照) | 确定性模式十步 loss 持续下降且跨 backend 全部通过，第 1/2/10 步梯度/参数通过；logits 第 2–10 步仍超原阈值，长期训练未验收 |

这组证据支持“切分改变舍入结果，经 AdamW/BF16 写回及层间传播放大”的解释，未发现本组
replica 聚合或 optimizer 实现错误。原 `rtol=0.02 / atol=0.002` 不变；机制定位、逐项精度通过、
完整训练通过和干净性能测量分别记录，不能用其中一项覆盖另一项。

## 多步数值与计时诊断

本轮基于拉取后的 `megamoe-sun@ee5a696554133d99c80aa9029e5cdd9ab039c1ac`，继续
[单后端驻留报告](#单后端驻留与首次精度) 第 4.4 节留下的诊断。
本轮不修改模型、optimizer、MegaMoe runtime 或原精度阈值；确定性模式、FP32 归约、低学习率
均是单列的诊断对照，不替换默认 baseline。

结论：默认模式有两 backend 共有的非确定性，确定性控制可恢复各自重复性，但跨 backend
仍有从 expert dW 开始的可重复差异，多步 logits/loss 尚未通过原门槛。低 LR 只改善本组
十步轨迹；默认计时的竞争记录仍不对称，不能发布新合并 commit 的性能增量。

### 1. 版本与实验合同

拉取版本合并了 compact runtime descriptor、planner cost model、replica pruning、receive window
等工作；Qwen model、benchmark、config、isolated runner 相对 `a3e1b51b` 未变。
先用当前源码重新构建 Torch/Ascend910B native payload，避免用旧二进制解释新的 descriptor ABI。
旧 payload 已备份，构建成功后实验期间不再改动源码或 native 文件。

- 硬件：8 × Ascend910B3，单卡 64 GiB；CANN 9.2.0，Torch / TorchNPU 2.9.1，Transformers 5.5.3。
- 原始配置：EP8，8 层，H5120 / I1792 / T4096 / V32000，E48 / K8，shared-B1，capacity=1.5，
  Dirichlet alpha=0.1、route seed=42，model seed=42；每 rank 固定自己的 batch，逐步重复。
- Attention：16 个 query heads、2 个 KV heads、head dimension 320；无 dropout。
- 原 optimizer：BF16 model + FP32 main parameters / AdamW，LR=1e-3，weight decay=0.01，
  betas=(0.9, 0.999)，eps=1e-8。每个训练 step 均执行 zero-grad、F+B、dense grad sync、optimizer。
- 沿用 `rtol=0.02 / atol=0.002` 的全量 FP32 比较；归一化误差为
  `abs(actual-reference) / (0.002 + 0.02 * abs(reference))`，大于 1 才超原阈值。
- 一次只启动一个后端的八卡进程组。所有 rank 都记录输入、初始/最终 canonical 参数 SHA256、
  source/native identity、逐步 loss 和绝对时间，不只看 rank 0。

共同 identity map 摘要，算法为 `sha256(json.dumps(map, sort_keys=True).encode()).hexdigest()`：

```text
source_sha256: 4ef0fc3fc3ffc695e03d8491e4242524b6aeb3140395eb74460e83a0bd9fb0c4
native_sha256: 5c43d54081781262bb5903d369c9a6ad8e612374085c60cd5520a487ad841ff9
```

### 2. 已隔离的数值差异来源

#### 2.1 不更新参数，common 自重复仍有梯度差异

对完整八层 common 连续做三次固定权重 F+B，optimizer 对象仍按原配置构造，但诊断中显式
将 `step()` 替换为空操作；检查初始/最终全量参数 hash 相同。这个对照不属于训练性能测量。

- 三次全量 logits 在所有 rank 上逐位一致；标量 loss 的归约差异不超过 2.8610e-6。
- dense 同步前已有梯度差异：跨 rank 并集 77/83 个 canonical 张量非逐位一致，最大绝对差
  3.8147e-6。最末层 Q/K/V projection 等开始出现差异，而 lm_head 本地梯度在所有 rank
  的三次结果中逐位一致。
- BF16 dense 同步后，lm_head 梯度不再逐位一致。全部同步后梯度仍通过原阈值，不能把
  “通过 atol”理解为没有可被 optimizer 放大的差异。
- 当前 MegaMoe 两次固定权重 F+B 与 common reference 对照：初始/最终权重、输入完全相同，
  logits 逐位一致；所有 rank 的全量梯度均通过原阈值，两步最大绝对差分别为
  2.3842e-6 / 2.8610e-6。两后端 source/native 指纹一致。

这排除了“必须先更新参数才产生差异”，也不支持把所有早期差异都归因于 MegaMoe expert 路径。

#### 2.2 固定真实梯度输入，单独复放 dense collective

从上述 common 首步 CPU reference 读取每个 rank 的同步前梯度，按原顺序选取 67 个 dense
张量，四次各自复制同一组输入，执行原 `all_reduce_coalesced(SUM)` 和除以 8。
不运行模型、attention、MoE 或 optimizer，比较完整 lm_head 梯度的 163,840,000 个元素。
下表范围取第 2–4 次相对第 1 次复放；八个 rank 的统计相同。

| 独立复放模式 | 非逐位相同元素数 | 最大梯度绝对差 | 梯度符号翻转数 | 预测首步 AdamW 更新最大差 |
| --- | ---: | ---: | ---: | ---: |
| 原 BF16 归约 | 56,303,540–57,523,144 | 3.8147e-6 | 17,746–18,047 | 1.9589e-3 |
| BF16 + `torch.use_deterministic_algorithms(True)` | 0 | 0 | 0 | 0 |
| FP32 归约后 cast 回 BF16 | 5–7 | 2.9802e-8 | 0 | 1.2751e-6 |

因此 BF16 dense collective 本身就能产生足以改变微小梯度符号的差异；不需要 attention 输入
变化或 MegaMoe 参与。FP32 对照显著降低差异，但不是严格逐位确定，且没有把生产归约改为 FP32。
这里只比较了完整 lm_head 输出，不声称其余 66 个 dense 输出也全部通过逐位验证。

#### 2.3 独立 attention backward 也有非确定性

使用模型实际调用的 `run_qwen3_moe_flash_attention`，固定合成 BF16 Q/K/V 和输出梯度，
配置 `[batch=1, query_heads=16, kv_heads=2, sequence=4096, head_dim=320]`，
causal / BNSD / sparse_mode=2 / dropout=0，每种模式连续四次 F+B。

| 对照 | forward | dQ 最大绝对差 | dK 最大绝对差 | dV 最大绝对差 |
| --- | --- | ---: | ---: | ---: |
| 默认 | 逐位一致 | 2.4414e-4 | 3.90625e-3 | 3.90625e-3 |
| 确定性模式 | 逐位一致 | 0 | 0 | 0 |

这确认实际 attention adapter 所调用的反向路径可以独立产生差异，与完整模型最末层梯度的
观察一致。输入是**同形状合成张量**，不是捕获的真实 Q/K/V；上表绝对差不能直接换算成
整模型 loss 差，也没有完成每一条模型反向边的穷尽归因。

#### 2.4 为什么微小梯度差会变成约 0.002 的参数差

零动量状态的首步 AdamW（忽略实现舍入）为：

```text
w_new = w * (1 - lr * weight_decay) - lr * g / (abs(g) + eps)
```

当两次归约使接近零的 `g` 符号相反、且 `abs(g)` 相对 eps 足够大，更新差可接近 `2 * lr`，
本配置就是 0.002。把已经 BF16 归约的结果再转成 FP32 main grad，不能追回归约阶段丢失的信息。

历史小形状 common/common reference 提供了实际 optimizer 的交叉检查：初始 lm_head 权重
完全一致，最大梯度差 1.5259e-5、472 个符号翻转，更新后最大参数差 0.0019989014；CPU
首步公式与实际 BF16 更新的最大差仅 1.5259e-5。expert 0 的 gate-up 对照中，梯度最大差
9.5367e-7、4 个符号翻转，更新后最大差 9.765625e-4，CPU 首步公式与实际结果一致。
这组历史证据来自重构前后的 common，不是当前八层新训练；支持 AdamW 的数值放大机制，
没有发现 optimizer 实现错误。

### 3. 完整模型训练对照

#### 3.1 默认模式：自重复同样不能通过多步比较

按 common A1、MegaMoe B1、MegaMoe B2、common A2 顺序，各做 10 个真实 optimizer step。
第 2 步保存或比较全量 loss/logits、83 个梯度和 83 个更新后 canonical 模型参数；第 1、3–10
步记录全部 rank 的 loss，最终再记录全部模型参数 hash。FP32 master / moments 没有做逐步
全量 oracle，因此本轮不是所有 10 步、所有 optimizer 状态的完整精度验收。

四组配置、seed、对应 rank 的输入/初始参数、source/native map 完全相同；各组运行前后指纹
未变。各组自身的 replicated dense 初始/最终参数在八个 rank 间均完全一致，但跨独立运行
的最终参数不同。所有记录的 loss 均有限。

第 2 步对 A1 的全量比较如下；数值为跨八个 rank 的最大归一化误差：

| 候选 | loss/logits | 梯度 | 更新后模型参数 |
| --- | ---: | ---: | ---: |
| common A2 | **16.0054，失败** | 0.029711，通过 | **1.876729，失败** |
| MegaMoe B1 | **15.9694，失败** | 0.022477，通过 | **1.856529，失败** |
| MegaMoe B2 | **15.4316，失败** | 0.029848，通过 | **1.846680，失败** |

forward 的失败来自 logits，三组最大绝对差均为 0.125；第 2 步 loss 本身仍在原阈值内。
更新后参数最大绝对差均为 0.00390625。common/common 已出现同级别失败，因此不能把
这些独立轨迹间的失败单独归因于新合并的 MegaMoe 改动，也不能继续称默认多步精度已通过。
本轮第 2 步以 common A1 为统一 anchor，B1/B2 未另存全量候选梯度做直接互比；
MegaMoe 自重复结论来自全 10 步 loss 和最终全部模型参数 hash，不冒充其第 2 步完整互比。

下面是记录后在 CPU 上计算的八 rank loss 均值，不是额外施加给训练的 loss reduction：

| optimizer step | common A1 | MegaMoe B1 | MegaMoe B2 | common A2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 10.539054 | 10.539055 | 10.539054 | 10.539054 |
| 2 | 0.989696 | 0.989683 | 0.989798 | 0.989789 |
| 3 | 0.083473 | 0.083493 | 0.083478 | 0.083446 |
| 4 | 0.072547 | 0.072488 | 0.072601 | 0.072690 |
| 5 | 1.501870 | 1.497062 | 1.450180 | 1.484002 |
| 6 | 1.264459 | 1.238857 | 1.224548 | 1.253639 |
| 7 | 5.115752 | 5.085871 | 5.020524 | 5.035926 |
| 8 | 7.681335 | 7.767803 | 7.860365 | 7.721008 |
| 9 | 8.563452 | 8.310156 | 8.119903 | 8.752827 |
| 10 | 8.604605 | 8.206761 | 7.998721 | 8.582307 |

对每个对应 rank 按原阈值比较 loss，A1/A2、B1/B2、A1/B1、A1/B2 都从第 5 步首次失败。
仅看最终 rank-zero loss 或跨 rank 平均值会漏掉部分更早的分歧；原始 JSON 保留全部 rank。

#### 3.2 确定性模式对照

唯一执行模式变化是启动时调用 `torch.use_deterministic_algorithms(True)`，仍为原 LR=1e-3，
同样完成 A1/B1/B2/A2 四组各 10 步，使用同一第 2 步 snapshot 合同。
它不是基于警告忽略或放宽阈值的“确定性”；验证结果如下：

| 比较 | 八 rank × 十步 loss | 最终全部 canonical 模型参数 | 第 2 步全量对照 |
| --- | --- | --- | --- |
| common A1/A2 | 逐位一致 | hash 全部一致 | logits、梯度、更新后参数全部逐位一致 |
| MegaMoe B1/B2 | 逐位一致 | hash 全部一致 | 两组分别与 common A1 比较；未另做候选全量互比 |
| common A1 / MegaMoe B1 或 B2 | 第 6 步首次有 rank 超原 loss 阈值 | 不同 | logits 失败；梯度和更新后参数通过 |

跨 backend 第 2 步最大 logits 绝对差为 0.0625、归一化误差 10.239767；梯度最大绝对差
3.0518e-5、归一化误差 0.014966；更新后参数最大绝对差 0.00390625，但这次发生在更宽
reference-relative 容差的元素上，最大归一化误差 0.785867，故通过原阈值。
绝对最大差相同，不意味着是否通过也相同。

确定性模式恢复了本组实测的 backend 内部重复性，**没有消除可重复的跨 backend 差异**，
也没有消除 loss 回升。八 rank 的均值从第 4 步 common/MegaMoe 的 0.072696/0.072610，
升到第 10 步的 8.620320/8.424216。不能把确定性模式当成跨 backend 精度修复或收敛修复。

#### 3.3 确定性首步定位与低学习率控制

补做 common / MegaMoe 各一次确定性首步训练，保存首次 optimizer 更新之前的完整梯度，
再比较更新后全部模型参数。输入、初始参数与 source/native 指纹完全相同：

| 首步阶段 | 八 rank 的结果 |
| --- | --- |
| loss / logits | 全量逐位一致 |
| 67 个 dense 梯度 | 全量逐位一致 |
| expert 梯度 | 跨 rank 非逐位张量并集为全部 16 个 expert 张量；最大绝对差 9.5367e-7 |
| 更新后 dense 模型参数 | 全量逐位一致 |
| 更新后 expert 模型参数 | 最大绝对差 1.2207e-4，最大归一化误差 0.052787，通过原阈值 |

首步梯度最大归一化误差为 0.000476，仍通过原阈值；但“通过”不代表首步 expert 权重相同。
结合第 3.2 节可重复的第二步 logits 失败，下一步定位应从 **expert dW 的累加/舍入路径**
开始，而不是继续泛化怀疑 attention 或 dense 同步。这里没有直接检查每条隐藏状态的 dX，
不能由 dense 参数梯度一致推导任意中间张量都逐位相同。

代码候选范围是 common 的 `components/functional/npu_grouped_swiglu.py` 中
`_NPUGroupedMatmul.backward`（grouped dW），与 MegaMoe 自定义 dW、physical copy / replica
贡献的 FP32 聚合及最终 cast。上述证据尚不能唯一归因到分块累加顺序、copy 切分或某一侧
舍入，也没有证明哪一侧数学结果错误；不能为贴齐 reference 而提前把 replica contribution
降为 BF16，破坏已有的 FP32 聚合合同。

再做两次默认执行模式的 common 控制，仅将 LR 改为 1e-4，不开启确定性。它们只记录
10 步 loss 和输入/初始/最终参数指纹，不做全量梯度 oracle。两次在所有 rank 上均逐步
下降，八 rank 均值如下：

| step | common A1，LR=1e-4 | common A2，LR=1e-4 |
| --- | ---: | ---: |
| 1 | 10.539054 | 10.539054 |
| 4 | 6.884703 | 6.884747 |
| 7 | 3.079109 | 3.079263 |
| 10 | 0.683563 | 0.683609 |

十步对应 rank 的 loss 全部通过原阈值，最大绝对差 5.0521e-4、最大归一化误差 0.012971；
loss 和最终模型参数仍非逐位一致。该控制支持学习率敏感性，不证明长期收敛，也没有验证
低 LR 的跨 backend 全量精度。原 LR=1e-3、原 rtol/atol 和默认执行模式均未修改。

### 4. 计时与竞争的可解释边界

诊断 worker 在每步 barrier / device synchronize 后记录 `time.time_ns()` 和
`time.perf_counter_ns()`，执行与原 benchmark 相同的训练 step，再同步并记录结束时间。
父进程不断保存带采样起止绝对时间的 `npu-smi info` 原始输出，并尽可能记录 PID 命令行。
每步用八个 rank 最早开始到最晚结束的区间，保守匹配与该区间重叠的采样帧，排除自己的
八个 worker PID，留下外部进程。

第 2 步全量 snapshot 用于精度诊断，不参与稳态计时；第 4–10 步不做 snapshot。
统计口径为每步取 rank-max，再取七步 median。快照落盘与比较在稳态之外，但这仍是带
额外 warmup 诊断的运行，不冒充原 shell 的无改动重跑。固定权重测试计时也不作为性能结果。

当前默认组观测值如下。四组稳态窗口均没有采样失败帧，但竞争记录不对称：

| 默认组 | 七步 median（ms） | 与稳态窗口相交的外部 PID 采样 |
| --- | ---: | --- |
| common A1 | 1289.887 | 七步全部命中，每步 45–46 个外部 PID |
| MegaMoe B1 | 714.911 | 七步全部命中，NPU 4 上的 PID 54264，约 136 MiB 驻留 |
| MegaMoe B2 | 715.593 | 七步均未命中 |
| common A2 | 1206.259 | 七步均未命中 |

同 backend 的 common A1/A2 相差约 83.6 ms，与采样到的环境差异共现。B1 的小额进程驻留
不证明它在执行计算。B2/A2 可作为较干净的相邻观测，但默认组没有两轮共同干净的 ABBA，也没有
匹配的旧版本独占重跑，因此这里不发布新合并 commit 的加速/回退比例。

单列的确定性组四次 median 为 common A1 **1220.890 ms**、MegaMoe B1 **727.858 ms**、
MegaMoe B2 **727.702 ms**、common A2 **1219.385 ms**；其 28 个稳态窗口均未采到外部
PID，且没有采样失败帧。它提供了当前版本较稳定且可审计的对照，但执行模式不同，不能
混入默认组统计，也不能与历史默认结果相减来估算本轮合并的性能增量。

采样未命中外部 PID 不等于硬件独占；约 0.5 秒的采样空隙、命令自身采样区间及短命进程都
有盲区。外部 PID 与慢步重叠也只是竞争证据，不能仅凭共现量化其因果耗时。未终止或修改
任何外部 Jenkins / NPU 作业。

首次 `train_default_a1` 尝试在模型初始化后因 `npu-smi` 采样超过 8 秒而被临时父脚本终止，
未完成训练 step，不计入结果。保留该尝试到 `train_default_a1_census_timeout/` 和
`training_default.log`；修正临时采样脚本，超时返回 `returncode=124 / coverage=unknown`，
继续训练并统计缺测帧，不能把缺测当作无竞争。重跑日志为 `training_default_retry.log`。

### 5. 验证与原始证据

本轮已有 CPU 回归：MegaMoe UT、module UT 和 Qwen model/config/snapshots/isolated UT，
`193 passed / 202 subtests`，26.95 秒。当前 native 完整构建成功；固定权重 common 和
MegaMoe、默认/确定性训练矩阵、确定性首步和低 LR 控制共 14 组模型实验均正常退出。
112 份 rank 结果的输入、初始参数、模型配置和运行前后 source/native identity 一致；
optimizer LR 的控制差异单列。正常退出不代表数值对照通过，失败项见第 3 节。

本地保留的 `megamoe_qwen_multistep_diagnosis_20260908.json` 包含全部 rank 的
逐步 loss、原阈值比较结果、时间/竞争摘要、首步定位、独立复放，以及 318 份保留原始文件的
SHA256。该摘要与原始 JSON、全量模型/梯度张量均不入 git，存放在下述本地诊断目录；
reference 的逐张量 hash 元数据也列入本地校验清单。

本轮原始日志、全量 CPU references、逐 rank / step JSON、采样原文与临时诊断脚本均在：

```text
/tmp/qwen-multistep-diagnosis.kI6Fza/
  megamoe_qwen_multistep_diagnosis_20260908.json
  native_build.log
  previous_native_lib/
  pulled_cpu_ut.log
  historical_adamw.json
  fixed_common/                  fixed_common_reference/
  fixed_mega/
  train_default_a1/              train_default_b1/               train_default_b2/       train_default_a2/
  train_deterministic_a1/        train_deterministic_b1/         train_deterministic_b2/ train_deterministic_a2/
  first_deterministic_common/    first_deterministic_mega/
  train_low_lr_a1/               train_low_lr_a2/
  collective_bf16/               collective_deterministic/       collective_fp32/
  attention_model_shape_default.json
  attention_model_shape_deterministic.json
  worker.py                     run_case.py                     summarize.py
  collective_replay.py           attention_replay.py             run_training.py
  run_followup_controls.py       training_overview.py
```

临时脚本复用当前模型、训练 step、snapshot oracle，不改生产模块；大体积 CPU references 不入 git。
`attention_default.json` / `attention_deterministic.json` 是早先 40/8 heads、D128 的探索性复放，
不作为本报告实际模型形状的证据。早期 fixed-common census 的结构化 PID 解析遗漏了表格竖线，
最终汇总已从保留的完整 raw table 重新解析，不能用旧帧的空 `processes` 字段声称无竞争。

本机复现实验入口如下。脚本拒绝覆盖已有 case/reference，因此重跑时先通过 `mktemp -d`
建立新的诊断目录并放入同版本脚本，不直接复用已完成的结果目录：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HYPER_PARALLEL_PLATFORM=torch
export OMP_NUM_THREADS=1
QWEN_DIAG_PY=/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python
QWEN_DIAG_DIR=/tmp/qwen-multistep-diagnosis.kI6Fza

"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_training.py" --group default
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_training.py" --group deterministic
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/run_followup_controls.py"
"$QWEN_DIAG_PY" "$QWEN_DIAG_DIR/training_overview.py"
```

以上入口顺序执行，不同时启动多个本轮八卡进程组。parent 为每组选择独立 MASTER / SHMEM
端口，HCCL NPU socket range 为 `62000-62063`。跟随控制中的低 LR 两组不选择任何 tensor
snapshot step；`loss_only_no_tensor_references` 仅满足临时 worker 复用原 parser 的单后端参数
要求，不会生成或读取不存在的张量 reference。对应 loss 比较、初始/最终 hash 和运行身份
由逐 rank JSON 完成，不宣称该控制验证了所有梯度或 optimizer 状态。

### 6. 本轮后的优先级

后续已完成实际第 8 层的固定输入 / dY 复放，见
[expert dW 切分与舍入诊断](#expert-dw-切分与舍入)：B0 单波与 native
逐位一致，B1 residual 只在复制过的 expert；实际 owner FP32 聚合和最终 cast 与 CPU 重算
一致。该后续报告已将下一优先级调整为 AdamW FP32 更新与 BF16 参数写回的放大定位。
以下保留本轮结束时的原始优先级，不能将完成的单层定位外推为完整模型多步精度已通过。

1. **先拆 expert dW 的可重复残差。** 固定相同权重、路由和实际层输入/输出梯度，逐段比较
   原生 grouped dW、自定义 dW、physical copy 部分和 owner 聚合结果；用 B0/B1 对照区分
   kernel 累加顺序与 replica 切分/最终 cast。必要时用受控的小块高精度计算判定舍入误差，
   不提前削减 FP32 contribution 精度，也不将本轮 tensor-level 定位冒充具体 kernel 的定罪。
2. **精度与性能分开验收。** 确定性模式可用于复现数值问题，但当前跨 backend 多步检查仍失败。
   修复后重新验证原配置的 logits、梯度、更新后模型参数，并扩展 FP32 main/moments 与更多
   更新步的状态对照。低 LR 或放宽 rtol/atol 都不能替代这项验收。
3. **再评估默认重复性的低成本改善。** FP32 dense 归约是有独立证据支持的候选，不是已经
   验证的整网修复。若实现，应核对现有 main-grad 生命周期和有界 bucket，避免多一套长驻
   全量梯度池；同步评估通信量、峰值 HBM、snapshot/optimizer 的数值合同，不能静默改 baseline。
4. **稳定测量之后再选调度优化。** 若要判断拉取 commit 的性能增减，需要旧/新代码分别匹配
   native payload，在相同执行模式、相同配置下取得可审计的八卡配对窗口，再采全 rank profile。
   本轮既不终止其他作业，也不从进程驻留或离散采样推断严格独占和具体竞争耗时。

本轮仅将诊断报告入仓，证据摘要保留在本地；不实施上述候选修复，不新增结构性 UT；按约定直接本地
commit，不使用 autogit，不 push。

## Expert dW 切分与舍入

本轮接续 [多步数值诊断](#多步数值与计时诊断) 的最高优先级，
拆分 native grouped dW、MegaMoe 单波/多波，以及 B1 physical-copy 聚合的影响。
基于 `caf71330`，生产 Python / native 相对 `ee5a6965` 未变，沿用上一轮重新构建的 payload。
本轮不修改 runtime、optimizer、学习率或精度阈值，也不发布性能数据。

结论：本次真实层复放中，B0 单波与 native 的输出、dX、dW 全部逐位一致；B0 多波差异仅在
wave 切开的 expert，B1 差异仅在复制过的 expert。B1 的实际 FP32 owner 聚合和最终 cast 均与
CPU 重算逐位一致。证据支持切分改变累加/舍入路径，未发现本次 replica 聚合错误；
**这不是完整模型多步精度已修复的结论**。

### 1. 固定的不是合成随机梯度，而是真实首步的层边界

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

### 2. 全量 expert 梯度对照

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

### 3. 差异精确落在哪些 expert

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

### 4. FP64 点积与代码路径交叉检查

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

### 5. 验证范围、原始数据与复现

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

### 6. 完成此步后的优先级

后续已完成实际 fused AdamW 的 main/moments/BF16 写回复放，见
[AdamW 放大诊断](#adamw-与-bf16-写回)：等梯度的两种参数布局逐位一致，
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

## AdamW 与 BF16 写回

本轮接续 [expert dW 切分诊断](#expert-dw-切分与舍入)，基于
`megamoe-sun@7891f94d`。按当前选择保留 replica / FP32 partial 路径，不实施高代价的
native-exact dW 重算或重新切分。生产 Python / native 相对 `ee5a6965` 未变。

结论：在真实第 8 层权重/梯度的独立 optimizer 复放中，**没有发现 AdamW、参数布局或
FP32 main 生命周期错误**。同梯度时两种布局的 main/moments/model 全部逐位一致；换入
B1 梯度后，差异经小梯度区域的 AdamW 更新敏感度和 BF16 写回边界两阶段作用，首步出现
738 个不同的 BF16 参数元素。两侧首步写回均与之前各自完整模型的该层结果逐位一致。
这定位了本层首步参数差异，尚未完成下一次 forward 的因果干预或整网多步精度验收。

### 1. 实验合同：真实梯度、真实 optimizer，不运行模型反馈

使用上一轮保存的原配置确定性首步数据：第 8 层（索引 7）、EP8、H5120 / I1792、E48 / K8，
每 rank 六个 home experts。初始权重相同，native / B1 梯度来自固定实际 hidden、route、dY
的复放，全部数据指纹与保留的 capture 对上。

三组各构造独立 optimizer，按顺序在八个 rank 执行：

| 对照 | 实际参数布局 | 输入梯度 | 目的 |
| --- | --- | --- | --- |
| native_ref | NativeMoeExperts，三个 EP-sharded DTensor 参数 | native | 复现原始更新 |
| mega_same_grad | MegaMoeExperts，两个 packed 本地参数 | 同一份 native 梯度 | 排除布局/optimizer 差异 |
| mega_b1_grad | MegaMoeExperts，两个 packed 本地参数 | B1 实际梯度 | 只改变 dW |

复用 `qwen_moe_benchmark.py::_build_workload` 的原 optimizer factory：
`Float16OptimizerWithFloat16Params` → `ChainedOptimizer` → 项目 `core.optimizer.adamw.AdamW`
→ `torch._fused_adamw_`。不是用 CPU AdamW 或另一个优化器替代设备执行。

LR=1e-3、weight decay=0.01、betas=(0.9, 0.999)、eps=1e-8、amsgrad/maximize=False，
BF16 model / FP32 main / FP32 moments。沿用确定性模式和原 `rtol=0.02 / atol=0.002`。

每个 optimizer 连续更新三次，**每次重复喂同一份捕获梯度**，不做新的 forward、backward、
MoE 或 dense gradient synchronization。共 24 个独立 optimizer 实例、72 次 rank-local 更新。
它用于隔离更新/写回及状态延续，绝不是完整模型三步训练、收敛或性能测试。

### 2. 已排除的实现问题

1. 三组每步 BF16 model grad 转 FP32 main grad 都与输入梯度的直接 FP32 转换逐位一致；
   model.grad 随后被消费，main_grad 没有意外残留，下一步 zero-grad 正常清理。
2. fused AdamW 的 group step 依次为 1 / 2 / 3。第二、三步的 FP32 main 在更新前与上一步
   更新后的全量指纹相同，没有重新从 BF16 model 初始化 main。
3. 每步 BF16 model 写回与实际 FP32 main 的直接 BF16 cast 全量逐位一致。
4. native_ref / mega_same_grad 的 main、一阶/二阶 moment、BF16 model，在八 rank、三次
   更新中全量逐位一致，排除本次 packed / transpose / DTensor 布局造成不同 optimizer 结果。
5. native_ref / mega_b1_grad 首步写回的 gate-up / down，各自在八 rank 上与历史完整模型
   `first_deterministic_common` / `first_deterministic_mega` 的第 8 层最终参数指纹完全相同。
   原权重及 source/native identity 同样匹配，不是只对齐了某一个最大误差。

全量比较覆盖该层 1321205760 个逻辑权重元素，每个状态阶段均比较完整 gate-up / down。
B1 与 native 的任何状态差异，都只出现在输入梯度不同的 40566 个坐标内；输入梯度相同处
没有出现新的 optimizer 数值差异。以上是本层、固定梯度三次更新的结论，不外推到任意缺失
梯度、checkpoint、参数 offload 或其他模型的 optimizer 生命周期。

### 3. 全量状态差异

以下为 mega_b1_grad 相对 native_ref，元素数跨八 rank 求和，最大绝对差取全量最大值。
三步重复的是首步梯度，因此输入 dW 最大差始终为 4.7683716e-7。

| 固定梯度更新次数 | FP32 main 不同元素 | FP32 main 最大差 | BF16 model 不同元素 | BF16 model 最大差 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 40245 | 6.4931810e-6 | 738 | 1.2207031e-4 |
| 2 | 40462 | 1.2986362e-5 | 1338 | 1.2207031e-4 |
| 3 | 40552 | 1.9479543e-5 | 1940 | 1.2207031e-4 |

每一步的一阶、二阶 moment 都有 40566 个不同元素，与输入梯度差异坐标数一致：

| 更新次数 | 一阶 moment 最大差 | 二阶 moment 最大差 |
| --- | ---: | ---: |
| 1 | 4.7683898e-8 | 1.0618242e-13 |
| 2 | 9.0600224e-8 | 2.1225903e-13 |
| 3 | 1.2922465e-7 | 3.1823155e-13 |

所有比较状态有限，仍通过原 tensor 阈值。三步 main 最大归一化误差为 0.00878502，model
为 0.0527872，均小于 1。这个宽容差通过不代表 BF16 参数相同，也不能撤销此前完整模型
第二步 logits、后续 loss 的失败记录。

### 4. 第一阶段：AdamW 对小梯度敏感

零初始化 moments 的首步理想更新为：

```text
u(g) = lr * g / (abs(g) + eps)
w_main_new = (1 - lr * weight_decay) * w_initial - u(g)
du/dg = lr * eps / (abs(g) + eps)^2
```

小梯度区域的最大斜率为 `lr / eps = 100000`。因此是否符号翻转并不是唯一条件；
同号的小梯度也可能把约 1e-10 的梯度差变成约 1e-6 的更新差。

首步 main 最大差的真实位置是 rank 5、packed gate-up 展平索引 40626214：

| 项目 | native | B1 |
| --- | ---: | ---: |
| 初始权重 | 0.01025390625 | 0.01025390625 |
| 梯度 | -2.5756889954e-9 | -2.4738255888e-9 |
| 更新后 FP32 main | 0.0104586184025 | 0.0104521252215 |
| BF16 model | 0.01043701171875 | 0.01043701171875 |

此处梯度差只有 1.0186341e-10，main 差为 6.4931810e-6，但 BF16 写回仍相同。
说明“最大 dW 差的位置”“最大 main 差的位置”“最大 BF16 参数差的位置”不是一回事；
不能把三个全局最大值直接相除来解释逐元素放大。

六个符号翻转位置的梯度绝对值最大仅 5.5252e-11，远小于 eps；首步 main 最大差为
5.7875e-6。这组 expert 残差**没有实测出接近 2*LR=0.002 的更新差**，不能把此前默认
dense collective 非确定性控制中的结果照搬到本组。

### 5. 第二阶段：跨 BF16 舍入边界

BF16 写回经常掩盖小的 main 差异，但当两个 main 值位于同一舍入边界的两侧，就会产生
不同的模型参数。一个首步最大 BF16 差的实际例子：rank 5、down 展平索引 19032926。

| 项目 | native | B1 |
| --- | ---: | ---: |
| 初始权重 | -0.026611328125 | -0.026611328125 |
| 梯度 | -2.0489096642e-8 | -2.0372681320e-8 |
| 更新后 FP32 main | -0.0259390473366 | -0.0259403046221 |
| BF16 model | -0.02587890625 | -0.0260009765625 |

两个 BF16 值的中点是 **-0.02593994140625**，两个 main 值恰在两侧：
梯度差 1.1641532e-10 → main 差 1.2572855e-6 → BF16 差 1.2207031e-4。
这里的 BF16 cast 与 CPU 直接 cast 完全一致，不是拷贝精度错误。

首步 738 个 BF16 差异中，671 个相差一个相邻 BF16 间隔，均核对到 main 跨越中点；
另外 67 个相差多个间隔，不能笼统称作全部“一个 ULP”。仅在 BF16 不同的坐标上计算
`BF16 参数差 / main 差`，中位数约 7.29，最大约 4681.14，也有小于 1 的位置。
量化既可能掩盖，也可能扩大差异，不是每个元素都被统一放大。

FP32 main 并未被 BF16 写回覆盖。后续真实训练的反馈路径应区分为：

```text
dW 残差 → FP32 AdamW 更新差 → BF16 模型权重差 → 下一次 forward / backward → 新的梯度差
```

本轮直接验证到 BF16 写回。固定梯度三次更新的 main 差近似线性增加，不能冒充上面最后
两条模型反馈边的实测，也不能据此解释此前所有层、多步 loss 回升的幅度。

### 6. FP64 公式复核

保存全部 40566 个输入梯度不同坐标的初始权重、两组梯度及每步实际 main/moments/model。
这不是抽样：全量状态比较已经确认差异没有出现在这些坐标之外。

在 CPU 上使用 FP64 按固定梯度的理想 AdamW 公式计算：

```text
d = 1 - lr * weight_decay
m_t = (1 - beta1^t) * g
v_t = (1 - beta2^t) * g^2
w_t = d^t * w_0 - lr * g / (abs(g) + eps) * sum(d^i, i=0..t-1)
```

两组取较大值，实际设备 main 到 FP64 理想值的最大距离分别为
2.291e-9 / 6.445e-9 / 5.962e-9；三步一阶 moment 最大距离不超过 6.345e-12，二阶不超过
4.780e-16。该有限精度实现误差远小于本组跨梯度的最大 main 差，支持正常 AdamW 数值
敏感度的解释，没有发现足以解释当前差异的错误更新公式。该公式复核不要求 fused FP32
运算与 FP64 理想公式逐位一致，也未将新阈值代替原全量 tensor 比较。

### 7. 证据、复现和下一步

全部运行正常退出，源码/native identity 前后、跨 rank 与之前捕获一致。生产代码和 native
未改动，不需要重构建或新增结构性 UT；本轮不提供性能/HBM 测量，也不改变 LR、eps、
确定性默认值或原精度阈值。原始 JSON 和张量不入 git，保留在：

```text
/tmp/qwen-adamw-diagnosis.DI8gFa/
  worker.py          run_case.py       probe_util.py
  analyze.py         audit.py
  summary.json       audit.json        analysis.log
  replay/launch.json
  replay/worker.log
  replay/census.jsonl
  replay/result/rank{0..7}.json
  replay/result/rank{0..7}/native_ref/step{1..3}.pt
  replay/result/rank{0..7}/{native_ref,mega_same_grad,mega_b1_grad}/step{1..3}.json
  replay/result/rank{0..7}/mega_b1_grad/step{1..3}_changed_gradient_points.pt
```

native_ref 保留各步全量 main/moments/model 张量；候选保留全量比较、指纹，以及 B1 的
全部梯度差异坐标状态，不额外保存三套完整候选状态。`audit.py` 验证 24 个 optimizer 实例
的输入与原 capture、首步写回与两组完整模型 reference 的一致性，另记录脚本和摘要 SHA256。
`summary.json` SHA256 为 `710bff3de26c64aa97ff50f0fbfc9d0cd2d4dec460e6cd9a3d986786d50c5f4a`。

本机 venv Python 为 `/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python`，启动
`run_case.py --directory <新诊断根目录>/replay --python <venv Python>`；
先 source CANN 和当前 payload 的 `set_env.bash`，设置 `HYPER_PARALLEL_PLATFORM=torch`、
`OMP_NUM_THREADS=1`。入口无额外 worker 参数，会顺序运行三组固定梯度更新。
脚本读取上述上一轮 capture 的绝对路径；`analyze.py` / `audit.py` 汇总同目录下的 `replay`。
重跑须用 `mktemp -d` 建立新目录、复制脚本并保持对应目录布局；脚本拒绝覆盖已有结果。

后续已完成 [首步 expert 参数干预](#参数干预与层间传播)：只恢复
不同的 expert BF16 参数，MegaMoe 第二次 logits/loss 与历史 native 逐位一致；在 native
反向注入相同差异，也与 MegaMoe 对应组的各层输出逐位一致。末层的 738 个差异值已链接到
下一次 forward，但完整模型后续多步 loss gate 仍未通过。以下保留本轮结束时的原始优先级；
当前下一项转为显式 eps 稳定性控制，不改默认参数或原阈值。

完成此步时的优先级：

1. **下一次 forward 的参数干预。** 固定同一更新后 reference 状态，只替换已确认的 expert
   BF16 参数差异，比较 logits/loss；与恢复这些参数的对照一起量化因果贡献，再扩展到其余层。
   当前只直接链接了第 8 层首步，尚不能声称它单独解释整模型第二步 logits 失败。
2. **保留当前 AdamW 和 FP32 main 生命周期。** 不把正常舍入敏感度当作已发现的 optimizer
   bug，不重置 moments、不把 main 从 BF16 重载，也不为逐位对齐引入高代价 dW 重算。
   如后续评估 eps/LR 稳定性，只能作为显式控制，不能静默修改 baseline 或冒充精度修复。
3. **性能工作继续独立验收。** 原默认 attention / dense collective 非确定性、旧/新代码
   配对计时窗口等边界仍在；不从本轮 optimizer 复放推导性能、收敛或完整多步精度已通过。

本轮只将报告与索引更新直接本地 commit，不使用 autogit，不 push。

## 参数干预与层间传播

本轮接续 [AdamW 与 BF16 写回诊断](#adamw-与-bf16-写回)，基于
`megamoe-sun@f70d3fdf`。生产 Python / native 相对 `ee5a6965` 未变；保留当前 replica、FP32
partial 聚合和 optimizer 实现，不做 native-exact dW 重算，不改变默认超参数或精度阈值。

关键结果：在 MegaMoe 上，把首步更新后不同的 expert BF16 参数恢复为 native 值，第二次
logits/loss 就与历史 native 全量逐位一致；重新注入差异则重新出现分歧。末层已经定位的
738 个参数差异单独就能使 32 个 logits 元素超过原阈值，但不能代表其余七层的贡献。
反向在 native 注入同一组差异，也与 MegaMoe 对应组的全部层输出、norm、logits/loss
逐位一致。本轮验证的是首步参数到第二次 forward 的因果关系，不是完整模型多步精度修复。

### 1. 实验合同与隔离方式

沿用原始八层 Qwen example、8 × Ascend910B3、Torch/TorchNPU 2.9.1、CANN 9.2.0、
Transformers 5.5.3：EP8，H5120 / I1792 / T4096 / V32000，E48 / K8，shared-B1，
capacity=1.5，Dirichlet alpha=0.1、route/model seed=42，固定每个 rank 的原 batch。
BF16 model / FP32 main / FP32 moments，AdamW LR=1e-3、weight decay=0.01、
betas=(0.9, 0.999)、eps=1e-8。确定性模式仍只是诊断控制，不改生产默认值。

每个 backend 用独立进程组和自己的模型、optimizer；两个 EP8 进程组顺序运行，不同时驻留：

1. 用当前 benchmark 的 `_build_backend_model`、`_build_workload`、`_optimizer_step`
   执行一次真实完整训练 step，包括 backward、dense 同步和实际 fused AdamW 更新。
   核对输入、初始参数、首步 loss/logits、更新后全部 canonical 参数和 source/native identity
   与之前确定性首步记录一致。
2. 从 MegaMoe 首步更新后参数中提取相对历史 native reference 的**全部差异坐标和值**。
   dense 参数必须全量相同；只有 expert 的实际不同元素用于注入或恢复，不改其他值。
   末层坐标、native 值、B1 值逐位核对上一轮实际 AdamW 的 BF16 写回结果。
3. 固定首步状态，切换 BF16 expert 参数后，再调用原训练 step，但仅在临时脚本中把
   `optimizer.step()` 替换为空操作。保留正常的 F+B、dense 同步、zero-grad 以及计算图释放，
   避免改用另一条 inference 路径；本轮只分析 forward，不把这些重复 F+B 计作训练更新。
4. 每组前后核对**全模型**参数指纹：选中层恰好等于 B1 首步值，其余恰好等于 native 首步值，
   执行探针前后参数不变。全部探针前后的 FP32 main、两个 moments 全量 hash 和 group step
   也比较；不从 BF16 参数重载 main，不推进 optimizer，不使用下一轮训练反馈。

这种只替换 BF16 model 而保留原 main/moments 的状态是刻意的因果干预，不能作为一致的训练
checkpoint 继续 optimizer step。实验结束销毁模型；不把它当成恢复 optimizer 的做法。

MegaMoe 共 14 个探针：原始未插桩状态、全层注入、全量恢复、八个逐层单独注入、
仅前七层注入、再次全量恢复、再次全层注入。native 共五个：reference 未插桩、reference
插桩、仅末层注入、全层注入、再次恢复。插桩保存八个 decoder layer 输出、最终 norm 输出、
loss 和全量 logits；每个 backend 都有无插桩的 forward 对照。

原比较仍是 FP32 `abs(actual-reference) / (0.002 + 0.02 * abs(reference))`，大于 1 超限。
另报逐位不同元素数与 RMSE，用于解释误差分布，**不替换原 gate**。logits 覆盖八 rank 的
1048576000 个元素，计数求和、最大误差取全局最大，loss 按对应 rank 比较，不只看均值。

### 2. 实际干预规模

全部 dense 首步参数相同。八层 routed experts 合计 10569646080 个逻辑参数中，有
**11227 个 BF16 元素不同**；每层的最大绝对差都为 1.220703125e-4。

| 模型层（从 1 开始） | BF16 参数差异元素数 |
| --- | ---: |
| 1 | 2046 |
| 2 | 1380 |
| 3 | 1635 |
| 4 | 1133 |
| 5 | 1535 |
| 6 | 1518 |
| 7 | 1242 |
| 8 | 738 |
| 合计 | 11227 |

这不是随机扰动、对最大差位置的抽样或给整层统一加一个偏移；每个注入值都是原 B1 首步
实际参数值。gate/up packed 坐标显式映射到 native 的 w1/w3，down 映射到 native w2；
每组参数 hash 验证该映射的全量结果，不能假设 canonical 的临时拼接张量是可写参数别名。

### 3. 第二次 logits/loss 的变化

以下均为 MegaMoe 执行路径，与原 native 首步更新后的第二次 forward 比较。只注入第 N 层
表示其余七层及全部 dense 参数都是 native reference。表中的 RMSE 覆盖全部 rank/logits。

| 注入范围 | logits 最大差 | 最大归一化误差 | logits 超限元素 | logits RMSE | 对应 rank loss 最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无，全部恢复 native | 0 | 0 | 0 | 0 | 0 |
| 仅第 1 层 | 0.125 | 10.0202 | 111669098 | 4.17404e-3 | 2.13206e-4 |
| 仅第 2 层 | 0.0625 | 9.42333 | 91498210 | 3.78021e-3 | 1.44958e-4 |
| 仅第 3 层 | 0.0625 | 8.59579 | 70015900 | 3.35633e-3 | 2.11895e-4 |
| 仅第 4 层 | 0.0625 | 6.65221 | 48527531 | 2.91790e-3 | 1.59323e-4 |
| 仅第 5 层 | 0.0625 | 6.24185 | 28363224 | 2.47088e-3 | 1.52826e-4 |
| 仅第 6 层 | 0.125 | 4.17603 | 11511821 | 2.01447e-3 | 1.58131e-4 |
| 仅第 7 层 | 0.0625 | 2.91826 | 1195403 | 1.49118e-3 | 7.81417e-5 |
| 仅第 8 层 | 0.0625 | 1.54400 | 32 | 4.01474e-4 | 3.85046e-5 |
| 前七层，恢复末层 | 0.0625 | 10.2398 | 111605854 | 4.17287e-3 | 1.08659e-4 |
| 全八层 | 0.0625 | 10.2398 | 111605047 | 4.17286e-3 | 1.15275e-4 |

所有注入组的本次 loss 差仍通过原阈值，不能称为第二步 loss gate 失败；logits 则都已失败。
末层单独注入产生 71411298 个不同 logits，但只有 32 个超限；全层注入产生 801878157 个
不同 logits，111605047 个超限。不能只凭最大绝对差相同（0.0625）说它们影响相当。

各层的因果影响**不可相加**：只注入第 1 或第 6 层的最大绝对差反而大于全层一起注入。
本组浅层单独注入的 logits RMSE 更大，但各层扰动坐标、数量和数值不同，不能由此估计
统一的“每多经过一层放大多少倍”。恢复末层也不能修复已经由其余层引起的第二步分歧。

### 4. 隐藏状态与恢复对照

八个单层干预、每个八 rank，首个不同 decoder 输出都恰好出现在被注入层；它之前的层
输出逐位一致。特别是末层 738 个参数干预，前七层输出仍相同，差异从第 8 层输出出现，
再传播到 norm、logits/loss。这补齐了上一轮仅验证到 BF16 写回、尚未直接验证的下一条边。

只注入第 1 层时，该层输出 RMSE 为 2.73177e-4，到第 8 层输出变为 9.42436e-3；后七层
参数均保持 native 值，也没有新的 optimizer 更新。这直接观察到同一次 forward 中扰动沿层
传播，不能把这段变化误写为多次 AdamW 更新累积，或由不同层张量尺度推导统一的放大倍率。

MegaMoe 全量恢复后，logits/loss 与之前 native 第二步完整 CPU reference 逐位一致；
全层注入、全量恢复各自再次执行时，八层输出、norm、logits/loss 均复现对应的首次结果。
全层注入的有/无插桩 logits/loss 也一致。原参数到全层注入之间虽然经过额外固定状态 F+B，
仍通过这个自重复控制，不能把恢复一致归因于恰好走了一次不同的运行序列。

随后在实际 NativeMoeExperts backend 上执行相同的 BF16 参数干预。两 backend 的三个
同权重状态，八 rank 结果如下：

| 同权重状态 | 全模型参数 | 八层输出与最终 norm | logits/loss |
| --- | --- | --- | --- |
| 全部 native 首步参数 | 全量相同 | 全量逐位一致 | 全量逐位一致 |
| 仅注入末层 738 个 B1 参数差异 | 全量相同 | 全量逐位一致 | 全量逐位一致 |
| 注入全部 11227 个 B1 参数差异 | 全量相同 | 全量逐位一致 | 全量逐位一致 |

native 自身的无插桩、插桩 reference 及注入后的恢复也完全一致；其 reference 的 logits/loss
复现历史第二步。因而本组第二步分歧可以由首步权重差异重现并通过权重恢复消除，没有发现
额外的、同权重条件下仍存在的 forward backend 差异。该结论限定于本组配置和这些干预状态，
不是任意路由/权重下两种 kernel 都逐位等价的证明。

### 5. 验证边界与后续优先级

已经直接连接的证据是：

```text
真实 expert dW 残差 → 实际 FP32 AdamW 更新差 → BF16 model 写回差 → 第二次 forward 差异
```

末层的前三段在前两轮分别验证，本轮对其 738 个不同写回值完成下一次 forward 干预；
其余层使用真实完整模型的首步参数差异，证明这些参数差异对第二次 forward 的影响，不冒充
其每个 dW 元素都已完成 native/replica/FP64 分解。

固定状态 F+B 产生的新梯度没有做全量精度 oracle；本轮没有执行干预后的新 optimizer 更新，
也未推进第三步及以后的反馈，没有解释完历史第 6 步开始超 loss 阈值及后续 loss 回升的幅度。
仍是固定 batch、随机权重、
合成固定路由的八层 example，不能外推为真实预训练长期收敛结论。原配置跨 backend 的
logits gate 仍失败；解释了失败的数值路径不等于修复或放宽这个 gate。

完成此步后的优先级调整：

1. **降低没有新证据支持的 AdamW 实现错误排查优先级。** 保留 FP32 main、moments、
   replica contribution 合同，不为了逐位对齐重写正确的更新/写回，不引入高代价 dW 重算。
2. **下一项改为显式数值稳定性控制。** 优先单列 eps 敏感性实验：用真实梯度和原 fused
   AdamW 验证参数差异是否降低，再检查真实多步 loss/logits；必要时加入已表现敏感的 LR
   控制。改变 eps/LR 是超参数实验，不是修复现有实现，不能静默改默认、放宽阈值，或用
   新超参数的通过替代原配置的失败。不能只优化最大参数差而忽略训练轨迹。
3. **性能 gate 独立推进。** 本轮有大量 CPU snapshot、逐层同步与固定状态 F+B，不发布
   wall time、HBM 或新 commit 加速比例。默认 attention/dense collective 非确定性和旧/新
   配对计时窗口仍是原边界，不混入本次确定性因果对照。

后续已执行四档 eps 首步复放和 eps=1e-7 的完整模型十步控制，并对照其他 trainer 的
AdamW 用法；较小的 FP32 main 最大差没有消除 BF16/训练分歧。当前优先级已转向单变量
LR 控制及 trainer 梯度合同核验，详见 [eps 与 trainer 对照](#eps-稳定性与-trainer-契约)。

### 6. 本地证据与复现

两个八卡进程组正常退出，完成两次真实完整首步训练和 19 组固定状态 F+B（152 个 rank-local
探针）。八 rank 的身份、初始状态、首步历史链接、干预范围、参数不变、main/moments 不变、
插桩和恢复控制均通过；三个同权重跨 backend 状态的全部保存输出逐位一致。临时脚本通过
`py_compile`；生产源码未改，不将本轮诊断正常退出冒充原 logits gate 已通过。

独立 `audit.py` 重新读取全部 152 份 CPU trace，核对每个张量的 shape/dtype/SHA256；
再从 native 完整 reference 和稀疏注入值重建全部 128 个 rank-local canonical expert
张量，其 hash 与历史 B1 首步完整参数一致。八 rank 全部通过，报告表格也与汇总逐行核对。
MegaMoe / native 各保留 373 / 273 个进程采样帧，均无未知帧或观测到的外部 PID；这只是
离散采样结果，不作为严格独占或性能测量的声明。

原始 JSON、稀疏参数差异、全量 CPU forward traces 和临时脚本只保留本地，不入 git：

```text
/tmp/qwen-parameter-intervention.oUmFcD/
  worker.py       probe.py       run_case.py       run_all.py
  analyze.py      audit.py
  mega/           native/
    launch.json   worker.log     census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}/result.json
    result/rank{0..7}/<probe>/forward.pt
    result/rank{0..7}/<probe>/result.json
  mega/result/rank{0..7}/deltas.pt
  mega/result/rank{0..7}/delta_metrics.json
  analysis_rank{0..7}.json
  summary.json    manifest.json  audit.json
  analysis.log    analysis_parallel.log    analysis_rank{2..7}.log
  analyze_serial.py                       audit.log
```

读取的历史 references 为 `/tmp/qwen-multistep-diagnosis.kI6Fza/` 的
`first_deterministic_reference`、`first_deterministic_{common,mega}` 和
`train_deterministic_reference`；末层写回坐标来自
`/tmp/qwen-adamw-diagnosis.DI8gFa/replay/result/rank{0..7}/mega_b1_grad/`。
source/native 指纹须与这些捕获一致；脚本检测到不一致会停止，不能用新源码混跑旧 reference。

CPU 汇总最初串行，完成 rank 0–2 后只停止本轮自有分析进程，改为按 rank 并行并复用这三份
完整结果；原脚本与串行日志保留在 `analyze_serial.py` / `analysis.log`，最终汇总日志为
`analysis_parallel.log`。没有中断、重跑或改变已经完成的 NPU 探针。`manifest.json` 记录
脚本和逐 rank JSON 的 SHA256。`summary.json` SHA256 为：

```text
03e2cf3d658dc643ff1655094b63607cbe6fb3d826931aada7574d4445bc1b47
```

本机复现时用 `mktemp -d` 建新目录，将本轮六个 `.py` 脚本复制过去，然后在仓库根目录：

```bash
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HYPER_PARALLEL_PLATFORM=torch
export OMP_NUM_THREADS=1
/tmp/megamoe-qwen-shared.0MYlWz/venv/bin/python <新诊断目录>/run_all.py
```

入口顺序启动 MegaMoe 和 native，不复用进程组；父脚本选择独立 MASTER/SHMEM 端口，
HCCL socket range 为 `62000-62063`，拒绝覆盖已有 case。只终止自己启动且失败/超时的
进程组，不操作外部作业。逐帧保存进程采样，采样无命中不等于硬件严格独占。

本轮不改生产代码，不需要 native 重构建或新增结构性 UT。仅将报告和索引直接本地 commit，
不用 autogit，不 push；诊断 JSON、张量和脚本不入仓。

## eps 稳定性与 trainer 契约

接续 [expert 参数干预](#参数干预与层间传播)，基于
`megamoe-sun@0681a29b`。生产 Python / native 相对 `ee5a6965` 未变。本轮只做诊断，
不改变默认 eps、LR、梯度归约、容差或 replica / FP32 main 合同。

结论：**增大 eps 不能据本轮结果作为精度修复。** 四档真实梯度复放中，FP32 main 最大差
下降，但 BF16 model 的最大差、不同元素数、RMSE 没有同时改善；`eps=1e-7` 的完整模型
仍在第二步 logits、第六步 loss 超原阈值。其他 trainer 使用相同的 core fused AdamW，
主要差异是优化器选择、LR、分组、调度、裁剪和梯度归约，不是另一套特殊 AdamW 公式。

### 1. 保持不变的实验合同

硬件/运行时沿用前几轮：EP8、Ascend 910B3、Torch/TorchNPU 2.9.1、CANN 9.2.0，
editable import 指向当前 checkout，native payload 仍为当前源码构建产物。初始权重、
batch、配置及 source/native identity 均链接到此前确定性控制的指纹。

原 example 为八层 connected Qwen model，H5120 / I1792、E48 / K8、每 rank 4096 tokens，
Dirichlet 0.1 固定路由、shared-B1、capacity factor 1.5。BF16 model、FP32 main/moments，
AdamW 的 LR=1e-3、weight decay=0.01、betas=(0.9, 0.999)。确定性模式仅是诊断控制。
原 `rtol=0.02 / atol=0.002` 保留，归一化误差大于 1 表示超过此阈值。

通过临时、局部 optimizer factory 覆盖只改变 eps，并检查实际 leaf defaults 和每个
param group；不是修改报告标签而继续运行原 eps，也不是用 CPU optimizer 替代设备更新。
所有临时脚本、JSON、张量只保存于 `/tmp/qwen-eps-stability.iUEwDO/`。

### 2. 四档 eps：固定真实梯度、独立首步更新

使用此前实际第八层的初始权重、native 梯度和 B1 梯度，每档分别运行：native layout +
native gradient、MegaMoe layout + 同一 native gradient、MegaMoe layout + 实际 B1 gradient。
每组重新创建 model/optimizer，仅做一次原 fused AdamW 更新，不运行新的 F+B。
四档 × 三组 × 八 rank，共 96 个独立 rank-local optimizer 实例/更新。

全量比较覆盖该层 1321205760 个逻辑权重。相同梯度的两种布局，在全部 eps 下的 main、
两个 moments、BF16 model 均逐位一致；eps=1e-8 的全部状态指纹复现上一轮记录。
梯度转 FP32、group step=1、BF16 直接 cast 和梯度消费均符合原合同。同一 eps 下 B1 对
native 的状态差异始终限定在输入梯度不同的 40566 个坐标内；其他位置没有新增状态差异。
三组各自固定梯度的完整 moments 指纹均不随 eps 改变。

#### 2.1 跨 backend 残差

下表均为 B1 相对同 eps 的 native，元素数跨八 rank 求和，最大差取全量最大值；
RMSE 的分母为全层权重元素数，不是仅取差异位置。

| eps | FP32 main 不同元素 | main 最大差 | main RMSE | BF16 model 不同元素 | model 最大差 | model RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-8 | 40245 | 6.493181e-6 | 4.102406e-9 | 738 | 1.220703e-4 | 3.228289e-8 |
| 1e-7 | 40566 | 1.931563e-6 | 3.723977e-9 | 826 | 1.220703e-4 | 3.307562e-8 |
| 1e-6 | 40513 | 1.944602e-6 | 3.709006e-9 | 815 | 1.220703e-4 | 3.501729e-8 |
| 1e-5 | 39098 | 1.912005e-6 | 3.688080e-9 | 763 | 1.220703e-4 | 3.074808e-8 |

例如 1e-7 把 main 最大差降低约 70.3%，但 BF16 不同元素从 738 增到 826，RMSE 也略增。
1e-5 的 BF16 RMSE 下降约 4.8%，但最大差未变、不同元素数仍高于原配置。
不能只根据 main 最大差选择所谓“精度更好”的 eps：BF16 舍入边界的位置也随正常更新改变。

独立 CPU FP64 复核全部 40566 个梯度差异坐标，实际 main 相对首步理想公式
`w * (1 - lr * wd) - lr * g / (abs(g) + eps)` 的最大误差均不超过 2.291e-9。
不同 eps 下没有出现新的更新公式、布局或写回错误证据。

#### 2.2 增大 eps 同时改变正常更新

以下比较同一 native 梯度下，新 eps 的首步更新与原 eps=1e-8 更新；它不是 backend 残差。

| 新 eps | native main 不同元素 | main 最大差 | main RMSE | native BF16 model 不同元素 | model 最大差 | model RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-7 | 770703350 | 5.194941e-4 | 9.291866e-5 | 323327282 | 6.103516e-4 | 9.374245e-5 |
| 1e-6 | 770703350 | 8.181827e-4 | 2.798714e-4 | 645727012 | 8.544922e-4 | 2.779793e-4 |
| 1e-5 | 770703350 | 9.386940e-4 | 5.238589e-4 | 770600102 | 9.765625e-4 | 5.191292e-4 |

改变正常更新的幅度远大于本次 backend 残差。因此在完整训练前明确选择新档中改动最小的
1e-7 做动态控制，不根据后续 loss 事后挑档，也不推荐把更大 eps 直接设为默认值。

### 3. 完整模型十步：eps=1e-7 未消除分歧

使用原 `_optimizer_step`，所有参数组统一设为 eps=1e-7，不只修改末层 expert；每步都执行
真实 F+B、原 BF16 dense all-reduce、FP32-main AdamW 和 BF16 写回。其余合同不变。
按 native A1 / MegaMoe B1 / MegaMoe B2 / native A2 顺序独立驻留，避免双套设备训练状态。

A1 导出十步完整 loss/logits，以及首两步全量 83 个 canonical 梯度/更新后参数。后三组
按原阈值比较；每组另外记录全部阶段指纹和末步完整 model/main/moments 指纹。
首步 forward 与历史 eps=1e-8 逐位相同，native 首步全部梯度也不变，符合 eps 尚未参与更新。

首两步全量梯度与更新后参数仍通过原 tensor 阈值，但这不阻止下一次 logits 失败：

| 步数/阶段 | 不同张量数（跨 rank 名称并集 / 83） | 最大绝对差 | 最大归一化误差 | 原 gate |
| --- | ---: | ---: | ---: | --- |
| 1 / gradients | 16 | 9.536743e-7 | 0.000476256 | 通过 |
| 1 / updated parameters | 16 | 1.220703e-4 | 0.0527872 | 通过 |
| 2 / gradients | 83 | 3.051758e-5 | 0.0149530 | 通过 |
| 2 / updated parameters | 67 | 3.906250e-3 | 0.708363 | 通过 |

首组跨 backend 对照如下。loss 列为八 rank 均值，但精度判定逐 rank 执行，不能以均值抵消差异。

| 步数 | native mean loss | MegaMoe mean loss | loss 最大归一化误差 | loss 超限 rank | forward 最大归一化误差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 10.539054 | 10.539054 | 0 | 无 | 0 |
| 2 | 0.980820 | 0.980826 | 0.010356 | 无 | 10.664919 |
| 3 | 0.072665 | 0.072634 | 0.043739 | 无 | 12.898761 |
| 4 | 0.080321 | 0.080216 | 0.216489 | 无 | 34.934174 |
| 5 | 2.049541 | 2.045538 | 0.384933 | 无 | 98.762787 |
| 6 | 1.920564 | 1.913746 | 1.304379 | 1 / 2 / 6 / 7 | 65.732635 |
| 7 | 7.506670 | 7.521816 | 0.330558 | 无 | 95.236740 |
| 8 | 7.314979 | 7.475851 | 4.819952 | 2 / 5 | 834.811951 |
| 9 | 8.189187 | 8.025925 | 1.505480 | 0 / 2 / 5 | 1428.254639 |
| 10 | 8.594246 | 8.602003 | 0.927607 | 无 | 1384.625366 |

第二步 forward 失败来自 logits，不是 loss；后续 loss 并非一旦超限就每步持续失败。
与同源码/初始状态的历史 eps=1e-8 确定性对照相比，首次 logits / loss 失败仍为第 2 / 6 步。
两 backend 的 loss 仍显著回升。动态轨迹已被 eps 改变，不能把某一步差值下降当作原配置已修复。

独立重读第二步的 1048576000 个 logits，eps=1e-7 有 111865143 个超过原阈值，RMSE 为
0.004180811；历史 eps=1e-8 分别为 111605047 / 0.004172864，最大绝对差均为 0.0625。
该步不仅最大归一化误差没有改善，超限数量和 RMSE 也略增。末层独立 eps 复放的首步
BF16 写回与本次完整模型对应参数在两侧八 rank 全部逐位一致，连接了两个实验的实际状态。

### 4. 其他模块 trainer 的实际 AdamW 用法

#### 4.1 共享实现，不同训练合同

`BaseTrainer._build_optimizer` 通过 YAML target 的 builder 构造 optimizer；AdamW builder
及 Muon 的 fallback AdamW 最终都调用 `get_hyper_optimizer`，进入同一个
`core.optimizer.adamw.AdamW` → `torch._fused_adamw_`。配置中的 `adamw_` 前缀在 factory
剥离，未填写的 beta/eps 采用构造默认值。开启 `fp32_main_params` 时，trainer 使用的
`Float16OptimizerWithFloat16Params` 与 benchmark 是同一个 wrapper。

| 项目 | 当前 Qwen benchmark | Qwen3-MoE trainer recipe | Qwen2-MoE FSDP 精度专用用例 |
| --- | --- | --- | --- |
| optimizer | 所有参数纯 AdamW | 矩阵走 Muon，其余 fallback AdamW | 纯 AdamW |
| AdamW 基础 LR | 1e-3 | 1e-5；Muon 为 3e-5 | 1e-4 |
| AdamW betas / eps | (0.9, 0.999) / 1e-8 | 相同 | 相同，由 core 默认补齐 |
| weight decay | 所有参数一组 0.01 | AdamW 分 0.01 / 0 两组；Muon 为 0 | 全部为 0 |
| no-decay 规则 | 无，norm 也有 decay | 名称包含 bias / norm / ln_ | builder 同样分组，但两组 decay 均为 0 |
| FP32 main wrapper | 始终开启 | YAML 未开启；配置默认 False | 显式 True |
| 梯度归约/保留 | dense BF16 SUM/8，再转 FP32 | 由 trainer/FSDP policy 配置 | FP32 reduce，并保留 FP32 main_grad |
| 裁剪 | 无 | max_grad_norm=1 | 测试循环 max_norm=inf，仅比较 norm |
| LR scheduler | 无 | cosine + 1 步 warmup | 专用测试循环无 scheduler |

这里的 Qwen3 recipe 是预训练 Qwen3-30B-A3B、真实数据、默认 FSDP8/EP1、全层 checkpoint
的三步示例，不是本次随机八层、固定 batch/路由的 EP8 benchmark。
`split_muon_adamw_params` 把 ndim>=2 且名称不包含 embed/lm_head/out_head/out_proj.bias
等保留关键词的参数分给 Muon，包含普通 expert 矩阵；所以不能称该配方已经证明“expert
使用另一种 AdamW 设置即可稳定”。三个 `examples/training_demo/train_parallel_*.yaml`
也采用相同的 Muon/fallback AdamW 配置模式。

用一个包含 embedding、norm、gate 和 expert 矩阵的小 CPU 模型，实际解析 YAML 并构造
AdamW，验证前缀、参数分组、默认 eps/betas，以及 scheduler 顺序。该 dry-run 不产生梯度、
不启动 NPU、不执行 fused 更新，不冒充完整 trainer 训练。

Qwen3 recipe 的 `lr_config.lr=3e-5` 是调度比例的基准，不会把 AdamW 基础 LR 变成 3e-5。
当前 LambdaLR 初始化及 trainer 的“optimizer.step 后 scheduler.step”顺序，使三次更新
所用的 AdamW LR 实测为 **0 → 1e-5 → 5e-6**，第三次调度后为 0。第一步零 LR 不等于
跳过 backward 或 moments 更新；因此不能把这份三步 recipe 与恒定 LR=1e-3 的十步轨迹等同。

另查 `compile.GraphTrainer`：它在 `_init_optimizer` 中构造的是 `torch.optim.Adam`，
默认 LR=1e-4，可选 `grad_clip`，并不是上述 AdamW 路径，不能混作 AdamW 稳定性证据。

#### 4.2 同一个 FP32-main wrapper，不代表同一个梯度路径

trainer 配置在 `fp32_main_params=True` 时强制要求 `reduce_dtype='float32'`；FSDP adapter
进一步设置 `apply_grad_on_fp32_main_grad=True`。专用精度用例明确检查归约后 model.grad
为空、main_grad 为 FP32，随后 `prepare_grads`、`step_with_ready_grads` 并检查直接 cast 写回。
当前 benchmark 没有 FSDP；dense BF16 grad 先归约/平均，wrapper 才转 FP32，不能恢复已发生
的 BF16 归约舍入。这个区别值得单独验证，但不推翻此前确定性首步 dense 梯度逐位一致的证据。

TextTrainer/VLMTrainer 都先做 micro-batch 累积与 FSDP 同步，再裁剪、optimizer.step、
zero_grad、scheduler.step。FP32-main wrapper 本身没有加入动态 loss scaler 或自动裁剪；
`mixed_precision.enabled` 也是与 `fp32_main_params` 分开的配置项。

梯度归一化也必须连同 loss 合同看：trainer 用有效 token 计数加权 backward loss，配合
FSDP 的 DP/CP/TP reduction/scaling；benchmark 用每 rank 本地平均 CE，再手工平均 dense
梯度。不能把 trainer 的缩放因子脱离 mesh/loss 语义移植，也不能据两个 backend 都通过
首步比较，就声称已验证其与真实 trainer 的全局目标完全一致。

#### 4.3 不可直接照搬全局裁剪

项目 `clip_grad_norm_` 优先读 main_grad，通信来自梯度/参数的 DTensor placements：
Shard 统计跨 rank 求和，plain tensor 被当作 replicated，不发起该分片归约。
NativeMoeExperts 是 EP-sharded DTensor，而当前 MegaMoe expert 是 packed 本地 Parameter。
如果对两侧直接调用同一 helper，MegaMoe 的其他 rank expert 统计不会自动进入 norm。

后续控制需要显式区分 dense 副本和 home expert 分片，统一原有梯度归一化口径后，
使 dense 只计一次、各 rank home expert 统计合并一次；不能把 physical replica 再计为
独立参数，也不能因统计域不同而给两 backend 施加不同的 clip coefficient。应先用 max_norm=inf 验证
norm 与聚合定义，再尝试实际裁剪。以上是移植约束，本轮没有实现裁剪或宣称其已改善 loss。

#### 4.4 源码入口

- [benchmark optimizer/step](../../../../../examples/torch/multicore/mega_moe/qwen_moe_benchmark.py)
- [BaseTrainer](../../../../trainer/base.py)、[TextTrainer](../../../../trainer/text_trainer.py)、
  [VLMTrainer](../../../../trainer/vlm_trainer.py)
- [optimizer builders](../../../../components/optim/builders.py)、
  [parameter groups](../../../../components/optim/parameter_groups.py)、
  [FP32-main wrapper](../../../../components/optim/mixed_precision_optimizer.py)
- [optimizer factory](../../../optimizer/__init__.py)、[fused AdamW](../../../optimizer/adamw.py)、
  [scheduler](../../../optimizer/lr_scheduler.py)
- [Qwen3 recipe](../../../../models/qwen3_moe/recipes/train.yaml)、
  [Qwen2 精度配置](../../../../../tests/torch/dualmode_trainer/fsdp_integaration/test_yamls/qwen2_tp_cp_ep_fsdp_global_accuracy.yaml)、
  [精度测试循环](../../../../../tests/torch/dualmode_trainer/fsdp_integaration/_test_fsdp_accuracy.py)
- [trainer config 校验](../../../../trainer/config/trainer.py)、
  [FSDP adapter](../../../../distributed/_builder/fsdp_adapter.py)、
  [loss aggregation](../../../../trainer/runtime/metrics.py)、
  [Torch clip 实现](../../../../platform/torch/clip_grad.py)、
  [GraphTrainer](../../../../compile/trainer.py)

### 5. 完成本步后的优先级

1. **不把增大 eps 纳入默认优化。** 保留原 baseline 和失败记录，不再根据较小的 main
   最大差推断训练稳定。当前证据没有支持重写 AdamW 或高代价 native-exact dW。
2. **下一项优先单变量 LR=1e-4 控制。** 这是现有纯 AdamW 精度用例的量级，也是此前
   native-only 短轨迹曾表现稳定的档位；应补齐两个 backend 的真实多步 loss/logits、
   梯度/参数与自重复。先保持 eps=1e-8、WD=0.01、无裁剪/调度，不能把同时换 WD、
   Muon、warmup 的结果误认作 LR 效果；新合同通过也不替代原合同验收。
3. **随后分别核验 trainer 对齐项。** 梯度全局归一化及 norm 定义先于实际裁剪；FP32
   dense 归约需独立记录精度、通信和 HBM 成本；no-decay 分组、WD、warmup 各自单列。
   不把整套 trainer recipe 一次搬入 benchmark 后再倒推是哪一项有效。
4. **性能 gate 独立。** 本轮有完整 CPU snapshot、全量 hash/oracle，不能发布 latency、
   HBM 增量或新加速比例；原有竞争/确定性模式边界不变。

后续已执行其中的单变量 LR=1e-4 控制，见
[低 LR 多步对照](#低-lr-多步对照)：eps/WD 与原梯度路径不变，
两侧十步 loss 持续下降且跨 backend loss 全部通过，但 logits 仍从第二步失败。
这更新下一阶段优先级，不改变本报告 eps 实验的原始合同或失败结论。

### 6. 验证与本地证据

四个独立八卡训练进程组均正常退出，完成 320 次 rank-local 真实更新。native A1/A2 与
MegaMoe B1/B2 在八 rank 的以下自重复检查全部通过：

- 十步 loss 和全量 logits 指纹逐位相同。
- 首两步全部 canonical 梯度/更新后参数指纹逐位相同。
- 第十次更新后的完整 model、FP32 main、一阶/二阶 moment 指纹逐位相同；group step=10。
- source/native、初始权重、batch、配置、实际 optimizer defaults 和 worker 脚本身份一致。

自重复的最终状态比较是同 backend 的全量 hash，不是跨 backend 最终 main/moments 的
数值 oracle；第 3--10 步没有保存全量梯度/更新后参数，不声称十步所有训练状态均通过
跨 backend 精度或有限值验收。每步 loss/logits 则都做了原全量有限值/精度检查。

独立审计重新读取 112 份 native references、80 份 MegaMoe forward 文件，核对 2976 个
张量的 shape/dtype/实际字节 SHA256 及 reference contract。由这些完整值重新计算的十步
forward 最大差、归一化误差及 pass/fail 与运行时 oracle 一致；loss 与 logits 分开判断。
原参考值没有被候选覆盖。四档复放的全部差异坐标、实际 optimizer 设置及 FP64 首步公式
另行复核通过；空差异 rank 的审计空集处理修正后重跑，不影响设备复放的原始记录。

进程采样情况如下。外部 PID 为不属于本组八个 worker 的设备进程，观测到的部分命令为
外部 pytest；不干预这些任务。无未知采样帧不代表连续严格独占。

| 进程组 | 采样帧数 | 有外部 PID 的帧数 | 未知帧数 |
| --- | ---: | ---: | ---: |
| 四档独立层复放 | 341 | 0 | 0 |
| native A1 | 283 | 58 | 0 |
| MegaMoe B1 | 494 | 70 | 0 |
| MegaMoe B2 | 452 | 0 | 0 |
| native A2 | 279 | 0 | 0 |

因此四组只用于本次数值控制，不称为严格独占的性能 ABBA。两 backend 在不同采样条件下
各自逐位复现支持本轮数值结果，但不构成竞争不影响一般精度/性能的保证。

临时脚本语法检查、CPU trainer 配置/scheduler dry-run、上述设备诊断及独立数据审计通过。
报告表格逐项与本地汇总核对，Markdown 链接与 diff whitespace 检查通过。生产源码未改，
不新增 UT，不把诊断正常退出冒充原跨 backend logits/loss gate 通过。

本地证据布局如下；JSON、张量、临时脚本均不入 git：

```text
/tmp/qwen-eps-stability.iUEwDO/
  replay_probe/             training_probe/
  run_training.py          trainer_optimizer_probe.py
  summarize_replay.py      audit_replay.py
  summarize_training.py    audit_training.py    verify_report.py
  replay/
    launch.json            worker.log           census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}/eps{1e-08,1e-07,1e-06,1e-05}/
      native.json          same_grad.json       b1.json
      changed_gradient_points.pt
  train_{a1,b1,b2,a2}/
    launch.json            worker.log           census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}_step{1..10}.json
  train_b1/result/rank{0..7}/step{1..10}_forward.pt
  training_reference/rank{0..7}/
    step{1..10}_forward.{pt,json}
    step{1,2}_{gradients,updated_parameters}.{pt,json}
  replay_summary.json      replay_audit.json
  training_summary.json    training_audit.json
  trainer_optimizer_probe.json                 manifest.json
```

输入沿用 `/tmp/qwen-expert-dw.9pLDND/`、`/tmp/qwen-adamw-diagnosis.DI8gFa/` 和
`/tmp/qwen-multistep-diagnosis.kI6Fza/` 的实际 capture/reference，未重新生成替代原梯度。
复放/训练 launcher 及原始命令、环境见各组 `launch.json`；脚本、汇总及报告 SHA256
汇集在本地 `manifest.json`。关键汇总身份为：

| 文件 | SHA256 |
| --- | --- |
| replay_summary.json | `327deb3d760ecdcd4b69aceda3f8a3ad3e7de245ebf8fbf3090cf86588b5492e` |
| training_summary.json | `6c938de2885e475bc4d13740b66c9d64376b2b219711daea31d10f01a60be9eb` |
| training_audit.json | `bdec8d78ce5b6a90aa0220d26cb6106e026eb2a29671313ea1d78a326e0b0d2c` |

## 低 LR 多步对照

本轮接续 [eps 稳定性与 trainer 对照](#eps-稳定性与-trainer-契约)，基于
`megamoe-sun@8870070c`。生产 Python/native 相对 `ee5a6965` 未变。只通过 benchmark 已有的
`--learning-rate 1e-4` 改变实验 LR，不修改默认值、optimizer 实现或梯度路径。

结论：**LR=1e-4 控制中，两侧十步 loss 均持续下降，跨 backend loss 全部通过，但
logits 仍未通过。** 第 1、2、10 步的全量梯度/参数仍通过原 tensor 阈值。这支持高 LR 是
此前短轨迹不稳定的敏感因素，不等于 expert dW 舍入、原 LR 配置或默认模式精度已修复。

### 1. 合同与覆盖范围

保留原八层 connected Qwen example：EP8、H5120 / I1792、E48 / K8，每 rank 4096 tokens，
vocab 32000、shared-B1、capacity factor 1.5、Dirichlet 0.1 固定路由及相同初始权重/batch。
AdamW 仍为 eps=1e-8、WD=0.01、betas=(0.9, 0.999)，BF16 model / FP32 main/moments。
不加 scheduler、warmup、裁剪、no-decay 分组，也不切换 Muon。

环境仍为 8 张 64 GiB Ascend 910B3，CANN 9.2.0、Torch/Torch-NPU 2.9.1、
Transformers 5.5.3；editable import 已确认指向本 checkout。

本轮使用确定性模式，比较对象为此前同模式 LR=1e-3 的控制，而不是把确定性开关也当作
LR 的效果。此前 `train_low_lr_a1/a2` 是默认非确定性模式下的 native-only 数据：可作为
选择此 LR 档位的线索，但不是本轮逐位参考，也未验证 MegaMoe。

八卡按 native A1 / MegaMoe B1 / MegaMoe B2 / native A2 顺序独立驻留，每组执行十次原
`_optimizer_step`：真实 F+B → 原 BF16 dense SUM/8 → FP32-main fused AdamW → BF16 写回。
没有 optimizer monkey-patch、空更新或只改报告标签。实际 defaults/param group 的
LR、eps、WD、betas 和逐步递增的 group step 均检查。

采集/比较范围：

- 每步完整 loss/logits，原 `rtol=0.02 / atol=0.002`，逐 rank 判定，不用平均 loss 抵消误差。
- 第 1、2、10 步全部 83 个 canonical 梯度与更新后参数，比上一轮增加末步全量比较。
- 末步完整 model/main/moments 指纹；额外全量检查 FP32 main/moments 的 dtype 和有限值。
- source/native、配置、初始参数/batch 指纹，以及两 backend 的首步 forward/梯度历史链接。

第 3--9 步没有保存全量梯度/更新后参数；末步 main/moments 只做同 backend 自重复 hash
及有限值检查，没有保存全量值做跨 backend 的数值 oracle。不能把这些未覆盖项写成已验收。

`/tmp` 只余约 201 GiB，不足以增加末步快照。本轮原始数据保存到仓库外的
`/home/weizheng/work/zfr/qwen-lr-stability.LGYINg/`，旧证据不删除，JSON/张量/临时脚本不入 git。

### 2. 十步 loss 与 forward

以下为 A1/B1 的八 rank 均值；精度判定逐 rank 执行。归一化误差为
`abs(actual - native) / (0.002 + 0.02 * abs(native))`，大于 1 才超原阈值。

| 步数 | native mean loss | MegaMoe mean loss | loss 最大归一化误差 | loss 超限 rank | forward 最大归一化误差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 10.539054 | 10.539054 | 0.000000 | 无 | 0.000000 |
| 2 | 9.093883 | 9.093871 | 0.000368 | 无 | 10.825780 |
| 3 | 7.857574 | 7.857608 | 0.000941 | 无 | 12.620228 |
| 4 | 6.884751 | 6.884725 | 0.001972 | 无 | 11.838282 |
| 5 | 5.545509 | 5.545509 | 0.002370 | 无 | 12.522596 |
| 6 | 4.390320 | 4.390457 | 0.005130 | 无 | 11.882232 |
| 7 | 3.079215 | 3.079110 | 0.005403 | 无 | 12.586909 |
| 8 | 2.047648 | 2.047724 | 0.007032 | 无 | 12.060124 |
| 9 | 1.233722 | 1.233623 | 0.012705 | 无 | 13.001662 |
| 10 | 0.683569 | 0.683580 | 0.009494 | 无 | 12.221617 |

两 backend 的每个 rank 均连续十步 loss 下降。八 rank × 十步的 loss 最大绝对差为
4.620552e-4，最大归一化误差为 0.0127054；不是只在平均后通过。第 2--10 步的 forward
失败仍来自 logits，不能用 loss 的通过覆盖；末步 logits 最大绝对差为 0.125。

独立读取完整 logits 后重算如下，两步各覆盖八 rank 合计 1048576000 个元素，且每个 rank
都存在 logits 超限。不同元素数与超限元素数不是同一指标：

| logits 步数 | 不同元素数 | 超原阈值元素数 | 最大绝对差 | RMSE |
| --- | ---: | ---: | ---: | ---: |
| 2 | 799743683 | 110224179 | 0.03125 | 0.004159040 |
| 10 | 833466173 | 148913179 | 0.125 | 0.005027591 |

同源码、同初始状态、同确定性模式的历史 LR=1e-3 对照，native/MegaMoe 平均 loss 在
第四步为 0.072696 / 0.072610，之后明显回升，第十步为 8.620320 / 8.424216；跨 backend
loss 从第六步有 rank 超阈值。本轮新 LR 的初期 loss 下降较慢，但十步内没有重现回升或
loss 超限。比较的是不同优化轨迹，不能把低 LR 的末步参数直接当作原 LR 的精度参考。

首步两侧全量 forward 与历史逐位一致，全部梯度也分别复现各自历史指纹。LR 首次参与
optimizer 更新之后，轨迹才变化。eps/betas/WD 配置未动；AdamW 的梯度项和 decoupled
decay 项都会随 LR 缩放，本轮没有进一步分解两者的贡献。

### 3. 梯度、参数与自重复

不同张量数是跨 rank 的 canonical 名称并集，不是不同元素数：

| 步数/阶段 | 不同张量数（并集 / 83） | 最大绝对差 | 最大归一化误差 | 原 gate |
| --- | ---: | ---: | ---: | --- |
| 1 / gradients | 16 | 9.536743e-7 | 0.000476256 | 通过 |
| 1 / updated parameters | 16 | 1.220703e-4 | 0.0527872 | 通过 |
| 2 / gradients | 83 | 3.814697e-6 | 0.001905586 | 通过 |
| 2 / updated parameters | 66 | 2.441406e-4 | 0.1055743 | 通过 |
| 10 / gradients | 83 | 7.629395e-6 | 0.003784381 | 通过 |
| 10 / updated parameters | 66 | 1.953125e-3 | 0.4040196 | 通过 |

首步梯度不依赖 LR；首步 BF16 参数最大差也仍为 1.220703e-4，与历史 LR=1e-3 相同。
这不证明不同元素数或整体误差相同，但说明降低 LR 不保证降低 BF16 最大差。继续保留
原 replica FP32 partial 聚合、FP32 main/moments 和 BF16 写回，不为逐位对齐引入高代价重算。

native A1/A2 和 MegaMoe B1/B2 在全部八 rank 各自通过以下自重复检查：

- 十步 loss 和完整 logits 指纹逐位一致。
- 第 1、2、10 步全部 canonical 梯度/更新后参数指纹逐位一致。
- 第十次更新后完整 model、FP32 main、一阶/二阶 moment 指纹逐位一致。

四组末步 main/moments 均通过全量 FP32 dtype/有限值检查，实际 optimizer group step=10。
这是同 backend 的逐位自重复，不是两 backend 的末步 main/moments 数值相同。

### 4. 解释边界与下一步

低 LR 下固定 batch 的十步下降曲线，不等于真实数据长期收敛；降低 LR 是改变优化轨迹，
不是消除 expert dW 舍入，也不能替代原 LR=1e-3 合同的精度验收。loss、logits 和训练状态
各自按原阈值报告，不把其中一项的通过扩展到其他项。

本轮有完整 CPU snapshot、hash/oracle 和额外状态检查，不发布新的 latency/HBM 或内核
加速比例；确定性控制与默认模式、性能测量继续分开。

完成本步后的优先级调整为：

1. **先验证全局梯度归一化和 norm，不继续盲目扫 LR/eps。** 保留 LR=1e-4 为显式的
   稳定短轨迹控制，不改变生产默认值。用可容纳的全局 batch/unsharded 小模型参考核对
   本地平均 CE、dense SUM/EP 和 home expert 梯度对应的全局目标；两个 EP backend
   互相通过，不足以证明这一合同正确。本轮尚未运行该参考，不能先断言存在缩放 bug。
2. **归一化明确后，先测 norm，再裁剪。** dense 副本只计一次、home expert 分片合并
   一次，不重复统计 physical replica；先用 max_norm=inf 验证两 backend 与全局参考
   的 norm，再尝试有限阈值。不能直接给 plain packed expert 套用依赖 DTensor
   placements 的 helper。trainer 的有效 token 加权与 mesh scaling 也须作为整体核对。
3. **FP32 dense 归约独立控制。** 它与 BF16 归约后转 FP32 并非相同数值路径，须单列
   完整训练状态精度、通信量和 HBM 成本。WD/no-decay、warmup/调度继续分项，不一次
   搬入整套 recipe，也不把降低 LR 的效果算到通信或 optimizer 实现优化上。
4. **保留未验收项。** 本轮未覆盖低 LR 的默认非确定性 MegaMoe 多步、自重复及更长
   真实数据轨迹；原 LR 多步 logits/loss 失败记录不撤销。干净性能窗口与无诊断探针的
   完整模型 profile 仍独立验收，不用本轮训练诊断耗时推断性能。

### 5. 本地证据与验证

四个独立八卡进程组均正常退出，完成 320 次 rank-local 真实更新。全部 32 份最终 rank
记录通过 source/native、配置、初始参数/batch、实际 optimizer 设置、worker 身份及
逐步 group step 核验。生产源码/native 未改，旧参考值未被新控制覆盖。

独立 CPU 审计重新读取 128 份 native reference 和 80 份 MegaMoe forward 文件，核对
4304 个张量的 shape/dtype/实际字节 SHA256，reference schema/contract 与运行时指纹
均一致。完整 logits/loss 的有限值、最大差、归一化误差及原阈值 pass/fail 独立重算，
与运行时 forward oracle 一致；loss 和 logits 分开汇总，不用前者覆盖后者的失败。

进程采样如下：a1/a2 为 native，b1/b2 为 MegaMoe。外部 PID 指不属于本组八个 worker
的设备进程；覆盖检查要求每张设备有解析出的进程行或明确的无进程行，否则记为未知。

| 进程组 | 采样帧数 | 有外部 PID 的帧数 | 未知帧数 |
| --- | ---: | ---: | ---: |
| a1 | 350 | 0 | 0 |
| b1 | 661 | 0 | 0 |
| b2 | 703 | 0 | 0 |
| a2 | 349 | 0 | 0 |

采样未见外部设备进程，不代表连续严格独占。即使本轮没有观测到竞争，CPU 快照/hash/
oracle 与额外状态检查仍使其不适合作为性能 ABBA；不据此更新旧的 latency/HBM 结论。

临时脚本语法、四组设备诊断、独立完整快照审计、报告数值表格/链接和 diff whitespace
检查通过。只修改 Markdown 报告与索引，不新增 UT；诊断正常退出不等于 logits gate 通过。

本地证据布局如下；以下均在仓库外，git 只记录报告与索引：

```text
/home/weizheng/work/zfr/qwen-lr-stability.LGYINg/
  training_probe/
    worker.py             run_case.py           probe_util.py
  run_training.py         summarize_training.py
  audit_training.py       verify_report.py
  train_{a1,b1,b2,a2}/
    launch.json           worker.log            census.jsonl
    result/rank{0..7}.json
    result/rank{0..7}_step{1..10}.json
  train_b1/result/rank{0..7}/step{1..10}_forward.pt
  training_reference/rank{0..7}/
    step{1..10}_forward.{pt,json}
    step{1,2,10}_{gradients,updated_parameters}.{pt,json}
  training_summary.json   training_audit.json   manifest.json
```

模型、输入、forward/梯度的历史指纹链接到
`/tmp/qwen-multistep-diagnosis.kI6Fza/` 和 `/tmp/qwen-eps-stability.iUEwDO/`；
本轮参考由新 native A1 生成，未覆盖旧 LR/eps 数据。每组的完整启动命令及通信环境见
`launch.json`，诊断 worker 直接调用原 benchmark 的 workload/optimizer-step，脚本身份
随结果保存。生产入口为
[Qwen benchmark](../../../../../examples/torch/multicore/mega_moe/qwen_moe_benchmark.py)及
[现有 LR 参数](../../../../../examples/torch/multicore/mega_moe/qwen_moe_config.py)。

脚本、汇总及本报告 SHA256 汇集在本地 `manifest.json`。关键汇总身份为：

| 文件 | SHA256 |
| --- | --- |
| training_summary.json | `a55bc488474bf90f33149e69efb875f92986124e4811a98918f39ca8efb9b9a2` |
| training_audit.json | `c58ee24b6c288d4046200f11f73db691ccf62ce5915743963d71df4c6552bec2` |

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

远端新增的六轮诊断原文固定在提交 `ba9a83a2943ffdf8f27691f3e3a07738752e7971`，使用同一路径前缀：

| 原文件 | 本文锚点 |
| --- | --- |
| megamoe_qwen_multistep_diagnosis_20260908.md | [多步数值与计时诊断](#多步数值与计时诊断) |
| megamoe_qwen_expert_dw_diagnosis_20260908.md | [Expert dW 切分与舍入](#expert-dw-切分与舍入) |
| megamoe_qwen_adamw_diagnosis_20260908.md | [AdamW 与 BF16 写回](#adamw-与-bf16-写回) |
| megamoe_qwen_parameter_intervention_20260908.md | [参数干预与层间传播](#参数干预与层间传播) |
| megamoe_qwen_eps_stability_20260908.md | [eps 稳定性与 trainer 契约](#eps-稳定性与-trainer-契约) |
| megamoe_qwen_lr_stability_20260908.md | [低 LR 多步对照](#低-lr-多步对照) |

例如恢复未调整标题和链接的完整原文：

```bash
git show ba9a83a2943ffdf8f27691f3e3a07738752e7971:hyper_parallel/core/multicore/doc/megamoe_qwen_lr_stability_20260908.md
```
