# Qwen 完整模型 shared-B 接入与训练验收

**结论：完整模型示例现已显式支持跨层 shared-B。** 平衡、Dirichlet 倾斜、learned TopK 三组
八卡训练回归通过，保留原始 `rtol=0.02 / atol=0.002`，没有修改 MegaMoe native、planner 或
梯度累加规则。本轮是 connected model 的正确性覆盖，不是新的优化加速结论。

前一轮 receive-only 优化已提交为 `7dba0a70`；其独立 16 层 MoE 偏斜 full-step 改善 4.086%，
不能直接外推本轮的 attention → MoE → loss 整模型。背景见
[receive-only 报告](megamoe_receive_only_overlap_20260908.md)。

## 1. 实现与修复

- 从 `qwen_moe_model.py` 提取 backend-free `qwen_moe_config.py`，旧配置类/类型名仍可从模型模块导入。
  保持默认 8 层、H5120/I1792、T4096、V32000、E48/K8、EP8，不把小测试形状改成默认值。
- 新增显式 `--shared-physical-weights`，必须 B>0，且与 legacy compact 互斥；传入每层
  `MegaMoeExperts` 后复用现有 `share_execution_resources()`。home 参数与 optimizer state 不共享。
  新增层数、宽度、intermediate、词表、序列长度 CLI，非法配置在 distributed init 前检查。
- 真实入口原有 `position_ids=[sequence]` 不符合 Transformers RoPE 的 `[batch, sequence]` 合同，
  CPU 测试先复现失败后修为保留 batch 轴；实际 NPU attention 路径通过。
- `components.modules` 原先急切导入 DSA，导致只使用 Qwen RMSNorm 也要求安装
  `omni_training_custom_ops`。沿用项目 functional 包的按需导出模式，保留全部 13 个公开名称、
  类型检查导入和对象缓存。显式请求缺失依赖的模块仍抛原异常，不吞错、不假造可用实现。
- 修复 first-step expert update 探针：原逻辑仅比较第一个本地参数最前面 4096 个值，冷专家可能
  完全不更新，即使其他专家已正确更新。现在有界采样所有层、所有参数的每个本地专家；有梯度的
  owner 必须观察到更新，冷 owner 允许不变，但全局必须至少有一次专家更新。两个 collective
  无条件进入，不能因本地结果不同导致 collective 序列不一致。
- 更新探针仍是 smoke check，不宣称逐元素更新验收。原来的全量 loss/logits、所有参数梯度和
  更新后参数比较完全保留，阈值不变。探针及梯度快照只存在于 first-step validation，不进入稳态。
- JSON 增加完整 config、初始化 seed 和计时后的实际每-rank B-pool bytes；验证各层同一 resource
  group。这是逻辑持有字节数，不是 allocator peak，也不能证明没有临时 HBM 峰值。

## 2. 验证环境和范围

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

## 3. 复现与证据

完整示例说明见 [README](../../../../examples/torch/multicore/mega_moe/README.md)。
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

## 4. 本轮后优先级

1. 原 H5120/I1792/T4096 宽度的两层/四层正确性已补齐（第 5 节）。接下来先拆开 benchmark 的
   双模型/optimizer 常驻，再建立 connected model 的完整 optimizer-step matched ABBA、全 rank
   HBM 和真实路由/层间反向 profile。不直接外推小形状或独立层结果。
2. 从整模型 trace 重新定位剩余 receive/mixed batch 等待。rank 1 晚邻居窗口仍只是独立层证据；
   mixed send/receive 必须保留 native dW producer、owner FP32 聚合和 shared-B 复用依赖。
3. 后续仍需逐 step 换路由、connected checkpoint、更大层数以及 B/capacity 曲线。不因跨层 B
   变大就取消 residual multi-wave，也不无条件加入额外 persistent pool。

本轮没有证明优化达到上限；后续工作仍以正确性、端到端实测和 HBM 成本决定取舍。

## 5. 原目标宽度复验与基准内存限制

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

**优先级因此调整：** 下一轮先将 full-model accuracy 与 performance runner 解耦；common/MegaMoe
分别使用独立 fresh process，在 CPU 侧保存/比较逐 tensor 的初始化、loss/logits、梯度及更新后
参数，保留原 oracle。计时只驻留一个 backend，完成对称 warmup 与 ABBA，再采单模型全 rank HBM。
不能只删除 common 的 optimizer 或快照而让两侧计时占用不对称，也不能以取消每步 AdamW、
降低精度或缩小默认形状来掩盖基准限制。这个方案尚未实现，不计作 HBM 优化收益。

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

## 6. 本次停止边界

按用户要求在 2026-09-08 03:00（Asia/Shanghai）前收尾，保留已验收并直接本地提交的改动，
不 push，不在截止前仓促修改 mixed-batch/native 调度。**这是时间边界，不是优化上限。**
下一轮首项是第 5 节的单 backend 驻留基准改造；新的整模型性能结论必须等匹配 ABBA 和
单模型 HBM 完成后再给出。原 receive-only 的独立层收益与本轮 connected model 正确性证据分别保留。
