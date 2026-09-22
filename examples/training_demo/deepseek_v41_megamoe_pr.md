# DSV4.1 文本与 VLM 训练接入 MegaMoe

Trainer 基线为 `trainer_dev` 的 `edd4fae9`。在该基线上，按原顺序重放 `master`
截至 `605d5aa5` 的 multicore 及对应测试提交，再引入
[PR #847](https://github.com/mindspore-ai/hyper-parallel/pull/847) 的 `73f09787`，
再从 `megamoe-push-pull` 引入 `01839418` 的外部专家权重接口，
DSV4.1 Trainer 适配合并为顶部的一条提交；底层 multicore 来源提交独立保留。

重放范围包含 SHMEM stream-enqueue、MegaMoe 内存和大任务图、profiling、native
库加固、静态检查及对应测试。两个全仓 Platform/MindSpore 清理提交仅提取 multicore
和对应测试路径，保留 trainer_dev 的其余框架代码。打包元数据和 multicore 的 CI
检查名单随各自原提交完整保留。

引入 PR #847 前，multicore、其 UT 和 ST 目录均与上述 master 快照完全一致。
PR #847 尚未合入 master；当前以该 master 快照加 PR #847 的语义合并结果为对照，
上游实际合入后仍需重新比较。冲突处理保留 master 的字段读取、缓存刷新范围及代码结构，
增加 clipped SwiGLU 分支。适配与此文档位于 multicore 目录之外。

master 的示例使用 `components.modules.moe.GroupedExperts`；trainer_dev 尚未迁移该类，
因此在此路径提供对现有 Trainer 实现的导出，保持 multicore 示例和 ST 原文一致。

## 适配范围

- 保留 DSV4.1 原 router、路由权重缩放和 shared experts，使用全局 expert ID 调用 MegaMoe。
- routed 和 shared experts 的 limit 必须相同且为有限正数。发布配置的 `limit=10` 原样保留，
  不添加 limit=0 覆盖，也不将输出 clamp 当作中间激活 clamp 的替代。
- HF 权重 `[E,2I,H]`、`[E,H,I]` 转为 native 布局 `[E,H,2I]`、`[E,I,H]`。
  保留 `gate_up_proj`、`down_proj` checkpoint 名称，通过可逆 Transpose 转换恢复 HF 布局。
- 在现有 `adapter/replacements.py` 中定义专家参数 holder，不新增独立 MegaMoe 适配文件。
  在模型替换阶段创建 holder，沿用 Trainer 的 EP 切分、FSDP 包装和 optimizer。
  executor 使用 `create_parameters=False`，不分配占位权重。每次前向从当前参数取得本地
  Tensor，通过 `forward(..., expert_weights=(gate_up, down))` 直接传入；executor 不缓存
  外部权重。模型树中仍只有 holder 的两份专家参数，避免重复 checkpoint 和 optimizer 参数。
- MoE 调用仍经过 `module.experts(...)`，保留其嵌套 FSDP hooks。多层可共享上游执行资源，
  训练结束时在销毁分布式进程组前关闭资源。文本和 VLM 共用 `BaseTrainer` 的配置、
  workspace 共享和资源关闭步骤，不另建 MegaMoe Trainer 子类。
- 文本批次复用上游 `TextParallelBatch` 和 `DeepseekV41Runtime`，传入 V4.1 的
  `SharedCompressedPackedSequence`，保留 packed sample 边界。

multicore 的实现来自上述 master 提交、PR #847 和 `megamoe-push-pull` 的外部权重接口。
permute-grad 沿用上游对 `torch_npu.npu_moe_token_permute_grad_v2` 的调用，
使用已提供此接口的 `torch_npu 2.9.0.post6`，不添加自定义 permute-grad Torch binding。
外部权重接口保留默认自持参数行为，仅在显式传入权重时使用调用方参数。
不引入 push/pull 切换、动态容量、Muon 或 Indexer 优化。

## 配置与运行

复用现有入口和 YAML，不新增 MegaMoe 专用训练文件：

- 文本：`examples/training_demo/train_text.py` 和 `train_deepseek_v41_online.yaml`。
- VLM：`scripts/train_vl.py` 和 `examples/training_demo/train_deepseek_v41_vlm_online.yaml`。

两套配置的 `megamoe` 默认均为 `false`：保留原 `TextTrainer` / `VLMTrainer`、native EP、
数据管线及模型尺寸，不加载 multicore。原文本 EP16、VLM EP8/E16 默认值保持不变。
在 runtime 模块补齐现有文本 YAML 已引用的 `DeepseekV41TextBatch`。

VLM 接法参考 `megamoe-deepseek-v4.1` 的 `1b619fbc`：复用专家替换、原 `image_mask`
router、workspace 共享及关闭顺序。原分支由 benchmark 脚本手动配置；这里将配置和资源管理
接到共用的 `BaseTrainer`，正常文本/VLM 入口均可通过 `--megamoe=true` 启用。
保留原 `bias_vl` 与文本 bias 选路、shared experts、视觉编码器和图像插入流程。

启用后，专家执行 token 容量由解析后的配置统一计算：取 `max_seq_len` 与 packing
`token_budget` 的较大值，再向上对齐到 128；未指定 budget 时使用
`micro_batch_size × max_seq_len`。Omni packing 允许单个样本超过选择 budget，
因此容量至少覆盖完整的 `max_seq_len`。TP/CP/PP 必须均为 1，EP 组覆盖整个 world
且保持相同 rank 顺序。

变长 batch 仅在专家边界补零到固定容量，计算后裁回真实长度。补位使用均匀分布的全局
expert ID 和零路由权重，不进入原 router，不改变图像坐标、attention packed 边界或 loss
输入，也不贡献真实 token 和参数梯度。超过配置 token 容量直接报错，不截断输入。
这种补位会增加短序列的 MoE 计算和通信；性能报告需同时记录真实及补位后的 token 数。

以下文本命令覆盖为 EP8、全局 E48、每卡 6 个专家，保留四层、H5120/I2304、TopK6，
每卡专家执行 token 容量为 4096。

示例设置 `expert_capacity_factor=2.0`，使用上游固定容量检查。超出容量会显式报错，不丢弃
或截断路由。应根据真实输入检查接收量；可增大固定容量，或用 `null` 预留最坏路由容量，
后者会增加显存。所有共享执行资源的模块必须在第一次前向之前配置完成。

按构建文档从当前分支重新构建 native payload，并激活其 `set_env.bash`。不能复用旧分支
的 native 库。准备同版本模型配置、tokenizer、Engram assets 和 JSONL 文本后运行：

```bash
HYPER_PARALLEL_PLATFORM=torch torchrun --standalone --nproc-per-node=8 \
  -m examples.training_demo.train_text \
  examples/training_demo/train_deepseek_v41_online.yaml \
  --megamoe=true \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --training.global_batch_size=8 \
  --model.num_routed_experts=48 \
  --model.config_path="$MODEL_DIR" \
  --model.engram_assets_path="$ENGRAM_ASSETS" \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path="$MODEL_DIR" \
  --dataset.data_path="$TRAIN_DATA"
```

VLM 使用原入口，下面同样覆盖为 EP8/E48；视觉层数、数据变换、Omni packing 和 loss
设置沿用现有 VLM YAML：

```bash
HYPER_PARALLEL_PLATFORM=torch torchrun --standalone --nproc-per-node=8 \
  scripts/train_vl.py \
  examples/training_demo/train_deepseek_v41_vlm_online.yaml \
  --megamoe=true \
  --accelerator.ep_size=8 \
  --fsdp_config.dp_shard_size=8 \
  --training.global_batch_size=8 \
  --model.num_routed_experts=48 \
  --model.config_path="$MODEL_DIR" \
  --model.engram_assets_path="$ENGRAM_ASSETS" \
  --dataset.model_assets.config_path="$MODEL_DIR" \
  --dataset.data_path="$VLM_TRAIN_DATA"
```

该 crop 从配置初始化；加载实际 checkpoint 时应另行验证转换、保存和恢复后的训练轨迹。

## 验证

CPU 回归覆盖：真实 checkpoint 转换往返、meta 初始化、连续两次调用时使用当前权重、
输出与全部梯度、异常后不保留外部权重、无重复参数、router 调用次数、嵌套 hooks、
packed sample 边界、limit 和并行配置检查。另验证开关缺省和显式 `false` 的配置一致，
仍选择原训练器和 native EP；两种 Trainer 都在分布式配置归一化前处理 MegaMoe 开关。
新增变长 token 补位前后的输出、输入/路由/专家权重梯度对照，真实 V4.1 `bias_vl` 选路、
图像 token 梯度及 shared expert 梯度检查。CPU 数值测试以独立参考替代 native executor，
验证适配语义，不作为 native kernel 或整网 NPU 精度结论。关闭资源前先执行训练回调。

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/core/multicore \
  tests/ut/auto_models/models/deepseek_v41/test_megamoe_training.py \
  tests/ut/trainer
```

当前环境为 Torch 2.9.0、torch_npu 2.9.0.post6，已确认提供
`npu_moe_token_permute_grad_v2`。接入 VLM 后重新执行上述 CPU 回归；
结果为 **126 passed、397 subtests passed**，无失败。没有跳过缺失接口相关测试或放宽阈值。
独立进程验证文本和 VLM 默认 YAML 及共用初始化路径，`megamoe=false` 保持配置不变，
且不会导入 multicore。`git diff --check` 通过。适配与测试文件 pylint 通过；
`BaseTrainer` 的检查报告 3 条成员声明问题，涉及 `data_transform` 和 `num_micro_batches`。
对修改前源码重复检查得到相同 3 条报告，本轮未新增 pylint 问题。

额外收集整个 DSV4.1 UT 目录时，`test_deepseek_v41_crop.py` 因导入 batching 包中不存在的
`ParallelBatch` 而失败。该测试及 batching 代码均与 trainer_dev 基线相同；本轮未修改或
跳过该用例，上述通过数量仅对应列出的回归范围。

切换 Torch/torch_npu 版本后，native payload 需要从当前源码重新构建，再验证设备加载
及 NPU 精度。此前自定义 binding 的构建和双卡测试结果不作为当前实现的验证证据。

旧基线 EP8/E48、limit10 的 NPU 检查未通过。定位发现，仅移植 PR #847 时遗漏其基线依赖：
trainer_dev 的 `getTaskDesc()` 没有读取 `extra_value_0`，导致设备端 clipping 分支选择
不确定。PR #847 原始基线已具备该读取逻辑；此次完整重放 master 历史将它一并引入。

NPU 验证应先检查无 clipping 和 limit=1/10 的前后向，再以 EP8/E48、固定权重和路由
对照独立 FP32 参考，检查输出、输入梯度、router 权重梯度和本地专家权重梯度，输入须
实际触发 clamp。逐算子对照使用各算子的实际输入，区分算子误差与 BF16 中间舍入累积。
整块精度通过后运行 Trainer 前向、反向和 optimizer step。不得通过放宽阈值掩盖问题。
新基线尚未取得 MegaMoe 整块 NPU 精度或文本/VLM 整网通过结论；VLM 还需在目标设备上
验证视觉编码器到 loss 的完整前反向和 optimizer step。

性能需在精度通过后单独测量：固定 canonical 权重、数据、学习率计划和任务队列设置，
基线使用 trainer_dev 原生 EP，候选使用本适配，按独立进程 ABBA 顺序运行。
同时报告整步吞吐、四层 routed MoE 前后向合计和两种显存口径：Torch allocator peak
与整卡 HBM 采样峰值。MoE 诊断应单独运行，避免同步计时影响整步结果。
不沿用旧实验分支的整网加速比；本分支暂不声明性能收益。
