# DeepSeek-V4.1 MegaMoe 适配计划

## 目标与当前状态

在 `upstream/deepseek-v4.1-preview` 的模型和训练框架上接入现有 `MegaMoeExperts`，
先验证关闭 SwiGLU clamp 后的 routed-expert 前后向精度，再接入完整训练链路。
首轮验证约定 `swiglu_limit=0.0`；baseline 和 MegaMoe 使用相同的配置、权重和路由。

已完成分支准备、提交迁移、独立依赖环境、limit override、无 clamp 激活分支、
独立 MoE adapter、native 重编译及 block 精度入口。独立 CPU FP32 oracle 已补齐，
两轮各 16 组 NPU 矩阵已完成。P1 已确认采用同状态 FP32 数值判据，并补充真实 owner EP 对照。
P2 已新增 Trainer replacement、训练配方及多层 EP/FSDP 验证入口；具体通过范围与未完成项见
[训练接入报告](deepseek_v41_training_report.md)。P3 拓扑/性能扩展尚未完成。

## 分支与提交来源

- 目标基线：`upstream/deepseek-v4.1-preview`，`8786d9439b95bcfd151e0ae40d9cb1211a2de3e0`。
- 来源分支：`megamoe-push-pull`，迁移起点为 `c1334800593da18929348340a60a5a07014f33cd`。
- 工作分支：`feat/deepseek-v4.1-megamoe`。
- 完整迁移后代码提交：`cf6de38e`；追加 optimizer 回退后的代码提交：`3ff76bdb`。
- 按依赖顺序完整 cherry-pick 以下 14 个非 merge 提交，未拆分提交内容；
  每对提交的 `git patch-id --stable` 均一致。

| 来源提交 | 新提交 | 内容 |
| --- | --- | --- |
| `f7439d98` | `69079596` | SHMEM 通信栈重构及相关检查配置 |
| `83be5c96` | `3b1f47fe` | MegaMoe 显存、ready 复用及大任务图 |
| `fda6c58c` | `7de75ffd` | MegaKernel profiling |
| `81128611` | `b2bda570` | 清理 backward 绑定 |
| `23e5baf6` | `4eaa762a` | profiling runtime 测试适配 |
| `c53926d9` | `c3216156` | native 库加固及 wheel commit 元数据 |
| `e17359d1` | `14a25b51` | SHMEM 测试环境激活 |
| `ea0e29b7` | `45238df3` | metadata-only permutation gradient 的 ACLNN 调用 |
| `3b337a56` | `09e5601d` | ready 失败注入测试 |
| `8ade7bb6` | `924a97c2` | MegaMoe 显存优化及 optimizer 直接参数回拷 |
| `1b03f88c` | `81d10545` | push/pull transport 统一 |
| `5cb09343` | `bf131212` | 移除固定 expert 数量限制 |
| `148bc6a3` | `88788127` | SHMEM EP subgroup bootstrap |
| `c1334800` | `cf6de38e` | transport 集成回归 |

为保留原提交边界，`c53926d9` 包含的 `setup.py` 修改，以及 `8ade7bb6` 包含的
`components/optim/mixed_precision_optimizer.py` 修改和参数回拷测试均先完整迁入。
随后按要求在来源分支新增一个回退提交，再完整同步到本分支：

| 来源提交 | 新提交 | 回退内容 |
| --- | --- | --- |
| `2632ba92` | `3ff76bdb` | 恢复 optimizer 先 `.to(device, dtype)` 再 `copy_` 的参数回拷，并删除新增的 `test_parameter_copy.py` |

因此本分支相对 DSV4.1 基线不再包含 optimizer 文件和该测试的改动；wheel 元数据修改保留。
当前分支没有引入来源分支中无关的 FSDP、DTensor、Platform、compile 等重构提交。

## 已确认的接入边界

| 位置 | 当前行为 | 适配要求 |
| --- | --- | --- |
| [DSV4.1 router/model](../../../models/deepseek_v41/modeling_deepseek_v41.py) | router 返回 logits、weights、indices；支持文本/图像 correction bias | 保留 router 原语义及 `image_mask`，只替换 routed experts 执行 |
| [DSV4.1 EP adapter](../../../models/deepseek_v41/adapter/expert_parallel.py) | 通用 EP dispatch/combine 后执行本地 experts，再加 shared experts | MegaMoe 自带 dispatch/combine；新路径必须在完整 routed 分支入口切换，避免重复通信 |
| [MegaMoeExperts](../modules/mega_moe/module.py) | 接收 hidden states、全局 expert IDs 和 weights；持有本地 expert 参数 | 绑定正确 EP subgroup，保留 weights 梯度；shared experts 先沿用模型实现 |
| [配置构造](../../../../examples/training_demo/cropped_deepseek_v41.py) | 从源 `text_config["swiglu_limit"]` 传入 HF config | 已支持构造前显式覆盖，并保留源值；普通 baseline 默认仍沿用源值 |
| [参数布局与通用 EP](../../../distributed/expert_parallel/experts.py) | fused `gate_up_proj [E,2I,H]`、`down_proj [E,H,I]` | MegaMoe 参数为 `[E_local,H,2I]`、`[E_local,I,H]`；需要显式转换 |
| [模块替换](../../../models/replacement.py) | 默认要求参数身份、注册名和 state dict 不变；支持 `make_transforms()` | 参数改名、转置必须提供 weight conversion，不能直接赋值后声称兼容 checkpoint |
| [DSV4.1 注册](../../../models/deepseek_v41/adapter/registration.py) | 声明 Engram、视觉及 decoder 的 FSDP 包装/执行顺序 | 保留声明；新 experts FQN、EP 参数元数据和嵌套 FSDP 单元必须一致 |

DSV4.1 router 不能替换成 Qwen 示例里的 softmax router。当前实现可使用
`sqrtsoftplus`，correction bias 只参与 expert 选择；权重来自无偏置的 scores，
TopK 大于 1 时归一化，再乘 `routed_scaling_factor`。
输出合并与 router 权重不能重复缩放，shared experts 也不能重复相加。

## SwiGLU limit 的首轮处理

仓库的 [DSV4.1 测试](../../../../tests/ut/auto_models/models/deepseek_v41/test_deepseek_v41_crop.py)
明确设置 `swiglu_limit=10.0`，EP adapter 将 HF experts 的 `_apply_gate` 传给本地计算，
并因 clamp 语义拒绝 `use_grouped_gemm=true`。当前 MegaMoe 公共 API 和
[SwiGLU task 配置](../tasks/swiglu.py) 没有 limit 参数。

首轮落实方式：

1. 在 DSV4.1 验证配置构造入口增加显式 `swiglu_limit` override，并从模型构造入口透传。
   MegaMoe 精度 recipe 设置 `0.0`；同一份 baseline recipe 也设置 `0.0`。
   原模型资产的 `config.json` 保留原值，报告记录源值和生效值。
2. 使用实际依赖版本的 HF `_apply_gate` 确认 `0` 表示关闭 clamp。
   用正负大幅值输入检查输出及梯度是否等于普通 `silu(gate) * up`，
   不能把“截断到零”误当成“关闭截断”。若依赖不支持此约定，需显式实现验证用无 clamp 分支。
3. 在创建 routed/shared expert 实例前应用配置；检查实例是否缓存 limit。
   配置中的零必须落实到两条执行路径，仅修改 `model.config` 不足以证明生效。
4. MegaMoe adapter 对非零 limit 明确拒绝。保留普通 DSV4.1 baseline 的 limit=10 回归；
   不能以本轮结果宣称已经验证原始 clamp 模型精度。

已在隔离环境实测 Transformers 5.13.0：HF 的零值仍会执行 clamp，并将 up 截断为零。
当前实现仅对 limit=0 的实例绑定无 clamp 方法，同时覆盖 routed/shared 分支；
limit=10 保留 HF 方法及参数身份。CPU 测试包含大幅值输出/梯度、配置透传和真实模型构造。

## 分阶段实施

### P0：准备模型依赖和可复现输入

- 准备隔离的、包含 `transformers.models.deepseek_v4` 的依赖环境。
  目标分支的 [环境记录](../../../../docs/guide/trainer/current_hf_model_environment.md)
  使用 Transformers 5.13.0；本机当前为 4.57.6，不能直接沿用文档中的其他主机路径。
- 固定模型 config、Engram 资产及 tokenizer 的 revision/hash。
  使用现有四层 crop，先关闭 vision、TP、CP、PP、compile 和 activation checkpoint。
  实际 H/I、全局 experts、TopK、每 rank token 数全部写入结果。
- 按当前 checkout 重编译和激活 Multicore native payload，核对 editable import、
  component root、vendor 路径、Torch/torch-npu/CANN 版本；旧分支的二进制不自动构成当前验证。
- 执行上述 limit=0 配置与无 clamp 数学检查，再开始模型比较。

交付：可复现环境清单、实际生效配置、DSV4.1 CPU baseline UT 结果。

### P1：独立 MoE block 的精度闭环

- 新增最小 DSV4.1 MoE 验证入口，位于 `core/multicore/examples/mega_moe/`；
  使用真实 DSV4.1 router 和 shared experts，baseline 为原 DSV4.1 执行路径。
- 固定同一份全局初始权重，按 EP rank 截取连续 expert 区间，再将最后两维转置为
  MegaMoe layout 并 contiguous。校验 gate/up 顺序；初始化、梯度比较和 checkpoint 逆变换一致。
  每次 forward 都转换整份参数会引入额外开销，不作为最终方案。
- `local_num_tokens` 为进入 MoE 的实际本地 token 数，必须满足当前 128 对齐约束；
  首轮使用固定 batch/sequence，变长、尾 batch 和 padding 另列用例。
- `expert_capacity_factor=None` 保持 lossless；保留 learned routing，不用均匀路由替代。
  覆盖空 expert、热点 expert、不均衡、TopK 边界及多个连续 step。
- 先测 EP1，再 EP2/EP4；绑定 `ep_mesh.get_group("ep")`，不能假设 EP group 等于 WORLD。
  分别运行 push 和 pull，按实际接收量验证容量与多 wave 行为。

交付：output、dX、routing-weight gradient、expert dW、router/shared gradient 和
optimizer 更新后的参数比较。路由离散索引不要求梯度；correction bias 的梯度有无须与 baseline 一致。

### P2：接入 Trainer、EP 和 FSDP

- 在 `models/deepseek_v41/adapter/` 增加可选 MegaMoe routed forward，
  保留普通 EP baseline。不得将 MegaMoe 挂进已经完成 EP dispatch 的本地 expert 回调。
- 使用现有 module replacement 与 `make_transforms()`/weight conversion 机制处理
  `gate_up_proj/down_proj` 到 native 参数布局的转换，并实现初始化、加载和保存闭环。
- Trainer replacement 保留 `gate_up_proj/down_proj` 名称，沿用 expert-axis `Shard(0)`
  和 source-shard metadata；独立 block 的 `gate_up_weight/down_weight` 不进入 planner。
  先保证每 rank 持有完整本地 experts，不引入 expert 内 TP。
- 验证替换发生在参数分片和 optimizer 构造之前，meta materialization 后 BF16、连续性和形状正确；
  FSDP 每次 unshard 的当前参数供执行使用，不缓存过期 storage/view。
- 保留 `mlp.experts` 的嵌套 FSDP 单元及 DSV4.1 adapter 执行顺序，验证 root 不重复管理 expert 参数。
  运行四层 crop，比较 logits、loss、全参数梯度和至少三个 optimizer step。

交付：可切换 baseline/MegaMoe 的训练 recipe、checkpoint round-trip 及完整精度结果。

### P3：扩展拓扑、资源共享与性能

- 增加 TP/SP、CP、PP subgroup，再加入 VLM `image_mask`/`bias_vl`。
  以进入 MoE 边界后的 token 布局计算容量，核对 EP 分组与 shared 分支通信归属。
- 串行层按资源兼容性共享 execution resources；覆盖多层 backward、gradient accumulation、
  activation checkpoint 重入、close/recreate 和多 EP subgroup。
  在 PP 存在并发或重叠执行时，先证明资源生命周期和互斥关系再启用共享。
- 精度通过后做同配置、同 native payload 的 fresh-process ABBA，分别统计编译/初始化和稳态。
  NPU 作业经过空闲队列与最终占用复查，记录期间的设备使用者；混入其他进程的性能块重跑。
- 报告 peak allocated/reserved、SHMEM heap 和 workspace，区分参数/activation 与通信常驻空间。
  limit=10 的 native forward/backward 支持另立后续阶段。

## 验证与验收

| 层级 | 检查 | 通过条件 |
| --- | --- | --- |
| 分支迁移 | 14 对 patch-id、目录 diff、基线祖先关系 | 提交完整，Multicore 源码和测试与来源一致 |
| CPU | 配置 override、激活边界、权重双向映射、替换/分片元数据 | 正常路径一致，非零 limit 和不支持配置明确报错 |
| NPU block | 固定 routes 与真实 router 两组；forward/backward/update | 相同初始状态，逐项有限性及误差检查 |
| NPU model | 四层 crop、loss、所有应有梯度、optimizer state、保存恢复 | 多步一致，无丢失/重复参数或通信 |
| 拓扑扩展 | EP subgroup、TP/CP/PP、VLM、重算/资源复用 | 各组合独立记录，不从单一配置外推 |

初始 BF16 比较阈值沿用现有 MegaMoe ST 的 `rtol=2e-2, atol=2e-3`，
同时输出最大绝对误差、相对误差和失败 tensor 名；这不是预先保证 DSV4.1 会通过。
权重初始化/布局变换应精确一致。超阈值先定位数值与语义来源，不为通过测试直接放宽阈值。

迁移验证记录（2026-09-17，完整迁移后、optimizer 回退前 `cf6de38e`）：

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/core/multicore \
  tests/ut/auto_models/trainer/test_parameter_copy.py \
  tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py
```

结果：`140 passed, 616 subtests passed`。其中 Multicore 为 124 项，optimizer 为 16 项。
上面的参数回拷测试文件已在后续回退中删除，该命令仅记录当时的验证范围。
有已有环境/pytest marker 警告。HyperParallel editable import 指向当前 checkout。

optimizer 和测试回退后，在新分支重跑
`tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py`：
`15 passed, 18 subtests passed`。amend 前后的代码 tree 完全一致。
optimizer 文件和已删除测试相对 DSV4.1 基线无 diff，Multicore 源码/测试相对来源分支无 diff。
`git diff --check`、文档本地链接检查通过。

分支准备阶段，旧环境的 DSV4.1 CPU UT 曾在 collection 阶段报
`ModuleNotFoundError: No module named 'transformers.models.deepseek_v4'`。
实施阶段已使用独立 Transformers 5.13.0 环境解决，旧环境未修改。

## 实施记录（2026-09-17）

- 隔离环境使用 Transformers 5.13.0、Torch/torch-npu 2.9.1、torchdata 0.11.0。
  当前 checkout 的 editable import 已确认；CANN 9.1.0 的 910B native payload 已重编译。
- 本机 CANN 的第三方 vendor 配置不可读，编译使用仅链接同版本内置文件的独立 OPP 视图。
  runtime 改回原始 OPP 路径后通过启动；未修改系统安装或其他用户的 vendor。
- [block adapter](../../../models/deepseek_v41/adapter/megamoe.py) 保留原 router/shared 模块，
  按 group-local rank 转换一次完整 expert 权重；拒绝非零 limit、hash routing 和错误布局。
  参数名称发生变化，因此该 adapter 限于独立 block；后续 Trainer 接入使用独立的、保留 HF 参数名的 replacement。
- [精度入口](../examples/mega_moe/deepseek_v41_precision.md) 覆盖真实 learned routing、
  hotspot/空 expert、push/pull、EP1/EP2/EP4、两个 strided EP2 subgroup、视觉路由、TopK=1/8。
  每组连续三步 SGD，保留原 BF16 门槛和完整逐 tensor 诊断。
- [独立 FP32 oracle](../examples/mega_moe/deepseek_v41_oracle.py) 使用 CPU FP32 原语和 autograd，
  不调用 HF expert 或 native 实现；固定实际离散路由，使用每条路径当前权重重算连续函数。
  expert 梯度在实际 EP group 内以 FP32 归约；同时比较一次 SGD 更新。
  该 oracle 不是独立连续三步的 FP32 训练轨迹，TopK 离散选择仍由 BF16 baseline 精确对比覆盖。
- 按用户指示允许 busy NPU，继续记录进程占用。当前运行仅作精度验证，没有性能结论。
- 原 BF16 `rtol=0.02, atol=0.002` 判据保留：热点和部分多卡用例有少量 expert dW 超阈值。
  FP32 复核还发现未改动的 shared expert BF16 梯度也会触发同一逐元素门槛。
  因而必须分别报告数值误差与集成语义检查，不能通过放宽阈值直接宣称验收完成。

CPU 最终全量范围为 `tests/ut/auto_models/models/deepseek_v41` 与
`tests/ut/core/multicore`，结果 `156 passed, 610 subtests passed`。
其中新增文件聚焦测试为 `10 passed, 13 subtests passed`，包含独立 FP32 oracle 和
HF text / V4.1 text / V4.1 visual router；原有 limit=10 和 Multicore 回归同时通过。
该阶段四层模型配置/构造只在 CPU 测试覆盖；后续 P2 进展见训练接入报告。

同状态的 16 组均满足初始参数/路由 IDs/梯度存在性精确匹配。
MegaMoe 对 FP32 的最坏相对 L2 为 0.5177%，峰值归一化误差为 1.0054%；
HF BF16 对 FP32 分别为 0.5482% 和 1.1716%。原 BF16 逐元素硬门槛通过 5/16 组。
用户已确认同状态 FP32 判据：每 tensor 相对 L2 <= 1%、峰值归一化误差 <= 2%，
精确检查形状、梯度存在性、初始权重和路由 IDs，双方 baseline 使用相同标准。
默认入口及 ST 已切换到此判据，原 BF16 差异报告和显式旧门槛模式保留。
本阶段不更改通用 optimizer，也不恢复已删除的参数回拷测试。

FP32 数值结果、独立训练轨迹的路由分歧，以及同状态验证协议见
[block 精度实施记录](deepseek_v41_block_precision_report.md)。

算子偏差追踪入口及数值证据见 [算子精度定位报告](deepseek_v41_operator_precision_report.md)。
本轮只改变验收入口和诊断工具；融合 SwiGLU、归约边界的替换只用于诊断副本。

## Trainer 接入进展（2026-09-17）

新增 [Trainer expert replacement](../../../models/deepseek_v41/adapter/megamoe_training.py)
与 [可选训练配方](../../../../examples/training_demo/train_deepseek_v41_megamoe.yaml)。
所有层在首轮 forward 前登记执行规格，使固定 SHMEM heap 一次覆盖完整模型。
专家参数只由原 `mlp.experts` FSDP 单元持有，每次 forward 使用当轮 unshard 的参数。
专家梯度按已有框架的 EDP mesh 归约，未新增 EP dW all-reduce，未修改通用 optimizer。

[训练接入报告](deepseek_v41_training_report.md) 区分 owner EP 数值验收、完整 Trainer 冒烟、
模型权重 checkpoint，以及尚未验证的融合注意力/mHC 与 optimizer checkpoint 恢复。
