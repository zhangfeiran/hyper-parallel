# DeepSeek-V4.1 MegaMoe Trainer 接入

## 实现边界

[可选配方](../../../../examples/training_demo/train_deepseek_v41_megamoe.yaml) 基于原四层
DSV4.1 text 配方，增加 expert replacement 和整个 MoE 边界的 local compute factory。
当前限定 TP=CP=1、learned routing、固定本地 token 数；验证使用 `swiglu_limit=0`，
同时关闭 routed/shared 的 clamp。原始 limit=10 模型不属于本次精度结论。

[replacement](../../../models/deepseek_v41/adapter/megamoe_training.py) 保留
`mlp.experts.gate_up_proj/down_proj` 名称与 expert 维，参数布局改成
`[E,H,2I]` / `[E,I,H]`，由可逆 `WeightConverter(Transpose)` 连接 HF checkpoint。
参数转换发生在分片、FSDP 和 optimizer 创建前；meta materialization 后显式初始化。
`initializer_range` 默认 0.02，使用不同源初始化标准差时须同步修改 replacement 配置。

原 `mlp.experts` 是唯一参数持有者，仍是嵌套 FSDP 单元。
无参数 MegaMoe executor 每次读取当前 unshard 后的 expert 参数，不保存参数副本或旧 view。
所有层在 forward 前登记静态执行规格，首次 native 调用才分配资源并按全部层计算 SHMEM heap。
当前各层拥有独立 workspace；串行层共享 workspace 留待后续。

归约契约沿用框架：MegaMoe 完成 token dispatch、owner expert compute 和 token combine；
expert dW 不额外做 EP all-reduce。同一 expert 的副本由 EDP/FSDP mesh 归约，
`reduce_dtype=float32` 只控制通信精度，不代表 kernel 以 FP32 输出局部 dW。
训练 ST 明确检查 expert FSDP mesh 只包含 `edp_shard/edp_replicate`。
Router/shared/dense 参数继续由原 dense FSDP 域管理，未修改通用 optimizer。

## Owner EP 数值对照

[block 入口](../examples/mega_moe/deepseek_v41_precision.md) 默认改为真实
`deepseek_v41_ep_compute_fn`：A2A 将 token 发到 expert owner 后计算，再返回源 rank。
该参考没有额外的 BF16 expert dW all-reduce。
历史 full-HF-replicated 诊断仍可通过 `--reference hf_replicated` 运行。
独立 CPU FP32 oracle 仍将各源 rank 数学梯度以 FP32 求和，构造 owner 梯度参考。

6 个 NPU 用例通过：push/pull × learned EP2、hotspot EP2、visual learned EP1 subgroup。
每组包含三步同状态输出、dX、路由权重梯度、所有参数梯度及 SGD 更新检查。
FP32 判据保持每 tensor 相对 L2 ≤1%、峰值归一化误差 ≤2%，以及精确结构/路由检查。

| 对照路径 | 最坏相对 L2 | 最坏峰值归一化误差 |
| --- | ---: | ---: |
| DSV4.1 owner EP BF16 → FP32 | 0.6063% | 1.0453% |
| MegaMoe → FP32 | 0.5196% | 0.9052% |

这些是固定路由、每步同状态的数值证据，不是完整模型独立训练轨迹相等的证明。

## Trainer 与 checkpoint 验证

[训练 ST](../../../../tests/torch/multicore/_test_deepseek_v41_megamoe_training.py)
使用真实 `TextTrainer` 构建路径、四层 DSV4.1 crop、Engram、Full/Reindex attention、
原路由/shared experts、Muon/AdamW、梯度裁剪和学习率调度。
H=512、I=128、E=4、TopK=2，每 rank 每 micro-batch 128 tokens，每步累积两次，连续三步。
输入为确定性 token 数据，非真实预训练数据；MegaMoe 使用实际 910B kernel。

EP2/EDP1/push 已通过三步训练，所有 rank 的八个 expert 参数 tensor 均更新且有限。
第 2 步将模型 DCP checkpoint 写盘，破坏当前权重，再恢复并逐元素检查一致，随后完成第 3 步。
CPU 测试另验证完整 recipe 的 HF checkpoint 反向转置与初始参数逐元素一致。
四卡 EP2/EDP2/pull、FP32 参数存储也通过相同三步训练及模型 checkpoint 检查。
BF16 参数存储/FP32 主参数的四卡专项覆盖同样通过。

| World / EP / EDP shard | Transport | 参数存储 | 结果 |
| --- | --- | --- | --- |
| 2 / 2 / 1 | push | FP32 | 三步、累积、权重 checkpoint 通过 |
| 4 / 2 / 2 | pull | FP32 | 三步、累积、权重 checkpoint 通过 |
| 4 / 2 / 2 | pull | BF16 + FP32 主参数 | 三步、累积、权重 checkpoint 通过 |

训练 ST 显式将注意力、Sinkhorn、mHC post 选为仓库已有的等价 PyTorch 数学实现，仍在 NPU 上执行。
这是隔离 MegaMoe 接入的测试边界：本机最初缺少额外 CANN 算子；激活已有 Omni vendor 后，
小模型的融合 attention/mHC post 调用仍失败。配方保留原 fused 路径，运行需匹配的依赖和形状。
不能把本 ST 通过表述为所有融合算子或原始大模型已验证。

## 当前未覆盖及发现的问题

- 完整模型 logits/所有梯度对独立 FP32 参考、长期训练轨迹、真实 4K 配方尚未验收。
- 全量 optimizer checkpoint 恢复未通过：尝试保存 model+optimizer 后，恢复
  `model.layers.0.attn_hc.scale` 的 Adam `exp_avg/exp_avg_sq` 出现 `[2]` 与 `[1]` shard shape mismatch。
  该参数不属于 MegaMoe。本轮只验证模型权重 checkpoint，没有修补通用 DCP/optimizer，
  不能据此宣称完整训练状态断点续训通过。
- TP/CP/PP 组合、完整 VLM Trainer、跨拓扑 checkpoint、共享 workspace、性能仍待验证。
- 用户已允许 busy NPU；本轮保留占用记录，只作正确性验证，不报告性能收益。

## 复现

激活当前 checkout 的 editable 安装、CANN 和 Torch multicore payload 后执行：

```bash
export HYPER_PARALLEL_PLATFORM=torch OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python -m pytest -q tests/torch/multicore/test_deepseek_v41_megamoe.py
python -m pytest -q tests/ut/auto_models/models/deepseek_v41 tests/ut/core/multicore
```

实际 text 训练入口为 `python -m examples.training_demo.train_deepseek_v41_megamoe <recipe.yaml>`，
用 torchrun 启动。该入口在 Trainer 完成 callbacks 后、销毁 EP process group 前，
按模型顺序释放 native/SHMEM 资源。示例 recipe 的资产路径是占位符；
先按训练 demo 的流程准备模型/Engram/tokenizer/数据，并激活兼容的额外融合算子。
`local_num_tokens` 必须等于每次 MoE 边界实际 token 数，且为 128 的倍数；
改变 micro-batch 或打包长度时同步更新，当前不支持动态长度。

本轮原始日志与 JSON 位于本机
`/home/feiran/doc/dsv41-megamoe-implementation-20260917.sORKhN/`：
`owner-ep-st-results`、`real-trainer-v5`、`training-final-st-results`、
`training-dtypes-st-results`、`training-lifecycle-st-results`。
`real-trainer-v4.log` 保留 optimizer checkpoint 的失败证据。

最终检查：CPU `160 passed, 627 subtests passed`；NPU block+Trainer 矩阵
`8 passed`，追加参数存储 dtype 矩阵 `2 passed`；带有序 SHMEM 关闭的四卡 BF16
Trainer 用例 `1 passed`。这些追加运行覆盖后续代码变化，不代表额外的独立精度场景。
Pylint 与仓库静态检查通过；静态检查提示本机未安装 codespell。
