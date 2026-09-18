# DeepSeek-V4.1 MegaMoe 适配说明

## 变更范围

本分支以 `upstream/trainer_dev` (`162aa8e1`) 为实际基线，适配 DSV4.1 的
learned routing MoE 到现有 Torch MegaMoe。代码适配提交截至 `f4ae1a7a`；
本文件只汇总实现契约和验收方法。四层训练 crop 用于验证完整训练链路，不能
等同于官方 40 层模型。

MegaMoe 部分主要继承自 `megamoe-push-pull` (`2632ba92`) 的代码线，随后重放
到 `trainer_dev` 基线。由于重放时基线和周边平台代码不同，提交 SHA 已变化，
但核心 patch 可以按以下代表性映射核对：

| `megamoe-push-pull` | 当前分支 | 内容 |
| --- | --- | --- |
| `f7439d98` | `f3237463` | SHMEM one-sided runtime |
| `83be5c96` | `acb12a7f` | workspace/大任务图和显存优化 |
| `fda6c58c` | `2640a41d` | MegaKernel profiling |
| `ea0e29b7` | `d1f8ba84` | permutation gradient 的 ACLNN 适配 |
| `8ade7bb6` | `8a90fab8` | MegaMoe memory savings |
| `1b03f88c` | `c852d237` | push/pull transport |
| `5cb09343` | `343c9a8f` | 移除固定 expert 上限 |
| `148bc6a3` | `ba4c4f4a` | subgroup SHMEM bootstrap |
| `c1334800` | `9ecab8b4` | push/pull 集成测试 |
| `2632ba92` | `56223e9a` | optimizer parameter-copy 回退 |

在这条 MegaMoe 代码线上，当前分支再叠加 DSV4.1 FP32 oracle、算子定位、
Trainer/FSDP replacement、四层训练 benchmark 和 `swiglu_limit=10` 支持。

当前实现保持 DSV4.1 的 routed/shared expert 语义、TopK router、shared
expert、attention、Engram、indexer、mHC 和优化器配置。TP/CP/PP 仍限制为 1，
EP 可以是 WORLD 或显式子组。

## 适配方式

### 独立 MoE block

`DeepseekV41MegaMoe` 保留源 block 的 router 和 shared experts，只替换 routed
experts。构造前要求完整 HF expert 权重，按 EP group-local rank 截取连续专家，
一次性将 HF 布局
`[E, 2I, H]`/`[E, H, I]` 转为 native 布局
`[E_local, H, 2I]`/`[E_local, I, H]`。运行时不复制旧参数，native executor
读取当前参数。

### Trainer/FSDP

`DeepseekV41TrainingExperts` 在参数分片和 optimizer 创建前完成 replacement，
保留 `gate_up_proj`、`down_proj` 名称，通过 `WeightConverter(Transpose)` 完成
checkpoint 双向布局转换。原 `mlp.experts` 仍是参数唯一持有者和嵌套 FSDP
单元；每次 forward 使用当前 unshard 参数，不缓存过期 view。

`deepseek_v41_megamoe_compute_fn` 只替换整个 routed 分支，router 和 shared
branch 仍由 DSV4.1 adapter 执行。所有兼容层在首次 native forward 前登记静态
规格，串行层可以共享一个 execution workspace；关闭必须在所有 backward 完成
后按层有序调用 `close()`。

### SwiGLU limit

DSV4.1 使用 released `swiglu_limit=10`。当前 adapter 对 routed/shared 两支都
要求相同的有限正数，并把该值传入 MegaMoe forward/backward graph；native
clipped SwiGLU 与 HF 的正向/反向语义一致。适配代码和训练 recipe 不再添加
`swiglu_limit=0` 覆盖，也不使用 `limit=1` 作为模型默认值。

### EP 梯度归约

owner EP 的 dispatch 将 token 发到 expert owner，owner 对收到的全局 token 计算
该 expert 的输出和 dW；同一 expert 的 dW 不再额外做一次 EP all-reduce。Trainer
中的 expert 副本归约沿用既有 EDP/FSDP mesh，通信 dtype 仍由框架的
`reduce_dtype` 控制。MegaMoe 负责 dispatch、owner compute、combine，不改变
通用 optimizer 或 dense DP 梯度归约。

### Push / pull

`MegaMoeExperts(..., dispatch_mode="push" | "pull")` 在构造时选定模式，所有
EP rank 必须一致，且不同模式不共享同一 workspace。

- `push`：按 lossless 最大接收容量分配对称 SHMEM receive 区，使用 PUT dispatch。
- `pull`：SHMEM 保存本地 source，接收 rows 使用普通 HBM，dispatch 使用 GET；
  combine 仍走 PUT。
- 每次独立 torchrun 使用新的
  `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`，进程组销毁前显式关闭所有
  MegaMoe executor。

## 精度验收

精度入口为
`hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_precision`，
Trainer 入口为
`examples.training_demo.benchmark_deepseek_v41_megamoe`。两条路径都固定初始
权重、输入和离散 route；比较 output、selected IDs、dX、router/shared 梯度、
每个 expert 梯度以及一步 optimizer update。

主判据是同状态独立 CPU FP32 oracle：每个 tensor 的 relative L2 不超过 1%，
最大绝对误差除以 reference 最大值不超过 2%；同时要求 shape、finite、梯度
存在性、初始权重和 route IDs 精确一致。HF BF16 逐元素比较仍作为诊断保留，
但不通过直接放宽阈值来掩盖 BF16 舍入差异。

已归档的 NPU limit=10 复测结果：

- `/home/feiran/doc/dsv41-megamoe-swiglu-limit-20260918-retry2.json`：EP2、
  learned route、FP32 oracle，`passed=true`，HF 和 MegaMoe 均通过 FP32 判据。
- `/home/feiran/doc/dsv41-megamoe-swiglu-clamp-probe-final-20260918.json`：
  真实超限/未超限输入的 clipped SwiGLU probe，`passed=true`。该 probe 使用
  `limit=1` 验证算子边界，不代表模型配置；模型配置仍为 10。

上述 JSON 的源码身份是 `85a5ae97` 加各自记录的 staged diff；该 diff 随后
提交为当前的 `f4ae1a7a`。native Torch adapter SHA256 为
`35d6923f5cbe280eb60e00d44a37248e7042bda7b439d09874c9f3be63ad416a`。因此这些
结果证明的是与当前提交相同的源码树，但后续重跑仍应重新记录当前 checkout 的
SHA 和 payload hash。

CPU 回归覆盖 DSV4.1 model adapter、Multicore、replacement 和 oracle；历史完整
范围为 `156 passed, 610 subtests passed`。NPU 结果必须同时记录 branch SHA、
源码文件 hash、Torch/torch-npu/Transformers、CANN、active OPP 和 native
payload hash；旧 payload 或旧 branch 的 JSON 不能直接作为当前验收。

## 性能与 HBM 测试方法

性能比较只使用 fresh-process、同一 canonical rank-local BF16 权重和确定性
输入。EP8/E48 的含义是全局 48 个 routed experts、每卡 6 个；原始尺寸裁剪保持
`H=5120`、`I=2304`、`TopK=6`、每卡 4096 tokens，四层用于整网链路验证。

### 整网

先由 owner EP 写一次权重，再对 owner、MegaMoe push、MegaMoe pull 分别启动独立
torchrun。稳定测试建议 `warmup=8`、`steps=10`、`schedule_steps=18`：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.benchmark_deepseek_v41_megamoe \
  --model-dir /path/to/DeepSeek-V4.1-Flash \
  --engram-assets /path/to/engram.json \
  --backend owner_ep --experts 48 --tokens 4096 \
  --warmup 8 --steps 10 --schedule-steps 18 \
  --weights /path/to/canonical-weights --output /path/to/owner

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.benchmark_deepseek_v41_megamoe \
  --model-dir /path/to/DeepSeek-V4.1-Flash \
  --engram-assets /path/to/engram.json \
  --backend megamoe --dispatch-mode push --experts 48 --tokens 4096 \
  --warmup 8 --steps 10 --schedule-steps 18 \
  --weights /path/to/canonical-weights --output /path/to/megamoe-push

# pull 只需将 dispatch-mode 改为 pull，并使用独立 SHMEM endpoint/output。
```

每步在前后同步 NPU，按各 rank 最大 step time 汇总 mean/median 和 global
tokens/s。不要删除慢样本；出现外来或无法归属 NPU 进程时，整组 ABBA 失效并重跑。
MoE 加速比用相同输入和权重的模块边界诊断或
`deepseek_v41_benchmark` 计算，不能把单层 MoE 倍率直接外推为整网倍率。

### HBM 口径

结果 JSON 中的 `peak_allocated_bytes` 和 `peak_reserved_bytes` 是
`torch.npu.max_memory_*` 的 allocator 峰值；它们不包含所有 SHMEM/外部 allocator，
不能直接称为整卡 HBM。测试期间另用 `npu-smi info` 或 `npu-smi info -t proc-mem`
按秒采样每张卡的 HBM used/total 和进程归属，报告两类数值：Torch allocator
峰值，以及卡侧物理 HBM 峰值。两者必须注明采样边界和是否含 SHMEM heap。

### 当前性能边界

历史归档中有 `limit=0` 的 EP8/E48 四层整网和单层 MoE 结果；这些数字用于解释
实现收益，不能当作当前 `limit=10` 的性能验收。当前分支本次尝试的 EP8/E48
push/pull 整网复测因设备持续被其他训练进程占用，在 benchmark 启动前取消，
没有新增性能或 HBM 结论。重新验收时必须按上述流程在 limit=10、同一 native
payload 下分别完成 owner、push、pull。

## 验收命令

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/auto_models/models/deepseek_v41 \
  tests/ut/core/multicore \
  tests/ut/auto_models/trainer/test_mixed_precision_optimizer.py

git diff --check
```

NPU block、Trainer smoke、checkpoint round-trip、push/pull 和性能结果分别归档；
一个通过的 block oracle 不代表完整 40 层训练收敛，也不代表 TP/CP/PP/VLM 已验收。
