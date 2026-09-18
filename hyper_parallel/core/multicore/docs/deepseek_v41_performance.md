# DeepSeek-V4.1 单机裁剪性能验证

## 可复现入口

`examples.training_demo.benchmark_deepseek_v41_megamoe` 通过真实 Trainer 构建四层
DSV4.1 text crop，使用生产融合 attention、Sinkhorn、mHC-post，以及 Engram、Full/Reindex indexer、
Muon/AdamW、BF16 模型参数和 FP32 主参数。mHC pre 保留配方中的分阶段 Torch 实现；未启用 activation checkpoint 或 compile。
两种后端都设置 routed/shared `swiglu_limit=0`。输入是确定性合成 token ID，
不是预训练权重或真实语料，不用于宣称模型质量或收敛。

基线 `owner_ep` 是本分支原有 DSV4.1 owner A2A EP，expert 计算为逐 expert Torch
matmul（`use_grouped_gemm=false`）；`megamoe` 使用融合多核 dispatch/compute/combine。
两个后端保持相同 dense FSDP、expert EP/EDP、非 MoE 模块、优化器和输入。
默认串行四层共享一个 MegaMoe workspace，未启用容量丢 token。

正式比较应使用新进程 A/B/B/A；先单独生成 canonical rank-local BF16 权重，再为
全部计时进程读取同一目录。加载 MegaMoe 时仅转置 expert 矩阵末两维，同时刷新
FP32 主参数。读取/写入初始化权重的时间不计入 step。

```bash
# 先激活对应 CANN、Omni attention/mHC vendor、当前 checkout 的 multicore payload。
export HYPER_PARALLEL_PLATFORM=torch OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HYPER_PARALLEL_SHMEM_HEAP_SIZE
# 每轮指定未占用的 SHMEM bootstrap 端口。
export HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://127.0.0.1:28662

# 每次 torchrun 用独立 SHMEM bootstrap endpoint；本例需要预先准备的本地资产。
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m examples.training_demo.benchmark_deepseek_v41_megamoe \
  --model-dir /path/to/DeepSeek-V4.1-Flash --engram-assets /path/to/engram.json \
  --backend owner_ep --experts 128 --tokens 4096 --warmup 0 --steps 1 \
  --weights /path/to/canonical-weights --write-weights --output /path/to/initialize

# 正式 A/B/B/A 分别设 backend=owner_ep,megamoe,megamoe,owner_ep，输出到不同目录。
# 全部去掉 --write-weights，使用相同的 --weights、--warmup 8、--steps 10。
```

每步计时覆盖完整前反向、梯度裁剪、优化器更新和标准 Trainer callback，前后同步 NPU。
rank 间 barrier 在计时外，以各 rank 最大耗时统计均值与中位数，输出真实 global tokens/s。
JSON 保存逐步 loss/grad_norm、峰值 allocated/reserved，以及后端、规模和权重目录。
Trainer callback 的合成数据 token 计数可能为零；性能使用 JSON 中明确的
`world_size * tokens / median_seconds`；汇总吞吐另使用全部计时步均值，
不采用该 callback 的 tokens/s。Torch allocated/reserved 不含所有 SHMEM/外部 allocator，
不能等同于整卡 HBM。
初始化期间不采样吞吐；性能比较须保留整段设备占用记录，出现外来/无法归属进程时
整组 ABBA 作废，正确性证据单独保留。

## 原尺寸与裁剪边界

[官方 Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/config.json)
保持 H=5120、expert I=2304、TopK=6、64 个 attention heads、
head_dim=512、RoPE dim=64、mHC mult=4、vocab=129280。
裁剪深度 40→4，Engram bucket 16M→4096（规范化词表映射来自官方 tokenizer）。
192 experts / EP8 对应原配方 384 experts / EP16 的每卡 24 experts。
4 层中的 Full/Reindex 重映射和 indexer loss coefficient 沿用已有验证 crop，
不能等同于完整 40 层模型。

## 完整 384 专家的 MoE 前反向

单层入口 `hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_benchmark`
保留 H=5120、I=2304、E=384、TopK=6、shared expert=1、路由缩放 1.5。
EP8 下每卡持有 48 个专家；在 CPU 只初始化本 rank 的 canonical HF-layout 权重，
再为 MegaMoe 转置，避免每卡临时加载全局 expert 权重。

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_benchmark \
  --backend owner_ep --experts 384 --tokens 4096 --warmup 5 --steps 10 \
  --output /path/to/moe-owner-A1
```

以 `owner_ep, megamoe, megamoe, owner_ep` 顺序启动独立进程。每 rank 的
`initial_weight_sha256` 和 `route_sha256` 必须在四次运行间一致；同时记录全 EP
域各 owner 的接收量。这是正常 learned routing，未人为平衡路由。
测量 router、routed/shared experts 的完整前反向；不包含优化器或 router/shared
参数的 dense DP 归约，不能把该加速比表述为整模型训练加速。
固定输入与权重用于稳定测量。原始 H/I/TopK 的独立 FP32 验收和大 E/T 下的
BF16 同状态补充检查分开记录。可用 `--evidence-dir /path/to/bf16-evidence` 在计时后
写出 owner EP 的 output/dX/全部参数梯度，后续进程读取并比较；该选项不改变
计时范围，Torch peak 在额外检查前保存。

## 原始宽度精度与融合算子验证

- 原始 H=5120 / I=2304 / TopK=6，EP2、E=8、T=128 的独立 CPU FP32
  oracle 检查通过；覆盖输出、dX、路由权重梯度、全部 MoE 参数梯度和同状态 SGD 更新。
- owner EP BF16 最坏相对 L2 0.5422%，MegaMoe 0.5132%；两者最坏峰值归一化
  误差均为 0.9303%（shared expert down projection 梯度）。门槛保持 1% / 2%。
- 四层 128 experts / EP8 / T4096 的生产 attention/Sinkhorn/mHC-post 融合路径前反向及
  Muon/AdamW 更新通过。两后端首步各 rank 的 BF16 loss 一致，global grad norm
  分别约 2.91591 / 2.91550；这不替代全模型每项梯度的 FP32 验收。
- 小模型此前的融合算子失败源于不支持的形状：本机 sparse attention 要求
  head_dim=512、RoPE dim=64，mHC post grad 支持 mult=4/6/8；原始形状可运行。
- 本轮 CPU 回归为 50 项、49 subtests；静态检查通过。
- 默认共享 workspace 后，四卡 EP2/EDP2、pull、BF16 + FP32 主参数的 Trainer
  回归通过：三步训练、每步两次梯度累积、模型权重 checkpoint 保存/恢复和有序资源关闭。
  该小模型 ST 使用 portable attention/mHC，原始形状融合路径由上述 4K Trainer 单独覆盖。

原 192 experts / EP8 / T1024 配置接近显存上限。普通 allocator 在第二步失败，
expandable segments 下 owner EP 完成三步，但 MegaMoe 第三步仍出现驱动 OOM。
因此完整训练正式比较采用 128 experts，未将未完成配置报告为稳定性能。
所有容量配置维持默认 lossless，未丢弃 token、未缩小 H/I/TopK。

## 首轮完整训练 ABBA（3 warmup + 5 measured）

8×910B3 64GB，4 层 / EP8 / E128 / T4096，固定 canonical 初始权重；
四个进程组均正常退出，监测无外来或无法归属的进程。以下保留全部计时样本。

| 运行 | 后端 | 均值 ms/step | 中位数 ms/step | Torch peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| A1 | owner EP | 6891.76 | 6891.60 | 55.93 |
| B1 | MegaMoe | 6867.82 | 6423.96 | 54.56 |
| B2 | MegaMoe | 6682.19 | 6428.00 | 54.56 |
| A2 | owner EP | 6894.79 | 6896.02 | 55.93 |

全部 10 个 measured steps 的均值为 6893.28 → 6775.01 ms，
吞吐增加 1.75%；两轮中位数平均为 6893.81 →
6425.98 ms，对应吞吐增加 7.28%。
MegaMoe 两轮的第 4–5 步仍包含较慢样本并伴随 reserved memory 增长，不能删掉
这些样本后把中位数收益当作这组全部步骤的平均收益。首轮只表明稳定末段更快，
整个短窗口的平均收益较小；后续长预热复测如下。

## 完整训练复测（8 warmup + 10 measured）

保持上述四层 EP8/E128/T4096 配置和 canonical 权重，重新启动完整 ABBA。
四轮正常退出，占用记录均无外来/无法归属进程；没有剔除任何计时样本。

| 运行 | 后端 | 均值 ms/step | 中位数 ms/step | Torch peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| A1 | owner EP | 6888.36 | 6894.61 | 55.93 |
| B1 | MegaMoe | 6525.89 | 6420.42 | 54.56 |
| B2 | MegaMoe | 6535.53 | 6424.09 | 54.56 |
| A2 | owner EP | 6887.31 | 6890.70 | 55.93 |

每个后端全部 20 个 measured steps 平均 **6887.83 → 6530.71 ms**，
完整训练吞吐增加 **5.47%**，step 延迟下降 **5.18%**。
两轮中位数平均 6892.66 → 6422.25 ms，对应吞吐增加 7.32%，仅作补充统计。
Torch peak allocated 减少约 1.37 GiB；该指标不包含所有外部 allocator。

增加预热没有消除全部慢样本：B1/B2 的第 12 步分别为 7526.86 / 7489.50 ms。
两轮均在 rank 2 出现同一现象：reserved 从约 56.46 GiB 降至 41.73 GiB，
对应 plog 记录 20 MiB 物理分配失败，以及 `NPUCachingAllocator` 释放缓存后重新分配。
该重试成功，全部后续步骤完成。记录支持显存压力和缓存回收与慢样本相关，
尚未通过 profiler 拆分各部分开销；不能声称只是预热，也不能删除这些样本。
本轮没有修改通用 allocator 或 optimizer。后续可针对该分配重试单独定位和优化。

另按 EP8、全局 E48（每卡 6 experts）复测四层整网，完整 step 吞吐增加 3.57%；
整网内四层 MoE 前反向为 509.43 → 315.86 ms，即 1.61x。配置、计时口径和
原始证据见 [EP8/E48 整网内 MoE 计时](deepseek_v41_ep8_e48_attribution.md)。

## 大尺寸 MoE 单层 ABBA

初次 EP8/E384 及后续 EP4/E192 试跑有外来或无法归属的 NPU 进程，完整计时组
均归档为无效性能证据。EP4/E192 的数值对照保留：两次 MegaMoe 各 36 项
output/dX/参数梯度检查通过，最坏相对 L2 0.4910%、峰值归一化误差 1.1364%；
owner EP 重复运行与参考逐元素一致。这是补充 BF16 对照，未冒充大 E/T 的 FP32 oracle。

随后固定物理卡 2/3/4/5，重新完整采集 `block-ep4-clean` ABBA，四轮占用记录
均无外来/无法归属进程。E384/EP8 → E192/EP4，保持每卡 48 experts 和 T4096；
H/I/TopK、路由方式、缩放、shared expert 不变。所有 rank 的初始权重与路由
SHA256 在四轮完全一致，owner 接收量为 `[24812, 24584, 24304, 24604]`。
五步预热、十步计时；数值检查与初始化均不计入性能。

| 运行 | 后端 | 均值 ms | 中位数 ms | Torch peak allocated GiB |
| --- | --- | ---: | ---: | ---: |
| A1 | owner EP | 708.050 | 707.690 | 9.982 |
| B1 | MegaMoe push | 46.450 | 46.424 | 7.855 |
| B2 | MegaMoe push | 46.458 | 46.528 | 7.855 |
| A2 | owner EP | 708.092 | 708.240 | 9.982 |

全部 measured steps 平均 708.071 → 46.454 ms，
前反向加速 **15.24×**。对照是当前 DSV4.1 逐 expert matmul 路径，
不是优化后的 grouped-GEMM MoE；也未包含 optimizer 或 dense DP 梯度归约。
这组的 expert 数、EP degree 与四层训练组不同，不能将该倍率外推为端到端收益。

## 环境与证据索引

- 设备：Ascend 910B3，64GB/卡；CANN 9.1.0。
- Torch / torch-npu 2.9.1、Transformers 5.13.0，当前 checkout 的 editable 安装。
- 模型资产固定 revision `dba1be0a40aa45a94ad051997016db3960a90277`，config SHA256
  `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`。
- 全部训练使用 BF16 模型参数、FP32 主参数和原配方的 Muon/AdamW。
  `TASK_QUEUE_ENABLE=0`、`OMP_NUM_THREADS=1`、`expandable_segments:True`；仅测试 push 性能。
- Torch adapter SHA256 `35d6923f5cbe280eb60e00d44a37248e7042bda7b439d09874c9f3be63ad416a`；
  native opapi SHA256 `a49b453c8c7e25323acdc71294aed33ae27545c4ff85ee8ec33af89dfbc9be3e`。
  本轮没有改动 C++/device kernel，payload 与前轮一致。

本地证据根目录：`/home/feiran/doc/dsv41-megamoe-scale-20260917.FpKkkO`。
`source-environment.json` 保存被测 Python 源码及 native `.so/.o` 的 SHA256；
`model-assets.json` 保存官方资产 revision/哈希；`fused-registration.json` 核对
Omni mHC-post 和 Sinkhorn 实际注册，mHC pre 的分阶段实现不作全融合声明。

- `fullwidth-oracle.json`：原始 H/I/TopK 的独立 FP32 数值验收。
- `trainer-summary.json`：短预热完整训练 ABBA，包括全部慢样本。
- `trainer-steady-summary.json`：8 步预热、10 步计时的完整 ABBA，主要端到端性能结论。
- `trainer-steady-B*-plog-*`：两次第 12 步缓存释放/分配重试的驱动及 allocator 记录。
- `shared-workspace-training.log`、`shared-workspace-training-results/`：共享 workspace 训练回归。
- `block-ep4-{B1,B2,A2}/rank*.json`：大 E/T 的 BF16 输出、dX、参数梯度补充对照。
- `block-ep4-clean-summary.json`：干净的单层性能 ABBA。
- 各运行的 `*-ownership.json` 与逐 rank `result.json`：进程归属、计时和显存原始记录。
- `block-summary.json`、`block-ep4-summary.json` 的 `valid=false`：受干扰整组，未用于性能结论。

所有结论限于上述随机初始化 text crop、临时 limit=0 和当前 owner EP 基线。
未宣称完整 40 层模型、limit=10、真实语料收敛或最优 grouped-GEMM 基线的收益。
