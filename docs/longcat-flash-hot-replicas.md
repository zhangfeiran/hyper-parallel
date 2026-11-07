# LongCat Flash 热专家副本接入与验收

当前客户交付分支 `feat/longcat-flash-repro` 已支持 LongCat 原生 EOI 权重布局的热专家副本，包括权重发送、物理槽位路由和 FP32 专家梯度回传。
原交付提交 `a124e1304` 的历史参考结果仍然使用 B0；此处是独立实现与验收，不能把旧结果写成已启用副本。
切换到本优化分支需要重新编译 native，不能直接沿用原客户分支的二进制。客户自建 Torch/PTA 保持原版本。

## 配置

沿用[接入指南](../examples/longcat_flash_repro/INTEGRATION.zh-CN.md)中的权重分片和生命周期设置，
在创建 `LongcatFlashMegaMoeExperts` 时增加：

```python
replacement = LongcatFlashMegaMoeExperts(
    module=source_hf_experts,
    context={"ep_size": ep_size, "ep": ep_size > 1},
    backend="megamoe",
    dispatch_mode="push",  # pull 也已通过下述连续更新验证
    expert_weight_layout="EOI",
    swiglu_precision="fused",
    routing_backend="triton",
    initial_token_capacity=24576,  # 本文 T2048、K12；其他负载按实际情况设置
    replica_slots_per_rank=2,
    replica_granularity="expert",
    replica_transport="shmem_signal_kernel_gradient_bands",
    replica_min_rows=1024,
    replica_minimum_load_reduction=0.4,
)
```

每个 EP rank 仍然拥有 8 个真实专家；B2 额外提供最多 2 个执行客槽，不能把全局专家数量从 E32 改成 E40。
HF 原始参数名、EOI 存储、unused Gate/Up 尾部和优化器所属 rank 均保持原接入契约。
identity 专家仍执行加权恒等映射，不创建权重副本，不进入真实专家 GEMM。
`direct_weight_gradients=True` 的 B0 专用快速路径在 B>0 时自动使用普通参数前缀梯度路径。

不满足最低收益或无实际 guest 的路由继续在 owner 上执行。
默认 40% 是接收行峰值的规划门限；它不等同于端到端加速保证，因为权重发送和梯度回传也有成本。
每次 forward 的权重可见性及每次 backward 的梯度归还仍由副本 lease/ready/ack 协议管理。

## 完整专家与 token 分片的区别

LongCat 默认 `replica_granularity="expert"`：同一逻辑专家的所有 source 行交给同一个执行位置，
保留原生 GMM 的专家内归约顺序。这允许把一个 owner 的多个热专家移到不同 rank。
通用 `MegaMoeExperts` 默认仍然是 `replica_granularity="token"`。

原 token 分片路径在 EIO 和 EOI 两种布局上得到相同的一个 near-zero dW 元素严格失败，
相对 L2 约 `1.45e-5`；不能因为整体误差小而把这个结果标成严格通过。
因此本次 LongCat 验收采用完整专家模式，未放宽 `2e-5 + 0.05*abs(reference)` 的逐元素门限。

完整专家模式有明确边界：它不能拆分单个超大热专家。
若一个专家本身超过硬接收容量，且 owner 也无法容纳，会明确报错；不会丢行或暗中切回 token 分片。
当前完整专家模式使用 CPU counts 规划器，并关闭不适用于该粒度的旧 token 分片成本标定。
接收容量、最低峰值改善和 checkpoint 重算策略隔离检查仍然生效。

## 已完成的 NPU 验证

环境为 CANN8.5.0、aarch64、910C/Ascend910_9382、Python3.11、Torch2.9/PTA2.9。
客户 x86_64 二进制需要在其编译环境现场构建；本文不是客户 x86 NPU 的实测结果。

实验保持 H6144、专家 I2048、TopK12、T2048/rank；EP4、E32、Z16，即每卡 8 个真实专家、4 个 identity 份额。
裁剪只减少 EP 对应的全局专家数及模型深度，不缩减维度或每卡专家数。
原生对照使用 NPU GMM、`npu_swiglu`、permute/unpermute 和 AllToAll；两侧使用相同有序 Triton 路由。

| 实现 | 连续更新 | 实际副本步骤 | 输出、dX、dP、Gate/Up dW、Down dW | 更新后参数 |
| --- | --- | --- | --- | --- |
| EOI Push B2 | 10 步 × 4 rank | 每 rank 7 步 | 每步全值 max_abs=0 | 每步完整哈希一致 |
| EOI Pull B2 | 10 步 × 4 rank | 每 rank 7 步 | 每步全值 max_abs=0 | 每步完整哈希一致 |
| EIO Push B2 | 10 步 × 4 rank | 每 rank 7 步 | 每步全值 max_abs=0 | 每步完整哈希一致 |

连续验证保持同一个副本执行器存活，使用两套独立参数和 FP32 SGD master。
路由覆盖热 owner 切换、普通路由、一个 source 全 identity、全局 identity；全局 identity 时参数梯度保持 None。
全局 identity 不更新专家参数，其他步骤的 unused Gate/Up 尾部梯度为零。
无副本 B0 的 Push/Pull 也在最终 native 库上额外验证，4 个 rank 的全部比较值误差均为 0。
这是专家组件及独立优化器更新验收，不代表完整 LongCat Trainer、多节点 EP64 或完整层数验收。

固定热点组件中，8 个热真实专家都属于 rank0；每个专家在 4 个 source 上各有 2048 行。
B0 接收峰值为 65536 行。实际迁移 6 个完整专家后，4 个 rank 均为 16384 行。
4 个 rank 的最终 SHMEM 状态均为 `Uninitialized/reference_count=0`，运行退出码为 0，所属设备进程已退出。

CPU 核心回归为 658 项测试、3401 个子用例通过，另有 8 项交付入口测试及 8 个子用例通过，覆盖 EOI 梯度带容量、无损行计数、guest 预算、硬容量、
单热专家回退、随机多 owner 规划以及完整专家/分片模式的 checkpoint 策略隔离。
新增文件的 pylint 和复杂度检查通过；合并进来的 expert 分支仍有独立的全量风格门禁遗留项，不能据此宣称全分支门禁通过。

## 性能对照

使用相同完整权重、输入、路由与上游梯度，每次 fresh process，10 次预热、30 次正式样本。
每步取全部 rank 的最大耗时；每进程取全部样本中位数，再对固定 ABBA 和 BAAB 的进程配对比值取几何平均。
保留慢样本，没有在看到结果后更换统计口径。普通路由的 3 次控制预跑冻结了 2% 最低耗时改善门限。

| 场景 | 控制路径前后向 | 候选路径前后向 | 配对加速 | 耗时下降 |
| --- | --- | --- | --- | --- |
| 普通路由、B0：冻结的 expert 控制 vs 本次最终优化 | 32.225 ms | 29.583 ms | 1.089 倍 | 8.20% |
| 受控热点：本次 B0 vs 本次 B2 完整专家副本 | 67.085 ms | 41.315 ms | 1.624 倍 | 38.41% |

普通路由保持 B0，不把副本带来的负载分散混入列宽优化收益。
普通路由的前向为 12.737 vs 12.667 ms，0.55% 的差异未达到预先冻结的 2% 门限；本轮确证收益在前后向路径。
同样使用新路由前端的原生大算子路径单次 fresh-process 对照为普通路由 40.902 ms、受控热点 104.590 ms，
这些单次对照来自更新默认 lane 之前的同宽度、同路由前端版本，不具有上表最终版本重复配对比较相同的验收强度。
热点用固定 bias 集中选择 8 个真实专家和 4 个 identity，未把该分布表述为自然训练路由。
普通路由优化的门限来自相同普通路由预跑；热点列是独立的重复配对观测，没有把该门限挪作热点保证。
以上是专家组件结果，不能改写为完整模型或客户 x86 的速度。

Profiler 中，专家输出梯度 kernel 从约 3.05–4.18 ms 降到最终版本的 1.03–1.25 ms，符合完整路径的下降方向。
本轮也移植了 expert 分支截至 `db372c1` 的 4 个新增提交，包括默认 active gradient lanes，实际解析到 `device_active_lanes`。
最终 profile 中 W1 仍在 CombineGrad 后约 2.60–2.63 ms 完成；当前数据不支持宣称额外通算/CV 掩盖收益。
上表是最终组合实现的配对结果，没有把各单项优化相加。

逐进程样本、固定门限、全部 rank 终态审计及原始报告校验值见
[验收数据](longcat-flash-hot-replicas-results.json)。

所有最终性能进程均隔离库环境，并校验 `/proc/self/maps` 中实际加载的 native 库来自对应工作树。
首轮普通路由对照存在库路径切换污染风险，已排除；最终表格仅采用重新运行并通过来源检查的结果。

## 客户运行连续验证

在对应分支的仓库根目录、客户已验证的 Python3.11/CANN8.5.0 环境下重新编译，
并确认 editable 安装指向当前目录。构建会使用客户的 CPU 架构，不随分支分发本机的 native 二进制：

```bash
source "$CANN_ENV"
bash build.sh --multicore on --soc-list ascend910_93 --strict on
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

在客户自行预留的 4 个逻辑 NPU 上，显式设置 `ASCEND_RT_VISIBLE_DEVICES`，输出目录使用外部新目录。
需要可用的 Triton-Ascend3.2.0；命令不安装依赖，不替换客户 Torch/PTA，不依赖资源等待器。

```bash
export HCCL_DETERMINISTIC=true
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/torch/models/longcat_flash/_validate_longcat_replicas.py \
  --backend push --dispatch-mode push --layout EOI --budget 2 --updates 10 \
  --ep 4 --tokens 2048 --real 32 --zero 16 \
  --hidden 6144 --intermediate 2048 --top-k 12 \
  --routing-backend triton --swiglu-precision fused \
  --initial-token-capacity 24576 --output "$REPLICA_OUTPUT"
```

另起进程，把 `--dispatch-mode` 改成 `pull` 并使用新的输出目录即可验证 Pull。
每个 rank 写出终态 JSON 和逐步报告；应检查全部 rank，而非只检查 rank0。
自动 ST 入口为 `tests/torch/models/longcat_flash/test_longcat_flash.py::test_longcat_hot_replicas`，
其中同时运行 EOI Push/Pull 的连续更新验证。

旧 `examples/longcat_flash_repro/run.sh` 冻结了历史源码哈希，针对旧参考结果。
它在本优化分支上会主动拒绝源码不一致；不能删除该检查后把新结果当成旧实验复现。
本分支的热副本验收使用以上独立入口。

性能入口为 `tests/torch/models/longcat_flash/_benchmark_longcat_replicas.py`，沿用上述维度和设备参数，
把脚本名换为该入口，删除 `--dispatch-mode/--layout/--updates`，增加 `--fixture hot --warmup 10 --samples 30`。
分别用 `--backend push --budget 0`、`--backend push --budget 2`、`--backend native --budget 0`，每项单独新建进程与输出目录。
普通学习路由改为 `--fixture normal`。比较时两边使用同一版本的路由前端与 fused SwiGLU 语义。

## 回退与检查

需要回退调度时，在新进程中设置 `HYPER_PARALLEL_MOE_GRADIENT_QUEUE_ORDER=original`。
关闭热副本使用 `replica_slots_per_rank=0`。客槽为运行期存储，不改变逻辑专家数或 checkpoint 参数键。
若切回历史复现分支，请同时使用与该历史源码匹配的 native 产物，按其原入口进行源码、ABI 和 NPU 精度检查。
