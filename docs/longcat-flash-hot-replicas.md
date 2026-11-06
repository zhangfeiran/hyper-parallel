# LongCat Flash 热专家副本接入与验收

本地优化分支已支持 LongCat 原生 EOI 权重布局的热专家副本，包括权重发送、物理槽位路由和 FP32 专家梯度回传。
原 `feat/longcat-flash-repro` 分支的历史参考结果仍然使用 B0；此处是独立实现与验收，不能把旧结果写成已启用副本。
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
这是专家组件及独立优化器更新验收，不代表完整 LongCat Trainer、多节点 EP64 或完整层数验收。

固定热点组件中，8 个热真实专家都属于 rank0；每个专家在 4 个 source 上各有 2048 行。
B0 接收峰值为 65536 行。实际迁移 6 个完整专家后，4 个 rank 均为 16384 行。
4 个 rank 的最终 SHMEM 状态均为 `Uninitialized/reference_count=0`，运行退出码为 0，所属设备进程已退出。

CPU 回归为 643 项测试、3377 个子用例通过，覆盖 EOI 梯度带容量、无损行计数、guest 预算、硬容量、
单热专家回退、随机多 owner 规划以及完整专家/分片模式的 checkpoint 策略隔离。
新增文件的 pylint 和复杂度检查通过；合并进来的 expert 分支仍有独立的全量风格门禁遗留项，不能据此宣称全分支门禁通过。

## 性能对照

受控热点的 B0/B2 fresh-process ABBA 和反向确认已完成，前后向分别为 66.830 ms 和 41.183 ms，
配对几何平均加速 1.623 倍。普通路由对照的首轮存在 shell 库路径切换污染风险，已排除，正在独立环境重新确认。
最终数据单独归档后更新本节；不使用该首轮结果作为普通路由优化验收。

## 客户运行连续验证

先按仓库的 native 构建说明重新编译本分支，并确认 editable 安装指向当前目录：

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
source "$CANN_ENV"
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
