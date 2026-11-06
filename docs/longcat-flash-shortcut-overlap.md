# LongCat Flash 完整逻辑层的 shortcut 重叠实验

本实验将 shortcut MoE 放到副流，与 dense/attention 主分支重叠，得到原宽度完整逻辑层前后向 **5.23%** 的正式耗时下降。
它是可复跑的逻辑层诊断入口，尚未替换默认 decoder，也未验收 FSDP、激活重算或完整 Trainer。

## 计算路径和调度

HF LongCat 的一个逻辑层包含两组 MLA attention、两组 dense MLP 和一个 shortcut MoE。
attention0 残差相加并完成归一化后，MoE 和 dense0 读取同一份归一化输入。
MoE 的结果到 dense0、attention1、dense1 都完成之后才加入主分支。

```mermaid
flowchart LR
    A[attention0 与残差] --> N[post-attention norm0]
    N --> R[副流：router]
    R --> M[副流：route 与 MegaMoE]
    N --> D0[主流：dense0 与残差]
    D0 --> A1[主流：attention1 与残差]
    A1 --> D1[主流：dense1 与残差]
    M --> J[等待 shortcut 后相加]
    D1 --> J
```

调度先在副流提交 router，再在主流提交 dense0，随后继续提交副流专家计算。
这样 dense0 的设备工作已经排队，可以覆盖路由中的 CPU 元数据等待和部分通信。
只在最终 shortcut 相加之前合流，保留输入生产事件和跨流分配的生命周期记录。

直接改变执行流会改变 BF16 输入梯度的累加顺序，原始多流实验因此严格失败。
当前诊断在分叉处显式保留原先的 `(dense_gradient + expert_gradient) + router_gradient` 顺序。
该顺序对本例不是可忽略的细节：未经修正的分叉子图输入梯度最多相差 0.25，尽管前向和各参数梯度一致。
完整层的串行重复实验还受到非确定性计算影响，故本表两侧均显式启用 `torch.use_deterministic_algorithms(True)`。
没有将非确定性串行控制视为合格参考，也没有放宽原严格门限。

## 正式对照

环境为本地 aarch64、910C/Ascend910_9382、CANN8.5.0、Python3.11、Torch2.9/PTA2.9。
本次完整层正式实验包含 [路由反向优化](longcat-flash-cv-tuning.md)和冻结至
`de127408c6ac3d053d12ca222b7e7669203515b3` 的 expert 分支增量。
两侧使用同一份运行源码和 native 库，仅改变串行/有序多流调度。

| 参数 | 实际配置 |
| --- | --- |
| 逻辑层深度 | 1；包含两组 MLA、两组 dense MLP、shortcut MoE |
| 隐藏宽度 / 专家中间宽度 | 6144 / 2048 |
| dense 中间宽度 | 12288 |
| EP / 全局真实专家 / identity 专家 | 4 / 32 / 16 |
| 每 rank 真实专家 / identity 份额 | 8 / 4 |
| 每 rank tokens / TopK | 2048 / 12 |
| 数据与权重 | BF16；专家使用 fused SwiGLU；固定学习路由 |
| 热副本 | B0；本表不混入负载分散收益 |

各进程保留完整权重、输入及上游梯度哈希。
先独立运行 3 个串行 pilot，冻结 2% 最低耗时下降门限，再运行
`A1 B1 B2 A2 B3 A3 A4 B4`，A 为串行，B 为有序多流。
每进程 10 次预热、30 个正式样本，每步取全部 4 rank 最大耗时，每进程取全部样本中位数，固定配对比值取几何平均。
没有剔除慢样本，也没有调整统计口径。

| 实现 | 4 个独立进程的前后向中位数 | 中位数的几何平均 |
| --- | --- | --- |
| 串行 | 88.235 / 88.437 / 88.151 / 88.422 ms | 88.311 ms |
| 有序多流 | 83.888 / 83.268 / 83.585 / 84.036 ms | 83.694 ms |

固定配对加速 **1.05517 倍**，耗时下降 **5.228%**。
全部 8 个正式进程、每个进程的全部 4 rank 都独立对照串行图，输出、输入梯度和所有参数梯度 `max_abs=0`。
源码字节、benchmark 字节、实际加载库路径、全部操作数和样本均经过审计。
全部 rank 关闭 SHMEM 后为 `Uninitialized/reference_count=0`；任务退出 0，所属 NPU 进程退出。
完整数据与校验值见 [逻辑层验收数据](longcat-flash-shortcut-overlap-results.json)。

## 时间线中的实际覆盖

较早的 prefix-only 版本的独立 profiler 对照确认了如下区间交叠，不能将该采集直接当成本表最新版本的计时：

- 有序多流中，AIV route kernel 与主分支 MatMul 的相交区间为每 rank 2.910–3.145 ms。
- HCCL 元数据通信与主分支 MatMul 的相交区间为每 rank 1.006–1.217 ms。
- rank0 的 shortcut Combine 在约 24.28 ms 完成，而最后一组主分支前向 MatMul 约 30.50 ms 完成。
  该单步中 shortcut 没有成为最终前向合流的尾部。

这些是已插桩算子时间区间，不是硬件占用率，彼此也不能相加当作净加速。
这里的 HCCL 交叠来自路由 agreement/counts 交换；专家数据走 MegaMoE 内部 SHMEM，不能据此宣称其数据通信被额外覆盖了同样长的时间。
同时，时间线中主分支与专家 GMM 仍存在 CUBE 竞争，不能假设同时提交的计算都能并行执行。
正式端到端逻辑层收益采用前一节未插桩的独立进程对照。

## 复跑

先按[热副本分支接入说明](longcat-flash-hot-replicas.md)准备与当前源码匹配的 native 构建及 editable 环境。
本入口不安装或替换客户 Torch/PTA，不包含本机二进制，也不依赖设备等待器。
客户 x86 NPU 仍需使用其已验证环境现场编译并运行验证。
在自行预留的 4 个逻辑 NPU 上设置 `ASCEND_RT_VISIBLE_DEVICES`：

```bash
source "$CANN_ENV"
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HCCL_DETERMINISTIC=true OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/torch/models/longcat_flash/_benchmark_longcat_shortcut_overlap.py \
  --mode serial --backend push --ep 4 --tokens 2048 \
  --warmup 10 --samples 30 --output "$LAYER_OUTPUT"
```

另起独立进程，使用新输出目录，将 `--mode serial` 改成 `--mode ordered`。
正式评估需要按上述正反交错顺序完成，而非只取两次单跑。
入口在每个进程计时前比较完整输出与全部梯度，任一 rank 不一致都会拒绝计时并报错。
隐藏宽度、dense/专家中间宽度和 TopK 固定；EP 改变时按每 rank 8/4 自动调整全局 E/Z。

该入口构造单个逻辑层，不加载客户 checkpoint，不执行优化器更新，不启用 FSDP 或激活重算。
接入实际训练还需要验证父模块的 FSDP hook、参数物化、跨流重算和梯度生命周期；本轮没有通过修改默认 decoder 规避这些验证。
