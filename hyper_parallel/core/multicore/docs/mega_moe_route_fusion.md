# MegaMoe permute / unpermute 融合

`MegaMoeExperts` 可以把路由端点放入 MegaKernel 的 Dispatch 或终止 epilogue，
前向和反向分别选择。常规输入仍是原始 `[T, H]` hidden states、`[T, K]` Top-K IDs 和 Router weights。

## HBM 存储约束

前向启用 `permute` 或 `both` 时，路由准备只对 `[T * K, 1]` 标量 route-slot IDs 排序，
生成专家顺序索引和逆映射。`PreparedTopKRoute.routed_tokens` 为 `None`，
不创建发送侧 `[T * K, H]` permute 临时张量。
Dispatch 从 `original_x[route_slot // K]` 读入 UB，随后直接写入目标 rank 的专家接收区。

反向启用 `permute` 或 `both` 时，Dispatch 直接读取 `[T, H]` 的 `dY`，
在 UB 内乘 Router weight 并发送，不创建发送侧 `[T * K, H]` 加权 `dY` 临时张量。
Router weight 梯度在所有本地 Dispatch 读完权重后计算，使用独立输出，保留前向概率。

专家接收区、FFN 激活、Combine 返回的专家输出及其训练所需快照仍位于 HBM。
这些存储服务于专家计算和反向，不能计为已消除的发送侧 permute 临时张量。
例如 `T=4096, H=5120, K=8, BF16` 的一个完整 permute 张量为 320 MiB；
实际峰值下降还取决于其他张量的重叠存活时间，需单独测量。

## 选择融合端点

```python
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=96,
    top_k=8,
    ep_size=16,
    ep_group=ep_group,
    dispatch_mode="push",
    forward_fusion_mode="permute",
    backward_fusion_mode="legacy",
)
```

| mode | 前向 | 反向 |
| --- | --- | --- |
| `legacy` | 独立 permute / weighted unpermute | 独立加权 dY / dX reduction |
| `permute` | 原始 token gather 与 Dispatch 融合 | dY 加权、gather 与 Dispatch 融合，同时计算 Router weight 梯度 |
| `unpermute` | weighted unpermute 作为图内 epilogue | dX reduction 作为图内 epilogue |
| `both` | 融合前向两端 | 融合反向两端 |

构造参数为 `None` 时，按以下优先级解析：显式构造参数、对应方向的环境变量、
公共环境变量、默认值。支持的静态 push 形状默认使用前向 `permute`、反向 `legacy`。
反向融合模式使用独立的对称 Combine 返回区，经过调度校验后也可以复用接收区存放 gate dX。
融合会改变归约、快照和调度的开销，完整训练峰值及性能仍需按实际形状单独测量。

```bash
# 完整 legacy 对照；每次修改模式后重新构造模块或启动新进程。
export HYPER_PARALLEL_MEGA_MOE_FUSION_MODE=legacy

# 独立选择前后向；方向设置覆盖公共设置。
export HYPER_PARALLEL_MEGA_MOE_FORWARD_FUSION_MODE=permute
export HYPER_PARALLEL_MEGA_MOE_BACKWARD_FUSION_MODE=both
```

当前融合范围为静态 push、`K <= 16`、`T * K <= 2**24`。
后一个边界保证标量 FP32 route-slot IDs 精确表示整数。
Pull、动态 token 和超出范围的形状默认使用 legacy；显式要求这些形状执行融合会报错。
融合模式参与资源共享兼容性和 EP rank 配置一致性校验。

DSV4.1 的 `DeepseekV41MegaMoeExperts` 适配器默认接受动态 token，因此默认进入 legacy 路径。
只有确认每次输入行数固定时，才在模块替换配置中显式设置静态模式：

```yaml
replace_module:
  _target_: hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts
  local_num_tokens: 4096
  dynamic_tokens: false
  forward_fusion_mode: permute
  backward_fusion_mode: legacy
```

静态模式会检查实际输入行数；不同长度输入不能共用这一配置。

## 生命周期与诊断

共享 workspace 在整个原生调用和结果恢复期间持有 lease。
前向保存独立的专家输入；反向需要保留计算图时复制可写输出，最终反向才能复用已消费的专家输出快照。
前向 unpermute 复用 down projection 保存专家输出前，同时等待远端接收完成和本地 Combine 读完。
返回的 token 输入梯度始终拥有独立存储，后续层复用 workspace 不会改写它。
外置反向路由梯度消费专家输出快照后释放局部引用；在分配返回的 token 梯度前释放已消费的反向临时缓冲区。
当本地接收行数少于 `T * K` 时，反向 Combine 仍保留完整的发送侧行范围。

当前的 [MegaKernel Profiling](mega_kernel_profiling.md) 接口继续生效。
路由 reduction 涉及多个专家，因此不指定单一 expert owner。
L1 dump 对非连续 gather 的源 token 视图明确拒绝，用户可以选择本地 GMM 输入。

Native payload 与 Torch adapter 必须一起重新构建、激活，并启动新 Python 进程。
反向 native ABI 使用 CANN 自有计算 workspace，并通过原 byte-workspace 参数传入对称 Combine 返回区，
保持 CANN 输入数量不超过 32；逻辑端口 32 单独描述该返回区，避免覆盖接收的 dY。
原有 Python 低层 wrapper 的位置参数保持兼容。

## 验证

CPU 单测验证 16 种前后向模式组合的调度依赖、标量路由映射、轻负载接收 rank 的 Combine 范围、
Router 权重保存和输入梯度存储所有权：

```bash
python -m pytest -q tests/ut/core/multicore/modules/mega_moe/test_fusion.py
```

NPU 验收应比较相同权重、路由、梯度、EP 拓扑和新进程运行，检查每个 rank 的输出、
`dX`、Router weight 梯度、两组专家权重梯度以及保留计算图后的重复反向。
性能使用关闭 profiler 的 rank-max step 计时，并单独报告 Torch allocator 峰值与外部 SHMEM 存储。
