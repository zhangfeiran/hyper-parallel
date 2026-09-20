# MoE 多核并行使用指南

Multicore 是独立的 Torch-only 组件，提供芯片内多核 MPMD 并行能力，结合核级内存语义单边通信，增强 MoE 通算掩盖和 MAC 利用率。

用户从独立组件 `hyper_parallel.core.multicore` 显式导入 `MegaMoeExperts`。
SHMEM 由 Multicore 在内部管理，不提供独立用户接口。

- [构建与交付](docs/build.md)
- [架构设计与扩展](docs/architecture.md)
- [MegaKernel Profiling 使用指南](../../../docs/guide/mega_kernel_profiling.md)
- [MegaKernel Profiling 设计与实现](docs/mega_kernel_profiling_design_and_usage.md)
- [私有 SHMEM 生命周期](docs/shmem.md)
- [完整 MegaMoE 样例](examples/mega_moe/README.md)

## 核心概念

多核并行是 HyperMPMD 的核心能力之一，从集群级 MPMD（Pipeline 并行）扩展到芯片内多核 MPMD：

- **O0**：通过框架层 host CPU 侧的调度，支持 cube、vector、单边通信算子分核执行
- **O1**：调度下沉到 AICore，支持 cube、vector、单边通信算子分核执行，进一步提升性能

HyperParallel 基于多核并行实现 MoE 通算掩盖（Multicore MoE-FFN）：将 MoE-FFN 的五个算子
（AllToAll-Dispatch、GMM1、SwiGLU、GMM2、AllToAll-Combine）融合为一个 kernel，由 AIC（AI Cube）和
AIV（AI Vector）核同时执行，实现通信与计算的细粒度重叠。

## 接口概览

多核并行模块位于 `hyper_parallel/core/multicore/`，包含以下组件：

| 组件 | 说明 |
|------|------|
| `modules/` | 多核并行模块实现 |
| `ops/` | 多核并行算子 |
| `scheduler/` | 多核并行调度器 |
| `tasks/` | 任务编排 |
| `torch/` | Torch native binding |
| `shmem/` | 仅供本组件使用的私有单边通信 |

源码构建生成一个同时包含正反向 kernel 的
`hyper_parallel_multicore_nn` vendor，并与构建环境对应的框架 adapter 一起进入 wheel 或本地 native payload。
Torch adapter 是通过 `torch.ops.load_library()` 加载的普通共享库。

---

## Torch managed API

Torch 模型通过 `MegaMoeExperts` 执行 Router 选出的专家，当前支持 Ascend NPU 上的 Torch BF16 训练。
先按[构建与交付](docs/build.md)选择 `--multicore on`，激活 CANN 和 native payload，
并为所有 EP rank 配置相同的 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`（如 `tcp://<rank-zero-ip>:<port>`）。

在业务进程中选定当前 NPU、初始化 HCCL 进程组后，以下为两卡 EP 示例：

```python
import torch
import torch.distributed as dist
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    local_num_tokens=1024,
    hidden_size=512,
    intermediate_size=128,
    num_experts=4,
    top_k=2,
    ep_size=dist.get_world_size(),
    ep_group=dist.group.WORLD,
).to(device="npu", dtype=torch.bfloat16)

output = experts(hidden_states, topk_ids, topk_weights)
```

### 输入与权重

- EP 必须覆盖整个默认 world；每 rank 的 token 数 `T` 固定且为 128 的倍数，专家数 `E` 可被 EP 整除。
  执行计划采用静态 tiling，构造后的 `T/H/I/E/K/EP` 固定；其他 shape、芯片及后端需单独验证。
- `hidden_states` 扁平后的 token 数为 `T`，`topk_ids`、`topk_weights` 均为 `[T, K]`，
  ID 使用全局专家编号。
- 可选的 `tokens_per_expert` 是同 device 上 `[E]` 的精确本 rank histogram。
  省略时内部统计；传入时由调用方保证与当前 ID 一致。contiguous INT32 可直接使用，
  INT64 或非连续输入会转换。
- 本地参数布局为 `gate_up_weight: [E/EP, H, 2I]`、`down_weight: [E/EP, I, H]`。
  普通 gate/up/down 权重转换时，gate、up 分别转置后在末维拼接，down 转置最后两维。

Router 已有精确计数时，可直接传入：

```python
output = experts(
    hidden_states,
    topk_ids,
    topk_weights,
    tokens_per_expert=tokens_per_expert,
)
```

### 容量

`expert_capacity_factor=None` 是默认值，接收容量为 `EP * T * K` 向上对齐到 128，保证 lossless。
该容量用于各 rank 对称分配的 SHMEM 接收区。计算中间张量和待反向保存的 dispatch、
up-projection、activation 按本 rank 本次实际接收量分配；无接收时保留一行 ABI 占位。
源端 permute/combine 输出仍为 `T * K` 行。每次 forward 保存独立的接收数据和容量，支持后续路由变化。

分配前将已交换的各 rank 负载一次读取到 Host，同时用于本地定尺寸和全局溢出检查。
这也适用于默认 lossless 模式，会增加一次 Device-to-Host 等待，以减少计算和保存区的容量余量。
SHMEM heap 的预留仍由配置接收容量决定，不会随本次实际接收量缩小。

显式设置不小于 1 的有限 factor 时，容量改为 `ceil(T * K * factor)` 再对齐。
超过容量时，所有 EP rank 在进入 native kernel 前报 `capacity overflow`。
应根据显存和路由负载选择容量，确保显式容量覆盖实际接收量。

### 资源共享与关闭

相同配置的串行层可在首次 forward 前共享 workspace，各层参数、梯度和待反向激活仍独立：

```python
MegaMoeExperts.share_execution_resources(layer.mlp.experts for layer in model.layers)
```

共享资源只允许串行提交；跨 stream 时调用方须建立输入 tensor 的依赖，不支持并发线程调用。
checkpoint/recompute 和 `retain_graph=True` 的完整算子正确性暂未验证。

资源在首次执行时自动登记。普通 worker 入口可增加 `@managed_run`，在完整迭代边界增加
`lifecycle_checkpoint()`，无需逐层调用 `close()`，也无需用 `with` 包裹训练主体：

```python
from hyper_parallel.core.multicore import managed_run, lifecycle_checkpoint

@managed_run
def main():
    initialize_distributed()
    model = build_model()
    for batch in batches:
        train_step(model, batch)
        lifecycle_checkpoint()
```

装饰器在正常返回时先统一关闭资源，再销毁默认进程组。若现有入口已经调用
`dist.destroy_process_group()`，将该行替换为
`multicore.shutdown(destroy_process_group=True)`，且只在正常收尾或已协调的停止路径执行。
不能先销毁进程组再交给装饰器清理。已有框架可直接在模型卸载回调调用 `collect_resources()`，
在任务正常结束回调调用 `shutdown()`，无需接管原入口或信号处理器。

`collect_resources()` 仅释放所有 rank 上都没有模块成员、且没有未完成 backward 的组；
`shutdown()` 则关闭全部组，并使仍存活的模块不可再次执行。GC finalizer 只更新本地成员关系，
不执行 NPU 操作。图通过弱引用跟踪，成功的非保留 backward 结束后解除使用权；保留图或未执行
backward 的图继续阻止释放。旧版本引擎若没有图保留标志查询接口，则保守等待 context 被 GC。
生命周期检查不会访问 saved tensors，因此不会触发 activation offload 的恢复钩子。

当回收最后一组 workspace 时，manager 保留一份 SHMEM runtime 引用，使后续模型复用已有 bootstrap
和固定 heap；已回收的对称分配不再占用 heap 内的分配额度。最终 `shutdown()` 才释放这份引用。
因此 checkpoint 后固定 heap 仍可能占用设备显存，但资源组和活跃分配不会随反复建模累积。
新模型超过现有固定 heap 容量时仍会报错，需要结束当前生命周期并重新配置 heap。

这三个生命周期入口都必须由完整 WORLD 的各 rank 在相同安全点、以相同顺序调用，包含本地没有
待回收资源的 rank。按首次绑定顺序和资源配置核对各 rank 的资源清单，只有全体同意后才释放。
关闭失败保留尚未释放的句柄；跨 rank 部分关闭或通信失败不能盲目重试，应终止并重启该作业。
checkpoint 会交换 Host 元数据，应放在计时区间之外；模型层的稳态 forward/backward 不新增
生命周期 collective。仍可使用原有幂等 `close()`，但所有 rank 必须保持同样的关闭顺序。

### 中断与异常退出

`managed_run` 显式接管默认 SIGINT/SIGTERM 处理器，退出时恢复；遇到框架已安装的自定义处理器会
拒绝覆盖，应改用框架回调。信号处理器只记录请求，各 rank 在下一个 `lifecycle_checkpoint()`
共同决定停止，然后关闭资源并以 `128 + signal` 退出。没有 checkpoint 的长任务不会立即响应
合作式停止；进入清理后再次收到信号也不会在处理器中重入 native teardown。

未知训练异常、失联 rank、卡住的设备调用不走自动 collective 清理；原异常向外传播，由进程外
launcher/watchdog 终止其余 worker。回收通信使用现有进程组及其 timeout，本组件不提供能中断
native 调用的进程内定时器。必须为作业配置进程外超时终止策略。
SIGKILL/OOM kill 无法捕获，也不能执行 Python 清理；不能承诺对称内存 finalize 或设备立即恢复。
本实现不在 `atexit`、`__del__` 或信号回调里发起 collective。

SHMEM Python层以进程级引用计数统一管理Runtime生命周期。每个MegaMoe执行资源组建立时配对调用一次
内部`shmem.acquire()`，关闭时在workspace释放全部对称Tensor后调用一次`shmem.release()`；
`share_execution_resources`的相同配置层共享同一组及workspace，因此只形成一个SHMEM引用。非最后一个
`release()`只减少本地计数，进程内最后一个引用才执行跨rank关闭（仅丢弃模块对象会等待后续安全点回收）。未来
MegaMHC、MegaDSA等Multicore特性复用同一SHMEM Runtime时，也通过同一配对接口共享这套进程级计数，
不在各消费者内重复实现生命周期协调。
重新开启生命周期时，所有 rank 完成关闭后使用新的 `HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT`，
`HYPER_PARALLEL_SHMEM_HEAP_SIZE` 等堆配置在下一生命周期首次获取引用时重新生效。

完整 Qwen 接入及启动方式见 [MegaMoe 示例](examples/mega_moe/README.md)。MegaKernel 内部阶段采集方式见
[MegaKernel Profiling 使用指南](../../../docs/guide/mega_kernel_profiling.md)。

---

## 性能建议

1. **dispatch ↔ compute 掩盖**：MoE 的 AllToAll dispatch 与 expert compute 在不同核上并发，是最核心的掩盖收益
2. **单边通信**：基于内存语义的单边通信（Symmetric Memory）避免传统集合通信的同步开销
3. **RATR 通信重排**：通过 Rank-Aware Tile Reordering 将 AllToAll 流量在时间轴上均匀分散，避免多源 Rank 同时涌向同一目标，降低尾延迟
4. **O0 vs O1**：O0 通过 host CPU 调度，O1 调度下沉到 AICore，性能更高但实现难度更大
