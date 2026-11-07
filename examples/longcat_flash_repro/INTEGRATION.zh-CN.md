# 客户项目接入 LongCat Flash MegaMoE

本文面向已能运行 LongCat Flash 的客户项目，说明如何用当前分支的 MegaMoE 替换专家执行。
当前客户交付分支为 `feat/longcat-flash-repro`，已包含 EOI 热副本、路由反向和 expert 调度增量。
原始接口基于模型提交 `ca3ce4194`，核对过的 HF 实现来自 Transformers5.13.1；当前优化验收见
[调优说明](../../docs/longcat-flash-cv-tuning.md)，热副本配置见[热副本说明](../../docs/longcat-flash-hot-replicas.md)。
环境准备、x86架构检查和独立复现命令见 [复现指南](README.zh-CN.md)。

## 1. 替换范围和数据流

推荐接入 `LongcatFlashMegaMoeExperts`。它保留 HF experts 的三个位置参数和参数名，处理真实专家、identity分支、权重布局及梯度。
客户继续使用原来的 attention、dense MLP、Router、loss、优化器和训练调度。

```text
hidden_states ── 原 Router ── TopK 的全局 expert IDs、原始权重
      │                              │
      ├── 真实专家 slots ── compact ── MegaMoE EP/GMM/SwiGLU ──┐
      │                                                       ├── 按原 token 合并 ── output
      └── identity slots ── 本地输入乘对应路由权重 ─────────────┘
```

LongCat 的“零专家”是 **identity专家**，其贡献为输入乘路由权重，并非恒零输出。
Router仍在真实专家与identity专家的完整集合上做TopK；保留原来的FP32 softmax、选择偏置、缩放及路由权重，
只剔除真实专家执行中的identity slots，不能对剩下的真实专家权重重新归一化，也不能截断原TopK。

记真实专家数为E、identity专家数为Z、EP大小为P：真实ID为 `[0, E)`，identity ID为 `[E, E+Z)`。
发给adapter的ID始终是全局ID，不能先对P或每卡专家数取模。专家归属以 **EP组内rank** 的顺序确定。

| 场景 | E / Z / P | 每个逻辑NPU的真实专家 / identity份额 | H / 专家I / Router TopK |
| --- | --- | --- | --- |
| 原始专家规模 | 512 / 256 / 64 | 8 / 4 | 6144 / 2048 / 12 |
| 单层训练复现 | 128 / 64 / 16 | 8 / 4 | 6144 / 2048 / 12 |
| 组件复现 | 32 / 16 / 4 | 8 / 4 | 6144 / 2048 / 12 |

identity份额表示路由规模和HF保留的未使用参数行，不表示identity需要跨卡计算。
缩小实验时按EP同比例缩E/Z、保持每卡8/4和原始宽度，模型仅减少层数；不能在EP16仍配置E512后声称是同一每卡负载。
历史冻结版本验收了后两行；当前新增优化的验收范围见对应调优报告，不能将旧 Trainer 结果移作新版本证明。原始EP64规模需要另行验收。

## 2. 公共调用与推荐选项

```python
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts

replacement = LongcatFlashMegaMoeExperts(
    module=source_hf_experts,
    context={"ep_size": ep_size, "ep": ep_size > 1},
    backend="megamoe",
    dispatch_mode="push",
    expert_weight_layout="EOI",
    swiglu_precision="fused",
    routing_backend="triton",
    optimize_tasks=True,
    direct_weight_gradients=True,
    initial_token_capacity=24576,
    replica_slots_per_rank=0,
)
```

这里 `source_hf_experts` 是 **完整逻辑形状** 的 HF `LongcatFlashExperts`，可以在CPU或meta上构造。
EP本地权重已经切好的客户项目使用第4节，不能把局部张量伪装成全局HF模块传给上面的构造流程。

| 参数 | 接入含义 |
| --- | --- |
| `backend="megamoe"` | MegaMoE执行；`native`用于对照，NPU上使用GMM/permute/AllToAll/unpermute等算子 |
| `dispatch_mode="push"` | 参考单层Trainer使用的传输方式；`pull`可单独建模块比较 |
| `expert_weight_layout="EOI"` | 使用Torch Linear的 `[expert, out, in]` 布局，避免每次forward转置拷贝 |
| `swiglu_precision="fused"` | **默认值**；原生路径调用 `torch_npu.npu_swiglu`，MegaMoE内嵌CANN `swi_glu/swi_glu_grad` |
| `routing_backend="triton"` | 参考性能结果所用的有序gather/combine；依赖Triton-Ascend3.2.0，要求NPU BF16输入及FP32路由权重 |
| `routing_backend="torch"` | adapter的默认路由实现，可先用于功能接入；精度阶段和性能比较两侧使用同一选择 |
| `optimize_tasks=True` | 使用当前路由的任务边界和单专家任务寻址，默认开启 |
| `direct_weight_gradients=True` | EOI、无副本且任务优化开启时，保留完整Gate/Up梯度形状并清零unused尾部，默认开启 |
| `initial_token_capacity=24576` | 对每卡T2048、TopK12的初始compact-real行预留；变长输入可自动增长，不丢token；其他T应按实际负载设置 |
| `replica_slots_per_rank=0` | 默认无副本；当前 `feat/longcat-flash-repro` 分支已支持EOI热副本，LongCat默认按完整专家迁移，见[热副本接入与验收](../../docs/longcat-flash-hot-replicas.md) |

原生Router的TopK仍是12。adapter压紧真实slots以后，内部使用 `num_experts=E, top_k=1` 的执行器，
逐assignment执行；不要把内部K=1误改为12，也不要把identity ID直接传给通用 `MegaMoeExperts`。

实际forward契约为：

```python
weights, ids = moe.router(hidden_states)[-2:]
shape = hidden_states.shape
output = moe.experts(
    hidden_states.reshape(-1, shape[-1]),  # BF16，计算所在NPU上的 [T, H]
    ids,                                # 同设备的int32/int64 [T, K]，全局ID
    weights,                            # 同设备的FP32 [T, K]，保留autograd
).reshape(shape)
```

保留Router输出的真实权重；不要 `detach()`、转NumPy或重建为无梯度Tensor。
`swiglu_precision="hf"` 是显式的HF BF16中间舍入兼容选项，不能混用两种语义比较性能。
即使选择 `hf`，HF逐专家 `F.linear` 与原生GMM也可能有BF16末位差异；本次主验收对照为原生GMM。

## 3. HF模块尚未做EP切分：先替换，再切分

接入顺序如下：

1. 建立模型和显式EP进程组；设备选择和组创建沿用客户框架。
2. 在EP参数切分之前，以目标EP大小构造第2节的adapter，赋给 `moe.experts`。
3. 对adapter的 `gate_up_proj`、`down_proj` 在专家轴0做EP切分，保留下面的归属顺序。
   Router输出仍覆盖E+Z，各EP rank使用一致的Router状态；dense及Router的同步策略由客户原框架继续管理。
4. 由客户框架完成FSDP、参数物化和checkpoint加载；加载时应用第5节的packing转换。
5. 在每层真实专家权重已具备正确EP归属后，首次forward前调用一次：

   ```python
   moe.experts.configure(ep_group=ep_group, ep_size=ep_size, top_k=moe.router.top_k)
   ```

6. 完成模块和参数注册后再创建优化器。训练过程中每次forward使用当前物化权重，不缓存初次读取的BF16副本。

原HF `LongcatFlashMoE.forward` 已按第2节的顺序调用experts，因此保留其Router和forward即可。
自定义MoE forward若还执行旧的dispatch/permute/AllToAll，应将整段专家执行替换为adapter调用，避免重复通信。

当前模型adapter要求 **TP=CP=PP=1**。通用MegaMoE底层的能力不能自动视为该LongCat模型接入已支持的并行组合。
本分支不支持在MegaMoE动态token/heap增长路径上使用图捕获；先沿用复现配置的 `compile.enabled=false`。

## 4. 客户已有EP本地权重：保留Parameter绑定

本节适用于客户项目已持有正确EP归属的本地 `nn.Parameter`，且尚未构建FSDP封装和优化器。
下面的工厂只在meta上创建HF形状描述，不分配全局专家权重；随后绑定客户的本地参数。
`gate_up_proj` 必须已经是 `[本rank真实行, 本rankunused行]` 的顺序，行数和第5节一致。
函数不切分权重、不迁移优化器状态，也不隐式转换dtype。

```python
import torch
import torch.distributed as dist
from torch import nn
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts

from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts


def bind_local_longcat_experts(
    config: LongcatFlashConfig,
    gate_up_proj: nn.Parameter,
    down_proj: nn.Parameter,
    ep_group: object,
    *,
    routing_backend: str = "triton",
    initial_token_capacity: int = 24576,
) -> LongcatFlashMegaMoeExperts:
    """Bind materialized local parameters before FSDP and optimizer construction."""
    ep_size = dist.get_world_size(ep_group)
    real, zero = config.n_routed_experts, config.zero_expert_num or 0
    if real <= 0 or real % ep_size or zero % ep_size:
        raise ValueError("Real and identity counts must be divisible by EP")
    expected_gate = ((real + zero) // ep_size, 2 * config.expert_ffn_hidden_size, config.hidden_size)
    expected_down = (real // ep_size, config.hidden_size, config.expert_ffn_hidden_size)
    for parameter, expected in ((gate_up_proj, expected_gate), (down_proj, expected_down)):
        if not isinstance(parameter, nn.Parameter) or parameter.is_meta:
            raise ValueError("Pass existing materialized local Parameters")
        if tuple(parameter.shape) != expected or not parameter.is_contiguous():
            raise ValueError(f"Expected contiguous local EOI parameter with shape {expected}")
    if gate_up_proj.device != down_proj.device or gate_up_proj.dtype != down_proj.dtype:
        raise ValueError("Local expert parameters must share dtype and device")
    with torch.device("meta"):
        source = LongcatFlashExperts(config)
        result = LongcatFlashMegaMoeExperts(
            module=source,
            context={"ep_size": ep_size, "ep": ep_size > 1},
            backend="megamoe",
            dispatch_mode="push",
            expert_weight_layout="EOI",
            swiglu_precision="fused",
            routing_backend=routing_backend,
            optimize_tasks=True,
            direct_weight_gradients=True,
            initial_token_capacity=initial_token_capacity,
            replica_slots_per_rank=0,
        )
    result.gate_up_proj = gate_up_proj
    result.down_proj = down_proj
    if ep_size > 1 and zero:
        result.register_buffer(
            "_checkpoint_ep_size",
            torch.tensor([ep_size], dtype=torch.int64, device=gate_up_proj.device),
        )
    return result
```

调用方将返回值设为原MoE的experts，恢复原来的train/eval状态，完成FSDP封装后再调用 `configure`。
这条路径复用原Parameter对象；客户checkpoint loader仍负责全局到本地的权重归属和packing，不应再次对局部张量做全局packing。

如果客户的Gate/Up只有每卡8个真实行，需要在checkpoint导入和模型构建阶段明确补齐并保留4个unused行后再使用该adapter。
如果原权重是分离的 `w_gate/w_up`，或使用EIO布局，需要在构建优化器前转换并注册为正确的新Parameter，
同时转换checkpoint；已有优化器的主参数与动量不会由adapter自动迁移。

希望完全保留客户自定义参数格式时，可直接使用
[通用MegaMoeExperts](../../hyper_parallel/core/multicore/modules/mega_moe/module.py)：
设置 `create_parameters=False`、`num_experts=E`、`expert_weight_layout="EOI"`，每次forward传当前的
`expert_weights=(gate_up, down)`。但调用方必须自行接入真实slot压紧、identity合并及有序输入梯度，
不能直接将HF Router的E+Z个ID交给通用执行器。可参考
[adapter实现](../../hyper_parallel/models/longcat_flash/adapter/experts.py) 和
[masked_expert_forward](../../hyper_parallel/core/multicore/modules/mega_moe/assignments.py)。
这属于额外集成工作，不能用通用执行器一次调用替代完整的LongCat adapter契约。

## 5. 权重布局、checkpoint与梯度

EOI表示 `[expert, out_features, in_features]`：

| 参数 | HF全局形状 | adapter做EP后每rank的形状 |
| --- | --- | --- |
| `gate_up_proj` | `[E+Z, 2I, H]` | `[(E+Z)/P, 2I, H]`，前E/P行执行、后Z/P行保留 |
| `down_proj` | `[E, H, I]` | `[E/P, H, I]` |

例如EP64/E512/Z256时，每卡Gate/Up是 `[12, 4096, 6144]`，Down是 `[8, 6144, 2048]`。
Gate/Up内的通道顺序是 **先Gate、后Up**；分离权重需沿输出轴拼接，不能交错拼接。
EOI要求dense contiguous存储；如果客户原权重为EIO，需要在构建阶段转换轴序，避免每次forward复制整份专家权重。

HF全局Gate/Up的行顺序为 `[全部E个真实行, 全部Z个unused行]`。
adapter在EP切分前将其转换为：

```text
[rank0真实行, rank0 unused行, rank1真实行, rank1 unused行, ...]
```

否则直接等分HF全局Gate/Up，后面的rank会拿到错误的专家或纯unused行。
Down只含真实行，按相同的真实专家归属在轴0切分。
[ExpertOwnershipPacking](../../hyper_parallel/models/longcat_flash/adapter/layout.py) 提供正反转换：

```python
from hyper_parallel.models.longcat_flash.adapter.layout import ExpertOwnershipPacking

packing = ExpertOwnershipPacking(real_experts, zero_experts, ep_size)
packed_gate = packing.apply(full_hf_gate)
restored_hf_gate = packing.reverse_op.apply(packed_gate)
```

转换作用于完整逻辑权重，不能把已packed的checkpoint再次pack。
adapter参数名仍是 `gate_up_proj`、`down_proj`，但EP>1且E/Z均非零时还会保存 `_checkpoint_ep_size`。
**同名不意味着可以直接互相load state_dict**：HF到adapter要pack，导出HF要逆转换并移除marker。
通过本库 `apply_module_replacements(..., weights_mapping=conversions)` 接入时，保留登记的转换记录供checkpoint层使用；
直接赋值模块的客户框架需自行调用上述转换并维护marker。

改变EP大小时，先还原完整HF逻辑行序，再按新EP pack并切分；不能只改marker或把旧EP局部权重直接重新均分。
完整导出还需保留所有unused行，不可只保存真实Gate/Up的前缀。

训练时注意以下约束：

- MegaMoE计算输入及当前专家权重为BF16，Router权重为FP32；参考训练采用FP32初始化、主参数、归约和loss。
- FSDP负责在调用前提供**完整的本地EP专家矩阵**，不能将仍被FSDP切成片段的矩阵直接交给kernel。
  当前adapter识别本库DTensor；其他框架的DTensor/flat parameter需在物化边界接好当前Tensor与autograd关系。
- `.detach()`、`.data`或缓存旧的物化权重会切断梯度或使用过期值。保留输入、Router权重和Gate/Up、Down的全部梯度链。
- unused行保留完整梯度形状且尾部为零。某个真实专家未命中时，其梯度数值为零；
  全EP均只命中identity时，真实专家保持 `grad=None`，避免无意触发AdamW权重衰减。
- 本rank没有真实token不等于可以跳过调用：其他rank可能向它发送真实assignment，所有EP rank仍须保持一致的调用顺序。
- Router和dense参数的复制/分片/梯度同步沿用原框架；不要对专家参数额外跨EP all-reduce，混合不同专家的梯度。

## 6. 已使用HyperParallel Trainer时的最小配置

在原训练配置的 `plan_overrides` 中加入或合并下面三项，保留客户原attention/dense/loss设置。
这些项分别负责experts替换、EP参数切分、EP组绑定；只加第一项不能完成分布式接入。

```yaml
plan_overrides:
- match: model.layers.*.mlp.experts
  replace:
    kind: module
    source_type: transformers.models.longcat_flash.modeling_longcat_flash.LongcatFlashExperts
    implementation:
      _target_: hyper_parallel.models.longcat_flash.adapter.experts.LongcatFlashMegaMoeExperts
      backend: megamoe
      dispatch_mode: push
      expert_weight_layout: EOI
      swiglu_precision: fused
      routing_backend: triton
      optimize_tasks: true
      direct_weight_gradients: true
      initial_token_capacity: 24576
      replica_slots_per_rank: 0
- match: model.layers.*.mlp
  in_src:
    hidden_states:
      tp: replicate
      cp: shard(1)
      ep: replicate
  in_dst:
    hidden_states:
      tp: replicate
      cp: shard(1)
      ep: replicate
  out_src:
    tp: replicate
    cp: shard(1)
    ep: replicate
  out_dst:
    tp: replicate
    cp: shard(1)
    ep: replicate
  region_dispatch: false
  params:
    experts.gate_up_proj:
      ep: shard(0)
    experts.down_proj:
      ep: shard(0)
- match: model.layers.*.mlp.forward
  replace:
    kind: method
    implementation:
      _target_: hyper_parallel.models.longcat_flash.adapter.parallel.longcat_flash_ep_compute_fn
```

同时设置实际 `accelerator.ep_size`，并保持 `tp_size=cp_size=pp_size=1`。
精度验收和复现对照沿用FP32主参数及归约、BF16计算、`cast_forward_inputs: false`，
完整配置见 [megamoe.yaml](templates/trainer/megamoe.yaml)；其余框架参数不应仅为替换experts而擅自改动。

## 7. 多层共享与退出

多个串行层可以共享执行资源，专家权重与优化器状态仍逐层独立。
使用HyperParallel模型/Trainer路径时由模型materialization hook登记；客户手工绑定可在所有层configure后、首次forward前调用：

```python
from hyper_parallel.models.longcat_flash.adapter.materialization import register_materialization

register_materialization(model)
```

该函数在模型首次forward前收集已configure的执行器并共享兼容资源，还登记RoPE的可重建buffer。
要求各层配置、EP成员和调用顺序兼容；并行重入、不同stream并发执行的层不能直接套用串行共享。
使用通用执行器时，对兼容串行实例显式调用 `MegaMoeExperts.share_execution_resources(executors)`，必须早于首次执行。
一个进程内SHMEM同时只支持一套有序EP成员，不能在活跃资源之间切换互不相同的EP组。

最后一次backward或推理完成后，在所有rank以同样顺序关闭每层资源，再销毁process group：

```python
torch.npu.synchronize()
for layer in model.model.layers:
    layer.mlp.experts.close_runtime()
dist.destroy_process_group()
```

客户自定义模型树应按自己的稳定层序遍历；关闭必须发生在最后的反向和异步使用完成之后。
不要依赖Python析构完成分布式资源释放，也不要在仍需backward时提前关闭workspace。

## 8. 客户侧验收顺序

1. 按[当前版本指南](README.zh-CN.md)重新编译匹配的 native、核对 editable 来源和实际加载库，再运行独立历史前端精度入口。
   保留客户已验证的Torch/PTA；x86 native必须由客户环境构建。旧 `run.sh --check-only` 只对应历史冻结提交。
2. 在客户项目中固定同一份输入、权重、上游梯度、原Router ID和权重，对照原生GMM路径：
   核对输出、dX、dRouter权重、完整Gate/Up及Down梯度，覆盖混合/全真实/全identity/本rank空真实路由。
   CPU缩小形状可检查接口；NPU验收保留H6144/I2048、TopK12和每卡8/4，仅缩层数或按EP裁剪专家规模。
3. 使用现场同一checkpoint、输入与路由运行连续更新，比较loss、裁剪前梯度范数、更新后的模型和FP32主参数。
   历史 Trainer 复现约定为10次更新×16rank；当前热副本组件连续更新入口见[热副本验收](../../docs/longcat-flash-hot-replicas.md)。
   客户全模型、更换优化器或框架后的路径需要独立验证。
4. 精度通过后再测性能：两侧相同EP、形状、初始化、checkpoint/recompute和SwiGLU/路由语义，
   预热后记录所有rank完成时间及rank-max样本，保留慢样本，正式耗时关闭profiler。
5. 与参考结果比较时区分“只替换MoE”和参考配置中同时启用attention/dense适配的完整单层Trainer。
   x86/910C、客户PTA和客户全网络的结果以现场报告为准，组件收益不能直接换算成全网络吞吐。

源码入口：[LongCat experts](../../hyper_parallel/models/longcat_flash/adapter/experts.py)、
[EP绑定](../../hyper_parallel/models/longcat_flash/adapter/parallel.py)、
[原生NPU对照](../../hyper_parallel/models/longcat_flash/adapter/npu.py)、
[组件精度worker](../../tests/torch/models/longcat_flash/_compare_longcat_flash.py)。
