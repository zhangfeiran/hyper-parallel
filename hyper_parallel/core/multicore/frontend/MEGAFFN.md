# Dense MegaFFN training integration

The target is a trainable BF16 dense SwiGLU FFN after attention, with independent
model/optimizer alignment and measured gains over the existing packed
`SwiGLUMLP`. Dense FFN has no router, expert ownership or dispatch/combine transport.

## Semantic and training interface

`frontend/examples/dense_ffn.py` describes packed gate/up Matmul, dense SwiGLU and
down Matmul. `DenseSpec` binds symbolic dimensions without EP topology. Lowering
derives dependencies, bindings and conservative intermediate lifetimes from SSA.
Canonical primitive identities and trusted provider admission are mandatory.

```python
from hyper_parallel.core.multicore import MegaFFN

ffn = MegaFFN(hidden_size=1024, intermediate_size=4096).to("npu")
output = ffn(tokens)  # contiguous BF16 [..., 1024]
output.float().square().mean().backward()
```

Parameters are gate/up `[H, 2I]` packed columns and down `[I, H]`. Ordinary
`nn.Module`, autograd, optimizer, state dict and checkpoint/recompute APIs apply.
Norm and residual composition stay at the decoder boundary. Bias and TP
collectives are outside the initial dense contract.

Caller-owned local weights use `MegaFFN(..., create_parameters=False)` and
`ffn(tokens, weights=(local_gate_up, local_down))`. No borrowed weights are cached
or copied. Autograd state belongs to each invocation; close rejects future
forwards while pending backward remains valid.

## Existing Trainer replacement and checkpoints

`modules/mega_ffn/adapter.py` uses the existing `@module_replacement` factory.
Apply before sharding/optimizer creation with the normal weight mapping list:

```python
import torch

from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)

rules = [ModuleReplacementSpec(("model.layers.*.mlp",), MegaFFNAdapter, SourceMLP)]
plan = compile_module_replacements(model, rules)
model, mapping = apply_module_replacements(model, plan, weights_mapping=[])
model.to(dtype=torch.bfloat16, device="npu")
```

The factory packs bias-free SiLU gate/up/down weights once, preserving source
dtype/device until placement and all training flags. Checkpoint transforms reverse
concatenation/transposition exactly. Tied projections and different gate/up
`requires_grad` policies are rejected. The two-card BF16 FSDP/SGD component gate
passed changing-weight, recompute and exact checkpoint-restoration checks.

## Native execution and evidence boundary

The providers are native Torch Matmul and NPU packed SwiGLU with exact
forward/backward dispatcher schemas. Generated C++ forward/VJP adapters follow
SSA, including transpose handling and reverse gradient accumulation. The builder
seals generated source, compiler identity, Torch version/ABI and library hash.
The loader verifies them before binding operators.

Native adapters use one Python/native call per direction and enqueue provider
kernels on the current stream. The module defaults to this host-stream backend.
`Program.compile()` produces source artifacts and does not build a binary.

The explicit `resident_tiles` candidate lowers row-separable BF16 SSA into paired
AIC/AIV private queues. Each cube owns its token tiles and both vector lanes
participate in activation joins, including an empty lane at a one-row tail.
Event publication synchronizes scalar and DMA pipelines explicitly. Consumers
invalidate the event cache line and reload its GM value through a volatile
pointer on every poll, including while another core is still producing it.
Bounded producer prefetch follows the same queue order in the CPU scheduling
checker and native kernel. A single mixed kernel executes forward Matmul,
FP32 SwiGLU followed by BF16 storage, and Matmul. Generic row-local fanout and
transposed right weights are admitted without an FFN body matcher. Row mixing,
left transpose and unsupported storage are rejected before execution.

```python
from pathlib import Path

from hyper_parallel.core.multicore import MegaFFN
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig

execution = DenseExecutionConfig(
    backend="resident_tiles", cann_root=Path("/path/to/cann"), soc="Ascend910B3"
)
ffn = MegaFFN(1024, 4096, execution=execution).to("npu")
```

The existing replacement API accepts
`context={"mega_ffn_execution": execution}`. The selected SDK supplies core count,
workspace bytes and raw `TCubeTiling` size; these values are not assumed from
another SDK or device. The builder seals source, SDK headers/scripts/libraries,
device compiler and TorchNPU ABI plus the packed binary and host libraries. It
uses caller-owned caches and never modifies native vendor installations. The
launcher registers the exact binary lazily, checks registration failures and
submits through TorchNPU's task queue in stream order.

Backward currently reads the exact resident invocation's packed/activation cache
through the independently generated native provider VJP. Weight gradients retain
their full token contraction; they are not summed from rounded BF16 tile partials.
Intermediates, events and workspace belong to each invocation. Checkpoint and
close use the same module lifecycle as the host backend. Cached metadata has an
initialization event and all device storage is recorded on the launch stream.

Native build/package/load, real host SDK tiling, CPU queue replay and saved-state
protocol tests have passed. The one-card resident component matrix also passed
BF16 output and gradient comparisons for empty/tail inputs, checkpoint, pending
backward after close and two nondefault streams. The two-card BF16 FSDP/SGD
component gate also passed. Independent complete-model numerical/performance
acceptance remains separate. Resident backward, scratch-buffer reuse and
device tuning remain pending; retained forward buffers are intentionally not
aliased while backward needs them. CPU native execution is a reference.

The installed `npu_ffn` interface restricts gated activations to FP16 inference;
it is not admitted as a BF16 training shortcut. Successful host compilation and
CPU gradients do not establish NPU numerical correctness or performance.

## Final acceptance requirements

`examples/mega_ffn/qwen_dense_model.py` composes the FFN after existing Qwen GQA
attention and normalization in a complete dense LM. It supports conventional
three-projection (`common`), existing packed (`packed`) and AST (`mega_ffn`) models
through the same replacement mechanism.

1. NPU forward, dX and both weight gradients; tails, empty tokens and recompute.
2. FSDP/local-weight lifecycle and checkpoint roundtrip on actual devices.
3. Independent trajectories, FP32 main parameters and moments. Initial state is
   aligned once, with no later cross-run resynchronization.
4. Fresh-process native/native controls and ABBA comparisons against both
   baselines, measuring complete optimizer steps and peak memory with exact
   source/framework/device ownership recorded.
5. Resident-worker/tile-pipeline execution with buffer reuse and RAW/WAR/WAW
   dependencies, followed by measured gains. A smaller AST or compiled host
   adapter alone does not satisfy this item.

Long device runs use the user's `npu_wait_and_run.sh` helper with its locks,
health checks and final idle recheck.

## Reproducible acceptance entry

The benchmark advances separate models and FP32-master AdamW optimizers. It
records logits, losses, gradients, BF16 parameters, FP32 master parameters, both
moments and exact step counters. All initial states must match exactly, with no
later synchronization. A failing row remains in JSON and stops the run.

The supervisor starts every native control and ABBA slot in a fresh process,
using the existing idle helper for each job. It records sampled NPU process
ownership, Linux process start identity and host CPU ticks. Unknown or foreign
device processes invalidate a round. Timed rounds need ownership observations
and no host busy interval above 20 percent. Increase the iteration count if a
short measurement window contains no observer sample.

```bash
python hyper_parallel/core/multicore/examples/mega_ffn/acceptance.py run \
  --helper "$HOME/doc/npu_wait_and_run.sh" \
  --root /tmp/megaffn-dense-qwen-acceptance --device 0 --blocks 5 -- \
  --steps 100 --warmup 100 --iterations 100 --dense-backend resident_tiles
```

Both `common` and `packed` baselines get independent native/native numerical and
performance controls. Bootstrap confidence intervals resample process blocks,
not correlated steps within one process. A performance pass needs 100 independent
numerical steps, native-control drift at most 2 percent with a confidence interval
including zero, and a positive candidate interval exceeding both 1 percent and
the native noise envelope. Raw per-step samples, peak allocator memory, first-step
time, source/vendor/library hashes and optimizer/data/config identities are retained.
The acceptance analyzer also requires a resident forward payload for every
candidate decoder layer. `--dense-backend host_stream` remains a diagnostic mode
and cannot clear that resident requirement.

This evaluates a fixed-data, randomly initialized complete training graph. It
does not establish long-term real-data convergence or FSDP acceptance. Device
launch and execution failures remain failed gates.

`tests/torch/multicore/test_dense_ffn.py` exposes separate one-card resident and
two-card FSDP component cases. The one-card case also runs two nondefault streams
with independent borrowed weights and a shared executable plan, checking output,
dX, both dW tensors and unchanged input version counters. It closes the module
after both forwards, before their backwards, and waits on each stream before
reading gradients. Both component matrices have passed their device gates.
The two-card case checks BF16 gradients, repeated weight
updates across unshard/reshard, recompute and exact packed checkpoint restoration
against existing `SwiGLUMLP`. It uses independent SGD components, so FP32-master
AdamW under FSDP and full-model distributed training remain additional acceptance.
Launch it through the same idle helper with exactly two selected devices.
