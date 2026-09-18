# DeepSeek-V4.1 MegaMoe block precision

This example compares a standalone DeepSeek-V4.1 MoE block with
`DeepseekV41MegaMoe`. It retains the V4.1 learned router and shared experts;
only routed expert execution changes to `MegaMoeExperts`. This standalone entrypoint
does not install Trainer replacements. The separate Trainer path and its validation
boundaries are described in the [integration report](../../docs/deepseek_v41_training_report.md).

## Activation and dependencies

Use Transformers **5.13.0**, with the checkout installed in editable mode, and
build/activate its Torch multicore payload against the active CANN installation.
The example uses BF16 on Ascend 910B. The CPU tests also need the repository's
Trainer dependencies, including `torchdata`.

The example uses the released `swiglu_limit=10`. HF 5.13 clips the gate at the
positive limit and the up projection symmetrically; the model adapter passes the
same value to MegaMoe for both routed and shared experts. The validation crop
preserves the source model's limit without a separate zero-limit override.

## Run

```bash
source /path/to/CANN/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export HYPER_PARALLEL_PLATFORM=torch
export OMP_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://127.0.0.1:29571
python -m torch.distributed.run --standalone --nproc-per-node=2 --module \
  hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_precision \
  --dispatch-mode push --route learned --acceptance fp32 \
  --output output/deepseek_v41_megamoe.json
```

Use an available bootstrap port for each independent job. Activate the original
runtime OPP path even if a private OPP view was needed during compilation.

Defaults: 128 local tokens, H=512, I=128, E=8, TopK=2, three SGD steps at
learning rate 0.01, `sqrtsoftplus` routing, and routed scaling 1.7. Weights are
seeded independently of rank; inputs and loss derivatives vary by rank and step.
The crop's complete attention, mHC and Engram layers are not executed here.

Useful variations:

- `--reference owner_ep` (default): the actual DSV4.1 token A2A/owner-compute EP path.
- `--reference hf_replicated`: the earlier full-HF-experts-per-rank diagnostic,
  with a manual BF16 expert-gradient all-reduce.
- `--dispatch-mode pull`: integrated pull transport.
- `--route hotspot`: every token selects experts `0..TopK-1`; the remaining
  experts are empty. Gate parameters have no gradient in this diagnostic case.
- `--vision`: alternating text/image routing using V4.1 correction biases.
- `--ep-size 2` with four processes: disjoint strided groups `[0,2]` and `[1,3]`.
- `--top-k 1` or `--top-k 8`: selection boundary cases with E=8.
- `--tokens`: positive multiple of 128; capacity remains lossless.
- `--swiglu-limit`: positive clamp limit; the default is the released value 10.
- `--clamp-probe`: use exact projection values beyond and within the limit to
  validate clipped forward/backward math without BF16 threshold-mask ambiguity.
- `--synchronize-step-weights`: start each step from identical current HF weights.
  This is automatic with the default `--acceptance fp32`. The previous native
  optimizer update is compared before the next copy. Use `--acceptance hf_bf16
  --fp32-oracle` without this flag to observe independent trajectories, which may
  cross different TopK boundaries after small BF16 gradient differences.

## Reference and evidence

The BF16 reference uses the original expert container with the configured
SwiGLU limit and the actual `DeepseekV41TopKRouter`. Both paths start from
identical global weights. MegaMoe takes a contiguous expert slice in **group-local** rank order
and transposes `[E,2I,H]` / `[E,H,I]` to `[E_local,H,2I]` / `[E_local,I,H]` once.

The comparison checks exact initial parameter mapping (and exact state alignment
every step when synchronization is enabled). It includes output, selected IDs, input gradient, routing-weight
gradient, every parameter gradient, and every parameter after each SGD step.
The default reference calls `deepseek_v41_ep_compute_fn`: tokens are sent to
the owning rank before local expert compute, and outputs return through A2A.
Its local expert gradients already include incoming tokens from all EP ranks;
there is no additional EP expert-gradient reduction. Only `hf_replicated` uses
the historical BF16 expert-gradient all-reduce. Router and shared gradients
remain local in this standalone block test. FP32 oracle partial gradients are
separately summed in FP32 to construct the mathematical reference.

For compatibility, JSON keys named `hf` and `hf_bf16_passed` identify the selected
BF16 reference; `config.reference` records which implementation actually ran.
The operator/reduction trace tools intentionally retain `hf_replicated`.

The default `--acceptance fp32` evaluates independent CPU FP32 matrix operations,
SiLU, score functions and autodiff. It calls neither HF experts nor MegaMoe.
For each backend and each step it copies that backend's **current** weights and
inputs, fixes the actual discrete expert IDs, and recomputes differentiable
router scores. Hotspot routing uses the supplied weights as independent leaves.
It sums expert gradients in FP32 within the actual EP group before slicing the
native layout. It checks the one-step SGD update from this same state. This is
not a separate three-step FP32 training trajectory and does not independently
validate discrete TopK selection; the HF comparison checks selected IDs exactly.

Acceptance checks **every tensor on every rank and step** independently against
FP32: relative L2 <= 1% and max absolute error / reference max absolute value <=
2%. Shapes, gradient presence, initial weights and selected IDs must match;
values must be finite. A zero reference requires an exactly zero candidate.
Both HF BF16 and MegaMoe must meet these same bounds. This approved block
criterion is an engineering regression budget, not a convergence guarantee.

JSON retains original `rtol=0.02, atol=0.002` elementwise comparisons and counts.
`hf_bf16_passed` reports the original cross-backend gate, `elementwise_passed`
retains each FP32 comparison's old elementwise outcome, and `fp32_passed` gives
the new per-backend result. `passed` and exit status use the selected acceptance
mode. `--acceptance hf_bf16` explicitly restores the original hard gate.

The report records source file hashes (including uncommitted files), git HEAD,
diff hash, native artifact hashes, framework versions, dependency path, CANN,
OPP path and device name. These are correctness runs, with no timing claim.

## Tests

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/auto_models/models/deepseek_v41/test_megamoe.py
python -m pytest -q tests/torch/multicore/test_deepseek_v41_megamoe.py
```

The system-test launcher imports no framework. The block matrix requires two
visible NPUs; the Trainer EP2/EDP2 case requires four. The launcher activates
the native payload and launches a separate worker. It selects the
approved FP32 gate with synchronized state and retains all BF16 diagnostics.
See the [adaptation plan](../../docs/deepseek_v41_adaptation_plan.md) for the
current validation status and the remaining Trainer/FSDP work.

## Operator diagnosis

With the same activated environment, run `deepseek_v41_operator_trace` instead
of `deepseek_v41_precision` using `--nproc-per-node=1`. It supports `--route
hotspot|learned` and `--dispatch-mode push|pull` with the same shape arguments.
It copies actual native scratch before reuse, compares each operator against
CPU FP32 using that operator's actual inputs, and replays the chain with
ordinary NPU matmuls plus either HF SiLU/multiply or fused `npu_swiglu`.
The output JSON and `.stepN.pt` snapshots are diagnostic artifacts, not timing
results. The full-block intervention changes only a temporary reference copy.

For EP2/EP4 run `deepseek_v41_reduction_trace` with that many processes. This
probe requires WORLD=EP and text routing. It compares original HF BF16,
fused-SwiGLU HF with BF16 gradient reduction, and FP32 partial dW reduction
rounded once at the end. These partials use actual BF16 intermediate inputs;
they are **operator-local references**, not the end-to-end FP32 oracle.

See the [operator diagnosis](../../docs/deepseek_v41_operator_precision_report.md)
for source evidence, causal interventions and the remaining precision boundary.
