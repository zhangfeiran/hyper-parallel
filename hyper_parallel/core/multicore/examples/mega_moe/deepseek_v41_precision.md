# DeepSeek-V4.1 MegaMoe block precision

This example compares a standalone DeepSeek-V4.1 MoE block with
`DeepseekV41MegaMoe`. It retains the V4.1 learned router and shared experts;
only routed expert execution changes to `MegaMoeExperts`. It does not install
Trainer replacements or claim FSDP/checkpoint compatibility.

## Activation and dependencies

Use Transformers **5.13.0**, with the checkout installed in editable mode, and
build/activate its Torch multicore payload against the active CANN installation.
The example uses BF16 on Ascend 910B. The CPU tests also need the repository's
Trainer dependencies, including `torchdata`.

The example explicitly uses `swiglu_limit=0`. HF 5.13 clamps unconditionally:
zero would otherwise zero the up projection. The model adapter binds unclamped
methods to **both** routed and shared experts while retaining their parameters.
Positive limits keep the HF implementation; the MegaMoe adapter rejects them.
The crop builder accepts `swiglu_limit=0.0` and records the original value as
`v41_source_swiglu_limit`, without editing the source model assets.

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
  --dispatch-mode push --route learned --fp32-oracle --synchronize-step-weights \
  --output output/deepseek_v41_megamoe.json
```

Use an available bootstrap port for each independent job. Activate the original
runtime OPP path even if a private OPP view was needed during compilation.

Defaults: 128 local tokens, H=512, I=128, E=8, TopK=2, three SGD steps at
learning rate 0.01, `sqrtsoftplus` routing, and routed scaling 1.7. Weights are
seeded independently of rank; inputs and loss derivatives vary by rank and step.
The crop's complete attention, mHC and Engram layers are not executed here.

Useful variations:

- `--dispatch-mode pull`: integrated pull transport.
- `--route hotspot`: every token selects experts `0..TopK-1`; the remaining
  experts are empty. Gate parameters have no gradient in this diagnostic case.
- `--vision`: alternating text/image routing using V4.1 correction biases.
- `--ep-size 2` with four processes: disjoint strided groups `[0,2]` and `[1,3]`.
- `--top-k 1` or `--top-k 8`: selection boundary cases with E=8.
- `--tokens`: positive multiple of 128; capacity remains lossless.
- `--synchronize-step-weights`: start each step from identical current HF weights.
  The previous native optimizer update is compared before the next copy. Omit
  this flag to observe independent optimizer trajectories, which may cross
  different TopK boundaries after small BF16 gradient differences.

## Reference and evidence

The HF BF16 path uses the original expert container with the unclamped activation
and the actual `DeepseekV41TopKRouter`. Shared experts are unchanged apart from
the explicit zero-limit convention. Both paths start from identical global
weights. MegaMoe takes a contiguous expert slice in **group-local** rank order
and transposes `[E,2I,H]` / `[E,H,I]` to `[E_local,H,2I]` / `[E_local,I,H]` once.

The comparison checks exact initial parameter mapping (and exact state alignment
every step when synchronization is enabled). It includes output, selected IDs, input gradient, routing-weight
gradient, every parameter gradient, and every parameter after each SGD step.
The full-expert HF oracle sees only local inputs, so its expert gradients are
summed within the EP group before comparison with native local expert gradients.
Router and shared gradients remain local on both paths.

`--fp32-oracle` additionally evaluates independent CPU FP32 matrix operations,
SiLU, score functions and autodiff. It calls neither HF experts nor MegaMoe.
For each backend and each step it copies that backend's **current** weights and
inputs, fixes the actual discrete expert IDs, and recomputes differentiable
router scores. Hotspot routing uses the supplied weights as independent leaves.
It sums expert gradients in FP32 within the actual EP group before slicing the
native layout. It checks the one-step SGD update from this same state. This is
not a separate three-step FP32 training trajectory and does not independently
validate discrete TopK selection; the HF comparison checks selected IDs exactly.

JSON contains separate BF16 and FP32 comparisons, finite-value checks, relative
L2 error, maximum absolute error, failing-element counts and the worst ratio to
`atol + rtol * abs(reference)`. The default remains `rtol=0.02, atol=0.002`.
`passed` and the process exit status use the existing HF BF16 gate;
`fp32_passed` reports each backend against FP32 independently. Adding the oracle
does not relax acceptance. BF16 rounding can cause isolated element failures,
including in the unchanged shared-expert baseline. Inspect the full report
before interpreting an aggregate pass/fail as an integration defect.

The report records source file hashes (including uncommitted files), git HEAD,
diff hash, native artifact hashes, framework versions, dependency path, CANN,
OPP path and device name. These are correctness runs, with no timing claim.

## Tests

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/auto_models/models/deepseek_v41/test_megamoe.py
python -m pytest -q tests/torch/multicore/test_deepseek_v41_megamoe.py
```

The system-test launcher imports no framework. It requires two visible NPUs,
activates the native payload, and launches a separate worker. It retains the
strict BF16 gate, including hotspot cases; failures are not skipped or hidden.
See the [adaptation plan](../../docs/deepseek_v41_adaptation_plan.md) for the
current validation status and the remaining Trainer/FSDP work.
