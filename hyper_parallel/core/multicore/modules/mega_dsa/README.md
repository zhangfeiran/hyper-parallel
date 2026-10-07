# megaDSA development baseline

This package implements the P0 foundations of the 2026-10-07 megaDSA plan:
logical metadata, a small CPU numerical oracle, and an explicit single-card
enhance reference boundary. No `MegaDsaCore` device runtime, SHMEM consumer,
mixed-worker adapter, or production fallback is enabled yet.

## Implemented contracts

- `DsaBatchMeta` declares the global packed token namespace, sequence boundaries,
  ordered CP membership, root PE mapping, owner-local addresses and invocation
  identity. Q/K storage may be reordered or noncontiguous. Valid lengths do not
  inherit MegaMoE's 128-token alignment restriction.
- `sparse_attention_reference` evaluates absorbed MQA with a shared compressed
  K/V tensor, an explicit model attention scale, and ordinary Torch autograd.
  It preserves both key and value gradient contributions. It rejects missing
  selected remote KV rather than treating a local buffer as the full context.
- `indexer_reference` computes weighted ReLU dot products, preserving signed,
  already-scaled merge weights. Exact selection requires the full global key
  set. Its deterministic tie rule is descending score followed by ascending
  global token ID; equivalence to CANN's tie behavior remains to be established.
- `selected_kl_reference` normalizes the teacher per main head, then sums and
  L1-normalizes across heads. Teacher gradients are detached; gradients reach
  only index Q, index K and merge weights. `DsaLossNormalization` explicitly
  compensates a declared downstream CP sum or average. Projection input detach
  and the trainer's auxiliary upstream scale remain the caller's responsibility.
- The offline backend probe records package versions, exact registered schemas
  and custom-library SHA256 without launching device work.
- `CannDsaLayout` translates global packed indices into CANN's sequence-local
  namespace, removes illegal slots and stably compacts valid entries before the
  first `-1`. Backend lengths omit the metadata's initial zero. It rejects
  incomplete/reordered Q/K or CP>1 before stock right-down masking can use an
  incorrect query offset.
- `CannDsaReference` wires indexer, sparse attention and forward-computed KL
  derivatives through the installed enhance ABI. It reuses the registered
  sparse-attention autograd and passes one shared K/V tensor. KL receives Python
  length lists, saves only its three indexer gradients per invocation, and
  applies the declared global normalization and upstream scale once.
- `attention_bsnd` provides an explicit model boundary with Nkv=1 and zero RoPE
  output padding, preserving the model's projection/autograd ownership.

The initial CANN reference support matrix is CP=1 with complete, ordered packed
TND storage; BF16; C=512, Dr=64, K=2048; main heads 32/64/128; index heads
8/16/32/64 and Di=128. These checks describe the development adapter's intended
range, not a completed hardware support certification. Native statistics stay
opaque `[1,T,H]` FP32 buffers, separate from the CPU oracle's `[T,H]` buffers.

The reference layouts are `[Tq,H,C]`, `[Tk,C]`, `[Tq,H,Dr]`, `[Tk,Dr]` and
int32 `[Tq,K]` indices. The only accepted index namespace is `global_packed`.
The metadata cumulative lengths include an initial zero; CANN length arguments
have a separate adapter contract. Unknown IDs and duplicate selections fail.
`-1`, cross-sequence and future slots do not participate in attention or KL.
Rows with no legal candidates produce zero output, zero loss and zero gradients,
with softmax max/LSE `-inf` and sum zero. These are oracle conventions; backend
empty-row behavior must still be compared on device.

FP64 is preserved for gradcheck; other floating-point inputs are promoted to
FP32. The reference materializes selected KV and index scores and accepts CPU
tensors only. Its output contains only compressed values, without CANN's RoPE
backward padding. Its max/sum layout is `[Tq,H]`, and LSE uses natural logarithms;
these buffers must not be passed directly into a CANN backward operator.

## Running validation

Install this checkout into an isolated environment with `pip install -e .`, then
verify `hyper_parallel.__file__` before running:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/ut/core/multicore/mega_dsa
OMP_NUM_THREADS=1 python -m pytest -q tests/ut/core/multicore
python hyper_parallel/core/multicore/examples/mega_dsa_backend_probe.py
```

Once a matching enhance OPP is explicitly activated, run the standalone device
validator through the idle gate, in a process with this checkout installed:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 \
NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python hyper_parallel/core/multicore/examples/mega_dsa_cann_validate.py \
  --output /tmp/mega_dsa_cann_validation.json
```

The validator records output, natural-log LSE, all four attention input gradients,
KL loss and all three indexer gradients against the FP32 oracle, plus main/index
gradient isolation. Failures retain a JSON stage/traceback and a nonzero process
exit. The short packed case covers full legal histories; it is not evidence of
long-context Top-K selection, CP, projection-parameter acceptance or performance.
BF16 acceptance thresholds still require primitive-specific calibration.

CPU acceptance compares against independent dense and unabsorbed MLA formulas,
including all attention inputs, W_UK, W_UV and output projection gradients.
Separate key/value gradients must sum to the merged compressed-KV gradient.
Other cases cover fixed-selection gradcheck, signed merge weights, packed causal
boundaries, padding, reordered KV, interior/zigzag Q shards, detached teachers,
index projection isolation, coefficient/upstream scaling, and unequal CP query
contributions under declared sum/average reducers. CP tests simulate the objective
in a single CPU process; they do not validate collectives or remote transport.

## Remaining P0 gates

The first environment probe on the branch base reported Torch `2.9.0+cpu`,
TorchNPU `2.9.0.post6`, omni training custom ops `1.0`, and CANN environment paths
pointing to `9.1.0`. All three enhance forward/KL names were registered. This
establishes registration only; it does not establish device support or the build
provenance of the installed package relative to `dependencies.lock.json`.

The installed KL schema declares `actual_seq_qlen` and `actual_seq_klen` as
`SymInt[]?`. A dispatcher diagnostic accepted both Python lists and CPU Tensor
lengths through argument parsing, then stopped because KL has no Meta kernel.
Thus the schema difference alone does not establish a wrapper failure. The new
reference passes precomputed Python lists to avoid a device-to-host length
conversion at invocation time; no existing model wrapper is changed.

The first actual NPU call initialized Ascend910B3 and allocated its BF16 inputs,
then failed in indexer loading: `aclnnLightningIndexerEnhance` and its workspace
entry were unavailable in the active op-api search path. The active CANN 9.1
library exposes stock DSA names, which do not substitute for enhance symbols.
An existing local enhance OPP exports the required names, but its `version.info`
reports compiler 8.5.1; it was not activated under the 9.1 baseline. A matching
isolated OPP build and explicit activation are required before device acceptance.

P0 still needs real model wiring/parameter-gradient acceptance, a loadable CANN
reference with forward/backward comparison, hardware-calibrated BF16 tolerances,
training loss-scale integration and real index trace profiling. Follow those
gates before SHMEM coexistence (P1) or mixed-team extraction (P2). CPU mocks of
the CANN boundary validate ABI/autograd wiring only; they establish no NPU
numerical or performance result.
