# megaDSA development baseline

This package begins P0 of the 2026-10-07 megaDSA implementation plan. It supplies
logical metadata and a small CPU numerical oracle. No `MegaDsaCore` device runtime,
SHMEM consumer, mixed-worker adapter, or production fallback is enabled yet.

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
`SymInt[]?`, while the existing model wrapper passes Tensor lengths. This requires
an actual ABI call and a deliberate adapter decision before backend integration.
No existing model wrapper is changed by this baseline.

P0 still needs the real model boundary, explicit backend index-namespace and
RoPE-padding adapters, single-card CANN forward/backward and parameter-gradient
comparison, hardware-calibrated BF16 tolerances, training loss-scale integration,
and real index trace profiling. Follow those gates before SHMEM coexistence (P1)
or mixed-team kernel extraction (P2). No NPU numerical or performance claim is
made by the CPU oracle.
