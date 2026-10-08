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
  and custom-library SHA256 without launching device work. It separately records
  the active custom OPP compiler version and op-api hash; Python registration
  alone does not identify the runtime vendor payload.
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
- `CannDsaSelection` admits only CPU-prepared external snapshots or the native
  indexer's output. All execution entry points reject raw Tensor indices.
  External snapshots are validated before upload; owned native storage is bound
  to one prepared layout and checked for mutation using Tensor version metadata.
  Device-indexer admission and execution do not read device values on the host.
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
device tests now show the locked native backward/KL do not implement this
empty-row contract. See the selected-set matrix below before using the CANN
reference with externally supplied selections.

External CANN selections require CPU int32 `[T,2048]` snapshots. Preparation
rejects unknown IDs and duplicate non-padding IDs, filters known future and
cross-sequence IDs, compacts legal keys, and requires exactly
`min(2048, causal prefix length)` legal keys per row. Underfilled and empty
selections fail before upload or native dispatch. The general CPU oracle
continues to support arbitrary legal selected sets and empty rows.

```python
# External snapshot preparation runs outside the training hot path.
selection = backend.prepare_selection(cpu_global_indices)
output, stats = backend.attention(q_nope, compressed_kv, q_rope, k_rope, selection)

# Native indexer output already carries the native candidate-count contract.
selection = backend.indexer(index_q, index_k, merge_weight)
output, stats = backend.attention(q_nope, compressed_kv, q_rope, k_rope, selection)

# Oracle/trace capture is an explicit device-to-host operation.
cpu_global_indices = selection.to_global_indices().cpu()
```

`attention_bsnd` and `kl_loss` require the same admitted-selection type. A
selection cannot be passed to a different layout instance, even with identical
metadata. Source snapshots and exported global indices do not alias admitted
storage. The native indexer path trusts that primitive's count/uniqueness
contract; numerical device validation remains necessary, particularly for
long contexts and ties. This restriction is a development API change; no model
component or native payload is modified.

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

The initial NPU call failed to resolve the enhance op-api symbols. An isolated
build from Omni ops commit `14a3ed57aa9fd56c9dcb7ec282423711db4bce80` then produced
the four target operators with `custom_opp_compiler_version=9.1.0`, including 16
device object files. This reference dependency is separate from the Multicore
`dependencies.lock.json` sources. Its op-api SHA256 is
`5da9e2033e2f6dec138c73b6cce12c44cd986e96f84fecc50fa0362cdbf54fa0`.

The shared toolkit's vendor registry was unreadable during offline compilation.
A private OPP view linked its CANN 9.1 contents and supplied an owned empty vendor
registry for the compiler. Device execution must retain the original CANN 9.1
`ASCEND_OPP_PATH`: a runtime probe showed the private view failed to load built-in
tiling, while the original path computed `arange` and multiplication correctly.
Activate the isolated vendor through `ASCEND_CUSTOM_OPP_PATH` and its op-api
directory through `LD_LIBRARY_PATH` before importing Torch/NPU. No shared toolkit
permission, registry or installation change is needed.

### Single-fixture device evidence

On Ascend910B3, CP=1, packed lengths `(3,5)`, BF16, main H=32, index H=8,
C=512, Dr=64, Di=128 and K=2048, the host-dispatched enhance reference completed
indexer, sparse attention forward/backward and selected KL. KL did not alter the
main attention gradients. Native indices used the declared sequence-local
namespace and converted to the expected global packed history sets.

| Quantity | Enhance vs FP32 oracle relative L2 |
| --- | ---: |
| Compressed output | 0.00134942 |
| q_nope gradient | 0.01267673 |
| Compressed K/V gradient | 0.00170272 |
| q_rope gradient | 0.01279705 |
| k_rope gradient | 0.01372216 |
| KL loss | 0.00024537 |
| Index Q gradient | 0.00334042 |
| Index K gradient | 0.00303772 |
| Merge weight gradient | 0.00238999 |

A second device run used stock `torch.ops.npu` indexer, SFA, explicit SFA grad
and KL primitives from the installed CANN 9.1 runtime. Stock and enhance Top-K
sets matched and covered the complete legal history. For this fixture, each
enhance quantity passed bounds derived from stock-vs-oracle errors: relative L2
at most `max(1.25 * stock_error, 1e-4)`, max absolute error at most
`max(1.25 * stock_error, 1e-12)`, and cosine at least `stock_cosine - 1e-6`.
LSE also passed. The largest enhance attention-gradient relative L2 was 0.01372,
with max absolute error 2.41e-9; stock k_rope relative L2 was 0.01935.

This is acceptance of one complete-history fixture, calibrated to these stock
primitives. It is not a general BF16 tolerance, long-context Top-K equivalence,
CP>1 result, parameter-gradient acceptance or performance measurement. The
standalone validator still reports measurements pending calibration because it
does not perform the separate stock comparison itself.

### Selected-set matrix and auxiliary scaling

The original unrestricted matrix ran five fixed selections on the same packed
BF16 shape, comparing enhance and stock primitives independently with the FP32
oracle. Its failures below remain evidence of the native limitation. The
current `mega_dsa_cann_matrix.py` verifies pre-dispatch rejection for the four
unsupported cases and numerical acceptance for supported CPU-prepared and
native-indexer selections. It uses the per-fixture calibration rule above and rejects a stock calibration
whose relative L2 error is at least one (error as large as the reference signal).
This sanity condition prevents mutually incorrect native results from providing
a permissive error bound. Empty output/query derivatives/sum must be exactly
zero, and empty LSE must be `-inf`. Native maximum values are also recorded,
without equating their opaque representation with the oracle maximum.

| Fixture | Device result |
| --- | --- |
| Complete legal history per query | Passed all attention/indexer gradient and loss checks |
| Only current and sequence-start keys | Failed backward/KL oracle acceptance |
| All slots `-1` | Forward zero; backward/KL nonfinite; native sum incorrectly one |
| Empty first query of each packed sequence | Failed; nonfinite backward/KL |
| Leading holes plus known future/cross-sequence IDs | Compacted forward correct; same partial-set backward/KL failure |

The partial-set output relative L2 was 0.00142, but KL was approximately 0.18357
instead of the oracle's 0.00001685. Enhance main-query gradient relative L2 was
16.66 and merge-weight gradient relative L2 was 80.67. Stock reproduced the
large errors and empty-row failures. These are unsupported native candidate
counts, not ordinary BF16 rounding.

The locked native SFA backward computes `actualSelectedBlockCount` from
`min(selectedBlockCount, causal block count)`, without counting non-padding
indices. KL likewise sets `s2RealSize = min(kSize, s2SparseLen)`; when the causal
prefix fits K, `mergeKv` is false and it reads dense key storage. Its required
selection therefore contains exactly `min(K, causal prefix length)` unique
legal keys per row. Stable compaction does not satisfy this precondition for
an underfilled or empty row. The development adapter now enforces this
precondition through CPU preparation or native-indexer provenance. General
selected sets and empty rows are rejected; supporting them on device would
require a separate native change. The native limitation remains unchanged.

The same device matrix reuses the existing `aux_loss_auto_scale` and
`set_aux_loss_scale`. Complete-history checks pass for `grad_aux=1/7` with
`loss_coeff=0.3`, and `grad_aux=7` with `loss_coeff=0`. Forward values and all four
main gradients remain unchanged; only the three indexer gradients receive the
declared auxiliary multiplier, independently of the main objective multiplier
of 13. This validates the helper/bridge composition for one fixture, not an
end-to-end trainer or DP/PP accumulation trajectory.

Run the matrix with the same explicitly activated OPP and device idle gate:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 \
NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python hyper_parallel/core/multicore/examples/mega_dsa_cann_matrix.py \
  --output /tmp/mega_dsa_cann_matrix.json
```

Expected unsupported counts are reported as `rejected_before_dispatch`, with
`native_invoked=false`; the four expected rejections are contract passes, not
device numerical acceptance. The whole report sets `reference_contract_passed`
and `status=contract_verified` only when all expected rejections, supported
numerical checks and auxiliary-scaling checks pass. Unexpected admission,
supported numerical failure or an execution error produces a nonzero exit and
retains stage/traceback. It does not rerun unsupported raw native calls; the
earlier failure reports remain preserved separately.

### Offline index traces

`profile_index_trace` accepts an explicitly captured CPU int32 `[Tq,K]` snapshot
and `DsaBatchMeta`. For query tiles of 16/32/64/128 it records distinct keys,
owner reference/distinct histograms, owner-local contiguous storage runs,
adjacent-tile and invocation-wide reuse, deduplicated read counts, remote
fractions and payload byte estimates. Ownership uses ordered CP indices rather
than WORLD ranks, and query tiles follow actual Q storage order. Packed/future
slots are masked; unknown and duplicate IDs fail. Missing local KV is allowed
because final global selections may name remote tokens.

The device matrix profiles a synthetic enhance-indexer snapshot. It is not a
real model trace. The function never transfers device tensors implicitly;
the caller must request capture outside the hot path. Reuse counts describe
opportunities, not measured cache hits. Byte estimates assume one fetch per
distinct key per tile; actual transport requests, allocation peaks and buffer
lifecycles need separate runtime instrumentation.

### Parameter-preserving HP model boundary

`CannDsaReferenceAttention` replaces an already constructed
`DeepseekV32DSAAttention` through the existing declarative `replace_module`
mechanism. It retains the original Parameters, child modules, state dict,
training state and module aliases. Its constructor does not repeat weight
fusion, and `make_transforms()` returns no additional checkpoint conversion.
The inherited input projections, norms, RoPE, key absorption, value
restoration and output projection remain the model implementation. Only the
indexer, sparse attention and selected-KL boundary uses the prepared reference.

The replacement requires TP=CP=1 and K=2048, C=512, Dr=64. Each forward must
receive `dsa_reference=prepared_backend`, with the original attention scale and
matching layer metadata. Prepared metadata defines packed causal boundaries;
optional `actual_seq_len` must agree with it. Passing the prepared
`layout.length_tensor` is checked by identity without reading device values.
Alternate packed-length aliases are rejected. There is no automatic backend
construction or fallback. The caller continues to set the existing auxiliary
loss scale, and `aux_loss_auto_scale` attaches KL exactly once under the
original training/freeze/coefficient gates.

This targets the existing HP attention class, rather than a Transformers
module directly. It does not yet supply the prepared backend through an
end-to-end trainer invocation. The validation fixture constructs the real HP
class with random smaller projection dimensions; it is not a pretrained
DeepSeek-V3.2 model or a full-model training result.

CPU FP64 tests compare the absorbed boundary with independently expanded
per-head K/V dense MLA over complete causal histories. Output, hidden-state
gradients and all 11 Parameters align, including both slices of the key/value
up-projection. Separate tests cover replacement identity/state dict/aliases,
KL-only isolation, freeze/zero-coefficient/evaluation gates, non-reentrant
checkpoint recomputation and unequal 3/5-token microbatch accumulation against
the global packed mean. The accumulation test is local; distributed DP/PP and
AMP behavior remain separate acceptance work. CPU tests explicitly select an
oracle backend and mock only the optional NPU RoPE primitive.

### BF16 model calibration and held-out failures

`mega_dsa_model_validate.py` measures the actual HP boundary on one NPU with
packed sequence lengths 3 and 5. It compares output, loss, hidden gradients,
all 11 Parameters and the separate W_UK/W_UV slices against unabsorbed FP32 MLA
using identical BF16-quantized weights and inputs. The stock SFA/KL comparison
uses the same measured enhance selection. This isolates the selected operator
calibration; it does not establish long-context Top-K or tie equivalence.

The validator runs seven modes per seed: joint objectives with `grad_aux=1/7`,
KL-only with `grad_aux=7`, zero coefficient, freeze, evaluation and
non-reentrant checkpoint. Gradient availability must match the oracle exactly.
The main objective multiplier is 13; auxiliary scaling is controlled separately.

Model calibration uses relative L2, maximum absolute error and normalized
angular distance `sqrt(2 * (1 - cosine))`. The angular bound uses the same
1.25 multiplier as L2, with a 1e-4 floor; max-absolute error has a 1e-12 floor.
Three calibration seeds (20261007/17/18) determine per-quantity, per-objective
maximum stock-reference error bounds. The bounds are frozen and hashed before
two held-out seeds (20261027/28). Candidate measurements cannot enlarge the
bounds; both stock and enhance must satisfy the frozen bounds. Pointwise
stock comparisons are also retained in the report.

The recorded CANN 9.1 model run **failed acceptance: 22 of 35 cases passed,
13 failed**. All 21 calibration cases passed the frozen envelope. The first
held-out seed passed KL-only but failed hidden-gradient max-absolute bounds in
the six main-objective modes. The second held-out seed failed all seven modes,
with stock itself exceeding some frozen bounds. Gradient availability and
the expected isolation pattern passed in all 35 cases. These results do not
establish model parameter-gradient acceptance or a general BF16 tolerance.

Read-only follow-up diagnosis found one indexer ReLU sign crossing between
FP32 dot products on device BF16-projected states and full FP32 model-projected
states in the second held-out fixture. Its `indexer.wq_b.weight` gradients
matched stock exactly while differing from the FP32 model oracle. This observation
identifies projection/precision sensitivity; it is not a kernel-internal
activation trace or proof that every failed quantity has the same cause.
Earlier cosine-margin and pointwise max-absolute failures remain separate
evidence. The native source and payload were not changed for this model round.

Run the model validator with the same activated OPP and device idle gate:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 \
NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python hyper_parallel/core/multicore/examples/mega_dsa_model_validate.py \
  --output /tmp/mega_dsa_model_validation.json
```

The validator preserves measurements and returns a nonzero exit on failed
acceptance. Its captured selections come from real HP projections of random
fixtures, so their offline traces are not trained-model locality evidence.

### Layer diagnosis with identical inputs and cotangents

`mega_dsa_layer_diagnose.py` separates two sources of model error without
changing the frozen model acceptance bounds. It captures the real model's
seven projected states, selection and sparse-output cotangent. Enhance, stock
and the CPU FP32 primitive oracle then receive identical projected values,
selected keys and output cotangent. Selected KL uses the same detached teacher
inputs, coefficient 0.3 and auxiliary multiplier 7. This removes different
downstream model errors from the primitive comparison.

A separate comparison fixes the captured state cotangents and measures the
main/index projection VJPs through the real BF16 NPU model and FP32 CPU model.
These measurements include projection quantization and backward rounding;
they are not isolated GEMM or norm-kernel acceptance. CPU absorbed and
unabsorbed FP32 formulas are also compared. The diagnostic's staged execution
is checked against the original model boundary before interpreting metrics.
It does not install capture hooks in production model calls.

For the two previously failing held-out seeds, staged execution exactly
matched the original NPU boundary's output, loss, hidden and all 11 parameter
gradients. Both identical-state primitive comparisons passed pointwise stock
calibration. Their maximum enhance input-gradient relative L2 was 0.00352.
Fixed-cotangent projection VJP relative L2 ranged from 0.00184 to 0.00359,
while the CPU absorbed/unabsorbed model comparison reached approximately
1.12e-6. The second seed's indexer ReLU sign crossing was reproduced.

An offline sensitivity intervention held the FP32 projection Jacobian fixed
and forced the indexer ReLU branch mask predicted by FP32 dots on the captured
BF16 states. For the second seed, `wq_b.weight` relative L2 between the mapped
native cotangent and the CPU KL reference fell from 0.08423 to 0.00966. The
first seed had no crossing and its error stayed 0.00836. This is a
counterfactual diagnostic of branch sensitivity, not a production mask policy
or a replacement KL objective. It does not explain the first seed's hidden
gradient failure or eliminate all second-seed error.

These observations support further investigation of projection precision and
composition sensitivity in these fixtures. They do not prove every failed
model quantity has the same cause, establish a general primitive tolerance,
or turn the original 22/35 model result into acceptance. Diagnostic reports
use `status=diagnosis_measured` and keep `model_acceptance_changed=false` and
`p0_complete=false`; metric failures remain in the report.

Use the same activated OPP and idle gate to run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 \
NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python hyper_parallel/core/multicore/examples/mega_dsa_layer_diagnose.py \
  --output /tmp/mega_dsa_layer_diagnosis.json \
  --snapshot-dir /tmp/mega_dsa_layer_snapshots
```

The optional snapshot directory stores explicit offline CPU captures of random
fixture weights, inputs, projected states, cotangents and selection, with
SHA256 identities in the report. The default seeds are 20261027 and 20261028;
`--seeds` selects other diagnostic fixtures. Neither these short histories nor
their snapshots provide trained-model locality or long-context Top-K evidence.

### Output restoration and hidden-gradient controls

The layer diagnostic also captures sparse output, restored per-head values,
their cotangents and the final model-output cotangent. Restoration snapshots
remain detached CPU values and retain the original quantized model weights.
`mega_dsa_restore_diagnose.py` replays value up-projection, `o_proj`, and their
composition separately with identical captured inputs and fixed cotangents.
It records BF16 NPU versus FP32 CPU measurements and gradient availability.
The combined replay is compared with the original sparse cotangent, direct
W_UV gradient and `o_proj` gradient. Its W_UK slice must remain zero, since
key absorption belongs to the main projection graph rather than restoration.

Hidden-gradient controls then use an FP32 projection Jacobian with captured
native cotangents, FP32 sparse VJPs, FP32 restoration VJPs, and a complete FP32
downstream computation on the captured BF16 projected states. Each control is
measured against the full FP32 absorbed model's hidden gradient. The controls
change several precision boundaries and combine states/Jacobians from
different paths; their errors are not additive component attribution or
gradients of a proposed production model. KL inputs remain detached from the
hidden path. These controls do not change model calibration or acceptance.

CPU tests independently expand the value formula per head, check restoration
parameter reachability and the zero key-gradient slice, and verify separate
restoration VJPs compose correctly. With all stages in FP32, the hidden controls
must recover the same full-model hidden gradient. Snapshots additionally store
the actual hidden gradient and restoration capture for offline replay.

The CANN 9.1 restoration run used the two existing holdouts and a new
independent seed, 20261038. All three staged model captures and combined
restoration VJPs matched their original boundary measurements exactly, including
the sparse cotangent, W_UV and `o_proj` gradients. The restoration W_UK slice
was exactly zero. Identical-state SFA/KL comparisons passed pointwise stock
calibration for all three seeds. These are diagnostic consistency checks, not
frozen model acceptance for the new seed.

With fixed identical inputs and cotangents, value/output input-VJP relative L2
was approximately 0.00163--0.00167, and the combined restoration input VJP was
0.00233--0.00234. The full native hidden gradient and the FP32 downstream
control on BF16 projected states had the following relative L2 against the
FP32 absorbed hidden gradient:

| Seed | Native full hidden | FP32 downstream on BF16 projected states |
| --- | --- | --- |
| 20261027 | 0.00551 | 0.00223 |
| 20261028 | 0.00464 | 0.00245 |
| 20261038 | 0.00455 | 0.00256 |

Intermediate controls did not monotonically reduce maximum absolute error.
These results support composition/rounding sensitivity across several
boundaries; they do not identify one defective operator or justify dropping
the max-absolute criterion. The original 22/35 model result remains failed.

### Long-history selection and exact cutoff ties

`mega_dsa_indexer_validate.py` runs complete CP=TP=1 native query/key storage
with packed lengths 2176 and 2240, Hi=8, Di=128 and K=2048. For every query it
checks the unfiltered raw native indices for legal sequence-local IDs, exact
native cardinality, uniqueness and trailing padding. Checking before global-ID
conversion prevents an invalid native ID from disappearing into a masked slot.
It compares all-row enhance/stock selected sets and an enhance repeat run.

The increasing and signed-decreasing fixtures encode integer ranks with two
BF16 coordinates so their FP32 dot products stay distinct above K. Their
analytic winners are checked for every row. A concentrated fixture puts all
K strictly preferred winners in the first K positions of each sequence;
its all-row winner set must be complete. Its prefix partition is hypothetical,
not an executed CP owner or a distributed communication test. The signed case
requires negative merge weights to affect ordering without extra activation.

The all-zero fixture uses a predeclared tie rule: any legal cutoff-tied subset
is allowed, but every strictly better candidate is mandatory. Raw cardinality
and uniqueness remain strict. FP32 certificates and stable global-ID oracle
matches are reported separately. Random-input set differences require a
structural ReLU-zero certificate for every exchanged ID: both selections must
contain every strictly better candidate at cutoff zero, and every exchanged
key's head dots must remain strictly negative under a conservative FP32
accumulation error bound. Nonzero/near-cutoff differences are not admitted by
a tolerance. At most 32 extra mismatch queries are evaluated; uncertified
differences fail the gate. CPU score certificates use full key candidates for
20 base queries plus those extra queries, including packed starts and prefixes
just below/at/above K.

The first CANN 9.1 run **reported four accepted cases and random_signed failed**
because random-input sets were compared strictly without tie diagnosis. This
original report and source remain preserved. All raw native checks passed for every row
of every case. The three analytic fixtures matched all-row winners, stock and
repeat results. The all-zero fixture had 320 enhance/stock set differences,
all in histories longer than K, while both sets satisfied exact tie membership.
The enhance repeat was identical, and its sampled sets matched the ascending
global-ID CPU oracle. These observations do not guarantee that tie policy for
other shapes or versions.

The random fixture had one enhance/stock set difference at global query 2070;
the repeat enhance result was identical. The original 20 sampled FP32
certificates passed, so the all-row stock comparison caught a difference outside
the oracle sample. Follow-up diagnosis found 2044 strictly better candidates
and nine zero-cutoff candidates competing for four slots. Both selections
contained all strictly better candidates; they exchanged four zero-score IDs.
Their FP32/FP64 scores were exactly zero. Native returned scores were also
zero, and enabling score return retained each backend's selected set. Returned
BF16 equality alone would not prove an internal FP32 tie.

The structural certificate checks the exchanged keys' FP64 head dots using
the standard FP32 accumulation bound `gamma_n * sum(abs(products))`, with
`gamma_n = n*u/(1-n*u)` and `u=2^-24`. Their minimum negative margin after
this bound was approximately 0.000981; all head ReLUs are inactive. The report
separates strict set agreement from this exact zero-tie equivalence, applies
the same tie semantics to random inputs, and retains the first failed gate.
This is conditional on FP32 dot accumulation, as used by the locked indexer;
it does not establish equivalence for arbitrary floating-point near ties.

The final run reports `selection_verified_with_ties`: all five membership
cases passed, while strict enhance/stock set equality passed only the three
analytic cases. All raw native hashes were identical to the initial run; the
classification changed to account for the proven random ReLU-zero tie, not
because the operator or data changed. Every enhance repeat was identical.
The original BF16 model acceptance remains 22/35 and failed.

Run with the same activated payload and idle gate:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 \
NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python hyper_parallel/core/multicore/examples/mega_dsa_indexer_validate.py \
  --output /tmp/mega_dsa_indexer_validation.json \
  --snapshot-dir /tmp/mega_dsa_indexer_snapshots
```

Optional snapshots store CPU inputs and sampled global selections; full raw
native selections are hashed in the report. Failed selection or execution
returns nonzero and preserves the report. This validates indexer selection
only: long-history SFA/KL derivatives, distributed CP/TP, trained-model locality
and performance remain separate work.

The candidate-count boundary remains restricted; general underfilled/empty
device semantics remain unavailable. P0 still needs model BF16 calibration
that generalizes to held-out inputs, parameter-gradient device acceptance,
random long-context selection agreement and long-history SFA/KL derivatives,
end-to-end trainer backend plumbing and
DP/PP/AMP accumulation validation, and trained-model index traces with memory
lifecycle measurements. Follow those gates before SHMEM coexistence (P1) or
mixed-team extraction (P2).

## Explicit CP correctness baseline

[`cp_reference.py`](cp_reference.py) adds `DsaCpLayout` and
`CannDsaCpReference`. This is an explicitly selected development baseline for
CP=1/2/4, independent of the planned native fused worker. It gathers complete
Q **and** KV because the installed right-down CANN mask cannot represent
partial Q's global query offsets. Each rank evaluates full, globally ordered
packed attention/indexer/KL and selects its original local query rows. Local
sequence lengths are never substituted for global causal positions.

`DsaCpLayout` validates ordered process-group membership, global metadata,
invocation identity and complete owner-local Q/KV coverage during collective
construction. Local Q and KV may have different storage orders; uneven shards
use padding which is removed by a prepared global permutation. Host metadata
and device permutations are prepared outside execution. Communication uses
one packed differentiable all-gather for the declared main/indexer fields and
one reduce-scatter in backward. Remote gradient contributions are summed in
FP32 before conversion to the owner input dtype; this cannot recover rounding
already performed inside each native BF16 backward. FP64 is preserved for the
CPU exchange gradcheck.

The multi-output autograd boundary preserves absent main teacher gradients
in KL-only backward. Shared compressed K/V still receives the native dK+dV
sum once. Full-query KL is replicated on every rank, so each returned KL
contribution is divided by CP size. Sum **detached** contributions for reporting;
backpropagate each rank's own scalar. `DsaLossNormalization` retains its explicit
downstream reducer divisor. Do not insert a differentiable scalar all-reduce
or apply an additional auxiliary scale inside this boundary. Projection-input
detach and the model's original auxiliary gradient scale remain caller-owned.

All ranks must use the same prepared field schema, selection, loss coefficient,
normalization, requires-grad masks and forward/backward schedule. Construction
performs metadata object collectives; execution does not read tensor contents
on the host. External selections require identical complete CPU snapshots on
all ranks, prepared outside execution; generated selections use the native
indexer. Existing cardinality restrictions apply to both paths.

After activating the validated enhance payload and editable checkout, run the
validator through the idle gate (example for four cards):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPU_WAIT_VISIBLE_DEVICES=0,1,2,3 \
NPU_WAIT_NUM_CARDS=4 NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m hyper_parallel.core.multicore.examples.mega_dsa_cp_validate \
  --output-dir /tmp/mega_dsa_cp4
```

The [validator](../../examples/mega_dsa_cp_validate.py) writes per-rank evidence
for short packed lengths `(3,10)`: contiguous, strided and zigzag owners,
independent query/key orders, all seven input gradients, native TopK and
prepared external selections, KL-only gradient absence, zero coefficient,
non-reentrant checkpoint and retained-graph backward. Retained backward checks
each incoming gradient against the first using the declared tolerance, then
checks exact accumulation of the two actual incoming gradients; it does not
assume bitwise repeatability of native SFA reductions. CP parity uses declared
pointwise `rtol=0.02, atol=2e-5` against identical-state unsharded native execution.
CPU FP32 comparisons are measurements, not a new full-model acceptance gate.
The lightweight [ST launcher](../../../../../tests/torch/multicore/test_mega_dsa_cp.py)
contains CP=2/4 cases; it requires the caller's activated enhance payload.

This baseline does not implement SHMEM, selected-key communication, mixed
AIC/AIV worker groups, fused SFA/indexer/KL launch, model CP/TP integration or a
performance optimization. It does not change the original BF16 model's failed
22/35 acceptance or complete P0/P1/P2/P3. The next native work must establish
shared-root byte budgets and leases, then extract callable mixed-worker tiles
with explicit global causal offsets and verify full backward against this CP
baseline. Long-history sparse SFA/KL gradients and complete training steps
remain separate validation gates.

## Shared-root resource implementation

[Shared SHMEM root](../../docs/shared_shmem_root.md) adds frozen byte budgets,
consumer-owned allocation accounting, a root-wide serial lease and monotonic
heap generation. `MegaMoeExperts.bind_shmem_root` preserves the MoE-specific
buffer layout; [workspace.py](workspace.py) adds DSA owner publication, staging,
FP32 peer-exclusive gradient stripes and isolated events in one symmetric arena.
Declare all consumers before binding, use identical ordered CP/EP/root members,
and close every bound consumer before closing the root. Growth-capable push
configurations are rejected during planning; pull and full-bound push are fixed.

The workspace supplies storage for future native tiles. Publication is detached
and enqueue-only; ready/ACK, gradient routing and native DSA computation remain
separate implementation work. Caller-owned saved activations must be republished
under a fresh backward lease rather than retained as views of reusable scratch.

## P2 mixed SFA probe

The experimental [mixed schedule](mixed_tile.py) invokes the locked 910B SFA
implementation inside one persistent `HyperDsaMixedTile` launch. A physical
team has one AIC and its two AIV partners. Its cube publishes a logical
partition ticket with a local pipeline barrier before cross-core flag 13.
Both vectors invalidate the ticket cache line and consume a volatile GM load
after the flag. Flag 14
joins both vectors after tile completion before the cube reuses its scratch.
The upstream SFA reserves flags 3 through 12 for its internal pipeline; the
wrapper uses separate IDs and separate member cache lines.

Logical query partition IDs are independent of physical scratch group IDs.
Each tile creates and destroys its own `TPipe`; the TND adapter removes global
output initialization and its `SyncAll`. The adapter retains the locked SFA
arithmetic and host tiling. The ACLNN probe bridge uses L0 Contiguous and
ViewCopy to restore logical input/output views from flat Torch storage; these
extra transfers are outside any performance claim. It is hash-verified and applied to an exported
source copy; dependency checkouts remain unchanged.

This probe supports BF16, TND, CP1, C=512, Dr=64, H=32/64, Nkv=1, K=2048,
complete ordered packed Q/K, external complete causal selections, and 910B3.
Prepare the selection using the P0 reference contract. Device-produced
selection correctness remains a separate gate. Mutable output/stat/trace
buffers must have independent storage. Inputs with gradients are rejected;
this forward probe has no autograd backend. Invalid device configuration
leaves the NaN-initialized probe outputs and member records incomplete rather
than entering a mismatched team wait.

The compute group count is 1 through 19; at least one physical group remains
reserved. Reserved groups currently exit: no communication progress worker is
implemented. Repeated rounds overwrite the same output and stress local
scratch and event reuse; they are not multiple training steps.

After activating the native payload as described in [build](../../docs/build.md), run:

```bash
python -m hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate \
  --output mixed_sfa_report.json --long-history
```

The validator covers 1/2/7/19 groups, one/three traversals, H32/H64, packed
(3,10)/(19,29,33), and optional length 513 crossing the sparse KV tile boundary.
It compares output/max/sum to stock native CP1 at rtol=0.02, atol=2e-5 and
output/LSE to the CPU FP32 oracle at rtol=0.02, atol=0.002. Every group member
must independently report the declared count, checksum and final logical ID;
reserved groups must have no compute evidence. Outputs start as NaN to expose
incomplete partition coverage.

P2 remains incomplete until LI, complete backward, alternating task types and
communication pressure pass. This probe does not establish multi-card fused
DSA, model CP integration, training acceptance or performance improvement.
