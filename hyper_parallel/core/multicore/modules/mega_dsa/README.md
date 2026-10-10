# megaDSA development baseline

This package follows the 2026-10-07 megaDSA plan. It includes logical metadata,
CPU numerical oracles, single-card and CP CANN reference boundaries, a frozen
shared-root DSA workspace, and experimental callable mixed SFA/LI forward
probes. A production `MegaDsaCore` fused training backend and automatic
fallback policy remain unimplemented.

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

The original locked native SFA backward computes `actualSelectedBlockCount` from
`min(selectedBlockCount, causal block count)`, without counting non-padding
indices. KL likewise sets `s2RealSize = min(kSize, s2SparseLen)`; when the causal
prefix fits K, `mergeKv` is false and it reads dense key storage. Its required
selection therefore contains exactly `min(K, causal prefix length)` unique
legal keys per row. Stable compaction does not satisfy this precondition for
an underfilled or empty row. The development adapter now enforces this
precondition through CPU preparation or native-indexer provenance. General
selected sets and empty rows are rejected; supporting them on device would
require separate native changes. The external CP main-attention adapter below now contains a selected-count
correction; this does not change the KL/indexer selection restriction.

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

P2 remains incomplete until complete backward, alternating task types and
communication pressure pass. This probe does not establish multi-card fused
DSA, model CP integration, training acceptance or performance improvement.

## P2 mixed SFA backward probe

The experimental [training probe](mixed_attention.py) combines the callable
SFA forward with a [locked gradient adapter](../../ops/sparse_flash_attention_grad_mixed.patch).
It runs initialization, non-deterministic computation and output conversion
as three kernels on one stream. Initialization completes the global FP32
gradient zeroing before any compute group starts accumulation. Output
conversion starts after all compute groups finish. No reserved physical
group participates in a full-chip barrier.

Logical query partitions retain their original ownership; physical groups
own reusable MM and scatter scratch. The wrapper reserves cross-core flags
13/14, separately from the gradient tile's flags 0 through 8. Each phase
records all three group members' completion. The retained workspace is
ordinary HBM with a conservative bound checked against host tiling; it is
not a shared-root allocation or a communication buffer.

The raw interface returns separate K and V gradients. The autograd interface
adds those contributions once for the shared compressed K/V input. Each
forward saves independent output and softmax statistics and versioned input
tensors; each backward obtains fresh accumulation scratch. `retain_graph`
therefore repeats a complete three-phase backward. Higher derivatives,
deterministic algorithms, model integration and SHMEM communication remain
outside this probe's contract.

Support matches the forward probe's CP1 BF16 geometry, with one logical
traversal per phase. After [building and activating the payload](../../docs/build.md), run:

```bash
python -m hyper_parallel.core.multicore.examples.mega_dsa_mixed_grad_validate \
  --output mixed_sfa_grad_report.json --long-history
```

The validator checks all five raw gradients against stock CANN and all four
merged input gradients against an independent CPU FP32 oracle at
rtol=0.02, atol=2e-5. It retains every numerical comparison, including stock
CANN against FP32, before reporting an acceptance failure. It also checks
forward output, custom against stock autograd with shared K/V, repeated
backward and all three phase traces. Numerical summaries accumulate relative
L2 and cosine in CPU FP64; elementwise acceptance retains the declared
FP32 comparison thresholds. Native contract checks and FP32 acceptance are
reported separately, with the complete report failing if either fails.
Each fixture also holds two different forwards until reverse-order backward
on a second stream, with explicit stream waits, and runs non-reentrant
checkpoint recomputation. These checks compare against independent stock
invocations and do not establish shared-root or concurrent stream support.
Long history includes 4096 tokens to exercise the optimized scatter path;
only three query rows have nonzero cotangents in that fixture to bound the
independent CPU graph. Short fixtures use nonzero cotangents on every row.
Passing native-to-native comparisons alone does not establish FP32 gradient
acceptance or complete P2.

## P2 mixed LI probe

The experimental [LI adapter](mixed_indexer.py) preserves the locked 910B
weighted ReLU, signed merge-weight and exact global Top-K calculations. The
[mixed LI patch](../../ops/lightning_indexer_mixed.patch) introduces explicit
logical partition IDs and physical scratch IDs. Cube and both vectors reuse
physical MM double buffers; each logical partition retains its partial Top-K
lists and LD parameters in a separate, caller-owned uint8 arena. At H64/K2048,
the pinned arena budget is 47,226,880 bytes (45 MiB plus 40 KiB). Dependency
checkouts remain unchanged; the lock verifies the patch before source assembly.

Main and LD merge execute as two kernels on the same stream. The main kernel
publishes every logical partition's partials before the merge kernel runs.
Merge initialization preserves those partials and parameters. This replaces
the original full-chip `SyncAll()` boundary without requiring idle physical
groups to join a barrier. It is a host-scheduled completion boundary; the two
phases are not a single fused kernel or a communication progress worker.

Initial support is 910B3, BF16 query/key, BF16 or FP32 already-scaled weights,
H_index=64, D_index=128, Nkv=1, K=2048, full ordered CP1 packed TND with positive
sequence lengths and causal mode 3. Prepare cumulative lengths through the
P0 `CannDsaLayout` contract. The native boundary checks shape, dtype,
contiguity, workspace capacity and independent mutable storage. All inputs
must be detached. The probe has no autograd path. Each call retains independent
indices, values and phase traces; optional scratch reuse is serial on one
stream, or requires caller-established cross-stream events.

After [building and activating the payload](../../docs/build.md), run:

```bash
python -m hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate \
  --output mixed_li_report.json --long-history
```

The validator runs 1/2/7/19 groups and repeats each complete main/merge pair
three times. It covers short packed sequences and optional packed lengths
2176/2240 with increasing, signed decreasing, concentrated, all-tied and random
signed scores. It checks every index row for causal legality, uniqueness,
full cardinality and trailing padding; compares indices including order and
BF16 values bitwise to stock CP1; and checks all-row analytic winners for the
ordered/concentrated fixtures. Both phases must independently certify Cube
and both Vector task completion and agree on logical LD ownership. Long
history must execute at least one LD merge. There is no relaxed tie threshold.

P2 still requires complete backward, alternating callable task types and
communication pressure validation. CP fused execution and full training
acceptance remain separate from this single-card forward probe.

### LI device phase closure

The ABI 2 [fused LI probe](mixed_indexer.py) runs main and LD merge in one
native launch. Each compute group's Cube and both Vectors complete their DMA
and publish an arrival in their own cache line. One Vector in the immediately
following reserved group polls all arrivals and publishes the phase release.
Merge consumes the retained logical partials only after this release. The
reserved group's Cube and second Vector wait for release, keeping the full
mixed team resident through phase closure; other reserved groups remain
idle. Release belongs to the progress Vector's own cache line. This provides
a device completion boundary for a future LI→SFA
task chain. It does not transfer data between ranks.

Each invocation owns freshly zeroed int64 phase records `[2,20,64]`, even
when its retained scratch is reused. Main uses epoch 1 and merge epoch 2 in
separate records; no persistent event array is reused across invocations.
The validator requires all three members' arrival, the progress Vector's
exact release epoch and arrival count, its entry and all reserved members'
exit records, the existing task/LD evidence, and
zero compute evidence from every other reserved group. Old payloads reject
the fused probe before allocation. The existing two-launch probe accepts
ABI 1 or 2 and remains the host reference path.

```bash
python -m hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate \
  --output fused_li_report.json --long-history --fused
```

This runs the same stock bitwise, signed-weight, tie, all-row contract and
long-history LD checks as the two-launch validator, across 1/2/7/19 groups
and three complete invocations with serial scratch reuse. Cross-operator
alternation and native CP communication pressure are exercised by the fused
forward probes below. Fused backward remains a separate validation gate.


### LI main/merge and SFA in one kernel

The [fused forward probe](fused_forward.py) executes the complete chain in
one native `HyperDsaFusedForward` mixed AICore kernel. The same compute groups
traverse 20 logical LI partitions, merge their retained Top-K partials, then
traverse 20 logical SFA partitions. SFA consumes the final sequence-local
indices directly. There is no host launch between these phases. One reserved
Vector releases each phase after the Cube and both Vectors in every compute
group have completed their work and published arrival. Its whole mixed team
remains resident through all three closures.

The host adapter uses CANN's `OpTilingContextBuilder` to invoke both existing
locked tilers with their original IR input and attribute order. The source
assembler exports their original tiling declarations for CANN to generate
the composite device structure; math, selection order and workspace budgets
remain those of the existing callable tiles. The fused operator registers
weight-dtype templates with and without CP transport. The optional block-table input is
absent in both child contexts. Lengths describe complete positive CP1 packed
sequences; the probe does not accept sharded Q/KV.

Support is BF16 absorbed Q/KV with H32/H64, C512, RoPE64, LI H64/D128,
BF16 or FP32 already-scaled indexer weights, causal TND and K2048 on 910B3.
Compressed KV is shared between SFA K and V. The native boundary validates
shapes, dtypes, contiguous single-device storage and independent mutable
buffers. The raw probe requires detached states and preserves the supplied
attention scale. It returns owned indices, values, attention and FP32
maximum/sum. Optional LI scratch can be reused only after a complete
invocation on the same stream or with explicit event dependencies.

Each invocation owns fresh zeroed trace storage `[3,20,64]`. The CPU trace
decoder checks epochs 1/2/3, every compute member's arrival and logical tasks,
the reserved team's entry/release/exit and exact arrival count, matching LI
LD ownership across main/merge, and no stale LI evidence in SFA. The caller
must explicitly take CPU snapshots; the forward path never transfers traces
to the host.

```bash
python -m hyper_parallel.core.multicore.examples.mega_dsa_fused_forward_validate \
  --output fused_dsa_forward_report.json --long-history
```

The validator covers H32/H64 and 1/2/7/19 compute groups, repeats each complete
invocation three times with serial LI scratch reuse, and checks every output
against a separately executed stock LI→SFA chain with zero numerical
tolerance. Packed lengths 2176/2240 additionally require nonzero logical LD
merge participation. Signed weights, all ties, concentrated scores and random
BF16/FP32 weights retain the indexer's analytic and all-row legality checks.

The 910B3/CANN 9.1 validation covered 504 complete fused invocations across
short and long matrices, including a fresh-process replay through the ST
launcher. All five outputs matched the independent stock chain with zero
numerical tolerance; every long invocation executed nine logical LD merges.
Existing LI/SFA forward regressions passed. The separate backward regression
retained its incomplete FP32 elementwise acceptance: 36/64 autograd checks
passed at rtol=0.02, atol=2e-5, matching the stock shared-KV result. Native
stock-gradient, retained-graph and lifecycle checks passed, but the backward
ST still returns failure. The fusion gate does not clear that failure.

This is a CP1 forward and cross-operator scheduling gate. The existing SFA
backward probe remains separately scheduled. The CP transport gate is described
below. Owner gradient return, selected-set KL, fused backward and full-model training
acceptance remains required before claiming a complete MegaDSA backend.


### Native CP KV pull inside fused LI/SFA forward

The experimental [CP forward probe](fused_cp.py) adds real symmetric-memory
KV transport to the same mixed kernel. It requires adapter ABI 2, one node,
MTE-reachable peers, identical ordered CP/root membership and direct root PE
mapping. Geometry and math support match the CP1 fused probe. Preparation
validates global ownership and allows different, reordered Q/K storage,
uneven shards and owners with no rows.

Each invocation replicates only query-side fields through the existing CP
gather. Index-K, shared compressed KV and K-RoPE are published into the frozen
workspace in canonical owner-offset order. The reserved progress Vector
publishes READY and verifies every peer's generation, layer, microbatch,
invocation and layout signature. It pulls the complete global index-K into
ordinary invocation-owned HBM before compute begins. Once every compute
member has entered LI, it pulls compressed KV and K-RoPE while LI runs.
All transfers complete before the LI-main phase releases. SFA consumes these
owned buffers and final indices without a host launch between phases.

The progress worker uses private 8 KiB UB staging with explicit MTE completion
events; it does not consume a compute worker's tile storage. READY and
completed-read ACK occupy separate 128-byte cache lines. Each reader publishes
ACK only after all owner fields have been copied, and each owner waits for
every reader before kernel completion permits source reuse. The epoch belongs
to the workspace lifetime, survives adapter recreation and can be reserved
only inside the invocation lease. The lease covers query collectives,
publication, device execution and output restoration; root completion events
order reuse across streams.

The result owns local-Q outputs, global packed indices, all three global key
buffers, three phase traces and a separate transport trace. Explicit CPU
decoding checks exact epochs, ACK counts, local/remote byte counts and all
compute startup records. Long-history validation requires a strict intersection
between a completed LI compute interval and the main-KV transfer interval.
These timestamps certify simultaneous progress; they are not a speedup
measurement. This baseline pulls complete KV and executes replicated Q;
sparse KV fetch and query-local native tiling remain future work.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate \
  --output-dir fused_cp_report --long-history

# Framework-free launchers exercise independent two- and four-rank processes.
python -m pytest -q tests/torch/multicore/test_mega_dsa_fused_cp.py
```

The validator compares all five local outputs against a separately executed
stock LI/SFA chain at zero tolerance and all global key storage bits exactly.
It covers H32/H64, BF16/FP32 signed weights, packed lengths 3/10 and 2176/2240,
contiguous/strided/zigzag/empty owners, four compute-group counts and three
streams. Repeated invocations change source values and logical identity;
adapters are recreated while the same arena and monotonic epoch are retained.

The final 910B3/CANN 9.1 matrix passed 108 invocations per rank at CP1/2/4,
756 rank invocations in total. All 168 long-history rank invocations certified
LI/main-KV interval overlap and nine logical LD merges. Each group submits
three complete invocations on different streams before observing results;
all retained outputs and copied key buffers still match their own source
values exactly. CP2/CP4 passed through the formal ST launchers. Existing
CP2/CP4 reference and CP1 fused-forward regressions passed separately; the
eight pre-existing LI/SFA/gradient device objects retained their hashes.

This entry remains a raw detached forward probe. The main-attention backward
and autograd bridge are described below; selected-set KL and a production model
backend remain separate. Previous FP32 backward and BF16 model acceptance
failures remain open.


### Single-kernel CP backward and FP32 owner return

The [backward probe](fused_cp_backward.py) consumes a forward-produced saved
state. Global Q/KV/RoPE, final native indices, output and statistics live in
ordinary invocation-owned HBM. Admission binds saved state to its producer,
geometry/scale/schedule, tensor versions and prepared metadata. Local Q/KV
storage orders remain fixed even though the cross-rank layout signature omits
rank-specific permutations. Manually synthesized saved state and mutations
fail before native dispatch.

One `HyperDsaFusedGrad` mixed kernel executes the locked initialize, compute
and post tiles, with all-member phase closure by a reserved progress Vector.
Its host adapter calls the existing gradient tiler through an original-IR
child context; it retains the original scalar attributes, geometry checks,
workspace requirement and mathematical variant. The original separately
launched backward API remains available.

Only this member's local Q rows receive nonzero cotangents. After compute
closes, the progress worker reads dK/dV FP32 accumulators using the original
tiler's serialized offsets. It merges compressed dK+dV once, preserves
K-RoPE separately and writes complete owner rows into the sender's exclusive
inbox stripe. Each sender publishes write completion only after all MTE
stores finish. Every owner waits for all senders, adds stripes in ascending
rank order in FP32 and publishes ACK after consuming them. Source/inbox
reuse waits for every ACK. There are no cross-rank floating-point atomics.

Owner gradients return in the original local KV order and cast to BF16 only
after the ordered owner reduction. Q/Q-RoPE gradients use their independent
local Q permutation. A private 12 KiB progress UB is independent of the
compute groups' tile buffers. The experimental raw result additionally
exports FP32 pre-cast partials for independent transport certification; this
diagnostic materialization is included in the prototype's cost.

`FusedDsaCpAttentionProbe.attention(invocation, main_states, index_states)`
provides an explicit first-order main-attention autograd bridge. Native LI
selects the keys, while main attention returns gradients only for Q, shared
compressed KV, Q-RoPE and K-RoPE. Indexer inputs stay detached from this
objective; selected-set KL must establish its own gradient path and declared
loss normalization. All original and native saved tensors go through
`save_for_backward`; context metadata does not hold a second tensor cache,
so saved-tensor hooks and non-reentrant checkpoint can reconstruct the VJP.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_backward_validate \
  --output-dir fused_cp_backward_report --long-history
python -m pytest -q tests/torch/multicore/test_mega_dsa_fused_cp_backward.py
```

The validator independently checks exact FP32 owner bits against captured
peer partials, separate Q/K ordering and original three-launch accumulators.
Native gradient comparisons retain the existing rtol=0.02, atol=2e-5 checks
and separately report zero-tolerance bitwise results. A fixed-state control
found BF16 variation in stock and the original three-launch path themselves:
the maximum observed absolute difference was 7.62939453125e-6 and relative
L2 was 2.0951559621530854e-5. The fused path stayed in the same measured range.
These controls explain why universal gradient bitwise identity is not
certified; they do not clear the independent FP32 oracle or model failures.

CP1/CP2/CP4 matrices passed 108 invocations per rank, 756 rank invocations
in total, including 168 long-history invocations. Coverage includes H32/H64,
unequal rank cotangents, short/long packed histories, empty owners and all
four compute-group counts. All owner FP32 bits matched the independent
ordered peer reduction. Accumulators compared with the original three-launch
path had maximum relative L2 7.836094912391413e-7 and maximum absolute error
4.887580871582031e-6. The separate native-gradient bitwise diagnostic was
nonexact in 162/756 rank invocations; the established pointwise checks passed.

Fresh CP2/CP4 ST launchers passed. CP1/CP2/CP4 lifecycle checks passed for
zigzag and empty owners: four main gradients, indexer isolation, retained
graphs, delayed backward after different-layer source republication,
two-stream raw backward submission before observation and non-reentrant
checkpoint. The existing CP1 fused-forward ST passed again. All ten previously
accepted DSA device objects retained their hashes. These are primitive and
transport gates; external-Top-K production installation, selected-set KL,
independent model acceptance and full-step performance remain incomplete.


### External Top-K CP core: compacted subsets and empty-row autograd

The parameter-free [MegaDsaCore](module.py) admits caller-provided global
packed Top-K without running LI or KL. Preparation accepts CPU int32
`[local Q,2048]`, rejects unknown IDs and duplicates, gathers owner query
rows collectively, masks packed/causal violations and compacts `-1` to the
native tail. The owned selection exports independent storage. CPU preparation
is outside the hot path; arbitrary device-produced selection admission remains
part of the planned indexer integration.

The new native `HyperDsaCpAttention` pulls only compressed KV and K-RoPE,
then executes the original SFA tile in the same mixed kernel. It retains
complete ready/read-ACK closure and invocation-owned full activations. Q/Q-RoPE
are replicated in global packed order; owner-local Q and KV permutations stay
independent. Backward reuses the existing FP32 owner-return backend.

An external subset may contain fewer legal IDs than `min(K,causal_length)`.
The original gradient tile inferred its compute/scatter count from that
expression instead of the compacted valid prefix, allowing additional `-1`
entries into its gather. Stock backward also fails the independent FP32
oracle for this fixture; native-to-native parity is insufficient proof.

The locked [selected-count patch](../../ops/sparse_flash_attention_grad_selection_count.patch)
now bounds both compute and scatter by the compacted valid prefix. Complete
selections keep the final-slot fast path; underfilled selections use a binary
search. Gradient capability versions are now 2. The public training entry checks
this capability before forward dispatch and rejects old payloads. Prepared
subsets and all-empty selections pass the Python training admission contract.
CP1/CP2/CP4 device validation now covers compacted subsets, zero-count
rows and all four main gradients under the numerical criteria below.
KL/indexer restrictions and prior model acceptance results remain unchanged.

The CP forward adapter also requires ABI 2. After all tile writers finish,
disjoint vector-owned empty query rows receive exact zero output and denominator
inside the same kernel. This corrects stock SFA's finite empty-softmax sentinel
residue, without adding a host mask or another operator launch. Maximum retains
the native opaque representation. Nonempty output/max/sum remain subject to
bitwise stock comparison; empty output/sum follow the independent zero contract.
The validator records stock empty-row differences separately.

The [validator](../../examples/mega_dsa_core_cp_validate.py) has an explicit
`--forward-only` mode to certify legal external forward sets without claiming
padded backward acceptance. Small-case FP32 measurements remain visible.
CPU tests cover admission, packed compaction, independent exports, saved-state
hooks and old-payload rejection. They do not establish device autograd acceptance.

On 910B3/CANN 9.1, the forward-only CP1/CP2/CP4 matrix passed 88
invocations per rank: 616 rank invocations, including 112 long-history calls.
Native outputs and max/sum matched stock exactly. The matrix covers H32/H64,
contiguous/strided/zigzag/empty owners, external holes/packed violations,
all-empty selected sets, owner-zero sets and groups 1/2/7/19. All eleven
previously accepted DSA device objects retained their hashes. The CPU multicore
regression passed 300 tests and 985 subtests. These historical results certify forward
and the original admission/guard contracts; padded backward and its device autograd
were still unaccepted at that historical forward-only checkpoint. The current
selected-count and empty-row validation results are recorded below.


For the existing complete-cardinality count contract, the external core now has
real [CP1/CP2/CP4 ST launchers](../../../../../tests/torch/multicore/test_mega_dsa_core_cp.py).
The validator can explicitly choose `--selection-fixture complete --backward`.
This keeps the unresolved additional-padding requirement visible; choosing a
fixture does not change selection admission or native execution support.

The complete-selection ST passed 72 invocations per rank, 504 rank
forward/backward invocations in total, including 112 long-history invocations.
Every FP32 owner result matched independent ordered peer aggregation bitwise;
all five native gradients passed the original three-launch pointwise threshold
(rtol=0.02, atol=2e-5). Fourteen per-rank lifecycle tests covered zigzag/empty
owners, raw/autograd parity, retained graphs, delayed backward after another
publication, non-reentrant checkpoint and cross-stream backward. Native max/sum
remain nondifferentiable. The activated native payload was unchanged.

Small-case gradients also retain independent FP32 oracle measurements. The
largest gradient relative L2 was 0.0033812034965602666. Out of 1568 measured
four-gradient pointwise checks, 720 failed the recorded rtol=0.02, atol=2e-5;
these are not hidden by native parity. Neither this validation nor CPU analysis
of a proposed valid-prefix search establishes padded/empty-selection backward,
full-model acceptance, second-order derivatives, KL/indexer gradients or
performance. ABI 2 replaces the former additional-padding guard with a native
capability check. The current corrected payload passes the declared device
relative-L2 and exact-zero criteria for the tested external core fixtures.

The ST launchers additionally provide `test_mega_dsa_core_cp1_padded`,
`test_mega_dsa_core_cp2_padded` and `test_mega_dsa_core_cp4_padded`. They exercise
holes, owner-zero and all-empty sets, with both holes and all-empty lifecycle
checks. Small cases require independent FP32 relative L2 at most 0.02 for the
output and four main gradients, while retaining all original pointwise metrics.
All-empty output, denominator and gradients must be exactly zero. The fixed L2
threshold permits the observed BF16 near-zero pointwise differences; it does
not certify the recorded pointwise failures or full-model acceptance. These new
padded STs pass on CP1/CP2/CP4 after the selected-count correction and native
empty-forward canonicalization.

The corrected-payload complete matrix passed 504 rank forward/backward
invocations (112 long-history) and 14 rank lifecycle checks. The padded matrix
passed 616 rank invocations (112 long-history) and 28 rank lifecycle checks.
It includes 56 all-empty rank invocations with exact zero output, denominator
and four main gradients. Maximum retains the original native representation.
Nonempty rows matched stock output/max/sum bitwise; empty output/sum deliberately
use the independent zero contract because stock preserves finite-sentinel
residue. FP32 owner returns matched independent ordered peer sums bitwise.

Small-case independent FP32 gradient relative L2 reached at most
0.0033812034965602666 for complete selections and 0.003920230594267788 for padded
selections, below the declared 0.02 bound. The original pointwise criterion
(rtol=0.02, atol=2e-5) still failed 720/1568 complete and 868/2016 padded gradient
measurements. These failures remain visible; this is neither full pointwise
alignment nor full-model acceptance. Long-history correctness uses the original
three-launch gradient comparison and independent owner reduction, without a
full FP32 attention oracle at those sizes. KL/indexer gradients, model CP/TP,
second-order differentiation and performance remain outside this acceptance.


An additional all-empty CP1/CP2/CP4 matrix passed 504 rank invocations,
including 112 long-history invocations with 4416 packed tokens and H32/H64.
All output/denominator/four-gradient values were exactly zero. Fourteen rank
lifecycle checks independently required exact zero through retained graphs,
delayed backward, non-reentrant checkpoint and cross-stream execution.
The combined corrected-payload evidence is 1624 rank invocations, 336 long-history
invocations and 56 rank lifecycle checks. CPU regression passed 140 tests and
218 subtests. These results certify the tested external-core relative-L2,
empty-set and transport contracts; the pointwise/model limitations above remain.


### P4 training composition: native selection plus selected KL

The initial parameter-free [MegaDsa](module.py) composition combined the existing CP native
LI/Top-K/SFA forward with native main backward and the locked enhance selected
KL operator. It returns `(local_compressed_output, auxiliary_loss)`. The model
still owns projections, RoPE, state dictionaries and `aux_loss_auto_scale`.
Main H32/H64, index H64/Di128, BF16 C512/RoPE64, signed already-scaled
BF16 merge weights with KL enabled and one-node CP are explicit initial
dimensions. FP32 merge weights are accepted only with loss_coeff=0.

The fused forward retains its owned global index Q/K/weight tensors for the
immediate KL call. No second index-input all-gather or CPU selection admission
is needed. Native LI supplies complete causal Top-K provenance. This training
composition does not admit external underfilled selections into stock KL;
`MegaDsaCore` remains the entry for external subsets and empty selected rows.

KL computes and saves its three derivatives once in forward. Backward scales
those saved derivatives by the upstream auxiliary gradient once and returns
index Q/K/weight gradients through a shared FP32 owner reduce-scatter, restoring
independent Q/K storage orders. Main Q/shared KV/Q-RoPE/K-RoPE use the existing
native FP32 owner-return path. Both paths keep tensors under autograd saved
hooks; no shared scratch or module-level last-result cache owns backward data.
LM-only backward leaves all index gradients absent. KL-only backward leaves
all main teacher gradients absent. A zero coefficient skips the KL operator
and gives an exact zero auxiliary loss and index derivatives.

Each rank evaluates the replicated global selected KL and returns its
`1 / CP_size` contribution, following `CannDsaCpReference`. Explicit
`DsaLossNormalization(global_valid_queries, reducer_divisor)` applies the
actual downstream parameter-reducer compensation. Summing detached rank loss
contributions gives the declared global normalized objective. Members must use
matching objective/require-grad masks and backward schedules. The caller
applies the existing `aux_loss_auto_scale` once, rather than deriving auxiliary
scale from the main cotangent.

```python
from hyper_parallel.core.multicore.modules.mega_dsa.module import MegaDsa
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization

attention = MegaDsa(
    workspace, invocation, heads=32, attention_scale=model_scaling,
    schedule=schedule,
    normalization=DsaLossNormalization(global_valid_queries, reducer_divisor),
    loss_coeff=loss_coeff,
)
out, auxiliary_loss = attention(
    query, compressed_kv, query_rope, key_rope,
    index_query, index_key, scaled_merge_weight, batch_meta,
)
```

The [training validator](../../examples/mega_dsa_training_cp_validate.py) checks
stock selection sets, original CANN bitwise output, CP1 enhance gradients and
loss, independent FP32 main/KL
measurements and global loss contributions. Its declared native pointwise
threshold is rtol=0.02/atol=2e-5. Independent FP32 gradients/loss must be finite
and have relative L2 at most 0.02 or maximum absolute error at most 2e-5;
original pointwise results are retained separately. Fixtures cover H32/H64,
BF16 KL weights, FP32 zero-KL weights, explicit FP32-KL rejection,
uneven/empty owners, objective isolation, coefficients,
upstream auxiliary scales and reducer divisors, retained graphs, checkpoint,
delayed backward and cross-stream execution. CPU contract tests pass; The CP1/CP2/CP4 short packed training matrix passes on the validated payloads.
Omni enhance output differences are recorded independently; the original CANN
SFA is the bitwise output comparator. The smoke's enhance-output relative L2
was 0.0017897501040310402, with pointwise comparison failing. The independent
FP32 output relative L2 was 0.0016449038835654839; it also failed pointwise.
The smoke's maximum independent seven-gradient relative L2 was
0.003362107087384327, and KL loss relative error was 4.245591711149349e-06.

This is the first P4 composition, with a separate host-dispatched KL launch
and collective index-gradient return. KL tile fusion, device sparse
request/count generation, sparse gradient transport, model CP/TP integration,
full training and performance acceptance remain subsequent work.


Selected KL requires the existing reference Omni CANN vendor in addition to
the activated multicore vendor. Activate both `ASCEND_CUSTOM_OPP_PATH` entries
and both op-api library directories before importing Torch/TorchNPU. The
multicore adapter is explicitly preloaded from its own payload. The validator
records the actual loaded op-api libraries; an absent enhance KL symbol is an
environment failure, not a numerical result. The zero-coefficient path avoids
the optional KL operator.


The locked enhance KL op definition and tiling require all main/index/weight
inputs to share BF16/FP16 dtype. LI's FP32 weight capability therefore does not
imply FP32 KL support. `MegaDsa` rejects a nonzero-KL FP32 weight request before
native forward, preserving the caller's precision. Supporting FP32 KL needs a
separately validated native adapter; this composition performs no downcast.


The P4 composition matrix passed 16 positive training scenarios and one FP32-KL
rejection per rank on CP1/CP2/CP4: 112 positive rank scenarios, seven rejections,
147 native fused-forward invocations and 105 native main-backward invocations.
All 707 available seven-gradient comparisons with the CP1 enhance baseline
passed rtol=0.02/atol=2e-5. LM-only and KL-only gradients remain absent on the
opposite branch. Zero coefficient and zero upstream auxiliary scale produce
exact zero index gradients. CP reducer-divisor cases, empty owners, retained
graphs, checkpoint, delayed publication and cross-stream backward all pass.
Both expected op-api library paths were observed in the loaded process maps.

The independent FP32 seven-gradient maximum relative L2 was
0.003572747141616831; KL loss maximum relative error was
1.9810393961618714e-05 (maximum absolute error 2.130400389432907e-08).
All 315 independent FP32 index Q/K/weight measurements passed the original
pointwise threshold. FP32 output maximum relative L2 was 0.002212485804360936.
Original pointwise comparisons still failed for 86/707 FP32 gradient measurements and 96/112
FP32 output measurements. These remain recorded alongside native successes;
this does not establish full pointwise or model alignment. The short-matrix
CPU regression passed 149 tests and 224 subtests. Long-history complete LM+KL
and truncated TopK training are covered by the extended matrix below. This
training composition retains the host KL path; standalone callable phases
are described after that matrix.

The training validator also provides `--long-history`. Its fixtures include
packed lengths `(3, 2113)` and `(2176, 2240)`, giving respectively 65 and 320
query rows with causal histories longer than K=2048. It checks the complete
stock selected sets and bitwise BF16 TopK values, per-query selected counts,
packed boundaries, native main/KL derivatives and all seven independent FP32
derivatives. The extended matrix includes joint objectives, checkpoint with
explicit reducer compensation, and KL-only with empty owners. These are
component training checks; full-model and performance acceptance remain
separate requirements.

The [CPU training oracle](../../examples/mega_dsa_training_oracle.py)
backpropagates bounded query chunks into shared full-KV leaves and concatenates
independent query derivatives. It computes dense QK scores, gathers the original
selected slots, and scatters the selected probabilities for the PV matrix
multiply. This avoids expanded C512 KV scatter-backward on CPU while retaining
the original selected-set objective, global KL normalization and every query
contribution. Its score storage grows with the full key count and is bounded
by the query chunk size; this is an offline checker, not a production backend.
The [oracle cross-check](../../../../../tests/ut/core/multicore/mega_dsa/test_training_oracle.py)
compares output, KL and all seven derivatives with unpartitioned expanded-KV
FP32 autograd. It covers signed weights, unordered slots, padding and empty
rows, packed/future masking, joint/LM-only/KL-only/zero-coefficient objectives,
upstream auxiliary scales, reducer divisors and chunk sizes 1/3/7.

Run the extended fixtures with both explicitly activated vendors and the
device idle gate, using `--smoke` for the first H32 joint fixture:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPU_WAIT_VISIBLE_DEVICES=0,1,2,3 \
HCCL_IF_BASE_PORT=65120 HCCL_NPU_SOCKET_PORT_RANGE=62208-62271 \
NPU_WAIT_NUM_CARDS=4 NPU_WAIT_POLL_SECONDS=60 bash ~/doc/npu_wait_and_run.sh \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate \
  --output-dir /tmp/mega_dsa_training_long_cp4 --long-history
```

Formal extended launchers are `test_mega_dsa_training_long_cp1`,
`test_mega_dsa_training_long_cp2` and `test_mega_dsa_training_long_cp4` in the
[training ST](../../../../../tests/torch/multicore/test_mega_dsa_training_cp.py).
On a shared host, assign unused HCCL host and NPU port ranges to each concurrent
group. Device idle gating does not reserve the host HCCL listening ports.

The extended CP1/CP2/CP4 matrix passes on the same 910B3/CANN 9.1 native
payload: seven rank reports, 28 training scenarios, 42 fused forwards and 21
native main backwards. All 168 available native seven-gradient comparisons
pass rtol=0.02/atol=2e-5. Stock selected sets, BF16 TopK values and original
CANN output are exact, including the 65/320 genuinely truncated query rows in
the two fixtures. Joint LM+KL, checkpoint with reducer compensation and
KL-only gradient isolation with empty owners all pass.

All 168 independent FP32 gradient measurements satisfy the existing finite
relative-L2/maximum-absolute criterion. Maximum gradient relative L2 is
0.0032989832834116107, and maximum output relative L2 is
0.0024818912675734555. All 84 independent FP32 index Q/K/weight measurements
also pass the original pointwise threshold. Original pointwise failures
remain for 17/168 gradient measurements and 23/28 output measurements. KL loss
maximum relative error is 1.1491664595111996e-05, with maximum absolute error
8.149072527885437e-09. CPU regression passes 151 tests and 258 subtests.

The first CP4 attempt failed during HCCL initialization because the default
host/NPU communication ports were occupied. The isolated-port CP4 retry
passes; that initial environment failure is preserved separately. The
original expanded-KV long smoke also passed and remains separate from the
intentionally interrupted slow CPU-oracle matrix. Native framework and all
12 DSA kernel objects retain their validated hashes; no native rebuild was
performed in this round. These are component training results, not full-model
acceptance, KL tile fusion, CP8/CP×TP acceptance or complete-step performance.

## Selected-KL callable phases

The [mixed KL probe](mixed_kl.py) extracts the locked CANN selected-KL implementation
into initialization, compute and post-processing launches. Each compute team contains
one AIC and two AIVs. Logical query partitions and loss partials retain their own IDs
while the team reuses its physical gather/matmul scratch. Initialization clears the
FP32 key accumulators and logical loss partials; post-processing runs after the
compute launch, reduces the loss and converts the accumulated index-key gradient.
The compute tile contains no whole-chip barrier. Its internal cross-core flags are
separate from mixed-group dispatch/completion flags.

Packed cumulative lengths enter the native API as host constants through CANN's
`aclIntArray` conversion. They come from prepared batch metadata, without a device
readback. Every phase trace owns independent storage because host tiling inspects
the CANN storage shape. The native API normalizes contiguous tensor views and
restores output views using the same ACL conventions as mixed SFA.

The public probe preserves the P0 BF16 selection contract. The raw native callable
also has an FP32 merge-weight variant with a matching FP32 weight derivative; its
standalone validation is separate from `MegaDsa` admission. Deterministic mode and
per-phase traversal repeats are rejected. Results retain their own raw derivatives,
scalar loss, scratch and phase traces across later submissions. No auxiliary loss
coefficient or upstream cotangent is applied inside these phases.

The [validator](../../examples/mega_dsa_mixed_kl_validate.py) compares the scalar loss
and three raw indexer derivatives with stock CANN and an independent FP32 selected-set
oracle. BF16 cases additionally compare the explicitly activated Omni enhance KL.
It covers H32/H64, signed BF16/FP32 weights, packed causal padding, group counts
1/2/7/19, and three distinct invocations retained together. `--long-history` adds
packed lengths `(3,2113)` and `(2176,2240)` with genuine K2048 truncation. Original
pointwise metrics are saved alongside the existing FP32 relative-L2 criterion.
FP32 fixtures include weights outside the BF16 grid. `--weight-dtype bf16` or
`--weight-dtype fp32` selects a focused matrix; the default checks both variants.

The BF16 short/long matrix passes 48 cases. All 192 loss/index-Q/index-K/weight
measurements pass the original pointwise threshold against each of stock CANN,
Omni and the independent FP32 oracle. Maximum relative L2 against the oracle is
0.0024263695767346456. The genuinely truncated fixtures contain 65 and 320 query
rows whose causal history exceeds K2048.

After a complete source rebuild and private payload activation, the FP32-weight
short/long matrix with values outside the BF16 grid passes 48 cases and all 192
stock/oracle pointwise measurements. Maximum relative L2 is
8.857039325370368e-05 against stock CANN and 0.002421630389632869 against the FP32
oracle. The subsequent short BF16/FP32 matrix passes 48 cases and all 192 comparisons
per baseline. Every case verifies three complete phase traces, including physical
scratch reuse and results retained across later submissions. CPU regression passes
153 tests and 258 subtests. The 14 DSA device objects retain their verified hashes,
including all 12 previously accepted DSA objects; the new KL variants add two objects.
These standalone primitive results do not clear the existing full-model BF16
composition acceptance gap or establish training speedup.

Run with the private native payload, CANN and comparison vendor activated, through
the device idle gate:

```bash
NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 NPU_WAIT_POLL_SECONDS=60 \
  bash ~/doc/npu_wait_and_run.sh python -m \
  hyper_parallel.core.multicore.examples.mega_dsa_mixed_kl_validate \
  --output /tmp/mega_dsa_mixed_kl.json --long-history
```

The [extended ST launcher](../../../../../tests/torch/multicore/test_dsa_mixed_kl.py)
keeps framework imports in the worker. Standalone callable-phase acceptance does not
establish KL integration into the CP task DAG. The training backend at that milestone
used its explicitly activated host KL reference path and excluded nonzero-KL FP32
weights. The following six-phase adapter adds a distinct native training path.

## Six-phase LI/SFA/KL training kernel

The [fused training probe](fused_training.py) composes LI main, LI merge, SFA,
KL initialization, KL compute and KL post-processing into one native launch. A
reserved progress group releases each phase only after every compute group has
completed its logical tasks. KL consumes the selected indices and SFA max/sum
after their producing phases close. Each invocation owns independent LI/KL scratch,
nine arithmetic outputs and six phase traces. The KL tile retains its logical
query/loss partition IDs while reusing physical scratch; it contains no whole-chip
barrier. The native API takes prepared host cumulative lengths without reading
selection or length tensors back to the host.

The composite has BF16 and FP32 merge-weight variants for both single-card math
and CP transport. Other states remain BF16, main heads are 32/64, index heads are
64, index width is 128, compressed width is 512, RoPE width is 64 and K is 2048.
FP32 weights are preserved and produce an FP32 weight derivative. Deterministic
KL and repeated per-phase traversals are outside this adapter's contract.

`MegaDsa` defaults to `kl_backend="native"` for the six-phase CP forward. It saves the
raw loss and three index derivatives under autograd saved-tensor hooks, applies
the declared coefficient/global normalization/CP contribution once, and applies
the incoming auxiliary cotangent once during backward. Main backward continues
to use the existing native owner-return kernel. LM-only and KL-only gradient
isolation, independent Q/K owner orders and delayed backward use the same saved
state contract. A zero coefficient uses the three-phase LI/SFA kernel and exact
zero index derivatives. `kl_backend="reference"` selects the separately activated
Omni primitive and requires BF16 weights when KL is enabled. CP members must agree
on backend, coefficient, normalization and their forward/backward schedule.

The native production path needs the activated multicore payload and CANN. The
explicit reference backend and validators additionally need the Omni comparison
vendor. Both paths currently replicate packed Q, pull full KV and return index
gradients with FP32 collectives. Selected request/count generation and sparse
owner-gradient transport remain open; the six-phase kernel does not establish
those transport features or a complete-step speedup.

The [six-phase validator](../../examples/mega_dsa_fused_training_validate.py)
checks all five forward outputs against stock CANN, raw KL loss and three
derivatives against stock CANN and an independent FP32 oracle, and all six
ordered device releases. BF16 weights also compare against Omni. It covers signed
BF16/FP32 weights, including FP32 values outside the BF16 grid, H32/H64, group
counts 1/2/7/19 and three distinct results retained across later submissions.
`--long-history` adds packed lengths `(3,2113)` and `(2176,2240)` with 65/320
genuinely truncated query histories. Numerical thresholds match the existing
KL probes; original pointwise results remain separately recorded.

```bash
NPU_WAIT_VISIBLE_DEVICES=0 NPU_WAIT_NUM_CARDS=1 NPU_WAIT_POLL_SECONDS=60 \
  bash ~/doc/npu_wait_and_run.sh python -m \
  hyper_parallel.core.multicore.examples.mega_dsa_fused_training_validate \
  --output /tmp/mega_dsa_fused_training.json --long-history

NPU_WAIT_VISIBLE_DEVICES=0,1,2,3 NPU_WAIT_NUM_CARDS=4 NPU_WAIT_POLL_SECONDS=60 \
  bash ~/doc/npu_wait_and_run.sh python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --module hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate \
  --output-dir /tmp/mega_dsa_fused_training_cp4 --kl-backend native
```

The [single-card ST](../../../../../tests/torch/multicore/test_dsa_fused_training.py)
keeps framework imports in the worker. The existing CP training launchers use the
native backend by default in their validator; `--kl-backend reference` retains the
earlier comparison matrix. Native short fixtures additionally exercise nonzero-KL
FP32 weights with joint, KL-only and checkpoint objectives plus an explicitly
selected BF16 reference case. Each native nonzero-KL forward must emit all six
phase traces and saved KL outputs.

The complete native build passes and adds two fused-training device objects
containing the four BF16/FP32 and CP/non-CP variants;
all 20 previously accepted device objects retain their hashes. The short
single-card matrix passes 48 cases, 240 exact forward-output comparisons and
192 raw loss/gradient measurements against each applicable baseline. Maximum
relative L2 is 0.0024263695767346456 against the independent FP32 oracle. CPU
regression passes 158 tests and 268 subtests, including raw derivative ownership,
reference bypass, FP32 weight gradients, checkpoint and retained-graph scaling.
The combined short/long single-card matrix also passes 96 cases: 480 forward
outputs are exact and all 384 raw KL measurements per baseline pass the original
pointwise threshold. Maximum relative L2 is 0.00010109051444359716 against stock
and 0.0024263695767346456 against the FP32 oracle.

The CP1/CP2/CP4 short native training matrix passes all seven rank reports:
140 positive training scenarios and seven expected FP32-KL rejections through
the explicitly selected reference backend. The matrix includes 189 forward
invocations and 126 native main-backward invocations. All 875 available native
seven-gradient comparisons pass the original pointwise threshold. Of the 189
forwards, 168 use six phases and 21 use the explicit reference or zero-KL path. Stock TopK
sets/values and original CANN outputs are exact; isolation, zero coefficients,
zero auxiliary cotangents, reducer compensation, empty owners, retained graphs,
checkpoint, delayed publication and cross-stream execution pass. BF16 reference
and nonzero-KL FP32 native cases pass alongside native BF16 cases.

All 875 independent FP32 gradient measurements satisfy the existing finite
relative-L2/maximum-absolute criterion; maximum relative L2 is
0.0035769540586985246. Original pointwise failures remain for 107/875 gradient
measurements and 120/140 output measurements. Maximum FP32 output relative L2
is 0.0022124857085496536. All 140 FP32 loss measurements pass pointwise, with
maximum relative error 3.0094478258633895e-05 and maximum absolute error
3.236345946788788e-08. These component results preserve the earlier distinction
between native pointwise agreement and independent FP32 relative-L2 acceptance.
The native six-phase long-history CP1/CP2/CP4 matrix also passes all seven rank
reports: 28 training scenarios, 42 six-phase forwards and 21 native main backwards.
All 168 available native gradient measurements and 28 loss measurements pass
the original pointwise threshold. The packed fixtures contain 65/320 queries
with genuinely truncated histories; stock selection sets/values and outputs are
exact. All 168 independent FP32 gradient measurements satisfy the existing
L2/absolute criterion, with maximum relative L2 0.003298960380207957. Original
pointwise failures remain for 17/168 gradient and 23/28 output measurements.
All 28 FP32 loss measurements pass pointwise; maximum relative error is
7.684610066669985e-06 and maximum absolute error is 6.28642737865448e-09.
After selecting native KL as the module default, CPU regression over megaDSA
and shared-root SHMEM passes 191 tests and 282 subtests. The reference fixture
selects its backend explicitly, while the native tests exercise the constructor
default and confirm FP32 derivative dtype and one-time auxiliary scaling.
Full-model BF16 acceptance, CP8/CP×TP and complete-step performance remain separate
requirements.


## Device-generated selected main-KV requests

`MegaDsa(..., kv_transfer="selected")` selects the experimental request/count
producer inside the existing composite operator with the native backend.
Nonzero KL executes nine phases; `loss_coeff=0` skips its arithmetic and executes
six phases. `kv_transfer="full"` remains the default. The initial selected mode gathers the
same complete packed Q/index-Q/weights and pulls full index K for exact selection.
It retains global-key destination buffers and generates the main-KV union from
the actual native TopK output. The union may cover all keys, so this mode does
not promise lower traffic or compact saved activation storage.

Three additional stages run after LI merge: membership initialization, atomic
membership construction and deterministic request packing/main-KV pull. SFA and
KL wait for the reserved progress group to close those stages. Native indices
resolve through prepared per-query sequence starts, with padding/future slots
excluded. Membership accumulates aligned int32 atomic counts, preserving repeated
slots without neighboring-key cache-line races. Occurrence capacity is limited
to int32, admitting at most `(2**31-1)//2048` global queries.

The [CPU descriptor oracle](selected_requests.py) independently builds a selected
set, preserves packed destination addresses and coalesces runs only when owner,
owner-local source and destination are all consecutive. Native request capacity
is the global key count; the device publishes the actual count after descriptor
writes. The progress Vector consumes that count to perform real BF16 C512/RoPE64
pulls, then completes the existing read ACK protocol. Each invocation owns its
membership, descriptor table, counts and six or nine phase records. Request/count/TopK
readback occurs only in explicit offline validation.

The training validator accepts `--kv-transfer selected`. It compares descriptors,
all membership counts, epochs and local/remote main bytes with the CPU oracle,
checks descriptor publication before transfer, and requires all declared phase
releases before accepting stock output and owner gradients. Reference fixtures
explicitly report full transfer. `--zero-kl` exercises selected zero-coefficient
execution across the lifecycle matrix. Framework-free [CP1/CP2/CP4 launchers](../../../../../tests/torch/multicore/test_mega_dsa_selected_training_cp.py)
provide short and genuinely truncated long-history matrices.

```bash
NPU_WAIT_VISIBLE_DEVICES=0,1,2,3 NPU_WAIT_NUM_CARDS=4 NPU_WAIT_POLL_SECONDS=60 \
  bash ~/doc/npu_wait_and_run.sh python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --module hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate \
  --output-dir /tmp/mega_dsa_selected_training_cp4 --kv-transfer selected --long-history
```

The complete candidate native build passes and preserves all 20 non-training
DSA/MoE device object hashes. The original six-phase path is retained as a
separately tested baseline. Final CPU megaDSA/shared-root regression passes
200 tests and 307 subtests, including transfer byte/timestamp decoding and
prepared-address mutation rejection before a lease or native submission. The first selected CP1 device smoke passes descriptor,
byte-count, nine-phase and seven-gradient checks: 61 admitted selected slots
produce 13 unique keys and 13 descriptors, with 14,976 main bytes. Its query
scope covers all keys, giving the same main byte count as full pull. The full/selected CP1/CP2/CP4 short matrices both pass seven rank reports and
140 positive scenarios plus seven explicit reference FP32-KL rejections. All
875 native gradient measurements per mode pass original pointwise bounds. The
selected mode has 168 nine-phase forwards; reference/zero-KL cases retain 21
baseline forwards. All 875 independent FP32 gradient measurements satisfy the
existing L2/absolute criterion, with maximum relative L2 0.0035769540586985246;
107/875 gradient and 120/140 output measurements retain original pointwise failures.

The selected random long-history CP1/CP2/CP4 matrix also passes seven rank reports,
28 scenarios, 42 nine-phase forwards and 21 main backwards. All 168 native gradient
measurements pass pointwise. Independent FP32 gradient maximum relative L2 is
0.003298960380207957; original pointwise failures remain for 17/168 gradients and
23/28 outputs. All 28 FP32 loss measurements pass pointwise, with maximum relative
error 7.684610066669985e-06. Device descriptors/counts and actual bytes match the
CPU oracle in every forward. All sampled random unions cover the full key domain,
so these fixtures establish no main-byte reduction. Compute groups 1/2/19 each
pass CP1/CP2/CP4 short smoke, adding 21 rank scenarios alongside group 7.

`--selection-fixture hotset` creates strictly positive scores for each sequence's
first K keys and zero scores for its cold suffix through the actual native
indexer. It preserves complete causal K2048 provenance. The long fixtures then
request 2051 of 2116 keys and 4096 of 4416 keys, respectively. Their main bytes
are 2,362,752 and 4,718,592, saving 74,880 and 368,640 bytes against full pull.
The hotset CP1/CP2/CP4 matrices pass all seven rank reports: 28 scenarios, 42
nine-phase forwards and 21 main backwards. All 168 native gradients and 28 loss
measurements pass original pointwise bounds. All independent FP32 gradients
satisfy the existing L2/absolute criterion; maximum relative L2 is
0.013767521826761008. Pointwise failures remain for 17/168 FP32 gradients and
23/28 outputs. All FP32 loss measurements pass pointwise, with maximum relative
error 2.1349173793687763e-05. Unrequested C/RoPE rows remain initialized to NaN;
all 16 CP4 hotset scenarios explicitly verify that poison remains while output
and gradients stay finite. Supplementary CP1/CP4 poison smokes run separately.
Full training-step performance remains unmeasured; full pull remains the default.
Device sparse gradient owner return, selected external-TopK admission,
rank-local query/tile scope, CP8/CP×TP and model acceptance remain open.


## Selected transfer with zero KL

`MegaDsa(..., loss_coeff=0, kv_transfer="selected")` uses the same composite operator
with a `with_kl=false` tiling attribute. It runs LI main/merge, membership init/build,
request pack/pull and SFA, with read ACK on the final sixth phase. The host skips KL
child tiling, and the kernel never accesses KL scratch or derivative destinations.
The dedicated forward bridge creates empty auxiliary descriptors rather than global
zero derivative buffers, while preserving the existing training entry signatures.

Autograd saves caller inputs and main attention state for version checks and delayed
or retained backward. An explicit auxiliary backward returns zeros in each caller's
local shape and dtype without an index-gradient owner collective. Main-only backward
leaves index gradients absent. The forward loss is exact FP32 zero; hard selection
still has no main-objective derivative.

The training validator accepts `--kv-transfer selected --zero-kl`. The short matrix
covers BF16/FP32 merge weights, empty owners, checkpoint, retained/delayed backward,
objective isolation and stream changes. Long hotset fixtures retain complete native
K2048 selection and explicitly check unfetched C/RoPE poison and actual byte counts.
Actual CANN 9.1/Ascend910b build and execution accept the empty auxiliary descriptors.
The 20 nontraining device objects retain their previous hashes. The current CPU
regression passes 203 tests and 312 subtests; lint, native formatting, ST import
isolation, documentation links and the AGENTS catalog pass.

Selected zero-KL CP1/CP2/CP4 short matrices pass seven rank reports and 133 scenarios,
including BF16/FP32 weights, empty owners and the lifecycle cases above. All 182
forwards close six phases; 119 main backwards and 826 native gradient comparisons
pass the original pointwise bounds. All auxiliary losses are exactly zero. The
independent FP32 criterion passes, with maximum gradient relative L2
0.003189607634470013; 100/826 gradient and 113/133 output pointwise failures remain.

The zero-KL long hotset matrices pass seven rank reports and 28 scenarios, with 42
six-phase forwards and 21 main backwards. All 168 native gradient comparisons pass
pointwise bounds. Unrequested C/RoPE rows remain NaN in every scenario; the selected
sets and byte counts match the 2051/2116 and 4096/4416-key fixtures above, saving
74,880 and 368,640 main bytes. The H64 long fixture also uses FP32 merge weights.
Independent FP32 gradients meet the existing criterion, with maximum relative L2
0.0026095253024382647; 17/168 gradient and 23/28 output pointwise failures remain.
All 28 auxiliary losses are exact zero.

The rebuilt payload also passes 48 standalone six-phase arithmetic cases and fresh
full/selected CP1/CP2/CP4 short regressions, each with 140 positive scenarios, seven
explicit reference FP32-KL rejections and 875 native pointwise gradient comparisons.
In the selected regression, zero-coefficient cases now execute six selected phases;
nonzero native KL retains nine phases and the explicit reference fixture uses full
transfer. Source, activated payload and actually loaded library hashes are recorded
with the per-rank evidence. These results do not close the full-model BF16 gap or
establish complete-step speedup; full transfer remains the default.
