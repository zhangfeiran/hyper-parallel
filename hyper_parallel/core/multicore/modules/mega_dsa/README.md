# megaDSA development baseline

## 2026-10-12 阶段提交状态

本轮将 native TP/KL 与私有梯度累加器、可选 FP32 VJP 诊断、CP/TP 训练接入、
模型梯度/SGD reference、selected CPU oracle 和文档按依赖分别保存为本地提交。
CPU 回归为 `420 tests / 862 subtests`；Ruff、Pylint、Markdown 格式和 catalog
检查通过。通用 cpplint 和 Lizard 检查仍有告警，明细保存在
`build/native/validation/mega-dsa-commit-20261012/`，完整静态门禁尚未通过。

冻结的模型 reference 仍为 `held_out_failed`，完整模型数值与性能验收尚未完成。
H128 原型的设备证据保存在 validation 目录；本次提交不扩展正式 native H128
接纳范围。L8192 运行因用户要求而终止，CP8 后续验收未完成，均不计为通过。
实现 goal 保持暂停；本次提交整理未重新启动 NPU 任务。

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

## Compact local-query indexer preparation

[Local query layout](local_query.py) packs caller-local rows in sequence/global-position
order, stores genuine local cumulative query counts and sequence-relative logical
positions, and restores native output to caller order. Empty sequence buckets retain
repeated cumulative values; an empty owner has zero query/weight/position rows.
Prepared metadata is checked for both mutation and tensor replacement before native
loading. The local indexer probe performs no Q/index-Q/weight collective and reads
canonical complete index K.

The existing mixed-indexer operator accepts an optional query-position tensor. Its
arch22 vector mask uses each row's explicit position plus one, preserving exact
causal candidates for interior and zigzag rows. For sorted distinct positions p_i
in a sequence of length L with m local queries, p_i <= L-m+i: the existing right-down
Cube work partition is conservative and never omits a true candidate. This first
implementation retains those conservative key scans; it does not yet implement
position-aware cost partitioning or streaming index K.

Local LI main/merge retain the reserved mixed team's two device phase closures and
scratch reuse. Empty owners skip arithmetic while participating in both closures,
with no fabricated query row or length. Actual CANN 9.1/Ascend910b build and execution
accept zero-row query/weight/output descriptors while closing both phases. The
full-query product backend remains the accepted training path.
The framework-free local-indexer launchers cover concurrent CP1/CP2/CP4 ownership,
packed empty buckets, explicit interior positions, BF16/FP32 weights and native
K2048 truncation. This preparatory probe has no SHMEM transport or autograd, and
therefore does not establish local-query LM+KL training acceptance.

The complete local-query path still requires selected membership over local rows,
SFA/main backward with empty-owner ready/ACK, local loss contributions without a
replicated 1/CP factor, and index-K FP32 partial owner sums before casting. Sparse
gradient membership must follow selected indices rather than zero-valued gradients.
CP8/CP×TP, model stability, coexistence and complete-step performance remain open.

The local LI CP1/CP2/CP4 matrices pass seven rank reports and 350 scenarios, with
700 forwards and same-stream scratch reuse. They include 140 long-history cases,
175 FP32-weight cases (70 long), 50 empty-owner cases, groups 1/7/19 and 82,098
rows whose exact causal bound is strictly smaller than right-down. All tested
selected sets and BF16 values match full stock LI exactly; independent explicit
position/random-short and analytic-long oracles pass. Empty owners allocate zero
arithmetic scratch. No local query/weight collective is performed by the probe.

The rebuilt payload passes 36 full-LI short/long regression cases, 48 standalone
six-phase arithmetic cases and fresh full/selected CP4 training regressions. Each
training mode passes 80 positive scenarios, four expected reference FP32-KL
rejections and 500 native pointwise gradient comparisons. Four supplementary CP4
identity reports capture the actual framework bridge and host libraries, alongside
per-rank numerical evidence. Source/payload/library hashes match; patch context
normalization preserves all five assembled LI source files byte-for-byte. Fourteen
unaffected device objects keep their previous hashes. CPU regression passes 209
tests and 362 subtests; lint, edited native formatting, launcher import isolation,
documentation links and the AGENTS catalog pass.

These are local-indexer and existing-training regression results. They do not prove
local-query SFA/KL/backward, reduced end-to-end memory, full-model BF16 acceptance
or training-step speedup. Those remaining contracts follow the implementation
sequence above; the original multi-card CP and fused-execution goal remains active.

## Local-query fused CP forward

[原始 CP forward adapter](fused_cp.py) 提供 `prepare_local_queries(query_ids)` 和
`forward_local(invocation, main_states, index_states)`。冷准备支持完整 owner query 集合或调用方指定顺序的
子集，包括 CP1 上的空 tile。每次调用仍发布该 owner 的完整 index K、compressed KV 和 key RoPE。
Query、index-query 和 merge weights 保留实际本地行数；这条前向路径不执行 query gather。

Native launch 闭合六个阶段：indexer main、indexer merge、membership init、membership build、
request pack/pull 和 SFA。Membership 根据显式 global query IDs 和序列起点解析本地 selected slots。
SFA adapter 使用显式 query position 确定 sparse prefix 和因果边界。空 query tile 跳过 LI/SFA 算术
及 SFA child tiling，LI retained 算术 scratch 为零，但仍参与全部六个阶段闭合和 peer ready/read-ACK 交换。

零算术 scratch 仅指调用方的 LI retained buffer。MIX kernel 的生成入口仍会清零系统 workspace，
因此 tiling 始终申请至少 `PlatformAscendC::GetLibApiWorkSpaceSize()` 返回的系统空间。非空 query 使用
SFA child tiling 返回的更大空间；空 query 不能因跳过 SFA tiling 而把 kernel workspace 设为零，
否则入口初始化可能越界覆盖相邻张量。

输出和保存的激活由各次调用独立持有；输出置换恢复调用方的 query 顺序。Main-KV 目标缓冲区初始化为
NaN，便于验证未获取的行保持 poisoned 状态。Transport trace 记录完整 index-K 字节数及实际去重后的
main-KV 字节数；空 query tile 的 main-KV 字节数应为零。保存结果还持有 native 使用的 runtime buffers，
避免异步执行期间释放 metadata 或 retained scratch。
Local saved state 还持有本次 forward 使用的 `DsaLocalQueryLayout`；backend 重新准备另一块 tile
不会改变该状态的 query lengths、positions 或恢复顺序。使用保存状态前会核对这些张量的地址与版本。

[设备验证器](../../examples/mega_dsa_local_cp_validate.py) 将本地 LI selected sets 和 values 与 stock LI
逐项比较，按 sparse slot 顺序对齐 stock SFA，使用独立 FP32 attention oracle 检查每个本地 query，
并验证 selected descriptors、通信字节和阶段闭合。Framework-free CP1/CP2/CP4 launcher 覆盖 owner
置换、interior 和 first-sequence tile、空 owner/tile、H32/H64，以及 BF16/FP32 merge weights。
`--long-history` 额外覆盖 native K2048 截断和未拉取的 cold main-KV 行。

CANN 9.1 / Ascend 910B3 上的冻结载荷已通过 CP1/CP2/CP4 完整短、长序列 raw forward 验收，
每个 rank 各执行 320 次调用，合计 2240 次 rank-forward，三个 launcher 均正常退出。
独立 FP32 oracle 的最大 relative L2 为 `0.002419553899592081`，限值 `.02`。
标准构建成功，22 个 device objects 与冻结载荷相同；补齐独立冻结目录所需的相邻 SHMEM 载荷后，
打包载荷的 CP1 补充 smoke 正常退出，owner/empty tile 的四次 forward 均通过。

该 raw 接口尚未安装 KL 或 autograd。保存状态显式标为 local；现有 global-query backward 在提交前
拒绝该状态。训练路径迁移到 local query 前，仍需完成 local-query main backward 与 FP32 owner
accumulation 的独立数值验收，并接通 local LM+KL normalization 和 autograd。
当前已验收的产品训练路径仍使用 global query scope；
CP8/CP×TP、模型验收、稳定性、共存及完整训练步性能仍待验证。

## Local-query main backward 原始接口

[反向 adapter](fused_cp_backward.py) 新增 `backward_local(forward, grad_output)`，仅接收该 backend
产生的 local saved state。调用方的 BF16 cotangent 经保存的 query layout 打包，native 使用实际本地
query 行数和本次 forward 的显式序列位置；后续重新准备另一块 tile 不会替换这些位置或排列。
全局 query 的旧接口保持独立，local ABI 不可用时在申请 workspace lease 前拒绝调用。

Query scratch 按实际 query 行数计算，FP32 dK/dV 累加区和 partials 按完整 key 行数计算。
空 query 不调用非空 SFA child tiler，也不构造虚拟 query；它初始化完整 key 的零 partials，参与
initialize、compute、post/owner-return 三阶段闭合，并接收其他 rank 对该 owner 的梯度贡献。
Kernel 系统 workspace 与调用方算术 scratch 分别申请。结果持有 metadata，避免异步操作期间释放。

本版使用完整 key 的 owner 归还：每个 peer 的 FP32 dK/dV 先合并一次，再按 rank 顺序累加到 owner，
最后转换为 BF16 并恢复 owner 的输入排列。该原始接口不负责 autograd 或 KL；主注意力的
实验性 autograd 接口见下一节。该原始接口默认使用完整归还；后文分别说明 main 与 index-K
的可选稀疏归还。原始接口的证据不构成完整 local LM+KL 模型或训练步性能验收。

[设备验证器](../../examples/mega_dsa_local_cp_backward_validate.py) 比较 local 和 global-query
原始梯度、检查独立 FP32 oracle、精确 FP32 owner 累加顺序和字节数，并覆盖延迟 backward、重复
backward、跨 stream、owner/interior/first-sequence/empty tile。CP2/CP4 系统测试入口保持 framework-free。
独立 FP32 门槛为逐元素 `rtol=.02, atol=2e-5`；验证器记录所有 oracle 差异并将失败汇总为失败状态，
同时记录 global-query 基线的 oracle 误差，诊断信息不能替代验收。

当前 CPU 回归通过 186 tests、389 subtests，最终 native 构建成功。
CP1/CP2/CP4 每个 rank 各完成 144 次 backward，合计 1008 次调用、5040 个原始 native 梯度比较，
全部通过三阶段闭合和精确 FP32 owner 位模式检查；加载的候选库 SHA256 与冻结载荷一致。
独立 FP32 门槛仍失败，三个 runner 均返回非零。CP1 的 compressed-KV 有 108 个失败场景，
key RoPE 有 72 个，global-query 基线的对应失败场景数相同。CP2/CP4 也保留各 rank 的逐字段
误差和基线对照。Local backward 的独立数值验收仍未通过。

## BF16 主梯度的独立参考校准

[校准工具](../../examples/mega_dsa_grad_reference_calibrate.py) 只调用 stock SFA primitive，
不运行 local CP 候选。工具使用四个独立种子，其中最后一个仅用于验证前三个种子的观测范围；
覆盖 H32/H64、owner/interior/first-sequence/singleton/empty cotangent scope，分别保存原始 K/V
梯度、FP32 合并的 shared-C 梯度和 stock autograd 的 BF16 合并梯度。

CPU oracle 使用 FP64，独立计算 K/V 后相加，并与 shared-C oracle 比较。两个 oracle 的数学
等价性采用 `rtol=1e-11, atol=1e-12` 验证，避免把 FP32 加法顺序差异误判为数学错误。
BF16 比较仍记录原逐元素门槛以及 relative L2、cosine、max abs。若 oracle 梯度范数为零，
normalized metrics 不适用，观测范围只使用绝对误差；single-prefix 场景专门覆盖此情况。

当前独立校准完成 40 个场景、80 组重复观测，两个 FP64 oracle 全部一致，stock raw 与 autograd
在已有逐元素门槛下均一致。最后一个种子仅有 31/40 个字段落在前三个种子的全部观测范围内，
因此这些范围尚不能作为验收阈值。报告显式标为 `measured`、`accepted_tolerance_profile=false`，
不会改变候选的失败记录或自动激活默认载荷，也不能替代模型、长序列或完整训练验收。

## Local-query 主注意力 autograd 与 checkpoint

`FusedDsaCpAttentionProbe.attention_local()` 返回四个主输入的 first-order 梯度：local Q、
owner-local shared compressed-KV、local Q-RoPE、owner-local K-RoPE。Indexer 输入保持 detached，
不由主损失安装梯度路径；KL、auxiliary loss normalization 和完整模型集成仍需另行接通。

调用方显式传入 `prepare_local_queries()` 返回的 cold layout，避免 checkpoint 重算依赖 backend
后来准备的其他 tile。Backend 仅用弱引用登记已准备的 layout，拒绝其他 backend 或自行构造的
layout；原始 forward、saved backward 和重算均检查 metadata 地址及版本。

```python
query_layout = forward_backend.prepare_local_queries(local_query_ids)
attention = FusedDsaCpAttentionProbe(forward_backend)

def local_attention(q, c, q_rope, k_rope, *, saved_layout=query_layout,
                    saved_invocation=invocation, saved_index=index_states):
    # 显式保存本次 tile 身份，随后准备其他 tile 不会改变重算的排列或位置。
    return attention.attention_local(
        saved_invocation, (q, c, q_rope, k_rope), saved_index, query_layout=saved_layout)

output = checkpoint(local_attention, *main_states, use_reentrant=False)
```

Native 激活与原始输入通过 `save_for_backward()` 保存，参与 saved-tensor hooks 和版本检查。
空 query 的原始输出与 Native saved output 是同一个 Tensor；autograd bridge 返回独立的
detached Tensor 引用，共享存储而不复制数据，避免 PyTorch 给返回值安装 autograd metadata
时把 Native 输入标为 `requires_grad=True`。Native 保存的 output、max/sum 和输入仍保持 detached。

本轮 CPU 回归通过 191 tests、395 subtests，覆盖旧 layout 重算、重复 backward、空输出别名、
原始输入与 metadata 修改拒绝、first-order 限制、indexer 梯度隔离及旧 layout 的释放。
CP1 首轮 NPU smoke 的 owner/interior/first-sequence 场景通过，empty 场景暴露上述输出引用问题；
修正后的 CP1 smoke 通过。CP1/CP2/CP4 完整生命周期矩阵的七个 rank 每个各通过 24 个场景，
共 168 个场景、672 次 backward 和 2016 个主梯度比较；候选库 SHA256 与冻结载荷一致。
`--autograd-only` 仅验证主 autograd 生命周期，不清除独立数值门槛或模型验收失败。

## 按 CP query mask 冻结的参考数值契约

[CPU 参考精度契约](reference_precision.py) 对非零 oracle 梯度同时检查 relative L2、cosine error
和按 oracle RMS 归一化的 max abs；零范数使用单独的绝对误差规则，不使用人为截断的相对分母。
NaN/Inf、未校准的非零区间、修改后的策略或预算、缺失的 scope、重复样本，以及运行时版本或
参考库身份变化均拒绝验收。消费者重新拟合各字段预算并重新检查独立验证种子，不能只信任
producer 的 `accepted_tolerance_profile` 标志。

`mega_dsa_grad_reference_calibrate --cp-reference-profile` 使用新的三个校准种子和两个独立验证
种子，覆盖 H32/H64、CP1/CP2/CP4 各 rank、三种 owner 分布和五种 query mask，另随机化合法
selected slots 的排列。策略在采样前冻结：误差范数和 RMS max error 的 margin 为 `2`，
cosine error 对角度误差是二次量，margin 为 `4`；zero-oracle `atol=2e-5` 保持原有绝对门槛。
Oracle 始终使用 FP64，K/V 分别求导再合并与 shared-C oracle 一致。

本轮独立参考采集的 1050 个场景、210 个 scope bucket、840 个梯度字段均通过新的 held-out
验证，stock raw 与 autograd 对照一致。该契约仅覆盖 `(3,10)`、BF16、`H32/H64`、`K2048`
的声明合成分布；不接受长序列、其他数据分布或模型梯度。

候选验证器显式传入 `--reference-profile` 和独立保存的 `--profile-policy-sha256` 后才使用该契约。
同时检查各 peer 的 FP64 oracle 之和与 owner FP32 梯度：误差界包含独立参考的 partial budgets
和按 rank 顺序 FP32 相加的 `gamma(n)` 舍入项。Legacy pointwise 结果继续记录；默认验证入口
仍使用原有逐元素门槛，训练默认路径不因参考契约通过而自动迁移。
使用该契约的 CP1 矩阵已通过 144 次 backward、720 个 native 对照、576 个校准 partial
检查和 288 个 owner 误差界检查，实际加载库 SHA256 与冻结载荷一致。旧 pointwise gate 仍失败，
本次通过只证明上述短序列合成分布中的校准契约。补充 CP4 矩阵的四个 rank 均通过，各完成
144 次 backward，共通过 2880 个 native 对照、2304 个校准 partial 检查和 1152 个 owner
误差界检查。补充 CP2 矩阵则失败：rank 0 的 `H32/strided/owner` Q 字段在三种 schedule、
两次重复中均越界，RMS-normalized max abs 为 `0.08319747182188637`，冻结上限为
`0.07223937633149653`。其 Q 与 global-query native 对照的 max abs 为零；两 rank 的
1440 个 native 对照和 576 个 owner 误差界检查全部通过。参考包络对该输入的覆盖仍需诊断，
不扩大现有阈值，不清除失败记录。原设备上的 CP2 完整矩阵也已完成，在同样六个 Q 检查中
复现相同失败指标；两个物理设备组合的报告独立保存。原设备上的 CP4 任务仍在等待。

诊断复跑在同一冻结预算下分别记录 local 和 global-query primitive baseline 的校准指标与
位模式：失败 owner Q 的两条路径位精确一致，所有误差指标相同且同样越界。CPU FP64 oracle
最近 BF16 可表示梯度的 RMS-normalized max error 为 `0.024266288937042725`，低于冻结
上限，因此最终 BF16 转换本身不足以解释该失败。校准包络在这些原语输入上的覆盖仍未关闭。

校准工具支持 `--fit-seeds` 和 `--held-out-seeds`，以逗号分隔的整数列表指定完整采样计划。
两组必须非空、各自唯一且互不重叠；producer 在保存 policy 和运行设备采样前验证，consumer
按冻结的全部 seed/bucket 对重新验证覆盖，不依赖固定的场景数。Held-out 数据不参与拟合，
任一字段的独立验证失败都会拒绝 profile。

新的 v2 采样计划固定为 16 个 fit 种子和 8 个 held-out 种子，覆盖原有 210 个 bucket、5040
个场景；误差 margin、zero-oracle 门槛、输入分布和选择规则与 v1 相同。它只运行 stock
primitive，fixture、FP64 oracle 和测量函数与 v1 的源码 hash 一致。运行时 policy 与独立
保存的采样前 policy 逐字一致。v1 的候选失败报告继续保留；v2 须完成全部独立验证后才可用于
候选重放，不自动改变默认训练路径或模型验收。

v2 已完成 5040 个场景，840 个字段的全部 held-out 检查通过；consumer 重新拟合预算并
核对完整覆盖、采样前策略、核心 source 和实际 reference library hashes。使用该已验证
profile 的 CP1/CP2/CP4 候选重放全部通过：七个 rank 各完成 144 次 backward，合计
1008 次调用、5040 个 native 对照、4032 个校准 partial 检查和 2016 个 owner 误差界检查。
三阶段通信闭合、FP32 owner 位模式及实际加载库 SHA256 全部通过；v1 的 CP2 失败继续保留，
原逐元素 gate 仍失败。该验收只覆盖声明的短序列 BF16 分布，尚非完整模型或训练步验收。

硬件 worker 可通过 `HP_DSA_REFERENCE_PROFILE` 和 `HP_DSA_REFERENCE_POLICY_SHA256` 显式
指定独立校准档案及采样前保存的策略 commitment；必须同时提供。Worker 使用同一 consumer
重算预算、验证 scope 和运行环境。未提供时保留原逐元素检查，`autograd-only` 不接受数值
profile，也不改变其独立生命周期验收范围。

## Local-query KL 的 indexer 梯度归还与目标契约

`DsaCpLayout.reduce_local_index_gradients()` 接收 detached FP32 partials：caller-ordered local
index-Q、完整 global-packed index-K、caller-ordered local merge weight。仅 index-K 进入一次
owner reduce-scatter；Q 和 weight 保持本地 query 排列，支持完整 owner 集合、query subset、
空 tile 和空 owner。K 按声明的 owner KV 存储顺序恢复，并在求和后转换为原输入 dtype。
FP64 CPU oracle 保留 FP64 accumulation。该默认路径使用完整 K partial buffer；后文的
`sparse_index_return=True` 使用独立设备端稀疏归还。
Layer/microbatch/invocation 可以推进并复用同一 owner geometry；过时 heap generation、
其他 topology/layout ID 或 KV 存储排列仍会拒绝。
调用身份复用修正通过 CPU 回归和 CP2 HCCL 的 8 个场景、24 个字段位精确检查。

该归还接口不再乘 loss coefficient、auxiliary upstream scale 或除以 CP size。真正的 local
KL producer 应先以 `loss_coeff * normalization.local_sum_scale` 缩放本地有效 query 的 KL sum，
再在 first-order backward 乘一次 `grad_loss`，随后按上述接口归还 K。分母始终是 global valid
query 数，query tile 或空 rank 不改变它。声明为 average 的后续复制参数 reducer 则由
`DsaLossNormalization.reducer_divisor` 补偿。

[CPU 目标契约](../../../../../tests/ut/core/multicore/mega_dsa/test_local_training_contract.py) 将
CP1/CP2/CP4 的 local LM+KL contribution 与未分片 FP64 objective 比较，覆盖不等长、zigzag、
packed 边界、重排 sparse slots、空 rank、空 tile、sum/average reducer、非单位且负的 aux
upstream scale、`loss_coeff=0`、KL-only 和 freeze-main。比较全部主输入与 index/output projection
参数梯度，验证 detach-before-index-projection；另设置反例检测 local mean 平均及重复 CP 除法。

[设备归还验证器](../../../../../tests/torch/multicore/_test_mega_dsa_local_index_owner.py) 独立测试
真实 HCCL 的 FP32 K owner sum、BF16/FP32 目标 dtype、caller/owner 排列及空 rank 的集体参与。
CP1/CP2/CP4 的七个 rank 共通过 28 个场景、84 个字段的位精确比较，source SHA256 与冻结
清单一致。相关 CPU 回归通过 65 tests、59 subtests，新增代码通过 pylint 和 launcher import 检查。
本节实现的是 local KL 的归还基础及数学契约；raw local KL producer 与 local LM+KL autograd
见后续章节，不能将这些接口检查作为完整训练或模型数值验收。

## KL 的 FP32 index-K partial 导出

`MixedKlResult.key_partials` 从每次调用独立持有的 retained workspace 导出 FP32 `[N,128]`
view。`N` 为完整 key 数，main head 数由 producer 保存；decoder 检查 dtype、连续性、对齐和
精确容量，拒绝错误 geometry。View 保持原存储的生命周期；跨 stream 读取前需等待 submission
stream 的事件，保存不受后续 workspace 修改影响的梯度时需 clone 或 scale 到独立 Tensor。

该 plane 位于锁定非 deterministic KL kernel 的 FP32 scatter-add 区，post phase 从这里读取
并转换出原 BF16 index-K。接口导出转换前的数据，不把 BF16 输出转换回 FP32 冒充 partial。
主 backward 的 Native ABI 和已冻结载荷均未改动。

设备验证覆盖 H32/H64、BF16/FP32 merge weight、groups=1/2/7/19 和三次独立调用，共 48 个
场景、79872 个 key 元素。FP32 view 共享 retained 存储，该轮普通 Torch BF16 转换与 Native 输出
位精确一致，原 native/stock 及独立 oracle 检查均通过。后续 local-query 完整矩阵发现精确中点
不能用普通 Torch 转换验证：锁定 post kernel 使用 `RoundMode::CAST_ROUND`，中点向远离零的
方向舍入，而 Torch CPU/NPU 转换使用 ties-to-even。验证器通过
`mixed_kl_native_key_cast()` 对完成的 CPU FP32 快照解码 Native 转换；不改变 Native 算术或
训练 bridge 的 owner 求和及目标 dtype 转换。两个设备快照共 3328 个元素与该位参考完全一致，
包括原来普通 Torch 转换失败的两个负数精确中点。CPU 检查另覆盖正负中点、相邻位模式和 signed zero。
上述设备证据覆盖完整 query 的 standalone KL；local query producer 具有独立的验收入口，
local LM+KL autograd 的接口与 CPU 证据见后续章节，设备验收尚未完成。

## 实际 local-query 的 raw Native KL

`local_kl_gradients(forward)` 消费 `forward_local()` 产生的 invocation-owned saved state。
Main teacher 的 Q/c/RoPE 保持 detached；输入必须保留 forward 的 ownership、geometry、
index Tensor version 和冷准备 query layout。保存状态或静态位置被修改时，在提交 Native 前拒绝。
该接口不执行 Q/weight collective，也不从 backend 的最新 query tile 推导旧调用的位置。

新增 `HyperDsaLocalMixedKl` / `dsa_local_mixed_kl_out` 使用实际 local query 数、各 packed sequence
的真实累计 query 数，以及显式 sequence-relative query positions。没有 query 的 sequence 可以
重复累计数。锁定 KL kernel 的 compute 和 vector/scatter 均使用同一位置计算 causal prefix，
避免用 right-down 的 `L-M+i` 接纳 compact Top-K 后面的 `-1` slots。全 query ABI 保持独立。

`LocalKlResult` 持有每次调用独立的三阶段 trace、retained workspace、raw KL sum 和 index
derivatives。Q/merge-weight 梯度恢复 caller-local 顺序；K 输出及 FP32 partials 保持完整 global
packed 顺序。Retained 容量按全局 K 数分配，FP32 K 直接读取 native scatter-add plane。空 tile
不提交 KL arithmetic，返回真实零行 Q/weight 和独立的完整 FP32 零 K contribution；后续训练
bridge 仍须让空 rank 参与 owner reduction。Raw producer 不乘 loss coefficient、normalization
或 downstream cotangent。

CPU 准入、行序与所有权检查位于
[test_local_kl.py](../../../../../tests/ut/core/multicore/mega_dsa/test_local_kl.py)。设备入口为
[mega_dsa_local_kl_validate.py](../../examples/mega_dsa_local_kl_validate.py)，CP1/CP2/CP4 launcher
为 [test_mega_dsa_local_kl.py](../../../../../tests/torch/multicore/test_mega_dsa_local_kl.py)。设备检查
将 local subset 与独立 FP32 selected-KL oracle 及零化其他 query contribution 的全 query Native
baseline 比较，同时检查转换前 FP32 K、三阶段闭包、重复调用、空 tile 和重复累计 query 数。

当前完整 native 构建与 megaDSA CPU 回归已通过；raw local KL 的 CP1/CP2/CP4 完整矩阵通过
七个 rank 的 2016 个场景，每 rank 288 个。覆盖 H32/H64、BF16/FP32 merge weight、groups=1/7/19、
contiguous/strided/empty ownership、owner/interior/first-sequence/empty query tiles，以及保留两次
forward 后的独立校验。运行时加载的库、冻结载荷的 87 个文件及提交验收前的验证源码 SHA
均已核对。上述短 packed `(3,10)` 矩阵不建立长上下文或完整训练正确性。
该独立 producer 已接入下述 local LM+KL autograd；后文给出 `fused_kl=True` 的 forward
单 kernel task graph 及两条可选稀疏归还路径。长上下文、模型验收和 full-step 性能/HBM
证据分别记录。

## Local LM+KL 的 first-order 训练桥

`FusedDsaLocalCpTrainingProbe` 在冷准备阶段集体声明 `mode`、`loss_coeff` 和
`DsaLossNormalization`，拒绝不同 CP member 的 objective 契约不一致。支持 `lm_only`、
`kl_only` 和 `lm_plus_kl`。调用 `forward(invocation, main_states, index_states,
query_layout=...)` 返回 caller-ordered compressed output 与本 tile 的辅助 loss contribution。
Query layout 必须显式捕获，后续 tile 的准备不能改变 checkpoint replay 的行序或位置。

辅助分支在 forward 保存 `loss_coeff * normalization.local_sum_scale` 缩放后的三组独立
FP32 derivatives；K 取自 raw producer 的转换前 plane。Backward 乘一次 `grad_loss`，仅将
完整 global-packed K contribution 归还 owner，求和后转回原 dtype。分母始终是全局有效 query
数，后续复制参数的 sum/average reducer 由声明的 `reducer_divisor` 补偿，无额外 CP 除法。
Main 分支使用已有 local Native backward，KL teacher 与 hard selection 不进入 main VJP。
调用者在 index projection 前 detach hidden，以阻断 auxiliary 对共享 hidden 输入的梯度。

对于 `lm_plus_kl`，某 rank 的 main 或 auxiliary cotangent 缺失时，以本地零贡献参与对应的
owner return，仍可接收其他 rank 对 owner-local K/C/RoPE 的贡献。模式由所有 rank 冷声明；
`lm_only` 不计算 KL，`kl_only` 不提交 main backward。共同声明的 `loss_coeff=0` 跳过 KL
arithmetic、完整 FP32 auxiliary scratch 和 index-K exchange，返回 index 输入的零梯度。

七个原始输入、八个 Native saved activations 和三个缩放后的 KL derivatives 全部通过
`save_for_backward` 保存，支持 saved-tensor hooks、retained backward 与 non-reentrant
checkpoint；拒绝原输入/位置/geometry 修改以及 higher-order backward。后续调用无法覆盖
旧调用已保存的 K contribution。每个 rank 必须进入相同模式的 forward/backward 调用序列，
包括空 owner，并使用兼容的 requires-grad 策略。

[训练桥 CPU 检查](../../../../../tests/ut/core/multicore/mega_dsa/test_local_training.py) 覆盖三种
mode、BF16/FP32 merge weight、sum/average normalization、负 aux upstream、零 coefficient、
空 query、缺失本地 cotangent 时的远端 owner contribution、retained backward、checkpoint、
准入和版本保护。新增桥接后，全套 megaDSA CPU 回归通过 227 tests、514 subtests。
上述证据证明 Python 梯度桥的调用、生命周期与缩放契约；raw local KL、完整 LM+KL 的
完整 CP1/CP2/CP4 Native 数值和模型验证仍需完成，不构成训练性能或单 kernel 融合的验收。

## Local training 的 projection VJP 验收

[mega_dsa_local_training_validate.py](../../examples/mega_dsa_local_training_validate.py) 将真实
local training bridge 接至七个输入投影与 output projection。检查 shared hidden 和全部八个
projection 参数的梯度，并按声明的 sum/average reducer 归并 CP contribution。矩阵包括三种
mode、零 coefficient、负 upstream、query subset/empty owner、retained backward 和 checkpoint。
框架无关的 [CP launcher](../../../../../tests/torch/multicore/test_mega_dsa_local_training.py) 提供
CP1/CP2/CP4 入口。

[FP64 oracle](../../examples/mega_dsa_local_training_oracle.py) 固定 Native activation 与 compressed
output 的数值，独立求 linear projection、selected SFA/KL 的 Jacobian；这些固定值不替换
对应导数。此契约验证梯度桥和 projection VJP 的组合，不测 BF16 projection/output 的 forward
composition 误差，因此设备报告继续设置 `full_model_acceptance=False`。

在候选运行前冻结 pointwise `rtol=.02, atol=2e-5` 与非零 oracle 的 `relative_l2<=.02`，
要求二者同时通过。零 oracle 使用独立绝对规则；非有限值、缺失梯度及不应存在的梯度均拒绝。
[闭式 chain-rule CPU 检查](../../../../../tests/ut/core/multicore/mega_dsa/test_local_training_projection_oracle.py)
覆盖每个参数、hidden detach、padding、全局 normalization 和小非零梯度被绝对容差掩盖的反例。
全套 megaDSA CPU 回归通过 238 tests、540 subtests。

首轮 H32/FP32 weight、strided CP1/CP2/CP4、LM+KL smoke 均在第一场景未通过严格 pointwise gate：hidden
relative L2 为 `0.0012694554419907047`、max abs 为 `5.795180502238782e-05`；index-K projection
relative L2 为 `0.0019266434951632776`、max abs 为 `6.58863912048023e-05`。其余已比较字段通过，
失败报告保留。较小的整体 relative L2 不清除逐点失败，不能以该场景作为完整训练或模型通过证据。

CP1 的实际 activation 梯度与转换前 owner FP32 快照进一步分解了该失败。Hidden projection
矩阵乘的 max abs 误差为 `1.0158698472706451e-08`，最终 BF16 转换贡献的 max abs 为
`3.854824412469031e-05`，转换前 Native 误差的 max abs 为 `3.9305716180389194e-05`；
使用转换前梯度的闭式诊断对照通过逐点门槛。Index-K projection 矩阵乘误差为
`2.137980459337996e-09`，转换贡献为 `5.8836731359708516e-05`，转换前 Native 误差为
`3.866686460949568e-05`；转换前诊断对照仍未通过逐点门槛。这些诊断没有替换训练梯度，也
没有放宽门槛；仍需区分锁定基线误差与训练桥新增误差。

原始 raw KL 全矩阵的 CP1 曾在 217 个通过场景后被上述舍入校验假设拒绝。CP2/CP4 的原始
运行出现 HCCL collective 序列不一致：部分 rank 进入验证失败后的 close，其他 rank 继续
下一调用的冷准备。验证控制现通过
[collective_validation_check](../../examples/mega_dsa_validation_control.py) 汇总 rank-local
检查异常，在下一次 Native 调用或集体 teardown 前让全部 member 得到同一失败判定。该机制
只处理本地验证异常；HCCL 自身故障仍由 launcher/process group 恢复。原失败日志保留，
使用修正后的转换校验与异常同步重新执行完整 CP1/CP2/CP4 矩阵并通过。

同一 CP1 activation/output 快照另在完整 query 的 stock CANN SFA/KL 与原 mixed KL 路径上
执行诊断。Stock SFA forward 与捕获的 output 位精确一致；mixed KL 三组原始梯度与 stock
位精确一致，loss 最大差为 `1.1920928955078125e-07`。Local bridge 的 index-K owner FP32
与完整 query 的转换前梯度最大差为 `3.725290298461914e-09`；index-K projection 参数梯度
与该完整 query 基线的最大差为 `2.137980459337996e-09`。完整 query 基线对 FP64 的 index-K
projection 逐点检查同样失败，因此本快照的失败可在独立于 local bridge 的路径上复现。
Main hidden 的首轮 stock 对照将 K/V 各自转换 BF16 后相加，与 bridge 的 FP32 合并后一次
转换存在不同的舍入边界，不能据此证明两种 main VJP 契约等价。

进一步通过 `mixed_sfa_gradient_partials()` 解码独立完整 query mixed SFA reference 的转换前
FP32 dK/RoPE `[N,576]` 与 dV `[N,512]`。Decoder 按锁定 arch22 TND tiler 的每逻辑核
scratch、512-byte 对齐 dQ、dK 和 dV 顺序确定 offset；拒绝空 query、错误 geometry、dtype、
非连续/未对齐及截断存储。View 保持 retained workspace 的生命周期，消费前需遵守 producer
stream 完成顺序；不能从 BF16 输出回转来构造该 reference。

上述 CP1 快照使用 reference 的 FP32 dK+dV 合并和一次 BF16 转换后，bridge hidden 参数链的
最大差为 `1.0158698472706451e-08`；转换前 compressed owner 最大差为
`1.3969838619232178e-09`，key-RoPE owner 最大差为 `7.275957614183426e-12`。此独立
reference 的 hidden 与 index-K projection 对原 FP64 逐点门槛同样失败。该证据覆盖本次
同状态短序列诊断，不替代其他形状、CP topology 或完整模型的数值验收。

## 独立 projection reference 校准

[mega_dsa_projection_reference_calibrate.py](../../examples/mega_dsa_projection_reference_calibrate.py)
按实现计划的 reference-to-reference 口径，在采样前冻结 16 个拟合 seed、8 个独立 held-out
seed、normalized margin=2、cosine margin=4 和 zero-oracle `atol=2e-5`。数据来自完整 query
stock LI/SFA、独立 mixed SFA/KL 原语及固定 Native 数值的 FP64 projection oracle；不调用
local training bridge，候选 CP 数值不进入拟合或 held-out 验证。

校准范围为 packed `(3,10)`、H32/H64、BF16/FP32 merge weight、完整 global query、
`lm_only`/`kl_only`/`lm_plus_kl`、coefficient `.3` 和 auxiliary upstream `-2.5`。每 fixture
重复两次，main 采用转换前 FP32 dK+dV 合并后一次 BF16 转换，index-K 采用 FP32 scale、
upstream 后一次转换。Reference 保留原 pointwise 检查的每个结果，独立计算 relative L2、
cosine 和 RMS-scaled max abs envelope；held-out 数据不能扩大拟合出的预算。

两个独立非 deterministic backward 的原始梯度采用固定 `rtol=.02, atol=2e-5` 且
`relative_l2<=.02` 的数值一致性检查；同次调用的 FP32 plane→Native post 输出则使用字节级
校验，包含 signed zero。校准入口输出 policy SHA、完整样本、各字段预算、held-out 判定和
运行库 SHA，失败仍保存报告。该独立校准完成 576 份样本、12 个 H/weight/mode 组合，所有
held-out 字段均通过预先冻结的 envelope；原 pointwise 检查仍记录 360 个未通过字段。

Local training validator 可显式传入 `--reference-profile` 与独立冻结的
`--profile-policy-sha256`。CPU-only 的
[profile 准入](projection_precision.py) 重算全部拟合预算及 held-out 判定，核对完整
seed/repeat/field 覆盖、policy、Torch 版本和运行库 SHA，拒绝候选数据、缺失/重复样本、
预算改写或 held-out 失败。调用方另外检查完整 global query 覆盖、默认投影 geometry 和
已校准 objective；不覆盖的 subset/shape/objective 直接拒绝，不能静默套用预算。

显式 profile 模式报告 `calibrated_reference`，并独立保留 `independent_fp64_vjp` 与
`original_pointwise_passed`。未提供 profile 时继续使用原逐点门槛。Profile 的通过不能清除
原逐点失败，或完整模型与 BF16 forward composition 的未通过状态。

[完整 owner 矩阵入口](../../examples/mega_dsa_local_training_profile_validate.py) 覆盖 H32/H64、
BF16/FP32 weight、contiguous/strided/zigzag/empty ownership、三种 mode、sum/average reducer、
plain/retain_graph/checkpoint。各 rank 的 query 合集必须仍是完整 global query；subset 与其他
objective 不进入此 profile 的支持范围。

首轮 CP4 owner 矩阵在每 rank 210 个通过步骤后，H64/BF16、empty ownership 的 LM+KL 场景
触发 coupled objective envelope：误差 `2.135412876386189e-08`，相消后的目标值约
`0.003913455709050252`，normalized error 超过该字段预算；全部梯度字段均通过。原失败记录
保留。合成 scalar loss 的检查进一步使用独立 LM objective、KL loss 的 reference 预算，并
通过 triangle inequality 传播误差，加上两次 FP32 乘加与有序 peer 求和的 analytical gamma
bound。推断的 main component 和 auxiliary component 必须各自通过对应误差界；梯度预算
不变。报告同时保留原 coupled normalized 判定与 component bound，不能将该 scalar 规则用于
近零参数梯度。CPU 检查覆盖相消变号、component corruption、非有限值及预算保持不变。

上述完整 owner 矩阵随后在 CP1/CP2/CP4 全部通过：CP1 每 rank 144 步，CP2/CP4 每 rank
288 步，七个 rank 合计 1872 步。固定 reference profile、consumption policy、采样/候选源码和
运行库 SHA 均已核对，使用同一份未激活的冻结 Native 载荷。全套 CPU 回归通过 246 tests、
567 subtests；profile 准入、库身份、scalar cancellation 与反例检查通过。原 pointwise 失败和
首轮 scalar envelope 失败仍保留；这些结果只接受已声明的短序列 fixed-Native-value VJP
契约，不清除 subset、零 coefficient、长上下文、完整 BF16 模型或 F+B+SGD 的待验收范围。

## Actual local-query 的 LI/pull/SFA/KL 单 kernel

`forward_local(..., selected_kl=True)` 在已有 selected CP composite 中执行九阶段：LI main、
LI merge、membership init/build、request pack/pull、SFA、KL init/compute/post。Native host
单独传递实际 local query 的 INT64 cumulative lengths；LI/SFA 的 INT32 lengths、显式
sequence-relative positions 和 global query IDs 继续保留。KL compute 与 vector 使用同一
position override，不能把完整 key 的 cumulative lengths 当作 compact query 的长度。

每次调用独立持有 raw KL derivatives/loss、retained workspace 和转换前 FP32 global-packed
index-K view。Q/weight derivatives 恢复 caller 顺序。空 query 仍执行九阶段闭包与 ACK，但
不启动 KL 算术、不分配 KL scratch，返回真实零行 Q/weight 和独立 FP32 零 K contribution。

`FusedDsaLocalCpTrainingProbe(..., fused_kl=True)` 在冷准备阶段集体声明该选择；KL-enabled
forward 消费 composite 产生的导数，不再提交 standalone KL 三阶段。Coefficient、global
normalization 和 upstream 仍由原训练桥应用一次，index-K 求和与最终 dtype 转换契约不变。
`lm_only` 或零 coefficient 不执行 KL。默认独立提交路径继续作为显式对照，未启用 fused
模式时不会静默切换。Higher derivatives、完整模型、长上下文和性能仍需各自验收。

[mega_dsa_local_fused_training_validate.py](../../examples/mega_dsa_local_fused_training_validate.py)
比较单 kernel 与独立 local forward + raw KL，并检查 CPU oracle、FP32 plane→post 字节转换、
九阶段 device closure、selected 请求/字节与 ACK、空 tile 和重复调用。该入口不要求 selected
pull 与 LI 重叠：selected 请求依赖最终 Top-K，性能归因应独立采集。

该原始融合矩阵在 CP1/CP2/CP4 全部通过：每 rank 288 个 case，七个 rank 合计 2016 个
rank-case。覆盖 H32/H64、BF16/FP32 merge weight、三种 ownership、1/7/19 compute groups、
owner/interior/first-sequence/empty query tile 及两次调用。每次只有一个 Native submission；
九阶段闭包和 ready/ACK 均通过，运行库与冻结载荷的 SHA 已核对。原首轮 trace 验证入口缺少
`require_overlap` 参数的失败记录保留；修正验证调用后执行完整矩阵。

完整 owner 的 projection 验收可使用
[训练矩阵入口](../../examples/mega_dsa_local_training_profile_validate.py) 的 `--fused-kl`。
入口显式记录 `fused_kl` 和每次捕获的 forward 阶段数；启用 KL 的训练调用应为九阶段，
LM-only 和未启用 KL 的独立调用应为六阶段。需先在相同冻结载荷上重新运行独立 reference
校准，再传入其 profile 和 policy SHA；单次 submission 本身不构成完整训练步性能收益。

相同冻结载荷重新完成 576 样本、12 个精度桶的独立 reference 校准后，融合训练的完整
owner 矩阵在 CP1/CP2/CP4 全部通过：CP1 每 rank 144 步，CP2/CP4 每 rank 288 步，合计
1872 个 rank-step。Checkpoint 的两次启用 KL 调用均为九阶段；retain_graph 在中间另一
次六阶段 forward 后仍正确消费原 saved state。原 pointwise 失败独立保留，判定使用既定
reference profile 和 scalar component bound。旧全 query composite 另通过 48 个 case，
旧 selected-training 接口通过单卡 smoke。载荷保持隔离，未作为默认 active payload。

[mega_dsa_local_training_subset_validate.py](../../examples/mega_dsa_local_training_subset_validate.py)
另外比较 query subset 和零 coefficient 的完整参数 VJP。入口重建参与 query 的固定 Native
activation/output 数值，将未参与 query 的 Q、Q-RoPE、IQ、merge weight 与输出 cotangent
置零，在完整 query 的 stock SFA 和独立 mixed SFA/KL 原语上计算对照。Main dK+dV 和
index-K 均先使用 FP32，再按既定 owner 契约转换一次；index projection 继续与 hidden detach。

该入口只声明 packed `(3,10)` 的 Native-to-Native 参数链对照，保留 pointwise、relative L2
和梯度隔离检查；零 coefficient 的辅助 loss 与参数梯度额外要求精确零。它不消费完整 query
的 FP64 profile，不清除原 FP64 逐点失败，不接受 BF16 forward composition 或完整模型。
`--smoke` 覆盖 interior、first-sequence、零 coefficient、KL-only 零贡献和全空 checkpoint；
完整矩阵另覆盖两种 heads/weight dtype、四种 ownership、sum/average 和三种 replay。
该 Native-to-Native 入口的五个定向单卡 smoke 与 CP1/CP2/CP4 完整矩阵均通过：CP1 每
rank 1152 步，CP2/CP4 每 rank 2304 步，
合计 14976 个 rank-step。全部参数链、梯度隔离、零 coefficient 的精确零及 replay 检查
通过，源码和运行库 SHA 已核对。该结果接受所声明的 masked global Native 对照契约；
FP64 oracle、完整模型和 BF16 forward composition 的验收状态保持独立。

## Main gradient 的稀疏归还

`FusedDsaCpBackwardProbe(..., sparse_return=True)` 选择 Native gradient transport protocol 2，
支持 CP 最大为八，要求 `dsa_sparse_cp_grad_version()==1`。冷准备集体声明 mode；旧载荷或
声明改变在进入 Native lease 前拒绝。默认 protocol 1 继续返回完整梯度行。

Protocol 2 在 ready 前清零本 owner 上每个 sender 独占 inbox 的有效 main 区域，再对
FP32 合并后的 `[dc, dk_rope]` 逐行检测，仅省略数值全零的行。Sender 在传输结束后发布
每个 destination 的 INT64 行数；owner 仍按 sender `0..CP-1` 求和，最终由 Torch 转换
一次 BF16，发布 read-complete ACK 后才允许复用。省略的 signed-zero 行归一为零。

计数借用每个 sender stripe 首行的 index 预留区；不增加 symmetric heap allocation。
消息仍保持原来的 704 个 FP32 槽位，main 的 576 个槽位有效；index-K 没有加入该协议。
Trace 分开记录实际行 payload、counter 的逻辑控制字节、清零与 owner 扫描的 HBM 字节，
不能将 payload 减少解释为完整 backward 或 step 加速。全域 FP32 partial scratch 也仍保留。

`FusedDsaLocalCpTrainingProbe(..., sparse_main_return=True)` 可与 `fused_kl=True` 组合。该
选择参与冷准备声明和 forward 前的能力检查；main 的 owner 求和使用 protocol 2，而
index-K 默认继续完整 owner reduction，或通过后文的 `sparse_index_return=True` 切换归还。
训练矩阵入口提供 `--sparse-main-return`，仍需使用
当前载荷独立校准的 reference profile。

[mega_dsa_sparse_main_return_validate.py](../../examples/mega_dsa_sparse_main_return_validate.py)
对照同载荷的完整归还，并从同次 producer 的 FP32 dK/dV partials 独立计算固定 sender
顺序的 owner sum，要求转换前数值精确一致。额外检查实际 nonzero 行与传输计数、三阶段
闭包和 ACK、非零→零→非零复用。CP1/CP2/CP4 的定向 smoke 均通过，各 rank 12 个 case；
完整多组/ownership 矩阵及长历史 `(3,2113)` 也在 CP1/CP2/CP4 全部通过，各 rank 300 个
case，合计 2100 个 rank-case。覆盖 1/7/19 compute groups、H32/H64、四种 ownership 和
四种 query tile；长历史使用 H32/strided/7 groups。同次 producer 的转换前 FP32 owner sum
逐项数值精确一致，dense 对照、实际 nonzero payload 计数及复用均通过。

同一隔离载荷重新完成 576 样本、12 个精度桶的独立 reference 校准后，`fused_kl=True`
与 `sparse_main_return=True` 的完整 owner 融合训练矩阵在 CP1/CP2/CP4 全部通过。CP1
每 rank 144 步，CP2/CP4 每 rank 288 步，合计 1872 个 rank-step；源码、运行库 SHA、
forward 阶段数及原 pointwise 结果均已核对。该证据仍限定在短序列 fixed-Native-value
projection VJP；性能、完整模型和发布计划中的 8K/32K/128K 需要独立验收。

## Index-K gradient 的设备端稀疏归还

`FusedDsaLocalCpTrainingProbe(..., sparse_index_return=True)` 将已乘 coefficient、
normalization 和 upstream 的 global-packed FP32 index-K partials 交给独立 native
归还算子。每个 sender 只发送数值非零的 `[128]` 行；`+0`、`-0` 可省略，NaN/Inf
仍作为非零数据发送。owner 按 CP sender `0..CP-1` 顺序作 FP32 汇总，再由 Torch
转换一次到输入 dtype 并恢复 caller KV 顺序。该接口不再次缩放，不调用完整 K
reduce-scatter。默认路径继续使用已有完整归还。

该协议复用每个 sender 的 `[704]` FP32 inbox 行中最后 128 个元素；行计数位于
该 sender stripe 第一行的 main-gradient 区域。owner 在发布 ready 前清零有效 owner
行，sender 发布数据和计数后进入 write barrier，owner 汇总后通过 ACK 才允许复用。
main 与 index 归还必须以各 CP 成员一致的顺序执行，使用独立 epoch 和 protocol
identity。支持范围为 CP≤8；每个成员都参与，包括空 query 和空 owner。零 coefficient
在全体成员上没有 KL 贡献，保持现有直接返回零梯度的行为。

[mega_dsa_sparse_index_return_validate.py](../../examples/mega_dsa_sparse_index_return_validate.py)
以受控 FP32 partials 独立检查固定 sender 顺序的 owner sum、真实非零行与 512-byte
payload 计数、ready/write/ACK 和非零→零→非零复用。完整 owner 的独立 reference
矩阵提供 `--fused-kl`、`--sparse-main-return`、`--sparse-index-return` 三个开关；
subset/zero-coefficient 的 Native-to-Native 矩阵固定启用融合 KL，并提供两条稀疏归还开关。
正式冻结载荷包含 91 个文件，已完成 CP1/CP2/CP4 的回传矩阵：每 rank 80 个 case，
合计 560 个 rank-case。覆盖短 packed `(3,10)` 的四种 ownership、长历史 `(3,2113)`、
四种 query tile，以及受控 partials 的 signed zero/NaN/Inf 和真实融合 KL partials。
固定 sender 顺序的转换前 FP32 owner sum 逐项数值精确一致，main/index 交替复用、
实际非零行 payload 计数及 ready/write/ACK 均通过。trace 分别记录每行 512-byte 数据
payload 和逻辑 counter 控制字节；链路流量与延迟由独立性能测量记录。

同载荷完成 576 个样本、12 个精度桶的独立 projection reference 校准，保持原 consumption
policy。设备验收与源码/载荷 SHA 存于
`build/native/validation/mega-dsa-index-return-fixed-20261011/`。

同一载荷上，`fused_kl=True`、`sparse_main_return=True`、`sparse_index_return=True` 的
完整 owner 训练矩阵在 CP1/CP2/CP4 全部通过，合计 1872 个 rank-step；涵盖 H32/H64、
BF16/FP32 merge weight、四种 ownership、LM-only/KL-only/LM+KL、sum/average reducer
及 plain/retained/checkpoint。验收保留原 pointwise 结果，使用上述独立校准的 fixed-Native-value
projection VJP consumption policy；这不是完整 BF16 projection composition 验收。

另以 masked global Native VJP 校验 subset/zero-coefficient 矩阵，合计 14976 个 rank-step
全部通过；增加 coefficient `0/.3`、owner/interior/first_sequence/empty 四种 query tile，
并要求关闭的 auxiliary 梯度及 loss 精确为零。该矩阵使用 Native-to-Native 参考，
不借用完整 query 的 FP64 校准预算。Retained backward 插入一轮六阶段 forward 后仍消费
原保存状态；checkpoint 的两轮启用 KL forward 均为九阶段。

以上回传、校准与训练证据不建立完整模型、CP8、8K/32K/128K 或训练性能验收。

## CP 子组映射到共同 SHMEM root

`MegaDsaWorkspace(..., cp_group=cp_group)` 支持在覆盖 CP 成员的共同 root 上声明 DSA
arena。`DsaBatchMeta.cp_ranks` 使用 CP group 的有序 global ranks，`cp_rank` 是组内
ordinal，`root_pes` 由这些 global ranks 在 `root.members` 中的位置确定。Native forward、
main-gradient return 和 index-K return 在远端地址与传输拓扑查询前转换 CP ordinal，
owner inbox 的 sender 顺序继续使用 CP ordinal，不使用 root PE 作为 stripe 下标。

所有 root 成员必须以相同顺序声明同名 consumer、相同容量和布局，并参与 bind、对称
allocation/free 和 close。`dist.GroupMember.NON_GROUP_MEMBER` 成员保留 arena 和生命周期
责任，禁止 prepare/submit CP work。不同 CP 子组可使用同一声明的 arena；每个子组内部
须遵守一致的调用顺序，root 级 free/close 前须保证全部子组已完成。当前租约在同一进程
中串行交接，不提供跨 consumer 的并发执行或 heap growth。

映射协议要求 `dsa_cp_root_mapping_version()==1`，在申请 lease 前检查；旧载荷拒绝
mapped submission。原有 CP==root 的 18-word metadata 保留；映射路径使用 27-word
metadata，扩展内容为 root size 和八个 CP→root PE（未用槽位为 `-1`），支持 CP≤8。
Forward/index return 分别使用映射 protocol `2`，main dense/sparse return 分别为 `3/4`。

[mega_dsa_root_mapping_validate.py](../../examples/mega_dsa_root_mapping_validate.py)
已在四卡冻结载荷上完成 72 个 rank-case：CP `(1,3)` 与 root `(0,1,2,3)` 的非连续映射、
非参与成员的分配/释放，以及 CP `(1,3)` 与 `(0,2)` 两个子组分别执行。覆盖四种 query tile、
非零→零→非零复用、融合 LI/SFA/KL forward、main dense/sparse 对照和 index-K 稀疏归还。
每个 case 的转换前 FP32 owner sum 精确一致。验收与载荷身份位于
`build/native/validation/mega-dsa-root-mapping-20261011/`。

MegaMoE 的 shared-root adapter 当前要求 EP 与 root 的有序 membership 相同，因此该映射
为 CP2/EP4/root4 的真实共存提供 DSA 路径。EP 严格子组的 PE 转换、热副本远端地址转换、
CP8、CP×TP、完整模型和训练性能需要分别验收。

[真实 CP2/EP4 共存 worker](../../../../../tests/torch/multicore/_test_mapped_cp_ep_shared_root.py)
在同一冻结载荷上通过八种场景：MoE pull / 预留完整 lossless capacity 的 push、只运行
CP `(1,3)` / 同时分别运行两个 CP 子组，以及两种 consumer close 顺序。每步先保存两轮
MoE forward，再在另一 stream 执行融合 DSA LI/SFA/KL 和 main/index 归还，最后反序执行
MoE backward。MoE output、input/route/expert gradients 及 SGD 后参数与 common EP reference
对照通过。合计 1296 个融合 DSA rank-case、144 个 MoE F+B+SGD rank-step（288 对
forward/backward）；关闭一个 consumer 后，剩余 consumer 的实际执行仍通过。
每个场景的三次固定形状采样中，Torch allocated bytes 和外部 SHMEM heap bytes 保持一致；
分配计数遵守 `5→4/1→0`，最终 native reference count 归零。此短程 plateau 证据不替代
长步稳定性与完整模型训练。结果见同目录 `coexistence_audit.json`。

新载荷也重新通过原 18-word 协议的 CP1/CP2/CP4 回归，共 560 个 rank-case，覆盖
长历史 `(3,2113)`、signed zero/NaN/Inf、实际融合 KL partials 和 main/index 交替复用；
FP32 owner sum 与实际加载库 SHA 均已核对。

CP8 在同一冻结载荷上的 index-K 归还矩阵已通过：每 rank 80 个 case，合计 640 个
rank-case；短 packed `(3,10)` 的四种 ownership、长历史 `(3,2113)`、受控 signed zero/
NaN/Inf、真实融合 KL partials、main/index 交替复用均完成。转换前 FP32 index-K owner sum
逐项数值精确一致，实际加载库 SHA 匹配冻结载荷。证据见该目录 `cp8_index_audit.json`。
此 CP8 矩阵校验 main-gradient 的阶段/通信 trace；main-gradient 数值、训练 autograd 与
SGD、CP×TP 和长上下文性能还需各自完成验收。

## 映射 CP 子组的训练 autograd 与 SGD

[mega_dsa_mapped_training_validate.py](../../examples/mega_dsa_mapped_training_validate.py)
使用覆盖四个 root 成员的 workspace，只在 CP `(1,3)` 或两个独立 CP 子组中执行训练。
快照收集和 SGD 的梯度归约均使用 backend 的 CP group，非参与成员只加入 root 级准备、
结束确认与关闭。两个子组使用不同的 projection fixture seed，以检查各组独立的数据路径。

数值对照采用 masked global Native VJP，在固定 Native activation/output 值下检查 hidden
和全部八个 projection 参数的梯度；不使用完整 query 的 FP64 reference profile。随后执行
真实 CP SUM 与 downstream divisor 补偿，并以 `lr=1e-3` 做 SGD，检查归约值和更新后的
参数。HCCL 的梯度缓冲在提交前转为 contiguous。该 fixture 是有限维 projection 边界，
仍不建立完整 BF16 projection composition 或完整模型训练验收。

四卡 smoke 已通过 48 个 rank-step，涵盖 LM-only/KL-only/LM+KL、coefficient `0/.3`、
负 auxiliary upstream `-2.5`、sum/average reducer、四种 query tile、plain/retain_graph/
checkpoint。启用 KL 的 forward 为九阶段；retained backward 插入另一轮六阶段 forward，
checkpoint 的两次启用 KL forward 均为九阶段。关闭的 auxiliary 梯度精确为零，实际
SGD 检查通过。证据存于 `build/native/validation/mega-dsa-mapped-training-20261011/smoke_audit.json`。
完整 H32/H64、BF16/FP32 merge weight、四种 ownership 的 13824 个 rank-step 矩阵另行验收。

完整映射 CP2 训练矩阵现已通过，共 13824 个 rank-step；H32/H64、BF16/FP32 merge weight、
四种 ownership、三种 objective、coefficient `0/.3`、sum/average reducer、四种 query tile
以及 plain/retained/checkpoint 均完成。每个 case 的独立 masked global Native VJP 和实际
SGD 更新检查通过，关闭的 auxiliary 梯度精确为零，阶段数、源码及加载载荷 SHA 已核对。
完整证据见 `build/native/validation/mega-dsa-mapped-training-20261011/full_audit.json`。

CP8 main-gradient dense/sparse return 的完整数值矩阵也已通过：每 rank 300 个 case，合计
2400 个 rank-case，覆盖 H32/H64、四种 ownership、1/7/19 compute groups、四种 query tile、
非零→零→非零复用和长历史 `(3,2113)`。同次 producer 的转换前 FP32 owner sum 精确一致，
dense/sparse 对照和实际加载库 SHA 匹配；证据见 root-mapping 验收目录 `cp8_main_audit.json`。

## 真实 HP attention 的融合 CP 接线

[FusedDsaCpAttention](fused_model_boundary.py) 是显式 opt-in replacement，替换已构造的
`DeepseekV32DSAAttention`。它保留原 Parameters、children、state dict、训练状态、RoPE、
Q/KV latent projections、W_UK absorption、W_UV restoration 和 `o_proj`，不再次运行会创建
fused Parameter 的源构造器。模型仍由 caller 准备 workspace、training probe、invocation、
query layout、参数归约和 auxiliary upstream scale。

调用要求 `dsa_training`、`dsa_invocation` 和 `dsa_query_layout`；本阶段支持完整 owner-local
hidden shard，Q/KV 使用同一 caller row order。RoPE embeddings 必须对应这些 global token
positions。支持 H32/H64、Hi64、K2048、C512、Dr64、Di128；Native dtype/协议准入由显式
training probe 检查。TP 当前拒绝，training/freeze/coefficient gates 必须与冷准备的 probe
一致；KL 使用融合阶段。该边界保持 first-order backward 契约，无自动 backend 构造或
fallback。空 owner 仍提交 CP work，输出恢复使用显式维度，避免零行的 `reshape(...,-1)`。

CPU 测试通过真实 HP 投影与显式 oracle 检查：参数/child identity、checkpoint keys、完整
FP64 absorbed/unabsorbed output 与 hidden/全部参数梯度、freeze/zero/eval、checkpoint、
错误声明提前失败及空 owner 的执行接线。硬件验收由
[mega_dsa_fused_model_cp_validate.py](../../examples/mega_dsa_fused_model_cp_validate.py) 记录，
对照独立 stock LI/SFA/KL 的完整模型路径；CPU 接线验收不代表 BF16 模型验收。

真实 BF16 HP attention fixture 的 CP1/CP2 Native-to-Native 验收已通过，共 48 个模型
rank-case。使用独立 stock lightning indexer 重新计算选择，再运行 stock SFA/KL 和原模型
投影；逐项比较 caller-order output、KL、hidden、全部 11 个参数梯度、W_UK/W_UV，
并比较真实 SGD 后参数。原 pointwise `rtol=.02, atol=2e-5` 与非零 relative L2 `.02`
规则均通过，已比较字段的最大 relative L2 为 `0.005896421274430613`。
覆盖 joint plain/retained/checkpoint、KL-only、freeze、zero coefficient、eval、zero
auxiliary，以及 strided/empty ownership；CP2 的空 owner 仍完成执行与 backward。

硬件接线发现并修正了 inactive-input 的 autograd 边：`lm_only` 在进入 custom Function
前 detach 三个 index states，`kl_only` detach 四个 main states。原值仍供 LI 或 detached
teacher 使用，关闭分支的梯度保持 absent，且不会向上游 native normalization backward
提交 undefined cotangent。该修正不改变 active 分支的 Native 数值、归还协议或缩放。
模型证据和前两次接线失败记录位于 `build/native/validation/mega-dsa-fused-model-20261011/`，
审核结果为 `model_audit.json`。修正后的映射训练 smoke 另通过 48 个 rank-step；完整矩阵
使用新源码身份在 `build/native/validation/mega-dsa-inactive-branches-20261011/` 重新验收。

此模型证据是 13-token、hidden32、H32/Hi64 的真实 HP attention fixture 与独立 stock
BF16 模型对照；它不清除历史 FP32 model-composition 验收的失败，也不代表完整 LM、
预训练 checkpoint、H64/H128、CP×TP、长上下文或训练性能验收。

inactive-input 接线修正后的完整映射训练矩阵也已通过 13824 个 rank-step，阶段图、
Native VJP、实际 SGD、源码/加载载荷 SHA 均匹配，结果见新验收目录 `full_audit.json`。
旧矩阵和旧模型验收保留各自源码身份，不混用运行期间修改的验证脚本。

[mega_dsa_mapped_model_validate.py](../../examples/mega_dsa_mapped_model_validate.py) 将真实
attention 对照扩到四卡共同 root 的 CP `(1,3)` 和 `(0,2)`。模型快照、数值检查和 SGD
归约使用 CP group，root 全员负责 bind、完成/异常确认和 close。覆盖只有一个子组参与
和两个子组分别执行、strided/empty ownership；两个子组使用不同的模型权重，以检查
独立数据路径。四卡硬件矩阵现已通过 96 个模型 rank-case；独立 stock 的选择集合、
output/KL、hidden/全部参数梯度、W_UK/W_UV 与实际 SGD 更新均通过原数值规则。
已比较字段的最大 relative L2 为 `0.0059023659705557344`。启用 KL 的 plain forward
为九阶段，retained/checkpoint 两次 forward 均为九阶段，关闭 KL 为六阶段。源码身份、
冻结载荷和实际加载库 SHA 已核对；证据为
`build/native/validation/mega-dsa-mapped-model-20261011/model_audit.json`。
该结果仍限于 13-token、hidden32、H32/Hi64 的 BF16 attention fixture，不代表完整 LM、
独立 FP32 composition、CP×TP、长上下文、模型与 MoE 共存或性能验收。

## TP head 汇总边界

[DsaTpHeadCollectives](tp_collectives.py) 提供显式 Torch TP group 的数学/通信边界，
暂未接入 native fused CP worker。准备阶段检查 TP peers 的 ordered query/candidate IDs、
packed lengths 和 invocation 声明一致；CP group ranks 可以不同。执行不 all-gather heads。

- `index_scores` 在每个候选 tile 内计算 local-head ReLU 与原模型的 signed merge weight，
  以 FP32 SUM 汇总 TP score 后才交给全域 Top-K。`sum_selected_scores` 为每个 query 的
  selected slots 提供同样的汇总与 VJP；TP peers 必须使用完全相同的 slot 顺序。
- `teacher` 先逐 main head softmax，再 TP SUM head probabilities，最后 L1 normalize；
  teacher detach，padding/跨序列/未来位置由 caller 提供的 validity 排除，空行贡献零。
- score SUM 的 backward 是 identity：各 TP rank 对同一 replicated KL objective 各自
  backward 一次，不再 SUM cotangent 或除 TP size。原 CP reducer divisor、loss coefficient
  和 auxiliary upstream scale 各执行一次；该边界不自动改变 CP loss 契约。
- `sum_shared_gradients` 在 CP owner return 后，将 compressed KV、K RoPE、index K 等
  shared-field partials 打包为一个 FP32 TP SUM，再由 caller 接回原投影并 cast。None 保持
  absent；所有 TP peers 的 presence mask 与 backward schedule 必须一致。

[mega_dsa_tp_collectives_validate.py](../../examples/mega_dsa_tp_collectives_validate.py)
用四进程 Gloo 验证 CP `(0,2)/(1,3)` × TP `(0,1)/(2,3)`，真实按 token/head 切分输入，
逐 tile 汇总 score 后维护 exact Top-K；对照未分片 FP64 oracle 的选择、output/KL 和
全部七类输入梯度。矩阵包含 strided/empty ownership、candidate tile `1/2/7`、
coefficient `0/.37`、CP reducer divisor `1/2` 和 upstream `-2.5`，共 96 个 rank-case
通过，数值阈值为 `rtol=1e-10, atol=1e-12`。独立单测另覆盖“各局部 Top-K 都未选择、
head SUM 后却成为全局最佳”的反例，以及不同 head 数的 teacher 汇总顺序。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m hyper_parallel.core.multicore.examples.mega_dsa_tp_collectives_validate \
  --output /tmp/mega-dsa-cp2-tp2
```

审核结果在 `build/native/validation/mega-dsa-tp-collectives-20261011/collectives_audit.json`；
这是 CPU FP64 通信与数学证据。native LI 的 pre-Top-K score 交换、native KL 的 teacher
交换、真实 TP 模型接线和 NPU CP×TP 验收仍待实现；`FusedDsaCpAttention` 继续拒绝 TP。

### Native TP tile 传输 probe

[TpTileSumProbe](tp_tile_sum.py) 对应独立算子 `HyperDsaTpTileSum`，用于先验收 LI score 和
KL teacher 交换所需的 bounded FP32 SUM 协议。它使用包含所有参与组的 SHMEM root，
支持显式 TP subgroup-to-root PE mapping；inactive root 成员保持相同的 reserve/bind/free
顺序。该 probe 当前独立于 LI/SFA/KL，不构成 native CP×TP attention 接线。

每个 runnable vector slot 持有一个 source tile 与十条独立 control cache line：ready、
geometry/invocation signature 和最多八个读者 ACK。slot 数最多 40，width 为八对齐的
`8..2048` FP32 元素。slot 在下一轮写入前等待上一轮所有 ACK，peer 按固定 TP 顺序
累加；最后一轮也等待所有读者结束。scratch 仅随 slots/width 增长，与 scan rounds
无关；identity 包含 invocation、layer、microbatch、heap generation 和 ordered-row signature。

准备后的 metadata 检查 Tensor version，单个 arena 的 epoch 单调递增，修改 metadata、
重用旧 epoch、错误 dtype/shape、active autograd input 或缺少载荷 capability 都提前拒绝。
运行取得 shared-root lease，并在当前 stream 记录完成事件。saved activation/output
由 caller 持有；关闭时 root 全员同步并释放一个 arena。

[mega_dsa_tp_tile_sum_validate.py](../../examples/mega_dsa_tp_tile_sum_validate.py) 准备四卡
TP1、单个 `(1,3)` subgroup、两个 independent subgroups 和 TP4 矩阵，包含 slots/width
`(1,8)/(7,512)/(40,2048)`、rounds `1/17/257`、跨 stream 的 nonzero→zero→nonzero
复用、NaN/Inf 与最终 ready/ACK 检查。对照为固定 TP 顺序的 CPU FP32 SUM，阈值
`rtol=0, atol=0, equal_nan=True`。矩阵的运行状态与冻结载荷记录在
`build/native/validation/mega-dsa-tp-tile-sum-20261011/`。

完整 native 构建及四卡传输矩阵已通过，共 378 个 rank-case；所有输出、zero replay 和
最终 ready/ACK 检查通过，固定顺序 CPU FP32 SUM 精确一致。源码、95 个冻结载荷文件
以及实际加载库 SHA 已核对，审核结果为 `tile_sum_audit.json`。构建使用同一套 CANN
built-in 文件的本地 OPP 视图，首轮全局 vendor configuration 权限失败与首次 worker
导入路径错误均单独归档。CPU megaDSA/SHMEM 回归通过 314 个测试、668 个 subtest。

同一新载荷的真实 BF16 attention CP subgroup 回归任务独立记录在 `model_regression/`，
现已通过 96 个模型 rank-case，原数值规则、选择集合、SGD、阶段数和载荷身份均匹配；
审核结果为 `model_regression_audit.json`，对应 `25afe218` 的冻结传输载荷。以上传输通过
不建立 native LI pre-Top-K/KL teacher 的集成验收，
也不建立真实模型 CP×TP、完整 LM、长上下文数值或性能验收。

### Native LI 的 pre-Top-K TP 汇总

[TpLocalIndexerProbe](tp_indexer.py) 将 TP head SUM 接到锁定 LI vector tile 的
`DoReduce` 与排序之间。原 Hi64 按 TP 分片，各 shard 补零到 native H64；只复制 caller
原有的 signed BF16/FP32 merge weight，不增加 head/TP divisor 或其他缩放。每个候选
tile 先通过 SHMEM 累加所有 TP partial scores，再进入原 Sort/Top-K 和 LD merge。
不使用 local Top-K union，也不 all-gather heads 或完整 score 表。

TP 传输状态持续跨越该 physical vector 的全部 logical tickets 和 query/key tiles；
其 round 声明是上界。结束时只等待实际发布的最后一轮 ACK，零轮 slot 无须等待一个
不存在的 ready。source/control arena 随 compute vector 数量增长，width 固定 512。
原 mixed/local LI entry 的签名保持兼容，新增 `dsa_tp_local_indexer_out` 使用显式 arena
和 metadata；其他 composite LI closures 保留不启用 TP hook 的路径。

准备阶段要求完整有序的 replicated global index K，按 global query IDs 准备 compact
Q 顺序、真实 packed cumulative lengths 和 sequence-relative positions，并在 TP group
内核对同一 query/key/invocation 声明。CP membership 可以不同。输出将 native 的
sequence-relative IDs 转成 `global_packed`，再恢复 caller 的 query row order。准备 tensor
修改、旧 epoch、错误布局或缺少载荷 capability 都在提交前拒绝。空 owner 仍执行完整
phase closure，保存 selection/output 与 reusable scratch 分离。

[mega_dsa_tp_indexer_validate.py](../../examples/mega_dsa_tp_indexer_validate.py) 对照独立
stock full-head LI，覆盖 CP4×TP1、CP2×TP2、CP1×TP4、strided/interior/empty query tiles、
BF16/FP32 signed merge weights、1/7/19 compute groups 和重复执行。长历史使用 `(3,2113)`；
4097-key 反例中，一个候选在两个 head shards 都排在 local K2048 之外，head SUM 后
却成为全局第一。每个 case 检查选择集合、BF16 values、global causal/packed IDs、
phase graph 和最终 ACK；需要 LD 证据的条件使用实际 query 的因果长度。

新载荷完整 native 构建已通过，四卡 smoke 的 48 个 rank-case 已通过 stock 选择/值精确
对照与协议检查；CPU megaDSA/SHMEM 回归通过 318 个测试、668 个 subtest。完整矩阵
也已通过 864 个 rank-case，每 rank 216 个，所有 stock 选择集合和 BF16 values 精确一致；
global causal/packed IDs、两阶段 closure、active slot 与最终 ACK 均通过。同载荷的
tile SUM 回归通过 378 个 rank-case。审核文件为 `full_audit.json` 和
`tile_regression_audit.json`，证据目录为
`build/native/validation/mega-dsa-tp-indexer-20261011/`。这仍是使用 replicated global K
的 native selection probe；TP KL teacher/predictor 汇总、共享梯度 TP 归还、真实模型
CP×TP 和训练性能尚待接线验收。H64 补零是明确的原型计算开销，未建立性能收益。

验证过程中保留了两类失败：LD 证据的触发条件起初使用全局长度，现改为当前 query
tile 的实际因果长度；原 4097-key 反例有大量并列分数，TP-disabled local LI 与 full-Q
stock 也选出不同 ID，但选中分数是精确 Top-K，4096 号全局第一仍正确。隔离诊断存于
`tie_diagnosis/`，原 worker/report 存于 `failed-tied-stock-selection/`。严格集合对照
使用精确二进制低位打破并列的反例：原 merge weights 保持不变，两个 local ranks 均为
2049，全局 rank 为 1，全部 4097 个 FP32 global scores 唯一；证明存于
`counterexample_proof.json`。该通过不建立并列边界与 stock 的 ID 集合严格等价契约。

本次 LI 载荷的真实 BF16 attention CP subgroup 回归另排队验收，记录为
`model_regression/`；不能以先前 `25afe218` 载荷的 96 个模型 case 替代。

### Native selected-KL 的 TP 汇总接线

[TpLocalKlProbe](tp_kl.py) 提供独立的 TP selected-KL primitive。主 attention heads
均匀分片，本地 H32/H64；原 Hi64 的 index query 和已经模型缩放的 signed merge weight
按 TP 分片后补零到 native Hi64。共享 compressed K、K RoPE 和 index K 使用完整
`global_packed` 顺序，CP fetch/复制由调用方先完成。

锁定 KL vector 的 teacher 路径先按各本地主 head 的 saved maximum/denominator
计算概率，再通过 bounded SHMEM slot 汇总 TP 概率，除以全局主 head 数。
predictor 路径先汇总 TP index-head logits，再执行选定集合上的 softmax。两个 vector
分别负责 teacher/predictor，使用独立 slot；传输状态跨越同一 physical vector 的全部
logical tickets，退出前等待实际发布轮次的 reader ACK。每个 slot 宽度固定为 K2048，
与序列长度无关。其他 composite KL closures 暂时不启用该 TP hook。

调用分为 initialize、带 TP 汇总的 compute、post 三个 native phase。返回的 index-Q
和 merge-weight 导数恢复 caller 行序和本地 head 数，index-K 的 FP32 partials 在归还
owner 前保留。该 primitive 返回未经模型 coefficient、CP loss normalization、upstream
cotangent 缩放的局部 VJP 和 TP 复制的 KL loss；predictor SUM 的 backward 为 identity，
不能再次汇总 cotangent 或额外除以 TP。共享 key 的 CP/TP 归约及最后一次 cast 尚待训练
桥接接线。

准备阶段验证 TP query/key 行声明、head geometry、invocation 和 heap generation，冻结
selection/config 的版本。调用方必须使用同一 TP LI producer 的相同有序 selected IDs，
每行包含完整的 `min(causal_length,2048)` 有效项；本接口不在热路径把 selection 读回
host 做数值检查。空 query 保留真实空行并返回零共享 key partials。

当前 CPU MegaDSA/SHMEM 回归为 **330 tests、668 subtests**，包括新 TP KL 与 TP owner
return 的各四个契约测试。完整 native 构建已通过并冻结 95 个载荷文件；TP KL 的四卡
NPU smoke 已完成 **48 rank cases**，覆盖 CP4×TP1、CP2×TP2、CP1×TP4，对比 full-head
stock KL 和独立 FP32 VJP，最大 oracle relative L2 为 **0.012041410369068542**。首轮完整矩阵
在 CP1×TP4、lengths=(3,2113)、groups=19、BF16 merge weights 的 case 出现数值失败：
梯度和 loss 同时偏离 stock 与 FP32 oracle，phase/ACK 通过。失败证据保存在
`build/native/validation/mega-dsa-tp-kl-20261011/failed-tp4-long-g19/`；正在核查 TP
有序 selected IDs 和对应计算，验收阈值保持不变。不能据此宣称 CP×TP 模型、完整 LM
或性能已通过。新的 LI frozen payload 已完成原有四卡有序 CP subgroup 模型回归：
**96 rank cases**，覆盖 stock 比较、selected set、SGD 与 phase 数量，报告及载荷哈希已核对；
证据保存在 `build/native/validation/mega-dsa-tp-indexer-20261011/model_regression_audit.json`。
该回归仍是 TP=1 的 BF16 attention fixture。

### CP owner FP32 梯度之后的 native TP 返回边界

[TpOwnerGradientReturnProbe](tp_owner_return.py) 接受已经完成 CP owner return 的
C `[N,512]`、K-RoPE `[N,64]`、index-K `[N,128]` FP32 梯度。C 必须已经合并 dK+dV，
每个字段均包含自己的 coefficient、CP normalization 和 upstream cotangent。它把三个
active 字段按固定顺序打包到普通 FP32 HBM tile，通过已有 width-512 bounded SHMEM
slot 执行一次 ordered TP SUM，随后才转换为原投影输入 dtype，不增加 TP/CP divisor。

冷准备声明 TP peer 的相同 caller owner-global-ID 行序、branch presence、invocation 和
heap generation；其他 CP group 的 ranks 可以不同。缺失分支用 `None`，空 owner 保留
真实零行并执行相同 epoch 协议。执行入口拒绝已经转为 BF16 的 partials，从而保留
FP32 正负贡献抵消后的小残差。它返回独立 owner FP32 sums 及最终转换的 gradients。

CPU 契约已验证多轮 packing、caller owner 行序、缺失分支、空 owner、旧 ABI 拒绝和
先求和后 cast 的抵消反例。新的四卡验证程序
[mega_dsa_tp_owner_return_validate.py](../../examples/mega_dsa_tp_owner_return_validate.py)
覆盖 TP1/2/4、slots=1/14/38、rows=0/5/17/2113，以及 main/index/全部缺失分支，检查
ordered FP32 SUM、最终 cast 与 final ACK。该矩阵已通过 **576 rank cases**，实际加载库
和 frozen payload 哈希已核对。它验证的是已经 CP-returned FP32 字段
进入 TP 返回边界的协议和 packing；当前输入是可精确审计的梯度 fixture，尚不能视为
native CP return 与 TP return 已在真实模型 backward 中联合通过。训练调用层仍待接线。

### Native CP index-K return 与 TP shared return 的联合调用

[CpTpSharedGradientReturnProbe](cp_tp_return.py) 在冷准备阶段同时核对 CP group 的
branch presence / invocation 和 TP group 的相同 owner-global-ID 行序。执行前检查所有
main owner FP32 字段、global-packed index-K partial 和 native ABI；index 分支启用时
先执行 `SparseIndexGradReturnProbe.return_owner`，随后将其 FP32 owner 输出与 C/K-RoPE
一起交给 `TpOwnerGradientReturnProbe`。两个 native transport 保持同一 submission
stream 依赖，最后一次 input dtype 转换发生在 TP SUM 之后。index 分支缺失时所有 CP
members 都跳过该 CP return，不改变 main 字段的 TP 返回。

新增 CPU 契约覆盖 CP → TP 调用顺序、combined dK+dV / index-K 各入 TP 一次、branch
缺失、不同 root 拒绝、CP 分支声明不一致，以及旧 TP ABI 在 CP 发布前拒绝。四卡程序
[mega_dsa_cp_tp_return_validate.py](../../examples/mega_dsa_cp_tp_return_validate.py)
已排队，将真实调用 mapped native CP index-K return 和 native TP SUM，覆盖 CP4×TP1、
CP2×TP2、CP1×TP4，反向 caller owner 行序、zero → nonzero 复用及缺失 index 分支。
该联合 smoke 已通过 **36 rank cases**，实际 mapped native CP index return → native TP
SUM 的 receipt、ordered FP32 值、最终 cast 与 ACK 均已核对。完整联合矩阵已完成 **1296 rank cases** 的数值与协议核查；
index partial 与 main owner 字段仍是可审计梯度 fixture，不能视为 native KL、main CP
backward 或真实模型 CP×TP 已联合验收。

上述 TP KL smoke 与 TP owner return 的审计分别保存在
`build/native/validation/mega-dsa-tp-kl-20261011/smoke_audit.json` 和
`build/native/validation/mega-dsa-tp-kl-20261011/owner_return_audit.json`。验收严格使用同一
95-file frozen payload；构建时的 README 已另存，更新后的说明不覆盖原证据。完整 TP KL
首轮矩阵的失败记录仍保留；后续两次按原同步方式完整重跑均通过 864 rank cases。新的 KL frozen payload 的 TP=1 BF16 模型 CP 回归已完成并审计
**96 rank cases**，包含 selected set、梯度、SGD 和 phase 数量；证据保存在
`build/native/validation/mega-dsa-tp-kl-20261011/model_regression_audit.json`。该结果
不覆盖 TP KL 或 CP×TP 模型。联合 CP → TP return 完整矩阵的 1296 个 rank case
已通过数值、receipt、cast 与 ACK 核查；其启动前 Python manifest 早于融合桥接编辑，
审计明确标记 `source_identity_verified=False`，不能宣称完整 source identity 验收。

### 融合 CP forward/training 内的 TP LI/KL hook

新的 `dsa_tp_local_cp_training_out` / `aclnnHyperDsaTpLocalCpTraining` 在现有 local CP
融合入口添加独立 LI/KL TP arena 和 metadata，支持 `with_kl=False` 的六阶段 forward
以及 `with_kl=True` 的九阶段训练计算。LI score SUM 位于每个候选 tile 的排序之前；
KL teacher 概率和 predictor logits 的 SUM 位于各自归一化之前。通信状态保持跨 logical
tickets，phase release 前等待实际发布轮次的 ACK。coordinator 不占 TP compute slot。

TP LI/KL 必须使用相同 epoch、TP/root 映射和 invocation/row signature，KL transport
仅在 KL 启用时出现。其他 CP 入口通过缺省空 TP inputs 保留原调用 ABI。CPP tiling
schema 新增两个 arena byte 字段，嵌套 KL、with-KL 和新增尾字段的写入偏移已对应更新。

该路径的完整 native 构建已通过，载荷冻结在
`build/native/validation/mega-dsa-tp-fused-cp-20261011/payload`。当前仅建立编译与 ABI
证据，Python 的共同 TP invocation 与单 root lease 已接线，模型训练接线仍待完成。新融合
载荷的 NPU 数值 smoke 已通过 **48 rank cases**，包含 CP4×TP1、CP2×TP2、CP1×TP4
的六/九阶段调用、完整因果 selected set、有序 TP selection、stock 输出和 FP32 KL VJP
比较；审计见 `build/native/validation/mega-dsa-tp-fused-cp-20261011/smoke_audit.json`。已有 TP KL / CP → TP return 矩阵继续使用先前冻结的独立
载荷，其源文件和 compiled source closure 已另存，后续编译不会替换该证据。

### 融合 CP×TP 的 Python 提交边界

[FusedDsaCpTpForwardProbe](fused_cp_tp.py) 组合已有 local CP probe 与预声明的 TP LI/KL
consumer。冷准备要求 TP peers 的逻辑 Q 行、CP ordinal、owner offsets、owner-local KV
行序、head geometry、invocation、heap generation 和 KL branch 一致；不同 CP group
可以有不同的 global rank 名称。两个 TP transport 使用同一 row signature 和 epoch，
分别保留 width-512 LI 和 width-2048 KL 的有界 source/control arena。

执行时原 Hi64 按 TP 分头，内部补零到 native Hi64，保留 signed merge weights 和原
scale。一次 CP workspace lease 包含 CP owner publication、LI TP SUM、selected KV pull、
SFA 和可选 KL。`consume_in_root_lease` 只在同 root 的活动 owner lease 内消耗 TP epoch；
不增加第二个 lease，CP completion event 保护本次 kernel 引用的全部 arena。返回时恢复
caller 行序和原本地 index heads，保留 FP32 index-K partials，最后归还仍由独立 backward
调用层负责。

本轮 MegaDSA/SHMEM CPU 回归已通过 **335 tests、677 subtests**，Ruff、Pylint、文档
catalog 与 diff whitespace 检查通过。CPU 契约覆盖六/九阶段的一次提交、真实空 Q、
monotonic TP epoch、metadata 修改和旧 ABI
在发布前拒绝，以及 TP2/4/8 在 BF16/FP32 signed weights 下的 padding 和 VJP head 恢复。
四卡验证程序 [mega_dsa_fused_cp_tp_validate.py](../../examples/mega_dsa_fused_cp_tp_validate.py)
检查 CP4×TP1、CP2×TP2、CP1×TP4 的融合输出、selected-KL stock/FP32 比较和 phase closure。
该 smoke 仅覆盖 detached primitive，尚不证明 CP×TP 模型 backward、SGD 或性能。

TP KL 定向复现增加计算前的 selection host 同步后，失败配置的 8 个 rank case 通过，
有序 TP selected IDs 一致。原失败仍保留。验证程序通过显式 `--check-tp-selection` 区分
该诊断同步；缺省保留原执行同步方式。按原方式重跑的完整矩阵已通过 **864 rank cases**，报告明确记录
`pre_kl_selection_sync=False`；审计在
`build/native/validation/mega-dsa-tp-kl-20261011/full_recheck_audit.json`。原失败根因尚未
解释，本次完整通过仍不能作为原数值问题已修复或稳定性验收完成的依据。

### CP×TP 的一阶训练桥接

[FusedDsaLocalCpTpTrainingProbe](local_tp_training.py) 将融合 local CP×TP forward 的原始
index VJP 接入 Torch autograd。主损失只返回 Q/C/Q-RoPE/K-RoPE 梯度，selected KL 只返回
IQ/IK/merge-weight 梯度；调用方仍需在 index 投影前 detach 主 hidden states。KL 的
coefficient、CP normalization 和 upstream cotangent 各应用一次，TP replicated loss
不增加 divisor，也不再次 SUM 局部 IQ/weight cotangent。

Main backward 使用 `backward_local(..., defer_owner_cast=True)` 保留 combined dK+dV
和 K-RoPE 的 CP owner FP32 sum。Index-K 的已缩放 global-packed FP32 partial 经 native
CP return 后，与两个 main 字段一起执行一次 owner TP SUM，再转换为输入投影 dtype。
retained backward 每次准备新 owner TP epoch。缺失 upstream cotangent 使用零贡献并保持
声明的 CP/TP 调用顺序；lm-only、kl-only 和零 coefficient 隔离 inactive 分支。

本轮 CPU MegaDSA/SHMEM 回归通过 **340 tests、680 subtests**。测试覆盖 retained backward、
七组 VJP 的缩放/分支隔离、修改输入和高阶梯度拒绝，以及 shared owner cast deferral。
独立 full-head FP32 oracle 的六个完整 F+B fixture 也已核查（H64/H128 × 三种 objective）。
Ruff、Pylint 与 diff whitespace 检查通过。

四卡 [mega_dsa_cp_tp_training_validate.py](../../examples/mega_dsa_cp_tp_training_validate.py)
已通过并审计 **144 rank cases**，覆盖 CP4×TP1、CP2×TP2、CP1×TP4，strided/empty owner、
三种 objective 与两次 F+B+SGD。独立 FP32 七组梯度、loss/output、实际参数更新、native CP
main/index receipt、forward phase closure 与最终 TP reader ACK 均已核查；审计见
`build/native/validation/mega-dsa-tp-fused-cp-20261011/training_audit.json`。此阶段优化对象是七组投影后 primitive 输入，
尚不能视为真实模型 TP 投影、完整 LM 或性能验收。

融合 forward/KL smoke 首轮因验证器给完整 KV oracle 传入 owner-local KV 声明而终止；
原报告与启动源码快照保存在 `mega-dsa-tp-fused-cp-20261011/failed-oracle-kv-declaration/`。
修正验证调用的 KV namespace 后，使用相同 frozen native payload 与阈值重跑已通过
**48 rank cases**，实际加载库、95-file payload 和启动源码快照已核查。

### 真实 attention 的 TP 投影分片与 checkpoint 边界

[FusedDsaCpTpAttention](fused_tp_model_boundary.py) 消费尚未分片的 HP attention，在 optimizer、
DTensor/FSDP wrapping 之前，根据显式 `dsa_tp_group` 对五个投影权重进行真实 storage 分片：
Q-B、KV-B、IQ、merge-weight 按行分片，O 按列分片；对应 row bias 同步分片，O bias 保持
复制并在输出 TP SUM 后只加一次。Parameter 对象与 HP state_dict keys 保留，全局 heads
和原 Hi64 scaling 不变。QKV latent、norms、index-K projection 仍为复制参数。

[tp_projections.py](tp_projections.py) 从 TP 首个 member 同步初始参数值；Q latent 到本地
main heads 的输入 VJP 做 TP SUM，O 的复制输出 SUM 使用 identity VJP。共享 KV/index-K
梯度由 native training bridge 的 CP → TP return 完成，投影侧不重复 SUM。空 owner 使用
显式恢复维度。调用方负责 CP 参数初始化与梯度归约、统一 forward/backward schedule、
root lifetime 和 optimizer/FSDP 的后续接线。

`load_dsa_tp_state_dict` 将已经转换为 HP keys 的全局 checkpoint 切成当前 rank 的局部
权重，`gather_dsa_tp_state_dict` 在 TP group 上汇总还原全局 HP tensors。当前不把这些
接口宣称为已经接入 trainer 的分布式 checkpoint。准备阶段拒绝已有分片、梯度/hook、
wrapped 参数、不支持的几何，以及 head 参数与其他字段的 tied-storage 关系。

当前 MegaDSA/SHMEM CPU 回归通过 **345 tests、684 subtests**，Ruff、Pylint、文档 catalog
及 diff whitespace 检查通过。CPU 契约覆盖 TP2/4 的实际 storage 收缩、Parameter identity、
全局 load/gather round trip、
input-SUM/output-identity VJP、原 index scale、空 owner 与 O bias 梯度。两进程 Gloo 的
[mega_dsa_tp_model_cpu_validate.py](../../examples/mega_dsa_tp_model_cpu_validate.py) 另通过
**8 rank cases**：H128、Hi64、CP1×TP2，joint/freeze/zero/eval，对比独立 unabsorbed FP64
模型的输出、hidden 与全部参数梯度、SGD 后全局 checkpoint。最大绝对误差约
`8.05e-16`，审计见 `mega-dsa-tp-fused-cp-20261011/cpu_model_audit.json`。其稀疏计算为
显式 CPU execution double，尚不证明 NPU 模型精度、标准 hidden/Q-LoRA/V 维度、完整 LM
或性能。真实 NPU 模型仍待验证。

TP KL 在缺省原同步方式下的第二次完整矩阵也通过 **864 rank cases**，两次 fresh-process
完整通过的审计分别为 `full_recheck_audit.json` 和 `full_recheck_repeat_audit.json`。原失败
记录保留，根因尚未解释，不能将稳定性复验通过描述为已经修复该数值问题。

### CP×TP 真实 attention 的 NPU 模型验收入口

[mega_dsa_cp_tp_model_validate.py](../../examples/mega_dsa_cp_tp_model_validate.py)
把实际 HP 投影分片、融合 LI/SFA/KL、CP owner return 和 TP shared-gradient SUM
接入同一次 attention F+B+SGD。四卡矩阵为 CP4×TP1（全局 H64）、CP2×TP2
和 CP1×TP4（全局 H128），原 Hi64；每个拓扑覆盖交错 owner 和空 owner。
默认检查 joint plain/retain_graph/non-reentrant checkpoint，以及 freeze/zero/eval；
`--smoke` 先检查每个拓扑与 owner 模式的 joint plain，共 24 个 rank cases。

该入口分别比较独立 stock BF16 模型和 CPU FP32 unabsorbed 模型的输出、hidden、
全部 11 个参数梯度与 KL。参数梯度只在 CP group 内 FP32 SUM 一次，再执行 SGD；
通过显式 TP checkpoint gather 检查全局权重。模型使用 hidden32/Q-LoRA64/V16、
packed `(3, 10)`，因此不能替代标准 checkpoint、长序列、完整 LM 或性能验收。
所有字段沿用 `projection_vjp_metrics` 的既有 pointwise 和非零 relative L2 门槛；
stock 与 FP32 结果分别记录，数值失败保存输入、初始权重、索引和对应梯度，
不通过放宽门槛获得通过结论。设备结果仍须检查运行时 source snapshot、冻结 native
payload、实际加载库、前后向阶段和 owner ACK 后才能视为验收证据。

本轮 NPU smoke 已完成并审计 **24 个 rank cases**，结果为
`numerical_mismatch`，**0/24** 通过完整模型门槛。冻结载荷、加载库与运行时源码快照
身份已核对；三个拓扑、两个 owner 模式覆盖完整，selected sets、前后向阶段与 CP/TP
owner ACK 均通过。stock 对照的部分主分支输出或参数梯度未过 pointwise 门槛；
独立 FP32 对照中的 index-K、index norm 梯度相对 L2 约 5%–10%。所有 24 个 case
的 SGD 全局 checkpoint 与 stock 对照通过，但其中的 FP32 index norm bias 更新仍
有失败，不能凭 stock checkpoint 通过宣称模型数值已通过。

对应证据位于 `build/native/validation/mega-dsa-tp-fused-cp-20261011/` 的
`model_smoke/`、`model_smoke_audit.json`。首轮验收程序 CP 参数归约缺少连续性处理的
错误已修正，原日志与源码快照保留在 `failed-model-cp-gradient-contiguity/`。
[mega_dsa_cp_tp_model_diagnose.py](../../examples/mega_dsa_cp_tp_model_diagnose.py)
离线读取失败快照，分别比较 native/stock/FP32；`model_diagnosis/diagnosis.json`
与 `model_diagnosis_audit.json` 覆盖 24 份快照。stock 参考也有对应的 FP32 偏差；
三个拓扑的 CPU FP32 absorbed/unabsorbed 公式控制均通过，最大相对 L2
`9.490527503491542e-07`。这些证据不能证明具体数值根因，也不消除 native/stock
的 pointwise 失败。真实模型仍未验收通过，后续必须进一步定位投影、稀疏计算与归约。

完整模型矩阵也已执行并审计：**144 个 rank cases，完整数值通过 0/144**。
`model_full/` 与 `model_full_audit.json` 记录 joint plain/retained/checkpoint 和
freeze/zero/eval 的全部覆盖。所有 case 的 selected sets、原生前后向阶段、CP owner
归还与 TP ACK 均通过；**24/24 retained cases 的重复梯度逐位一致**。
SGD 全局 checkpoint 对 stock 通过 **144/144**，对独立 FP32 通过 **72/144**；
实际模型的 stock/FP32 输出或梯度门槛均未全部通过。因此这些记录证明声明矩阵内的
执行与保存状态回放，仍不证明完整模型数值验收、标准模型维度或训练性能。

首轮完整矩阵曾出现 rank 阶段分叉：rank 3 已进入 SHMEM free，其他 ranks 停在归约后
的设备同步；释放 barrier 也报错。该自有任务已终止，栈、日志及源码快照保存在
`failed-model-retained-collective-order/`，最初异常与根因未能确认。验收入口随后将
retained 梯度的本地严格比较改为记录结果，避免本地数值断言提前跳过后续 collectives；
在 SHMEM 释放前先保存异常。完整重跑无阶段分叉，retained 比较通过，仍不据此宣称
首轮分叉根因已修复。下一步需在相同投影状态与 cotangent 下进一步定位数值差异。

### CP×TP 实际状态的固定 cotangent 分层回放

[mega_dsa_cp_tp_boundary_capture.py](../../examples/mega_dsa_cp_tp_boundary_capture.py)
复用模型 smoke，在七组稀疏输入与压缩输出上注册显式诊断 hooks，保存进入实际融合
算子的值和实际 backward cotangent。快照全部 clone 到 CPU；该入口增加了同步，
不作为训练热路径或性能证据。24 份快照位于同一冻结载荷目录的
`model_boundary_capture/`，模型仍为原有 H64/H128、Hi64、hidden32/Q-LoRA64/V16。

[mega_dsa_boundary_snapshot.py](../../examples/mega_dsa_boundary_snapshot.py)
按显式 owner global IDs 和 TP head shard 重组快照，保留重复副本的差异记录，不再次
累加已完成 CP/TP 归约的梯度；缺失 token/head/参数分片直接报错。相关 CPU 测试覆盖
交错乱序 owner、空 owner、不同副本、row/column 参数分片和缺失分片。当前完整
MegaDSA/SHMEM CPU 回归通过 **349 tests、684 subtests**。

[mega_dsa_cp_tp_boundary_replay.py](../../examples/mega_dsa_cp_tp_boundary_replay.py)
分别支持 CPU oracle 与单卡 stock NPU 的固定状态回放，同时拆出固定稀疏 cotangent
的投影 VJP，以及固定压缩输入和输出 cotangent 的 value/O 投影 VJP。
`boundary_replay_cpu_exact/replay.json` 与 `boundary_replay_cpu_audit.json` 已核对源码
快照、冻结 payload 和 24 份采集文件 hash。六个全局拓扑/owner case 均有以下证据：

- Native 与 stock 的七组投影值、压缩输出及其 cotangent **逐位一致**；TP 副本、
  shared 梯度副本与 TP ordered selection 一致。
- 固定真实 Native 状态与 cotangent 后，七组输入梯度和 KL 对 CPU FP32 oracle
  通过既有门槛；压缩输出仍有 stock 同样的 pointwise 失败，relative L2 约 0.158%。
- 投影 VJP 和 value/O 投影回放仍有部分 pointwise 失败；其 normalized error 较小，
  不据此放宽门槛，也不把该结果视为完整模型通过。
- BF16 与 FP32 投影后，合法候选出现 4 或 8 处 ReLU 符号变化。保持 FP32 前向值、
  FP32 teacher 和投影 Jacobian，只把 ReLU backward branch 替换为实际 BF16 投影
  所决定的分支，index norm bias relative L2 从约 8.7%–9.7% 降到 0.63%–0.81%，
  前向 loss 逐位不变。该 counterfactual 解释了大部分 indexer 梯度差异；它不是生产
  precision policy，不能替代修复，也不能证明所有数值失败都有唯一相同根因。

上述证据区分了实际稀疏算子、非光滑投影组合以及投影/参数归约精度边界。
完整模型验收仍维持 **0/144**；后续 precision policy 或实现修改必须重新通过原有
模型矩阵，并另行验证标准模型维度和训练性能。

单卡 stock NPU 的固定状态回放也已完成并审计 **6/6 个全局 cases**，输入来自同一
24 份实际 CP×TP 快照。`boundary_replay_npu/replay.json` 与
`boundary_replay_npu_audit.json` 核对了快照 hash、运行时源码快照、当前源码、冻结
payload 和报告的实际加载库。在完全相同的实际投影值与 sparse-output cotangent 下，
压缩输出与 stock **逐位一致**，KL 与七组输入 VJP 全部通过既有门槛。该结论覆盖
实际稀疏算子边界，仍不能替代模型端输出、全部参数梯度或完整训练的验收。

首轮单卡回放使用了未带 ordinal 的 `npu` 布局设备声明，未通过 selection 的严格
设备检查；回放入口改为显式 `npu:0` 后重跑完成。失败日志与源码快照保留在
`failed-boundary-replay-unindexed-device/`，没有修改 reference 的设备契约。

### 完整全局模型的 reference-only BF16 校准

原研究计划 §10.1 要求先采集 reference-to-reference 误差，为 BF16 建立 normalized
门槛，不能对所有接近零的梯度统一指定 rtol。已有统一 pointwise 模型检查的
**0/144** 结果继续保留；它不能替代这项校准前置条件，也不能凭已有 candidate
误差反向拟合模型容差。

[model_reference_precision.py](model_reference_precision.py) 新增完整全局模型 profile 的
固定策略、逐字段拟合、独立 held-out 检查和重算校验。profile 只覆盖 packed `(3, 10)`、
H64/H128、Hi64、hidden32/Q-LoRA64/V16、K2048、joint/freeze/zero/eval，检查 output、
hidden、KL、全部 11 个参数梯度和 11 个 SGD 参数，共 25 个字段。owner-local tensor
和部分 head/参数 shard 不能使用该 profile；消费入口要求完整全局 shape、固定环境
和预留的独立 candidate seed。未经验证、被修改预算或梯度隔离改变的 profile 会被拒绝。

[mega_dsa_model_reference_calibrate.py](../../examples/mega_dsa_model_reference_calibrate.py)
仅采集完整全局 stock BF16 模型与独立 FP32 unabsorbed 模型，不调用融合 candidate。
拟合种子为 `20261101..20261116`，held-out 为 `20261131..20261138`，各重复两次。
`20261151..20261154` 仅预留给新的 candidate，不参与拟合或 held-out。这些整数是 RNG
seed。策略在设备采样前写入 policy JSON 和 SHA256，margin 为 `1.25`，cosine 使用
平方 margin；zero oracle 的绝对上限为 `2e-5`。原有默认模型 fixture 权重与已归档
基线逐位一致，新增显式 seed 参数仅用于独立采样。

本轮已完成并审计 **192 reference cases、384 samples、0 candidate samples**。
每份样本的 selected-set 与独立 FP32 absorbed/unabsorbed 公式检查均通过，源码快照、
冻结 payload、实际报告的加载库及拟合边界重算均通过。结果为 **`held_out_failed`**：
只有 `H128/joint` 的 `sgd/indexer.weights_proj.weight` 在 seed `20261136` 的两个重复
样本中超出边界，其他 199 个 bucket/field 组合通过。具体数值为：

| 指标 | 冻结上限 | held-out 实测 |
|---|---:|---:|
| relative L2 | `6.464746175296766e-05` | `6.485053464618428e-05` |
| cosine error | `2.0587892501922056e-09` | `2.084665884360959e-09` |
| max abs / RMS | `0.0004158845309730718` | `0.0003805172035307357` |

profile 的 `accepted_tolerance_profile=False`，不能用于放行 candidate，margin 未改变。
该记录位于冻结载荷目录的 `model_reference_calibration/` 与
`model_reference_calibration_audit.json`；首轮 JSON list 未转为 metadata tuple 的入口
错误保留在 `failed-model-reference-length-contract/`。后续校准或精度实现方案必须保留
这次失败，使用新的预声明独立样本，不能把该 held-out seed 加入拟合集来消除失败。

### CP×TP 延迟反向与梯度累积

[mega_dsa_cp_tp_accumulation_validate.py](../../examples/mega_dsa_cp_tp_accumulation_validate.py)
复用原生训练 validator 的 CP/TP 拓扑和声明，新增两个不同 microbatch 共享同一组七个
投影输出参数的实际 `.backward()` 累积与 SGD。两个前向全部完成后，分别以顺序
`(0, 1)` 和逆序 `(1, 0)` 反向，覆盖第二次前向覆盖 scratch 后的保存状态复用。
每个 microbatch 都使用独立的 `microbatch` 和 `invocation` 身份；owner TP epoch 按
反向执行次序递增，不能由保存的前向 epoch 推导。

两个 microbatch 均为 packed `(3, 10)`，主分支 cotangent 各乘 `0.5`，KL coefficient
保持 `0.3`、CP normalization 保持全局 query 数 `13`，auxiliary upstream 各为
`-1.75 / 2`。独立未分片 FP32 oracle 对两个目标分别求梯度再求和，不增加 TP divisor。
验收保留既有 primitive 门槛：输出、KL 和梯度的 relative L2 `<= 0.02` 或 max abs
`<= 2e-5`；SGD 参数使用 pointwise `rtol=0.02, atol=2e-5`。另外逐位检查实际 BF16
叶子累积等于两个已返回 BF16 VJP 按反向次序相加，避免只验证手工 FP32 累加。

四卡已通过 **144 rank-cases、288 次 microbatch 执行**：CP4×TP1、CP2×TP2、CP1×TP4，
strided/empty 所有权，lm_only/kl_only/lm_plus_kl 和两个反向次序全部覆盖。selected-set、
前向 phase、主分支 CP 返回、index CP 返回与每次共享梯度 TP reader ACK 均已核对。
全部对照字段的最大 relative L2 为 `0.0034958310625605674`。运行时源码快照与当前
源码、冻结 native payload、报告的实际加载库 SHA256 均验证一致。证据归档为
`accumulation/rank{0,1,2,3}.json`、`accumulation_cpu.log` 和 `accumulation_audit.json`。

该结果覆盖 BF16 投影输出叶子参数，两个 microbatch 的 query 数相同。实际模型的
全部共享投影权重梯度、不同 query 数的全局 normalization、AMP/PP、多步 HBM 和
完整 LM 性能仍需独立验收。模型级 **0/144** 和 reference profile 的
**`held_out_failed`** 结果保持不变。

### 实际模型共享参数的延迟反向、累积与 checkpoint

[mega_dsa_cp_tp_model_accumulation_validate.py](../../examples/mega_dsa_cp_tp_model_accumulation_validate.py)
将上面的七组投影输出叶子验证推进到真实 HP attention 的全部 11 个共享参数。
两个不同 hidden/cotangent microbatch 使用相同模型权重和 packed `(3, 10)`，先完成
两个前向，再按顺序、逆序或 checkpoint 重算后的逆序执行实际 `.backward()`。
主目标与 auxiliary upstream 均按两个 microbatch 的平均目标缩放，不增加 TP divisor。
参数叶子在 BF16 下实际累积，随后进行一次 FP32 CP SUM、转回参数 dtype 和 SGD；
TP head-sharded 参数只按其 head 归属比较，不再额外 TP SUM。

完整矩阵包含 CP4×TP1、CP2×TP2、CP1×TP4，strided/empty 和 joint/freeze/zero/eval，
每个组合包含 ordered/reverse/checkpoint 三种执行方式。首轮完成 **288 rank-cases、
576 个 microbatch**。原生 CP 返回与 TP reader ACK、参数身份、selected-set、checkpoint
的 microbatch 身份及新 invocation epoch 检查全部通过；这些协议检查不能替代数值验收。

另以两个分别执行的原生 microbatch 为对照，逐位检查 output、hidden 梯度和全部
参数的 BF16 累积。首轮 **220/288** 通过该额外逐位检查；68 个 case 的
`linear_qkv.weight` 不逐位相同，其中 56 个 case 同时涉及 `hidden/0`，8 个涉及
`kv_a_layernorm.weight`。这些差异全部通过已有数值门槛，最大 relative L2 分别约为
`7.97154e-5`、`2.71494e-4`、`3.05896e-5`。该观察本身不能证明 saved-state 损坏，
也不能把逐位失败改写为逐位通过。

与完整 stock/独立 FP32 模型比较，累积后的参数梯度仍有失败，完整数值验收为
**0/288**。SGD checkpoint 对 stock 为 **288/288**，对 FP32 为 **216/288**；后者的
72 个 joint case 均涉及 `indexer.k_norm.bias`。原有模型 **0/144** 和 reference-only
校准 **`held_out_failed`** 保留，本轮未使用失败的 profile 放行模型。

首轮运行时源码、冻结 payload 和报告的实际加载库经 SHA256 核对，证据为
`model_accumulation/rank{0,1,2,3}.json`、每个失败 case 的 `.pt` 快照和
`model_accumulation_audit.json`。运行源码快照保留在 `model_accumulation/runtime_sources/`；
后续验证器增加的独立重复对照不改写这份历史运行证据。

[mega_dsa_cp_tp_model_accumulation_repeat.py](../../examples/mega_dsa_cp_tp_model_accumulation_repeat.py)
进一步对 CP4×TP1、CP1×TP4 的 joint/reverse 和 strided/empty 做独立原生重复对照。
该聚焦矩阵完成 **16 rank-cases、32 对相同输入的独立 microbatch 重复**。延迟累积
逐位检查为 **15/16**；没有延迟反向的相同输入重复也在 CP4×TP1/empty/rank0 的
microbatch0 出现一次 `linear_qkv.weight` 差异，relative L2 为
`6.88700007086614e-5`、max abs 为 `7.62939453125e-6`，通过已有数值门槛。
这证明至少该字段的微小差异并非延迟反向所独有，不能仅凭逐位失败诊断保存状态损坏。
该控制没有复现 hidden/KV norm 的逐位差异，不能据此给全部失败指定唯一根因。
完整模型数值检查仍为 **0/16**，原 288 个 case 的结果不合并、不改写。

控制的独立源码快照、冻结 payload、报告的加载库和全部 case 覆盖已经核对，归档为
`model_accumulation_repeat/` 与 `model_accumulation_repeat_audit.json`。两个 NPU 任务均已
结束；非零退出来自归档的数值验收失败，协议运行与证据审计完成。

### O 投影在 TP SUM 之前保留 FP32

[fused_tp_model_boundary.py](fused_tp_model_boundary.py) 修正低精度 O 投影的提前舍入：
此前各 TP rank 先产生 BF16 局部 O 输出，再将这些已经舍入的值转为 FP32 SUM。
现在局部 O linear、TP SUM 和仅添加一次的 replicated bias 保留 FP32，最后转回
hidden activation dtype。FP32/FP64 输入保留原 dtype；局部区域显式关闭 autocast，
避免外层 AMP 再次把局部乘积降为 BF16。原 Parameter 对象和 checkpoint key 不变，
output SUM 的 VJP 仍为 identity，没有新的 TP divisor。

CPU 单测增加一个可精确计算的跨 rank 消去反例：局部乘积 `1 + 2**-8` 在 BF16 下
提前舍入为 `1`，与另一 rank 的 `-1` 相加会丢失小残差。FP32 局部乘积能保留
`2**-8`，bias 和全部 VJP 也按未分片目标计算；该测试同时覆盖外层 autocast。
MegaDSA 目录 CPU 回归为 **322 tests、687 subtests** 通过；两进程实际 TP2 投影
FP64 模型的 output、hidden、全部参数梯度与 global SGD checkpoint 为 **8/8 cases**
通过，最大绝对误差 `8.049116928532385e-16`。这些 CPU 证据不替代原生验收。

修改后重跑原有四卡完整模型矩阵 **144 rank-cases**，输出对完整 stock BF16 模型
全部 **逐位一致**，所有 case 的输出 max abs 均为 `0`。原有输出检查为 60/144
通过，新结果为 144/144，具体拓扑如下：

| 拓扑 | 修改前 stock 输出通过 | 修改后 stock 输出通过 | 修改前最大 relative L2 | 修改后最大 relative L2 |
|---|---:|---:|---:|---:|
| CP4×TP1 | 48/48 | 48/48 | 0 | 0 |
| CP2×TP2 | 12/48 | 48/48 | `0.00377995322638909` | 0 |
| CP1×TP4 | 0/48 | 48/48 | `0.0028984529412815767` | 0 |

每个拓扑/所有权组合抽取 rank0/joint/plain 的旧、新快照，共六对，确认 global
初始权重、hidden、RoPE embeddings、cotangent 和 selected-set 均相同。修改前完整
运行之后新增的 reference profile 模块未参与本轮验收，也没有使用失败的 profile。
完整模型数值验收仍为 **0/144**：stock 参数梯度的 `linear_qkv.weight`、部分
`kv_b_proj.weight`/`o_proj.weight` 失败继续保留，FP32 全模型门槛未改变。retained
反向的额外逐位检查为 **20/24**，四个 CP1×TP4/empty case 的 hidden/linear_qkv
存在微小差异，均通过既有数值门槛；原先 24/24 的历史记录不改写。

另重跑实际模型累积矩阵 **288 rank-cases、576 microbatches**，协议、参数身份、
selected-set 和 checkpoint 身份/epoch 检查全部通过。独立原生对照的额外逐位累积
检查为 **229/288**；这次计数变化不作为稳定性修复结论。完整模型数值验收仍为
**0/288**，SGD 对 stock 为 **288/288**、对 FP32 为 **216/288**。

归档为 `output_fp32_model/`、`output_fp32_model_accumulation/`、两者的独立审计 JSON、
`output_fp32_comparison.json`、`cpu_output_fp32_audit.json` 和 `output_fp32_cpu_all.log`。
两次原生矩阵完整执行后均因数值门槛失败退出；失败快照、源码快照、冻结 payload
与报告的加载库身份保留并核对。FP32 临时张量的 HBM 和整步性能尚未验收，不能把
消除输出误差写成性能收益或完整模型通过。

还对 Q latent 的 TP VJP 做了一个独立 CPU 消去反例，归档为
`query_latent_precision_cpu.json`：BF16 局部 VJP 先舍入、再 FP32 SUM 得到 `0`，
先保留 FP32 局部 VJP 再 SUM 得到正确的 `2**-8`，两个前向 projection 值逐位相同。
该记录仅为 CPU mocked-collective 证据；Q 投影的生产路径本轮未修改。

### 保留 BF16 Q-B 前向，局部 input VJP 在 TP SUM 前保留 FP32

[query_projection.py](query_projection.py) 新增一阶 Q-B projection autograd bridge，
由 [fused_tp_model_boundary.py](fused_tp_model_boundary.py) 的 `_project_query` 调用。
归一化的 Q latent 先转 FP32，再进入 TP input-copy boundary；前向重新使用原始
weight dtype 的 latent/weight/bias 执行 linear，保留原 BF16 前向。反向以 FP32
计算每个 head shard 的 input VJP，完成 TP SUM 后通过 latent cast 回到 BF16。
weight/bias VJP 保留原叶子 dtype，使用前向实际消耗的量化输入计算 weight VJP。
原 weight 经 `save_for_backward` 保留版本检查；空 owner 与冻结权重有明确路径，
二阶反向按现有训练接口的契约拒绝。没有新参数、checkpoint key 或 TP divisor。

本轮先验证了一个不保留 BF16 前向的候选：Q-B linear 直接使用 FP32。它在原生
144-case 矩阵中使 stock 输出检查从 144/144 降至 **60/144**，且未解决完整模型失败，
因此没有保留该实现。失败的完整矩阵、累积矩阵、源码和快照分别归档为
`query_fp32_model/`、`query_fp32_model_accumulation/` 与各自审计 JSON；结果不改写。

改用保留 BF16 前向的 bridge 后，重新完成 **144 rank-cases**。全部 stock 输出
再次逐位一致，max abs 为 `0`。CP1×TP4 的全部 **48** 个 case 中，
`q_a_layernorm.weight` 梯度也与 stock **逐位一致**，其 relative L2 从约
`0.003692027930759714` 降至 `0`。CP2×TP2 的该字段平均 relative L2 从
`0.003908475214308559` 降至 `0.0016336305551172805`；CP4×TP1 没有 TP 分片，
对应字段保持原结果。这些字段改善不能替代全部参数的验收。

完整模型验收仍为 **0/144**；stock `linear_qkv.weight`、部分 `kv_b_proj.weight` 和
`o_proj.weight` 的失败继续保留，FP32 模型 gate 不变。retained 反向的额外逐位检查
为 **19/24**，其余字段的微小变化均通过已有数值门槛；该计数未作为稳定性修复结论。

实际模型累积矩阵也重新完成 **288 rank-cases、576 microbatches**。协议、原参数
身份、selected-set 和 checkpoint 身份/epoch 检查通过；独立原生对照的额外逐位
累积检查为 **251/288**，完整模型数值验收仍为 **0/288**。SGD 对 stock 为
**288/288**，对 FP32 为 **216/288**。reference profile 仍为 `held_out_failed`，本轮
没有使用它放行候选，也没有改变 numerical gate。

CPU 覆盖包含保留 source forward、独立矩阵 VJP、实际量化输入、bias、空 owner、
冻结权重、权重版本变更、非法输入与二阶拒绝。目录回归 **327 tests、695 subtests**
通过，两进程真实 TP2 投影 FP64 模型 **8/8 cases** 通过。

修正版证据为 `query_source_model/`、`query_source_model_accumulation/`、各自的审计
JSON、`query_source_comparison.json`、`cpu_query_source_audit.json` 和
`query_source_cpu_all.log`。两个原生任务均完整执行，并以数值验收失败结束；归档
源码与当前源码、冻结 payload 和报告的加载库 SHA256 经核对。标准模型尺寸、
有效的完整模型 normalized profile、完整 LM、HBM 和整步性能仍待验收。

### 梯度预算传播到 BF16 SGD 的独立完整模型 profile

[model_sgd_precision.py](model_sgd_precision.py) 增加独立 schema 2 契约。
它保留 output、hidden、KL 和全部参数梯度的 reference-only 拟合；SGD 字段不再
根据 SGD 误差样本拟合，而是将已验证的梯度预算传播到一次 BF16 更新。
每个更新仍须通过完整参数梯度的准确性检查，以及从原始 BF16 参数和实际梯度在
同一设备独立执行一次 `add_(gradient, alpha=-0.01)` 的逐位对照。
物理算术边界比旧的经验 SGD 上限更保守，不能仅凭落入该边界放行模型。

对于参数元素数 `n`，独立 FP32 更新目标 `W` 和独立 FP32 梯度 `G`，令
`u=2**-8`、`uf=2**-24`、`eta=2**-134`、`c=(1+u)**2`。梯度预算给出的误差界为
`E2=||G||2 * gradient.relative_l2` 和
`Einf=||G||2/sqrt(n) * gradient.max_abs_over_rms`；零梯度使用既有绝对上限。
分别对 2 范数和无穷范数计算：

```text
pre = lr * (c * E + (c - 1) * ||G||)
      + 2 * uf * ||W|| + 4 * uf * lr * (||G|| + E)
      + underflow * (1 + u)
bound = (1 + u) * pre + u * ||W|| + underflow
```

其中 2 范数的 `underflow=eta*sqrt(n)`，无穷范数为 `eta`。
该界容纳 BF16 标量、乘积和加法舍入，并计入 FP32 reference 舍入。
没有梯度的字段必须完全不变；精确不变以零绝对差和独立单步控制验证，避免
cosine 的浮点除法将相同张量报告为约 `2e-16` 的角度误差。

新的独立拟合 seed 为 `20261201..20261216`，held-out 为 `20261231..20261238`，
candidate 预留 `20261251..20261254`，与原 profile 的三组 seed 均不相交。
margin 仍为 `1.25`、zero oracle 上限仍为 `2e-5`，不使用融合 candidate 拟合。
完整采样矩阵为 H64/H128、joint/freeze/zero/eval，每个 seed 重复两次，共
192 reference cases、384 samples。该矩阵仍为 hidden32/Q-LoRA64/V16、packed13
的真实 attention fixture，不能扩展为标准 checkpoint 或完整 LM 验收。

[mega_dsa_model_sgd_reference_calibrate.py](../../examples/mega_dsa_model_sgd_reference_calibrate.py)
的首次运行完成了全部 192 cases，但 CPU FP32 absorbed/unabsorbed 公式检查失败，
profile 为 `error`，不能使用。失败集中于 H64/seed `20261215` 和 H128/seed `20261201`
的 output，最大绝对差分别为 `1.0728836059570312e-6` 和 `1.0132789611816406e-6`，
relative L2 约 `7.6e-7`。原 `1e-6` 绝对门槛没有修改，原 JSON 和源码快照保留在
`model_sgd_reference_calibration/`，诊断为 `model_sgd_reference_failed_proof_audit.json`。

[mega_dsa_model_oracle_verify.py](../../examples/mega_dsa_model_oracle_verify.py)
将公式恒等验证与实际 FP32 reference 生成分开：完整 absorbed/unabsorbed 模型以
FP64 检查 output、hidden、KL 和全部参数梯度，预声明 relative L2 `<=1e-9`、
max abs `<=1e-11`；实际用于校准的每个 FP32 字段必须由原 unabsorbed 公式精确重算。
CPU reference 显式固定一个线程，避开多线程 LayerNorm backward 的微小重复差异，
环境报告记录线程数。FP64 值不替换 BF16 对 FP32 的实际校准目标。
两个曾失败的 fixture 的四种模式已通过 CPU 验证，篡改 FP32 目标的负例被拒绝。

修正后的 policy 在设备采样前冻结，SHA256 为
`e81b03d3507a2fed077b05a289bf7ef4069f30c3565439c20df964eb09559bec`。
新的 NPU 采样已完整结束，归档为 `model_sgd_reference_fp64_oracle/`：
**192 cases、384 samples、0 candidate samples**，原始结果为 `held_out_failed`。
修正精确不变参数的 cosine 误判后，另外生成 `profile_exact_inactive_check.json`，
仍为 `held_out_failed`，原始 profile 未覆盖。只有 18 个没有梯度的 SGD bucket/field
由误判改为通过；所有非 SGD 拟合上限和活跃 SGD 检查保持一致。
共有 5 个 bucket/field 的真实 held-out 失败、12 份字段样本，不能消费为有效 profile。
H64 的四种模式中，held-out seed `20261233` 的
`gradient/q_a_layernorm.weight` 在两个重复样本均出现独立 max abs/RMS 失败：
实测 `0.03144703612725351`，冻结上限 `0.02467929134678249`；relative L2
`0.006954169379146859` 与 cosine `2.2914843319621347e-5` 通过各自上限。
该真实梯度失败不属于完全相同参数的 cosine 舍入误判，不能用修正后者消除。
另一个失败为 H128/joint 的 KL loss：上限 relative L2/max abs/RMS 均为
`0.003388836819661733`，seed `20261233` 实测 `0.0039864109362938325`，
seed `20261234` 实测 `0.004096585819841678`，两个重复一致。
这些失败保留，margin 和 seed 分区未改变。

完整审计通过了全部 FP64 公式证明、精确 FP32 重算、selected-set、梯度可用性，
以及 **4224** 份参数单步 SGD 控制。冻结采样源码、冻结 sampler fitter 重算、
当前 fitter 重算、payload 和报告的加载库哈希均已核对；采样后只改变精确不变的
SGD 检查，未改变非 SGD 预算。证据为 `model_sgd_reference_fp64_oracle_audit.json`
和 `model_sgd_reference_fp64_held_out_failures.json`。原 schema 1 的 `held_out_failed` 和既有完整模型的失败记录保持原样。

[model_global_fields.py](model_global_fields.py) 在数值验收阶段拼回完整全局 25 个字段。
它要求完整 CP/TP 坐标、无重复且无遗漏的 token owner，以及精确一致的重复副本。
output/hidden 按全局 token ID 拼回；五类 head-sharded 参数按 TP 顺序拼接；已经
完成一次 CP FP32 SUM 的梯度和 SGD 参数只取验证一致后的一个副本，不能再次 CP SUM。
CPU 测试覆盖 strided/empty、O column shard、缺失 owner、副本不一致和梯度 None。

[mega_dsa_cp_tp_model_global_validate.py](../../examples/mega_dsa_cp_tp_model_global_validate.py)
使用预留的新 seed 执行真实原生 CP×TP F+B+SGD，再由
[model_sgd_evaluate.py](model_sgd_evaluate.py) 消费验证过的 profile。
它保留原有 pointwise gate 的结果，单独报告全局 normalized 结果。
reference 环境及其报告的加载库必须完全匹配；candidate 新加载的库只能来自显式
冻结的 payload 哈希声明，并完整保留在新运行报告中。
SGD 控制在原 NPU 后端执行，不能用 CPU 重放代替。
当前入口和 CPU 逻辑检查已完成，原生全局 normalized 矩阵须等待有效 profile。
`--metrics-only` 是单独的诊断模式：只采集完整全局指标和独立单步控制，所有
`global_reference.passed` 均为 `False`，报告 `profile_consumed_for_acceptance=False`。
它可使用完成采样但 held-out 失败的 reference 环境身份，不能消费失败预算进行放行。
四卡入口覆盖 CP4×TP1、CP2×TP2、CP1×TP4；八卡入口覆盖 CP8×TP1、CP4×TP2、CP2×TP4。
H128/TP8 的 local H16 超出当前 native H32/H64 契约，明确记录为不支持。
六组拓扑的完整 Cartesian 坐标已经 CPU mocked 检查；这不替代八卡原生验收。
最近一次目录 CPU 回归为 **339 tests、707 subtests** 通过。
本阶段没有完整 LM、长序列、HBM plateau 或完整训练步性能收益的验收结论。

四卡首次全局诊断已经完成 **24 rank-cases（6 个逻辑 case）**：
CP4×TP1、CP2×TP2、CP1×TP4，joint/plain，strided/empty，使用预留 seed `20261251`。
完整 25 个字段成功重建，四个 rank 报告的全局指标完全一致；所有重复副本精确一致，
且与独立 TP checkpoint gather 一致。selected-set、参数身份、前向 phase、CP backward、
owner return 和 TP reader ACK 已核对；全部 **264** 份参数单步 SGD 控制通过。
原有完整模型门槛仍为 **0/24**，`--metrics-only` 的全局验收标记始终为 `False`。

四卡归档为 `model_global_diagnostic_smoke/` 与 `model_global_diagnostic_smoke_audit.json`。
源码快照、当前源码、冻结 payload、报告的加载库和 reference 环境匹配已核对。
八卡同路径诊断也已完成 **48 rank-cases（6 个逻辑 case）**：
CP8×TP1、CP4×TP2、CP2×TP4，joint/plain，strided/empty，使用同一预留 seed `20261251`。
八个 rank 的完整全局指标完全一致，全局重建、副本、独立 TP checkpoint、全部
**528** 份参数 SGD 控制和所有协议检查通过。原有完整模型门槛仍为 **0/48**，
全局 normalized 验收标记始终为 `False`，没有使用失败的 reference profile 放行。
八卡归档为 `model_global_cp8_diagnostic_smoke/` 与
`model_global_cp8_diagnostic_smoke_audit.json`；源码、payload、报告的库及 reference
环境身份已经审计。该结果证明有界 fixture 的实际八卡融合 F+B+SGD 路径执行完成，
不能替代全模式、多 seed、标准尺寸、完整 LM、HBM plateau 和整步性能验收。

### O 参数 VJP 在 CP SUM 前保留 FP32

[parameter_vjp.py](parameter_vjp.py) 提供显式 backward cycle 的 intermediate VJP 捕获。
当前由 [fused_tp_model_boundary.py](fused_tp_model_boundary.py) 的 O weight/bias 使用，
通过 context `dsa_capture_output_parameter_vjps=True` 单独开启，默认路径保持原策略。
O linear 与 bias 原来已经使用 FP32 参数视图计算；新路径在视图 cast 回原 Parameter
之前捕获 FP32 VJP，原前向、输入 VJP、Parameter 对象、叶子梯度和 checkpoint key 不变。

这解决一个可精确计算的 CP 反例：BF16 输入 `1.5` 与 cotangent `0.66796875` 的乘积
为 `1.001953125`，局部叶子先 cast BF16 会舍入为 `1`；另一 owner 的贡献为 `-1`。
先 cast 再 CP SUM 得到 `0`，未分片计算和 FP32 CP SUM 后一次 cast 都得到
`0.001953125`。捕获钩子只安装在 intermediate Tensor 上，不安装在原 Parameter 上，
不修改传入 VJP，不执行 collective，不增加 TP divisor。

调用方须在完整 backward 或累积序列前 `begin_backward()`，结束后
`finish_backward()`，随后 `take()` 仅消费一次。若使用 `.backward()`，原 BF16
叶子累积仍按原路径执行；优化器使用的是捕获的 FP32 参数梯度经过一次 CP SUM、
一次 cast 后的梯度。不能把该策略说成从已经舍入的叶子梯度恢复了精度。
几个延迟前向可以同属一个累积 cycle；retained backward 必须开始独立 cycle，
checkpoint 重算可以创建新的参数视图。并发 backward 不支持；失败 cycle 须显式
`discard()`。权重版本变化、未开始 cycle、未消费旧 cycle 和重复消费会报错。

[mega_dsa_cp_tp_model_global_validate.py](../../examples/mega_dsa_cp_tp_model_global_validate.py)
增加 `--fp32-output-parameter-gradients` 进行同一预留 seed、同一 fixture 生成方式的
新旧 NPU 对照。[mega_dsa_cp_tp_model_validate.py](../../examples/mega_dsa_cp_tp_model_validate.py)
将捕获结果交给现有外部 CP reducer，保留一次 CP SUM 和一次 Parameter dtype cast。
当前只有 O weight/bias 使用该来源，其他参数的归并策略未改变。

CPU 测试覆盖消去反例、延迟前向累积、retained cycles、权重版本、冻结/unused、
精度类型和消费生命周期；实际 TP O 投影验证前向、输入 VJP、原叶子梯度逐位一致，
捕获 weight/bias VJP 与独立 FP32 公式一致。目录回归为 **343 tests、711 subtests**
通过。该结果不替代原生数值验收或捕获缓冲区的 HBM/整步性能测量。

同一预留 seed `20261251` 的四卡与八卡完整对照已完成：

| root / 拓扑 | O 梯度旧 stock 检查 | O 梯度捕获后 stock 检查 | 完整模型旧 gate |
|---|---:|---:|---:|
| 四卡：CP4×TP1、CP2×TP2、CP1×TP4 | 16/24 | 24/24，max abs 全为 0 | 0/24 |
| 八卡：CP8×TP1、CP4×TP2、CP2×TP4 | 24/48 | 48/48，max abs 全为 0 | 0/48 |

所有完成 case 的 O 梯度都与未分片 stock BF16 模型逐位一致。
strided CP4×TP1 的全局 FP32 oracle relative L2 从 `0.004410577893927871`
降至 `0.00407486494498889`，CP8×TP1 从 `0.0043960741077561415` 降至同一值；
TP2/TP4 的 strided O 梯度也下降，empty 所有权及 CP1×TP4 的 O 指标保持原值。
其余参数未被该捕获策略接管，完整模型的 `linear_qkv.weight` / KV 等失败仍保留。
CP2×TP2/empty 与 CP4×TP2/empty 各观察到一项 `indexer.wk.weight` 的微小重复差异，
不作为这次 O 修改的修复结论。

新旧使用相同 seed、同一 fixture 生成源码、同一冻结 payload 和匹配的 reference
环境；所有独立 FP32 oracle 字段范数逐位一致。早期诊断未记录运行时输入 tensor
的字节哈希，因此这里只声明这些已核对的身份，不补写实际 tensor 字节哈希证明。
对照归档为 `output_parameter_vjp_comparison.json`，四卡新证据为
`model_output_vjp_diagnostic_smoke/` 及其审计，八卡新完整证据为
`model_output_vjp_cp8_diagnostic_repeat2/` 及其审计。
全部全局字段重建、参数身份、selected-set、phase、CP backward、owner return、TP ACK
和 SGD 单步控制通过。reference profile 仍为 `held_out_failed`，没有用于放行。

八卡首轮及第一次重复在 CP4×TP2/strided 被严格 TP loss 副本检查拦住，各只完成
16 rank-cases；loss 为 `0.003228132613003254` 与 `0.0032281330786645412`，
相差两个 FP32 ULP。两次失败、源码快照和实际标量副本保留，检查未放宽。
增加 CP SUM 前局部 loss 的记录后，独立聚焦 CP4×TP2 的 16 rank-cases 与第二次
完整重复的 48 rank-cases 通过严格重建；这不能指定先前差异的唯一根因或宣称已修复。
失败归档为 `model_output_vjp_cp8_diagnostic_smoke/`、
`model_output_vjp_cp8_diagnostic_repeat1/`，聚焦控制为 `model_output_vjp_loss_diagnose/`。
当前捕获仍为显式实验选项，完整多 seed/模式、原生累积/checkpoint 和 HBM 性能仍待验收。

### QKV 权重 VJP 在 CP SUM 前保留 FP32

[source_linear.py](source_linear.py) 提供保留 source dtype 前向的 linear bridge。
前向把捕获的全精度参数视图转换回原始 Parameter dtype，与原 `linear_qkv` 输入、
weight/bias 和输出 dtype 一致；反向按前向实际输入在 FP32 计算 weight/bias VJP，
传给 intermediate 捕获钩子，外部 CP reducer 归并后才做一次叶子 dtype cast。
输入 VJP 返回原输入 dtype；原 Parameter 经 `save_for_backward` 保留版本检查。
空 token、bias、FP32/FP64 和冻结/unused 由相同形状/精度契约处理，二阶反向拒绝。

context `dsa_capture_qkv_parameter_vjps=True` 开启该实验策略，并同时开启已验证的
O VJP 捕获；原参数对象、checkpoint keys 和 CP/TP normalization 不变。
对应全局 validator 参数为 `--fp32-qkv-parameter-gradients`。O-only 基线与 QKV+O
候选使用预留 seed `20261251`、相同 fixture 生成源码、冻结 payload 和 reference 环境。
本轮没有扩大 reference profile 或用 candidate 误差拟合上限。

CPU 覆盖 source 前向逐位一致、独立矩阵 input/weight/bias VJP、空 token、原始
Parameter 版本、非法 view、二阶拒绝以及既有 VJP 捕获生命周期。
目录回归为 **345 tests、717 subtests** 通过，Ruff/Pylint 和文档检查通过。

四卡完成 **24 rank-cases**，八卡完成 **48 rank-cases**，均为 joint/plain 和
strided/empty。以下为 strided case 的完整全局 `linear_qkv.weight` 对独立 FP32
oracle 的 relative L2；每个 case 的全局指标在所有 rank 上一致：

| root / 拓扑 | O-only 基线 | QKV+O 捕获后 |
|---|---:|---:|
| 四卡 CP4×TP1 | `0.004723537536181745` | `0.004421365727768605` |
| 四卡 CP2×TP2 | `0.004749302358986961` | `0.004436418725020978` |
| 八卡 CP8×TP1 | `0.00472185075355831` | `0.004421366247646982` |
| 八卡 CP4×TP2 | `0.004690298410071016` | `0.004436083648628957` |
| 八卡 CP2×TP4 | `0.004749302358986961` | `0.004436418725020978` |

empty 所有权和 CP1×TP4 的 QKV 指标保持基线。output、hidden 与 O 梯度的指标
在新旧每个 case 中完全一致；O 梯度仍与 stock 逐位一致。四卡 CP2×TP2/empty
另有一项 `indexer.wk.weight` 的微小重复差异，不归因为 QKV 捕获的修复。
QKV 对 stock 的原 gate 仍未通过，完整模型仍为 **0/24** 和 **0/48**；这仅确认
局部权重梯度在 CP 归并前舍入的一部分误差已消除，不能写成完整模型通过。

全局重建、副本精确检查、独立 TP checkpoint、selected-set、参数身份、phase、
CP backward、owner return、TP ACK 和全部 SGD 单步控制通过；采样源码快照与当前
源码、冻结 payload 和报告的加载库/环境身份已经审计。
归档为 `model_qkv_vjp_diagnostic_smoke/`、`model_qkv_vjp_cp8_diagnostic_smoke/`、
各自的审计 JSON，以及 `qkv_parameter_vjp_comparison_four.json` /
`qkv_parameter_vjp_comparison_eight.json`。现有 reference `held_out_failed` 保留。
多 seed/模式、原生累积/checkpoint、标准尺寸、完整 LM、HBM 和整步性能仍需验证。

### KV 权重的两条 VJP 在 CP SUM 前保留 FP32

[source_bmm.py](source_bmm.py) 保留 source dtype 的 batched matmul 前向，
在反向返回全精度 weight VJP。`dsa_capture_kv_parameter_vjps=True` 使 query absorption
与 value restoration 使用同一个 `kv_b_proj.weight` 全精度视图；两条分支先在该视图
汇合，再由现有外部 CP reducer 做一次 SUM 和一次叶子 dtype cast。原 Parameter
对象和版本检查保留，默认路径不变。输入 VJP 采用全精度运算后返回输入 dtype，
因此不能声明其余参数梯度与原 NPU BF16 batched matmul 完全不变。全局 validator
选项为 `--fp32-kv-parameter-gradients`。

CPU 回归为 **347 tests、723 subtests** 通过，覆盖共享权重的 key/value 切片、
转置视图、空 owner、BF16/FP32/FP64、独立矩阵导数、版本检查和二阶拒绝。
同一预留 seed `20261251`、冻结 payload、fixture 生成源码与 reference 环境下，
O+QKV 基线和 O+QKV+KV 候选的 joint/plain、strided/empty 对照结果为：

| root / 拓扑 | KV 梯度旧 stock 检查 | KV 捕获后 stock 检查 | 完整模型旧 gate |
|---|---:|---:|---:|
| 四卡：CP4×TP1、CP2×TP2、CP1×TP4 | 16/24 | 24/24 | 0/24 |
| 八卡：CP8×TP1、CP4×TP2、CP2×TP4 | 26/48 | 48/48 | 0/48 |

KV 梯度 stock 对照的最大绝对误差为 `3.814697265625e-06`，不能写成逐位一致。
strided CP4×TP1 对 FP32 oracle 的 relative L2 从 `0.0044795936982710526`
降至 `0.004163700525570364`；CP8×TP1 从 `0.0044681181698717045` 降至
`0.00416370076869766`。Q-B 梯度及部分 QKV/SGD 字段也有变化，记录在
`kv_parameter_vjp_comparison.json`，不扩大为其他数值缺陷已解决的结论。

四卡 `model_kv_vjp_diagnostic_smoke/` 与八卡 `model_kv_vjp_cp8_diagnostic_smoke/`
及各自审计确认全局重建、副本精确检查、selected-set、参数身份、通信阶段、
264/528 个 SGD 单步控制、冻结 payload、报告的加载库和环境身份通过。
完整模型、normalized reference 和性能验收均未通过，`held_out_failed` profile 保留。

### Q-B 参数捕获与 TP 输入梯度归约

`dsa_capture_query_parameter_vjps=True` 让 Q-B weight/bias 的本地 VJP 进入全精度
视图捕获，外部 CP reducer 归约后转换回原叶子 dtype。该选项同时启用 O 捕获，
其他投影捕获选项独立。validator 通过 `--fp32-query-parameter-gradients` 启用。
[source_linear.py](source_linear.py) 的 `full_precision_inputs=True` 接受 TP copy
边界后的全精度 normalized latent；前向仍转换为原 Parameter dtype，weight VJP
使用前向实际消费的量化输入，input VJP 保持全精度直到 TP SUM。
CPU 对照覆盖原 Q-B 前向、input/leaf VJP 一致，以及独立 FP32/FP64 weight/bias
公式、空 owner 和实际分片模型边界。目录回归为 **349 tests、729 subtests** 通过，
Ruff/Pylint、文档目录及本地链接检查通过。原生验证结果另行记录；该选项仍为实验策略。

Q-B 捕获的四卡与八卡原生对照均已结束，退出码为 0。候选为
O+QKV+KV+Q-B 捕获，基线为上一轮 O+QKV+KV 捕获；均为同一预留 seed
`20261251`、joint/plain、strided/empty、相同 fixture 生成方式与冻结 payload。
所有独立 FP32 oracle 字段的 expected norm 在新旧间一致。strided 全局 Q-B 梯度
对 FP32 oracle 的 relative L2 为：

| root / 拓扑 | 原 Q-B 参数叶子路径 | Q-B 捕获后 |
|---|---:|---:|
| 四卡 CP4×TP1 | `0.006239612093267434` | `0.006023493141016537` |
| 四卡 CP2×TP2 | `0.0063555870641981834` | `0.0061050339203446375` |
| 八卡 CP8×TP1 | `0.0062270969870052975` | `0.006023489786684151` |
| 八卡 CP4×TP2 | `0.006329585156638936` | `0.006105036370078737` |
| 八卡 CP2×TP4 | `0.0063555870641981834` | `0.0061050339203446375` |

empty 所有权及 CP1×TP4 的全部全局字段指标与基线相同。output、hidden、loss、
O 梯度指标在所有新旧 case 中一致。部分 strided QKV/KV 梯度出现微小重复差异，
不能归因为此次 Q-B 参数捕获改变输入 VJP；CPU 已对输入 VJP 做逐位对照，
NPU 指标比较本身不能证明中间 tensor 的逐位一致。

Q-B 梯度 stock 检查保持四卡 **24/24**、八卡 **48/48** 通过，最大绝对误差为
`9.5367431640625e-07`。O 梯度 stock 对照仍逐位一致，KV stock 检查仍全部通过。
完整模型旧 gate 仍为 **0/24**、**0/48**；normalized reference、完整模型和
性能验收均未通过。对照没有使用 candidate 拟合 profile，没有放宽已有 gate。

归档为 `model_query_vjp_diagnostic_smoke/`、`model_query_vjp_cp8_diagnostic_smoke/`、
各自审计 JSON 与 `query_parameter_vjp_comparison.json`。审计覆盖完整全局字段、
CP/TP 副本精确检查、独立 TP checkpoint、selected-set、参数身份、phase、
CP backward、owner return、TP ACK、264/528 个 SGD 单步控制，以及采样源码、
冻结 payload 和报告的加载库/环境。当前仍仅验收一个 seed 的 joint/plain 诊断，
多模式、多 seed、原生累积/checkpoint、标准尺寸、完整 LM、HBM 和整步性能待验证。

### 多 microbatch 的参数 VJP 捕获与 checkpoint

[mega_dsa_cp_tp_model_accumulation_validate.py](../../examples/mega_dsa_cp_tp_model_accumulation_validate.py)
增加 `--fp32-parameter-gradients`，同时启用 O、QKV、KV 和 Q-B 捕获。两个
microbatch 的前向先保存，再按 ordered/reverse 顺序执行 backward；整个序列
共用一个捕获周期。non-reentrant checkpoint 在重算中创建的新视图使用同一周期，
最后一次性消费 FP32 梯度，做一次实际参数 CP SUM 和叶子 dtype 转换。
独立逐 microbatch 捕获在消费后清空，不与 delayed 周期混用；失败周期显式 discard。

检查分别记录原 BF16 叶子累积、FP32 中间捕获累积、外部 CP 归约结果、独立单步
SGD 控制及 native phase/owner-return/ACK 记录。CP 归约控制另做一条验证路径，
不会把控制路径的 collective 计为实际参数重复归约。完整模型数值 gate 保留；
`--metrics-only` 只允许收集诊断，报告始终不放行 normalized reference、完整模型或
性能，也不生成大体积失败 tensor 文件。validator 可在四卡和八卡 root 上执行
CP×TP1/2/4；TP8 的 local H16 仍不在当前原生 support matrix 内。

CPU 消去反例覆盖两次独立 backward、两个顺序、non-reentrant checkpoint、
source dtype 输入及 TP 边界的全精度输入。原 BF16 叶子累积为 0，而捕获 FP32
梯度为 `0.001953125`，与独立公式一致；输入 VJP 正确，捕获只能消费一次。
原生结果另行记录，以上实现契约不能替代数值与性能验收。

该路径的 CPU 回归为 **350 tests、735 subtests** 通过，Ruff/Pylint 和文档检查通过。
四卡 `model_parameter_vjp_accumulation_smoke/` 与八卡
`model_parameter_vjp_accumulation_cp8_smoke/` 均完成 joint 模式下所有
strided/empty、ordered/reverse/checkpoint case，进程退出码为 0。

| 诊断检查（rank-cases） | 四卡 | 八卡 |
|---|---:|---:|
| 完成两个 microbatch | 72/72 | 144/144 |
| 原叶子及输入/输出与逐 microbatch 路径逐位一致 | 67/72 | 143/144 |
| 捕获 FP32 梯度与逐 microbatch 捕获之和逐位一致 | 59/72 | 129/144 |
| 捕获 CP 归约结果与独立逐 microbatch 控制逐位一致 | 52/72 | 88/144 |
| 综合严格累积契约 | 44/72 | 74/144 |
| 全部 11 个 SGD 单步控制 | 72/72 | 144/144 |
| 完整模型旧数值 gate | 0/72 | 0/144 |

所有 case 的 native phase、两次 backward、owner return/ACK、Parameter 身份和
selected-set 检查通过；SGD 单步控制分别为 792/1584 个，全部逐位一致。
捕获 FP32 本地累积的失败均出现在 `linear_qkv.weight`，最大绝对误差约
`4.202127456665039e-06`，另有约 `1e-9` 的微小重复差异。CP 归约控制中
Q-B/KV/O 也有差异，部分 case 本地捕获已逐位一致；原始失败和严格判定保留，
不能仅凭进程退出码把累积或 checkpoint 写成已完整验收。

两个采样 producer 在任务期间保持冻结；初次审计核对采样源码与当时工作树、
冻结 payload 和报告的加载库。随后增加同源 CP SUM 重复控制与聚焦 CLI，
修改发生在两组任务及初次审计结束之后。各自
`*_post_run_source_transition.json` 记录这一 producer 差异，历史结论以保存的
`runtime_sources/` 和审计为准，不能宣称旧 producer 与后续源码仍逐字节相同。

`--topology`、`--pattern`、`--replay` 可聚焦单个诊断。
`same_source_cp_reduction_control` 在实际 CP SUM 前复制同一份连续 FP32 捕获
输入，另做一次 CP SUM 后比较叶子 dtype 结果，用于定位归约重复差异。
原逐 microbatch 控制和全部严格数值判定保留；归因结果另行记录。

聚焦 `model_parameter_vjp_accumulation_cp_sum_repeat/` 已完成四卡
CP4×TP1/strided/ordered 的 4 rank-cases，退出码为 0。四个 rank 的本地捕获
FP32 累积均与独立逐 microbatch 路径逐位一致，但同一份连续 FP32 输入重复
执行 CP SUM 后，Q-B/KV 的叶子 dtype 结果仍不逐位一致：

| 字段 | 同源 CP SUM 重复的 max abs | relative L2 |
|---|---:|---:|
| `q_b_proj.weight` | `2.384185791015625e-07` | `2.909260338410053e-06` |
| `kv_b_proj.weight` | `4.76837158203125e-07` | `3.963241037593411e-07` |

四个 rank 的重复差异指标相同，O/QKV 的同源 CP 控制逐位一致。
Q-B/KV 的常规点误差检查通过，但严格逐位检查未通过；两种判定分别保留。
该控制排除了“全部 CP 控制差异都来自本地捕获或 checkpoint”的解释，
只定位到同源 CP 归约/最终转换这一段，不指定 HCCL 的唯一内部根因。
此前累积诊断的失败计数和完整模型失败仍保留，未放宽阈值。

聚焦审计 `model_parameter_vjp_accumulation_cp_sum_repeat_audit.json` 核对源码
快照与当前源码、冻结 payload、报告的加载库、协议/参数身份/selected-set
和全部 44 个 SGD 单步控制。下一步需要以固定顺序 FP32 归约控制进一步分离
归约重复误差与本地 QKV/input VJP 的重复差异；不能将控制方案的正确性直接
当作完整模型或融合训练性能通过。

### 有界分块的固定顺序参数 SUM

[parameter_sum.py](parameter_sum.py) 提供 caller-owned、first-order 的
`ordered_parameter_sum`：只接受 detached FP32/FP64 contribution、显式 CP group
及正整数 chunk size。每块做 device all-gather，等待 work handle 完成，再按
group rank 0 到 P−1 顺序加到全精度结果；不做 mean、不转换叶子 dtype，不修改
输入，不复制梯度到 CPU。调用者保证 shape/dtype、调用顺序和 chunk size 在
所有参与成员上相同。单成员及零元素输入不执行 collective。

默认 `chunk_elements=262144`；通信 scratch 上限为 `P×chunk_elements` 个
全精度元素，CP8/FP32 为 8 MiB。输出及非连续输入的 contiguous copy 另计，
这不是完整模型 HBM 峰值。该实现使用更多通信和设备加法，只是显式实验归约
策略，未测量性能前不替代默认 collective SUM。

累积 validator 的 `--parameter-reduction-order group_rank` 将实际参数归约、
独立逐 microbatch 控制和同源重复控制都接到这一顺序；默认仍为 `collective`。
两条控制只用于验证，不叠加到实际参数梯度。原来的 CP SUM 次数、aux scale、
SGD 控制、逐位累积契约和完整模型 gate 均保留。CPU mocked tests 覆盖延迟
collective completion、读取前 wait、尾块、scratch 复用、非连续输入、CP2/4/8、
FP32/FP64、单成员/空张量、非法 dtype/梯度/group/chunk 和消去顺序反例。
原生数值、HBM 与性能证据另行记录。

固定顺序 reducer 的目录回归为 **354 tests、744 subtests** 通过，Ruff/Pylint、
文档目录/链接及 diff 检查通过。四卡 `model_ordered_parameter_accumulation_smoke/`
与八卡 `model_ordered_parameter_accumulation_cp8_smoke/` 在 joint 模式完成全部
strided/empty、ordered/reverse/checkpoint 诊断，均退出码为 0。

| 检查（rank-cases） | 四卡 | 八卡 |
|---|---:|---:|
| 同一份 FP32 输入重复固定顺序 CP SUM，所有捕获字段逐位一致 | 72/72 | 144/144 |
| 捕获 CP 归约结果与独立逐 microbatch 控制逐位一致 | 52/72 | 128/144 |
| 捕获 FP32 本地累积与独立捕获之和逐位一致 | 53/72 | 132/144 |
| 原叶子及输入/输出与独立路径逐位一致 | 61/72 | 136/144 |
| 综合严格累积契约 | 44/72 | 124/144 |
| 全部 SGD 单步控制 | 72/72 | 144/144 |
| 完整模型旧数值 gate | 0/72 | 0/144 |

同源归约检查对应 **288/576 个参数控制，全部逐位一致**。独立 CP 控制的
Q-B/KV/O 失败全部消失；该控制剩余失败均为 `linear_qkv.weight`，四卡 20、
八卡 16 rank-cases。捕获本地累积的失败也仅为 QKV，四卡 19、八卡 12；
最大 absolute error 分别约 `4.202127456665039e-06`、
`3.1441450119018555e-06`。原叶子/input 检查仍有 QKV、hidden，四卡还有
一项 `kv_a_layernorm.weight` 差异，不能将参数归约重复性改善写成全部误差消失。

上一轮 collective SUM 的综合严格契约为 44/72、74/144，本轮为 44/72、124/144。
两轮使用相同 fixture 生成方式和冻结 payload，但有本地 numerical repeat 差异，
因此这些 aggregate 数字不是稳定训练收益或性能证据。当前结论限于全部同源
固定顺序 CP 控制逐位一致，及独立控制中 Q-B/KV/O 的逐位失败消失。
没有用 candidate 拟合 profile，没有修改原数值阈值。

各自审计 JSON 记录完整覆盖、采样源码快照与当时工作树、冻结 payload、报告的
加载库、phase/owner-return/ACK、参数身份、selected-set 和 792/1584 个
SGD 单步控制。对比为 `ordered_parameter_accumulation_comparison.json`。
完整模型、normalized reference、完整 LM 和性能验收均未通过；固定顺序 reducer
保持实验选项。下一步继续定位本地 QKV/input VJP 的重复差异，而不扩大该
归约控制的通过结论。

### TP 投影 SUM 的固定顺序实验策略

[tp_projections.py](tp_projections.py) 的 `DsaTpProjectionCollectives` 增加
`reduction_order='group_rank'`，复用有界 FP32/FP64 gather+顺序加法。
初始化时将归约策略加入原有 collective row declaration，拒绝成员间策略不一致。
原默认策略 `collective` 保留；固定顺序只接管 Torch Q-B 输入 VJP 的 TP SUM
和 O 投影 partial output 的 TP SUM，原生 fused index/KL 与 shared owner-return
的实现不受该选项接管。

模型 context 为 `dsa_projection_reduction_order='group_rank'`，非法本地策略在
物理参数分片之前拒绝。Q-B copy 仍保持 source 前向与 input-SUM backward；
O output 仍保持 row-parallel SUM 前向和 identity backward，dtype 转换位置不变。
不对 TP 或 CP 多除一次 size，也不调整 aux upstream。累积 validator 通过
`--projection-reduction-order group_rank` 启用，与 CP 参数归约选项独立。

CPU 对照覆盖 BF16/FP32/FP64、两个 autograd 规则、full-precision detached
operand、活动 autograd edge 与非法 dtype 拒绝、成员策略不一致、非法模型策略
不修改原 Parameter storage。原生结果另行记录；固定顺序策略仍为实验选项，
不能仅凭可重复性宣称数值验收或整步性能通过。

TP 固定顺序投影的目录回归为 **356 tests、747 subtests** 通过，Ruff/Pylint
和文档检查通过。四卡 `model_ordered_tp_projection_accumulation_smoke/`、
八卡 `model_ordered_tp_projection_accumulation_cp8_smoke/` 均完成同一
joint、strided/empty、ordered/reverse/checkpoint 矩阵，退出码为 0。

| 检查（rank-cases） | 四卡 | 八卡 |
|---|---:|---:|
| 同源固定顺序 CP 参数控制逐位一致 | 72/72 | 144/144 |
| 捕获 CP 归约与独立控制逐位一致 | 40/72 | 120/144 |
| 本地捕获 FP32 累积逐位一致 | 44/72 | 126/144 |
| 原叶子及输入/输出逐位一致 | 52/72 | 138/144 |
| 综合严格累积契约 | 32/72 | 108/144 |
| 全部 SGD 单步控制 | 72/72 | 144/144 |
| 完整模型旧数值 gate | 0/72 | 0/144 |

CP 控制与本地捕获的剩余失败仍只有 QKV。上一轮只固定 CP 参数 SUM 时的
严格契约为 44/72、124/144；本轮投影策略没有改善 aggregate 数字。两轮有
数值重复差异，不能据此指定唯一根因或称为性能退化，但当前没有默认启用
TP 固定顺序策略的证据。各自审计核对矩阵、源码/payload/报告加载库、
phase/owner-return/ACK、Parameter 身份、selected-set、792/1584 个 SGD
单步控制和全部同源 CP 控制。完整模型与性能验收仍未通过。

### 实际 RMSNorm 的同状态 VJP 重放

[mega_dsa_norm_vjp_trace.py](../../examples/mega_dsa_norm_vjp_trace.py) 在
实际 `nn.RMSNorm` 的前向和 intermediate tensor 上安装临时 observer，捕获
输入、原权重值、输出、上游梯度与输入 VJP。原算子及其反向保持不变；
被观察的 norm 输入只能供该 norm 分支使用。两个 isolated forward/backward
与 delayed forward/backward 分别标记，按前向顺序对齐，不按反向完成顺序配对。

反向完成后使用捕获的输入/权重/上游梯度，重复原同设备 RMSNorm VJP，
并记录 source forward/input VJP 的逐位比较和独立 CPU FP64 数学导数误差。
权重快照保持前向值，后续 SGD 不改变重放状态。全部 hooks 在诊断结束时
移除；仅输出 JSON 指标，不保存完整模型 tensor 文件。此诊断增加复制、
同步和重放开销，不能用于性能测量，FP64 数学对照也不是拟合后的验收 profile。
CPU 覆盖 reverse delayed backward、上游 cotangent 改变、独立公式、权重
更新后仍重放前向状态、缺失记录/非法配置以及 hook 清理。原生结果另行记录。

包含 norm observer 的目录 CPU 回归为 **359 tests、747 subtests** 通过，
Ruff/Pylint 和文档检查通过。两组四卡聚焦任务均完成、退出码为 0：
`model_norm_vjp_cp4_empty/` 为 CP4×TP1/empty/ordered；
`model_norm_vjp_tp4_strided/` 为 CP1×TP4/strided/ordered。
均沿用 FP32 参数捕获、固定顺序 CP 参数 SUM 与 TP 投影 SUM。

每个 rank 的两个实际 norm 各捕获 2 isolated + 2 delayed backward 状态，
每个状态重放 3 次；合计 **192 次同状态重放**。所有重放的 source 输出和
input VJP 都逐位一致，独立 FP64 input/weight VJP 公式指标有限。
CP4×TP1/empty 的两个 norm 在 isolated/delayed 间全部输入、权重、输出、
upstream 与 input VJP 逐位一致；该次聚焦 rank0 的严格累积契约也通过。
这不消除先前完整矩阵中保留的重复差异。

CP1×TP4/strided 的 Q norm 全部字段一致；KV norm 的输入、权重和输出一致，
但两个 microbatch 的 upstream 已不同，只有第二个 microbatch 的 input VJP
跟随出现差异。四个 rank 的这些指标相同：

| KV norm 字段 | max abs | relative L2 |
|---|---:|---:|
| microbatch 0 upstream | `7.275957614183426e-12` | `3.5396259756165135e-10` |
| microbatch 1 upstream | `2.3283064365386963e-10` | `1.167264045428176e-08` |
| microbatch 1 input VJP | `9.313225746154785e-10` | `2.5513183796476023e-08` |

该次 TP4 聚焦严格累积契约仍未通过。证据只证明所捕获差异在 KV norm 的
upstream 已存在，且相同状态的 norm 自身可重复；不能指定 native SFA、
shared dK+dV、owner return 或其他上游环节为唯一根因，也不能从本次观测
推断之前所有 case 都相同。临时 observer 增加复制和改变执行时序，禁止用其
结果报告性能。完整模型仍未通过，原完整矩阵失败不被聚焦结果替代。

各自审计核对采样源码/payload/报告加载库、保存的 diagnostic driver 字节哈希、
原协议/参数身份/selected-set 与 44 个 SGD 单步控制。诊断汇总为
`norm_vjp_diagnostic_summary.json`，TP 策略比较为
`ordered_tp_projection_accumulation_comparison.json`。下一步在 compressed KV
梯度进入 norm 之前进一步拆分原生梯度与归还环节，继续固定状态/上游的对照。

### compressed KV 梯度的逐阶段捕获与同状态重放

[mega_dsa_gradient_stage_trace.py](../../examples/mega_dsa_gradient_stage_trace.py)
为 isolated/delayed 的 microbatch 0/1 保留 detached、无输入别名的阶段快照。
整数 indices 只作逐位比较，浮点输入/梯度另外计算 CPU FP64 误差指标，
inactive 字段显式为 None。缺失阶段、重复记录、schema/shape/dtype 不一致
均拒绝作为完整证据；诊断不分配验收预算。

实际模型诊断 driver 捕获 native backward 的输入 states、forward stats、
indices、cotangent，随后捕获 native gradients、FP32 partials、CP owner
结果，以及 shared return 的输入、TP FP32 输出和最终 dtype 输出。
Native backward 与 shared return 都在原权重梯度/SGD 比较完成之后，使用
保存的实际状态或复制的同一份 partials，分别重放两次。每次重新准备 transport
epoch，验证 CP backward/return 回执和 TP ACK，不重用已消费的 invocation。
已缩放的 partials 不再次应用 aux、CP 或 TP 系数。

同步保留 norm observer，防止仅比较 kernel 输出就遗漏到 norm 边界的差异。
全部信息输出为 JSON 指标，diagnostic driver 自身保存并记录字节哈希；
临时 observer 在完成时清理。此路径增加复制、同步和额外 native 调用，
禁止用于性能比较，也不替代完整模型原 gate。CPU 测试覆盖快照所有权、
反向顺序配对、整数/inactive 字段和不完整/重复/不兼容 trace 的拒绝。
原生结果另行记录。

目录 CPU 回归为 **361 tests、747 subtests** 通过，Ruff/Pylint 和文档检查通过。
四卡 `model_gradient_stage_cp4_empty/`（CP4×TP1/empty/ordered）与
`model_gradient_stage_tp4_strided/`（CP1×TP4/strided/ordered）均结束，退出码为 0。
审计验证源码快照、driver 字节哈希、冻结 payload、报告加载库和原协议/
参数身份/selected-set/SGD 控制。每组另外完成 **32 次 native backward**
和 **32 次 shared return** 同状态重放；所有 native CP 回执与 shared TP ACK
epoch 都经过验证且按顺序递增，不使用旧 invocation。

TP4 的 isolated/delayed native 输入 states、indices、forward stats 和 cotangent
全部逐位一致。差异先出现在 native 的 FP32 `main_partials`、CP owner C/RoPE
结果，随后传入 shared return；TP FP32 输出和最终 C 的 BF16 转换承接差异，
KV norm 的 upstream/input VJP 也相应不同。Q norm 仍一致。
同一份 shared 输入重放的 **64 次控制（两组合计）所有输出逐位一致**，
因此这批结果没有显示 shared CP-index/TP return 自身增加重复差异。

相同 native 保存状态和 cotangent 的重放则仍有 FP32 partial 差异。CP4 中
活跃 rank0 的 8 次重放全部不同，其他三个无 query 的 rank 结果为零并一致；
TP4 的全部 32 次重放都存在 FP32 partial 差异，个别 BF16 dK/dV 也不同。
以下为每组全部同状态 native 重放的最大误差：

| 字段 | CP4 empty max abs / max relative L2 | TP4 strided max abs / max relative L2 |
|---|---:|---:|
| `main_partials` | `2.3283064365386963e-10` / `6.372360286006728e-08` | `1.7462298274040222e-10` / `6.466776836742346e-08` |
| CP owner C | `2.3283064365386963e-10` / `6.818939244814029e-08` | `1.7462298274040222e-10` / `6.951657133085547e-08` |
| CP owner RoPE | `5.820766091346741e-11` / `6.213835825398099e-08` | `5.820766091346741e-11` / `8.041896852239745e-08` |

无 query 的 CP rank 保存的部分未使用 compressed/key-RoPE 输入含非有限值，
`torch.equal` 的失败不证明这些输入发生有效数值变化；审计单独记录这些
非有限比较，排除它们参与活跃计算差异的归因。CP4 活跃 rank0 与 TP4 的
有限 native 输入均逐位一致，native 重放输出全部有限。

当前证据将这些样本的重复差异定位到 native backward 内部及其 FP32 partial
输出，不指定具体原子累加、混合 worker 分组或其他内部机制为唯一原因。
当前 backend 本身明确拒绝 `torch.are_deterministic_algorithms_enabled()`，
因为它保留 native 非确定性数学；本次诊断没有把微小误差改写成协议故障或
完整模型通过。原数值 gate 和完整矩阵失败均保留。

审计汇总为 `gradient_stage_diagnostic_summary.json`，每组的
`*_audit.json` 包含阶段失败计数与 rank0 指标。下一步对主梯度 worker 分组
做独立控制，区分计算调度与固定状态误差；控制结果不能直接作为生产默认
或训练性能证据。

### 主梯度 compute group 数量的控制

累积 validator 增加 `--main-compute-groups`，范围为 `MixedSfaSchedule` 已声明
的 1–19，默认仍为 7。配置在准备 root/workspace 前验证，并记录在每个
scenario 与顶层报告中。它改变主 attention 的 logical partition 到 physical
compute group 的调度，不调整模型尺寸、TP/CP loss scaling、Parameter/
checkpoint 身份、FP32 捕获、原数值 gate 或 payload。前向和对应 backward
使用同一 frozen schedule；同状态重放保留该 schedule。

单 group 是针对计算调度的诊断控制，不是默认优化或完整 CP/融合目标的替代。
保留上一轮 g7 两组证据，新轮 g1 使用相同输入生成方式和 diagnostic driver，
再次比较 native partial、CP owner、shared TP、norm 以及协议回执。两轮
producer 差异在各自 post-run source transition 中记录，历史源码快照不覆盖。
原生控制结果另行记录；没有只凭低并行度下通过就默认降低生产并行度。

首次 g1 两组任务在 fused CP/TP 构造阶段退出，尚未执行模型计算。原因是
新 group 参数未同步 LI/KL 的槽数，原 `FusedDsaCpTpForwardProbe` 契约要求
`slots == 2*compute_groups`。错误保留在 `model_gradient_stage_g1_cp4_empty/`
和 `model_gradient_stage_g1_tp4_strided/`，不计为 native 数值或协议结果。

准备代码已按既有契约从 schedule 派生 LI/KL 两个 transport 的 slots；
owner-return 的独立 14-slot transport 保持原容量。默认 g7 因此仍使用 14
个 LI/KL slots，g1 使用 2 个；报告同时记录两个数值。CPU 增加 g1/2/7/19
对匹配 LI/KL 容量的构造检查及错误容量拒绝。新的重跑使用独立 repeat1
目录，不覆盖失败或基线证据。该控制同时改变调度及其必须联动的 LI/KL
容量，不能描述成其他物理配置完全不变。

联动 slots 修正后的目录回归为 **362 tests、751 subtests** 通过，Ruff/Pylint
和文档检查通过。`model_gradient_stage_g1_cp4_empty_repeat1/` 与
`model_gradient_stage_g1_tp4_strided_repeat1/` 均结束，退出码为 0。
全部原始 backward 和 64 次同状态 native 重放的设备回执确认：
`compute_groups=1`，每个阶段仍遍历全部 **20 logical tasks**，每组
1 AIC/2 AIV，CP ready/write/ACK 和 shared TP ACK epoch 全部有效。

| 同状态控制 | 默认 g7 的 native FP32 partial 非逐位次数 | g1 非逐位次数 |
|---|---:|---:|
| CP4×TP1/empty，32 次 native 重放 | 8（活跃 rank0；其他 rank 无 query） | 0/32 |
| CP1×TP4/strided，32 次 native 重放 | 32 | 0/32 |

两组 g1 的 native raw gradients、FP32 partials、CP owner 输出及同源 shared
输出全部逐位一致。有限值的 isolated/delayed 阶段比较也全部一致；无 query
rank 的未使用非有限输入继续单独记录。Q/KV norm 输入、权重、输出、
upstream 和 input VJP 在两个阶段间全部逐位一致。两组严格累积契约分别
为 **4/4、4/4 rank-cases**；完整模型旧数值 gate 仍为 **0/4、0/4**。

这证明在该小尺寸 fixture、两种拓扑和 ordered 执行方式中，调度与联动
容量的控制能消除所观察的 native 重复误差；不能宣称所有形状/seed/模式
确定性，不能指定唯一内部机制，也不能把低并行度控制替代完整融合目标。
默认仍为 g7，性能/HBM 尚未测量，完整模型验收没有放行。

各自审计包含 32 次 native 回执、32 次 shared ACK、全部逻辑工作覆盖、
原参数身份/selected-set/44 个 SGD 单步控制、采样源码/driver/payload/
报告加载库。汇总为 `gradient_stage_single_group_summary.json`，新旧对比为
`gradient_stage_group_count_comparison.json`。最初两个构造失败仍保留。
下一步将 backward 的调度配置与 forward/LI/KL 容量分离，使用同一保存
状态对不同 backward group 数做控制，进一步排除前向及联动容量的混杂。

### 独立 backward schedule 与相同 forward 状态控制

[fused_cp_backward.py](fused_cp_backward.py) 的 `FusedDsaCpBackwardProbe`
增加可选 `schedule`，默认沿用 forward schedule。独立 schedule 在 CP group
内冷声明一致性，要求 `rounds=1`；runtime config 保留身份与版本检查，
配置/调度在提交前被改动则在 lease 前拒绝。native 调用使用 backward config，
其三阶段回执也按 backward schedule 解码，仍要求覆盖全部 20 logical tasks。
保存的 forward geometry/activation 版本校验保留，forward 与 LI/KL 容量
不会因额外 backward probe 而改变。返回的 runtime buffers 保持 config 生命周期。

新的诊断先按默认 forward=7、LI/KL=14 slots 执行原模型，再针对每份实际
保存状态和同一 cotangent，交错调用 backward g7/g1，各重放两次。两个
backward probe 共用 workspace 的递增 epoch，使用独立 runtime config；
比较原始输出差异，并将每个 group 的第二次输出与该 group 第一次输出
逐位对照。shared return 控制保留原调用配置，不重复应用损失缩放。

CPU 验证 g1 backward 直接消费 g7 forward 保存状态、native 获得 g1 config、
forward config 不变、FP32 owner cast deferral 保留；还覆盖非法 rounds/
schedule、CP 成员调度不一致、runtime config 和 schedule 修改在 lease 前
拒绝。该配置用于隔离诊断，默认训练调度未修改，原生结果另行记录。

独立 backward schedule 的目录 CPU 回归为 **364 tests、753 subtests**
通过，Ruff/Pylint 和文档检查通过。`model_backward_schedule_cp4_empty/`
与 `model_backward_schedule_tp4_strided/` 均完整结束、退出码为 0。
每组 forward 固定 g7、LI/KL 固定 14 slots，不重新生成 forward 或
改变保存状态；同一个 cotangent 按 g7→g1→g7→g1 顺序重放。

| 第二次与第一次同 group 输出比较 | CP4×TP1/empty | CP1×TP4/strided |
|---|---:|---:|
| backward g7 的 FP32 partial 非逐位对数 | 4/16（活跃 rank 的 4 份状态） | 16/16 |
| backward g1 的任意 native 输出非逐位对数 | 0/16 | 0/16 |

每组完成 64 次 native backward 重放、32 次 shared return 重放，
合计 128 次 native 回执和 64 次 shared TP ACK。每次设备回执按实际
backward schedule 解码，确认遍历全部 20 logical tasks，CP ready/write/
ACK 与递增 epoch 有效。全部 shared 同状态输出保持逐位一致。

这些结果将前向计算、LI/KL 容量与保存状态的混杂因素排除后，仍观察到
backward g7 的重复误差，而 g1 在这些样本中可重复。范围只覆盖小 fixture、
两种拓扑、四份保存状态及各两次重放；不指定 native 原子累加或具体
内部机制为唯一根因，不宣称任意形状/seed 确定性。低并行度是诊断控制，
不作为默认生产策略或原多卡融合目标的替代。

两组审计核对 producer 源码、driver 字节哈希、冻结 payload、报告加载库、
原参数身份/selected-set/SGD 控制、全部 logical-task/CP/TP 回执。汇总为
`backward_schedule_control_summary.json`。原模型仍按默认调度运行，其
完整数值 gate 没有放行，normalized reference 与性能验收仍未通过。
下一步把独立 backward schedule 接入实际训练 bridge 的显式实验配置，
验证完整 CP×TP 累积/checkpoint 矩阵并继续处理模型精度差异。

### 训练 bridge 的独立 backward schedule

[local_tp_training.py](local_tp_training.py) 的
`FusedDsaLocalCpTpTrainingProbe` 增加 `backward_schedule`，默认仍跟随
forward。该参数与 mode、loss coefficient、normalization、branch presence
及 sparse-return 一起在 CP/TP 上冷声明一致性，`rounds=1`；调用期间
修改声明会拒绝。实际 main backward backend 使用所选 schedule，
indexer/KL forward、LI/KL slots、CP/TP shared return 和 loss scaling 保留。

累积 validator 的 `--backward-compute-groups` 与 `--main-compute-groups`
独立，报告记录实际 backward group 数及原 backward 设备回执。
CPU 验证显式 g1 backend 消费原 g7 forward、main/index VJP 系数仍应用
一次、schedule 修改和 CP/TP 成员不一致被拒绝。

本轮硬件矩阵扩大到 joint/freeze/zero/eval 四种模式，以及 ordered/
reverse/non-reentrant checkpoint、strided/empty 和全部 CP×TP1/2/4。
forward 固定 g7、LI/KL 固定 14 slots，main backward 为 g1，FP32 参数
捕获与固定顺序 CP 参数归约启用；TP 投影使用默认 collective 策略。
其结果仍为诊断，完整模型/reference/performance gate 不通过则不放行。

四卡完整模式矩阵 `model_training_backward_g1_full/` 已结束，退出码为 0。
审计核对 72 个逻辑场景、288 rank-cases、576 rank-microbatches；
joint/freeze/zero/eval 各 72/72，ordered/reverse/checkpoint 各 96/96
严格累积契约通过。15 项原模型输出/hidden/参数累积检查、4 项 FP32
捕获检查与独立 CP 归约控制全部逐位一致；1152 个同源 CP 参数控制、
3168 个 SGD 单步控制通过。每次 main backward 三阶段均报告 g1、
20 logical tasks、每组 1 AIC/2 AIV；CP ready/write/ACK、TP owner epoch、
参数身份与选择集合有效。源码快照、冻结 payload 和报告加载库哈希匹配。

完整模型旧数值 gate 仍为 **0/288**，normalized reference、性能与
标准 checkpoint 验收未放行。576 个微批的 stock output/hidden/loss 对照
全部通过，stock SGD 为 288/288；累积 stock 参数仍在
`linear_qkv.weight`、`o_proj.weight` 全部失败。FP32 oracle 的主模型
参数/输出仍有失败；joint 的 indexer 参数及 SGD 后 `k_norm.bias` 也
保留失败。该结果只证明该 fixture/seed 的训练累积可重复，不能将
FP32 参数捕获与 BF16 stock 梯度求和的不同舍入路径视作逐位等价，
不能由局部重复性推导完整模型数值正确性。

审计为 `model_training_backward_g1_full_audit.json`，按模式残差字段
为 `model_training_backward_g1_full_residual_fields.json`；八卡同配置
结果另行记录。默认 backward 仍跟随 forward g7，g1 仍为显式诊断配置。

八卡 `model_training_backward_g1_cp8_full/` 随后结束，退出码为 0；
CP8×TP1、CP4×TP2、CP2×TP4 的 72 个逻辑场景、576 rank-cases、
1152 rank-microbatches 均完成。四种模式各 144/144、三种 replay 各
192/192 严格累积契约通过，2304 个同源 CP 参数控制、6336 个 SGD
单步控制逐位一致；协议/身份/选择与源码/payload/报告加载库审计通过。
完整模型旧 gate 仍为 **0/576**。两组共 864/864 严格契约通过，完整
模型为 0/864，不能据此启用默认低并行度策略或宣称融合训练加速。
八卡审计为 `model_training_backward_g1_cp8_full_audit.json`。

### 累积验收的显式 fixture seed

`mega_dsa_cp_tp_model_accumulation_validate.py` 增加 `--seed`。指定时
模型、重建投影、hidden、cotangent 分别使用 seed、seed+1、seed+2、
seed+3，仍按相同两微批 roll 和目标缩放执行；报告同时保存基 seed
与四个实际 seed，每个 case 保存基 seed。非法/溢出 seed 在输出目录
创建与 NPU/分布式初始化之前拒绝。不指定时保持原模型/投影/输入/
cotangent 的 20261007/733/20261008/20261011，数值 gate 不变。

默认种子的两组任务与源码审计均完成后才应用该选项；原报告与源码
快照保留，差异记录在
`training_backward_schedule_fixture_seed_post_run_source_transition.json`
及同目录 `training_backward_schedule_fixture_seed.patch`。下一轮使用
预留 candidate seed 20261252，完整复跑四卡/八卡四模式矩阵；该候选
结果不参与 reference profile 拟合，结果另行记录。

### 多组 backward 的 FP32 私有累加实验

当前组装的 CANN VecOp 在共享 dK/dV workspace 上执行
`SetAtomicAdd<float>()`。此前同状态控制只证明 g7 与 g1 的重复性不同，
不能指定该累加为唯一根因。为保留多组计算，新增显式的 native 实验路径：

- 默认 `partials[Tk,1088]` 继续使用共享累加；
  `partials[groups+1,Tk,1088]` 声明分组私有累加，groups 范围 1..19。
- slot0 保留原 CP transport 的 interleaved 输出；每个其他 slot 为该
  compute group 的连续 dK `[Tk,576]` 与 dV `[Tk,512]`，包含 RoPE。
  两个 AIV 在 phase0 分片清零，phase1 的 tile 只向所属 group 写入。
- phase1 完成后，预留 progress vector 使用固定 4096-float chunk，
  按 group0→group1→… 归并至原 retained dK/dV workspace。发布独立
  ready 后，post-cast reader 才能消费；随后仍执行原 CP owner return。
- 元数据、workspace、参数 identity、loss 缩放及 owner cast 语义保留。
  新能力由 `dsa_group_partials_version()` 声明；新增 patch 固定 SHA，
  应用于锁定源码副本。

额外普通 HBM 为 `groups × Tk × 1088 × sizeof(float)`，归并 UB 为
32 KiB；这只是源码容量，不是实测 HBM 或性能结果。group 内仍可能
存在累加顺序差异，不宣称任意形状的确定性，不放行 Torch 全局
deterministic mode。

该轮初次落地时仅完成 native 源码接入与 C++ 风格检查，尚未完成编译、加载、
Python bridge 配置/新增 ready 回执解码、数值重复性及 F+B+SGD/HBM
验收。独立构建目录为 `build/native/validation/mega-dsa-group-partials-20261011/`；
首个构建因隔离脚本的 SHMEM 输出路径遗漏而失败，保留 `build.log`，
路径修正后的构建记录在 `build_attempt2.log`。现有验收使用的 SHMEM
payload 字节哈希复查一致；冻结 payload 继续保留。

预留 seed 20261252 的四卡完整矩阵
`model_training_backward_g1_seed20261252_full/` 已结束，退出码为 0；
72 个逻辑场景、288 rank-cases、576 rank-microbatches 完成，四模式
各 72/72、三种 replay 各 96/96 严格累积契约通过。1152 个同源 CP
控制、3168 个 SGD 控制逐位一致；协议/身份/选择、实际四个 seed、
源码快照、冻结 payload 与报告加载库哈希审计通过。完整模型旧 gate
仍为 0/288，未消费 profile 或将候选用于拟合。审计为
`model_training_backward_g1_seed20261252_full_audit.json`，八卡任务仍
保留原句柄等待设备门禁。

私有累加构建的 `build_attempt2.log` 记录 host 编译成功后，CANN
编译工具导入因系统 OPP `vendors/config.ini` 不可读而失败；保留日志。
`build_attempt3.log` 使用私有 OPP 视图，其 built-in/库/版本链接到
同一 CANN，vendors 使用独立目录，系统文件未修改。该构建尚未验收。

`build_attempt3.log` 已通过 OPP 导入与 host 编译，设备编译发现
组装漏复制 `dsa_group_partials.h`。已在组装规则中补齐该 header，
并将其加入必需文件检查。`header-closure-proof/` 的实际锁定源码
组装通过，header 字节与本树一致，两个 vendor setter 均存在；
记录为 `header_closure_proof.json`。这仍不是 native 编译或 NPU
数值验收，后续构建与 bridge 接入另行记录。

补齐 closure 后的完整隔离构建已启动，记录在 `build_attempt4.log`。
原始 native 输入保存在 `native_sources_attempt4/`，对应哈希为
`native_source_identity_attempt4.json`；此时仍未声明编译/加载/硬件
验收通过。

seed 20261252 的八卡完整矩阵
`model_training_backward_g1_cp8_seed20261252_full/` 已结束，退出码为 0；
72 个逻辑场景、576 rank-cases、1152 rank-microbatches 完成。四模式
各 144/144、三种 replay 各 192/192 严格累积契约通过，2304 个
同源 CP 控制、6336 个 SGD 控制逐位一致。协议/身份/选择/实际 seed
及源码/payload/报告加载库审计通过，完整模型旧 gate 仍为 0/576。
两组 seed 20261252 合计严格契约 864/864，完整模型 0/864。

两组任务和审计结束后，训练 bridge 增加显式
`group_private_partials=False`，CLI 对应 `--group-private-partials`。
该配置在 CP/TP 冷声明一致；修改配置或缺少新 native 能力在发布/
lease 前拒绝。native 获得完整三维 scratch，公开 result.partials
仍为 slot0 的二维 canonical 输出，完整 scratch 在 runtime buffers
中保持异步生命周期。新回执校验要求 phase0/1 的私有归并 ready 为
0、phase2 为 3，随后仅在克隆的 CPU 快照中移除该扩展字段，再执行
原有完整 phase/CP owner 校验；原始回执不修改，默认模式仍拒绝扩展
控制字。源码差异记录为新构建目录中的
`group_private_bridge_final_source_transition.json` 和
`group_private_bridge_final.patch`。

CPU 目录回归为 **370 tests、769 subtests** 通过，覆盖 g1/g7、
local/empty Q、canonical 输出、scratch 生命周期、能力/成员不一致/
配置修改在 lease 前拒绝，以及 ready 缺失/提前/错误 epoch 拒绝。
Ruff/Pylint、C++ 风格检查通过；尚无新路径的硬件数值或性能结论。

完整隔离构建的第四次设备编译发现 nested tiling 缺少 synthetic
注册，补齐后目标编译进一步暴露 SDK 不支持当前标量 Min 重载与
copy 参数窄化。修正为有界标量分支和显式 uint32 字节数后，
`fused_grad_scalar_fix.log` 记录 fused backward kernel 编译成功。
605 个实际 vendor 源码文件与重新从 lock 组装的源码逐字节一致，
记录为 `final_source_byte_identity_recheck.json`；两个 MoE closure 中
未被 include 的 runtime header 副本在编译结束后刷新，并明确记入
该证明。恢复打包的第一次安装因缺少聚合 binary index 而拒绝，
保留 `build_recovery.log`，正在完成 CMake binary 聚合目标。

下一项硬件控制在同一 g7 forward 保存状态和 cotangent 上交错重放
shared g7、private g7、shared g1，各重复两次；同时校验归并 ready、
全部 logical tasks、CP/TP epoch、native/owner 输出和 shared return。
构建、加载及该重放全部通过前，不改变默认配置。

CMake binary 聚合目标完整结束、退出码为 0，保留
`binary_index_completion.log`。再次恢复安装/打包的
`build_recovery_complete.log` 完整结束、退出码为 0，保留正常 vendor
符号/kernel object/metadata/binary index 与 host ELF 检查，并编译
配套 Torch adapter；冻结 component payload 为新目录 `payload/` 的
95 个文件，记录 `payload_identity.json`。重用的 vendor 输入已与
fresh lock assembly 核对 605 文件字节一致；此前失败日志全部保留。

加载检查确认本树 Python、Torch 2.9.0+cpu/TorchNPU 2.9.0.post6、
fused grad ABI2、local CP grad ABI1、group partial ABI1。SHMEM binding
实际导入来自当前 payload 路径，已与新冻结 payload 做字节对照，
SHA256 相同；记录 `load_identity.log` 和
`shmem_framework_byte_identity.json`，不将该文件对照当作历史
process-map 全量证明。

两个同状态硬件控制 `private_group_cp4_empty/`、
`private_group_tp4_strided/` 均已完整结束并审计。
每组重放 96 次 native backward、32 次 shared return，native
覆盖 shared g7/private g7/shared g1 三种模式；有意义的第二次对
第一次比较每模式 16 对，第一轮自比较不计入重复性结论。
producer Python 与 driver/冻结 payload/实际源码 closure/报告加载库
及全部 CP/TP/新归并 ready 回执由
`audit_group_partial_control.py` 核对，结果见下节。
完整模型/reference/HBM/性能 gate 保持未通过。

### private g7 同状态 NPU 重放结果

`private_group_cp4_empty/` 与 `private_group_tp4_strided/` 均完整结束、
退出码为 0；各自在相同 forward g7、LI/KL 14 slots、保存对象与
cotangent 上交错重放 shared g7/private g7/shared g1，各两次。

| 第二次与第一次同模式比较 | CP4×TP1/empty | CP1×TP4/strided |
|---|---:|---:|
| shared g7 的任意 native 输出非逐位对数 | 4/16（活跃 rank0 的 4 对） | 16/16 |
| private g7 的任意 native 输出非逐位对数 | 0/16 | 0/16 |
| shared g1 的任意 native 输出非逐位对数 | 0/16 | 0/16 |

CP empty 另外 12 对来自无 query rank；第一轮自比较不计入上述
重复性结论。private g7 的所有 native raw gradients、FP32 partials
及 CP owner C/RoPE 均逐位可重复。两组合计 192 次 native backward
回执、64 次 shared return：全部 20 logical tasks、实际 group 数、
CP ready/write/ACK、private 归并 ready 与递增 CP/TP epoch 有效，
所有 shared 同状态输出逐位一致。producer/driver/冻结 payload/
605 文件源码 closure/报告加载库审计通过。

private g7 与原始 shared g7 输出有微小差异，不能宣称二者逐位
等价；CP4 控制的 FP32 partial 最大绝对差为
2.3283064365386963e-10、最大 relative L2 为
7.25078116991897e-08，未使用的零范数样本另计。该实验仅覆盖
一个 fixture seed、两个拓扑、四份保存状态及各两次重放，不能
宣称任意形状确定性或将观察归因于唯一内部机制；完整模型/
reference/HBM/性能验收仍未放行。原模型 case 使用 shared g7，
其严格累积分别为 0/4、4/4，完整模型均 0/4；这些原模型 case
不能计作 private g7 的训练 bridge 验收。

审计为 `private_group_cp4_empty_audit.json`、
`private_group_tp4_strided_audit.json`，汇总为
`group_private_repeatability_summary.json`。实际训练矩阵
`private_group_training_seed20261252_full/` 和
`private_group_training_cp8_seed20261252_full/` 均已完成：
forward/backward 均 g7，private partials 开启，LI/KL 14 slots，
四模式/三种 replay/两种分片/全部 CP×TP1/2/4，FP32 参数捕获和
固定顺序 CP 参数归约开启，TP 投影使用默认 collective。
分别 288、576 rank-cases；结果见下节，默认配置不改变。

### private g7 完整训练矩阵结果

上述四卡、八卡矩阵均完整结束、退出码为 0，并通过源码、冻结
payload、报告加载库及协议回执审计。fixture seed 为 20261252，
两 microbatch，覆盖 joint/freeze/zero/eval、ordered/reverse/
non-reentrant checkpoint、strided/empty 和 CP×TP1/2/4。

| 验收项 | 四卡 | 八卡 |
|---|---:|---:|
| 严格梯度累积契约 | 288/288 | 576/576 |
| 完整模型原有数值 gate | 0/288 | 0/576 |
| 同源 CP 参数归约逐位控制 | 1152/1152 | 2304/2304 |
| 单次 SGD 参数更新逐位控制 | 3168/3168 | 6336/6336 |

每个 case 的 native 累积、捕获的局部 FP32 参数 VJP、独立 CP
参数归约控制和 SGD 更新控制均通过；main backward 保持 g7，
全部 20 logical tasks 与 private 归并 ready 回执有效。该矩阵
证明所覆盖配置的训练 bridge 严格契约，不代表完整模型验收。
四卡各模式的残差通过数与失败字段计数，与同 seed 的 shared g1
矩阵相同；这不证明每个残差值相同或已定位唯一误差来源。

审计分别为 `private_group_training_seed20261252_full_audit.json`
和 `private_group_training_cp8_seed20261252_full_audit.json`；
残差分类对照为 `private_g7_vs_shared_g1_fourcard_residuals.json`。
reference 使用独立 NPU stock 路径和经 FP64 验证的 oracle；
现有归一化 reference profile 仍为 `held_out_failed`，未消费为
验收依据。完整全局 25 字段诊断覆盖四个预留 candidate seeds，
保持原有 gate、profile 和默认配置，当前结果如下。

### private g7 完整全局模型诊断

完整全局诊断已接入 `--group-private-partials`，report 与每个 case
记录启用状态，实际 backward 回执还要求 private 归并 ready。
首轮四卡 `private_group_global_full4/` 在首个 case 的 reference
环境身份检查失败、退出码为 1；没有完成的 global case。
旧 profile 的 `ASCEND_CUSTOM_OPP_PATH` 和加载库身份绑定旧冻结
payload，新 private payload 不能直接消费该 profile。八卡首轮
门禁在实际执行前取消，避免继续使用已知不匹配的 profile。

独立重新校准使用 `model_sgd_reference_private_payload/`，保持
相同 policy SHA、fit/held-out seeds、两次重复、四模式、H64/H128
及 FP64 oracle 验证，共 192 reference cases、384 samples。
它只运行 stock/reference 路径，不使用 private g7 candidate 的
数值拟合；旧 profile 和失败记录均保留。四卡、八卡诊断使用
匹配校准环境的 `_matchedref` 独立目录；
即使独立 profile 通过，metrics-only 诊断也不自动放行模型验收。

新 payload 独立校准现已完成全部 192 cases、384 samples；
producer 源码、冻结 payload、报告加载库、独立 fitter 重算、
全部 oracle/选择集/隔离检查和 4224 次单 AXPY 更新控制审计通过。
profile 仍为 `held_out_failed`，未改变 policy 或阈值规则：
H64 四模式的 `gradient/q_a_layernorm.weight` 及 H128 joint
的 `loss` 未通过 held-out，与旧环境的失败字段一致。
记录为 `model_sgd_reference_private_payload_audit.json`。

使用该匹配环境的全量诊断目录为：
`private_group_global_full4_matchedref/` 和
`private_group_global_full8_matchedref/`。每组覆盖四个预留
candidate seeds、四模式、三种 CP×TP 拓扑、两种分片，共
96 logical cases；四卡 384、八卡 768 rank-cases，每个 case
保留完整 25 字段与 11 参数的单次 SGD 控制。四卡已完成终态
审计，八卡在 loss 副本检查失败，见下方诊断；完整模型及性能
验收不改变。

四卡 `_matchedref` 全量矩阵现已退出 0、终态审计通过：四个
candidate seeds、四模式、CP4×TP1/CP2×TP2/CP1×TP4 和
strided/empty，共 96 logical cases、384 rank-cases。
全部完整全局 25 字段重建、CP/TP replica 一致性、独立 TP
checkpoint gather、private 归并 ready、20 logical tasks、选择集
与原参数身份验证通过；各 rank 的全局指标相同。4224 次单次
SGD 参数更新控制均逐位通过。

原有完整模型数值 gate 为 0/384。SGD checkpoint 的 stock
比较为 384/384，FP32 比较为 336/384；这些 checkpoint 结果
不能替代输出/hidden/loss/所有参数梯度验收。非 joint 模式
梯度 relative L2 最大约 0.006773；joint 的最大梯度残差位于
indexer：`k_norm.bias` 为 0.148735、`wq_b.weight` 为 0.114739。
记录为 `private_group_global_full4_matchedref_audit.json` 和
`private_group_global_full4_matchedref_residual_summary.json`；
这些极值仅为诊断证据，不定义新的数值阈值。

四卡 joint 模式中，五个 indexer 参数各自 96 个 rank/shard
样本对独立 stock 的原有 pointwise 与 relative L2 检查均通过。
各参数最大 candidate-to-stock rank/shard relative L2 为：
`k_norm.bias` 0.006264、`wq_b.weight` 0.004087、`wk.weight`
0.004705、`k_norm.weight` 0.005206、`weights_proj.weight`
0.004437。记录为
`private_group_global_full4_matchedref_joint_stock_comparison.json`，
含源审计和 rank report SHA256；该表是局部参数 shard 对 stock
的比较，不与完整全局 FP32 指标直接等同，也不证明唯一误差
机制。它支持先核对同 seed stock 对 FP32 oracle 的 joint 残差。

八卡 `_matchedref` 已退出 1：完成首个 seed 的 CP8×TP1 两种
分片、四模式，共 8 logical cases、64 rank-cases 后，在
CP4×TP2/strided/joint 的完整全局重建失败。
`replica_loss_cp4_tp2_20261251_joint.json` 显示四个局部 loss
贡献在 TP 两侧逐位相同，但各自 CP collective SUM 结果为
0.003228132613003254 与 0.0032281330786645412，相差 2 ULP。
已保留失败报告、源码快照及
`private_group_global_full8_matchedref_failure_audit.json`；
该未完成矩阵不能计作八卡全量验收。

冷模型诊断已改用现有 `ordered_parameter_sum`，按 CP
group-rank 顺序汇总 detached FP32 loss。局部 TP loss 副本先
经全 root 集体检查逐位一致，再进入 CP SUM；receipt 记录
`local_tp_replicas_bit_exact`、`cp_reduction_order=group_rank`
和一次 CP 归约。该修改不改变 native loss、梯度或训练 scaling，
也不放宽全局副本检查。正数标量的归约顺序与两 ULP 差异拒绝
回归共 9 tests、11 subtests 通过，Ruff/Pylint 通过。
旧结果到新源码的 SHA256/patch 记录为
`ordered_loss_source_transition.json`。

新的完整矩阵 `private_group_global_ordered_loss_full4/`、
`private_group_global_ordered_loss_full8/` 已提交门禁，仍覆盖每组
96 logical cases 和全部原有数值检查。四卡已退出 0、终态审计
通过，384 rank-cases 的完整全局字段和 loss 副本一致，4224 次
单次 SGD 控制逐位通过；原有完整模型 gate 仍为 0/384，stock/
FP32 checkpoint 通过数仍为 384/384、336/384。
八卡已退出 1，完成 35 logical cases、280 rank-cases；72 个
已完成 joint rank-cases 的固定顺序 loss 回执有效，原 loss
失败点已通过。随后在 seed 20261252、CP4×TP2/strided/eval 的
`gradient/linear_qkv.weight` 副本一致性检查失败。
旧失败未保存梯度张量，不能直接归因于某一种内部机制；
记录为 `private_group_global_ordered_loss_full8_failure_audit.json`。

全局入口现支持 `--parameter-reduction-order collective/group_rank`，
默认仍为 `collective`。控制模式复用已有 FP32 group-rank SUM，
保持一次 CP 参数归约后 cast，再执行一次 SGD；report 和 case
记录选择。全局重建失败时另外输出指定字段的副本是否相同、
不同元素数和误差指标，避免只留下字段名。
归约/重建/参数 VJP 捕获的 CPU 回归为 12 tests、15 subtests，
Ruff/Pylint 通过。源码转换记录为
`ordered_parameters_source_transition.json`。

新的 `private_group_global_ordered_parameters_full4/` 和
`private_group_global_ordered_parameters_full8/` 已提交门禁，显式
使用 `--parameter-reduction-order group_rank`，继续覆盖每组
96 logical cases。两组现均退出 0 并通过硬件终态审计；原有
阈值、默认 collective 模式和完整模型验收状态不改变。

| 固定顺序 loss 与参数 SUM 的全量诊断 | 四卡 | 八卡 |
|---|---:|---:|
| logical cases | 96 | 96 |
| 完整全局字段与协议检查有效的 rank-cases | 384/384 | 768/768 |
| 单次 SGD 参数更新逐位控制 | 4224/4224 | 8448/8448 |
| 原有完整模型数值 gate | 0/384 | 0/768 |
| stock checkpoint 比较 | 384/384 | 768/768 |
| FP32 checkpoint 比较 | 336/384 | 672/768 |

两组均覆盖四个预留 seeds、四模式、strided/empty 和 TP1/2/4；
完整全局 25 字段重建、CP/TP replica 严格一致、独立 checkpoint
gather、private 归并 ready、20 logical tasks、选择集与参数身份
均有效，所有 rank 的全局指标一致。loss 和参数归约顺序的
receipt、源码快照、冻结 payload、报告加载库和 reference
环境身份审计通过。该控制模式已通过此前遇到的两处副本
一致性失败点，不代表默认 collective 在任意形状逐位可重复。

审计为 `private_group_global_ordered_parameters_full4_audit.json`、
`private_group_global_ordered_parameters_full8_audit.json`，汇总为
`private_group_global_ordered_parameters_summary.json`。合计
1152 rank-cases、12672 次单 AXPY 控制；完整模型 gate 仍为
0/1152，归一化 reference profile 仍为 `held_out_failed`，
未消费为验收依据。模型数值、标准维度/full LM、长上下文、
HBM 与完整训练步性能验收仍未完成。
同 seed 的独立 stock 对照已完整结束、退出 0 并通过审计：
`reserved_joint_stock_diagnosis/`，仅在相同四个预留 seed 和
H64/H128 上测独立 stock 对 FP32 oracle 的 joint 残差，保留
FP64 oracle 验证并观测投影后 index states 的符号变化。
该诊断不消费或拟合 reference profile；CPU 点积的符号变化
只描述所捕获数值状态，不作为 native 内部累积或 ReLU 分支
执行的证明。8 个 joint cases、88 次单次 SGD 参数更新控制，
以及全部 oracle/选择集/隔离、源码/driver/冻结 payload/报告库
和 reference 环境身份检查均通过。记录为
`reserved_joint_stock_diagnosis_audit.json`。

stock 对 FP32 的 `k_norm.bias` 梯度 relative L2 在
H64/20261251 为 0.097003、H128/20261254 为 0.147701；
对应四卡 private g7 完整全局指标范围分别为
0.096647–0.096830、0.148648–0.148735。
这表明相近量级的大残差也存在于独立 stock 路径，尚不能将其
归因于 CP、融合或唯一内部机制。两组 stock 的该 bias SGD
更新 relative L2 分别为 0.097465、0.147951，不能用其他
checkpoint 参数的通过掩盖这个字段。

四个 seed/H64/H128 的投影状态点积符号变化数分别为
8/4/6/5 和 2/0/7/6；每个 case 观测 3904 个合法 query/head/key
组合，点积在 CPU FP64 上从捕获的数值状态计算。
同 seed 对照记录为
`private_group_global_full4_matchedref_stock_global_residual_comparison.json`，
含两份源审计 SHA256；五个配对 indexer 梯度字段的 FP32 oracle
expected norm 一致，只证明这组指标可配对，不证明梯度张量
逐位相同或 native 的实际 ReLU 分支。

### 主干 VJP 与 compressed 梯度精度边界

`mega-dsa-main-vjp-20261012/` 的三个八卡控制均退出 0 并完成
审计：seed 20261252、CP4×TP2、strided、eval、private g7、
四项 FP32 参数 VJP 捕获、固定顺序 loss/参数 SUM。
每组仅一个 logical case、8 rank-cases，不能扩称全矩阵验收。
运行源码/driver/冻结 payload/报告库、全局重建和协议身份有效。

`linear_vjp_cp4tp2_seed20261252_eval_audit.json` 显示
`linear_qkv` 的输入与前向输出对 stock 逐位相同，但 incoming
cotangent 的 relative L2 为 0.002948。固定 native cotangent
后，独立 BF16 linear 的权重 VJP 与 native leaf 逐位相同，
独立 FP32 linear 的 hidden VJP 在转回 BF16 后也逐位相同。
捕获的 FP32 局部权重 VJP 按 CP 顺序相加，相对独立全局 FP32
收缩的 relative L2 为 1.961799e-08，最大绝对差
9.313226e-10；该控制不支持将误差归因于 linear 本身。

`upstream_vjp_cp4tp2_seed20261252_eval_audit.json` 显示 Q/KV
norm 的输入、前向输出对 stock 逐位相同；使用各自实际
upstream 的 norm 自重放 VJP 也逐位相同。KV norm upstream
的 relative L2 为 0.002860、input VJP 为 0.003024；Q norm
对应为 0.000201、0.000300。main compressed 输入梯度差异
为 0.002860；分别将全局 FP32 DK/DV 转成 BF16 再相加的
诊断控制，对 stock 的差异为 0.000107，而合并后一次 cast
的控制与当前 native compressed VJP 逐位相同。

`compressed_rounding_cp4tp2_seed20261252_eval_audit.json` 的
counterfactual 只在诊断进程中改变返回给 autograd 的 C 梯度：
全局 DK 和 DV 分别 BF16 cast，再做 BF16 add；实际 native
FP32 owner 传输与协议仍执行。其影响为：

| 比较项 | 原路径 | C 梯度舍入 counterfactual |
|---|---:|---:|
| linear incoming cotangent 对 stock relative L2 | 0.002948 | 0.000132 |
| linear 权重梯度对 stock relative L2 | 0.003517 | 0.000239 |
| hidden 梯度对 stock relative L2 | 0.003284 | 0.000322 |
| stock 模型比较 | 0/8 | 8/8 |
| FP32 模型比较 | 0/8 | 0/8 |
| 完整模型原有 gate | 0/8 | 0/8 |

counterfactual 的 FP32 比较新增 `kv_a_layernorm.weight` 失败，
不能以恢复 stock 比较为由采用该舍入路径。保留原 FP32 owner
契约，未修改 native/模型运行源码；报告新增
`main_compressed_vjp_precision`，显式区分 native 的 FP32 合并/
CP/TP SUM 后一次 activation cast、stock 的逐分支 activation
cast 后 add，以及无中间 activation cast 的 FP32 oracle。
元数据改动前后的快照转换为
`precision_declaration_source_transition.json`，诊断汇总为
`compressed_rounding_counterfactual_summary.json`；原有阈值、
profile 和完整模型/HBM/性能 gate 均未放行。

### 非融合 FP32 compressed VJP 控制

[`mixed_fp32_control.py`](mixed_fp32_control.py) 提供显式诊断 API
`mixed_sfa_fp32_backward_control`：使用完整 packed query、已保存的
真实 SFA output/max/sum 和实际 BF16 cotangent，不执行 CP/TP 通信。
它从 mixed backward 的锁定 workspace 读取 FP32 dK/RoPE 和 dV，
在 FP32 中合并 dK+dV 后只做一次 activation dtype 转换。
H128 按两个 H64 调用切分 query heads、cotangent 和原始统计量；
共享 dK、dV、key RoPE 的跨 head 累加也在 FP32 中完成。

返回的 FP32 planes 与梯度拥有独立 storage；同 stream 复制/相加
后才复用 native scratch，调用方仍须显式等待后再读取 CPU 快照。
返回每个 head chunk 的三阶段 trace，需逐份验证，不安装 autograd
backend。group 原子累加仍可能不重复，不承诺任意 shape 确定性。

该控制与融合候选共享锁定 vendor gradient tile 实现，只用于隔离
融合、通信和精度边界；不能替代独立 stock/CPU FP32/FP64 oracle，
也不能拟合或消费新的验收 profile。现有 stock SFA gradient 输出
接口只支持 BF16/FP16，不提供独立的 FP32 dK/dV output。
完整模型数值、HBM 和训练步性能要求继续保留。

真实 NPU 控制 `unfused_fp32_cp4tp2_seed20261252_eval_attempt4/`
已退出 0，并由 `unfused_fp32_cp4tp2_seed20261252_eval_attempt4_audit.json`
完成审计：一个 H128、CP4×TP2、strided、eval logical case，
seed 20261252，8 rank-cases、88 次单 AXPY 参数更新控制。
原有 native/private merge/owner return 协议、选择集、参数身份、
源码/driver/冻结 payload/报告库及 reference 环境检查均通过。

在同一实际 native cotangent 下按 g1/g7/g7/g1 重放非融合控制，
每次执行两个 H64 chunks，共验证 24 份三阶段 trace。四次重放
的 query、compressed、query RoPE、key RoPE 最终 BF16 VJP，
全部与当前融合 CP4×TP2 的全局重建结果逐位一致。
原始 FP32 dK/dV 相对融合路径的 relative L2 为
7.086e-08–7.647e-08 / 7.118e-08–7.332e-08，并非逐位等价。
g1 第二次重放的 FP32 planes 逐位相同；g7 planes 有约
6.861e-08 / 6.917e-08 的重复差异，但该案例的最终 BF16
VJP 仍逐位相同。此结果不推广为任意 shape 的原子确定性。

| VJP | 非融合控制对固定状态 CPU FP32 relative L2 |
|---|---:|
| query | 0.002901 |
| compressed | 0.002140 |
| query RoPE | 0.002822 |
| key RoPE | 0.003051 |

同一 cotangent 的 stock 重放与非融合控制在 query/query RoPE/
key RoPE 上逐位相同；compressed 的 relative L2 为 0.002862，
保留逐分支 BF16 转换与 FP32 合并后转换的精度契约差别。
该案例未观察到额外的融合/通信最终 VJP 差异，不能据此定位
完整模型的全部误差或放行独立数值验收。完整模型 gate 仍为
0/8，profile 未用于验收；汇总为
`unfused_fp32_boundary_control_summary.json`。

同一成功运行的 CPU fixture 另作固定状态 FP32/FP64 连续路径
对照：output relative L2 为 6.065e-08，四项 VJP 为
1.884e-07–2.933e-07，记录于
`fixed_state_cpu_precision_control_verified.json`。这里比较实际
BF16 表示的主干状态和固定 native cotangent，不替代完整模型
的 unabsorbed oracle。最初三次诊断失败分别独立归档；首次异常
分配后 rank 间进入不同 collective 的现场栈与主动终止原因均
保留，不作为候选 HBM 测量或数值通过证据。

### 四卡、八卡与 joint 固定状态 VJP 扩展

`mega-dsa-joint-vjp-20261012/` 的 `joint_boundary_full4/` 与
`joint_boundary_full8/` 均退出 0 并通过审计。四卡覆盖 CP4×TP1
（H64）和 CP2×TP2（H128），八卡覆盖 CP4×TP2（H128）；
每个拓扑都使用 seed 20261252/20261254、joint/eval、strided、
plain replay。合计 12 logical cases、64 rank-cases、704 次
单 AXPY 参数控制及 240 份非融合三阶段 trace。原有 native
private merge、CP/TP owner return、选择集、参数身份、全局重建、
replica、源码/driver/冻结 payload/报告库/reference 环境检查通过。
审计分别为 `joint_boundary_full4_audit.json` 和
`joint_boundary_full8_audit.json`，汇总为 `joint_boundary_summary.json`。

每个 case 用实际 native cotangent 做 g1/g7/g7/g1 非融合主干重放。
四项 VJP 共 192 次比较，182 次逐位相同。query、query RoPE 和
key RoPE 各 48/48 相同；compressed 为 38/48。两个四卡 eval
的 g7 样本差异 relative L2 为 5.923e-09 / 6.467e-07；八卡
seed 20261254 的 joint/eval 中，四次 compressed 对照均有
1.293e-06、max abs 5.960e-08 的差异。后两个 case 用 native
自身 FP32 dK/dV 按全局平面先相加再 cast 的控制，也观察到这个
差异，说明“保留 FP32 后只 cast 一次”不保证不同求和结合次序
逐位一致；未据此改动 native 合并/归还或将其归为唯一根因。

joint 另捕获三个 indexer 状态与实际 VJP，使用固定同一主干
teacher、选择集、loss coefficient/normalization 和 upstream
做 stock 与 CPU FP32/FP64 对照。6 个 joint cases 的 18 项
stock 自重放均逐位相同；CPU FP32/FP64 VJP relative L2 为
4.876e-07–8.195e-07。H128 的 CP2 与 CP4 在两个 seed 上的
indexer 对照指标完全相同，只证明配对指标一致，不证明所保存
张量逐位相同。主 SFA 的四项 VJP 与 indexer VJP 分开捕获，
KL teacher 在源码和 CPU oracle 中均 detached。

八卡对同一 stock 原始 indexer VJP 另做 FP32 coefficient/upstream
乘法、最后一次 BF16 cast 的缩放控制：

| seed | query 对 native | weight 对 native | key 对 native relative L2 |
|---|---|---|---:|
| 20261252 | 逐位相同 | 逐位相同 | 0.002641 |
| 20261254 | 逐位相同 | relative L2 4.648e-06 | 0.002677 |

原 stock 缩放对应 query/weight relative L2 约为
0.002712/0.002512 和 0.002734/0.002305。该控制表明缩放的
中间 BF16 舍入解释了这两个案例的 query 差异以及大部分 weight
差异；stock 原始 key 梯度已经 BF16，无法从中恢复 native 的
FP32 累加器，因此不以这个控制宣称 key owner return 验收。
八卡 joint 的 `fixed_index_cpu_fixture.pt` 保留实际状态、raw stock
梯度、native 梯度、两种缩放结果和 CPU FP32/FP64 对照。

报告新增 `indexer_vjp_precision`，明确 native IQ/weight 的 FP32
缩放后一次 cast、IK 的 FP32 缩放/CP/TP SUM 后一次 cast，以及
stock 的 activation dtype raw VJP 和两次 activation dtype 乘法。
它只是报告契约，未改动运行梯度；本次运行快照到新报告字段的
源码转换保存在 `indexer_precision_source_transition.json`。
四卡 root 额外重复了一份 indexer 诊断字段；审计核对它与所有
rank 共享的内容一致后，显式去重比较主要指标，未宣称原始 JSON
完全相同。完整模型 gate 仍为 0/64，profile 未用于验收，标准
维度/full LM、长上下文、HBM 和完整训练步性能要求仍未完成。

### indexer LayerNorm 上游与 ReLU 区域控制

`mega-dsa-keynorm-vjp-20261012/keynorm_boundary_full8/` 的两个
CP4×TP2、strided、joint cases（seed 20261252/20261254）退出 0，
并通过 `keynorm_boundary_full8_audit.json` 审计。16 rank-cases、
176 次单 AXPY 参数更新控制、6 项 stock LayerNorm 自重放、24 项
native 局部 LayerNorm 自重放以及 4 项 CP 参数求和控制通过。
所有实际 native 局部 input/weight/bias VJP 对独立固定状态重放
逐位相同，局部 BF16 leaf 梯度按原 CP 顺序 FP32 求和、转换后，
也与实际 global leaf 逐位相同。原有完整模型 gate 仍为 0/16。

| seed | norm upstream 对完整 FP32 relative L2 | native bias 对完整 FP32 | 固定 native upstream 的 FP32 norm bias 对完整 FP32 |
|---|---:|---:|---:|
| 20261252 | 0.007170 | 0.007464 | 0.006634 |
| 20261254 | 0.110166 | 0.147916 | 0.148644 |

这组控制不支持仅修改 LayerNorm 参数归约来消除 seed 20261254
的大残差：它已经存在于进入 norm backward 的 upstream；固定
这个 upstream 后的独立全局 FP32 norm，仍保留约 0.1486 的 bias
差异。未采用 LayerNorm 精度修改。

随后从实际保存的 BF16 主干/indexer 状态及完整 FP32 模型重构
状态，在 CPU 公式中分别替换 teacher、index states 和 ReLU
backward mask。`cpu_index_precision_attribution_extended.json`
及其审计保留了两个 seed、各 12 个变量控制。两个 seed 都有
3904 个合法 query/head/key 点积；在 CPU FP32 和 FP64 算出的
状态分支变化数分别都是 0 / 6。这里没有捕获 native 内部真实
ReLU 分支，不以 CPU 分支数宣称 native 的唯一执行机制。

seed 20261254 的 bias relative L2 对照为：

| 独立 CPU 控制 | bias 对完整 FP32 relative L2 |
|---|---:|
| 实际 BF16 表示状态 | 0.147450 |
| 仅替换 FP32 teacher | 0.147440 |
| 仅替换 FP32 index query | 0.009409 |
| 仅替换 FP32 index key | 0.011799 |
| 替换全部 FP32 index states、保留原 teacher | 0.000538 |
| 只替换 ReLU backward mask、保留原 primal loss | 0.006359 |

backward-only mask 控制属于诊断数学，不是可采用的运行梯度。
同 mask 自重放的 raw VJP 逐位相同，换 mask 后的 primal loss
也逐位相同。CPU 控制采用单线程；多线程版本的同 mask 梯度重放曾未逐位
一致，单线程版本通过，未唯一定位该内部差异。这不改变候选
运行。重新构造的完整 FP32
upstream/bias 对已归档 FP32 oracle 并非逐位相同，relative L2
约为 3.54e-07–4.20e-07 / 6.49e-07–7.90e-07，明确保留在报告中，
没有修改验收阈值或以此放行 profile。

“全部 FP32 index 投影后再 BF16 cast”也只是变量控制：它把
seed 20261254 的 bias 差异降到 0.004950，却把 seed 20261252
从 0.005736 扩大到 0.021378。因此未采用这个转换方案。
锁定 stock KL 定义的 index query/key 只支持 FP16/BF16，local
mixed KL 的对应输入只支持 BF16；允许 FP32 merge weight 不代表
允许 FP32 index query/key。该能力来自当前锁定定义及 Torch
wrapper 源码检查，不是 FP32 native 运行验收。

汇总为 `keynorm_boundary_summary.json`，CPU 源码/fixture/自重放/
primal/指标审计为 `cpu_index_precision_attribution_audit.json`。
本轮没有改变 native/model 梯度，没有采用新的精度路径，没有
拟合或消费 profile；完整数值、标准维度/full LM、长上下文、HBM
与完整训练步性能验收继续保留。

### FP32 index selected-KL 原语与实际 CP/TP 归约

`fp32_index_kl.py` 与 `HyperDsaFp32IndexKl` 增加显式 FP32 index 路径：
index Q `[Tq,H<=128,128]`、完整 K `[Tk,128]` 和已缩放的 signed weights
`[Tq,H]` 均保留 FP32。该路径接收调用者提供的 detached teacher 和固定
`global_packed` selection `[Tq,2048]`，不隐式改变 LI selection 或模型投影精度。

- phase 0 计算本地 index heads 的 `sum(weight * ReLU(Q @ K))`；padding logits 为零。
- 调用者在 TP 内 SUM logits，再将相同的 global logits 与 teacher 交给 phase 1。
- phase 1 在有效 selection 上计算稳定 softmax、每 query KL 和 FP32 dQ/dK/dweight。
  对非单位 teacher mass，logit VJP 使用 `p_student * sum(teacher) - teacher`。
- dK 是完整 key buffer 上本地 query/head 的贡献。调用者先 CP SUM、再 TP SUM，
  然后进行 owner return 与声明的投影精度转换；没有额外 TP divisor。
- coefficient、全局 valid-query normalization 和 loss cotangent 由训练接入层各应用一次。
- 空 query shard 不提交 native kernel，仍返回完整的 FP32 零 dK 参与归约。
- selection、forward inputs 和 logits 有 saved-version 检查；非 FP32 states 提前拒绝。
  teacher 的有限性、非负性与 padding 零值是调用者准备分布时的前置契约。

当前是独立 native 原语。模型/autograd backend、FP32 投影和融合 teacher/selection
生产者尚未接入；现有融合模型默认行为保持原来的精度声明。native dK 使用 FP32
atomics，拒绝 deterministic algorithms；不承诺不同 invocation 的逐位一致性。

2026-10-12 验证目录：
`build/native/validation/mega-dsa-fp32-index-kl-20261012/`。
独立 payload 与 CANN kernel、Torch adapter 已编译、加载，canonical/assembled
新算子源码、runtime Python、payload 和已报告加载库均经字节审计。
编译阶段使用自有 OPP view 读取 vendor 配置，设备运行使用原 CANN OPP；
这两个环境用途分开记录。

单卡覆盖 packed `(3,42)` 的正/负/零点积、零 teacher 项、teacher mass `0.75`，
以及保留 seed `20261252`、`20261254` 的完整 FP32 index states 和 detached teacher。
三个样例各运行两次，并用两份 head shard 检查 global logits 与局部 VJP 的合成。
CPU FP32/FP64 独立 oracle 与设备输出的固定逐元素门槛为
`rtol=2e-5, atol=2e-6`，没有拟合或消费模型误差 profile。
保留种子的 dQ/dK/dweight relative L2 约为 `4.6e-7` 至 `7.7e-7`，
loss relative L2 约为 `2.9e-6`。

实际 HCCL `CP2×TP2` 的三个样例分别运行 `strided`、`last_empty`，
合计 24 个 rank 样例，其中 6 个 rank 样例为空 query；全部通过独立 CPU
完整 VJP 对照和每 block phase receipt 检查。
CP 与 TP SUM 后的 dK 在四个 rank 间逐位一致；五项指标的最大 relative L2
分别约为 logits `3.05e-7`、loss `4.80e-6`、dQ `6.55e-7`、dK `6.02e-7`、
dweight `1.06e-6`。空 query rank 同样得到完整归约后的 dK。

实际 HCCL `CP4×TP2` 同样运行三个样例与两种分片，合计 48 个 rank 样例，
其中 6 个 rank 样例为空 query；全部通过。五项指标的最大 relative L2 分别约为
logits `3.72e-7`、loss `7.85e-6`、dQ `1.43e-6`、dK `5.75e-7`、dweight `1.56e-6`。
归约后的 dK 在八个 rank 间逐位一致；四卡与八卡共 72 个 rank 样例，
不等长 packed、交错 CP 分片与空 rank 的归约协议均通过。

加载失败、逻辑 view/flat storage 几何失败和首轮 VJP receipt 失败均保留在该目录。
ACLNN 封装已补 `ACLNN_API` 导出及 `Contiguous`/`ViewCopy`；receipt 改用各 AIV
独立的 32-byte MTE3 写入，修正后的严格设备检查通过。
以上原语与归约证据不构成完整 LM+KL 模型验收，也不证明长序列性能或融合收益。

### 显式 FP32 KL 训练分支

`FusedDsaLocalCpTpTrainingProbe(..., fp32_index_kl=True)` 与模型替换 context
`dsa_fp32_index_kl=True` 必须同时声明，默认均为 `False`。该分支保留原 BF16
index projections 供融合 LI 生产 selection，另行计算 FP32 index projections
供辅助 KL 使用。主干 attention 与其 backward 继续使用现有融合路径。

模型侧从 detached hidden 和原始参数值重新计算 FP32 Q-latent/RMSNorm，再计算
FP32 index Q、index K/LayerNorm、merge weights 和 RoPE。主干 Q-latent 参数、
主干 hidden 不接收 KL 梯度；index 参数对象保持原身份。显式 FP32 区域关闭
外层 autocast，避免 Float inputs 被线性层重新降为 BF16。

`Fp32IndexKlProbe.from_saved_attention()` 仅接受真实融合 forward 的 local
saved-state receipt，验证原生产者的 token、版本与 row layout，并将
sequence-relative IDs 在设备上转换到 caller-ordered `global_packed`。
该入口没有任意 device selection tensor 的绕过路径，也不把 selection 拷回 host。

`fp32_fused_kl.py` 负责以下步骤：

1. 冷声明验证完整 CP owner key storage，预先建立 global packed permutation。
2. 运行时 gather detached FP32 index K；空 owner 仍参与同一个 gather。
3. 在 BF16 主干已保存状态的表示值上计算 FP32 selected teacher head probabilities，
   TP SUM 后显式归一化 teacher。每 query scratch 有界，不构造整个 batch 的
   `[Tq,H,2048,512]` 张量。
4. native FP32 index logits 经 TP SUM 后计算 raw FP32 KL VJP。
5. 训练 autograd 保存当前 invocation 的 raw VJP，coefficient/global-query
   normalization 在 forward 各应用一次，loss cotangent 在 backward 应用一次。
6. dK 通过现有 native CP owner return，再与主干 shared fields 一起执行一次 TP SUM；
   返回 FP32 index activation VJP，最后由投影 autograd 转换到原 Parameter dtype。

FP32 K gather 没有 autograd SUM；共享 key 梯度的 CP 与 TP SUM 由原 owner-return
协议各承担一次，防止 gather backward 再次归约。retained backward 使用独立
native return epoch；checkpoint replay 重新准备 forward invocation。

`mega_dsa_cp_tp_model_global_validate.py --fp32-index-kl` 声明该候选并记录
BF16 LI/FP32 KL 的不同精度边界。模型 reference policy、held-out profile 与
原有数值门禁不随此开关修改。该分支当前增加独立 HCCL gather/SUM、teacher
计算与 native KL 提交；它不是新的设备内完整融合或性能默认选择。

CPU mocked tests 已覆盖 reordered/empty CP owner gather、异步 handle 完成前
不读取输出、teacher head mass 合成、真实 saved selection 版本校验、FP32
投影与独立 Float 模型一致、原 Parameter 身份与 KL 梯度隔离，以及 retained
backward 的一次缩放和 FP32 owner return。386 个测试、777 个子测试通过。
模型 NPU F+B+SGD 的新证据归档在
`build/native/validation/mega-dsa-fp32-training-20261012/`；结果须以该目录的
terminal rank reports 和源码/payload 审计为准，CPU 测试不能替代设备验收。

FP32 训练分支的独立解析审计保存在验收目录的
`initial_packets_analytic_audit.json`。该审计只使用归档的四份首轮 fixed-state
packet，以 NumPy FP64 显式计算 softmax、teacher-mass KL、ReLU mask 与
Q/K/weight 导数，不调用 autograd。它与 CPU FP64 autograd 的最大 relative L2
约为 `1.83e-15`；设备梯度的最大 relative L2 约为 `7.77e-7`，loss 约为 `6.19e-6`。
这是固定 teacher/global logits 的原始 KL 数学核对，不包含完整模型接受结论。
后续四卡、八卡归档也由 `audit_fp32_packets.py` 使用相同固定门槛重新核对。

新 payload 的 custom OPP 路径与旧 reference profile 不同，严格环境检查已拒绝
直接复用旧 profile。验收使用 `reference_new_payload/` 重新采集独立 stock
reference，保持原 policy、fit/held-out seeds、重复次数和 margin，保留旧 profile。
诊断流程只有在确认完整 reference sampling 和源文件身份后才开始模型矩阵。

### FP32 KL 完整模型执行与数值诊断结果

2026-10-12 的独立 reference 重采已完成 192 个 case / 384 个样本，
4224 次 BF16 SGD single-AXPY control 与所有 independent model oracle proof
通过，原 policy SHA256 保持
`e81b03d3507a2fed077b05a289bf7ef4069f30c3565439c20df964eb09559bec`。
新 profile SHA256 为
`466210a5c0ebaf59de6a7c55e5cc18b2806a9faa451a09e70bb6e03d434d452f`，
状态仍为 `held_out_failed`，没有数值接受能力。新旧 profile 均保留，
未修改 margin，也未用融合候选测量拟合 reference。

| 拓扑 | rank case | single-AXPY control | fixed-state FP32 KL control |
| --- | ---: | ---: | ---: |
| CP2×TP2 / 四卡 | 64 | 704 | 64 |
| CP4×TP2 / 八卡 | 128 | 1408 | 128 |

两种拓扑分别覆盖 seed `20261252`、`20261254`，`strided` / `empty`，
joint 的 plain / retained / non-reentrant checkpoint，以及 eval plain。
完整 F+B+SGD 执行、新 branch 的 native receipts、CP return 与 TP ACK、
retained 梯度逐位重放、完整 25-field global reconstruction、CP/TP replicas、
独立 checkpoint gather、原 Parameter 身份和原 BF16 LI selection 均经检查。
运行退出码、审计退出码均为 0；矩阵以 `metrics_only` 运行，完整模型接受仍为 false。

设备 raw FP32 KL 梯度对 CPU FP32 oracle 的最大 relative L2：
四卡 dQ `5.41e-7`、dK `5.68e-7`、dweight `8.93e-7`；
八卡 dQ `1.05e-6`、dK `1.31e-6`、dweight `1.47e-6`。
归档的全部 192 个 fixed-state packet 另经 NumPy FP64 显式解析导数核对，
它不调用 autograd；该检查与 CPU FP64 autograd 的结果一致。

与之前相同 seed、topology 和 pattern 的完整全局诊断对照，
seed `20261254` 的 indexer `k_norm.bias` 梯度 relative L2 从约 `0.148`
下降到 `0.00185` 至 `0.00297`；`wq_b.weight` 从约 `0.115`
下降到 `0.00176` 至 `0.00248`。两个 seed 的 index 参数梯度和 KL loss
均改善。主干 `q_a_layernorm.weight` 的 relative L2 指标与该对照相同，
此项仅比较归档指标，不是新旧完整梯度 tensor 的逐位比较。

结果、源码和 105-file payload 审计、223-file runtime Python identity、
加载库校验、独立 reference 审计及同 fixture 数值对照分别保存在验收目录的
`full4_audit.json`、`full8_audit.json`、`reference_new_payload_audit.json`、
`full4_analytic_packet_audit.json`、`full8_analytic_packet_audit.json` 和
`fp32_training_summary.json`。

当前候选解决了所测 joint indexer 梯度的主要精度偏差，仍保留独立 HCCL
及 teacher/native KL 提交。完整模型数值门禁、标准 checkpoint 维度、
长序列/HBM/性能以及更完整设备融合尚未完成，默认开关保持关闭。

### Query norm 的 FP32 VJP 候选

`FusedDsaCpTpAttention` 的 context 可显式设置 `dsa_fp32_query_vjp=True`。
该候选要求同时启用 QKV、KV 和 query 参数 VJP capture，并且原
`q_a_layernorm` 必须是直接 scale 的一维 `nn.RMSNorm`，具有 affine weight
和明确的正 epsilon。默认值为 false。

该路径保持原 BF16 QKV、Q-B、absorbed Q 和 RoPE 前向值：QKV 与 Q-B
的原始 rounded 值用 FP32 表示，absorption 仍以原 BF16 operands 计算。
Absorption 的输入 VJP、Q-B 的输入 VJP/TP SUM、query norm 的输入 VJP
和 QKV 参数 VJP 之间不再插入 BF16 cast。Query norm 的 scale VJP 通过
原 Parameter 的 FP32 view capture，在 CP 参数归并之后才转换为叶子 dtype。
KV norm、native main backward 和 RoPE 的源 dtype 边界仍按原路径执行。

`source_rms_norm_with_fp32_vjp` 使用保存的 BF16 norm 输入，在 FP32 中计算
直接 scale RMSNorm 的解析一阶导数。它保留原 Parameter identity/version，
支持空 owner、延迟 backward、retain_graph 和 non-reentrant checkpoint，
拒绝二阶 backward。这是明确声明的混合精度导数，不等同于 stock BF16
backward，也不代表原始完整 FP32 模型的数值接受。

`mega_dsa_cp_tp_model_global_validate.py --fp32-query-vjp` 记录该开关。
CPU mocked suite 已通过 392 个测试、796 个子测试；其中新检查分别验证
BF16 前向值、FP32 norm cotangent、独立 FP64 导数、原 Parameter 身份、
checkpoint/retained 重放和源参数修改检测。设备诊断归档在
`build/native/validation/mega-dsa-fp32-query-vjp-20261012/`，须等完整
terminal reports、源码/payload 审计和独立 packet 检查完成后评价候选。

2026-10-12 的 query VJP 候选四卡、八卡执行和独立审计均以退出码 0 完成。
两种拓扑都覆盖 seed `20261252` / `20261254`、`strided` / `empty`，
joint plain / retained / non-reentrant checkpoint，以及 eval plain。

| 拓扑 | rank case | 精确 single-AXPY 更新 | fixed-state FP32 KL control | FP32 norm VJP packet |
| --- | ---: | ---: | ---: | ---: |
| CP2×TP2 / 四卡 | 64 | 704 | 64 | 80 |
| CP4×TP2 / 八卡 | 128 | 1408 | 128 | 160 |

240 份 norm packet（其中 80 份为空 owner）均通过独立 NumPy FP64
解析导数和 CPU FP64 autograd 核对；设备 FP32 norm VJP 的最大 relative L2
为 `9.103e-8`，最大 absolute error 为 `3.655e-10`。解析检查使用固定
`rtol=2e-5, atol=2e-6`，没有拟合 margin。全部 192 份 KL packet 也重新
通过原独立解析检查。

另将本轮与上一 FP32 index KL 候选的同 fixture packet 直接比较：
1920 组 fields 的 selected main states、完整 index states、teacher、logits
和 indices 逐位一致。原始 main key staging 的未请求 rows 可保留 NaN
poison，因此 main key 比较只覆盖实际 selected rows，并单独要求这些
rows 全部有限；它不将未读取 scratch 当作逻辑模型状态。

完整 global gradient 诊断中的 query norm scale relative L2 如下：

| fixture seed | 上一候选在两种 CP 拓扑/布局的 relative L2 | 本轮 relative L2 |
| --- | ---: | ---: |
| `20261252` | `0.006533`–`0.006772` | `0.005856` |
| `20261254` | `0.005949`–`0.006024` | `0.005717` |

Q-B、QKV 和 hidden 梯度指标也小幅改善；output、KL loss 与 KV-B
梯度的归档 relative L2 指标相同。此处的指标比较不声称完整 tensor 逐位相等。
完整 F+B+SGD、原 Parameter 身份、LI selection、CP/TP replicas、25-field
重构、checkpoint gather、retained 梯度逐位重放、native ready/write/ACK
和 index owner return 均经原严格机械检查。

证据位于上述验收目录的 `full4_audit.json`、`full8_audit.json`、
`full4_norm_packet_audit.json`、`full8_norm_packet_audit.json`、
`full4_analytic_packet_audit.json`、`full8_analytic_packet_audit.json`
和 `query_vjp_summary.json`。每轮冻结 224 个 runtime Python 文件和
105-file native payload；reference source transition 独立列出仅影响候选
路径的五个修改文件，新 norm 模块也纳入 candidate source identity。
独立 stock sampler 的依赖、policy 和 profile bytes 均未变，profile 仍为
`held_out_failed`，本轮只使用 metrics，没有完整模型数值接受或性能接受。
默认开关保持关闭，主干 native/activation 精度、完整模型门禁、标准维度、
长序列/HBM/完整 step 性能及设备融合仍需继续推进。

### Compressed KV 到 norm 的 FP32 VJP 候选

`dsa_fp32_compressed_vjp=True` 扩展上一 query VJP 候选，要求同时启用
`dsa_fp32_query_vjp` 及其三个投影 capture。原 `kv_a_layernorm` 必须是
具有 affine scale 和明确正 epsilon 的一维 `nn.RMSNorm`。训练 probe
也必须显式设置 `fp32_compressed_vjp=True`；该配置参与 CP/TP cold
collective declaration，构造后修改配置会在 native backward 前失败。
默认开关保持关闭。

该路径将原 BF16 KV norm 前向值表示为 FP32 C autograd input。进入 native
forward 时仅将其恢复为 BF16，hard selection、teacher 与 native main
前向仍使用原 BF16 states；同时保存原 FP32 input 和独立 BF16 native
saved state，保留延迟 backward 的输入版本校验。Native backward 的
FP32 dK+dV 经过原 CP owner return 和 TP SUM 各一次后，以 FP32 直接
返回 C input。KV norm 的解析一阶 VJP、QKV activation VJP 和 KV norm
scale capture 之间不再插入 BF16 cast。KV norm scale 的原 Parameter
仍在 CP 参数归并后才转换为叶子 dtype。

此开关只扩展共享 C 到 KV norm 的精度边界；Q/Q-RoPE/K-RoPE native
梯度和 RoPE 的 source dtype 边界仍按原路径执行。它没有增加一次
CP/TP 归约，也没有改变损失系数、全局 token normalization 或 KL
梯度隔离。`mega_dsa_cp_tp_model_global_validate.py --fp32-compressed-vjp`
记录此候选，原 reference policy/profile/数值门槛不随该开关改变。

CPU mocked 检查覆盖未被 BF16 舍入的 owner sum、source forward 一致、
FP32 KV norm scale VJP 与独立 FP64 公式、原 Parameter 身份、plain /
retained / non-reentrant checkpoint、原始 C input 修改检测以及配置和
输入 dtype 拒绝。设备诊断归档在
`build/native/validation/mega-dsa-fp32-compressed-vjp-20261012/`，完整
NPU 执行和独立 packet 审计完成之前，不据此声明模型数值接受。

2026-10-12 的 compressed VJP 候选四卡、八卡完整执行与独立审计均以
退出码 0 完成；CPU mocked 全套为 397 个测试、808 个子测试通过。
两个拓扑均覆盖 seed `20261252` / `20261254`、`strided` / `empty`，
joint plain / retained / non-reentrant checkpoint，以及 eval plain。

| 拓扑 | rank case | 精确 single-AXPY 更新 | fixed-state KL control | query + KV norm packet |
| --- | ---: | ---: | ---: | ---: |
| CP2×TP2 / 四卡 | 64 | 704 | 64 | 160 |
| CP4×TP2 / 八卡 | 128 | 1408 | 128 | 320 |

480 份 norm packet 各含 FP32 upstream、原 BF16 norm input、FP32 scale
和设备 VJP，其中 query/KV norm 各 240 份，空 owner 共 160 份。
全部经过独立 NumPy FP64 解析导数与 CPU FP64 autograd 双重核对；
设备 norm VJP 最大 relative L2 为 `9.103e-8`，最大 absolute error
为 `8.050e-10`。固定门槛仍为 `rtol=2e-5, atol=2e-6`。192 份 KL
packet 也通过原固定门槛与独立解析检查。与上一 query VJP 候选相比，
1920 组 selected main / complete index states、teacher、logits 和 indices
逐位一致；main key staging 仍只比较实际 selected 且有限的 rows。

同 fixture 的完整 global gradient 诊断如下，范围覆盖两个 seed、两个
CP 拓扑及两种布局的 joint plain；eval plain 也有对应独立记录。

| 字段 | 上一 query VJP 候选 relative L2 范围 | 本轮 compressed VJP 范围 |
| --- | ---: | ---: |
| `kv_a_layernorm.weight` | `0.004508`–`0.005364` | `0.004214`–`0.004480` |
| `linear_qkv.weight` | `0.004548`–`0.004603` | `0.003891`–`0.004003` |
| `hidden` | `0.004395`–`0.004471` | `0.004103`–`0.004134` |

Query norm 的归档 relative L2 指标与上一候选相同；该指标比较不等于
完整梯度 tensor 的逐位对照。原 Parameter 身份、原 LI selection、CP/TP
replicas、25-field 全局重构、checkpoint gather、retained 梯度重放、
ready/write/ACK 和 owner-return 检查均通过。归约次数仍为 CP、TP 各一次。

证据保存在本轮验收目录的 `full4_audit.json`、`full8_audit.json`、
`full4_norm_packet_audit.json`、`full8_norm_packet_audit.json`、
`full4_analytic_packet_audit.json`、`full8_analytic_packet_audit.json`
与 `compressed_vjp_summary.json`。
冻结身份为 224 个 runtime Python 文件和 105-file native payload；
reference source transition 独立列出六个候选路径修改文件。
Reference stock sampler 的依赖、policy/profile bytes 均未修改，profile
继续为 `held_out_failed`，本轮只记录 metrics。完整模型数值门禁、
标准维度、长序列/HBM/完整训练步性能及进一步设备融合仍未完成。

锁定源码还显示 dQ workspace 在 post 前保存 FP32 累加值，ND post
才应用 attention scale 并转换为 BF16。该源码观察记录在
`next_query_workspace_source_observation.json`；实际 device layout、scale
次数、BF16 post 重建和 scratch 生命周期尚需单独验收，不能将此观察
当作下一 query VJP 路径已经实现或通过设备验证。

### Native dQ workspace 的 FP32 VJP 候选

`dsa_fp32_native_query_vjp=True` 依赖 compressed/query VJP 两个已有候选，
使 absorbed Q 的原 BF16 forward 值以 FP32 表示。Training probe 同时声明
`fp32_native_query_vjp=True`，该配置参与 CP/TP cold declaration；不同
precision 的 Q input 或构造后配置修改会在 native backward 前失败。
Native forward 仍恢复为原 BF16 Q/C，LI、teacher 和 main primal 不改变。

`query_workspace.scaled_query_workspace_vjps` 只解码锁定 arch22 gradient
ABI 2/local CP ABI 1 的 TND C512/Dr64/H32/H64 workspace。每 logical core
的 mm1/mm2、selected K/V 和 p/ds scratch 之后，dQ plane 按 ND
`[local_T,local_H,576]` 保存 FP32 累加值。解码器按锁定 ND post 的 FP32
attention scale 缩放一次，拆分 Q/RoPE 并生成独立、连续的 FP32 tensors；
不保留对可复用 scratch 的 view。Empty query 不解释特殊 empty layout。
调用方必须使用 producer stream 或显式 event 保证计算完成。

本轮将 FP32 dQ 经保存的 local query layout 恢复 caller 顺序，直接送回
absorption 与 Q-B 的 FP32 VJP。Q-RoPE、K-RoPE 仍返回原 BF16 native
梯度；C 保留已有 FP32 owner return。原 Parameter、损失缩放、归约次数
和 KL 梯度隔离不变；默认开关关闭。

CPU mocked 全套为 403 个测试、830 个子测试通过，覆盖 H32/H64、20-core
cap 两侧的 ND 布局、Q/RoPE 拆分、FP32 scale、空 query、截断/错位
workspace、scratch 覆写、source BMM 前向值与未舍入 cotangent、caller
顺序、checkpoint/retained 和 precision declaration。设备证据归档在
`build/native/validation/mega-dsa-fp32-native-query-vjp-20261012/`，须经过
terminal reports 与独立 packet 审计后才评价候选。

2026-10-12 的 native query VJP 候选四卡、八卡完整执行和所有独立审计
均以退出码 0 完成。覆盖两个保留种子、strided/empty，以及 joint 的
plain/retained/non-reentrant checkpoint 和 eval plain。

| 拓扑 | rank case | 精确 single-AXPY 更新 | FP32 query packet | query/KV norm packet | fixed-state KL packet |
| --- | ---: | ---: | ---: | ---: | ---: |
| CP2×TP2 / 四卡 | 64 | 704 | 80 | 160 | 64 |
| CP4×TP2 / 八卡 | 128 | 1408 | 160 | 320 | 128 |

全部 240 份 query packet 以 NumPy FP32 独立计算 raw workspace 的原
post scale，结果逐位一致。重建 BF16 post 时必须使用锁定 native
`CAST_ROUND` 的 nearest/ties-away-from-zero 规则，不能用 Torch 默认
的 ties-to-even 替代。本轮 80 个中点 cells 在这两种 BF16 转换规则下
不同，均经过原始 FP32 bit pattern 核对；使用 native 规则后，Q 与
Q-RoPE post 的全部字段逐位一致。这是 workspace 布局、scale 和 cast
的验证，不等于独立完整 attention 数学导数接受。

首次诊断误用 Torch BF16 转换，在某个 rank 的 post 断言失败后进入
root 错误汇总，而其余 rank 仍等待 subgroup prepare。watchdog 失败
和确认 collective 顺序分歧后停止的诊断均保留原始现场。修正诊断
后，post 数值检查在完整 collective 序列结束后对归档 packet 执行，
没有扩大 rtol/atol，也没有修改设备梯度或原 reference profile。

另完成 32 次单卡设备 geometry 控制：H32/H64，T=1/3/21/41，g1/g7，
每组重复两次；覆盖 20-core cap 两侧、packed sequence 和 scratch
overwrite。该控制使用既有 stock SFA 前向及 host 三阶段 mixed grad，
按其实际协议检查 `validate_mixed_trace`。32 份 query/RoPE post 均可
按 native rounding rule 逐位重建，已返回 FP32 tensors 不受后续
scratch 清零影响。此控制不宣称新增完整模型 H32 拓扑接受或性能收益。

相对上一 compressed VJP 候选，完整 global gradient 的 relative L2
在两个种子、两种拓扑及两种布局的 joint plain 中如下：

| 字段 | 上一候选范围 | 本轮范围 |
| --- | ---: | ---: |
| `q_a_layernorm.weight` | `0.005717`–`0.005857` | `0.005328`–`0.005748` |
| `q_b_proj.weight` | `0.005742`–`0.005863` | `0.005587`–`0.005708` |
| `kv_b_proj.weight` | `0.004211`–`0.004440` | `0.004203`–`0.004428` |
| `linear_qkv.weight` | `0.003891`–`0.004003` | `0.003884`–`0.003989` |
| `hidden` | `0.004103`–`0.004134` | `0.004051`–`0.004112` |

1920 组 selected main / complete index states、teacher、logits 和 indices
与上一候选逐位一致；原 Parameter 身份、CP/TP replicas、25-field
重构、checkpoint gather、retained 梯度逐位重放、ready/write/ACK 和
owner return 均通过原严格检查。480 份 norm VJP 和 192 份 KL packet
也通过原固定门槛与独立解析导数检查。

证据索引为本轮目录的 `native_query_vjp_summary.json`、四卡/八卡
`audit`、`query_packet_audit`、`norm_packet_audit`、`analytic_packet_audit`
及 `query_geometry_audit.json`。冻结 225 个 runtime Python 文件和
105-file native payload；本轮没有 native rebuild。独立 reference 的
policy/profile bytes 和 stock sampler 依赖未修改，仍使用
`held_out_failed` profile 记录 metrics。完整模型数值接受、标准维度、
长序列/HBM/完整 step 性能和进一步设备融合仍未完成，默认开关关闭。

### FP32 RoPE VJP 实验路径（默认关闭）

`dsa_fp32_rope_vjp=True` / `--fp32-rope-vjp` 要求同时启用
FP32 query、compressed、native query VJP 及其参数捕获前置条件。
`source_rope.py` 调用原有 BF16 RoPE 算子生成前向结果，以 FP32 表示这些值，
反向使用 FP32 的 rotary map 转置。支持 half 和 interleave（输出转换为 half layout）；
position embeddings 保持原值且必须不可训练。输入和频率版本由 saved tensor 校验，
只支持一阶反向。

训练桥从已验证的 native ND workspace 读取 FP32 dQ-RoPE，恢复 caller query 顺序；
K-RoPE 使用既有 FP32 CP owner 返回和一次 TP SUM，在到达投影前保留 FP32。
这不改变 native 前向值、Top-K、teacher、KL 或原 Parameter 身份。
空 owner 在模型入口沿用 `position_embeddings=None`，无需执行 native rotary 算子。

设备原语对照共 16 个 case：half/interleave、B=1/2、T=1/3、单 batch 或逐 batch
频率、BF16/FP32 频率。前向与原算子逐元素一致，FP32 VJP 与独立 FP64 autograd
Jacobian 最大相对 L2 误差为 `4.270451069290456e-8`。

4 卡 CP2×TP2 的 64 个 rank case 已完成，704 次 SGD 单次 AXPY 核对通过，
包括 strided/empty、两枚 reserved seed、plain/retained/checkpoint 和 eval。
独立 NumPy FP64 RoPE 审计覆盖全部 60 个实际执行的 rotary backward packet；
另外 20 个 empty-query backward 按模型入口契约绕过 RoPE。
640 个 selected main/complete index/teacher/logit/indices 字段与上一轮逐元素一致。

数值变化不一致：q_a_layernorm.weight 相对 L2 在 seed 254 从约 `0.005328`
下降到 `0.004980`，在 seed 252 从约 `0.005747` 上升到 `0.006019`。
因此该路径仅作为显式实验选项，不作为默认优化或全模型精度通过的依据。
沿用冻结 reference policy/profile，未调整阈值、重新拟合或消费失败的 profile。
本节的设备原语通过和多卡协议通过均不等于全模型验收或性能收益。

原语 packet、模型报告和独立审计归档于
`build/native/validation/mega-dsa-fp32-rope-vjp-20261012/`。

8 卡 CP4×TP2 也已完成：128 个 rank case、1408 次精确 SGD 更新、128 份独立 KL
packet、320 份 norm packet、160 份 native query workspace packet，以及 100 份实际
执行的 RoPE backward packet。60 个 empty-query backward 按入口契约绕过 RoPE。
NumPy FP64 rotary 转置审计最大相对 L2 为 `3.218126455099013e-8`；全部 1280 个
selected main/complete index/teacher/logit/indices 字段与上一轮逐元素一致。
4/8 卡的 q-norm 数值变化方向一致，hidden 误差同样呈现一枚 seed 改善、另一枚变差。

当前源码 CPU 回归为 `407 tests / 856 subtests`，Ruff/Pylint、文档目录检查和
`git diff --check` 通过。源码快照包含 226 个运行时 Python 文件，native payload
保持 105 个文件的冻结身份，本轮未修改或重建 native 二进制。
完整对照见该归档目录的 `rope_vjp_summary.json`；全模型验收仍为 false，
冻结 profile 仍为 `held_out_failed`。后续仍需完成标准模型/长序列训练、
PP/FSDP/热副本共存、HBM 和完整训练步性能验收，以及目标范围内进一步设备融合。

### 标准投影维度与显式 packed 长度诊断

`examples/mega_dsa_standard_model_common.py` 新增标准维度随机权重 fixture：
H128、Hi64、hidden7168、Q-LoRA1536、V128、C512、Dr64、Di128、K2048。
配置来自 DeepSeek-V3.2 的固定 revision
`a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`；归档同时保存原始 config、
inference 源码、下载 URL 和 SHA256。没有加载预训练权重，也没有运行完整 LM。
主/index RoPE 布局及 YaRN attention scale、FP32 correction-band 频率表显式保留；
packed sequence 使用序列内 position，重新开始的序列从 position=0 计算频率。

`examples/mega_dsa_standard_model_validate.py` 使用显式配置、payload 和输出目录，
运行实际 HP attention 的 CP×TP 前向、反向、全参数 SGD 以及独立 checkpoint gather。
它采用原 BF16 前向、FP32 index KL/Q/norm/C VJP，RoPE VJP 保留原路径。
`--lengths`、`--pattern`、`--replay` 可分别声明 packed 长度、strided/empty owner
和 plain/retained/checkpoint。当前独立未吸收 dense oracle 要求每条序列长度
不超过 K=2048；超过该范围会明确拒绝，不能用完整历史 oracle 替代真实 selected DSA。

全局字段重建和仅诊断的 SGD 指标收集支持 `declared_shapes`，由完整 source/template
参数形状提供。默认仍使用原小模型形状；数值 acceptance evaluator、旧 reference
policy 和阈值不变。声明另一套形状不会消费旧预算，也不会返回数值通过。

标准维度 CPU FP64 absorbed/unabsorbed 对照验证了 output、loss、hidden 和全部
11 个参数梯度，共 14 个字段；最大相对 L2 为 `1.822512308262805e-15`，
最大绝对误差为 `1.4988010832439613e-15`，使用预先固定的 `rtol=1e-9 / atol=1e-11`。

设备验证已完成 4 卡 CP2×TP2 的两个 case：packed13/strided/plain 和
packed17/empty/checkpoint。共 8 个 rank case、88 次完整参数 single-AXPY 核对，
精确 CP/TP replica、选择集合、Parameter 身份、梯度归还 ACK 及独立 TP checkpoint
重建通过。源码快照包含 228 个运行时 Python 文件，native payload 仍为冻结的
105 个文件，实际加载库及 editable import 路径另行记录并核对。

本轮当前源码 CPU 回归为 `411 tests / 860 subtests`；最终受影响文件 Pylint/Ruff
和 `git diff --check` 通过。报告为 `diagnostic_finished`，全模型数值验收为 false。
没有将标准 attention 层的短序列执行证据算作长上下文、预训练 checkpoint、
完整 LM、HBM 稳态或性能验收。

完整证据位于 `build/native/validation/mega-dsa-standard-shapes-20261012/`，
包括 `full4_plain_audit.json`、`full4_checkpoint_audit.json` 和
`standard_fp64_oracle_proof.json`。

8 卡 CP4×TP2 的标准维度 packed13/strided/plain case 随后完成并通过同一机械审计：
8 个 rank case、88 次完整参数 single-AXPY 核对。4/8 卡同 fixture 的 output、loss、
q-norm 相对 L2 分别一致为约 `0.00385832 / 0.000620831 / 0.00579294`；
hidden 分别约 `0.00446416 / 0.00446392`。这些数字仅是显式数值诊断，不是新预算。
本轮三种设备 case 合计 16 个 rank case、176 次精确更新，汇总见
`standard_geometry_summary.json`。8 卡 root 的 CP4×TP2 不替代 CP8×TP1：
后者在标准 H128 下仍需扩展并设备验证当前仅接纳 local H32/H64 的 native 路径。

### 实际 selected set 的有界 CPU oracle

`selected_cpu_reference.py` 提供 `selected_attention_cpu` 和 `selected_kl_cpu`。
输入必须为同 dtype 的 CPU FP32/FP64；实现逐 query 展开实际选中的 KV，
反向重算 softmax，不保留 `[Tq,H,K]` probability 或 `[Tq,K,C]` selected KV 图。
显式 per-head K/V 支持未吸收 MLA；共享 C 同时传入 K/V 两条 autograd edge 时，
Torch 将独立 dK+dV 加回原 C 一次。原 tensor 和 indices 的 version 由 saved tensor
检查，支持普通/retained/non-reentrant checkpoint 的一阶 VJP；统计量不可微。
逐 query scratch 有界，完整投影激活、输出和参数梯度仍随模型规模增长。

KL 在同一 selected set 上计算每个主 head 的 probability，head sum 后 L1
归一化并 detach teacher；ReLU 只作用于 index dot，merge weights 保留符号。
loss coefficient、global normalization 和 downstream auxiliary cotangent 各应用一次。
空候选行输出/损失/梯度为零；packed/future filtering、重复 ID 和 missing causal KV
遵循小 CPU oracle 的契约。现有 CANN reference 的 K2048/有效候选数量契约未放宽。

`examples/mega_dsa_selected_model_reference.py` 在实际 selection 上展开独立
W_UK/W_UV 并运行未吸收模型，保留全部原参数梯度。
标准维度验证入口的 `--selected-oracle` 先执行 fused 路径，按 CP owner 重建其
真实 global packed indices，再计算 selected oracle。stock indexer 继续独立生成
selection；选择集合对齐单独报告，不由 CPU oracle 重新选 Top-K 或补成完整历史。
reference 重建和计算的异常在成员间同步传播，再进入后续梯度归约。

CPU 检查覆盖普通 autograd 的主/KL 梯度、独立逐 head K/V 数学、gradcheck、
C alias dK+dV、teacher 梯度隔离、signed weights、empty rows、retained/checkpoint、
source version 和 CPU autocast 精度。另有 4100-token 历史、K2048 的实际选中子集测试：
未选中的巨大 value 不影响输出或梯度，也不保留完整 query×head×K 图。
当前回归为 `420 tests / 862 subtests`，受影响文件 Ruff/Pylint 通过。

本轮归档目录为 `build/native/validation/mega-dsa-selected-cpu-oracle-20261012/`。
CPU 原语和小模型对照通过不代表长序列设备或全模型数值验收通过。

标准维度 CP2×TP2 的单序列 L2050/K2048 case 已完成：4 个 rank case、44 次
完整参数 single-AXPY 核对，所有 actual/independent stock selection sets 对齐。
每个 rank 归档都包含 2 个合法历史超过 K 的 query，oracle 消费的是实际选中的
2048 个 key；候选数量、去重、packed causal 范围和 stock 集合均独立审计通过。
output/hidden/loss/q-norm 的相对 L2 约为
`0.00446395 / 0.00491987 / 5.20917e-7 / 0.00540626`，仅为诊断数字，未拟合预算。
源码快照包含 230 个运行时 Python 文件，native payload 仍为冻结的 105 个文件。

首次 head-strided CPU oracle 的未完成运行保存在该目录的
`first_head_strided_oracle_run/`，不计入通过结果。随后在原 tensor version/edge
保持原样的前提下，逐 head selected KV 使用连续工作布局；共享 KV 使用矩阵
乘法，减少重复逐 head matvec。CPU 同几何微测的 dot 结果逐位一致，
这些微测不构成 native 或完整训练步的性能结论。布局修复后的完整 CPU 回归仍为
`420 tests / 862 subtests`。L8192 的完整设备结果需单独验收，不由 L2050 外推。
