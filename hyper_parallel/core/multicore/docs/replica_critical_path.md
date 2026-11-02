# Replica device critical-path model

The CPU replica planner can score bounded candidates with an execution-specific
device dependency model. The predictor models device work; the quota solver still
runs on the host. The first template supports MegaMoE Push/Pull with
`shmem_signal_kernel_gradient_bands`. Native and other transports retain their
existing scoring.

The business API remains `replica_slots_per_rank=B_max`. Adaptation is on by
default, with the existing hidden `HYPER_PARALLEL_REPLICA_COST_MODEL` switch.
There is no new model argument or adaptive flag.

## Modeled dependencies

`DeviceCriticalPathCalibration.estimate()` schedules immutable quota layouts on
exclusive Cube and calibrated communication resource domains:

1. Credit precedes weight delivery. Measured fanout batches share their owner's
   communication endpoints. Complete-expert readiness is the default unit.
2. Home compute and ordered guest compute share each rank's Cube queue. Guest
   compute waits for its weights. Copy/compute overlap is credited only when
   `overlap_verified` is true in the matching calibration.
3. Checkpoint recompute repeats the forward template. Verified resident backward
   weight reuse removes backward weight and credit tasks.
4. Each gradient band waits for its guest producer and a conservative complete
   home producer fence. The same expert element retains its peer reduction order.
   Owner workers join W2 before processing W13; the final ACK sweep waits for all
   owned projections, peers and bands.
5. Guest buffer release waits for its owner's ACK. The score exposes final owner
   readiness, phase makespans and separately calibrated host overhead.

Without measured band data, gradient return starts after complete backward
compute. Explicit bands contain `(projection, ready_fraction, local_ms, remote_ms)`:
projection 0 is W2 and projection 1 is W13. Both projections are required, fractions
must be ordered within each projection, and early release requires verified
overlap. Fractions describe progress through the guest's complete backward
compute. Band indices map to the configured one to four return workers.

The optional task trace uses phase-local timestamps. Recompute has the forward
makespan and repeats that template; it does not retain another copy of the trace.
This coarse model is advisory. It is not a tile simulator, a mathematical upper
bound, or a complete FSDP/optimizer/whole-network timeline.

## Admission and memory

The true B0 reference uses separate owner-only compute curves. A zero-guest B>0
decision includes a separate measured rejection overhead; that overhead cannot
be counted as savings available to replicas. A replica must beat the true B0
reference and the current candidate by the configured gain margin, including at
least twice the calibration uncertainty.

The planner compares finite budget skeletons and completed candidates, followed
by the existing bounded locality candidate. It does not use the legacy separable
lower bound or two-rank incremental quota refinement for this model, because
changing one rank can change another owner's shared-resource queue. Candidate
enumeration has no rank-local wall-clock cutoff.

Push distinguishes the soft receive target from safe storage allowance. A matching
critical-path profile can compare candidates within committed receive storage or
an explicit hard Heap budget. With no explicit Heap limit it retains the existing
bounded growth allowance, enlarged only to already committed capacity. It does
not query instantaneous free HBM. Allocation checks and capacity-required replicas
remain active; `B_max` still reserves physical guest pools.

## Calibration and exact matching

Existing package JSON files remain valid. A new entry adds a `critical_path`
object inside `model`; selector fields remain `soc_name`, `num_experts`, `top_k`.
The object has these fields:

| Field | Meaning |
|---|---|
| `schema_version` / `template` | `1` / `replica_ready_ack_v1` |
| `execution_signature` | SHA256 of the deployed execution contract |
| `home_forward_ms`, `home_backward_ms` | True B0 per-expert compute curves starting at `(0, 0)` |
| `intra_node_gradient_ms`, `inter_node_gradient_ms` | Full physical gradient service per expert when no bands are supplied |
| `communication_domains` | One nonnegative endpoint-domain label per EP rank; labels are scoped by node |
| `overlap_verified` | Explicit evidence for overlap; defaults to false |
| `gradient_bands`, `gradient_workers` | Optional measured band services and the deployed return-worker count |
| `credit_ms`, `weight_ack_ms`, `gradient_ack_ms` | Protocol readiness and completion service durations |
| `planner_host_ms`, `rejection_host_ms` | Observed host work for active and zero-guest paths |
| `uncertainty_ms` | One-sided prediction uncertainty used for admission |

The parent model's compute curves describe enabled-path compute service, and its
weight/link/fanout fields describe physical weight service. Its legacy exposed
gradient residual and legacy planner/replica overhead fields are ignored by the
critical-path scorer. Do not attach this object to old residual fits without
revalidating their compute and transfer interpretation. Physical matrix return
time and exposed fused-kernel gradient tail are different quantities.

The signature binds backend, H/I, EP, expert count, TopK, transport, B_max, weight
layout, activation clamp, kernel splits, normalized node mapping, checkpoint mode,
BF16 weights, FP32 partials, execution template, and the actual adapter/vendor
binary contents. Installation prefixes are excluded. Matching profiles and native
payload identity join the existing EP resource-manifest agreement; no additional
steady-state collective is added. Signatures are obtained by the worker from the
live module and resources, rather than guessed from an offline configuration.

Exact critical profiles are preferred. An unmatched signature falls back to the
legacy matching table or the existing load/capacity policy. The existing
9382/9392 shared-calibration rule remains. No production critical-path coefficients
are shipped by this implementation: synthetic correctness profiles must never be
installed as performance calibrations.

## Device collection and validation

After activating the matching CANN and native payload before importing Torch,
run the self-contained worker with a normal distributed launcher:

```bash
python -m torch.distributed.run --nproc-per-node=4 \
  tests/torch/expert_parallel/_benchmark_replica_critical_path.py \
  --output /path/to/component-validation --synthetic-correctness
```

This explicitly synthetic mode verifies automatic model selection, real guest
execution, checkpoint replay, ragged/empty ranks, outputs, input/router/weight
gradients, and AdamW updates for Push/Pull x B0/B1/B6. It provides numerical and
runtime integration evidence, not calibrated prediction accuracy or speedup.

For EP64 device observations, use the cluster's normal launcher to supply 64 or
128 ranks and the deployed environment. For example, pass this worker payload:

```bash
tests/torch/expert_parallel/_benchmark_replica_critical_path.py \
  --output /path/to/observations --hidden 5120 --intermediate 2304 \
  --experts 384 --ep-size 64 --top-k 6 --tokens 4096 \
  --repeats 4 --observations-only
```

All EP groups execute the same six configurations in one process lifetime, with
balanced, intermediate, hot, ragged and empty route phases. Observation mode keeps
finite-value checks and avoids retaining full-width CPU precision references.
An optional `--profile-directory` loads candidate JSON files privately in the
worker; installed package files remain untouched.

Each rank records the exact effective model, topology, receive allowance, source
histogram, slots, dispatch, CPU planning samples, cache reuse, native bypass count,
and fused forward/recompute/backward device events. Events around the fused kernel
exclude surrounding module/control work. Whole component wall time is recorded
separately. This resolves decision provenance before comparing predictions.

Use `_benchmark_replica_links` for independent full-matrix link service samples.
Band-ready fractions, shared-domain contention, ACK tails and overlap require
matching device timeline evidence; do not infer all of them from a fused-kernel
duration. Fit and test on disjoint histograms, preserve slow samples, and compare
profiler-off fresh-process rank-max medians before accepting whole-model gains.
