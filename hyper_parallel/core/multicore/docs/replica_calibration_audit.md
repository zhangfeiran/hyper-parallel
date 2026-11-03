# Strict offline replica calibration and static-band v2 reference

This workflow reads a frozen capture and writes independent analysis artifacts.
It never installs coefficients, changes a running bundle, or acquires NPUs.

## Raw completion and provider-only holdout

```bash
python -m hyper_parallel.core.expert_parallel.hot_replica.calibration_audit \
  --run-root /path/to/replica-critical-path-run --output /path/to/separate-audit
```

The audit requires all node stage exits and all per-rank worker completion
records. Directory existence is not training evidence. Terminal NPU snapshots
are reported, but their presence does not certify PID cleanup.

Physical provider services fit one EP group and validate on disjoint EP groups.
Actual topology, matrix bytes, adapter identity, software/hardware selectors and
sample completeness are checked. The result retains every sample and signed,
absolute and relative holdout errors. These provider observations are neither
fused-kernel exposed residuals nor a complete execution identity. Candidate
records remain `installable=false` while missing full execution binding.

`replay_decision` validates immutable source counts, slots, dispatch, capacity
and execution identity before calling the exact original CPU planner. Fresh
replay mismatches remain explicit. Recompute-only records are separate because
receive allowance can legitimately change after the original plan was frozen.

`validate_fit_sample` requires native, host and residency identities, a matching
verified schedule contract, actual decisions, F/R/B device observations, host
wall and exposed costs, clean sampling and a valid clock domain. Missing fields
are rejected; no synthetic values are supplied. `decision_metrics` reports
wrong direction, false admissions, missed benefits, uncovered samples and
rejected identities rather than hiding them behind average fit error.

Observation mode is explicit even when native binary identity is unchanged.
Production-off, external Level0 and internal causal instrumentation must not
silently share a calibration domain.

## Why the first dependency template is not a production calibration target

The deployed static executor traverses each worker's band then all owned edges,
in stable target-rank/slot order. An expert-specific heap permits readiness
reordering that the native queue does not. A worker completes its W2 work before
its W13 work, but only worker0 joins all workers and publishes ACKs per projection.
Other workers can enter W13 before the W2 ACK sweep. W2 ACK does not wait for W13.

Owner and guest producer flags are independently required, including no-tile
contributions. Band ordinals and groups restart within each projection. Up to
four groups distribute work across all physical workers; they are not four
physical workers. Weight SDMA uses a different engine from AIV return work
(MTE2 remote reads, Vector FP32 add, MTE3 writes).

Coefficients cannot compensate for these different dependency graphs. The
original template remains unchanged for reproducibility; the offline v2
reference is separate and does not replace production scoring.

## Stable v2 interface

`static_band_reference.py` owns scheduling semantics. Its pure data interfaces are:

| Interface | Contract |
|---|---|
| `gradient_band_layout(rows, columns)` | Whole-row FP32 bands, ragged final band, at most 64 |
| `static_band_queue(plan, rank, worker, projection, bands, physical_workers)` | Immutable `ReturnOperation` tuple; no timing required |
| `BandObservation` | Owner/guest readiness per physical producer and service per participating return worker, with explicit epochs and clock domain |
| `static_band_reference(...)` | Fixed worker queues, per-worker projection progression, per-projection ACK quorum and final incoming ACK waits |

Observation keys are `(projection, owner_rank, target_rank, target_slot, band)`.
Projection 0 is W2, projection 1 is W13. Readiness arrays include every physical
producer. Unassigned stripe services can be `None`; missing assigned service or
any producer readiness remains uncovered. ACK service is an explicit argument.

Owner gradient readiness and kernel completion are separate outputs. Local AIC
dW may finish after return-worker quorum, and final guest retention also waits
incoming ACKs. This reference evaluates declared logical elapsed durations; it
does not assume an unmeasured interconnect arbitration policy or calibrated
production timings. It always reports `production_calibrated=false`.

Publication visibility may lie inside a producer's data-fence/publication
interval. A consumer may observe a flag before the publisher samples its end
counter. Do not force consumer-ready to follow every publication interval end,
clamp a signed diagnostic difference, or subtract raw counters across ranks.
Censored or unaligned readiness must not be converted into an exact scalar.

Host acceleration may replace immutable knot interpolation and original-order
serial accumulation, preserving exact floating-point results. It must not change
these queues, candidates, ACKs, cache state or plan objective. A host path change
has its own calibration identity. Residency queries must be immutable and must
not call prepare/prefetch, consume credits, mutate epochs or alter LRU state.

## Validation boundary

Hand-calculated tests cover blocked heads, per-worker W13 overlap, per-projection
ACK, physical workers versus groups, guest-slot insertion order, multiple owners,
ragged geometry and missing/stale producer observations. Real observed queues can
be checked with `static_band_queue` without inventing readiness times.

Stateful component screening preserves the original load windows and slow
samples. For each matched backend and EP group, reduce per-step rank maxima before
median/p95. Profile steps are declared and excluded globally. F/R/B kernel events,
component wall time, planning wall time and exposed host time remain separate.
Do not treat this screening as fresh-process ABBA or whole-model non-regression.

## Bounded candidate reachability audit

```bash
python scripts/audit_replica_candidates.py --fixtures /path/to/actual-push-fixtures \
  --output /path/to/separate-candidate-audit.json
```

The script requires exact original-plan reproduction, equal source histograms,
equal effective legacy cost models and matching policy fields other than the
recorded theoretical capacity bound. It pads a captured B1 layout into the B6
physical configuration without redistributing any source quota, then checks its
original hard/receive constraints and scores it under the same B6 model.

An isolated serial trace records thin candidates and actual scoring reachability.
A separately labeled CPU probe relaxes only the intermediate search bound while
retaining the receive allowance and rechecking the selected result against the
original bound. This distinguishes an excluded feasible completion from an
unfavorable score. It does not apply a planner fix, change coefficients, execute
the alternative layout or establish device performance.
