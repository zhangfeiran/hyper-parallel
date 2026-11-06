# Script-only calibration for another model or allocation

The initial automated contract supports BF16 MegaMoE Push/Pull, EIO expert
weights, FP32 expert partials, actual checkpoint replay, and normal-prefetch
residency disabled. Valid saved recompute-to-backward generations still reuse
their payload. Native is an optional end-to-end control, not a calibrated backend
in this pipeline.

The model-owned JSON manifest contains `hidden_size`, `intermediate_size`,
`num_experts`, `top_k`, `ep_size`, `local_num_tokens`, `checkpointed: true`,
`dispatch_modes`, `replica_caps`, `checkpoint_contract`, and `swiglu_limit`. The
checkpoint contract is `reentrant` for a no-grad original forward and grad-enabled
replay, as used by PanGu; `nonreentrant` is the default for Hyper checkpointing.
The original call-boundary grad mode is remembered in the shared invocation and
bound into the same execution signature for both phases. It may also contain the model
family as descriptive metadata; selection never relies on its name. For PanGu
504B the relevant dimensions are H5120/I1792/E384/K8. The initial comparison uses
EP64 and maximum source tokens4096.

After the normal cluster launcher activates its isolated Python, CANN and native
payload, run this same command on every node:

```bash
bash scripts/run_replica_autocalibration_cluster.sh \
  --manifest /shared/run/model-shape.json --output /shared/run/calibration
```

Scheduler variables supply NNODES, NODE_RANK and MASTER_ADDR; no job ID is
required. `DRY_RUN=1` prints all four phases without importing Torch. The worker
reuses the existing NUMA launcher before framework imports. No CANN profiler is
collected. The default matrix uses Push/Pull x B0/B1/B2/B6, twelve bounded route
frames, one warmup and three recorded samples per frame.

The script owns collection, fitting, private model-in-loop verification and
publication. All nodes and all worker ranks must complete each capture before
the fitter starts. A peer failure stops the other local worker groups. Quality
rejection after complete collection exits20, reports `collection_complete: true`
and `calibration_status: rejected`, and publishes no catalog. Runtime or input
failures exit1. The outer launcher can continue explicitly with heuristic controls.

## Observation and fit semantics

F/R events surround the whole public MegaMoE invocation, including prefetch,
route preparation and output restoration. An outer backward event interval
contains recompute and backward; subtracting the measured replay interval gives
the remaining backward service. The versioned measurement contract is
`rank_cumulative_f_r_b_v2`: add F/R/B on each rank, then take the rank maximum.
The old sum of three separate phase maxima is permanently retained as a diagnostic;
different ranks can own these maxima, so their sum is not elapsed component time.
A whole-component event stops after backward and before optimizer work. Every
rank's cumulative F/R/B must fit inside that interval within timing tolerance.
Host wall samples including the optimizer remain separate diagnostics. Events
use rank-local stream clocks; no cross-device timestamp alignment is assumed.

The model uses group component maxima for active expert tasks, expert rows,
local/remote matrix edges, token rows and active-path overhead. Convex nonnegative
least squares fits the cumulative-rank target. Basis selection uses TRAIN-only
cross-validation, grouping the same route kind and source-size regime across
all repeats, EP groups and B0/positive-cap observations. An unidentifiable basis
is rejected in every fold. Structurally impossible link classes are excluded.
The retained basis has full rank and bounded condition number. Cold observations
are retained; the gain margin includes worst underprediction from grouped CV.

The internal `effective_group_ms` field selects the same scorer for runtime
admission and quota trials, including owner-only candidates. Its six nonnegative
coefficients are empirical; they do not identify physical GEMM, ready/ACK services
or overlap. New group tables use Python scoring because the legacy native scorer
does not implement this aggregation; unrelated host primitives may remain native.
Legacy row curves retain coverage and conservative lower-bound roles. Tables
without this field preserve legacy behavior. No DSV41 coefficient is rescaled.

Frame indices congruent to2 modulo3 are untouched holdouts. Publication requires
held-out relative error, worst underprediction, and harmful false-admission gates,
followed by another actual runtime capture loading only the candidate tables.
Every non-balanced lookup must resolve the exact candidate; balanced owner-only
bypass is recorded and allowed. Holdouts never refit coefficients. Failed fits
can be diagnosed offline using the retained capture. Old captures are never
relabelled as v2 acceptance data. A fresh v2 collection and a separate fresh
model-in-loop verification are required before publishing the new tables.

## Artifacts and matching

`collect/` and `verify/` retain group observations and every rank's completion.
`nodes/node_N/` retains stage logs and terminal status. Candidate model JSON files
are in `models/`; only successful private verification atomically creates
`models/catalog.json` with `status: validated`. Published directories are immutable.

Subsequent training in the same allocation selects the catalog internally:

```bash
export HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY=0
export HYPER_PARALLEL_REPLICA_METADATA_REUSE=0
export HYPER_PARALLEL_REPLICA_CALIBRATION_DIR=/shared/run/calibration/models
```

The business module still only needs the replica cap; cost selection and bounded
adaptation remain enabled by default. The external directory is a hidden launch
setting. Missing shape/signature coverage retains the normal heuristic. An
explicitly configured corrupt, unvalidated or source-mismatched catalog fails
closed rather than silently installing its coefficients.

Profiles bind each cap separately to live H/I/E/K/EP, global world size, maximum
agreed source-token count, Cube count, layout/clamp/splits, checkpoint context,
normalized topology, schedule and native adapter/vendor contents. The catalog also
binds execution Python sources, optional host extension bytes, Torch/NPU/CANN
versions, residency/metadata flags and scheduler master/worker allocation scope.
Maximum source tokens are observed by group agreement; ragged frames retain the
declared maximum rather than fabricating a token signature. Prefixes are excluded
from source and native content identity. Allocation-scoped catalogs should be
recollected after changing the cluster allocation or fabric.

Prediction gates validate a component model, not an end-to-end training speedup.
Concurrent FSDP/TP traffic and model scheduling can change exposed services. The
cluster wrapper should optionally continue into profiler-off native/B0/cost-model
PanGu controls and report rank-max step medians before accepting whole-model gains.
