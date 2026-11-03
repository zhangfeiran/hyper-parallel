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
the fitter starts. A peer failure stops the other local worker groups. Failed
gates retain observations and logs, exit nonzero and publish no catalog.

## Observation and fit semantics

F/R events surround the whole public MegaMoE invocation, including prefetch,
route preparation and output restoration. An outer backward event interval
contains recompute and backward; subtracting the measured replay interval gives
the remaining backward service. Maxima are taken separately for F, R and B. The
events use rank-local stream clocks; no cross-device timestamp alignment is
assumed. Component wall samples include the optimizer and are retained separately.

The first model is an empirical rank-local full-step service predictor. Convex
nonnegative least squares fits all rank observations with an identifiable six
term basis: active expert tasks, expert rows, local/remote matrix edges, token
rows and active-path overhead. A training-only four term basis is considered if
the richer basis is unidentifiable. A structurally impossible link class, such
as inter-node copies for an entirely single-node EP group, is excluded. Every
retained design must have full rank and bounded condition number.

The runtime legacy scorer encodes this effective full-step predictor with shared
F/B terms. This allocation does not identify physical GEMM, gradient-band ready
fractions, ACK services or overlap. The artifact explicitly makes no physical
service or verified-overlap claim. Raw separate F/R/B measurements remain available
for future richer models. No old DSV41 coefficient is scaled to the new model.

Frame indices congruent to2 modulo3 are untouched holdouts. Publication requires
held-out relative error, worst underprediction, and harmful false-admission gates,
followed by another actual runtime capture loading only the candidate tables.
Every non-balanced lookup must resolve the exact candidate; balanced owner-only
bypass is recorded and allowed. Holdouts never refit coefficients. Failed fits
can be rerun offline after improving the scripts using the retained capture.

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
