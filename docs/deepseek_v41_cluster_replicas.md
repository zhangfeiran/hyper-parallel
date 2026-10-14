# DeepSeek V4.1 cluster smoke with expert replicas

`bash scripts/run_deepseek_v41_cluster.sh` runs native, push and pull in separate,
sequential distributed processes. Each case uses the same random initialization,
logical experts, topology and replica budget. A failed case stops the sequence.

The defaults are EP64, 16 dies per node, TP=CP=PP=1, MBS1, four decoder layers,
one vision layer, E384, original widths and the original Engram table retained
by that decoder prefix. Activation checkpointing is `full`. Node count and rank
come from the PanGu environment; dense FSDP spans the world and expert FSDP
spans WORLD_SIZE/EP_SIZE. The 128-die configuration therefore uses eight nodes,
dense FSDP128 and expert FSDP2. MTP/DSpark heads are outside this adapter.

The new runner uses the current branch's Omni data path. It retains the frozen
image conversations and the released image ceiling of 1024. Native and MegaMoe
execute each packed sample's actual token count (at most 4096). MegaMoe reserves
communication capacity for 4096 tokens without adding synthetic routes.
This is not a timing continuation of the older
September data/runtime bundle.

## Replica budget

`REPLICA_SLOTS_PER_RANK` is B, the maximum extra guest experts per EP rank.
The cluster launcher defaults to E/EP, which is six for E384/EP64. It configures
native and both MegaMoe transports explicitly; the general library default
remains B=0. All cases use CPU planning, HCCL P2P weight/gradient transfer,
`replica_min_rows=0` and `replica_minimum_load_reduction=0.0`. Balanced routes
can naturally need no active guests. B limits available slots, not mandatory copies.

With R=64, H=E/R=6, S=4096, K=6, the average routed row count is A=S*K=24576.
The source planner's conservative aligned receive bounds are:

| B | Maximum receive rows | Bound / A | Guest BF16 weights + FP32 gradients per die |
|---|---:|---:|---:|
| 0 | 1572864 | 64.000 | 0 MiB |
| 1 | 1314944 | 53.505 | 202.5 MiB |
| 2 | 1056896 | 43.005 | 405 MiB |
| 4 | 540800 | 22.005 | 810 MiB |
| 5 | 282752 | 11.505 | 1012.5 MiB |
| 6 | 24576 | 1.000 | 1215 MiB |

For 0<B<H the bound is align128(min(U, ceil(((H-B)*U+B*A)/H)+R-1)),
where U=R*S*min(K,H). B>=H permits exact balance. B>6 cannot improve this
bound for the stated topology, while it reserves more storage.

Each expert has 3*5120*2304 parameters: 67.5 MiB BF16 weights plus 135 MiB
FP32 gradients. The guest pool is shared across serial layers and creates no
optimizer parameters or states. These are payload sizes, not total HBM peaks.
Home FP32 gradient partials, owner receive scratch, native weight-layout copies,
activation tensors, FSDP/optimizer buffers and allocator reserve are additional.

For the particular synthetic route where all tokens select the same six home
experts, CPU planning reaches max loads of 49152 for B=1 and 24576 for B>=2.
That example does not prove B=2 can bound every possible learned route.
B=6 is the initial memory-conservative setting; evaluate smaller B against
actual route and full-step measurements after the distributed smoke succeeds.

Push's general initial factor remains 1.25, but B=6 caps its effective capacity
at 24576 rows (factor 1.0). Push's two token buffers therefore need 480 MiB BF16
plus events/alignment; pull's two source-sized buffers have the same payload.
Both also allocate rank-local compute intermediates. CPU planner is required
here because EP64/E384 exceeds the current optional device planner's capacity.

Native replicas preserve DSV4.1's FP32 clamp/SwiGLU and inclusive clamp boundary
derivatives. Expert calls remain inside their FSDP module boundaries, and guest
partials return to the original owners. MegaMoe accepts ordered EP subgroups;
it no longer requires EP to span the whole training world.

## Bundle and launch

A bundle contains `source`, matching rebuilt Ascend 910C native payload under
`source/build/native/payload`, pinned runtime wheels, model/tokenizer metadata,
Engram metadata, frozen image data, CANN installation helpers and `SHA256SUMS`.
Source and native payload must be updated together for the replica runtime ABI.
The processor requires `assets/model/inference/config.json`; when migrating an
older bundle, its seven vision/image fields can be derived directly from the
released nested `config.json` and must be recorded in bundle provenance.

Run on every node of an eight-node / 128-die job:

```bash
RUN_ID=dsv41-depth4-ep64-128-b6-dynamic-005 \
NUM_CARDS=16 EP_SIZE=64 EDP_SHARD_SIZE=2 \
REPLICA_SLOTS_PER_RANK=6 \
NUM_HIDDEN_LAYERS=4 VISION_NUM_HIDDEN_LAYERS=1 \
ACTIVATION_CHECKPOINT=full TRAIN_ITERS=40 WARMUP_STEPS=10 \
ROUTE_DIAGNOSTICS=1 HBM_INTERVAL=1 \
bash /work/z00961864/hyper-dsv4.1-depth4-dynamic-20261014/source/scripts/run_deepseek_v41_cluster.sh
```

Use a fresh RUN_ID for each attempt. `DRY_RUN=1` validates topology and prints
all three commands without device work. `REPLICA_SLOTS_PER_RANK=0` selects the
ordinary no-copy paths for a separately labeled comparison (also set
`ROUTE_DIAGNOSTICS=0`); the prior extreme
hotspot OOM remains possible with that setting.

Weights are generated directly into local parameter shards with canonical
name/global-index SplitMix64 initialization and seed 42. No weight checkpoints
are written. Checkpoint saving and restoration are disabled for this smoke.

The default protocol has ten warmup steps followed by thirty measured steps.
Both phases update the optimizer with the same forty-step LR horizon in every
case. Per-rank records contain actual MoE input rows, phase and UTC timing
boundaries. With `ROUTE_DIAGNOSTICS=1`, the harness retains existing immutable
CPU planner results and summarizes them after timing: logical expert rows,
home and balanced destination loads, active guest slots, guest compute rows
and weight-transfer count. Planner calls during checkpoint recomputation are
attributed to their actual model layer; no extra device count copy is added.

Every node samples physical HBM and classifies device PIDs against its own
launcher. The sampler records gaps, failed queries and foreign/unresolved PIDs.
Its exit status is separate from the training exit status. After all case
cleanup sentinels exist, node zero writes `RUN_ROOT/summary.json`, comparing
all-rank initialization and actual token rows, reducing full-step timings by
rank maximum and reporting whole-run versus measured-window physical peaks.
HBM or timing comparisons with incomplete monitoring or foreign PIDs are
marked ineligible. SHMEM heap, Torch allocator and physical HBM remain separate.
The sampler covers initialization as well; allocator peaks start after model
initialization and include separate warmup and measured-window statistics.

CPU tests, meta construction, recipe resolution and native compilation do not
establish distributed numerical correctness, full-step performance or HBM fit.
A successful cluster check requires all three case statuses, all-rank step and
result files, finite loss/gradients, and allocator plus external SHMEM/device
memory observations. Replica capacity addresses received-token skew; it does
not guarantee that every other model allocation will fit.
