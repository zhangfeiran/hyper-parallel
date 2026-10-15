# DeepSeek-V4.1 Host Engram and cluster experiments

This extension sits on top of the core DSV4.1 MegaMoe adapter and bounded dynamic
token support and the merged expert-hot-replica implementation. Native, push and pull all use
the same model, initialization, optimizer policy and data order.

## Host Engram

The recipe selects `HostEngramModule` through the existing `replace_module`
contract. The replacement preserves table FQNs, logical hash addresses and EP
ownership. Each owner holds its FP32 table on CPU; only requested rows move to
the accelerator, with BF16 lookup outputs for the BF16 projection. Dense FSDP
and the dense optimizer exclude these parameters.

Backward accumulates sparse CPU row gradients. Ranks sharing an EP owner
synchronize those gradients before a joint dense/sparse norm and clipping step.
CPU SparseAdam owns Host updates. Its learning rate, betas and epsilon come from
the benchmark's AdamW configuration. SparseAdam does not apply dense AdamW's
updates or weight decay to untouched rows: compare native/push/pull within the
same storage policy rather than assuming Host and device tables have identical
multi-step optimizer semantics.

Owner rows use canonical global element indices and the same deterministic
SplitMix64/Box-Muller initialization as dense benchmark parameters. The seed is
independent of the EP/FSDP placement and expert matrix transposition. Generation
uses bounded CPU chunks and creates no weight snapshots. Optimizer FP32 masters
are refreshed after benchmark initialization.

The generic external-state lifecycle delegates model-specific validation and
loading to the adapter. Host sidecars store owner weights and optional sparse
optimizer moments; DCP stores dense state. Restore validates topology, key sets,
file checksums and optimizer scope before consuming state. The cluster benchmark
disables checkpoint saving; `prepare_engram_host_rows.py` is an optional
pretrained-row conversion tool and is not used by random initialization.

Host training requires meta construction, FP32 model initialization and optimizer
masters, BF16 FSDP parameters, FP32 gradient reduction and disabled automatic
forward-input casting. Placement validation, compilation and asynchronous Host
checkpointing are unsupported and fail explicitly.

## Decoder and attention recomputation

`ACTIVATION_CHECKPOINT=full` selects model-adapter safe regions and sets their
count from the requested decoder depth. Each decoder layer is a region. The
benchmark explicitly selects `ATTENTION_RECOMPUTE_TIER=all`, including Full,
Reindex and Reuse attention. This leaves the library's normal default tier
unchanged. Existing shared-state replay pins discrete attention choices and
avoids publishing cross-layer state twice. The visual tower is outside decoder
recomputation. Cluster VLM recipes set `root_owned_vision=true`: the root FSDP
unit owns visual parameters so differing per-rank image counts cannot change
the number of parameter collectives inside the image loop. Normal recipes keep
their existing visual FSDP units. Native vision SDPA retains a batch dimension
to permit the fused attention path. `ACTIVATION_CHECKPOINT=off` provides the non-recompute comparison.

## Cluster launch and workload

`scripts/run_deepseek_v41_cluster.sh` reads the node count and node rank from
PanGu scheduler variables. Each node must execute the same command with one
fresh shared `RUN_ID`. Defaults for an eight-node, 16-die-per-node allocation are:

| Setting | Default |
| --- | --- |
| World / EP | 128 / 64 |
| Dense / expert FSDP | 128 / 2 |
| TP / CP / PP | 1 / 1 / 1 |
| Micro / global batch | 1 / 128 |
| Decoder / vision depth | 4 / 1 |
| Width / expert intermediate / experts | released values / released values / 384 |
| Engram | original retained-layer tables, CPU owners |
| Expert replicas | B0 by default; `REPLICA_SLOTS_PER_RANK=1` enables B1 |
| Steps / warmup | 40 / 10 |
| Maximum sequence / actual token bound | 4096 / 4096 |
| Cache clearing | every step; `EMPTY_CACHE_STEPS=0` disables it |

The default is a depth crop, not full-model training. `--full-model` in the
worker command restores released parameter dimensions and validates original
Engram metadata; explicit depth arguments still crop the model. Full depths
require `NUM_HIDDEN_LAYERS=40 VISION_NUM_HIDDEN_LAYERS=32` and separate memory
acceptance. DSpark is supported by the underlying model but disabled in these
backbone benchmark recipes.

The launcher runs six independent cases in order: dynamic native/push/pull,
then packed native/push/pull. VLM dynamic mode selects one complete sample with
`token_budget=1` and `min_buffered_samples=1`; it does not truncate the sample to
one token. Packed mode uses budget 4096 and a 16-sample first-fit buffer. Actual
rows can differ by rank and need not equal 4096. The adapter's `local_num_tokens`
argument is an upper bound passed to generic MegaMoe's `max_local_num_tokens`;
no artificial rows are added. Text mode uses the existing online text loader;
its prepared source records determine sequence length, so the two packing
labels do not represent different text packing implementations.

Use `MODALITY=text` for `TextTrainer` and `assets/data/train_text.jsonl`, or
`MODALITY=vlm` for `VLMTrainer` and `assets/data/train.jsonl`. `MODEL_DIR`, `DATA`,
`ENGRAM` and `NATIVE_LIB` can override bundle inputs. A prepared bundle must
contain the matching source and native payload, source-only model/tokenizer
assets, Engram metadata, finite-horizon data, offline runtime wheels, a checksum
manifest and its CANN installation helper. Do not reuse a native payload from an
older MegaMoe implementation. Existing work bundles are not updated by creating
this worktree.

From a prepared bundle, run this on every allocated node:

```bash
RUN_ID=unique-job-id MODALITY=vlm EP_SIZE=64 \
  bash source/scripts/run_deepseek_v41_cluster.sh
```

`DRY_RUN=1` prints commands and checks bundle inputs without installing packages
or using devices. Node-local source and virtual environments keep runtime files
off shared work storage. The launcher verifies package versions, import origin,
native registration and checksums before starting workers.

## Timing, profiling and failure evidence

`MOE_TIMING=1` records native, push and pull on the same MoE boundary: router,
routed experts and shared experts. Current-stream events observe initial
forward, replay forward and output-gradient to input-gradient spans. They do
not retain activations or synchronize each layer. Backward can include replay
and omit work remaining on other streams or weight-gradient tails; phases are
not additive. The summary sums layers within each rank before taking the step's
rank maximum. Full-network ratios use synchronized training-step wall time.

`scripts/run_dsv41_push_profile_cluster.sh` reuses the common launcher with
`PROFILE=1 MODALITY=text`. It runs packed native then packed push, profiling
local rank zero on each node during steps `[3, 5)`. Override modality or
`PROFILE_START_STEP` / `PROFILE_END_STEP` as needed. The default scheduler
horizon remains the requested training horizon. Profiler summaries always mark
the timings diagnostic and ineligible for speedup acceptance.

Each case owns its worker process group. Peer failure triggers cleanup of that
case's workers; all nodes must publish cleanup status before advancing. The
coordinator never stops unrelated jobs. A cleanup timeout stops progression.
Partial summaries retain completed comparisons while the final job status
remains nonzero if a required case fails.

Rank results include actual MoE rows, initialization fingerprints, step timing,
allocator peaks and SHMEM state. Per-node monitoring separately records physical
HBM and process ownership. Missing ranks, inconsistent workload, foreign or
unresolved PIDs, unhealthy devices or insufficient HBM samples invalidate the
relevant comparison. Finite loss/gradnorm and reported differences do not
constitute tensor-level numerical acceptance. Reduced CPU and four-rank HCCL
fixtures do not establish full-model fit or 128-die performance.

## Optional expert replicas

The DSV4.1 native and MegaMoe adapters accept `replica_slots_per_rank` and
`replica_minimum_load_reduction`. B0 keeps ordinary expert ownership; B1 permits
one guest expert slot per EP rank. Current DSV4.1 recipes use the CPU planner,
P2P weight/gradient transport and FP32 clamped SwiGLU semantics. The explicit
admission threshold defaults to zero, preserving the earlier balance-oriented
policy; this differs from the generic component's conservative default.

`--route-diagnostics` records host placement plans after each step, including
source tokens, home and balanced destination loads, guest slots and transfers.
A B1 budget does not imply every layer or step activates a guest. Full-layer
recomputation may call the planner again, and diagnostics retain its call count.
Dynamic token schedules retain bounded plan and runtime-image caches; evicted
native images remain protected by stream recording.
