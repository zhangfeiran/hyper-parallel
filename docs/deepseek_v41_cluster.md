# DeepSeek-V4.1 full-backbone cluster comparison

Run the same command on **every node** of one PanGu 910C job:

```bash
RUN_ID=dsv41-full-001 bash /work/z00961864/hyper-dsv4.1-full-20260921/source/scripts/run_deepseek_v41_cluster.sh
```

Use a fresh, identical `RUN_ID` on all nodes for every invocation. The launcher
reads `MA_NUM_HOSTS` (or counts `VC_WORKER_HOSTS`), `RANK_AFTER_ACC` /
`VC_TASK_INDEX`, and the first worker host. `NNODES`, `NODE_RANK`, `MASTER_ADDR`,
`NUM_CARDS` and `MASTER_PORT_BASE` can override scheduler values. A missing
multi-node topology fails before installation. `DRY_RUN=1` prints the launch
commands without installing anything or accessing NPUs.

The synchronized development path is
`/home/ma-user/work/z00961864/hyper-dsv4.1-full-20260921`; the same shared mount
is `/work/z00961864/hyper-dsv4.1-full-20260921` on the cluster.

## Scope and defaults

- Released backbone: 40 decoder layers, 32 vision blocks, E384, TopK6, H5120,
  I2304, full Engram at layers 1 and 14 (384006168 / 384016682 logical rows),
  image-token ceiling 1024, original candidate/indexer hierarchy.
- The existing adapter does not implement MTP/DSpark prediction heads. This is
  full backbone scale, not validation of every released pretraining objective.
- TP1 / CP1 / PP1, `EP_SIZE=16`, dense FSDP = total ranks, expert FSDP = ranks / EP.
  EP16 keeps MoE groups within a 16-die node; FSDP spans nodes. Override `EP_SIZE`
  explicitly to study cross-node MoE; it must divide both total ranks and E384.
- MBS1, global batch = total ranks, 4096 padded input tokens, BF16 compute,
  FP32 model initialization/reduction/main parameters, Muon + AdamW.
- Native grouped GEMM, MegaMoE push and MegaMoE pull run sequentially in fresh
  torchrun processes. Each uses `TRAIN_ITERS=100`, checkpointing off, swap off,
  compile off and no profiler. Push uses the lossless capacity bound; pull has
  no push capacity factor. Forty MegaMoE layers share execution resources.
- The same frozen 512 JSONL records and JPEGs as the cropped report are bundled.
  The model/tokenizer and full Engram assets are bundled; no pretrained weights.
  With MBS1 and this fixed dataset, total ranks must not exceed 512.

## Random initialization without weight files

`RANDOM_INIT_SEED=42` selects a stateless generator keyed by parameter name and
canonical global element index. Local ranks generate only their own parameter
shards, using bounded CPU chunks; MegaMoE expert indices are transposed back to
the native layout. Replicated parameters receive identical values. Random
parameters use N(0, 0.02); norm/Engram gate/mHC scale values remain one, and
bias/mHC base/attention sink values remain zero. Embedding padding stays zero.
Optimizer main parameters are reloaded afterwards. No canonical weight files
or preparation run are created. This intentionally defines a new random
initialization and does not reproduce the cropped run's saved weights.

Each rank writes `initialization.json` with generator version, seed, local
parameter count and an order-independent FP32 bit-sum checksum. For identical
topology, compare these records across all three backends before accepting
training comparisons. The checksum is an audit aid, not a precision criterion.

## Runtime and artifacts

The launcher verifies `SHA256SUMS`, copies the source to node-local storage,
installs bundled wheels in a fresh Python 3.11 venv and installs that source
editable. It uses the existing PanGu CANN installer and shared CANN/Omni Ops
packages to install CANN 9.1.0 node-locally. `CANN_ENV_FILE` and `MHC_ENV_FILE`
can select an existing matching installation. It validates runtime versions,
Hyper import origin, CANN version and native registration before torchrun.

`NODE_LOCAL_ROOT` defaults to `${MODELARTS_JOB_DIR:-/tmp}/dsv41-RUN_ID`.
`PYTHON_BIN` can select the Python 3.11 interpreter used to create the venv;
otherwise the launcher checks the PanGu PyTorch-2.9.0 / PyTorch-2.6.0 environment
paths. Torch and torch_npu are both pinned to 2.9.1, torchvision to 0.24.1 and
Transformers to 5.13.1. The native payload is the fingerprint-verified direct-pull
snapshot from the supplied report, not the unrelated PanGu Hyper wheel.

Shared results default to `/work/z00961864/dsv41-runs/RUN_ID/`:

- `nodes/node_N/launcher.log`, per-case `command.txt`, `run.log`, `exit_status.txt`,
  `npu_before.txt`, `npu_after.txt`.
- `native|push|pull/rankN/recipe.yaml`, `engram.json`, `initialization.json`,
  `steps.jsonl`, `result.json`.

Step files retain Trainer timing and synchronized whole-step maximum-rank
seconds separately. Allocated/reserved peaks are framework counters, not
sampled physical HBM. This launcher does not supply profiler or HBM sampling
acceptance and a single native/push/pull sequence is not an ABBA speedup result.

## Validation status

Full meta construction: 748,979,956,848 backbone parameters, no device allocation.
Local CPU initialization checks cover split/chunk/transpose consistency,
constants, seed separation and optimizer reload. Existing crop tests pass.
The delivered bundle is checked through offline installation, runtime/native
imports, full-model recipe dry-runs and scheduler-variable launch dry-runs.
No full-model NPU execution, memory-fit result, numerical acceptance or
performance conclusion is claimed. Full scale needs sufficient HBM; the
launcher does not shrink the model or change recomputation automatically.

## Suggested EP64 starting topology

For 16 dies per node and 64 GiB HBM per die, start with 24 nodes (384 ranks):
TP1 / CP1 / PP1 / EP64, dense FSDP384, expert FSDP6, MBS1 / GBS384 / 4K.
Each EP group spans four nodes. Each rank computes six routed experts; expert
FSDP6 evenly divides that expert state. Use the same topology and recomputation
mode for all three backends.

```bash
RUN_ID=dsv41-ep64-24n-001 EP_SIZE=64 ACTIVATION_CHECKPOINT=full \
  bash /work/z00961864/hyper-dsv4.1-full-20260921/source/scripts/run_deepseek_v41_cluster.sh
```

The scheduler must allocate 24 nodes; the command still reads the actual node
count automatically. `ACTIVATION_CHECKPOINT` accepts `off` (the original default)
or `full`. For this KV-shared architecture, Hyper's full mode uses submodule
checkpointing to preserve shared-attention state; it is not whole-layer attention
recomputation. The full EP64 training combination remains unverified on NPUs.

Static state accounting from the meta model gives 551,040,925,696 Muon parameters
and 197,939,031,152 AdamW parameters. Assuming FP32 parameter/gradient storage,
one FP32 Muon momentum or two FP32 AdamW moments, the ideal evenly sharded state
is about 47.4 GiB/die at 12 nodes, or 23.7 GiB/die at 24 nodes. These estimates
exclude activations, unsharded parameters, prefetch, optimizer temporaries,
communication, allocator overhead and replicated/uneven-shard differences.
EP64 lossless push additionally reserves about 15.23 GiB/die for its two SHMEM
data tensors, before events and heap alignment. At balanced routing, 40 layers'
saved MoE dispatch/up/activation tensors alone total about 22.03 GiB/die without
recomputation. More FSDP ranks do not reduce this per-rank activation term.

24 nodes with recomputation is an initial resource budget, not a demonstrated
memory-fit or throughput result. Start with a short three-backend smoke using
`TRAIN_ITERS=3`, then use a fresh RUN_ID with the default 100 steps. The short run
has a different LR horizon and must not be included in the 100-step comparison.
