# MegaMoe Qwen optimizer benchmark

This directory contains a focused end-to-end training benchmark:

- `qwen_moe_model.py` defines a random-weight Qwen MoE model with learned TopK
  routing, routed experts, shared experts, attention, and a language-model loss.
- `qwen_moe_benchmark.py` builds identically initialized common-MoE and
  MegaMoe variants, checks their full-model accuracy, and runs `zero_grad`,
  forward, backward, dense-gradient synchronization, and an AdamW step.
- `qwen_moe_config.py` provides backend-free shape, routing, and storage options.
- `run_qwen_moe_benchmark.sh` activates CANN and the native payload, verifies
  the editable install, and launches the eight-rank job.

It is a runnable integration example, not a profiler. It records actual shared-B
pool bytes after timing, but not allocator peaks, traces, or artifact inventories.

## MegaMoe integration

The Qwen Router computes `topk_ids`, `topk_weights`, and
`tokens_per_expert`. The exact Router histogram is passed to
`MegaMoeExperts`, so MegaMoe does not repeat the `bincount` operation.

Each decoder layer keeps independent expert parameters and optimizer state.
All serial MegaMoe layers call `share_execution_resources()` before first
forward and therefore reuse one SHMEM runtime/workspace. The benchmark calls
`QwenMoeModel.close()` before destroying the process group.

`--shared-physical-weights --replica-slots-per-rank B` additionally shares one
BF16 replica-weight pool and one FP32 replica-gradient pool across layers. It
requires `B > 0` and excludes `--compact-physical-weights`. Home parameters and
optimizer states remain independent; this is not cross-layer weight tying.
The option is explicit and off by default. Balanced routing may never activate
replicas, in which case the lazily allocated B pool correctly remains empty.

The public `MegaMoeExperts` API and this benchmark default
`expert_capacity_factor` to `None`, which reserves the maximum lossless receive
capacity. An explicit factor such as `1.5` bounds the per-wave receive workspace;
the current implementation uses lossless multi-wave execution for residual
overload that replica planning does not remove.

## Workload

| Item | Value |
| --- | --- |
| Topology | TP1, EP8 |
| Dtype | BF16 |
| Batch / sequence per rank | 1 / 4096 |
| Hidden size / layers | 5120 / 8 |
| Routed experts / TopK | 48 / 8 |
| Expert intermediate size | 1792 |
| Vocabulary | 32000 |
| Optimizer | HyperParallel AdamW, FP32 master / BF16 model |
| Comparison order | common A, then MegaMoe B |
| Default timing | 3 warmup + 5 measured optimizer steps |

Memory caveat: the current comparison keeps **both** backends and both AdamW
optimizers resident. At the default eight-layer shape, the model/master/moment/
main-gradient tensors alone require about 73.2 GiB per rank, before activations
or workspaces. This is not a single-model HBM measurement. Use the smaller
correctness command below on 64 GiB devices; isolated-backend performance runs
are the next benchmark improvement. Original-width two- and four-layer
correctness checks have passed; the default eight-layer pair has not been run.

## Single-node run

Build the Torch multicore payload, install this checkout in editable mode, and
install the example's optional Transformers dependency (validated with
`transformers==5.5.3`). Qwen does not require DeepSeek-only custom operators.
Run:

```bash
CANN_ENV_FILE=/path/to/CANN/set_env.sh \
bash examples/torch/multicore/mega_moe/run_qwen_moe_benchmark.sh \
  --output output/qwen_moe_benchmark.json
```

For a smaller connected two-layer shared-B correctness run on eight NPUs:

```bash
bash examples/torch/multicore/mega_moe/run_qwen_moe_benchmark.sh \
  --shared-physical-weights --replica-slots-per-rank 1 \
  --expert-capacity-factor 1.5 --routing dirichlet --router-load-alpha 0.1 \
  --num-layers 2 --hidden-size 1024 --intermediate-size 256 \
  --vocab-size 2048 --sequence-length 1024 \
  --warmup-steps 1 --measured-steps 2 --output output/qwen_shared_smoke.json
```

Use `--routing balanced` or `--routing topk` for balanced or learned-router
coverage. The short run is a correctness check, not a performance comparison
between optimization revisions.

Useful environment overrides are `NATIVE_ENV_FILE`, `SHMEM_LIB_DIR`,
`ASCEND_RT_VISIBLE_DEVICES`, `MASTER_PORT`, `SHMEM_HOST`, `SHMEM_PORT`, `PYTHON_BIN`,
`TORCHRUN_BIN`, and `TIMEOUT_SECONDS`.

## Multi-node run

Launch the same command on every node with a shared `MASTER_ADDR` and distinct
`NODE_RANK`. For two four-device nodes, set:

```bash
NNODES=2 NPROC_PER_NODE=4 NODE_RANK=<0-or-1> \
MASTER_ADDR=<rank-zero-host> SHMEM_HOST=<rank-zero-ip> \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
bash examples/torch/multicore/mega_moe/run_qwen_moe_benchmark.sh
```

`NNODES * NPROC_PER_NODE` must equal EP8. This minimal benchmark uses one
SHMEM communicator over the full Torch world. The launcher sets
`SHMEM_IP_PORT` from `SHMEM_HOST` (defaulting to `MASTER_ADDR`) and
`SHMEM_PORT`; subgroup endpoint discovery is intentionally outside this PR.

## Output

Rank zero writes one JSON file containing the complete configuration and seed,
topology and shape, per-rank shared-pool bytes,
optimizer settings, common/MegaMoe accuracy checks, per-backend first-step
validation, steady-state latency samples and medians, their B/A ratio, measured
losses, and Torch runtime versions. Accuracy starts from exactly equal local
parameters and compares full-model loss/logits, all local gradients, and all
parameters after one AdamW step. The model parameters remain BF16 while
`Float16OptimizerWithFloat16Params` updates FP32 master parameters through
HyperParallel AdamW and copies them back to the model. All comparisons use
`rtol=2e-2` and `atol=2e-3`.
The wrapper's historical `Float16` class name denotes the low-precision model
path; this benchmark explicitly uses BF16 model parameters and FP32 master
parameters.
Gradient snapshots are collected only for the compared first step, so they do
not affect steady-state timing. Timing is plain A/B, not A/B/B/A, and uses the
rank-maximum complete optimizer-step latency after independent warmup. This
random-weight benchmark validates integration; it does not establish checkpoint
convergence.
