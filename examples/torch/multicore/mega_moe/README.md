# MegaMoe Qwen optimizer benchmark

This directory contains a focused end-to-end training benchmark:

- `qwen_moe_model.py` defines a random-weight Qwen MoE model with learned TopK
  routing, routed experts, shared experts, attention, and a language-model loss.
- `qwen_moe_benchmark.py` builds identically initialized common-MoE and
  MegaMoe variants, checks their full-model accuracy, and runs `zero_grad`,
  forward, backward, dense-gradient synchronization, and an AdamW step.
- `qwen_moe_config.py` provides backend-free shape, routing, and storage options.
- `qwen_moe_isolated.py` launches fresh common/MegaMoe processes in A/B/B/A order.
- `qwen_moe_snapshots.py` stores CPU references and compares tensors in bounded
  FP32 chunks, without keeping a second model on the device.
- `run_qwen_moe_benchmark.sh` activates CANN and the native payload, verifies
  the editable install, and defaults to the isolated comparison.

It is a runnable integration example, not a profiler. Isolated runs record
per-rank steady-state allocator peaks, external SHMEM heap bytes, actual shared-B
pool bytes, training-state bytes, and source/native fingerprints outside timing.
They do not collect traces.

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
| Comparison order | fresh-process common A1, MegaMoe B1/B2, common A2 |
| Default timing | 3 warmup + 5 measured optimizer steps per process |

The default comparison keeps only one backend and its optimizer resident per
process. The old paired path is available with `BENCHMARK_MODE=paired`; at eight
layers, its two sets of model/master/moment/main-gradient tensors alone need
about 73.2 GiB per rank, before activations or workspaces. Do not use that path
as a single-model HBM measurement.

CPU references are retained under a fresh artifact directory. At the original
eight-layer width they need approximately 100 GiB across all eight ranks. Use
`--artifacts-dir` to select storage with sufficient space; snapshot I/O and CPU
comparison are outside steady-state timing. These are precision-test artifacts,
not restartable training checkpoints or optimizer-state comparisons.

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
  --order ab --warmup-steps 1 --measured-steps 2 --output output/qwen_shared_smoke.json
```

Use `--routing balanced` or `--routing topk` for balanced or learned-router
coverage. The short run is a correctness check, not a performance comparison
between optimization revisions.

Useful environment overrides are `NATIVE_ENV_FILE`, `SHMEM_LIB_DIR`,
`ASCEND_RT_VISIBLE_DEVICES`, `MASTER_PORT`, `SHMEM_HOST`, `SHMEM_PORT`, `PYTHON_BIN`,
`TORCHRUN_BIN`, `BENCHMARK_MODE`, and `TIMEOUT_SECONDS`. By default the selected
Python runs `-m torch.distributed.run`, so a separate `torchrun` executable is not
required in an isolated venv. `--timeout-seconds` bounds each child process group;
`TIMEOUT_SECONDS` bounds the whole shell-launched comparison. Cancellation reaps
only the parent-owned child group, never unrelated device jobs.

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
Each node retains its own rank-local references and validates its local child
result. Only node zero writes the final aggregate. Run the isolated parent
directly; wrapping it in `torchrun` would create nested process groups and is
rejected. Single-node EP8 is the validated hardware topology for this change;
the multi-node orchestration has not been hardware-validated.

## Output

The final JSON contains complete child results and raw log paths, configuration,
source/native fingerprints, per-rank memory ledgers, and precision results for
every compared run. A1 exports rank-local CPU references; B1/B2/A2 must all match
the same workload and complete every comparison phase before an aggregate is
written. Inputs and initial parameters are compared exactly. Full-model
loss/logits, all local gradients, and all updated parameters retain the original
`rtol=2e-2` and `atol=2e-3` after one AdamW step. The model parameters remain BF16 while
`Float16OptimizerWithFloat16Params` updates FP32 master parameters through
HyperParallel AdamW and copies them back to the model.
The wrapper's historical `Float16` class name denotes the low-precision model
path; this benchmark explicitly uses BF16 model parameters and FP32 master
parameters.
Gradient snapshots are collected on CPU only for the compared first step and
released before the remaining warmup and measurement. Memory peaks reset after
warmup. `conservative_total_peak_bytes` adds the allocator's peak allocated bytes
and the external SHMEM heap once; B pools are already in the allocator and
workspace views are already in SHMEM. This accounting excludes opaque HCCL/driver
allocations. `final_device_used_bytes` is a device-wide sample after measurement,
not a runtime peak or a process-exclusive statistic. Source fingerprints cover
this example's Python files and `hyper_parallel/**/*.py`; native fingerprints
cover the built payload's `.so` and `.o` files. Workers verify the editable import
path and reject identity changes within or between runs.

Timing uses the rank maximum for each complete optimizer step, a median
within each process, then the mean of those process medians for each backend;
it does not pool samples from different process lifetimes. Use `--order ab` for
a short correctness run. The old `BENCHMARK_MODE=paired` format and oracle remain
available for regression checks. This random-weight benchmark does not establish
checkpoint convergence, and common/MegaMoe ratios are not optimization-revision
speedups.

For the original-width eight-layer shared-B1/capacity-1.5/Dirichlet acceptance,
per-rank memory costs, competing-process observations, and remaining multi-step
numerical limits, see the [isolated benchmark report](../../../../hyper_parallel/core/multicore/doc/megamoe_qwen_isolated_benchmark_20260908.md).
