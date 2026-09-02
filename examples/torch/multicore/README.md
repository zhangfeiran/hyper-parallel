# MegaMoe Qwen optimizer benchmark

This directory contains a synthetic eight-layer Qwen MoE optimizer-step
validation and benchmark. The model reuses HyperParallel's Qwen-compatible
Qwen flash-attention kernel, `RMSNorm`, `RotaryEmbedding`, and `SwiGLUMLP`
building blocks. The example keeps only a thin QKV projection shell around the
fused attention operation, the routed-expert integration, and a language-model
shell. It compares the managed API with the production native fallback built
from large `torch_npu` permute, grouped-matmul, and unpermute operators.

## Integration boundary

`QwenMoeBlock` owns the learned TopK router and shared expert. Its routed
experts are selected as follows:

- `mega_moe` constructs `MegaMoeExperts` with the fixed model shape and the
  default capacity factor of `1.5`;
- `native` constructs the production `NativeMoeExperts` reference class. The
  example does not carry another native MoE implementation.

Each decoder layer owns independent expert parameters and state-dict entries.
After all eight layers are constructed, the MegaMoe variant calls
`share_execution_resources()` once, before the first forward, so serial layers
share one bounded runtime/workspace. There is no model-side `finalize()` call;
the worker process owns native resources for its lifetime.

The model and benchmark do not construct or inspect private runtime, route, or
workspace objects. Kernel split fields remain source-level tuning points and
are not benchmark arguments.

## Fixed workload

| Item | Value |
| --- | --- |
| Hardware topology | 8 ranks, TP1, EP8 |
| Data type | BF16 |
| Batch / sequence | 1 / 1024 per rank |
| Hidden size | 2048 |
| Decoder layers | 8 |
| Routed experts / TopK | 16 / 8 |
| Expert intermediate | 512 |
| Receive capacity factor | 1.5 |
| Optimizer | AdamW |
| Timed operation | zero-grad, forward, loss, backward, dense-grad sync, optimizer step |
| Default measurement | 5 warmup + 10 measured steps |

Both backends start in fresh processes with the same model seed and rank-local
input seeds. The first warmup verifies finite loss and gradients and confirms
that both dense and expert parameters update. The fifth warmup samples stable
optimizer-step memory.

## Run

Build and install the native payload, install this checkout in editable mode,
then run the managed backend:

```bash
CANN_ENV_FILE=/path/to/CANN/set_env.sh \
bash examples/torch/multicore/benchmark_qwen_moe_mega_moe.sh \
  --backend mega_moe \
  --warmup-steps 5 \
  --measured-steps 10 \
  --tag minimal_a
```

Run the native baseline in a fresh process:

```bash
CANN_ENV_FILE=/path/to/CANN/set_env.sh \
bash examples/torch/multicore/benchmark_qwen_moe_mega_moe.sh \
  --backend native \
  --warmup-steps 5 \
  --measured-steps 10 \
  --tag native_a
```

For a formal comparison, use independent A/B/B/A processes rather than
alternating backends in one process:

```bash
for spec in \
  "mega_moe minimal_a" \
  "native native_a" \
  "native native_b" \
  "mega_moe minimal_b"
do
  set -- $spec
  CANN_ENV_FILE=/path/to/CANN/set_env.sh \
  bash examples/torch/multicore/benchmark_qwen_moe_mega_moe.sh \
    --backend "$1" --tag "$2"
done
```

Useful environment overrides are:

- `CANN_ENV_FILE`: selected CANN activation script;
- `NATIVE_ENV_FILE`: multicore payload activation script;
- `SHMEM_LIB_DIR`: symmetric-memory runtime library directory;
- `ASCEND_RT_VISIBLE_DEVICES`: eight visible device IDs;
- `MASTER_PORT` and `SHMEM_PORT`: process-group and SHMEM ports;
- `PYTHON_BIN` and `TORCHRUN_BIN`: executable overrides;
- `TIMEOUT_SECONDS`: whole-launch timeout.

The wrapper verifies that `hyper_parallel` imports from this checkout instead
of relying on `PYTHONPATH` to mask a stale editable installation.

## Results and memory meaning

Rank zero writes
`<output-dir>/optimizer_step_<tag-or-backend>.json`; every rank also writes a
`.rank<N>.json` raw result. The aggregate contains:

- complete fixed topology, model shape, dtype, optimizer, runtime versions,
  and fingerprints for the activated native vendor, op library, framework
  adapter, and build manifest;
- first-step loss, finite-gradient status, and dense/expert update checks;
- cross-rank maximum optimizer-step latency samples plus P50, P90, minimum,
  and maximum;
- cross-rank maximum memory for model initialization, first optimizer step,
  and a stable optimizer step.

Memory reports both PyTorch allocator allocated/reserved peaks and
`torch.npu.mem_get_info()` total-minus-free HBM sampled every 10 ms. The latter
is intentionally separate because SHMEM heap allocations need not appear in
framework allocator counters. Sampling is done on an untimed warmup, so it does
not perturb the ten latency samples.

The random-weight model validates integration and optimizer execution; it does
not establish checkpoint convergence or application-level throughput.
