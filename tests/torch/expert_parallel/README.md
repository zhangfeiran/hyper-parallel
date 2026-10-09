# Expert Parallelism Combination Tests

This directory contains distributed tests that validate the interaction between
basic Expert Parallelism (EP) and other parallel strategies (DP, TP, CP).

## Test Objective

Verify that ExpertParallel and ExpertTensorParallel work correctly when combined
with Data Parallel (FSDP), Tensor Parallel, and Context Parallel dimensions.

## Validation Method

All tests follow the same pattern:

1. Build a standalone MoE model (no parallelism) as the reference
2. Build a parallel MoE model with the target strategy applied
3. Run forward + backward on both models with the same input
4. Compare outputs and gradients within tolerance

## Test Coverage

| Template Name | dp | ep | tp | cp | Min Cards | Validation Scope |
|---------------|----|----|----|----|-----------|------------------|
| ep-only       | 1  | 2  | 1  | 1  | 2         | Basic EP functionality |
| tp-only       | 1  | 1  | 2  | 1  | 2         | Expert-internal TP |
| dp-ep         | 2  | 2  | 1  | 1  | 4         | DP + EP interaction |
| ep-tp         | 1  | 2  | 2  | 1  | 4         | EP + TP interaction |
| dp-ep-tp      | 2  | 2  | 2  | 1  | 8         | DP + EP + TP 3D interaction |
| dp-ep-cp      | 2  | 2  | 1  | 2  | 8         | EP + CP dimension compatibility (only validates EP works with a CP-included mesh, not CP communication itself) |
| dp-ep-cp-with-attention | 2 | 2 | 1 | 2 | 8 | **Real CP communication** with Self-Attention + MoE block |

### 2-Card Runnable Cases (32 total)

| Test Group | Count | Parameter Variations |
|------------|-------|----------------------|
| EP Base | 6 | num_experts: 2/4/8, top_k: 1/2 |
| EP + grouped_mm | 6 | num_experts: 2/4/8, top_k: 1/2 |
| EP + Shared Expert | 6 | num_experts: 2/4/8, top_k: 1/2 |
| TP-only | 8 | num_experts: 2/4, top_k: 1/2, hidden_dim: 64/128 |
| Validation | 6 | Valid/invalid mesh configurations |

## Running Tests

### 2-Card Environment

```bash
pytest tests/torch/expert_parallel/test_combinations.py -v -k "test_2card_group"
```

## Constraints

| Constraint | Description |
|------------|-------------|
| num_experts % ep == 0 | Experts must be evenly divisible across EP ranks |
| hidden_dim % tp == 0 | Hidden dimension must be divisible by TP degree |
| dp % ep == 0 (when dp > 1) | Data parallel degree must be divisible by EP degree |
| dp * ep * tp * cp == world_size | Product of all dimensions must equal total devices |

## Dirichlet hot-replica sweep

`dirichlet_routes.py` preserves the seeded Dirichlet sampling, water filling and
distinct TopK construction from `megamoe-sun`. Its original 55 alpha/seed pairs
are replayed in the same shuffled order by every variant. Smaller alpha usually
increases imbalance; compare realized home-destination skew rather than assuming
alpha uniquely determines rank load. The skew is measured before replication.

The worker `_benchmark_dirichlet_replica.py` compares native grouped experts and
MegaMoe with B=0 or B=1 replica slots per rank. Defaults are EP16/E96, 16 independent
MoE layers, 4096 tokens/rank, H5120/I1792, TopK8 and BF16. Each step executes all
checkpointed forwards, reverse-order backwards and a full AdamW update. The
scope excludes attention and a router network. Both B=1 variants use the current
CPU planner and default HCCL P2P transport; MegaMoe uses push with initial capacity factor
1.5 and the current automatic growth policy. Execution storage and the bounded
replica pool are shared across layers, and high-watermarks survive route changes.
Native retains the original NPU token-permute and weighted token-unpermute
operators around the current grouped-expert EP entry.

Activate CANN and this checkout's native payload before starting Python, and
ensure the editable install resolves to this checkout. Run the following under
the available device idle gate with all 16 devices selected:

```bash
python scripts/run_dirichlet_replica_sweep.py --output /path/to/smoke --phase smoke
python scripts/run_dirichlet_replica_sweep.py --output /path/to/accept --phase accept
python scripts/run_dirichlet_replica_sweep.py --output /path/to/sweep --phase sweep
python scripts/plot_dirichlet_replica_sweep.py \
    --input /path/to/sweep --output /path/to/dirichlet_curves
```

Smoke runs check two small layers at two load distributions. Acceptance compares
production-size output, input/router gradients and all three expert weight
gradients against native B=0 for balanced, intermediate and highly skewed routes;
the rank-max relative L2 threshold is 1%. Acceptance is separate from timed runs.
The sweep uses two fresh processes per variant in mirrored variant order, with
3 warmup and 7 measured steps per route. Source, native payload, initial weight,
input and route hashes are recorded alongside all-rank results.
`--phase full` runs acceptance, the sweep and plotting in order. `--resume` keeps
only complete runs whose physical monitor is clean and whose recorded source and
native payload hashes still match; it archives rejected attempts before retrying.

Select only MegaMoe B=1 and its kernel-gradient replica transport with:

```bash
python scripts/run_dirichlet_replica_sweep.py --output /path/to/kernel_gradient \
    --phase full --variants megamoe-b1 --replica-transport shmem_signal_kernel_gradient
```

A selected variant still runs acceptance and two independent sweeps. Automatic
four-series plotting runs only when all four variants are selected. Reusing older
baselines requires auditing unchanged model/runtime sources, native payload,
initial weights, inputs, routes and measurement boundaries. The worker records
the actual provider and W2/W13 readiness separately from the requested mode.

For a targeted comparison, the worker accepts `--pairs` and
`--replica-target-load`. The latter sets MegaMoe's planner target in received
rows per rank, independently of its initial push buffer capacity. For example,
`--replica-target-load 32768` aligns EP16/E96 routing to the average load while
retaining the initial capacity factor 1.5. Its default retains capacity-based
push planning. Re-run compared variants with the same selected pair order so
their weight-update histories agree.

`--diagnose` captures separate Torch/NPU and internal MegaKernel traces for one
checkpointed MoE layer after timing finishes. It records original forward,
recompute and backward, actual placement, provider flags and inclusive host
spans. These profiler-on traces diagnose overlap and readiness waits; they do
not replace profiler-off 16-layer step samples.

Latency is the median of rank-max step durations. Accounted HBM is rank-max
allocator peak plus the full external SHMEM heap, including optimizer state;
reserved allocator memory and sampled physical card/process HBM are separate
fields. Physical monitoring records PIDs and launcher ancestry. The plotter
requires complete, matched all-rank runs and rejects foreign processes, unhealthy
devices and missing physical samples. It writes CSV, JSON, PNG and SVG artifacts.
| sequence_length % cp == 0 | CP attention tests require sequence length divisible by cp |
| CP support | Only dimension compatibility is validated; full CP communication requires attention module (covered by cp-with-attention test) |
