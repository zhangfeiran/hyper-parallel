# Multicore examples

The current Qwen training benchmark and standard EP reference are documented in
[mega_moe/README.md](mega_moe/README.md). The benchmark uses FP32 main parameters
and AdamW; it does not use the removed standalone native expert implementation.

## Bounded replicas and compact storage

Set `--expert-capacity-factor 1.5 --replica-slots-per-rank 1` in the current
`mega_moe/run_qwen_moe_benchmark.sh` entrypoint to enable the bounded hot-expert
pool. Routes that fit one receive window use the replica-free plan. Overflow
activates placement, and any residual skew is handled losslessly by multi-wave
execution. The API's default capacity remains `None`, reserving the lossless EP
maximum.

`--compact-physical-weights` opts into storing replica suffixes beside home
parameters. The first active placement can rebind parameter storage while
preserving parameter identity, optimizer ownership, and state-dict shape.
Do not keep raw storage aliases across that transition or modify parameters
between forward and backward. Live graphs retain their required storage;
incompatible placements use isolated snapshots. Optimizer updates invalidate
the physical-weight cache and require a refresh.

The learned route uses a fixed-size INT32 histogram to avoid data-dependent
output sizing while retaining zero-count expert entries.
