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
parameters. The default `eager_on_first_forward` strategy prepares storage before
the first forward graph. `lazy_on_first_replica` delays preparation until an
active placement without an outstanding home-only graph. Both preserve
parameter identity, optimizer ownership, and state-dict shape.
Do not keep raw storage aliases across layout preparation or modify parameters
between forward and backward. Live graphs retain their required storage;
incompatible placements use isolated snapshots. Optimizer updates invalidate
the physical-weight cache and require a refresh.

The learned route uses a fixed-size INT32 histogram to avoid data-dependent
output sizing while retaining zero-count expert entries.

## Routed-expert skew curves

`plot_mega_moe_dirichlet.py` renders a separate, expert-layer-only comparison:
native, MegaMoe with zero replica slots per rank (`B=0`), and MegaMoe with one
replica slot per rank (`B=1`). Both MegaMoe variants retain lossless multi-wave
overflow handling. The B1 legend states whether compact physical weights or
snapshot storage was measured; these storage modes must not be mixed.

Pass repeated sweep JSON files from independent processes:

```bash
python examples/torch/multicore/plot_mega_moe_dirichlet.py \
  --native output/native_a.json output/native_b.json \
  --b0 output/b0_a.json output/b0_b.json \
  --b1 output/b1_a.json output/b1_b.json \
  --output-dir output/plots
```

Rendering requires `matplotlib`. Use `--no-render` to validate inputs and
export CSV/JSON with only the Python standard library; neither operation
initializes an accelerator or imports a training framework.

Each input is a performance sweep with `variant`, `kind`, `source`,
`configuration`, and `points` fields. Points use the routed-expert ST's
`shape`, `topology`, `route`, `rank_max_fwd_bwd_ms`, `validation`, and `memory`
records. Source identity includes the commit, worker/adapter hashes, and
native artifact hashes. The tool rejects unmatched alpha/seed sets, different
logical routes or shapes, incompatible settings, duplicate samples, and
inconsistent memory accounting instead of silently joining unlike runs.

The outputs are `dirichlet_skew_latency_hbm.png`, an SVG, a point-level CSV,
and `dirichlet_skew_summary.json`. The upper panel shows median forward plus
backward latency; the lower panel shows total peak HBM per rank. Both use the
same **original home-destination skew**, before replica placement. Dashed wave
boundaries describe B0 only; B1's physical skew and wave counts in the CSV
are planner predictions, not a substitute for a kernel trace. Lines connect
measured routes rather than extrapolating a fitted performance model.

Latency pools equal-length samples from repeated processes, while retaining
each process median separately. HBM is the maximum over repeats of
`max_rank(allocator_peak + full_external_SHMEM_heap)`. Workspace tensor views
are not added again, and the result is not an eight-card sum or whole-device
`npu-smi` occupancy. A layer reused across routes retains its workspace and
physical-weight high-watermarks, which remain part of this memory total.

This fixed-route F+B workload excludes optimizer updates and activation
checkpointing. In particular, B1 can reuse same-version physical weights
after warmup; the curve does not include the refresh cost following an
optimizer update. Use the complete optimizer-step example above when that
cost or full-model memory is the question.
