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
overflow handling. The B1 legend identifies compact, snapshot, or cross-layer
shared B storage; these storage modes must not be mixed.

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
records. Source identity includes the commit, worker/adapter hashes, Torch
MegaMoe Python implementation digest (including dirty-tree edits), and native
artifact hashes. The tool rejects unmatched alpha/seed sets, different
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

### Checkpointed multi-layer sweep

`benchmark_mega_moe_layers.py` measures 16 independent MoE layers by default,
using EP8, 4096 tokens/rank, H5120, I1792, E48, TopK8, BF16, and capacity
factor 1.5. All layer forwards precede reverse-order backwards. Each layer
has its own parameters and input storage, with reentrant activation checkpoint.
This exercises cross-layer resource lifetimes; it is not a connected transformer
and excludes attention and router-network computation. Every layer receives the
same original route in a step, making the skew coordinate unambiguous.

Both MegaMoe variants share execution resources before first forward. B1 uses
`shared_physical_weights=True`, leaving home parameters intact and sharing only
the B-slot weight/FP32-gradient pools. `torch.optim.AdamW` updates parameters
every iteration (`lr=1e-4`, `betas=(0.9, 0.999)`, `eps=1e-8`,
`weight_decay=0.01`, `foreach=False`, `fused=False`). The latency plot uses the
directly measured full step (zero-grad + F+B + AdamW update), while JSON and CSV
also retain F+B-only latency. Both synchronized wall timers use the same start;
cross-rank timing aggregation happens after both stop, not between phases.
The full-step rank-max is not a sum of phase rank-maxima. Subsequent replica
refresh and checkpoint recomputation are timed. HBM includes
all 16 layers' home parameters/gradients, inputs, live outputs, retained resources,
AdamW states/update temporaries, and the full SHMEM heap. State is initialized
before resetting peak statistics. The per-rank optimizer-state byte/dtype ledger
is diagnostic, not an extra term added to the allocator peak. Standard AdamW
uses parameter-dtype moments (BF16 here); this example does not add FP32 master
parameters or model a full transformer's optimizer memory.

After activating matching native artifacts and verifying the editable checkout,
run each variant in a fresh, idle-gated process, for example:

```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29500 \
  examples/torch/multicore/benchmark_mega_moe_layers.py \
  --variant b1 --layers 16 --warmup 3 --measured 7 --output output/b1_a.json
```

The default route set is the same 55 alpha/seed pairs used by the single-layer
curve, including high-skew W3/W4 routes. `--pairs 10000:42,0.005:943` selects a
short smoke sweep; `--limit 4` replays the first four routes of the full shuffled
sequence without changing their order. Use the same command/configuration for native, b0, and b1,
then repeat in reverse order (`native, b0, b1, b1, b0, native`). Pass all six JSON
files to the plot command above. Each point validates finite outputs and all
input/router/parameter gradients before warmup and after all measured updates.
The post-measurement check also validates parameters and AdamW state, outside
both timing and the HBM peak window; this is not a numerical oracle comparison.
Run the shared-weight and compact-transition precision regressions
separately when changing runtime code.

The plotter rejects mixed layer counts, checkpoint modes, optimizer protocols,
and workload scope, including whether latency includes the optimizer update.
Latency is measured for the complete multi-layer step, not single-layer latency
multiplied by the layer count.

The legacy single-layer fixed-route F+B workload excludes optimizer updates and activation
checkpointing. In particular, B1 can reuse same-version physical weights
after warmup; the curve does not include the refresh cost following an
optimizer update. Use the complete optimizer-step example above when that
cost or full-model memory is the question.
