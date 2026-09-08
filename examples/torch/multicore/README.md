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

The multi-layer worker's v3 Python digest includes the standard EP and common
expert sources, and excludes generated `build`/`prebuild` trees. Both ST helper
files are fingerprinted, and native payloads remain fingerprinted separately.
Results from before the minimal-API rebase use a different native implementation
and must not be reused as measurements of the current code.

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

All variants start from the same seeded standard EP expert weights for each
layer; MegaMoe copies the rank-local shard into its packed layout before timing.
Native parameters and optimizer states are DTensors: validation and the HBM
ledger inspect their local shards, not the global logical tensor size.

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

Add `--profile-output output/b1_trace.json` to capture an optional rank-zero
CPU/NPU Chrome trace. After **all** timing and memory measurements, the worker
warms the first selected route with an AdamW update and profiles one extra full
step. This also covers replica refresh after the update. Diagnostic annotations
cover count exchange, planning/remapping, cache agreement, weight P2P, fused
forward/backward launches, backward allocation, replica-gradient return and AdamW.
The wrappers are installed only inside the profiling context and always restored;
disabled profiling adds no annotations or synchronization to library hot paths.
CPU scopes are inclusive launch/wait times, not device compute times, and must not
be added together. Use the NPU timeline for asynchronous attribution and independent
profiler-free runs for final performance/HBM comparisons. The output path must be
new and distinct from the benchmark JSON.

Use `--profile-p2p` for a separate, post-update diagnostic step on every EP rank.
It works independently of the Chrome profiler and stores `diagnostic_p2p` in the
result JSON. Both weights and gradients retain their original communication:
the example temporarily observes transfer builders, batch submission and the
existing completion waits, without adding device-wide synchronization inside
F+B. It restores all wrappers even on failure. No observer, timing events or
payload metadata are created in the default path.

Each nonempty batch reports its phase (`weight_refresh` or `gradient_return`),
phase-call index, peer, operation, shape, dtype and actual tensor bytes. All ranks
must agree on matching sends/receives for each phase round, peer, shape and dtype.
Metadata retains no communication tensors. Empty batches and cache-hit waits
create no events. Device events are read only after the diagnostic step completes.
Three caller-stream timestamps distinguish:

- `launch_to_ready_ms`: before submission to after the existing completion wait;
- `pre_wait_ms`: before submission to entry at the wait site;
- `exposed_wait_ms`: entry at the wait site to readiness for the next consumer.

Separate CPU timers cover descriptor preparation, submission and `handle.wait()`.
The device spans include host enqueue gaps and instrumentation overhead: they
are **not** pure wire time, and `pre_wait_ms` is an opportunity window rather than
measured useful-compute overlap. Per-phase sums and rank maxima are not predicted
end-to-end savings. The serial layer example requires the wait to use the same
caller stream as its submission; unsupported cross-stream attribution fails
instead of reporting a misleading overlap. Final performance/HBM comes from the
ordinary pre-diagnostic measurements and independent profiler-free runs.

For receive-only shared-B ranks, submission and `Work.wait()` run on the control
stream. Their event interval measures that stream's wait; the compute stream's
exposed wait ends at its later join. `analyze_mega_moe_gradient_timeline.py`
uses the same-execution payloads stored in `diagnostic_trace.capture` with
`--profile-all-ranks`. It supports arbitrary rank/layer counts, empty batches,
multiple receive windows and multiple execution resource groups. It separates
control-stream waits, compute-stream joins, owner adds and window reuse fences.
Missing flows, mismatched capture identities and uncalibrated cross-host clocks
are rejected. Legacy EP8/16-layer traces retain their existing reader.

For a small diagnostic workload, keep tokens divisible by 128 and select shape,
window and sharing explicitly (default shape and sharing are unchanged):

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  -m examples.torch.multicore.benchmark_mega_moe_layers \
  --variant b1 --layers 3 --resource-groups 2 --gradient-receive-slots 1 \
  --tokens 512 --hidden 1024 --intermediate 1024 --experts 48 \
  --pairs 0.005:943 --warmup 0 --measured 1 --output output/windows.json \
  --profile-output output/windows-trace.json --profile-all-ranks --profile-p2p \
  --profile-kernel-cycles --profile-tasks

python -m examples.torch.multicore.analyze_mega_moe_gradient_timeline output/windows.json
```

The trace collector also reads the profiler's existing allocator records and
correlates them to CPU scopes. Unattributed events remain explicit; external
SHMEM/HCCL memory and profile-off peaks are separate. A partial JSON left by a
failed vendor exporter is rejected on all participating ranks.

For MegaMoe variants, `--profile-kernel-cycles` runs a separate extra post-update
step after all measurements and writes `diagnostic_kernel_cycles` into the JSON.
It gathers per-core counters from every EP rank. The shared workspace retains
only the **last launch in each direction**: layer zero's forward checkpoint
recomputation and backward for the default sharing configuration, not all layers.
With multiple groups, `captures` identifies the last launch of every group and
direction. Add `--profile-tasks` to retain detailed queues in the same extra step.
The existing `counters` field remains
backward-only; `forward_counters` separates up/down GMM, hidden dispatch, combine,
local compute (SwiGLU/epilogue), dependency waits and synchronization. Counters use
system cycles; per-category worker maxima are not additive wall time.
The option temporarily enables native instrumentation and restores its environment
even on failure. An inherited `HP_MEGA_MOE_KERNEL_CYCLE_PROFILE=1` is rejected by
this timed benchmark, as is inherited `HP_MEGA_MOE_TASK_PROFILE=1`: unset them
and use the diagnostic options instead. The unified offline import entry and
clock/coverage contracts are documented in the
[cost-model guide](../../../hyper_parallel/core/multicore/doc/theoretical_cost_model.md#关联采集与多窗口归因).

Forward diagnostics lazily allocate a separate 4 KiB per shared resource group,
so subsequent backward launches cannot overwrite them. This buffer is reused;
disabled launches neither allocate nor clear it. Ordinary runs therefore add no
persistent HBM. Diagnostic allocations happen after the timing/HBM window and
are not part of the reported default memory peak. Rebuild the native multicore
payload after updating: the private forward operator ABI has gained a byte
workspace input and a `cycle_profile` attribute. The public `MegaMoeExperts`
call signature and existing runtime descriptor positions are unchanged.

The multi-layer adapter reports the actual home-runtime bypass for routes already
inside one receive window: no active replica bindings and unchanged destination
loads, even with a configured B budget. It does not report the ST helper's
hypothetical exact-mean placement as if those replicas executed.

The legacy single-layer fixed-route F+B workload excludes optimizer updates and activation
checkpointing. In particular, B1 can reuse same-version physical weights
after warmup; the curve does not include the refresh cost following an
optimizer update. Use the complete optimizer-step example above when that
cost or full-model memory is the question.
