# Bounded expert hot replicas

This opt-in feature keeps logical expert ownership unchanged while assigning a
bounded number of temporary expert replicas to other EP ranks. The planner,
route contract and sparse transport live here and do not import multicore or
multi-wave scheduling.

## Entry points

Native grouped experts use the existing EP strategy:

```python
from hyper_parallel.core.expert_parallel import ExpertParallel

ExpertParallel(replica_slots_per_rank=1).apply(experts, ep_mesh)
```

`experts` is `components.modules.moe.GroupedExperts`. Its `w1`, `w2`, `w3`
parameters and state-dict layout remain unchanged. The current adapter supports
BF16 NPU SwiGLU, expert TP=1, synchronous `all_to_all`. Unsupported asynchronous
combine and deredundency combinations are rejected at construction.

Multicore exposes the same B setting on `MegaMoeExperts`:

```python
from hyper_parallel.core.multicore import MegaMoeExperts

experts = MegaMoeExperts(
    local_num_tokens=4096,
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=24,
    top_k=8,
    ep_size=4,
    ep_group=ep_group,
    replica_slots_per_rank=1,
    dispatch_mode="push",
    initial_capacity_factor=1.25,
    capacity_growth_factor=1.25,
)
```

Both push and pull execute a single physical expert schedule. B=0 retains the
existing path. Hot replication requires distinct, in-range expert IDs within
each token's TopK selection. Multicore validates this collectively before sparse
weight communication. Native receives the existing expert-major count contract.

## Capacity and placement

Let R be the EP degree, H the home experts per rank, S the tokens per source,
K the distinct TopK count, and b=min(B,H). Put A=S*K and U=R*S*min(K,H).

The integer planner first balances work with at most one incoming original owner
per receiver. It retains floor(b*m/H) rows from each balanced transfer of m rows,
using its largest b expert segments. It then uses remaining guest slots and
existing replicas for further moves that decrease overload. Per-source,
per-logical-expert counts are conserved exactly; no rows are dropped.
For each fixed placement, source-rank affinity fills local quotas first, reaching
the maximum possible local row count before assigning remaining remote traffic.

For 0 < b < H, a conservative bound is:

```text
Cmax = align128(min(U, ceil(((H-b)*U + b*A) / H) + R - 1))
```

For B>=H, Cmax=align128(A). MegaMoe B=0 keeps its original global-token bound,
including its existing routing contract.

Push still allocates from `initial_capacity_factor` and grows with
`capacity_growth_factor`. Both initial allocation and geometric headroom are
capped at Cmax. An invalid demand is rejected before the capacity fast path.
Explicit heap budgets retain the existing minimum-growth retry. A planner target
based on the current capacity avoids unnecessary copies for a route that already
fits. Pull continues to use source-sized symmetric storage and rank-local receive
scratch.

## Training and transport

Only selected owner-to-guest weight slices travel through HCCL P2P. Transfer order
is deterministic and every asynchronous handle is waited before consumption.
The group shares a persistent B-slot pool per device, dtype and matrix layout.
The pool borrows home matrices and stores only guest weights and FP32 guest dW.
Exclusive leases and completion events order reuse across streams; different
layers can share storage without retaining their parameters in the pool. Native
runs separate home and guest GMM segments. Multicore uses a versioned address
extension to select home/guest matrices and gradient outputs. B=0 keeps the dense
kernel ABI. Rebuild the multicore native payload from this revision before
using split weights. Native's existing W1/W3 packing is still required.

Ordinary guest allocations survive push heap replacement. Set
`MegaMoeExperts(..., replica_transport="shmem")` to opt into one-sided weight puts
and owner-side FP32 gradient gets. A symmetric B-slot inbox is included in heap
accounting, collective allocation/free, and growth preflight. Each invocation
binds the current inbox after a rebuild. No remote FP32 atomic fan-in is used.
This barrier-based provider uses publication/consumption barriers and an extra
staging copy. P2P remains the default; one-sided transport is opt-in.

`replica_transport="shmem_signal"` selects direct symmetric B-slot execution
weights and FP32 guest dW. The kernel reads/writes these views directly, removing
the intermediate inbox copies and ordinary guest pool. A persistent provider
owns the stream lease and protocol epoch. The receiver grants credit only after
its previous slot consumer, owners put weights then publish ready, and the
receiver acknowledges arrival. For gradients, each guest publishes its finished
FP32 outputs; owners get and accumulate them, then acknowledge completed reads.
All incoming credits are enqueued before any outgoing wait, avoiding wait cycles.
Signals for distinct channel/peer/slot tuples occupy separate 64-byte cache lines.
Initialization and rare int32 epoch rollover use host barriers; steady-state
prefetch/return uses stream-ordered signals without host barriers. This does not
yet overlap communication with home-expert computation.

`replica_transport="shmem_signal_sdma"` uses the same direct slots and signal
protocol, with ACL asynchronous copies against SHMEM-mapped peer addresses.
It requires direct peer mapping and rejects unmapped peers. Native callers select
it with `SignalReplicaTransport(..., use_sdma=True)` and a runtime whose put/get
accept that keyword. The SHMEM runtime also exposes `put/get(..., use_sdma=True)`.
This option preserves stream ordering and allocation validation; it does not add
staging buffers or change the planner and capacity policy. P2P remains the default.

`replica_transport="shmem_signal_sdma_parallel"` additionally prefetches outgoing
weights on one lazily created stream per target peer. Native callers select
`SignalReplicaTransport(..., use_sdma=True, parallel_prefetch=True)`. Each copy
stream waits for source preparation and all incoming credits; received weights
are acknowledged on the caller stream before it joins outgoing copies. The pool
lease therefore covers all copy streams, including when owners or caller streams
change. Source tensors are recorded on their copy stream for allocator lifetime.
Gradient reads and FP32 accumulation retain their original deterministic order.
This mode adds stream/event resources but no expert staging buffers or symmetric
heap bytes. It does not overlap prefetch with home-expert computation. All ranks
must select the same mode; the runtime must support calls on different streams.

`replica_transport="shmem_signal_sdma_bidir"` also reads and accumulates guest
gradients on one stream per projection matrix. Native callers add
`parallel_gradients=True` to the provider options above. Each projection preserves
its original target-rank accumulation order in FP32 and has a separate output,
so no two streams update the same gradient matrix. All projections join the
caller before it acknowledges remote reads or releases the slot lease.

This mode lazily caches one FP32 expert gradient across projections per provider,
independent of B and EP size: `4 * sum(prod(matrix_shape))` bytes, or `12 * D * I`
bytes for packed W13/W2. It allocates only on ranks that read remote gradients and
reuses that scratch across peers, slots and calls. The public
`gradient_scratch_bytes` property reports allocated tensor bytes. This is a local
allocator cache, separate from symmetric B slots and the SHMEM heap budget;
it is released with the provider after all streams finish. It is not a measured
net peak-HBM increase relative to transient buffers in the ordered implementation.
The original P2P, serial SDMA and weight-only parallel SDMA options remain available.

`replica_transport="shmem_signal_sdma_overlap"` retains bidirectional parallel
copies and lets MegaMoe home computation begin before guest weights arrive.
This overlap is specific to MegaMoe; native uses eager HCCL P2P weight copies.
The shared `prefetch_weights(..., overlap=True)` scope returns leased tensors
with `wait_weights()` and optional `weight_ready=(base_address, epoch)` metadata.
Ordinary eager providers retain their existing behavior. The MegaMoe fused
consumer honors the per-slot ready metadata.
The initial implementation overlaps home GMM1 in forward and home activation
gradient matmul in backward, preserving the existing task order.

Credits and any source contiguity conversion run before the copy streams fork.
Those streams then issue only SDMA weight and ready-word copies: publication
does not need an AIV helper to run alongside a fused kernel. Readiness uses an
existing cache line per guest slot, adding no symmetric heap bytes. The shared
lease joins copies and acknowledges receivers before releasing storage, even
when the caller has no guest rows. All ranks must use the same mode.

Multicore deferred consumers require split-weight runtime ABI v3, which appends
the ready base and epoch to the v2 layout. Matching forward/backward kernels
must be rebuilt; an older payload does not support this optional mode. The
new kernels continue accepting v2 for eager prefetch.

`replica_transport="shmem_signal_sdma_projection"` adds independent matrix
publications. The MegaMoe provider selects `projection_ready=True`,
including that option in `signal_storage_bytes(...)`. The shared view exposes
`projection_ready=((matrix0_base, matrix1_base, ...), epoch)`;
`wait_weights(matrix_index)` waits only for that projection and `wait_weights()`
joins every projection. Each publication has its own 64-byte cache line per
slot, adding `64 * B * matrix_count` symmetric bytes. These lines do not alias
the peer credit, whole-slot ready, or ACK channels.

Forward copies W13 then W2; backward copies W2 then W13. Each matrix publishes
ready immediately after its own SDMA copy, allowing the first guest matmul to
start while the other matrix is still being copied. Multicore uses split runtime
ABI v4 with both ready bases and the epoch; the matching kernels also accept v2
and v3. Slot release still waits for every matrix consumer and all copy streams.
The default remains P2P; selecting a transport requires measuring the complete
training step for the intended shape and route.

Signal storage and the persistent provider are freed/recreated together by the
heap manager. Autograd saves neither symmetric addresses nor an old provider for
multicore. Compatible MegaMoe layers must share execution resources to share
symmetric slots. Native layers share the ordinary HCCL P2P guest pool.

One-sided transport and weight-copy/home-compute overlap are MegaMoe capabilities.
Native uses `ExpertParallel(..., replica_slots_per_rank=B, replica_min_rows=...)`
with the shared planner, eager HCCL P2P weight prefetch and FP32 gradient return.
It does not accept a replica transport provider or initialize a SHMEM runtime.
The shared planner remains outside multicore and independent of its runtime.

Each autograd invocation saves its immutable route and original owner weights.
Backward re-prefetches that route, allowing other forwards to run before it.
FP32 guest gradients return in target-rank rounds, bounding an owner's incoming
scratch to B expert slots per projection. They are added to the owner's FP32
partial before gradients reach the original parameter boundary and its hooks.
No replica parameters or optimizer states are registered.

Multicore emits FP32 dW directly from its grouped kernel. The native CANN
non-quantized grouped-matmul API rejects BF16-input/FP32-output dW on the validated
backend. The native adapter therefore uses grouped forward/dX and ordinary FP32
matmul for dW, with boundaries taken from the plan rather than device readbacks.

Expert sort keys use exact FP32 values when the expert-ID range fits 24 bits;
permutation indices always remain integers. Larger expert-ID ranges use integer
sorting. CPU planning runs after count exchange; the optional device planner keeps
quotas and source runs on the NPU and reads back one control summary.

## Current implementation limits

| Capability | Implemented execution | Validation boundary |
| --- | --- | --- |
| Shared CPU planner (default) | Native and MegaMoe push/pull | CPU policy/capacity tests and NPU training ST |
| Shared device planner | Independent Ascend AIV; native and MegaMoe | Exact CPU parity, retained plans and two-stream NPU ST; one control-summary readback remains |
| Native hot replicas | Ordinary HCCL P2P, FP32 guest-gradient merge | Forward/backward, optimizer, cross-layer and reversed-backward ST; no one-sided provider |
| MegaMoe projection readiness | SDMA weight copies overlap home computation | Projection and kernel-gradient ST, including ready publication after fused consumer submission |
| MegaMoe early gradient return | W2/W13 readiness and FP32 owner accumulation | Kernel-gradient ST; end-to-end benefit requires a separate matched benchmark |
| Push receive capacity | Dynamic growth bounded by theoretical maximum | Growth with a retained forward, then reversed backward |
| Offline calibrated quota refinement | CPU planner only | Device planning rejects this option; cost estimates are approximate |
| Multi-node, expert TP, graph capture, FSDP/DP composition | No support claim from these tests | Dedicated composition ST required |

The device planner retains its documented topology/scratch and int32-quota limits.
Implemented capabilities and correctness tests do not establish a performance gain.
Pool leases reject overlapping host submissions rather than allocating extra B
slots. Native parameter packing and per-expert FP32 dW matmuls remain costs to
measure. Barrier RMA additionally reserves a B-slot FP32 symmetric inbox. Signal RMA
places the guest weights and gradients directly in symmetric memory, plus
`5 * ep_size * B * 64` signal bytes and matrix alignment padding. Allocation size
changes are not a measured end-to-end peak-HBM claim.

No end-to-end speedup or peak-HBM reduction is claimed by precision validation.

## Tests

CPU planner/capacity tests are in
[`test_hot_replica.py`](../../../../tests/ut/core/expert_parallel/test_hot_replica.py).
Distributed native/push/pull launchers are in
[`test_hot_replica.py`](../../../../tests/torch/expert_parallel/test_hot_replica.py).
The worker additionally accepts `--backend`, `--budget`, `--replica-transport`,
`--replica-planner={cpu,device}`, `--default-group`, and `--result-dir` for
controlled fresh-process validation. Tests compare outputs, input/router/weight
gradients, SGD updates and momentum, and reversed backward of live distinct plans.
Small shapes also exercise independent layers sharing the guest pool, with
different weights and reversed backward.

The worker can select production dimensions with `--tokens`, `--hidden`,
`--intermediate`, and `--top-k`. `--same-backend-reference` compares multicore
replication with the same compute backend at B=0; the default reference is native B=0.
`--fp32-reference` selects an explicit native B=0 FP32-dW oracle without patching
production hooks. It is mutually exclusive with `--same-backend-reference`.
The FP32 oracle is a correctness control, not a production-baseline speed comparison.

Named launchers cover default WORLD and noncontiguous subgroup P2P, native device
planning, push/pull device planning with projection and kernel-gradient transport,
and K=1/2/3/4/5/6/8 deferred backward. The latter checks distinct legal experts and
two actual saved plans with replicas; the full active-replica acceptance uses
B>0 and `--replica-min-rows=0`. The push K>=5/B1 case checks real growth while an
older forward remains live. Planner parity probes retain twelve plans on two streams.
Results record requested and observed configuration, source identity, EP membership,
CANN/Torch versions and loaded payload hashes/ABI where exported.

For example, after building and activating the payload as described below:

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_test_hot_replica.py \
  --backend push --budget 1 --top-k 8 --replica-planner device \
  --replica-transport shmem_signal_kernel_gradient --fp32-reference \
  --result-dir ./logs/hot_replica/push-device-kernel-gradient
```

Incoming gradient partials retain the per-element precision gate. Deferred
backward also checks exact BF16 accumulation of captured partials, because
cancellation can amplify relative error in a final BF16 gradient. Comparison
failures are reported collectively before cleanup to avoid stranding other ranks.

Signal tests additionally rotate owners twelve times over two NPU streams before
checking results, with a delayed rank on each iteration. CPU protocol tests
exercise arbitrary rank progress with many calls queued ahead. The worker accepts
`--benchmark-iterations` for warmed forward/backward/SGD timing on a fixed hot
route; performance comparisons require fresh-process paired runs and ownership
auditing, separate from the precision results.

## Filtering small expert copies

`ExpertParallel(..., replica_min_rows=N)` and
`MegaMoeExperts(..., replica_min_rows=N)` use the same host planner policy.
The default `N=0` preserves token-balancing placement. With a positive threshold,
the planner prefers returning smaller copies to their home owner. If capacity
requires some of that work to remain remote, it first tries consolidating those
rows into already selected copies. It creates no additional replica and keeps
mandatory copies even when their row count is below the threshold.

After filtering, retained copies may absorb more of their original owner's rows.
Each move stops at the pair's load balance, so the global receive peak cannot
increase. This preserves replica positions and full-expert weight/gradient bytes;
token traffic and compute timing can still change. The current refinement uses
row counts rather than a calibrated backend cost model.

The threshold is a workload-specific proxy for exposed copy cost. Calibrate it
with complete training-step comparisons; a fixed row threshold is not a complete
compute/communication cost model. Source histograms and all logical TopK choices
are unchanged, and original owners still receive the complete FP32 gradient sum.

Multicore supplies its proven receive bound from the exact `S/K/B` shape. Native
has expert-major histograms without original `S/K` metadata. For equal source
sizes, the planner uses `T=S*K` and `max(count)<=S` to choose the largest feasible
integer `K` dividing `T`. The receive bound is nonincreasing in `K` for fixed `T`,
so the resulting bound is safe for every compatible distinct-TopK shape. Unequal
source sizes use the constructive formula with observed home loads. A copy is
removed only if the new plan respects the bound and does not exceed the original
home-only peak load.

Returning work home can increase receive capacity and reduce remote weight/grad
bytes. Push retains dynamic growth up to the same theoretical maximum; the policy
does not change the resident guest budget `B` or allocate another weight cache.
Planner policy is per module and does not change the layout of shared execution
storage. All EP ranks must use the same policy for a given invocation.

## Consuming invocation-owned gradients

The native and multicore adapters allocate fresh FP32 home gradients for each
backward. They pass `consume=True` to the shared `return_gradients(...)` helper,
allowing accumulation directly into those buffers instead of cloning all home
experts. Only detached FP32 buffers with exclusive invocation ownership may be
consumed. They must not alias parameters, saved activations, guest slots, other
projections, or previously returned gradients. The caller must use the returned
tensors and must not reuse the original gradient values afterward.

The ordinary `return_gradients` provider method keeps its non-consuming contract.
A provider explicitly supports ownership transfer by implementing
`return_gradients_owned(gradients, guests, route)`. P2P and the built-in one-sided
providers support it; providers without that method receive their original
three-argument call. All paths keep the same FP32 peer accumulation order.

On parallel signal transport, the producer stream still records the fork event
after gradient production and publication. Copy streams wait for that event,
record the output tensors on their streams, and join before ACK and lease
release. The returned home gradients are ordinary invocation allocations; they
do not belong to the reusable guest pool or symmetric heap.

## W2 return inside multicore backward

`shmem_signal_kernel_gradient` keeps projection SDMA weight prefetch and uses
one-sided MTE reads for W2 gradients inside the fused backward kernel. The
adapter first verifies that every Cube executes W2Grad before ActGrad and that
the expert's ActGrad event joins all its Cube workers. Unknown schedules retain
late return. The shared planner and receive capacity policy are unchanged.

The otherwise idle even AIV workers accumulate disjoint blocks of each home W2
matrix, preserving target-rank order for every FP32 element. Worker zero
publishes all local guest readiness before waiting for remote producers. After
all workers finish, it acknowledges remote reads and waits for this rank's guest
readers before the kernel can complete. Odd AIV and Cube queues do not depend
on these workers. Python then returns W13 through the ordinary transport.

The v5 runtime extension points to invocation-owned metadata and cache-line
completion words. It reuses the provider's ready/ACK channels with a distinct
epoch and adds no full expert inbox. This mode is available only to MegaMoe;
native returns both projections through HCCL P2P after local backward computation.


## Reuse planned receive loads

A hot-replica route already contains exact CPU `destination_loads`. MegaMoe uses
these for both rank-local intermediate allocation (`max(1, local_load)`) and the
global push growth check, avoiding a second device sum and host readback of the
uploaded dispatch counts. Routes without a replica plan retain their count
readback. Push still grows dynamically up to the theoretical maximum capacity;
pull keeps its preallocated bound. Count-exchange waits and routing offsets are
unchanged.

## Measured cost refinement

`ExpertReplicaCostModel` optionally refines the existing replica quotas after
capacity planning. Pass the same immutable model to every rank through
`ExpertParallel(replica_cost_model=model)` or
`MegaMoeExperts(replica_cost_model=model)`. Calibration must match hidden and
intermediate dimensions, EP size, backend and transport. Native calibration uses
ordinary HCCL P2P. Version 2 requires explicit `calibration_version=2`, FP32 weight
partials, and a compatible schedule identity (`native_grouped_v1`,
`fixed_queue_v1`, or `fixed_queue_w13_first_v1`). The schedule defaults to the
identity implied by backend/transport. Regenerate old calibration data; merely
changing its version does not validate it against the new model.

Without measured schedule-prefix windows, all modes charge the full exposed
weight and gradient transfers. Later home work cannot hide a stall at the first
guest dependency in a fixed queue. Incoming and outgoing transfers are counted
separately and added; neither full-duplex bandwidth nor multi-peer contention is
inferred. The score is `max(rank forward) + max(rank backward)`, shared by
reporting and quota refinement. This is a conservative phase heuristic, not a
claim that real execution has an intervening global barrier or a guaranteed
latency bound. Stage-prefix overlap discounts remain unsupported.

Supply monotonic `(rows, milliseconds)` forward/backward tables starting at
`(0, 0)`, full weight-copy cost per replica, exposed gradient-return cost and
optional remote-token cost. Include measurement uncertainty and additional host
planning time in `minimum_gain_ms`. These are approximate costs: measure complete
forward/backward/optimizer steps separately before enabling a model in training.
Absent calibration or expert counts outside the measured range retain the
original quota policy.

Refinement searches only existing replica edges, may remove a replica, and
accepts only a strict reduction of the same phase score beyond `minimum_gain_ms`. Receive rows
may increase within the existing capacity limit. The B slot budget, lossless
routing and push dynamic growth with a theoretical upper bound are preserved.

Routing uploads invocation-owned runs and dispatch counts in one aligned buffer.
For a globally replica-free plan it maps logical IDs directly to home physical
slots, including holes reserved by B. CPU plan derivatives are cached only on
the immutable plan; device metadata and weights are never cached across calls.

## Shared fused device planner

Set `replica_planner="device"` on either `ExpertParallel` or `MegaMoeExperts`.
The default remains `"cpu"`. The device solver implements the same integer
placement policy, including `target_load`, `replica_min_rows`, capacity-safe
small-copy consolidation and retained-edge rebalancing. Offline calibrated
cost refinement requires the CPU planner and is rejected with the device backend.

The optional Ascend 910B kernel builds independently of multicore and SHMEM.
After activating the CANN environment, run from the repository root:

```bash
cmake -S hyper_parallel/core/expert_parallel/hot_replica/_device_kernel \
  -B build/replica-planner
cmake --build build/replica-planner -j 4
cmake --install build/replica-planner \
  --prefix "$PWD/hyper_parallel/core/expert_parallel/hot_replica/_device_kernel"
```

To package the optional library, install it to
`build/replica-planner-payload/core/expert_parallel/hot_replica/_device_kernel`
and use the existing wheel build option
`HYPER_PARALLEL_NATIVE_OUTPUT_ROOT="$PWD/build/replica-planner-payload"`.
When also packaging multicore, stage both components under the same payload root.
An unbuilt source installation must run the CMake commands with the prefix pointing
to its installed `hot_replica/_device_kernel` directory. Loading is lazy;
CPU planning and ordinary native EP do not load this library. This first fused
implementation uses 180 KiB of AIV scratch and accepts topologies satisfying
`5*EP*E + 3*E + 4*EP <= 23040`, including EP16/E256. Larger topologies are rejected
before launch. Count values and arithmetic overflow are checked on device.

One device task consumes the gathered histograms and writes physical placement,
destination counts, rank splits, int32 dispatch quotas and preordered source runs.
The runs and int32 counts feed routing without extra sorting, quota gather or
count conversion kernels. Only the compact control prefix is read back. Each call
owns its output storage; later calls and different streams cannot overwrite a
retained route. Producers must establish normal current-stream dependencies;
the launch records tensor lifetimes on that stream. No graph cache or mutable
replay workspace is needed.

Remap and count consumers use the device tensors directly. One compact control
copy provides physical ownership, destination boundaries and rank splits for
host P2P calls, native allocations and push capacity growth. This remains a
host-controlled execution boundary. Native imports no multicore or one-sided
transport implementation, and push retains growth up to the theoretical bound.

MegaMoe's `shmem_signal_kernel_gradient` mode also returns W13 inside backward
when the generated Cube schedule proves per-expert readiness. W13Grad precedes
dX on every Cube; the complete dX event fences the W13 output. W2 and W13 use
distinct epochs and completion storage, and the kernel joins all remote readers
before releasing the guest lease. Unrecognized schedules retain late return.


## Controlled performance and diagnostics

Run the fresh-process worker through the same four-card environment and native
payload as the correctness tests:

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_benchmark_hot_replica.py \
  --backend push --budget 1 --replica-transport shmem_signal_sdma_projection \
  --replica-planner cpu --result-dir /tmp/replica-benchmark
```

The default EP4/E24 shape has six home experts per rank, S=512, D=5120,
I=1792, K=8. The worker times complete forward/backward/SGD steps on directly
owned MegaMoe parameters, with fixed router probabilities. Rank maximum is
reduced outside the timed interval. It records lazy initialization, first hot
route/growth, balanced, steady hot and rotating-hot distributions separately.
Warm windows must keep capacity and SHMEM epoch constant. Audit the recorded
plans and capacities before comparing variants; timing does not establish
numerical correctness.

Compare adjacent modes in separate fresh processes in ABBA order: B=0, B=1
CPU/P2P, bidirectional SDMA, slot overlap, projection readiness, kernel gradient
return, then CPU/device with the selected transport. Keep shape, routes, plan,
precision and warmed capacity identical for mechanism comparisons. B=0 computes
BF16 weight partials while B=1 uses FP32 partials, so that first comparison is an
actual production baseline comparison, not an isolated communication result.
Optimizer and direct parameter ownership are identical within this worker;
absolute timings from different drivers are not interchangeable.

Add `--diagnose` to collect an extra, untimed TorchNPU trace, internal cycle trace,
and inclusive host/stream intervals. The spans cover count gather, solver/AIV
launch, control D2H, remap, metadata upload, invocation metadata and transport
submission. They can nest, overlap, and include queue gaps; do not sum them into
step latency or treat stream intervals as pure device kernel time. The host
observer excludes profiler teardown and plan-audit readbacks. Use the framework
trace for kernel duration. Internal `ReplicaWeightReadyWait` records
identify rank, projection, slot, epoch, consumer stage, logical expert and owner
peer. `epoch` is not a task number. Only the profiling path records cycle pairs;
normal execution adds no synchronization or poll-loop logging. The trace checks
for dropped records and missing wait records on active guest ranks.

Memory snapshots distinguish Torch allocated/reserved peaks from the external
SHMEM heap and list guest weight/FP32 gradient payload bytes, their shared backing
storage, gradient scratch, expected home FP32 partials, and saved nonparameter storage. Subcategories
can overlap and must not be added again. Saved storage is a lower bound, not all
activation memory. Only a constant SHMEM reservation in a no-growth window may
be added to that window's Torch peak; peaks from different steps are not summed.

For a separate candidate-plan experiment, first generate calibration:

```bash
torchrun --standalone --nproc-per-node=4 tests/torch/expert_parallel/_benchmark_replica_calibration.py \
  --replica-transport shmem_signal_sdma_projection --result-dir /tmp/replica-calibration
```

Then pass `--cost-model /tmp/replica-calibration/cost-model.json` to the CPU
benchmark and compare against an uncalibrated CPU run in fresh-process ABBA
order. Calibration measures single-expert F/B, full weight copy, both gradient
projections, and host refinement overhead. Its additive per-expert tables
include launch/control costs and approximate link contention; they are not a
whole-schedule oracle. Actual candidate plans, complete step distributions and
capacity changes must be reported separately from transport ablations. A lower
model score alone is insufficient evidence for enabling calibration in training.

The precision worker accepts the same `--cost-model` file. Validate its output,
dX, router-probability gradients and owner dW against `--fp32-reference` before
using a changed plan. The reference never receives the candidate calibration.


For a MegaMoe B>0 invocation whose global plan has no transfers, forward and
backward skip the guest pool lease, guest gradient clearing and gradient return.
The invocation still saves its immutable route and computes FP32 home weight
partials. Its runtime keeps the existing v2 home-expert addressing marker with
null guest pointers: the marker also selects per-expert GMM addressing and
cannot be omitted. The global transfer list controls this path; an owner with
outgoing transfers must participate even when it has no local guest work.
