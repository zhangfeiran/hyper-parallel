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

Signal storage and the persistent provider are freed/recreated together by the
heap manager. Autograd saves neither symmetric addresses nor an old provider for
multicore. Native callers must keep their externally supplied provider/storage
alive until all queued consumers finish and reinitialize after replacing a heap.
Compatible multicore layers must share execution resources to share direct
symmetric slots; native layers share the injected provider.

Native can inject `ExpertParallel(..., replica_transport=provider)`. The shared
[`OneSidedReplicaTransport`](one_sided.py) requires an externally leased symmetric
uint8 inbox and a runtime exposing `put`, `get`, and `host_barrier`. Its inbox must
hold `B * max(expert_matrix_numel) * 4` bytes. The direct
[`SignalReplicaTransport`](signal_transport.py) also needs `signal` and
`wait_signal`, with transfer completion and signal operations ordered on the
calling stream. Size its 64-byte-aligned uint8 allocation with
`signal_storage_bytes(matrix_shapes, B, ep_size, weight_element_size)`, then pass
`SignalReplicaTransport(runtime, storage, B, ep_size)`. All ranks must invoke the
same provider call sequence. Runtime PE numbering must match EP-group local
ranks. The caller owns initialization, heap budget, exclusive submission and
teardown. Both native providers and the planner remain independent of multicore;
the tests explicitly inject its SHMEM runtime for NPU validation.

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
sorting. The current planner still runs on the host after the count exchange.

## Current implementation limits

The planner still performs one host readback after count exchange. Device-side
planning and transport/compute overlap remain future work. Multi-node operation,
expert TP, graph capture and FSDP composition need dedicated system tests before
being advertised as supported.
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
The worker additionally accepts `--backend`, `--budget`, `--replica-transport`, and `--result-dir` for
controlled fresh-process validation. Tests compare outputs, input/router/weight
gradients, SGD updates and momentum, and reversed backward of live distinct plans.
Small shapes also exercise independent layers sharing the guest pool, with
different weights and reversed backward.

The worker can select production dimensions with `--tokens`, `--hidden`,
`--intermediate`, and `--top-k`. `--same-backend-reference` compares multicore
replication with the same transport at B=0; the default reference is native B=0.
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
