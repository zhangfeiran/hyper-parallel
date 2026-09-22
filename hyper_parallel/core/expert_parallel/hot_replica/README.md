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
Weights use ordinary allocations, so push heap rebuilds do not invalidate guest
storage or require unmanaged SHMEM allocations.

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

The eager executor materializes a dense H+B weight tensor and re-prefetches for
backward. It does not yet implement segmented home/guest kernel pointers, a shared
persistent B-slot weight pool, or one-sided RMA transport. These require separate
lifetime and performance validation; the present P2P allocations safely coexist
with dynamic push growth. Multi-node operation, expert TP, graph capture and FSDP
composition need dedicated system tests before being advertised as supported.

No end-to-end speedup or peak-HBM reduction is claimed by precision validation.

## Tests

CPU planner/capacity tests are in
[`test_hot_replica.py`](../../../../tests/ut/core/expert_parallel/test_hot_replica.py).
Distributed native/push/pull launchers are in
[`test_hot_replica.py`](../../../../tests/torch/expert_parallel/test_hot_replica.py).
The worker additionally accepts `--backend`, `--budget`, and `--result-dir` for
controlled fresh-process validation. Tests compare outputs, input/router/weight
gradients, SGD updates and momentum, and reversed backward of live distinct plans.

The worker can select production dimensions with `--tokens`, `--hidden`,
`--intermediate`, and `--top-k`. `--same-backend-reference` compares multicore
replication with the same transport at B=0; the default reference is native B=0.
Incoming gradient partials retain the per-element precision gate. Deferred
backward also checks exact BF16 accumulation of captured partials, because
cancellation can amplify relative error in a final BF16 gradient. Comparison
failures are reported collectively before cleanup to avoid stranding other ranks.
