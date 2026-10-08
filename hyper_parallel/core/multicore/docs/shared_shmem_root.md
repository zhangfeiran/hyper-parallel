# Frozen shared SHMEM root

`SharedShmemRoot` in [consumer.py](../shmem/consumer.py) provides an explicit,
process-owned registry for serial MoE/DSA consumers. Declare all consumers before
binding any of them. Their independent byte budgets are summed and rounded to
2 MiB physical pages before the first native initialization. An explicit root
limit or the existing `HYPER_PARALLEL_SHMEM_HEAP_SIZE` must cover this complete
sum. Binding freezes the consumer order, layouts and heap size across ranks.

The first implementation requires the same ordered EP, CP and SHMEM root
membership. It supports one device per process and serial invocation, including
serial switches between streams. Each process holds one native root reference;
consumer binding does not acquire another reference. Unregistered acquire,
release, allocation and one-sided submissions are rejected while this root is
active. Existing standalone SHMEM/MoE lifecycles use their original path.

## Prepare resources

Use the established process groups and native payload activation described in
[build.md](build.md). Construct the modules and configure any existing
`share_execution_resources` grouping before reserving their root consumers.
The MoE group keeps its original source/routed/event allocations; it is not
packed into the DSA arena and DSA specs never enter the MoE budget formula.

```python
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec, MegaDsaWorkspace,
)

root = SharedShmemRoot(device, root_group=ep_group)
moe.bind_shmem_root(root, "moe", dtype=activation_dtype)
dsa_workspace = MegaDsaWorkspace(
    root,
    "dsa",
    DsaWorkspaceSpec(
        source_tokens=source_capacity,
        staging_tokens=staging_capacity,
        owner_gradient_tokens=gradient_capacity,
        cp_size=cp_size,
        dtype=activation_dtype,
    ),
)
dsa_workspace.bind()
```

`device` is the explicitly indexed local NPU. All root members must make the
same declarations, bindings, allocations, frees and collective closes in the
same order. DSA source and gradient capacities cover the largest owner shard,
including uneven shards; padding is internal scratch, not a global token ID.

MoE pull has fixed symmetric source storage. Shared-root push must preallocate
its entire lossless receive bound, including the expert replica capacity bound
when enabled. A growth-capable push configuration is rejected before reserving
its consumer. Both the fixed MoE adapter and the native lifecycle prohibit
shared-root heap rebuilding. Standalone MoE still uses its grow-only manager.

## Lease, publication and backward lifetime

The root inserts a stream wait on the previous consumer's completion event.
Concurrent consumer execution, nested execution leases and graph capture are
rejected. Symmetric allocation/free is setup/teardown work and is forbidden
during a lease. Budget and allocation ownership are checked before native
allocation/free; bookkeeping is committed only after a successful native call.

DSA owns one aligned byte arena with separate owner c/K-RoPE/index-K publication,
selected-key staging, FP32 gradient inbox and event views. Inbox stripes are
exclusive per source peer. Independent events occupy 64-byte cache lines.
The storage layout provides space for a native protocol; it does not implement
ready/ACK signaling or gradient reduction by itself.

Prepare `DsaBatchMeta` using `root.generation`, complete owner-local KV IDs,
actual group-local CP rank and direct CP-to-root PE mapping. Prepare the device
owner-storage permutation outside execution:

```python
invocation = dsa_workspace.prepare(batch_meta)
with dsa_workspace.lease(invocation, direction="forward"):
    dsa_workspace.publish_owner_states(invocation, (compressed_kv, key_rope, index_key))
    # The native execution must consume published values and complete its remote ACK protocol here.
```

Publication enqueues detached copies in owner-offset order. It does not provide
an autograd bridge, remote readiness, a barrier or a host synchronization. The
caller/native execution must keep the lease until all remote reads and ACKs are
ordered before completion. A backward invocation takes a new lease and
republishes caller-owned saved or recomputed activations. Scratch views cannot
be the only saved backward state. Layout, layer, microbatch, invocation,
direction and heap generation form the lease's logical identity.

Heap generation increases on successful native initialization/reinitialization.
Shared-root participants agree on the prospective generation before bootstrap;
old generations cannot execute against new addresses. Cross-consumer rebuild
and transparent rebinding are outside this version.

## Close and validation

Close modules/workspaces in a consistent collective order, then call
`root.close()`. Closing MoE must preserve DSA's allocation and vice versa.
The root rejects close until all bound consumers have freed their allocations
and retired their leases; only the root releases the final native reference.
Close is explicit and idempotent, rather than relying on Python finalizers.

The [system test](../../../../tests/torch/multicore/test_shared_shmem_root.py)
compares MoE output, input/route gradients and both expert weight gradients with
common MoE while alternating DSA publication/remote reads on the same root. It
covers independent Q/K ownership order, uneven shards, delayed backward,
serial stream switching, stable memory and both close orders. Run it after acquiring the
local device idle gate; the launcher itself expects already activated
CANN/native artifacts.

This resource implementation does not claim fused DSA attention, sparse
communication performance, model CP/TP integration, CP differing from EP,
concurrent execution or graph capture. The replicated CP correctness baseline
and failed BF16 model acceptance remain separate evidence. The next native
stage must extract mixed AIC/AIV tiles with explicit global causal offsets,
then validate their complete forward/backward against that baseline.

Validated on Ascend 910B3 with a fresh CANN 9.1.0 native payload: CP2/EP2 and
CP4/EP4 each ran pull and full-bound push with both consumer close orders.
Every scenario ran eight rounds with two outstanding MoE forwards, reverse
backward, changing layer/microbatch/invocation publication values and serial
stream switching. MoE output, input/route gradients and both expert weight
gradients matched common MoE at its existing `rtol=0.02, atol=0.002` gate;
DSA remote reads matched exactly. Each scenario retained one native reference,
five symmetric allocations, a fixed 2 MiB heap and stable generation. Warm
Torch live allocation variation was zero in this fixture. Generation advanced
from 1 through 4 across the four completed lifecycles on every rank. Both close
orders preserved the other consumer's allocations and ended in Uninitialized.
The existing standalone MoE level0 precision test also passed. These are small
resource/coexistence fixtures, not full DSA numerical or training acceptance.
