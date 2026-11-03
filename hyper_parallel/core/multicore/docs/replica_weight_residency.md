# Resident replica weights

Signal transports can reuse an unchanged expert payload already present in a guest slot.
The existing `replica_slots_per_rank` cap, matrix storage, gradient storage and public
constructors are unchanged. The eager P2P transport keeps its existing behavior.

## Owner-authorized reuse

Every participating rank observes the same immutable global transfer sequence. Each
owner keeps bounded host metadata for its last writes to `(target_rank, target_slot)`.
A transfer by another owner invalidates that slot's evidence; inactive slots retain it.

An entry identifies the owner's expert slice and the live source tensors, source
storages, parameter version counters, data pointers, offsets, shapes, strides, dtypes
and devices. References to tensors and storages are weak. Metadata cannot retain an
autograd graph, a parameter allocation or an additional expert matrix.

On a hit, only the owner skips the BF16 matrix puts. It still waits for the current
credit, publishes every current projection-ready word, and waits for reader ACKs.
The receiver follows its normal protocol and never guesses whether a remote owner
updated its parameters. No collective or device-to-host decision is added.

Payload evidence is committed only after the existing protocol succeeds. Exceptions,
epoch rollover and replacement providers invalidate evidence. Storage rebinding or
parameter updates also miss. Unversioned inference tensors always use the copy path.
HyperParallel FSDP assigns a fresh materialization token when all-gather refreshes
an unsharded value while preserving its autograd counter. That token also invalidates
replica evidence. Retaining an unsharded value across accumulation can retain hits.
As with autograd's saved-tensor checks, supported parameter mutations must maintain
Torch version counters; raw writes that bypass those counters are not certified.

## Accumulation and recomputation

Payload identity excludes token counts, microbatch IDs and checkpoint invocation IDs.
The current invocation still has independent routes, gradients, epochs and completion
state. Different token work may reuse the same expert matrix without reusing a route.

This permits forward-to-recompute and cross-microbatch hits during gradient
accumulation when the parameter version remains unchanged. An optimizer update makes
the affected owner's old evidence miss. FSDP storage or tensor replacement also misses
conservatively rather than assuming that a reused address contains the old values.

The existing saved-generation backward fast path remains intact. Other calls retain
fresh transport epochs even when every matrix put hits the owner cache.

## Scope and diagnostics

The cache reuses fixed slots. It does not move weights to inactive slots, reserve
per-layer copies, delay gradient return or change per-element FP32 reduction order.
Layers sharing one pool can overwrite each other's weights, so a larger GAS does not
guarantee more hits. Physical B, runtime routing and the actual call sequence determine
the opportunity.

`SignalReplicaTransport._last_weight_copy_counts` contains local owner copy and hit
edge counts for the latest normal prefetch. A saved-generation backward bypasses that
prefetch and leaves both counts zero. This private diagnostic is not a physical link
counter. Collection should occur outside timed steps.

Cost calibration must distinguish executed weight copies from resident hits and must
include any remaining signal traffic and resource contention. Fewer matrix bytes do
not by themselves establish an end-to-end speedup.
