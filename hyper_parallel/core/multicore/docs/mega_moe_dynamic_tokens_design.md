# Bounded dynamic tokens in MegaMoe

`MegaMoeExperts` accepts exactly one token-size argument:

- `local_num_tokens=T` preserves fixed-size execution, including its existing tile constraints.
- `max_local_num_tokens=T_cap` accepts actual source lengths from zero through `T_cap`, independently on every EP rank.

Hidden size, intermediate size, experts, top-k, EP membership, dispatch mode, communication splits, dtype and gradient
participation remain fixed and consistent across the EP group. Dynamic execution requires eager mode; graph capture is
rejected before native execution.

## Storage and schedules

The dynamic bound is rounded up to 128 for symmetric source/return storage. No synthetic tokens, routes or probabilities
are appended to the model input. A forward exchanges actual lengths and gradient participation with expert counts in one
asynchronous gather. All ranks validate token limits, counts and gradient participation before starting a MegaKernel.

A task plan covers the group maximum actual length, rounded up to 128, with a minimum of 128 even for an all-empty group.
The execution-resource owner caches at most four plans. Autograd retains each invocation's plan independently, so eviction
does not invalidate deferred or retained backward. Tiling tensors and runtime configurations record their consuming stream.
The current ACLNN base launches prepared runtime tensors directly; there is no additional per-length runtime-image cache.

SHMEM reservation depends on the configured bound, not the current task plan. Push retains managed lossless capacity growth
and the `initial_capacity_factor` / `capacity_growth_factor` API. Pull rejects explicit capacity factors and allocates received
expert activations in ordinary HBM. Both transports reserve symmetric source/return regions using the aligned bound.
Changing the actual length does not itself rebuild the heap.

## Empty and unequal sources

A zero-token rank still participates in routing and expert forward/backward: it can receive work from other ranks.
The minimum local receive allocation is one row for a non-null ABI argument; this does not introduce a logical route.
For pull, permutation and backward staging use only the actual prefix of the reserved symmetric source buffer. An empty
prefix retains the symmetric allocation's base address when entering native code, because remote addresses are derived
from that base. Empty token permutation/unpermutation operators are bypassed, while empty input and router gradients remain
connected to autograd.

Native PUT/GET scheduling still publishes completion signals for zero-length transfers. SwiGLU tail tiling is reset on each
call so a small tail cannot restrict a later, larger invocation using the same plan.

## Ownership and validation

Modules sharing execution resources must have compatible static specifications and execute serially. Each forward saves
its own route metadata, received activation tensors and backward plan. Shared scratch remains protected by workspace
leases, completion events and the base runtime's graph-lifetime tracking. Resource teardown and push heap rebuild retain
the lifecycle guarantees documented in the [component README](../README.md).

`tokens_per_expert` remains a trusted router histogram. Callers must supply valid, distinct top-k expert IDs and matching
counts. Group configuration, forward order, grad mode and the input/expert/router `requires_grad` policy must agree. A rank
must not independently skip a layer or backward. Token overflow and inconsistent participation fail before native launch.

## Tests

CPU unit tests cover constructor bounds, route metadata, unequal/all-empty sources, plan-cache eviction, empty symmetric
pointers, pull prefix staging, deferred backward and checkpoint replay without acquiring an accelerator.

`tests/torch/multicore/test_mega_moe.py::test_mega_moe_dynamic_tokens` runs push and pull on eight ranks against independent
FP32 and rounded-BF16 references, including changing shapes, hotspots, empty ranks, frozen inputs/weights and shared layers.
The DSV4.1 launcher adds a saturated positive-clamp reference and all-gradient comparison using external expert parameters.
These are operator/adapter tests; they do not establish full-model memory fit or training performance.
