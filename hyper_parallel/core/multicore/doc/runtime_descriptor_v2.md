# MegaMoe graph-sized runtime descriptors

This increment removes the fixed 25,600-task / 1,024-event construction limits and adds an optional tensor-table wire representation. Logical tasks, queue permutations, triggers, dependencies, ACKs and wave ordering remain unchanged. It does not skip empty expert tasks or remove synchronization.

## Host allocation and event lifetime

The host reserves the sum of graph task counts, one termination slot, and 23 extra queue positions for every shared phase-barrier descriptor. Capacities are rounded to 16. Small graphs retain the original `RuntimeConfigC` host structure; larger graphs use the same fields with sized ctypes arrays. Device offsets remain uint32, so layouts at or above 4 GiB are rejected before allocation.

For `Ephysical = E + EP * B` and `Nphysical = Ephysical / EP`, the largest existing backward graph and its readiness/route events need ordinary IDs through `2 * Ephysical + 9 * Nphysical + 10`. The event capacity is:

```text
event_capacity = max(1024, align_up(2 * Ephysical + 9 * Nphysical + 11, 16) + 16)
phase_barrier = event_capacity - 2
wave_barrier  = event_capacity - 1
ready_base_bytes = 4 * event_capacity
route_arrival_base_bytes = ready_base_bytes + (EP + 1) * 64
```

The final 16-counter cache line is reserved for barriers. The configured plan, its home-only fallback, forward and both backward variants share one event capacity. Workspace reuse rejects a different event layout. Per-call clearing preserves the persistent ready generation; arrivals and copy ACKs are cleared separately. SHMEM heap sizing and the framework-free HBM model use the same production formula.

## Wire format

The original unmarked fixed image and the previous bit-31 dynamic-capacity image remain readable. Extended images set both bits 31 and 30 of header word 2; the low 30 bits contain task capacity. All integers are little-endian.

| Byte offset | Content |
| --- | --- |
| 0 | task count, unchanged including the existing forward termination convention |
| 4 | worker count |
| 8 | capacity flags and task capacity |
| 12 | extra config value, including the ready-handshake magic |
| 16 | event capacity |
| 20 | tensor-template count (zero for extended dense images) |
| 24 | task stride: 576 for dense, 128 for tensor-table records |
| 28 | format version: 2 |
| 32–63 | reserved zero bytes |
| 64 onward | trigger counts, task records, event descriptors, queue counts, three queues, dynamic data, group list, atomic values, ready metadata |
| aligned tail | 64-byte-aligned tensor templates |
| image end | optional existing 40-byte split-weight pointer extension |

A compact task contains its original first six uint32 words, eight `(tensor_template_id, base_ptr_offset)` pairs, and its original last ten words. Each template preserves all 16 TensorDesc words except `base_ptr_offset`, which is restored from the task record. Even unused tensor slots are preserved. Records and templates are cache-line aligned. C++ validation runs once at worker initialization.

Managed online plans use tensor-table records by default (`HP_MEGA_MOE_COMPACT_RUNTIME=1`); `0` selects dense records for A/B comparison. The generic offline serializer still defaults to dense for compatibility. Expanded event/task allocation works in either mode. Rebuild the multicore native payload before running this ABI; old native binaries cannot read extended descriptors.

`compact_physical_weights` controls B-slot weight storage independently of this descriptor format.

## Memory and cost accounting

The plan audit builds the real graph using meta tensors and reports eager configured/home plan storage. It separately counts a warmed upper bound for all three split-weight runtime caches: each cache is a full `torch.cat(base_config, pointer_metadata)` allocation, retained alongside its base. The predictor consumes this upper bound when an audit is supplied. Each audit describes one execution-resource group on rank zero; callers with several independent groups must account for each group.

Descriptor traffic decreases without changing task count or expert FLOPs. The old calibrated descriptor service time cannot be scaled by the 128/576 byte ratio: template cache hits, scalar decoding, dependency polling and critical-path overlap affect elapsed time. Use profile-off A/B/B/A for throughput, and separate task/cycle profiles for attribution. EP16/32/64 graph construction and network simulation remain distinct from real multi-node execution.

## Validation entry points

```bash
HYPER_PARALLEL_PLATFORM=torch python -m pytest -q \
  tests/ut/core/multicore/mega_moe/test_runtime_serialization.py \
  tests/ut/core/multicore/mega_moe/test_workspace.py

HP_MEGA_MOE_COMPACT_RUNTIME=1 HYPER_PARALLEL_PLATFORM=torch \
  python examples/torch/multicore/audit_mega_moe_plan.py \
  --ranks 16,32,64 --slots 0,1,2,3 --cube-cores 20 \
  --output /tmp/megamoe-compact-audit.json
```

The serialization tests compile the actual `runtime_config.hpp` reader with a host compiler, round-trip all descriptor bytes, verify queues/events, exercise old fixed and dynamic formats, and locate the split-weight extension after the template pool. They include real E384 graphs that exceed each old limit, optimized replay flags and terminal references beyond the old task capacity. They do not substitute for NPU numerical and lifecycle tests.

## Decoder calibration and worker topology

The decoder fit uses each worker's full `seen` count and its descriptor-decode cycle counter. It does not divide by the number of sampled records, and excludes dependency waits and held-out route pairs. It rejects mixed runtime formats, native payloads or worker counts. GEMM/transfer fits already present in the hardware JSON are preserved when only the decoder is fitted.

```bash
python examples/torch/multicore/predict_mega_moe.py \
  --descriptor-profiles /path/to/profile/rank0.json /path/to/profile/rank1.json \
  --train-pairs 10000:42,0.1:7,0.005:943 \
  --hardware /path/to/hardware.json --output /tmp/hardware-with-decoder.json
```

`Hardware.vector_workers` replaces the old fixed divisor of 24. Its portable default remains 24, while the local 910B3 profiles identify 20 Cube and 20 active Vector workers. Plan audits accept `--cube-cores`; the predictor requires this topology to match the supplied hardware calibration. Neither a 24-worker descriptor audit nor its byte total should be silently substituted for a 20-worker target.

A lower decode service time does not imply the same end-to-end percentage gain. The current cost model still stages rank-level work, and its balanced-route overprediction indicates missing overlap. Prediction reports retain calibration domains and simulation limitations.
