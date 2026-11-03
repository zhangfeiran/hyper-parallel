# Bounded replica dependency observations

The existing `mega_kernel_profile` switch enables these records. Ordinary execution
keeps the frozen static-band traversal and the per-element FP32 peer reduction order.
This observer does not admit replicas, change the cost model, change weight
residency, reorder bands, or introduce a collective. A separate profiled traversal
allocates bounded head observations and probes, while preserving the same fixed
band and owned-edge order. It must not be used for performance acceptance.

## Device record contract

Records retain the existing 32-byte `CycleTraceRecord` layout. Descriptor
`0x11000 + projection * 0x100 + kind` uses projection 0 for W2 and 1 for W13.
The packed task field contains EP-local peer rank in bits 14+, producer core in
bits 8–13, and band in bits 0–7. Producer core 63 and band 255 are sentinels.
`stage_task_index` carries the actual protocol epoch. The owner field contains
EP-local rank in bits 16+ and physical expert in bits 0–15; expert 65535 is a
projection-wide sentinel. The collector also retains the containing invocation,
step, worker, global EP-group rank list, and Host lease transfer map.

| Kind | Decoded name | Meaning |
|---|---|---|
| 1 | ProducerReady | Start after the real FIX_MTE2 output fence; end after ready publication and DCCI |
| 2 | HeadPendingWindow | First head observation until readiness is confirmed; includes service of other bands |
| 3 | ConsumerReadyObserved | First successful check by this consuming worker |
| 4 | GradientReturn | Existing FP32 remote DataCopy/add work for this worker's disjoint elements |
| 5 | IndependentReadyObserved | Another owner is ready and the blocked head is rechecked as still unready |
| 6 | ReplicaReturnsMerged | All replica-return workers completed this projection; other local owner dW may still run |
| 7 | AckIssued | Owner submitted the existing guest-release signal |
| 8 | AckObserved | Guest's local wait interval for the existing ACK |
| 9 | EmptyProducerReady | This producer core has no tiles for the band, not necessarily an empty expert |
| 10 | LocalNotReadyObserved | First failed head check stopped at a local producer core |
| 11 | GuestNotReadyObserved | First failed head check stopped at a guest producer core |

At most one independent-owner opportunity is emitted for each head/band. Only the
leader within a worker subgroup probes; at most four probes are allowed, each
checks up to four later edges, and subsequent probes are separated by at least
5000 local cycles. A missing opportunity does not prove that no opportunity exists.
No record is written on every poll. Producers publish once per core/band as before.

Weight prefetch uses Host `put(..., use_sdma=True)`. Gradient returns use the
reserved AIV workers' remote-pointer `DataCopyPad` MTE2, FP32 vector add, and MTE3
writeback. These are distinct execution resources; the return record is not a
standalone runtime SDMA operation.

`WorkerTasksComplete` is the worker's task-queue completion observation.
`WorkerComplete` also includes that worker's existing completion handshake, or the
reserved return worker's existing ACK path. These are observed lower bounds for
hardware kernel completion, not an outer runtime kernel timestamp. Parameter
post-accumulate NPU events in the collector are separate stream markers and do not
claim FSDP, optimizer, or complete model-gradient readiness.

## Weight prefix and clock domains

The exclusive profiler installs a private process-wide Host observer visible to
Python autograd threads. Lease identity remains thread-local. Host records are
bounded at 16384 per window and retain only primitive metadata: transfer map,
phase, provider identity, credit-wait submission, projection copy submission,
consumer-wait submission, actual weight epoch, and actual backward generation
reuse hit/miss. No lazy histogram or Tensor is retained by the observer.

`ReplicaWeightReadyWait` is the existing Device-side projection wait immediately
before the consumer obtains its weight pointer. Its end is a first-use readiness
boundary. A Host copy-submission interval is not an SDMA copy duration. Runtime
stream waits before kernel launch require an outer runtime trace to measure their
actual Device duration; Host observations alone cannot measure this prefix. The
cluster launcher enables `--outer-capture` on EP-local ranks 0, 16, and 63 only,
covering a hot owner and ranks on other nodes. These Level0 captures use a
separate runtime clock and must not be joined to raw internal cycles without
calibration. Keep them until analysis finishes, then delete their raw data.

Every Device event declares a per-rank Device clock domain. Host observations use
`host_monotonic:pid...` and are exported separately. Only subtract timestamps in
the same calibrated domain. Across ranks, pair producer/publication, consumer,
return, and ACK by EP-group global ranks, lease transfer map, protocol epoch,
projection, logical expert, and band. Do not subtract remote cycles to invent a
network latency. The phase and protocol epoch are independent: backward can reuse
a previous weight epoch while reserving new gradient epochs.

## Static executor port and validation

This observation port starts at the frozen A source, not the previous experimental
ready-cursor branch. The original `ProcessGradientBands` body is unchanged after
removing its diagnostic-only early branch. The profiled traversal finishes every
owned edge for one subgroup band before entering its next band. No later ready
owner or band is consumed early, including when the observer detects an opportunity.
The original blocking `WaitBandProducer` remains unchanged; nonblocking readiness
helpers serve diagnostic observation only.

The static Host harness compiles the production traversal and validates 74 worker/
band combinations with both profile modes, exact nonassociative FP32 peer order,
ragged final bands, bounded observations, and an independently delayed producer.
Publisher tests compile the actual final-output fence and empty-core contribution
path. Host capture tests cover inactive windows, nested/thread-local leases,
exclusive installation, and bounded drops. No new NPU result is implied by these tests.

Rebuild both HyperMegaMoe and HyperMegaMoeGrad for the target compute unit after
applying the native headers, worker changes, and the grouped-matmul patch. The
updated dependency lock pins the patch bytes. Host registration and the existing
32-byte cycle-record ABI are unchanged, but the new event decoder and capacity
budget must ship with the new payload. Never combine this capture with calibration
coefficients from a different native fingerprint or diagnostic mode.

The source graph builders, planner, model, policy, route code and cost coefficients
are unchanged. `plan.py` only enlarges lazy diagnostic record budgets. The private
signal transport hooks only read actual lease, epoch, copy submission and reuse
facts; residency identity and its lifecycle remain owned by the residency feature.

The frozen capture worker already present in this source can consume the public
internal profiler in an explicitly active window; the normalized postprocessor
`summarize_replica_causal.py` is also provided. Detailed capture setup and native
fingerprints belong to the experiment manifest, not a production calibration table.
