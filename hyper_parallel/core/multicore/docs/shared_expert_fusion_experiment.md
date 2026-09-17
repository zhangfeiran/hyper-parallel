# Shared expert fusion on the current MegaMoE runtime

## Scope

The implementation starts from Hyper `2632ba92` (`megamoe-push-pull`) and
PanGu `f91f5f874` (`feat/megamoe`). The historical Hyper worktree at `8f14cc16`
was inspected for its ABI and scheduling lessons; it was not cherry-picked.

The acceptance shape is the 505B family: sequence 4096, microbatch 2, TP1/EP16,
8192 local tokens, hidden size 5120, routed/shared intermediate size 1792,
384 routed experts, TopK8, BF16, gradient accumulation fusion disabled, matching
PanGu `validate_megamoe_perf.sh` and the M-D-M runner. Timing covers one complete MoE layer's forward
and backward, including route preparation, permutation, the shared branch,
output addition, and input/parameter/router gradients. It excludes the router
projection, attention, optimizer, and DP gradient reduction. These results must
not be presented as whole-model training speedup.

## Implementation

`MegaMoeExperts(shared_intermediate_size=...)` accepts caller-owned dense
`shared_gate_up_weight` with shape `[2I, H]` and `shared_down_weight` with shape
`[H, I]`. Both must be contiguous and match the BF16 input device and dtype.
Zero shared width preserves the existing public behavior. The shared weights
are not registered or initialized a second time.

The shared up/SwiGLU/down computations and their backward matrix products run
inside the existing forward/backward megakernels. Shared tensors use ordinary
local NPU storage, with no extra shared-expert communication. Routed/shared
output addition and dX addition remain ordinary Torch operations. Saved tensors
retain each microbatch's shared activations; the workspace lease and current
stream continue to protect communication storage.

Both native ops append a pointer table and a packed shared runtime buffer after
the profiler input. Existing tiling/event/profile slots remain unchanged. The
pointer table starts with scheduling flags followed by tensor addresses; shared
TensorDesc entries use `tensor_type=2`, `dynamic_dim` as the address index, and
`base_ptr_offset` as the element offset. The loader requires shared ABI version
2. Rebuild and activate both the vendor and Torch adapter together.

The runtime adds separate shared events beyond the routed ready/completion
lanes. Its default shared placement is `idle`:

- Forward: put shared up/down Cube rounds after routed Cube work, overlapping
  the routed Combine tail.
- Backward: run shared dActivation at Cube entry during the initial Dispatch
  wait; put shared dW2/dX/dW1 at the routed W2Grad/ActGrad phase boundary.
- Run shared SwiGLU and its gradient on the previously idle even AIV workers.
  Odd AIV workers retain the routed queue. The shared vector suffix is excluded
  from their loop, while the odd worker owning termination still executes it.

The internal experiment runner also retains `tail`, `overlap`, and `dispatch`
placements. `tail`, `overlap`, and `idle` have real-device checks; `dispatch`
has only queue/event simulation coverage and is not a performance recommendation.

PanGu enables this path with `--mega-moe-fuse-shared-expert`, or
`accelerate_config.moe.fuse_shared_expert: true`. It defaults off and requires
`--moe-backend=mega_moe`, TP1, a configured shared expert,
`--no-gradient-accumulation-fusion`, and no MLP-only recompute. Existing BF16/SwiGLU/bias/dropout/quantization restrictions still
apply. Parameter ownership, checkpoint names, and default RNG behavior remain
in PanGu.

## Profiler findings

The original 256-record per-core cap truncated the target trace by 332550
records. The host/device cap is now 16384; actual allocation remains derived
from queue requirements and rounded to 16 records. Truncated traces are rejected
by the experiment runner. The final `idle` target captures contain 355182 records
on each of ranks 0 and 15, with zero dropped records.

Representative rank-0 measurements below are diagnostic traces, with profiling
on. Stage spans include gaps across workers and are not unprofiled latency.

| Forward scheduling | Shared up ends | Shared SwiGLU span | Shared down ends | Mean Cube wait before shared down |
| --- | ---: | ---: | ---: | ---: |
| Tail on routed AIV | 13.855 ms | 14.913–17.864 ms | 18.343 ms | 4.05 ms |
| Late Combine insertion on routed AIV | 13.944 ms | 13.950–16.849 ms | 17.339 ms | See raw trace |
| Dedicated even AIV (`idle`) | 14.084 ms | 14.196–14.335 ms | 14.825 ms | 0.28 ms |

With `idle`, backward shared SwiGLUGrad runs at 0.907–1.178 ms instead of
5.334–7.458 ms in the routed AIV insertion. It no longer occupies the routed
vector queue. Shared Cube work still consumes real Cube time; independent graph
branches alone do not imply free overlap. Long initial waits on the even AIV
workers are expected because those workers await the shared Cube producer.

The initial tail experiment also used a synchronous pointer-table upload.
Subsequent runs use a pinned host table and a nonblocking copy on the current
stream. Therefore the early tail-to-overlap wall-time difference cannot be
attributed solely to placement. The earlier exploratory ABBA overlapped a host
build and is explicitly excluded from final performance acceptance.

## Validation and reproducibility

Local environment: Ascend910 (`ascend910_93`), 16 logical NPUs with 64 GiB each,
CANN 9.1.0, Torch 2.9.1, torch_npu 2.9.1. Runs use the cooperating per-device
`npu_wait_and_run.sh` locks, two idle samples, and a final 30-second idle window.
Each benchmark records command, source patch, native hashes, runtime versions,
NPU health, process ownership, seven output/gradient signatures, and peak HBM.

The repository entry point is
`examples/torch/multicore/shared_expert_profile.py`. After editable installation,
build and activate the native component before importing Torch:

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910_93
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
export LD_LIBRARY_PATH="$PWD/build/native/payload/hyper_parallel/core/multicore/shmem/lib:$PWD/build/native/payload/hyper_parallel/core/multicore/shmem/lib/shmem:$LD_LIBRARY_PATH"
export HYPER_PARALLEL_PLATFORM=torch
export SHMEM_IP_PORT=tcp://127.0.0.1:62828
# Launch through the site's NPU idle gate. Omit --fused for the paired baseline.
python -m torch.distributed.run --nproc-per-node=16 --master-port=62818 \
  examples/torch/multicore/shared_expert_profile.py \
  --output /path/to/result --fused --warmup 5 --steps 50
```

Use `--check --profile` for a separate numerical/trace run. Use
`--route dirichlet` for a fixed global Dirichlet(alpha=10) expert load with
independent local token samples. Performance uses profiler-off ABBA, five warmup
and fifty measured steps per leg, independently restarted processes, identical
routes/weights/gradient inputs, and the maximum rank latency at each step.
The runner's p95 uses linear interpolation. Do not include `--check` in timed
acceptance runs because its reference allocation changes the memory pool.

Hyper multicore UT: 127 passed, 634 subtests passed. PanGu adapter UT: 31 passed;
focused MoE argument/configuration UT: 73 passed. Queue simulation checks exact task
coverage and forward progress, including the separate even-AIV suffix. Real
small and target `idle` comparisons have zero maximum absolute error for output,
dX, router gradient, both routed weights, and both shared weights on all ranks.

The PanGu worker `tests/group3/integrated_tests/model/_test_megamoe_shared_expert.py`
checks strict checkpoint loading, three SGD updates, two outstanding forwards
per step, reverse-order backward, and real FP32 DDP gradient buffers before DP
reduction. This is not a full trainer or distributed optimizer acceptance test.
An additional pre-guard diagnostic at PanGu `fcd023334` tested direct FP32
accumulation; the restriction and failed numerical evidence are detailed below.

Machine-local raw results, traces, build logs, scripts, and the stage-span plot:
`/home/ma-user/zfr/shared-profiler-results-20260917/`.
## Final profiler-off ABBA

Each row pools 100 measured rank-maximum samples per mode from two independent
restarts. Both forward and backward, including the shared expert, are timed.

| Routing | Standalone median | Fused median | Time reduction | Standalone p95 | Fused p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced | 61.707 ms | 60.168 ms | 2.49% | 63.229 ms | 61.676 ms |
| Dirichlet(alpha=10) | 64.707 ms | 62.865 ms | 2.85% | 66.611 ms | 63.760 ms |

The paired median reductions are 2.79% / 2.25% for balanced routing and
3.02% / 2.44% for Dirichlet. All eight legs use identical source patches, runner,
runtime, and 15 native binary hashes. All seven signatures match exactly on
all 16 ranks within each routing workload. Every start/end health snapshot
reports 16 healthy NPUs and zero foreign workers.

| Routing | Allocated peak, standalone/fused | Reserved peak, standalone/fused |
| --- | ---: | ---: |
| Balanced | 5.7770 / 5.7857 GiB | 6.9668 / 6.7383 GiB |
| Dirichlet(alpha=10) | 5.9826 / 5.9913 GiB | 7.1504 / 6.9453 GiB |

These are Torch NPU allocator peaks, not total device memory including external
SHMEM allocations. The allocated increase is 8.93 MiB in both workloads;
reserved memory is an allocator measurement, not a live-tensor size estimate.
Raw device HBM snapshots are retained alongside the allocator measurements.

The final balanced PanGu integration again has exact output/dX/router/FP32
main-gradient agreement on all 16 ranks across three updates and two outstanding
forwards per step.

### Unsupported direct FP32 shared gradient accumulation

The exploratory FP32-accumulation check first stopped because the standalone
PanGu baseline could not load `libatb.so`. Activating the installed ATB 8.5.2
C++ ABI 1 library with CANN 9.1 allowed the additional numerical experiment to
run. Its three-update elementwise comparison failed: maximum absolute error
0.0625 and maximum relative L2 0.003849. The first step already showed shared
dW rounding differences: the independent baseline accumulates FP32 matmul
outputs, while the integrated kernel returns BF16 weight gradients for DDP to
add to FP32 buffers. The tolerance was not relaxed.

Both argument validation and the PanGu adapter now reject shared fusion with
`gradient_accumulation_fusion=True`. The requested acceptance workload remains
valid: the existing 505B/M-D-M runners explicitly pass
`--no-gradient-accumulation-fusion`. Supporting direct FP32 shared accumulation
requires a separate native output/accumulation path and is outside the measured
configuration. No performance result here uses the additional ATB environment.

The original batch exited nonzero only at this extra check, after all eight ABBA
legs and the accepted PanGu integration had completed. `validate-final.py`
checks each completed leg independently and emits `final/acceptance.json` plus
`final/ACCEPTED_SCOPE_PASSED`; the unsupported combination stays explicitly
failed in that report.

Acceptance artifacts under the result root:

- `final/balanced/abba_summary.json` and `final/dirichlet/abba_summary.json`.
- `final/<route>-<A1|B1|B2|A2>/`: commands, patches, binary hashes, per-rank
  samples/signatures, and start/end NPU health.
- `final/pull-small/`, `final/dirichlet-check/`: numerical comparisons.
- `final/pangu-integration/`: accepted training-state comparison;
  `final/pangu-grad-accumulation/`: retained failed direct-FP32 comparison.
- `final-validation-v2.sh`, `run_env.sh`, and `validate-final.py`: exact orchestration and audit.
- `atb-validation-environment.json`, `complete-validation-atb.sh`, and `run_atb.sh`: extra-check provenance.
- `placement-spans-final.png` / `.svg` and the complete raw internal cycle traces.

The fusion switch remains opt-in. Pull has a small real-device correctness check;
its production-shaped performance, other TP sizes, other hardware/toolchains,
and full-model throughput have not been measured in this experiment.
