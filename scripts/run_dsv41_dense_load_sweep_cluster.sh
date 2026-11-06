#!/usr/bin/env bash
# Compare near-balanced loads before later hotspots with an automatic SHMEM Heap.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo "Run DSV4.1 Text8 on 8 nodes x 16 dies, EP64 x expert FSDP2."
    echo "Default stages: native_b0,push_b0,push_b2,push_b6,pull_b0,pull_b2,pull_b6."
    echo "Use the dense frozen schedule: 21 loads, 5 primer + 2 settling/7 measured steps per load."
    echo "SHMEM is automatic; inherited fixed Heap budgets and repacking controls are cleared."
    echo "No profiler, cost table, metadata replay, complete search or projected quotas."
    echo "DRY_RUN=1 prints all worker commands without installation or device use."
    exit 0
fi
: "${DSV41_ROUTE_SCHEDULE:?DSV41_ROUTE_SCHEDULE must name the frozen dense route schedule}"
export RUN_ID=${RUN_ID:-dsv41-text8-dense-alpha-8x16-manual-001}
export STAGES=${STAGES:-native_b0,push_b0,push_b2,push_b6,pull_b0,pull_b2,pull_b6}
export TRAIN_ITERS=${TRAIN_ITERS:-194} WARMUP_STEPS=${WARMUP_STEPS:-5}
export DSV41_CALIBRATION_E2E=0 REPLICA_SCHEDULE_E2E=0 DSV41_COMPLETE_GUEST_SEARCH_E2E=1
export HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY=0 HYPER_PARALLEL_REPLICA_METADATA_REUSE=0
export HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH=0 HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH=0
export HYPER_PARALLEL_REPLICA_RANK_LOCAL_COMM=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
unset HYPER_PARALLEL_SHMEM_HEAP_SIZE HYPER_PARALLEL_SHMEM_FIXED_HEAP_REPACK
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_replica_e2e_cluster.sh" "$@"
