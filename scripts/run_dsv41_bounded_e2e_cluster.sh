#!/usr/bin/env bash
# Measure bounded adaptive replicas through the full Trainer before further model calibration.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo "Run seven pure-timing Trainer stages on 8 nodes x 16 dies, EP64."
    echo "Order: Push B0/B2/B6, Pull B0/B2/B6, Native B0. No calibration phase is launched."
    echo "Each stage runs 35 steps with balanced/medium/hot windows; use rank-max window medians."
    echo "B is an adaptive upper bound. The current graph has no matching accepted cost table."
    echo "Host Engram, full decoder/attention recompute, fused attention, NUMA and one 4 GiB heap."
    echo "No profiler, module timing or route observation hooks. DRY_RUN=1 prints the actual commands."
    exit 0
fi
export RUN_ID=${RUN_ID:-dsv41-text8-bounded-e2e-20261104-001}
export STAGES=${STAGES:-push_b0,push_b2,push_b6,pull_b0,pull_b2,pull_b6,native_b0}
export DSV41_CALIBRATION_E2E=0 REPLICA_SCHEDULE_E2E=0
export HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY=0 HYPER_PARALLEL_SHMEM_HEAP_SIZE=4294967296
export HYPER_PARALLEL_REPLICA_CALIBRATION_DIR= HYPER_PARALLEL_REPLICA_REQUIRE_CALIBRATION=0
unset HYPER_PARALLEL_REPLICA_CALIBRATION_AUDIT_DIR DSV41_CALIBRATION_STATUS
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
export NODE_LOCAL_ROOT=${NODE_LOCAL_ROOT:-${MODELARTS_JOB_DIR:-/tmp}/dsv41-${RUN_ID}}
export DSV41_REPLICA_LOCAL_ROOT=${NODE_LOCAL_ROOT}/benchmark
: "${DSV41_ROUTE_SCHEDULE:?DSV41_ROUTE_SCHEDULE must name the frozen route-window schedule}"
export TRAIN_ITERS=35 WARMUP_STEPS=5
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_replica_e2e_cluster.sh" "$@"
