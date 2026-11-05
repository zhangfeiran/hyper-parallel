#!/usr/bin/env bash
# Calibrate the current graph in this allocation, then compare full Trainer policies.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo "Run on 8 nodes x 16 dies, EP64. Calibrate the replica caps in the bundle manifest."
    echo "Collection includes bounded frames and the frozen balanced/medium/hot decoder route templates."
    echo "The first EP group fits; a disjoint EP group and a fresh model-in-loop capture validate."
    echo "Trainer: native B0, Push/Pull B0, and adjacent heuristic/calibrated pairs; 35 steps each."
    echo "A quality rejection skips only that modeled stage; runtime failures stop the job."
    echo "All cases share a 4 GiB SHMEM budget. Normal prefetch misses; valid R-to-B reuse is retained."
    echo "No profiler or module/route timing is installed. DRY_RUN=1 prints commands without device use."
    exit 0
fi
export RUN_ID=${RUN_ID:-dsv41-text8-autocal-20261103-001}
export DSV41_CALIBRATION_E2E=1 REPLICA_SCHEDULE_E2E=1
export HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY=0
export HYPER_PARALLEL_SHMEM_HEAP_SIZE=4294967296
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
: "${DSV41_ROUTE_SCHEDULE:?DSV41_ROUTE_SCHEDULE must name the frozen route-window schedule}"
export TRAIN_ITERS=35 WARMUP_STEPS=5
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_replica_e2e_cluster.sh" "$@"
