#!/usr/bin/env bash
# Compare compute-only and communication-prefix schedules through the full safe8 Trainer.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo "Run 11 uninstrumented full Trainer stages on 8 nodes x 16 dies, EP64."
    echo "Baselines: native B0, Push B0, Pull B0. Adjacent old/new pairs: Push/Pull B1 and B6."
    echo "35 steps: 5 hot primer, then balanced/alpha1/alpha0.1, each with 2 settling + 8 measured steps."
    echo "Use the frozen route schedule; safe8 text, Host Engram, full recompute, NUMA, fused attention."
    echo "Both schedules keep the same adaptive policy, hidden cost switch and normal weight residency."
    echo "No profiler, route diagnostics, module timing or fixed-plan injection is installed."
    echo "DRY_RUN=1 validates and prints all commands without device use or installation."
    exit 0
fi
export RUN_ID=${RUN_ID:-dsv41-text8-comm-e2e-20261103-001}
export REPLICA_SCHEDULE_E2E=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
: "${DSV41_ROUTE_SCHEDULE:?DSV41_ROUTE_SCHEDULE must name the frozen route-window schedule}"
export TRAIN_ITERS=${TRAIN_ITERS:-35} WARMUP_STEPS=${WARMUP_STEPS:-5}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_replica_e2e_cluster.sh" "$@"
