#!/usr/bin/env bash
# Isolate optional Mega kernels with native abort capture and the existing full-width workload.
set -euo pipefail
export RUN_ID=${RUN_ID:-dsv41-text8-abort-capture-${MA_JOB_ID:-manual-001}}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-64100}
export DSV41_CRASH_CAPTURE=1 CLEAN_INCLUDE_PULL=${CLEAN_INCLUDE_PULL:-0}
# Keep fatal stacks enabled; opt into verbose CANN logs only for a targeted investigation.
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3} ASCEND_SLOG_PRINT_TO_STDOUT=0
export PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1
unset ASCEND_LAUNCH_BLOCKING
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_numa_clean_profile_cluster.sh" "$@"
