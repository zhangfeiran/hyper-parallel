#!/usr/bin/env bash
# Include mirrored MegaMoe Pull repeats with fixed NUMA placement and tail profiling.
set -euo pipefail
export RUN_ID=${RUN_ID:-dsv41-text8-numa-clean8-push-pull-tail-l0-${MA_JOB_ID:-manual-001}}
export CLEAN_INCLUDE_PULL=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_numa_clean_profile_cluster.sh" "$@"
