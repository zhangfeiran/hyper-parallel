#!/usr/bin/env bash
# Compare native, MoE push, MoE pull and all-on with Level0 traces after clean steps.
set -euo pipefail
export RUN_ID=${RUN_ID:-dsv41-text8-tail-l0-push-pull-${MA_JOB_ID:-manual-001}}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-63300}
export CLEAN_INCLUDE_PULL=1 CLEAN_PROFILE_TAIL=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_dsv41_clean_abba_cluster.sh" "$@"
