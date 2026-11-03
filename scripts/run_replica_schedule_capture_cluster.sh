#!/usr/bin/env bash
# Collect matched fixed-plan graph controls before installing any new calibration.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo "Run 10 fixed-plan original/suffix graph controls on 8 nodes x 16 dies, EP64."
    echo "Each case: 6 warmup + 8 uninstrumented samples, then 3 warmup + 1 OFF/1 ON causal step."
    echo "Normal prefetch always misses; saved recompute-to-backward generation reuse is retained."
    echo "Pull fixtures are explicit nonzero diagnostics, not observed production decisions."
    echo "DRY_RUN=1 prints the stage table without framework imports, installation or device use."
    exit 0
fi
capture_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export BUNDLE_ROOT=${BUNDLE_ROOT:-$(dirname -- "${capture_source}")}
export RUN_ID=${RUN_ID:-replica-queue-fixed-ep64-20261103-001}
export PYTHONDONTWRITEBYTECODE=1
export PHASES=schedule ORDER=AB REPLICA_SCHEDULE_CAPTURE=1
export HYPER_PARALLEL_REPLICA_METADATA_REUSE=0
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_replica_ad_cluster.sh" "$@"
