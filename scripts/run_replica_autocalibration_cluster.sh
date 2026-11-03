#!/usr/bin/env bash
# Reuse the caller's activated cluster runtime; the coordinator imports no framework.
set -euo pipefail
replica_source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${replica_source_root}"
replica_python=${PYTHON_BIN:-python3}
if [[ ${DRY_RUN:-0} == 1 ]]; then
    exec "${replica_python}" scripts/replica_autocalibration.py "$@" --dry-run
fi
exec "${replica_python}" scripts/replica_autocalibration.py "$@"
