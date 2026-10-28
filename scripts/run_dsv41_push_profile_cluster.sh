#!/usr/bin/env bash
# Profile the same native/push text workload using the common cluster launcher.
set -euo pipefail
export PROFILE=1 MODALITY=${MODALITY:-text}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepseek_v41_cluster.sh" "$@"
