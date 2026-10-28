#!/usr/bin/env bash
# Compare all-off, MoE-only and all-on in six clean fresh-process runs.
set -euo pipefail
ulimit -c 0
export RUN_ID=${RUN_ID:-dsv41-text8-clean-abc-cba-${MA_JOB_ID:-manual-001}}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-63100}
export CANN_INSTALL_ROOT=${CANN_INSTALL_ROOT:-/tmp/pangu-cann-${RUN_ID}}
export NUM_CARDS=${NUM_CARDS:-16} EP_SIZE=64 EDP_SHARD_SIZE=2
export NUM_HIDDEN_LAYERS=8 REPLICA_SLOTS_PER_RANK=0 MODALITY=text
export ENGRAM_STORAGE_BACKEND=host ACTIVATION_CHECKPOINT=full ATTENTION_RECOMPUTE_TIER=all
export TRAIN_ITERS=40 WARMUP_STEPS=10 CLEAN_MEGA_ABBA=1 MEGA_AB=0 PROFILE=0 MOE_TIMING=0
export EMPTY_CACHE_STEPS=1 HBM_INTERVAL=${HBM_INTERVAL:-1}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepseek_v41_cluster.sh" "$@"
