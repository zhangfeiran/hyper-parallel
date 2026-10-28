#!/usr/bin/env bash
# Compare inherited CPU scheduling and NUMA CPU partitions with identical training.
set -euo pipefail
ulimit -c 0
export RUN_ID=${RUN_ID:-dsv41-text8-cpu-placement-${MA_JOB_ID:-manual-001}}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-63500}
export CANN_INSTALL_ROOT=${CANN_INSTALL_ROOT:-/tmp/pangu-cann-${RUN_ID}}
export NUM_CARDS=${NUM_CARDS:-16} EP_SIZE=64 EDP_SHARD_SIZE=2
export NUM_HIDDEN_LAYERS=8 REPLICA_SLOTS_PER_RANK=0 MODALITY=text
export ENGRAM_STORAGE_BACKEND=host ACTIVATION_CHECKPOINT=full ATTENTION_RECOMPUTE_TIER=all
export TRAIN_ITERS=100 WARMUP_STEPS=20 CPU_PLACEMENT_ABBA=1 CLEAN_MEGA_ABBA=1
export CLEAN_INCLUDE_PULL=0 CLEAN_PROFILE_TAIL=0 MEGA_AB=0 PROFILE=0 MOE_TIMING=0
export EMPTY_CACHE_STEPS=1 HBM_INTERVAL=${HBM_INTERVAL:-1}
export PYTHONDONTWRITEBYTECODE=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepseek_v41_cluster.sh" "$@"
