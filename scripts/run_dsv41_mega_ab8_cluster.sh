#!/usr/bin/env bash
# Run all-off, MegaMoe-only, then all-on at matched eight-layer scope.
set -euo pipefail
export RUN_ID=${RUN_ID:-dsv41-text8-three-mega-ab-${MA_JOB_ID:-manual-001}}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-63700}
export CANN_INSTALL_ROOT=${CANN_INSTALL_ROOT:-/tmp/pangu-cann-${RUN_ID}}
export NUM_CARDS=${NUM_CARDS:-16} EP_SIZE=${EP_SIZE:-64}
export NUM_HIDDEN_LAYERS=8 REPLICA_SLOTS_PER_RANK=0 MODALITY=text
export ENGRAM_STORAGE_BACKEND=host ACTIVATION_CHECKPOINT=full ATTENTION_RECOMPUTE_TIER=all
export TRAIN_ITERS=${TRAIN_ITERS:-40} WARMUP_STEPS=${WARMUP_STEPS:-10}
export MEGA_AB=1 PROFILE=0 MOE_TIMING=1 EMPTY_CACHE_STEPS=1
export HBM_INTERVAL=${HBM_INTERVAL:-1} PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepseek_v41_cluster.sh" "$@"
