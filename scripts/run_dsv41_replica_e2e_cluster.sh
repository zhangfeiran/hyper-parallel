#!/usr/bin/env bash
# Compare safe8 DSV native/Push/Pull with adaptive B0/B1/B6 caps.
set -euo pipefail
ulimit -c 0
if [[ ${1:-} == --help ]]; then
    echo "Run on 8 nodes x 16 dies; NNODES/NODE_RANK/MASTER_ADDR are detected from scheduler variables."
    echo "Default stages: native_b0,push_b0,pull_b0,native_b1,push_b1,pull_b1,native_b6,push_b6,pull_b6."
    echo "STAGES=push_b0,push_b1,pull_b0,pull_b1 selects a subset in that order."
    echo "TRAIN_ITERS=20 WARMUP_STEPS=5; safe8 text, EP64, expert FSDP2, NUMA, full replay, Host Engram."
    echo "REPLICA_TRANSPORT=shmem_signal_kernel_gradient_bands by default for MegaMoe; native always uses P2P."
    echo "Profiler, module events, route diagnostics and batch hooks are disabled; B is an adaptive cap."
    echo "RUN_ID must be fresh. DRY_RUN=1 prints the commands without device use or installation."
    exit 0
fi
export RUN_ID=${RUN_ID:-dsv41-text8-replica-e2e-manual-001}
export MASTER_PORT_BASE=${MASTER_PORT_BASE:-28600}
export CANN_INSTALL_ROOT=${CANN_INSTALL_ROOT:-/tmp/pangu-cann-${RUN_ID}}
export NUM_CARDS=16 EP_SIZE=64 EDP_SHARD_SIZE=2
export NUM_HIDDEN_LAYERS=8 REPLICA_SLOTS_PER_RANK=0 MODALITY=text
export ENGRAM_STORAGE_BACKEND=host ACTIVATION_CHECKPOINT=full ATTENTION_RECOMPUTE_TIER=all
export TRAIN_ITERS=${TRAIN_ITERS:-20} WARMUP_STEPS=${WARMUP_STEPS:-5}
export REPLICA_E2E=1 CLEAN_MEGA_ABBA=0 CLEAN_INCLUDE_PULL=0 CLEAN_PROFILE_TAIL=0
export CPU_PLACEMENT_ABBA=0 CLEAN_NUMA_TAIL=0 DSV41_CRASH_CAPTURE=0 MEGA_AB=0 PROFILE=0 MOE_TIMING=0
export REPLICA_TRANSPORT=${REPLICA_TRANSPORT:-shmem_signal_kernel_gradient_bands}
export HYPER_PARALLEL_REPLICA_HOST_NATIVE=1 HYPER_PARALLEL_REPLICA_COST_MODEL=1
export HYPER_PARALLEL_REPLICA_METADATA_REUSE=0
export EMPTY_CACHE_STEPS=1 HBM_INTERVAL=${HBM_INTERVAL:-1}
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_deepseek_v41_cluster.sh" "$@"
