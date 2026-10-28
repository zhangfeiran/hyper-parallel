#!/usr/bin/env bash
# Run the released V4.1 backbone through native, push and pull on a PanGu cluster.
set -euo pipefail

HYPER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BUNDLE_ROOT=${BUNDLE_ROOT:-$(dirname -- "${HYPER_ROOT}")}
SHARED_ROOT=${SHARED_ROOT:-$(dirname -- "${BUNDLE_ROOT}")}
DRY_RUN=${DRY_RUN:-0}
PROFILE=${PROFILE:-0}
MEGA_AB=${MEGA_AB:-0}
CLEAN_MEGA_ABBA=${CLEAN_MEGA_ABBA:-0}
CLEAN_INCLUDE_PULL=${CLEAN_INCLUDE_PULL:-0}
CLEAN_PROFILE_TAIL=${CLEAN_PROFILE_TAIL:-0}
CPU_PLACEMENT_ABBA=${CPU_PLACEMENT_ABBA:-0}
CLEAN_NUMA_TAIL=${CLEAN_NUMA_TAIL:-0}
DSV41_CRASH_CAPTURE=${DSV41_CRASH_CAPTURE:-0}
for flag in CLEAN_INCLUDE_PULL CLEAN_PROFILE_TAIL CPU_PLACEMENT_ABBA CLEAN_NUMA_TAIL DSV41_CRASH_CAPTURE; do
    [[ ${!flag} == 0 || ${!flag} == 1 ]] || { echo "ERROR: Invalid ${flag}" >&2; exit 1; }
done
[[ ${DSV41_CRASH_CAPTURE} == 0 || ${CLEAN_NUMA_TAIL} == 1 ]] || {
    echo "ERROR: Native crash capture requires the fixed NUMA workload" >&2; exit 1;
}
[[ ${CLEAN_NUMA_TAIL} == 0 || ${CLEAN_MEGA_ABBA}:${CLEAN_PROFILE_TAIL}:${CPU_PLACEMENT_ABBA} == 1:1:0 ]] || {
    echo "ERROR: Fixed NUMA tail requires clean comparisons and tail profiling" >&2; exit 1;
}
[[ ${CPU_PLACEMENT_ABBA} == 0 || ${CLEAN_MEGA_ABBA}:${CLEAN_INCLUDE_PULL}:${CLEAN_PROFILE_TAIL} == 1:0:0 ]] || {
    echo "ERROR: CPU placement requires clean comparisons without pull or profiling" >&2; exit 1;
}
[[ ${CLEAN_INCLUDE_PULL}:${CLEAN_PROFILE_TAIL} == 0:0 || ${CLEAN_MEGA_ABBA} == 1 ]] || {
    echo "ERROR: Pull repeats and tail profiling require CLEAN_MEGA_ABBA=1" >&2; exit 1;
}
[[ ${CLEAN_MEGA_ABBA} == 0 || ${CLEAN_MEGA_ABBA} == 1 ]] || { echo "ERROR: Invalid CLEAN_MEGA_ABBA" >&2; exit 1; }
[[ ${CLEAN_MEGA_ABBA} == 0 || ${MEGA_AB}:${PROFILE} == 0:0 ]] || {
    echo "ERROR: CLEAN_MEGA_ABBA requires MEGA_AB=0 and PROFILE=0" >&2; exit 1;
}
[[ ${MEGA_AB} == 0 || ${MEGA_AB} == 1 ]] || { echo "ERROR: Invalid MEGA_AB" >&2; exit 1; }
[[ ${MEGA_AB} == 0 || ${PROFILE} == 0 ]] || { echo "ERROR: MEGA_AB requires PROFILE=0" >&2; exit 1; }
[[ ${PROFILE} == 0 || ${PROFILE} == 1 ]] || { echo "ERROR: Invalid PROFILE" >&2; exit 1; }
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash source/scripts/run_deepseek_v41_cluster.sh
Run the same command on every node in one PanGu job.
NNODES: MA_NUM_HOSTS, otherwise the count of VC_WORKER_HOSTS (required).
NODE_RANK: RANK_AFTER_ACC, otherwise VC_TASK_INDEX (required for multiple nodes).
MASTER_ADDR: first VC_WORKER_HOSTS entry; MASTER_PORT_BASE defaults to 61300.
NUM_CARDS=16 EP_SIZE=64 TRAIN_ITERS=40 WARMUP_STEPS=10; dense FSDP=WORLD_SIZE.
Case order: dynamic native/push/pull, then packed native/push/pull.
Steps 1-10 warm up; steps 11-40 are measured in each independent case. MegaMoe executes actual token rows.
MODALITY=vlm or text selects the production Trainer and dataset.
MOE_TIMING=1 records forward/replay/backward current-stream spans per MoE layer.
CLEAN_MEGA_ABBA=1 runs all-off, MoE-only, all-on, all-on, MoE-only, all-off without module timing or profiler.
CLEAN_INCLUDE_PULL=1 adds mirrored MoE-only pull repeats for eight cases.
CLEAN_PROFILE_TAIL=1 measures steps 11-35, warms the profiler at 37 and captures Level0 steps 38-39.
CPU_PLACEMENT_ABBA=1 compares inherited CPUs and NUMA CPU partitions, Native/Push twice each, 100/20 steps.
CLEAN_NUMA_TAIL=1 binds CPUs before initialization, measures steps 21-100, and profiles Level0 steps 103-104.
CASE_SYNC_TIMEOUT=600 bounds the cleanup barrier after workers have exited.
HBM_INTERVAL=1 samples physical HBM and process ownership on every node.
ACTIVATION_CHECKPOINT=full. ATTENTION_RECOMPUTE_TIER=all; expert replicas are disabled.
Full checkpointing recomputes decoder layers and all attention roles.
PROFILE=1 selects packed native/push only and profiles one local rank per node.
PROFILE_START_STEP=3 PROFILE_END_STEP=5 (exclusive); profiler timings are diagnostic.
EMPTY_CACHE_STEPS=1 preserves the preceding cluster memory policy; 0 disables it.
ENGRAM_STORAGE_BACKEND=host keeps EP owner tables and SparseAdam state on CPU.
Set ENGRAM_STORAGE_BACKEND=device for the device-table comparison.
NUM_HIDDEN_LAYERS=4 VISION_NUM_HIDDEN_LAYERS=1; optional full depths are 40 and 32.
EDP_SHARD_SIZE defaults to WORLD_SIZE/EP; it is 2 for EP64 on 128 dies.
RUN_ID defaults to MA_JOB_ID (required); use a fresh ID for each run.
DRY_RUN=1 prints all six commands without installing packages or using devices.
PYTHON_BIN may select a Python 3.11 base interpreter; runtime wheels are installed
into a node-local venv. CANN_ENV_FILE and MHC_ENV_FILE may select an existing CANN
9.1.0 / Omni Ops installation; otherwise use the shared PanGu CANN installer.
NODE_LOCAL_ROOT defaults to MODELARTS_JOB_DIR/dsv41-RUN_ID (or /tmp/dsv41-RUN_ID).
Canonical random FP32 shards are generated in memory; logs/results stay shared.
The model defaults to a 4+1 depth crop with full widths/E384/Engram, BF16, MBS1, 4K,
Muon+AdamW. DSpark remains disabled in this backbone benchmark.
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo "ERROR: unknown argument: $1" >&2; exit 1; }
die() { echo "ERROR: $*" >&2; exit 1; }
worker_hosts=${VC_WORKER_HOSTS:-}
if [[ -z ${NNODES:-${MA_NUM_HOSTS:-}} && -n ${worker_hosts} ]]; then
    IFS=, read -r -a hosts <<< "${worker_hosts}"
    NNODES=${#hosts[@]}
fi
NNODES=${NNODES:-${MA_NUM_HOSTS:-}}
NODE_RANK=${NODE_RANK:-${RANK_AFTER_ACC:-${VC_TASK_INDEX:-}}}
MASTER_ADDR=${MASTER_ADDR:-${worker_hosts%%,*}}
NUM_CARDS=${NUM_CARDS:-16}
EP_SIZE=${EP_SIZE:-64}
TRAIN_ITERS=${TRAIN_ITERS:-40}
WARMUP_STEPS=${WARMUP_STEPS:-10}
MODALITY=${MODALITY:-vlm}
ATTENTION_RECOMPUTE_TIER=${ATTENTION_RECOMPUTE_TIER:-all}
EMPTY_CACHE_STEPS=${EMPTY_CACHE_STEPS:-1}
MOE_TIMING=${MOE_TIMING:-1}
CASE_SYNC_TIMEOUT=${CASE_SYNC_TIMEOUT:-600}
HBM_INTERVAL=${HBM_INTERVAL:-1}
ACTIVATION_CHECKPOINT=${ACTIVATION_CHECKPOINT:-full}
ENGRAM_STORAGE_BACKEND=${ENGRAM_STORAGE_BACKEND:-host}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-4}
VISION_NUM_HIDDEN_LAYERS=${VISION_NUM_HIDDEN_LAYERS:-1}
[[ ${ACTIVATION_CHECKPOINT} == off || ${ACTIVATION_CHECKPOINT} == full ]] || die "Invalid ACTIVATION_CHECKPOINT"
[[ ${ENGRAM_STORAGE_BACKEND} == device || ${ENGRAM_STORAGE_BACKEND} == host ]] || die "Invalid ENGRAM_STORAGE_BACKEND"
MASTER_PORT_BASE=${MASTER_PORT_BASE:-61300}
RUN_ID=${RUN_ID:-${MA_JOB_ID:-}}
[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || die "Set RUN_ID to a fresh shared job identifier"
for key in NNODES NUM_CARDS EP_SIZE TRAIN_ITERS MASTER_PORT_BASE NUM_HIDDEN_LAYERS VISION_NUM_HIDDEN_LAYERS; do
    [[ ${!key} =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer"
done
[[ ${WARMUP_STEPS} =~ ^[0-9]+$ && ${WARMUP_STEPS} -lt ${TRAIN_ITERS} ]] || die "Invalid WARMUP_STEPS"
[[ ${MODALITY} == vlm || ${MODALITY} == text ]] || die "Invalid MODALITY"
[[ ${EMPTY_CACHE_STEPS} =~ ^[0-9]+$ ]] || die "Invalid EMPTY_CACHE_STEPS"
[[ ${REPLICA_SLOTS_PER_RANK:-0} == 0 ]] || die "Expert replicas are not supported on this branch"
[[ ${MOE_TIMING} == 0 || ${MOE_TIMING} == 1 ]] || die "Invalid MOE_TIMING"
[[ ${CASE_SYNC_TIMEOUT} =~ ^[1-9][0-9]*$ ]] || die "Invalid CASE_SYNC_TIMEOUT"
[[ ${HBM_INTERVAL} =~ ^[1-9][0-9]*$ ]] || die "HBM_INTERVAL must be a positive integer"
if [[ ${NNODES} == 1 ]]; then
    NODE_RANK=${NODE_RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
[[ ${NODE_RANK} =~ ^[0-9]+$ && ${NODE_RANK} -lt ${NNODES} ]] || die "Invalid NODE_RANK"
[[ -n ${MASTER_ADDR} ]] || die "MASTER_ADDR or VC_WORKER_HOSTS is required"
WORLD_SIZE=$((NNODES * NUM_CARDS))
(( EP_SIZE >= 2 && WORLD_SIZE % EP_SIZE == 0 && 384 % EP_SIZE == 0 )) || die "EP must divide WORLD_SIZE and 384"
EDP_SHARD_SIZE=${EDP_SHARD_SIZE:-$((WORLD_SIZE / EP_SIZE))}
[[ ${EDP_SHARD_SIZE} =~ ^[1-9][0-9]*$ ]] || die "Invalid EDP_SHARD_SIZE"
(( WORLD_SIZE / EP_SIZE % EDP_SHARD_SIZE == 0 )) || die "EDP_SHARD_SIZE must divide WORLD_SIZE/EP"
(( NUM_HIDDEN_LAYERS >= 4 && NUM_HIDDEN_LAYERS <= 40 )) || die "Language depth must be in [4, 40]"
(( VISION_NUM_HIDDEN_LAYERS <= 32 )) || die "Vision depth must be in [1, 32]"
(( MASTER_PORT_BASE + 223 < 65536 )) || die "MASTER_PORT_BASE is too large"
RUN_ROOT=${RUN_ROOT:-${SHARED_ROOT}/dsv41-runs/${RUN_ID}}
NODE_LOCAL_ROOT=${NODE_LOCAL_ROOT:-${MODELARTS_JOB_DIR:-/tmp}/dsv41-${RUN_ID}}
case ${NODE_LOCAL_ROOT} in /work/*|/home/ma-user/work/*) die "NODE_LOCAL_ROOT must not use shared work storage";; esac
NODE_OUTPUT=${RUN_ROOT}/nodes/node_${NODE_RANK}
RANDOM_INIT_SEED=${RANDOM_INIT_SEED:-42}
MODEL_DIR=${MODEL_DIR:-${BUNDLE_ROOT}/assets/model}
if [[ ${MODALITY} == text ]]; then
    DATA=${DATA:-${BUNDLE_ROOT}/assets/data/train_text.jsonl}
else
    DATA=${DATA:-${BUNDLE_ROOT}/assets/data/train.jsonl}
fi
ENGRAM=${ENGRAM:-${BUNDLE_ROOT}/assets/engram_full.json}
NATIVE_LIB=${NATIVE_LIB:-${HYPER_ROOT}/build/native/payload/hyper_parallel/core/multicore/lib}
if [[ ${CLEAN_MEGA_ABBA} == 1 ]]; then
    (( WORLD_SIZE == 128 && EP_SIZE == 64 && EDP_SHARD_SIZE == 2 )) || die "Clean ABC-CBA requires 128 dies, EP64, expert FSDP2"
    if [[ ${CPU_PLACEMENT_ABBA} == 1 ]]; then
        (( TRAIN_ITERS == 100 && WARMUP_STEPS == 20 && NUM_HIDDEN_LAYERS == 8 )) || die "CPU placement requires 8 layers and 100/20 steps"
    elif [[ ${CLEAN_NUMA_TAIL} == 1 ]]; then
        (( TRAIN_ITERS == 105 && WARMUP_STEPS == 20 && NUM_HIDDEN_LAYERS == 8 )) || die "Fixed NUMA tail requires 8 layers and 105/20 steps"
    else
        (( TRAIN_ITERS == 40 && WARMUP_STEPS == 10 && NUM_HIDDEN_LAYERS == 8 )) || die "Clean ABC-CBA requires 8 layers and 40/10 steps"
    fi
    [[ ${MODALITY}:${MOE_TIMING}:${ENGRAM_STORAGE_BACKEND} == text:0:host ]] || die "Clean ABC-CBA requires text, no module timing and Host Engram"
fi

if [[ -z ${PYTHON_BIN:-} ]]; then
    for candidate in /home/ma-user/anaconda3/envs/PyTorch-{2.9.0,2.6.0}/bin/python; do
        if [[ -x ${candidate} ]]; then PYTHON_BIN=${candidate}; break; fi
    done
fi
BASE_PYTHON=${PYTHON_BIN:-python3}
PYTHON_BIN=${NODE_LOCAL_ROOT}/venv/bin/python
for path in "${MODEL_DIR}/config.json" "${DATA}" "${ENGRAM}" "${NATIVE_LIB}/set_env.bash"; do
    [[ -s ${path} ]] || die "Missing bundle input: ${path}"
done
data_records=$(wc -l < "${DATA}")
(( data_records >= WORLD_SIZE * TRAIN_ITERS )) || die "Dataset is too short for the requested training horizon"

if [[ ${DRY_RUN} != 1 ]]; then
    mkdir -p "${NODE_OUTPUT}" "${NODE_LOCAL_ROOT}"
    mkdir "${NODE_OUTPUT}/started" || die "Run already started for this node; use a fresh RUN_ID"
    exec > >(tee "${NODE_OUTPUT}/launcher.log") 2>&1
    (cd "${BUNDLE_ROOT}" && sha256sum --quiet -c SHA256SUMS)
    if [[ ${DSV41_CRASH_CAPTURE} == 1 ]]; then
        export DSV41_NATIVE_CAPTURE_LIB=${BUNDLE_ROOT}/tools/native/libdsv41_capture.so
        PYTHONPATH=${HYPER_ROOT} "${BASE_PYTHON}" -m examples.training_demo.deepseek_v41.deepseek_v41_capture_worker \
            --preflight --output "${NODE_OUTPUT}/capture-preflight"
    fi
    if [[ -z ${CANN_ENV_FILE:-} ]]; then
        export CANN_PACKAGE_ROOT=${CANN_PACKAGE_ROOT:-${SHARED_ROOT}/cann-9.1.0-packages}
        export OMNI_OPS_PACKAGE_ROOT=${OMNI_OPS_PACKAGE_ROOT:-${SHARED_ROOT}/omni-ops-cann910-packages}
        export CANN_INSTALL_ROOT=${CANN_INSTALL_ROOT:-${MODELARTS_JOB_DIR:-/tmp}/pangu-cann-9.1.0}
        bash "${BUNDLE_ROOT}/scripts/install_cluster_cann.sh"
        CANN_ENV_FILE=${CANN_INSTALL_ROOT}/cann/set_env.sh
    fi
    MHC_ENV_FILE=${MHC_ENV_FILE:-$(dirname -- "${CANN_ENV_FILE}")/opp/vendors/omni_training_custom_transformer/bin/set_env.bash}
    [[ -f ${CANN_ENV_FILE} && -f ${MHC_ENV_FILE} ]] || die "CANN / Omni Ops activation scripts are missing"
    unset RANK_TABLE_FILE RANK_ID RANK_SIZE PYTHONPATH LD_LIBRARY_PATH
    unset ASCEND_CUSTOM_OPP_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH TOOLCHAIN_HOME
    set +u
    source "${CANN_ENV_FILE}"
    source "${MHC_ENV_FILE}"
    set -u
    "${BASE_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
    cp -a "${HYPER_ROOT}" "${NODE_LOCAL_ROOT}/source"
    HYPER_ROOT=${NODE_LOCAL_ROOT}/source
    NATIVE_LIB=${HYPER_ROOT}/build/native/payload/hyper_parallel/core/multicore/lib
    set +u
    source "${NATIVE_LIB}/set_env.bash"
    if [[ ${DSV41_CRASH_CAPTURE} == 1 ]]; then
        FUSED_ATTN_ENV_FILE=${FUSED_ATTN_ENV_FILE:-${BUNDLE_ROOT}/fused_ops/vendors/dsv41_fused_transformer/bin/set_env.bash}
        [[ -f ${FUSED_ATTN_ENV_FILE} ]] || die "Missing fused attention operator environment: ${FUSED_ATTN_ENV_FILE}"
        source "${FUSED_ATTN_ENV_FILE}"
        fused_vendor=$(cd -- "$(dirname -- "${FUSED_ATTN_ENV_FILE}")/.." && pwd -P)
        # A host-only package can resolve an incompatible built-in device binary with the same operator name.
        for op in lightning_indexer_v2 sparse_flash_mla sparse_flash_mla_grad sparse_lightning_indexer_kl_loss_grad; do
            [[ -s ${fused_vendor}/op_impl/ai_core/tbe/kernel/config/ascend910_93/${op}.json ]] || \
                die "Missing matched fused device kernels for ${op}; install the precompiled package"
        done
    fi
    set -u
    "${BASE_PYTHON}" -m venv "${NODE_LOCAL_ROOT}/venv"
    # Uninstall seeded packages so obsolete distribution entry points cannot survive.
    "${PYTHON_BIN}" -m pip install --no-index --no-deps --force-reinstall "${BUNDLE_ROOT}"/wheels/*.whl
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

versions = [
    distribution.version
    for distribution in metadata.distributions()
    if distribution.metadata.get("Name", "").lower() == "setuptools"
]
if versions != ["83.0.0"]:
    raise RuntimeError(f"Expected exactly one setuptools 83.0.0 distribution, found {versions}; use a fresh RUN_ID")
print("Packaging preflight passed: exactly one setuptools 83.0.0 distribution", flush=True)
PY
    "${PYTHON_BIN}" -m pip install --use-pep517 --no-index --no-deps --no-build-isolation -e "${HYPER_ROOT}"
    # Prefer the pinned wheel over the extension bundled on the CANN Python path.
    venv_site=$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    export PYTHONPATH="${venv_site}${PYTHONPATH:+:${PYTHONPATH}}"
    export TORCH_EXTENSIONS_DIR="${NODE_LOCAL_ROOT}/torch-extensions"
    export PATH="${NODE_LOCAL_ROOT}/venv/bin:${PATH}"
    export HYPER_PARALLEL_PLATFORM=torch PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
    export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_CARDS - 1)))}
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200} HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
    export HYPER_ROOT
    cd "${HYPER_ROOT}"
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path
import hyper_parallel
import torch
import torch_npu
import torchvision
import omni_training_custom_ops
from transformers.models.deepseek_v4 import DeepseekV4Config
from examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm import main
from hyper_parallel.core.multicore.torch.ops import _load_native
expected = {'torch': '2.9.1', 'torch-npu': '2.9.1', 'transformers': '5.13.1', 'torchvision': '0.24.1'}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise RuntimeError(f'{package}: expected {version}, got {actual}')
root = Path(os.environ['HYPER_ROOT']).resolve()
if Path(hyper_parallel.__file__).resolve() != root / 'hyper_parallel/__init__.py':
    raise RuntimeError(f'Wrong Hyper import: {hyper_parallel.__file__}')
cann = Path(os.environ['ASCEND_HOME_PATH']) / 'share/info/runtime/version.info'
if 'Version=9.1.0' not in cann.read_text().splitlines():
    raise RuntimeError(f'Expected CANN 9.1.0: {cann}')
_load_native()
if os.environ.get('DSV41_CRASH_CAPTURE') == '1':
    import cann_ops_transformer
    from hyper_parallel.components.functional.lightning_indexer import _load_indexer_op as load_indexer
    from hyper_parallel.components.functional.sparse_flash_mla import _load_attention_ops as load_attention
    from hyper_parallel.components.functional.sparse_lightning_indexer_kl_loss_grad import _load_kl_ops as load_kl
    extension = Path(cann_ops_transformer.__file__).resolve()
    venv_site = Path(os.environ['PYTHONPATH'].split(os.pathsep)[0]).resolve()
    if not extension.is_relative_to(venv_site):
        raise RuntimeError(f'Expected the pinned fused attention extension in {venv_site}, got {extension}')
    load_indexer()
    load_attention()
    load_kl()
if os.environ.get('CLEAN_MEGA_ABBA') == '1':
    from hyper_parallel.core.multicore.torch.mega_gate_ops import _load_native as load_mega_gate
    load_mega_gate()
print(f'Runtime import and native registration passed: {hyper_parallel.__file__}', flush=True)
PY
fi

SHMEM_HOST=${SHMEM_HOST:-${MASTER_ADDR}}
if [[ ${DRY_RUN} != 1 ]]; then
    SHMEM_HOST=$(getent ahostsv4 "${SHMEM_HOST}" | awk 'NR==1 {print $1}')
    [[ -n ${SHMEM_HOST} ]] || die "Cannot resolve SHMEM_HOST to IPv4"
fi
export MASTER_ADDR NODE_RANK
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
if [[ ${CLEAN_MEGA_ABBA} == 1 ]]; then
    export HYPER_ROOT NNODES NUM_CARDS EP_SIZE EDP_SHARD_SIZE MASTER_PORT_BASE
    export RUN_ROOT MODEL_DIR DATA ENGRAM NATIVE_LIB SHMEM_HOST
    clean_options=()
    if [[ ${CLEAN_INCLUDE_PULL} == 1 ]]; then clean_options+=(--include-pull); fi
    if [[ ${CLEAN_PROFILE_TAIL} == 1 ]]; then clean_options+=(--tail-profile); fi
    if [[ ${CLEAN_NUMA_TAIL} == 1 ]]; then clean_options+=(--numa-tail); fi
    if [[ ${DRY_RUN} == 1 ]]; then
        clean_options+=(--dry-run)
        export PYTHONPATH=${HYPER_ROOT}
        PYTHON_BIN=${BASE_PYTHON}
    fi
    clean_module=examples.training_demo.deepseek_v41.deepseek_v41_clean_abba
    if [[ ${CPU_PLACEMENT_ABBA} == 1 ]]; then
        clean_module=examples.training_demo.deepseek_v41.deepseek_v41_cpu_abba
    fi
    if [[ ${DSV41_CRASH_CAPTURE} == 1 ]]; then
        clean_module=examples.training_demo.deepseek_v41.deepseek_v41_crash_capture
        clean_options=()
        if [[ ${DRY_RUN} == 1 ]]; then clean_options+=(--dry-run); fi
        export DSV41_NATIVE_CAPTURE_LIB=${BUNDLE_ROOT}/tools/native/libdsv41_capture.so
    fi
    exec "${PYTHON_BIN}" -m "${clean_module}" "${clean_options[@]}"
fi
publish_summary() {
    local summary_status=0
    local summary_module=examples.training_demo.deepseek_v41.summarize_deepseek_v41_cluster
    local summary_options=(--matrix)
    if [[ ${PROFILE} == 1 ]]; then
        summary_module=examples.training_demo.deepseek_v41.summarize_deepseek_v41_profile
        summary_options=()
    fi
    if [[ ${MEGA_AB} == 1 ]]; then
        summary_module=examples.training_demo.deepseek_v41.summarize_deepseek_v41_mega_ab
        summary_options=()
    fi
    "${PYTHON_BIN}" -m "${summary_module}" \
        --run-root "${RUN_ROOT}" --nodes "${NNODES}" --cards-per-node "${NUM_CARDS}" \
        --steps "${TRAIN_ITERS}" --warmup-steps "${WARMUP_STEPS}" "${summary_options[@]}" \
        > "${RUN_ROOT}/summary.log" 2>&1 || summary_status=$?
    printf '%s\n' "${summary_status}" > "${NODE_OUTPUT}/summary_exit_status.txt"
    return "${summary_status}"
}

indices=(0 1 2 3 4 5)
benchmark_module=examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm
if [[ ${MEGA_AB} == 1 ]]; then indices=(3 4 5); fi
if [[ ${PROFILE} == 1 ]]; then
    indices=(3 4)
    benchmark_module=examples.training_demo.deepseek_v41.benchmark_deepseek_v41_profile
    export PROFILE_START_STEP=${PROFILE_START_STEP:-3} PROFILE_END_STEP=${PROFILE_END_STEP:-5}
fi
for index in "${indices[@]}"; do
    labels=(native push pull)
    if [[ ${MEGA_AB} == 1 ]]; then labels=(native moe_only push); fi
    label=${labels[index % 3]}
    packing=dynamic
    (( index < 3 )) || packing=packed
    backend=owner_ep
    mode=push
    if [[ ${label} == push || ${label} == pull ]]; then backend=megamoe; mode=${label}; fi
    if [[ ${label} == moe_only ]]; then backend=megamoe; mode=push; fi
    export HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT="tcp://${SHMEM_HOST}:$((MASTER_PORT_BASE + 100 + index))"
    export HCCL_IF_BASE_PORT=$((MASTER_PORT_BASE + 200 + index * 4))
    command=("${PYTHON_BIN}" -m torch.distributed.run --nnodes="${NNODES}" --node-rank="${NODE_RANK}"
        --nproc-per-node="${NUM_CARDS}" --master-addr="${MASTER_ADDR}" --master-port="$((MASTER_PORT_BASE + index))"
        --rdzv-conf=timeout=7200
        -m "${benchmark_module}"
        --full-model --experts 384 --ep-size "${EP_SIZE}" --steps "${TRAIN_ITERS}"
        --warmup-steps "${WARMUP_STEPS}" --packing-mode "${packing}"
        --num-hidden-layers "${NUM_HIDDEN_LAYERS}" --vision-num-hidden-layers "${VISION_NUM_HIDDEN_LAYERS}"
        --modality "${MODALITY}" --attention-recompute-tier "${ATTENTION_RECOMPUTE_TIER}"
        --empty-cache-steps "${EMPTY_CACHE_STEPS}"
        --edp-shard-size "${EDP_SHARD_SIZE}"
        --activation-checkpoint "${ACTIVATION_CHECKPOINT}"
        --engram-storage-backend "${ENGRAM_STORAGE_BACKEND}"
        --model-dir "${MODEL_DIR}" --engram-assets "${ENGRAM}" --data "${DATA}"
        --random-init-seed "${RANDOM_INIT_SEED}" --native-lib "${NATIVE_LIB}" --output "${RUN_ROOT}/${packing}/${label}"
        --backend "${backend}" --dispatch-mode "${mode}")
    [[ ${MOE_TIMING} == 0 ]] || command+=(--moe-timing)
    if [[ ${MEGA_AB} == 1 ]]; then
        mega_kernels=off
        [[ ${label} == native ]] || mega_kernels=on
        [[ ${label} != moe_only ]] || mega_kernels=moe
        command+=(--mega-kernels "${mega_kernels}")
    fi
    echo "Case=${packing}/${label} world=${WORLD_SIZE} EP=${EP_SIZE} FSDP=${WORLD_SIZE} EDP_SHARD=${EDP_SHARD_SIZE}"
    echo "Depth=${NUM_HIDDEN_LAYERS}+${VISION_NUM_HIDDEN_LAYERS} E384 TP1 CP1 MBS1 maxS4096 B=0 modality=${MODALITY}"
    echo "Engram=${ENGRAM_STORAGE_BACKEND} activation_checkpoint=${ACTIVATION_CHECKPOINT} decoder_attention_tier=${ATTENTION_RECOMPUTE_TIER}"
    echo "Warmup=${WARMUP_STEPS} measured=$((TRAIN_ITERS - WARMUP_STEPS)) actual_tokens=true"
    printf '%q ' "${command[@]}"; printf '\n'
    [[ ${DRY_RUN} == 1 ]] && continue
    case_dir=${RUN_ROOT}/${packing}/nodes/node_${NODE_RANK}/${label}
    mkdir -p "${case_dir}"
    printf '%q ' "${command[@]}" > "${case_dir}/command.txt"
    printf '\n' >> "${case_dir}/command.txt"
    controller_status=0
    "${PYTHON_BIN}" -m examples.training_demo.deepseek_v41.coordinate_deepseek_v41_cluster \
        --run-root "${RUN_ROOT}" --packing-mode "${packing}" --label "${label}" \
        --nodes "${NNODES}" --node-rank "${NODE_RANK}" --cards-per-node "${NUM_CARDS}" \
        --interval "${HBM_INTERVAL}" --sync-timeout "${CASE_SYNC_TIMEOUT}" -- "${command[@]}" \
        > "${case_dir}/controller.log" 2>&1 || controller_status=$?
    if [[ ${NODE_RANK} == 0 ]]; then
        publish_summary || echo "ERROR: Summary publication failed; see ${RUN_ROOT}/summary.log" >&2
    fi
    (( controller_status == 0 )) || die "Case cleanup barrier failed: ${packing}/${label}; see ${case_dir}/controller.log"
done

if [[ ${DRY_RUN} != 1 && ${NODE_RANK} == 0 ]]; then
    publish_summary
    "${PYTHON_BIN}" - "${RUN_ROOT}/summary.json" <<'PY_FINAL'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
if not report['complete']:
    raise SystemExit('Some cases failed; completed comparisons remain in summary.json')
PY_FINAL
fi
