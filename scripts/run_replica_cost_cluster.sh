#!/usr/bin/env bash
# Bootstrap isolated historical-route MoE cost-model controls.
set -euo pipefail
HYPER_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BUNDLE_ROOT=${BUNDLE_ROOT:-$(dirname -- "${HYPER_ROOT}")}
SHARED_ROOT=${SHARED_ROOT:-$(dirname -- "${BUNDLE_ROOT}")}
DRY_RUN=${DRY_RUN:-0}
die() { echo "ERROR: $*" >&2; exit 1; }
if [[ ${1:-} == --help ]]; then
    echo "Run on 8 nodes x 16 dies. Detect NNODES/NODE_RANK/MASTER_ADDR from scheduler variables."
    echo "Collect B0 and previous/current bounded-candidate controls, B1/B2/B6, warmup12 and samples12."
    echo "No scheduler job ID, training assets or CANN profiler is required."
    exit 0
fi
[[ $# == 0 ]] || die "Unknown argument: $1"
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
MASTER_PORT_BASE=${MASTER_PORT_BASE:-63600}
RUN_ID=${RUN_ID:-replica-candidates-ep64-20261028-001}
[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid RUN_ID"
for key in NNODES NUM_CARDS EP_SIZE MASTER_PORT_BASE; do
    [[ ${!key} =~ ^[1-9][0-9]*$ ]] || die "${key} must be a positive integer"
done
[[ ${NODE_RANK} =~ ^[0-9]+$ && ${NODE_RANK} -lt ${NNODES} ]] || die "Invalid NODE_RANK"
[[ -n ${MASTER_ADDR} ]] || die "MASTER_ADDR or VC_WORKER_HOSTS is required"
(( NNODES == 8 && NUM_CARDS == 16 && EP_SIZE == 64 )) || die "This matrix requires 8 nodes x 16 dies and EP64"
(( MASTER_PORT_BASE + 447 < 65536 )) || die "MASTER_PORT_BASE is too large"
RUN_ROOT=${RUN_ROOT:-${SHARED_ROOT}/dsv41-runs/${RUN_ID}}
NODE_LOCAL_ROOT=${NODE_LOCAL_ROOT:-${MODELARTS_JOB_DIR:-/tmp}/replica-${RUN_ID}}
case ${NODE_LOCAL_ROOT} in /work/*|/home/ma-user/work/*) die "NODE_LOCAL_ROOT must not use shared work storage";; esac
NODE_OUTPUT=${RUN_ROOT}/nodes/node_${NODE_RANK}
if [[ -z ${PYTHON_BIN:-} ]]; then
    for candidate in /home/ma-user/anaconda3/envs/PyTorch-{2.9.0,2.6.0}/bin/python; do
        if [[ -x ${candidate} ]]; then PYTHON_BIN=${candidate}; break; fi
    done
fi
BASE_PYTHON=${PYTHON_BIN:-python3}
PYTHON_BIN=${NODE_LOCAL_ROOT}/venv/bin/python
if [[ ${DRY_RUN} != 1 ]]; then
    mkdir -p "${NODE_OUTPUT}" "${NODE_LOCAL_ROOT}"
    mkdir "${NODE_OUTPUT}/started" || die "Run already started for this node; use a fresh RUN_ID"
    exec > >(tee "${NODE_OUTPUT}/launcher.log") 2>&1
    monitor_pid=
    finish() {
        status=$?
        trap - EXIT
        if [[ -n ${monitor_pid} ]]; then
            kill "${monitor_pid}" 2>/dev/null || true
            wait "${monitor_pid}" 2>/dev/null || true
        fi
        npu-smi info > "${NODE_OUTPUT}/npu-terminal.txt" 2>&1 || true
        printf '%s\n' "${status}" > "${NODE_OUTPUT}/exit_status.txt"
        exit "${status}"
    }
    trap finish EXIT
    (cd "${BUNDLE_ROOT}" && sha256sum --quiet -c SHA256SUMS)
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
    cp -al "${HYPER_ROOT}" "${NODE_LOCAL_ROOT}/source-previous"
    for module in planner cost; do
        relative=hyper_parallel/core/expert_parallel/hot_replica/${module}.py
        cp --remove-destination "${BUNDLE_ROOT}/controls/previous/${relative}" \
            "${NODE_LOCAL_ROOT}/source-previous/${relative}"
    done
    # Reset to the CANN/Omni environment before activating each source-specific payload.
    replica_cann_library_path=${LD_LIBRARY_PATH:-}
    replica_cann_opp_path=${ASCEND_CUSTOM_OPP_PATH:-}
    NATIVE_LIB=${HYPER_ROOT}/build/native/payload/hyper_parallel/core/multicore/lib
    set +u
    source "${NATIVE_LIB}/set_env.bash"
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
print(f'Runtime import and native registration passed: {hyper_parallel.__file__}', flush=True)
PY
fi

SHMEM_HOST=${SHMEM_HOST:-${MASTER_ADDR}}
if [[ ${DRY_RUN} != 1 ]]; then
    SHMEM_HOST=$(getent ahostsv4 "${SHMEM_HOST}" | awk 'NR==1 {print $1}')
    [[ -n ${SHMEM_HOST} ]] || die "Cannot resolve SHMEM_HOST to IPv4"
    npu-smi info > "${NODE_OUTPUT}/npu-before.txt"
    (
        while true; do
            date -u +%FT%TZ
            npu-smi info
            ps -eo pid,ppid,lstart,args
            sleep 5
        done
    ) > "${NODE_OUTPUT}/npu-process-monitor.log" 2>&1 &
    monitor_pid=$!
fi
export MASTER_ADDR NODE_RANK
export HYPER_PARALLEL_REPLICA_COST_MODEL=1 HP_EP_STATIC_SPLITS=0
export DSV41_CPU_PLACEMENT=numa DSV41_PLACEMENT_DRY_RUN=0
export PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1
if [[ ${DRY_RUN} == 1 ]]; then PYTHON_BIN=${BASE_PYTHON}; fi
cases=(
    01_native_b0_a:native:0:fanout:1
    02_push_b0:push:0:fanout:1
    03_push_b1_previous:push:1:previous:1
    04_push_b1_load:push:1:fanout:0
    05_push_b1_current:push:1:fanout:1
    06_push_b2_current:push:2:fanout:1
    07_push_b6_previous:push:6:previous:1
    08_push_b6_current:push:6:fanout:1
    09_pull_b0:pull:0:fanout:1
    10_pull_b1_previous:pull:1:previous:1
    11_pull_b1_load:pull:1:fanout:0
    12_pull_b1_current:pull:1:fanout:1
    13_pull_b2_current:pull:2:fanout:1
    14_pull_b6_previous:pull:6:previous:1
    15_pull_b6_current:pull:6:fanout:1
    16_native_b0_b:native:0:fanout:1
)
index=0
for entry in "${cases[@]}"; do
    IFS=: read -r label backend budget variant cost_enabled <<< "${entry}"
    port=$((MASTER_PORT_BASE + index * 16))
    source_root=${NODE_LOCAL_ROOT}/source
    if [[ ${variant} != fanout ]]; then source_root=${source_root}-${variant}; fi
    if [[ ${DRY_RUN} != 1 ]]; then
        export LD_LIBRARY_PATH="${replica_cann_library_path}"
        export ASCEND_CUSTOM_OPP_PATH="${replica_cann_opp_path}"
        NATIVE_LIB=${source_root}/build/native/payload/hyper_parallel/core/multicore/lib
        set +u
        source "${NATIVE_LIB}/set_env.bash"
        set -u
        "${PYTHON_BIN}" -m pip install --use-pep517 --no-index --no-deps --no-build-isolation -e "${source_root}"
        cd "${source_root}"
        # Validate the editable and native payload in a fresh process before launching ranks.
        "${PYTHON_BIN}" - "${source_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import hyper_parallel
import torch
from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.core.multicore.torch.ops import _load_native

root = Path(sys.argv[1]).resolve()
if Path(hyper_parallel.__file__).resolve() != root / 'hyper_parallel/__init__.py':
    raise RuntimeError(f'Wrong Hyper import for {root}: {hyper_parallel.__file__}')
vendor, adapter = get_multicore_paths()
native = root / 'build/native/payload/hyper_parallel/core/multicore/lib'
if vendor != native / 'vendors/hyper_parallel_multicore_nn' or not adapter.is_relative_to(native):
    raise RuntimeError(f'Wrong native payload for {root}: vendor={vendor}, adapter={adapter}')
_load_native()
if torch.npu.is_initialized():
    raise RuntimeError('Runtime preflight unexpectedly initialized NPU')
print(json.dumps(dict(source_root=str(root), native_vendor=str(vendor), native_adapter=str(adapter),
                      native_adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                      registered=True, npu_initialized=False)), flush=True)
PY
    fi
    command=(env HYPER_PARALLEL_REPLICA_COST_MODEL=${cost_enabled}
        HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT=tcp://${SHMEM_HOST}:$((port + 1))
        HCCL_IF_BASE_PORT=$((port + 2)) "${PYTHON_BIN}" -m torch.distributed.run
        --nnodes "${NNODES}" --node-rank "${NODE_RANK}" --nproc-per-node "${NUM_CARDS}"
        --master-addr "${MASTER_ADDR}" --master-port "${port}" --rdzv-conf=timeout=7200
        -m tests.torch.expert_parallel._replica_cost_worker --ep-size "${EP_SIZE}"
        --hidden 5120 --intermediate 2304 --tokens 4096 --top-k 6
        --backend "${backend}" --budget "${budget}" --replica-minimum-load-reduction 0.0 --iterations 12 --warmup 12
        --route-file "${BUNDLE_ROOT}/routes.json" --result-dir "${RUN_ROOT}/cases/${label}")
    if [[ ${DRY_RUN} == 1 ]]; then
        "${BASE_PYTHON}" -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${command[@]}"
    else
        echo "Starting replica cost control: ${label} source=${source_root}"
        "${command[@]}" > "${NODE_OUTPUT}/${label}.log" 2>&1
        echo "Finished replica cost control: ${label}"
    fi
    index=$((index + 1))
done
