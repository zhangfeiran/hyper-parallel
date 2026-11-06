#!/usr/bin/env bash
set -eo pipefail
addon_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
customer_python=python
customer_repo=
cann_env=
native_env=
args=("$@")
while (($#)); do
  case "$1" in
    --python) customer_python=$2; shift 2 ;;
    --repo) customer_repo=$2; shift 2 ;;
    --cann-env) cann_env=$2; shift 2 ;;
    --native-env) native_env=$2; shift 2 ;;
    -h|--help) exec "$customer_python" "$addon_dir/launch.py" --help ;;
    *) shift ;;
  esac
done
if [[ -z $customer_repo || -z $cann_env ]]; then
  echo 'Required: --repo REPOSITORY --cann-env CANN_ENV_SCRIPT' >&2
  exit 2
fi
if [[ -z $native_env ]]; then
  native_env=$customer_repo/build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
fi
if [[ ! -f $cann_env || ! -f $native_env ]]; then
  echo 'CANN/native environment script is missing. Specify --cann-env and --native-env.' >&2
  exit 2
fi
source "$cann_env"
source "$native_env"
export HCCL_DETERMINISTIC=true HCCL_HOST_SOCKET_PORT_RANGE=auto OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$customer_python" "$addon_dir/launch.py" "${args[@]}" --resolved-native-env "$native_env"
