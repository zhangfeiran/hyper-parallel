# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by law, distributed on an "AS IS" BASIS, WITHOUT WARRANTIES.

_hp_mhc_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
_hp_mhc_vendor="${_hp_mhc_root}/vendors/hyper_parallel_multicore_mhc_v1"
if [[ ! -f "${_hp_mhc_root}/manifest.json" || ! -f "${_hp_mhc_vendor}/op_api/lib/libcust_opapi.so" ]]; then
    echo "[HP-MHC-PAYLOAD-MISSING] incomplete payload: ${_hp_mhc_root}" >&2
    unset _hp_mhc_root _hp_mhc_vendor
    return 1 2>/dev/null || exit 1
fi
export HP_MHC_PAYLOAD_ROOT="${_hp_mhc_root}"
export ASCEND_CUSTOM_OPP_PATH="${_hp_mhc_vendor}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${_hp_mhc_vendor}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _hp_mhc_root _hp_mhc_vendor
