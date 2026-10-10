# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""在独立 CPU 进程中查询 SDK 内置 MatMulV2；不导入框架或 TBE Python。"""

from __future__ import annotations

import ctypes
import hashlib
import json
import platform
import sys
from pathlib import Path


def _query_library(request):
    sdk = Path(request["cann_root"])
    base = sdk / "opp/built-in/op_impl/ai_core/tbe"
    # AscendC 独立构建可能关闭 RPATH；先按封存的 SDK 路径加载桥接库的依赖。
    common = ctypes.CDLL(str(base / f"op_host/lib/linux/{platform.machine()}/libophost_comm_legacy.so"))
    initializer = ctypes.CDLL(request["library"])
    initialize = initializer.hyper_parallel_dense_compile_platform
    initialize.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    initialize.restype = ctypes.c_int
    status = initialize(request["soc"].encode(), request["cube_workers"])
    if status:
        raise ValueError(f"稠密参考 tiling 平台初始化失败：{status}")
    native = ctypes.CDLL(str(sdk / "lib64/libregister.so"))
    legacy = ctypes.CDLL(str(base / f"op_tiling/lib/linux/{platform.machine()}/liboptiling.so"))
    host_path = base / f"op_host/lib/linux/{platform.machine()}/libophost_legacy.so"
    host = ctypes.CDLL(str(host_path))
    register = host.TbeLoadSoAndSaveToRegistry
    register.argtypes = [ctypes.c_char_p]
    register.restype = None
    register(str(host_path).encode())
    query = native.DoOpTilingForCompile
    query.argtypes = [ctypes.c_char_p] * 6 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_char_p]
    query.restype = ctypes.c_char_p
    return query, (initializer, native, legacy, host, common)


def _tensor(name, shape):
    return {"name": name, "shape": list(shape), "ori_shape": list(shape), "dtype": "bfloat16",
            "format": "ND", "ori_format": "ND"}


def _query_shape(query, request, shape):
    rows, columns, contracted, transpose = shape
    sdk = Path(request["cann_root"])
    metadata = sdk / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/mat_mul"
    metadata /= f"MatMulV2_ND_ND_FP16_FP16_false_{str(transpose).lower()}_all.json"
    compile_info = json.loads(metadata.read_text(encoding="utf-8"))["compileInfo"]
    compile_info["hardware_info"]["CORE_NUM"] = request["cube_workers"]
    right = (columns, contracted) if transpose else (contracted, columns)
    inputs = [_tensor("x1", (rows, contracted)), _tensor("x2", right), None, None]
    outputs = [_tensor("y", (rows, columns))]
    attrs = [{"name": "transpose_x1", "dtype": "bool", "value": False},
             {"name": "transpose_x2", "dtype": "bool", "value": transpose},
             {"name": "offset_x", "dtype": "int", "value": 0},
             {"name": "_op_impl_mode_enum", "dtype": "int", "value": 4}]
    extra = {"op_name": "", "deterministic": request["deterministic"], "deterministic_level": 0,
             "_op_aicore_num": str(request["cube_workers"]), "_op_vectorcore_num": str(2 * request["cube_workers"])}
    info = json.dumps(compile_info).encode()
    buffer = ctypes.create_string_buffer(65536)
    response = query(b"MatMulV2", info, hashlib.sha1(info).hexdigest().encode(), json.dumps(inputs).encode(),
                     json.dumps(outputs).encode(), json.dumps(attrs).encode(), buffer, len(buffer), None,
                     json.dumps(extra).encode())
    if response is None:
        raise ValueError("稠密参考 tiling 查询未返回状态")
    status = json.loads(response)
    if status.get("ret_code") != 0:
        raise ValueError(f"稠密参考 tiling 查询失败：shape={shape}, status={status}")
    return {"shape": shape, "run_info": json.loads(buffer.value),
            "compile_info_sha256": hashlib.sha256(info).hexdigest()}


def _native_family(initializer, request, shape):
    select = initializer.hyper_parallel_dense_native_family
    select.argtypes = [ctypes.c_char_p] + [ctypes.c_uint32] * 6
    select.restype = ctypes.c_int
    choices = [select(request["soc"].encode(), request["cube_workers"], *shape, split) for split in (0, 1)]
    if any(choice not in (0, 1) for choice in choices):
        raise ValueError(f"稠密原生内核族查询失败：shape={shape}, choices={choices}")
    family = "ambiguous" if choices[0] != choices[1] else ("MatMulV3" if choices[0] else "MatMulV2")
    return {"native_family": family, "v3_with_split_k_false_true": choices}


def main() -> None:
    """从标准输入读取已验证的构建参数，将完整查询结果写入标准输出。"""
    request = json.load(sys.stdin)
    query, libraries = _query_library(request)
    results = [{**_query_shape(query, request, shape), **_native_family(libraries[0], request, shape)}
               for shape in request["shapes"]]
    if not libraries:
        raise ValueError("稠密参考 tiling 库未保持有效")
    print(json.dumps({"format_version": 2, "operator": "native_dense_matmul", "soc": request["soc"],
                      "cube_workers": request["cube_workers"], "deterministic": request["deterministic"],
                      "scope": "sdk_host_query", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
