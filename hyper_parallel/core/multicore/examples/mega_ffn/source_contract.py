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
"""不导入框架的 Source-NK 完整训练执行记录检查。"""

from __future__ import annotations

from collections import Counter

ACCESS_REASONS = {"sdk_k_block_access", "native_matmul_v3_sequential", "sdk_single_k_block_sequential",
                  "sdk_single_partition_sequential", "sdk_full_l1_k_sequential"}


def _matrix_sources(tiling, expected):
    references, reasons = tiling.get("matrix_references"), tiling.get("access_reasons")
    if not isinstance(references, list) or not isinstance(reasons, list) or len(references) != len(reasons):
        return False
    sources = []
    transposes = {"dense.linear": [False, True], "dense.linear.input_grad": [False, False],
                  "dense.linear.weight_grad": [True, False]}
    for reference, reason in zip(references, reasons):
        if reference is None:
            if reason != "no_matrix":
                return False
            continue
        if not isinstance(reference, dict):
            return False
        source, matrix = reference.get("source"), reference.get("matrix")
        if (source not in transposes or reason not in ACCESS_REASONS or not isinstance(matrix, list)
                or reference.get("coordinates_transposed") is not (source == "dense.linear.weight_grad")
                or len(matrix) != 5 or any(type(value) not in (int,) or value <= 0 for value in matrix[:3])
                or any(type(value) not in (bool,) for value in matrix[3:]) or matrix[3:] != transposes[source]):
            return False
        sources.append(source)
    return Counter(sources) == Counter(expected)


def resident_source_entry(entry: dict[str, object], baseline: str) -> bool:
    """检查相应 Source-NK 语义的真实常驻前向与完整反向记录。

    Args:
        entry: 从执行清单保存到 JSON 后读出的单个形状记录。
        baseline: 原始 common 或 packed 数值契约。

    Returns:
        只有数值来源、完整梯度、设备调用及无主机补算均满足契约时返回 True。
        此结构检查还需要独立的数值轨迹、源码身份和进程归属证据。
    """
    if baseline not in ("common", "packed"):
        raise ValueError("Source-NK acceptance requires the common or packed contract")
    projections = 3 if baseline == "common" else 2
    native = entry.get("native") or {}
    forward, runtime = native.get("tiling") or {}, native.get("backward_runtime") or {}
    calls, host, payloads = runtime.get("native_calls"), runtime.get("host_calls"), runtime.get("native_payloads")
    forward_caps = {"matmul_nt", "silu", "multiply"} if baseline == "common" else {"matmul_nt", "swiglu"}
    backward_caps = {"matmul_nn", "matmul_tn", "silu_grad", "multiply", "add"} if baseline == "common" else {
        "matmul_nn", "matmul_tn", "swiglu_grad"}
    if (native.get("execution_mode") != "resident_dense_tile_candidate" or forward.get("direction") != "forward"
            or native.get("backward_execution_mode") != "resident_dense_backward"
            or type(calls) not in (int,) or calls <= 0 or type(host) not in (int,) or host != 0
            or runtime.get("unbound_capabilities") != [] or not isinstance(payloads, list) or not payloads
            or set(forward.get("required_capabilities", ())) != forward_caps
            or not _matrix_sources(forward, ["dense.linear"] * projections)):
        return False
    for payload in payloads:
        plan, tiling = payload.get("plan") or {}, payload.get("tiling") or {}
        inputs, outputs = plan.get("input_mask"), plan.get("output_mask")
        if (payload.get("execution_mode") != "resident_backward_candidate"
                or not isinstance(inputs, list) or len(inputs) != projections + 1
                or any(value is not True for value in inputs) or outputs != [True] or outputs[0] is not True
                or set(tiling.get("required_capabilities", ())) != backward_caps
                or not _matrix_sources(tiling, ["dense.linear.input_grad", "dense.linear.weight_grad"] * projections)):
            return False
    return True
