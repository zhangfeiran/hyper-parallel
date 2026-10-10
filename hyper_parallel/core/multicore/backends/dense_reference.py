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
"""隔离的 SDK 参考查询与数据无关的 K 维访问参数。"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

ACCESS = struct.Struct("<6I")
REFERENCE_FIELDS = (
    "k_ori", "m_ori", "n_ori", "k", "m", "n", "batch_single_core", "m_single_core", "n_single_core",
    "batch_dim", "n_dim", "m_dim", "k_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time",
    "kal0_factor", "kbl0_factor", "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times",
    "batch_l1_factor", "batch_ub_l0_time", "batch_cub", "out_branch_flag", "bias_flag", "hf32_flag",
    "datatype_bf16", "al1_db", "bl1_db", "l0c_db", "l2_cache_flag", "close_k_shift",
)


def matrix_geometry(shape: tuple) -> tuple[int, int, int, bool, bool]:
    """规范化旧右转置 ABI 和显式双侧转置 ABI，拒绝布尔维度或整数标志。"""
    if type(shape) not in (tuple,) or len(shape) not in (4, 5):
        raise ValueError("Dense matrix geometry requires four or five exact tuple fields")
    geometry = (*shape[:3], False, shape[3]) if len(shape) == 4 else shape
    if (any(type(value) not in (int,) or value < 0 for value in geometry[:3])
            or any(type(value) not in (bool,) for value in geometry[3:])):
        raise ValueError("Dense matrix geometry requires integer dimensions and exact boolean transposes")
    return geometry


def _sequential_reference(fields, key, contracted):
    if (fields["datatype_bf16"] != 1 or fields["bias_flag"] or fields["hf32_flag"]
            or fields["batch_dim"] != 1 or fields["k_dim"] != 1 or fields["kl1_times"] != 1
            or fields["k"] != (contracted + 15) // 16 or fields["k"] == 0
            or ((key >> 10) & 31) != 0 or key >> 18 != 0):
        return None
    if min(fields["k_l0"], fields["kal1_16"], fields["kbl1_16"]) >= fields["k"]:
        return "sdk_single_k_block_sequential"
    # SDK 的单个 M/N 分区没有核间 K 重排，完整 L1 内的多个 L0 块仍按 K 顺序累加。
    if (fields["m_dim"] == fields["n_dim"] == fields["kal1_factor"] == fields["kbl1_factor"] == 1
            and fields["k_l0"] > 0
            and fields["kal1_16"] == fields["kal0_factor"] * fields["k_l0"]
            and fields["kbl1_16"] == fields["kbl0_factor"] * fields["k_l0"]
            and min(fields["kal1_16"], fields["kbl1_16"]) >= fields["k"]):
        return "sdk_single_partition_sequential"
    return None


def _full_l1_reference(fields, key, shape, cube_workers):
    rows, columns, contracted = shape[:3]
    attachment = ((key >> 3) & 3, (key >> 5) & 3)
    if (attachment not in ((2, 0), (0, 2)) or columns % 16 or contracted % 16
            or max(columns, contracted) > 65535 or fields["datatype_bf16"] != 1
            or fields["bias_flag"] or fields["hf32_flag"] or fields["batch_dim"] != 1 or fields["k_dim"] != 1
            or fields["k"] != contracted // 16 or fields["k"] == 0
            or ((key >> 10) & 31) != 0 or key >> 18 != 0):
        return None
    full, partial = ("b", "a") if attachment == (2, 0) else ("a", "b")
    full_k, partial_k = fields[f"k{full}l1_16"], fields[f"k{partial}l1_16"]
    if (full_k != fields["k"] or partial_k <= 0 or full_k % partial_k or fields["k_l0"] <= 0
            or fields[f"k{full}l1_factor"] != 1
            or fields[f"k{partial}l1_factor"] * partial_k != full_k
            or fields["kl1_times"] * partial_k != full_k
            or fields[f"k{full}l0_factor"] * fields["k_l0"] != full_k
            or fields[f"k{partial}l0_factor"] * fields["k_l0"] != partial_k):
        return None
    m_span = 16 * fields["m_single_core"] * fields["m_al1"] * fields["m_l0"]
    n_span = 16 * fields["n_single_core"] * fields["n_bl1"] * fields["n_ub_l0_time"] * fields["cub_n1"]
    if (min(m_span, n_span, fields["m_dim"], fields["n_dim"]) <= 0
            or fields["m_dim"] * fields["n_dim"] > min(cube_workers, 24)
            or m_span * fields["m_dim"] < rows or n_span * fields["n_dim"] < columns):
        return None
    # SDK 满载分支的最外层 K 覆盖整次收缩，shift_block_access 对这个单迭代轴不改变顺序。
    return "sdk_full_l1_k_sequential"


def reference_access(result: dict[str, object], shape: tuple,
                     cube_workers: int, *, allow_right_transpose: bool = False,
                     allow_left_transpose: bool = False,
                     allow_sequential_reference: bool = False) -> tuple[bytes, str]:
    """从 BF16 ND 模板的原始字段生成 M/N 分区、K 块大小和禁用标志。

    Args:
        result: 同一矩阵的 SDK 主机查询记录。
        shape: 行数、列数、收缩维度及转置标志；兼容旧四字段 ABI。
        cube_workers: SDK 查询得到的目标 Cube 数。
        allow_right_transpose: 仅由具有可转置 B 加载回调的独立反向 worker 启用。
        allow_left_transpose: 仅由具有可转置 A 加载回调的共享矩阵核心启用。
        allow_sequential_reference: 显式 Linear 可准入单 K 块、单分区或完整 L1 K 的顺序模板。

    Returns:
        固定 24 字节的设备访问参数及准入原因。
    """
    if any(type(flag) not in (bool,) for flag in
           (allow_left_transpose, allow_right_transpose, allow_sequential_reference)):
        raise ValueError("稠密参考的右转置加载能力必须为精确布尔值")
    if tuple(result["shape"]) != shape:
        raise ValueError("稠密参考查询的矩阵形状不匹配")
    run = result["run_info"]
    try:
        raw = bytes.fromhex(run["tiling_data"])
        fields = dict(zip(REFERENCE_FIELDS, struct.unpack("<38i", raw)))
    except (KeyError, TypeError, ValueError, struct.error) as exc:
        raise ValueError("稠密参考 tiling 的字段 ABI 不匹配") from exc
    rows, columns, contracted, transpose_left, transpose = matrix_geometry(shape)
    if (fields["m_ori"], fields["n_ori"], fields["k_ori"]) != shape[:3]:
        raise ValueError("稠密参考 tiling 的原始维度不匹配")
    key = run.get("tiling_key")
    if not isinstance(key, int) or isinstance(key, bool) or key < 0 or any(value < 0 for value in fields.values()):
        raise ValueError("稠密参考 tiling 含有非法模板或字段")
    family = result.get("native_family")
    if family not in ("MatMulV2", "MatMulV3", "ambiguous"):
        raise ValueError("稠密参考 tiling 缺少有效原生内核族")
    choices = result.get("v3_with_split_k_false_true")
    valid_choices = (isinstance(choices, (tuple, list)) and len(choices) == 2
                     and all(isinstance(choice, int) and not isinstance(choice, bool) and choice in (0, 1)
                             for choice in choices))
    if not valid_choices:
        raise ValueError("稠密参考 tiling 的内核族选择条件无效")
    expected_family = "ambiguous" if choices[0] != choices[1] else ("MatMulV3" if choices[0] else "MatMulV2")
    if family != expected_family:
        raise ValueError("稠密参考 tiling 的内核族和选择结果不一致")
    if family != "MatMulV2":
        reason = "native_matmul_v3_sequential" if family == "MatMulV3" else "ambiguous_native_family"
        return bytes(ACCESS.size), reason
    if (allow_sequential_reference and (not transpose or allow_right_transpose)
            and (not transpose_left or allow_left_transpose) and not (transpose and transpose_left)
            and (reason := (_sequential_reference(fields, key, contracted)
                            or _full_l1_reference(fields, key, shape, cube_workers))) is not None):
        return bytes(ACCESS.size), reason
    # 位布局来自选定 SDK 的 gemm_tilingcase；满载、split-K 和其他布局尚未证明可复现。
    compatible = ((not transpose or allow_right_transpose) and (not transpose_left or allow_left_transpose)
                  and not (transpose and transpose_left) and columns % 16 == 0 and contracted % 16 == 0
                  and max(columns, contracted) <= 65535
                  and fields["datatype_bf16"] == 1 and fields["bias_flag"] == fields["hf32_flag"] == 0
                  and fields["batch_dim"] == fields["k_dim"] == 1
                  and ((key >> 3) & 3) == 2 and ((key >> 5) & 3) == 2
                  and ((key >> 10) & 31) == 0 and key >> 18 == 0)
    if not compatible:
        return bytes(ACCESS.size), "unsupported_reference_template"
    m_span = 16 * fields["m_single_core"] * fields["m_al1"] * fields["m_l0"]
    n_span = 16 * fields["n_single_core"] * fields["n_bl1"] * fields["n_ub_l0_time"] * fields["cub_n1"]
    m_extent, n_extent = fields["m_dim"], fields["n_dim"]
    k_chunk = 16 * max(fields["kal1_16"], fields["kbl1_16"])
    if (min(m_span, n_span, m_extent, n_extent, k_chunk) == 0 or contracted % k_chunk
            or m_extent * n_extent > min(cube_workers, 24)
            or m_span * m_extent < rows or n_span * n_extent < columns
            or max(m_span, n_span, k_chunk) > 2**31 - 1):
        return bytes(ACCESS.size), "unsupported_reference_partition"
    return ACCESS.pack(m_span, n_span, m_extent, n_extent, k_chunk, fields["close_k_shift"]), "sdk_k_block_access"


class DenseReferenceTiler:
    """一次性 CPU 子进程查询，不改变训练进程的全局 SDK 编译平台。"""

    def __init__(self, cann_root: Path, library: Path, soc: str, cube_workers: int,
                 deterministic: bool) -> None:
        """保留调用方已经验证过的 SDK、桥接库和实际执行配置。"""
        self.request = {"cann_root": str(Path(cann_root).resolve()), "library": str(Path(library).resolve()),
                        "soc": soc, "cube_workers": cube_workers, "deterministic": deterministic}

    def query(self, shapes: tuple[tuple, ...]) -> dict[str, object]:
        """查询整图矩阵的原生参考分区；不会出现在前向热路径。

        Args:
            shapes: 整图中所有非空矩阵的具体形状。
        """
        script = Path(__file__).resolve().parents[1] / "_build/query_dense_reference.py"
        for shape in shapes:
            matrix_geometry(shape)
        request = {**self.request, "shapes": shapes}
        process = subprocess.run([sys.executable, str(script)], input=json.dumps(request), text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if process.returncode:
            raise ValueError(f"稠密参考 tiling 子进程失败：{process.returncode}\n{process.stderr}")
        try:
            record = json.loads(process.stdout.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError("稠密参考 tiling 子进程返回了非法记录") from exc
        expected = {"format_version": 2, "operator": "native_dense_matmul", "scope": "sdk_host_query",
                    **{key: self.request[key] for key in ("soc", "cube_workers", "deterministic")}}
        if any(record.get(key) != value for key, value in expected.items()) or len(record["results"]) != len(shapes):
            raise ValueError("稠密参考 tiling 查询身份或矩阵数量不匹配")
        for result, shape in zip(record["results"], shapes):
            reference_access(result, shape, self.request["cube_workers"])
        return record
