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
"""CPU-only snapshots for isolated Qwen backend comparisons, never used during timing."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Torch-only example serialization; importing this module never initializes a device.
import torch  # pylint: disable=forbidden-backend-import

_SCHEMA_VERSION = 1
_CHUNK_SIZE = 1_048_576


def save_snapshot(
    path: Path,
    contract: dict[str, Any],
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    """Write owned, compact CPU values without replacing an existing reference.

    The caller must finish all snapshot writes before marking its run complete.
    Transfers are blocking and the stored values cannot alias device parameters.
    """
    if path.exists():
        raise FileExistsError(f"Qwen reference already exists: {path}")
    values = {}
    for name, tensor in tensors:
        if name in values:
            raise ValueError(f"duplicate Qwen tensor name: {name!r}")
        values[name] = tensor.detach().to(device="cpu", copy=True).contiguous()
    if not values:
        raise ValueError("Qwen snapshots must contain tensors.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save({"schema_version": _SCHEMA_VERSION, "contract": contract, "tensors": values}, stream)
    return {"status": "reference_saved", "tensor_count": len(values), "path": str(path)}


def _tensor_errors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    chunk_size: int,
) -> tuple[bool, float, float]:
    """Use the original reference-relative FP32 oracle in bounded CPU chunks."""
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    passed, maximum_absolute, maximum_normalized = True, 0.0, 0.0
    for offset in range(0, reference.numel(), chunk_size):
        expected_raw = reference_flat[offset:offset + chunk_size]
        actual_raw = candidate_flat[offset:offset + chunk_size]
        expected = expected_raw.float() if reference.is_floating_point() else expected_raw.double()
        actual = actual_raw.float() if candidate.is_floating_point() else actual_raw.double()
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        close = torch.equal(actual_raw, expected_raw) if not (rtol or atol) else bool(
            torch.isclose(actual, expected, rtol=rtol, atol=atol).all()
        )
        passed = passed and finite and close
        if not finite:
            maximum_absolute = maximum_normalized = math.inf
            continue
        difference = (actual - expected).abs()
        tolerance = atol + rtol * expected.abs()
        normalized = difference / tolerance.clamp_min(torch.finfo(torch.float32).tiny)
        maximum_absolute = max(maximum_absolute, float(difference.max()))
        maximum_normalized = max(maximum_normalized, float(normalized.max()))
    return passed, maximum_absolute, maximum_normalized


def compare_snapshot(
    path: Path,
    contract: dict[str, Any],
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    rtol: float,
    atol: float,
    chunk_size: int = _CHUNK_SIZE,
) -> dict[str, Any]:
    """Compare one rank/phase against a CPU reference without a second device model.

    Reference tensors are memory mapped. Only one candidate tensor and bounded
    FP32 comparison chunks are staged on the CPU at a time. Rank reduction is
    the distributed worker's responsibility.
    """
    if chunk_size <= 0 or not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("Qwen comparison requires positive chunks and finite nonnegative tolerances.")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != _SCHEMA_VERSION or payload.get("contract") != contract:
        raise ValueError(f"Qwen snapshot contract mismatch: {path}")
    references = payload["tensors"]
    if not isinstance(references, dict) or not references:
        raise ValueError(f"Qwen snapshot has no tensor map: {path}")
    result = {
        "passed": True, "tensor_count": len(references), "rtol": rtol, "atol": atol,
        "max_absolute_error": 0.0, "max_absolute_error_tensor": next(iter(references)),
        "max_normalized_error": 0.0, "max_normalized_error_tensor": next(iter(references)),
    }
    seen = set()
    for name, tensor in tensors:
        if name not in references or name in seen:
            raise ValueError(f"unexpected or duplicate Qwen tensor: {name!r}")
        seen.add(name)
        reference = references[name]
        if not isinstance(reference, torch.Tensor) or reference.device.type != "cpu":
            raise ValueError(f"Qwen reference {name!r} is not a CPU tensor.")
        if reference.shape != tensor.shape or reference.dtype != tensor.dtype:
            raise ValueError(f"Qwen snapshot shape/dtype mismatch for {name!r}: "
                             f"reference={reference.shape}/{reference.dtype}, actual={tensor.shape}/{tensor.dtype}")
        candidate = tensor.detach().to(device="cpu", copy=True).contiguous()
        passed, absolute, normalized = _tensor_errors(reference, candidate, rtol=rtol, atol=atol, chunk_size=chunk_size)
        result["passed"] = result["passed"] and passed
        for metric, value in (("absolute", absolute), ("normalized", normalized)):
            if value > result[f"max_{metric}_error"]:
                result[f"max_{metric}_error"] = value
                result[f"max_{metric}_error_tensor"] = name
        del candidate
    if seen != references.keys():
        raise ValueError(f"missing Qwen tensors: {sorted(references.keys() - seen)}")
    return result
