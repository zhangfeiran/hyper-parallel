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

"""Lightweight launchers for public MegaMoe layer system tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore_test_env import (
    multicore_framework_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)
from tests.common.parallel_case import TorchCase, parallel_run
from tests.common.port_utils import allocate_port

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe.py")
_PRECISION_WORLD_SIZE = 2
_PERFORMANCE_WORLD_SIZE = 4


def _prepare_torch_multicore_test_environment() -> None:
    """Activate multicore and require a payload built for the Torch framework."""
    prepare_multicore_test_environment()
    if not multicore_framework_adapter_is_available("torch"):
        pytest.skip(
            "MegaMoe Torch ST requires a wheel or PYTHONPATH payload built with "
            "--multicore torch or --multicore all"
        )


def _run_precision_worker(monkeypatch) -> None:
    """Run the two-rank precision worker with an isolated SHMEM endpoint."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_level0_balanced",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_level0_precision(monkeypatch) -> None:
    """Compare output, dX, route dW, W1 dW, and W2 dW with native MoE."""
    _run_precision_worker(monkeypatch)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_receive_capacity_fallback(monkeypatch) -> None:
    """Compare the two-rank overflowing fallback with native expert parallelism."""
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_level0_capacity_fallback",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


def _run_performance_backend(
    monkeypatch,
    result_dir: Path,
    backend: str,
    tag: str,
    npu_port_range: str,
) -> dict:
    """Run one fresh four-rank backend process and load its result."""
    result_path = result_dir / f"{tag}.json"
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PERF_BACKEND", backend)
    monkeypatch.setenv("HP_MEGA_MOE_PERF_WARMUP", "5")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_MEASURED", "10")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_RESULT", str(result_path))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", npu_port_range)
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_fwd_bwd_performance",
                    num_proc=_PERFORMANCE_WORLD_SIZE,
                )
            ],
            global_num_proc=_PERFORMANCE_WORLD_SIZE,
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_representative_performance(monkeypatch, tmp_path: Path) -> None:
    """Record a native/MegaMoe F+B baseline in independent four-rank runs."""
    _prepare_torch_multicore_test_environment()
    specs = (
        ("native", "native", "62000-62063"),
        ("mega_moe", "mega_moe", "62100-62163"),
    )
    results = {
        tag: _run_performance_backend(
            monkeypatch,
            tmp_path,
            backend,
            tag,
            port_range,
        )
        for backend, tag, port_range in specs
    }
    reference = results["native"]
    for tag, result in results.items():
        assert result["shape"] == reference["shape"], (
            f"performance shape drifted for {tag}: expected={reference['shape']}, got={result['shape']}"
        )
        assert result["topology"] == reference["topology"], (
            f"performance topology drifted for {tag}: "
            f"expected={reference['topology']}, got={result['topology']}"
        )
        assert result["device_healthy"], f"performance device health failed for {tag}: result={result}"
        assert all(result["validation"].values()), (
            f"performance validation failed for {tag}: validation={result['validation']}"
        )

    native_median = results["native"]["rank_max_fwd_bwd_ms"]["median"]
    mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
    assert native_median > 0.0, f"native median latency must be positive, got {native_median}."
    assert mega_median > 0.0, f"MegaMoe median latency must be positive, got {mega_median}."
    ratio = mega_median / native_median
    print(
        "MegaMoe representative A/B: "
        f"native_ms={native_median}, mega_ms={mega_median}, ratio={ratio:.6f}"
    )
