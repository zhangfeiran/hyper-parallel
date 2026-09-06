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
import os
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
_COMMON_WORKER = str(Path(__file__).resolve().parent / "_test_mega_moe_common.py")
_PRECISION_WORLD_SIZE = 2
_PERFORMANCE_WORLD_SIZE = 4


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_shared_weight_pool_lifecycle(monkeypatch) -> None:
    """Compare shared staging with native across layers and backward replay.

    Feature: cross-layer physical-weight staging
    Description: cover deferred graphs, checkpoint, retained graphs and updates
    Expectation: stable storage and native-equivalent outputs and gradients
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    worker = str(Path(__file__).with_name("_test_mega_moe_shared_weights.py"))
    with without_inherited_rank_environment():
        parallel_run(
            [TorchCase(worker, "test_shared_weight_pool_lifecycle", num_proc=_PRECISION_WORLD_SIZE)],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


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
    monkeypatch.setenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(64 * 1024 * 1024))
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _COMMON_WORKER,
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
    """Compare output, dX, route dW, W1 dW, and W2 dW with common MoE."""
    _run_precision_worker(monkeypatch)


def _run_performance_worker(monkeypatch, result_dir: Path) -> dict:
    """Run common/MegaMoe A/B in one four-rank process group."""
    result_path = result_dir / "routed_expert_ab.json"
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_PERF_WARMUP", "3")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_MEASURED", "5")
    monkeypatch.setenv("HP_MEGA_MOE_MAX_PERF_RATIO", "1.10")
    monkeypatch.setenv("HP_MEGA_MOE_PERF_RESULT", str(result_path))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "62000-62063")
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _COMMON_WORKER,
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
    """Enforce one steady-state common/MegaMoe routed-expert A/B threshold."""
    _prepare_torch_multicore_test_environment()
    comparison = _run_performance_worker(monkeypatch, tmp_path)
    results = comparison["results"]
    reference = results["common"]
    for tag, backend_result in results.items():
        assert backend_result["shape"] == reference["shape"], (
            f"performance shape drifted for {tag}: "
            f"expected={reference['shape']}, got={backend_result['shape']}"
        )
        assert backend_result["topology"] == reference["topology"], (
            f"performance topology drifted for {tag}: "
            f"expected={reference['topology']}, got={backend_result['topology']}"
        )
        assert backend_result["device_healthy"], (
            f"performance device health failed for {tag}: result={backend_result}"
        )
        assert all(backend_result["validation"].values()), (
            f"performance validation failed for {tag}: "
            f"validation={backend_result['validation']}"
        )

    common_median = results["common"]["rank_max_fwd_bwd_ms"]["median"]
    mega_median = results["mega_moe"]["rank_max_fwd_bwd_ms"]["median"]
    assert common_median > 0.0, (
        f"common median latency must be positive, got {common_median}."
    )
    assert mega_median > 0.0, f"MegaMoe median latency must be positive, got {mega_median}."
    ratio = mega_median / common_median
    assert ratio <= comparison["maximum_allowed_ratio"], (
        "MegaMoe routed-expert performance exceeded the A/B threshold: "
        f"ratio={ratio}, maximum={comparison['maximum_allowed_ratio']}."
    )
    print(
        "MegaMoe representative A/B: "
        f"common_ms={common_median}, mega_ms={mega_median}, ratio={ratio:.6f}"
    )


def _run_acceptance_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int) -> dict:
    """Launch one real-device acceptance worker and require its rank evidence."""
    _prepare_torch_multicore_test_environment()
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path)))
    result_path = result_dir / f"{case}.json"
    worker = "_test_mega_moe_resources.py" if cards == 2 else "_test_mega_moe_defaults.py"
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(cards))
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(result_path))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    if cards == 2:
        monkeypatch.setenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(64 * 1024 * 1024))
    else:
        monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    with without_inherited_rank_environment():
        parallel_run(
            [TorchCase(str(Path(__file__).with_name(worker)), case, num_proc=cards)],
            global_num_proc=cards,
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["ranks"]) == cards, f"expected {cards} rank records, got {len(result['ranks'])}."
    return result


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_shared_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Shared execution resources and public SHMEM ownership.

    Description: Compare real serial stacks and alternate streams across repeated steps.
    Expectation: Outputs, gradients and updates match, memory stabilizes and the last owner finalizes.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_shared_resource_acceptance", 2)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_default_interface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Feature: Default and explicit capacity without Router-provided counts.

    Description: Compare identical representative shapes in ABBA order, including Router timing.
    Expectation: Outputs and gradients agree; stable latency, peak memory and software identity are recorded.
    """
    _run_acceptance_worker(monkeypatch, tmp_path, "test_mega_moe_default_capacity_acceptance", 4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_receive_capacity_fallback(monkeypatch) -> None:
    """
    Feature: kernel-internal multi-wave overflow precision
    Description: compare a two-rank route above receive capacity with native MoE
    Expectation: all forward and backward tensors satisfy tolerances
    """
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


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_non_power_of_two_topk_multi_wave(monkeypatch) -> None:
    """
    Feature: non-power-of-two route-slot decoding across waves
    Description: execute an overflowing Top-K=3 route in two waves
    Expectation: forward and backward tensors match native MoE
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_non_power_of_two_topk_multi_wave",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_hot_expert_replication(monkeypatch) -> None:
    """
    Feature: bounded hot-expert replication
    Description: rebalance an overflowing route with one replica slot per rank
    Expectation: one-wave output and owner-reduced gradients match native MoE
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_hot_expert_replication",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_deferred_backward_hot_expert_replication(monkeypatch) -> None:
    """
    Feature: persistent replica pool with multiple live autograd graphs
    Description: reuse one physical binding, then update only its owner rank
    Expectation: shared graphs and collective refresh match native execution
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_deferred_backward_hot_expert_replication",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_lazy_compact_home_to_replica_deferred_backward(monkeypatch) -> None:
    """Validate the lazy compact transition with a live home-only graph.

    Feature: lazy compact storage with deferred backward
    Description: Execute home forward A and first replica forward B before either backward
    Expectation: Both gradients match native and compact rebind waits for graph release
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_lazy_compact_home_to_replica_deferred_backward",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_eager_compact_home_to_replica_deferred_backward(monkeypatch) -> None:
    """Validate stable eager compact storage with a live home-only graph.

    Feature: eager compact storage with deferred backward
    Description: Prepare storage before home A, then execute replica B before backward
    Expectation: Both gradients match native and the prepared pointers remain stable
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_eager_compact_home_to_replica_deferred_backward",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_replication_with_residual_multi_wave(monkeypatch) -> None:
    """
    Feature: bounded replication with residual multi-wave execution
    Description: leave overflow after one replica slot and execute two waves
    Expectation: output and owner-reduced gradients match native MoE
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_replication_with_residual_multi_wave",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_checkpointed_capacity_fallback(monkeypatch) -> None:
    """
    Feature: checkpointed one-wave and kernel-internal multi-wave execution
    Description: replay balanced and overflowing routes with both checkpoint modes
    Expectation: every W1/W2 output and backward tensor matches the native oracle
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_checkpointed_capacity_fallback",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_retained_graph_capacity_fallback(monkeypatch) -> None:
    """
    Feature: retained autograd graph with kernel-internal multi-wave execution
    Description: execute backward twice from one overflowing forward graph
    Expectation: both complete backward results match the native MoE oracle
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_retained_graph_capacity_fallback",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_retained_graph_one_wave(monkeypatch) -> None:
    """
    Feature: retained autograd graph with one-wave hidden redispatch
    Description: execute backward twice from one balanced fused-route graph
    Expectation: both complete backward results match the native MoE oracle
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_retained_graph_one_wave",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_deferred_backward_capacity_fallback(monkeypatch) -> None:
    """
    Feature: multiple live W1/W2 graphs sharing one MegaMoe workspace
    Description: enqueue two forwards per wave count before either backward
    Expectation: each W1/W2 forward and backward tensor matches its native oracle
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_deferred_backward_capacity_fallback",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
@pytest.mark.parametrize("atomic_dw", ["0", "1"])
def test_mega_moe_mixed_deferred_backward_capacity(monkeypatch, atomic_dw: str) -> None:
    """
    Feature: outstanding W1 and W2 graphs sharing one workspace
    Description: mix both wave counts before running backwards in either order
    Expectation: output and all gradients match native with both dW scratch policies
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PRECISION_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_ATOMIC_DW_ACCUMULATION", atomic_dw)
    monkeypatch.setenv("HP_MEGA_MOE_REUSE_RETAINED_DW_SCRATCH", "0")
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_mixed_deferred_backward_capacity",
                    num_proc=_PRECISION_WORLD_SIZE,
                )
            ],
            global_num_proc=_PRECISION_WORLD_SIZE,
        )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_mega_moe_mixed_wave_replay(monkeypatch) -> None:
    """
    Feature: repeated kernel-internal multi-wave execution
    Description: reuse one layer across one hundred W1-W2-W1-W4 sequences
    Expectation: every forward and backward tensor matches native
    """
    _prepare_torch_multicore_test_environment()
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    monkeypatch.setenv("HP_MEGA_MOE_WORLD_SIZE", str(_PERFORMANCE_WORLD_SIZE))
    monkeypatch.setenv("HP_MEGA_MOE_MIXED_WAVE_REPETITIONS", "100")
    monkeypatch.setenv("SHMEM_IP_PORT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        parallel_run(
            [
                TorchCase(
                    _WORKER,
                    "test_mega_moe_mixed_wave_replay",
                    num_proc=_PERFORMANCE_WORLD_SIZE,
                )
            ],
            global_num_proc=_PERFORMANCE_WORLD_SIZE,
        )


def _run_performance_backend(
    monkeypatch,
    result_dir: Path,
    backend: str,
    tag: str,
    npu_port_range: str,
    settings: dict[str, str] | None = None,
) -> dict:
    """Run one fresh eight-rank backend process and load its result."""
    result_path = result_dir / f"{tag}.json"
    monkeypatch.delenv("SYMMETRIC_MEMORY_HEAP_SIZE", raising=False)
    environment = {
        "HP_MEGA_MOE_WORLD_SIZE": str(_PERFORMANCE_WORLD_SIZE),
        "HP_MEGA_MOE_PERF_BACKEND": backend,
        "HP_MEGA_MOE_PERF_WARMUP": "5",
        "HP_MEGA_MOE_PERF_MEASURED": "10",
        "HP_MEGA_MOE_PERF_RESULT": str(result_path),
        "HP_MEGA_MOE_PERF_ROUTE_PROFILE": "dirichlet",
        "HP_MEGA_MOE_PERF_DIRICHLET_ALPHA": "10000",
        "HP_MEGA_MOE_PERF_DIRICHLET_SEED": "42",
        "HP_MEGA_MOE_PERF_CLUSTERED_EXPERTS": "12",
        "HP_MEGA_MOE_PERF_REPLICA_SLOTS": "0",
        "HP_MEGA_MOE_CAPACITY_FACTOR": "1.5",
        "HCCL_NPU_SOCKET_PORT_RANGE": npu_port_range,
        "SHMEM_IP_PORT": f"tcp://127.0.0.1:{allocate_port()}",
    }
    environment.update(settings or {})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
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
def test_mega_moe_extended_representative_performance(monkeypatch, tmp_path: Path) -> None:
    """
    Feature: representative MegaMoe performance baseline
    Description: run native and MegaMoe forward-backward in separate eight-rank jobs
    Expectation: both runs remain healthy and report positive comparable latency
    """
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


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_mega_moe_placement_performance_matrix(monkeypatch, tmp_path: Path) -> None:
    """
    Feature: bounded-placement performance regimes
    Description: measure home, multi-wave, replica, and residual E48/K8 routes
    Expectation: every route reaches its planned wave regime with finite gradients
    """
    _prepare_torch_multicore_test_environment()
    common_skew = {
        "HP_MEGA_MOE_PERF_ROUTE_PROFILE": "clustered",
        "HP_MEGA_MOE_PERF_CLUSTERED_EXPERTS": "12",
        "HP_MEGA_MOE_CAPACITY_FACTOR": "1.0",
        "HP_MEGA_MOE_PERF_WARMUP": "3",
        "HP_MEGA_MOE_PERF_MEASURED": "6",
    }
    specs = (
        (
            "home_only",
            "62200-62263",
            {},
            "home-only",
            1,
        ),
        (
            "multi_wave_only",
            "62300-62363",
            common_skew,
            "multi-wave-only",
            5,
        ),
        (
            "replica_only",
            "62400-62463",
            {**common_skew, "HP_MEGA_MOE_PERF_REPLICA_SLOTS": "2"},
            "replica-only",
            1,
        ),
        (
            "replica_residual",
            "62500-62563",
            {**common_skew, "HP_MEGA_MOE_PERF_REPLICA_SLOTS": "1"},
            "replica+residual-multi-wave",
            2,
        ),
    )
    results = {}
    for tag, port_range, settings, expected_mode, expected_waves in specs:
        result = _run_performance_backend(
            monkeypatch,
            tmp_path,
            "mega_moe",
            tag,
            port_range,
            settings,
        )
        route = result["route"]
        assert result["device_healthy"], (
            f"performance device health failed for {tag}: result={result}"
        )
        assert all(result["validation"].values()), (
            f"performance validation failed for {tag}: "
            f"validation={result['validation']}"
        )
        assert route["execution_mode"] == expected_mode, (
            f"performance mode mismatch for {tag}: "
            f"expected={expected_mode}, got={route['execution_mode']}"
        )
        assert route["expected_wave_count"] == expected_waves, (
            f"performance wave count mismatch for {tag}: "
            f"expected={expected_waves}, got={route['expected_wave_count']}"
        )
        median = result["rank_max_fwd_bwd_ms"]["median"]
        assert median > 0.0, (
            f"performance median must be positive for {tag}: got={median}"
        )
        results[tag] = result

    clustered_loads = results["multi_wave_only"]["route"]["destination_loads"]
    for tag in ("replica_only", "replica_residual"):
        actual_loads = results[tag]["route"]["destination_loads"]
        assert actual_loads == clustered_loads, (
            f"clustered logical route drifted for {tag}: "
            f"expected={clustered_loads}, got={actual_loads}"
        )
    summary = {
        tag: {
            "mode": result["route"]["execution_mode"],
            "waves": result["route"]["expected_wave_count"],
            "median_ms": result["rank_max_fwd_bwd_ms"]["median"],
            "peak_bytes": result["memory"]["rank_max"][
                "conservative_total_peak_bytes"
            ],
        }
        for tag, result in results.items()
    }
    print("MegaMoe placement performance matrix: " + json.dumps(summary, sort_keys=True))
