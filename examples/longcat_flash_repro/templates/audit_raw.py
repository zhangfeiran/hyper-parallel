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
"""Independently re-read every rank artifact and recompute reported timing samples."""

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE = "__REPO__/hyper_parallel/__init__.py"


def load(path: Path) -> Any:
    """Load a structured experiment artifact.

    Args:
        path: Artifact path to read or write.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def audit_monitor(folder: Path, known: set[int]) -> None:
    """Check recorded device process ownership.

    Args:
        folder: Case directory containing archived rank records.
        known: Worker process IDs owned by this case.
    """
    samples = load(folder / "monitor.json")
    assert samples and all(row["valid"] for row in samples)
    observed = {p["pid"] for row in samples for p in row["processes"]}
    owned = known | {p["pid"] for row in samples for p in row["processes"] if p["belongs_to_case"] is True}
    assert known <= observed <= owned


def audit_expert_backend(folder: Path, terminal: Any) -> str:
    """Verify expert backend, activation and capacity metadata.

    Args:
        folder: Case directory containing archived rank records.
        terminal: Rank terminal-state record.
    """
    arm = (
        "native"
        if folder.name == "numerical-native"
        or folder.name.endswith("-native")
        and not folder.name.endswith("-fast-native")
        else ("megamoe" if folder.name.endswith("-megamoe") else "fast-native")
    )
    expected_backend = {
        "backend": "megamoe" if arm == "megamoe" else "native",
        "swiglu_precision": "fused",
        "routing_backend": "torch" if arm == "native" else "triton",
        "optimize_tasks": True,
        "direct_weight_gradients": arm != "native",
    }
    assert len(terminal["expert_backends"]) == 1 and all(
        value == expected_backend for value in terminal["expert_backends"].values()
    )
    if arm == "megamoe":
        assert len(terminal["heap_state"]) == 1
        assert all(
            all(
                (
                    v["source_capacity"] == v["initial_source_capacity"] == 24576,
                    v["epoch"] == 0,
                    not v["growth_records"],
                    not v["forward_hf_bf16"],
                    not v["backward_hf_bf16"],
                )
            )
            for v in terminal["heap_state"].values()
        )
    if arm == "megamoe":
        assert any("__NATIVE_ROOT__/" in path for path in terminal["native_library_maps"])
    return arm


def audit_trainer_rank(folder: Path, rank: int, terminal: Any, rows: Any, summary: Any, expected: int) -> int:
    """Verify one Trainer rank and count its aligned updates.

    Args:
        folder: Case directory containing archived rank records.
        rank: Logical rank within the selected process group.
        terminal: Rank terminal-state record.
        rows: Raw aligned per-step records.
        summary: Complete case summary.
        expected: Expected number of aligned updates.
    """
    count = 0
    assert terminal["rank"] == rank and terminal["world"] == 16 and terminal["source"] == SOURCE
    assert terminal["all_main_fp32_finite"] and terminal["shmem_after_close"]["reference_count"] == 0
    assert not Path(f"/proc/{terminal['pid']}").exists()
    assert terminal["HCCL_DETERMINISTIC"] == "true"
    assert terminal["parameters"]["model.layers.0.mlp.experts.gate_up_proj"]["local_shape"] == [12, 4096, 6144]
    assert terminal["parameters"]["model.layers.0.mlp.experts.down_proj"]["local_shape"] == [8, 6144, 2048]
    arm = audit_expert_backend(folder, terminal)
    assert [row["step"] for row in rows] == list(range(2, expected + 2))
    if arm == "megamoe":
        assert all(row["heap_epochs"] == rows[5]["heap_epochs"] for row in rows[5:])
    assert rows == summary["steps"][rank]
    for row in rows:
        assert math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])
        assert row["objective_dtypes"] == ["torch.float32"]
        count += 1
    return count


def audit_trainer() -> dict[str, Any]:
    "Check every Trainer terminal record and raw timing sample."
    reference = None
    processes = steps = 0
    for path in sorted((ROOT / "trainer").glob("*/exit.json")):
        folder = path.parent
        if folder.name == "initialize":
            continue
        assert load(path)["exit_code"] == 0
        summary = load(folder / "summary.json")
        ranks = [load(folder / f"terminal-rank{rank}.json") for rank in range(16)]
        audit_monitor(folder, {r["pid"] for r in ranks})
        raw = [
            [json.loads(line) for line in (folder / f"steps-rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
            for rank in range(16)
        ]
        expected = 10 if folder.name.startswith("numerical-") else 15
        for rank, (terminal, rows) in enumerate(zip(ranks, raw)):
            steps += audit_trainer_rank(folder, rank, terminal, rows, summary, expected)
        for index in range(expected):
            maximum = max(rows[index]["local_seconds"] for rows in raw)
            assert all(abs(rows[index]["max_rank_seconds"] - maximum) <= max(1e-6, maximum * 1e-6) for rows in raw)
        if folder.name == "numerical-native":
            reference = raw
        processes += 1
    assert processes == 12 and reference is not None
    return {
        "processes": processes,
        "ranks_per_process": 16,
        "raw_step_rows_checked": steps,
        "raw_source": str(ROOT / "trainer"),
        "all_rank_terminal_and_rank_max_verified": True,
    }


def audit_component_rank(row: dict, rank: int, case: dict) -> None:
    """Verify component terminal state, physical shapes and fused operators.

    Args:
        row: Complete worker rank report.
        rank: Logical rank in the selected group.
        case: Component process metadata.
    """
    assert row["rank"] == rank and row["hyper_parallel_source"] == SOURCE
    assert row["shape"] == {
        "ep": 4,
        "tokens": case["tokens"],
        "hidden": 6144,
        "intermediate": 2048,
        "real": 32,
        "zero": 16,
        "top_k": 12,
    }
    assert row["parameter_local_shapes"]["gate_up_proj"] == [12, 4096, 6144]
    assert row["parameter_local_shapes"]["down_proj"] == [8, 6144, 2048]
    assert row["swiglu_precision"] == "fused"
    assert row["direct_weight_gradients"] == case["direct_weight_gradients"]
    assert row["routing_backend"] == case["routing_backend"]
    if case["backend"] == "native":
        assert all(
            row["native_operator_census"][name] > 0
            for name in ["npu_grouped_matmul", "npu_swiglu", "npu_moe_token_permute", "npu_moe_token_unpermute"]
        )
    activation = row["activation_execution"]
    if case["backend"] == "native":
        assert row["native_operator_census"]["silu"] == 0 and activation["operator_calls"] > 0
    else:
        assert not activation["forward_hf_bf16"] and not activation["backward_hf_bf16"]
    assert row["routing"]["real_source_rows"] + row["routing"]["identity_source_rows"] == case["tokens"] * 12
    assert row["finite_gradients"] and row["unused_gate_rows_gradient_zero"]
    assert row["shmem_after_close"]["reference_count"] == 0
    assert not Path(f"/proc/{row['pid']}").exists()


def audit_component() -> dict[str, Any]:
    "Check every component rank and recompute rank maximum timings."
    data = load(ROOT / "component/results.json")
    assert len(data["cases"]) == 15
    count = 0
    for case in data["cases"]:
        folder = ROOT / "component" / case["case"]
        assert load(folder / "exit.json")["exit_code"] == 0
        ranks = [load(folder / f"result/rank{rank}.json") for rank in range(4)]
        audit_monitor(folder, {r["pid"] for r in ranks})
        for rank, row in enumerate(ranks):
            audit_component_rank(row, rank, case)
        real = sum(row["routing"]["real_source_rows"] for row in ranks)
        assert sum(row["routing"]["owner_received_rows"] for row in ranks) == real
        assert all(
            len(row["routing"]["global_real_expert_counts"]) == 32
            and sum(row["routing"]["global_real_expert_counts"]) == real
            for row in ranks
        )
        for phase in ["forward", "forward_backward"]:
            values = [max(row["phases"][phase]["local_ms"][i] for row in ranks) for i in range(30)]
            assert values == case["phases"][phase]["rank_max_ms"]
            assert statistics.median(values) == case["phases"][phase]["p50_ms"]
            count += len(values)
    return {
        "processes": 15,
        "ranks_per_process": 4,
        "rank_max_samples_recomputed": count,
        "all_rank_terminal_and_shape_checks_verified": True,
    }


if __name__ == "__main__":
    report = {"trainer": audit_trainer(), "component": audit_component(), "status": "PASS"}
    (ROOT / "raw-evidence-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)
