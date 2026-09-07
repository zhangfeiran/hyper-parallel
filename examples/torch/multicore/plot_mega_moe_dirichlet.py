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
"""Plot route-matched native, MegaMoe B=0 and MegaMoe B=1 layer sweeps.

B denotes replica slots per rank. The shared horizontal coordinate is the
original home-destination skew, before replica placement. Only rendering needs
matplotlib; validation and aggregation use the standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VARIANTS = ("native", "b0", "b1")
GIB = 1 << 30
_LOGICAL_FIELDS = (
    "profile", "dirichlet_alpha", "dirichlet_seed", "expert_loads",
    "destination_loads", "realized_destination_skew", "receive_capacity", "baseline_wave_count",
)
_SOURCE_FIELDS = ("git_head", "worker_sha256", "adapter_sha256", "native_artifacts")
_STACK_DEFAULTS = {"layers": 1, "checkpoint": "none", "scope": "single expert layer", "optimizer": "none"}


@dataclass(frozen=True)
class CurveDataset:
    """Matched route statistics with the measurement provenance."""

    rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {value!r}.")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}.")
    return float(value)


def _logical_signature(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": point["shape"], "topology": point["topology"],
        "route": {name: point["route"][name] for name in _LOGICAL_FIELDS},
    }


def _validate_point(point: dict[str, Any], variant: str) -> tuple[float, int]:
    route = point["route"]
    alpha = _number(route["dirichlet_alpha"], "alpha", minimum=1e-300)
    seed = route["dirichlet_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    budget = int(variant == "b1")
    if route["replica_slots_per_rank"] != budget:
        raise ValueError(f"Replica budget mismatch for {variant}.")
    expected_backend = "native" if variant == "native" else "mega_moe"
    if point["backend"] != expected_backend:
        raise ValueError(f"Backend mismatch for {variant}.")
    samples = point["rank_max_fwd_bwd_ms"]["samples"]
    if not samples or len(samples) != point["measured_steps"]:
        raise ValueError("Latency sample count does not match measured_steps.")
    for sample in samples:
        _number(sample, "latency sample", minimum=1e-300)
    if not all(point["validation"].get(name) is True for name in ("gradients_present", "gradients_finite")):
        raise ValueError("Performance record did not validate gradients.")
    loads = route["destination_loads"]
    shape, topology = point["shape"], point["topology"]
    if len(loads) != topology["ep"] or topology["ep"] != topology["world_size"]:
        raise ValueError("Destination loads do not match EP topology.")
    if sum(loads) != shape["local_num_tokens"] * shape["top_k"] * topology["ep"]:
        raise ValueError("Destination loads do not preserve the routed row count.")
    skew = max(loads) / statistics.mean(loads)
    if not math.isclose(_number(route["realized_destination_skew"], "skew", minimum=1), skew):
        raise ValueError("Reported skew disagrees with original destination loads.")
    memory = point["memory"]
    if len(memory["per_rank"]) != topology["ep"]:
        raise ValueError("Memory rank count does not match EP topology.")
    totals = []
    for rank in memory["per_rank"]:
        allocated = _number(rank["peak_allocated_bytes"], "allocator peak")
        heap = _number(rank["symmetric_heap_bytes"], "SHMEM heap")
        total = _number(rank["conservative_total_peak_bytes"], "total peak HBM")
        if total != allocated + heap:
            raise ValueError("HBM accounting must be allocator peak plus SHMEM, without double count.")
        totals.append(total)
    if memory["rank_max"]["conservative_total_peak_bytes"] != max(totals):
        raise ValueError("HBM rank-max does not match per-rank totals.")
    return alpha, seed


def _load_variant(paths: Sequence[Path], variant: str) -> tuple[dict, dict]:
    records: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    header = None
    resolved = [path.resolve() for path in paths]
    if not resolved or len(resolved) != len(set(resolved)):
        raise ValueError(f"{variant} requires nonempty, distinct input files.")
    for path in resolved:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["variant"] != variant or payload["kind"] != "performance":
            raise ValueError(f"Expected {variant} performance sweep in {path}.")
        source = {name: payload["source"][name] for name in _SOURCE_FIELDS}
        current_header = {"source": source, "configuration": payload["configuration"]}
        config = payload["configuration"]
        layers = config.get("layers", 1)
        if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
            raise ValueError("layers must be a positive integer.")
        if header is not None and header != current_header:
            raise ValueError(f"Mixed source or configuration within {variant}.")
        header = current_header
        keys = set()
        for point in payload["points"]:
            key = _validate_point(point, variant)
            if (point.get("layers", 1) != layers
                    or point.get("checkpoint", "none") != config.get("checkpoint", "none")):
                raise ValueError("Layer count or checkpoint record disagrees with the configuration.")
            if (point["warmup_steps"] != payload["configuration"]["warmup_steps"]
                    or point["measured_steps"] != payload["configuration"]["measured_steps"]):
                raise ValueError("Timing record disagrees with the sweep configuration.")
            if key in keys:
                raise ValueError(f"Duplicate alpha/seed route {key} in {path}.")
            if records[key] and point["route"] != records[key][0]["route"]:
                raise ValueError(f"Route changed between repeated {variant} runs at {key}.")
            keys.add(key)
            records[key].append(point)
        if not keys:
            raise ValueError(f"Empty sweep in {path}.")
    if any(len(points) != len(paths) for points in records.values()):
        raise ValueError(f"Route sets differ between repeated {variant} runs.")
    return dict(records), header


def _aggregate(points: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [sample for point in points for sample in point["rank_max_fwd_bwd_ms"]["samples"]]
    ordered = sorted(samples)
    return {
        "latency_ms": statistics.median(samples),
        "p90_ms": ordered[math.ceil(len(ordered) * 0.9) - 1],
        "process_medians_ms": [
            statistics.median(point["rank_max_fwd_bwd_ms"]["samples"]) for point in points
        ],
        "samples": len(samples),
        "hbm_gib": max(
            point["memory"]["rank_max"]["conservative_total_peak_bytes"] for point in points
        ) / GIB,
        "planned_waves": points[0]["route"]["expected_wave_count"],
        "planned_physical_skew": points[0]["route"]["physical_destination_skew"],
    }


def load_results(inputs: Mapping[str, Sequence[Path]]) -> CurveDataset:
    """Validate and aggregate repeated, matched three-variant measurements.

    Args:
        inputs: Native, b0 and b1 JSON files, one file per independent run.

    Returns:
        Route-level pooled latency medians, maximum observed HBM and provenance.

    Raises:
        TypeError: A numeric measurement has an invalid type.
        ValueError: Routes, shapes, sources, measurement settings or accounting differ.
    """
    if set(inputs) != set(VARIANTS):
        raise ValueError("Supply exactly native, b0 and b1 inputs.")
    if len({len(paths) for paths in inputs.values()}) != 1:
        raise ValueError("All variants must have the same number of independent runs.")
    loaded = {variant: _load_variant(inputs[variant], variant) for variant in VARIANTS}
    reference_records, reference_header = loaded["native"]
    for variant, (records, header) in loaded.items():
        if records.keys() != reference_records.keys():
            raise ValueError(f"Route sets differ for {variant}.")
        if header["source"] != reference_header["source"]:
            raise ValueError(f"Source or native binary mismatch for {variant}.")
        config = header["configuration"]
        for key in ("capacity_factor", "warmup_steps", "measured_steps", "shuffle_seed", "optimizer_step",
                    "weight_refresh", "workspace_lifetime", "runtime_environment"):
            if config[key] != reference_header["configuration"][key]:
                raise ValueError(f"Measurement configuration mismatch for {variant}: {key}.")
        for key, default in _STACK_DEFAULTS.items():
            if config.get(key, default) != reference_header["configuration"].get(key, default):
                raise ValueError(f"Measurement configuration mismatch for {variant}: {key}.")
        if config["replica_slots_per_rank"] != int(variant == "b1"):
            raise ValueError(f"Configuration replica budget mismatch for {variant}.")
        if variant != "b1" and config["compact_physical_weights"]:
            raise ValueError("Compact physical weights are only meaningful for B=1.")
        if config.get("shared_physical_weights", False) and (variant != "b1" or config["compact_physical_weights"]):
            raise ValueError("Shared B requires B=1 and excludes compact physical weights.")
    rows = []
    workload = None
    for key, references in reference_records.items():
        signature = _logical_signature(references[0])
        current_workload = (signature["shape"], signature["topology"], signature["route"]["receive_capacity"])
        if workload is not None and workload != current_workload:
            raise ValueError("Mixed shapes, topology or receive capacity in the curve.")
        workload = current_workload
        row = {
            "alpha": key[0], "seed": key[1],
            "original_skew": signature["route"]["realized_destination_skew"],
            "baseline_waves": signature["route"]["baseline_wave_count"],
        }
        for variant, (records, _) in loaded.items():
            if any(_logical_signature(point) != signature for point in records[key]):
                raise ValueError(f"Logical route or shape mismatch at {key} for {variant}.")
            for name, value in _aggregate(records[key]).items():
                row[f"{variant}_{name}"] = value
        rows.append(row)
    rows.sort(key=lambda row: (row["original_skew"], row["alpha"], row["seed"]))
    reference = next(iter(reference_records.values()))[0]
    metadata = {
        "source": reference_header["source"],
        "shape": reference["shape"], "topology": reference["topology"],
        "capacity_skew": reference["route"]["receive_capacity"] / statistics.mean(
            reference["route"]["destination_loads"]
        ),
        "configuration": {variant: header["configuration"] for variant, (_, header) in loaded.items()},
        "input_files": {variant: [str(path.resolve()) for path in paths] for variant, paths in inputs.items()},
        "independent_runs_per_variant": len(inputs["native"]),
        "hbm_definition": "max_rank(allocator peak + full external SHMEM heap); maximum over repeated runs",
        "latency_definition": "median of pooled rank-max F+B samples; independent process medians retained separately",
        "x_definition": "original home-destination max(rows) / mean(rows), before replication",
        "line_definition": "connected route measurements, not a fitted performance model",
    }
    return CurveDataset(rows, metadata)


def _summary(dataset: CurveDataset) -> dict[str, Any]:
    by_wave = {}
    for wave in sorted({row["baseline_waves"] for row in dataset.rows}):
        selected = [row for row in dataset.rows if row["baseline_waves"] == wave]
        by_wave[str(wave)] = {
            "points": len(selected),
            **{
                variant: {
                    "median_latency_ms": statistics.median(row[f"{variant}_latency_ms"] for row in selected),
                    "median_hbm_gib": statistics.median(row[f"{variant}_hbm_gib"] for row in selected),
                    "median_latency_reduction_vs_native_pct": statistics.median(
                        100 * (1 - row[f"{variant}_latency_ms"] / row["native_latency_ms"]) for row in selected
                    ),
                }
                for variant in VARIANTS
            },
        }
    return {
        **dataset.metadata, "sample_count": len(dataset.rows),
        "original_skew_range": [dataset.rows[0]["original_skew"], dataset.rows[-1]["original_skew"]],
        "by_baseline_wave": by_wave,
    }


def _render(dataset: CurveDataset, output_dir: Path) -> None:
    # Matplotlib is optional: inspecting and validating measurements must not require it.
    from matplotlib import pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.switch_backend("Agg")
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    metadata = dataset.metadata
    shape, topology = metadata["shape"], metadata["topology"]
    compact = metadata["configuration"]["b1"]["compact_physical_weights"]
    shared = metadata["configuration"]["b1"].get("shared_physical_weights", False)
    storage = "shared across layers" if shared else ("compact" if compact else "snapshot")
    labels = ("Native", "MegaMoe B=0", f"MegaMoe B=1 ({storage})")
    config = metadata["configuration"]["native"]
    layers = config.get("layers", 1)
    checkpoint = config.get("checkpoint", "none")
    colors = ("#454545", "#0072B2", "#D55E00")
    skew = [row["original_skew"] for row in dataset.rows]
    for axis, metric in zip(axes, ("latency_ms", "hbm_gib")):
        for variant, label, color in zip(VARIANTS, labels, colors):
            axis.plot(skew, [row[f"{variant}_{metric}"] for row in dataset.rows],
                      color=color, label=label, linewidth=1.8, marker="o", markersize=3.5, alpha=0.9)
        axis.grid(True, color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", ncol=3, fontsize=10)
        axis.set_ylim(bottom=0)
        for wave in range(1, max(row["baseline_waves"] for row in dataset.rows)):
            boundary = wave * metadata["capacity_skew"]
            axis.axvline(boundary, color="#888888", linestyle=":", linewidth=0.9)
            axis.text(boundary + 0.025, 0.04, f"B=0: W{wave + 1}",
                      transform=axis.get_xaxis_transform(), fontsize=9, color="#666666")
    axes[0].set_ylabel("Median forward + backward (ms)")
    weight_state = "weights updated every step" if config["optimizer_step"] else "warmed weight cache"
    axes[0].set_title(
        f"{layers} MoE layer(s); checkpoint={checkpoint}; {weight_state}; optimizer excluded", fontsize=11,
    )
    axes[1].set_ylabel("Total peak HBM / rank (GiB)")
    axes[1].set_title("Allocator peak + full SHMEM heap; persistent high-watermarks retained", fontsize=11)
    axes[1].set_xlabel("Original routing skew S = max home-destination rows / mean rows (before replication)")
    figure.suptitle(
        f"Native / MegaMoe B=0 / B=1 across routing skew — {layers} MoE layer(s)\n"
        f"EP{topology['ep']}  |  {shape['local_num_tokens']} tokens/rank  |  "
        f"H{shape['hidden_size']}  I{shape['intermediate_size']}  E{shape['num_experts']}  "
        f"K{shape['top_k']}  {shape['dtype']}", fontsize=14,
    )
    figure.text(
        0.5, 0.015,
        f"{len(dataset.rows)} matched routes; {metadata['independent_runs_per_variant']} fresh processes/variant; "
        f"{config['warmup_steps']} warmup + {config['measured_steps']} measured steps/process/route.\n"
        "B = replica slots/rank. Lines connect measured route medians, not fitted predictions. "
        "HBM is rank-max, not an eight-card sum.",
        ha="center", fontsize=9, color="#555555",
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.94))
    for extension in ("png", "svg"):
        figure.savefig(output_dir / f"dirichlet_skew_latency_hbm.{extension}", dpi=180)
    plt.close(figure)


def write_results(dataset: CurveDataset, output_dir: Path, *, render: bool = True) -> None:
    """Write a CSV, provenance summary and optionally the two-panel PNG/SVG.

    Args:
        dataset: Validated matched measurements returned by load_results.
        output_dir: Destination directory, created if needed.
        render: Whether to render figures using the optional matplotlib dependency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dirichlet_skew_latency_hbm_points.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dataset.rows[0]))
        writer.writeheader()
        writer.writerows(dataset.rows)
    (output_dir / "dirichlet_skew_summary.json").write_text(
        json.dumps(_summary(dataset), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if render:
        _render(dataset, output_dir)


def main() -> None:
    """Replot three route-matched sweep variants without initializing an accelerator."""
    parser = argparse.ArgumentParser(description=__doc__)
    for variant in VARIANTS:
        parser.add_argument(f"--{variant}", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-render", action="store_true", help="Validate/export without requiring matplotlib.")
    args = parser.parse_args()
    dataset = load_results({variant: getattr(args, variant) for variant in VARIANTS})
    write_results(dataset, args.output_dir, render=not args.no_render)
    print(f"Wrote {len(dataset.rows)} matched routes to {args.output_dir}")


if __name__ == "__main__":
    main()
