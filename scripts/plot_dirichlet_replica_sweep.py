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
"""Validate all-rank sweeps and render four measured latency/HBM curves."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
VARIANTS = ("native-b0", "native-b1", "megamoe-b0", "megamoe-b1")
LABELS = ("Native B=0", "Native B=1", "MegaMoe B=0", "MegaMoe B=1")
COLORS = ("#555555", "#8c56a2", "#0077b6", "#ed7c12")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_points(directory):
    ranks, identities = [], []
    terminal = _read(directory / "terminal.json")
    if terminal["returncode"]:
        raise ValueError(f"Run failed: {directory}")
    for rank in range(16):
        data = _read(directory / f"rank{rank}" / "complete.json")
        if data["points"] != 55:
            raise ValueError(f"Expected 55 routes at {directory}/rank{rank}")
        ranks.append({(point["alpha"], point["seed"]): point for point in data["results"]})
        identities.append(_read(directory / f"rank{rank}" / "identity.json"))
    monitors = [_read_line(line) for line in (directory / "npu-monitor.jsonl").read_text().splitlines()]
    if any(point["foreign_pids"] or not point["healthy"] or point["returncode"] for point in monitors):
        raise ValueError(f"Unhealthy, contaminated or incomplete physical monitor: {directory}")
    for rank, points in enumerate(ranks):
        if points.keys() != ranks[0].keys():
            raise ValueError(f"Rank {rank} route matrix differs")
        for key, point in points.items():
            if not point["finite_validation"] or point["route_sha256"] != ranks[0][key]["route_sha256"]:
                raise ValueError(f"Route validation failed: {directory}, {rank}, {key}")
            if point["step_ms"]["samples"] != ranks[0][key]["step_ms"]["samples"]:
                raise ValueError("Rank-max timings differ across ranks")
    return ranks, identities, monitors


def _read_line(line):
    return json.loads(line)


def _physical(points, monitors):
    windows = [window for point in points for window in point["windows"]]
    samples = [sample for sample in monitors if any(sample["monotonic_ns"] <= window["end_ns"] and
                                                    sample["end_ns"] >= window["start_ns"] for window in windows)]
    if not samples or any(len(sample["processes"]) != 16 for sample in samples):
        raise ValueError("Every route needs physical HBM samples with 16 owned worker processes")
    return {"physical_card_hbm_gib": max(chip["hbm_mb"] for sample in samples for chip in sample["chips"]) / 1024,
            "physical_process_hbm_gib": max(process["memory_mb"] for sample in samples
                                            for process in sample["processes"]) / 1024,
            "physical_samples": len(samples)}


def aggregate(root: Path) -> tuple[list[dict], dict]:
    """Require two clean, matched fresh runs per variant and all 16 ranks.

    Args:
        root: Directory containing repeat1/repeat2 variant runs.

    Returns:
        Route statistics and the common execution configuration.
    """
    loaded = {(repeat, variant): _run_points(root / f"repeat{repeat}-{variant}")
              for repeat in (1, 2) for variant in VARIANTS}
    baseline = loaded[1, "native-b0"][1]
    native_payload = loaded[1, "megamoe-b0"][1]
    configuration = baseline[0]["configuration"]
    for (_, variant), (_, identities, _) in loaded.items():
        for rank, identity in enumerate(identities):
            for field in ("initial_weights", "input_sha256", "gradient_sha256", "sha", "source_manifest_sha256",
                          "torch", "torch_npu", "cann", "measurement_scripts"):
                if identity[field] != baseline[rank][field]:
                    raise ValueError(f"Unmatched {field} for {variant}, rank {rank}")
            for field in ("layers", "tokens", "experts", "hidden", "intermediate", "top_k", "warmup", "measured"):
                if identity["configuration"][field] != configuration[field]:
                    raise ValueError(f"Unmatched configuration {field}")
            if variant.startswith("megamoe"):
                for field in ("adapter", "kernels"):
                    if identity[field] != native_payload[rank][field]:
                        raise ValueError(f"Unmatched native payload {field}, rank {rank}")
    keys = loaded[1, "native-b0"][0][0].keys()
    rows = []
    for key in keys:
        baseline_point = loaded[1, "native-b0"][0][0][key]
        row = {"alpha": key[0], "seed": key[1], "skew": baseline_point["skew"],
               "route_sha256": baseline_point["route_sha256"]}
        for variant in VARIANTS:
            latencies, hbm, physical_card, physical_process, transfers = [], [], [], [], []
            medians = []
            for repeat in (1, 2):
                ranks, _, monitors = loaded[repeat, variant]
                points = [rank[key] for rank in ranks]
                if any(point["route_sha256"] != row["route_sha256"] for point in points):
                    raise ValueError(f"Route mismatch {key} {variant}")
                physical = _physical(points, monitors)
                latencies.extend(points[0]["step_ms"]["samples"])
                medians.append(points[0]["step_ms"]["median"])
                hbm.append(max(point["memory"]["total_peak_bytes"] for point in points) / 2**30)
                physical_card.append(physical["physical_card_hbm_gib"])
                physical_process.append(physical["physical_process_hbm_gib"])
                if points[0]["replica_plan"] is not None:
                    transfers.append(points[0]["replica_plan"]["transfers"])
            row.update({f"{variant}_step_ms": statistics.median(latencies),
                        f"{variant}_repeat1_ms": medians[0], f"{variant}_repeat2_ms": medians[1],
                        f"{variant}_hbm_gib": statistics.median(hbm),
                        f"{variant}_physical_card_hbm_gib": max(physical_card),
                        f"{variant}_physical_process_hbm_gib": max(physical_process),
                        f"{variant}_replica_transfers": max(transfers, default=0)})
        rows.append(row)
    return sorted(rows, key=lambda row: row["skew"]), configuration


def render(rows: list[dict], configuration: dict, output: Path) -> None:
    """Write measured curves and a CSV with allocator and physical HBM separately."""
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        x_values = [row["skew"] for row in rows]
        for axis, field in zip(axes, ("step_ms", "hbm_gib", "physical_card_hbm_gib")):
            axis.plot(x_values, [row[f"{variant}_{field}"] for row in rows], color=color, marker="o",
                      markersize=3, label=label, linewidth=1.8)
    figure.suptitle(f"Native / MegaMoe B=0 / B=1 across routing skew — {configuration['layers']} MoE layers\n"
                   f"EP16 | {configuration['tokens']} tokens/rank | H{configuration['hidden']} "
                   f"I{configuration['intermediate']} E{configuration['experts']} K{configuration['top_k']} BF16",
                   fontsize=15)
    axes[0].set_title("Reentrant checkpoint; weights updated every step; full AdamW update included")
    axes[0].set_ylabel("Median rank-max training step (ms)")
    axes[1].set_title("Allocator peak + full external SHMEM heap; high-watermarks retained")
    axes[1].set_ylabel("Accounted HBM / rank (GiB)")
    axes[2].set_title("Physical card HBM sampled during measured steps; includes device/runtime baseline")
    axes[2].set_ylabel("Physical HBM / chip (GiB)")
    axes[2].set_xlabel("Original routing skew S = max home-destination rows / mean rows (before replication)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left", ncol=4)
    figure.text(0.5, 0.015, "55 matched routes; 2 fresh processes/variant; 3 warmup + 7 measured steps/process/route.\n"
                "B = replica slots/rank, shared across layers. Measured route medians; HBM is rank-max.\n"
                "Scope: independent MoE layers, forward + backward + AdamW; no attention or router network.",
                ha="center", fontsize=10, color="#555555")
    figure.tight_layout(rect=(0, 0.075, 1, 0.94))
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    """Render only a complete matrix with matching routes and uncontaminated monitoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, configuration = aggregate(args.input)
    render(rows, configuration, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"configuration": configuration, "rows": rows}, indent=2)
                                               + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
