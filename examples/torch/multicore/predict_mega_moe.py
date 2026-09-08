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
"""Predict or sweep MegaMoe latency/HBM entirely on CPU, using explicit calibration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from examples.torch.multicore.megamoe_simulator.cost import Hardware, Options
from examples.torch.multicore.megamoe_simulator.estimate import predict
from examples.torch.multicore.megamoe_simulator.model import Training
from examples.torch.multicore.megamoe_simulator.network import Fabric
from examples.torch.multicore.megamoe_simulator.profile import (
    calibrate,
    calibrate_descriptor_service,
    calibrate_transfer_windows,
    hardware_dict,
    summarize_tasks,
    summarize_trace,
)
from examples.torch.multicore.megamoe_simulator.workload import Shape, make_counts


def main() -> None:
    """Produce machine-readable assumptions, predictions and feasibility filters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=64)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=1792)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--slots", default="0,1,2,3")
    parser.add_argument("--capacity-factors", default="1.25,1.5")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--training", type=Path, help="Training JSON overrides layers, steps, checkpoint, pool and updates"
    )
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--inter-gbps", type=float, default=25.0)
    parser.add_argument("--inter-latency-us", type=float, default=10.0)
    parser.add_argument("--routing", choices=("balanced", "hotspot", "dirichlet"), default="hotspot")
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument(
        "--schedule-audit", type=Path, help="Results from audit_mega_moe_plan; checks exact shape/options"
    )
    parser.add_argument("--options", type=Path)
    parser.add_argument(
        "--calibrate",
        type=Path,
        help="JSON with train samples and source/device identity",
    )
    parser.add_argument("--task-profile", type=Path, help="Summarize one collected task profile")
    parser.add_argument(
        "--descriptor-profiles", type=Path, nargs="+", help="Fit decoder service from raw rank profiles",
    )
    parser.add_argument("--train-pairs", default="10000:42", help="Route identifiers admitted into the descriptor fit")
    parser.add_argument(
        "--trace",
        type=Path,
        help="Summarize a Chrome trace without adding nested scopes",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh result path")
    if args.ranks <= 0 or args.ranks_per_node <= 0:
        parser.error("ranks and ranks-per-node must be positive")
    hardware = Hardware(**json.loads(args.hardware.read_text())) if args.hardware else Hardware()
    if args.calibrate:
        calibration = json.loads(args.calibrate.read_text())
        fitted = calibrate(calibration["samples"], hardware, identity=calibration["identity"])
        if calibration.get("transfer_samples"):
            fitted = calibrate_transfer_windows(
                calibration["transfer_samples"], fitted, identity=calibration["transfer_identity"]
            )
        output = hardware_dict(fitted)
    elif args.descriptor_profiles:
        output = hardware_dict(calibrate_descriptor_service(
            [json.loads(path.read_text()) for path in args.descriptor_profiles], hardware,
            train_pairs=set(args.train_pairs.split(",")),
        ))
    elif args.task_profile:
        output = summarize_tasks(json.loads(args.task_profile.read_text()))
    elif args.trace:
        output = summarize_trace(json.loads(args.trace.read_text()))
    else:
        shape = Shape(args.experts, args.tokens, args.top_k, args.hidden, args.intermediate)
        counts = (
            json.loads(args.counts.read_text())["counts"]
            if args.counts
            else make_counts(shape, args.ranks, routing=args.routing)
        )
        fabric = Fabric(
            tuple(rank // args.ranks_per_node for rank in range(args.ranks)),
            inter_gbps=args.inter_gbps,
            inter_latency_us=args.inter_latency_us,
        )
        options = Options(**json.loads(args.options.read_text())) if args.options else Options()
        training = (
            Training(**json.loads(args.training.read_text()))
            if args.training
            else Training(layers=args.layers, steps=1)
        )
        audits = json.loads(args.schedule_audit.read_text()) if args.schedule_audit else []
        results = []
        for factor in map(float, args.capacity_factors.split(",")):
            for slots in map(int, args.slots.split(",")):
                selected_shape = replace(shape, capacity_factor=factor)
                matches = [
                    row
                    for row in audits
                    if row["ranks"] == args.ranks
                    and row.get("num_cube_cores", 24) == hardware.vector_workers
                    and row["slots"] == slots
                    and row["shape"] == vars(selected_shape)
                    and all(row["options"][key] == getattr(options, key) for key in ("inplace_backward", "atomic_dw"))
                ]
                if args.schedule_audit and len(matches) != 1:
                    raise ValueError("schedule audit must contain exactly one entry matching each shape and option set")
                selected_options = (
                    replace(options, plan_storage_bytes=matches[0].get(
                        "total_warmed_plan_upper_bound_bytes", matches[0].get("total_plan_tensor_bytes", 0),
                    ))
                    if matches
                    else options
                )
                result = predict(
                    selected_shape,
                    counts,
                    slots,
                    fabric,
                    hardware,
                    training,
                    selected_options,
                )
                result["runtime_audit"] = matches[0] if matches else {"buildable": None, "scope": "not audited"}
                results.append(result)
        feasible = [index for index, result in enumerate(results) if result["memory"]["fits_budget_with_reserve"]]
        ranking = sorted(feasible, key=lambda index: results[index]["latency_ms"])
        runnable = [index for index in ranking if results[index]["runtime_audit"]["buildable"] is True]
        output = {
            "results": results,
            "ranking_with_audited_buildability": runnable,
            "ranking_within_memory_budget": ranking,
            "recommendation_scope": "scenario ranking; check descriptor_audit and calibration coverage before running",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
