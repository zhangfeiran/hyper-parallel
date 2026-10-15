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
"""Audit full-network rank-max timing, actual rows and node HBM channels."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

_LABELS = ("native", "push", "pull")


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _physical_hbm(folder: Path, start: str, finish: str) -> dict[str, Any]:
    try:
        return _read_physical_hbm(folder, start, finish)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'eligible': False, 'reason': f'Invalid HBM artifacts: {type(error).__name__}: {error}'}


def _read_physical_hbm(folder: Path, start: str, finish: str) -> dict[str, Any]:
    monitor = folder / "hbm_summary.json"
    samples = folder / "hbm_samples.jsonl"
    if not monitor.exists() or not samples.exists():
        return {"eligible": False, "reason": "Missing HBM monitor artifacts"}
    quality = _read(monitor)
    first, last = datetime.fromisoformat(start), datetime.fromisoformat(finish)
    valid = []
    for line in samples.read_text().splitlines():
        row = json.loads(line)
        stamp = datetime.fromisoformat(row["time"])
        if row["valid"] and first <= stamp <= last:
            valid.append((stamp, max(device["used_mib"] for device in row["devices"].values())))
    boundaries = [first, *(stamp for stamp, _ in valid), last]
    gap = max((b - a).total_seconds() for a, b in zip(boundaries, boundaries[1:]))
    eligible = (bool(valid) and quality["monitoring_sufficient"] and gap <= 30
                and not quality["foreign_pids"] and not quality["unresolved_pids"]
                and not quality["unhealthy_states"]
                and (folder / "monitor_exit_status.txt").exists()
                and (folder / "monitor_exit_status.txt").read_text().strip() == "0")
    return {"eligible": eligible, "quality": quality, "measured_valid_samples": len(valid),
            "measured_max_gap_seconds": gap,
            "measured_peak_mib": max((used for _, used in valid), default=None),
            "whole_run_peak_mib": quality["max_device_peak_mib"]}


def _timing_stats(values: list[float]) -> dict:
    return {'rank_max_ms_by_step': values, 'mean_ms': statistics.mean(values),
            'median_ms': statistics.median(values), 'minimum_ms': min(values), 'maximum_ms': max(values)}


def _moe_statistics(records: list[dict], warmup: int) -> dict:
    rows = [result['steps_detail'][warmup:] for result in records]
    observed = [row.get('moe_timing') for rank_rows in rows for row in rank_rows]
    if not all(observed):
        return {'eligible': False, 'reason': 'Missing MoE event records'}
    layers = sorted(observed[0]['layers'])
    if not layers or any(sorted(row['layers']) != layers or row['incomplete_intervals'] for row in observed):
        return {'eligible': False, 'reason': 'Missing layers or incomplete event pairs'}
    phases = sorted({phase for row in observed for layer in row['layers'].values() for phase in layer})
    per_layer, phase_sums = {}, {}
    for phase in phases:
        for row in observed:
            if any(phase not in row['layers'][layer] for layer in layers):
                return {'eligible': False, 'reason': f'Phase {phase} missing on some layers/ranks/steps'}
            for layer in layers:
                interval = row['layers'][layer][phase]
                if not math.isfinite(interval['current_stream_ms']) or interval['current_stream_ms'] < 0:
                    raise RuntimeError('Invalid MoE event duration')
        for layer in layers:
            values = [max(rank_rows[index]['moe_timing']['layers'][layer][phase]['current_stream_ms']
                          for rank_rows in rows) for index in range(len(rows[0]))]
            per_layer.setdefault(layer, {})[phase] = _timing_stats(values)
        # Sum within each rank first; never sum independent per-layer rank maxima.
        values = [max(sum(rank_rows[index]['moe_timing']['layers'][layer][phase]['current_stream_ms']
                          for layer in layers) for rank_rows in rows) for index in range(len(rows[0]))]
        phase_sums[phase] = _timing_stats(values)
    return {'eligible': True, 'method': 'current_stream_events', 'layers': per_layer, 'phase_sums': phase_sums,
            'scope': 'MoE includes router, routed/shared experts and replica paths; backward ends at input grad',
            'phases_additive': False, 'backward_may_include_replay': True}


def _summarize_case(run_root: Path, node_root: Path, nodes: int, cards_per_node: int,
                    steps: int, warmup_steps: int, label: str) -> tuple[dict, list]:
    world = nodes * cards_per_node
    records = []
    for rank in range(world):
        folder = run_root / label / f"rank{rank}"
        result = _read(folder / "result.json")
        rows = result["steps_detail"]
        if run_root.name in ("dynamic", "packed") and result.get("packing_mode", run_root.name) != run_root.name:
            raise RuntimeError(f"Packing mode does not match result directory: {folder}")
        expected_backend = "owner_ep" if label == "native" else "megamoe"
        if result.get("backend", expected_backend) != expected_backend:
            raise RuntimeError(f"Backend does not match case label: {folder}")
        if result["rank"] != rank or result["world_size"] != world or len(rows) != steps:
            raise RuntimeError(f"Incomplete or wrong-rank result: {folder}")
        if [row["step"] for row in rows] != list(range(1, steps + 1)):
            raise RuntimeError(f"Wrong step sequence: {folder}")
        if any(row["phase"] != ("warmup" if index < warmup_steps else "measured")
               for index, row in enumerate(rows)):
            raise RuntimeError(f"Wrong measured window: {folder}")
        if not result["finite"] or not all(math.isfinite(row[key]) for row in rows for key in ("loss", "grad_norm")):
            raise RuntimeError(f"Nonfinite training scalars: {folder}")
        saved = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        if saved != rows:
            raise RuntimeError(f"Step artifacts disagree: {folder}")
        records.append(result)
    times = []
    for step in range(warmup_steps, steps):
        rows = [result["steps_detail"][step] for result in records]
        recorded = rows[0]["max_rank_seconds"]
        if any(row["max_rank_seconds"] != recorded for row in rows):
            raise RuntimeError(f"Rank-max timing disagrees: {label} step {step + 1}")
        if abs(max(row["local_seconds"] for row in rows) - recorded) > 1e-4:
            raise RuntimeError(f"Rank-max timing does not match local timers: {label} step {step + 1}")
        times.append(recorded)
    node_hbm = []
    for node in range(nodes):
        folder = node_root / f"node_{node}" / label
        complete = _read(folder / "case_complete.json")
        if complete["training_exit_status"] != 0 or not complete.get("cleanup_ok", True):
            raise RuntimeError(f"Training failed: {folder}")
        local_rows = records[node * cards_per_node]["steps_detail"]
        node_hbm.append(_physical_hbm(folder, local_rows[warmup_steps]["started_at"],
                                      local_rows[-1]["finished_at"]))
    route_rows = [route for result in records for row in result["steps_detail"][warmup_steps:]
                  for route in row["replica_routes"].values()]
    routes = None
    if route_rows:
        routes = {
            "home_receive_max": max(max(row["home_destination_rows"]) for row in route_rows),
            "balanced_receive_max": max(max(row["balanced_destination_rows"]) for row in route_rows),
            "weight_transfers_min_max": [min(row["weight_transfers"] for row in route_rows),
                                         max(row["weight_transfers"] for row in route_rows)],
            "active_guests_per_rank_min_max": [min(min(row["active_guest_slots_by_ep_rank"]) for row in route_rows),
                                               max(max(row["active_guest_slots_by_ep_rank"]) for row in route_rows)],
        }
    actual_tokens = [value for result in records for row in result["steps_detail"][warmup_steps:]
                     for value in row["moe_input_rows"].values()]
    case = {
        "status": "complete", "moe_timing": _moe_statistics(records, warmup_steps),
        "completed_ranks": world, "steps_per_rank": steps, "all_finite": True,
        "measured_rank_max_seconds": times,
        "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
        "minimum_seconds": min(times), "maximum_seconds": max(times),
        "actual_token_min_max": [min(actual_tokens), max(actual_tokens)],
        "peak_allocated_bytes": max(result["peak_allocated_bytes"] for result in records),
        "peak_reserved_bytes": max(result["peak_reserved_bytes"] for result in records),
        "measured_peak_allocated_bytes": max(result["measurement"]["peak_allocated_bytes"] for result in records),
        "measured_peak_reserved_bytes": max(result["measurement"]["peak_reserved_bytes"] for result in records),
        "shmem_rank0": records[0]["shmem_state"], "route_summary": routes,
        "node_hbm": node_hbm, "hbm_eligible": all(node["eligible"] for node in node_hbm),
        "measured_physical_peak_mib": max((node["measured_peak_mib"] for node in node_hbm
                                           if node.get("measured_peak_mib") is not None), default=None),
        "whole_run_physical_peak_mib": max((node["whole_run_peak_mib"] for node in node_hbm
                                            if node.get("whole_run_peak_mib") is not None), default=None),
    }
    fingerprints = [{"initialization": _read(run_root / label / f"rank{rank}" / "initialization.json"),
                           "rows": [(row["loss"], row["grad_norm"], row["moe_input_rows"])
                                    for row in result["steps_detail"]]} for rank, result in enumerate(records)]
    return case, fingerprints


def _compare(cases: dict, comparisons: dict, label: str) -> dict:
    matching = all(a["initialization"] == b["initialization"]
                   for a, b in zip(comparisons["native"], comparisons[label]))
    token_match = all(x[2] == y[2] for a, b in zip(comparisons["native"], comparisons[label])
                      for x, y in zip(a["rows"], b["rows"]))
    differences = [max(abs(x[index] - y[index]) for a, b in zip(comparisons["native"], comparisons[label])
                       for x, y in zip(a["rows"], b["rows"])) for index in (0, 1)]
    comparison = {
        "initialization_match": matching, "actual_token_rows_match": token_match,
        "maximum_loss_difference": differences[0], "maximum_grad_norm_difference": differences[1],
        "observed_native_speed_ratio": cases["native"]["mean_seconds"] / cases[label]["mean_seconds"],
        "performance_eligible": matching and token_match and cases[label]["hbm_eligible"]
                                and cases["native"]["hbm_eligible"],
    }
    comparison['available'] = True
    comparison['numerical_validation'] = 'finite_scalars_and_differences_reported_tensor_parity_not_checked'
    native_moe, candidate_moe = cases['native']['moe_timing'], cases[label]['moe_timing']
    comparison['moe_current_stream_speed_ratios'] = {}
    if native_moe['eligible'] and candidate_moe['eligible'] and matching and token_match:
        for phase in native_moe['phase_sums']:
            if phase in candidate_moe['phase_sums']:
                denominator = candidate_moe['phase_sums'][phase]['mean_ms']
                if denominator > 0:
                    comparison['moe_current_stream_speed_ratios'][phase] = (
                        native_moe['phase_sums'][phase]['mean_ms'] / denominator)
    return comparison


def summarize(run_root: Path, nodes: int, cards_per_node: int, steps: int,
              warmup_steps: int, *, allow_partial: bool = False, node_root: Path | None = None) -> dict[str, Any]:
    """Audit complete cases, retaining valid pairs when another backend fails."""
    report: dict[str, Any] = {'world_size': nodes * cards_per_node, 'warmup_steps': warmup_steps,
                              'measured_steps': steps - warmup_steps, 'cases': {}, 'comparisons': {}}
    fingerprints = {}
    for label in _LABELS:
        try:
            case, signature = _summarize_case(run_root, node_root or run_root / 'nodes', nodes,
                                               cards_per_node, steps, warmup_steps, label)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            if not allow_partial:
                raise
            report['cases'][label] = {'status': 'incomplete_or_failed',
                                       'reason': f'{type(error).__name__}: {error}'}
        else:
            report['cases'][label] = case
            fingerprints[label] = signature
    for label in ('push', 'pull'):
        report['comparisons'][f'{label}_vs_native'] = (
            _compare(report['cases'], fingerprints, label) if 'native' in fingerprints and label in fingerprints
            else {'available': False, 'performance_eligible': False, 'reason': 'A required case is incomplete'})
    report['complete'] = len(fingerprints) == len(_LABELS)
    return report


def summarize_matrix(run_root: Path, nodes: int, cards_per_node: int,
                     steps: int, warmup_steps: int) -> dict:
    """Keep dynamic and packed comparisons separate and publish completed groups early."""
    groups = {}
    for packing in ('dynamic', 'packed'):
        # Node artifacts share a root but retain the same grouping as rank artifacts.
        node_root = run_root / packing / 'nodes'
        groups[packing] = summarize(run_root / packing, nodes, cards_per_node,
                                    steps, warmup_steps, allow_partial=True, node_root=node_root)
        (run_root / packing).mkdir(exist_ok=True, parents=True)
        _publish(run_root / packing / 'summary.json', groups[packing])
    _publish_tables(run_root, groups)
    return {'groups': groups, 'complete': all(group['complete'] for group in groups.values()),
            'case_order': [f'{packing}/{label}' for packing in groups for label in _LABELS]}


def _publish_tables(run_root: Path, groups: dict) -> None:
    fields = ['packing', 'case', 'status', 'tokens_min', 'tokens_max', 'step_mean_ms', 'step_speedup_vs_native',
              'moe_forward_ms', 'moe_recompute_ms', 'moe_backward_to_input_grad_ms',
              'moe_forward_speedup', 'moe_recompute_speedup', 'moe_backward_to_input_grad_speedup',
              'hbm_eligible', 'physical_peak_mib', 'performance_eligible', 'loss_max_abs_diff', 'grad_max_abs_diff']
    destination = run_root / 'summary.tsv'
    temporary = destination.with_suffix('.tsv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, dialect='excel-tab')
        writer.writeheader()
        for packing, report in groups.items():
            for label, case in report['cases'].items():
                row = {'packing': packing, 'case': label, 'status': case['status']}
                if case['status'] == 'complete':
                    native = label == 'native'
                    comparison = report['comparisons'].get(f'{label}_vs_native', {})
                    row.update(tokens_min=case['actual_token_min_max'][0], tokens_max=case['actual_token_min_max'][1],
                               step_mean_ms=case['mean_seconds'] * 1000,
                               step_speedup_vs_native=1.0 if native else comparison.get('observed_native_speed_ratio'),
                               hbm_eligible=case['hbm_eligible'], physical_peak_mib=case['measured_physical_peak_mib'],
                               performance_eligible=comparison.get('performance_eligible', False),
                               loss_max_abs_diff=comparison.get('maximum_loss_difference'),
                               grad_max_abs_diff=comparison.get('maximum_grad_norm_difference'))
                    for phase, column in (('forward', 'forward'), ('recompute_forward', 'recompute'),
                                          ('backward_to_input_grad', 'backward_to_input_grad')):
                        timing = case['moe_timing'].get('phase_sums', {}).get(phase, {})
                        row[f'moe_{column}_ms'] = timing.get('mean_ms')
                        row[f'moe_{column}_speedup'] = (1.0 if native and timing else
                            comparison.get('moe_current_stream_speed_ratios', {}).get(phase))
                writer.writerow(row)
    temporary.replace(destination)
    destination = run_root / 'moe_layers.tsv'
    temporary = destination.with_suffix('.tsv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, dialect='excel-tab')
        writer.writerow(['packing', 'case', 'layer', 'phase', 'rank_max_mean_ms', 'rank_max_median_ms'])
        for packing, report in groups.items():
            for label, case in report['cases'].items():
                for layer, phases in case.get('moe_timing', {}).get('layers', {}).items():
                    for phase, values in phases.items():
                        writer.writerow([packing, label, layer, phase, values['mean_ms'], values['median_ms']])
    temporary.replace(destination)


def _publish(destination: Path, value: dict) -> None:
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)


def main() -> None:
    """Wait for node cleanup sentinels and atomically publish the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--cards-per-node", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--matrix", action="store_true", help="Publish both groups including partial results")
    args = parser.parse_args()
    if min(args.nodes, args.cards_per_node, args.steps) < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("Invalid topology or measured window")
    if args.matrix:
        result = summarize_matrix(args.run_root, args.nodes, args.cards_per_node, args.steps, args.warmup_steps)
        _publish(args.run_root / "summary.json", result)
        print(json.dumps(result, indent=2))
        return
    deadline = time.monotonic() + 120
    pending = [args.run_root / "nodes" / f"node_{node}" / label / "case_complete.json"
               for node in range(args.nodes) for label in _LABELS]
    while not all(path.exists() for path in pending):
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for node case cleanup artifacts")
        time.sleep(1)
    result = summarize(args.run_root, args.nodes, args.cards_per_node, args.steps, args.warmup_steps)
    destination = args.run_root / "summary.json"
    temporary = args.run_root / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({label: {key: result["cases"][label][key]
                             for key in ("completed_ranks", "mean_seconds", "actual_token_min_max", "hbm_eligible")}
                      for label in _LABELS}, indent=2))


if __name__ == "__main__":
    main()
