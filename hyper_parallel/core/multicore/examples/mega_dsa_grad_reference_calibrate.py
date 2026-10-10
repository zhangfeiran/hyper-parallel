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
"""Measure stock BF16 gradient error independently of the local CP candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

import hyper_parallel
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _difference
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import _tile
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_grad_validate import _SCALE, _stock
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.reference import sparse_attention_reference
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    fit_reference_budget, gradient_error_metrics, reference_budget_passes, validate_reference_seed_sets,
)

_SEEDS = (20261011, 20261012, 20261013, 20261014)
_NAMES = ("query", "compressed_kv", "query_rope", "key_rope")
_MODES = ("owner", "interior", "first_sequence", "singleton", "empty")
_PROFILE_FIT_SEEDS = (20261020, 20261021, 20261022)
_PROFILE_HELD_OUT_SEEDS = (20261023, 20261024)
_PROFILE_MARGIN = 2.
_PROFILE_ZERO_ATOL = 2e-5


def _selection(meta, seed=None):
    selected = torch.full((meta.global_valid_queries, 2048), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    for token in range(meta.global_valid_queries):
        sequence, position = meta.sequence_position(token)
        ids = torch.arange(meta.global_cu_seqlens[sequence], token + 1)
        if generator is not None:
            ids = ids[torch.randperm(len(ids), generator=generator)]
        selected[token, :position + 1] = ids
    return selected


def _separate_oracle(states, selected, cotangent, rows):
    query, compressed, query_rope, key_rope = states
    shapes = (query.shape, compressed.shape, compressed.shape, query_rope.shape, key_rope.shape)
    if not rows:
        return tuple(torch.zeros(shape, dtype=torch.float64) for shape in shapes)
    inputs = tuple(tensor.double().clone().requires_grad_() for tensor in
                   (query[list(rows)], compressed, compressed, query_rope[list(rows)], key_rope))
    q_state, key, value, q_rope, k_rope = inputs
    indices = selected[list(rows)].long()
    valid = indices >= 0
    safe = indices.masked_fill(~valid, len(key))
    selected_key, selected_value, selected_rope = (
        torch.cat((tensor, tensor.new_zeros((1, tensor.shape[-1]))))[safe] for tensor in (key, value, k_rope))
    scores = (torch.einsum("thc,tkc->thk", q_state, selected_key)
              + torch.einsum("thd,tkd->thk", q_rope, selected_rope)) * _SCALE
    probability = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
    output = torch.einsum("thk,tkc->thc", probability, selected_value)
    gradients = torch.autograd.grad(output, inputs, grad_outputs=cotangent[list(rows)].double())
    full = [torch.zeros(shape, dtype=torch.float64) for shape in shapes]
    full[0][list(rows)], full[3][list(rows)] = gradients[0], gradients[3]
    full[1], full[2], full[4] = gradients[1], gradients[2], gradients[4]
    return tuple(full)


def _fixture(meta, heads, seed, mode):
    generator = torch.Generator().manual_seed(seed + heads)
    total = meta.global_valid_queries
    shapes = ((total, heads, 512), (total, 512), (total, heads, 64), (total, 64))
    states = tuple((torch.randn(shape, generator=generator) * .1).bfloat16() for shape in shapes)
    cotangent = torch.zeros_like(states[0])
    rows = (tuple(token for token in meta.q_global_ids if token in meta.global_cu_seqlens[:-1])
            if mode == "singleton" else _tile(meta, mode).q_global_ids)
    dy = (torch.randn((len(rows), heads, 512), generator=generator) * .1).bfloat16()
    cotangent[list(rows)] = (dy * (meta.cp_rank + 1)).bfloat16()
    return states, cotangent, rows


def _shared_oracle(meta, states, selected, cotangent, rows):
    full = tuple(torch.zeros_like(tensor, dtype=torch.float64) for tensor in states)
    if not rows:
        return full
    query, compressed, query_rope, key_rope = states
    inputs = tuple(tensor.double().clone().requires_grad_() for tensor in
                   (query[list(rows)], compressed, query_rope[list(rows)], key_rope))
    output, _ = sparse_attention_reference(*inputs, selected[list(rows)], replace(meta, q_global_ids=rows),
                                         attention_scale=_SCALE)
    gradients = torch.autograd.grad(output, inputs, grad_outputs=cotangent[list(rows)].double())
    full[0][list(rows)], full[2][list(rows)] = gradients[0], gradients[2]
    return full[0], gradients[1], full[2], gradients[3]


def _measure(meta, layout, native, selected, heads, seed, mode):
    states, cotangent, rows = _fixture(meta, heads, seed, mode)
    width = max(end - start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
    oracle_selection = selected[:, :width]
    separated = _separate_oracle(states, oracle_selection, cotangent, rows)
    base_meta = layout.batch_meta
    shared = _shared_oracle(base_meta, states, oracle_selection, cotangent, rows)
    merged_oracle = (separated[0], separated[1] + separated[2], separated[3], separated[4])
    for actual, expected in zip(merged_oracle, shared):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    device_states = tuple(tensor.to(layout.device) for tensor in states)
    samples = []
    for repeat in range(2):
        _, raw, autograd = _stock(device_states, native, layout.length_tensor, cotangent.to(layout.device))
        raw_cpu = tuple(tensor.cpu().float() for tensor in raw)
        merged = (raw_cpu[0], raw_cpu[1][:, 0] + raw_cpu[2][:, 0], raw_cpu[3], raw_cpu[4][:, 0])
        autograd_cpu = tuple(tensor.cpu().float() for tensor in autograd)
        raw_vs_autograd = [_difference(actual, expected, require_close=False)
                           for actual, expected in zip(merged, autograd_cpu)]
        samples.append({"repeat": repeat,
                        "raw_vs_fp64_separate_oracle": [
                            _difference(actual[:, 0] if index in (1, 2, 4) else actual, expected,
                                        require_close=False)
                            for index, (actual, expected) in enumerate(zip(raw_cpu, separated))],
                        "raw_merged_vs_fp64_oracle": [
                            _difference(actual, expected, require_close=False)
                            for actual, expected in zip(merged, shared)],
                        "autograd_vs_fp64_oracle": [
                            _difference(actual, expected, require_close=False)
                            for actual, expected in zip(autograd_cpu, shared)],
                        "raw_merged_vs_autograd": raw_vs_autograd})
        projected_raw = (merged[0][list(rows)], merged[1], merged[2][list(rows)], merged[3])
        projected_oracle = (shared[0][list(rows)], shared[1], shared[2][list(rows)], shared[3])
        samples[-1]["profile_metrics"] = [gradient_error_metrics(actual, expected)
                                          for actual, expected in zip(projected_raw, projected_oracle)]
    return {"heads": heads, "seed": seed, "tile": mode, "query_rows": len(rows),
            "partition": "held_out" if seed == _SEEDS[-1] else "calibration",
            "separate_and_shared_fp64_oracles_agree": True, "samples": samples}


def _profile_budgets(cases, fit_seeds, held_out_seeds):
    grouped = {}
    for case in cases:
        grouped.setdefault(case["bucket"], []).append(case)
    results = {}
    for name, members in grouped.items():
        fields = {}
        for index, field in enumerate(_NAMES):
            fitting = [sample["profile_metrics"][index] for member in members
                       if member["seed"] in fit_seeds for sample in member["samples"]]
            validation = [sample["profile_metrics"][index] for member in members
                          if member["seed"] in held_out_seeds for sample in member["samples"]]
            budget = fit_reference_budget(fitting, margin=_PROFILE_MARGIN, zero_atol=_PROFILE_ZERO_ATOL)
            fields[field] = {"limits": asdict(budget), "held_out_samples": len(validation),
                             "held_out_passed": bool(validation) and all(
                                 reference_budget_passes(metric, budget) for metric in validation)}
        results[name] = fields
    return results


def run_cp_reference_profile(
    report: dict, output: Path, *, fit_seeds: tuple[int, ...] = _PROFILE_FIT_SEEDS,
    held_out_seeds: tuple[int, ...] = _PROFILE_HELD_OUT_SEEDS,
) -> None:
    """Freeze a CP-mask reference-only policy before sampling new calibration and held-out seeds.

    The resulting budgets cover this synthetic short BF16 fixture distribution only.
    They cannot accept long sequences, model gradients or a different backend identity.

    Args:
        report: Mutable reference-only evidence record.
        output: Measurement archive; the policy is saved beside it before device sampling.
        fit_seeds: Independent seeds used to fit the unchanged reference margins.
        held_out_seeds: Disjoint seeds used only for validation, never to expand budgets.
    """
    fit_seeds, held_out_seeds = validate_reference_seed_sets(fit_seeds, held_out_seeds)
    policy = {"fit_seeds": fit_seeds, "held_out_seeds": held_out_seeds,
              "margin": _PROFILE_MARGIN, "cosine_error_margin": _PROFILE_MARGIN**2,
              "zero_atol": _PROFILE_ZERO_ATOL, "lengths": (3, 10),
              "cp_sizes": (1, 2, 4), "heads": (32, 64), "patterns": ("contiguous", "strided", "empty"),
              "tiles": _MODES, "sparse_count": 2048, "attention_scale": _SCALE,
              "fixture_distribution": "normal states and dy std=.1; BF16; dy scaled by rank+1",
              "selection": "all legal history with independently randomized slot order",
              "candidate_measurements": False, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    policy_text = json.dumps(policy, sort_keys=True, indent=2) + "\n"
    output.with_suffix(".policy.json").write_text(policy_text, encoding="utf-8")
    report.update(status="running", cases=[], policy=policy,
                  policy_sha256=hashlib.sha256(policy_text.encode()).hexdigest(), accepted_tolerance_profile=False,
                  candidate_measurements=False, package_path=hyper_parallel.__file__,
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    torch.npu.set_device(0)
    base = DsaBatchMeta.packed((3, 10))
    layout = CannDsaLayout(base, "npu:0")
    for heads in (32, 64):
        for size in (1, 2, 4):
            for pattern in ("contiguous", "strided", "empty"):
                for rank in range(size):
                    meta = _metadata((3, 10), pattern, size, rank)
                    for mode in _MODES:
                        for seed in (*fit_seeds, *held_out_seeds):
                            selected = _selection(base, seed)
                            native = layout.global_to_sequence_indices(selected.to(layout.device))
                            bucket = f"H{heads}/CP{size}/{pattern}/rank{rank}/{mode}"
                            report["stage"] = {"bucket": bucket, "seed": seed}
                            case = _measure(meta, layout, native, selected, heads, seed, mode)
                            case.update(bucket=bucket, cp_size=size, cp_rank=rank, owner_pattern=pattern,
                                        partition="calibration" if seed in fit_seeds else "held_out")
                            report["cases"].append(case)
                            if len(report["cases"]) % 20 == 0:
                                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    budgets = _profile_budgets(report["cases"], fit_seeds, held_out_seeds)
    consistent = all(metric["passed"] for case in report["cases"] for sample in case["samples"]
                     for metric in sample["raw_merged_vs_autograd"])
    expected_cases = (len(fit_seeds) + len(held_out_seeds)) * len(policy["heads"]) * sum(policy["cp_sizes"]) \
        * len(policy["patterns"]) * len(policy["tiles"])
    validated = len(report["cases"]) == expected_cases and consistent and all(
        field["held_out_passed"] for fields in budgets.values() for field in fields.values())
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                 if any(name in line for name in ("libopapi.so", "libcust_opapi.so", "libtorch_npu.so"))}
    report.update(status="validated" if validated else "failed", stage="complete", case_count=len(report["cases"]),
                  budgets=budgets, accepted_tolerance_profile=validated,
                  stock_raw_and_autograd_pointwise_consistent=consistent,
                  loaded_library_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                         for name in libraries})


def _envelopes(cases):
    envelopes = []
    for heads in (32, 64):
        for mode in _MODES:
            fitting = [sample for case in cases if case["heads"] == heads and case["tile"] == mode
                       and case["partition"] == "calibration" for sample in case["samples"]]
            held_out = [sample for case in cases if case["heads"] == heads and case["tile"] == mode
                        and case["partition"] == "held_out" for sample in case["samples"]]
            fields = {}
            for index, name in enumerate(_NAMES):
                metrics = [sample["raw_merged_vs_fp64_oracle"][index] for sample in fitting]
                normalized = [metric for metric in metrics if metric["relative_l2_applicable"]]
                envelope = {"max_relative_l2": max(metric["relative_l2"] for metric in normalized)
                            if normalized else None,
                            "max_abs": max(metric["max_abs"] for metric in metrics),
                            "min_cosine": min(metric["cosine"] for metric in normalized) if normalized else None}
                validation = [sample["raw_merged_vs_fp64_oracle"][index] for sample in held_out]
                envelope["held_out_inside_observed_envelope"] = all(
                    metric["max_abs"] <= envelope["max_abs"]
                    and (not metric["relative_l2_applicable"] or (
                        envelope["max_relative_l2"] is not None
                        and metric["relative_l2"] <= envelope["max_relative_l2"]
                        and metric["cosine"] >= envelope["min_cosine"])) for metric in validation)
                fields[name] = envelope
            envelopes.append({"heads": heads, "tile": mode, "fields": fields})
    return envelopes


def run_calibration(report: dict, output: Path) -> None:
    """Measure separate K/V, shared C and stock autograd without loading a CP candidate.

    Observed envelopes are measurements, not accepted tolerance budgets. Held-out
    seeds never enlarge the envelope and do not change the existing candidate gate.
    """
    torch.npu.set_device(0)
    meta = DsaBatchMeta.packed((3, 10))
    layout = CannDsaLayout(meta, "npu:0")
    selected = _selection(meta)
    native = layout.global_to_sequence_indices(selected.to(layout.device))
    report.update(status="running", cases=[], candidate_measurements=False, tolerance_gate_changed=False,
                  seeds=_SEEDS, lengths=(3, 10), attention_scale=_SCALE,
                  oracle_dtype="float64", metric_comparison_dtype="float32",
                  package_path=hyper_parallel.__file__, torch_version=torch.__version__,
                  torch_npu_version=torch_npu.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for heads in (32, 64):
        for seed in _SEEDS:
            for mode in _MODES:
                report["stage"] = {"heads": heads, "seed": seed, "tile": mode}
                report["cases"].append(_measure(meta, layout, native, selected, heads, seed, mode))
                output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                 if any(name in line for name in ("libopapi.so", "libcust_opapi.so", "libtorch_npu.so"))}
    consistent = all(metric["passed"] for case in report["cases"] for sample in case["samples"]
                     for metric in sample["raw_merged_vs_autograd"])
    report.update(status="measured", stage="complete", case_count=len(report["cases"]),
                  stock_raw_and_autograd_pointwise_consistent=consistent,
                  observed_envelopes=_envelopes(report["cases"]), accepted_tolerance_profile=False,
                  loaded_library_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                         for name in libraries})


def _parse_seeds(value):
    try:
        return tuple(int(seed.strip()) for seed in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error


def main() -> None:
    """Archive independent stock reference measurements and any exact failing stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp-reference-profile", action="store_true")
    parser.add_argument("--fit-seeds", type=_parse_seeds)
    parser.add_argument("--held-out-seeds", type=_parse_seeds)
    args = parser.parse_args()
    if not args.cp_reference_profile and (args.fit_seeds is not None or args.held_out_seeds is not None):
        parser.error("explicit seed sets require --cp-reference-profile")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "independent stock BF16 main-gradient calibration"}
    try:
        if args.cp_reference_profile:
            run_cp_reference_profile(report, args.output,
                                     fit_seeds=args.fit_seeds if args.fit_seeds is not None else _PROFILE_FIT_SEEDS,
                                     held_out_seeds=(args.held_out_seeds if args.held_out_seeds is not None
                                                     else _PROFILE_HELD_OUT_SEEDS))
        else:
            run_calibration(report, args.output)
        if args.cp_reference_profile and not report["accepted_tolerance_profile"]:
            raise RuntimeError("independent held-out reference profile validation failed")
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
