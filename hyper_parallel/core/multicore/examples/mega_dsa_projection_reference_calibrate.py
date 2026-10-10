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
"""Reference-only full-query BF16 projection VJP calibration at fixed Native values."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import _loaded_library_identity
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    ProjectionGeometry, project_local_states, projection_fixture, projection_vjp_metrics, projection_vjp_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout, CannDsaStats
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    mixed_sfa_backward_probe, mixed_sfa_gradient_partials,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import _submit_mixed_kl, mixed_kl_native_key_cast
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    fit_reference_budget, gradient_error_metrics, reference_budget_passes, validate_reference_seed_sets,
)

_SCALE = 192**-.5
_FIELDS = ("hidden", "query_projection", "compressed_projection", "query_rope_projection", "key_rope_projection",
           "index_query_projection", "index_key_projection", "merge_projection", "output_projection")
_FIT_SEEDS = tuple(range(20261040, 20261056))
_HELD_OUT_SEEDS = tuple(range(20261060, 20261068))


def _project_gradients(main_gradients, index, hidden, weights, output_gradient, mode):
    active = ((main_gradients if mode != "kl_only" else (None,)*4)
              + (index if mode != "lm_only" else (None,)*3))
    hidden_gradient = None
    if mode != "kl_only":
        hidden_gradient = sum(gradient.float().flatten(1) @ weight
                              for gradient, weight in zip(main_gradients, weights[:4]))
    parameter_gradients = tuple(None if gradient is None else gradient.float().flatten(1).T @ hidden
                                for gradient in active)
    output_parameter_gradient = (None if mode == "kl_only" else output_gradient)
    return (hidden_gradient, *parameter_gradients, output_parameter_gradient)


def _assert_reference_bits(actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise RuntimeError("reference dtype or shape mismatch")
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise RuntimeError("reference storage bit mismatch")


def _measure(geometry, dtype, seed, device):
    meta = DsaBatchMeta.packed((3, 10))
    layout = CannDsaLayout(meta, device)
    fixture = projection_fixture(meta.global_valid_queries, geometry, seed=seed)
    cpu_hidden, cpu_weights, targets = fixture
    hidden = cpu_hidden.to(device)
    weights = tuple(weight.to(device) for weight in cpu_weights)
    rows = torch.arange(meta.global_valid_queries, device=device)
    states = project_local_states(hidden, weights, geometry, rows, rows, dtype)
    config = MixedSfaSchedule(7).runtime_config(device)
    native = torch.ops.npu.npu_lightning_indexer(
        states[4], states[5][:, None], states[6], actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query="TND", layout_key="TND", sparse_count=2048,
        sparse_mode=3, return_value=True)[0]
    output = torch.ops.npu.npu_sparse_flash_attention(
        states[0], states[1][:, None], states[1][:, None], native, _SCALE,
        actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_kv=layout.length_tensor,
        query_rope=states[2], key_rope=states[3][:, None], layout_query="TND", layout_kv="TND",
        sparse_block_size=1, sparse_mode=3, attention_mode=2, return_softmax_lse=True)
    observed = output[0].detach().requires_grad_()
    output_weight = weights[7].detach().requires_grad_()
    prediction = observed.float().flatten(1) @ output_weight.T
    main_loss = (prediction-targets.to(device)).square().sum()/meta.global_valid_queries
    cotangent, output_gradient = torch.autograd.grad(main_loss, (observed, output_weight))
    indices = layout.sequence_to_global_indices(native).cpu()[:, :max((3, 10))]
    cpu_states, cpu_output = tuple(tensor.cpu() for tensor in states), output[0].cpu()
    expected = {mode: projection_vjp_reference(
        cpu_hidden, cpu_weights, targets, cpu_states, cpu_output, indices, meta, geometry, mode=mode,
        coefficient=.3, auxiliary=-2.5, attention_scale=_SCALE)
                for mode in ("lm_only", "kl_only", "lm_plus_kl")}
    samples = []
    for repeat in range(2):
        raw, _, retained = mixed_sfa_backward_probe(
            states[:4], native, layout.length_tensor, config, cotangent, output, _SCALE, MixedSfaSchedule(7))
        key, value = mixed_sfa_gradient_partials(retained, query_tokens=13, key_tokens=13, heads=geometry.heads)
        main_gradients = (raw[0], (key[:, :512]+value).bfloat16(), raw[3], key[:, 512:].bfloat16())
        kl = _submit_mixed_kl(states[:4], states[4:], native, layout.cumulative_lengths,
                              CannDsaStats(output[1], output[2]), config, _SCALE)
        scale = .3/meta.global_valid_queries
        index = ((kl.gradients[0].float()*scale*-2.5).bfloat16(),
                 (kl.key_partials*scale*-2.5).bfloat16(), (kl.gradients[2].float()*scale*-2.5).to(dtype))
        stock_main = torch.ops.npu.npu_sparse_flash_attention_grad(
            states[0], states[1][:, None], states[1][:, None], native, cotangent, *output, _SCALE, 1,
            query_rope=states[2], key_rope=states[3][:, None], actual_seq_qlen=layout.length_tensor,
            actual_seq_kvlen=layout.length_tensor, layout="TND", sparse_mode=3, attention_mode=0)
        stock_kl = torch.ops.npu.npu_sparse_lightning_indexer_grad_kl_loss(
            states[0], states[1][:, None], states[4], states[5][:, None], states[6], native,
            output[1], output[2], _SCALE, query_rope=states[2], key_rope=states[3][:, None],
            actual_seq_qlen=list(layout.cumulative_lengths), actual_seq_klen=list(layout.cumulative_lengths),
            layout="TND", sparse_mode=3)
        torch.npu.synchronize()
        stock_consistency = [projection_vjp_metrics(actual, reference)
                             for actual, reference in zip(raw, stock_main)]
        for plane, converted in ((key[:, :512], raw[1][:, 0]), (key[:, 512:], raw[4][:, 0]),
                                  (value, raw[2][:, 0]), (kl.key_partials, kl.gradients[1])):
            _assert_reference_bits(mixed_kl_native_key_cast(plane.cpu()), converted.cpu())
        stock_consistency.extend(projection_vjp_metrics(actual, reference) for actual, reference in
                                 zip(kl.gradients, (stock_kl[0], stock_kl[1][:, 0], stock_kl[2])))
        if not all(metric["passed"] for metric in stock_consistency):
            raise RuntimeError(f"non-deterministic mixed/stock reference consistency failed: {stock_consistency}")
        for mode, (reference_objective, reference_loss, reference_gradients) in expected.items():
            gradients = _project_gradients(main_gradients, index, hidden, weights, output_gradient, mode)
            loss = kl.loss*scale if mode != "lm_only" else kl.loss.new_zeros(())
            objective = loss*-2.5 if mode == "kl_only" else main_loss.detach()
            if mode == "lm_plus_kl":
                objective = objective+loss*-2.5
            metrics = {"objective": gradient_error_metrics(objective.cpu(), reference_objective),
                       "auxiliary_loss": gradient_error_metrics(loss.cpu(), reference_loss)}
            pointwise = {"objective": projection_vjp_metrics(objective, reference_objective),
                         "auxiliary_loss": projection_vjp_metrics(loss, reference_loss)}
            for name, actual, reference in zip(_FIELDS, gradients, reference_gradients):
                if reference is None:
                    if actual is not None:
                        raise RuntimeError("reference projection gradient isolation failed")
                    continue
                metrics[name] = gradient_error_metrics(actual.cpu(), reference)
                pointwise[name] = projection_vjp_metrics(actual, reference)
            samples.append({"seed": seed, "repeat": repeat, "mode": mode, "heads": geometry.heads,
                            "weight_dtype": str(dtype), "metrics": metrics, "original_pointwise": pointwise,
                            "reference_native_post_bits_exact": True,
                            "raw_gradients_stock_consistency": stock_consistency})
        del kl, retained, key, value
    return samples


def _budgets(samples, fit_seeds, held_out_seeds):
    groups = {}
    for sample in samples:
        bucket = f"H{sample['heads']}/{sample['weight_dtype']}/{sample['mode']}"
        groups.setdefault(bucket, []).append(sample)
    budgets = {}
    for bucket, members in groups.items():
        fields = {}
        for field in members[0]["metrics"]:
            fitting = [member["metrics"][field] for member in members if member["seed"] in fit_seeds]
            held_out = [member["metrics"][field] for member in members if member["seed"] in held_out_seeds]
            budget = fit_reference_budget(fitting, margin=2., zero_atol=2e-5)
            fields[field] = {"limits": asdict(budget), "held_out_samples": len(held_out),
                             "held_out_passed": bool(held_out) and all(
                                 reference_budget_passes(metric, budget) for metric in held_out)}
        budgets[bucket] = fields
    return budgets


def run_calibration(report: dict, output: Path) -> None:
    """Freeze independent seeds and margins before sampling a complete-query primitive reference.

    This scope covers packed (3,10), complete global queries, H32/H64 and BF16/FP32
    merge weights. It does not certify CP subsets, a full model or performance.
    Local CP candidate results are never inputs to fitting or held-out validation.
    """
    validate_reference_seed_sets(_FIT_SEEDS, _HELD_OUT_SEEDS)
    policy = {"schema": 1, "scope": "complete-query fixed-Native-value projection VJP reference",
              "candidate_measurements": False, "fit_seeds": _FIT_SEEDS, "held_out_seeds": _HELD_OUT_SEEDS,
              "margin": 2., "cosine_margin": 4., "zero_atol": 2e-5, "lengths": (3, 10),
              "heads": (32, 64), "weight_dtypes": ("torch.bfloat16", "torch.float32"),
              "query_scope": "complete global queries", "modes": ("lm_only", "kl_only", "lm_plus_kl"),
              "coefficient": .3, "auxiliary": -2.5, "repeats": 2, "attention_scale": _SCALE,
              "raw_stock_consistency": {"non_deterministic": True, "rtol": .02,
                                        "atol": 2e-5, "relative_l2": .02},
              "main_owner_contract": "FP32 dK+dV then one BF16 cast",
              "index_owner_contract": "FP32 K scale then upstream then one BF16 cast"}
    policy_text = json.dumps(policy, sort_keys=True, indent=2)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".policy.json").write_text(policy_text)
    report.update(policy=policy, policy_sha256=hashlib.sha256(policy_text.encode()).hexdigest(), samples=[],
                  candidate_measurements=False, full_model_acceptance=False, status="sampling")
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    for heads in policy["heads"]:
        for dtype in (torch.bfloat16, torch.float32):
            for seed in _FIT_SEEDS+_HELD_OUT_SEEDS:
                report["stage"] = {"heads": heads, "weight_dtype": str(dtype), "seed": seed}
                report["samples"].extend(_measure(ProjectionGeometry(heads=heads), dtype, seed, device))
    budgets = _budgets(report["samples"], _FIT_SEEDS, _HELD_OUT_SEEDS)
    passed = all(field["held_out_passed"] for fields in budgets.values() for field in fields.values())
    report.update(status="validated" if passed else "held_out_failed", budgets=budgets,
                  accepted_tolerance_profile=passed, loaded_library_sha256=_loaded_library_identity(),
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    if not passed:
        raise RuntimeError("independent held-out projection reference exceeds the frozen calibration policy")


def main() -> None:
    """Persist the reference profile and failed samples without changing any candidate gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "full_model_acceptance": False}
    try:
        run_calibration(report, args.output)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
