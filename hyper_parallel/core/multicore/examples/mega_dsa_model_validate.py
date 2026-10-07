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
"""Single-NPU HP DSA model parameter validation with an unabsorbed FP32 oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cann_matrix import _calibrate
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
    CannDsaSelection,
    CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.trace import profile_index_trace


class _ObservedReference(CannDsaReference):
    """Device-validator capture only; production model calls do not request host snapshots."""

    def __init__(self, layout: CannDsaLayout, *, attention_scale: float) -> None:
        """Initialize per-forward selection and auxiliary-loss observations."""
        super().__init__(layout, attention_scale=attention_scale)
        self.last_selection = None
        self.last_loss = None

    def indexer(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    ) -> CannDsaSelection:
        """Record the admitted selection without copying device content to the host."""
        self.last_selection = super().indexer(index_query, index_key, merge_weight)
        return self.last_selection

    def kl_loss(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        main_inputs: tuple[torch.Tensor, ...], topk_indices: CannDsaSelection, stats: CannDsaStats,
        *, normalization: DsaLossNormalization, loss_coeff: float = 1.0,
    ) -> torch.Tensor:
        """Record the raw auxiliary scalar for KL-only numerical acceptance."""
        self.last_loss = super().kl_loss(index_query, index_key, merge_weight, main_inputs, topk_indices, stats,
                                         normalization=normalization, loss_coeff=loss_coeff)
        return self.last_loss


class _StockAttention(torch.autograd.Function):
    """Explicit stock SFA VJP for calibration of the same model/selection."""

    @staticmethod
    def forward(
        ctx: Any, query: torch.Tensor, compressed: torch.Tensor, qr: torch.Tensor, kr: torch.Tensor,
        selection: CannDsaSelection, backend: CannDsaReference,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Save native buffers for the stock primitive's explicit backward."""
        native = backend._native_indices(selection)
        lengths = backend.layout.length_tensor
        output, maximum, denominator = torch.ops.npu.npu_sparse_flash_attention(
            query, compressed[:, None], compressed[:, None], native, backend.attention_scale,
            actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
            query_rope=qr, key_rope=kr[:, None], layout_query="TND", layout_kv="TND", sparse_block_size=1,
            sparse_mode=3, attention_mode=2, return_softmax_lse=True,
        )
        ctx.backend = backend
        ctx.save_for_backward(query, compressed, qr, kr, native, output, maximum, denominator)
        ctx.mark_non_differentiable(maximum, denominator)
        return output, maximum, denominator

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_maximum: torch.Tensor | None, _grad_denominator: torch.Tensor | None,
    ) -> tuple:
        """Add the stock K and V derivatives once for the shared compressed input."""
        query, compressed, qr, kr, native, output, maximum, denominator = ctx.saved_tensors
        backend = ctx.backend
        gradients = torch.ops.npu.npu_sparse_flash_attention_grad(
            query, compressed[:, None], compressed[:, None], native, grad_output.contiguous(),
            output, maximum, denominator, backend.attention_scale, 1, query_rope=qr, key_rope=kr[:, None],
            actual_seq_qlen=backend.layout.length_tensor, actual_seq_kvlen=backend.layout.length_tensor,
            layout="TND", sparse_mode=3, attention_mode=2,
        )
        grad_query, grad_key, grad_value, grad_qr, grad_kr = gradients
        return grad_query, (grad_key + grad_value)[:, 0], grad_qr, grad_kr[:, 0], None, None


class _StockKl(torch.autograd.Function):
    """Explicit stock forward-computed KL derivatives for model calibration."""

    @staticmethod
    def forward(
        ctx: Any, iq: torch.Tensor, ik: torch.Tensor, weight: torch.Tensor, main_inputs: tuple[torch.Tensor, ...],
        selection: CannDsaSelection, stats: CannDsaStats, backend: CannDsaReference, scale: float,
    ) -> torch.Tensor:
        """Compute stock selected KL with the same detached teacher and declared scale."""
        query, compressed, qr, kr = (tensor.detach() for tensor in main_inputs)
        gradients = torch.ops.npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query, compressed[:, None], iq, ik[:, None], weight, backend._native_indices(selection),
            stats.maximum, stats.denominator, backend.attention_scale, query_rope=qr, key_rope=kr[:, None],
            actual_seq_qlen=list(backend.layout.cumulative_lengths),
            actual_seq_klen=list(backend.layout.cumulative_lengths),
            layout="TND", sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
        )
        grad_iq, grad_ik, grad_weight, loss = gradients
        ctx.save_for_backward(grad_iq * scale, grad_ik[:, 0] * scale, grad_weight * scale)
        return loss.reshape(()) * scale

    @staticmethod
    def backward(ctx: Any, grad_loss: torch.Tensor) -> tuple:
        """Apply the upstream scalar once to the three stock indexer derivatives."""
        return (*(gradient * grad_loss for gradient in ctx.saved_tensors), *((None,) * 5))


class _StockReference(_ObservedReference):
    """Same projected model, fixed enhance selection, stock SFA/KL primitives only."""

    def __init__(self, layout: CannDsaLayout, *, attention_scale: float, indices: torch.Tensor) -> None:
        """Prepare the measured candidate selection outside the model invocation."""
        super().__init__(layout, attention_scale=attention_scale)
        self.last_selection = self.prepare_selection(indices)

    def indexer(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
    ) -> CannDsaSelection:
        """Use the fixed measured selection; this calibration is not a Top-K equivalence test."""
        del index_query, index_key, merge_weight
        return self.last_selection

    def attention(
        self, query: torch.Tensor, compressed_kv: torch.Tensor, query_rope: torch.Tensor,
        key_rope: torch.Tensor, topk_indices: CannDsaSelection,
    ) -> tuple[torch.Tensor, CannDsaStats]:
        """Run stock SFA with an explicit VJP and native statistics."""
        self._validate_main((query, compressed_kv, query_rope, key_rope))
        output, maximum, denominator = _StockAttention.apply(
            query.contiguous(), compressed_kv.contiguous(), query_rope.contiguous(), key_rope.contiguous(),
            topk_indices, self,
        )
        return output, CannDsaStats(maximum, denominator)

    def kl_loss(
        self, index_query: torch.Tensor, index_key: torch.Tensor, merge_weight: torch.Tensor,
        main_inputs: tuple[torch.Tensor, ...], topk_indices: CannDsaSelection, stats: CannDsaStats,
        *, normalization: DsaLossNormalization, loss_coeff: float = 1.0,
    ) -> torch.Tensor:
        """Run stock KL with explicit per-invocation saved indexer gradients."""
        self.last_loss = _StockKl.apply(index_query.contiguous(), index_key.contiguous(), merge_weight.contiguous(),
                                        main_inputs, topk_indices, stats, self,
                                        loss_coeff * normalization.local_sum_scale)
        return self.last_loss


def _configure(model, mode):
    model.train(mode != "eval")
    model.freeze_dsa = mode == "freeze"
    model.dsa_loss_coeff = 0 if mode == "zero" else 0.3


def _collect(model, hidden, output, loss, *, gradients=None) -> dict:
    names = ("hidden", *dict(model.named_parameters()))
    if gradients is None:
        gradients = (hidden.grad, *(parameter.grad for parameter in model.parameters()))
    result = {name: None if gradient is None else gradient.detach().float().cpu()
              for name, gradient in zip(names, gradients)}
    result["output"] = output.detach().float().cpu()
    result["kl_loss"] = None if loss is None else loss.detach().float().cpu()
    projection = result["kv_b_proj.weight"]
    if projection is not None:
        projection = projection.reshape(model.num_heads, model.qk_nope_head_dim + model.v_head_dim, 512)
        result["W_UK"] = projection[:, :model.qk_nope_head_dim]
        result["W_UV"] = projection[:, model.qk_nope_head_dim:]
    return result


def _device_case(template, hidden_cpu, embeddings_cpu, meta, mode, auxiliary_scale, *, indices=None):
    source = copy.deepcopy(template).to("npu:0")
    _configure(source, mode)
    model = CannDsaReferenceAttention(module=source)
    hidden = hidden_cpu.to("npu:0").requires_grad_()
    embeddings = tuple(tensor.to("npu:0") for tensor in embeddings_cpu)
    layout = CannDsaLayout(meta, "npu:0")
    backend = (_ObservedReference(layout, attention_scale=model.scaling) if indices is None
               else _StockReference(layout, attention_scale=model.scaling, indices=indices))
    set_aux_loss_scale(torch.tensor(float(auxiliary_scale), device="npu:0"))

    def _forward(value):
        return model(value, position_embeddings=embeddings, dsa_reference=backend,
                     actual_seq_len=layout.length_tensor)[0]

    output = checkpoint(_forward, hidden, use_reentrant=False) if mode == "checkpoint" else _forward(hidden)
    objective = backend.last_loss * auxiliary_scale if mode == "kl_only" else output.float().square().mean() * 13
    gradients = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True)
    torch.npu.synchronize()
    selected = backend.last_selection.to_global_indices().cpu()
    return _collect(model, hidden, output, backend.last_loss, gradients=gradients), selected


def _cpu_case(template, hidden_cpu, embeddings_cpu, meta, indices, mode, auxiliary_scale):
    model = copy.deepcopy(template).float()
    _configure(model, mode)
    hidden = hidden_cpu.float().requires_grad_()
    embeddings = tuple(tensor.float() for tensor in embeddings_cpu)
    with patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
               side_effect=_rotary_mul_oracle):
        output, loss = unabsorbed_model_reference(model, hidden, embeddings, meta, indices)
        if mode == "kl_only":
            objective = loss * auxiliary_scale
        else:
            objective = output.square().mean() * 13
            if loss is not None:
                objective = objective + loss * auxiliary_scale
        gradients = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True)
    return _collect(model, hidden, output, loss, gradients=gradients)


def _apply_angular_bound(check: dict) -> None:
    """Apply the stock-calibrated normalized angular bound to one quantity."""
    # For unit vectors, squared L2 distance is 2*(1-cosine). Apply the same
    # 1.25 L2 multiplier to angular error, rather than an absolute cosine margin.
    actual, reference, limits = check["enhance_vs_oracle"], check["stock_vs_oracle"], check["limits"]
    if reference["reference_norm"] == 0:
        limits["cosine"] = None
        angular_pass = True
    else:
        angular = math.sqrt(2 * max(0.0, 1 - reference["cosine"]))
        actual["angular_l2"] = math.sqrt(2 * max(0.0, 1 - actual["cosine"]))
        reference["angular_l2"] = angular
        limits["angular_l2"] = max(1.25 * angular, 1e-4)
        limits["cosine"] = 1 - limits["angular_l2"] ** 2 / 2
        angular_pass = actual["angular_l2"] <= limits["angular_l2"]
    check["passed"] = (
        check["stock_calibration_usable"] and actual["finite"] and angular_pass
        and actual["relative_l2"] <= limits["relative_l2"] and actual["max_abs"] <= limits["max_abs"]
    )


def _compare(candidate, stock, oracle) -> dict:
    present = {name: value for name, value in oracle.items()
               if value is not None and candidate.get(name) is not None and stock.get(name) is not None}
    checks = _calibrate({name: candidate[name] for name in present}, {name: stock[name] for name in present},
                        present, torch.ones(1, dtype=torch.bool))
    for check in checks.values():
        _apply_angular_bound(check)
    availability = {name: (candidate.get(name) is None) == (stock.get(name) is None) == (value is None)
                    for name, value in oracle.items()}
    return {"checks": checks, "gradient_availability": availability,
            "accepted": all(availability.values()) and all(check["passed"] for check in checks.values())}


def _calibration_limits(cases: dict) -> dict:
    limits = {}
    for name, case in cases.items():
        objective = name.split(",", 1)[1]
        quantities = limits.setdefault(objective, {})
        for quantity, check in case["checks"].items():
            if not check["stock_calibration_usable"]:
                raise ValueError(f"unusable stock model calibration: {name}/{quantity}")
            stock = check["stock_vs_oracle"]
            candidate_limits = {
                "relative_l2": max(1.25 * stock["relative_l2"], 1e-4),
                "max_abs": max(1.25 * stock["max_abs"], 1e-12),
                "angular_l2": max(1.25 * stock.get("angular_l2", 0), 1e-4),
            }
            previous = quantities.setdefault(quantity, candidate_limits.copy())
            for metric, value in candidate_limits.items():
                previous[metric] = max(previous[metric], value)
    return limits


def _apply_calibration(case: dict, limits: dict) -> dict:
    case["pointwise_accepted"] = case["accepted"]
    for quantity, check in case["checks"].items():
        bound = limits[quantity]
        check["pointwise_limits"] = check["limits"]
        check["limits"] = {**bound, "cosine": 1 - bound["angular_l2"] ** 2 / 2}
        for backend, field in (("enhance", "enhance_vs_oracle"), ("stock", "stock_vs_oracle")):
            observed = check[field]
            check[f"{backend}_within_calibration"] = (
                observed["finite"] and observed["relative_l2"] <= bound["relative_l2"]
                and observed["max_abs"] <= bound["max_abs"]
                and observed.get("angular_l2", 0) <= bound["angular_l2"]
            )
        check["passed"] = (check["stock_calibration_usable"] and check["stock_within_calibration"]
                           and check["enhance_within_calibration"])
    case["accepted"] = all(case["gradient_availability"].values()) and all(
        check["passed"] for check in case["checks"].values())
    return case


def run_model_validation(report: dict) -> None:
    """Validate all model parameters with joint/isolated/gated/checkpoint objectives."""
    # Optional device extension is activated only by this explicit validator.
    __import__("torch_npu")
    torch.npu.set_device(0)
    report["environment"] = probe_environment()
    meta = DsaBatchMeta.packed((3, 5))
    report["cases"] = {}
    report["index_traces"] = {}
    frozen = None
    for seed in (20261007, 20261017, 20261018, 20261027, 20261028):
        template = build_model_fixture(dtype=torch.bfloat16, seed=seed)
        hidden, embeddings = model_inputs(meta, dtype=torch.bfloat16, seed=seed + 1)
        report["parameter_shapes"] = {name: list(parameter.shape) for name, parameter in template.named_parameters()}
        for mode, auxiliary_scale in (("joint", 1), ("joint", 7), ("kl_only", 7), ("zero", 7),
                                      ("freeze", 7), ("eval", 7), ("checkpoint", 7)):
            name = f"seed={seed},{mode},grad_aux={auxiliary_scale}"
            report["stage"] = name
            candidate, indices = _device_case(template, hidden, embeddings, meta, mode, auxiliary_scale)
            stock, _ = _device_case(template, hidden, embeddings, meta, mode, auxiliary_scale, indices=indices)
            oracle = _cpu_case(template, hidden, embeddings, meta, indices, mode, auxiliary_scale)
            report["cases"][name] = _compare(candidate, stock, oracle)
            if frozen is not None:
                _apply_calibration(report["cases"][name], frozen[name.split(",", 1)[1]])
            if mode == "joint" and auxiliary_scale == 1:
                report["index_traces"][str(seed)] = {
                    "provenance": "actual HP DSA projections; random small fixture, not a pretrained model",
                    "profile": profile_index_trace(indices, meta, kv_bytes_per_token=1152),
                }
            print(f"{name}: accepted={report['cases'][name]['accepted']}", flush=True)
        if seed == 20261018:
            frozen = _calibration_limits(report["cases"])
            report["calibration_limits"] = frozen
            report["calibration_sha256"] = hashlib.sha256(
                json.dumps(frozen, sort_keys=True).encode("utf-8")).hexdigest()
            for name, case in report["cases"].items():
                _apply_calibration(case, frozen[name.split(",", 1)[1]])
    report["accepted_for_model_fixture"] = all(case["accepted"] for case in report["cases"].values())
    report["status"] = "accepted" if report["accepted_for_model_fixture"] else "failed_acceptance"
    report["stage"] = "complete"


def main() -> None:
    """Persist model acceptance, exact environment and failures with a nonzero error status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"status": "running", "stage": "device_init", "cp_size": 1, "tp_size": 1,
              "sequence_lengths": [3, 5], "dtype": "bfloat16", "performance_measurement": False,
              "fixture": "randomly initialized existing HP DeepseekV32DSAAttention; not a pretrained HF model",
              "oracle": "FP32 unabsorbed dense MLA and detached selected-set KL on identical quantized inputs/weights",
              "calibration_seeds": [20261007, 20261017, 20261018], "holdout_seeds": [20261027, 20261028],
              "calibration": "per quantity and objective: 1.25*maximum stock error across three calibration seeds; "
                             "relative/angular L2 and max-abs floors 1e-4/1e-4/1e-12; angular L2=sqrt(2*(1-cosine)); "
                             "freeze before holdouts; both stock and candidate must pass; stock relative L2<1",
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        run_model_validation(report)
        if not report["accepted_for_model_fixture"]:
            raise RuntimeError("model fixture did not pass calibrated parameter-gradient acceptance")
    except Exception as error:
        report["status"] = "error" if report["status"] == "running" else report["status"]
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        set_aux_loss_scale(torch.tensor(1.0))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
