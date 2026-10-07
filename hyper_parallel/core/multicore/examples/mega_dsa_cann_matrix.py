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
"""Single-NPU selected-set matrix with per-fixture stock CANN calibration."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from hyper_parallel.components.functional.aux_loss import (
    aux_loss_auto_scale,
    set_aux_loss_scale,
)
from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cann_validate import _metrics
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.trace import profile_index_trace

_MAIN_NAMES = ("q_nope", "compressed_kv", "q_rope", "k_rope")
_INDEX_NAMES = ("index_q", "index_k", "merge_weight")


def _selection(case: str, meta: DsaBatchMeta) -> torch.Tensor:
    indices = torch.full((meta.global_valid_queries, 2048), -1, dtype=torch.int32)
    for token in range(meta.global_valid_queries):
        start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
        if case == "complete":
            selected = list(range(start, token + 1))
        elif case == "empty" or (case == "mixed_empty" and token == start):
            selected = []
        else:
            selected = [token] if token == start else [token, start]
        if case == "holes":
            # A known future/cross-sequence ID plus a leading hole must be filtered before native dispatch.
            selected = [-1, (token + 1) % meta.global_valid_queries] + selected
        indices[token, :len(selected)] = torch.tensor(selected, dtype=torch.int32)
    return indices


def _oracle(values: tuple, indices: torch.Tensor, backend: CannDsaReference) -> dict:
    main_inputs = tuple(tensor.float().requires_grad_() for tensor in values[:4])
    index = tuple(tensor.float().requires_grad_() for tensor in values[4:])
    meta = backend.layout.batch_meta
    output, stats = sparse_attention_reference(*main_inputs, indices, meta, attention_scale=backend.attention_scale)
    main_grad = torch.autograd.grad(output.square().mean(), main_inputs)
    loss = selected_kl_reference(
        *index, *main_inputs, indices, meta, attention_scale=backend.attention_scale,
        normalization=DsaLossNormalization(meta.global_valid_queries), loss_coeff=0.3,
    )
    index_grad = torch.autograd.grad(loss * 7, index)
    return {"output": output, "lse": stats.lse, "kl_loss": loss,
            **dict(zip(_MAIN_NAMES, main_grad)), **dict(zip(_INDEX_NAMES, index_grad))}


def _enhance(values: tuple, indices: torch.Tensor, backend: CannDsaReference) -> tuple[dict, dict]:
    main_inputs = tuple(tensor.to(backend.layout.device).requires_grad_() for tensor in values[:4])
    index = tuple(tensor.to(backend.layout.device).requires_grad_() for tensor in values[4:])
    selected = indices.to(backend.layout.device)
    output, stats = backend.attention(*main_inputs, selected)
    output.float().square().mean().backward()
    main_grad = tuple(tensor.grad.detach().clone() for tensor in main_inputs)
    loss = backend.kl_loss(
        *index, main_inputs, selected, stats,
        normalization=DsaLossNormalization(backend.layout.batch_meta.global_valid_queries),
        loss_coeff=0.3,
    )
    (loss * 7).backward()
    torch.npu.synchronize()
    # Nonfinite main derivatives fail numerical acceptance separately; unchanged NaNs do not imply a KL leak.
    isolation = all(torch.allclose(tensor.grad.cpu(), before.cpu(), rtol=0, atol=0, equal_nan=True)
                    for tensor, before in zip(main_inputs, main_grad))
    return ({"output": output, "lse": stats.lse[0], "kl_loss": loss,
             **dict(zip(_MAIN_NAMES, main_grad)), **dict(zip(_INDEX_NAMES, (tensor.grad for tensor in index)))},
            {"main_gradient_isolation": isolation, "maximum": stats.maximum[0], "sum": stats.denominator[0]})


def _stock(values: tuple, indices: torch.Tensor, backend: CannDsaReference) -> dict:
    query, compressed, query_rope, key_rope, index_query, index_key, weight = (
        tensor.to(backend.layout.device) for tensor in values)
    layout = backend.layout
    selected = layout.global_to_sequence_indices(indices.to(layout.device))
    output, maximum, denominator = torch.ops.npu.npu_sparse_flash_attention(
        query, compressed[:, None], compressed[:, None], selected, backend.attention_scale,
        actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_kv=layout.length_tensor,
        query_rope=query_rope, key_rope=key_rope[:, None], layout_query="TND", layout_kv="TND",
        sparse_block_size=1, sparse_mode=3, attention_mode=2, return_softmax_lse=True,
    )
    dout = (2 * output.float() / output.numel()).bfloat16().contiguous()
    grad_query, grad_key, grad_value, grad_qr, grad_kr = torch.ops.npu.npu_sparse_flash_attention_grad(
        query, compressed[:, None], compressed[:, None], selected, dout, output, maximum, denominator,
        backend.attention_scale, 1, query_rope=query_rope, key_rope=key_rope[:, None],
        actual_seq_qlen=layout.length_tensor, actual_seq_kvlen=layout.length_tensor,
        layout="TND", sparse_mode=3, attention_mode=2,
    )
    grad_iq, grad_ik, grad_weight, loss = torch.ops.npu.npu_sparse_lightning_indexer_grad_kl_loss(
        query, compressed[:, None], index_query, index_key[:, None], weight, selected, maximum, denominator,
        backend.attention_scale, query_rope=query_rope, key_rope=key_rope[:, None],
        actual_seq_qlen=list(layout.cumulative_lengths), actual_seq_klen=list(layout.cumulative_lengths),
        layout="TND", sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
    )
    torch.npu.synchronize()
    scale = 0.3 / layout.batch_meta.global_valid_queries
    return {"output": output, "lse": maximum[0].float() + denominator[0].float().log(),
            "kl_loss": loss.reshape(()) * scale,
            **dict(zip(_MAIN_NAMES, (grad_query, (grad_key + grad_value)[:, 0], grad_qr, grad_kr[:, 0]))),
            **dict(zip(_INDEX_NAMES, ((grad_iq * scale) * 7, (grad_ik[:, 0] * scale) * 7,
                                     (grad_weight * scale) * 7)))}


def _calibrate(candidate: dict, baseline: dict, oracle: dict, nonempty: torch.Tensor) -> dict:
    checks = {}
    for name, expected in oracle.items():
        actual = candidate[name].detach().cpu()
        stock = baseline[name].detach().cpu()
        if name == "lse":
            actual, stock, expected = actual[nonempty], stock[nonempty], expected[nonempty]
        if expected.numel() == 0:
            continue
        observed, reference = _metrics(actual, expected), _metrics(stock, expected)
        # A stock error at least as large as the signal cannot calibrate BF16 acceptance.
        usable = reference["finite"] and reference["relative_l2"] < 1
        limits = {"relative_l2": max(reference["relative_l2"] * 1.25, 1e-4),
                  "max_abs": max(reference["max_abs"] * 1.25, 1e-12),
                  "cosine": reference["cosine"] - 1e-6}
        checks[name] = {
            "enhance_vs_oracle": observed, "stock_vs_oracle": reference, "limits": limits,
            "stock_calibration_usable": usable,
            "passed": usable and observed["finite"]
            and observed["relative_l2"] <= limits["relative_l2"] and observed["max_abs"] <= limits["max_abs"]
            and observed["cosine"] >= limits["cosine"],
        }
    return checks


def _empty_contract(candidate: dict, stats: dict, empty: torch.Tensor) -> dict:
    output = candidate["output"].detach().cpu()[empty]
    query_grads = tuple(candidate[name].detach().cpu()[empty]
                        for name in ("q_nope", "q_rope", "index_q", "merge_weight"))
    maximum, denominator = stats["maximum"].detach().cpu()[empty], stats["sum"].detach().cpu()[empty]
    # K gradients are shared across queries; a key can still be used by a nonempty row.
    return {"query_count": int(empty.sum()), "output_zero": bool((output == 0).all()),
            "query_gradients_zero": all(bool((gradient == 0).all()) for gradient in query_grads),
            "lse_is_negative_infinity": bool(torch.isneginf(candidate["lse"].detach().cpu()[empty]).all()),
            "maximum_is_negative_infinity": bool(torch.isneginf(maximum).all()),
            "sum_zero": bool((denominator == 0).all())}


def _auxiliary_scaling(values: tuple, backend: CannDsaReference) -> dict:
    meta = backend.layout.batch_meta
    indices = _selection("complete", meta)
    oracle = _oracle(values, indices, backend)
    stock = _stock(values, indices, backend)
    report = {}
    try:
        for auxiliary_scale, coefficient in ((1, 0.3), (7, 0.3), (7, 0)):
            main_inputs = tuple(tensor.to(backend.layout.device).requires_grad_() for tensor in values[:4])
            index = tuple(tensor.to(backend.layout.device).requires_grad_() for tensor in values[4:])
            selected = indices.to(backend.layout.device)
            output, stats = backend.attention(*main_inputs, selected)
            loss = backend.kl_loss(
                *index, main_inputs, selected, stats,
                normalization=DsaLossNormalization(meta.global_valid_queries), loss_coeff=coefficient,
            )
            set_aux_loss_scale(torch.tensor(float(auxiliary_scale), device=backend.layout.device))
            attached = aux_loss_auto_scale(output, loss)
            gradients = torch.autograd.grad(attached.float().square().mean() * 13,
                                            (*main_inputs, *index), retain_graph=True)
            lm_gradients = torch.autograd.grad(output.float().square().mean() * 13, main_inputs)
            torch.npu.synchronize()
            factor = auxiliary_scale / 7 * coefficient / 0.3
            checks = _calibrate(dict(zip(_INDEX_NAMES, gradients[4:])),
                                {name: stock[name] * factor for name in _INDEX_NAMES},
                                {name: oracle[name] * factor for name in _INDEX_NAMES},
                                torch.ones(meta.global_valid_queries, dtype=torch.bool))
            unchanged = torch.equal(attached, output)
            isolated = all(torch.equal(actual, expected) for actual, expected in zip(gradients[:4], lm_gradients))
            report[f"grad_aux={auxiliary_scale},coefficient={coefficient}"] = {
                "checks": checks, "forward_unchanged": unchanged, "main_gradients_unchanged": isolated,
                "accepted": unchanged and isolated and all(check["passed"] for check in checks.values()),
            }
    finally:
        set_aux_loss_scale(torch.tensor(1.0))
    return report


def run_matrix(report: dict) -> None:
    """Run complete/partial/empty/mixed/holey selections with primitive calibration."""
    # Optional extension is loaded only for an explicitly requested device run.
    __import__("torch_npu")
    torch.npu.set_device(0)
    report["environment"] = probe_environment()
    report["device"] = torch.npu.get_device_name(0)
    meta = DsaBatchMeta.packed((3, 5))
    backend = CannDsaReference(CannDsaLayout(meta, "npu:0"), attention_scale=192**-0.5)
    generator = torch.Generator().manual_seed(20261007)
    shapes = ((8, 32, 512), (8, 512), (8, 32, 64), (8, 64), (8, 8, 128), (8, 128), (8, 8))
    values = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in shapes)
    report["cases"] = {}
    for case in ("complete", "partial", "empty", "mixed_empty", "holes"):
        report["stage"] = case
        selected = _selection(case, meta)
        legal = backend.layout.sequence_to_global_indices(
            backend.layout.global_to_sequence_indices(selected.to(backend.layout.device))).cpu()
        nonempty = (legal >= 0).any(-1)
        oracle = _oracle(values, selected, backend)
        candidate, stats = _enhance(values, selected, backend)
        stock = _stock(values, selected, backend)
        checks = _calibrate(candidate, stock, oracle, nonempty)
        empty = _empty_contract(candidate, stats, ~nonempty)
        accepted = (all(check["passed"] for check in checks.values()) and stats["main_gradient_isolation"]
                    and all(empty[name] for name in ("output_zero", "query_gradients_zero", "sum_zero",
                                                     "lse_is_negative_infinity")))
        report["cases"][case] = {"checks": checks, "empty_rows": empty,
                                 "main_gradient_isolation": stats["main_gradient_isolation"],
                                 "accepted_for_fixture": accepted}
        print(f"{case}: accepted_for_fixture={accepted}", flush=True)
    report["stage"] = "index_trace"
    index = tuple(tensor.to(backend.layout.device) for tensor in values[4:])
    selected = backend.indexer(*index).detach().cpu()
    report["index_trace"] = {"provenance": "synthetic packed fixture through enhance indexer",
                              "profile": profile_index_trace(selected, meta, kv_bytes_per_token=(512 + 64) * 2)}
    report["stage"] = "auxiliary_scaling"
    report["auxiliary_scaling"] = _auxiliary_scaling(values, backend)
    report["accepted_for_matrix"] = (all(case["accepted_for_fixture"] for case in report["cases"].values())
                                     and all(case["accepted"] for case in report["auxiliary_scaling"].values()))
    report["status"] = "accepted" if report["accepted_for_matrix"] else "failed_acceptance"
    report["stage"] = "complete"


def main() -> None:
    """Persist stage, calibrated metrics and failures; propagate unsuccessful acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "running", "stage": "device_init", "cp_size": 1, "sequence_lengths": [3, 5],
              "seed": 20261007, "main_heads": 32, "index_heads": 8, "dtype": "bfloat16",
              "compressed_dim": 512, "rope_dim": 64, "index_dim": 128,
              "sparse_count": 2048, "performance_measurement": False,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "threshold_rule": "per fixture: relative L2/max abs <= 1.25*stock error with floors 1e-4/1e-12; "
                                "cosine >= stock cosine - 1e-6; stock relative L2 must be < 1; "
                                "exact zero for empty output/query grads/sum and LSE=-inf"}
    try:
        run_matrix(report)
        if not report["accepted_for_matrix"]:
            raise RuntimeError("selected-set matrix did not pass numerical acceptance")
    except Exception as error:
        report["status"] = "error" if report["status"] == "running" else report["status"]
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
