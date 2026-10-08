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
"""Real mixed SFA forward/backward gate with independent shared-K/V oracle."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    MixedDsaCoreProbe,
    mixed_sfa_backward_probe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    sparse_attention_reference,
)

_SCALE = 192**-0.5
_NAMES = ("query", "compressed_kv", "query_rope", "key_rope")


def _fixture(lengths: tuple[int, ...], heads: int, device: torch.device) -> tuple:
    meta = DsaBatchMeta.packed(lengths)
    layout = CannDsaLayout(meta, device)
    total = meta.global_valid_queries
    generator = torch.Generator().manual_seed(20261008 + total + heads)
    shapes = ((total, heads, 512), (total, 512), (total, heads, 64), (total, 64))
    states = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in shapes)
    selected = torch.full((total, 2048), -1, dtype=torch.int32)
    for query in range(total):
        seq, pos = meta.sequence_position(query)
        start = max(meta.global_cu_seqlens[seq], query + 1 - 2048)
        count = min(2048, pos + 1)
        selected[query, :count] = torch.arange(start, query + 1, dtype=torch.int32)
    if max(lengths) <= 513:
        rows = tuple(range(total))
        cotangent = (torch.randn(shapes[0], generator=generator) * 0.1).bfloat16()
    else:
        # Bound the independent CPU graph while still exercising the long native scatter path.
        rows = tuple(sorted({start + pos for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:])
                             for pos in (0, 2048, end - start - 1) if pos < end - start}))
        cotangent = torch.zeros(shapes[0], dtype=torch.bfloat16)
        cotangent[list(rows)] = (torch.randn((len(rows), heads, 512), generator=generator) * 0.1).bfloat16()
    reference = CannDsaReference(layout, attention_scale=_SCALE)
    selection = reference.prepare_selection(selected)
    return meta, layout, states, selected, selection, cotangent, rows


def _oracle(meta: DsaBatchMeta, states: tuple, selected: torch.Tensor,
            cotangent: torch.Tensor, rows: tuple[int, ...]) -> tuple:
    q, compressed, qr, kr = states
    inputs = tuple(tensor.float().requires_grad_() for tensor in (q[list(rows)], compressed, qr[list(rows)], kr))
    output, _ = sparse_attention_reference(*inputs, selected[list(rows)], replace(meta, q_global_ids=rows),
                                         attention_scale=_SCALE)
    gradients = torch.autograd.grad((output * cotangent[list(rows)].float()).sum(), inputs)
    grad_query, grad_c, grad_qr, grad_kr = gradients
    full_q = torch.zeros_like(q, dtype=torch.float32)
    full_qr = torch.zeros_like(qr, dtype=torch.float32)
    full_q[list(rows)], full_qr[list(rows)] = grad_query, grad_qr
    return output.detach(), (full_q, grad_c, full_qr, grad_kr)


def _stock(states: tuple, native: torch.Tensor, lengths: torch.Tensor, cotangent: torch.Tensor) -> tuple:
    inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in states)
    query, compressed, query_rope, key_rope = inputs
    key = compressed[:, None, :]
    output = torch.ops.npu.npu_sparse_flash_attention(
        query, key, key, native, _SCALE, actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
        query_rope=query_rope, key_rope=key_rope[:, None, :], layout_query="TND", layout_kv="TND",
        sparse_block_size=1, sparse_mode=3, attention_mode=2, return_softmax_lse=True)
    with torch.no_grad():
        gradients = torch.ops.npu.npu_sparse_flash_attention_grad(
            query, key, key, native, cotangent, *output, _SCALE, 1,
            query_rope=query_rope, key_rope=key_rope[:, None, :], actual_seq_qlen=lengths,
            actual_seq_kvlen=lengths, layout="TND", sparse_mode=3, attention_mode=0)
    autograd_gradients = torch.autograd.grad(output[0], inputs, grad_outputs=cotangent)
    return output, gradients, autograd_gradients


def _difference(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> dict:
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    metric_actual, metric_expected = actual.double(), expected.double()
    delta = metric_actual - metric_expected
    norm_product = (metric_actual.norm() * metric_expected.norm()).clamp_min(1e-30)
    return {"passed": bool(close.all()), "mismatched_elements": int((~close).sum()),
            "elements": actual.numel(), "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / metric_expected.norm().clamp_min(1e-30)),
            "cosine": float((metric_actual * metric_expected).sum() / norm_product), "rtol": rtol, "atol": atol}


def _merged_gradients(gradients: tuple) -> tuple:
    """Match the shared compressed-K/V autograd boundary with one addition."""
    return gradients[0], (gradients[1] + gradients[2])[:, 0], gradients[3], gradients[4][:, 0]


def _run_case(states: tuple, layout: CannDsaLayout, native: torch.Tensor, selection: object,
              cotangent: torch.Tensor, rows: tuple, oracle: tuple, stock: tuple,
              schedule: MixedSfaSchedule) -> dict:
    output, stock_gradients, stock_autograd = stock
    retained = None
    raw_reports = []
    config = schedule.runtime_config(layout.device)
    for repeat in range(2):
        gradients, traces, retained = mixed_sfa_backward_probe(
            states, native, layout.length_tensor, config, cotangent, output, _SCALE, schedule, retained)
        torch.npu.synchronize()
        evidence = [validate_mixed_trace(trace.cpu(), schedule) for trace in traces]
        metrics = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                   for name, actual, expected in zip(("query", "key", "value", "query_rope", "key_rope"),
                                                     gradients, stock_gradients)}
        merged = _merged_gradients(gradients)
        oracle_metrics = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                          for name, actual, expected in zip(_NAMES, merged, oracle[1])}
        raw_reports.append({"repeat": repeat, "trace": evidence, "stock_gradients": metrics,
                            "fp32_oracle_gradients": oracle_metrics})
    inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in states)
    core = MixedDsaCoreProbe(layout, attention_scale=_SCALE, schedule=schedule)
    result = core.attention(*inputs, selection)
    loss = (result.float() * cotangent.float()).sum()
    actual_gradients = torch.autograd.grad(loss, inputs, retain_graph=True)
    repeated_gradients = torch.autograd.grad(loss, inputs)
    torch.npu.synchronize()
    output_metrics = _difference(result, output[0], rtol=0, atol=0)
    oracle_output = _difference(result[list(rows)], oracle[0], rtol=0.02, atol=0.002)
    autograd = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                for name, actual, expected in zip(_NAMES, actual_gradients, oracle[1])}
    autograd_stock = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                      for name, actual, expected in zip(_NAMES, actual_gradients, stock_autograd)}
    retain = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
              for name, actual, expected in zip(_NAMES, repeated_gradients, actual_gradients)}
    native_comparisons = [output_metrics, *autograd_stock.values(), *retain.values()]
    comparisons = [*native_comparisons, oracle_output, *autograd.values()]
    for raw in raw_reports:
        comparisons.extend(raw["stock_gradients"].values())
        comparisons.extend(raw["fp32_oracle_gradients"].values())
        native_comparisons.extend(raw["stock_gradients"].values())
    return {"groups": schedule.compute_groups, "passed": all(item["passed"] for item in comparisons),
            "native_contract_passed": all(item["passed"] for item in native_comparisons),
            "raw_backward_reuse": raw_reports,
            "autograd": autograd, "autograd_stock": autograd_stock,
            "retain_graph": retain, "stock_forward": output_metrics,
            "fp32_oracle_forward": oracle_output}


def _lifecycle(states: tuple, layout: CannDsaLayout, native: torch.Tensor, selection: object,
               cotangent: torch.Tensor, stock: tuple) -> dict:
    """Check independent saved states, reverse backward, serial streams and checkpoint recomputation."""
    core = MixedDsaCoreProbe(layout, attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
    first = tuple(tensor.detach().clone().requires_grad_() for tensor in states)
    second = tuple((tensor.detach() * (0.75 if index == 0 else 1)).requires_grad_()
                   for index, tensor in enumerate(states))
    second_stock = _stock(tuple(tensor.detach() for tensor in second), native, layout.length_tensor, cotangent)
    forward_stream, backward_stream = torch.npu.Stream(), torch.npu.Stream()
    forward_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(forward_stream):
        first_output = core.attention(*first, selection)
        second_output = core.attention(*second, selection)
    backward_stream.wait_stream(forward_stream)
    with torch.npu.stream(backward_stream):
        second_gradients = torch.autograd.grad(second_output, second, grad_outputs=cotangent)
        first_gradients = torch.autograd.grad(first_output, first, grad_outputs=cotangent)
    torch.npu.current_stream().wait_stream(backward_stream)
    torch.npu.synchronize()
    comparisons = {}
    for label, actual, baseline in (("first", first_gradients, stock),
                                     ("second", second_gradients, second_stock)):
        merged = baseline[2]
        comparisons[label] = {name: _difference(value, expected, rtol=0.02, atol=2e-5)
                              for name, value, expected in zip(_NAMES, actual, merged)}
    checkpoint_output = checkpoint(core.attention, *first, selection, use_reentrant=False)
    checkpoint_gradients = torch.autograd.grad(checkpoint_output, first, grad_outputs=cotangent)
    torch.npu.synchronize()
    comparisons["checkpoint"] = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                                 for name, actual, expected in zip(_NAMES, checkpoint_gradients, first_gradients)}
    forward = {"first": _difference(first_output, stock[0][0], rtol=0, atol=0),
               "second": _difference(second_output, second_stock[0][0], rtol=0, atol=0),
               "checkpoint": _difference(checkpoint_output, first_output, rtol=0, atol=0)}
    metrics = [*forward.values(), *(value for comparison in comparisons.values() for value in comparison.values())]
    return {"passed": all(value["passed"] for value in metrics), "compute_groups": 7,
            "reverse_backward_serial_streams": comparisons, "forward": forward}


def run_validation(report: dict, *, long_history: bool, smoke: bool = False) -> None:
    """Validate all gradients, shared K/V ownership and complete three-phase participation."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    report["device"] = torch.npu.get_device_name(0)
    if "910B3" not in report["device"]:
        raise RuntimeError("mixed SFA training probe initially supports verified 910B3 hardware")
    fixtures = [((3, 10), 32)] if smoke else [((3, 10), 32), ((19, 29, 33), 64)]
    if long_history:
        fixtures.extend([((513,), 32), ((4096,), 32)])
    report["cases"] = []
    report["lifecycle"] = []
    for lengths, heads in fixtures:
        report["stage"] = {"lengths": lengths, "heads": heads, "operation": "reference"}
        meta, layout, cpu, selected, selection, cotangent, rows = _fixture(lengths, heads, device)
        oracle = _oracle(meta, cpu, selected, cotangent, rows)
        states = tuple(tensor.to(device) for tensor in cpu)
        cotangent = cotangent.to(device)
        native = layout.global_to_sequence_indices(selected.to(device))
        stock = _stock(states, native, layout.length_tensor, cotangent)
        merged = _merged_gradients(stock[1])
        stock_oracle = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                        for name, actual, expected in zip(_NAMES, merged, oracle[1])}
        shared_kv_oracle = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
                            for name, actual, expected in zip(_NAMES, stock[2], oracle[1])}
        report["stage"] = {"lengths": lengths, "heads": heads, "operation": "lifecycle"}
        lifecycle = _lifecycle(states, layout, native, selection, cotangent, stock)
        lifecycle.update(lengths=lengths, heads=heads)
        report["lifecycle"].append(lifecycle)
        for groups in ((1,) if smoke else (1, 2, 7, 19)):
            report["stage"] = {"lengths": lengths, "heads": heads, "groups": groups}
            result = _run_case(states, layout, native, selection, cotangent, rows, oracle, stock,
                               MixedSfaSchedule(groups))
            result.update(lengths=lengths, heads=heads, nonzero_gradient_query_ids=rows,
                          stock_fp32_oracle_gradients=stock_oracle,
                          stock_shared_kv_autograd_fp32_oracle_gradients=shared_kv_oracle)
            report["cases"].append(result)
    passed = all(case["passed"] for case in (*report["cases"], *report["lifecycle"]))
    report.update(status="passed" if passed else "failed", stage="complete", case_count=len(report["cases"]))
    if not passed:
        raise RuntimeError("mixed SFA acceptance failed; all completed comparisons are retained in the report")


def main() -> None:
    """Persist the failing stage and completed evidence without changing thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "CP1 mixed SFA forward/first-order backward",
              "communication_progress": False, "performance_measurement": False}
    try:
        run_validation(report, long_history=args.long_history, smoke=args.smoke)
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if report["status"] != "failed":
            report["status"] = "error"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
