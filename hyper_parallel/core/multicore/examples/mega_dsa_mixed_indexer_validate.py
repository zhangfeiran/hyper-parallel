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
"""Single-NPU mixed LI probe including retained long-history Top-K merge."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_indexer import (
    mixed_indexer_forward_probe,
    mixed_indexer_fused_forward_probe,
    validate_fused_indexer_traces,
    validate_mixed_indexer_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
)

_K = 2048


def _fixture(lengths: tuple[int, ...], case: str, device: torch.device) -> tuple:
    meta = DsaBatchMeta.packed(lengths)
    layout = CannDsaLayout(meta, device)
    tokens = meta.global_valid_queries
    generator = torch.Generator().manual_seed(20261008 + tokens)
    if case in ("random_signed", "random_signed_fp32_weights"):
        shapes = ((tokens, 64, 128), (tokens, 1, 128), (tokens, 64))
        states = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in shapes)
        if case == "random_signed_fp32_weights":
            states = (states[0], states[1], states[2].float())
    else:
        query = torch.zeros(tokens, 64, 128)
        key = torch.zeros(tokens, 1, 128)
        weight = torch.zeros(tokens, 64)
        positions = torch.tensor([meta.sequence_position(token)[1] for token in range(tokens)])
        if case in ("increasing", "signed_decreasing"):
            key[:, 0, 0], key[:, 0, 1] = positions // 128, positions % 128
            query[:, 0, 0], query[:, 0, 1] = 128, 1
            weight[:, 0] = 1
            if case == "signed_decreasing":
                query[:, 1, 0], query[:, 1, 1] = 256, 2
                weight[:, 1] = -1
        elif case == "concentrated":
            query[:, 0, 0] = 1
            key[:, 0, 0] = torch.where(positions < _K, 4, 1)
            weight[:, 0] = 1
        elif case != "all_tied":
            raise ValueError("unknown mixed LI fixture")
        states = (query.bfloat16(), key.bfloat16(), weight.bfloat16())
    return meta, layout, states, tuple(state.to(device) for state in states)


def _contract(raw: torch.Tensor, meta: DsaBatchMeta) -> dict:
    indices = raw[:, 0]
    positions = torch.tensor([meta.sequence_position(token)[1] for token in range(meta.global_valid_queries)])
    present = indices >= 0
    ordered = indices.sort(dim=-1).values
    valid = ((indices == -1) | (present & (indices <= positions[:, None]))).all(dim=1)
    unique = ~((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any(dim=1)
    count = present.sum(dim=1) == (positions + 1).clamp_max(_K)
    trailing = ~(present[:, 1:] & ~present[:, :-1]).any(dim=1)
    failed = ~(valid & unique & count & trailing)
    if bool(failed.any()):
        raise RuntimeError(f"mixed LI index contract failed: query IDs {failed.nonzero().flatten().tolist()}")
    return {"queries_checked": indices.shape[0], "legal_unique_full_cardinality_trailing_padding": True}


def _analytic(raw: torch.Tensor, meta: DsaBatchMeta, case: str) -> bool | None:
    if case not in ("increasing", "signed_decreasing", "concentrated"):
        return None
    positions = torch.tensor([meta.sequence_position(token)[1] for token in range(meta.global_valid_queries)])[:, None]
    slots = torch.arange(_K)[None]
    begin = (positions + 1 - _K).clamp_min(0) if case == "increasing" else torch.zeros_like(positions)
    expected = torch.where(slots < (positions + 1).clamp_max(_K), slots + begin, -1).to(torch.int32)
    torch.testing.assert_close(raw[:, 0].sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0)
    return True


def _run_case(meta: DsaBatchMeta, layout: CannDsaLayout, states: tuple,
              schedule: MixedSfaSchedule, baseline: tuple, case: str, fused: bool) -> dict:
    # Repeating complete pairs tests scratch reuse without destroying live merge state.
    retained = None
    repeats = []
    forward = mixed_indexer_fused_forward_probe if fused else mixed_indexer_forward_probe
    validate = validate_fused_indexer_traces if fused else validate_mixed_indexer_traces
    for repeat in range(3):
        indices, values, traces, retained = forward(
            *states, layout.length_tensor, schedule, retained)
        torch.npu.synchronize()
        snapshots = tuple(trace.cpu() for trace in traces)
        lengths = tuple(end - start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
        evidence = validate(snapshots, schedule, require_ld=max(lengths) > _K)
        actual = (indices.cpu(), values.cpu())
        torch.testing.assert_close(actual[0], baseline[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[1].view(torch.int16), baseline[1].view(torch.int16), rtol=0, atol=0)
        repeats.append({"repeat": repeat, "trace": evidence,
                        "stock_indices_order_bitwise_equal": True, "stock_values_bitwise_equal": True,
                        "stock_values_storage_bits_equal": True,
                        "raw_contract": _contract(actual[0], meta),
                        "analytic_all_row_winners": _analytic(actual[0], meta, case)})
    return {"groups": schedule.compute_groups, "device_phase_closure": fused, "repeats": repeats}


def run_validation(report: dict, *, long_history: bool, fused: bool = False) -> None:
    """Require stock bitwise parity and all-row legal Top-K across group schedules."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    report["device"] = torch.npu.get_device_name(0)
    if "910B3" not in report["device"]:
        raise RuntimeError("the initial mixed LI probe is restricted to verified 910B3 hardware")
    fixtures = [((3, 10), "random_signed"), ((3, 10), "random_signed_fp32_weights"),
                ((19, 29, 33), "increasing")]
    if long_history:
        fixtures.extend(((2176, 2240), case) for case in
                        ("increasing", "signed_decreasing", "concentrated", "all_tied",
                         "random_signed", "random_signed_fp32_weights"))
    report["cases"] = []
    for lengths, case in fixtures:
        report["stage"] = {"lengths": lengths, "case": case, "operation": "stock reference"}
        meta, layout, _, states = _fixture(lengths, case, device)
        baseline = torch.ops.npu.npu_lightning_indexer(
            *states, actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_key=layout.length_tensor,
            layout_query="TND", layout_key="TND", sparse_count=_K, sparse_mode=3, return_value=True)
        baseline = tuple(tensor.cpu() for tensor in baseline)
        for groups in (1, 2, 7, 19):
            report["stage"] = {"lengths": lengths, "case": case, "groups": groups}
            result = _run_case(meta, layout, states, MixedSfaSchedule(groups), baseline, case, fused)
            result.update(lengths=lengths, fixture=case)
            report["cases"].append(result)
    report.update(status="passed", stage="complete", case_count=len(report["cases"]))


def main() -> None:
    """Persist the failing stage and completed evidence before propagating errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "CP1 mixed LI forward probe", "device_phase_closure": args.fused,
              "backward": False, "communication_progress": False, "performance_measurement": False}
    try:
        run_validation(report, long_history=args.long_history, fused=args.fused)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
