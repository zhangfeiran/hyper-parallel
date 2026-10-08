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
"""Verify single-kernel LI/merge/SFA handoff, ordered packed inputs and scratch reuse."""

import argparse
import json
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import (
    _analytic,
    _contract,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import (
    _fixture as indexer_fixture,
)
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import (
    _reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_forward import (
    fused_dsa_forward_probe,
    validate_fused_dsa_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule

_SCALE = 192**-0.5


def _main_states(tokens: int, heads: int, device: torch.device) -> tuple:
    generator = torch.Generator().manual_seed(20261008 + tokens + heads)
    shapes = ((tokens, heads, 512), (tokens, 512), (tokens, heads, 64), (tokens, 64))
    return tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16().to(device) for shape in shapes)


def _baseline(index_states: tuple, main_states: tuple, layout) -> tuple:
    selection = torch.ops.npu.npu_lightning_indexer(
        *index_states, actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_key=layout.length_tensor,
        layout_query="TND", layout_key="TND", sparse_count=2048, sparse_mode=3, return_value=True)
    forward = _reference(main_states, selection[0], layout)
    return tuple(tensor.cpu() for tensor in (*selection, *forward))


def _run_case(meta, layout, index_states: tuple, main_states: tuple,
              schedule: MixedSfaSchedule, baseline: tuple, fixture: str, require_ld: bool) -> dict:
    retained = None
    repeats = []
    previous = None
    names = ("indices", "values", "attention", "maximum", "sum")
    for repeat in range(3):
        result = fused_dsa_forward_probe(index_states, main_states, layout.length_tensor,
                                         _SCALE, schedule, retained)
        *outputs, traces, retained = result
        torch.npu.synchronize()
        evidence = validate_fused_dsa_traces(tuple(trace.cpu() for trace in traces), schedule, require_ld=require_ld)
        actual = tuple(tensor.cpu() for tensor in outputs)
        for name, tensor, expected in zip(names, actual, baseline):
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(actual[1].view(torch.int16), baseline[1].view(torch.int16), rtol=0, atol=0)
        if previous is not None:
            for tensor, expected in zip(outputs, previous):
                if tensor.is_set_to(expected):
                    raise RuntimeError("separate fused invocations alias output storage")
        previous = outputs
        repeats.append({"repeat": repeat, "trace": evidence,
                        "stock_exact_outputs": list(names), "values_storage_bits_equal": True,
                        "raw_index_contract": _contract(actual[0], meta),
                        "analytic_all_row_winners": _analytic(actual[0], meta, fixture)})
    return {"groups": schedule.compute_groups, "repeats": repeats}


def run_validation(report: dict, *, long_history: bool) -> None:
    """Require exact stock parity after all three native phases on four mixed group schedules."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    report["device"] = torch.npu.get_device_name(0)
    if "910B3" not in report["device"]:
        raise RuntimeError("fused DSA currently requires verified 910B3 hardware")
    fixtures = [((3, 10), "random_signed"), ((3, 10), "random_signed_fp32_weights"),
                ((19, 29, 33), "increasing")]
    if long_history:
        fixtures.extend(((2176, 2240), fixture) for fixture in
                        ("increasing", "signed_decreasing", "concentrated", "all_tied",
                         "random_signed", "random_signed_fp32_weights"))
    report["cases"] = []
    for lengths, fixture in fixtures:
        meta, layout, _, index_states = indexer_fixture(lengths, fixture, device)
        for heads in (32, 64):
            report["stage"] = {"lengths": lengths, "fixture": fixture, "heads": heads, "op": "stock reference"}
            main_states = _main_states(meta.global_valid_queries, heads, device)
            baseline = _baseline(index_states, main_states, layout)
            for groups in (1, 2, 7, 19):
                report["stage"] = {"lengths": lengths, "fixture": fixture, "heads": heads, "groups": groups}
                result = _run_case(meta, layout, index_states, main_states, MixedSfaSchedule(groups),
                                   baseline, fixture, max(lengths) > 2048)
                result.update(lengths=lengths, fixture=fixture, heads=heads)
                report["cases"].append(result)
    report.update(status="passed", stage="complete", case_count=len(report["cases"]),
                  invocation_count=sum(len(case["repeats"]) for case in report["cases"]))


def main() -> None:
    """Persist completed checks and the failing stage for fresh-process replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "CP1 single-kernel LI-main/merge/SFA forward",
              "backward": False, "communication_progress": False, "performance_measurement": False}
    try:
        run_validation(report, long_history=args.long_history)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
