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
"""Execute real callable SFA tasks on reusable mixed groups; forward-only P2 probe."""

import argparse
import json
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.

from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import (
    MixedSfaSchedule,
    validate_mixed_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.torch.ops import _load_native

_SCALE = 192**-0.5


def _fixture(lengths: tuple[int, ...], heads: int, device: torch.device) -> tuple:
    meta = DsaBatchMeta.packed(lengths)
    layout = CannDsaLayout(meta, device)
    tokens = meta.global_valid_queries
    generator = torch.Generator().manual_seed(20261008 + tokens + heads)
    shapes = ((tokens, heads, 512), (tokens, 512), (tokens, heads, 64), (tokens, 64))
    cpu = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in shapes)
    selected = torch.full((tokens, 2048), -1, dtype=torch.int32)
    for query in range(tokens):
        sequence, position = meta.sequence_position(query)
        start = meta.global_cu_seqlens[sequence]
        selected[query, :position + 1] = torch.arange(start, query + 1, dtype=torch.int32)
    native = layout.global_to_sequence_indices(selected.to(device))
    states = tuple(tensor.to(device) for tensor in cpu)
    return meta, layout, selected, native, cpu, states


def _reference(states: tuple, native: torch.Tensor, layout: CannDsaLayout) -> tuple:
    query, compressed, query_rope, key_rope = states
    key = compressed[:, None, :]
    return torch.ops.npu.npu_sparse_flash_attention(
        query, key, key, native, _SCALE,
        actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_kv=layout.length_tensor,
        query_rope=query_rope, key_rope=key_rope[:, None, :], layout_query="TND", layout_kv="TND",
        sparse_block_size=1, sparse_mode=3, attention_mode=2, return_softmax_lse=True)


def _difference(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> dict:
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    delta = actual - expected
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-30)),
            "bitwise_equal": bool(torch.equal(actual, expected)), "rtol": rtol, "atol": atol}


def _run_case(states: tuple, native: torch.Tensor, layout: CannDsaLayout,
              schedule: MixedSfaSchedule, baseline: tuple, oracle: tuple) -> dict:
    query, compressed, query_rope, key_rope = states
    key = compressed[:, None, :]
    config = schedule.runtime_config(query.device)
    trace = schedule.new_trace(query.device)
    output = torch.full_like(query, float("nan"))
    stats_shape = (1, query.shape[0], query.shape[1])
    maximum = torch.full(stats_shape, float("nan"), dtype=torch.float32, device=query.device)
    denominator = torch.full_like(maximum, float("nan"))
    torch.ops.hyper_parallel.dsa_mixed_tile_out(
        query, key, key, native, layout.length_tensor, layout.length_tensor,
        query_rope, key_rope[:, None, :], config, trace, _SCALE, output, maximum, denominator)
    torch.npu.synchronize()
    evidence = validate_mixed_trace(trace.cpu(), schedule)
    metrics = {name: _difference(actual, expected, rtol=0.02, atol=2e-5)
               for name, actual, expected in zip(("output", "maximum", "sum"),
                                                 (output, maximum, denominator), baseline)}
    oracle_out, oracle_stats = oracle
    lse = maximum + denominator.log()
    oracle_metrics = {"output": _difference(output, oracle_out, rtol=0.02, atol=0.002),
                      "lse": _difference(lse[0], oracle_stats.lse, rtol=0.02, atol=0.002)}
    return {"schedule": {"groups": schedule.compute_groups, "rounds": schedule.rounds},
            "trace": evidence, "stock_native_cp1": metrics, "fp32_oracle": oracle_metrics}


def run_validation(report: dict, *, long_history: bool) -> None:
    """Run single/multiple groups, repeated tasks, packed boundaries and two head shapes."""
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    report["device"] = torch.npu.get_device_name(0)
    if "910B3" not in report["device"]:
        raise RuntimeError("the initial mixed SFA probe is restricted to verified 910B3 hardware")
    _load_native()
    if torch.ops.hyper_parallel.dsa_mixed_tile_version() != 1:
        raise RuntimeError("mixed SFA adapter ABI mismatch; rebuild this checkout's payload")
    fixtures = [((3, 10), 32), ((19, 29, 33), 64)]
    if long_history:
        fixtures.append(((513,), 32))
    report["cases"] = []
    for lengths, heads in fixtures:
        meta, layout, selected, native, cpu, states = _fixture(lengths, heads, device)
        report["stage"] = {"lengths": lengths, "heads": heads, "operation": "references"}
        baseline = _reference(states, native, layout)
        oracle = sparse_attention_reference(*(tensor.float() for tensor in cpu), selected, meta,
                                            attention_scale=_SCALE)
        for groups in (1, 2, 7, 19):
            for rounds in (1, 3):
                report["stage"] = {"lengths": lengths, "heads": heads, "groups": groups, "rounds": rounds}
                result = _run_case(states, native, layout, MixedSfaSchedule(groups, rounds), baseline, oracle)
                result.update({"lengths": lengths, "heads": heads})
                report["cases"].append(result)
    report.update(status="passed", stage="complete", case_count=len(report["cases"]))


def main() -> None:
    """Persist failures as well as successful numerical and member-participation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "CP1 external TopK SFA mixed-group forward probe",
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
