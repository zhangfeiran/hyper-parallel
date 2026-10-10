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
"""Single-launch LI/TopK/SFA/KL arithmetic with six device phase handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

import hyper_parallel
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import _inputs, _metrics, _SCALE
from hyper_parallel.core.multicore.examples.mega_dsa_training_oracle import training_reference
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout, CannDsaStats, _selected_kl_gradients,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import _reference
from hyper_parallel.core.multicore.modules.mega_dsa.fused_training import (
    fused_dsa_training_probe, validate_fused_training_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


def _case(report, heads, lengths, device, weight_dtype):
    meta = DsaBatchMeta.packed(lengths)
    layout = CannDsaLayout(meta, device)
    original, cotangent = _inputs(heads, torch.bfloat16, lengths)
    fixtures = []
    for repeat in range(3):
        factor = 1 + repeat * .125
        cpu = tuple((tensor * factor).bfloat16() if field in (0, 6) else tensor
                    for field, tensor in enumerate(original))
        weight = cpu[6].to(weight_dtype)
        if weight_dtype == torch.float32:
            weight = weight * 1.003 + .000013
        cpu = (*cpu[:6], weight)
        full = tuple(tensor.to(device) for tensor in cpu)
        native, values = torch.ops.npu.npu_lightning_indexer(
            full[4], full[5][:, None], full[6], actual_seq_lengths_query=layout.length_tensor,
            actual_seq_lengths_key=layout.length_tensor, layout_query="TND", layout_key="TND",
            sparse_count=2048, sparse_mode=3, return_value=True)
        attention, maximum, denominator = _reference(full[:4], native, layout)
        stats = CannDsaStats(maximum, denominator)
        indices = layout.sequence_to_global_indices(native).cpu()
        oracle = training_reference(cpu, cotangent, indices, DsaLossNormalization(sum(lengths), sum(lengths)),
                                    1., "kl_only", 1., lengths, attention_scale=_SCALE)
        stock = torch.ops.npu.npu_sparse_lightning_indexer_grad_kl_loss(
            full[0], full[1][:, None], full[4], full[5][:, None], full[6], native,
            stats.maximum, stats.denominator, _SCALE, query_rope=full[2], key_rope=full[3][:, None],
            actual_seq_qlen=list(layout.cumulative_lengths), actual_seq_klen=list(layout.cumulative_lengths),
            layout="TND", sparse_mode=3)
        stock = (stock[0], stock[1][:, 0], stock[2], stock[3].reshape(()))
        baseline = (_selected_kl_gradients(*full[4:], full[:4], native, stats, layout, _SCALE)
                    if weight_dtype == torch.bfloat16 else stock)
        fixtures.append((full, (native, values, attention, maximum, denominator),
                         baseline, (*oracle[2][4:], oracle[1]), stock))
    for groups in (1, 2, 7, 19):
        schedule = MixedSfaSchedule(groups)
        report["stage"] = {"heads": heads, "lengths": lengths, "groups": groups, "weight_dtype": str(weight_dtype)}
        retained = []
        for full, _, _, _, _ in fixtures:
            result = fused_dsa_training_probe(full[4:], full[:4], layout, _SCALE, schedule)
            retained.append(result)
        torch.npu.synchronize()
        for repeat, result in enumerate(retained):
            forward, baseline, oracle_values, stock = fixtures[repeat][1:]
            actual = (*result.index_gradients, result.loss)
            native_metrics, fp32_metrics = [], []
            for name, value, expected, independent in zip(
                    ("index_query", "index_key", "merge_weight", "loss"), actual, baseline, oracle_values):
                measurement = _metrics(value, expected)
                native_metrics.append({"name": name, **measurement})
                fp32 = _metrics(value, independent)
                fp32_metrics.append({"name": name, **fp32})
            case = {"heads": heads, "lengths": lengths, "groups": groups, "repeat": repeat,
                    "weight_dtype": str(weight_dtype),
                    "native_backend": "omni" if weight_dtype == torch.bfloat16 else "stock",
                    "native": native_metrics, "fp32": fp32_metrics,
                    "stock": [_metrics(value, expected) for value, expected in zip(actual, stock)],
                    "trace": validate_fused_training_traces(tuple(trace.cpu() for trace in result.traces), schedule,
                                                          require_ld=max(lengths) > 2048),
                    "forward_exact": [bool(torch.equal(value.cpu(), expected.cpu()))
                                      for value, expected in zip(result.forward, forward)]}
            report["cases"].append(case)
            if not all(case["forward_exact"]):
                raise RuntimeError(f"fused LI/SFA original stock mismatch: {case['forward_exact']}")
            if not all(item["pointwise_pass"] for item in case["stock"]):
                raise RuntimeError(f"fused KL original stock mismatch: {case['stock']}")
            if not all(item["pointwise_pass"] for item in native_metrics):
                raise RuntimeError(f"fused KL native mismatch: {native_metrics}")
            if not all(item["relative_l2"] <= .02 or item["max_abs"] <= 2e-5 for item in fp32_metrics):
                raise RuntimeError(f"fused KL independent FP32 mismatch: {fp32_metrics}")


def run_validation(report: dict, *, long_history: bool = False, weight_dtype: torch.dtype | None = None) -> None:
    """Run complete selected KL across group counts and independent retained invocations."""
    if weight_dtype not in (None, torch.bfloat16, torch.float32):
        raise ValueError("KL validation weights must be BF16 or FP32")
    torch.npu.set_device(0)
    fixtures = [(32, (3, 10)), (64, (3, 10))]
    if long_history:
        fixtures += [(32, (3, 2113)), (64, (2176, 2240))]
    for heads, lengths in fixtures:
        for dtype in ((torch.bfloat16, torch.float32) if weight_dtype is None else (weight_dtype,)):
            _case(report, heads, lengths, torch.device("npu:0"), dtype)
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
              if any(name in line for name in ("libcust_opapi.so", "libcust_opmaster_rt2.0.so",
                                               "libhyper_parallel_mega_moe_torch.so"))}
    report.update(status="passed", stage="complete", torch_version=torch.__version__,
                  torch_npu_version=torch_npu.__version__, package_path=hyper_parallel.__file__,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  loaded_library_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                         for path in sorted(loaded)})


def main() -> None:
    """Save every phase result, including failed metrics, before claiming numerical acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-history", action="store_true")
    parser.add_argument("--weight-dtype", choices=("bf16", "fp32"), help="Focus one variant; default validates both")
    args = parser.parse_args()
    report = {"status": "running", "cases": [], "scope": "single-launch LI/TopK/SFA/KL six-phase DAG"}
    try:
        run_validation(report, long_history=args.long_history,
                       weight_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}.get(args.weight_dtype))
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
