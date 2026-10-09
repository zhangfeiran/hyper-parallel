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
"""CP training composition: native selection, seven gradients and separated LM/KL objectives."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata, _SCALE
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import _reference as stock_attention
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaStats, _SelectedKlFunction, _load_custom_ops,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.module import MegaDsa
from hyper_parallel.core.multicore.modules.mega_dsa.reference import selected_kl_reference, sparse_attention_reference
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_LENGTHS = (3, 10)
_FIELDS = ("query", "compressed", "query_rope", "key_rope", "index_query", "index_key", "merge_weight")


def _inputs(heads, weight_dtype):
    generator = torch.Generator().manual_seed(20261009 + heads)
    shapes = ((13, heads, 512), (13, 512), (13, heads, 64), (13, 64), (13, 64, 128), (13, 128), (13, 64))
    tensors = tuple((torch.randn(shape, generator=generator) * .1).bfloat16() for shape in shapes)
    return (*tensors[:6], tensors[6].to(weight_dtype)), (torch.randn(shapes[0], generator=generator) * .01).bfloat16()


def _objective(output, loss, cotangent, mode, auxiliary):
    if mode == "kl_only":
        return loss * auxiliary
    main_loss = (output * cotangent).sum()
    return main_loss if mode == "lm_only" else main_loss + loss * auxiliary


def _metrics(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = actual.double() - expected.double()
    return {"max_abs": float(delta.abs().max()) if delta.numel() else 0.,
            "relative_l2": float(delta.norm() / expected.double().norm().clamp_min(1e-30)),
            "pointwise_pass": bool(torch.allclose(actual, expected, rtol=.02, atol=2e-5))}


def _full_native(inputs, cotangent, result, layout, normalization, coefficient, mode, auxiliary):
    full = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    saved = result.saved
    output, maximum, denominator = _load_custom_ops().npu_sparse_flash_attention_enhance(
        full[0], full[1][:, None], full[1][:, None], saved.indices, _SCALE,
        block_table=None, actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_kv=layout.length_tensor, query_rope=full[2], key_rope=full[3][:, None],
        sparse_block_size=1, layout_query="TND", layout_kv="TND", sparse_mode=3,
        attention_mode=2, return_softmax_lse=True)
    if coefficient == 0:
        loss = sum(tensor.float().sum() * 0 for tensor in full[4:])
    else:
        loss = _SelectedKlFunction.apply(*full[4:], full[:4], saved.indices,
                                        CannDsaStats(maximum, denominator), layout, _SCALE,
                                        coefficient * normalization.local_sum_scale)
    _objective(output, loss, cotangent, mode, auxiliary).backward()
    return output, loss, tuple(tensor.grad for tensor in full)


def _cpu_reference(inputs, cotangent, indices, normalization, coefficient, mode, auxiliary):
    full = tuple(tensor.detach().float().cpu().requires_grad_() for tensor in inputs)
    meta = DsaBatchMeta.packed(_LENGTHS)
    output, _ = sparse_attention_reference(*full[:4], indices, meta, attention_scale=_SCALE)
    loss = selected_kl_reference(*full[4:], *full[:4], indices, meta, attention_scale=_SCALE,
                                 normalization=normalization, loss_coeff=coefficient)
    _objective(output, loss, cotangent.float().cpu(), mode, auxiliary).backward()
    return output, loss, tuple(tensor.grad for tensor in full)


def _run_case(report, device, pattern, heads, weight_dtype, mode, auxiliary, divisor):
    meta = _metadata(_LENGTHS, pattern, dist.get_world_size(), dist.get_rank())
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(dist.get_world_size())))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-training", DsaWorkspaceSpec(capacity, 13, capacity, dist.get_world_size()))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    try:
        prepared = workspace.prepare(meta)
        normalization = DsaLossNormalization(13, divisor)
        coefficient = 0. if mode == "coeff_zero" else .3
        model = MegaDsa(workspace, prepared, heads=heads, attention_scale=_SCALE, schedule=MixedSfaSchedule(7),
                        normalization=normalization, loss_coeff=coefficient)
        cpu, cotangent = _inputs(heads, weight_dtype)
        full = tuple(tensor.to(device) for tensor in cpu)
        cotangent = cotangent.to(device)
        local = tuple(tensor[list(meta.kv_global_ids if field in (1, 3, 5) else meta.q_global_ids)]
                      .clone().requires_grad_() for field, tensor in enumerate(full))
        local_cotangent = cotangent[list(meta.q_global_ids)]
        forwards, backwards = [], []
        native_forward = model.core.backend.forward
        native_backward = model.core.backward_backend._backward_saved

        def _forward(*args):
            value = native_forward(*args)
            forwards.append(value)
            return value

        def _backward(*args):
            value = native_backward(*args)
            backwards.append(value)
            return value

        if mode == "fp32_rejected":
            with patch.object(model.core.backend, "forward") as launch:
                try:
                    model(*local, meta)
                except ValueError as error:
                    if "KL requires BF16" not in str(error):
                        raise
                else:
                    raise RuntimeError("FP32 selected KL must reject before native submission")
                launch.assert_not_called()
            report["cases"].append({"pattern": pattern, "heads": heads, "weight_dtype": str(weight_dtype),
                                    "mode": mode, "status": "rejected_before_native_dispatch"})
            return
        with patch.object(model.core.backend, "forward", side_effect=_forward), \
                patch.object(model.core.backward_backend, "_backward_saved", side_effect=_backward):
            output, loss = model(*local, meta)
            if mode == "checkpoint":
                output, loss = checkpoint(lambda *values: model(*values, meta), *local, use_reentrant=False)
            elif mode == "delayed":
                model(*(tensor.detach() + .0625 for tensor in local), replace(meta, layer=3, microbatch=4))
            objective = _objective(output, loss, local_cotangent, mode, auxiliary)
            if mode == "cross_stream":
                event = torch.npu.Event()
                event.record()
                with torch.npu.stream(torch.npu.Stream()):
                    event.wait()
                    objective.backward()
            else:
                objective.backward(retain_graph=mode == "retain")
            if mode == "retain":
                first = tuple(tensor.grad.detach().clone() for tensor in local)
                for tensor in local:
                    tensor.grad = None
                objective.backward()
                for actual, expected in zip(local, first):
                    torch.testing.assert_close(actual.grad, expected, rtol=.02, atol=2e-5)
        torch.npu.synchronize()
        raw = forwards[1] if mode == "checkpoint" else forwards[0]
        native_layout = model.core.backend.native_layout
        indices = native_layout.sequence_to_global_indices(raw.saved.indices).cpu()
        stock_indices, _ = torch.ops.npu.npu_lightning_indexer(
            full[4], full[5][:, None], full[6], actual_seq_lengths_query=native_layout.length_tensor,
            actual_seq_lengths_key=native_layout.length_tensor, layout_query="TND", layout_key="TND",
            sparse_count=2048, sparse_mode=3, return_value=True)
        torch.testing.assert_close(raw.saved.indices.cpu().sort(dim=-1).values,
                                   stock_indices.cpu().sort(dim=-1).values, rtol=0, atol=0)
        baseline = _full_native(full, cotangent, raw, native_layout,
                                normalization, coefficient, mode, auxiliary)
        oracle = _cpu_reference(full, cotangent, indices, normalization, coefficient, mode, auxiliary)
        gradients, oracle_gradients = {}, {}
        for field, (name, tensor, expected, reference) in enumerate(zip(_FIELDS, local, baseline[2], oracle[2])):
            rows = list(meta.kv_global_ids if field in (1, 3, 5) else meta.q_global_ids)
            if expected is None:
                if tensor.grad is not None:
                    raise RuntimeError(f"{name}: gradient isolation failed")
                gradients[name] = {"absent": True}
                continue
            torch.testing.assert_close(tensor.grad.float().cpu(), expected[rows].float().cpu(), rtol=.02, atol=2e-5)
            gradients[name] = _metrics(tensor.grad, expected[rows])
            metric = _metrics(tensor.grad, reference[rows])
            if (not math.isfinite(metric["relative_l2"]) or not math.isfinite(metric["max_abs"])
                    or (metric["relative_l2"] > .02 and metric["max_abs"] > 2e-5)):
                raise RuntimeError(f"{name}: independent FP32 mismatch {metric}")
            if coefficient == 0 or (auxiliary == 0 and field >= 4):
                if field >= 4 and bool(tensor.grad.count_nonzero()):
                    raise RuntimeError("disabled KL must return exact zero index gradients")
            oracle_gradients[name] = metric
        summed_loss = loss.detach().clone()
        dist.all_reduce(summed_loss)
        stock_output, _, _ = stock_attention(full[:4], raw.saved.indices, native_layout)
        torch.testing.assert_close(output.cpu(), stock_output[list(meta.q_global_ids)].cpu(), rtol=0, atol=0)
        output_metric = _metrics(output, oracle[0][list(meta.q_global_ids)])
        if (not math.isfinite(output_metric["relative_l2"]) or not math.isfinite(output_metric["max_abs"])
                or (output_metric["relative_l2"] > .02 and output_metric["max_abs"] > 2e-5)):
            raise RuntimeError(f"independent FP32 output mismatch {output_metric}")
        torch.testing.assert_close(summed_loss.float().cpu(), baseline[1].detach().float().cpu(), rtol=.02, atol=2e-5)
        loss_metric = _metrics(summed_loss, oracle[1])
        if (not math.isfinite(loss_metric["relative_l2"]) or not math.isfinite(loss_metric["max_abs"])
                or (loss_metric["relative_l2"] > .02 and loss_metric["max_abs"] > 2e-5)):
            raise RuntimeError(f"independent FP32 KL loss mismatch {loss_metric}")
        traces = [model.core.backend.validate_trace(value, tuple(trace.cpu() for trace in value.phase_traces),
                                                    value.transport_trace.cpu(), require_overlap=False)
                  for value in forwards]
        gradient_traces = [model.core.backward_backend.validate_trace(
            value, tuple(trace.cpu() for trace in value.phase_traces), value.transport_trace.cpu())
            for value in backwards]
        report["cases"].append({"pattern": pattern, "heads": heads, "weight_dtype": str(weight_dtype),
                                "mode": mode, "auxiliary": auxiliary, "reducer_divisor": divisor,
                                "kl_reference": "exact_zero" if coefficient == 0 else "enhance_native",
                                "selection_stock_set_exact": True, "output_stock_exact": True,
                                "output_enhance": _metrics(output, baseline[0][list(meta.q_global_ids)]),
                                "output_fp32": output_metric, "loss_native": _metrics(summed_loss, baseline[1]),
                                "loss_fp32": _metrics(summed_loss, oracle[1]), "gradients_native": gradients,
                                "gradients_fp32": oracle_gradients, "forward_traces": traces,
                                "backward_traces": gradient_traces})
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, output_dir: Path, *, smoke: bool = False) -> None:
    """Check global KL normalization and all seven owner gradients on CP1/CP2/CP4."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[],
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  custom_opp_path=os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                  library_path=os.environ.get("LD_LIBRARY_PATH"))
    cases = [("strided", 32, torch.bfloat16, "joint", 7, 1)]
    if not smoke:
        cases += [("zigzag", 64, torch.bfloat16, "joint", 1, 1),
                  ("empty", 32, torch.bfloat16, "joint", 7, 1),
                  ("strided", 32, torch.bfloat16, "lm_only", 7, 1),
                  ("strided", 32, torch.bfloat16, "kl_only", 7, 1),
                  ("strided", 32, torch.bfloat16, "coeff_zero", 7, 1),
                  ("strided", 32, torch.bfloat16, "joint", 0, 1),
                  ("zigzag", 64, torch.bfloat16, "checkpoint", 7, 1),
                  ("empty", 64, torch.bfloat16, "checkpoint", 7, 1),
                  ("zigzag", 32, torch.bfloat16, "retain", 7, 1),
                  ("zigzag", 64, torch.bfloat16, "delayed", 7, 1),
                  ("zigzag", 32, torch.bfloat16, "cross_stream", 7, 1),
                  ("empty", 32, torch.bfloat16, "kl_only", 1, 1),
                  ("strided", 64, torch.bfloat16, "joint", 7, dist.get_world_size()),
                  ("empty", 64, torch.bfloat16, "joint", 7, dist.get_world_size()),
                  ("strided", 64, torch.float32, "coeff_zero", 7, 1),
                  ("zigzag", 64, torch.float32, "fp32_rejected", 1, 1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        report["stage"] = tuple(str(item) for item in case)
        _run_case(report, torch.device(f"npu:{local_rank}"), *case)
        (output_dir / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
              if "libcust_opapi.so" in line}
    report.update(status="passed", stage="complete", loaded_op_api_libraries=sorted(loaded))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist per-rank evidence including native, independent FP32 and lifecycle results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "P4 native selection plus host selected KL"}
    try:
        run_validation(report, args.output_dir, smoke=args.smoke)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
