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
"""Torchrun CP baseline validation with per-rank evidence; no fused/performance claim."""

import argparse
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cann_validate import _metrics
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import (
    CannDsaCpReference,
    DsaCpLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)

_NAMES = ("query", "compressed_kv", "query_rope", "key_rope", "index_query", "index_key", "merge_weight")
_LENGTHS = (3, 10)
_SHAPES = ((13, 32, 512), (13, 512), (13, 32, 64), (13, 64), (13, 8, 128), (13, 128), (13, 8))
_RTOL = 0.02
_ATOL = 2e-5


def _metadata(pattern: str, rank: int, size: int, invocation: int) -> DsaBatchMeta:
    total = sum(_LENGTHS)
    if pattern == "contiguous":
        owners = tuple(token * size // total for token in range(total))
    elif pattern == "strided":
        owners = tuple(token % size for token in range(total))
    else:
        owners = tuple(min(token, total - 1 - token) % size for token in range(total))
    owner_ids = tuple(tuple(token for token in reversed(range(total)) if owners[token] == owner)
                      for owner in range(size))
    offsets = tuple(owner_ids[owner].index(token) for token, owner in enumerate(owners))
    queries = tuple(reversed(owner_ids[rank]))
    queries = queries[1:] + queries[:1]
    return DsaBatchMeta((0, 3, 13), queries, owner_ids[rank], owners, offsets, cp_ranks=tuple(range(size)),
                        root_pes=tuple(range(size)), cp_rank=rank, layout_id=pattern, invocation=invocation)


def _objective(output: torch.Tensor, loss: torch.Tensor, cotangent: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "kl_only":
        return loss * 7
    return (output.float() * cotangent).sum() + loss * 7


def _assert_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    torch.testing.assert_close(actual.detach().float().cpu(), expected.detach().float().cpu(), rtol=_RTOL, atol=_ATOL)
    return _metrics(actual, expected)


def _cpu_oracle(cpu: tuple, indices: torch.Tensor, cotangent: torch.Tensor, mode: str) -> tuple:
    inputs = tuple(tensor.float().requires_grad_() for tensor in cpu)
    meta = DsaBatchMeta.packed(_LENGTHS)
    output, _ = sparse_attention_reference(*inputs[:4], indices, meta, attention_scale=192**-0.5)
    loss = selected_kl_reference(*inputs[4:], *inputs[:4], indices, meta, attention_scale=192**-0.5,
                                 normalization=DsaLossNormalization(13), loss_coeff=0 if mode == "coeff_zero" else 0.3)
    _objective(output, loss, cotangent, mode).backward()
    return output, loss, tuple(tensor.grad for tensor in inputs)


def _full_native(cpu: tuple, cotangent: torch.Tensor, mode: str, device: torch.device) -> tuple:
    inputs = tuple(tensor.to(device).requires_grad_() for tensor in cpu)
    backend = CannDsaReference(CannDsaLayout(DsaBatchMeta.packed(_LENGTHS), device), attention_scale=192**-0.5)
    selection = backend.indexer(*inputs[4:])
    output, stats = backend.attention(*inputs[:4], selection)
    loss = backend.kl_loss(*inputs[4:], inputs[:4], selection, stats, normalization=DsaLossNormalization(13),
                           loss_coeff=0 if mode == "coeff_zero" else 0.3)
    _objective(output, loss, cotangent.to(device), mode).backward()
    return output, loss, tuple(tensor.grad for tensor in inputs), selection.to_global_indices().cpu()


def _local_inputs(cpu: tuple, meta: DsaBatchMeta, device: torch.device) -> tuple:
    return tuple(tensor[list(meta.kv_global_ids if field in (1, 3, 5) else meta.q_global_ids)]
                 .to(device).requires_grad_() for field, tensor in enumerate(cpu))


def _retained_backward(backend: CannDsaCpReference, local: tuple, normalization: DsaLossNormalization,
                       loss_coeff: float, cotangent: torch.Tensor) -> dict:
    """Validate native repeat tolerance and exact accumulation of actual incoming gradients."""
    # Capture actual incoming gradients: native SFA reductions need not be bitwise repeatable.
    second = backend.forward(local[:4], index_inputs=local[4:], normalization=normalization, loss_coeff=loss_coeff)
    for tensor in local:
        tensor.grad = None
    incoming = [[] for _ in local]
    handles = [tensor.register_hook(lambda gradient, values=values: values.append(gradient.detach().clone()))
               for tensor, values in zip(local, incoming)]
    _objective(second.output, second.kl_loss, cotangent, "joint").backward(retain_graph=True)
    first = tuple(tensor.grad.detach().clone() for tensor in local)
    _objective(second.output, second.kl_loss, cotangent, "joint").backward()
    retained = {}
    for name, tensor, before, values, handle in zip(_NAMES, local, first, incoming, handles):
        handle.remove()
        if len(values) != 2:
            raise AssertionError(f"{name}: expected two incoming gradients")
        torch.testing.assert_close(before, values[0], rtol=0, atol=0)
        retained[name] = _assert_metrics(values[1], values[0])
        torch.testing.assert_close(tensor.grad, values[0] + values[1], rtol=0, atol=0)
        tensor.grad = before
    return retained


def _run_case(pattern: str, mode: str, invocation: int, device: torch.device) -> dict:
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(pattern, rank, size, invocation)
    generator = torch.Generator().manual_seed(20261008)
    cpu = tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16() for shape in _SHAPES)
    cotangent = torch.randn(_SHAPES[0], generator=generator) * 0.01 / 13
    full_output, full_loss, native_gradients, indices = _full_native(cpu, cotangent, mode, device)
    cpu_output, cpu_loss, cpu_gradients = _cpu_oracle(cpu, indices, cotangent, mode)
    backend = CannDsaCpReference(DsaCpLayout(meta, device), attention_scale=192**-0.5)
    local = _local_inputs(cpu, meta, device)
    normalization = DsaLossNormalization(13)
    coeff = 0 if mode == "coeff_zero" else 0.3
    selection = backend.prepare_selection(indices) if mode == "external" else None
    result = backend.forward(local[:4], index_inputs=local[4:], selection=selection,
                             normalization=normalization, loss_coeff=coeff)
    torch.testing.assert_close(result.global_indices.cpu().sort(dim=-1).values,
                               indices[list(meta.q_global_ids)].sort(dim=-1).values, rtol=0, atol=0)
    output, loss = result.output, result.kl_loss
    if mode == "checkpoint":
        def _recompute(*fields):
            item = backend.forward(fields[:4], index_inputs=fields[4:], normalization=normalization, loss_coeff=coeff)
            return item.output, item.kl_loss
        output, loss = checkpoint(_recompute, *local, use_reentrant=False)
    local_cotangent = cotangent[list(meta.q_global_ids)].to(device)
    _objective(output, loss, local_cotangent, mode).backward()
    retained = None
    if mode == "retain_graph":
        retained = _retained_backward(backend, local, normalization, coeff, local_cotangent)
    torch.npu.synchronize()
    gradients, cpu_comparison = {}, {}
    for field, (name, actual, expected, oracle) in enumerate(zip(_NAMES, local, native_gradients, cpu_gradients)):
        rows = list(meta.kv_global_ids if field in (1, 3, 5) else meta.q_global_ids)
        if expected is None:
            if actual.grad is not None:
                raise AssertionError(f"{name}: expected absent teacher gradient")
            gradients[name] = {"gradient_absent": True}
        else:
            gradients[name] = _assert_metrics(actual.grad, expected[rows])
            cpu_comparison[name] = _metrics(actual.grad, oracle[rows])
    summed_loss = loss.detach().clone()
    dist.all_reduce(summed_loss)
    return {"pattern": pattern, "mode": mode, "status": "passed", "owner_counts": backend.layout.counts,
            "q_global_ids": meta.q_global_ids, "kv_global_ids": meta.kv_global_ids,
            "output_vs_cp1": _assert_metrics(output, full_output[list(meta.q_global_ids)]),
            "kl_sum_vs_cp1": _assert_metrics(summed_loss, full_loss), "gradients_vs_cp1": gradients,
            "retained_backward_repeat": retained,
            "cpu_fp32_measurements": {"output": _metrics(output, cpu_output[list(meta.q_global_ids)]),
                                       "kl": _metrics(summed_loss, cpu_loss), "gradients": cpu_comparison}}


def run_validation(report: dict, output_dir: Path) -> None:
    """Execute CP=1/2/4 short packed fixtures with all seven gradients and recomputation."""
    # Device validation explicitly opts into the optional TorchNPU extension.
    __import__("torch_npu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), device=torch.npu.get_device_name(local_rank),
                  environment=probe_environment(), cases=[])
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("contiguous", "strided", "zigzag"):
        for mode in ("joint", "kl_only", "coeff_zero", "external", "checkpoint", "retain_graph"):
            report["stage"] = f"{pattern}/{mode}"
            report["cases"].append(_run_case(pattern, mode, len(report["cases"]), device))
            path = output_dir / f"rank{dist.get_rank()}.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report.update(status="passed", stage="complete")
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Preserve each rank's diagnostics on execution or parity failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {"status": "running", "stage": "init", "sequence_lengths": _LENGTHS,
              "cp1_parity_threshold": {"rtol": _RTOL, "atol": _ATOL}, "performance_measurement": False,
              "fused_execution": False, "cpu_fp32_acceptance": "measurement only; no model acceptance change"}
    try:
        run_validation(report, args.output_dir)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"rank": rank, "status": report["status"], "stage": report["stage"]}))


if __name__ == "__main__":
    main()
