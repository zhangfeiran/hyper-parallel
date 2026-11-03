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
"""Validate installed MegaMoE B=1 against an independent Torch expert reference.

Run outside the checkout after activating CANN and the wheel's custom OPP::

    torchrun --standalone --nproc-per-node=4 -m \
        hyper_parallel.core.multicore.examples.moe.b1_smoke --dispatch-mode push
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch.nn import functional

import hyper_parallel
from hyper_parallel.core.expert_parallel.hot_replica.host import _native_backend
from hyper_parallel.core.multicore import MegaMoeExperts


class _ReferenceMatmul(torch.autograd.Function):
    """Use ordinary BF16 matmul while preserving FP32 reference weight gradients."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Keep the same input dtype and accumulation path as BF16 expert GEMMs.

        Args:
            ctx: Autograd context retaining the reference inputs.
            hidden: BF16 matrix of token rows.
            weight: FP32 reference expert matrix.
        """
        ctx.save_for_backward(hidden, weight)
        return hidden @ weight.to(hidden.dtype)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate local expert weight gradients before the cross-rank FP32 sum.

        Args:
            ctx: Context containing the forward matrices.
            gradient: BF16 output gradient.
        """
        hidden, weight = ctx.saved_tensors
        grad_input = (gradient.float() @ weight.transpose(0, 1)).to(hidden.dtype)
        grad_weight = hidden.float().transpose(0, 1) @ gradient.float()
        return grad_input, grad_weight


def _execution_info(output: torch.Tensor) -> tuple[dict[str, Any], Any]:
    """Inspect the executed autograd invocation so fallback cannot pass this smoke."""
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if hasattr(node, "replica_route"):
            route = node.replica_route
            provider = node.workspace.replica_provider
            return {
                "executed_budget": node.plan.spec.replica_slots_per_rank,
                "transfers": 0 if route is None else len(route.plan.transfers),
                "provider": None if provider is None else type(provider).__name__,
            }, provider
        pending.extend(edge[0] for edge in node.next_functions)
    raise RuntimeError("No MegaMoE invocation found in the output graph")


def _reference(
    hidden: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
    gate_up: torch.Tensor, down: torch.Tensor,
) -> torch.Tensor:
    """Match BF16 forward boundaries while accumulating expert gradients in FP32."""
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(gate_up.shape[0]):
        rows, columns = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        projected = _ReferenceMatmul.apply(hidden[rows], gate_up[expert])
        gate, up = projected.float().chunk(2, dim=-1)
        activation = (functional.silu(gate) * up).to(hidden.dtype)
        values = _ReferenceMatmul.apply(activation, down[expert]).float()
        result = result.index_add(0, rows, values * probabilities[rows, columns, None])
    return result.to(hidden.dtype)


def _copy_owned_weights(layer: MegaMoeExperts, gate_ref: torch.Tensor, down_ref: torch.Tensor) -> None:
    """Compare each invocation at identical weights despite optimizer rounding drift."""
    for owned, reference in ((layer.gate_up_weight, gate_ref), (layer.down_weight, down_ref)):
        shards = [torch.empty_like(owned) for _ in range(dist.get_world_size())]
        dist.all_gather(shards, owned.detach())
        with torch.no_grad():
            reference.copy_(torch.cat(shards).float())


def _check(actual: torch.Tensor, expected: torch.Tensor, name: str, *, elementwise: bool = True) -> dict:
    """Require finite values, relative L2 and the public precision tolerances."""
    actual, expected = actual.detach().float(), expected.detach().float()
    difference = actual - expected
    relative = float(difference.norm() / expected.norm().clamp_min(1e-8))
    failure = None
    try:
        if not bool(torch.isfinite(actual).all()) or relative > .01:
            raise AssertionError(f"relative L2 {relative}")
        if elementwise:
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
    except AssertionError as error:
        failure = f"rank={dist.get_rank()} {name}: {error}"
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    if any(failures):
        raise AssertionError(str(failures))
    return {"relative_l2": relative, "max_abs": float(difference.abs().max())}


def _write_result(
    args: argparse.Namespace, rank: int, world: int, layer: MegaMoeExperts, host: Any, records: list[dict],
) -> None:
    """Persist numerical checks and installed library identity for each rank."""
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "pid": os.getpid(), "world": world, "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "dispatch_mode": args.dispatch_mode,
              "shape": {"tokens": 2048, "hidden": 128, "intermediate": 128, "top_k": 2},
              "transport": layer.replica_transport, "host_abi": host.abi_version(), "records": records,
              "venv": sys.prefix, "package": hyper_parallel.__file__, "host_library": host.__file__,
              "torch_file": torch.__file__, "torch_npu_file": torch_npu.__file__,
              "maps": sorted({line.split()[-1]
                              for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                              if "/" in line and ("Ascend" in line or "hyper_parallel" in line)})}
    (args.result_dir / f"rank{rank:02}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rank == 0:
        print(json.dumps({"B1_SMOKE": "PASS", "mode": args.dispatch_mode,
                          "transport": layer.replica_transport, "steps": args.steps}), flush=True)


def main() -> None:
    """Run active default-transport B=1 forward, backward and AdamW from the wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--result-dir", type=Path, default=Path("b1-smoke-results"))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    layer = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world < 2:
            raise ValueError("B=1 acceptance requires at least two ranks")
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        tokens, hidden_size, intermediate, top_k = 2048, 128, 128, 2
        experts = world * 6
        generator = torch.Generator().manual_seed(715)
        gate_cpu = torch.randn(experts, hidden_size, 2 * intermediate, generator=generator)
        gate_cpu = (gate_cpu / math.sqrt(hidden_size)).to(torch.bfloat16)
        down_cpu = torch.randn(experts, intermediate, hidden_size, generator=generator)
        down_cpu = (down_cpu / math.sqrt(intermediate)).to(torch.bfloat16)
        gate_ref = gate_cpu.to(device=device, dtype=torch.float32).requires_grad_()
        down_ref = down_cpu.to(device=device, dtype=torch.float32).requires_grad_()
        layer = MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate, num_experts=experts,
            top_k=top_k, ep_size=world, ep_group=dist.group.WORLD,
            dispatch_mode=args.dispatch_mode, replica_slots_per_rank=1,
        ).to(device=device, dtype=torch.bfloat16)
        owned = slice(rank * 6, (rank + 1) * 6)
        with torch.no_grad():
            layer.gate_up_weight.copy_(gate_cpu[owned])
            layer.down_weight.copy_(down_cpu[owned])
        host = _native_backend()
        if host is None or host.abi_version() != 3:
            raise RuntimeError("Wheel is missing replica host ABI3")
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.001, foreach=False)
        reference_optimizer = torch.optim.AdamW((gate_ref, down_ref), lr=.001, foreach=False)
        torch.manual_seed(712 + rank)
        values = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
        ids = torch.arange(tokens * top_k, device=device).reshape(tokens, top_k).remainder(top_k).to(torch.int32)
        counts = torch.bincount(ids.flatten().long(), minlength=experts).to(torch.int32)
        probabilities = torch.full((tokens, top_k), 1 / top_k, device=device)
        upstream = torch.randn_like(values).div_(tokens)
        records = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            _copy_owned_weights(layer, gate_ref, down_ref)
            actual_input = values.detach().clone().requires_grad_()
            expected_input = values.detach().clone().requires_grad_()
            actual_probs = probabilities.detach().clone().requires_grad_()
            expected_probs = probabilities.detach().clone().requires_grad_()
            expected = _reference(expected_input, ids, expected_probs, gate_ref, down_ref)
            actual = layer(actual_input, ids, actual_probs, tokens_per_expert=counts)
            execution, provider = _execution_info(actual)
            if execution["executed_budget"] != 1 or not execution["transfers"]:
                raise RuntimeError(f"B=1 must execute real copies: {execution}")
            expected.backward(upstream.float())
            actual.backward(upstream)
            dist.all_reduce(gate_ref.grad)
            dist.all_reduce(down_ref.grad)
            checks = {"output": _check(actual, expected, "output")}
            checks.update(
                input_grad=_check(actual_input.grad, expected_input.grad, "dx"),
                probability_grad=_check(actual_probs.grad, expected_probs.grad, "dprob"),
                gate_up_grad=_check(layer.gate_up_weight.grad, gate_ref.grad[owned], "dW13", elementwise=False),
                down_grad=_check(layer.down_weight.grad, down_ref.grad[owned], "dW2", elementwise=False),
            )
            matrices = list(getattr(provider, "last_kernel_gradient_matrices", ()))
            if sorted(matrices) != [0, 1]:
                raise RuntimeError(f"Default bands transport did not return both projections: {matrices}")
            optimizer.step()
            reference_optimizer.step()
            checks.update(
                gate_up_updated=_check(layer.gate_up_weight, gate_ref[owned], "updated W13", elementwise=False),
                down_updated=_check(layer.down_weight, down_ref[owned], "updated W2", elementwise=False),
            )
            records.append({"step": step, "execution": execution, "kernel_gradient_matrices": matrices,
                            "checks": checks})
        torch.npu.synchronize()
        _write_result(args, rank, world, layer, host, records)
    finally:
        if layer is not None:
            layer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
