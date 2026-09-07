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
"""Connected shared-B training with overlapping graphs and optimizer updates."""

from functools import partial
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.platform import torch as multicore_adapter
from tests.torch.expert_parallel import _test_mega_moe_common as baseline
from tests.torch.expert_parallel._mega_moe_utils import start_shmem_lifetime, write_evidence


def _new_stack(shape: baseline.MoeShape, capacity: float) -> torch.nn.ModuleList:
    """Keep home parameters independent while enabling one replica slot per layer."""
    return torch.nn.ModuleList([
        MegaMoeExperts(
            local_num_tokens=shape.local_num_tokens, hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size, num_experts=shape.num_experts,
            top_k=shape.top_k, ep_size=shape.ep_size, ep_group=dist.group.WORLD,
            expert_capacity_factor=capacity, replica_slots_per_rank=1, shared_physical_weights=True,
        ).to(device=baseline.DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    ])


def _routes(shape: baseline.MoeShape) -> list[tuple[torch.Tensor, ...]]:
    """Alternate home-only, hot-home and shifted-hot placements within each graph."""
    balanced = baseline.make_balanced_route(shape)
    positions = torch.arange(shape.local_num_tokens * shape.top_k, device=baseline.DEVICE)
    routes = [balanced]
    for offset in (0, shape.top_k):
        ids = (positions.remainder(shape.top_k) + offset).reshape(shape.local_num_tokens, shape.top_k)
        counts = torch.bincount(ids.flatten(), minlength=shape.num_experts).to(torch.int32)
        routes.append((ids, balanced[1], counts))
    return routes


def _optimizer(layers: torch.nn.ModuleList) -> torch.optim.AdamW:
    """Use real optimizer state and refresh replica versions after every step."""
    return torch.optim.AdamW(
        layers.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, foreach=False, fused=False,
    )


def _train_step(
    layers: torch.nn.ModuleList,
    optimizer: torch.optim.AdamW,
    shape: baseline.MoeShape,
    routes: list[tuple[torch.Tensor, ...]],
    step: int,
    *,
    recompute: bool,
) -> dict[str, torch.Tensor]:
    """Retain two connected graphs and compare all gradients only after both backwards."""
    optimizer.zero_grad(set_to_none=True)
    hidden, upstream = baseline.make_data(shape)
    upstream = upstream / shape.local_num_tokens**0.5
    inputs = [(hidden * (1 + microbatch / 8)).detach().requires_grad_(True) for microbatch in range(2)]
    weights, outputs = [], []
    for microbatch, hidden_states in enumerate(inputs):
        output = hidden_states
        for index, layer in enumerate(layers):
            ids, probabilities, counts = routes[(step + microbatch + index) % len(routes)]
            route_weight = probabilities.clone().requires_grad_(True)
            weights.append(route_weight)
            forward = partial(layer, tokens_per_expert=counts)
            output = (
                checkpoint(forward, output, ids, route_weight, use_reentrant=True)
                if recompute else forward(output, ids, route_weight)
            )
        outputs.append(output)
    for output in reversed(outputs):
        output.backward(upstream)
    torch.npu.synchronize()
    tensors = {f"output/{index}": tensor for index, tensor in enumerate(outputs)}
    tensors.update({f"dx/{index}": tensor.grad for index, tensor in enumerate(inputs)})
    tensors.update({f"droute/{index}": tensor.grad for index, tensor in enumerate(weights)})
    tensors.update({f"dw/{index}": parameter.grad for index, parameter in enumerate(layers.parameters())})
    snapshots = {}
    for name, tensor in tensors.items():
        assert tensor is not None, f"rank={baseline.RANK}: {name} expected a gradient, got {tensor}."
        baseline.assert_finite(name, tensor)
        snapshots[name] = tensor.detach().cpu()
    optimizer.step()
    for index, parameter in enumerate(layers.parameters()):
        baseline.assert_finite(f"weight/{index}", parameter)
        snapshots[f"weight/{index}"] = parameter.detach().cpu()
        for name, tensor in optimizer.state[parameter].items():
            baseline.assert_finite(f"optimizer/{index}/{name}", tensor)
            snapshots[f"optimizer/{index}/{name}"] = tensor.detach().cpu().clone()
    return snapshots


def _compare_case(capacity: float, recompute: bool) -> dict:
    """Compare private and shared resource groups with identical connected training."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=512, num_experts=8, top_k=4)
    torch.manual_seed(61_000 + baseline.RANK)
    independent = _new_stack(shape, capacity)
    shared = _new_stack(shape, capacity)
    shared.load_state_dict(independent.state_dict())
    MegaMoeExperts.share_execution_resources(list(shared))
    optimizers = [_optimizer(layers) for layers in (independent, shared)]
    routes = _routes(shape)
    wave_counts, home_expert_counts = set(), set()
    launch = multicore_adapter.mega_moe

    def record_launch(*args: Any, **kwargs: Any) -> Any:
        """Observe scalar coverage without replacing kernels or retaining launch tensors."""
        wave_counts.add(kwargs["wave_count"])
        home_expert_counts.add(kwargs["home_experts"])
        return launch(*args, **kwargs)

    try:
        with patch.object(multicore_adapter, "mega_moe", new=record_launch):
            for step in range(3):
                expected = _train_step(independent, optimizers[0], shape, routes, step, recompute=recompute)
                observed = _train_step(shared, optimizers[1], shape, routes, step, recompute=recompute)
                for name, actual in observed.items():
                    try:
                        torch.testing.assert_close(actual, expected[name], rtol=2e-2, atol=1e-5)
                    except AssertionError as error:
                        raise AssertionError(
                            f"rank={baseline.RANK} capacity={capacity} checkpoint={recompute} "
                            f"step={step} {name}: {error}"
                        ) from error
        expected_waves = {1, 2} if capacity == 1.0 else {1}
        assert wave_counts == expected_waves, f"expected waves={expected_waves}, got {wave_counts}."
        assert home_expert_counts == {0, 4}, f"expected home and shared-B launches={{0, 4}}, got {home_expert_counts}."
        return {
            "capacity_factor": capacity, "checkpoint": "reentrant" if recompute else "none",
            "layers": 3, "outstanding_forwards": 6, "optimizer": "AdamW", "optimizer_steps": 3,
            "wave_counts": sorted(wave_counts), "home_expert_counts": sorted(home_expert_counts),
            "compared_tensors_per_step": len(observed), "outputs_gradients_weights_optimizer_match": True,
        }
    finally:
        for layer in [*independent, *shared]:
            layer.close()


def test_mega_moe_connected_shared_gradients() -> None:
    """Validate owned backward outputs with checkpoint, routing changes and updates."""
    cases = [_compare_case(capacity, recompute) for capacity in (1.5, 1.0) for recompute in (False, True)]
    write_evidence({"cases": cases, "rtol": 2e-2, "atol": 1e-5})
