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
"""Distributed shared-weight lifecycle regressions using the native MoE oracle."""

import json
import os
from dataclasses import fields
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from tests.torch.expert_parallel import _test_mega_moe as moe
from tests.torch.expert_parallel import _test_mega_moe_common as common

# Reuse the canonical worker's native oracle, route builders and BF16 tolerances.
# pylint: disable=protected-access


def _shape() -> moe._MoeShape:
    """Allow the same regression to run at small and production shapes."""
    return moe._MoeShape(
        local_num_tokens=int(os.getenv("HP_MEGA_MOE_SHARED_TOKENS", "128")),
        hidden_size=int(os.getenv("HP_MEGA_MOE_SHARED_HIDDEN", "512")),
        intermediate_size=int(os.getenv("HP_MEGA_MOE_SHARED_INTERMEDIATE", "128")),
        num_experts=48 if moe.dist.get_world_size() == 8 else 8,
        top_k=8 if moe.dist.get_world_size() == 8 else 4,
        ep_size=moe.dist.get_world_size(),
    )


def _layers(shape: moe._MoeShape, *, shared: bool = True, compact: bool = False) -> tuple[list, list]:
    """Construct distinct layer parameters and a single shared staging owner."""
    candidates, references = [], []
    for index in range(2):
        native = common.new_common_moe(shape)
        with torch.no_grad():
            common._local_tensor(native.w1).mul_(1.0 - index * 0.1)
            common._local_tensor(native.w3).mul_(1.0 - index * 0.1)
        mega = moe.MegaMoeExperts(
            local_num_tokens=shape.local_num_tokens,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            ep_size=shape.ep_size,
            ep_group=moe.dist.group.WORLD,
            expert_capacity_factor=float(os.getenv("HP_MEGA_MOE_CAPACITY_FACTOR", "1.0")),
            replica_slots_per_rank=int(os.getenv("HP_MEGA_MOE_SHARED_SLOTS", "1")),
            shared_physical_weights=shared,
            compact_physical_weights=compact,
        ).to(device=moe._DEVICE, dtype=torch.bfloat16)
        common._copy_common_weights_to_mega(native, mega)
        candidates.append(mega)
        references.append(native)
    moe.MegaMoeExperts.share_execution_resources(candidates)
    return candidates, references


def _capture(layer, hidden, route_weights, output) -> moe._RunResult:
    """Capture one completed backward without retaining shared scratch views."""
    return moe._RunResult(*(
        tensor.detach().clone()
        for tensor in (output, hidden.grad, route_weights.grad,
                       layer.gate_up_weight.grad, layer.down_weight.grad)
    ))


def _cyclic_route(shape: moe._MoeShape, active_experts: int, offset: int = 0) -> tuple:
    """Construct exact route histograms without requiring K <= local experts."""
    positions = torch.arange(shape.local_num_tokens * shape.top_k, device=moe._DEVICE)
    ids = ((positions % active_experts + offset) % shape.num_experts).reshape(
        shape.local_num_tokens, shape.top_k
    ).to(torch.int32)
    weights = torch.full(ids.shape, 1.0 / shape.top_k, device=moe._DEVICE, dtype=torch.float32)
    counts = torch.bincount(ids.flatten().long(), minlength=shape.num_experts).to(torch.int32)
    return ids, weights, counts


def _check_deferred(
    candidates, references, shape, routes, *, check_storage: bool = True, checkpoint_reentrant: bool = False
) -> int:
    """Overwrite one pool across layers and placements while old graphs live."""
    hidden, grad_output = moe._make_data(shape)
    inputs, weights, outputs, expected = [], [], [], []
    layer_indices = (0, 0, 1, 0, 1)
    parameter_state = [
        (weight.data_ptr(), weight._version)
        for layer in candidates for weight in layer.parameters()
    ]
    for index, route in zip(layer_indices, routes):
        expected.append(moe._run_layer(references[index], hidden, *route, grad_output))
        inputs.append(hidden.detach().clone().requires_grad_(True))
        weights.append(route[1].detach().clone().requires_grad_(True))
        function = partial(candidates[index], tokens_per_expert=route[2])
        if checkpoint_reentrant:
            output = checkpoint(function, inputs[-1], route[0], weights[-1], use_reentrant=True)
        else:
            output = function(inputs[-1], route[0], weights[-1])
        outputs.append(output)
    manager = candidates[0]._get_execution_resources(hidden).replica_manager
    pool_ptrs = tuple(weight.data_ptr() for weight in manager._shared_weights) if check_storage else ()
    gradient_ptrs = None
    for position in reversed(range(len(outputs))):
        layer = candidates[layer_indices[position]]
        layer.zero_grad(set_to_none=True)
        outputs[position].backward(grad_output)
        torch.npu.synchronize(moe._DEVICE)
        actual = _capture(layer, inputs[position], weights[position], outputs[position])
        moe._assert_results_close(expected[position], actual, weight_grad_wave_count=shape.ep_size)
        # A home-only backward can precede the first replica backward. The
        # shared gradient pool is deliberately allocated only when first used.
        if check_storage and manager._shared_gradients is not None:
            current_ptrs = tuple(gradient.data_ptr() for gradient in manager._shared_gradients)
            if gradient_ptrs is None:
                gradient_ptrs = current_ptrs
            assert current_ptrs == gradient_ptrs, f"B gradient pool changed: {current_ptrs} != {gradient_ptrs}"
        for parameter in layer.parameters() if check_storage else ():
            gradient = parameter.grad
            assert gradient.untyped_storage().nbytes() == gradient.numel() * gradient.element_size(), (
                f"gradient retains staging: storage={gradient.untyped_storage().nbytes()}, "
                f"logical={gradient.numel() * gradient.element_size()}"
            )
    if check_storage:
        assert gradient_ptrs is not None, f"replica backwards never initialized the B gradient pool: {gradient_ptrs}"
    current_state = [
        (weight.data_ptr(), weight._version)
        for layer in candidates for weight in layer.parameters()
    ]
    assert parameter_state == current_state, f"parameter state changed: {parameter_state} != {current_state}"
    actual_ptrs = tuple(weight.data_ptr() for weight in manager._shared_weights) if check_storage else ()
    assert pool_ptrs == actual_ptrs, f"staging changed: {pool_ptrs} != {actual_ptrs}"
    return len(outputs)


def test_shared_weight_pool_lifecycle(*, checkpoint_modes: tuple[bool, ...] = (False, True)) -> None:
    """Validate mixed home/replica graphs, layer reuse, checkpoint and updates."""
    shape = _shape()
    candidates, references = _layers(shape)
    balanced = _cyclic_route(shape, shape.num_experts, moe._RANK * shape.local_experts)
    one_wave = _cyclic_route(shape, shape.top_k)
    skewed = _cyclic_route(shape, min(shape.num_experts // 2, 2 * shape.local_experts))
    shift = shape.num_experts // 2
    shifted = ((skewed[0] + shift) % shape.num_experts, skewed[1], skewed[2].roll(shift))
    cold_home = _cyclic_route(shape, shape.num_experts - shape.local_experts, shape.local_experts)
    routes = (balanced, one_wave, skewed, shifted, cold_home)
    slots = int(os.getenv("HP_MEGA_MOE_SHARED_SLOTS", "1"))
    summaries = [
        moe._summarize_performance_route(shape, route[2], slots, {"profile": name})
        for name, route in zip(("balanced", "hot_topk", "hot_two_home_groups", "shifted", "cold_home"), routes)
    ]
    if moe._RANK == 0:
        print("SHARED_WEIGHT_ROUTES=" + json.dumps(summaries, sort_keys=True), flush=True)
    if slots == shape.local_experts and shape.ep_size == 8:
        maximum_slots_used = max(sum(expert >= 0 for expert in row) for row in summaries[-1]["replica_slot_bindings"])
        assert maximum_slots_used == slots, f"cold-home route did not fill B: {maximum_slots_used} != {slots}"
    checks = _check_deferred(candidates, references, shape, routes)
    checks += _check_deferred(
        candidates, references, shape, routes, checkpoint_reentrant=True
    )
    rejected_repeated_backwards = 0
    hidden, grad_output = moe._make_data(shape)
    for route in (balanced, skewed):
        reference = moe._run_layer(references[0], hidden, *route, grad_output)
        for reentrant in checkpoint_modes:
            if moe._RANK == 0:
                print(f"SHARED_WEIGHT_CHECKPOINT skewed={route is skewed} reentrant={reentrant}", flush=True)
            actual = moe._run_checkpointed_layer(
                candidates[0], hidden, *route, grad_output, use_reentrant=reentrant
            )
            moe._assert_results_close(reference, actual, weight_grad_wave_count=shape.ep_size)
            checks += 1
        runtime = candidates[0]._get_execution_resources(hidden).plan
        reuses_saved_storage = (
            route is skewed and summaries[2]["expected_wave_count"] > 1 and runtime.reuse_retained_dw_scratch
        )
        if reuses_saved_storage:
            # The existing retained-scratch mode deliberately invalidates a
            # second backward after overwriting saved activations. Sharing B
            # must preserve that guard, not silently reuse corrupted storage.
            try:
                moe._run_retained_graph_layer(candidates[0], hidden, *route, grad_output)
            except RuntimeError as error:
                if "modified by an inplace operation" not in str(error):
                    raise
                rejected_repeated_backwards += 1
            else:
                raise AssertionError("retained scratch accepted a second backward after overwriting saved storage")
        else:
            for actual in moe._run_retained_graph_layer(candidates[0], hidden, *route, grad_output):
                moe._assert_results_close(reference, actual, weight_grad_wave_count=shape.ep_size)
                checks += 1
    with torch.no_grad():
        candidates[0].gate_up_weight.add_(0.03125)
        candidates[0].down_weight.sub_(0.015625)
        common._local_tensor(references[0].w1).add_(0.03125)
        common._local_tensor(references[0].w3).add_(0.03125)
        common._local_tensor(references[0].w2).sub_(0.015625)
    reference = moe._run_layer(references[0], hidden, *skewed, grad_output)
    actual = moe._run_layer(candidates[0], hidden, *skewed, grad_output)
    moe._assert_results_close(reference, actual, weight_grad_wave_count=shape.ep_size)
    manager = candidates[0]._get_execution_resources(hidden).replica_manager
    expected_pool_bytes = slots * shape.hidden_size * shape.intermediate_size * 3 * 2
    pool_bytes = sum(weight.untyped_storage().nbytes() for weight in manager._shared_weights)
    gradient_bytes = sum(gradient.untyped_storage().nbytes() for gradient in manager._shared_gradients)
    assert pool_bytes == expected_pool_bytes, f"weight pool is not B-only: {pool_bytes} != {expected_pool_bytes}"
    expected_gradient_bytes = expected_pool_bytes * 2
    assert gradient_bytes == expected_gradient_bytes, (
        f"gradient pool is not FP32 B-only: {gradient_bytes} != {expected_gradient_bytes}"
    )
    result = {
        "passed_groups": checks + 1,
        "rejected_repeated_backwards": rejected_repeated_backwards,
        "compared_tensors": [field.name for field in fields(moe._RunResult)],
        "ep_size": shape.ep_size,
        "shared_pool_bytes": pool_bytes,
        "shared_gradient_pool_bytes": gradient_bytes,
        "replica_slots_per_rank": slots,
        "route_summaries": summaries,
        "layer_count": len(candidates),
        "checkpoint_modes": list(checkpoint_modes),
    }
    if moe._RANK == 0:
        print("SHARED_WEIGHT_PRECISION=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        test_shared_weight_pool_lifecycle(
            checkpoint_modes=(True,) if os.getenv("HP_MEGA_MOE_SHARED_REENTRANT_ONLY") == "1" else (False, True)
        )
    finally:
        if moe.dist.is_initialized():
            moe.dist.destroy_process_group()
