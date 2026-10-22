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
"""Local expert FFNs with native weights and a Multicore fused backward."""

import ctypes
from functools import lru_cache, partial
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_parallel.core.multicore.modules.mega_moe.backward.tiling_tables import (
    get_swiglu_grad_tiling_bytes, get_w1_grad_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes, get_up_proj_tiling_bytes,
)
from hyper_parallel.core.multicore.modules.mega_moe.plan import _resize_swiglu_tiling, _tensor_from_bytes
from hyper_parallel.core.multicore.profiler.profiling import (
    _prepare_mega_kernel_runtime_config, _set_mega_kernel_profile_metadata,
)
from hyper_parallel.core.multicore.profiler.profiler import _enable_runtime_config_tensor, prepare_mega_kernel_call
from hyper_parallel.core.multicore.scheduler.config import (
    EVENT_INVALID_ID, RuntimeConfigC, SwiGluTilingDataC, TaskAiCoreType, TaskDescC,
    TaskSplitValue, TaskType, mega_moe_event_capacity,
)
from hyper_parallel.core.multicore.scheduler.graph import OpType
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config
from hyper_parallel.core.multicore.tasks.utils import add_dynamic_data
from hyper_parallel.core.multicore.torch import ops


def _add_empty_grad_tasks(config: RuntimeConfigC, values: TaskSplitValue, hidden: int,
                          intermediate: int, experts: int, cores: int) -> None:
    """Initialize only empty expert matrices while Cube starts its first GMM."""
    for worker in range(cores):
        task = TaskDescC()
        task.task_type = TaskType.TASK_ZERO_EMPTY_EXPERTS
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs, task.num_outputs = 1, 2
        task.inputs[0].dim[0] = experts
        task.inputs[0].input_position = 19
        task.inputs[0].data_type = 8
        for output, position, rows, columns in ((0, 6, hidden, intermediate), (1, 18, 2 * intermediate, hidden)):
            descriptor = task.outputs[output]
            descriptor.input_position, descriptor.data_type = position, 2
            descriptor.dim[0], descriptor.dim[1], descriptor.dim[2] = experts, rows, columns
        task.dependent_event, task.trigger_event = EVENT_INVALID_ID, 5
        task.task_index, task.task_split_num = worker, cores
        task.profile_desc_id, task.profile_owner_id = 0x30005, 0
        config.all_tasks[worker] = task
        config.vector_task_indices[worker] = worker
    config.task_index_num[1] = cores
    values.pre_task_num, values.pre_vector_task_num = cores, cores


def _bind_native_layout(task: TaskDescC) -> None:
    """Bind dY.T @ X and native untransposed weights to reused GMM descriptors."""
    output = task.outputs[0].input_position
    if output in (8, 12):
        task.inputs[1].transpose_flag = 0
    if output in (6, 18):
        first = type(task.inputs[0]).from_buffer_copy(bytes(task.inputs[0]))
        task.inputs[0] = task.inputs[1]
        task.inputs[1] = first
        task.inputs[0].transpose_flag = 1
        task.inputs[1].transpose_flag = 0


def _local_backward_config(tokens: int, hidden: int, intermediate: int, experts: int, cores: int) -> RuntimeConfigC:
    """Reuse existing tasks while binding native gradient layouts and local roots."""
    values = TaskSplitValue(tp=1, ep=1, seq_size=max(512, tokens), all_expert_num=experts, top_k=1)
    graph = build_backward_graph(values, hidden_size=hidden, intermediate_size=intermediate, num_cube_cores=cores)
    operators = [graph.get_op(name) for name in ("act_grad", "w2_grad", "swiglu_grad", "gate_grad", "w1_grad")]
    for operator in operators:
        operator.task_num = (tokens + 511) // 512 if operator.op_type == OpType.SWIGLU_GRAD else cores
        if operator.op_type == OpType.SWIGLU_GRAD:
            operator.split_value = 512
    task_count = cores + sum(operator.task_num for operator in operators)
    config = allocate_runtime_config(task_count, mega_moe_event_capacity(experts, 1), experts)
    config.num_workers = 2 * cores
    _add_empty_grad_tasks(config, values, hidden, intermediate, experts, cores)
    for operator in operators:
        operator.fill_config.fill(config, operator, values)
    config.task_num = task_count
    config.atomic_add_values[0] = 1
    add_dynamic_data(config, values, 19)
    for queue, kind in ((config.cube_task_indices, 0), (config.vector_task_indices, 1)):
        for index in queue[:config.task_index_num[kind]]:
            task = config.all_tasks[index]
            if task.task_type == TaskType.TASK_ZERO_EMPTY_EXPERTS:
                continue
            output = task.outputs[0].input_position
            task.dependent_event = EVENT_INVALID_ID if output in (6, 8) else (1 if output == 10 else 2)
            task.trigger_event = {8: 1, 6: 4, 10: 2, 12: 3, 18: 4}[output]
            task.profile_desc_id = 0x30000 + output
            task.profile_owner_id = 0
            if task.task_type == TaskType.TASK_GROUPED_MATMUL:
                task.extra_value_0 = 1
            else:
                task.extra_value_1 = 1
            _bind_native_layout(task)
    for event, count in ((1, cores), (2, config.task_index_num[1] - cores), (3, cores), (4, 2 * cores), (5, cores)):
        config.all_event_num_triggers[event] = count
    _set_mega_kernel_profile_metadata(
        config, kernel_name="LocalFfnBackward", owner_label="Local",
        stage_names={0x30000 + position: name for position, name in
                     ((5, "ZeroEmptyExperts"), (8, "dActivation"), (6, "dW2"),
                      (10, "SwiGLUGrad"), (12, "dInput"), (18, "dW1"))})
    return config


@lru_cache(maxsize=16)
def _backward_plan(tokens: int, hidden: int, intermediate: int, experts: int,
                   device: torch.device, stream_id: int):
    """Cache scratch and tiling per stream; allocate completion counters per call."""
    del stream_id  # Each stream needs its own device-mutated tiling and runtime scratch.
    cores = torch.npu.get_device_limit(device.index)["cube_core_num"]
    config = _local_backward_config(tokens, hidden, intermediate, experts, cores)
    common = {"num_groups": experts, "num_cube_cores": cores}
    tilings = (
        get_up_proj_tiling_bytes(4096, hidden_size=hidden, intermediate_size=intermediate // 2, **common),
        get_down_proj_tiling_bytes(4096, hidden_size=hidden, intermediate_size=2 * intermediate, **common),
        get_w1_grad_tiling_bytes(4096, hidden_size=2 * intermediate, intermediate_size=hidden // 2, **common),
        get_w1_grad_tiling_bytes(4096, hidden_size=hidden, intermediate_size=intermediate // 2, **common),
        _resize_swiglu_tiling(_swiglu_tiling(intermediate), cores),
    )
    runtime = _prepare_mega_kernel_runtime_config(
        config, tensor_factory=partial(_tensor_from_bytes, device=device),
        profile_tensor_factory=_enable_runtime_config_tensor, rank=0, device=device, device_id=device.index)
    return runtime, tuple(_tensor_from_bytes(data, device) for data in tilings), config.event_capacity


def _swiglu_tiling(intermediate: int) -> bytes:
    """Keep the existing UB tile while extending the row loop to 512 rows."""
    data = get_swiglu_grad_tiling_bytes(128, intermediate_size=intermediate)
    tiling = SwiGluTilingDataC.from_buffer_copy(data[:ctypes.sizeof(SwiGluTilingDataC)])
    tiling.rowLen = 512
    return bytes(tiling) * 49


class _RoutedFfn(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Keep original parameters owned by the enclosing FSDP/optimizer."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                counts: torch.Tensor) -> torch.Tensor:
        """Save local compute inputs.

        Args:
            ctx: Autograd context.
            inputs: Expert-major token rows.
            weight1: Native gate/up compute weights.
            weight2: Native down compute weights.
            counts: Device expert counts.
        """
        first = torch.ops.npu.npu_grouped_matmul(
            [inputs], [weight1.transpose(1, 2)], group_list=counts, group_type=0, group_list_type=1, split_item=2)[0]
        activation = torch.ops.npu.npu_swiglu(first, dim=-1)
        output = torch.ops.npu.npu_grouped_matmul(
            [activation], [weight2.transpose(1, 2)], group_list=counts,
            group_type=0, group_list_type=1, split_item=2)[0]
        ctx.save_for_backward(inputs, weight1, weight2, first, activation, counts)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Overlap empty-expert initialization and SwiGLU gradients with Cube work.

        Args:
            ctx: Saved local FFN context.
            grad_output: Expert-major output derivatives.
        """
        inputs, weight1, weight2, first, activation, groups = ctx.saved_tensors
        hidden, intermediate = inputs.shape[1], activation.shape[1]
        bound = max(512, 1 << (inputs.shape[0] - 1).bit_length())
        stream = torch.npu.current_stream(inputs.device)
        runtime, tilings, events = _backward_plan(
            bound, hidden, intermediate, weight1.shape[0], inputs.device, stream.npu_stream)
        grad_output = grad_output.contiguous()
        dw1, dw2 = torch.empty_like(weight1), torch.empty_like(weight2)
        dactivation, dfirst, dinputs = torch.empty_like(activation), torch.empty_like(first), torch.empty_like(inputs)
        counters = torch.zeros(events, dtype=torch.int32, device=inputs.device)
        # These BF16 compute paths use the CANN reserve, with no user scratch.
        workspace = torch.empty(512, dtype=torch.uint8, device=inputs.device)
        call = prepare_mega_kernel_call(
            runtime, direction="backward", fallback_event_counters=counters.view(torch.uint8))
        try:
            ops.mega_moe_grad_with_profile_buffer(
                grad_output, groups, grad_output, groups, counters,
                activation, dw2, weight2, dactivation, first, dfirst, weight1, dinputs, dinputs,
                groups, groups, counters, inputs, dw1, groups,
                *tilings, workspace, workspace, call.runtime_config, call.event_counters, call.profile_buffer,
                0, 1, weight1.shape[0], hidden, inputs.shape[0])
            call.complete()
        except Exception:
            call.cancel()
            raise
        return dinputs, dw1, dw2, None


def routed_ffn(inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
               counts: torch.Tensor) -> torch.Tensor:
    """Compute expert-major SwiGLU FFNs with one fused backward launch.

    Args:
        inputs: Contiguous BF16 expert-major rows, including repeated Top-K rows.
        weight1: Contiguous BF16 native gate/up weights shaped [E, 2I, H].
            H and I must be positive multiples of 32.
        weight2: Contiguous BF16 native down weights shaped [E, H, I].
        counts: Contiguous device int64 counts for all E experts, including
            empty experts. Counts must be nonnegative and sum to inputs.shape[0].
            Their values stay on the device; the caller owns this routing contract.

    Returns:
        Expert-major BF16 outputs; inputs and both original weights retain gradients.
    """
    _validate_shapes(inputs, weight1, weight2)
    _validate_storage(inputs, weight1, weight2, counts)
    if inputs.shape[0] == 0:
        return inputs + weight1.reshape(-1)[:0].sum() + weight2.reshape(-1)[:0].sum()
    return _RoutedFfn.apply(inputs, weight1, weight2, counts)


def _validate_shapes(inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> None:
    """Validate native geometry without reading device values."""
    if inputs.device.type != "npu":
        raise ValueError("routed_ffn requires an Ascend NPU with torch_npu")
    if inputs.ndim != 2 or weight1.ndim != 3 or weight2.ndim != 3:
        raise ValueError("Expected 2D expert-major inputs and 3D native weights")
    experts, twice_intermediate, hidden = weight1.shape
    if (twice_intermediate % 64 or hidden % 32 or min(experts, twice_intermediate, hidden) <= 0
            or inputs.shape[1] != hidden
            or tuple(weight2.shape) != (experts, hidden, twice_intermediate // 2)):
        raise ValueError("Incompatible native FFN weight dimensions")


def _validate_storage(inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                      counts: torch.Tensor) -> None:
    """Validate dtype, device and storage contracts."""
    experts = weight1.shape[0]
    if counts.dtype != torch.int64 or tuple(counts.shape) != (experts,) or not counts.is_contiguous():
        raise ValueError("counts must be a contiguous int64 device vector with E entries")
    if any(tensor.dtype != torch.bfloat16 or tensor.device != inputs.device or not tensor.is_contiguous()
           for tensor in (inputs, weight1, weight2)) or counts.device != inputs.device:
        raise ValueError("FFN tensors must be contiguous BF16 tensors on the same NPU")
