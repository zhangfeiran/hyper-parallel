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
"""Keep native-row-order weight GEMMs inside the composed backward kernel."""

from __future__ import annotations

import copy
import ctypes
from dataclasses import replace

import torch

from hyper_parallel.core.multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_parallel.core.multicore.scheduler.config import (
    RuntimeConfigC, TaskAiCoreType, TaskDescC, TaskSplitValue, TaskType, TensorDescC, validate_runtime_config,
)
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config

from .spec import MegaMoeSpec


def supports_native_weight_gradients(spec: MegaMoeSpec) -> bool:
    """Limit exact fused dW to the deployed official GEMM geometry validated against native."""
    return (spec.route_policy == "torch" and spec.expert_weight_layout == "EIO"
            and (spec.hidden_size, spec.intermediate_size, spec.num_experts, spec.ep_size, spec.top_k,
                 spec.num_cube_cores) == (5120, 2304, 96, 16, 6, 24))


def ordered_workspace_views(workspace: torch.Tensor, rows: int, intermediate: int,
                            hidden: int, include_input: bool = False) -> tuple[torch.Tensor, ...]:
    """Expose logical dU, activation and dY views over invocation-owned byte storage."""
    offset, result = 0, []
    for width in (intermediate * 2, intermediate, hidden) + ((hidden,) if include_input else ()):
        size = rows * width * 2
        result.append(workspace[offset:offset + size].view(torch.bfloat16).reshape(rows, width))
        offset = (offset + size + 127) // 128 * 128
    return tuple(result)


def allocate_ordered_workspace(order: torch.Tensor, intermediate: int, hidden: int = 5120, *, include_input: bool = False) -> torch.Tensor:
    """Own all ordered dW operands and a fresh row map without a synchronous host header copy."""
    offset = 0
    for width in (intermediate * 2, intermediate, hidden) + ((hidden,) if include_input else ()):
        offset = (offset + order.numel() * width * 2 + 127) // 128 * 128
    workspace = torch.empty(max(4096, offset + order.numel() * 8), dtype=torch.uint8, device=order.device)
    workspace.view(torch.int64)[offset // 8:offset // 8 + order.numel()].copy_(order)
    return workspace


def _copy_node(name: str, inputs: list[TensorSpec], outputs: list[TensorSpec], swiglu: OperatorNode) -> OperatorNode:
    """Reuse the existing expert-row tiling for a data-only indexed copy."""
    return OperatorNode(
        name=name, op_type=OpType.INDEXED_COPY, inputs=inputs, outputs=outputs,
        param_positions=[tensor.param_position for tensor in inputs + outputs],
        split_value=swiglu.split_value,
        split_spec=SplitSpec(split_inputs=None, task_num_fn=swiglu.split_spec.task_num_fn,
                             split_output_dims=[0] * len(outputs)),
        tiling_position=-1, fill_config=None,
        diagnostic_name="NativeOrderW2" if name.endswith("w2") else "NativeOrderW1",
    )


def _ordered_graph(graph: ComputeGraph, values: TaskSplitValue, spec: MegaMoeSpec) -> ComputeGraph:
    """Represent independent dX and dW paths with their actual operand ports."""
    nodes = {node.name: node for node in copy.deepcopy(graph).topological_sort()}
    for node in nodes.values():
        node.predecessors, node.successors = [], []
    swiglu = nodes["swiglu_grad"]
    activation = replace(swiglu.inputs[0], name="native_activation", param_position=33, transpose=True)
    dy = replace(nodes["gate_grad"].outputs[0], name="native_dy", param_position=34, transpose=False)
    copy2 = _copy_node("native_order_w2", [replace(nodes["w2_grad"].inputs[0], transpose=False),
                                          nodes["dispatch"].outputs[0]],
                       [replace(activation, transpose=False), dy], swiglu)
    du = replace(swiglu.outputs[0], name="native_du", param_position=26, transpose=False)
    copy1_inputs, copy1_outputs = [swiglu.outputs[0]], [du]
    source = nodes["w1_grad"].inputs[0]
    if spec.dispatch_mode == "pull":
        copy1_inputs.append(replace(source, transpose=False))
        source = replace(source, name="native_input", param_position=35)
        copy1_outputs.append(replace(source, transpose=False))
    copy1 = _copy_node("native_order_w1", copy1_inputs, copy1_outputs, swiglu)
    nodes["w2_grad"].inputs = [activation, dy, nodes["w2_grad"].inputs[2]]
    nodes["w2_grad"].param_positions = [33, 34, 19, 6]
    nodes["w1_grad"].inputs = [source, du, nodes["w1_grad"].inputs[2]]
    nodes["w1_grad"].param_positions = [source.param_position, 26, 19, 18]
    result = ComputeGraph()
    for node in (nodes["dispatch"], copy2, nodes["w2_grad"], nodes["act_grad"], swiglu,
                 nodes["gate_grad"], copy1, nodes["combine"], nodes["w1_grad"]):
        result.add_op(node)
    for source_name, target_name in (
        ("dispatch", "native_order_w2"), ("native_order_w2", "w2_grad"), ("dispatch", "act_grad"),
        ("act_grad", "swiglu_grad"), ("swiglu_grad", "gate_grad"), ("swiglu_grad", "native_order_w1"),
        ("gate_grad", "combine"), ("native_order_w1", "w1_grad"),
    ):
        result.add_edge(source_name, target_name)
    result.propagate_splits(values)
    return result


def _expanded_runtime(original: RuntimeConfigC, task_count: int, local_experts: int) -> RuntimeConfigC:
    """Preserve runtime bounds and protocol metadata while growing only task storage."""
    result = allocate_runtime_config(task_count, original.event_capacity, local_experts)
    for name, _ in RuntimeConfigC._fields_:
        if name in ("task_capacity", "event_capacity", "task_num"):
            continue
        value = getattr(original, name)
        if isinstance(value, ctypes.Array):
            getattr(result, name)[:] = value[:]
        else:
            setattr(result, name, value)
    result.task_num = task_count
    result.all_event_num_triggers[:] = original.all_event_num_triggers[:]
    result.all_events[:] = original.all_events[:]
    result.dynamic_data = original.dynamic_data
    result.atomic_add_values[:] = original.atomic_add_values[:]
    return result


def _indexed_copy_task(template: TaskDescC, node: OperatorNode, dependency: int,
                       trigger: int, intermediate: int) -> TaskDescC:
    """Retain per-expert task coordinates and declare the precise copy input/output ports."""
    result = TaskDescC.from_buffer_copy(bytes(template))
    result.task_type = TaskType.TASK_NATIVE_BIT_REORDER
    result.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
    result.num_inputs, result.num_outputs = len(node.inputs), len(node.outputs)
    result.dependent_event, result.trigger_event = dependency, trigger
    result.extra_value_0, result.extra_value_1, result.extra_value_2 = intermediate * 2, 0, 0
    result.tiling_data_position = 0xFFFFFFFF
    for descriptors, tensors in ((result.inputs, node.inputs), (result.outputs, node.outputs)):
        for index, tensor in enumerate(tensors):
            descriptor = TensorDescC()
            descriptor.input_position, descriptor.data_type = tensor.param_position, tensor.dtype_size
            descriptor.tensor_type, descriptor.dynamic_shape = tensor.tensor_type, 1
            descriptor.dim[0], descriptor.dim[1] = tensor.shape[:2]
            descriptors[index] = descriptor
    return result


def build_native_backward(graph: ComputeGraph, original: RuntimeConfigC, values: TaskSplitValue,
                          spec: MegaMoeSpec) -> tuple[ComputeGraph, RuntimeConfigC]:
    """Prepare dW operands while independent ActGrad runs, preserving transport dX order."""
    stages: dict[str, list[int]] = {}
    cursor = 0
    for node in graph.topological_sort():
        stages[node.name] = list(range(cursor, cursor + node.task_num))
        cursor += node.task_num
    if cursor + 1 != original.task_num or original.all_tasks[cursor].task_type != TaskType.TASK_TERMINATE:
        raise RuntimeError("Native ordered dW requires one terminal task after the baseline graph.")
    swiglu_tasks = stages["swiglu_grad"]
    result_graph = _ordered_graph(graph, values, spec)
    result = _expanded_runtime(original, original.task_num + 2 * len(swiglu_tasks), spec.local_experts)
    result.native_order_intermediate = spec.intermediate_size
    used = max(index for index, count in enumerate(original.all_event_num_triggers) if count)
    first_event = (used + 8 + 15) // 16 * 16
    if first_event + 2 * spec.local_experts + 8 > result.event_capacity:
        raise RuntimeError("Native ordered dW events exceed the baseline counter workspace.")
    local, cores = spec.local_experts, spec.num_cube_cores
    copy2_event, copy1_event = first_event, first_event + local
    copies_per_expert = len(swiglu_tasks) // local
    for expert in range(local):
        result.all_event_num_triggers[copy2_event + expert] = copies_per_expert
        result.all_event_num_triggers[copy1_event + expert] = copies_per_expert
    ids: dict[int, int] = {}
    added: dict[str, list[int]] = {}
    cursor = 0
    for node in result_graph.topological_sort():
        added[node.name] = []
        for index in range(node.task_num):
            if node.name.startswith("native_order_"):
                template = original.all_tasks[swiglu_tasks[index]]
                expert = index // copies_per_expert
                copy2 = node.name.endswith("w2")
                producer = original.all_tasks[stages["w2_grad" if copy2 else "gate_grad"][expert * cores]]
                task = _indexed_copy_task(template, node, producer.dependent_event,
                                          (copy2_event if copy2 else copy1_event) + expert,
                                          spec.intermediate_size)
                task.extra_value_2 = int(spec.dispatch_mode == "pull")
            else:
                old_id = stages[node.name][index]
                ids[old_id] = cursor
                task = TaskDescC.from_buffer_copy(bytes(original.all_tasks[old_id]))
                expert = task.task_index // cores
                if node.name == "w2_grad":
                    task.inputs[0].input_position, task.inputs[1].input_position = 33, 34
                    task.dependent_event = copy2_event + expert
                elif node.name == "w1_grad":
                    task.inputs[0].input_position = 35 if spec.dispatch_mode == "pull" else 17
                    task.inputs[1].input_position = 26
                    task.dependent_event = copy1_event + expert
            result.all_tasks[cursor] = task
            added[node.name].append(cursor)
            cursor += 1
    terminal_id = original.task_num - 1
    ids[terminal_id] = cursor
    result.all_tasks[cursor] = original.all_tasks[terminal_id]
    cube = [task_id for name in ("act_grad", "w2_grad", "gate_grad", "w1_grad") for task_id in added[name]]
    result.cube_task_indices[:len(cube)] = cube
    count = original.task_index_num[2]
    result.mix_task_indices[:count] = [ids[index] for index in original.mix_task_indices[:count]]
    original_vector = list(original.vector_task_indices[:original.task_index_num[1]])
    vector = []
    for name in ("dispatch", "native_order_w2", "swiglu_grad", "native_order_w1", "combine"):
        if name.startswith("native_order_"):
            vector.extend(added[name])
        else:
            members = set(stages[name])
            vector.extend(ids[index] for index in original_vector if index in members)
    vector.extend(ids[index] for index in original_vector if index == terminal_id)
    result.vector_task_indices[:len(vector)] = vector
    result.task_index_num[:] = original.task_index_num[:]
    result.task_index_num[1] = len(vector)
    validate_runtime_config(result, values, cores)
    validate_mega_kernel_diagnostic_contract(result_graph, result, 36)
    return result_graph, result
