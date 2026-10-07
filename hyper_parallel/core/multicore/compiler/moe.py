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
"""Validated MoE region lowering through the original fill/finalize contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hyper_parallel.core.multicore.ir.program import ProgramIR, Value
from hyper_parallel.core.multicore.language.types import (
    DType,
    RaggedTensorType,
    RouteMetadataType,
    TensorListType,
)
from hyper_parallel.core.multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import TaskSplitValue
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph

_STAGES = ("moe.dispatch", "moe.grouped_matmul", "moe.swiglu_packed", "moe.grouped_matmul", "moe.combine")
_LEGACY_NAMES = ("dispatch", "up_proj", "swiglu", "down_proj", "combine")


@dataclass(frozen=True)
class MoeRecipe:
    """The fixed legacy numerical recipe with AST identities and source provenance."""

    ir: ProgramIR
    limit: float | None

    @property
    def fingerprint(self) -> str:
        """Hash the structural program without checkout-dependent source locations."""
        data = json.loads(self.ir.dump())
        for operation in data["operations"]:
            operation.pop("source")
            operation.pop("call_chain")
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def build_graph(self, spec: MegaMoeSpec) -> ComputeGraph:
        """Instantiate fixed native nodes and derive dependency edges from the AST.

        Args:
            spec: Existing module's shape/topology/capacity contract.
        """
        self.validate_spec(spec)
        values = TaskSplitValue(tp=1, ep=spec.ep_size, seq_size=spec.local_num_tokens,
                                all_expert_num=spec.num_experts, top_k=spec.top_k,
                                dispatch_mode=spec.dispatch_mode)
        template = build_forward_graph(
            values, dispatch_sv=spec.dispatch_split, swiglu_sv=spec.swiglu_split,
            combine_sv=spec.combine_split, hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size, num_cube_cores=spec.num_cube_cores,
            swiglu_limit=self.limit,
        )
        nodes = [template.get_op(name) for name in _LEGACY_NAMES]
        graph = ComputeGraph()
        for node in nodes:
            node.predecessors.clear()
            node.successors.clear()
            graph.add_op(node)
        for producer, consumer in _edges(self.ir):
            graph.add_edge(nodes[producer], nodes[consumer])
        return graph

    def validate_spec(self, spec: MegaMoeSpec) -> None:
        """Validate logical shapes against physical bindings without freezing route counts.

        Args:
            spec: Rank-local native specification, including guest execution slots.
        """
        if not isinstance(spec, MegaMoeSpec):
            raise TypeError("MoE planning requires a bound MegaMoeSpec")
        if self.limit != spec.swiglu_limit:
            raise ValueError("Program SwiGLU limit must match the MegaMoe module specification")
        dimensions = {}
        shapes = ((spec.routed_slots, spec.hidden_size),
                  (spec.hidden_size, 2 * spec.intermediate_size),
                  (spec.intermediate_size, spec.hidden_size), (spec.local_experts,))
        for value, shape in zip(self.ir.inputs, shapes):
            declared = value.type.shape
            if declared is None:
                continue
            for expected, actual in zip(declared, shape):
                if isinstance(expected, str):
                    if dimensions.setdefault(expected, actual) != actual:
                        raise ValueError(f"MoE symbolic dimension {expected} disagrees with native bindings")
                elif expected is not None and expected != actual:
                    raise ValueError(f"MoE shape mismatch for {value.name}: {declared}, native {shape}")


def match_moe_region(ir: ProgramIR) -> MoeRecipe:
    """Accept only the original dispatch/GMM/SwiGLU/GMM/combine numerical contract.

    Args:
        ir: Canonically registered typed semantic computation.
    """
    if ir.numeric_policy != "preserve_numeric_order" or len(ir.inputs) != 4 or len(ir.operations) != 5:
        raise ValueError("MoE compatibility requires the complete five-stage region and four runtime inputs")
    if tuple(operation.logical_name for operation in ir.operations) != _STAGES:
        raise ValueError("Unsupported MoE primitive sequence")
    _validate_inputs(ir.inputs)
    routed, weight1, weight2, metadata = ir.inputs
    dispatch_op, first, activation, second, combine_op = ir.operations
    received, groups = dispatch_op.outputs
    limit = dict(activation.arguments)["limit"]
    expected = (
        {"routed_x": routed, "route_meta": metadata},
        {"value": received, "weights": weight1, "groups": groups},
        {"packed": first.outputs[0], "layout": "gate_up", "limit": limit},
        {"value": activation.outputs[0], "weights": weight2, "groups": groups},
        {"projected": second.outputs[0], "route_meta": metadata},
    )
    _validate_operations(ir.operations, expected)
    if ir.outputs != combine_op.outputs or ir.returns_tuple:
        raise ValueError("MoE region must return only the combined routed tensor")
    return MoeRecipe(ir, limit)


def _validate_inputs(inputs):
    routed, weight1, weight2, metadata = inputs
    if not isinstance(routed.type, RaggedTensorType) or routed.type.dtype != DType.BF16:
        raise ValueError("MoE routed input must be BF16 RaggedTensor")
    if any(not isinstance(value.type, TensorListType) or value.type.dtype != DType.BF16
           for value in (weight1, weight2)) or not isinstance(metadata.type, RouteMetadataType):
        raise ValueError("MoE requires two BF16 TensorLists and RouteMetadata in native binding order")


def _validate_operations(operations, expected):
    for operation, arguments in zip(operations, expected):
        if operation.version != 1 or dict(operation.arguments) != arguments:
            raise ValueError(f"MoE operands do not match the native recipe at {operation.logical_name}")
        if any(effect.kind != "read" for effect in operation.effects if effect.value_id not in {
            output.id for output in operation.outputs
        }):
            raise ValueError("MoE compatibility rejects mutating primitive inputs")


def _edges(ir):
    producers, ancestors, edges = {}, {}, []
    for index, operation in enumerate(ir.operations):
        dependencies = {
            producers[value.id] for _, value in operation.arguments
            if isinstance(value, Value) and value.id in producers
        }
        immediate = {parent for parent in dependencies
                     if not any(parent in ancestors[other] for other in dependencies if other != parent)}
        edges.extend((parent, index) for parent in sorted(immediate))
        ancestors[index] = dependencies | {ancestor for parent in dependencies for ancestor in ancestors[parent]}
        for output in operation.outputs:
            producers[output.id] = index
    return tuple(edges)
