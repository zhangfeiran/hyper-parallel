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
"""Validate Gate Route semantics before selecting pinned legacy worker templates."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass

from hyper_parallel.core.multicore.frontend.diagnostics import FrontendError
from hyper_parallel.core.multicore.ir.program import Operation, ProgramIR, Value
from hyper_parallel.core.multicore.language.types import DType


@dataclass(frozen=True)
class GateRouteRecipe:
    """Checked numerical graph and source nodes for each legacy forward stage."""

    logits: Value
    bias: Value
    scores: Value
    indices: Value
    selected_scores: Value
    denominator: Value | None
    weights: Value
    index_output: Value
    top_k: int
    scale: float
    stage_sources: tuple[tuple[Operation, ...], ...]


class _RouteMatcher:
    def __init__(self, ir: ProgramIR) -> None:
        """Index canonical tensor producers for structural recipe matching."""
        self.ir = ir
        self.producers = {value.id: operation for operation in ir.operations for value in operation.outputs}
        self.visited = set()

    def _node(self, value, logical_name):
        operation = self.producers.get(value.id) if isinstance(value, Value) else None
        if operation is None:
            raise ValueError(f"Legacy Gate Route requires a {logical_name} tensor producer")
        if operation.logical_name != logical_name or operation.version != 1 or len(operation.outputs) != 1:
            raise FrontendError(f"Legacy Gate Route requires {logical_name}.v1 here", operation.source)
        self.visited.add(operation.outputs[0].id)
        return operation

    @staticmethod
    def _arguments(operation):
        return dict(operation.arguments)

    @staticmethod
    def _require(operation, condition, message):
        if not condition:
            raise FrontendError(message, operation.source)

    def _normalization(self, selected, top_k):
        if top_k == 1:
            return selected, None, ((), (), ())
        divide = self._node(selected, "gate.divide")
        arguments = self._arguments(divide)
        gathered = arguments["left"]
        denominator = arguments["right"]
        addition = self._node(denominator, "gate.add")
        added = self._arguments(addition)
        self._require(
            addition,
            type(added["right"]) in (float,) and added["right"] == 1.0e-20,
            "Legacy Gate Route requires epsilon=1.0e-20 after reduction",
        )
        reduction = self._node(added["left"], "gate.reduce_sum")
        reduced = self._arguments(reduction)
        self._require(
            reduction,
            reduced == {"value": gathered, "axis": -1, "keepdim": True},
            "Legacy Gate Route requires selected-score row sums with keepdim=True",
        )
        return gathered, denominator, ((reduction,), (addition,), (divide,))

    def match(self) -> GateRouteRecipe:
        """Validate the Route graph and return its complete native-stage provenance."""
        if self.ir.numeric_policy != "preserve_numeric_order" or len(self.ir.outputs) != 2 or not self.ir.returns_tuple:
            raise ValueError("Legacy Gate Route requires preserved numerical order and weights/indices tuple outputs")
        weights, index_output = self.ir.outputs
        scale_op = self._node(weights, "gate.multiply")
        scale_args = self._arguments(scale_op)
        self._require(
            scale_op,
            type(scale_args["right"]) in (int, float) and math.isfinite(scale_args["right"]),
            "Legacy Gate Route requires a finite static scale",
        )
        try:
            struct.pack("<f", scale_args["right"])
        except (OverflowError, struct.error) as exc:
            raise FrontendError("Legacy Gate scale must be representable as FP32", scale_op.source) from exc
        cast = self._node(index_output, "gate.cast")
        cast_args = self._arguments(cast)
        self._require(cast, cast_args["dtype"] == DType.INT64, "Legacy Gate indices must be cast to int64")
        indices = cast_args["value"]
        topk = self._node(indices, "gate.topk_indices")
        topk_args = self._arguments(topk)
        count = topk_args["k"]
        self._require(
            topk,
            type(count) in (int,) and count > 0 and topk_args["axis"] == -1 and topk_args["sorted"] is False,
            "Legacy Gate topk requires positive k, axis=-1 and sorted=False",
        )
        gathered, denominator, normalization_sources = self._normalization(scale_args["left"], count)
        gather = self._node(gathered, "gate.gather")
        gather_args = self._arguments(gather)
        scores = gather_args["value"]
        self._require(
            gather,
            gather_args["indices"] == indices and gather_args["axis"] == -1,
            "Legacy Gate gather must use the selected indices and original scores",
        )
        selection = self._node(topk_args["value"], "gate.add")
        select_args = self._arguments(selection)
        self._require(selection, select_args["left"] == scores, "Legacy Gate selection must add bias to route scores")
        detached = self._node(select_args["right"], "gate.stop_gradient")
        bias = self._arguments(detached)["value"]
        square_root = self._node(scores, "gate.sqrt")
        softplus = self._node(self._arguments(square_root)["value"], "gate.softplus")
        logits = self._arguments(softplus)["value"]
        self._validate_inputs(logits, bias)
        if len(self.visited) != len(self.ir.operations):
            raise ValueError("Legacy Gate Route cannot lower additional operations or effects")
        sources = ((softplus,), (square_root,), (selection, detached), (topk,), (gather,))
        sources += normalization_sources + ((scale_op,), (cast,))
        return GateRouteRecipe(
            logits,
            bias,
            scores,
            indices,
            gathered,
            denominator,
            weights,
            index_output,
            count,
            float(scale_args["right"]),
            sources,
        )

    def _validate_inputs(self, logits, bias):
        if (
            not isinstance(logits, Value)
            or not isinstance(bias, Value)
            or {logits.id, bias.id} != {value.id for value in self.ir.inputs}
        ):
            raise ValueError("Legacy Gate Route requires exactly logits and bias runtime inputs")
        if (
            logits.type.dtype != DType.FP32
            or bias.type.dtype != DType.FP32
            or len(logits.type.shape) != 2
            or bias.type.shape != (logits.type.shape[1],)
        ):
            raise ValueError("Legacy Gate Route requires FP32 logits[T,E] and FP32 bias[E]")
        self._validate_effects()

    def _validate_effects(self):
        # Logical access changes cannot be hidden by matching only an operation name.
        for operation in self.ir.operations:
            args = self._arguments(operation)
            expected_reads = {value.id for value in args.values() if isinstance(value, Value)}
            reads = {effect.value_id for effect in operation.effects if effect.kind == "read"}
            writes = {effect.value_id for effect in operation.effects if effect.kind == "write"}
            output_ids = {value.id for value in operation.outputs}
            self._require(
                operation,
                reads == expected_reads
                and writes == output_ids
                and all(effect.kind in ("read", "write") for effect in operation.effects),
                "Legacy Gate semantic primitives require pure read/new-result effects",
            )


def match_gate_route(ir: ProgramIR) -> GateRouteRecipe:
    """Prove a semantic graph matches the existing text Route numerical contract.

    Args:
        ir: Typed computation in preserved numerical order.
    """
    return _RouteMatcher(ir).match()


def resolve_gate_shape(recipe: GateRouteRecipe, signature: Mapping[str, int]) -> tuple[int, int]:
    """Resolve positive native dimensions without reading any runtime tensor.

    Args:
        recipe: Validated Route graph with static or symbolic tensor dimensions.
        signature: Static bindings for the symbolic input dimensions.
    """
    required = {dimension for dimension in recipe.logits.type.shape if isinstance(dimension, str)}
    if set(signature) != required:
        raise ValueError(f"Gate shape signature requires exactly these symbols: {sorted(required)}")
    dimensions = tuple(
        signature[dimension] if isinstance(dimension, str) else dimension for dimension in recipe.logits.type.shape
    )
    if any(type(dimension) not in (int,) or not 0 < dimension < 1 << 32 for dimension in dimensions):
        raise ValueError("Gate token and expert dimensions must be positive uint32 integers")
    if recipe.top_k > dimensions[1]:
        raise ValueError("Gate top_k exceeds the resolved expert dimension")
    return dimensions
