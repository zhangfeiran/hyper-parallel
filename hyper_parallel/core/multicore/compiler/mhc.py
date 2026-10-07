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
"""Validate the shifted MHC semantic fork and expand the original six-stage template."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hyper_parallel.core.multicore.compiler._mhc_forward import build_mega_mhc_graph
from hyper_parallel.core.multicore.ir.program import ProgramIR, Value
from hyper_parallel.core.multicore.language.types import DType, TensorType
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph

STAGE_NAMES = ("mhc_post", "mhc_norm_cast", "mhc_projection", "mhc_input_mix", "mhc_mapping", "mhc_rms_norm")


@dataclass(frozen=True)
class MhcRecipe:
    """The shifted native numerical contract with canonical SSA operands."""

    ir: ProgramIR
    hc_eps: float
    norm_eps: float
    num_iters: int

    @property
    def fingerprint(self) -> str:
        """Hash semantic types/constexprs without source paths or device pointers."""
        value = json.loads(self.ir.dump())
        for operation in value["operations"]:
            operation.pop("source")
            operation.pop("call_chain")
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def validate_spec(self, spec: MhcSpec) -> None:
        """Unify native flattened shapes, including the projection's 4H dimension.

        Args:
            spec: Static flattened token, hidden and hardware contract.
        """
        dimensions = {}
        rows, hidden = spec.token_count, spec.hidden_size
        shapes = ((rows, 4, hidden), (rows, hidden), (rows, 4), (rows, 4), (rows, 4, 4),
                  (24, 4 * hidden), (3,), (24,), (hidden,))
        for value, shape in zip(self.ir.inputs, shapes):
            for declared, actual in zip(value.type.shape, shape):
                if isinstance(declared, str):
                    if dimensions.setdefault(declared, actual) != actual:
                        raise ValueError(f"MHC symbolic dimension mismatch for {value.name}: {declared}")
                elif declared != actual:
                    raise ValueError(f"MHC native shape mismatch for {value.name}: {value.type.shape}, {shape}")

    def build_graph(self, spec: MhcSpec) -> tuple[ComputeGraph, object]:
        """Expand the validated mapping decomposition with the original ring/wave policy.

        Args:
            spec: Shape, physical core and tile contract.
        """
        self.validate_spec(spec)
        template, topology = build_mega_mhc_graph(
            spec.token_count, spec.hidden_size, spec.num_cube_cores, spec.token_tile)
        nodes = {name: template.get_op(name) for name in STAGE_NAMES}
        graph = ComputeGraph()
        for name in ("mhc_post", "mhc_norm_cast", "mhc_projection", "mhc_input_mix", "mhc_rms_norm", "mhc_mapping"):
            node = nodes[name]
            node.predecessors.clear()
            node.successors.clear()
            graph.add_op(node)
        for producer, consumer in _expanded_edges(self.ir):
            graph.add_edge(nodes[producer], nodes[consumer])
        return graph, topology

    def stage_sources(self) -> dict[str, object]:
        """Map decomposed native stages back to the semantic operation defining them."""
        post, mapping, mixing, rms = self.ir.operations
        return {"mhc_post": post.source, "mhc_norm_cast": mapping.source, "mhc_projection": mapping.source,
                "mhc_mapping": mapping.source, "mhc_input_mix": mixing.source, "mhc_rms_norm": rms.source}

    def backward_sources(self) -> dict[str, object]:
        """Associate each fused reverse stage with the semantic operation(s) it differentiates."""
        post, mapping, mixing, rms = self.ir.operations
        return {"grad_output_init": tuple(operation.source for operation in self.ir.operations),
                "rms_norm_grad": rms.source,
                "mhc_grad_prev_a_and_mapping": (mixing.source, mapping.source),
                "mhc_grad_phi_rms": mapping.source, "mhc_grad_prev_x_and_post": post.source}


def match_mhc_region(ir: ProgramIR) -> MhcRecipe:
    """Accept only the full five-output, previous-pre shifted boundary.

    Args:
        ir: Canonically registered, typed semantic source.
    """
    if (ir.numeric_policy != "preserve_numeric_order" or len(ir.inputs) != 9 or len(ir.operations) != 4
            or not ir.returns_tuple):
        raise ValueError("MHC compatibility requires nine inputs and the complete four-operation shifted boundary")
    if tuple(operation.logical_name for operation in ir.operations) != (
            "mhc.post", "mhc.mapping", "mhc.input_mix", "mhc.rms_norm"):
        raise ValueError("Unsupported shifted MHC primitive order")
    _validate_input_types(ir.inputs)
    residual, output, pre, post, matrix, phi, alpha, bias, weight = ir.inputs
    post_op, mapping, mixing, rms = ir.operations
    attributes = dict(mapping.arguments)
    hc_eps, norm_eps, iterations = attributes["hc_eps"], attributes["norm_eps"], attributes["num_iters"]
    expected = (
        {"residual": residual, "previous_output": output, "previous_post": post, "previous_residual": matrix},
        {"updated": post_op.outputs[0], "phi": phi, "alpha": alpha, "bias": bias,
         "hc_eps": hc_eps, "norm_eps": norm_eps, "num_iters": iterations},
        {"updated": post_op.outputs[0], "previous_pre": pre},
        {"value": mixing.outputs[0], "weight": weight, "eps": norm_eps},
    )
    for operation, arguments in zip(ir.operations, expected):
        if operation.version != 1 or dict(operation.arguments) != arguments:
            raise ValueError(f"Shifted MHC native operand mismatch: {operation.logical_name}")
        if any(effect.kind != "read" for effect in operation.effects
               if effect.value_id not in {value.id for value in operation.outputs}):
            raise ValueError("MHC compatibility rejects mutating primitive inputs")
    if ir.outputs != (post_op.outputs[0], *mapping.outputs, rms.outputs[0]):
        raise ValueError("MHC must return updated residual, next mixes and block input in native order")
    return MhcRecipe(ir, hc_eps, norm_eps, iterations)


def _validate_input_types(inputs):
    ranks = (3, 2, 2, 2, 3, 2, 1, 1, 1)
    dtypes = (DType.BF16, DType.BF16, DType.FP32, DType.FP32, DType.FP32,
              DType.FP32, DType.FP32, DType.FP32, DType.BF16)
    for value, rank, dtype in zip(inputs, ranks, dtypes):
        if not isinstance(value.type, TensorType) or value.type.dtype != dtype or len(value.type.shape) != rank:
            raise ValueError(f"MHC native input type mismatch: {value.name}")


def _expanded_edges(ir):
    roots = ("mhc_post", "mhc_norm_cast", "mhc_input_mix", "mhc_rms_norm")
    leaves = ("mhc_post", "mhc_mapping", "mhc_input_mix", "mhc_rms_norm")
    producers, edges = {}, [("mhc_norm_cast", "mhc_projection"), ("mhc_projection", "mhc_mapping")]
    for index, operation in enumerate(ir.operations):
        for _, value in operation.arguments:
            if isinstance(value, Value) and value.id in producers:
                edge = (producers[value.id], roots[index])
                if edge not in edges:
                    edges.append(edge)
        for value in operation.outputs:
            producers[value.id] = leaves[index]
    # NormCast precedes InputMix under the preserved AIV gap-filling policy.
    edges.append(("mhc_norm_cast", "mhc_input_mix"))
    edges.remove(("mhc_post", "mhc_input_mix"))
    return tuple(edges)
