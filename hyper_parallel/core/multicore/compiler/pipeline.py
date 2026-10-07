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
"""Separate semantic matching, scheduling and family ABI emission."""

from __future__ import annotations

from collections.abc import Mapping

from hyper_parallel.core.multicore.backends.legacy import gate_runtime_image
from hyper_parallel.core.multicore.compiler.gate import (
    match_gate_route,
    resolve_gate_shape,
)
from hyper_parallel.core.multicore.ir.program import ProgramIR
from hyper_parallel.core.multicore.ir.schedule import (
    HardwareSpec,
    PipelineScheduleIR,
    PipelineStage,
    RowPartition,
    WorkerPipeline,
)
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.plan import Binding, KernelPlan, RuntimeImage

_POST_CALLS = (
    "HyperMegaGateLinearIndex",
    "HyperMegaGateScatterSelectedGrad",
    "HyperMegaGateDoubleRouteScores",
    "HyperMegaGateSqrtInputGrad",
    "HyperMegaGateSoftplusV2Grad",
    "HyperMegaGateAddGrad(optional_direct)",
)
_SAVED_STATE = ("logits", "expert_indices", "route_scores", "selected_scores", "normalization_denominator")


def _runtime(abi, template, stages):
    return RuntimeImage(
        template.kernel_name,
        stages,
        gate_runtime_image(abi, template),
        gate_runtime_image(abi, template, profiled=True),
    )


def _stages(template, source_groups, fallback):
    return tuple(
        PipelineStage(
            logical_name,
            display_name,
            tuple(operation.source for operation in source_groups[index]) or (fallback,),
            "native_primitive" if source_groups[index] else "retained_legacy_k1_stage",
        )
        for index, (logical_name, display_name) in enumerate(zip(template.logical_stages, template.stage_names))
    )


def compile_worker_pipeline(
    ir: ProgramIR, schedule: WorkerPipeline, signature: Mapping[str, int], topology: HardwareSpec
) -> KernelPlan:
    """Compile a supported Route program into explicit host images and binding metadata.

    Args:
        ir: Typed semantic program.
        schedule: Ordered row-owner AIV execution mode.
        signature: Bindings for symbolic token/expert dimensions.
        topology: Caller-provided available hardware workers.
    """
    if not isinstance(schedule, WorkerPipeline) or not isinstance(topology, HardwareSpec):
        raise TypeError("Gate compatibility lowering requires WorkerPipeline and HardwareSpec")
    recipe = match_gate_route(ir)
    token_count, expert_count = resolve_gate_shape(recipe, signature)
    abi = family_abi("gate")
    forward = abi.pipeline("forward")
    backward = abi.pipeline("backward_k1" if recipe.top_k == 1 else "backward")
    fallback = recipe.stage_sources[4][0].source
    stages = _stages(forward, recipe.stage_sources, fallback)
    worker_count = min(token_count, topology.available_aiv_workers, 48)
    rows_per_worker = (token_count + worker_count - 1) // worker_count
    partitions = tuple(
        RowPartition(
            worker, worker * rows_per_worker, min(rows_per_worker, max(0, token_count - worker * rows_per_worker))
        )
        for worker in range(worker_count)
    )
    schedule_ir = PipelineScheduleIR(stages, partitions, rows_per_worker)
    backward_stages = tuple(
        PipelineStage(logical, display, (fallback,), "explicit_native_backward_recipe")
        for logical, display in zip(backward.logical_stages, backward.stage_names)
    )
    bindings, backward_bindings = _bindings(recipe)
    return KernelPlan(
        abi,
        schedule_ir,
        _runtime(abi, forward, stages),
        _runtime(abi, backward, backward_stages),
        bindings,
        backward_bindings,
        _SAVED_STATE,
        _POST_CALLS,
        recipe.top_k,
        recipe.scale,
        token_count,
        expert_count,
    )


def _bindings(recipe):
    # Preserve raw native kernel order, including text-only vision aliases and implicit caches.
    values = (
        recipe.logits.id,
        recipe.bias.id,
        recipe.bias.id,
        None,
        None,
        None,
        recipe.weights.id,
        recipe.index_output.id,
        recipe.scores.id,
        recipe.selected_scores.id,
        recipe.denominator.id if recipe.denominator is not None else None,
        None,
        None,
    )
    names = (
        "logits",
        "text_bias",
        "vision_bias(text_alias)",
        "image_mask_placeholder",
        "runtime_config",
        "profile_buffer",
        "routing_weights",
        "expert_indices",
        "route_scores",
        "selected_scores",
        "normalization_denominator",
        "workspace",
        "tiling",
    )
    bindings = tuple(Binding(name, position, value) for position, (name, value) in enumerate(zip(names, values)))
    grad_names = (
        "selected_scores",
        "normalization_denominator",
        "grad_routing_weights",
        "route_scores",
        "expert_indices",
        "runtime_config",
        "profile_buffer",
        "selected_score_grad",
        "zero_score_grad",
        "workspace",
        "tiling",
    )
    grad_values = (
        recipe.selected_scores.id,
        recipe.denominator.id if recipe.denominator is not None else None,
        None,
        recipe.scores.id,
        recipe.indices.id,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    backward_bindings = tuple(
        Binding(name, position, value) for position, (name, value) in enumerate(zip(grad_names, grad_values))
    )
    return bindings, backward_bindings
