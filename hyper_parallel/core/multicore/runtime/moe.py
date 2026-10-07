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
"""CPU MoE plans with complete legacy images and fixed-stride queue provenance."""

from __future__ import annotations

import json
import struct
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any

from hyper_parallel.core.multicore.compiler.moe import MoeRecipe
from hyper_parallel.core.multicore.ir.program import SourceSpan
from hyper_parallel.core.multicore.modules.mega_moe.backward.storage import (
    prepare_w13_overlap_schedule,
)
from hyper_parallel.core.multicore.modules.mega_moe.plan import (
    _build_runtime_artifacts,
    build_mega_moe_plan,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.profiler.profiling import (
    _prepare_mega_kernel_runtime_config,
)
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.moe_native import verify_moe_native


@dataclass(frozen=True)
class TaskDAGImage:
    """Full native wire images, physical queues and source spans for one direction."""

    normal: bytes
    profiled: bytes
    queues: tuple[tuple[int, ...], ...]
    worker_counts: tuple[int, ...]
    stages: tuple[tuple[str, int, int, SourceSpan], ...]
    completion_event: int
    protocol_version: int

    def worker_queues(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Return the exact per-worker FIFO visits using the native fixed queue stride."""
        return tuple(tuple(queue[worker::count] for worker in range(count))
                     for queue, count in zip(self.queues, self.worker_counts))


@dataclass(frozen=True)
class MoeKernelPlan:
    """A TaskDAG compatibility plan; dynamic route counts and native pointers stay external."""

    recipe: MoeRecipe
    spec: MegaMoeSpec
    forward: TaskDAGImage
    backward: TaskDAGImage
    backward_no_replica: TaskDAGImage | None

    def materialize(self, device: Any) -> Any:
        """Materialize through the existing MoE tiling/profiling bridge.

        Args:
            device: NPU receiving the original forward/backward resources.

        Returns:
            Existing MegaMoePlan, consumed by the original autograd/workspace path.
        """
        verify_moe_native()
        return build_mega_moe_plan(self.spec, device, forward_graph_factory=self.recipe.build_graph)

    def export_manifest(self) -> dict[str, object]:
        """Export capacities, physical bindings, source identity and the explicit backward boundary."""
        return {
            **family_abi("moe").export_manifest(), "native_status": "unbound",
            "program_fingerprint": self.recipe.fingerprint,
            "schedule": "moe_ratr_v1", "execution_mode": "fixed_stride_task_dag",
            "spec": {field.name: getattr(self.spec, field.name) for field in fields(self.spec)
                     if field.name != "ep_group"},
            "row_capacities": {"source": self.spec.routed_slots, "receive_initial": self.spec.receive_capacity,
                               "receive_maximum": self.spec.maximum_receive_capacity},
            "runtime_counts": "borrowed_device_group_list",
            "backward_recipe": "original_moe_autograd_and_replica_return",
            "forward_bindings": {
                "routed_x": (2,), "w13": (5,), "w2": (9,),
                "route_meta": (1, 3, 4, 6, 10, 13, 14, 15),
                "received": (0,), "packed": (7,), "activated": (8,), "projected": (11,), "combined": (12,),
            },
        }

    def explain(self) -> str:
        """Dump native queues, stage ranges and preserved transport handshakes."""
        return json.dumps({"manifest": self.export_manifest(),
                           "forward": _describe(self.forward), "backward": _describe(self.backward),
                           "backward_no_replica": _describe(self.backward_no_replica)}, indent=2, sort_keys=True)


def compile_moe_plan(recipe: MoeRecipe, spec: MegaMoeSpec) -> MoeKernelPlan:
    """Compile original finalized runtime variants from an AST-derived forward graph.

    Args:
        recipe: Validated five-stage semantic program.
        spec: Existing module's rank-local topology and static shapes.
    """
    forward_graph, forward, backward_graph, regular = _build_runtime_artifacts(
        spec, overlap_w13=False, forward_graph_factory=recipe.build_graph,
    )
    backward, fallback = regular, None
    if spec.replica_slots_per_rank and spec.replica_transport == "shmem_signal_kernel_gradient":
        overlap = deepcopy(regular)
        if prepare_w13_overlap_schedule(overlap, spec.local_experts, spec.num_cube_cores):
            backward, fallback = overlap, regular
    return MoeKernelPlan(
        recipe, spec, _image(forward, forward_graph, recipe, False, spec),
        _image(backward, backward_graph, recipe, True, spec),
        _image(fallback, backward_graph, recipe, True, spec) if fallback is not None else None,
    )


def _image(config, graph, recipe, backward, spec):
    offset = family_abi("moe").structure("RuntimeConfigC").offset("cycle_profiling_enabled")

    def _enable(wire):
        enabled = bytearray(wire)
        struct.pack_into("<I", enabled, offset, 1)
        return bytes(enabled)

    runtime = _prepare_mega_kernel_runtime_config(
        config, tensor_factory=bytes, profile_tensor_factory=_enable, rank=spec.rank_id, device_id=0,
    )
    stages, first = [], 0
    sources = {name: operation.source for name, operation in zip(
        ("dispatch", "up_proj", "swiglu", "down_proj", "combine"), recipe.ir.operations,
    )}
    inverse = {"dispatch": "combine", "combine": "dispatch",
               "act_grad": "down_proj", "w2_grad": "down_proj", "swiglu_grad": "swiglu",
               "gate_grad": "up_proj", "w1_grad": "up_proj"}
    for operation in graph.topological_sort():
        source = sources[inverse.get(operation.name, operation.name)] if backward else sources[operation.name]
        stages.append((operation.name, first, operation.task_num, source))
        first += operation.task_num
    queues = tuple(tuple(int(task) for task in queue[:int(count)]) for queue, count in zip(
        (config.cube_task_indices, config.vector_task_indices, config.mix_task_indices), config.task_index_num[:3],
    ))
    return TaskDAGImage(runtime.normal_tensor, runtime.profile_tensor, queues,
                        (spec.num_cube_cores, 2 * spec.num_cube_cores, spec.num_cube_cores), tuple(stages),
                        int(config.completion_event), int(config.protocol_version))


def _describe(image):
    if image is None:
        return None
    return {
        "bytes": len(image.normal), "completion_event": image.completion_event,
        "protocol_version": image.protocol_version, "queues": image.queues,
        "worker_counts": image.worker_counts,
        "stages": [{"stage": name, "first_task": first, "task_count": count, "source": str(span)}
                   for name, first, count, span in image.stages],
    }
