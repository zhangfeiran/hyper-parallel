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
"""Complete MHC family images with fork, ring/cache and macro-join contracts."""

from __future__ import annotations

import ctypes
import json
import struct
from dataclasses import asdict, dataclass
from typing import Any

from hyper_parallel.core.multicore.compiler._mhc_backward import (
    build_event_layout as build_grad_event_layout,
)
from hyper_parallel.core.multicore.compiler._mhc_backward import (
    build_mega_mhc_grad_graph,
    make_grad_tiling,
    order_vector_tasks_by_stage,
    resolve_grad_token_tile,
)
from hyper_parallel.core.multicore.compiler._mhc_forward import (
    build_event_layout,
    resolve_token_tile,
)
from hyper_parallel.core.multicore.compiler.mhc import MhcRecipe
from hyper_parallel.core.multicore.profiler.profiling import (
    _apply_mega_kernel_profile_graph,
    _prepare_mega_kernel_runtime_config,
    _ProfileSpec,
)
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.mhc_native import MhcExecutable
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from hyper_parallel.core.multicore.scheduler.config import init_task_split_value
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config


@dataclass(frozen=True)
class MhcRuntimeImage:
    """Native images and physical FIFO/event tables for one direction."""

    normal: bytes
    profiled: bytes
    queues: tuple[tuple[int, ...], ...]
    workers: tuple[int, ...]
    tasks: tuple[tuple[int, int, int, int, int, int], ...]
    thresholds: tuple[int, ...]
    stages: tuple[tuple[str, int, int], ...]

    def simulate(self) -> dict[str, object]:
        """Check fixed-stride FIFO progress against the original counter thresholds."""
        fifos = [queue[worker::count] for queue, count in zip(self.queues, self.workers) for worker in range(count)]
        cursors, counters, visited = [0] * len(fifos), [0] * len(self.thresholds), []
        while len(visited) < sum(len(queue) for queue in fifos):
            progress = False
            for worker, queue in enumerate(fifos):
                if cursors[worker] == len(queue):
                    continue
                task_id = queue[cursors[worker]]
                _, dependent, trigger, _, _, _ = self.tasks[task_id]
                if dependent != 0xFFFFFFFF and counters[dependent] < self.thresholds[dependent]:
                    continue
                if trigger != 0xFFFFFFFF:
                    counters[trigger] += 1
                cursors[worker] += 1
                visited.append(task_id)
                progress = True
            if not progress:
                raise ValueError("MHC fixed-stride FIFO has an unsatisfied head dependency")
        return {"visited": visited, "counters": counters, "instruction_simulation": False}


@dataclass(frozen=True)
class MhcKernelPlan:
    """Host compatibility result with explicit saved-state and scratch sub-buffers."""

    recipe: MhcRecipe
    spec: MhcSpec
    forward: MhcRuntimeImage
    backward: MhcRuntimeImage | None
    token_tile: int
    grad_token_tile: int
    ring_slots: int
    macro_join: tuple[tuple[int, ...], ...]

    def materialize(self, device: str = "npu:0", *, payload_root: Any = None) -> MhcExecutable:
        """Bind native MHC resources after family/artifact and device checks.

        Args:
            device: Ascend device for the original family-native workers.
            payload_root: Isolated build-produced MHC payload.
        """
        return MhcExecutable(self, device, payload_root)

    def export_manifest(self) -> dict[str, object]:
        """Export semantic identity and explicit hidden scratch/cache contracts."""
        rows, hidden = self.spec.token_count, self.spec.hidden_size
        return {
            **family_abi("mhc").export_manifest(), "native_status": "unbound",
            "program_fingerprint": self.recipe.fingerprint, "schedule": "shifted_mhc_v1",
            "spec": asdict(self.spec), "token_tile": self.token_tile,
            "grad_token_tile": self.grad_token_tile, "ring_slots": self.ring_slots,
            "ring_bytes": self.ring_slots * self.token_tile * 4 * hidden * 4,
            "ring_reuse": "NormCast(i) waits for Projection(i-ring_slots)",
            "semantic_fork": [["mhc_post", "mhc_norm_cast"], ["mhc_post", "mhc_input_mix"]],
            "policy_edge": ["mhc_norm_cast", "mhc_input_mix"],
            "shifted_input_mix": "previous_pre", "macro_join": self.macro_join,
            "backward_recipe": "original_MHC_prepare_phi_rms_previous_x_post",
            "cache_buffers": [
                {"name": "new_residual", "slot": 12, "shape": (rows, 4, hidden), "dtype": "bf16"},
                {"name": "hc_before_norm", "slot": 17, "shape": (rows, 24), "dtype": "fp32"},
                {"name": "inv_rms", "slot": 18, "shape": (rows, 1), "dtype": "fp32"},
                {"name": "sum_out", "slot": 19, "shape": (40, rows, 4), "dtype": "fp32"},
                {"name": "norm_out", "slot": 20, "shape": (40, rows, 4, 4), "dtype": "fp32"},
                {"name": "mixed_input", "slot": 21, "shape": (rows, hidden), "dtype": "bf16"},
                {"name": "rms_rstd", "slot": 22, "shape": (rows, 1), "dtype": "fp32"},
            ],
            "scratch_buffers": [
                {"name": "x_cast_ring", "binding": "user_workspace", "dtype": "fp32"},
                {"name": "projection_scratch", "binding": "native_tiling_workspace", "dtype": "fp32"},
            ],
            "legacy_tensor_spec_slot_17": "placeholder; physical caches and scratch remain separate",
            "saved_state_lifetime": "per_autograd_invocation",
        }

    def explain(self) -> str:
        """Describe queues, stage source spans, ring ownership and backward joins."""
        return json.dumps({"manifest": self.export_manifest(),
                           "forward": _describe(self.forward, self.recipe.stage_sources()),
                           "backward": _describe(self.backward, self.recipe.backward_sources())},
                          sort_keys=True, indent=2)


def _config(graph, topology, spec):
    nodes = graph.topological_sort()
    config = allocate_runtime_config(1 + sum(node.task_num for node in nodes))
    config.num_workers = spec.num_vector_cores
    init_task_split_value(topology)
    for node in nodes:
        node.fill_config.fill(config, node, topology)
    config.task_num = sum(node.task_num for node in nodes)
    config.atomic_add_values[0] = 1
    return config


def _image(graph, config, spec, backward):
    _apply_mega_kernel_profile_graph(config, graph, _ProfileSpec(
        kernel_name="HyperMegaMhcGrad" if backward else "HyperMegaMhc", owner_label="TokenPartition"))
    def _enable(data):
        result = bytearray(data)
        struct.pack_into("<I", result, 20, 1)
        return bytes(result)
    runtime = _prepare_mega_kernel_runtime_config(
        config, tensor_factory=bytes, profile_tensor_factory=_enable, rank=0, device_id=0)
    queues = tuple(tuple(int(value) for value in queue[:count]) for queue, count in zip(
        (config.cube_task_indices, config.vector_task_indices, config.mix_task_indices), config.task_index_num[:3]))
    tasks = tuple((int(task.task_type), int(task.dependent_event), int(task.trigger_event),
                   int(task.task_index), int(task.task_split_value), int(task.extra_value_2))
                  for task in config.all_tasks[:config.task_num])
    stages, first = [], 0
    for node in graph.topological_sort():
        stages.append((node.name, first, node.task_num))
        first += node.task_num
    if ctypes.sizeof(config.grouped_matmul_group_list) != 512 * 8:
        raise ValueError("MHC requires the pinned fixed 512-entry scratch layout")
    return MhcRuntimeImage(runtime.normal_tensor, runtime.profile_tensor, queues,
                           (spec.num_cube_cores, spec.num_vector_cores, spec.num_cube_cores),
                           tasks, tuple(config.all_event_num_triggers), tuple(stages))


def compile_mhc_plan(recipe: MhcRecipe, spec: MhcSpec) -> MhcKernelPlan:
    """Lower the original token-tiled forward and backward family without device access.

    Args:
        recipe: Validated shifted five-output computation.
        spec: Static shape, physical topology, tile and backward contract.
    """
    recipe.validate_spec(spec)
    tile = resolve_token_tile(spec.token_count, spec.token_tile, spec.num_cube_cores)
    tile_count = (spec.token_count + tile - 1) // tile
    layout = build_event_layout(tile_count)
    if layout.final + 8 > 1024:
        raise ValueError("MHC legacy event boundary lacks atomic counter vector padding")
    graph, topology = recipe.build_graph(spec)
    forward = _image(graph, _config(graph, topology, spec), spec, False)
    backward, joins, grad_tile = None, (), resolve_grad_token_tile(spec.token_count, spec.grad_token_tile)
    if spec.need_backward:
        tiling = make_grad_tiling(spec.token_count, grad_tile)
        if build_grad_event_layout(tiling.tile_count, tiling.macro_count).final + 8 > 1024:
            raise ValueError("MHC backward event boundary lacks atomic counter vector padding")
        graph, topology = build_mega_mhc_grad_graph(
            spec.token_count, spec.hidden_size, grad_tile, num_vector_cores=spec.num_vector_cores)
        config = _config(graph, topology, spec)
        order_vector_tasks_by_stage(config)
        backward = _image(graph, config, spec, True)
        tiling = make_grad_tiling(spec.token_count, grad_tile)
        joins = tuple(tuple(index for index in range(tiling.tile_count) if tiling.macro_for_tile(index) == macro)
                      for macro in range(tiling.macro_count))
    plan = MhcKernelPlan(recipe, spec, forward, backward, tile, grad_tile,
                         min(tile_count, 4 * spec.num_cube_cores * 32 // tile), joins)
    plan.forward.simulate()
    if plan.backward is not None:
        plan.backward.simulate()
    return plan


def _describe(image, sources):
    if image is None:
        return None
    return {"bytes": len(image.normal), "queues": image.queues, "workers": image.workers,
            "stages": [{"stage": name, "first_task": first, "task_count": count, "source": str(sources.get(name, ""))}
                       for name, first, count in image.stages]}
