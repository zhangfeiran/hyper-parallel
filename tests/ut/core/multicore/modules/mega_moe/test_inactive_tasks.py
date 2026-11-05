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
"""Real runtime queue checks for unmapped guest suffix pruning."""

from collections import Counter
from types import SimpleNamespace
import struct
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import inactive_tasks
from hyper_parallel.core.multicore.modules.mega_moe import function as kernels
from hyper_parallel.core.multicore.diagnostics.common.call import DiagnosticCall, LaunchArguments
from hyper_parallel.core.multicore.diagnostics.dump.api import TensorDumpConfig, _TensorDumpCall
from hyper_parallel.core.multicore.diagnostics.dump import l1
from hyper_parallel.core.multicore.modules.mega_moe.inactive_tasks import (
    apply_guest_task_prefix, build_guest_task_prefixes, select_guest_task_prefix,
)
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts, _prepare_diagnostics
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskType
from hyper_parallel.core.multicore.scheduler.runtime import serialize_runtime_config


def apply_prefix(config: RuntimeConfigC, prefix: inactive_tasks.GuestTaskPrefix) -> RuntimeConfigC:
    """Apply exactly the byte ranges submitted to the native runtime image."""
    image = bytearray(serialize_runtime_config(config))
    for offset, payload in prefix.patches:
        image[offset:offset + len(payload)] = payload
    return type(config).from_buffer_copy(image)


class TestGuestTaskPrefixes(unittest.TestCase):
    """Remove only unused suffix work and preserve native physical dependencies."""

    def test_b1_unmapped_senders_omit_local_combine_with_shared_incoming_thresholds(self) -> None:
        """A remote B1 guest cannot force every sender to execute its empty slot."""
        ep, home, local = 4, 6, 7
        mappings = tuple(tuple(range(rank * home, (rank + 1) * home))
                         + ((0,) if rank == 1 else (-1,)) for rank in range(ep))
        for mode in ('push', 'pull'):
            for direction in (1, 3):
                changed, removed = [], []
                for rank in range(ep):
                    spec = MegaMoeSpec(1024, 128, 128, ep * local, 2, 1.25, 8192, ep, None, rank, 24,
                                       dispatch_mode=mode, replica_slots_per_rank=1, logical_num_experts=24)
                    config = _build_runtime_artifacts(spec)[direction]
                    prefixes = build_guest_task_prefixes(config, home_experts=home, ep_size=ep, dispatch_mode=mode,
                                                         rank_local_communication=True)
                    prefix = select_guest_task_prefix(prefixes, mappings[rank], all_slot_to_logical=mappings)
                    image = apply_prefix(config, prefix)
                    original = set(config.vector_task_indices[:config.task_index_num[1]])
                    retained = set(image.vector_task_indices[:image.task_index_num[1]])
                    combine = [config.all_tasks[task] for task in original - retained
                               if config.all_tasks[task].task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                               and config.all_tasks[task].inputs[3].input_position in (15, 16)]
                    self.assertEqual(bool(combine), rank != 1)
                    changed.append(image)
                    removed.append(len(combine))
                self.assertGreater(sum(removed), 0)
                signals = [Counter() for _ in range(ep)]
                events = [set() for _ in range(ep)]
                for rank, config in enumerate(changed):
                    for task_id in (*config.cube_task_indices[:config.task_index_num[0]],
                                    *config.vector_task_indices[:config.task_index_num[1]]):
                        task = config.all_tasks[task_id]
                        if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                            tiles = task.task_split_num // (ep * local)
                            target = task.task_index // (local * tiles)
                            signals[target][task.trigger_event] += 1
                            events[target].add(task.trigger_event)
                        else:
                            signals[rank][task.trigger_event] += 1
                            if task.task_type == TaskType.TASK_SHMEM_GET_MEM:
                                events[rank].add(task.trigger_event)
                for rank, counts in enumerate(signals):
                    for event in events[rank]:
                        self.assertEqual(changed[rank].all_event_num_triggers[event], counts[event],
                                         f'{mode=} {direction=} {rank=} {event=}')


    def test_other_senders_invalidate_cache_with_unchanged_local_and_global_prefix(self) -> None:
        """A changed incoming count must rewrite a reused image even for identical prefixes."""
        spec = MegaMoeSpec(1024, 128, 128, 28, 2, 1.25, 8192, 4, None, 0, 24,
                           replica_slots_per_rank=1, logical_num_experts=24)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6, ep_size=4, rank_local_communication=True)
        local = (0, 1, 2, 3, 4, 5, -1)
        base = torch.frombuffer(bytearray(serialize_runtime_config(config)), dtype=torch.uint8)
        image, cache, thresholds = base.clone(), {}, []
        with patch.object(inactive_tasks, '_copy_device_bytes', wraps=inactive_tasks._copy_device_bytes) as copy:
            for mapped_ranks in (1, 2, 3, 1):
                mappings = (local,) + tuple(tuple(range(rank * 6, (rank + 1) * 6))
                                           + ((0,) if rank <= mapped_ranks else (-1,)) for rank in range(1, 4))
                prefix = select_guest_task_prefix(prefixes, local, all_slot_to_logical=mappings)
                self.assertEqual((prefix.active_experts, prefix.communication_experts), (6, 7))
                self.assertEqual(prefix.incoming_combine, (4 - mapped_ranks) * prefix.dropped_combine // 4)
                calls = copy.call_count
                apply_guest_task_prefix(image, prefix, base=base, cache=cache)
                self.assertEqual(copy.call_count, calls + 4)
                self.assertEqual(image.numpy().tobytes(), bytes(apply_prefix(config, prefix)))
                thresholds.append(apply_prefix(config, prefix).all_event_num_triggers[prefix.combine_event])
                apply_guest_task_prefix(image, prefix, base=base, cache=cache)
                self.assertEqual(copy.call_count, calls + 4)
        self.assertLess(thresholds[0], thresholds[1])
        self.assertLess(thresholds[1], thresholds[2])
        self.assertEqual(thresholds[0], thresholds[3])


    def test_ep64_b1_incoming_counts_include_all_unmapped_senders(self) -> None:
        """Nonuniform EP64 B1 placement reduces receiver signals without changing mapped work."""
        ep, home, local, guests = 64, 6, 7, 10
        mappings = tuple(tuple(range(rank * home, (rank + 1) * home))
                         + (((rank + 1) % ep * home,) if rank < guests else (-1,)) for rank in range(ep))
        for mode in ('push', 'pull'):
            spec = MegaMoeSpec(4096, 128, 128, ep * local, 6, 1.25, 38400, ep, None, 17, 24,
                               dispatch_mode=mode, replica_slots_per_rank=1, logical_num_experts=384)
            artifacts = _build_runtime_artifacts(spec)
            for direction in (1, 3):
                config = artifacts[direction]
                prefixes = build_guest_task_prefixes(config, home_experts=home, ep_size=ep, dispatch_mode=mode,
                                                         rank_local_communication=True)
                prefix = select_guest_task_prefix(prefixes, mappings[17], all_slot_to_logical=mappings)
                changed = apply_prefix(config, prefix)
                self.assertGreater(prefix.dropped_combine, 0)
                self.assertEqual(prefix.incoming_combine, (ep - guests) * prefix.dropped_combine // ep)
                kept = set(changed.cube_task_indices[:changed.task_index_num[0]]) | set(
                    changed.vector_task_indices[:changed.task_index_num[1]])
                combine_tasks = 0
                local_signals = 0
                for task_id in kept:
                    task = config.all_tasks[task_id]
                    if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                        if task.inputs[3].input_position in (15, 16):
                            combine_tasks += 1
                    else:
                        local_signals += int(task.trigger_event == prefix.combine_event)
                        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
                            local_signals += int(task.extra_value_1 == prefix.combine_event)
                incoming = combine_tasks + prefix.dropped_combine - prefix.incoming_combine
                self.assertEqual(changed.all_event_num_triggers[prefix.combine_event], incoming + local_signals)


    def test_ep64_rotated_peer_rounds_remain_symmetric(self) -> None:
        """EP64 and 24 workers must remove whole rounds of both geometries."""
        spec = MegaMoeSpec(4096, 128, 128, 768, 6, 1.25, 38400, 64, None, 17, 24,
                           replica_slots_per_rank=6, logical_num_experts=384)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6, ep_size=64)
        mappings = tuple(tuple(range(rank * 6, (rank + 1) * 6)) + (0,) + (-1,) * 5 for rank in range(64))
        prefix = select_guest_task_prefix(prefixes, mappings[17], all_slot_to_logical=mappings)
        changed = apply_prefix(config, prefix)
        retained = set(changed.vector_task_indices[:changed.task_index_num[1]])
        groups = Counter()
        original = list(config.vector_task_indices[:config.task_index_num[1]])
        communication = 0
        for task_id in original:
            task = config.all_tasks[task_id]
            if task_id not in retained and task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                tiles = task.task_split_num // 768
                groups[task.inputs[3].input_position, task.task_index // tiles] += 1
                communication += 1
        self.assertGreater(communication, 0)
        for (stage, group), count in groups.items():
            self.assertTrue(all(groups[stage, group % 12 + peer * 12] == count for peer in range(64)))
        queue = list(changed.vector_task_indices[:changed.task_index_num[1]])
        positions = {task_id: index for index, task_id in enumerate(original)}
        self.assertTrue(all(positions[task_id] % 24 == index % 24 for index, task_id in enumerate(queue)))

    def test_communication_prefix_changes_restore_with_the_same_local_mapping(self) -> None:
        """A remote mapping change invalidates the body cache even for a fixed local prefix."""
        spec = MegaMoeSpec(1024, 128, 128, 48, 2, 1.25, 8192, 4, None, 0, 24,
                           replica_slots_per_rank=6, logical_num_experts=24)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6, ep_size=4)
        local = (0, 1, 2, 3, 4, 5) + (-1,) * 6
        base = torch.frombuffer(bytearray(serialize_runtime_config(config)), dtype=torch.uint8)
        image, cache = base.clone(), {}
        for common in (7, 9, 12, 6, 7):
            remote = (6, 7, 8, 9, 10, 11) + tuple(0 if slot == common - 1 else -1 for slot in range(6, 12))
            mappings = (local, remote, local, local)
            prefix = select_guest_task_prefix(prefixes, local, all_slot_to_logical=mappings)
            self.assertEqual(prefix.active_experts, 6)
            self.assertEqual(prefix.communication_experts, common)
            apply_guest_task_prefix(image, prefix, base=base, cache=cache)
            self.assertEqual(image.numpy().tobytes(), bytes(apply_prefix(config, prefix)))
            with patch.object(inactive_tasks, '_copy_device_bytes') as copy:
                apply_guest_task_prefix(image, prefix, base=base, cache=cache)
                copy.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Global'):
            select_guest_task_prefix(prefixes, local, all_slot_to_logical=(local,))


    def test_global_unmapped_suffix_removes_communication_with_matched_remote_thresholds(self) -> None:
        """Every retained remote signal matches the destination's completion threshold."""
        for mode in ('push', 'pull'):
            for cores in (1, 8, 24):
                ep, home, local = 4, 6, 12
                mappings = tuple(tuple(range(rank * home, (rank + 1) * home))
                                 + ((0, -1, -1, -1, -1, -1) if rank == 1 else (-1,) * 6)
                                 for rank in range(ep))
                graphs = []
                for rank in range(ep):
                    spec = MegaMoeSpec(1024, 128, 128, ep * local, 2, 1.25, 8192, ep, None, rank, cores,
                                       dispatch_mode=mode, replica_slots_per_rank=6, logical_num_experts=24)
                    _, forward, _, backward = _build_runtime_artifacts(spec)
                    graphs.append((forward, backward))
                for direction in (0, 1):
                    changed = []
                    for rank, pair in enumerate(graphs):
                        config = pair[direction]
                        prefixes = build_guest_task_prefixes(config, home_experts=home,
                                                              ep_size=ep, dispatch_mode=mode)
                        prefix = select_guest_task_prefix(prefixes, mappings[rank], all_slot_to_logical=mappings)
                        self.assertEqual(prefix.communication_experts, 7)
                        self.assertEqual(prefix.active_experts, 7 if rank == 1 else 6)
                        changed.append(apply_prefix(config, prefix))
                        original = list(config.vector_task_indices[:config.task_index_num[1]])
                        retained = list(changed[-1].vector_task_indices[:changed[-1].task_index_num[1]])
                        positions = {task_id: index for index, task_id in enumerate(original)}
                        self.assertTrue(all(positions[task_id] % cores == index % cores
                                            for index, task_id in enumerate(retained)))
                        removed = [config.all_tasks[task_id] for task_id in set(original) - set(retained)]
                        communication = [task for task in removed if task.task_type in (
                            TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM)]
                        self.assertGreater(len(communication), 0)
                        for task in communication:
                            tiles = task.task_split_num // (ep * local)
                            self.assertGreaterEqual((task.task_index // tiles) % local, 7)
                    signals = [Counter() for _ in range(ep)]
                    communication_events = [set() for _ in range(ep)]
                    for rank, config in enumerate(changed):
                        for task_id in (*config.cube_task_indices[:config.task_index_num[0]],
                                        *config.vector_task_indices[:config.task_index_num[1]]):
                            task = config.all_tasks[task_id]
                            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                                tiles = task.task_split_num // (ep * local)
                                target = task.task_index // (local * tiles)
                                signals[target][task.trigger_event] += 1
                                communication_events[target].add(task.trigger_event)
                            else:
                                signals[rank][task.trigger_event] += 1
                                if task.task_type == TaskType.TASK_SHMEM_GET_MEM:
                                    communication_events[rank].add(task.trigger_event)
                    for rank, counts in enumerate(signals):
                        for event in communication_events[rank]:
                            total = counts[event]
                            self.assertEqual(changed[rank].all_event_num_triggers[event], total,
                                             f'{mode=} {cores=} {direction=} {rank=} {event=} {total=}')

    def test_real_graphs_preserve_workers_descriptors_and_event_accounting(self) -> None:
        """Each retained task stays on its original worker with identical operands."""
        for mode in ('push', 'pull'):
            for cores, budget, tokens in ((1, 1, 128), (8, 2, 128), (24, 6, 1024)):
                spec = MegaMoeSpec(tokens, 128, 128, 4 * (6 + budget), 2, 1.25, 8192, 4, None, 0, cores,
                                   dispatch_mode=mode, replica_slots_per_rank=budget, logical_num_experts=24,
                                   replica_transport='shmem_signal_kernel_gradient_bands')
                _, forward, _, backward = _build_runtime_artifacts(spec)
                for config in (forward, backward):
                    before = bytes(config)
                    prefixes = build_guest_task_prefixes(config, home_experts=6)
                    for active in range(6, spec.local_experts + 1):
                        with self.subTest(mode=mode, cores=cores, budget=budget,
                                          backward=config is backward, active=active):
                            prefix = prefixes[active - 6]
                            changed = apply_prefix(config, prefix)
                            self.assertEqual(bytes(config), before)
                            self.assertEqual(changed.task_capacity, config.task_capacity)
                            self.assertEqual(bytes(changed.dynamic_data), bytes(config.dynamic_data))
                            self.assertEqual(bytes(changed.all_tasks), bytes(config.all_tasks))
                            self.assertEqual(bytes(changed.all_events), bytes(config.all_events))
                            self.assertEqual(changed.ready_event, config.ready_event)
                            self.assertEqual(changed.completion_event, config.completion_event)
                            removed = []
                            for queue_name, count_index in (('cube_task_indices', 0), ('vector_task_indices', 1)):
                                original = list(getattr(config, queue_name)[:config.task_index_num[count_index]])
                                retained = list(getattr(changed, queue_name)[:changed.task_index_num[count_index]])
                                positions = {task_id: index for index, task_id in enumerate(original)}
                                keep = set(retained)
                                self.assertEqual(retained, [task_id for task_id in original if task_id in keep])
                                self.assertTrue(all(positions[task_id] % cores == index % cores
                                                    for index, task_id in enumerate(retained)))
                                removed.extend(config.all_tasks[task_id] for task_id in original if task_id not in keep)
                            reductions = Counter()
                            for task in removed:
                                self.assertIn(task.task_type, (TaskType.TASK_GROUPED_MATMUL,
                                                              TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD))
                                expert = task.task_index // (task.task_split_num // spec.local_experts)
                                self.assertGreaterEqual(expert, active)
                                reductions[task.trigger_event] += 1
                                if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
                                    reductions[task.extra_value_1] += 1
                            for event, needed in enumerate(config.all_event_num_triggers):
                                self.assertEqual(changed.all_event_num_triggers[event], needed - reductions[event])
                                self.assertGreaterEqual(changed.all_event_num_triggers[event], 0)
                            self.assertEqual(prefix.dropped_cube + prefix.dropped_vector, len(removed))
                            if active == spec.local_experts:
                                self.assertEqual(bytes(changed), before)
                            elif cores == 24:
                                self.assertGreater(prefix.dropped_cube, 0)
                                self.assertGreater(prefix.dropped_vector, 0)

    def test_mapped_empty_guests_and_holes_keep_physical_slot_identity(self) -> None:
        """Mapped slots remain scheduled independently of their current row counts."""
        spec = MegaMoeSpec(128, 128, 128, 12, 2, 1.25, 512, 1, None, 0, 8,
                           replica_slots_per_rank=6, logical_num_experts=6)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6)
        self.assertEqual(select_guest_task_prefix(prefixes, (0, 1, 2, 3, 4, 5, -1, -1, 2, -1, -1, -1))
                         .active_experts, 9)
        self.assertEqual(select_guest_task_prefix(prefixes, (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1))
                         .active_experts, 6)
        self.assertIs(select_guest_task_prefix(prefixes, None), prefixes[-1])
        with self.assertRaisesRegex(ValueError, 'physical'):
            select_guest_task_prefix(prefixes, (0, 1))
        with self.assertRaisesRegex(ValueError, 'home'):
            build_guest_task_prefixes(config, home_experts=0)

    def test_prefix_changes_restore_body_and_rebuilt_images_do_not_hit_stale_cache(self) -> None:
        """Serial F/R/B images reuse only their own exact prefix and preserve the base."""
        spec = MegaMoeSpec(1024, 128, 128, 12, 2, 1.25, 4096, 1, None, 0, 8,
                           replica_slots_per_rank=6, logical_num_experts=6)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6)
        original = serialize_runtime_config(config)
        base = torch.frombuffer(bytearray(original), dtype=torch.uint8)
        suffix = torch.tensor([213, 214, 215], dtype=torch.uint8)
        image = torch.cat((base, suffix))
        cache = {}
        with patch.object(inactive_tasks, '_copy_device_bytes', wraps=inactive_tasks._copy_device_bytes) as copy:
            for active in (7, 9, 6, 12, 7):
                prefix = prefixes[active - 6]
                self.assertIs(apply_guest_task_prefix(image, prefix, base=base, cache=cache), image)
                expected = serialize_runtime_config(apply_prefix(config, prefix))
                self.assertEqual(image[:base.numel()].numpy().tobytes(), expected)
                self.assertEqual(base.numpy().tobytes(), original)
                torch.testing.assert_close(image[-3:], suffix)
                calls = copy.call_count
                apply_guest_task_prefix(image, prefix, base=base, cache=cache)
                self.assertEqual(copy.call_count, calls)
            replacement = torch.cat((base, suffix))
            apply_guest_task_prefix(replacement, prefixes[1], base=base, cache=cache)
            self.assertEqual(copy.call_count, calls + 4)
            self.assertEqual(replacement.numpy().tobytes(), image.numpy().tobytes())
            cache.clear()
            apply_guest_task_prefix(replacement, prefixes[1], base=base, cache=cache)
            self.assertEqual(copy.call_count, calls + 8)
        with self.assertRaisesRegex(ValueError, 'exclusive'):
            apply_guest_task_prefix(base, prefixes[0], base=base)

    def test_partial_copy_failure_invalidates_the_previous_prefix(self) -> None:
        """A failed 7-to-9 upload cannot make a later 7 retry skip restoration."""
        spec = MegaMoeSpec(1024, 128, 128, 12, 2, 1.25, 4096, 1, None, 0, 8,
                           replica_slots_per_rank=6, logical_num_experts=6)
        config = _build_runtime_artifacts(spec)[1]
        prefixes = build_guest_task_prefixes(config, home_experts=6)
        base = torch.frombuffer(bytearray(serialize_runtime_config(config)), dtype=torch.uint8)
        runtime = base.clone()
        cache = {}
        apply_guest_task_prefix(runtime, prefixes[1], base=base, cache=cache)
        expected = runtime.clone()
        original = inactive_tasks._copy_device_bytes
        submissions = 0

        def copy(data: bytes, destination: torch.Tensor) -> None:
            """Fail after one real partial update without swallowing the exception."""
            nonlocal submissions
            submissions += 1
            if submissions == 2:
                raise RuntimeError('Injected metadata submission failure')
            original(data, destination)

        with patch.object(inactive_tasks, '_copy_device_bytes', side_effect=copy):
            with self.assertRaisesRegex(RuntimeError, 'Injected metadata'):
                apply_guest_task_prefix(runtime, prefixes[3], base=base, cache=cache)
        self.assertNotIn(id(base), cache)
        self.assertFalse(torch.equal(runtime, expected))
        with patch.object(inactive_tasks, '_copy_device_bytes', wraps=original) as restore:
            apply_guest_task_prefix(runtime, prefixes[1], base=base, cache=cache)
        self.assertEqual(restore.call_count, 4)
        torch.testing.assert_close(runtime, expected)

    def test_l1_keeps_requested_padding_tasks_and_profile_retains_pruning(self) -> None:
        """A real L1 record plan remains scheduled through the native launch boundary."""
        spec = MegaMoeSpec(1024, 128, 128, 12, 2, 1.25, 4096, 1, None, 0, 8,
                           replica_slots_per_rank=6, logical_num_experts=6)
        graph, config, bgraph, bconfig = _build_runtime_artifacts(spec)
        forward, _ = _prepare_diagnostics(spec, graph, config, bgraph, bconfig, 'cpu')
        parameters = [torch.zeros(1, dtype=torch.bfloat16) for _ in range(16)]
        parameters[6] = torch.zeros(12, dtype=torch.int64)
        dump_config = TensorDumpConfig('/tmp/unused-dump-output', level='L1', stages='SwiGLU', io='input', owner_ids=7)
        prepared = l1._prepare_l1_dump(forward._runtime, graph, parameters, dump_config, rank=0, device='cpu')
        prefix = build_guest_task_prefixes(config, home_experts=6)[1]
        expected = {record.task_id for record in prepared.records}
        self.assertTrue(expected)
        metadata = SimpleNamespace(group_list=parameters[6])
        for name in ('dispatch_target_off', 'dispatch_src_off', 'dispatch_size',
                     'combine_target_off', 'combine_src_off', 'combine_size'):
            setattr(metadata, name, torch.zeros(1, dtype=torch.int64))
        arguments = LaunchArguments(prepared.runtime_config, torch.zeros(16), prepared.buffer, torch.zeros(16))

        def image(base: torch.Tensor, data: bytes, *_args, **_kwargs) -> torch.Tensor:
            """Emulate only the current-stream metadata append on CPU storage."""
            return torch.cat((base, torch.frombuffer(bytearray(data), dtype=torch.uint8)))

        for preserves in (True, False):
            with self.subTest(preserves=preserves):
                call = (_TensorDumpCall(prepared_l1=prepared, arguments=arguments) if preserves
                        else DiagnosticCall(arguments))
                self.assertEqual(call.preserves_task_queues, preserves)
                plan = SimpleNamespace(spec=spec, fwd_runtime_config=forward.runtime_config,
                    fwd_diagnostics=SimpleNamespace(is_active=True, launch=lambda *_a, **_kw: call),
                    up_proj_tiling=None, swiglu_tiling=None, down_proj_tiling=None)
                execution = kernels._ForwardExecution(
                    dispatch=parameters[0], source=parameters[0], combine=parameters[0],
                    event_counters=torch.zeros(16), gmm_workspace=parameters[0],
                    intermediates=kernels._ForwardIntermediates(parameters[0], parameters[0], parameters[0]),
                    task_prefix=prefix)
                with (patch.object(kernels, '_runtime_image', side_effect=image),
                      patch.object(kernels.multicore_ops, 'mega_moe_with_diagnostic_buffer') as native):
                    kernels._launch_forward_kernel(plan, metadata, parameters[0],
                                                   torch.zeros(6, 1), torch.zeros(6, 1), execution)
                submitted = native.call_args.args[20]
                actual = type(config).from_buffer_copy(submitted[:prepared.runtime_config.numel()].numpy().tobytes())
                scheduled = set(actual.cube_task_indices[:actual.task_index_num[0]]) | set(
                    actual.vector_task_indices[:actual.task_index_num[1]])
                if preserves:
                    self.assertTrue(expected <= scheduled)
                else:
                    self.assertTrue(expected.isdisjoint(scheduled))
                # Simulate status writes only for tasks that the submitted queue can execute.
                for record in prepared.records:
                    status = l1.L1_DUMP_STATUS_CAPTURED if record.task_id in scheduled else 0
                    start = record.record_offset + 7 * 4
                    prepared.buffer[start:start + 4] = torch.tensor(list(struct.pack('<I', status)), dtype=torch.uint8)
                if preserves:
                    self.assertEqual(len(l1._decode_l1_dump(prepared)), len(prepared.records))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'record was not captured'):
                        l1._decode_l1_dump(prepared)


if __name__ == '__main__':
    unittest.main()
