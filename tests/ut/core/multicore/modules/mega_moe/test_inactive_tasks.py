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
