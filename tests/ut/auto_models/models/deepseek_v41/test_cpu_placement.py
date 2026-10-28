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
"""Verify scheduler CPU restrictions and independent mirrored treatment commands."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import deepseek_v41_cpu_abba as abba
from examples.training_demo.deepseek_v41 import deepseek_v41_cpu_worker as worker


class TestCpuPlacement(unittest.TestCase):
    """Reject overlapping CPU domains and protect the experimental controls."""

    def test_restricted_uneven_nodes_partition_without_overlap(self) -> None:
        """Use only scheduler-allowed CPUs even when nodes have unequal capacity."""
        nodes = {0: [2, 3], 2: [9, 10, 11, 12, 13, 14], 4: []}
        plan = worker.placement_plan(nodes, 4)
        cpus = [cpu for row in plan for cpu in row['cpus']]
        self.assertEqual(sorted(cpus), [2, 3, 9, 10, 11, 12, 13, 14])
        self.assertEqual(len(cpus), len(set(cpus)))
        for row in plan:
            self.assertTrue(row['cpus'])
            self.assertTrue(set(row['cpus']).issubset(nodes[row['numa_node']]))

    def test_insufficient_or_overlapping_cpus_fail(self) -> None:
        """Do not silently run multiple workers on the same restricted CPU."""
        for nodes, workers in [({0: [4]}, 2), ({0: [4], 1: [4]}, 2), ({}, 1)]:
            with self.assertRaises(ValueError):
                worker.placement_plan(nodes, workers)
        with self.assertRaises(ValueError):
            worker.parse_cpu_list('5-2')
        self.assertEqual(worker.parse_cpu_list('2-4,9,12-13'), [2, 3, 4, 9, 12, 13])

    def test_mirrored_commands_preserve_model_and_observer_controls(self) -> None:
        """Only CPU placement and MegaMoe vary across all eight fresh processes."""
        matrix = abba.case_matrix()
        treatments = [(case.placement, case.mega_kernels) for case in matrix]
        self.assertEqual(treatments, list(reversed(treatments)))
        self.assertEqual(len(set(treatments)), 4)
        env = {key: 'value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                       'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT')}
        env.update(MASTER_PORT_BASE='63500', NODE_RANK='7')
        with patch.dict(os.environ, env):
            commands = [abba.command_for(case, index, abba.Workload()) for index, case in enumerate(matrix)]
        for case, command in zip(matrix, commands):
            self.assertIn(f'DSV41_CPU_PLACEMENT={case.placement}', command)
            self.assertEqual(command[command.index('--steps') + 1], '100')
            self.assertEqual(command[command.index('--warmup-steps') + 1], '20')
            self.assertEqual(command[command.index('--experts') + 1], '384')
            self.assertEqual(command[command.index('--ep-size') + 1], '64')
            self.assertEqual(command[command.index('--num-hidden-layers') + 1], '8')
            self.assertNotIn('--moe-timing', command)
            self.assertNotIn('--capacity-factor', command)
            self.assertIn('examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker', command)
        self.assertEqual(len({command[-1] for command in commands}), 8)

    def test_worker_and_controller_import_without_frameworks(self) -> None:
        """The parent and worker preamble must not import device frameworks."""
        code = ('from examples.training_demo.deepseek_v41 import deepseek_v41_cpu_abba, '
                'deepseek_v41_cpu_worker; import sys; '
                'assert not {"torch", "torch_npu", "hyper_parallel"}.intersection(sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_real_worker_affinity_is_set_before_framework_import(self) -> None:
        """Exercise the actual process affinity without starting NPU runtimes."""
        with TemporaryDirectory() as directory:
            env = {**os.environ, 'DSV41_CPU_PLACEMENT': 'numa', 'DSV41_PLACEMENT_DRY_RUN': '1',
                   'LOCAL_RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'RANK': '0'}
            subprocess.run([sys.executable, '-m',
                            'examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker',
                            '--output', directory], env=env, check=True)
            record = json.loads((Path(directory) / 'rank0/cpu-placement.json').read_text())
            self.assertEqual(record['before']['affinity'], record['expected_affinity'])
            self.assertTrue(record['placement_before_framework_import'])
            self.assertFalse(record['npu_distance_used'])

    def test_profile_dispatch_happens_after_actual_cpu_binding(self) -> None:
        """Select the profile entry without importing a framework before CPU placement."""
        with TemporaryDirectory() as directory:
            env = {**os.environ, 'DSV41_CPU_PLACEMENT': 'numa', 'DSV41_PLACEMENT_DRY_RUN': '0',
                   'LOCAL_RANK': '0', 'LOCAL_WORLD_SIZE': '1', 'RANK': '0', 'CLEAN_PROFILE_TAIL': '1'}
            code = '''import json, os, sys
from pathlib import Path
from examples.training_demo.deepseek_v41 import deepseek_v41_cpu_worker as worker
sys.argv = ['worker', '--output', sys.argv[1]]
def execute(name, run_name):
    path = Path(sys.argv[2]) / 'rank0/cpu-placement.json'
    placement = json.loads(path.read_text())
    assert sorted(os.sched_getaffinity(0)) == placement['expected_affinity']
    assert not {'torch', 'torch_npu', 'hyper_parallel'}.intersection(sys.modules)
    path.with_name('dispatch.json').write_text(json.dumps({'module': name, 'run_name': run_name}))
worker.runpy.run_module = execute
worker.main()
'''
            subprocess.run([sys.executable, '-c', code, directory], env=env, check=True)
            record = json.loads((Path(directory) / 'rank0/dispatch.json').read_text())
            self.assertEqual(record['module'],
                             'examples.training_demo.deepseek_v41.benchmark_deepseek_v41_profile')
            self.assertEqual(record['run_name'], '__main__')

    def test_snapshot_allows_threads_that_finish_during_profiler_teardown(self) -> None:
        """Keep surviving-thread evidence when an enumerated task has already exited."""
        with patch.object(worker.Path, 'iterdir', return_value=[Path('42'), Path('43')]), \
                patch.object(worker.os, 'sched_getaffinity',
                             side_effect=[ProcessLookupError(), {2, 3}, {2, 3}]):
            snapshot = worker._snapshot()
        self.assertEqual(snapshot['thread_affinities'], {'43': [2, 3]})
        self.assertEqual(snapshot['affinity'], [2, 3])
