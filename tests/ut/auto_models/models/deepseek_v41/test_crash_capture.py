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
"""Check diagnostic isolation, fresh-process capture and failed-case publication."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import deepseek_v41_crash_capture as capture
from examples.training_demo.deepseek_v41.deepseek_v41_capture_worker import capture_environment


class TestCrashCapture(unittest.TestCase):
    """Diagnostic commands must preserve the failed workload and report failures honestly."""

    def test_three_commands_preserve_scope_without_profiler(self) -> None:
        """Compare native, MegaMoe and all kernels with an identical optimizer horizon."""
        env = {key: '/value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                         'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT', 'DSV41_NATIVE_CAPTURE_LIB')}
        env.update(MASTER_PORT_BASE='64100', NNODES='8', NODE_RANK='7', NUM_CARDS='16')
        with patch.dict('os.environ', env):
            cases = capture.case_matrix()
            commands = [capture.capture_command(case, index) for index, case in enumerate(cases)]
        self.assertEqual([case.mega_kernels for case in cases], ['off', 'moe', 'on'])
        for case, command in zip(cases, commands):
            for key, value in {'--steps': '105', '--warmup-steps': '20', '--num-hidden-layers': '8',
                               '--ep-size': '64', '--edp-shard-size': '2', '--dispatch-mode': 'push',
                               '--mega-kernels': case.mega_kernels,
                               '--backend': 'owner_ep' if case.mega_kernels == 'off' else 'megamoe'}.items():
                self.assertEqual(command[command.index(key)+1], value)
            self.assertIn('DSV41_CPU_PLACEMENT=numa', command)
            self.assertIn('CLEAN_PROFILE_TAIL=0', command)
            self.assertIn('--use-fused-attention', command)
            self.assertIn('examples.training_demo.deepseek_v41.deepseek_v41_capture_worker', command)
            self.assertNotIn('--moe-timing', command)
        self.assertEqual(len({command[-1] for command in commands}), 3)
        self.assertIn('/native/', commands[0][-1] + '/')

    def test_pull_case_appends_without_changing_existing_controls(self) -> None:
        """Append a MoE-only pull run with the same workload and fused attention."""
        env = {key: '/value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                         'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT', 'DSV41_NATIVE_CAPTURE_LIB')}
        env.update(MASTER_PORT_BASE='64100', NNODES='8', NODE_RANK='7', NUM_CARDS='16')
        with patch.dict('os.environ', env):
            previous = [capture.capture_command(case, index) for index, case in enumerate(capture.case_matrix())]
            cases = capture.case_matrix(include_pull=True)
            commands = [capture.capture_command(case, index) for index, case in enumerate(cases)]
        self.assertEqual(commands[:3], previous)
        self.assertEqual([case.name for case in cases],
                         ['01_native', '02_moe_push', '03_all_on', '04_moe_pull'])
        pull = commands[-1]
        for key, value in {'--steps': '105', '--warmup-steps': '20', '--num-hidden-layers': '8',
                           '--experts': '384', '--ep-size': '64', '--edp-shard-size': '2',
                           '--dispatch-mode': 'pull', '--mega-kernels': 'moe', '--backend': 'megamoe'}.items():
            self.assertEqual(pull[pull.index(key) + 1], value)
        self.assertIn('--use-fused-attention', pull)
        self.assertNotIn('--capacity-factor', pull)
        self.assertTrue(pull[-1].endswith('/04_moe_pull/packed/pull'))

    def test_capture_environment_is_child_scoped(self) -> None:
        """Preserve existing preload libraries without mutating the parent environment."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            library = root/'capture.so'
            library.touch()
            env = {'DSV41_NATIVE_CAPTURE_LIB': str(library), 'LD_PRELOAD': '/existing.so'}
            result = capture_environment(root/'rank15', env)
            self.assertEqual(result['LD_PRELOAD'], f'{library}:/existing.so')
            self.assertEqual(env['LD_PRELOAD'], '/existing.so')
            self.assertEqual(result['DSV41_NATIVE_CAPTURE_LOG'], str(root/'rank15/native-stacks.log'))

    def test_failed_mode_preserves_capture_and_earlier_control(self) -> None:
        """A captured abort is observable and cannot be mistaken for training success."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for case in capture.case_matrix():
                base = root/'cases'/case.name/'packed'
                node = base/'nodes/node_0'/case.backend
                node.mkdir(parents=True)
                status = {'training_exit_status': 1 if case.mega_kernels == 'on' else 0, 'cleanup_ok': True}
                (node/'case_complete.json').write_text(json.dumps(status))
                folder = base/case.backend/'rank0'
                folder.mkdir(parents=True)
                (folder/'steps.jsonl').write_text('{}\n')
                if case.mega_kernels == 'on':
                    (folder/'native-stacks.log').write_text('DSV41_NATIVE_STACK_BEGIN\nDSV41_NATIVE_STACK_END\n')
            report = capture.publish(root, 1)
            self.assertTrue(report['collection_complete'])
            self.assertFalse(report['training_all_passed'])
            self.assertTrue(report['cases']['01_native']['training_passed'])
            self.assertTrue(report['cases']['02_moe_push']['training_passed'])
            self.assertTrue(report['cases']['03_all_on']['ranks']['0']['native_stack_complete'])
            self.assertFalse(report['profiler'])
