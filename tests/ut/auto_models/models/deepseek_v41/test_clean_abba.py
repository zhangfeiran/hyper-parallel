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
"""Protect mirrored clean comparisons, identity gates and observer cleanup."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from examples.training_demo.deepseek_v41 import deepseek_v41_clean_abba as clean
from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from examples.training_demo.deepseek_v41.benchmark_deepseek_v41_vlm import _BatchIdentity
from hyper_parallel.core.fully_shard.api import _extend_module_with_hsdp_interface
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts


class TestCleanAbba(unittest.TestCase):
    """Reject configuration drift and incomplete repeats before publishing ratios."""

    def tearDown(self) -> None:
        """Keep incremental publication caches isolated between cases."""
        clean._CASE_CACHE.clear()

    def test_module_counts_survive_expert_fsdp(self) -> None:
        """Count actual MegaMoe instances after FSDP dynamically extends their type."""
        model = torch.nn.Module()
        experts = torch.nn.Module()
        experts.__class__ = DeepseekV41MegaMoeExperts
        _extend_module_with_hsdp_interface(experts)
        model.experts = experts
        self.assertEqual(type(experts).__name__, 'HSDPDeepseekV41MegaMoeExperts')
        self.assertEqual(benchmark._mega_module_counts(model), {'mega_moe': 1, 'mega_mhc': 0, 'mega_gate': 0})

    def test_pull_has_separate_mirrored_commands(self) -> None:
        """Pull repeats use the same model and remain distinct from push in comparisons."""
        cases = clean.case_matrix(include_pull=True)
        self.assertEqual([case.backend for case in cases],
                         ['native', 'push', 'pull', 'push', 'push', 'pull', 'push', 'native'])
        self.assertEqual([case.mega_kernels for case in cases],
                         ['off', 'moe', 'moe', 'on', 'on', 'moe', 'moe', 'off'])
        env = {key: 'value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                       'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT')}
        env.update(MASTER_PORT_BASE='63100', NNODES='8', NODE_RANK='7', NUM_CARDS='16')
        with patch.dict('os.environ', env):
            commands = [clean.command_for(case, index, tail_profile=True) for index, case in enumerate(cases)]
        for case, command in zip(cases, commands):
            self.assertIn('CLEAN_PROFILE_TAIL=1', command)
            self.assertIn('examples.training_demo.deepseek_v41.benchmark_deepseek_v41_profile', command)
            self.assertEqual(command[command.index('--dispatch-mode')+1], case.dispatch_mode)
            self.assertNotIn('--moe-timing', command)
        self.assertEqual(len({command[-1] for command in commands}), 8)

    def test_push_pull_comparison_does_not_pool_both_transports(self) -> None:
        """Only the two requested transport repeats contribute to each side."""
        values = {'off': 10.0, 'moe': 8.0, 'moe_pull': 7.0, 'on': 6.0}
        cases = {case.name: {'status': 'complete', 'identity_sha256': 'same', 'hbm_eligible': True,
                             'measured_rank_max_seconds': [values[case.mode]] * 25}
                 for case in clean.case_matrix(include_pull=True)}
        comparison = clean._compare(cases, 'moe', 'moe_pull', include_pull=True)
        self.assertEqual(comparison['baseline_measured_steps'], 50)
        self.assertEqual(comparison['candidate_measured_steps'], 50)
        self.assertAlmostEqual(comparison['median_speed_ratio'], 8 / 7)

    def test_full_tail_publication_masks_profile_and_keeps_trace_coverage_separate(self) -> None:
        """Audit synthetic all-rank files through real summary and identity validation."""
        self._verify_tail_publication(False, 40, 10, 35, 37, [38, 39])
        clean._CASE_CACHE.clear()
        self._verify_tail_publication(True, 105, 20, 100, 102, [103, 104])
        clean._CASE_CACHE.clear()
        self._verify_tail_publication(True, 105, 20, 100, 102, [103, 104], include_pull=True)

    def _verify_tail_publication(self, numa_tail: bool, steps: int, warmup: int, clean_end: int,
                                 profile_warmup: int, profile_range: list[int],
                                 include_pull: bool | None = None) -> None:
        """Reject contaminated suffixes and bad affinity through actual artifact readers."""
        status = {'training_exit_status': 0, 'monitor_exit_status': 0, 'cleanup_ok': True, 'peer_failed': False}
        start = datetime(2026, 10, 19, tzinfo=timezone.utc)
        include_pull = not numa_tail if include_pull is None else include_pull
        matrix = clean.case_matrix(include_pull=include_pull)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for case in matrix:
                base = root/'cases'/case.name/'packed'
                for rank in range(2):
                    folder = base/case.backend/f'rank{rank}'
                    folder.mkdir(parents=True)
                    seconds = {'off': 10.0, 'moe': 8.0, 'moe_pull': 7.0, 'on': 6.0}[case.mode]
                    rows = [{'step': index+1, 'phase': 'warmup' if index < warmup else 'measured',
                             'local_seconds': seconds if index < clean_end else 1000.0,
                             'max_rank_seconds': seconds if index < clean_end else 1000.0,
                             'started_at': (start+timedelta(seconds=index*20)).isoformat(),
                             'finished_at': (start+timedelta(seconds=index*20+seconds)).isoformat(),
                             'loss': 12.0, 'grad_norm': 3.0, 'moe_input_rows': {'layer.mlp': 4096},
                             'replica_routes': {}, 'moe_timing': None,
                             'peak_allocated_bytes': 100 if index < clean_end else 900,
                             'peak_reserved_bytes': 200 if index < clean_end else 999,
                             'input_sha256': 'a'*64} for index in range(steps)]
                    evidence = {'profiler_level': 0, 'clean_steps_inclusive': [warmup + 1, clean_end],
                                'profile_steps_inclusive': profile_range, 'profiler_warmup_step': profile_warmup,
                                'profiling_enabled': rank == 0, 'trace_exported': rank == 0}
                    record = {'mega_kernels': case.mega_kernels, 'moe_timing': False,
                              'profiling_enabled': rank == 0, 'record_input_identity': True,
                              'mega_module_counts': {'mega_moe': 0 if case.mode == 'off' else 8,
                                                    'mega_mhc': 16 if case.mode == 'on' else 0,
                                                    'mega_gate': 8 if case.mode == 'on' else 0},
                              'num_hidden_layers': 8, 'experts': 384, 'ep_size': 64, 'edp_shard_size': 2,
                              'modality': 'text', 'packing_mode': 'packed', 'activation_checkpoint': 'full',
                              'attention_recompute_tier': 'all', 'engram_storage_backend': 'host',
                              'empty_cache_steps': 1, 'random_init_seed': 42, 'steps': steps, 'warmup_steps': warmup,
                              'rank': rank, 'world_size': 2, 'finite': True, 'steps_detail': rows,
                              'peak_allocated_bytes': 900, 'peak_reserved_bytes': 999,
                              'measurement': {'peak_allocated_bytes': 900, 'peak_reserved_bytes': 999},
                              'backend': 'owner_ep' if case.backend == 'native' else 'megamoe',
                              'dispatch_mode': case.dispatch_mode, 'capacity_factor': None,
                              'tail_profile': evidence, 'shmem_state': None}
                    for name, value in [('result.json', record), ('initialization.json', {'seed': 42}),
                                        ('profiling-config.json', evidence)]:
                        (folder/name).write_text(json.dumps(value))
                    (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                    if numa_tail:
                        cpus = [rank]
                        placement = {'mode': 'numa', 'rank': rank, 'local_rank': rank, 'local_workers': 2,
                                     'placement_before_framework_import': True, 'expected_affinity': cpus,
                                     'plan': [{'cpus': [0]}, {'cpus': [1]}], 'before': {'affinity': cpus},
                                     'after': {'affinity': cpus, 'thread_affinities': {'1': cpus}},
                                     'threads_with_affinity_outside_expected': {}}
                        (folder/'cpu-placement.json').write_text(json.dumps(placement))
                node = base/'nodes/node_0'/case.backend
                node.mkdir(parents=True)
                (node/'case_complete.json').write_text(json.dumps(status))
                (node/'monitor_exit_status.txt').write_text('0\n')
                (node/'hbm_summary.json').write_text(json.dumps(
                    {'monitoring_sufficient': True, 'foreign_pids': [], 'unresolved_pids': [],
                     'unhealthy_states': [], 'max_device_peak_mib': 4000}))
                samples = [{'time': (start+timedelta(seconds=second)).isoformat(), 'valid': True,
                            'devices': {'0': {'used_mib': 3000}}} for second in range(0, steps*20, 2)]
                (node/'hbm_samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in samples))
                trace = base.parent/'profiles'/case.backend/'rank0'
                trace.mkdir(parents=True)
                (trace/'trace_view.json').write_text('{}')
                (trace/'kernel_details.csv').write_text('Name\nsynthetic_fixture\n')
            options = {'include_pull': include_pull, 'tail_profile': True, 'numa_tail': numa_tail}
            report = clean.publish(root, 1, 2, **options)
            self.assertTrue(report['complete'])
            self.assertTrue(report['speedup_eligible'])
            self.assertAlmostEqual(report['comparisons']['off_vs_moe_only']['median_speed_ratio'], 10/8)
            self.assertEqual(report['comparisons']['off_vs_moe_only']['candidate_measured_steps'],
                             2 * (clean_end - warmup))
            if numa_tail:
                self.assertEqual(report['purpose'], f'numa_clean{len(matrix)}_tail_level0')
            if include_pull:
                self.assertAlmostEqual(report['comparisons']['off_vs_moe_pull']['median_speed_ratio'], 10/7)
                pull = report['comparisons']['moe_push_vs_pull']
                self.assertAlmostEqual(pull['median_speed_ratio'], 8/7)
                self.assertEqual(pull['candidate_measured_steps'], 2 * (clean_end - warmup))
            for case in report['cases'].values():
                self.assertEqual(case['measured_steps_inclusive'], [warmup + 1, clean_end])
                self.assertEqual(case['measured_peak_allocated_bytes'], 100)
                self.assertLess(case['mean_seconds'], 1000.0)
            (trace/'kernel_details.csv').unlink()
            report = clean.publish(root, 1, 2, **options)
            self.assertFalse(report['profiling_complete'])
            self.assertFalse(report['complete'])
            self.assertTrue(report['speedup_eligible'])
            if numa_tail:
                all_on = next(case for case in matrix if case.mega_kernels == 'on')
                bad = root/'cases'/all_on.name/'packed/push/rank1/cpu-placement.json'
                placement = json.loads(bad.read_text())
                placement['after']['thread_affinities'] = {'1': [0, 1]}
                bad.write_text(json.dumps(placement))
                clean._CASE_CACHE.clear()
                report = clean.publish(root, 1, 2, **options)
                self.assertEqual(report['cases'][all_on.name]['status'], 'incomplete_or_failed')
                self.assertFalse(report['comparisons']['off_vs_all_on']['eligible'])
                self.assertTrue(report['comparisons']['off_vs_moe_only']['eligible'])

    def test_fixed_numa_commands_keep_mirrored_modes_and_long_clean_window(self) -> None:
        """All modes bind before import and use an identical optimizer horizon."""
        env = {key: 'value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                       'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT')}
        env.update(MASTER_PORT_BASE='63900', NNODES='8', NODE_RANK='7', NUM_CARDS='16')
        with patch.dict('os.environ', env):
            with self.assertRaises(ValueError):
                clean.command_for(clean.case_matrix()[0], 0, numa_tail=True)
            for include_pull in (False, True):
                with self.subTest(include_pull=include_pull):
                    cases = clean.case_matrix(include_pull=include_pull)
                    commands = [clean.command_for(case, index, tail_profile=True, numa_tail=True)
                                for index, case in enumerate(cases)]
                    expected = ['off', 'moe', 'moe', 'on', 'on', 'moe', 'moe', 'off'] if include_pull else [
                        'off', 'moe', 'on', 'on', 'moe', 'off']
                    self.assertEqual([case.mega_kernels for case in cases], expected)
                    for case, command in zip(cases, commands):
                        self.assertIn('DSV41_CPU_PLACEMENT=numa', command)
                        self.assertIn('CLEAN_PROFILE_TAIL=1', command)
                        self.assertIn('CLEAN_NUMA_TAIL=1', command)
                        self.assertIn('examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker', command)
                        self.assertEqual(command[command.index('--steps')+1], '105')
                        self.assertEqual(command[command.index('--warmup-steps')+1], '20')
                        self.assertEqual(command[command.index('--dispatch-mode')+1], case.dispatch_mode)
                        self.assertNotIn('--moe-timing', command)
                    self.assertEqual(len({command[-1] for command in commands}), len(expected))

    def test_mirrored_modes_and_uninstrumented_commands(self) -> None:
        """Each mode has two fresh outputs and no event or profiler switches."""
        cases = clean.case_matrix()
        self.assertEqual([case.mega_kernels for case in cases], ['off', 'moe', 'on', 'on', 'moe', 'off'])
        env = {key: 'value' for key in ('HYPER_ROOT', 'SHMEM_HOST', 'MASTER_ADDR', 'MODEL_DIR',
                                       'ENGRAM', 'DATA', 'NATIVE_LIB', 'RUN_ROOT')}
        env.update(MASTER_PORT_BASE='63100', NNODES='8', NODE_RANK='7', NUM_CARDS='16')
        with patch.dict('os.environ', env):
            commands = [clean.command_for(case, index) for index, case in enumerate(cases)]
        self.assertEqual(len({command[-1] for command in commands}), 6)
        for case, command in zip(cases, commands):
            self.assertNotIn('--moe-timing', command)
            self.assertIn('--record-input-identity', command)
            self.assertEqual(command[command.index('--mega-kernels') + 1], case.mega_kernels)
            self.assertEqual(command[command.index('--backend') + 1],
                             'owner_ep' if case.mega_kernels == 'off' else 'megamoe')

    def test_pooling_precedes_median_and_identity_mismatch_rejects_ratio(self) -> None:
        """Pooling repeats must not become an average of their medians."""
        values = ([1, 2, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [4, 50, 100])
        cases = {case.name: {'status': 'complete', 'identity_sha256': 'same', 'hbm_eligible': True,
                             'measured_rank_max_seconds': times}
                 for case, times in zip(clean.case_matrix(), values)}
        comparison = clean._compare(cases, 'off', 'moe')
        self.assertEqual(comparison['baseline_median_seconds'], 3.5)
        self.assertEqual(comparison['candidate_median_seconds'], 1.5)
        self.assertAlmostEqual(comparison['median_speed_ratio'], 3.5 / 1.5)
        cases[clean.case_matrix()[4].name]['identity_sha256'] = 'different'
        comparison = clean._compare(cases, 'off', 'moe')
        self.assertFalse(comparison['eligible'])
        self.assertNotIn('median_speed_ratio', comparison)

    def test_bad_monitor_or_missing_repeat_cannot_publish_ratio(self) -> None:
        """A fast surviving run cannot replace both independent repeats."""
        cases = {case.name: {'status': 'complete', 'identity_sha256': 'same', 'hbm_eligible': True,
                             'measured_rank_max_seconds': [1.0]} for case in clean.case_matrix()}
        cases[clean.case_matrix()[4].name]['hbm_eligible'] = False
        self.assertFalse(clean._compare(cases, 'off', 'moe')['eligible'])
        cases[clean.case_matrix()[4].name] = {'status': 'incomplete_or_failed'}
        self.assertFalse(clean._compare(cases, 'off', 'moe')['eligible'])

    def test_effective_module_counts_are_required(self) -> None:
        """All-on arguments alone cannot prove that all replacements were installed."""
        case = clean.case_matrix()[2]
        record = {'mega_kernels': 'on', 'moe_timing': False, 'profiling_enabled': False,
                  'dispatch_mode': 'push', 'capacity_factor': None,
                  'record_input_identity': True, 'mega_module_counts': {'mega_moe': 8, 'mega_mhc': 16, 'mega_gate': 8},
                  'num_hidden_layers': 8, 'experts': 384, 'ep_size': 64, 'edp_shard_size': 2,
                  'modality': 'text', 'packing_mode': 'packed', 'activation_checkpoint': 'full',
                  'attention_recompute_tier': 'all', 'engram_storage_backend': 'host',
                  'empty_cache_steps': 1, 'random_init_seed': 42, 'steps': 40, 'warmup_steps': 10,
                  'steps_detail': [{'input_sha256': 'a' * 64} for _ in range(40)]}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'push/rank0'
            folder.mkdir(parents=True)
            (folder / 'initialization.json').write_text(json.dumps({'seed': 42}))
            path = folder / 'result.json'
            path.write_text(json.dumps(record))
            self.assertEqual(len(clean._case_identity(root, case, 1)), 64)
            record['mega_module_counts']['mega_gate'] = 0
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Unexpected workload or instrumentation'):
                clean._case_identity(root, case, 1)

    def test_failed_all_on_retains_valid_off_vs_moe(self) -> None:
        """A failed third mode does not hide a complete clean A/B comparison."""
        status = {'training_exit_status': 0, 'monitor_exit_status': 0, 'cleanup_ok': True, 'peer_failed': False}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for case in clean.case_matrix():
                if case.mega_kernels == 'on':
                    continue
                folder = root / 'cases' / case.name / 'packed/nodes/node_0' / case.backend
                folder.mkdir(parents=True)
                (folder / 'case_complete.json').write_text(json.dumps(status))
            result = {'status': 'complete', 'hbm_eligible': True, 'actual_token_min_max': [4096, 4096],
                      'measured_rank_max_seconds': [1.0], 'mean_seconds': 1.0, 'median_seconds': 1.0}
            with patch.object(clean.summary_base, '_summarize_case', return_value=(result, [])), \
                    patch.object(clean, '_case_identity', return_value='same'):
                report = clean.publish(root, 1, 1)
        self.assertFalse(report['complete'])
        self.assertTrue(report['comparisons']['off_vs_moe_only']['eligible'])
        self.assertFalse(report['comparisons']['off_vs_all_on']['eligible'])

    def test_batch_identity_has_no_gc_callback_and_restores_trainer(self) -> None:
        """Actual nested inputs remain comparable without adding GC observers."""
        @dataclass
        class Batch:
            """Represent the production dataclass container on CPU."""
            model_inputs: dict

        ids = torch.arange(4)
        def original(_iterator: Any) -> Batch:
            """Return fresh batch containers while preserving their tensor values."""
            return Batch({'input_ids': ids.clone()})
        base = SimpleNamespace(get_batch=original)
        callbacks = list(gc.callbacks)
        observer = _BatchIdentity(SimpleNamespace(base=base))
        try:
            base.get_batch(None)
            first = observer.snapshot()
            base.get_batch(None)
            self.assertEqual(first, observer.snapshot())
            ids[0] = 99
            base.get_batch(None)
            self.assertNotEqual(first, observer.snapshot())
            self.assertFalse(observer.batches)
            self.assertEqual(gc.callbacks, callbacks)
        finally:
            observer.close()
        self.assertIs(base.get_batch, original)


if __name__ == '__main__':
    unittest.main()
