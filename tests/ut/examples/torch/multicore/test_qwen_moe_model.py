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
"""CPU-only construction and input contracts for the optional full Qwen example."""

import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from examples.torch.multicore.mega_moe.qwen_moe_config import QwenMoeConfig
from examples.torch.multicore.mega_moe.qwen_moe_config import make_config, parse_args

# Only this model integration file requires the example's optional packages;
# configuration and namespace tests run without either dependency installed.
pytest.importorskip('transformers')
pytest.importorskip('torch_npu')

qwen_model = importlib.import_module('examples.torch.multicore.mega_moe.qwen_moe_model')
qwen_benchmark = importlib.import_module('examples.torch.multicore.mega_moe.qwen_moe_benchmark')


class _PositionRecorder(torch.nn.Module):
    """Inspect decoder inputs without invoking any NPU operator."""

    def __init__(self) -> None:
        """Retain only the last decoder input positions."""
        super().__init__()
        self.positions = None

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Keep the hidden states intact while retaining the positions for an assertion."""
        self.positions = positions
        return hidden


class TestQwenMoeModel(unittest.TestCase):
    """Exercise the actual CPU model factory without distributed setup or native execution."""

    @staticmethod
    def _config() -> QwenMoeConfig:
        return replace(QwenMoeConfig(), hidden_size=64, intermediate_size=32, num_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, shared_expert_intermediate_size=32,
                       max_seq_len=128, local_num_tokens=128, vocab_size=64,
                       replica_slots_per_rank=1, shared_physical_weights=True)

    def test_layers_share_resources_but_not_parameters(self) -> None:
        """The second decoder must join the same B resource group without aliasing home weights."""
        model = qwen_model.QwenMoeModel(self._config())
        first, second = (layer.mlp.experts for layer in model.layers)
        self.assertTrue(first.shared_physical_weights)
        self.assertTrue(second.shared_physical_weights)
        self.assertIs(first._resource_group, second._resource_group)
        self.assertNotEqual(first.gate_up_weight.data_ptr(), second.gate_up_weight.data_ptr())
        self.assertNotEqual(first.down_weight.data_ptr(), second.down_weight.data_ptr())
        model.close()

    def test_rotary_positions_keep_batch_and_sequence_axes(self) -> None:
        """RoPE must receive [batch, sequence], including more than one batch row."""
        model = qwen_model.QwenMoeModel(self._config())
        decoder = _PositionRecorder()
        with patch.object(model, 'layers', torch.nn.ModuleList([decoder])), \
                patch.object(model, 'norm', torch.nn.Identity()):
            output = model(torch.zeros((2, 64), dtype=torch.long))
        self.assertEqual(tuple(output['logits'].shape), (2, 64, 64))
        self.assertEqual(tuple(decoder.positions.shape), (2, 64))
        torch.testing.assert_close(decoder.positions, torch.arange(64).expand(2, -1))
        model.close()


class TestQwenBackendModel(unittest.TestCase):
    """Protect backend initialization and default dtype restoration."""

    @staticmethod
    def _shard_grouped(module: torch.nn.Module, _mesh: object) -> None:
        """Emulate rank-zero expert sharding without initializing a process group."""
        for name in ('w1', 'w2', 'w3'):
            parameter = getattr(module, name)
            setattr(module, name, torch.nn.Parameter(parameter[:parameter.shape[0] // 8].detach().clone()))

    def _common_model(self) -> torch.nn.Module:
        config = TestQwenMoeModel._config()
        with patch.object(qwen_model, 'init_device_mesh', return_value=object()), \
                patch.object(qwen_model.ExpertParallel, 'apply', side_effect=self._shard_grouped):
            model = qwen_model.build_qwen_moe_model(config, torch.device('cpu'), backend='common')
        self.addCleanup(model.close)
        return model

    def test_factory_preserves_initial_weights_and_dtype(self) -> None:
        """Conversion must happen after dense initialization, preserving the original RNG order."""
        config = TestQwenMoeModel._config()
        default_dtype = torch.get_default_dtype()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            mega = qwen_model.build_qwen_moe_model(config, torch.device('cpu'))
            self.addCleanup(mega.close)
            torch.manual_seed(42)
            common = self._common_model()
        self.assertEqual(torch.get_default_dtype(), default_dtype)
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in common.parameters()))
        self.assertEqual(common.layers[0].self_attn.rotary_emb.inv_freq.dtype, torch.float32)
        expected = dict(mega.named_canonical_tensors())
        actual = dict(common.named_canonical_tensors())
        self.assertEqual(list(actual), list(expected))
        self.assertEqual(len(actual), 23)
        for name, tensor in actual.items():
            with self.subTest(name=name):
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        self.assertTrue(all(isinstance(layer.mlp.experts, qwen_model.NativeMoeExperts) for layer in common.layers))
        self.assertEqual(len(common.expert_parameters()), 6)
        with patch.object(qwen_model.MegaMoeExperts, 'close') as close_mega:
            common.close()
            close_mega.assert_not_called()

    def test_factory_rejects_backend_before_construction_and_restores_dtype_on_error(self) -> None:
        """Invalid backends and failed model construction cannot leak a new default dtype."""
        config = TestQwenMoeModel._config()
        default_dtype = torch.get_default_dtype()
        with patch.object(qwen_model, 'QwenMoeModel', side_effect=RuntimeError('construction failed')) as construct:
            with self.assertRaisesRegex(ValueError, 'backend'):
                qwen_model.build_qwen_moe_model(config, torch.device('cpu'), backend='invalid')
            construct.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                qwen_model.build_qwen_moe_model(config, torch.device('cpu'))
        self.assertEqual(torch.get_default_dtype(), default_dtype)


class TestExpertUpdateProbe(unittest.TestCase):
    """Keep optimizer validation meaningful when skew leaves an expert or owner cold."""

    def test_later_expert_update_is_visible(self) -> None:
        """A cold first expert cannot hide a later expert's parameter update."""
        parameter = torch.zeros((2, 5000))
        before = qwen_benchmark._expert_update_sample(parameter).clone()
        parameter[1, 0] = 1
        gradients = {'experts.0.gate_up_weight': torch.ones_like(parameter)}
        with patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                patch.object(qwen_benchmark, '_rank_max', side_effect=lambda value, _: value):
            updated = qwen_benchmark._expert_parameters_updated([(parameter, before)], gradients, torch.device('cpu'))
        self.assertTrue(updated)
        self.assertEqual(tuple(before.shape), (2, 4096))

    def test_cold_owner_needs_a_global_update(self) -> None:
        """An unchanged cold owner passes only when another owner actually updates."""
        parameter = torch.zeros((2, 4))
        probes = [(parameter, parameter.clone())]
        gradients = {'experts.0.gate_up_weight': torch.zeros_like(parameter),
                     'dense.weight': torch.ones_like(parameter)}
        for global_update in (0.0, 1.0):
            with self.subTest(global_update=global_update), \
                    patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                    patch.object(qwen_benchmark, '_rank_max', return_value=global_update) as rank_max:
                updated = qwen_benchmark._expert_parameters_updated(probes, gradients, torch.device('cpu'))
                self.assertEqual(updated, bool(global_update))
                rank_max.assert_called_once_with(0.0, torch.device('cpu'))

    def test_active_owner_without_update_still_fails(self) -> None:
        """Another owner's update cannot mask an unchanged active owner."""
        parameter = torch.zeros((2, 4))
        gradients = {'experts.0.gate_up_weight': torch.ones_like(parameter)}
        with patch.object(qwen_benchmark, '_all_ranks_true', side_effect=lambda value, _: value), \
                patch.object(qwen_benchmark, '_rank_max', return_value=1.0) as rank_max:
            updated = qwen_benchmark._expert_parameters_updated(
                [(parameter, parameter.clone())], gradients, torch.device('cpu'),
            )
        self.assertFalse(updated)
        rank_max.assert_called_once_with(0.0, torch.device('cpu'))


class TestSingleBackendSnapshots(unittest.TestCase):
    """Mock collectives while exercising real CPU snapshot coordination and ownership."""

    @staticmethod
    def _gather(destination, local) -> None:
        """Emulate agreement with an identical peer, without distributed initialization."""
        destination[:] = [local] * len(destination)

    def test_incomplete_reference_fails_on_every_rank(self) -> None:
        """A directory with no completion marker is not an accepted baseline."""
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--backend', 'mega_moe', '--reference-dir', directory])
            snapshots = qwen_benchmark._AccuracySnapshots(args, make_config(args), 0, 2)
            with patch.object(qwen_benchmark.dist, 'all_gather_object', side_effect=self._gather), \
                    self.assertRaisesRegex(RuntimeError, 'snapshot errors by rank'):
                snapshots.phase('inputs', iter([('input_ids', torch.ones(2, dtype=torch.int64))]))

    def test_round_trip_checks_every_phase_before_completion(self) -> None:
        """A complete CPU reference supports a fresh process without preserving device tensors."""
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(qwen_benchmark.dist, 'all_gather_object', side_effect=self._gather):
            args = parse_args(['--backend', 'common', '--snapshot-dir', directory])
            writer = qwen_benchmark._AccuracySnapshots(args, make_config(args), 0, 2)
            for phase in qwen_benchmark._SNAPSHOT_PHASES:
                writer.phase(phase, iter([('value', torch.ones(2))]))
            self.assertFalse((Path(directory) / 'rank0/complete.json').exists())
            writer.finish()
            args = parse_args(['--backend', 'mega_moe', '--reference-dir', directory])
            reader = qwen_benchmark._AccuracySnapshots(args, make_config(args), 0, 2)
            for phase in qwen_benchmark._SNAPSHOT_PHASES:
                reader.phase(phase, iter([('value', torch.ones(2))]))
                self.assertTrue(reader.results[phase]['passed'])
            reader.finish()

    def test_peer_io_error_is_not_masked_by_local_success(self) -> None:
        """A rank-local filesystem failure must not let the other ranks continue the model."""
        def gather_with_error(destination: list, local: dict) -> None:
            """Report that the peer could not write its snapshot."""
            destination[:] = [local, {'error': 'disk full'}]

        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--backend', 'common', '--snapshot-dir', directory])
            writer = qwen_benchmark._AccuracySnapshots(args, make_config(args), 0, 2)
            with patch.object(qwen_benchmark.dist, 'all_gather_object', side_effect=gather_with_error), \
                    self.assertRaisesRegex(RuntimeError, 'disk full'):
                writer.phase('inputs', iter([('input_ids', torch.ones(2, dtype=torch.int64))]))
            self.assertFalse((Path(directory) / 'rank0/complete.json').exists())

    def test_cpu_gradient_finiteness_uses_only_scalar_collectives(self) -> None:
        """CPU snapshots must still participate in the original all-rank finite-gradient gate."""
        with patch.object(qwen_benchmark.dist, 'all_reduce') as reduce:
            self.assertTrue(qwen_benchmark._all_gradients_finite({'weight': torch.ones(3)}, torch.device('cpu')))
            self.assertFalse(qwen_benchmark._all_gradients_finite({'weight': torch.tensor([float('nan')])},
                                                               torch.device('cpu')))
        self.assertTrue(all(call.args[0].numel() == 1 for call in reduce.call_args_list))

    def test_memory_window_excludes_warmup_and_counts_external_heap_once(self) -> None:
        """Separate allocator peaks, the external heap, and a final device-wide sample."""
        args = parse_args(['--warmup-steps', '2', '--measured-steps', '2'])
        workload = MagicMock(device=torch.device('cpu'), world_size=8)
        events = []

        def timed_step(_workload: object) -> tuple[torch.Tensor, float]:
            """Record each synthetic optimizer step without running a device operation."""
            events.append('step')
            return torch.ones(()), 3.0

        with patch.object(qwen_benchmark, '_timed_step', side_effect=timed_step), \
                patch.object(qwen_benchmark, '_symmetric_heap_bytes', return_value=50, create=True), \
                patch.object(qwen_benchmark.torch.npu, 'synchronize'), \
                patch.object(qwen_benchmark.torch.npu, 'reset_peak_memory_stats',
                             side_effect=lambda _: events.append('reset')), \
                patch.object(qwen_benchmark.torch.npu, 'max_memory_allocated', return_value=100), \
                patch.object(qwen_benchmark.torch.npu, 'max_memory_reserved', return_value=120), \
                patch.object(qwen_benchmark.torch.npu, 'memory_allocated', return_value=90), \
                patch.object(qwen_benchmark.torch.npu, 'memory_reserved', return_value=110), \
                patch.object(qwen_benchmark.torch.npu, 'mem_get_info', return_value=(300, 500)), \
                patch.object(qwen_benchmark.dist, 'all_gather_object', side_effect=self._gather):
            result = qwen_benchmark._measure_backend(workload, {}, 10.0, args, collect_memory=True)
        self.assertEqual(events, ['step', 'reset', 'step', 'step'])
        memory = result['steady_state_memory_per_rank']
        self.assertEqual(len(memory), 8)
        self.assertEqual(memory[0]['symmetric_heap_bytes'], 50)
        self.assertEqual(memory[0]['conservative_total_peak_bytes'], 150)
        self.assertEqual(memory[0]['final_device_used_bytes'], 200)
        self.assertEqual(memory[0]['final_device_total_bytes'], 500)

    def test_single_backend_builds_one_model_and_closes_it(self) -> None:
        """The worker cannot retain the paired path's second model or device gradient snapshots."""
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(['--backend', 'mega_moe', '--snapshot-dir', directory,
                               '--output', directory + '/run.json'])
            config = make_config(args)
            model, workload = MagicMock(), MagicMock()
            snapshots = MagicMock(contract={'optimizer': {}}, results={}, reference=False)
            cpu = torch.device('cpu')
            first_step = ({'loss_finite': True}, 1.0, torch.ones(()), torch.ones((1, 2)), {'weight': torch.ones(2)})
            with patch.object(qwen_benchmark, '_runtime_identity', return_value={}), \
                    patch.object(qwen_benchmark, '_AccuracySnapshots', return_value=snapshots), \
                    patch.object(qwen_benchmark, '_build_batch', return_value=(torch.ones(2), torch.ones(2))), \
                    patch.object(qwen_benchmark, '_build_backend_model', return_value=model) as build, \
                    patch.object(qwen_benchmark, '_build_workload', return_value=workload) as attach, \
                    patch.object(qwen_benchmark, '_validate_first_step', return_value=first_step) as validate, \
                    patch.object(qwen_benchmark, '_measure_backend', return_value={}), \
                    patch.object(qwen_benchmark, '_resident_training_ledger',
                                 return_value={'resident_backend_count': 1}), \
                    patch.object(qwen_benchmark, '_shared_pool_ledger', return_value={}), \
                    patch.object(qwen_benchmark, '_all_ranks_true', return_value=True), \
                    patch.object(qwen_benchmark.dist, 'all_gather_object', side_effect=self._gather), \
                    patch.object(qwen_benchmark.dist, 'barrier'), \
                    patch.object(qwen_benchmark.torch.npu, 'synchronize'), \
                    patch.object(qwen_benchmark.torch.npu, 'empty_cache'):
                self.assertEqual(qwen_benchmark._run_single_backend(args, config, 0, 8, cpu), 0)
            build.assert_called_once_with(config, cpu, args.seed, 'mega_moe')
            attach.assert_called_once()
            validate.assert_called_once_with(workload, gradient_device=cpu)
            model.close.assert_called_once()
            payload = json.loads(Path(args.output).read_text(encoding='utf-8'))
            self.assertEqual(len(payload['resident_training_per_rank']), 8)

    def test_runtime_identity_includes_kernel_objects_and_checks_editable(self) -> None:
        """Native .o kernels must not disappear from the identity behind an unchanged adapter .so."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            worker = repo / 'examples/torch/multicore/mega_moe/qwen_moe_benchmark.py'
            package = repo / 'hyper_parallel/__init__.py'
            native = repo / 'build/native/payload/hyper_parallel'
            for path in (worker, package, native / 'adapter.so', native / 'kernel.o', native / 'ignored.txt'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('identity fixture', encoding='utf-8')
            with patch.object(qwen_benchmark, '__file__', str(worker)), \
                    patch.object(qwen_benchmark.hyper_parallel, '__file__', str(package)):
                identity = qwen_benchmark._runtime_identity()
                self.assertEqual(set(identity['native_sha256']),
                                 {'build/native/payload/hyper_parallel/adapter.so',
                                  'build/native/payload/hyper_parallel/kernel.o'})
                self.assertEqual(set(identity['source_sha256']),
                                 {'examples/torch/multicore/mega_moe/qwen_moe_benchmark.py',
                                  'hyper_parallel/__init__.py'})
                with patch.object(qwen_benchmark.hyper_parallel, '__file__', str(repo / 'stale/__init__.py')), \
                        self.assertRaisesRegex(RuntimeError, 'this checkout'):
                    qwen_benchmark._runtime_identity()
