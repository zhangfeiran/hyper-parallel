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
"""Exercise event collection on actual CPU autograd and checkpoint replay."""

from copy import deepcopy
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from examples.training_demo.deepseek_v41.deepseek_v41_moe_timing import MoeTiming
from hyper_parallel.core.activation_memory import ckpt_wrapper
from hyper_parallel.components.modules.shared_compressed_dsa_attention import SharedCompressedPackedSequence
from hyper_parallel.distributed.activation_checkpoint import _apply_activation_checkpointing
from tests.ut.auto_models.models.deepseek_v41.test_host_recompute import _tiny_model


class CpuEvent:
    """Resolve host timestamps for hardware-independent hook acceptance."""

    def __init__(self) -> None:
        """Initialize an unrecorded CPU event."""
        self.stamp = 0.0

    def record(self) -> None:
        """Record CPU time instead of touching an accelerator."""
        self.stamp = time.perf_counter()

    def elapsed_time(self, other: 'CpuEvent') -> float:
        """Match the accelerator event's millisecond interface."""
        return (other.stamp - self.stamp) * 1000


class _Moe(torch.nn.Module):
    def __init__(self) -> None:
        """Build independent routed and shared branches."""
        super().__init__()
        self.gate = torch.nn.Linear(4, 4)
        self.experts = torch.nn.Linear(4, 4)
        self.shared_experts = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Keep the gate and both expert branches in the autograd graph."""
        return self.gate(value).sigmoid() * torch.nn.functional.silu(self.experts(value)) + self.shared_experts(value)


class TestMoeTiming(unittest.TestCase):
    """Hooks preserve outputs and all gradients without inserting tensor views."""

    def test_checkpoint_phases_and_gradient_equivalence(self) -> None:
        torch.manual_seed(7)
        reference = torch.nn.Sequential()
        reference.add_module('mlp', _Moe())
        candidate = deepcopy(reference)
        model = torch.nn.Module()
        model.layer = ckpt_wrapper(candidate)
        collector = MoeTiming(model, event_factory=CpuEvent)
        self.addCleanup(collector.close)
        expected_input = torch.randn(3, 4, requires_grad=True)
        actual_input = expected_input.detach().clone().requires_grad_()
        expected = reference(expected_input)
        with patch.object(DefaultDeviceType, 'get_device_type', return_value='cpu'):
            actual = model.layer(actual_input)
            actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0, atol=0)
        for parameter, other in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
        record = collector.snapshot()
        self.assertEqual(record['incomplete_intervals'], [])
        self.assertEqual(len(record['layers']), 1)
        phases = next(iter(record['layers'].values()))
        self.assertEqual(set(phases), {'forward', 'recompute_forward', 'backward_to_input_grad'})
        for phase in phases.values():
            self.assertEqual(phase['calls'], 1)
            self.assertGreaterEqual(phase['current_stream_ms'], 0)
        collector.clear()
        self.assertEqual(collector.records, [])
        self.assertEqual(collector.grad_hooks, [])

    def test_tiny_packed_vlm_recompute_keeps_all_moe_intervals(self) -> None:
        # Model fixture exercises real decoder/CSA2 boundaries rather than fake checkpoint contexts.

        with TemporaryDirectory() as directory, patch.object(DefaultDeviceType, 'get_device_type', return_value='cpu'):
            model = _tiny_model(directory)
            _apply_activation_checkpointing(model, 'full', selection=SimpleNamespace(
                source='model_adapter_safe_regions', layer_count=4, layer_indices=None))
            collector = MoeTiming(model, event_factory=CpuEvent)
            try:
                token_types = torch.full((1, 32), -1, dtype=torch.long)
                for start in (2, 18):
                    token_types[0, start:start + 8] = torch.tensor([0, 1, 1, 2, 1, 1, 2, 3])
                result = model(
                    input_ids=torch.randint(1, 32, (1, 32)), token_types=token_types,
                    pixel_values=torch.randn(8, 3, 2, 2), image_patch_offsets=torch.tensor([0, 4, 8]),
                    image_vit_grid_hw=torch.tensor([[2, 2], [2, 2]]),
                    image_llm_grid_hw=torch.tensor([[2, 2], [2, 2]]), image_batch_indices=torch.tensor([0, 0]),
                    image_token_starts=torch.tensor([2, 18]), use_cache=False,
                    packed_seq_params=SharedCompressedPackedSequence(torch.tensor([0, 16, 32]), 0, 32, 32),
                ).logits
                result.float().square().mean().backward()
                observed = collector.snapshot()
                self.assertEqual(len(observed['layers']), 4)
                self.assertEqual(observed['incomplete_intervals'], [])
                for phases in observed['layers'].values():
                    self.assertEqual(phases['forward']['calls'], 1)
                    self.assertEqual(phases['backward_to_input_grad']['calls'], 1)
                    self.assertEqual(phases['recompute_forward']['calls'], 1)
            finally:
                collector.close()
