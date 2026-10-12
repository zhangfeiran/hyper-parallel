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
"""Physical head sharding, checkpoint round trips and distinct TP projection VJPs over CPU mocks."""

import copy
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional

from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_tp_model_boundary import (
    FusedDsaCpTpAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.tp_projections import (
    DsaTpProjectionCollectives,
    gather_dsa_tp_state_dict,
    load_dsa_tp_state_dict,
    shard_dsa_tp_projections,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)


class TestTpProjections(unittest.TestCase):
    """Keep source Parameter objects while allocating only the local head weight storage."""

    def setUp(self) -> None:
        """Mock explicit TP membership without starting a process group."""
        self.group = object()
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_projections.dist'
        self.addCleanup(patch.stopall)
        patch(prefix+'.is_initialized', return_value=True).start()
        self.members = patch(prefix+'.get_process_group_ranks', return_value=[0, 1]).start()
        self.rank = patch(prefix+'.get_rank', return_value=1).start()
        patch(prefix+'.all_gather_object', side_effect=self._gather_object).start()
        self.broadcast = patch(prefix+'.broadcast').start()

    @staticmethod
    def _gather_object(output, value, **_kwargs):
        output[:] = [value]*len(output)

    def test_local_storage_parameter_identity_and_checkpoint_round_trip(self):
        """TP2/4 physically shrink the five weights and gather/load every row bias in global order."""
        for size in (2, 4):
            with self.subTest(size=size):
                self.members.return_value = list(range(size))
                self.rank.return_value = size-1
                model = tp_model_fixture(dtype=torch.float32, projection_bias=True)
                original = dict(model.named_parameters())
                global_state = {name: value.clone() for name, value in model.state_dict().items()}
                global_numel = sum(value.numel() for value in original.values())
                layout = shard_dsa_tp_projections(model, self.group)
                self.assertEqual(layout.local_heads, 128//size)
                self.assertEqual(layout.local_index_heads, 64//size)
                self.assertLess(sum(value.numel() for value in model.parameters()), global_numel)
                self.assertEqual(set(model.state_dict()), set(global_state))
                for name, parameter in model.named_parameters():
                    self.assertIs(parameter, original[name])
                for name, axis, shape in layout.shards:
                    self.assertEqual(tuple(global_state[name].shape), shape)
                    expected = global_state[name].chunk(size, dim=axis)[size-1]
                    torch.testing.assert_close(model.get_parameter(name), expected, rtol=0, atol=0)
                load_dsa_tp_state_dict(model, {name: value+1 for name, value in global_state.items()})
                calls = iter(layout.shards)
                with patch('hyper_parallel.core.multicore.modules.mega_dsa.tp_projections.dist.all_gather',
                           side_effect=partial(self._gather_global, calls=calls, state=global_state, size=size)):
                    restored = gather_dsa_tp_state_dict(model)
                for name, value in restored.items():
                    torch.testing.assert_close(value, global_state[name]+1, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'already been sharded'):
                    shard_dsa_tp_projections(model, self.group)

    def _gather_global(self, output, value, *, calls, state, size, **_kwargs):
        name, axis, _shape = next(calls)
        target = (state[name]+1).chunk(size, dim=axis)
        torch.testing.assert_close(value, target[size-1], rtol=0, atol=0)
        for destination, source in zip(output, target):
            destination.copy_(source)

    def test_output_vjp_capture_preserves_forward_input_and_leaf_gradients(self):
        """Capture actual O weight/bias VJPs before casting without changing source arithmetic or Parameters."""
        models = [FusedDsaCpTpAttention(module=tp_model_fixture(heads=128, dtype=torch.bfloat16, projection_bias=True),
                  context={'dsa_tp_group': self.group, 'dsa_capture_output_parameter_vjps': enabled})
                  for enabled in (False, True)]
        generator = torch.Generator().manual_seed(20261251)
        original = torch.randn(1, 3, models[0].o_proj.in_features, generator=generator).bfloat16()
        upstream = torch.randn(1, 3, 32, generator=generator)
        observed = []
        for model in models:
            restored = original.clone().requires_grad_()
            with patch.object(model.tp_collectives, 'sum_output', side_effect=lambda value: value):
                output = model._project_output(restored)
                capture = model.parameter_vjp_capture
                if capture is not None:
                    capture.begin_backward()
                gradients = torch.autograd.grad(output, (restored, model.o_proj.weight, model.o_proj.bias), upstream)
                if capture is not None:
                    capture.finish_backward()
                observed.append((output, gradients))
        self.assertTrue(torch.equal(observed[0][0], observed[1][0]))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(observed[0][1], observed[1][1])))
        captured = models[1].parameter_vjp_capture.take()
        expected = upstream.reshape(-1, 32).T@original.float().reshape(-1, models[1].o_proj.in_features)
        torch.testing.assert_close(captured['o_proj.weight'], expected, rtol=0., atol=0.)
        torch.testing.assert_close(captured['o_proj.bias'], upstream.reshape(-1, 32).sum(0), rtol=0., atol=0.)
        self.assertIsNone(models[1].o_proj.weight._backward_hooks)
        self.assertIsNone(models[1].o_proj.bias._backward_hooks)

    def test_group_geometry_and_preexisting_grads_fail_before_mutation(self):
        """Bad geometry or live optimizer inputs cannot partially consume the source module."""
        model = tp_model_fixture(heads=64, dtype=torch.float32)
        pointers = tuple(value.data_ptr() for value in model.parameters())
        self.members.return_value = list(range(4))
        with self.assertRaisesRegex(ValueError, 'local H32/H64'):
            shard_dsa_tp_projections(model, self.group)
        self.assertEqual(pointers, tuple(value.data_ptr() for value in model.parameters()))
        self.members.return_value = [0, 1]
        model.q_b_proj.weight.grad = torch.zeros_like(model.q_b_proj.weight)
        with self.assertRaisesRegex(ValueError, 'optimizer state'):
            shard_dsa_tp_projections(model, self.group)
        self.broadcast.assert_not_called()
        self.assertEqual(pointers, tuple(value.data_ptr() for value in model.parameters()))

    def test_projection_input_sums_vjp_and_output_keeps_identity_vjp(self):
        """Input and output collectives cannot share the replicated-loss backward rule."""
        boundary = DsaTpProjectionCollectives(self.group, row_signature=('projection',))
        value = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        cotangent = torch.tensor([3., 5.], dtype=torch.float64)
        with patch.object(boundary, 'sum_detached', side_effect=lambda tensor: tensor*2) as summed:
            copied = boundary.copy_query_latent(value)
            torch.testing.assert_close(copied, value)
            gradient, = torch.autograd.grad((copied*cotangent).sum(), value)
        torch.testing.assert_close(gradient, cotangent*2)
        summed.assert_called_once()
        with patch.object(boundary, 'sum_detached', side_effect=lambda tensor: tensor*2) as summed:
            output = boundary.sum_output(value)
            torch.testing.assert_close(output, value*2)
            gradient, = torch.autograd.grad((output*cotangent).sum(), value)
        torch.testing.assert_close(gradient, cotangent)
        summed.assert_called_once()

    def test_ordered_projection_sum_keeps_input_sum_and_output_identity_vjps(self):
        """The optional ordered transport preserves both autograd rules and full-precision inputs."""
        boundary = DsaTpProjectionCollectives(self.group, row_signature=('projection',), reduction_order='group_rank')
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_projections.ordered_parameter_sum'
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                compute = torch.float64 if dtype == torch.float64 else torch.float32
                value = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
                cotangent = torch.tensor([3., 5.], dtype=dtype)
                with patch(prefix, side_effect=lambda tensor, _group: tensor*2) as summed:
                    copied = boundary.copy_query_latent(value)
                    gradient, = torch.autograd.grad(copied, value, cotangent)
                    torch.testing.assert_close(copied, value, rtol=0, atol=0)
                    torch.testing.assert_close(gradient, cotangent*2, rtol=0, atol=0)
                    self.assertEqual(summed.call_args.args[0].dtype, compute)
                    self.assertFalse(summed.call_args.args[0].requires_grad)
                with patch(prefix, side_effect=lambda tensor, _group: tensor*2) as summed:
                    output = boundary.sum_output(value)
                    gradient, = torch.autograd.grad(output, value, cotangent)
                    torch.testing.assert_close(output, value*2, rtol=0, atol=0)
                    torch.testing.assert_close(gradient, cotangent, rtol=0, atol=0)
                    self.assertEqual(summed.call_args.args[0].dtype, compute)
                    self.assertIs(summed.call_args.args[1], self.group)
        with self.assertRaisesRegex(ValueError, 'active autograd edge'):
            boundary.sum_detached(torch.ones(2, requires_grad=True))
        with self.assertRaisesRegex(ValueError, 'floating-point'):
            boundary.sum_detached(torch.ones(2, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, 'reduction order'):
            DsaTpProjectionCollectives(self.group, row_signature=(), reduction_order='mean')

    def test_projection_policy_is_collective_and_invalid_model_policy_preserves_storage(self):
        """Reject rank policy disagreement and reject local misuse before physical weight sharding."""
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_projections.dist'

        def _disagreement(output, value, **_kwargs):
            output[:] = [value, (value[0], 'collective')]

        with patch(prefix+'.all_gather_object', side_effect=_disagreement), \
                self.assertRaisesRegex(ValueError, 'disagree'):
            DsaTpProjectionCollectives(self.group, row_signature=(), reduction_order='group_rank')
        source = tp_model_fixture(dtype=torch.bfloat16)
        pointers = {name: parameter.data_ptr() for name, parameter in source.named_parameters()}
        with self.assertRaisesRegex(ValueError, 'reduction order'):
            FusedDsaCpTpAttention(module=source, context={
                'dsa_tp_group': self.group, 'dsa_projection_reduction_order': 'mean'})
        self.assertEqual(pointers, {name: parameter.data_ptr() for name, parameter in source.named_parameters()})

    def test_model_adapter_uses_local_projection_shapes_and_global_index_scale(self):
        """Real projections use H64/Hi32 after TP2 without changing the original Hi64 scale."""
        source = tp_model_fixture(dtype=torch.float64, projection_bias=True)
        parameters = dict(source.named_parameters())
        model = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': self.group})
        self.assertEqual(model.local_heads, 64)
        self.assertEqual(model.local_index_heads, 32)
        self.assertEqual(model.num_heads, 128)
        self.assertEqual(model.num_index_heads, 64)
        self.assertEqual(model.o_proj.in_features, 64*16)
        for name, value in model.named_parameters():
            self.assertIs(value, parameters[name])
        hidden = torch.randn(1, 3, 32, dtype=torch.float64, requires_grad=True)
        resid, states, weights = model._prepare_attention_states(hidden, None)
        self.assertEqual(states[0].shape, (1, 3, 64, 512))
        self.assertEqual(weights.shape, (64, 144, 512))
        iq, ik, merge = model._project_index_states(hidden, resid, None)
        self.assertEqual(iq.shape, (1, 3, 32, 128))
        self.assertEqual(ik.shape, (1, 3, 1, 128))
        expected = model.indexer.weights_proj(hidden.detach())*64**-.5*128**-.5
        torch.testing.assert_close(merge, expected, rtol=0, atol=0)
        self.assertEqual(model.make_transforms(), [])

    def test_fp32_index_projection_matches_float_model_and_preserves_parameter_identity(self):
        """Auxiliary projections keep FP32 states while detached main/input parameters receive no KL VJP."""
        self.rank.return_value = 0
        source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
        oracle = copy.deepcopy(source).float()
        identities = {name: id(parameter) for name, parameter in source.named_parameters()}
        model = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': self.group,
                                                           'dsa_fp32_index_kl': True})
        generator = torch.Generator().manual_seed(1254)
        hidden = torch.randn((1, 3, 32), generator=generator).bfloat16().requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model._project_fp32_index_states(hidden, None)
        resid = oracle._project_attention_states(hidden.detach().float())[0]
        expected = oracle._project_index_states(hidden.detach().float(), resid, None)
        expected = (expected[0][:, :, :model.local_index_heads], expected[1],
                    expected[2][:, :, :model.local_index_heads])
        for output, baseline in zip(actual, expected):
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, baseline, rtol=0, atol=0)
        self.assertEqual(identities, {name: id(parameter) for name, parameter in model.named_parameters()})
        actual_parameters = dict(model.named_parameters())
        expected_parameters = dict(oracle.named_parameters())
        gradients = torch.autograd.grad(sum(value.sum() for value in actual),
                                        (*actual_parameters.values(), hidden), allow_unused=True)
        reference = torch.autograd.grad(sum(value.sum() for value in expected),
                                        tuple(expected_parameters.values()), allow_unused=True)
        self.assertIsNone(gradients[-1])
        for (name, parameter), value, baseline in zip(actual_parameters.items(), gradients, reference):
            if not name.startswith('indexer.'):
                self.assertIsNone(value)
            else:
                if baseline.shape != parameter.shape:
                    baseline = baseline[:parameter.shape[0]]
                torch.testing.assert_close(value, baseline.bfloat16(), rtol=0, atol=0)

    def _model_invocation(self, model, *, empty=False):
        meta = DsaBatchMeta((0, 3), () if empty else (0, 1, 2), () if empty else (0, 1, 2),
                            (0, 0, 0), (0, 1, 2), cp_rank=1 if empty else 0,
                            cp_ranks=(0, 1) if empty else (0,), root_pes=(0, 1) if empty else (0,))
        workspace = SimpleNamespace(root=SimpleNamespace(generation=0), closed=False, participates=True)
        backend = SimpleNamespace(workspace=workspace,
                                   layout=SimpleNamespace(batch_meta=meta, device=torch.device('cpu')),
                                   heads=model.local_heads, scale=model.scaling)
        forward = SimpleNamespace(li=SimpleNamespace(group=self.group, members=(0, 1)),
                                   index_heads=model.local_index_heads)
        probe = object.__new__(FusedDsaLocalCpTpTrainingProbe)
        probe.forward_backend = backend
        probe.tp_forward = forward
        probe.mode = 'lm_only'
        probe.fp32_index_kl = False
        probe.fp32_compressed_vjp = False
        probe.fp32_native_query_vjp = False
        probe.fp32_rope_vjp = False
        probe.loss_coeff = model.dsa_loss_coeff
        cp = DsaWorkspaceInvocation(meta, torch.arange(len(meta.kv_global_ids)), workspace)
        invocation = FusedCpTpInvocation(forward, cp, DsaLocalQueryLayout(meta, 'cpu'), (), False)
        return probe, invocation

    def test_model_empty_owner_and_output_bias_are_applied_once(self):
        """Actual model forward admits zero rows and a replicated bias has no TP multiplier."""
        source = tp_model_fixture(dtype=torch.float64, projection_bias=True)
        model = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': self.group}).eval()
        for empty in (False, True):
            with self.subTest(empty=empty):
                probe, invocation = self._model_invocation(model, empty=empty)
                hidden = torch.randn(1, 0 if empty else 3, 32, dtype=torch.float64)
                with patch.object(probe, 'forward', side_effect=lambda _inv, main, _index:
                                  (main[0], torch.tensor(0.))), \
                        patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda tensor: tensor*2):
                    result, _ = model(hidden, dsa_training=probe, dsa_invocation=invocation)
                    gradient, = torch.autograd.grad(result.sum(), (model.o_proj.bias,))
                self.assertEqual(result.shape, hidden.shape)
                torch.testing.assert_close(gradient, torch.full_like(model.o_proj.bias, hidden.shape[1]),
                                           rtol=0, atol=0)

    def test_output_projection_preserves_cross_rank_residual_and_bias_vjp(self):
        """A half-ULP local product must survive cancellation, including with autocast enabled."""
        source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
        model = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': self.group})
        with torch.no_grad():
            model.o_proj.weight.zero_()
            model.o_proj.weight[0, 0] = 1.
            model.o_proj.weight[0, 1] = 2**-8
            model.o_proj.bias.zero_()
            model.o_proj.bias[0] = 2**-10
        restored = torch.zeros(1, 1, model.o_proj.in_features, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            restored[0, 0, :2] = 1.
        remote = torch.zeros(1, 1, 32, dtype=torch.float32)
        remote[0, 0, 0] = -1.
        with patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda value: value+remote) as summed:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = model._project_output(restored).to(restored.dtype)
            gradients = torch.autograd.grad(output.sum(), (restored, model.o_proj.weight, model.o_proj.bias))
        self.assertEqual(summed.call_args.args[0].dtype, torch.float32)
        expected = torch.zeros_like(output)
        expected[0, 0, 0] = 2**-8+2**-10
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        expected_restored = torch.zeros_like(restored)
        expected_restored[0, 0, :2] = torch.tensor([1., 2**-8])
        torch.testing.assert_close(gradients[0], expected_restored, rtol=0, atol=0)
        expected_weight = torch.zeros_like(model.o_proj.weight)
        expected_weight[:, :2] = 1.
        torch.testing.assert_close(gradients[1], expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(gradients[2], torch.ones_like(model.o_proj.bias), rtol=0, atol=0)

    def test_query_projection_sums_fp32_input_vjp_before_latent_cast(self):
        """Keep a half-ULP input-gradient residual while preserving BF16 projection values."""
        source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
        model = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': self.group})
        with torch.no_grad():
            model.q_b_proj.weight.zero_()
            model.q_b_proj.weight[0, 0] = 1.
            model.q_b_proj.weight[model.qk_head_dim, 0] = 2**-8
            model.q_b_proj.bias.zero_()
        resid = torch.zeros(1, 1, model.q_lora_rank, dtype=torch.bfloat16, requires_grad=True)
        with torch.no_grad():
            resid[0, 0, 0] = 1.
        remote = torch.zeros_like(resid, dtype=torch.float32)
        remote[0, 0, 0] = -1.
        expected_output = torch.zeros(1, 1, model.q_b_proj.out_features, dtype=torch.bfloat16)
        expected_output[0, 0, 0] = 1.
        expected_output[0, 0, model.qk_head_dim] = 2**-8
        with patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda value: value+remote) as summed:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = model._project_query(resid)
            gradients = torch.autograd.grad(output.sum(), (resid, model.q_b_proj.weight, model.q_b_proj.bias))
        self.assertEqual(summed.call_args.args[0].dtype, torch.float32)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        expected_resid = torch.zeros_like(resid)
        expected_resid[0, 0, 0] = 2**-8
        torch.testing.assert_close(gradients[0], expected_resid, rtol=0, atol=0)
        expected_weight = torch.zeros_like(model.q_b_proj.weight)
        expected_weight[:, 0] = 1.
        torch.testing.assert_close(gradients[1], expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(gradients[2], torch.ones_like(model.q_b_proj.bias), rtol=0, atol=0)

    def test_query_parameter_capture_uses_actual_tp_boundary(self):
        """The sharded module routes weight/bias VJPs to its capture without rounding before CP SUM."""
        source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
        model = FusedDsaCpTpAttention(module=source, context={
            'dsa_tp_group': self.group, 'dsa_capture_query_parameter_vjps': True})
        parameters = (model.q_b_proj.weight, model.q_b_proj.bias)
        resid = torch.linspace(-1, 1, 3*model.q_lora_rank).reshape(1, 3, -1).bfloat16().requires_grad_()
        with patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda value: value) as summed:
            output = model._project_query(resid)
            expected = functional.linear(resid, *parameters)  # pylint: disable=not-callable
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            upstream = torch.linspace(-1, 1, output.numel()).reshape(output.shape).bfloat16()
            model.parameter_vjp_capture.begin_backward()
            grads = torch.autograd.grad(output, (resid, *parameters), upstream)
            model.parameter_vjp_capture.finish_backward()
        observed = model.parameter_vjp_capture.take()
        expected_weight = upstream.float().reshape(3, -1).T@resid.detach().float().reshape(3, -1)
        torch.testing.assert_close(observed['q_b_proj.weight'], expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(observed['q_b_proj.bias'], upstream.float().sum((0, 1)), rtol=0, atol=0)
        torch.testing.assert_close(grads[1], expected_weight.bfloat16(), rtol=0, atol=0)
        self.assertIsNone(observed['o_proj.weight'])
        self.assertEqual(summed.call_args.args[0].dtype, torch.float32)


    def test_fp32_query_vjp_retains_source_states_and_norm_cotangent(self):
        """Query norm sees FP32 cotangents while every original main/index forward state stays BF16 exact."""
        for rows in (0, 3):
            with self.subTest(rows=rows):
                source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
                baseline = FusedDsaCpTpAttention(module=copy.deepcopy(source), context={'dsa_tp_group': self.group})
                parameters = dict(source.named_parameters())
                model = FusedDsaCpTpAttention(module=source, context={
                    'dsa_tp_group': self.group, 'dsa_capture_qkv_parameter_vjps': True,
                    'dsa_capture_kv_parameter_vjps': True, 'dsa_capture_query_parameter_vjps': True,
                    'dsa_fp32_query_vjp': True})
                hidden = torch.linspace(-1, 1, rows*32).reshape(1, rows, 32).bfloat16().requires_grad_()
                resid, states, _weights = model._prepare_attention_states(hidden, None)
                expected_resid, expected_states, _ = baseline._prepare_attention_states(hidden, None)
                torch.testing.assert_close(resid, expected_resid.float(), rtol=0, atol=0)
                for actual, expected in zip(states, expected_states):
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for actual, expected in zip(model._project_index_states(hidden, resid, None),
                                            baseline._project_index_states(hidden, expected_resid, None)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                cotangents = []
                resid.register_hook(lambda value, records=cotangents: records.append(value.detach().clone()))
                objective = sum(value.float().square().sum() for value in states)
                model.parameter_vjp_capture.begin_backward()
                with patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda value: value):
                    objective.backward()
                model.parameter_vjp_capture.finish_backward()
                captured = model.parameter_vjp_capture.take()
                self.assertEqual(cotangents[0].dtype, torch.float32)
                latent = source.linear_qkv(hidden.detach())[..., :model.q_lora_rank].double()
                inverse = torch.rsqrt(latent.square().mean(-1, keepdim=True)+model.q_a_layernorm.eps)
                expected = (cotangents[0].double()*latent*inverse).reshape(-1, model.q_lora_rank).sum(0)
                torch.testing.assert_close(captured['q_a_layernorm.weight'].double(), expected,
                                           rtol=2e-5, atol=2e-6)
                self.assertTrue(all(dict(model.named_parameters())[name] is parameter
                                    for name, parameter in parameters.items()))


    def test_fp32_compressed_norm_retains_original_primal_and_full_precision_weight_vjp(self):
        """KV norm's compute representation keeps source forward values and its unrounded derivative."""
        for rows in (0, 3):
            with self.subTest(rows=rows):
                source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
                baseline = FusedDsaCpTpAttention(module=copy.deepcopy(source), context={'dsa_tp_group': self.group})
                original = dict(source.named_parameters())
                model = FusedDsaCpTpAttention(module=source, context={
                    'dsa_tp_group': self.group, 'dsa_capture_qkv_parameter_vjps': True,
                    'dsa_capture_kv_parameter_vjps': True, 'dsa_capture_query_parameter_vjps': True,
                    'dsa_fp32_query_vjp': True, 'dsa_fp32_compressed_vjp': True})
                hidden = torch.linspace(-1, 1, rows*32).reshape(1, rows, 32).bfloat16().requires_grad_()
                resid, actual, _ = model._prepare_attention_states(hidden, None)
                expected_resid, expected, _ = baseline._prepare_attention_states(hidden, None)
                torch.testing.assert_close(resid, expected_resid.float(), rtol=0, atol=0)
                for index, (value, reference) in enumerate(zip(actual, expected)):
                    self.assertEqual(value.dtype, torch.float32 if index == 1 else torch.bfloat16)
                    torch.testing.assert_close(value, reference.to(value.dtype), rtol=0, atol=0)
                upstream = torch.linspace(-.2, .3, actual[1].numel()).reshape(actual[1].shape)
                model.parameter_vjp_capture.begin_backward()
                actual[1].backward(upstream)
                model.parameter_vjp_capture.finish_backward()
                observed = model.parameter_vjp_capture.take()
                latent = source.linear_qkv(hidden.detach())[..., model.q_lora_rank:model.q_lora_rank+512].double()
                inverse = torch.rsqrt(latent.square().mean(-1, keepdim=True)+model.kv_a_layernorm.eps)
                expected_weight = (upstream.squeeze(2).double()*latent*inverse).reshape(-1, 512).sum(0)
                torch.testing.assert_close(observed['kv_a_layernorm.weight'].double(), expected_weight,
                                           rtol=2e-5, atol=2e-6)
                self.assertTrue(all(dict(model.named_parameters())[name] is parameter
                                    for name, parameter in original.items()))


    def test_native_query_compute_representation_keeps_original_main_values(self):
        """Both Q and C carry FP32 autograd edges while the complete BF16 main primal remains identical."""
        for rows in (0,3):
            with self.subTest(rows=rows):
                source=tp_model_fixture(dtype=torch.bfloat16,projection_bias=True)
                baseline=FusedDsaCpTpAttention(module=copy.deepcopy(source),context={'dsa_tp_group':self.group})
                model=FusedDsaCpTpAttention(module=source,context={
                    'dsa_tp_group':self.group,'dsa_capture_qkv_parameter_vjps':True,
                    'dsa_capture_kv_parameter_vjps':True,'dsa_capture_query_parameter_vjps':True,
                    'dsa_fp32_query_vjp':True,'dsa_fp32_compressed_vjp':True,'dsa_fp32_native_query_vjp':True})
                hidden=torch.linspace(-1,1,rows*32).reshape(1,rows,32).bfloat16().requires_grad_()
                _,actual,_=model._prepare_attention_states(hidden,None)
                _,expected,_=baseline._prepare_attention_states(hidden,None)
                for index,(value,reference) in enumerate(zip(actual,expected)):
                    self.assertEqual(value.dtype,torch.float32 if index<2 else torch.bfloat16)
                    torch.testing.assert_close(value,reference.to(value.dtype),rtol=0,atol=0)


    def test_rope_compute_representation_keeps_parameter_identity_and_source_values(self):
        """RoPE inputs retain FP32 edges into query and fused QKV projections, including empty tokens."""
        for rows in (0, 3):
            with self.subTest(rows=rows):
                source = tp_model_fixture(dtype=torch.bfloat16, projection_bias=True)
                original = dict(source.named_parameters())
                baseline = FusedDsaCpTpAttention(module=copy.deepcopy(source), context={'dsa_tp_group':self.group})
                model = FusedDsaCpTpAttention(module=source, context={
                    'dsa_tp_group':self.group, 'dsa_capture_qkv_parameter_vjps':True,
                    'dsa_capture_kv_parameter_vjps':True, 'dsa_capture_query_parameter_vjps':True,
                    'dsa_fp32_query_vjp':True, 'dsa_fp32_compressed_vjp':True,
                    'dsa_fp32_native_query_vjp':True, 'dsa_fp32_rope_vjp':True})
                hidden = torch.linspace(-1, 1, rows*32).reshape(1, rows, 32).bfloat16().requires_grad_()
                _, actual, _ = model._prepare_attention_states(hidden, None)
                _, expected, _ = baseline._prepare_attention_states(hidden, None)
                for value, reference in zip(actual, expected):
                    self.assertEqual(value.dtype, torch.float32)
                    torch.testing.assert_close(value, reference.float(), rtol=0, atol=0)
                with patch.object(model.tp_collectives, 'sum_detached', side_effect=lambda value: value*2):
                    gradients = torch.autograd.grad(actual[2].sum()*1.003+actual[3].sum()*1.007, hidden)[0]
                self.assertTrue(torch.isfinite(gradients).all())
                self.assertTrue(all(dict(model.named_parameters())[name] is parameter
                                    for name, parameter in original.items()))
