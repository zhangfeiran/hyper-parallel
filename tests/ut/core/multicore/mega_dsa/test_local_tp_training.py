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
"""Autograd separation and precision for combined native CP/TP owner returns over CPU mocks."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumInvocation,
    TpTileSumSpec,
)
from tests.ut.core.multicore.mega_dsa import test_local_training as training_fixture


class TestLocalTpTraining(unittest.TestCase):
    """Reuse saved-attention fixtures while independently checking shared TP SUM and scaling."""

    def setUp(self) -> None:
        """Emulate two TP head partitions over one real CPU-mocked CP workspace."""
        self.fixture = training_fixture.TestLocalTraining()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.owner_transport = Mock()
        self.owner_transport.root = self.fixture.root
        self.owner_transport.members = (0, 1)
        self.owner_transport.participates = True
        self.owner_transport.spec = TpTileSumSpec(1, 512)
        self.owner_transport.arena = torch.empty(0, dtype=torch.uint8)
        self.owner_transport.last_epoch = 0
        self.owner_transport.prepare.side_effect = self._prepare_transport
        self.owner_transport.run.side_effect = self._tp_sum
        li = SimpleNamespace(root=self.fixture.root, members=(0, 1), group=object())
        self.forward = SimpleNamespace(cp=self.fixture.backend, li=li, kl=object(), forward=Mock())
        self.forward.forward.side_effect = self._forward
        patch('hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.dist.get_world_size',
              return_value=2).start()
        patch('hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.dist.all_gather_object',
              side_effect=self._gather).start()
        patch.object(FusedDsaLocalCpTpTrainingProbe, 'check_native_support').start()
        patch('hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return.TpOwnerGradientReturnProbe.'
              'check_native_support').start()

    @staticmethod
    def _gather(output, value, **_kwargs):
        output[:] = [value]*len(output)

    def _prepare_transport(self, rounds, epoch, identity):
        metadata = torch.zeros(24, dtype=torch.int64)
        return TpTileSumInvocation(self.owner_transport, rounds, epoch, 1, metadata, metadata._version, identity)

    def _tp_sum(self, partial, invocation):
        self.owner_transport.last_epoch = invocation.epoch
        self.assertEqual(partial.dtype, torch.float32)
        return partial*2

    def _forward(self, main, index, invocation):
        result = self.fixture._forward(invocation.cp, main, index, query_layout=invocation.local)
        raw = (torch.full_like(index[0], 2), self.fixture.key_partials.bfloat16(), torch.full_like(index[2], 5))
        return replace(result, kl_gradients=raw, kl_key_partials=self.fixture.key_partials.clone(),
                       kl_loss=torch.tensor(7.))

    def _main_backward(self, saved, shape, cotangent, *, local_queries, defer_owner_cast):
        self.assertTrue(local_queries)
        self.assertTrue(defer_owner_cast)
        value = self.fixture._main_backward(saved, shape, cotangent, local_queries=local_queries)
        factor = float(cotangent.flatten()[0]) if cotangent.numel() else 0.
        owner = (torch.full((3, 512), 1.003*factor), torch.full((3, 64), 7.003*factor))
        return SimpleNamespace(gradients=value.gradients, owner_fp32=owner, retained=torch.empty(0,dtype=torch.uint8))

    def _probe(self, mode='lm_plus_kl', coefficient=.7, backward_schedule=None, group_private_partials=False,
               fp32_index_kl=False, fp32_compressed_vjp=False, fp32_native_query_vjp=False, fp32_rope_vjp=False):
        probe = FusedDsaLocalCpTpTrainingProbe(self.forward, self.owner_transport,
            normalization=DsaLossNormalization(3, 2), loss_coeff=coefficient, mode=mode,
            backward_schedule=backward_schedule, group_private_partials=group_private_partials,
            fp32_index_kl=fp32_index_kl, fp32_compressed_vjp=fp32_compressed_vjp,
            fp32_native_query_vjp=fp32_native_query_vjp, fp32_rope_vjp=fp32_rope_vjp)
        patch.object(probe.backward_backend, '_backward_saved', side_effect=self._main_backward).start()
        patch.object(probe.shared_return.index_return, 'check_native_support').start()
        patch.object(probe.shared_return.index_return, 'return_owner', side_effect=self._index_cp_return).start()
        return probe

    def _index_cp_return(self, invocation, partial):
        self.assertEqual(partial.dtype, torch.float32)
        return SimpleNamespace(owner_fp32=partial[list(invocation.batch_meta.kv_global_ids)])

    def _inputs(self, probe):
        layout, inputs = self.fixture._inputs()
        inputs = (*inputs[:4], inputs[4][:, :32].detach().requires_grad_(), inputs[5],
                  inputs[6][:, :32].detach().requires_grad_())
        invocation = FusedCpTpInvocation(self.forward, self.fixture.invocation, layout, (),
                                         probe.index_active and not probe.fp32_index_kl)
        return invocation, inputs

    def test_fp32_kl_keeps_original_li_and_applies_scale_and_shared_returns_once(self):
        """FP32 auxiliary states remain separate from BF16 selection and retained backward stays FP32."""
        factory_path = 'hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.Fp32FusedSelectedKlProbe'
        with patch(factory_path) as factory:
            probe = self._probe(fp32_index_kl=True)
        invocation, original = self._inputs(probe)
        inputs = (*original[:4], *(tensor.float().detach().requires_grad_() for tensor in original[4:]))

        def _raw(_result, _main, index):
            return SimpleNamespace(vjp=SimpleNamespace(
                gradients=(torch.full_like(index[0], 1.003), self.fixture.key_partials.clone(),
                           torch.full_like(index[2], 5.003)), loss=torch.full((index[0].shape[0],), 3.5)))

        factory.return_value.gradients.side_effect = _raw
        output, loss = probe.forward(invocation, inputs[:4], inputs[4:], li_index=original[4:])
        objective = output.sum()+loss*-1.75
        first = torch.autograd.grad(objective, inputs, retain_graph=True)
        second = torch.autograd.grad(objective, inputs)
        self.assertFalse(invocation.selected_kl)
        self.assertEqual(factory.return_value.gradients.call_count, 1)
        self.assertEqual(self.owner_transport.run.call_count, 2)
        scale = .7*2/3*-1.75
        self.assertTrue(all(tensor.dtype == torch.float32 for tensor in first[4:]))
        torch.testing.assert_close(first[4], torch.full_like(inputs[4], 1.003*scale))
        torch.testing.assert_close(first[6], torch.full_like(inputs[6], 5.003*scale))
        forward_scale = probe.loss_coeff*probe.normalization.local_sum_scale
        expected_key = (self.fixture.key_partials*forward_scale*-1.75)[list(self.fixture.meta.kv_global_ids)]*2
        torch.testing.assert_close(first[5], expected_key, rtol=0, atol=0)
        self.assertEqual(self.forward.forward.call_args.args[1][0].dtype, torch.bfloat16)
        for actual, repeated in zip(first, second):
            torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        probe.fp32_index_kl = False
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.check_declaration()

    def test_joint_vjp_applies_scale_once_and_retained_backward_uses_fresh_tp_epoch(self):
        """Replicated loss uses identity local VJP; only shared keys receive the second TP contribution."""
        probe = self._probe()
        invocation, inputs = self._inputs(probe)
        output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
        objective = output.sum()+loss*-1.75
        first = torch.autograd.grad(objective, inputs, retain_graph=True)
        second = torch.autograd.grad(objective, inputs)
        self.assertEqual(self.owner_transport.last_epoch, 2)
        self.assertEqual(self.owner_transport.run.call_count, 2)
        scale = .7*2/3*-1.75
        torch.testing.assert_close(first[4], torch.full_like(inputs[4], 2*scale))
        torch.testing.assert_close(first[6], torch.full_like(inputs[6], 5*scale))
        expected = (self.fixture.key_partials*scale)[list(self.fixture.meta.kv_global_ids)]*2
        torch.testing.assert_close(first[5], expected.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(first[1], torch.full_like(inputs[1], 1.003*2), rtol=0, atol=0)
        for actual, replay in zip(first, second):
            torch.testing.assert_close(actual, replay, rtol=0, atol=0)
        self.forward.forward.assert_called_once()

    def test_group_private_training_configuration_is_declared_and_immutable(self):
        """The actual training bridge passes the storage protocol and rejects changing it after preparation."""
        probe = self._probe(group_private_partials=True)
        self.assertTrue(probe.backward_backend.group_private_partials)
        probe.check_declaration()
        probe.group_private_partials = False
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.check_declaration()
        for flag in (1, None, 'true'):
            with self.subTest(flag=flag), self.assertRaisesRegex(TypeError, 'must be a boolean'):
                self._probe(group_private_partials=flag)

    def test_independent_backward_schedule_preserves_scale_and_rejects_mutation(self):
        """The actual training bridge chooses g1 backward while retaining its g7 forward contract."""
        probe = self._probe(backward_schedule=MixedSfaSchedule(1))
        self.assertEqual(probe.forward_backend.schedule.compute_groups, 7)
        self.assertEqual(probe.backward_backend.schedule.compute_groups, 1)
        invocation, inputs = self._inputs(probe)
        output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
        grads = torch.autograd.grad(output.sum()+loss*-1.75, inputs)
        scale = .7*2/3*-1.75
        torch.testing.assert_close(grads[4], torch.full_like(inputs[4], 2*scale))
        torch.testing.assert_close(grads[1], torch.full_like(inputs[1], 1.003*2), rtol=0, atol=0)
        probe.backward_schedule = MixedSfaSchedule(2)
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.check_declaration()

    def test_backward_schedule_is_declared_consistently_across_cp_and_tp(self):
        """A caller cannot prepare different main backward schedules on participating peers."""
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.dist.all_gather_object'

        def _disagree(output, declaration, **_kwargs):
            different = list(declaration)
            different[5] = MixedSfaSchedule(2)
            output[:] = [declaration, tuple(different)]

        with patch(prefix, side_effect=_disagree), self.assertRaisesRegex(ValueError, 'disagree'):
            self._probe(backward_schedule=MixedSfaSchedule(1))
        for invalid in (1, MixedSfaSchedule(1, rounds=2)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'rounds=1'):
                self._probe(backward_schedule=invalid)

    def test_modes_and_zero_coefficient_keep_inactive_gradients_isolated(self):
        """Auxiliary-only and main-only calls leave no autograd edge into the other projections."""
        for mode, coefficient in (('lm_only', .7), ('kl_only', .7), ('kl_only', 0.)):
            with self.subTest(mode=mode, coefficient=coefficient):
                probe = self._probe(mode, coefficient)
                invocation, inputs = self._inputs(probe)
                output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
                objective = output.sum() if mode == 'lm_only' else loss
                gradients = torch.autograd.grad(objective, inputs, allow_unused=True)
                inactive = gradients[4:] if mode == 'lm_only' else gradients[:4]
                self.assertTrue(all(value is None for value in inactive))
                if coefficient == 0:
                    self.assertTrue(all(not bool(value.any()) for value in gradients[4:]))

    def test_saved_input_mutation_and_higher_order_reject_before_owner_return(self):
        """Saved-tensor hooks enforce versions and prevent unsupported second-order transport."""
        probe = self._probe()
        invocation, inputs = self._inputs(probe)
        output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
        with self.assertRaisesRegex(ValueError, 'first-order'):
            torch.autograd.grad(output.sum()+loss, inputs, create_graph=True)
        self.owner_transport.run.assert_not_called()
        output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
        with torch.no_grad():
            inputs[0].add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified'):
            torch.autograd.grad(output.sum()+loss, inputs)
        self.owner_transport.run.assert_not_called()

    def test_changed_objective_rejects_before_forward(self):
        """The collectively prepared mode and scale cannot silently change after initialization."""
        probe = self._probe()
        invocation, inputs = self._inputs(probe)
        probe.loss_coeff = .2
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.forward(invocation, inputs[:4], inputs[4:])
        self.forward.forward.assert_not_called()


    def test_fp32_compressed_input_keeps_native_primal_and_owner_sum(self):
        """CP/TP shared return reaches C without BF16 rounding under plain/retained/checkpoint backward."""
        for replay in ('plain', 'retained', 'checkpoint'):
            with self.subTest(replay=replay):
                probe = self._probe(mode='lm_only', fp32_compressed_vjp=True)
                invocation, inputs = self._inputs(probe)
                compressed = inputs[1].detach().float().requires_grad_()
                main = (inputs[0], compressed, *inputs[2:4])

                def _forward(value, backend=probe, prepared=invocation, states=main, index=inputs[4:]):
                    return backend.forward(prepared, (states[0], value, *states[2:]), index)[0]

                output = (checkpoint(_forward, compressed, use_reentrant=False)
                          if replay == 'checkpoint' else _forward(compressed))
                supplied = self.forward.forward.call_args.args[0]
                self.assertTrue(all(value.dtype == torch.bfloat16 for value in supplied))
                torch.testing.assert_close(supplied[1], compressed.bfloat16(), rtol=0, atol=0)
                gradients = []
                for _repeat in range(2 if replay == 'retained' else 1):
                    gradient = torch.autograd.grad(output.sum(), compressed, retain_graph=replay == 'retained')[0]
                    self.assertEqual(gradient.dtype, torch.float32)
                    torch.testing.assert_close(gradient, torch.full_like(compressed, 1.003)*2, rtol=0, atol=0)
                    self.assertFalse(torch.equal(gradient, gradient.bfloat16().float()))
                    gradients.append(gradient)
                if replay == 'retained':
                    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)

    def test_fp32_compressed_declaration_mutation_and_dtype_rejected(self):
        """The dtype contract is declared collectively and cannot change after preparation."""
        probe = self._probe(fp32_compressed_vjp=True)
        invocation, inputs = self._inputs(probe)
        with self.assertRaisesRegex(ValueError, 'declared Q/RoPE precision and FP32'):
            probe.forward(invocation, inputs[:4], inputs[4:])
        probe.fp32_compressed_vjp = False
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.check_declaration()
        for invalid in (1, None, 'true'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(TypeError, 'must be a boolean'):
                self._probe(fp32_compressed_vjp=invalid)


    def test_fp32_compressed_original_input_version_is_checked_before_native_backward(self):
        """The bridge saves the original compute input in addition to its distinct BF16 native copy."""
        probe = self._probe(mode='lm_only', fp32_compressed_vjp=True)
        invocation, inputs = self._inputs(probe)
        compressed = inputs[1].detach().float().requires_grad_()
        output, _ = probe.forward(invocation, (inputs[0], compressed, *inputs[2:4]), inputs[4:])
        with torch.no_grad():
            compressed.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
            torch.autograd.grad(output.sum(), compressed)
        self.assertEqual(self.fixture.main_calls, 0)


    def test_fp32_compressed_mode_and_loss_scaling_isolation(self):
        """Changing C's autograd dtype preserves main/index branch isolation and the once-only KL scale."""
        for mode, coefficient in (('lm_only', .7), ('kl_only', .7), ('kl_only', 0.), ('lm_plus_kl', .7)):
            with self.subTest(mode=mode, coefficient=coefficient):
                probe = self._probe(mode=mode, coefficient=coefficient, fp32_compressed_vjp=True)
                invocation, inputs = self._inputs(probe)
                compressed = inputs[1].detach().float().requires_grad_()
                inputs = (inputs[0], compressed, *inputs[2:])
                output, loss = probe.forward(invocation, inputs[:4], inputs[4:])
                objective = output.sum() if mode == 'lm_only' else loss*-1.75
                if mode == 'lm_plus_kl':
                    objective = objective+output.sum()
                gradients = torch.autograd.grad(objective, inputs, allow_unused=True)
                if mode == 'lm_only':
                    self.assertTrue(all(value is None for value in gradients[4:]))
                elif mode == 'kl_only':
                    self.assertTrue(all(value is None for value in gradients[:4]))
                if mode != 'lm_only':
                    scale = coefficient*2/3*-1.75
                    torch.testing.assert_close(gradients[4], torch.full_like(inputs[4], 2*scale))
                if mode != 'kl_only':
                    torch.testing.assert_close(gradients[1], torch.full_like(compressed, 1.003)*2, rtol=0, atol=0)


    def test_fp32_native_query_returns_decoded_gradient_in_original_caller_order(self):
        """The saved local layout restores an unrounded FP32 dQ under plain/retained/checkpoint backward."""
        prefix='hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.scaled_query_workspace_vjps'
        for replay in ('plain','retained','checkpoint'):
            with self.subTest(replay=replay):
                probe=self._probe(mode='lm_only',fp32_compressed_vjp=True,fp32_native_query_vjp=True)
                invocation,inputs=self._inputs(probe)
                query=inputs[0].detach().float().requires_grad_()
                compressed=inputs[1].detach().float().requires_grad_()
                packed=torch.arange(query.shape[0],dtype=torch.float32).view(-1,1,1)+2.003
                decoded=packed.expand_as(query).clone()
                expected=invocation.local.restore(decoded)

                def _forward(value,backend=probe,prepared=invocation,source=inputs,owner=compressed):
                    return backend.forward(prepared,(value,owner,*source[2:4]),source[4:])[0]

                output=(checkpoint(_forward,query,use_reentrant=False) if replay=='checkpoint' else _forward(query))
                with patch(prefix,return_value=(decoded,torch.empty(0))) as decode:
                    gradient=torch.autograd.grad(output.sum(),query,retain_graph=replay=='retained')[0]
                    torch.testing.assert_close(gradient,expected,rtol=0,atol=0)
                    self.assertEqual(gradient.dtype,torch.float32)
                    self.assertFalse(torch.equal(gradient,gradient.bfloat16().float()))
                    self.assertEqual(decode.call_args.kwargs['attention_scale'],self.fixture.backend.scale)
                    if replay=='retained':
                        second=torch.autograd.grad(output.sum(),query)[0]
                        torch.testing.assert_close(second,expected,rtol=0,atol=0)
                supplied=self.forward.forward.call_args.args[0]
                self.assertTrue(all(value.dtype==torch.bfloat16 for value in supplied))
        with self.assertRaisesRegex(ValueError,'requires FP32 compressed'):
            self._probe(fp32_native_query_vjp=True)
        probe=self._probe(fp32_compressed_vjp=True,fp32_native_query_vjp=True)
        probe.fp32_native_query_vjp=False
        with self.assertRaisesRegex(ValueError,'objective was modified'):
            probe.check_declaration()


    def test_fp32_rope_returns_local_query_and_once_reduced_owner_in_float(self):
        """Both rotary VJPs keep nonrepresentable values; native forward still consumes BF16."""
        probe = self._probe(mode='lm_only', fp32_compressed_vjp=True,
                            fp32_native_query_vjp=True, fp32_rope_vjp=True)
        invocation, original = self._inputs(probe)
        inputs = tuple(value.detach().float().requires_grad_() for value in original[:4])
        decoded_query = torch.full_like(inputs[0], 2.003)
        decoded_rope = torch.full_like(inputs[2], 3.003)
        output, _loss = probe.forward(invocation, inputs, original[4:])
        with patch('hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training.scaled_query_workspace_vjps',
                   return_value=(decoded_query, decoded_rope)):
            gradients = torch.autograd.grad(output.sum(), inputs)
        torch.testing.assert_close(gradients[2], invocation.local.restore(decoded_rope), rtol=0, atol=0)
        torch.testing.assert_close(gradients[3], torch.full_like(inputs[3], 7.003)*2, rtol=0, atol=0)
        self.assertTrue(all(value.dtype == torch.float32 for value in gradients))
        self.assertTrue(all(value.dtype == torch.bfloat16 for value in self.forward.forward.call_args.args[0]))
        with self.assertRaisesRegex(ValueError, 'requires FP32 native query'):
            self._probe(fp32_rope_vjp=True)
        probe.fp32_rope_vjp = False
        with self.assertRaisesRegex(ValueError, 'objective was modified'):
            probe.check_declaration()
