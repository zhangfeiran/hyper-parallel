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
"""CPU objective scaling, isolation and saved-state contracts for local CP training."""

from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp, local_training
from hyper_parallel.core.multicore.modules.mega_dsa.local_kl import LocalKlResult
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture

_CHECK_NATIVE_SUPPORT = local_training.FusedDsaLocalCpTrainingProbe.check_native_support


class TestLocalTraining(SharedRootFixture):
    """Use real CPU owner permutations while mocking native arithmetic and transport."""

    def setUp(self) -> None:
        """Prepare a caller query permutation and a different owner key storage order."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "local-training", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0, 1), kv_global_ids=(1, 2, 0),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        self.backend = fused_cp.FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                                       attention_scale=.1, schedule=MixedSfaSchedule(7))
        self.key_partials = torch.arange(3*128).reshape(3, 128).float()/4096+.03137
        self.raw_calls = 0
        self.main_calls = 0
        patch.object(local_training.FusedDsaLocalCpTrainingProbe, "check_native_support").start()
        patch.object(self.backend, "forward_local", side_effect=self._forward).start()
        patch.object(local_training, "local_kl_gradients", side_effect=self._raw).start()

    def _inputs(self, ids=(2, 0), dtype=torch.float32):
        layout = self.backend.prepare_local_queries(ids)
        count = len(ids)
        shapes = ((count, 32, 512), (3, 512), (count, 32, 64), (3, 64),
                  (count, 64, 128), (3, 128), (count, 64))
        inputs = tuple(torch.ones(shape, dtype=dtype if field == 6 else torch.bfloat16, requires_grad=True)
                       for field, shape in enumerate(shapes))
        return layout, inputs

    def _forward(self, invocation, main, index, *, query_layout):
        self.assertTrue(all(not tensor.requires_grad for tensor in (*main, *index)))
        count = main[0].shape[0]
        states = (query_layout.pack(main[0]), torch.zeros(3, 512, dtype=torch.bfloat16),
                  query_layout.pack(main[2]), torch.zeros(3, 64, dtype=torch.bfloat16))
        indices = torch.full((count, 1, 2048), -1, dtype=torch.int32)
        output = states[0].clone()
        maximum, denominator = torch.zeros(1, count, 32), torch.ones(1, count, 32)
        saved_tensors = (*states, indices, output, maximum, denominator)
        meta = replace(invocation.batch_meta, q_global_ids=query_layout.batch_meta.q_global_ids)
        saved = fused_cp.FusedCpSavedAttention(meta, states, indices, (output, maximum, denominator),
                                               tuple(t._version for t in saved_tensors), self.backend,
                                               (32, .1, self.backend.schedule), fused_cp._SAVED_ATTENTION_TOKEN,
                                               "local", query_layout)
        packed_index = (query_layout.pack(index[0]), torch.zeros(3, 128, dtype=torch.bfloat16),
                        query_layout.pack(index[2]))
        return fused_cp.FusedCpForwardResult(query_layout.restore(output), indices.clone(), indices.clone(),
                                              maximum, denominator, (), (), torch.zeros(32, dtype=torch.long),
                                              1, saved, packed_index)

    def _raw(self, forward):
        self.raw_calls += 1
        count = forward.output.shape[0]
        iq = torch.full((count, 64, 128), 2, dtype=torch.bfloat16)
        weight = torch.full((count, 64), 5, dtype=forward.index_states[2].dtype)
        partials = self.key_partials.clone() if count else torch.zeros_like(self.key_partials)
        return LocalKlResult((iq, partials.bfloat16(), weight), partials, torch.tensor(7. if count else 0.),
                              (), torch.empty(0, dtype=torch.uint8))

    def _main_backward(self, saved, shape, cotangent, *, local_queries):
        self.main_calls += 1
        self.assertTrue(local_queries)
        self.assertEqual(shape, tuple(cotangent.shape))
        saved.validate_versions()
        factor = float(cotangent.flatten()[0]) if cotangent.numel() else 0
        rows = len(saved.batch_meta.q_global_ids)
        gradients = tuple(torch.full(shape, factor*value, dtype=torch.bfloat16)
                          for shape, value in zip(((rows, 32, 512), (3, 512), (rows, 32, 64), (3, 64)), (2, 3, 5, 7)))
        return SimpleNamespace(gradients=gradients)

    def _probe(self, coefficient=.7, mode="lm_plus_kl", divisor=1, fused_kl=False, sparse_main_return=False,
               sparse_index_return=False):
        probe = local_training.FusedDsaLocalCpTrainingProbe(
            self.backend, normalization=DsaLossNormalization(3, divisor), loss_coeff=coefficient, mode=mode,
            fused_kl=fused_kl, sparse_main_return=sparse_main_return, sparse_index_return=sparse_index_return)
        patch.object(probe.backward_backend, "_backward_saved", side_effect=self._main_backward).start()
        return probe

    def test_fused_kl_consumes_forward_derivatives_without_resubmission(self):
        """Saved FP32 key partials and caller-ordered Q/weight derivatives receive one upstream scale."""
        layout, inputs = self._inputs()
        probe = self._probe(fused_kl=True)
        def _fused_forward(invocation, main, index, *, query_layout, selected_kl):
            self.assertTrue(selected_kl)
            forward = self._forward(invocation, main, index, query_layout=query_layout)
            raw = self._raw(forward)
            return replace(forward, kl_gradients=raw.gradients, kl_loss=raw.loss,
                           kl_key_partials=raw.key_partials, kl_retained=raw.retained)
        with patch.object(self.backend, "forward_local", side_effect=_fused_forward) as submit, \
             patch.object(local_training, "local_kl_gradients", side_effect=AssertionError("second KL submission")):
            output, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            gradients = torch.autograd.grad(output.sum()+loss*-1.75, inputs)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(self.raw_calls, 1)
        scale = .7/3*-1.75
        expected = (self.key_partials*scale)[list(self.meta.kv_global_ids)].bfloat16()
        torch.testing.assert_close(gradients[5], expected, rtol=0, atol=0)
        torch.testing.assert_close(gradients[4], torch.full_like(inputs[4], 2*scale))
        torch.testing.assert_close(gradients[6], torch.full_like(inputs[6], 5*scale))

    def test_sparse_index_returns_scaled_fp32_without_dense_collective(self):
        """Empty query members still receive owner sums; coefficient and upstream apply before transport."""
        for ids in ((2, 0), ()):
            with self.subTest(ids=ids):
                layout, inputs = self._inputs(ids)
                probe = self._probe(divisor=2, sparse_index_return=True)
                received = self.key_partials[list(self.meta.kv_global_ids)] * .1937
                with patch.object(probe.index_return_backend, "return_owner",
                                  return_value=SimpleNamespace(owner_fp32=received)) as returned, \
                        patch.object(self.backend.layout, "reduce_local_index_gradients",
                                     side_effect=AssertionError("dense owner collective")):
                    output, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
                    actual = torch.autograd.grad(output.sum()+loss*-1.75, inputs)
                returned.assert_called_once()
                partial = returned.call_args.args[1]
                self.assertEqual(partial.dtype, torch.float32)
                self.assertFalse(partial.requires_grad)
                expected = self.key_partials*(.7*2/3)*-1.75 if ids else torch.zeros_like(self.key_partials)
                torch.testing.assert_close(partial, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[5], received.bfloat16(), rtol=0, atol=0)
                torch.testing.assert_close(actual[4], torch.full_like(inputs[4], 2*(.7*2/3)*-1.75))

    def test_sparse_index_capability_rejects_before_forward(self):
        """Unsupported index return cannot begin forward or fall back to a dense collective."""
        layout, inputs = self._inputs()
        probe = self._probe(mode="kl_only", sparse_index_return=True)
        probe.check_native_support = MethodType(_CHECK_NATIVE_SUPPORT, probe)
        with patch.object(probe.index_return_backend, "check_native_support",
                          side_effect=RuntimeError("index-K return requires adapter ABI 1")), \
                patch.object(self.backend, "forward_local") as forward:
            with self.assertRaisesRegex(RuntimeError, "index-K return requires"):
                probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            forward.assert_not_called()

    def test_sparse_index_zero_coefficient_skips_return(self):
        """A collectively zero KL coefficient needs no index transport and produces exact zeros."""
        layout, inputs = self._inputs()
        probe = self._probe(coefficient=0., mode="kl_only", sparse_index_return=True)
        with patch.object(probe.index_return_backend, "return_owner",
                          side_effect=AssertionError("zero KL entered transport")):
            _, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            actual = torch.autograd.grad(loss, inputs[4:])
        for value, original in zip(actual, inputs[4:]):
            torch.testing.assert_close(value, torch.zeros_like(original), rtol=0, atol=0)

    def test_modes_scaling_zero_coefficient_and_empty_query(self):
        """One coefficient, one global denominator and one upstream scale reach each contribution."""
        for mode in ("lm_only", "kl_only", "lm_plus_kl"):
            for coefficient in (0., .7):
                for ids in ((2, 0), ()):
                    for dtype in (torch.bfloat16, torch.float32):
                        for divisor in (1, 2):
                            with self.subTest(mode=mode, coefficient=coefficient, ids=ids,
                                              dtype=dtype, divisor=divisor):
                                layout, inputs = self._inputs(ids, dtype)
                                probe = self._probe(coefficient, mode, divisor)
                                output, loss = probe.forward(self.invocation, inputs[:4], inputs[4:],
                                                              query_layout=layout)
                                objective = loss * -1.75 if mode == "kl_only" else (output*3).sum()
                                if mode == "lm_plus_kl":
                                    objective = objective+loss*-1.75
                                actual = torch.autograd.grad(objective, inputs, allow_unused=True)
                                scale = coefficient*divisor/3*-1.75
                                if mode == "kl_only":
                                    self.assertTrue(all(value is None for value in actual[:4]))
                                else:
                                    for value, source, factor in zip(actual[:4], inputs[:4], (2, 3, 5, 7)):
                                        torch.testing.assert_close(value,
                                                                   torch.full_like(source, 3*factor if ids else 0))
                                if mode == "lm_only":
                                    self.assertTrue(all(value is None for value in actual[4:]))
                                    self.assertFalse(loss.requires_grad)
                                else:
                                    expected = (torch.full_like(inputs[4], 2*scale),
                                                (self.key_partials[[1, 2, 0]]*scale).bfloat16() if ids else
                                                torch.zeros_like(inputs[5]), torch.full_like(inputs[6], 5*scale))
                                    for value, target in zip(actual[4:], expected):
                                        torch.testing.assert_close(value, target,
                                                                   rtol=4*torch.finfo(torch.float32).eps, atol=0)
                                    torch.testing.assert_close(
                                        loss.detach(), torch.tensor(7*coefficient*divisor/3 if ids else 0.),
                                        rtol=4*torch.finfo(torch.float32).eps, atol=0)

    def test_zero_coefficient_skips_raw_arithmetic_and_auxiliary_owner_exchange(self):
        """A common cold zero coefficient keeps zero index VJPs without global FP32 scratch or exchange."""
        layout, inputs = self._inputs()
        probe = self._probe(0., "kl_only")
        with patch.object(self.backend.layout, "reduce_local_index_gradients") as reduce:
            _, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            actual = torch.autograd.grad(loss, inputs, allow_unused=True)
        reduce.assert_not_called()
        self.assertEqual(self.raw_calls, 0)
        self.assertTrue(all(tensor is None for tensor in actual[:4]))
        for value, source in zip(actual[4:], inputs[4:]):
            torch.testing.assert_close(value, torch.zeros_like(source), rtol=0, atol=0)

    def test_precast_key_plane_and_no_upstream_do_not_lose_owner_contributions(self):
        """FP32 key contributions survive auxiliary scaling before the owner dtype conversion."""
        layout, inputs = self._inputs()
        probe = self._probe(1.3, "kl_only")
        _, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
        key = torch.autograd.grad(loss, inputs[5])[0]
        expected = (self.key_partials[[1, 2, 0]]*(1.3/3)).bfloat16()
        rounded = (self.key_partials.bfloat16().float()[[1, 2, 0]]*(1.3/3)).bfloat16()
        torch.testing.assert_close(key, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(key, rounded))

    def test_unused_auxiliary_still_participates_in_same_owner_collective(self):
        """A member with no local auxiliary cotangent still enters the prepared index owner sum."""
        layout, inputs = self._inputs()
        probe = self._probe()
        with patch.object(self.backend.layout, "reduce_local_index_gradients",
                          wraps=self.backend.layout.reduce_local_index_gradients) as reduce:
            output, _ = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            actual = torch.autograd.grad(output.sum(), inputs)
        reduce.assert_called_once()
        self.assertTrue(all(torch.equal(value, torch.zeros_like(value)) for value in reduce.call_args.args[0]))
        self.assertTrue(all(torch.equal(value, torch.zeros_like(value)) for value in actual[4:]))

    def test_missing_local_auxiliary_cotangent_receives_remote_owner_key_gradient(self):
        """Only the local contribution becomes zero; incoming key-owner contributions remain differentiable."""
        layout, inputs = self._inputs()
        probe = self._probe()
        received = torch.full_like(inputs[5], 2.03125, dtype=torch.float32)
        with patch.object(self.backend.layout, "reduce_owner_gradients", return_value=(received,)) as reduce:
            output, _ = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            actual = torch.autograd.grad(output.sum(), inputs)
        reduce.assert_called_once()
        torch.testing.assert_close(reduce.call_args.args[0][0], torch.zeros(3, 128), rtol=0, atol=0)
        torch.testing.assert_close(actual[5], received.bfloat16(), rtol=0, atol=0)
        for field in (4, 6):
            torch.testing.assert_close(actual[field], torch.zeros_like(inputs[field]), rtol=0, atol=0)

    def test_missing_local_main_cotangent_receives_remote_main_owner_gradients(self):
        """LM+KL members enter main owner return even when their objective uses only local auxiliary loss."""
        layout, inputs = self._inputs()
        probe = self._probe()
        gradients = (torch.zeros_like(inputs[0]), torch.full_like(inputs[1], 3),
                     torch.zeros_like(inputs[2]), torch.full_like(inputs[3], 7))
        with patch.object(probe.backward_backend, "_backward_saved",
                          return_value=SimpleNamespace(gradients=gradients)) as backward:
            _, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            actual = torch.autograd.grad(loss, inputs)
        backward.assert_called_once()
        torch.testing.assert_close(backward.call_args.args[2], torch.zeros_like(inputs[0]), rtol=0, atol=0)
        for value, expected in zip(actual[:4], gradients):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_retained_backward_uses_owned_scaled_derivatives_after_reprepare(self):
        """Later forwards or query preparations cannot replace saved gradient values or tile identity."""
        layout, inputs = self._inputs()
        probe = self._probe()
        output, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
        objective = output.sum()+loss*-2
        first = torch.autograd.grad(objective, inputs, retain_graph=True)
        self.key_partials.add_(1)
        newer, other = self._inputs((1,))
        probe.forward(self.workspace.prepare(replace(self.meta, invocation=2)), other[:4], other[4:],
                      query_layout=newer)
        second = torch.autograd.grad(objective, inputs)
        for actual, expected in zip(second, first):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_checkpoint_replay_retains_all_eighteen_saved_tensors(self, _device):
        """Non-reentrant checkpoint rebuilds scaled gradients under the original cold layout."""
        layout, inputs = self._inputs()
        probe = self._probe()
        def _run(*states):
            return probe.forward(self.invocation, states[:4], states[4:], query_layout=layout)
        output, loss = checkpoint(_run, *inputs, use_reentrant=False)
        self.backend.prepare_local_queries(())
        actual = torch.autograd.grad(output.sum()+loss, inputs)
        self.assertEqual(self.raw_calls, 2)
        torch.testing.assert_close(actual[5], (self.key_partials[[1, 2, 0]]*(.7/3)).bfloat16(), rtol=0, atol=0)

    def test_mutation_higher_order_and_metadata_changes_reject_before_owner_return(self):
        """Original inputs and all saved activation metadata remain subject to first-order version guards."""
        for failure in ("main", "index", "layout", "geometry", "higher_order"):
            with self.subTest(failure=failure):
                layout, inputs = self._inputs()
                probe = self._probe()
                output, loss = probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
                if failure in ("main", "index"):
                    with torch.no_grad():
                        inputs[1 if failure == "main" else 5].add_(1)
                    error, message = RuntimeError, "modified by an inplace operation"
                elif failure == "layout":
                    layout.positions.add_(0)
                    error, message = ValueError, "metadata was modified"
                elif failure == "geometry":
                    self.backend.scale = .2
                    error, message = ValueError, "unchanged forward geometry"
                else:
                    error, message = ValueError, "first-order"
                with patch.object(self.backend.layout, "reduce_local_index_gradients") as reduce, \
                        self.assertRaisesRegex(error, message):
                    torch.autograd.grad(output.sum()+loss, inputs, create_graph=failure == "higher_order")
                reduce.assert_not_called()
                self.backend.scale = .1

    def test_invalid_kl_abi_and_mutated_objective_reject_before_forward(self):
        """Training preflight cannot enter a transport lease with unsupported KL or a changed cold contract."""
        layout, inputs = self._inputs()
        probe = self._probe(.7, "kl_only")
        probe.check_native_support = MethodType(_CHECK_NATIVE_SUPPORT, probe)
        with patch.object(local_training, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_mixed_kl_version", return_value=2, create=True), \
                patch.object(self.backend, "forward_local") as forward:
            with self.assertRaisesRegex(RuntimeError, "local KL adapter ABI"):
                probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            forward.assert_not_called()
        probe.loss_coeff = 1.
        with patch.object(self.backend, "forward_local") as forward:
            with self.assertRaisesRegex(ValueError, "objective was modified"):
                probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            forward.assert_not_called()

    def test_sparse_main_return_requires_capability_before_forward(self):
        """The bound main return and its native capability are checked before forward transport."""
        layout, inputs = self._inputs()
        probe = self._probe(0., "lm_only", sparse_main_return=True)
        self.assertTrue(probe.backward_backend.sparse_return)
        probe.check_native_support = MethodType(_CHECK_NATIVE_SUPPORT, probe)
        with patch.object(probe.backward_backend, "check_local_native_support"), \
                patch.object(local_training, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_cp_grad_version", return_value=0, create=True), \
                patch.object(self.backend, "forward_local") as forward:
            with self.assertRaisesRegex(RuntimeError, "sparse main-gradient return"):
                probe.forward(self.invocation, inputs[:4], inputs[4:], query_layout=layout)
            forward.assert_not_called()
        probe.backward_backend.sparse_return = False
        with self.assertRaisesRegex(ValueError, "main-gradient return was modified"):
            _CHECK_NATIVE_SUPPORT(probe)

    def test_invalid_and_disagreeing_cold_objective_contracts_reject(self):
        """Normalization, coefficient and mode disagreements fail before a training native lease."""
        invalid = ((DsaLossNormalization(2), 1., "kl_only"), (DsaLossNormalization(3), -1., "kl_only"),
                   (DsaLossNormalization(3), float("nan"), "kl_only"), (DsaLossNormalization(3), 1., "other"))
        for normalization, coefficient, mode in invalid:
            with self.subTest(normalization=normalization, coefficient=coefficient, mode=mode), \
                    self.assertRaises(ValueError):
                local_training.FusedDsaLocalCpTrainingProbe(self.backend, normalization=normalization,
                                                           loss_coeff=coefficient, mode=mode)
        with patch.object(self.backend.layout, "cp_size", 2), \
                patch.object(local_training.dist, "all_gather_object"):
            with self.assertRaisesRegex(ValueError, "CP members disagree"):
                self._probe()
