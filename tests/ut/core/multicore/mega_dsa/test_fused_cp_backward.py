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
"""Saved state, local permutation, FP32 owner restoration and leased CP backward."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp, fused_cp_backward
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestFusedCpBackward(SharedRootFixture):
    """Exercise the real workspace and backward wrapper over a mocked CPU native call."""

    def setUp(self) -> None:
        """Prepare independently owned saved tensors and different Q/K local orders."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "cp-gradient", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        meta = replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0, 1), kv_global_ids=(1, 2, 0),
                       heap_generation=self.root.generation)
        invocation = self.workspace.prepare(meta)
        self.forward_backend = fused_cp.FusedDsaCpForwardProbe(self.workspace, invocation, heads=32,
                                                              attention_scale=0.1, schedule=MixedSfaSchedule(7))
        states = tuple(torch.ones(shape, dtype=torch.bfloat16) for shape in
                       ((3, 32, 512), (3, 512), (3, 32, 64), (3, 64)))
        indices = torch.full((3, 1, 2048), -1, dtype=torch.int32)
        forward = (torch.ones_like(states[0]), torch.ones(1, 3, 32), torch.ones(1, 3, 32))
        tensors = (*states, indices, *forward)
        saved = fused_cp.FusedCpSavedAttention(meta, states, indices, forward,
                                               tuple(tensor._version for tensor in tensors), self.forward_backend,
                                               (32, 0.1, MixedSfaSchedule(7)), fused_cp._SAVED_ATTENTION_TOKEN)
        self.forward = fused_cp.FusedCpForwardResult(torch.ones_like(states[0]), indices.clone(), indices.clone(),
                                                     forward[1].clone(), forward[2].clone(), (), (),
                                                     torch.zeros(32, dtype=torch.int64), 1, saved)
        self.backend = fused_cp_backward.FusedDsaCpBackwardProbe(self.forward_backend)

    def _local_forward(self, ids):
        self.forward_backend.prepare_local_queries(ids)
        layout = self.forward_backend.local_query_layout
        meta = replace(self.forward.saved.batch_meta, q_global_ids=ids)
        states = (self.forward.saved.states[0][:len(ids)].clone(), self.forward.saved.states[1].clone(),
                  self.forward.saved.states[2][:len(ids)].clone(), self.forward.saved.states[3].clone())
        indices = torch.full((len(ids), 1, 2048), -1, dtype=torch.int32)
        forward = (torch.ones_like(states[0]), torch.zeros(1, len(ids), 32), torch.ones(1, len(ids), 32))
        tensors = (*states, indices, *forward)
        saved = fused_cp.FusedCpSavedAttention(meta, states, indices, forward,
                                               tuple(tensor._version for tensor in tensors), self.forward_backend,
                                               (32, 0.1, MixedSfaSchedule(7)), fused_cp._SAVED_ATTENTION_TOKEN,
                                               "local", layout)
        output = layout.restore(forward[0]) if ids else forward[0]
        return replace(self.forward, output=output, saved=saved)

    def test_local_backward_keeps_real_q_and_complete_key_partials_after_reprepare(self):
        """A saved subset restores caller Q and owner K orders without query replication."""
        for ids in ((2, 0), ()):
            with self.subTest(ids=ids):
                forward = self._local_forward(ids)
                layout = forward.saved.local_query_layout
                self.forward_backend.prepare_local_queries((1,))
                cotangent = torch.tensor([5, 7][:len(ids)], dtype=torch.bfloat16)
                cotangent = cotangent[:, None, None].expand(-1, 32, 512).clone()
                epochs = []

                def _execute(*args):
                    self.assertEqual(len(args), 28)
                    self.assertIs(self.root.lease_owner, self.workspace.consumer)
                    self.assertEqual(args[0].shape[0], len(ids))
                    self.assertEqual(args[1].shape[0], 3)
                    self.assertEqual(args[26].shape, (3, 1088))
                    self.assertIs(args[8], layout.query_lengths)
                    self.assertIs(args[27], layout.positions)
                    torch.testing.assert_close(args[4], layout.pack(cotangent))
                    if not ids:
                        self.assertEqual(args[14].numel(), 13312)
                    epochs.append(int(args[22][2]))
                    args[16].copy_(args[4] * 2)
                    args[19].fill_(3)
                    args[17].fill_(999)
                    args[18].fill_(777)
                    args[25].copy_(torch.arange(3, dtype=torch.float32)[:, None].expand(3, 576) + .001)
                    args[26].fill_(17)

                with patch.object(fused_cp_backward, "_load_native"), \
                        patch.object(torch.ops.hyper_parallel, "dsa_local_cp_grad_version", return_value=1,
                                     create=True), \
                        patch.object(torch.ops.hyper_parallel, "dsa_local_cp_grad_out", new=_execute, create=True):
                    first = self.backend.backward_local(forward, cotangent)
                    second = self.backend.backward_local(forward, cotangent)
                self.assertEqual(epochs[1], epochs[0] + 1)
                self.assertEqual(int(first._runtime_buffers[0][2]), epochs[0])
                self.assertIsNot(first._runtime_buffers[0], second._runtime_buffers[0])
                torch.testing.assert_close(first.gradients[0], cotangent * 2)
                torch.testing.assert_close(first.owner_fp32[0][:, 0], torch.tensor([1.001, 2.001, .001]))
                torch.testing.assert_close(first.gradients[1], first.owner_fp32[0].bfloat16())
                self.assertEqual(first.owner_fp32[0].dtype, torch.float32)
                self.assertNotEqual(first.partials.data_ptr(), second.partials.data_ptr())
                self.assertIsNone(self.root.lease_owner)
                forward.saved.validate_versions()

    def test_local_admission_rejects_mutated_layout_and_stale_payload_before_lease(self):
        """Native availability cannot bypass saved positions or consume an epoch on failure."""
        forward = self._local_forward((2, 0))
        before = self.workspace.transport_epoch
        with patch.object(fused_cp_backward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_grad_version", return_value=0, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_local_cp_grad_out", create=True) as execute:
            with self.assertRaisesRegex(RuntimeError, "adapter ABI 1"):
                self.backend.backward_local(forward, forward.output)
            execute.assert_not_called()
        self.assertEqual(self.workspace.transport_epoch, before)
        self.assertIsNone(self.root.lease_owner)
        forward.saved.local_query_layout.positions.add_(0)
        with patch.object(fused_cp_backward, "_load_native") as load:
            with self.assertRaisesRegex(ValueError, "metadata was modified"):
                self.backend.backward_local(forward, forward.output)
            load.assert_not_called()

    def test_cotangent_and_owner_gradients_restore_independent_q_k_orders(self):
        """Native receives global Q cotangents while FP32 owner results follow original KV order."""
        cotangent = torch.tensor([5, 7, 11], dtype=torch.bfloat16)[:, None, None].expand(3, 32, 512).clone()
        epochs = []

        def _execute(*args):
            self.assertIs(self.root.lease_owner, self.workspace.consumer)
            self.assertEqual(self.workspace.active_invocation[-1], "backward")
            torch.testing.assert_close(args[4][:, 0, 0], torch.tensor([7, 11, 5], dtype=torch.bfloat16))
            self.assertIs(args[21], self.workspace.arena)
            epochs.append(int(args[22][2]))
            args[16].copy_(args[4] * 2)
            args[19].fill_(3)
            args[25].copy_(torch.arange(3, dtype=torch.float32)[:, None].expand(3, 576) + 0.001)
            args[26].fill_(17)

        with patch.object(fused_cp_backward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_grad_version", return_value=2, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_grad_out", side_effect=_execute, create=True):
            first = self.backend.backward(self.forward, cotangent)
            self.workspace.buffers["source_compressed"].fill_(-99)
            second = self.backend.backward(self.forward, cotangent)
        self.assertEqual(epochs, [1, 2])
        self.assertIsNone(self.root.lease_owner)
        torch.testing.assert_close(first.gradients[0][:, 0, 0], cotangent[:, 0, 0] * 2)
        torch.testing.assert_close(first.owner_fp32[0][:, 0], torch.tensor([1.001, 2.001, 0.001]))
        self.assertEqual(first.owner_fp32[0].dtype, torch.float32)
        torch.testing.assert_close(first.gradients[1], first.owner_fp32[0].bfloat16())
        self.assertNotEqual(first.retained.data_ptr(), second.retained.data_ptr())
        self.assertNotEqual(first.partials.data_ptr(), second.partials.data_ptr())
        self.forward.saved.validate_versions()

    def test_mutated_or_foreign_saved_state_rejects_before_native_dispatch(self):
        """Delay and retain_graph cannot silently consume mutated keys or another adapter's state."""
        with patch.object(fused_cp_backward, "_load_native") as load:
            foreign = replace(self.forward.saved, backend=object())
            with self.assertRaisesRegex(ValueError, "this backend"):
                self.backend.backward(replace(self.forward, saved=foreign), self.forward.output)
            self.forward.saved.states[1].add_(1)
            with self.assertRaisesRegex(ValueError, "modified after forward"):
                self.backend.backward(self.forward, self.forward.output)
            load.assert_not_called()

    def test_saved_admission_and_geometry_cannot_be_reinterpreted(self):
        """Reject manually synthesized state and scale changes before invoking any native code."""
        with patch.object(fused_cp_backward, "_load_native") as load:
            synthetic = replace(self.forward.saved, _token=None)
            with self.assertRaisesRegex(ValueError, "forward-produced"):
                self.backend.backward(replace(self.forward, saved=synthetic), self.forward.output)
            self.forward_backend.scale = 0.2
            with self.assertRaisesRegex(ValueError, "geometry/scale/schedule"):
                self.backend.backward(self.forward, self.forward.output)
            load.assert_not_called()

    def test_abi_and_cotangent_admission_do_not_claim_the_workspace(self):
        """Reject unsupported gradients and stale native payloads before entering a lease."""
        with self.assertRaisesRegex(ValueError, "BF16 output cotangent"):
            self.backend.backward(self.forward, self.forward.output.float())
        with patch.object(fused_cp_backward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_grad_version", return_value=0, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_grad_out", create=True) as execute:
            with self.assertRaisesRegex(RuntimeError, "adapter ABI"):
                self.backend.backward(self.forward, self.forward.output)
            execute.assert_not_called()
        self.assertIsNone(self.root.lease_owner)

    def test_autograd_main_inputs_and_indexer_isolation_survive_retained_backward(self):
        """The bridge detaches native inputs and returns each of four main gradients exactly once."""
        backend = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.forward.saved.states)
        index = tuple(torch.ones(1, requires_grad=True) for _ in range(3))
        invocation = self.workspace.prepare(self.forward.saved.batch_meta)

        def _forward(_invocation, main_states, index_states):
            self.assertTrue(all(not tensor.requires_grad for tensor in (*main_states, *index_states)))
            return self.forward

        gradients = tuple(torch.full_like(tensor, factor) for factor, tensor in enumerate(main, start=2))
        with patch.object(self.forward_backend, "forward", side_effect=_forward), \
                patch.object(backend.backward_backend, "_backward_saved",
                             return_value=SimpleNamespace(gradients=gradients)):
            output = backend.attention(invocation, main, index)
            first = torch.autograd.grad(output.sum(), main, retain_graph=True)
            second = torch.autograd.grad(output.sum(), main)
        for actual, replay, expected in zip(first, second, gradients):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(replay, expected)
        self.assertTrue(all(tensor.grad is None for tensor in index))

    def test_local_autograd_retains_original_layout_and_returns_only_main_gradients(self):
        """Repeated backward uses the saved local ABI after another query tile is prepared."""
        bridge = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        for ids in ((2, 0), ()):
            with self.subTest(ids=ids):
                result = self._local_forward(ids)
                layout = result.saved.local_query_layout
                main = tuple(tensor.clone().requires_grad_() for tensor in result.saved.states)
                index = tuple(torch.ones(1, requires_grad=True) for _ in range(3))
                gradients = tuple(torch.full_like(tensor, factor if ids else 0)
                                  for factor, tensor in enumerate(main, start=2))
                invocation = self.workspace.prepare(result.saved.batch_meta)

                def _forward(_invocation, states, index_states, *, query_layout):
                    self.assertIs(query_layout, layout)
                    self.assertTrue(all(not tensor.requires_grad for tensor in (*states, *index_states)))
                    return result

                def _backward(saved, shape, cotangent, *, local_queries):
                    self.assertTrue(local_queries)
                    self.assertEqual(shape, tuple(result.output.shape))
                    self.assertFalse(cotangent.requires_grad)
                    self.assertIs(saved.local_query_layout, layout)
                    self.assertTrue(all(not tensor.requires_grad for tensor in
                                        (*saved.states, saved.indices, *saved.forward)))
                    saved.validate_versions()
                    return SimpleNamespace(gradients=gradients)

                with patch.object(self.forward_backend, "forward_local", side_effect=_forward), \
                        patch.object(bridge.backward_backend, "_backward_saved", side_effect=_backward):
                    output = bridge.attention_local(invocation, main, index, query_layout=layout)
                    self.forward_backend.prepare_local_queries((1,))
                    first = torch.autograd.grad(output.sum(), (*main, *index), retain_graph=True, allow_unused=True)
                    second = torch.autograd.grad(output.sum(), main)
                for actual, replay, expected in zip(first[:4], second, gradients):
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(replay, expected)
                self.assertTrue(all(tensor is None for tensor in first[4:]))

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_local_checkpoint_replay_uses_explicit_layout_after_reprepare(self, _device_type):
        """The checkpoint callback retains cold tile identity while native activations use saved hooks."""
        bridge = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        result = self._local_forward((2, 0))
        layout = result.saved.local_query_layout
        main = tuple(tensor.clone().requires_grad_() for tensor in result.saved.states)
        index = (torch.ones(1),) * 3
        invocation = self.workspace.prepare(result.saved.batch_meta)
        gradients = tuple(torch.full_like(tensor, factor) for factor, tensor in enumerate(main, start=2))

        def _forward(*_args, query_layout):
            self.assertIs(query_layout, layout)
            return replace(result, output=result.output.detach().clone())

        def _attention(*states):
            return bridge.attention_local(invocation, states, index, query_layout=layout)

        with patch.object(self.forward_backend, "forward_local", side_effect=_forward) as forward, \
                patch.object(bridge.backward_backend, "_backward_saved",
                             return_value=SimpleNamespace(gradients=gradients)):
            output = checkpoint(_attention, *main, use_reentrant=False)
            self.assertFalse(hasattr(output.grad_fn, "result"))
            self.forward_backend.prepare_local_queries(())
            actual = torch.autograd.grad(output.sum(), main)
        self.assertEqual(forward.call_count, 2)
        for value, expected in zip(actual, gradients):
            torch.testing.assert_close(value, expected)

    def test_local_autograd_rejects_input_mutation_and_higher_order_before_native(self):
        """Detaching the native ABI must retain original input version and first-order protections."""
        bridge = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        for mode in ("main", "index", "higher_order", "layout"):
            with self.subTest(mode=mode):
                result = self._local_forward((2, 0))
                main = tuple(tensor.clone().requires_grad_() for tensor in result.saved.states)
                index = tuple(torch.ones(1, requires_grad=True) for _ in range(3))
                layout = result.saved.local_query_layout
                invocation = self.workspace.prepare(result.saved.batch_meta)
                with patch.object(self.forward_backend, "forward_local", return_value=result), \
                        patch.object(fused_cp_backward, "_load_native") as load:
                    output = bridge.attention_local(invocation, main, index, query_layout=layout)
                    if mode == "higher_order":
                        with self.assertRaisesRegex(ValueError, "first-order"):
                            torch.autograd.grad(output.sum(), main, create_graph=True)
                    elif mode == "layout":
                        layout.positions.add_(0)
                        with self.assertRaisesRegex(ValueError, "metadata was modified"):
                            torch.autograd.grad(output.sum(), main)
                    else:
                        with torch.no_grad():
                            (main[1] if mode == "main" else index[0]).add_(1)
                        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                            torch.autograd.grad(output.sum(), main)
                    load.assert_not_called()

    def test_autograd_rejects_original_input_mutation_before_native_backward(self):
        """Versioned original inputs remain protected even though native saved buffers are independent."""
        backend = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.forward.saved.states)
        invocation = self.workspace.prepare(self.forward.saved.batch_meta)
        with patch.object(self.forward_backend, "forward", return_value=self.forward), \
                patch.object(backend.backward_backend, "_backward_saved") as backward:
            output = backend.attention(invocation, main, (torch.ones(1),) * 3)
            with torch.no_grad():
                main[1].add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                output.sum().backward()
            backward.assert_not_called()

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_non_reentrant_checkpoint_recomputes_without_context_tensor_side_storage(self, _device_type):
        """Native activations use saved-tensor hooks so checkpoint can rebuild the first-order VJP."""
        backend = fused_cp_backward.FusedDsaCpAttentionProbe(self.forward_backend)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.forward.saved.states)
        index = (torch.ones(1),) * 3
        invocation = self.workspace.prepare(self.forward.saved.batch_meta)
        gradients = tuple(torch.full_like(tensor, factor) for factor, tensor in enumerate(main, start=2))

        def _forward(*_args):
            return replace(self.forward, output=self.forward.output.detach().clone())

        def _attention(*states):
            return backend.attention(invocation, states, index)

        with patch.object(self.forward_backend, "forward", side_effect=_forward) as forward, \
                patch.object(backend.backward_backend, "_backward_saved",
                             return_value=SimpleNamespace(gradients=gradients)):
            output = checkpoint(_attention, *main, use_reentrant=False)
            self.assertFalse(hasattr(output.grad_fn, "result"))
            actual = torch.autograd.grad(output.sum(), main)
        self.assertEqual(forward.call_count, 2)
        for value, expected in zip(actual, gradients):
            torch.testing.assert_close(value, expected)

    @staticmethod
    def _phase_traces():
        traces = []
        for epoch in (1, 2, 3):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(7):
                tasks = tuple(range(group, 20, 7))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 4] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1]])
                    trace[group, offset + 6] = epoch
            trace[7, [6, 21, 22, 24, 38]] = epoch
            trace[7, 25] = 21
            traces.append(trace)
        return traces

    def test_sparse_mode_is_collectively_declared_and_versioned(self):
        """Malformed and inconsistent cold modes cannot enter a native gradient lease."""
        for invalid in (None, 0, 1, "sparse"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                fused_cp_backward.FusedDsaCpBackwardProbe(self.forward_backend, sparse_return=invalid)
        def _disagree(peers, declaration, **_kwargs):
            peers[:] = [declaration, (False, None)]
        with patch.object(self.forward_backend.layout, "cp_size", 2), \
                patch.object(fused_cp_backward.dist, "all_gather_object", side_effect=_disagree), \
                self.assertRaisesRegex(ValueError, "disagree"):
            fused_cp_backward.FusedDsaCpBackwardProbe(self.forward_backend, sparse_return=True)
        sparse = fused_cp_backward.FusedDsaCpBackwardProbe(self.forward_backend, sparse_return=True)
        meta = sparse._metadata(self.forward.saved.batch_meta, 7)
        self.assertEqual(int(meta[1]), 2)
        with patch.object(fused_cp_backward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_cp_grad_version", return_value=0, create=True), \
                patch.object(self.workspace, "lease") as lease, self.assertRaisesRegex(RuntimeError, "adapter ABI 1"):
            sparse.backward(self.forward, torch.ones_like(self.forward.output))
        lease.assert_not_called()
        sparse.sparse_return = False
        with self.assertRaisesRegex(ValueError, "mode was modified"):
            sparse.backward(self.forward, torch.ones_like(self.forward.output))

    def test_sparse_trace_requires_row_receipts_and_clear_extent(self):
        """A sparse receipt proves its payload counts, deterministic owner extent and full completion."""
        sparse = fused_cp_backward.FusedDsaCpBackwardProbe(self.forward_backend, sparse_return=True)
        traces = self._phase_traces()
        transport = torch.zeros(32, dtype=torch.int64)
        transport[:19] = torch.tensor([2, 2, 1, 1, 2816, 0, 2816, 0, 100, 120, 130, 150,
                                      1, 2, 1, 0, 0, 0, 6912])
        transport[20:22] = torch.tensor([512, 7424])
        result = fused_cp_backward.FusedCpBackwardResult((), (), (), torch.empty(0), (), transport,
                                                         torch.empty(13568, dtype=torch.uint8), 2)
        proof = sparse.validate_trace(result, tuple(traces), transport)
        self.assertEqual(proof["sparse_return"]["skipped_zero_rows"], 2)
        self.assertEqual(proof["remote_sent_bytes"], 0)
        for word, value in ((0, 1), (1, 1), (2, 0), (3, 0), (4, 2817), (5, 1), (6, 0), (7, 2816),
                            (12, 4), (13, 0), (14, 4), (15, 1), (16, 8), (17, 8), (18, 0), (19, 1), (31, 1)):
            broken = transport.clone()
            broken[word] = value
            with self.subTest(word=word, value=value), self.assertRaises(ValueError):
                sparse.validate_trace(result, tuple(traces), broken)
        with self.assertRaises(ValueError):
            self.backend.validate_trace(result, tuple(traces), transport)

    def test_trace_rejects_missing_phases_and_incomplete_owner_receipts(self):
        """Independent synthetic phase records require full transport completion and valid scratch offsets."""
        traces = self._phase_traces()
        transport = torch.zeros(32, dtype=torch.int64)
        transport[:12] = torch.tensor([2, 2, 1, 1, 8448, 0, 8448, 0, 100, 120, 130, 150])
        transport[20:22] = torch.tensor([512, 7424])
        result = fused_cp_backward.FusedCpBackwardResult((), (), (), torch.empty(0), (), transport,
                                                         torch.empty(13568, dtype=torch.uint8), 2)
        proof = self.backend.validate_trace(result, tuple(traces), transport)
        self.assertTrue(proof["ready_write_ack"])
        for word, value in ((0, 1), (1, 1), (2, 0), (3, 0), (4, 0), (5, 1), (6, 0), (7, 1),
                            (8, 0), (9, 100), (10, 119), (11, 130), (12, 1), (19, 1),
                            (20, -1), (21, 6000), (22, 1), (31, 1)):
            broken = transport.clone()
            broken[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.backend.validate_trace(result, tuple(traces), broken)
        with self.assertRaisesRegex(ValueError, "phase snapshots"):
            self.backend.validate_trace(result, tuple(traces[:2]), transport)
