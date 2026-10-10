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
"""External selection admission, packed compaction and CP core autograd contracts."""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp, fused_cp_backward, module
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestMegaDsaCore(SharedRootFixture):
    """Use real CPU ownership/workspace preparation and only mock the native NPU operation."""

    def setUp(self) -> None:
        """Prepare two packed sequences and different local Q/K permutations."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "external-core", DsaWorkspaceSpec(4, 4, 4, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((2, 2)), q_global_ids=(3, 0, 2, 1), kv_global_ids=(1, 3, 0, 2),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        self.core = module.MegaDsaCore(self.workspace, self.invocation, heads=32,
                                       attention_scale=0.1, schedule=MixedSfaSchedule(7))
        self.indices = torch.full((4, 2048), -1, dtype=torch.int32)
        self.indices[:, :5] = torch.tensor([[3, -1, 0, 2, 1], [1, -1, 2, 0, 3],
                                            [0, -1, 3, 2, 1], [2, -1, 0, 1, 3]], dtype=torch.int32)
        self.states = tuple(torch.ones(shape, dtype=torch.bfloat16) for shape in
                            ((4, 32, 512), (4, 512), (4, 32, 64), (4, 64)))

    def test_selection_compacts_padding_and_masks_packed_causal_ids(self):
        """External IDs use global query positions rather than local row numbers or KV order."""
        selection = self.core.prepare_selection(self.indices)
        expected = torch.tensor([[3, 2, -1], [0, -1, -1], [2, -1, -1], [0, 1, -1]], dtype=torch.int32)
        torch.testing.assert_close(selection.to_global_indices()[:, :3], expected)
        self.indices.fill_(0)
        exported = selection.to_global_indices()
        exported.fill_(99)
        torch.testing.assert_close(selection.to_global_indices()[:, :3], expected)
        native = selection.native_indices(self.core.backend)
        native.fill_(99)
        torch.testing.assert_close(selection.to_global_indices()[:, :3], expected)
        self.assertEqual(self.core.state_dict(), {})

    def test_bad_selection_rejects_before_gather_or_native(self):
        """Reject unknown IDs, duplicates including masked IDs, device rank and wrong K."""
        candidates = [self.indices.float(), self.indices[:, :8], self.indices[None]]
        for token in (-2, 4):
            value = self.indices.clone()
            value[0, 0] = token
            candidates.append(value)
        duplicate = self.indices.clone()
        duplicate[1, 5] = 3
        candidates.append(duplicate)
        with patch.object(module, "_load_native") as load:
            for value in candidates:
                with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                    self.core.prepare_selection(value)
            with self.assertRaisesRegex(ValueError, "prepare_selection"):
                module.MegaDsaCoreSelection(self.core.backend, self.indices)
            load.assert_not_called()
        self.assertIsNone(self.root.lease_owner)

    def _execute(self, *args):
        self.assertIs(self.root.lease_owner, self.workspace.consumer)
        self.assertEqual(self.workspace.active_invocation[-1], "forward")
        self.assertIs(args[8], self.workspace.arena)
        args[1][:, 0].copy_(torch.arange(4)[:, None].expand(4, 512))
        args[3].fill_(2)
        args[13].copy_(args[0] * 3)
        args[14].fill_(5)
        args[15].fill_(7)

    def _native_patches(self):
        return (patch.object(module, "_load_native"),
                patch.object(torch.ops.hyper_parallel, "dsa_cp_attention_version", return_value=2, create=True),
                patch.object(torch.ops.hyper_parallel, "dsa_cp_attention_out", side_effect=self._execute, create=True))

    def test_raw_forward_owns_selection_and_publishes_only_main_sources(self):
        """Prepared indices and saved KV survive source republication and selection exports."""
        selection = self.core.prepare_selection(self.indices)
        self.workspace.buffers["source_index"].fill_(91)
        load, version, execute = self._native_patches()
        with load, version, execute:
            first = self.core.raw_forward(self.invocation, self.states, selection)
            second = self.core.raw_forward(self.invocation, self.states, selection)
        self.assertEqual((first.epoch, second.epoch), (1, 2))
        self.assertEqual(int(self.workspace.buffers["source_index"][0, 0]), 91)
        self.assertNotEqual(first.saved.indices.data_ptr(), second.saved.indices.data_ptr())
        self.workspace.buffers["source_compressed"].fill_(-19)
        first.saved.validate_versions()
        torch.testing.assert_close(first.output, self.states[0] * 3)
        torch.testing.assert_close(first.saved.states[1][:, 0], torch.arange(4).bfloat16())

    def test_old_forward_payload_rejects_before_workspace_lease(self):
        """An old forward adapter cannot silently retain finite empty-set sentinels."""
        selection = self.core.prepare_selection(torch.full_like(self.indices, -1))
        with patch.object(module, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_cp_attention_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_cp_attention_out", create=True) as execute:
            with self.assertRaisesRegex(RuntimeError, "ABI 2"):
                self.core.raw_forward(self.invocation, self.states, selection)
            execute.assert_not_called()
        self.assertIsNone(self.root.lease_owner)
        self.assertEqual(self.workspace.transport_epoch, 0)

    def test_forward_rejects_changed_layout_and_foreign_selection_before_load(self):
        """Prepared ownership and local storage orders cannot change under a live backend."""
        selection = self.core.prepare_selection(self.indices)
        with patch.object(module, "_load_native") as load:
            changed = replace(self.invocation, batch_meta=replace(self.meta, q_global_ids=(0, 1, 2, 3)))
            with self.assertRaisesRegex(ValueError, "local orders"):
                self.core.raw_forward(changed, self.states, selection)
            with self.assertRaisesRegex(TypeError, "admitted"):
                self.core.raw_forward(self.invocation, self.states, self.indices)
            selection._indices.add_(1)
            with self.assertRaisesRegex(ValueError, "modified"):
                self.core.raw_forward(self.invocation, self.states, selection)
            load.assert_not_called()

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_autograd_saved_hooks_recompute_without_hidden_tensor_cache(self, _device_type):
        """Checkpoint and retain_graph return exactly four gradients with immutable native stats."""
        complete = torch.full_like(self.indices, -1)
        for row, query in enumerate(self.meta.q_global_ids):
            sequence, position = self.meta.sequence_position(query)
            start = self.meta.global_cu_seqlens[sequence]
            complete[row, :position + 1] = torch.arange(start, query + 1, dtype=torch.int32)
        selection = self.core.prepare_selection(complete)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.states)
        gradients = tuple(torch.full_like(tensor, index) for index, tensor in enumerate(main, 2))
        load, version, execute = self._native_patches()
        calls = []

        def _backward(saved, shape, _cotangent):
            saved.validate_versions()
            calls.append(shape)
            return SimpleNamespace(gradients=gradients)

        with load, version, execute, patch.object(self.core.backward_backend, "check_native_support"), \
                patch.object(self.core.backward_backend, "_backward_saved", side_effect=_backward):
            out, stats = self.core(*main, selection, self.meta)
            self.assertFalse(stats.maximum.requires_grad)
            self.assertFalse(stats.denominator.requires_grad)
            first = torch.autograd.grad(out.sum(), main, retain_graph=True)
            second = torch.autograd.grad(out.sum(), main)
            recomputed = checkpoint(lambda *values: self.core(*values, selection, self.meta)[0],
                                    *main, use_reentrant=False)
            third = torch.autograd.grad(recomputed.sum(), main)
        for result in (first, second, third):
            for actual, expected in zip(result, gradients):
                torch.testing.assert_close(actual, expected)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.workspace.transport_epoch, 3)

    def test_old_gradient_payload_rejects_padded_training_before_forward_dispatch(self):
        """ABI 1 cannot silently enter training after admission of a compacted padded selection."""
        incomplete = torch.full_like(self.indices, -1)
        incomplete[0, 0] = 3
        selection = self.core.prepare_selection(incomplete)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.states)
        with patch.object(self.core, "raw_forward") as forward, \
                patch.object(fused_cp_backward, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_grad_version", return_value=1, create=True):
            with self.assertRaisesRegex(RuntimeError, "ABI 2"):
                self.core(*main, selection, self.meta)
            forward.assert_not_called()

    def test_padded_and_empty_selections_enter_training_with_current_gradient_abi(self):
        """Compacted subsets and empty rows preserve admission and all four gradient slots."""
        for empty in (False, True):
            with self.subTest(empty=empty):
                indices = torch.full_like(self.indices, -1)
                if not empty:
                    indices[0, 3] = 3
                selection = self.core.prepare_selection(indices)
                main = tuple(tensor.clone().requires_grad_() for tensor in self.states)
                gradients = tuple(torch.zeros_like(tensor) for tensor in main)
                load, version, execute = self._native_patches()

                def _backward(saved, _shape, _cotangent):
                    saved.validate_versions()
                    self.assertEqual(int((saved.indices >= 0).sum()), 0 if empty else 1)
                    return SimpleNamespace(gradients=gradients)

                with load, version, execute, patch.object(fused_cp_backward, "_load_native"), \
                        patch.object(torch.ops.hyper_parallel, "dsa_fused_grad_version", return_value=2, create=True), \
                        patch.object(self.core.backward_backend, "_backward_saved", side_effect=_backward):
                    output, stats = self.core(*main, selection, self.meta)
                    actual = torch.autograd.grad(output.sum(), main)
                for value, expected in zip(actual, gradients):
                    torch.testing.assert_close(value, expected)
                self.assertFalse(stats.maximum.requires_grad)
                self.assertFalse(stats.denominator.requires_grad)


class TestMegaDsaTraining(SharedRootFixture):
    """Check seven-gradient ownership, isolated objectives and hook-managed precomputed KL."""

    def setUp(self) -> None:
        """Prepare distinct Q/K orders and the locked BF16 merge-weight producer."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "joint-dsa", DsaWorkspaceSpec(4, 4, 4, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((2, 2)), q_global_ids=(3, 0, 2, 1), kv_global_ids=(1, 3, 0, 2),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        self.model = module.MegaDsa(self.workspace, self.invocation, heads=32, attention_scale=.1,
                                    schedule=MixedSfaSchedule(7), normalization=DsaLossNormalization(4), loss_coeff=.25,
                                    kl_backend="reference")
        self.inputs = tuple(torch.ones(shape, dtype=torch.bfloat16, requires_grad=True)
                            for shape in self.model.input_shapes)
        self.raw_index_gradients = tuple((torch.arange(4, dtype=tensor.dtype) + 1).reshape(
            4, *(1 for _ in tensor.shape[1:])).expand_as(tensor).clone() * factor
            for tensor, factor in zip(self.inputs[4:], (1, 10, 100)))
        self.main_gradients = tuple(torch.full_like(tensor, factor)
                                    for tensor, factor in zip(self.inputs[:4], (2, 3, 4, 5)))

    def _execute(self, *args):
        self.assertIs(self.root.lease_owner, self.workspace.consumer)
        for index in (1, 3, 5):
            args[index].fill_(1)
        args[12].fill_(-1)
        args[12][:, 0, 0] = 0
        args[13].fill_(0)
        args[14].copy_(args[2] * 3)
        args[15].fill_(0)
        args[16].fill_(1)

    def _execute_training(self, *args):
        self.assertIs(self.root.lease_owner, self.workspace.consumer)
        self.assertEqual(args[9].shape, (6, 20, 64))
        self.assertEqual(args[12], (2, 4))
        for index in (1, 3, 5):
            args[index].fill_(1)
        args[14].fill_(-1)
        args[14][:, 0, 0] = 0
        args[15].fill_(0)
        args[16].copy_(args[2] * 3)
        args[17].fill_(0)
        args[18].fill_(1)
        args[19].copy_(self.raw_index_gradients[0])
        args[20].copy_(self.raw_index_gradients[1][:, None])
        args[21].copy_(self.raw_index_gradients[2].to(args[21].dtype))
        args[22].fill_(8)

    @contextmanager
    def _patches(self):
        with ExitStack() as stack:
            for target in (fused_cp, fused_cp_backward):
                stack.enter_context(patch.object(target, "_load_native"))
            for name in ("dsa_fused_forward_version", "dsa_fused_grad_version"):
                stack.enter_context(patch.object(torch.ops.hyper_parallel, name, return_value=2, create=True))
            stack.enter_context(patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_forward_out",
                                            side_effect=self._execute, create=True))
            stack.enter_context(patch.object(torch.ops.hyper_parallel, "dsa_fused_training_version",
                                            return_value=1, create=True))
            stack.enter_context(patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_training_out",
                                            side_effect=self._execute_training, create=True))
            kl = stack.enter_context(patch.object(module, "_selected_kl_gradients", return_value=(
                *self.raw_index_gradients, torch.tensor(8.))))
            backward = stack.enter_context(patch.object(self.model.core.backward_backend, "_backward_saved",
                                                        return_value=SimpleNamespace(gradients=self.main_gradients)))
            yield kl, backward

    def _assert_index_gradients(self, actual, upstream=7):
        for index, (value, global_gradient) in enumerate(zip(actual, self.raw_index_gradients)):
            ids = self.meta.kv_global_ids if index == 1 else self.meta.q_global_ids
            expected = (global_gradient.to(self.inputs[index + 4].dtype) * (.25 / 4 * upstream))[list(ids)]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            self.assertEqual(value.dtype, self.inputs[index + 4].dtype)

    def test_joint_forward_saves_one_kl_evaluation_and_returns_all_seven_gradients(self):
        """The signed pre-scaled index inputs feed native KL once, with original owner orders."""
        with self._patches() as (kl, backward):
            output, loss = self.model(*self.inputs, self.meta)
            self.assertEqual(float(loss.detach()), .5)
            actual = torch.autograd.grad(output.sum() + loss * 7, self.inputs)
        kl.assert_called_once()
        backward.assert_called_once()
        self.assertFalse(any(tensor.requires_grad for tensor in kl.call_args.args[:3]))
        for value, expected in zip(actual[:4], self.main_gradients):
            torch.testing.assert_close(value, expected)
        self._assert_index_gradients(actual[4:])
        self.assertEqual(self.model.state_dict(), {})

    def test_lm_only_and_kl_only_keep_the_other_objective_gradients_absent(self):
        """Hard selection has no LM gradient, and KL never updates the main teacher inputs."""
        for kl_only in (False, True):
            with self.subTest(kl_only=kl_only), self._patches() as (_kl, backward):
                output, loss = self.model(*self.inputs, self.meta)
                actual = torch.autograd.grad(loss * 7 if kl_only else output.sum(), self.inputs, allow_unused=True)
                absent = actual[:4] if kl_only else actual[4:]
                self.assertTrue(all(value is None for value in absent))
                self.assertEqual(backward.call_count, 0 if kl_only else 1)
                if kl_only:
                    self._assert_index_gradients(actual[4:])

    def test_zero_coefficient_skips_native_kl_and_returns_exact_zero_index_gradients(self):
        """Zero KL retains a zero derivative path and cannot produce native NaN times zero."""
        self.model.loss_coeff = 0
        self.inputs = (*self.inputs[:6], self.inputs[6].detach().float().requires_grad_())
        with self._patches() as (kl, _backward):
            output, loss = self.model(*self.inputs, self.meta)
            actual = torch.autograd.grad(output.sum() + loss * 7, self.inputs)
            kl.assert_not_called()
        self.assertEqual(float(loss.detach()), 0)
        for value in actual[4:]:
            self.assertEqual(int(value.count_nonzero()), 0)

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_saved_hooks_checkpoint_retain_and_delayed_backward_own_kl_gradients(self, _device_type):
        """Repeated backward reuses saved derivatives; checkpoint recomputes them without a hidden cache."""
        with self._patches() as (kl, _backward):
            output, loss = self.model(*self.inputs, self.meta)
            self.workspace.buffers["source_compressed"].fill_(19)
            first = torch.autograd.grad(output.sum() + loss * 7, self.inputs, retain_graph=True)
            second = torch.autograd.grad(output.sum() + loss * 7, self.inputs)
            out, aux = checkpoint(lambda *values: self.model(*values, self.meta), *self.inputs, use_reentrant=False)
            third = torch.autograd.grad(out.sum() + aux * 7, self.inputs)
        for actual in (first, second, third):
            self._assert_index_gradients(actual[4:])
        self.assertEqual(kl.call_count, 3)

    def test_mutation_and_second_order_reject_instead_of_using_stale_derivatives(self):
        """Index inputs participate in saved tensor versioning even though LM selection is detached."""
        with self._patches():
            output, loss = self.model(*self.inputs, self.meta)
            with self.assertRaisesRegex(ValueError, "first-order"):
                torch.autograd.grad(loss, self.inputs[4], create_graph=True)
            with torch.no_grad():
                self.inputs[6].add_(1)
            with self.assertRaisesRegex(RuntimeError, "modified"):
                torch.autograd.grad(output.sum() + loss, self.inputs)

    def test_configuration_and_native_shape_errors_reject_before_launch(self):
        """Global loss count, coefficient and every index dimension are explicit contracts."""
        for normalization, coeff in ((DsaLossNormalization(3), .3), (DsaLossNormalization(4), -1)):
            with self.subTest(normalization=normalization, coeff=coeff), self.assertRaises(ValueError):
                module.MegaDsa(self.workspace, self.invocation, heads=32, attention_scale=.1,
                               schedule=MixedSfaSchedule(7), normalization=normalization, loss_coeff=coeff)
        with self._patches() as (kl, _backward), self.assertRaisesRegex(ValueError, "inputs must match"):
            self.model(*self.inputs[:6], self.inputs[6].double(), self.meta)
        kl.assert_not_called()

    def test_frozen_objective_outputs_do_not_invent_teacher_or_selection_gradients(self):
        """Output differentiability follows each objective's actual input branch."""
        for freeze_index in (False, True):
            with self.subTest(freeze_index=freeze_index), self._patches():
                inputs = tuple(tensor.detach().clone().requires_grad_(
                    index < 4 if freeze_index else index >= 4) for index, tensor in enumerate(self.inputs))
                output, loss = self.model(*inputs, self.meta)
                self.assertEqual(output.requires_grad, freeze_index)
                self.assertEqual(loss.requires_grad, not freeze_index)

    def test_fp32_kl_weight_rejects_before_forward_without_implicit_precision_conversion(self):
        """LI supports FP32 weights, while the Omni reference selected-KL schema requires BF16."""
        with patch.object(self.model.core.backend, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "KL requires BF16"):
                self.model(*self.inputs[:6], self.inputs[6].detach().float().requires_grad_(), self.meta)
            forward.assert_not_called()

    def test_owner_return_keeps_fp32_weight_derivatives_and_independent_q_k_orders(self):
        """The shared collective packs FP32 contributions before restoring each input dtype."""
        gradients = tuple(gradient.float() * .00314159 for gradient in self.raw_index_gradients)
        layout = self.model.core.backend.layout
        actual = layout.reduce_owner_gradients(
            gradients, ((64, 128), (128,), (64,)), (layout.query_order, layout.key_order, layout.query_order),
            (torch.bfloat16, torch.bfloat16, torch.float32))
        for field, (value, expected, dtype) in enumerate(zip(
                actual, gradients, (torch.bfloat16, torch.bfloat16, torch.float32))):
            rows = self.meta.kv_global_ids if field == 1 else self.meta.q_global_ids
            torch.testing.assert_close(value, expected[list(rows)].to(dtype), rtol=0, atol=0)

    def test_native_kl_uses_owned_forward_derivatives_and_scales_once(self):
        """Native BF16/FP32 KL skips host KL and restores both objectives in independent owner orders."""
        self.model = module.MegaDsa(self.workspace, self.invocation, heads=32, attention_scale=.1,
                                    schedule=MixedSfaSchedule(7), normalization=DsaLossNormalization(4), loss_coeff=.25)
        self.assertEqual(self.model.kl_backend, "native")
        for dtype in (torch.bfloat16, torch.float32):
            self.inputs = (*self.inputs[:6], self.inputs[6].detach().to(dtype).requires_grad_())
            for kl_only in (False, True):
                with self.subTest(dtype=dtype, kl_only=kl_only), self._patches() as (kl, backward):
                    output, loss = self.model(*self.inputs, self.meta)
                    self.assertEqual(float(loss.detach()), .5)
                    actual = torch.autograd.grad(loss * 7 if kl_only else output.sum() + loss * 7,
                                                self.inputs, allow_unused=True)
                    kl.assert_not_called()
                    self.assertEqual(backward.call_count, 0 if kl_only else 1)
                    if kl_only:
                        self.assertTrue(all(value is None for value in actual[:4]))
                    self._assert_index_gradients(actual[4:])

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_native_kl_checkpoint_and_retained_backward_preserve_raw_derivatives(self, _device_type):
        """Saved native derivatives survive later publication, repeated VJPs and recomputed forward."""
        self.model.kl_backend = "native"
        with self._patches() as (kl, _backward):
            output, loss = self.model(*self.inputs, self.meta)
            self.model(*(tensor.detach() + .25 for tensor in self.inputs), replace(self.meta, layer=3))
            first = torch.autograd.grad(output.sum() + loss * 7, self.inputs, retain_graph=True)
            second = torch.autograd.grad(output.sum() + loss * 7, self.inputs)
            output, loss = checkpoint(lambda *values: self.model(*values, self.meta),
                                      *self.inputs, use_reentrant=False)
            third = torch.autograd.grad(output.sum() + loss * 7, self.inputs)
            kl.assert_not_called()
        for actual in (first, second, third):
            self._assert_index_gradients(actual[4:])
