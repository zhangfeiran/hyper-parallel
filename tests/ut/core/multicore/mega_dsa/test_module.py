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

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa import module
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
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
                patch.object(torch.ops.hyper_parallel, "dsa_cp_attention_version", return_value=1, create=True),
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

    def test_autograd_saved_hooks_recompute_without_hidden_tensor_cache(self):
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

        with load, version, execute, patch.object(self.core.backward_backend, "_backward_saved", side_effect=_backward):
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

    def test_extra_padding_training_stops_before_unsafe_native_backward(self):
        """Known native count-contract failures cannot silently enter the public training path."""
        incomplete = torch.full_like(self.indices, -1)
        incomplete[0, 0] = 3
        selection = self.core.prepare_selection(incomplete)
        self.assertFalse(selection.backward_ready)
        main = tuple(tensor.clone().requires_grad_() for tensor in self.states)
        with patch.object(module, "_load_native") as load:
            with self.assertRaisesRegex(NotImplementedError, "selected-count fix"):
                self.core(*main, selection, self.meta)
            load.assert_not_called()
