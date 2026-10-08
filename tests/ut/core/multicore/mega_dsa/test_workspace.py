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
"""DSA scratch layout, detached publication and invocation/generation isolation."""

from dataclasses import replace

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestDsaWorkspace(SharedRootFixture):
    """Exercise the real registry/lifecycle over mocked native CPU allocations."""

    def setUp(self) -> None:
        """Declare one bounded DSA arena alongside the other root consumers."""
        super().setUp()
        spec = DsaWorkspaceSpec(8, 4, 8, 1, compressed_dim=8, rope_dim=4, index_dim=4)
        self.workspace = MegaDsaWorkspace(self.root, "dsa-workspace", spec)
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)

    def _prepare(self):
        self.workspace.bind()
        meta = replace(DsaBatchMeta.packed((3, 5)), kv_global_ids=(7, 0, 6, 1, 5, 2, 4, 3),
                       heap_generation=self.root.generation)
        return self.workspace.prepare(meta)

    def test_arena_alignment_and_peer_exclusive_fp32_inbox(self):
        """Independent fields are aligned and signals occupy separate 64-byte cache lines."""
        self.workspace.bind()
        base = self.workspace.arena.data_ptr()
        for name, offset, shape, dtype in self.workspace.layout:
            with self.subTest(name=name):
                self.assertEqual(offset % 512, 0)
                self.assertEqual(self.workspace.buffers[name].data_ptr() - base, offset)
                self.assertEqual(self.workspace.buffers[name].shape, shape)
                self.assertEqual(self.workspace.buffers[name].dtype, dtype)
        self.assertEqual(self.workspace.buffers["gradient_inbox"].dtype, torch.float32)
        self.assertEqual(self.workspace.buffers["events"].stride(-2) * 4, 64)
        self.assertEqual(self.native._empty.call_count, 1)

    def test_publication_restores_owner_offsets_and_detaches(self):
        """Saved activations are independent of scratch and retain their own autograd graph."""
        invocation = self._prepare()
        ids = torch.tensor(invocation.batch_meta.kv_global_ids, dtype=torch.bfloat16)
        states = tuple(ids[:, None].expand(-1, width).clone().requires_grad_() for width in (8, 4, 4))
        with self.workspace.lease(invocation, direction="forward"):
            self.workspace.publish_owner_states(invocation, states)
            for name in ("source_compressed", "source_rope", "source_index"):
                torch.testing.assert_close(self.workspace.buffers[name][:, 0], torch.arange(8, dtype=torch.bfloat16))
                self.assertFalse(self.workspace.buffers[name].requires_grad)
        self.assertTrue(all(tensor.grad is None for tensor in states))
        for tensor in states:
            tensor.sum().backward()
            torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))

    def test_invocation_and_direction_isolation(self):
        """A different layer/microbatch cannot publish during another invocation's lease."""
        first = self._prepare()
        other = self.workspace.prepare(replace(first.batch_meta, invocation=1, layer=2, microbatch=3))
        states = tuple(torch.zeros(8, width, dtype=torch.bfloat16) for width in (8, 4, 4))
        with self.workspace.lease(first, direction="forward"):
            with self.assertRaisesRegex(ValueError, "another invocation"):
                self.workspace.publish_owner_states(other, states)
            with self.assertRaisesRegex(RuntimeError, "active lease"):
                self.workspace.close()
        with self.workspace.lease(other, direction="backward"):
            self.workspace.publish_owner_states(other, states)
            self.assertEqual(self.workspace.active_invocation[-1], "backward")
        self.assertIsNone(self.workspace.active_invocation)

    def test_partial_wrong_root_capacity_and_stale_generation_rejected(self):
        """Publication metadata must cover this owner and address the current heap."""
        prepared = self._prepare()
        meta = prepared.batch_meta
        for invalid, reason in ((replace(meta, kv_global_ids=(0,)), "complete owner"),
                                (replace(meta, heap_generation=0), "heap_generation"),
                                (replace(meta, root_pes=(2,)), "membership")):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.workspace.prepare(invalid)
        larger = replace(DsaBatchMeta.packed((9,)), heap_generation=self.root.generation)
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.workspace.prepare(larger)

    def test_close_frees_once_and_invalidates_prepared_lease(self):
        """Workspace close drops every arena view and refuses later invocations."""
        prepared = self._prepare()
        self.workspace.close()
        self.workspace.close()
        self.assertIsNone(self.workspace.arena)
        self.assertEqual(self.workspace.buffers, {})
        self.native._free.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "bound live consumer"), \
             self.workspace.lease(prepared, direction="backward"):
            pass
