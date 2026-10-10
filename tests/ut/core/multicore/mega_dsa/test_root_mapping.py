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
"""CP subgroup/root PE separation, protocol metadata and admission over CPU storage."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp, index_return, workspace
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import FusedDsaCpBackwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestRootMapping(TestCase):
    """Mock only process groups; keep real owner permutations and metadata encoding."""

    def setUp(self) -> None:
        """Use WORLD ranks distinct from root PEs and CP ordinals."""
        self.cp_group, self.root_group = object(), object()
        self.root = SimpleNamespace(members=(3, 5, 7, 9), rank=1, group=self.root_group,
                                    generation=9, reserve=Mock(return_value=Mock()), lease_owner=None)
        self.addCleanup(patch.stopall)
        patch.object(workspace.dist, "is_initialized", return_value=True).start()
        patch.object(workspace.dist, "get_process_group_ranks", return_value=[5, 9]).start()
        patch.object(workspace.dist, "get_rank", return_value=0).start()
        def _gather(output, value, *, group):
            self.assertIs(group, self.cp_group)
            output[:] = [value]*len(output)
        self.gather = patch.object(workspace.dist, "all_gather_object", side_effect=_gather).start()
        self.workspace = workspace.MegaDsaWorkspace(self.root, "mapped", workspace.DsaWorkspaceSpec(2, 4, 2, 2),
                                                    cp_group=self.cp_group)
        self.workspace.bind = Mock()
        self.workspace.arena = torch.zeros(self.workspace.arena_bytes, dtype=torch.uint8)
        self.workspace.buffers["events"] = torch.zeros(2, 2, 4, 16, dtype=torch.int32)
        self.meta = replace(DsaBatchMeta.packed((4,)), cp_ranks=(5, 9), root_pes=(1, 3), cp_rank=0,
                            token_owners=(0, 1, 0, 1), token_local_offsets=(1, 1, 0, 0),
                            q_global_ids=(0, 2), kv_global_ids=(0, 2), heap_generation=9)
        self.invocation = self.workspace.prepare(self.meta)
        self.forward = fused_cp.FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                                       attention_scale=.1, schedule=MixedSfaSchedule(7))

    def test_cp_collectives_and_three_transport_protocols_use_explicit_root_pes(self):
        """Subgroup preparation never gathers on the encompassing root process group."""
        self.assertTrue(self.workspace.mapped_transport)
        self.assertEqual(self.workspace.root_pes, (1, 3))
        self.assertGreater(self.gather.call_count, 0)
        metadata = self.forward._metadata(self.invocation, 5)
        self.assertEqual(metadata.shape, (27,))
        self.assertEqual(metadata[1].item(), 2)
        self.assertEqual(metadata[3].item(), 0)
        self.assertEqual(metadata[18:].tolist(), [4, 1, 3, -1, -1, -1, -1, -1, -1])
        for sparse, protocol in ((False, 3), (True, 4)):
            backward = FusedDsaCpBackwardProbe(self.forward, sparse_return=sparse)
            metadata = backward._metadata(self.meta, 6)
            self.assertEqual(metadata[1].item(), protocol)
            self.assertEqual(metadata[18:].tolist(), [4, 1, 3, -1, -1, -1, -1, -1, -1])

    def test_index_return_uses_cp_owner_order_and_mapped_metadata(self):
        """Root PE one submits CP ordinal zero and receives caller-ordered owner FP32 rows."""
        probe = index_return.SparseIndexGradReturnProbe(self.forward)
        partial = torch.arange(4*128).reshape(4, 128).float()
        def _claim():
            self.root.lease_owner = self.workspace.consumer
        def _release():
            self.root.lease_owner = None
        def _native(gradient, _arena, metadata, _requests, _trace, owner):
            self.assertIs(gradient, partial)
            self.assertEqual(metadata[1].item(), 2)
            self.assertEqual(metadata[3].item(), 0)
            self.assertEqual(metadata[18:21].tolist(), [4, 1, 3])
            owner.copy_(gradient[[2, 0]])
        self.workspace.consumer.claim.side_effect = _claim
        self.workspace.consumer.release.side_effect = _release
        with patch.object(fused_cp, "_load_native"), patch.object(index_return, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_cp_root_mapping_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_index_grad_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_index_grad_out", side_effect=_native, create=True):
            result = probe.return_owner(self.invocation, partial)
        torch.testing.assert_close(result.owner_fp32, partial[[0, 2]], rtol=0, atol=0)
        self.assertIsNone(self.root.lease_owner)

    def test_bad_mapping_and_legacy_payload_reject_before_transport(self):
        """Valid-looking PE arrays cannot redirect data to other root members."""
        with self.assertRaisesRegex(ValueError, "root PE mapping"):
            self.workspace.prepare(replace(self.meta, root_pes=(0, 2)))
        with patch.object(fused_cp, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_cp_root_mapping_version", return_value=0, create=True), \
                self.assertRaisesRegex(RuntimeError, "root mapping ABI 1"):
            self.forward.check_mapping_support()
        self.workspace.consumer.claim.assert_not_called()
