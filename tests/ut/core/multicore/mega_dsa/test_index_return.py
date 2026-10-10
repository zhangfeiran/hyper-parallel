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
"""Index-K device return ownership, native admission and exact receipt validation."""

from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import index_return
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestIndexReturn(SharedRootFixture):
    """Use a real CPU workspace with native arithmetic mocked at its raw boundary."""

    def setUp(self) -> None:
        """Prepare a bound workspace and a caller KV permutation independent of native storage order."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "index-return", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((3,)), kv_global_ids=(1, 2, 0),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        self.forward = FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                              attention_scale=.1, schedule=MixedSfaSchedule(7))
        self.probe = index_return.SparseIndexGradReturnProbe(self.forward)

    def test_single_submission_restores_owner_order_and_keeps_results_independent(self):
        """Scaled input passes unchanged through a fresh lease and returns owned caller order."""
        partial = torch.arange(3*128).reshape(3, 128).float()/2048
        def _native(gradient, arena, metadata, requests, trace, owner):
            self.assertIs(self.root.lease_owner, self.workspace.consumer)
            self.assertIs(gradient, partial)
            self.assertIs(arena, self.workspace.arena)
            self.assertIs(requests, self.forward.requests)
            self.assertEqual(int(metadata[0]), index_return._INDEX_MAGIC)
            self.assertEqual(int(metadata[1]), 1)
            owner.copy_(gradient)
            trace[:4] = torch.tensor([int(metadata[2]), int(metadata[2]), 1, 1])
        with patch.object(index_return, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_index_grad_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_index_grad_out", side_effect=_native,
                             create=True) as submit:
            first = self.probe.return_owner(self.invocation, partial)
            preserved = first.owner_fp32.clone()
            second = self.probe.return_owner(self.invocation, partial)
        self.assertEqual(submit.call_count, 2)
        self.assertEqual(second.epoch, first.epoch+1)
        torch.testing.assert_close(first.owner_fp32, partial[[1, 2, 0]], rtol=0, atol=0)
        self.assertNotEqual(first.owner_fp32.data_ptr(), second.owner_fp32.data_ptr())
        second.owner_fp32.zero_()
        torch.testing.assert_close(first.owner_fp32, preserved, rtol=0, atol=0)

    def test_invalid_inputs_foreign_layout_and_old_payload_reject_before_lease(self):
        """Invalid global partials and unsupported adapters cannot publish any device state."""
        for value in (torch.zeros(2, 128), torch.zeros(3, 128, dtype=torch.float64),
                      torch.zeros(3, 128, requires_grad=True), torch.zeros(128, 3).T):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaisesRegex(ValueError, "FP32"):
                self.probe.return_owner(self.invocation, value)
        with patch.object(index_return, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_sparse_index_grad_version", return_value=0, create=True), \
                patch.object(self.workspace, "lease") as lease, self.assertRaisesRegex(RuntimeError, "adapter ABI 1"):
            self.probe.return_owner(self.invocation, torch.zeros(3, 128))
        lease.assert_not_called()
        foreign = replace(self.invocation, batch_meta=replace(self.meta, kv_global_ids=(0, 1, 2)))
        with patch.object(self.workspace, "lease") as lease, \
                self.assertRaisesRegex(ValueError, "unchanged prepared key ownership"):
            self.probe.return_owner(foreign, torch.zeros(3, 128))
        lease.assert_not_called()
        self.probe.owner_order = self.probe.owner_order.clone()
        with self.assertRaisesRegex(ValueError, "permutation was modified"):
            self.probe.return_owner(self.invocation, torch.zeros(3, 128))

    def test_trace_requires_all_receipts_counts_and_zero_clear_extent(self):
        """Independent receipts distinguish row payload, control counters and stale-data protection."""
        trace = torch.zeros(32, dtype=torch.int64)
        trace[:19] = torch.tensor([2, 2, 1, 1, 512, 0, 512, 0, 100, 120, 130, 150, 1, 2, 1, 0, 0, 0, 1536])
        result = index_return.IndexGradReturnResult(torch.zeros(3, 128), trace, 2)
        proof = self.probe.validate_trace(result, trace)
        self.assertEqual(proof["skipped_zero_rows"], 2)
        self.assertEqual(proof["owner_inbox_clear_bytes"], 1536)
        for word, value in ((0, 1), (1, 1), (2, 0), (3, 0), (4, 513), (5, 1), (6, 0), (7, 512),
                            (8, 0), (9, 100), (12, 4), (13, 0), (14, 4), (15, 1), (16, 8), (18, 0), (31, 1)):
            invalid = trace.clone()
            invalid[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.probe.validate_trace(result, invalid)
