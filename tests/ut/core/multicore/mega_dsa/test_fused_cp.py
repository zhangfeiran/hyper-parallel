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
"""Owner-run coverage, Q-only permutation and leased native CP submission contracts."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp
from hyper_parallel.core.multicore.modules.mega_dsa.cp_reference import DsaCpLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestQueryGather(unittest.TestCase):
    """Query-owned fields must all use Q storage ordering, including their owner gradients."""

    def test_query_fields_never_use_kv_order(self):
        """Exercise different Q/K storage permutations with an independent global-row oracle."""
        meta = DsaBatchMeta((0, 3), (2, 0, 1), (1, 2, 0), (0, 0, 0), (1, 2, 0))
        layout = DsaCpLayout(meta, "cpu")
        global_values = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        tensors = tuple((global_values[list(meta.q_global_ids)] * factor).requires_grad_() for factor in (1, 3))
        output = layout.gather_query_fields(tensors, ((2,), (2,)))
        for actual, factor in zip(output, (1, 3)):
            torch.testing.assert_close(actual, global_values * factor)
        coefficients = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        sum((value * coefficients).sum() for value in output).backward()
        for tensor in tensors:
            torch.testing.assert_close(tensor.grad, coefficients[list(meta.q_global_ids)])

    def test_runs_cover_global_ids_and_original_owner_offsets(self):
        """Expand runs independently and require exact owner/offset pairs for every destination row."""
        for owners, offsets in (((0, 0, 1, 1), (0, 1, 0, 1)),
                                ((0, 1, 0, 1), (1, 1, 0, 0)),
                                ((0, 0, 0, 0), (3, 2, 1, 0))):
            ids = tuple(index for index, owner in enumerate(owners) if owner == 0)
            meta = DsaBatchMeta((0, 4), ids, ids, owners, offsets, cp_ranks=(0, 1), root_pes=(0, 1))
            actual = {}
            for peer, source, destination, count in fused_cp.cp_owner_runs(meta):
                for step in range(count):
                    self.assertNotIn(destination + step, actual)
                    actual[destination + step] = (peer, source + step)
            self.assertEqual(actual, dict(enumerate(zip(owners, offsets))))


class TestFusedCpSubmission(SharedRootFixture):
    """Native transport cannot outlive ownership or inherit stale epochs from another adapter."""

    def setUp(self) -> None:
        """Bind one real workspace over mocked CPU SHMEM storage."""
        super().setUp()
        self.workspace = MegaDsaWorkspace(self.root, "cp-probe", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0, 1), kv_global_ids=(1, 2, 0),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)

    def _backend(self):
        return fused_cp.FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                              attention_scale=0.1, schedule=MixedSfaSchedule(7))

    def test_epoch_requires_lease_and_survives_adapter_recreation(self):
        with self.assertRaisesRegex(RuntimeError, "active invocation"):
            self.workspace.next_transport_epoch(self.invocation)
        with self.workspace.lease(self.invocation, direction="forward"):
            self.assertEqual(self.workspace.next_transport_epoch(self.invocation), 1)
        self._backend()
        with self.workspace.lease(self.invocation, direction="backward"):
            self.assertEqual(self.workspace.next_transport_epoch(self.invocation), 2)
            foreign = self.workspace.prepare(replace(self.meta, invocation=1))
            with self.assertRaisesRegex(RuntimeError, "active invocation"):
                self.workspace.next_transport_epoch(foreign)

    def test_query_gather_publication_and_native_launch_share_lease(self):
        backend = self._backend()
        query_ids = torch.tensor(self.meta.q_global_ids, dtype=torch.bfloat16)
        key_ids = torch.tensor(self.meta.kv_global_ids, dtype=torch.bfloat16)
        main = (query_ids[:, None, None].expand(3, 32, 512).clone(),
                key_ids[:, None].expand(3, 512).clone(), torch.zeros(3, 32, 64, dtype=torch.bfloat16),
                key_ids[:, None].expand(3, 64).clone())
        index = (torch.zeros(3, 64, 128, dtype=torch.bfloat16),
                 key_ids[:, None].expand(3, 128).clone(), torch.zeros(3, 64, dtype=torch.bfloat16))
        epochs = []

        def _execute(*args):
            self.assertIs(self.root.lease_owner, self.workspace.consumer)
            self.assertTrue(all(tensor.is_contiguous() for tensor in args[:11]))
            self.assertIs(args[17], self.workspace.arena)
            metadata = args[18]
            epochs.append(int(metadata[2]))
            torch.testing.assert_close(self.workspace.buffers["source_compressed"][:, 0],
                                       torch.arange(3, dtype=torch.bfloat16))
            for position in (1, 3, 5):
                args[position].fill_(1)
            args[12].fill_(-1)
            args[12][:, 0, 0] = 0
            args[13].fill_(0)
            args[14].copy_(args[2])
            args[15].fill_(0)
            args[16].fill_(1)

        with patch.object(fused_cp, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_forward_version", return_value=2, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_cp_forward_out", side_effect=_execute, create=True):
            first = backend.forward(self.invocation, main, index)
            second = self._backend().forward(self.invocation, main, index)
        self.assertEqual(epochs, [1, 2])
        self.assertIsNone(self.root.lease_owner)
        torch.testing.assert_close(first.output[:, 0, 0], query_ids)
        self.assertNotEqual(first.transport_trace.data_ptr(), second.transport_trace.data_ptr())
        self.assertNotEqual(first.global_keys[0].data_ptr(), second.global_keys[0].data_ptr())

    def test_transport_evidence_rejects_incomplete_reads_and_unordered_progress(self):
        """Reject corrupted ready/ACK, byte counts and intervals independently of native execution."""
        backend = self._backend()
        phases = []
        for epoch in (1, 2, 3):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(7):
                tasks = tuple(range(group, 20, 7))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 4] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1]])
                    trace[group, offset + 6] = epoch
                    if epoch == 1:
                        trace[group, offset + 7:offset + 10] = torch.tensor([1, 150, 300])
            trace[7, [6, 21, 22, 24, 38]] = epoch
            trace[7, 25] = 21
            phases.append(trace)
        transport = torch.tensor([5, 5, 1, 1, 768, 3456, 0, 0, 100, 120, 200, 250,
                                  0, 21] + [0] * 18, dtype=torch.int64)
        empty = torch.empty(0)
        result = fused_cp.FusedCpForwardResult(empty, empty, empty, empty, empty, (), (), transport, 5)
        evidence = backend.validate_trace(result, tuple(phases), transport, require_overlap=True)
        self.assertEqual(len(evidence["compute_members_overlapping_transfer"]), 21)
        for word, value in ((0, 4), (1, 4), (2, 0), (3, 0), (4, 767), (5, 3455),
                            (6, 1), (7, 1), (8, 0), (9, 99), (10, 119), (11, 200),
                            (12, 1), (13, 20), (14, 1), (31, 1)):
            broken = transport.clone()
            broken[word] = value
            with self.subTest(word=word), self.assertRaises(ValueError):
                backend.validate_trace(result, tuple(phases), broken, require_overlap=True)
        phases[0][0, 7] = 0
        with self.assertRaisesRegex(ValueError, "compute interval"):
            backend.validate_trace(result, tuple(phases), transport, require_overlap=True)
        phases[0][0, 7] = 1
        no_overlap = transport.clone()
        no_overlap[10:12] = torch.tensor([400, 450])
        backend.validate_trace(result, tuple(phases), no_overlap, require_overlap=False)
        with self.assertRaisesRegex(ValueError, "transfer overlap"):
            backend.validate_trace(result, tuple(phases), no_overlap, require_overlap=True)
