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
"""Native TP arena, metadata, epochs and mapped root lifecycle over CPU mocks."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestTpTileSumSpec(unittest.TestCase):
    """Keep transport identity and bounded scratch independent of candidate scan length."""

    def test_mapping_rounds_epochs_and_fixed_bytes(self):
        """Noncontiguous TP membership maps to root PEs with explicit unused padding."""
        for slots, width in ((1, 8), (7, 512), (40, 2048)):
            with self.subTest(slots=slots, width=width):
                spec = TpTileSumSpec(slots, width)
                first = spec.metadata(rounds=1, epoch=1, rank=1, root_pes=(1, 3), root_size=4,
                                      identity=(17, 2, 3, 1, 89))
                last = spec.metadata(rounds=2**32-1, epoch=2**31-1, rank=1, root_pes=(1, 3), root_size=4,
                                     identity=(17, 2, 3, 1, 89))
                self.assertEqual(len(first), 24)
                self.assertEqual(first[16:], (1, 3, -1, -1, -1, -1, -1, -1))
                self.assertEqual(first[11:16], (17, 2, 3, 1, 89))
                self.assertEqual(first[6:10], last[6:10])
                self.assertEqual(spec.control_offset % 64, 0)
                self.assertEqual(spec.arena_bytes % 512, 0)
                self.assertGreaterEqual(spec.arena_bytes, spec.control_offset + slots * 640)
                self.assertLessEqual(last[2] * 2**32 + last[10], 2**63-1)

    def test_reject_unsupported_geometry_and_metadata(self):
        """Fail before device submission for bad peer IDs, integer bounds or shape declarations."""
        for slots, width in ((0, 512), (41, 512), (True, 512), (1, 7), (1, 513), (1, 2056)):
            with self.subTest(slots=slots, width=width), self.assertRaises(ValueError):
                TpTileSumSpec(slots, width)
        valid = {'rounds': 17, 'epoch': 1, 'rank': 0, 'root_pes': (1, 3), 'root_size': 4, 'identity': (1, 2, 3, 4, 5)}
        invalid = ({'rounds': 0}, {'rounds': 2**32}, {'epoch': 0}, {'epoch': 2**31}, {'rank': 2},
                   {'root_pes': (1, 1)}, {'root_pes': (1, 4)}, {'root_pes': ()}, {'identity': ()},
                   {'identity': (1, 2, 3, 4, -1)}, {'root_size': True})
        for replacement in invalid:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                TpTileSumSpec().metadata(**(valid | replacement))


class TestTpTileSumProbe(SharedRootFixture):
    """Exercise real root leases and consumer allocation bookkeeping over CPU buffers."""

    def setUp(self) -> None:
        """Prepare one explicit singleton TP group and mocked native capability."""
        super().setUp()
        self.group = object()
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum'
        patch(prefix + '.dist.get_process_group_ranks', return_value=[0]).start()
        patch(prefix + '.dist.get_rank', return_value=0).start()
        patch(prefix + '._load_native').start()
        self.ops = Mock()
        self.ops.dsa_tp_tile_sum_version.return_value = 1
        self.ops.dsa_tp_tile_sum_out.side_effect = lambda partial, _arena, _meta, output: output.copy_(partial)
        patch(prefix + '.torch.ops.hyper_parallel', self.ops).start()
        self.probe = TpTileSumProbe(self.root, 'tp', TpTileSumSpec(1, 8), tp_group=self.group)
        self.consumers.append(self.probe.consumer)
        self.addCleanup(self.probe.close)

    def _prepare(self, epoch=1):
        self.probe.bind()
        return self.probe.prepare(rounds=3, epoch=epoch, identity=(1, 0, 0, self.root.generation, 7))

    def test_run_retains_metadata_releases_lease_and_rejects_epoch_reuse(self):
        """Repeated execution cannot observe stale ready/ACK flags or alias returned data."""
        invocation = self._prepare()
        partial = torch.randn(3, 1, 8)
        output = self.probe.run(partial, invocation)
        torch.testing.assert_close(output, partial)
        self.assertNotEqual(output.data_ptr(), partial.data_ptr())
        self.assertIsNone(self.root.lease_owner)
        self.assertEqual(self.probe.last_epoch, 1)
        self.assertEqual(self.native._empty.call_count, 1)
        self.assertEqual(invocation.metadata.numel(), 24)
        with self.assertRaisesRegex(ValueError, 'newer'):
            self.probe.run(partial, invocation)
        self.probe.run(partial, self._prepare(epoch=2))
        self.assertEqual(self.native._empty.call_count, 1)

    def test_capability_and_active_input_fail_before_claim(self):
        """Missing payload support does not consume a transport epoch or root lease."""
        invocation = self._prepare()
        with self.assertRaisesRegex(ValueError, 'detached'):
            self.probe.run(torch.randn(3, 1, 8, requires_grad=True), invocation)
        self.ops.dsa_tp_tile_sum_version.return_value = 0
        with self.assertRaisesRegex(RuntimeError, 'version 1'):
            self.probe.run(torch.randn(3, 1, 8), invocation)
        self.assertEqual(self.probe.last_epoch, 0)
        self.assertIsNone(self.root.lease_owner)
        self.ops.dsa_tp_tile_sum_out.assert_not_called()

    def test_modified_invocation_metadata_rejects_before_publication(self):
        """A mutated ready/ACK epoch or row signature cannot enter the native protocol."""
        invocation = self._prepare()
        invocation.metadata[2] = 99
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.probe.run(torch.randn(3, 1, 8), invocation)
        self.assertEqual(self.probe.last_epoch, 0)
        self.assertIsNone(self.root.lease_owner)
        self.ops.dsa_tp_tile_sum_out.assert_not_called()

    def test_fused_consumer_uses_existing_root_lease_and_rejects_reuse(self):
        """One completion event protects TP storage borrowed by a CP submission."""
        invocation = self._prepare()
        owner = self.consumers[0]
        owner.bind()
        with self.assertRaisesRegex(ValueError, 'active owner lease'):
            self.probe.consume_in_root_lease(invocation, owner)
        self.assertEqual(self.probe.last_epoch, 0)
        owner.claim()
        self.probe.consume_in_root_lease(invocation, owner)
        self.assertIs(self.root.lease_owner, owner)
        self.assertEqual(self.probe.last_epoch, 1)
        with self.assertRaisesRegex(ValueError, 'newer'):
            self.probe.consume_in_root_lease(invocation, owner)
        owner.release()
        self.assertIsNone(self.root.lease_owner)
        self.assertTrue(self.root.used)

    def test_inactive_member_allocates_and_closes_but_cannot_prepare(self):
        """Root-wide allocation order includes ranks outside the executing TP group."""
        inactive = TpTileSumProbe(self.root, 'inactive', TpTileSumSpec(1, 8),
                                  tp_group=dist.GroupMember.NON_GROUP_MEMBER)
        self.consumers.append(inactive.consumer)
        self.addCleanup(inactive.close)
        inactive.bind()
        self.assertFalse(inactive.participates)
        self.assertIsNotNone(inactive.arena)
        with self.assertRaisesRegex(ValueError, 'inactive'):
            inactive.prepare(rounds=3, epoch=1, identity=(0, 0, 0, self.root.generation, 0))
        inactive.close()
        self.assertEqual(self.native._free.call_count, 1)
