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
"""Fused TP declarations, single-root execution and caller-owned saved state contracts."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_cp_tp
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult,
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
    FusedDsaCpTpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class TestFusedCpTpSubmission(SharedRootFixture):
    """Exercise real consumer leases with CPU arena mocks and one fake native call."""

    def setUp(self) -> None:
        """Reserve every arena before any consumer binds and freezes the shared root."""
        super().setUp()
        group = object()
        prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum'
        patch(prefix+'.dist.get_process_group_ranks', return_value=[0]).start()
        patch(prefix+'.dist.get_rank', return_value=0).start()
        patch.object(fused_cp_tp.dist, 'all_gather_object', side_effect=self._gather).start()
        patch('hyper_parallel.core.multicore.modules.mega_dsa.fused_cp._load_native').start()
        for name in ('dsa_local_cp_forward_version', 'dsa_local_cp_training_version',
                     'dsa_tp_local_cp_training_version'):
            patch.object(torch.ops.hyper_parallel, name, return_value=1, create=True).start()
        self.workspace = MegaDsaWorkspace(self.root, 'cp-tp', DsaWorkspaceSpec(3, 3, 3, 1))
        self.li = TpTileSumProbe(self.root, 'li', TpTileSumSpec(14, 512), tp_group=group)
        self.kl = TpTileSumProbe(self.root, 'kl', TpTileSumSpec(14, 2048), tp_group=group)
        for item in (self.workspace, self.li, self.kl):
            self.consumers.append(item.consumer)
            self.addCleanup(item.close)
            item.bind()
        self.meta = replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0, 1),
                            heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        cp = FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                   attention_scale=.1, schedule=MixedSfaSchedule(7))
        self.backend = FusedDsaCpTpForwardProbe(cp, self.li, self.kl)

    @staticmethod
    def _gather(outputs, declaration, group):
        del group
        outputs[:] = [declaration]*len(outputs)

    @staticmethod
    def _states(ids):
        rows = torch.tensor(ids, dtype=torch.bfloat16)
        main = (rows[:, None, None].expand(-1, 32, 512).clone(), torch.zeros(3, 512, dtype=torch.bfloat16),
                rows[:, None, None].expand(-1, 32, 64).clone(), torch.zeros(3, 64, dtype=torch.bfloat16))
        index = (rows[:, None, None].expand(-1, 64, 128).clone(), torch.zeros(3, 128, dtype=torch.bfloat16),
                 rows[:, None].expand(-1, 64).clone())
        return main, index

    def _execute(self, *args):
        self.assertIs(self.root.lease_owner, self.workspace.consumer)
        self.assertEqual(self.li.last_epoch, int(args[-4][2]))
        enabled = args[-1]
        self.assertEqual(args[9].shape[0], 9 if enabled else 6)
        self.assertIs(args[-5], self.li.arena)
        self.assertEqual(self.kl.last_epoch, self.li.last_epoch if enabled else 0)
        if enabled:
            self.assertIs(args[-3], self.kl.arena)
            args[31].fill_(4)
            args[32].zero_()
            args[33].fill_(5)
            args[34].fill_(6)
        else:
            self.assertIsNone(args[-3])
            self.assertIsNone(args[-2])
            self.assertTrue(all(args[index].shape == (0,) for index in range(30, 35)))
        args[13].fill_(-1)
        args[15].copy_(args[2])
        args[16].zero_()
        args[17].fill_(1)

    def test_forward_and_kl_use_one_lease_and_preserve_empty_queries(self):
        """Both six and nine phases support true zero rows with monotonic TP consumption."""
        for epoch, (enabled, ids) in enumerate(((False, (2, 0)), (True, (2, 0)), (True, ())), start=1):
            with self.subTest(enabled=enabled, ids=ids):
                invocation = self.backend.prepare(self.invocation, epoch=epoch, query_ids=ids, selected_kl=enabled)
                if enabled:
                    self.assertEqual(invocation.transports[0].identity, invocation.transports[1].identity)
                with patch.object(torch.ops.hyper_parallel, 'dsa_tp_local_cp_training_out',
                                  side_effect=self._execute, create=True) as native:
                    result = self.backend.forward(*self._states(ids), invocation)
                native.assert_called_once()
                self.assertIsNone(self.root.lease_owner)
                result.saved.validate_versions()
                torch.testing.assert_close(result.output[:, 0, 0], torch.tensor(ids, dtype=torch.bfloat16))
                self.assertEqual(result.kl_loss is not None, enabled)
                if enabled:
                    torch.testing.assert_close(result.kl_gradients[0], torch.full_like(result.index_states[0], 4))
                with self.assertRaisesRegex(ValueError, 'newer'):
                    self.backend.forward(*self._states(ids), invocation)

    def test_metadata_mutation_and_old_abi_fail_before_epoch_consumption(self):
        """A failed validation cannot publish CP owner states or consume a TP transport epoch."""
        invocation = self.backend.prepare(self.invocation, epoch=1, query_ids=(2, 0), selected_kl=True)
        with patch.object(torch.ops.hyper_parallel, 'dsa_tp_local_cp_training_version', return_value=0), \
                self.assertRaisesRegex(RuntimeError, 'adapter ABI'):
            self.backend.forward(*self._states((2, 0)), invocation)
        self.assertEqual(self.li.last_epoch, 0)
        invocation.transports[1].metadata[2] = 17
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.backend.forward(*self._states((2, 0)), invocation)
        self.assertEqual(self.li.last_epoch, 0)
        self.assertIsNone(self.root.lease_owner)

    def test_tp_declaration_disagreement_prevents_preparation(self):
        """TP must agree on CP ordinal and rows before any device task can wait for peers."""
        def _mismatch(outputs, _declaration, group):
            del group
            outputs[:] = [('different rows',)]
        with patch.object(fused_cp_tp.dist, 'all_gather_object', side_effect=_mismatch), \
                self.assertRaisesRegex(ValueError, 'logical rows'):
            self.backend.prepare(self.invocation, epoch=1)
        self.assertEqual(self.li.last_epoch, 0)


class TestFusedCpTpHeadRestore(unittest.TestCase):
    """Check original Hi64 sharding independently of the mocked transport arithmetic."""

    def test_compute_group_changes_require_matching_li_and_kl_capacity(self):
        """A legal group schedule must resize both score transports before fused construction."""
        for groups in (1, 2, 7, 19):
            with self.subTest(groups=groups):
                root = object()
                cp = SimpleNamespace(workspace=SimpleNamespace(root=root), schedule=MixedSfaSchedule(groups))
                li = SimpleNamespace(root=root, members=(0, 1), participates=True,
                                     spec=TpTileSumSpec(2*groups, 512))
                kl = SimpleNamespace(root=root, members=(0, 1), spec=TpTileSumSpec(2*groups, 2048))
                backend = FusedDsaCpTpForwardProbe(cp, li, kl)
                self.assertEqual(backend.index_heads, 32)
                li.spec = TpTileSumSpec(2*groups+2, 512)
                with self.assertRaisesRegex(ValueError, 'matching LI root/slots'):
                    FusedDsaCpTpForwardProbe(cp, li, kl)
                li.spec = TpTileSumSpec(2*groups, 512)
                kl.spec = TpTileSumSpec(2*groups+2, 2048)
                with self.assertRaisesRegex(ValueError, 'matching root and TP members'):
                    FusedDsaCpTpForwardProbe(cp, li, kl)

    def test_signed_weight_padding_and_local_head_vjp_restore(self):
        """TP2/4/8 preserve signed scaled weights and remove only zero native padding."""
        for size in (2, 4, 8):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(size=size, dtype=dtype):
                    root = object()
                    local = DsaLocalQueryLayout(replace(DsaBatchMeta.packed((3,)), q_global_ids=(2, 0)), 'cpu')
                    cp = SimpleNamespace(workspace=SimpleNamespace(root=root), schedule=MixedSfaSchedule(7),
                                         layout=SimpleNamespace(device=torch.device('cpu')))
                    li = SimpleNamespace(root=root, members=tuple(range(size)), participates=True,
                                         spec=TpTileSumSpec(14, 512))
                    backend = FusedDsaCpTpForwardProbe(cp, li)
                    query = torch.arange(2*backend.index_heads*128).reshape(2, backend.index_heads, 128).bfloat16()
                    weights = torch.linspace(-.2, .2, 2*backend.index_heads).reshape(2, backend.index_heads).to(dtype)
                    key = torch.zeros(3, 128, dtype=torch.bfloat16)
                    def _execute(_invocation, main_states, index_states, *, local=local, backend=backend,
                                 query=query, weights=weights, **options):
                        del main_states
                        self.assertIs(options['query_layout'], local)
                        padded_query, owner_key, padded_weights = index_states
                        torch.testing.assert_close(padded_query[:, :backend.index_heads], query)
                        torch.testing.assert_close(padded_weights[:, :backend.index_heads], weights)
                        self.assertFalse(bool(padded_query[:, backend.index_heads:].any()))
                        self.assertFalse(bool(padded_weights[:, backend.index_heads:].any()))
                        empty = torch.empty(0)
                        return FusedCpForwardResult(empty, empty, empty, empty, empty, (), (), empty, 1,
                            index_states=(local.pack(padded_query), owner_key, local.pack(padded_weights)),
                            kl_gradients=(padded_query*2, owner_key, padded_weights*3), kl_loss=torch.tensor(1.))
                    cp.forward_local = Mock(side_effect=_execute)
                    invocation = FusedCpTpInvocation(backend, SimpleNamespace(batch_meta=local.batch_meta),
                                                      local, (), True)
                    result = backend.forward((), (query, key, weights), invocation)
                    torch.testing.assert_close(result.index_states[0], local.pack(query))
                    torch.testing.assert_close(result.index_states[2], local.pack(weights))
                    torch.testing.assert_close(result.kl_gradients[0], query*2)
                    torch.testing.assert_close(result.kl_gradients[2], weights*3)
                    self.assertEqual(result.index_versions, tuple(value._version for value in result.index_states))
                    cp.forward_local.assert_called_once()
