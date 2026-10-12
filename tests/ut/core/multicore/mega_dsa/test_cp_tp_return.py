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
"""CPU contracts for native CP index return followed by shared-gradient TP return."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.cp_tp_return import (
    CpTpSharedGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import _layout_signature
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import (
    IndexGradReturnResult,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerReturnInvocation,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)


class TestCpTpReturn(unittest.TestCase):
    """Verify cross-transport ordering, collective branch admission and unrounded CP output."""

    def setUp(self) -> None:
        """Prepare one CP owner and two mocked native transports on a shared root."""
        self.prefix = 'hyper_parallel.core.multicore.modules.mega_dsa.cp_tp_return'
        self.meta = DsaBatchMeta.packed((3,))
        root = object()
        self.workspace = Mock(root=root)
        self.cp = Mock()
        self.cp.workspace = self.workspace
        self.cp.backend.layout.group = object()
        self.cp.backend.layout.batch_meta = self.meta
        self.cp.backend.signature = _layout_signature(self.meta)
        self.tp = Mock()
        self.tp.transport.root = root
        self.tp.transport.arena = torch.empty(0, dtype=torch.uint8)
        self.backend = CpTpSharedGradientReturnProbe(self.cp, self.tp)
        self.cp_invocation = DsaWorkspaceInvocation(
            batch_meta=self.meta, owner_order=torch.arange(3), workspace=self.workspace)
        self.tp.prepare.side_effect = lambda meta, epoch, presence: TpOwnerReturnInvocation(
            self.tp, meta.kv_global_ids, presence, Mock())
        self.owner = torch.full((3, 128), 0.003)
        self.cp.return_owner.return_value = IndexGradReturnResult(self.owner, torch.zeros(32, dtype=torch.int64), 1)
        self.order = Mock()
        self.order.attach_mock(self.cp.return_owner, 'cp')
        self.order.attach_mock(self.tp.return_owner, 'tp')
        gather = patch(self.prefix+'.dist.all_gather_object')
        self.gather = gather.start()
        self.addCleanup(gather.stop)
        self.gather.side_effect = lambda output, value, **_kwargs: output.__setitem__(slice(None), [value])
        size = patch(self.prefix+'.dist.get_world_size', return_value=1)
        size.start()
        self.addCleanup(size.stop)

    def test_cp_output_stays_fp32_until_shared_tp_return(self):
        """Combined dK+dV and CP index owner output each enter TP exactly once."""
        invocation = self.backend.prepare(self.cp_invocation, epoch=1)
        index = torch.ones(3, 128)
        main = (torch.full((3, 512), 1.003), torch.full((3, 64), -1.0))
        result = self.backend.return_owner(index, main, invocation, input_dtypes=(torch.bfloat16,)*3)
        self.assertIs(result.index_cp.owner_fp32, self.owner)
        self.assertEqual([call[0] for call in self.order.mock_calls], ['cp', 'tp'])
        fields = self.tp.return_owner.call_args.args[0]
        self.assertIs(fields[0], main[0])
        self.assertIs(fields[1], main[1])
        self.assertIs(fields[2], self.owner)
        self.assertEqual(fields[2].dtype, torch.float32)
        self.cp.return_owner.assert_called_once_with(self.cp_invocation, index)

    def test_inactive_index_skips_cp_return_but_keeps_tp_main_fields(self):
        """A collectively absent auxiliary branch never enters native CP index publication."""
        invocation = self.backend.prepare(self.cp_invocation, epoch=1, presence=(True, True, False))
        main = (torch.zeros(3, 512), torch.zeros(3, 64))
        result = self.backend.return_owner(None, main, invocation, input_dtypes=(torch.bfloat16,)*3)
        self.assertIsNone(result.index_cp)
        self.cp.return_owner.assert_not_called()
        self.assertEqual(self.tp.return_owner.call_args.args[0][2], None)

    def test_invalid_fields_and_old_tp_abi_fail_before_cp_publication(self):
        """Cross-transport rejection is early enough to avoid partial CP submission."""
        invocation = self.backend.prepare(self.cp_invocation, epoch=1)
        main = (torch.zeros(3, 512), torch.zeros(3, 64))
        with self.assertRaisesRegex(ValueError, 'FP32'):
            self.backend.return_owner(torch.ones(3, 128).bfloat16(), main, invocation,
                                      input_dtypes=(torch.bfloat16,)*3)
        self.tp.check_native_support.side_effect = RuntimeError('old TP ABI')
        with self.assertRaisesRegex(RuntimeError, 'old TP ABI'):
            self.backend.return_owner(torch.ones(3, 128), main, invocation, input_dtypes=(torch.bfloat16,)*3)
        self.cp.return_owner.assert_not_called()
        self.tp.return_owner.assert_not_called()

    def test_cp_branch_disagreement_and_root_mismatch_rejected(self):
        """Neither mismatched CP branches nor different root heaps may be composed."""
        self.gather.side_effect = lambda output, value, **_kwargs: output.__setitem__(
            slice(None), [value, (False, False, False)])
        with patch(self.prefix+'.dist.get_world_size', return_value=2), \
                self.assertRaisesRegex(ValueError, 'CP members disagree'):
            self.backend.prepare(self.cp_invocation, epoch=1)
        self.tp.prepare.assert_not_called()
        self.tp.transport.root = object()
        with self.assertRaisesRegex(ValueError, 'same SHMEM root'):
            CpTpSharedGradientReturnProbe(self.cp, self.tp)
