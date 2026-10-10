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
"""CPU ownership, permutation and empty-tile contracts for local selected KL."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import local_kl, mixed_kl
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpForwardResult, FusedCpSavedAttention, _SAVED_ATTENTION_TOKEN, _layout_signature,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import DsaLocalQueryLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestLocalKl(unittest.TestCase):
    """Mock arithmetic while checking the production native ABI and storage contract."""

    def setUp(self) -> None:
        """Use compact CPU workspaces and record independent native phase buffers."""
        self.calls = []
        self.workspaces = []
        mocks = (patch.object(local_kl, "_load_native"),
                 patch.object(local_kl, "mixed_kl_workspace_bytes",
                              side_effect=lambda tokens, heads: 512+tokens*512),
                 patch.object(mixed_kl, "mixed_kl_workspace_bytes",
                              side_effect=lambda tokens, heads: 512+tokens*512),
                 patch.object(torch.ops.hyper_parallel, "dsa_local_mixed_kl_version", return_value=1, create=True),
                 patch.object(torch.ops.hyper_parallel, "dsa_local_mixed_kl_out",
                              side_effect=self._native, create=True))
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)

    @staticmethod
    def _forward(ids=(6, 0, 2), signature=1, weight_dtype=torch.float32):
        meta = replace(DsaBatchMeta.packed((3, 4)), q_global_ids=ids)
        local = DsaLocalQueryLayout(meta, "cpu")
        schedule = MixedSfaSchedule(7)
        backend = SimpleNamespace(signature=_layout_signature(meta), heads=32, scale=0.1, schedule=schedule,
                                  config=schedule.runtime_config("cpu"))
        count = len(ids)
        query = torch.full((count, 32, 512), signature, dtype=torch.bfloat16)
        states = (query, torch.zeros(7, 512, dtype=torch.bfloat16),
                  torch.zeros(count, 32, 64, dtype=torch.bfloat16), torch.zeros(7, 64, dtype=torch.bfloat16))
        index = (torch.zeros(count, 64, 128, dtype=torch.bfloat16), torch.zeros(7, 128, dtype=torch.bfloat16),
                 torch.zeros(count, 64, dtype=weight_dtype))
        indices = torch.full((count, 1, 2048), -1, dtype=torch.int32)
        output = torch.zeros_like(query)
        maximum = torch.zeros(1, count, 32)
        denominator = torch.ones_like(maximum)
        tensors = (*states, indices, output, maximum, denominator)
        saved = FusedCpSavedAttention(meta, states, indices, (output, maximum, denominator),
                                      tuple(tensor._version for tensor in tensors), backend,
                                      (32, 0.1, schedule), _SAVED_ATTENTION_TOKEN, "local", local)
        return FusedCpForwardResult(output, indices, indices.bfloat16(), maximum, denominator, (), (),
                                    torch.zeros(32, dtype=torch.long), 1, saved, index,
                                    index_versions=tuple(tensor._version for tensor in index))

    def _native(self, *args):
        self.assertEqual(len(args), 22)
        for tensor in (*args[:10], *args[18:]):
            self.assertFalse(tensor.requires_grad)
        query, key, retained, positions, phase = args[0], args[1], args[14], args[15], args[17]
        self.assertEqual(key.shape[0], 7)
        self.assertEqual(args[11], (3, 7))
        self.assertEqual(args[10][-1], query.shape[0])
        self.assertEqual(retained.numel(), 512+7*512)
        self.assertEqual(args[13].storage_offset(), 0)
        self.assertEqual(args[13].untyped_storage().nbytes(), 20*64*8)
        if phase == 0:
            retained[0] = 9
            self.workspaces.append(retained)
        else:
            self.assertEqual(int(retained[0]), 9)
        if phase == 2:
            args[18].copy_(positions[:, None, None].expand_as(args[18]))
            args[20].copy_(positions[:, None].expand_as(args[20]))
            plane = retained[512:].view(torch.float32).reshape(7, 128)
            plane.fill_(float(query[0, 0, 0])+1/1024)
            args[19].copy_(plane[:, None])
            args[21].fill_(float(query[0, 0, 0])*3)
        self.calls.append((phase, tuple(args[10])))

    def test_actual_cumulative_counts_explicit_positions_and_caller_order(self):
        """Repeated zero counts and caller permutations survive the three native phases."""
        for ids in ((6, 0, 2), (6,)):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(ids=ids, dtype=dtype):
                    forward = self._forward(ids, weight_dtype=dtype)
                    result = local_kl.local_kl_gradients(forward)
                    expected = torch.tensor([forward.saved.batch_meta.sequence_position(i)[1] for i in ids])
                    torch.testing.assert_close(result.gradients[0][:, 0, 0], expected.bfloat16())
                    torch.testing.assert_close(result.gradients[2][:, 0], expected.to(dtype))
                    self.assertEqual(result.loss.item(), 3)
                    self.assertEqual(result.key_partials.dtype, torch.float32)
                    self.assertFalse(torch.equal(result.key_partials, result.gradients[1].float()))
                    self.assertEqual(len({t.untyped_storage().data_ptr() for t in result.traces}), 3)
                    self.assertEqual(self.calls[-3:], [(i, forward.saved.local_query_layout.cumulative_queries)
                                                       for i in range(3)])

    def test_delayed_invocations_keep_fp32_key_and_loss_storage(self):
        """Later submissions cannot overwrite saved pre-cast key partials or raw loss."""
        first = local_kl.local_kl_gradients(self._forward(signature=1))
        second = local_kl.local_kl_gradients(self._forward(signature=3))
        self.assertIsNot(self.workspaces[0], self.workspaces[1])
        self.assertEqual(first.loss.item(), 3)
        self.assertEqual(second.loss.item(), 9)
        torch.testing.assert_close(first.key_partials, torch.full_like(first.key_partials, 1+1/1024), rtol=0, atol=0)

    def test_empty_tile_skips_arithmetic_with_owned_zero_global_key_contribution(self):
        """Empty ownership remains empty and supplies an independent FP32 owner-reduction contribution."""
        first = local_kl.local_kl_gradients(self._forward(()))
        second = local_kl.local_kl_gradients(self._forward(()))
        self.assertEqual(self.calls, [])
        self.assertEqual(first.gradients[0].shape, (0, 64, 128))
        self.assertEqual(first.key_partials.shape, (7, 128))
        self.assertEqual(first.loss.item(), 0)
        self.assertEqual(first.traces, ())
        self.assertEqual(first.retained.numel(), 0)
        self.assertNotEqual(first.key_partials.data_ptr(), second.key_partials.data_ptr())
        torch.testing.assert_close(first.key_partials, torch.zeros(7, 128), rtol=0, atol=0)

    def test_mutated_main_index_layout_and_geometry_reject_before_submission(self):
        """Admission failures cannot start partial native arithmetic."""
        for field in ("main", "index", "layout", "geometry", "global"):
            with self.subTest(field=field):
                forward = self._forward()
                if field == "main":
                    forward.saved.states[0].zero_()
                elif field == "index":
                    forward.index_states[1].zero_()
                elif field == "layout":
                    forward.saved.local_query_layout.positions.zero_()
                elif field == "geometry":
                    forward.saved.backend.scale = 0.2
                else:
                    forward = replace(forward, saved=replace(forward.saved, query_scope="global"))
                with self.assertRaises(ValueError):
                    local_kl.local_kl_gradients(forward)
        self.assertEqual(self.calls, [])

    def test_wrong_native_abi_rejects_even_empty_tile(self):
        """All CP members enforce the same payload admission before an empty-rank branch."""
        with patch.object(torch.ops.hyper_parallel, "dsa_local_mixed_kl_version", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "ABI 1"):
                local_kl.local_kl_gradients(self._forward(()))
        self.assertEqual(self.calls, [])
