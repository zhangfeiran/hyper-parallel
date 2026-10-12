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
"""FP32 CP storage, detached teacher, and authentic fused-selection contracts over CPU mocks."""

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fp32_fused_kl
from hyper_parallel.core.multicore.modules.mega_dsa.fp32_index_kl import (
    Fp32IndexKlProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from tests.ut.core.multicore.mega_dsa import test_local_training as training_fixture


class TestFp32FusedKl(unittest.TestCase):
    """No process group initialization or device runtime is used by these tests."""

    def test_owner_gather_waits_then_reorders_and_ignores_empty_peer_padding(self) -> None:
        """Native owner buffers may be reordered and an empty owner still joins the key gather."""
        for empty in (False, True):
            with self.subTest(empty=empty):
                members = (3, 7, 11)
                rank = 1 if empty else 0
                local_ids = () if empty else (2, 0)
                meta = DsaBatchMeta((0, 3), local_ids, local_ids, (0, 2, 0), (1, 0, 0),
                                    cp_ranks=members, root_pes=(0, 1, 2), cp_rank=rank)
                group = object()
                handle = Mock()

                def _declare(peers, declaration, **_kwargs):
                    peers[:] = [(declaration[0], ids) for ids in ((2, 0), (), (1,))]

                def _gather(output, local, completion=handle, **_kwargs):
                    self.assertEqual(local.shape, (2, 128))
                    output.fill_(torch.nan)
                    def _complete():
                        output[0].fill_(3.)
                        output[1].fill_(1.)
                        output[4].fill_(2.)
                    completion.wait.side_effect = _complete
                    return completion

                with patch.object(fp32_fused_kl.dist, 'get_process_group_ranks', return_value=members), \
                     patch.object(fp32_fused_kl.dist, 'get_rank', return_value=rank), \
                     patch.object(fp32_fused_kl.dist, 'all_gather_object', side_effect=_declare), \
                     patch.object(fp32_fused_kl.dist, 'all_gather_into_tensor', side_effect=_gather):
                    backend = fp32_fused_kl.Fp32OwnerKeyGather(meta, group, 'cpu')
                    key = torch.empty((0, 128)) if empty else torch.tensor([3., 1.])[:, None].expand(2, 128)
                    result = backend.gather(key)
                    torch.testing.assert_close(result, torch.tensor([1., 2., 3.])[:, None].expand(3, 128))
                    handle.wait.assert_called_once()
                    backend.global_order = backend.global_order.clone()
                    with self.assertRaisesRegex(ValueError, 'declaration changed'):
                        backend.gather(key)

    def test_teacher_head_mass_sums_before_normalization_and_has_no_main_autograd_edge(self) -> None:
        """Partitioned teacher head mass reconstructs the complete selected detached teacher."""
        generator = torch.Generator().manual_seed(130)
        main = (torch.randn((3, 4, 512), generator=generator), torch.randn((3, 512), generator=generator),
                torch.randn((3, 4, 64), generator=generator), torch.randn((3, 64), generator=generator))
        ids = torch.full((3, 2048), -1, dtype=torch.int32)
        for row in range(3):
            ids[row, :row + 1] = torch.arange(row + 1)
        full = fp32_fused_kl.selected_teacher_mass(main, ids, scale=.1)
        partials = [fp32_fused_kl.selected_teacher_mass((main[0][:, head], main[1], main[2][:, head], main[3]),
                                                     ids, scale=.1) for head in (slice(0, 2), slice(2, 4))]
        teacher = sum(partials)
        teacher /= teacher.sum(-1, keepdim=True)
        expected = full / full.sum(-1, keepdim=True)
        torch.testing.assert_close(teacher, expected)
        self.assertFalse(teacher.requires_grad)
        self.assertTrue(torch.equal(teacher.masked_select(ids < 0), torch.zeros_like(teacher.masked_select(ids < 0))))
        invalid = (main[0].requires_grad_(), *main[1:])
        with self.assertRaisesRegex(ValueError, 'detached'):
            fp32_fused_kl.selected_teacher_mass(invalid, ids, scale=.1)

    def test_native_saved_selection_restores_caller_rows_and_rejects_mutated_producer(self) -> None:
        """A device selection entry must carry the authentic fused producer's saved-state guards."""
        fixture = training_fixture.TestLocalTraining()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        layout, tensors = fixture._inputs()
        result = fixture._forward(fixture.invocation, tuple(t.detach() for t in tensors[:4]),
                                  tuple(t.detach() for t in tensors[4:]), query_layout=layout)
        saved = result.saved
        for row, query in enumerate(layout.global_ids):
            saved.indices[row, 0, :query + 1] = torch.arange(query + 1, dtype=torch.int32)
        saved = replace(saved, versions=tuple(t._version for t in (*saved.states, saved.indices, *saved.forward)))
        probe = Fp32IndexKlProbe.from_saved_attention(saved)
        self.assertEqual(probe.batch_meta.q_global_ids, (2, 0))
        self.assertEqual(probe.batch_meta.kv_global_ids, (0, 1, 2))
        self.assertEqual(probe.indices[0, :4].tolist(), [0, 1, 2, -1])
        self.assertEqual(probe.indices[1, :2].tolist(), [0, -1])
        with self.assertRaisesRegex(ValueError, 'native local'):
            Fp32IndexKlProbe.from_saved_attention(object())
        saved.indices.add_(1)
        with self.assertRaisesRegex(ValueError, 'modified'):
            probe._check_native()
