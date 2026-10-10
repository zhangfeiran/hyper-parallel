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
"""Local query position, caller order and native admission contracts."""

from dataclasses import replace
from itertools import combinations
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import local_query
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestLocalQuery(unittest.TestCase):
    """Use genuine local shapes and explicit logical positions, including empty sequence buckets."""

    @staticmethod
    def _layout(ids=(9, 2, 7, 0)):
        return local_query.DsaLocalQueryLayout(replace(DsaBatchMeta.packed((3, 4, 5)), q_global_ids=ids), "cpu")

    def test_sequence_positions_and_permutations_preserve_caller_rows(self):
        """Local rows pack by sequence, retaining logical positions and genuine repeated cumulative counts."""
        layout = self._layout()
        self.assertEqual(layout.global_ids, (0, 2, 7, 9))
        self.assertEqual(layout.cumulative_queries, (2, 2, 4))
        torch.testing.assert_close(layout.positions, torch.tensor([0, 2, 0, 2]))
        torch.testing.assert_close(layout.query_lengths, torch.tensor([2, 2, 4], dtype=torch.int32))
        value = torch.tensor(layout.batch_meta.q_global_ids)[:, None]
        packed = layout.pack(value)
        torch.testing.assert_close(packed[:, 0], torch.tensor(layout.global_ids))
        torch.testing.assert_close(layout.restore(packed), value)
        self.assertEqual(layout.causal_bounds(), ((1, 2), (3, 3), (1, 4), (3, 5)))

    def test_right_down_is_only_a_conservative_bound_for_every_sorted_subset(self):
        """Exhaustively prove the existing partition never omits the explicit-position candidate prefix."""
        for count in range(6):
            for ids in combinations(range(5), count):
                with self.subTest(ids=ids):
                    layout = local_query.DsaLocalQueryLayout(
                        replace(DsaBatchMeta.packed((5,)), q_global_ids=tuple(reversed(ids))), "cpu")
                    self.assertEqual(layout.cumulative_queries, (count,))
                    bounds = layout.causal_bounds()
                    self.assertTrue(all(exact <= conservative for exact, conservative in bounds))
                    self.assertEqual(tuple(exact for exact, _ in bounds), tuple(token + 1 for token in ids))

    def test_empty_owner_keeps_zero_counts_and_zero_tensor_shapes(self):
        """An empty owner carries no fake query, weight or position row."""
        layout = self._layout(())
        self.assertEqual(layout.cumulative_queries, (0, 0, 0))
        self.assertEqual(layout.positions.shape, (0,))
        self.assertEqual(layout.causal_bounds(), ())
        self.assertEqual(layout.restore(layout.pack(torch.empty(0, 64, 128))).shape, (0, 64, 128))

    def test_metadata_mutation_and_replacement_reject_before_loading_native(self):
        """Both in-place mutation and same-version replacement invalidate prepared address tensors."""
        for name in ("positions", "pack_order", "restore_order", "query_lengths", "key_lengths"):
            for replacement in (False, True):
                with self.subTest(name=name, replacement=replacement):
                    layout = self._layout()
                    tensor = getattr(layout, name)
                    if replacement:
                        setattr(layout, name, tensor.clone())
                    else:
                        tensor.add_(0)
                    with patch.object(local_query, "_load_native") as load, self.assertRaises(ValueError):
                        local_query.local_indexer_forward_probe(torch.empty(4,64,128,dtype=torch.bfloat16),
                            torch.empty(12,1,128,dtype=torch.bfloat16), torch.empty(4,64,dtype=torch.bfloat16),
                            layout, MixedSfaSchedule(7))
                    load.assert_not_called()

    def test_native_submission_packs_only_local_rows_and_restores_owned_outputs(self):
        """The native boundary receives real local lengths and positions, independent trace/output storage."""
        for ids in ((9, 2, 7, 0), ()):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(ids=ids, dtype=dtype):
                    layout = self._layout(ids)
                    query = torch.tensor(ids, dtype=torch.bfloat16)[:, None, None].expand(-1,64,128).clone()
                    weights = torch.ones(len(ids),64,dtype=dtype)
                    key = torch.ones(12,1,128,dtype=torch.bfloat16)

                    def _submit(*args):
                        self.assertEqual(args[0].shape[0], len(ids))
                        torch.testing.assert_close(args[0][:,0,0], torch.tensor(layout.global_ids,dtype=query.dtype))
                        self.assertIs(args[3], layout.query_lengths)
                        self.assertIs(args[4], layout.key_lengths)
                        self.assertIs(args[8], layout.positions)
                        self.assertEqual(args[7].numel(), local_query.MIXED_INDEXER_WORKSPACE_BYTES if ids else 0)
                        args[9].fill_(-1)
                        args[9][:,0,0] = torch.tensor(layout.global_ids,dtype=torch.int32)
                        args[10].fill_(1)
                        args[6].fill_(3)

                    with patch.object(local_query, "_load_native"), \
                            patch.object(torch.ops.hyper_parallel, "dsa_local_indexer_version",
                                         return_value=1, create=True), \
                            patch.object(torch.ops.hyper_parallel, "dsa_local_indexer_out",
                                         side_effect=_submit, create=True) as submit:
                        indices, values, traces, _scratch = local_query.local_indexer_forward_probe(
                            query,key,weights,layout,MixedSfaSchedule(7))
                    submit.assert_called_once()
                    torch.testing.assert_close(indices[:,0,0], torch.tensor(ids,dtype=torch.int32))
                    self.assertEqual(len(traces), 2)
                    self.assertTrue(bool((values == 1).all()))

    def test_invalid_states_reject_before_native_submission(self):
        """Local query count, complete key storage and detached training boundaries are explicit."""
        layout = self._layout()
        states = (torch.ones(4,64,128,dtype=torch.bfloat16), torch.ones(12,1,128,dtype=torch.bfloat16),
                  torch.ones(4,64,dtype=torch.float32))
        for field, value in ((0,states[0][:3]), (0,states[0].float()), (1,states[1][:4]),
                              (2,states[2].clone().requires_grad_())):
            invalid = list(states)
            invalid[field] = value
            with self.subTest(field=field), patch.object(local_query,"_load_native") as load, \
                    self.assertRaises(ValueError):
                local_query.local_indexer_forward_probe(invalid[0], invalid[1], invalid[2], layout, MixedSfaSchedule(7))
            load.assert_not_called()
