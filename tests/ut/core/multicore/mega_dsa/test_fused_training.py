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
"""Raw objective ownership and six-phase dependency failure checks without an NPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_dsa import fused_training
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule


class TestFusedTraining(unittest.TestCase):
    """Reject incomplete handoffs and retain independent raw derivatives across calls."""

    @staticmethod
    def _traces(groups):
        traces = []
        for epoch in range(1, 7):
            trace = torch.zeros(20, 64, dtype=torch.int64)
            for group in range(groups):
                tasks = tuple(range(group, 20, groups))
                trace[group, 0] = tasks[-1]
                for offset in (0, 16, 32):
                    trace[group, offset + 1:offset + 5] = torch.tensor(
                        [len(tasks), sum(task + 1 for task in tasks), tasks[-1], int(epoch < 3)])
                    trace[group, offset + 6] = epoch
            trace[groups, [6, 21, 22, 24, 38]] = epoch
            trace[groups, 25] = 3 * groups
            traces.append(trace)
        return tuple(traces)

    def test_dependency_closure_rejects_early_kl_and_stale_members(self):
        """KL cannot observe unfinished stats/initialization or reuse stale producer epochs."""
        traces = self._traces(7)
        schedule = MixedSfaSchedule(7)
        evidence = fused_training.validate_fused_training_traces(traces, schedule, require_ld=True)
        self.assertEqual(evidence["phase_order"],
                         ["li_main", "li_merge", "sfa", "kl_init", "kl_compute", "kl_post"])
        for phase, row, word, value in ((3, 7, 24, 3), (4, 7, 25, 20), (5, 3, 38, 5),
                                        (3, 4, 20, 1), (4, 4, 17, 0), (5, 8, 1, 1)):
            broken = tuple(trace.clone() for trace in traces)
            broken[phase][row, word] = value
            with self.subTest(phase=phase, row=row, word=word), self.assertRaises(ValueError):
                fused_training.validate_fused_training_traces(broken, schedule, require_ld=True)
        with self.assertRaisesRegex(ValueError, "six phase snapshots"):
            fused_training.validate_fused_training_traces(traces[:3], schedule, require_ld=True)

    @staticmethod
    def _run(signature, dtype):
        main = tuple(torch.full(shape, signature, dtype=torch.bfloat16)
                     for shape in ((2, 32, 512), (2, 512), (2, 32, 64), (2, 64)))
        index = (torch.ones(2, 64, 128, dtype=torch.bfloat16), torch.ones(2, 128, dtype=torch.bfloat16),
                 torch.ones(2, 64, dtype=dtype))
        layout = SimpleNamespace(length_tensor=torch.tensor([2], dtype=torch.int32), cumulative_lengths=(2,))
        return fused_training.fused_dsa_training_probe(index, main, layout, 0.1, MixedSfaSchedule(7))

    def test_single_launch_retains_raw_objective_and_distinct_invocations(self):
        """One submission produces selection, attention and raw KL without implicit loss scaling."""
        def _execute(*args):
            self.assertEqual(len(args), 23)
            self.assertEqual(args[9].shape, (6, 20, 64))
            self.assertEqual(args[12], (2,))
            self.assertFalse(args[10].is_set_to(args[11]))
            for output, multiplier in zip(args[14:], (1, 1, 1, 1, 1, 2, 3, 5, 7)):
                output.fill_(args[2][0, 0, 0].item() * multiplier)
        with patch.object(fused_training, "_load_native"), \
                patch.object(fused_training, "MIXED_INDEXER_WORKSPACE_BYTES", 64), \
                patch.object(fused_training, "mixed_kl_workspace_bytes", return_value=128), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_training_version", return_value=1, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_training_out",
                             side_effect=_execute, create=True) as op:
            first, second = self._run(1, torch.float32), self._run(3, torch.bfloat16)
        self.assertEqual(op.call_count, 2)
        for result, signature in ((first, 1), (second, 3)):
            for value, multiplier in zip(result.index_gradients, (2, 3, 5)):
                torch.testing.assert_close(value, torch.full_like(value, signature * multiplier))
            self.assertEqual(result.loss.item(), signature * 7)
            self.assertEqual(len(result.traces), 6)
            self.assertFalse(result.loss.requires_grad)
        self.assertEqual(first.index_gradients[2].dtype, torch.float32)
        self.assertEqual(second.index_gradients[2].dtype, torch.bfloat16)
        for before, after in zip((*first.forward, *first.index_gradients, *first.retained),
                                 (*second.forward, *second.index_gradients, *second.retained)):
            self.assertFalse(before.is_set_to(after))

    def test_incompatible_abi_and_deterministic_mode_reject_before_submission(self):
        """Reject a payload or execution mode that cannot honor the six-phase KL contract."""
        with patch.object(fused_training, "_load_native"), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_training_version", return_value=0, create=True), \
                patch.object(torch.ops.hyper_parallel, "dsa_fused_training_out", create=True) as op:
            with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
                self._run(1, torch.bfloat16)
            op.assert_not_called()
        with patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True), \
                patch.object(fused_training, "_load_native") as load:
            with self.assertRaisesRegex(ValueError, "non-deterministic"):
                self._run(1, torch.bfloat16)
            load.assert_not_called()
