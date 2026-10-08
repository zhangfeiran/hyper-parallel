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
"""Validate shadow residency invalidations independently of devices and transport."""

from dataclasses import replace
import unittest

from tests.common.mark_utils import arg_mark
from tests.torch.expert_parallel.replica_residency import ResidentWeight, ResidencyLedger


class TestReplicaResidency(unittest.TestCase):
    """Ensure a stable slot cannot conceal a changed layer, version or heap."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_same_slot_requires_matching_content_identity(self) -> None:
        """
        Feature: Potential weight residency audit
        Description: Keep the same guest slot while changing layer, expert, version, layout and heap.
        Expectation: Only identical caller-owned content is a potential ordered hit.
        """
        ledger, slot = ResidencyLedger(), ("shared-pool", 0, 0)
        weight = ResidentWeight("layer-a", 7, 10, ("bfloat16", (8, 16), (16, 1)), 0)
        self.assertEqual(ledger.inspect(slot, weight), "empty")
        ledger.commit(slot, weight)
        self.assertEqual(ledger.inspect(slot, weight), "ordered_hit")
        cases = ((replace(weight, layer="layer-b"), "layer_changed"),
                 (replace(weight, expert=8), "expert_changed"),
                 (replace(weight, generation=11), "weight_updated"),
                 (replace(weight, layout=("bfloat16", (8, 16), (1, 8))), "layout_changed"),
                 (replace(weight, heap_generation=1), "heap_changed"))
        for current, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(ledger.inspect(slot, current), reason)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_unknown_version_and_partial_write_never_hit(self) -> None:
        """
        Feature: Conservative residency audit
        Description: Observe missing generations, failed writes and separate projections.
        Expectation: Unknown or failed content is forgotten without affecting an untouched projection.
        """
        ledger = ResidencyLedger()
        slots = (("shared-pool", 0, 0), ("shared-pool", 0, 1))
        weight = ResidentWeight("layer-a", 7, 10, ("bfloat16", (8, 16), (16, 1)), 0)
        for slot in slots:
            ledger.commit(slot, weight)
        unknown = replace(weight, generation=None)
        self.assertEqual(ledger.inspect(slots[0], unknown), "unknown_generation")
        ledger.commit(slots[0], unknown)
        self.assertEqual(ledger.inspect(slots[0], weight), "empty")
        self.assertEqual(ledger.inspect(slots[1], weight), "ordered_hit")
        ledger.invalidate(slots)
        self.assertEqual(ledger.entries, {})
