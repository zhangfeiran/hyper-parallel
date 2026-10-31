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
"""Differential tests for the exact receiver/expert selection objective."""

import random
import unittest

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.expert_parallel.hot_replica.planner import _improve_copies


def _exhaustive_improve(copies: list[dict[int, int]], config: ExpertReplicaConfig, target: int) -> None:
    """Enumerate the documented move objective, including zero-key residency."""
    home = config.home_experts
    while True:
        loads = [sum(row.values()) for row in copies]
        for donor in sorted(range(config.ep_size), key=lambda rank: (-loads[rank], rank)):
            moves = []
            for receiver in range(config.ep_size):
                guests = sum(expert // home != receiver for expert in copies[receiver])
                for expert in range(donor * home, (donor + 1) * home):
                    existing = expert in copies[receiver]
                    if not existing and guests >= config.replica_slots_per_rank:
                        continue
                    rows = min(copies[donor][expert], loads[donor] - target, target - loads[receiver])
                    if rows > 0:
                        moves.append((rows, int(existing), -receiver, -expert))
            if moves:
                rows, _, negative_receiver, negative_expert = max(moves)
                receiver, expert = -negative_receiver, -negative_expert
                copies[donor][expert] -= rows
                copies[receiver][expert] = copies[receiver].get(expert, 0) + rows
                break
        else:
            return


class TestReplicaPlannerFast(unittest.TestCase):
    """Require exact choices against exhaustive enumeration, not just legality."""

    def test_random_initial_layouts_match_exhaustive_move_order(self) -> None:
        """Cover existing guests, full budgets, zero keys and empty owner quotas."""
        generator = random.Random(8087)
        for size, home in ((2, 2), (4, 3), (6, 6)):
            for budget in (0, 1, 2, 6):
                config = ExpertReplicaConfig(size * home, size, budget)
                for iteration in range(30):
                    copies = [{expert: generator.randrange(30) for expert in range(rank * home, (rank + 1) * home)}
                              for rank in range(size)]
                    for receiver in range(size):
                        candidates = [expert for expert in range(size * home) if expert // home != receiver]
                        for expert in generator.sample(candidates, min(budget, len(candidates))):
                            owner = expert // home
                            rows = generator.randrange(copies[owner][expert] + 1)
                            copies[owner][expert] -= rows
                            copies[receiver][expert] = rows
                    target = generator.randrange(max(map(lambda row: sum(row.values()), copies)) + 20)
                    expected, actual = [dict(row) for row in copies], [dict(row) for row in copies]
                    _exhaustive_improve(expected, config, target)
                    _improve_copies(actual, config, target)
                    with self.subTest(size=size, home=home, budget=budget, iteration=iteration):
                        self.assertEqual(actual, expected)
                        self.assertEqual([tuple(row.items()) for row in actual],
                                         [tuple(row.items()) for row in expected])

    def test_existing_copy_wins_a_clipped_quota_tie(self) -> None:
        """A smaller existing expert can beat the first larger new expert."""
        config = ExpertReplicaConfig(6, 3, 2)
        copies = [{0: 20, 1: 15}, {2: 0, 3: 0, 1: 0}, {4: 10, 5: 10}]
        expected = [dict(row) for row in copies]
        _exhaustive_improve(expected, config, 20)
        _improve_copies(copies, config, 20)
        self.assertEqual(copies, expected)
        self.assertEqual(copies[1][1], 15)

    def test_stalled_hottest_owner_can_be_discarded(self) -> None:
        """A rank overloaded by guests cannot gain movable home work later."""
        config = ExpertReplicaConfig(8, 4, 1)
        copies = [{0: 0, 1: 0, 2: 200}, {2: 140, 3: 0}, {4: 0, 5: 0}, {6: 0, 7: 0}]
        expected = [dict(row) for row in copies]
        _exhaustive_improve(expected, config, 50)
        _improve_copies(copies, config, 50)
        self.assertEqual(copies, expected)
        self.assertEqual(copies[2][2], 50)
        self.assertEqual(copies[3][2], 40)


if __name__ == '__main__':
    unittest.main()
