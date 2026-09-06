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
"""Unit tests for bounded Torch expert weight replicas."""

import unittest
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    build_constrained_expert_placement,
)
from hyper_parallel.core.multicore.modules.mega_moe import (
    replica as replica_module,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    MegaMoeSpec,
)
from hyper_parallel.core.multicore.modules.mega_moe.replica import (
    ExpertReplicaManager,
)


class TestExpertReplicaManager(unittest.TestCase):
    """Validate weight prefetch and owner-directed gradients on CPU."""

    @staticmethod
    def _config(rank_id: int) -> MegaMoeSpec:
        """Build a two-rank configuration with one replica slot per rank."""
        return MegaMoeSpec(
            expert_capacity_factor=1.5,
            local_num_tokens=2,
            hidden_size=2,
            intermediate_size=1,
            num_experts=4,
            top_k=2,
            receive_capacity=4,
            ep_size=2,
            ep_group="ep_group",
            rank_id=rank_id,
            num_cube_cores=24,
            replica_slots_per_rank=1,
        )

    @staticmethod
    def _hot_expert_zero_plan():
        """Replicate logical expert zero from rank zero onto rank one."""
        return build_constrained_expert_placement(
            ((4, 0, 0, 0), (4, 0, 0, 0)),
            1,
            receive_capacity=4,
        )

    def test_prefetch_populates_remote_replica_after_completion(self):
        """
        Feature: bounded asynchronous expert prefetch
        Description: receive one remote expert into rank one's fixed suffix slot
        Expectation: expose home rows followed by the completed remote weights
        """
        manager = ExpertReplicaManager(self._config(rank_id=1))
        plan = self._hot_expert_zero_plan()
        weight1 = torch.tensor(
            [[[10.0, 11.0], [12.0, 13.0]], [[20.0, 21.0], [22.0, 23.0]]],
            requires_grad=True,
        )
        weight2 = torch.tensor(
            [[[30.0, 31.0]], [[40.0, 41.0]]],
            requires_grad=True,
        )
        remote_weight1 = torch.full_like(weight1[0], 7.0)
        remote_weight2 = torch.full_like(weight2[0], 9.0)

        def complete_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Populate mocked receive buffers before returning completion."""
            receive_tensors = [
                transfer.tensor
                for transfer in transfers
                if transfer.operation == "receive"
            ]
            receive_tensors[0].copy_(remote_weight1)
            receive_tensors[1].copy_(remote_weight2)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_receives,
        ) as mock_launch:
            pending = manager.start_prefetch(weight1, weight2, plan)
            physical_weight1, physical_weight2 = pending.wait()

        self.assertEqual(tuple(physical_weight1.shape), (3, 2, 2))
        self.assertEqual(tuple(physical_weight2.shape), (3, 1, 2))
        self.assertTrue(torch.equal(physical_weight1[:2], weight1))
        self.assertTrue(torch.equal(physical_weight2[:2], weight2))
        self.assertTrue(torch.equal(physical_weight1[2], remote_weight1))
        self.assertTrue(torch.equal(physical_weight2[2], remote_weight2))
        transfers = mock_launch.call_args.args[0]
        self.assertEqual(
            tuple((transfer.operation, transfer.peer_rank) for transfer in transfers),
            (("receive", 0), ("receive", 0)),
        )

    def test_backward_reduces_replica_gradients_to_home_owner(self):
        """
        Feature: owner-directed replica gradient reduction
        Description: backpropagate through rank zero's physical expert pool
        Expectation: add rank one's replica gradients only to expert zero
        """
        manager = ExpertReplicaManager(self._config(rank_id=0))
        plan = self._hot_expert_zero_plan()
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        def complete_gradient_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Fill owner receive buffers with deterministic replica gradients."""
            self.assertTrue(
                all(transfer.tensor.is_contiguous() for transfer in transfers)
            )
            receive_tensors = [
                transfer.tensor
                for transfer in transfers
                if transfer.operation == "receive"
            ]
            if receive_tensors:
                receive_tensors[0].fill_(2.0)
                receive_tensors[1].fill_(3.0)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_gradient_receives,
        ):
            pending = manager.start_prefetch(weight1, weight2, plan)
            physical_weight1, physical_weight2 = pending.wait()
            (physical_weight1.sum() + physical_weight2.sum()).backward()

        self.assertTrue(torch.equal(weight1.grad[0], torch.full_like(weight1[0], 3.0)))
        self.assertTrue(torch.equal(weight1.grad[1], torch.ones_like(weight1[1])))
        self.assertTrue(torch.equal(weight2.grad[0], torch.full_like(weight2[0], 4.0)))
        self.assertTrue(torch.equal(weight2.grad[1], torch.ones_like(weight2[1])))

    def test_generation_changes_only_when_slot_binding_changes(self):
        """
        Feature: physical placement generations
        Description: reuse one slot map and then bind a different hot expert
        Expectation: retain the generation for reuse and advance it for rebinding
        """
        manager = ExpertReplicaManager(self._config(rank_id=0))
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)
        first_plan = self._hot_expert_zero_plan()
        second_plan = build_constrained_expert_placement(
            ((0, 0, 4, 0), (0, 0, 4, 0)),
            1,
            receive_capacity=4,
        )

        with patch.object(replica_module, "_launch_transfers", return_value=()):
            first = manager.start_prefetch(weight1, weight2, first_plan)
            reused = manager.start_prefetch(weight1, weight2, first_plan)
            rebound = manager.start_prefetch(weight1, weight2, second_plan)
            first.drain()
            reused.drain()
            rebound.drain()

        self.assertEqual(first.generation, 1)
        self.assertEqual(reused.generation, 1)
        self.assertEqual(rebound.generation, 2)
        self.assertEqual(manager.generation, 2)

    def test_pending_weights_cannot_materialize_twice(self):
        """
        Feature: physical pool ownership
        Description: attempt to attach autograd twice to one prefetched pool
        Expectation: reject the second materialization before storage is aliased
        """
        manager = ExpertReplicaManager(self._config(rank_id=0))
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        with patch.object(replica_module, "_launch_transfers", return_value=()):
            pending = manager.start_prefetch(
                weight1,
                weight2,
                self._hot_expert_zero_plan(),
            )
            pending.wait()
            with self.assertRaisesRegex(RuntimeError, "only once"):
                pending.wait()


if __name__ == "__main__":
    unittest.main()
