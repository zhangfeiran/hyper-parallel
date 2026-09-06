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

import gc
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

    def test_dense_physical_gradients_expose_home_prefix_views(self):
        """Reuse dense physical-gradient storage for owner-local gradients.

        Feature: replica-gradient fast path
        Description: reduce contiguous physical gradients on a replica target rank
        Expectation: return contiguous home-prefix views without another full copy
        """
        manager = ExpertReplicaManager(self._config(rank_id=1))
        plan = self._hot_expert_zero_plan()
        grad_weight1 = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        grad_weight2 = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)

        with patch.object(replica_module, "_launch_transfers", return_value=()):
            home_grad_weight1, home_grad_weight2 = manager.reduce_replica_gradients(
                plan,
                grad_weight1,
                grad_weight2,
            )

        self.assertEqual(home_grad_weight1.data_ptr(), grad_weight1.data_ptr())
        self.assertEqual(home_grad_weight2.data_ptr(), grad_weight2.data_ptr())
        self.assertTrue(home_grad_weight1.is_contiguous())
        self.assertTrue(home_grad_weight2.is_contiguous())
        self.assertTrue(torch.equal(home_grad_weight1, grad_weight1[:2]))
        self.assertTrue(torch.equal(home_grad_weight2, grad_weight2[:2]))

    def test_ready_pool_reuses_storage_until_a_parameter_changes(self):
        """
        Feature: generation-aware persistent physical pool
        Description: repeat one binding, then mutate an optimizer-owned weight
        Expectation: skip the repeated P2P and isolate the refreshed storage
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

        def complete_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Populate each newly allocated remote suffix deterministically."""
            receive_tensors = [
                transfer.tensor
                for transfer in transfers
                if transfer.operation == "receive"
            ]
            if receive_tensors:
                receive_tensors[0].fill_(7.0)
                receive_tensors[1].fill_(9.0)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_receives,
        ) as mock_launch:
            first_pending = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            )
            first_weight1, first_weight2 = first_pending.wait()
            first_home_weight1 = first_weight1[:2].clone()
            reused_weight1, reused_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()
            with torch.no_grad():
                weight1.add_(5.0)
            refreshed_weight1, refreshed_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()

        self.assertEqual(mock_launch.call_count, 2)
        self.assertIsNone(first_pending.physical_weight1.grad_fn)
        self.assertIsNotNone(first_weight1.grad_fn)
        self.assertEqual(first_weight1.data_ptr(), reused_weight1.data_ptr())
        self.assertEqual(first_weight2.data_ptr(), reused_weight2.data_ptr())
        self.assertNotEqual(first_weight1.data_ptr(), refreshed_weight1.data_ptr())
        self.assertNotEqual(first_weight2.data_ptr(), refreshed_weight2.data_ptr())
        self.assertTrue(torch.equal(first_weight1[:2], first_home_weight1))
        self.assertTrue(torch.equal(refreshed_weight1[:2], weight1))

    def test_inflight_pool_is_not_reused_before_transfer_completion(self):
        """
        Feature: asynchronous physical pool publication
        Description: request the same binding before its first transfer is drained
        Expectation: allocate separately and cache only the completed latest pool
        """
        manager = ExpertReplicaManager(self._config(rank_id=0))
        plan = self._hot_expert_zero_plan()
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        with patch.object(
            replica_module,
            "_launch_transfers",
            return_value=(),
        ) as mock_launch:
            first = manager.start_prefetch(weight1, weight2, plan)
            second = manager.start_prefetch(weight1, weight2, plan)
            first.drain()
            second.drain()
            third = manager.start_prefetch(weight1, weight2, plan)
            third.drain()

        self.assertEqual(mock_launch.call_count, 2)
        self.assertNotEqual(
            first.physical_weight1.data_ptr(),
            second.physical_weight1.data_ptr(),
        )
        self.assertEqual(
            second.physical_weight1.data_ptr(),
            third.physical_weight1.data_ptr(),
        )

    def test_completed_graph_refreshes_updated_weights_in_place(self):
        """
        Feature: optimizer-aware physical pool refresh
        Description: finish backward, update home weights, and prefetch again
        Expectation: reuse storage while copying the new home parameter values
        """
        manager = ExpertReplicaManager(self._config(rank_id=1))
        plan = self._hot_expert_zero_plan()
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        def complete_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Populate mocked remote suffixes for both generations."""
            for transfer in transfers:
                if transfer.operation == "receive":
                    transfer.tensor.fill_(7.0)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_receives,
        ) as mock_launch:
            physical_weight1, physical_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()
            first_pointers = (
                physical_weight1.data_ptr(),
                physical_weight2.data_ptr(),
            )
            (physical_weight1.sum() + physical_weight2.sum()).backward()
            del physical_weight1, physical_weight2
            gc.collect()
            with torch.no_grad():
                weight1.add_(2.0)
                weight2.add_(3.0)
            refreshed_weight1, refreshed_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()

        self.assertEqual(mock_launch.call_count, 3)
        self.assertEqual(refreshed_weight1.data_ptr(), first_pointers[0])
        self.assertEqual(refreshed_weight2.data_ptr(), first_pointers[1])
        self.assertTrue(torch.equal(refreshed_weight1[:2], weight1))
        self.assertTrue(torch.equal(refreshed_weight2[:2], weight2))

    def test_compact_pool_aliases_home_rows_and_reuses_storage(self):
        """Attach only replica suffix capacity to optimizer-owned parameters.

        Feature: compact physical-weight storage
        Description: materialize, update, and rematerialize one compact pool
        Expectation: preserve logical shapes while reusing aliased physical storage
        """
        manager = ExpertReplicaManager(
            self._config(rank_id=1),
            compact_physical_weights=True,
        )
        plan = self._hot_expert_zero_plan()
        weight1 = torch.tensor(
            [[[10.0, 11.0], [12.0, 13.0]], [[20.0, 21.0], [22.0, 23.0]]],
            requires_grad=True,
        )
        weight2 = torch.tensor(
            [[[30.0, 31.0]], [[40.0, 41.0]]],
            requires_grad=True,
        )
        original_weight1 = weight1.detach().clone()
        original_weight2 = weight2.detach().clone()
        old_pointers = weight1.data_ptr(), weight2.data_ptr()

        def complete_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Populate the compact replica suffix deterministically."""
            receive_tensors = [
                transfer.tensor
                for transfer in transfers
                if transfer.operation == "receive"
            ]
            if receive_tensors:
                receive_tensors[0].fill_(7.0)
                receive_tensors[1].fill_(9.0)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_receives,
        ) as mock_launch:
            physical_weight1, physical_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()
            first_pointers = physical_weight1.data_ptr(), physical_weight2.data_ptr()
            del physical_weight1, physical_weight2
            gc.collect()
            physical_weight1, physical_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()
            (physical_weight1.sum() + physical_weight2.sum()).backward()
            del physical_weight1, physical_weight2
            gc.collect()
            with torch.no_grad():
                weight1.add_(2.0)
                weight2.add_(3.0)
            refreshed_weight1, refreshed_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()

        self.assertEqual(tuple(weight1.shape), (2, 2, 2))
        self.assertEqual(tuple(weight2.shape), (2, 1, 2))
        self.assertEqual(tuple(refreshed_weight1.shape), (3, 2, 2))
        self.assertEqual(tuple(refreshed_weight2.shape), (3, 1, 2))
        self.assertNotEqual(weight1.data_ptr(), old_pointers[0])
        self.assertNotEqual(weight2.data_ptr(), old_pointers[1])
        self.assertEqual(refreshed_weight1.data_ptr(), weight1.data_ptr())
        self.assertEqual(refreshed_weight2.data_ptr(), weight2.data_ptr())
        self.assertEqual(first_pointers, (weight1.data_ptr(), weight2.data_ptr()))
        self.assertEqual(
            weight1.untyped_storage().nbytes(),
            refreshed_weight1.numel() * refreshed_weight1.element_size(),
        )
        self.assertEqual(
            weight2.untyped_storage().nbytes(),
            refreshed_weight2.numel() * refreshed_weight2.element_size(),
        )
        self.assertTrue(torch.equal(weight1, original_weight1 + 2.0))
        self.assertTrue(torch.equal(weight2, original_weight2 + 3.0))
        self.assertTrue(
            torch.equal(refreshed_weight1[2], torch.full_like(weight1[0], 7.0))
        )
        self.assertTrue(
            torch.equal(refreshed_weight2[2], torch.full_like(weight2[0], 9.0))
        )
        self.assertEqual(mock_launch.call_count, 3)

    def test_compact_pool_detects_parameter_mutation_before_backward(self):
        """Fail closed when compact aliased weights change under a live graph.

        Feature: compact physical-weight autograd contract
        Description: mutate a home row after a value-saving physical operation
        Expectation: backward reports the shared version-counter violation
        """
        manager = ExpertReplicaManager(
            self._config(rank_id=1),
            compact_physical_weights=True,
        )
        plan = self._hot_expert_zero_plan()
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        with patch.object(replica_module, "_launch_transfers", return_value=()):
            physical_weight1, physical_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                plan,
            ).wait()
            loss = physical_weight1.square().sum() + physical_weight2.square().sum()
            with torch.no_grad():
                weight1.add_(1.0)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                loss.backward()

    def test_compact_pool_snapshots_an_overlapping_placement(self):
        """Preserve a live compact graph across a different placement.

        Feature: compact physical-weight graph isolation
        Description: materialize two different placements before either backward
        Expectation: use a transient snapshot, then resume the compact pool
        """
        manager = ExpertReplicaManager(
            self._config(rank_id=1),
            compact_physical_weights=True,
        )
        first_plan = self._hot_expert_zero_plan()
        second_plan = build_constrained_expert_placement(
            ((0, 4, 0, 0), (0, 4, 0, 0)),
            1,
            receive_capacity=4,
        )
        weight1 = torch.ones((2, 2, 2), requires_grad=True)
        weight2 = torch.ones((2, 1, 2), requires_grad=True)

        def complete_receives(
            transfers: tuple[Any, ...],
            _group: Any,
        ) -> tuple[Any, ...]:
            """Fill only remote receive targets for both placements."""
            for transfer in transfers:
                if transfer.operation == "receive":
                    transfer.tensor.fill_(2.0)
            return ()

        with patch.object(
            replica_module,
            "_launch_transfers",
            side_effect=complete_receives,
        ):
            first_weight1, first_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                first_plan,
            ).wait()
            first_loss = first_weight1.square().sum() + first_weight2.square().sum()
            second_weight1, second_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                second_plan,
            ).wait()
            second_loss = second_weight1.square().sum() + second_weight2.square().sum()

            self.assertEqual(first_weight1.data_ptr(), weight1.data_ptr())
            self.assertNotEqual(second_weight1.data_ptr(), weight1.data_ptr())
            first_loss.backward()
            second_loss.backward()
            del first_loss, second_loss
            del first_weight1, first_weight2, second_weight1, second_weight2
            gc.collect()

            resumed_weight1, resumed_weight2 = manager.start_prefetch(
                weight1,
                weight2,
                second_plan,
            ).wait()

        self.assertEqual(resumed_weight1.data_ptr(), weight1.data_ptr())
        self.assertEqual(resumed_weight2.data_ptr(), weight2.data_ptr())

    def test_shared_layers_reuse_independent_cached_pools(self):
        """
        Feature: bounded shared-layer physical pool cache
        Description: execute two parameter pairs and revisit the first pair
        Expectation: retain one ready physical pool for each shared layer
        """
        manager = ExpertReplicaManager(
            self._config(rank_id=0),
            pool_cache_size=2,
        )
        plan = self._hot_expert_zero_plan()
        first_weights = (
            torch.ones((2, 2, 2), requires_grad=True),
            torch.ones((2, 1, 2), requires_grad=True),
        )
        second_weights = tuple(
            torch.full_like(weight, 2.0, requires_grad=True)
            for weight in first_weights
        )

        with patch.object(
            replica_module,
            "_launch_transfers",
            return_value=(),
        ) as mock_launch:
            first_physical = manager.start_prefetch(
                *first_weights,
                plan,
            ).wait()
            manager.start_prefetch(*second_weights, plan).wait()
            reused_physical = manager.start_prefetch(
                *first_weights,
                plan,
            ).wait()

        self.assertEqual(mock_launch.call_count, 2)
        self.assertEqual(
            tuple(weight.data_ptr() for weight in first_physical),
            tuple(weight.data_ptr() for weight in reused_physical),
        )

    def test_pool_cache_size_must_be_positive(self):
        """
        Feature: bounded physical pool cache configuration
        Description: construct managers with invalid cache sizes
        Expectation: reject nonpositive and noninteger values
        """
        for cache_size in (0, -1, True, 1.5, "1"):
            with (
                self.subTest(cache_size=cache_size),
                self.assertRaisesRegex(ValueError, "pool_cache_size"),
            ):
                ExpertReplicaManager(
                    self._config(rank_id=0),
                    pool_cache_size=cache_size,
                )

    def test_compact_physical_weights_must_be_boolean(self):
        """Reject ambiguous compact storage selectors.

        Feature: compact physical-weight configuration
        Description: construct replica managers with non-Boolean selectors
        Expectation: every invalid selector fails at the manager boundary
        """
        for compact_physical_weights in (0, 1, 1.5, "true", None):
            with (
                self.subTest(compact_physical_weights=compact_physical_weights),
                self.assertRaisesRegex(ValueError, "compact_physical_weights"),
            ):
                ExpertReplicaManager(
                    self._config(rank_id=0),
                    compact_physical_weights=compact_physical_weights,
                )

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
