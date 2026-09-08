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
"""Check simulation payloads against real manager transfer descriptors."""

import unittest
from collections import Counter
from types import SimpleNamespace

import torch

from examples.torch.multicore.megamoe_simulator.model import (
    Costs,
    Training,
    TrainingGraph,
)
from examples.torch.multicore.megamoe_simulator.workload import (
    Shape,
    build_plan,
    make_counts,
)
from hyper_parallel.core.multicore.modules.mega_moe.replica import ExpertReplicaManager


class TestProductionReplicaPayloads(unittest.TestCase):
    """Exercise actual descriptor builders using metadata-only tensors, with no communication."""

    def test_all_rank_weight_and_gradient_payloads_match_runtime(self):
        shape = Shape(experts=64, tokens_per_rank=128, top_k=2, hidden=16, intermediate=8)
        for ranks in (16, 32, 64):
            with self.subTest(ranks=ranks):
                plan = build_plan(make_counts(shape, ranks, hot_fraction=1), 1, shape.capacity)
                real_weights, real_gradients = Counter(), Counter()
                for rank in range(ranks):
                    spec = SimpleNamespace(rank_id=rank, local_experts=shape.experts // ranks, replica_slots_per_rank=1)
                    manager = ExpertReplicaManager(spec)
                    home = (
                        torch.empty(
                            (spec.local_experts, shape.hidden, 2 * shape.intermediate),
                            dtype=torch.bfloat16,
                            device="meta",
                        ),
                        torch.empty(
                            (spec.local_experts, shape.intermediate, shape.hidden), dtype=torch.bfloat16, device="meta"
                        ),
                    )
                    suffix = tuple(
                        torch.empty((1, *weight.shape[1:]), dtype=weight.dtype, device="meta") for weight in home
                    )
                    transfers = manager._prefetch_transfers(plan, *home, *suffix, pool_start=spec.local_experts)
                    self._count_sends(real_weights, transfers, rank)
                    gradients = tuple(weight.to(torch.float32) for weight in home)
                    replica_gradients = tuple(weight.to(torch.float32) for weight in suffix)
                    transfers, _ = manager._gradient_transfers(plan, *gradients, replica_gradients=replica_gradients)
                    self._count_sends(real_gradients, transfers, rank)
                graph = TrainingGraph(shape, [plan], Costs(), Training(layers=1, steps=1))
                tasks = graph.build()
                for phase, real in (("weight_prefetch", real_weights), ("replica_gradient", real_gradients)):
                    simulated = Counter(
                        (task.rank, task.target, task.size_bytes) for task in tasks if task.phase == phase
                    )
                    self.assertEqual(simulated, real)

    @staticmethod
    def _count_sends(counter, transfers, rank):
        for transfer in transfers:
            if transfer.operation == "send":
                counter[(rank, transfer.peer_rank, transfer.tensor.numel() * transfer.tensor.element_size())] += 1
