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
"""Canonical initialization must preserve Host owner row identity."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from examples.training_demo.deepseek_v41.deepseek_v41_random_init import initialize_random_shards
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.deepseek_v41.adapter.optim.host_sparse import HostSparseOptimizerCoordinator


class TestHostRandomInitialization(unittest.TestCase):
    """Compare EP owner chunks to one canonical unsharded table."""

    def test_owner_rows_match_unsharded_initialization(self) -> None:
        """Changing EP owner placement must preserve every global element."""
        reference = torch.nn.Module()
        reference.embed = torch.nn.Embedding(16, 3)
        trainer = SimpleNamespace(base=SimpleNamespace(model=reference, optimizer=Mock()))
        expected = initialize_random_shards(trainer, seed=42, backend="owner_ep")
        owners = []
        signatures = []
        for rank in range(2):
            table = HostEngramTable(
                source_weight=torch.nn.Parameter(torch.empty((16, 3), device="meta")), logical_rows=16,
            )
            table.weight = torch.nn.Parameter(torch.empty((8, 3), device="meta"))
            table.bind_planned_shard(ep_rank=rank, ep_size=2, ep_group=object())
            torch.utils.swap_tensors(table.weight, torch.nn.Parameter(torch.empty((8, 3))))
            model = torch.nn.Module()
            model.embed = table
            model._hp_external_state = SimpleNamespace(
                parameters={"embed.weight": table.weight}, tables={"embed": table},
            )
            trainer = SimpleNamespace(base=SimpleNamespace(model=model, optimizer=Mock()))
            with patch("examples.training_demo.deepseek_v41.deepseek_v41_random_init._CHUNK_ELEMENTS", 5):
                signatures.append(initialize_random_shards(trainer, seed=42, backend="owner_ep"))
            owners.append(table.weight.detach().clone())
        torch.testing.assert_close(torch.cat(owners), reference.embed.weight, rtol=0, atol=0)
        checksum = sum(int(item["fp32_bits_sum_mod_2_64"]) for item in signatures) % 2**64
        self.assertEqual(checksum, int(expected["fp32_bits_sum_mod_2_64"]))

    def test_coordinator_reloads_dense_masters_without_dense_host_state(self) -> None:
        """Random initialization must refresh only the dense FP32 masters."""
        dense = Mock(optimizers_dict={})
        parameter = torch.nn.Parameter(torch.ones((4, 2)))
        sparse = torch.optim.SparseAdam([parameter])
        optimizer = HostSparseOptimizerCoordinator(dense, sparse, {})
        optimizer.reload_model_params()
        dense.reload_model_params.assert_called_once_with()
        self.assertFalse(sparse.state)
