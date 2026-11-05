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

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from examples.training_demo.deepseek_v41.deepseek_v41_random_init import (
    canonical_indices, initialize_random_shards,
)
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

    def test_chunk_indices_match_global_layout_for_each_shard(self) -> None:
        """Preserve row-major global indices through partial and transposed shards."""
        shape = (4, 5, 7)
        slices = (
            ((1, 3), (0, 5), (0, 7)),
            ((1, 2), (2, 4), (0, 7)),
            ((1, 3), (2, 4), (1, 6)),
        )
        for area in slices:
            for transpose in (False, True):
                with self.subTest(area=area, transpose=transpose):
                    global_ids = np.arange(140, dtype=np.uint64)
                    global_ids = (global_ids.reshape(4, 7, 5).swapaxes(-1, -2) if transpose
                                  else global_ids.reshape(shape))
                    expected = global_ids[tuple(slice(begin, end) for begin, end in area)].reshape(-1)[3:10]
                    np.testing.assert_array_equal(canonical_indices(3, 7, shape, area, transpose), expected)

    def test_constant_parameter_needs_no_global_index_materialization(self) -> None:
        """Constant parameters retain their bits and signature without allocating indices."""
        model = torch.nn.LayerNorm(6)
        trainer = SimpleNamespace(base=SimpleNamespace(model=model, optimizer=Mock()))
        with patch("examples.training_demo.deepseek_v41.deepseek_v41_random_init.canonical_indices",
                   side_effect=AssertionError("Constant initialization built global indices")):
            report = initialize_random_shards(trainer, seed=42, backend="owner_ep")
        torch.testing.assert_close(model.weight, torch.ones(6), rtol=0, atol=0)
        torch.testing.assert_close(model.bias, torch.zeros(6), rtol=0, atol=0)
        self.assertEqual(int(report["fp32_bits_sum_mod_2_64"]), 6 * 0x3f800000)

    def test_bounded_cpu_prefetch_keeps_random_weights_and_signature(self) -> None:
        """Parallel generation preserves the sequential bit pattern and optimizer reload."""
        module = "examples.training_demo.deepseek_v41.deepseek_v41_random_init"
        reference = SimpleNamespace(base=SimpleNamespace(model=torch.nn.Embedding(64, 32), optimizer=Mock()))
        candidate = SimpleNamespace(base=SimpleNamespace(model=torch.nn.Embedding(64, 32), optimizer=Mock()))
        with (patch(f"{module}._CHUNK_ELEMENTS", 127), patch(f"{module}._INITIALIZATION_WORKERS", 1)):
            expected = initialize_random_shards(reference, seed=42, backend="owner_ep")
        with (patch(f"{module}._CHUNK_ELEMENTS", 127),
              patch(f"{module}.os.sched_getaffinity", return_value={0, 1}),
              patch(f"{module}.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pool):
            actual = initialize_random_shards(candidate, seed=42, backend="owner_ep")
        pool.assert_called_once_with(max_workers=2)
        candidate.base.optimizer.reload_model_params.assert_called_once()
        self.assertEqual(actual, expected)
        torch.testing.assert_close(candidate.base.model.weight, reference.base.model.weight, rtol=0, atol=0)

    def test_coordinator_reloads_dense_masters_without_dense_host_state(self) -> None:
        """Random initialization must refresh only the dense FP32 masters."""
        dense = Mock(optimizers_dict={})
        parameter = torch.nn.Parameter(torch.ones((4, 2)))
        sparse = torch.optim.SparseAdam([parameter])
        optimizer = HostSparseOptimizerCoordinator(dense, sparse, {})
        optimizer.reload_model_params()
        dense.reload_model_params.assert_called_once_with()
        self.assertFalse(sparse.state)
