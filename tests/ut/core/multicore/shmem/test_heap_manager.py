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
"""Generic byte reservations and collective layout agreement without MegaMoe."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.shmem import _heap_manager, _lifecycle
from tests.common.mark_utils import arg_mark


class TestShmemHeapManager(unittest.TestCase):
    """Check root coordination independently of any model-specific capacity policy."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_initial_layout_agreement(self) -> None:
        """Feature: initial layout agreement.
        Description: Compare peer dtype, budget, shape and reservation order.
        Expectation: Reject dtype, budget, shape and allocation order disagreement before initialization.
        """
        world = object()
        dist = SimpleNamespace(is_initialized=lambda: True, group=SimpleNamespace(WORLD=world),
                               get_rank=lambda _group: 0, get_world_size=lambda *_args: 2,
                               get_process_group_ranks=lambda _group: [0, 1], all_gather_object=Mock())
        reservations = ((object(), (512, ("first",), 1)), (object(), (1024, ("second",), 2)))
        for mismatch in (None, "shape", "order", "heap", "dtype"):
            def gather(peers: list, payload: tuple, **_kwargs: object) -> None:
                """Simulate a root peer proposing an incompatible reservation."""
                (heap, dtype, layouts), error = payload
                changes = {"shape": (heap, dtype, ((1, (), 1), layouts[1])),
                           "order": (heap, dtype, layouts[::-1]), "heap": ("different", dtype, layouts),
                           "dtype": (heap, "torch.float16", layouts)}
                peers[:] = [payload, (changes.get(mismatch, payload[0]), error)]

            with (self.subTest(mismatch=mismatch),
                  patch.object(_lifecycle, "_torch_modules", return_value=(torch, dist)),
                  patch.dict(_heap_manager.os.environ, {}, clear=True)):
                dist.all_gather_object.side_effect = gather
                if mismatch is None:
                    manager = _heap_manager.ShmemHeapManager(world, torch.empty(0), reservations)
                    self.assertEqual(manager.heap_bytes, 2 * 1024**2)
                    self.assertNotIn("HYPER_PARALLEL_SHMEM_HEAP_SIZE", _heap_manager.os.environ)
                else:
                    with self.assertRaisesRegex(RuntimeError, "differs across EP ranks"):
                        _heap_manager.ShmemHeapManager(world, torch.empty(0), reservations)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fixed_reservations_roll_back_and_owner_probe_errors_are_collective(self) -> None:
        """Feature: fixed reservations roll back and owner probe errors are collective.
        Description: Add a larger fixed reservation, then fail an owner state probe.
        Expectation: Preserve a fixed root layout and old heap when an owner cannot report its state.
        """
        key = object()
        dist = SimpleNamespace(is_initialized=lambda: False)
        with (patch.object(_lifecycle, "_torch_modules", return_value=(torch, dist)),
              patch.object(_lifecycle, "_retained_users", 0), patch.object(_lifecycle, "_shutdown_failed", False),
              patch.dict(_heap_manager.os.environ, {}, clear=True),
              patch.object(_heap_manager, "debug_state", return_value={"config": {"heap_size_bytes": 2 * 1024**2},
                                                                      "reference_count": 1, "active_allocations": []})):
            manager = _heap_manager.ShmemHeapManager(None, torch.empty(0), ((key, (512, (), None)),))
            owner = Mock()
            manager.bind(owner, key)
            with self.assertRaisesRegex(RuntimeError, "fixed SHMEM layout"):
                manager.reserve(((object(), (4 * 1024**2, (), None)),), allow_rebuild=False)
            self.assertEqual(len(manager.entries), 1)
            owner.heap_state.side_effect = RuntimeError("owner probe failed")
            with self.assertRaisesRegex(RuntimeError, "owner probe failed"):
                manager.replace(owner, (4 * 1024**2, (), None))
            owner.release_heap.assert_not_called()
            self.assertEqual(manager.state, "ready")
            self.assertEqual(manager.epoch, 0)
