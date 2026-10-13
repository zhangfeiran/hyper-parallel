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

from __future__ import annotations

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
            def gather(peers: list, payload: tuple, *, mismatch_case: str | None = mismatch,
                       **unused_kwargs: object) -> None:
                """Simulate a root peer proposing an incompatible reservation."""
                del unused_kwargs
                (heap, dtype, layouts), error = payload
                changes = {"shape": (heap, dtype, ((1, (), 1), layouts[1])),
                           "order": (heap, dtype, layouts[::-1]), "heap": ("different", dtype, layouts),
                           "dtype": (heap, "torch.float16", layouts)}
                peers[:] = [payload, (changes.get(mismatch_case, payload[0]), error)]

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
    def test_growth_preflight_and_failed_rebuild(self) -> None:
        """Feature: Heap replacement failure boundaries.
        Description: Reject unsafe ownership and then fail the native rebuild.
        Expectation: Preflight preserves the old heap; a failed rebuild blocks further execution.
        """
        key = object()
        dist = SimpleNamespace(is_initialized=lambda: False)
        with (patch.object(_lifecycle, "_torch_modules", return_value=(torch, dist)),
              patch.object(_lifecycle, "_retained_users", 0), patch.object(_lifecycle, "_shutdown_failed", False),
              patch.dict(_heap_manager.os.environ, {}, clear=True),
              patch.object(torch.npu, "synchronize") as synchronize,
              patch.object(torch.npu, "is_current_stream_capturing", return_value=False) as capture,
              patch.object(_lifecycle, "_reinitialize", side_effect=RuntimeError("native rebuild failed")) as rebuild,
              patch.object(_heap_manager, "debug_state", return_value={"config": {"heap_size_bytes": 2 * 1024**2},
                                                                      "reference_count": 1,
                                                                      "active_allocations": []}) as debug):
            manager = _heap_manager.ShmemHeapManager(None, torch.empty(0), ((key, (512, (), None)),))
            owner = Mock()
            manager.bind(owner, key)
            with self.assertRaisesRegex(RuntimeError, "fixed SHMEM layout"):
                manager.reserve(((object(), (4 * 1024**2, (), None)),), allow_rebuild=False)
            self.assertEqual(len(manager.entries), 1)
            for error in ("owner probe failed", "unmanaged SHMEM owner", "unmanaged SHMEM allocation",
                          "active workspace lease", "graph capture", "explicit heap budget", "sync failed"):
                with self.subTest(error=error):
                    owner.heap_state.side_effect = RuntimeError(error) if error == "owner probe failed" else None
                    owner.heap_state.return_value = (set(), error == "active workspace lease", False)
                    debug.return_value["reference_count"] = 2 if error == "unmanaged SHMEM owner" else 1
                    debug.return_value["active_allocations"] = (
                        [{"allocation_base": 1234}] if error == "unmanaged SHMEM allocation" else [])
                    capture.return_value = error == "graph capture"
                    manager.heap_limit = 2 * 1024**2 if error == "explicit heap budget" else None
                    synchronize.side_effect = RuntimeError(error) if error == "sync failed" else None
                    with self.assertRaisesRegex(RuntimeError, error):
                        manager.replace(owner, (4 * 1024**2, (), None))
                    self.assertEqual((manager.state, manager.epoch, manager.heap_bytes), ("ready", 0, 2 * 1024**2))
            owner.release_heap.assert_not_called()
            rebuild.assert_not_called()
            synchronize.side_effect = None
            with self.assertRaisesRegex(RuntimeError, "native rebuild failed"):
                manager.replace(owner, (4 * 1024**2, (), None))
            self.assertEqual((manager.state, manager.epoch), ("failed", 0))
            self.assertTrue(_lifecycle._shutdown_failed)
            with self.assertRaisesRegex(RuntimeError, "heap is failed"), manager.access():
                self.fail("A failed manager allowed execution")
