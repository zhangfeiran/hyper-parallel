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
"""Unit tests for the private SHMEM reference-counted lifecycle."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from hyper_parallel.core.multicore.shmem import _lifecycle
from tests.common.mark_utils import arg_mark


# These tests deliberately inspect the private state machine and mocked native entry points.
class TestRuntimeLifecycle(unittest.TestCase):
    """Validate lifecycle coordination without Torch distributed or NPU hardware."""

    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Install deterministic framework and Native boundaries."""
        self.world = object()
        self.dist = SimpleNamespace(
            group=SimpleNamespace(WORLD=self.world),
            is_initialized=Mock(return_value=True),
            get_world_size=Mock(return_value=2),
            get_process_group_ranks=Mock(return_value=[0, 1]),
            get_rank=Mock(return_value=0),
            barrier=Mock(),
            broadcast_object_list=Mock(),
            all_gather_object=Mock(
                side_effect=lambda output, value, **_kw: output.__setitem__(slice(None), [value] * 2)),
        )
        self.torch = SimpleNamespace(npu=SimpleNamespace(
            synchronize=Mock(),
            is_current_stream_capturing=Mock(return_value=False),
            current_device=Mock(return_value=0),
        ), device=lambda device_type, index: f"{device_type}:{index}")
        self.native = SimpleNamespace(
            _initialize=Mock(),
            _get_unique_id=Mock(return_value=b"group-bootstrap"),
            _validate_shutdown=Mock(),
            _shutdown=Mock(),
            _debug_state=Mock(return_value={
                "config": {"heap_size_bytes": 128, "data_engine": "mte"},
                "device_index": 0,
                "active_allocations": [],
            }),
        )
        self.framework_patch = patch.object(_lifecycle, "_torch_modules", return_value=(self.torch, self.dist))
        self.native_patch = patch.object(_lifecycle, "_load_native", return_value=self.native)
        self.engine_patch = patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte"})
        self.framework_patch.start()
        self.native_patch.start()
        self.engine_patch.start()
        self.addCleanup(self.framework_patch.stop)
        self.addCleanup(self.native_patch.stop)
        self.addCleanup(self.engine_patch.stop)
        self.addCleanup(self._reset_runtime)
        self._reset_runtime()

    @staticmethod
    def _reset_runtime() -> None:
        """Reset private process state after each isolated test."""
        _lifecycle._users = 0  # pylint: disable=protected-access
        _lifecycle._root_group = None  # pylint: disable=protected-access
        _lifecycle._root_uses_distributed = None  # pylint: disable=protected-access
        _lifecycle._root_ranks = None  # pylint: disable=protected-access
        _lifecycle._root_size = None  # pylint: disable=protected-access
        _lifecycle._device_index = None  # pylint: disable=protected-access
        _lifecycle._shutdown_failed = False  # pylint: disable=protected-access

    def test_single_process_runtime_is_reference_counted(self) -> None:
        """Keep one Root PE active until its final reference is released."""
        self.dist.is_initialized.return_value = False
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access

        _lifecycle.acquire()
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
        _lifecycle.acquire()
        self.assertEqual(_lifecycle._reference_count(), 2)  # pylint: disable=protected-access
        _lifecycle.release()
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access

        self.native._validate_shutdown.assert_not_called()
        self.torch.npu.synchronize.assert_not_called()
        self.native._shutdown.assert_not_called()

        _lifecycle.release()
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access

        self.native._initialize.assert_called_once_with(0, 1)
        self.native._validate_shutdown.assert_called_once_with()
        self.torch.npu.synchronize.assert_called_once_with()
        self.dist.barrier.assert_not_called()
        self.native._shutdown.assert_called_once_with()

    def test_equivalent_process_group_shares_runtime(self) -> None:
        """Count a distinct ProcessGroup object describing the same complete WORLD."""
        equivalent_world = object()

        _lifecycle.acquire(self.world)
        _lifecycle.acquire(equivalent_world)

        self.native._initialize.assert_called_once_with(0, 2)
        self.dist.barrier.assert_called_once_with(group=self.world)
        self.assertIs(_lifecycle._root_group, self.world)  # pylint: disable=protected-access
        _lifecycle.release()
        self.native._shutdown.assert_not_called()
        _lifecycle.release()

    def test_active_runtime_rejects_changed_data_engine(self) -> None:
        """Reject a new user before it can reinterpret the active device state."""
        _lifecycle.acquire(self.world)

        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma"}):
            with self.assertRaisesRegex(RuntimeError, "data engine differs"):
                _lifecycle.acquire(self.world)

        self.assertEqual(_lifecycle._reference_count(), 1)
        _lifecycle.release()

    def test_active_udma_runtime_accepts_the_same_data_engine(self) -> None:
        """Keep the shared QP engine fixed while adding an equivalent owner."""
        self.native._debug_state.return_value["config"]["data_engine"] = "mte_udma"

        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma"}):
            _lifecycle.acquire(self.world)
            _lifecycle.acquire(self.world)
            self.assertEqual(_lifecycle._reference_count(), 2)
            _lifecycle.release()
            _lifecycle.release()

    def test_different_root_is_rejected(self) -> None:
        """Reject a later ProcessGroup with a different ordered membership."""
        different_group = object()

        def ranks(group: object) -> list[int]:
            """Return complete WORLD only for the lifecycle's first ProcessGroup."""
            return [0, 1] if group is self.world else [1, 0]

        self.dist.get_process_group_ranks.side_effect = ranks
        _lifecycle.acquire(self.world)

        with self.assertRaisesRegex(RuntimeError, "different ordered membership"):
            _lifecycle.acquire(different_group)

        self.native._initialize.assert_called_once_with(0, 2)
        self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
        _lifecycle.release()

    def test_distributed_state_change_is_rejected(self) -> None:
        """Keep the framework mode fixed throughout one active lifecycle."""
        self.dist.is_initialized.return_value = False
        _lifecycle.acquire()
        self.dist.is_initialized.return_value = True

        with self.assertRaisesRegex(RuntimeError, "state changed"):
            _lifecycle.acquire()
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            _lifecycle.release()

        self.dist.is_initialized.return_value = False
        _lifecycle.release()

    def test_collective_runtime_operation_order(self) -> None:
        """Order Root convergence before initialization and quiescence before shutdown."""
        operations = Mock()
        operations.attach_mock(self.dist.barrier, "barrier")
        operations.attach_mock(self.native._initialize, "initialize")
        operations.attach_mock(self.native._validate_shutdown, "validate_shutdown")
        operations.attach_mock(self.torch.npu.synchronize, "synchronize")
        operations.attach_mock(self.native._shutdown, "shutdown")

        _lifecycle.acquire()
        _lifecycle.release()

        self.assertEqual(
            operations.mock_calls,
            [
                call.barrier(group=self.world),
                call.initialize(0, 2),
                call.validate_shutdown(),
                call.synchronize(),
                call.barrier(group=self.world),
                call.shutdown(),
            ],
        )

    def test_clean_release_allows_new_runtime(self) -> None:
        """Run Native initialization and shutdown again after a clean lifecycle."""
        for _ in range(2):
            _lifecycle.acquire()
            _lifecycle.release()

        self.assertEqual(self.native._initialize.call_count, 2)
        self.assertEqual(self.native._shutdown.call_count, 2)
        self.assertEqual(self.dist.barrier.call_count, 4)
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

    def test_entry_barrier_failure_is_retryable(self) -> None:
        """Retry when Root convergence fails before entering Native initialization."""
        self.dist.barrier.side_effect = RuntimeError("root convergence failed")

        with self.assertRaisesRegex(RuntimeError, "root convergence failed"):
            _lifecycle.acquire()

        self.native._initialize.assert_not_called()
        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        self.dist.barrier.side_effect = None
        _lifecycle.acquire()
        _lifecycle.release()

    def test_final_release_precondition_failure_is_retryable(self) -> None:
        """Retain lifecycle state when shutdown validation, sync, or barrier fails."""
        for operation in (self.native._validate_shutdown, self.torch.npu.synchronize, self.dist.barrier):
            with self.subTest(operation=operation):
                self._reset_runtime()
                self.native._validate_shutdown.reset_mock(side_effect=True)
                self.torch.npu.synchronize.reset_mock(side_effect=True)
                self.dist.barrier.reset_mock(side_effect=True)
                self.native._shutdown.reset_mock(side_effect=True)
                _lifecycle.acquire()
                self.dist.barrier.reset_mock()
                operation.side_effect = RuntimeError("quiescence failed")

                with self.assertRaisesRegex(RuntimeError, "quiescence failed"):
                    _lifecycle.release()

                self.assertEqual(_lifecycle._reference_count(), 1)  # pylint: disable=protected-access
                self.assertIsNotNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
                self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
                self.native._shutdown.assert_not_called()
                operation.side_effect = None
                _lifecycle.release()

    def test_native_initialize_failure_is_retryable(self) -> None:
        """Retry Native initialization after a failed attempt; only shutdown failure blocks."""
        self.native._initialize.side_effect = RuntimeError("initialize failed")

        with self.assertRaisesRegex(RuntimeError, "initialize failed"):
            _lifecycle.acquire()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access

        self.native._initialize.side_effect = None
        _lifecycle.acquire()
        self.assertEqual(self.native._initialize.call_count, 2)
        _lifecycle.release()

    def test_peer_shutdown_errors_are_coordinated(self) -> None:
        """A peer precheck failure blocks the barrier; a peer native failure is terminal."""
        _lifecycle.acquire()
        self.dist.barrier.reset_mock()
        with patch.object(_lifecycle._automatic, "exchange", return_value=[None, "peer busy"]):
            with self.assertRaisesRegex(RuntimeError, "peer busy"):
                _lifecycle.release()
        self.dist.barrier.assert_not_called()
        self.native._shutdown.assert_not_called()
        self.assertEqual(_lifecycle._reference_count(), 1)
        with patch.object(_lifecycle._automatic, "exchange", side_effect=[[None, None], [None, "peer finalize"]]):
            with self.assertRaisesRegex(RuntimeError, "peer finalize"):
                _lifecycle.release()
        self.native._shutdown.assert_called_once_with()
        self.assertTrue(_lifecycle._shutdown_failed)
        with self.assertRaisesRegex(RuntimeError, "Native shutdown failure"):
            _lifecycle.acquire()

    def test_native_shutdown_failure_blocks_future_acquire(self) -> None:
        """Clear Root ownership but reject a new lifecycle after Native shutdown fails."""
        _lifecycle.acquire()
        self.native._shutdown.side_effect = RuntimeError("finalize failed")

        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            _lifecycle.release()

        self.assertEqual(_lifecycle._reference_count(), 0)  # pylint: disable=protected-access
        self.assertIsNone(_lifecycle._root_uses_distributed)  # pylint: disable=protected-access
        self.assertTrue(_lifecycle._shutdown_failed)  # pylint: disable=protected-access
        with self.assertRaisesRegex(RuntimeError, "Native shutdown failure"):
            _lifecycle.acquire()

    def test_release_without_reference_is_rejected(self) -> None:
        """Expose an unmatched release instead of silently underflowing the user count."""
        with self.assertRaisesRegex(RuntimeError, "no active reference"):
            _lifecycle.release()

    def test_runtime_restart_preserves_collective_operation_order(self) -> None:
        """Rebuild explicit Allocations around one Native restart."""
        _lifecycle.acquire()
        operations = []
        clock = [10.0]
        self.torch.npu.synchronize.side_effect = lambda _device: operations.append("quiesce")

        def validate_shutdown() -> None:
            """Record validation and advance the deterministic stage clock."""
            operations.append("validate")
            clock[0] += 0.002

        def shutdown() -> None:
            """Record shutdown and advance the deterministic stage clock."""
            operations.append("finalize")
            clock[0] += 0.003

        self.native._validate_shutdown.side_effect = validate_shutdown
        self.native._shutdown.side_effect = shutdown
        current = Mock(shape=(4,), dtype=torch.float32, _base=None)
        current.data_ptr.return_value = 100
        current.numel.return_value = 4
        current.element_size.return_value = 4
        current.is_contiguous.return_value = True
        replacement = torch.empty(8, dtype=torch.float32)
        active_allocations = [{"allocation_base": 100, "allocation_bytes": 16}]

        def debug_state() -> dict:
            """Expose the mutable mocked Allocation registry."""
            return {
                "config": {"heap_size_bytes": 256, "data_engine": "mte"},
                "device_index": 0,
                "active_allocations": list(active_allocations),
            }

        self.native._debug_state.side_effect = debug_state

        def bootstrap() -> bytes:
            """Record bootstrap ordering and return one group identifier."""
            operations.append("bootstrap")
            return b"replacement-bootstrap"

        def initialize(*_args: object) -> None:
            """Record Native initialize ordering."""
            operations.append("initialize")

        self.native._get_unique_id.side_effect = bootstrap
        self.native._initialize.side_effect = initialize
        allocation = _lifecycle._AllocationReplacement(  # pylint: disable=protected-access
            current,
            _lifecycle._AllocationSpec((8,), torch.float32, 16),  # pylint: disable=protected-access
        )

        def free(tensor: object) -> None:
            """Record release ordering and clear the mocked registry."""
            del tensor
            operations.append("free")
            active_allocations.clear()

        with (
            patch("hyper_parallel.core.multicore.shmem._api.free", side_effect=free),
            patch("hyper_parallel.core.multicore.shmem._api.empty", side_effect=lambda *_a, **_k: replacement),
            patch.object(_lifecycle.time, "perf_counter", side_effect=lambda: clock[0]),
        ):
            result = _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(allocation,),
                preflight_hook=lambda: ("layout", 1),
            )

        self.assertEqual(
            operations,
            ["quiesce", "free", "validate", "finalize", "bootstrap", "initialize"],
        )
        self.assertEqual(
            set(result.stages_ms),
            {
                "preflight_ms", "quiesce_ms", "release_ms", "finalize_ms",
                "bootstrap_ms", "initialize_ms", "restore_ms", "commit_ms",
            },
        )
        self.assertEqual(result.tensors, (replacement,))
        self.assertAlmostEqual(result.stages_ms["finalize_ms"], 5.0)
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

    def test_runtime_restart_preflight_failure_preserves_runtime(self) -> None:
        """Converge ownership, device, Allocation and hook failures before teardown."""
        _lifecycle.acquire()
        _lifecycle.acquire()
        with self.assertRaisesRegex(RuntimeError, "one healthy Runtime user"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        _lifecycle.release()
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

        self.torch.npu.current_device.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "current NPU device"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.torch.npu.current_device.return_value = 0

        self.native._debug_state.return_value = {
            "config": {"heap_size_bytes": 128, "data_engine": "mte"},
            "device_index": 0,
            "active_allocations": [{"allocation_base": 1234, "allocation_bytes": 16}],
        }
        with self.assertRaisesRegex(RuntimeError, "cover every active"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        current = Mock(shape=(4,), dtype=torch.float32, _base=None)
        current.data_ptr.return_value = 1234
        current.numel.return_value = 4
        current.element_size.return_value = 4
        current.is_contiguous.return_value = True
        for alignment in (None, 0, 3):
            with self.subTest(alignment=alignment), self.assertRaisesRegex(RuntimeError, "alignment_bytes"):
                invalid_target = _lifecycle._AllocationSpec(  # type: ignore[arg-type]
                    (8,), torch.float32, alignment,
                )
                replacement = _lifecycle._AllocationReplacement(current, invalid_target)
                _lifecycle._restart(  # pylint: disable=protected-access
                    heap_size_bytes=256,
                    replacements=(replacement,),
                    preflight_hook=lambda: "state",
                )
        self.torch.npu.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "graph capture"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.torch.npu.is_current_stream_capturing.return_value = False
        self.native._debug_state.return_value = {
            "config": {"heap_size_bytes": 128, "data_engine": "mte"}, "device_index": 0, "active_allocations": [],
        }

        def peer_preflight_error(output: list, value: object, **_kwargs: object) -> None:
            """Inject one peer-side preflight failure."""
            output[:] = [value, (None, "RuntimeError: peer preflight failed")]

        self.dist.all_gather_object.side_effect = peer_preflight_error
        with self.assertRaisesRegex(RuntimeError, "peer preflight failed"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.torch.npu.synchronize.assert_not_called()
        self.native._shutdown.assert_not_called()
        self.assertFalse(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

    def test_runtime_restart_destructive_failure_is_terminal(self) -> None:
        """Invalidate the Runtime after Allocation release begins and fails."""
        _lifecycle.acquire()
        self.native._shutdown.side_effect = RuntimeError("finalize failed")
        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            _lifecycle._restart(  # pylint: disable=protected-access
                heap_size_bytes=256,
                replacements=(),
                preflight_hook=lambda: "state",
            )
        self.assertTrue(_lifecycle._shutdown_failed)  # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_default_group_does_not_select_the_active_subgroup(self) -> None:
        """Feature: Default-root collective selection.
        Description: Check WORLD layout while another subgroup owns the native Runtime.
        Expectation: Explicit None selects WORLD; an omitted group selects the active root.
        """
        subgroup = object()
        with patch.object(_lifecycle, "_root_group", subgroup):
            _lifecycle._reconfiguration_stage(Mock(), group=None, value="world layout")
            self.assertIsNone(self.dist.all_gather_object.call_args.kwargs["group"])
            _lifecycle._reconfiguration_stage(Mock())
            self.assertIs(self.dist.all_gather_object.call_args.kwargs["group"], subgroup)


if __name__ == "__main__":
    unittest.main()
