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
"""CPU contracts for post-measurement P2P attribution and lifetime isolation."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from examples.torch.multicore import mega_moe_p2p_profile as profiling


class _Event:
    """Model one caller-stream timestamp without an accelerator."""

    def __init__(self, clock: SimpleNamespace) -> None:
        """Keep the fake clock until an event is recorded."""
        self.clock = clock
        self.timestamp = None

    def record(self, stream: object) -> None:
        """Record the deterministic device clock, retaining no tensor storage."""
        del stream
        self.timestamp = self.clock.milliseconds

    def elapsed_time(self, end: "_Event") -> float:
        """Return fake device elapsed time, not CPU wall time."""
        return end.timestamp - self.timestamp


class TestP2PObserver(unittest.TestCase):
    """Count true payload and exposed waits without modifying transfers."""

    def setUp(self) -> None:
        """Supply a deterministic asynchronous transport and timing backend."""
        self.clock = SimpleNamespace(milliseconds=0.0)
        self.stream = object()
        self.npu = SimpleNamespace(
            Event=lambda **_kwargs: _Event(self.clock), current_stream=lambda _device: self.stream,
        )
        manager = SimpleNamespace(
            _prefetch_transfers=lambda transfers: transfers,
            _gradient_transfers=lambda transfers: (transfers, ()),
        )

        def launch(transfers: tuple, group: object) -> tuple:
            """Queue the fake transport and return opaque work handles."""
            del group
            self.clock.milliseconds += 2
            return (object(),) if transfers else ()

        def wait(handles: tuple) -> None:
            """Advance only the fake device clock at a nonempty wait site."""
            if handles:
                self.clock.milliseconds += 5

        self.replica = SimpleNamespace(ExpertReplicaManager=manager, _launch_transfers=launch, _wait_handles=wait)

    @staticmethod
    def _transfers(dtype: torch.dtype) -> tuple:
        """Describe one 2:1 gate/up-to-down pair in each direction."""
        return tuple(
            SimpleNamespace(operation=direction, peer_rank=peer, tensor=torch.ones(shape, dtype=dtype))
            for direction, peer in (("send", 1), ("receive", 2)) for shape in ((2, 6), (3, 2))
        )

    def test_payload_and_device_waits_are_separate_from_cpu_submission(self) -> None:
        """
        Feature: phase-specific P2P attribution
        Description: build BF16 weights and FP32 gradients, overlap work before waiting
        Expectation: bytes count each tensor once and exposed waits exclude overlap
        """
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with observer.observe(self.replica):
            for phase, dtype in (("weight_refresh", torch.bfloat16), ("gradient_return", torch.float32)):
                tensors = self._transfers(dtype)
                if phase == "weight_refresh":
                    transfers = self.replica.ExpertReplicaManager._prefetch_transfers(tensors)
                else:
                    transfers, _ = self.replica.ExpertReplicaManager._gradient_transfers(tensors)
                self.assertIs(transfers, tensors)
                handles = self.replica._launch_transfers(transfers, None)
                self.clock.milliseconds += 7
                self.replica._wait_handles(handles)
                self.replica._wait_handles(())
        result = observer.collect()
        for index, expected_bytes in enumerate((36, 72)):
            batch = result["batches"][index]
            self.assertEqual(batch["send_bytes"], expected_bytes)
            self.assertEqual(batch["receive_bytes"], expected_bytes)
            self.assertEqual(batch["launch_to_ready_ms"], 14)
            self.assertEqual(batch["pre_wait_ms"], 9)
            self.assertEqual(batch["exposed_wait_ms"], 5)
            self.assertEqual(len(batch["transfers"]), 4)
            self.assertEqual(batch["transfers"][0]["peer_rank"], 1)
        self.assertEqual(result["summary"]["gradient_return"]["batches"], 1)
        self.assertEqual(result["summary"]["weight_refresh"]["exposed_wait_ms"], 5)

    def test_noop_calls_create_no_events_or_payload(self) -> None:
        """
        Feature: home-only diagnostic isolation
        Description: execute empty transfer builders and cache-hit waits
        Expectation: empty operations retain original behavior without timing events
        """
        self.npu.Event = MagicMock(side_effect=AssertionError("unexpected device event"))
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with observer.observe(self.replica):
            transfers = self.replica.ExpertReplicaManager._prefetch_transfers(())
            self.replica._wait_handles(self.replica._launch_transfers(transfers, None))
        self.assertEqual(observer.collect()["batches"], [])
        self.npu.Event.assert_not_called()

    def test_observer_does_not_retain_transfers_or_tensor_storage(self) -> None:
        """
        Feature: diagnostic storage ownership
        Description: release communication descriptors after their normal wait
        Expectation: the observer retains only scalar metadata and timing events
        """
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with observer.observe(self.replica):
            transfers = self.replica.ExpertReplicaManager._prefetch_transfers(self._transfers(torch.bfloat16))
            references = [weakref.ref(transfer.tensor) for transfer in transfers]
            self.replica._wait_handles(self.replica._launch_transfers(transfers, None))
            del transfers
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(len(observer.collect()["batches"]), 1)

    def test_exception_restores_all_wrappers(self) -> None:
        """
        Feature: diagnostic exception isolation
        Description: fail after installing temporary communication wrappers
        Expectation: the original builder, launch and wait functions are restored
        """
        originals = (self.replica._launch_transfers, self.replica._wait_handles,
                     self.replica.ExpertReplicaManager._prefetch_transfers,
                     self.replica.ExpertReplicaManager._gradient_transfers)
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with observer.observe(self.replica):
                raise RuntimeError("injected")
        restored = (self.replica._launch_transfers, self.replica._wait_handles,
                    self.replica.ExpertReplicaManager._prefetch_transfers,
                    self.replica.ExpertReplicaManager._gradient_transfers)
        self.assertEqual(originals, restored)

    def test_unfinished_or_unknown_batches_are_rejected(self) -> None:
        """
        Feature: complete P2P accounting
        Description: launch unknown descriptors or collect before a recorded wait
        Expectation: missing attribution is reported rather than silently undercounted
        """
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with observer.observe(self.replica):
            with self.assertRaisesRegex(RuntimeError, "unclassified"):
                self.replica._launch_transfers(self._transfers(torch.bfloat16), None)
            transfers = self.replica.ExpertReplicaManager._prefetch_transfers(self._transfers(torch.bfloat16))
            handles = self.replica._launch_transfers(transfers, None)
            with self.assertRaisesRegex(RuntimeError, "unfinished"):
                observer.collect()
            self.replica._wait_handles(handles)
        self.assertEqual(len(observer.collect()["batches"]), 1)

    def test_post_update_capture_participates_on_every_rank(self) -> None:
        """
        Feature: representative P2P diagnostic lifecycle
        Description: run a nonzero rank through warm-update, capture, update and aggregation
        Expectation: both steps update weights and event reads occur after device synchronization
        """
        sweep = MagicMock()
        sweep.worker._RANK = 3
        sweep.shape.ep_size = 8
        observer = MagicMock()
        events = []
        sweep._train_step.side_effect = lambda _route: events.append("train")
        sweep.optimizer.step.side_effect = lambda: events.append("update")
        sweep.worker.dist.barrier.side_effect = lambda: events.append("barrier")
        sweep.worker.torch.npu.synchronize.side_effect = lambda _device: events.append("sync")
        observer.collect.side_effect = lambda: events.append("collect") or {"batches": [], "summary": {}}

        def gather(ranks: list, local: dict) -> None:
            """Fill distinct EP-rank rows without a distributed process group."""
            events.append("gather")
            ranks[:] = [{**local, "rank": rank} for rank in range(len(ranks))]

        sweep.worker.dist.all_gather_object.side_effect = gather
        with patch.object(profiling, "_P2PObserver", return_value=observer), \
                patch.object(profiling.importlib, "import_module", return_value=self.replica):
            result = profiling.profile_layer_p2p(sweep, (1, 2, 3))
        self.assertEqual(sweep._train_step.call_count, 2)
        self.assertEqual(sweep.optimizer.step.call_count, 2)
        self.assertEqual(len(result["per_rank"]), 8)
        sweep.worker.dist.all_gather_object.assert_called_once()
        self.assertEqual(sweep.worker.dist.all_gather_object.call_args.args[1]["rank"], 3)
        self.assertEqual(events, ["train", "update", "barrier", "sync", "train", "sync",
                                  "collect", "update", "sync", "gather"])

    def test_payload_balance_rejects_wrong_peer_dtype_or_round(self) -> None:
        """
        Feature: all-rank payload conservation
        Description: compare matching send/receive metadata and corrupt one field
        Expectation: only matching direction, peer, dtype and phase-round records pass
        """
        send = {"operation": "send", "peer_rank": 1, "bytes": 24, "dtype": "torch.float32", "shape": [2, 3]}
        receive = {**send, "operation": "receive", "peer_rank": 0}
        batch = {"phase": "gradient_return", "phase_call": 0, "transfers": [send]}
        other = {**batch, "transfers": [receive]}
        ranks = [{"rank": 0, "batches": [batch]}, {"rank": 1, "batches": [other]}]
        result = profiling._validate_payload_balance(ranks)
        self.assertEqual(result["sent_bytes"], {"weight_refresh": 0, "gradient_return": 24})
        for field, value in (("peer_rank", 1), ("dtype", "torch.bfloat16"), ("bytes", 12), ("shape", [6])):
            with self.subTest(field=field), patch.dict(receive, {field: value}):
                with self.assertRaisesRegex(RuntimeError, "payload mismatch"):
                    profiling._validate_payload_balance(ranks)
        with patch.dict(other, {"phase_call": 1}), self.assertRaisesRegex(RuntimeError, "payload mismatch"):
            profiling._validate_payload_balance(ranks)

    def test_wait_failure_remains_unfinished_and_restores_original(self) -> None:
        """
        Feature: complete diagnostics on communication failure
        Description: fail the original wait after launching a recorded batch
        Expectation: incomplete events cannot be reported as a successful capture
        """
        self.replica._wait_handles = MagicMock(side_effect=RuntimeError("wait failed"))
        original = self.replica._wait_handles
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            with observer.observe(self.replica):
                transfers = self.replica.ExpertReplicaManager._prefetch_transfers(self._transfers(torch.float32))
                self.replica._wait_handles(self.replica._launch_transfers(transfers, None))
        self.assertIs(self.replica._wait_handles, original)
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            observer.collect()

    def test_partial_setup_restores_earlier_builder(self) -> None:
        """
        Feature: profiling setup failure isolation
        Description: discover a missing second builder after patching the first
        Expectation: the successfully installed wrapper is removed before raising
        """
        manager = self.replica.ExpertReplicaManager
        original = manager._prefetch_transfers
        del manager._gradient_transfers
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with self.assertRaises(AttributeError):
            with observer.observe(self.replica):
                self.fail("profiling accepted a missing transfer builder")
        self.assertIs(manager._prefetch_transfers, original)

    def test_launch_failure_or_missing_handles_cannot_publish_complete_capture(self) -> None:
        """
        Feature: complete diagnostics on submission failure
        Description: fail a nonempty batch or return no completion handles
        Expectation: wrappers restore and collection rejects the unfinished batch
        """
        for error in (RuntimeError("launch failed"), None):
            with self.subTest(error=error):
                original = MagicMock(return_value=(), side_effect=error)
                self.replica._launch_transfers = original
                observer = profiling._P2PObserver(self.npu, "npu:0")
                with self.assertRaisesRegex(RuntimeError, "launch failed|no completion handles"):
                    with observer.observe(self.replica):
                        transfers = self.replica.ExpertReplicaManager._prefetch_transfers(
                            self._transfers(torch.float32),
                        )
                        self.replica._launch_transfers(transfers, None)
                self.assertIs(self.replica._launch_transfers, original)
                with self.assertRaisesRegex(RuntimeError, "unfinished"):
                    observer.collect()

    def test_invalid_device_durations_are_rejected(self) -> None:
        """
        Feature: valid device timing evidence
        Description: return nonfinite or negative elapsed time after a completed batch
        Expectation: invalid event results cannot be serialized as successful diagnostics
        """
        for duration in (float("nan"), float("inf"), -1.0):
            with self.subTest(duration=duration):
                observer = profiling._P2PObserver(self.npu, "npu:0")
                with observer.observe(self.replica):
                    transfers = self.replica.ExpertReplicaManager._prefetch_transfers(self._transfers(torch.float32))
                    self.replica._wait_handles(self.replica._launch_transfers(transfers, None))
                with patch.object(_Event, "elapsed_time", return_value=duration):
                    with self.assertRaisesRegex(RuntimeError, "invalid P2P device event durations"):
                        observer.collect()

    def test_cross_stream_wait_is_not_reported_as_same_stream_overlap(self) -> None:
        """
        Feature: caller-stream timing contract
        Description: move a pending batch wait to a different stream
        Expectation: reject the unsupported timing interpretation without publishing a summary
        """
        observer = profiling._P2PObserver(self.npu, "npu:0")
        with observer.observe(self.replica):
            transfers = self.replica.ExpertReplicaManager._prefetch_transfers(self._transfers(torch.float32))
            handles = self.replica._launch_transfers(transfers, None)
            self.stream = object()
            with self.assertRaisesRegex(RuntimeError, "caller stream"):
                self.replica._wait_handles(handles)
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            observer.collect()
