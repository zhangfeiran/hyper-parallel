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
"""Opt-in all-rank P2P payload and caller-stream readiness diagnostics."""

from __future__ import annotations

import importlib
import math
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any

_PHASES = ("weight_refresh", "gradient_return")
_METRICS = ("send_bytes", "receive_bytes", "prepare_cpu_ms", "launch_cpu_ms", "wait_cpu_ms",
            "launch_to_ready_ms", "pre_wait_ms", "exposed_wait_ms")


class _P2PObserver:
    """Observe serial batches without retaining tensors or changing communication."""

    def __init__(self, npu: Any, device: Any) -> None:
        """Keep only host metadata and three device events for each nonempty batch."""
        self.npu = npu
        self.device = device
        self.prepared: dict[int, dict[str, Any]] = {}
        self.pending: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []
        self.calls = dict.fromkeys(_PHASES, 0)

    def _builder(self, function: Any, phase: str) -> Any:
        @wraps(function)
        def build(*args: Any, **kwargs: Any) -> Any:
            """Snapshot payload metadata without retaining the returned tensors."""
            started = time.perf_counter()
            result = function(*args, **kwargs)
            elapsed = (time.perf_counter() - started) * 1000
            transfers = result[0] if phase == "gradient_return" else result
            call_index = self.calls[phase]
            self.calls[phase] += 1
            if transfers:
                metadata = [
                    {"operation": item.operation, "peer_rank": item.peer_rank,
                     "bytes": item.tensor.numel() * item.tensor.element_size(),
                     "dtype": str(item.tensor.dtype), "shape": list(item.tensor.shape)}
                    for item in transfers
                ]
                self.prepared[id(transfers)] = {
                    "phase": phase, "phase_call": call_index, "prepare_cpu_ms": elapsed,
                    "transfers": metadata,
                    "send_bytes": sum(item["bytes"] for item in metadata if item["operation"] == "send"),
                    "receive_bytes": sum(item["bytes"] for item in metadata if item["operation"] == "receive"),
                }
            return result

        return build

    def _launch(self, function: Any) -> Any:
        @wraps(function)
        def launch(transfers: tuple, group: Any) -> tuple:
            """Start the readiness span before the unchanged asynchronous batch."""
            if not transfers:
                return function(transfers, group)
            record = self.prepared.get(id(transfers))
            if record is None:
                raise RuntimeError("unclassified P2P batch in diagnostic step")
            record["stream"] = self.npu.current_stream(self.device)
            record["events"] = [self.npu.Event(enable_timing=True) for _ in range(3)]
            record["events"][0].record(record["stream"])
            started = time.perf_counter()
            handles = function(transfers, group)
            record["launch_cpu_ms"] = (time.perf_counter() - started) * 1000
            if not handles:
                raise RuntimeError("nonempty P2P batch returned no completion handles")
            del self.prepared[id(transfers)]
            self.pending[id(handles)] = record
            return handles

        return launch

    def _wait(self, function: Any) -> Any:
        @wraps(function)
        def wait(handles: tuple) -> Any:
            """Bracket the existing wait site without adding a device synchronization."""
            if not handles:
                return function(handles)
            record = self.pending.get(id(handles))
            if record is None:
                raise RuntimeError("unclassified P2P wait in diagnostic step")
            if self.npu.current_stream(self.device) != record["stream"]:
                raise RuntimeError("profiled P2P wait changed the caller stream")
            record["events"][1].record(record["stream"])
            started = time.perf_counter()
            result = function(handles)
            record["wait_cpu_ms"] = (time.perf_counter() - started) * 1000
            record["events"][2].record(record["stream"])
            del self.pending[id(handles)]
            self.completed.append(record)
            return result

        return wait

    @contextmanager
    def observe(self, replica: Any) -> Iterator[None]:
        """Patch only an extra example step, restoring even partially installed wrappers."""
        manager = replica.ExpertReplicaManager
        targets = (
            (manager, "_prefetch_transfers", lambda function: self._builder(function, "weight_refresh")),
            (manager, "_gradient_transfers", lambda function: self._builder(function, "gradient_return")),
            (replica, "_launch_transfers", self._launch),
            (replica, "_wait_handles", self._wait),
        )
        with ExitStack() as stack:
            for target, name, decorate in targets:
                original = getattr(target, name)
                stack.callback(setattr, target, name, original)
                setattr(target, name, decorate(original))
            yield

    def collect(self) -> dict[str, Any]:
        """Read events only after caller synchronization; report sums, not critical-path savings."""
        if self.prepared or self.pending:
            raise RuntimeError("unfinished P2P batches in diagnostic step")
        batches = []
        for record in self.completed:
            start, wait, ready = record["events"]
            durations = {"launch_to_ready_ms": start.elapsed_time(ready),
                         "pre_wait_ms": start.elapsed_time(wait), "exposed_wait_ms": wait.elapsed_time(ready)}
            if any(not math.isfinite(value) or value < 0 for value in durations.values()):
                raise RuntimeError(f"invalid P2P device event durations: {durations}")
            batches.append({key: value for key, value in record.items() if key not in {"stream", "events"}} | durations)
        summary = {}
        for phase in _PHASES:
            selected = [batch for batch in batches if batch["phase"] == phase]
            summary[phase] = {"batches": len(selected), "builder_calls": self.calls[phase],
                              **{name: sum(batch[name] for batch in selected) for name in _METRICS},
                              "max_exposed_wait_ms": max((batch["exposed_wait_ms"] for batch in selected), default=0)}
        return {"batches": batches, "summary": summary}


def _validate_payload_balance(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    """Require matching EP-local peers, phase rounds, shapes, dtypes and byte counts."""
    if sorted(rank["rank"] for rank in ranks) != list(range(len(ranks))):
        raise RuntimeError("P2P diagnostics must contain each EP rank exactly once")
    sent, received = Counter(), Counter()
    for rank in ranks:
        for batch in rank["batches"]:
            for transfer in batch["transfers"]:
                peer = transfer["peer_rank"]
                if transfer["operation"] not in {"send", "receive"} or not 0 <= peer < len(ranks):
                    raise RuntimeError(f"invalid P2P transfer metadata: {transfer}")
                source, target = (rank["rank"], peer) if transfer["operation"] == "send" else (peer, rank["rank"])
                key = (batch["phase"], batch["phase_call"], source, target,
                       transfer["dtype"], tuple(transfer["shape"]))
                counts = sent if transfer["operation"] == "send" else received
                counts[key] += transfer["bytes"]
    if sent != received:
        raise RuntimeError(f"P2P payload mismatch: unmatched sends={sent - received}, receives={received - sent}")
    return {"matching_peer_payloads": True,
            "sent_bytes": {phase: sum(size for key, size in sent.items() if key[0] == phase) for phase in _PHASES}}


def profile_layer_p2p(sweep: Any, route: tuple) -> dict[str, Any]:
    """Measure an extra updated-weight step on every EP rank, outside latency/HBM windows.

    Args:
        sweep: Initialized layer benchmark. All EP ranks must participate.
        route: Previously sampled route for the diagnostic step.

    Returns:
        Per-batch scalar payload metadata, CPU submission times and NPU event
        spans on the caller stream. No profiler is started and no device-wide
        synchronization is added inside the training step.

    Note:
        Launch-to-ready includes route preparation and host enqueue gaps, not
        just wire time. Exposed wait is what remains at the wait site, including
        instrumentation/enqueue overhead; it is not a pure network time or a
        predicted optimization gain. Pre-wait time is an opportunity window,
        not evidence that all communication overlapped useful computation.
    """
    # This private coupling is confined to diagnostics in the single-threaded
    # example. Normal library calls never construct an observer or its events.
    # pylint: disable=protected-access
    worker = sweep.worker
    sweep._train_step(route)
    sweep.optimizer.step()
    worker.dist.barrier()
    worker.torch.npu.synchronize(worker._DEVICE)
    observer = _P2PObserver(worker.torch.npu, worker._DEVICE)
    replica = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.replica")
    with observer.observe(replica):
        sweep._train_step(route)
    worker.torch.npu.synchronize(worker._DEVICE)
    local = {"rank": worker._RANK, **observer.collect()}
    sweep.optimizer.step()
    worker.torch.npu.synchronize(worker._DEVICE)
    ranks = [None] * sweep.shape.ep_size
    worker.dist.all_gather_object(ranks, local)
    return {
        "scope": "all nonempty P2P batches in one extra post-update F+B; no optimizer timing",
        "timing": "NPU caller-stream events and separate CPU scopes after all performance/HBM measurements",
        "caveat": "event spans include enqueue/instrumentation gaps; pre-wait is not measured compute overlap",
        "per_rank": ranks,
        "payload_balance": _validate_payload_balance(ranks),
    }
