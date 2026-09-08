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

from examples.torch.multicore.mega_moe_profile_context import (
    ProfileContext,
    new_capture_id,
    placement_metadata,
)

_PHASES = ("weight_refresh", "gradient_return")
_METRICS = ("send_bytes", "receive_bytes", "prepare_cpu_ms", "launch_cpu_ms", "wait_cpu_ms",
            "launch_to_ready_ms", "pre_wait_ms", "exposed_wait_ms")
_DEVICE_METRICS = ("launch_to_ready_ms", "pre_wait_ms", "exposed_wait_ms")


class _P2PObserver:
    """Observe serial batches without retaining tensors or changing communication."""

    def __init__(self, npu: Any, device: Any, *, context: Any = None, device_timing: bool = True) -> None:
        """Keep only host metadata and three device events for each nonempty batch."""
        self.npu = npu
        self.device = device
        self.prepared: dict[int, dict[str, Any]] = {}
        self.pending: dict[int, dict[str, Any]] = {}
        self.completed: list[dict[str, Any]] = []
        self.calls = dict.fromkeys(_PHASES, 0)
        self.context = context
        self.device_timing = device_timing
        self.empty_batches: list[dict[str, Any]] = []

    def _correlation(self, args: tuple, phase: str, metadata: list[dict], transfers: tuple) -> dict:
        """Match logical expert descriptors to the actual two wire tensors."""
        if self.context is None:
            return {}
        manager, description = args[:2]
        identity = self.context.metadata()
        if "layer" not in identity:
            raise ValueError("P2P batch lacks its layer identity")
        if phase == "weight_refresh":
            descriptors = description.replica_transfers
            identity.update(placement_metadata(manager, description))
            identity["batch_index"] = self.context.next_index(
                phase, identity["layer"], identity["direction"], identity["call_id"],
            )
        else:
            descriptors = description
        expected = []
        for entry in descriptors:
            if manager.spec.rank_id not in (entry.owner_rank, entry.target_rank):
                continue
            owner = manager.spec.rank_id == entry.owner_rank
            send = owner if phase == "weight_refresh" else not owner
            for tensor_role in ("gate_up", "down"):
                expected.append({
                    "operation": "send" if send else "receive",
                    "peer_rank": entry.target_rank if owner else entry.owner_rank,
                    "logical_expert": entry.logical_expert, "owner_slot": entry.owner_slot,
                    "replica_slot": entry.target_slot, "tensor_role": tensor_role,
                })
        if len(expected) != len(metadata):
            raise RuntimeError("P2P descriptor count does not match actual wire tensors")
        for item, descriptor, transfer in zip(metadata, expected, transfers):
            if any(item[name] != descriptor[name] for name in ("operation", "peer_rank")):
                raise RuntimeError("P2P descriptor order does not match actual wire tensors")
            item.update(descriptor)
            item["storage_address"] = transfer.tensor.untyped_storage().data_ptr()
            item["storage_offset_elements"] = transfer.tensor.storage_offset()
        return identity

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
            metadata = [
                    {"operation": item.operation, "peer_rank": item.peer_rank,
                     "bytes": item.tensor.numel() * item.tensor.element_size(),
                     "dtype": str(item.tensor.dtype), "shape": list(item.tensor.shape)}
                    for item in transfers
            ]
            correlation = self._correlation(args, phase, metadata, transfers)
            if transfers:
                self.prepared[id(transfers)] = {
                    **correlation,
                    "phase": phase, "phase_call": call_index, "prepare_cpu_ms": elapsed,
                    "transfers": metadata,
                    "send_bytes": sum(item["bytes"] for item in metadata if item["operation"] == "send"),
                    "receive_bytes": sum(item["bytes"] for item in metadata if item["operation"] == "receive"),
                }
            else:
                self.empty_batches.append({
                    **correlation, "phase": phase, "phase_call": call_index,
                    "prepare_cpu_ms": elapsed, "transfers": [], "send_bytes": 0, "receive_bytes": 0,
                })
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
            record["events"] = [self.npu.Event(enable_timing=True) for _ in range(3)] if self.device_timing else []
            if record["events"]:
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
            if record["events"]:
                record["events"][1].record(record["stream"])
            started = time.perf_counter()
            result = function(handles)
            record["wait_cpu_ms"] = (time.perf_counter() - started) * 1000
            if record["events"]:
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
            durations = dict.fromkeys(_DEVICE_METRICS)
            if record["events"]:
                start, wait, ready = record["events"]
                durations = {"launch_to_ready_ms": start.elapsed_time(ready),
                             "pre_wait_ms": start.elapsed_time(wait), "exposed_wait_ms": wait.elapsed_time(ready)}
            if any(value is not None and (not math.isfinite(value) or value < 0) for value in durations.values()):
                raise RuntimeError(f"invalid P2P device event durations: {durations}")
            batches.append({key: value for key, value in record.items() if key not in {"stream", "events"}} | durations)
        summary = {}
        for phase in _PHASES:
            selected = [batch for batch in batches if batch["phase"] == phase]
            summary[phase] = {"batches": len(selected), "builder_calls": self.calls[phase],
                              **{name: sum(batch[name] for batch in selected if batch[name] is not None)
                                 if self.device_timing or name not in _DEVICE_METRICS else None for name in _METRICS},
                              "max_exposed_wait_ms": max(
                                  (batch["exposed_wait_ms"] for batch in selected), default=0
                              ) if self.device_timing else None}
        return {"batches": batches, "empty_batches": self.empty_batches, "summary": summary,
                "device_timing": self.device_timing}


def _validate_payload_balance(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    """Require matching EP-local peers, phase rounds, shapes, dtypes and byte counts."""
    if sorted(rank["rank"] for rank in ranks) != list(range(len(ranks))):
        raise RuntimeError("P2P diagnostics must contain each EP rank exactly once")
    sent, received = Counter(), Counter()
    totals = Counter()
    contexts = [item.get("context") for item in ranks]
    correlated = any(contexts)
    if correlated:
        if not all(contexts):
            raise RuntimeError("P2P diagnostics mix correlated and legacy captures")
        identities = {(ctx["capture_id"], ctx["step"], ctx["world_size"]) for ctx in contexts}
        if len(identities) != 1 or contexts[0]["world_size"] != len(ranks):
            raise RuntimeError("P2P diagnostics have incompatible capture identities")
    for rank in ranks:
        seen_batches = set()
        if correlated and rank["context"]["rank"] != rank["rank"]:
            raise RuntimeError("P2P rank disagrees with its capture context")
        for batch in [*rank["batches"], *rank.get("empty_batches", [])]:
            if not correlated:
                continue
            for name in ("capture_id", "rank", "world_size", "step"):
                if batch.get(name) != rank["context"][name]:
                    raise RuntimeError(f"P2P batch has incompatible {name}")
            key = tuple(batch.get(name) for name in (
                "phase", "step", "layer", "resource_group", "direction", "call_id", "reduction_index", "batch_index",
            ))
            if key in seen_batches:
                raise RuntimeError("Duplicate P2P batch identity")
            seen_batches.add(key)
        for batch in rank["batches"]:
            for transfer in batch["transfers"]:
                peer = transfer["peer_rank"]
                if transfer["operation"] not in {"send", "receive"} or not 0 <= peer < len(ranks):
                    raise RuntimeError(f"invalid P2P transfer metadata: {transfer}")
                source, target = (rank["rank"], peer) if transfer["operation"] == "send" else (peer, rank["rank"])
                round_key = (batch["phase_call"],)
                expert_key = ()
                if correlated:
                    round_key = tuple(batch.get(name) for name in (
                        "step", "layer", "resource_group", "direction", "call_id", "reduction_index", "batch_index",
                        "placement_id",
                    ))
                    expert_key = tuple(transfer[name] for name in (
                        "logical_expert", "tensor_role", "owner_slot", "replica_slot",
                    ))
                key = (batch["phase"], round_key, source, target, expert_key,
                       transfer["dtype"], tuple(transfer["shape"]), transfer["bytes"])
                counts = sent if transfer["operation"] == "send" else received
                counts[key] += 1
                if transfer["operation"] == "send":
                    totals[batch["phase"]] += transfer["bytes"]
    if sent != received:
        raise RuntimeError(f"P2P payload mismatch: unmatched sends={sent - received}, receives={received - sent}")
    return {"matching_peer_payloads": True,
            "sent_bytes": {phase: totals[phase] for phase in _PHASES}}


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
    context = ProfileContext(rank=worker._RANK, world_size=sweep.shape.ep_size,
                             capture_id=new_capture_id(worker.dist, "p2p"), source=sweep.source)
    observer = _P2PObserver(worker.torch.npu, worker._DEVICE, context=context)
    replica = importlib.import_module("hyper_parallel.core.multicore.modules.mega_moe.replica")
    with context.observe(worker.torch, sweep.layers, checkpoint="reentrant"), observer.observe(replica):
        sweep._train_step(route)
    worker.torch.npu.synchronize(worker._DEVICE)
    local = {"rank": worker._RANK, "context": context.manifest(), **observer.collect()}
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
