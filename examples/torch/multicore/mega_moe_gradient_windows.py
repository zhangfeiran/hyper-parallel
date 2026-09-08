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
"""Attribute globally ordered gradient windows from one correlated all-rank trace."""

from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from typing import Any

from examples.torch.multicore.mega_moe_p2p_profile import _validate_payload_balance
from examples.torch.multicore.mega_moe_profile_context import source_id
from examples.torch.multicore.mega_moe_trace import (
    CALL_FIELDS,
    Trace,
    communication_events,
    contains,
    duration_ns,
    identity_key,
    interval,
    one,
    validate_stream_order,
)

_NATIVE = "aclnnHyperMegaMoeGrad_HyperMegaMoeGrad_HyperMegaMoeGrad"
_BATCH = "batch_send_recvAicpuKernel"


def _owner_events(trace: Trace, scope: dict) -> tuple[list[dict], list[dict], list[dict]]:
    devices = trace.devices(scope)
    kernels = [event for event in devices if event["args"]["Task Type"].startswith("KERNEL")]
    add_scopes = [event for event in trace.events if event.get("ph") == "X"
                  and event.get("name") == "aten::add_" and contains(scope, event)]
    adds = [device for device, host in trace.joined if device in kernels
            and any(contains(add_scope, host) for add_scope in add_scopes)]
    return kernels, adds, [event for event in devices if event["name"] == "EVENT_WAIT"]


def rank_windows(path: Path, payload: dict) -> dict:
    """Resolve every local window, including empty ones, against its exact launch scopes."""
    context = payload["context"]
    if payload.get("memory_events") and payload.get("memory_clock_domain") != "host_aligned_epoch_ns":
        raise ValueError("Memory records have an incomparable clock domain")
    trace = Trace(path, context)
    gradients = [batch for batch in [*payload["batches"], *payload.get("empty_batches", [])]
                 if batch["phase"] == "gradient_return"]
    payloads = {identity_key(batch): batch for batch in gradients}
    scopes = trace.select("batch")
    if len(payloads) != len(gradients) or {identity_key(scope["metadata"]) for scope in scopes} != set(payloads):
        raise ValueError("Missing, duplicate or unmatched gradient window scopes/payloads")
    rows = []
    call_indices = {}
    for scope in scopes:
        metadata = scope["metadata"]
        key = identity_key(metadata)
        gradient = payloads[key]
        for field in ("placement_id", "placement_generation", "group_index", "receive_only"):
            if gradient.get(field) != metadata.get(field):
                raise ValueError(f"Gradient payload disagrees with trace {field}")
        call = {field: metadata[field] for field in CALL_FIELDS}
        call_key = identity_key(metadata, CALL_FIELDS)
        call_indices.setdefault(call_key, len(call_indices))
        kernel_scope = one(trace.select("kernel", **call, kernel_direction="backward"), "backward kernel scope")
        native = one(trace.devices(kernel_scope, name=_NATIVE), "native gradient kernel flow")
        row = {**metadata, "backward_index": call_indices[call_key], "empty": not gradient["transfers"],
               "native_start_ns": native["start_ns"], "native_end_ns": native["end_ns"],
               "send_to": sorted({item["peer_rank"] for item in gradient["transfers"]
                                  if item["operation"] == "send"}),
               "receive_from": sorted({item["peer_rank"] for item in gradient["transfers"]
                                       if item["operation"] == "receive"}),
               "transfers": gradient["transfers"], "send_bytes": gradient["send_bytes"],
               "receive_bytes": gradient["receive_bytes"]}
        if row["empty"]:
            if trace.devices(scope, name=_BATCH):
                raise ValueError("Empty gradient window launched communication")
            rows.append(row)
            continue
        batch = one(trace.devices(scope, name=_BATCH), "P2P control kernel flow")
        named = {}
        for label, suffix in (("launch", "_launch_transfers"), ("wait", "_wait_handles")):
            named[label] = one([event for event in trace.select("scope", scope=f"replica.{suffix}")
                                if contains(scope, event)], f"window {label} scope")
        records, barriers, joins = communication_events(
            trace.joined, native, {"reduction": scope, **named},
            receive_only=all(item["operation"] == "receive" for item in gradient["transfers"]),
        )
        group = one(trace.select("gradient_group", **call, reduction_index=metadata["reduction_index"],
                                 group_index=metadata["group_index"]), "gradient owner group")
        following = [item["start_ns"] for item in scopes
                     if contains(group, item) and item["start_ns"] > scope["start_ns"]]
        owner_scope = scope | {"start_ns": scope["end_ns"], "end_ns": min(following, default=group["end_ns"])}
        kernels, adds, waits = _owner_events(trace, owner_scope)
        independent = (barriers[0]["pid"], barriers[0]["tid"]) != (native["pid"], native["tid"])
        reuse = [event for event in waits if (event["pid"], event["tid"]) ==
                 (barriers[0]["pid"], barriers[0]["tid"])] if independent else []
        if row["receive_from"] and not adds:
            raise ValueError("Missing owner accumulation kernel flows")
        if independent and not reuse:
            raise ValueError("Missing receive-window reuse dependency")
        owner_start, owner_end = interval(kernels)
        add_start, add_end = interval(adds)
        reuse_start, reuse_end = interval(reuse)
        row.update(
            batch_control_start_ns=batch["start_ns"], batch_control_end_ns=batch["end_ns"],
            ready_ns=max(event["end_ns"] for event in records),
            wait_start_ns=min(event["start_ns"] for event in joins),
            wait_end_ns=max(event["end_ns"] for event in joins),
            wait_task_count=len(joins), ready_record_count=len(records),
            p2p_stream_id=barriers[0]["tid"], caller_stream_id=native["tid"], independent_receive=independent,
            p2p_wait_start_ns=min(event["start_ns"] for event in barriers),
            p2p_wait_end_ns=max(event["end_ns"] for event in barriers),
            owner_start_ns=owner_start, owner_end_ns=owner_end,
            owner_kernel_names=[event["name"] for event in kernels],
            owner_accumulation_start_ns=add_start, owner_accumulation_end_ns=add_end,
            owner_accumulation_duration_ns=duration_ns(adds),
            window_reuse_start_ns=reuse_start, window_reuse_end_ns=reuse_end,
            window_reuse_duration_ns=duration_ns(reuse),
            control_stream_wait_duration_ns=duration_ns(barriers) if independent else 0,
            compute_stream_join_duration_ns=duration_ns(joins),
        )
        validate_stream_order(row)
        if independent and reuse_end < add_end:
            raise ValueError("Receive window can be reused before owner accumulation completes")
        rows.append(row)
    # Reusing the same allocation on another stream requires the preceding owner fence.
    previous = {}
    for row in rows:
        for transfer in row["transfers"]:
            if transfer["operation"] != "receive":
                continue
            storage = (row["resource_group"], transfer["storage_address"], transfer["storage_offset_elements"])
            prior = previous.get(storage)
            if (prior is not None and prior is not row
                    and identity_key(prior, CALL_FIELDS) == identity_key(row, CALL_FIELDS)):
                end = (prior["window_reuse_end_ns"] if prior["independent_receive"]
                       else prior["owner_accumulation_end_ns"])
                if row["ready_ns"] < end:
                    raise ValueError("Next receive starts before window reuse dependency completes")
            previous[storage] = row
    return {"rank": payload["rank"], "wall_offset_ns": trace.offset, "trace_events": len(trace.events),
            "unlinked_device_events": trace.unlinked_device_events, "rows": rows,
            "memory": trace.memory(payload.get("memory_events"))}


def analyze_windows(run: dict[str, Any]) -> dict[str, Any]:
    """Validate one execution and report per-window and per-call costs without fixed EP/layer sizes."""
    capture = run["diagnostic_trace"]["capture"]
    ranks = capture["per_rank"]
    _validate_payload_balance(ranks)
    traces = {item["rank"]: item["path"] for item in run["diagnostic_trace"]["per_rank"]}
    if len(traces) != len(run["diagnostic_trace"]["per_rank"]) or set(traces) != set(range(len(ranks))):
        raise ValueError("Timeline attribution requires exactly one trace for every EP rank")
    contexts = [item["context"] for item in ranks]
    if len({item["host"] for item in contexts}) != 1:
        raise ValueError("Cross-host monotonic clocks are incomparable without an explicit calibration")
    if len({item.get("source_id") for item in contexts}) != 1:
        raise ValueError("Trace ranks disagree on source/native identity")
    if any(item.get("source_id") != source_id(run["source"]) for item in contexts):
        raise ValueError("Trace capture disagrees with benchmark source/native identity")
    expected_layers = run["configuration"]["layers"]
    if any([entry["layer"] for entry in item["layers"]] != list(range(expected_layers)) for item in contexts):
        raise ValueError("Trace layer registry disagrees with benchmark configuration")
    per_rank = [rank_windows(Path(traces[item["rank"]]), item) for item in sorted(ranks, key=lambda item: item["rank"])]
    lookups = [{identity_key(row): row for row in item["rows"]} for item in per_rank]
    active = [row for item in per_rank for row in item["rows"] if not row["empty"]]
    for row in active:
        neighbors = set(row["send_to"] + row["receive_from"])
        peers = [lookups[peer].get(identity_key(row)) for peer in neighbors]
        if any(peer is None or peer["empty"] for peer in peers):
            raise ValueError("Missing matching peer window")
        ready = max(peer["ready_ns"] for peer in peers)
        row.update(last_neighbor_ready_ns=ready, late_neighbor_ns=max(0, ready - row["wait_start_ns"]),
                   after_neighbors_ready_ns=row["wait_end_ns"] - max(row["wait_start_ns"], ready))
        if row["after_neighbors_ready_ns"] < -10000:
            raise ValueError("Wait completes before same-execution peer readiness")
    origin = min((row["native_start_ns"] for item in per_rank for row in item["rows"]), default=0)
    summaries = []
    for item in per_rank:
        rows = [row for row in item["rows"] if not row["empty"]]
        natives = {identity_key(row, CALL_FIELDS): row for row in item["rows"]}
        summaries.append({"rank": item["rank"], "windows": len(rows),
                          "empty_windows": len(item["rows"]) - len(rows),
                          "native_duration_ms": sum(row["native_end_ns"] - row["native_start_ns"]
                                                    for row in natives.values()) / 1e6,
                          "wait_ms": sum(row["compute_stream_join_duration_ns"] for row in rows) / 1e6,
                          "control_stream_wait_ms": sum(row["control_stream_wait_duration_ns"] for row in rows) / 1e6,
                          "owner_accumulation_ms": sum(row["owner_accumulation_duration_ns"] for row in rows) / 1e6,
                          "window_reuse_ms": sum(row["window_reuse_duration_ns"] for row in rows) / 1e6,
                          "independent_receive_calls": sum(row["independent_receive"] for row in rows),
                          "late_neighbor_ms": sum(row["late_neighbor_ns"] for row in rows) / 1e6,
                          "after_neighbors_ready_ms": sum(row["after_neighbors_ready_ns"] for row in rows) / 1e6})
    keys = {identity_key(row) for row in active}
    spreads = [max(row["ready_ns"] for row in active if identity_key(row) == key)
               - min(row["ready_ns"] for row in active if identity_key(row) == key) for key in keys]
    # Keep absolute host timestamps and one explicit origin; never shift durations or task cycles.
    return {"schema_version": 2, "capture_id": capture["capture_id"], "source": run["source"],
            "clock_domain": "same_host_monotonic_raw_ns", "unit": "ns", "origin_monotonic_ns": origin,
            "scope": "diagnostic execution only; stream waits can overlap and must not be added as wall time",
            "summary": summaries, "per_rank": per_rank,
            "ready_spread_median_ms": median(spreads) / 1e6 if spreads else None,
            "ready_spread_mean_ms": mean(spreads) / 1e6 if spreads else None,
            "ready_spread_max_ms": max(spreads) / 1e6 if spreads else None}
