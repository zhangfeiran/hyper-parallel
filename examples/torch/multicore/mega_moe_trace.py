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
"""Exact Chrome flow parsing and contextual attribution for diagnostic executions."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from examples.torch.multicore.mega_moe_profile_context import parse_scope_name

CALL_FIELDS = ("step", "layer", "resource_group", "direction", "call_id")
BATCH_FIELDS = (*CALL_FIELDS, "reduction_index", "batch_index")


def identity_key(metadata: dict, fields: tuple = BATCH_FIELDS) -> tuple:
    """Require every component instead of pairing operations by list position."""
    return tuple(metadata[field] for field in fields)


def contains(scope: dict, host: dict) -> bool:
    """Attribute host submissions only on the exact process and thread."""
    return ((scope["pid"], scope["tid"]) == (host["pid"], host["tid"])
            and scope["start_ns"] <= host["start_ns"] < scope["end_ns"])


def nanoseconds(value: Any) -> int:
    """Convert decimal microseconds before subtracting an epoch offset."""
    return int(Decimal(str(value)) * 1000)


class Trace:
    """Read one calibrated host timeline; device cycles never enter this clock."""

    def __init__(self, path: Path, context: dict | None = None) -> None:
        """Load one raw clock calibration and verify the exported capture scopes."""
        roots = list(path.parent.glob(path.stem + "_*_ascend_pt"))
        if len(roots) != 1:
            raise ValueError(f"Expected one raw artifact root for {path}: {roots}")
        starts = list(roots[0].glob("PROF_*/host/start_info"))
        if len(starts) != 1:
            raise ValueError(f"Expected one CANN clock calibration: {starts}")
        clock = json.loads(starts[0].read_text(encoding="utf-8"), parse_float=Decimal)
        self.offset = nanoseconds(clock["collectionTimeBegin"]) - int(clock["clockMonotonicRaw"])
        document = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
        self.events = document["traceEvents"] if isinstance(document, dict) else document
        self.scopes = []
        starts, finishes = {}, {}
        for event in self.events:
            if "ts" in event:
                event["start_ns"] = nanoseconds(event["ts"]) - self.offset
                event["end_ns"] = event["start_ns"] + nanoseconds(event.get("dur", 0))
            if event.get("ph") == "X":
                metadata = parse_scope_name(event.get("name", ""))
                if metadata is not None:
                    if context is None or any(metadata.get(key) != context.get(key) for key in (
                        "capture_id", "rank", "world_size", "step", "source_id",
                    )):
                        raise ValueError("Trace scope disagrees with its capture/source identity")
                    self.scopes.append(event | {"metadata": metadata})
            if event.get("cat") != "async_npu":
                continue
            if event["ph"] == "s":
                if event["id"] in starts:
                    raise ValueError("Duplicate host flow identity")
                starts[event["id"]] = event
            elif event["ph"] == "f":
                key = (event["pid"], event["tid"], event["start_ns"])
                if key in finishes:
                    raise ValueError("Duplicate device flow endpoint")
                finishes[key] = event
        self.joined = []
        self.unlinked_device_events = 0
        for event in self.events:
            if event.get("ph") != "X" or not event.get("args", {}).get("Task Type"):
                continue
            finish = finishes.get((event["pid"], event["tid"], event["start_ns"]))
            if finish is None:
                self.unlinked_device_events += 1
                continue
            if finish["id"] not in starts:
                raise ValueError("Device flow has no host submission")
            self.joined.append((event, starts[finish["id"]]))
        self.scopes.sort(key=lambda event: event["start_ns"])
        if context is not None:
            roles = {"call", "kernel", "batch", "gradient_group", "reduction"}
            expected = Counter(json.dumps(item, sort_keys=True) for item in context["operations"]
                               if item["role"] in roles)
            observed = Counter(json.dumps(item["metadata"], sort_keys=True) for item in self.scopes
                               if item["metadata"]["role"] in roles)
            if expected != observed:
                raise ValueError("Missing or unexpected contextual trace scopes")

    def select(self, role: str, **fields: Any) -> list[dict]:
        """Select fully identified complete CPU intervals."""
        return [scope for scope in self.scopes if scope["metadata"]["role"] == role
                and all(scope["metadata"].get(key) == value for key, value in fields.items())]

    def devices(self, scope: dict, *, name: str | None = None) -> list[dict]:
        """Join by flow endpoint and containing CPU scope, never nearest timestamp."""
        return [device for device, host in self.joined if contains(scope, host)
                and (name is None or device["name"] == name)]

    def memory(self, framework_records: list[dict] | None = None) -> dict:
        """Keep allocator events in their trace domain with explicit attribution coverage."""
        events, unattributed = [], 0
        exported = ([{"pid": item["pid"], "tid": item["tid"], "args": item["args"],
                      "start_ns": item["timestamp_ns"] - self.offset} for item in framework_records]
                    if framework_records else [event for event in self.events if event.get("name") == "[memory]"])
        for event in exported:
            scopes = [scope for scope in self.scopes if contains(scope, event)]
            selected = min(scopes, key=lambda scope: scope["end_ns"] - scope["start_ns"], default=None)
            if selected is None:
                unattributed += 1
            events.append({"timestamp_ns": event["start_ns"], "args": event.get("args", {}),
                           "context": selected["metadata"] if selected is not None else None})
        return {"events": events, "attributed": len(events) - unattributed, "unattributed": unattributed,
                "coverage": "recorded allocator events only; not a total HBM ledger or profile-off peak",
                "available": bool(events), "unit": "bytes", "clock_domain": "host_monotonic_raw_ns"}


def one(events: list[dict], description: str) -> dict:
    """Reject missing or ambiguous evidence instead of choosing a nearby event."""
    if len(events) != 1:
        raise ValueError(f"Expected exactly one {description}; found {len(events)}")
    return events[0]


def interval(events: list[dict]) -> tuple[int | None, int | None]:
    """Bound a set of events; empty sets remain explicitly empty."""
    return (min((event["start_ns"] for event in events), default=None),
            max((event["end_ns"] for event in events), default=None))


def duration_ns(events: list[dict]) -> int:
    """Union intervals rather than double-counting nested or overlapping work."""
    total, end = 0, None
    for event in sorted(events, key=lambda item: item["start_ns"]):
        start = event["start_ns"] if end is None else max(end, event["start_ns"])
        total += max(0, event["end_ns"] - start)
        end = max(end or event["end_ns"], event["end_ns"])
    return total


def communication_events(
    joined: list[tuple[dict[str, Any], dict[str, Any]]],
    native: dict[str, Any],
    scopes: dict[str, dict[str, Any]],
    *,
    receive_only: bool,
    include_scope_end: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate a P2P wait from its eventual compute-stream join through exact scopes."""
    def within(scope: dict, host: dict) -> bool:
        """Retain the legacy inclusive-end convention only for legacy input."""
        return contains(scope, host) or (include_scope_end and
                                        (scope["pid"], scope["tid"], scope["end_ns"]) ==
                                        (host["pid"], host["tid"], host["start_ns"]))

    barriers = [device for device, host in joined if within(scopes['wait'], host)
                and device['name'] == 'EVENT_WAIT']
    streams = {(event['pid'], event['tid']) for event in barriers}
    if len(streams) != 1:
        raise ValueError('Non-unique P2P wait stream')
    stream = next(iter(streams))
    caller = (native['pid'], native['tid'])
    independent = stream != caller
    if independent and not receive_only:
        raise ValueError('Gradient send lost its native producer dependency')
    records = [device for device, host in joined if within(scopes['launch'], host)
               and (device['pid'], device['tid']) == stream and device['name'] == 'EVENT_RECORD']
    caller_barriers = ([device for device, host in joined
                        if within(scopes['reduction'], host) and not within(scopes['wait'], host)
                        and (device['pid'], device['tid']) == caller and device['name'] == 'EVENT_WAIT']
                       if independent else barriers)
    if not records or not caller_barriers:
        raise ValueError('Missing P2P ready record or caller-stream join')
    return records, barriers, caller_barriers


def validate_stream_order(row: dict[str, Any]) -> None:
    """Require native production and P2P completion before owner consumption."""
    if ((not row['independent_receive'] and row['ready_ns'] < row['native_end_ns'])
            or row['p2p_wait_start_ns'] < row['ready_ns']
            or row['wait_start_ns'] < row['native_end_ns']
            or row['wait_end_ns'] < row['p2p_wait_end_ns']):
        raise ValueError('Invalid producer/communication/consumer stream order')
    if row['owner_start_ns'] is not None and row['owner_start_ns'] < row['wait_end_ns']:
        raise ValueError('Owner starts before P2P completion')
