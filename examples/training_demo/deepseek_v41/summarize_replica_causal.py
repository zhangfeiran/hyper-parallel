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
"""Join protocol identities across ranks without subtracting remote clock cycles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from typing import Any


def _identity(args: dict[str, Any], lease: dict[str, Any]) -> int:
    """Map a physical owner/guest to an immutable logical expert."""
    expert, rank, home = args["physical_expert"], args["ep_local_rank"], lease["home_experts"]
    if expert is None:
        raise ValueError("Projection-wide completion cannot identify a band expert")
    if expert < home:
        return rank * home + expert
    matches = [edge["logical_expert"] for edge in lease["transfers"]
               if edge["target_rank"] == rank and edge["target_slot"] == expert]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous physical guest identity: {args}")
    return matches[0]


def _collect(folder: Path, world: int) -> tuple[list[dict[str, Any]], dict[tuple, list[dict[str, Any]]]]:
    """Require every rank and collect complete local observations by protocol identity."""
    manifests, groups = [], defaultdict(list)
    for rank in range(world):
        manifest = json.loads((folder / f"rank{rank}.json").read_text())
        if manifest["terminal_status"] or not manifest["tensor_hashes_exact"]:
            raise ValueError(f"Failed rank: {rank}")
        manifests.append(manifest)
        with gzip.open(folder / f"rank{rank}.events.json.gz", "rt") as stream:
            trace = json.load(stream)
        leases = {event["lease_id"]: event["metadata"]
                  for event in trace["replicaHostObservations"]["events"] if event["kind"] == "LeaseBegin"}
        for event in trace["traceEvents"]:
            args = event.get("args", {})
            if args.get("replica_protocol") != "causal_v1" or args["physical_expert"] is None:
                continue
            lease = leases[args["replica_lease_id"]]
            logical = _identity(args, lease)
            owner = logical // lease["home_experts"]
            peer = args["peer_ep_rank"] if args["ep_local_rank"] == owner else args["ep_local_rank"]
            peers = [peer]
            if args["task_stage"] in ("ProducerReady", "EmptyProducerReady") and args["ep_local_rank"] == owner:
                peers = [edge["target_rank"] for edge in lease["transfers"] if edge["logical_expert"] == logical]
            for target in peers:
                key = (tuple(manifest["ep_group_global_ranks"]), args["epoch"], args["projection"], logical,
                       owner, target, args["band"])
                groups[key].append({"rank": rank, "stage": args["task_stage"], "worker": args["block_id"],
                                    "clock_domain": args["clock_domain"], "producer_core": args["producer_core"],
                                    "start_cycle": args["start_cycle"], "end_cycle": args["end_cycle"]})
    return manifests, groups


def _band_facts(key: tuple, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the owner and guest timestamps in separate clock domains."""
    ranks, epoch, projection, logical, owner, peer, band = key
    owner_rank, guest_rank = ranks[owner], ranks[peer]
    producer = [event for event in events if event["stage"] in ("ProducerReady", "EmptyProducerReady")]
    returned = [event for event in events if event["stage"] == "GradientReturn"]
    owner_producer = [event for event in producer if event["rank"] == owner_rank]
    guest_producer = [event for event in producer if event["rank"] == guest_rank]
    owner_end = max((event["end_cycle"] for event in owner_producer), default=None)
    ready = [event for event in events if event["stage"] == "ConsumerReadyObserved"]
    first_ready = min((event["start_cycle"] for event in ready), default=None)
    independent = [event for event in events if event["stage"] == "IndependentReadyObserved"]
    return {"ep_group_global_ranks": ranks, "epoch": epoch, "projection": projection,
            "logical_expert": logical, "owner_global_rank": owner_rank, "guest_global_rank": guest_rank,
            "band": band, "producer_cores_owner": len({event["producer_core"] for event in owner_producer}),
            "producer_cores_guest": len({event["producer_core"] for event in guest_producer}),
            "owner_publication_end_cycle": owner_end,
            "guest_publication_end_cycle": max((event["end_cycle"] for event in guest_producer), default=None),
            "consumer_first_ready_owner_cycle": first_ready,
            "return_end_owner_cycle": max((event["end_cycle"] for event in returned), default=None),
            "owner_publication_to_observed_ready_cycles":
            first_ready - owner_end if first_ready is not None and owner_end is not None else None,
            "owner_failure_observations": sum(event["stage"] == "LocalNotReadyObserved" for event in events),
            "guest_failure_observations": sum(event["stage"] == "GuestNotReadyObserved" for event in events),
            "independent_ready_observations": len(independent),
            "return_workers": len({event["worker"] for event in returned}),
            "ack_issued_owner_cycle": max((event["end_cycle"] for event in events
                                           if event["stage"] == "AckIssued"), default=None),
            "ack_observed_guest_end_cycle": max((event["end_cycle"] for event in events
                                                 if event["stage"] == "AckObserved"), default=None),
            "cross_rank_latency": None,
            "notes": "Remote producer timing is causal only; compare cycles within each rank only"}


def summarize(folder: Path, world: int, output: Path) -> dict[str, Any]:
    """Persist paired band facts and all-rank capture quality before interpreting them."""
    manifests, grouped = _collect(folder, world)
    facts = [_band_facts(key, rows) for key, rows in grouped.items()]
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "paired-bands.json.gz", "wt") as stream:
        json.dump(facts, stream, separators=(",", ":"))
    events = Counter()
    rank_rows = []
    for manifest in manifests:
        backward = [invocation for invocation in manifest["observations"]["invocations"]
                    if invocation["direction"] == "backward"]
        for invocation in backward:
            events.update(invocation["event_counts"])
            rank_rows.append({"rank": manifest["rank"], "ep_rank": manifest["ep_rank"], **invocation})
    result = {"schema_version": 1, "expected_ranks": world, "complete_ranks": len(manifests),
              "actual_plan_hashes": sorted({item["actual_plan_sha256"] for item in manifests}),
              "dropped_records": sum(item["observations"]["dropped_records"] for item in manifests),
              "host_dropped": sum(item["observations"]["host_dropped"] for item in manifests),
              "tensor_hashes_exact": all(item["tensor_hashes_exact"] for item in manifests),
              "backward_events": dict(events), "rank_backward": rank_rows,
              "paired_band_count": sum(item["band"] is not None for item in facts),
              "paired_ack_count": sum(item["ack_issued_owner_cycle"] is not None
                                      and item["ack_observed_guest_end_cycle"] is not None for item in facts),
              "paired_returns_with_owner_and_guest_producers": sum(
                  item["return_workers"] > 0 and item["producer_cores_owner"] > 0
                  and item["producer_cores_guest"] > 0 for item in facts),
              "cross_rank_latency": None,
              "scope": "Executor diagnostic; no full-model performance or resource-contention claim"}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Analyze one completed case and retain only local-clock interval arithmetic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.world <= 0:
        raise ValueError("Expected world size must be positive")
    result = summarize(args.folder, args.world, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rank_backward"}))


if __name__ == "__main__":
    main()
