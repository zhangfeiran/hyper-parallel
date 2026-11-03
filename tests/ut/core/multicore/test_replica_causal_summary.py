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
"""Verify causal joins remain valid for clocks with arbitrary remote offsets."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from examples.training_demo.deepseek_v41.summarize_replica_causal import summarize


class TestReplicaCausalSummary(unittest.TestCase):
    """Owner publication fans out to each guest without inventing a remote latency."""

    def test_pair_owner_publication_with_each_guest_without_clock_subtraction(self) -> None:
        """A remote producer can have an arbitrary cycle origin, including a later one."""
        transfers = [{"logical_expert": 0, "owner_rank": 0, "owner_slot": 0,
                      "target_rank": rank, "target_slot": 6} for rank in (1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for rank in range(4):
                manifest = {"rank": rank, "ep_rank": rank, "ep_group_global_ranks": [0, 1, 2, 3],
                            "terminal_status": 0, "tensor_hashes_exact": True, "actual_plan_sha256": "fixed",
                            "observations": {"invocations": [], "dropped_records": 0, "host_dropped": 0}}
                (folder / f"rank{rank}.json").write_text(json.dumps(manifest))
                rows = []

                def event(stage: str, physical: int, peer: int, start: int, end: int) -> dict:
                    return {"ph": "X", "args": {"replica_protocol": "causal_v1", "band": 0,
                            "physical_expert": physical, "ep_local_rank": rank, "replica_lease_id": 0,
                            "epoch": 13, "projection": "W13", "peer_ep_rank": peer, "task_stage": stage,
                            "block_id": 0, "producer_core": 0, "clock_domain": f"device{rank}",
                            "start_cycle": start, "end_cycle": end}}

                if rank == 0:
                    rows.append(event("ProducerReady", 0, 0, 100, 110))
                    for peer in (1, 2):
                        rows.append(event("ConsumerReadyObserved", 0, peer, 135, 135))
                        rows.append(event("GradientReturn", 0, peer, 140, 150))
                elif rank in (1, 2):
                    rows.append(event("ProducerReady", 6, rank, 10**12, 10**12 + 10))
                payload = {"traceEvents": rows, "replicaHostObservations": {"events": [
                    {"kind": "LeaseBegin", "lease_id": 0,
                     "metadata": {"home_experts": 6, "transfers": transfers}}]}}
                with gzip.open(folder / f"rank{rank}.events.json.gz", "wt") as stream:
                    json.dump(payload, stream)
            result = summarize(folder, 4, folder / "facts")
            with gzip.open(folder / "facts/paired-bands.json.gz", "rt") as stream:
                paired = json.load(stream)
            self.assertEqual(result["paired_band_count"], 2)
            self.assertEqual(result["paired_returns_with_owner_and_guest_producers"], 2)
            self.assertEqual({row["guest_global_rank"] for row in paired}, {1, 2})
            self.assertTrue(all(row["owner_publication_to_observed_ready_cycles"] == 25 for row in paired))
            self.assertTrue(all(row["cross_rank_latency"] is None for row in paired))
            with self.assertRaises(FileNotFoundError):
                summarize(folder, 5, folder / "invalid")
