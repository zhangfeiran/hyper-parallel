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
"""Independent flow fixtures for multi-window correlation and strict cost imports."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from examples.torch.multicore.mega_moe_gradient_windows import analyze_windows
from examples.torch.multicore.mega_moe_profile_context import scope_name, source_id
from examples.torch.multicore.megamoe_simulator.profile_import import (
    cube_services,
    import_profiles,
    task_attribution,
)

_SOURCE = {"native_artifacts": {"kernel.so": "a" * 64}, "python_implementation": {"sha256": "b" * 64}}


class _TraceFixture:
    """Build exact flow evidence from a known host/device schedule."""

    def __init__(self, context: dict, legacy: bool) -> None:
        """Keep per-rank fixture state separate from the outer rank loop."""
        self.context = context
        self.legacy = legacy
        self.events = []

    def scope(self, role: str, metadata: dict, start: float, end: float) -> None:
        """Emit a CPU interval with an independently supplied identity."""
        self.context["operations"].append({"role": role, **metadata})
        name = scope_name(role, metadata)
        if self.legacy:
            if role not in {"kernel", "reduction", "scope"}:
                return
            name = {"kernel": "megamoe::function._launch_backward",
                    "reduction": "megamoe::replica.ExpertReplicaManager.reduce_replica_gradients",
                    "scope": "megamoe::" + metadata.get("scope", "")}[role]
        self.events.append({"name": name, "ph": "X", "pid": 1, "tid": 2,
                       "ts": start, "dur": end - start})

    def device(self, name: str, host: float, start: float, end: float, stream: int) -> None:
        """Connect an explicitly supplied host submission to one device event."""
        identity = len(self.events)
        self.events.extend([
            {"cat": "async_npu", "ph": "s", "id": identity, "pid": 1, "tid": 2, "ts": host},
            {"cat": "async_npu", "ph": "f", "id": identity, "pid": 9, "tid": stream, "ts": start},
            {"ph": "X", "name": name, "pid": 9, "tid": stream, "ts": start, "dur": end - start,
             "args": {"Task Type": "KERNEL_AICORE" if name in ("Add", "native") else "EVENT"}},
        ])
        if name == "native":
            self.events[-1]["name"] = "aclnnHyperMegaMoeGrad_HyperMegaMoeGrad_HyperMegaMoeGrad"


def _fixture(directory: Path, ranks: int = 3, *, mixed: bool = False, legacy: bool = False) -> dict:
    """Three layers, two shared resource groups, two payload windows and a local empty window."""
    layers = 16 if legacy else 3
    run = {"source": _SOURCE, "configuration": {"layers": layers}, "points": [],
           "diagnostic_trace": {"per_rank": [], "capture": {"schema_version": 2, "capture_id": "trace:fixture",
                                                            "per_rank": []}}}
    for rank in range(ranks):
        base = {"capture_id": "trace:fixture", "rank": rank, "world_size": ranks, "step": 0,
                "source_id": source_id(_SOURCE)}
        context = {**base, "host": "fixture-host",
                   "layers": [{"layer": i, "resource_group": i % 2} for i in range(layers)],
                   "operations": [], "last_launches": [], "clock_domains": {"trace": "host_monotonic_raw_ns"}}
        payload = {"rank": rank, "context": context, "batches": [], "empty_batches": []}
        recorder = _TraceFixture(context, legacy)
        events, scope, device = recorder.events, recorder.scope, recorder.device

        for index, layer in enumerate(reversed(range(layers))):
            start = 1000 + index * 1000
            call = {**base, "layer": layer, "resource_group": layer % 2, "direction": "backward", "call_id": 0,
                    "placement_id": "p", "placement_generation": 1}
            scope("call", call, start, start + 200)
            kernel = {**call, "kernel_direction": "backward", "wave_count": 2, "num_cube_cores": 2,
                      "logical_experts": [rank, 0]}
            scope("kernel", kernel, start + 1, start + 10)
            device("native", start + 5, start + 300, start + 400, 47)
            group = {**call, "reduction_index": 0, "group_index": 0, "owner_slot": None}
            scope("reduction", {**call, "reduction_index": 0}, start + 12, start + 180)
            scope("gradient_group", group, start + 15, start + 170)
            for window in range(1 if legacy else 3):
                transfers = []
                edges = ([(owner, (owner + 1) % ranks) for owner in range(ranks)] if mixed else [(0, window + 1)])
                if window < 2:
                    for owner, target in edges:
                        if rank not in (owner, target):
                            continue
                        for tensor in ("gate_up", "down"):
                            transfers.append({"operation": "receive" if rank == owner else "send",
                                              "peer_rank": target if rank == owner else owner,
                                              "logical_expert": owner, "owner_slot": 0, "replica_slot": 1,
                                              "tensor_role": tensor, "dtype": "torch.float32", "shape": [2, 2],
                                              "bytes": 16, "storage_address": 4096 if tensor == "gate_up" else 8192,
                                              "storage_offset_elements": 0})
                independent = bool(transfers) and all(item["operation"] == "receive" for item in transfers)
                metadata = {**group, "batch_index": window, "receive_only": independent}
                host = start + 20 + window * 55
                scope("batch", metadata, host, host + 35)
                batch = {**metadata, "phase": "gradient_return", "phase_call": index * 3 + window,
                         "transfers": transfers, "send_bytes": sum(item["bytes"] for item in transfers
                                                                    if item["operation"] == "send"),
                         "receive_bytes": sum(item["bytes"] for item in transfers if item["operation"] == "receive")}
                payload["batches" if transfers else "empty_batches"].append(batch)
                if not transfers:
                    continue
                scope("scope", {**metadata, "scope": "replica._launch_transfers"}, host + 5, host + 10)
                scope("scope", {**metadata, "scope": "replica._wait_handles"}, host + 15, host + 20)
                ready = start + (150 if independent and window == 0 else 420 + window * 110)
                stream = 44 if independent else 47
                device("EVENT_RECORD", host + 6, ready - 1, ready, stream)
                device("batch_send_recvAicpuKernel", host + 7, ready + 1, ready + 2, 60)
                device("EVENT_WAIT", host + 16, ready + 2, start + 470 + window * 110, stream)
                if independent:
                    device("EVENT_WAIT", host + 25, start + 430 + window * 110, start + 480 + window * 110, 47)
                if batch["receive_bytes"]:
                    events.append({"name": "aten::add_", "ph": "X", "pid": 1, "tid": 2,
                                   "ts": host + 40, "dur": 3})
                    device("Add", host + 41, start + 490 + window * 110, start + 500 + window * 110, 47)
                if independent:
                    device("EVENT_WAIT", host + 46, start + 501 + window * 110, start + 510 + window * 110, stream)
            events.append({"name": "[memory]", "ph": "i", "pid": 1, "tid": 2,
                           "ts": start + 50, "args": {"Bytes": 16}})
        path = directory / f"trace.rank{rank}.json"
        path.write_text(json.dumps(events), encoding="utf-8")
        raw = directory / f"{path.stem}_fixture_ascend_pt/PROF_fixture/host"
        raw.mkdir(parents=True)
        (raw / "start_info").write_text(json.dumps({"collectionTimeBegin": 1000, "clockMonotonicRaw": 1000000}))
        run["diagnostic_trace"]["per_rank"].append({"rank": rank, "path": str(path)})
        run["diagnostic_trace"]["capture"]["per_rank"].append(payload)
    if legacy:
        run["diagnostic_p2p"] = {"per_rank": run["diagnostic_trace"].pop("capture")["per_rank"]}
    return run


def _tasks() -> tuple[dict, dict]:
    context = {"num_cube_cores": 2, "logical_experts": [12], "direction": "recompute"}
    workers = []
    for kind in ("cube", "vector"):
        for core in range(2):
            records = [{"task_type": 104, "task_index": core, "tiling_position": position, "wave": 0,
                        "expert_rows": 32, "start_cycle": 100, "ready_cycle": 200 + core,
                        "compute_end_cycle": 1200 + core, "end_cycle": 1250 + core}
                       for position in (17, 19, 20, 21, 22, 23)] if kind == "cube" else []
            workers.append({"kind": kind, "core": core, "stride": 1, "seen": len(records),
                            "recorded": len(records), "records": records})
    return {"schema_version": 1, "unit": "system_cycle", "workers": workers}, context


class TestProfilerIntegration(unittest.TestCase):
    """Exercise attribution against known schedules, then corrupt independent evidence."""

    def test_windows_keep_empty_batches_groups_and_receive_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = analyze_windows(_fixture(Path(directory)))
        owner = result["per_rank"][0]
        self.assertEqual(len(owner["rows"]), 9)
        self.assertEqual(sum(row["empty"] for row in owner["rows"]), 3)
        self.assertEqual({row["resource_group"] for row in owner["rows"]}, {0, 1})
        self.assertEqual(result["summary"][0]["native_duration_ms"], 0.3)
        self.assertEqual(result["summary"][0]["owner_accumulation_ms"], 0.06)
        self.assertEqual(result["summary"][0]["window_reuse_ms"], 0.054)
        self.assertEqual(owner["memory"]["attributed"], 3)
        self.assertTrue(all(row["independent_receive"] for row in owner["rows"] if not row["empty"]))

    def test_mixed_and_more_ranks_do_not_depend_on_ep8_or_layer_order(self) -> None:
        for ranks, mixed in ((3, True), (16, False), (32, False), (64, False)):
            with self.subTest(ranks=ranks, mixed=mixed), tempfile.TemporaryDirectory() as directory:
                run = _fixture(Path(directory), ranks, mixed=mixed)
                run["diagnostic_trace"]["per_rank"].reverse()
                result = analyze_windows(run)
                self.assertEqual(len(result["summary"]), ranks)
                if mixed:
                    self.assertTrue(all(row["send_to"] and row["receive_from"] and not row["independent_receive"]
                                        for item in result["per_rank"] for row in item["rows"] if not row["empty"]))

    def test_missing_events_wrong_identity_and_clocks_are_rejected(self) -> None:
        for corruption in ("peer", "dtype", "bytes", "window", "clock", "source", "missing", "reuse"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                run = _fixture(Path(directory))
                payload = run["diagnostic_trace"]["capture"]["per_rank"][0]
                if corruption in ("peer", "dtype", "bytes"):
                    field, value = {"peer": ("peer_rank", 0), "dtype": ("dtype", "torch.bfloat16"),
                                    "bytes": ("bytes", 8)}[corruption]
                    payload["batches"][0]["transfers"][0][field] = value
                elif corruption == "window":
                    payload["batches"][0]["batch_index"] = 8
                elif corruption == "clock":
                    payload["context"]["host"] = "another-host"
                elif corruption == "source":
                    run["source"] = {"native_artifacts": {"kernel.so": "c" * 64}}
                else:
                    path = Path(run["diagnostic_trace"]["per_rank"][0]["path"])
                    events = json.loads(path.read_text(encoding="utf-8"))
                    if corruption == "missing":
                        events = [event for event in events if event.get("name") != "batch_send_recvAicpuKernel"]
                    else:
                        event = next(event for event in events if event.get("name") == "EVENT_WAIT"
                                     and event["tid"] == 44 and event["ts"] == 1501)
                        event["dur"] = -10
                    path.write_text(json.dumps(events), encoding="utf-8")
                with self.assertRaises((ValueError, RuntimeError)):
                    analyze_windows(run)

    def test_import_separates_profile_off_and_diagnostic_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = _fixture(Path(directory))
            result = import_profiles([run], expected_identity=_SOURCE)
            self.assertEqual(len(result["executions"]), 1)
            self.assertEqual(result["measurements"], [])
            self.assertEqual(result["timelines"][0]["capture_id"], "trace:fixture")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                import_profiles([run], expected_identity={})

    def test_six_gemm_phases_have_expert_readiness_without_wait_cost(self) -> None:
        profile, context = _tasks()
        result = task_attribution(profile, context)
        self.assertEqual({row["stage"] for row in result["expert_services"]}, {"dX", "dW", "recompute"})
        self.assertEqual({row["logical_expert"] for row in result["expert_services"]}, {12})
        self.assertEqual({row["compute_us"] for row in result["expert_services"]}, {20.0})
        self.assertEqual({row["ready_cycle"] for row in result["expert_services"]}, {201})
        self.assertIn("last launch", result["coverage"])

    def test_sampled_diagnostic_is_not_eligible_for_gemm_calibration(self) -> None:
        profile, context = _tasks()
        profile["workers"][0]["stride"] = 2
        result = task_attribution(profile, context)
        self.assertFalse(result["gemm_calibration_eligible"])
        self.assertEqual(result["expert_services"], [])
        with self.assertRaisesRegex(ValueError, "unsampled"):
            cube_services(profile, expected_cores=2)
        for mutation in ("missing_queue", "duplicate_queue", "missing_record", "duplicate_record"):
            profile, _ = _tasks()
            if mutation == "missing_queue":
                profile["workers"].pop(0)
            elif mutation == "duplicate_queue":
                profile["workers"][0]["core"] = 1
            elif mutation == "missing_record":
                profile["workers"][0]["records"].pop()
            else:
                queue = profile["workers"][0]
                queue["records"].append(copy.deepcopy(queue["records"][0]))
                queue["seen"] += 1
                queue["recorded"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cube_services(profile, expected_cores=2)

    def test_cost_worker_import_keeps_task_and_scope_steps_distinct(self) -> None:
        """Two diagnostic executions preserve their own clocks, placement and last-launch coverage."""
        profile, metadata = _tasks()
        base = {"rank": 0, "world_size": 2, "step": 0, "source_id": source_id(_SOURCE),
                "clock_domains": {"tasks": "rank_local_cycles", "scopes": "stream_elapsed_us"}}
        task = {**metadata, **base, "capture_id": "tasks:fixture", "layer": 0, "resource_group": 0}
        captures = {
            "tasks": {**base, "capture_id": "tasks:fixture", "last_launches": [task]},
            "scopes": {**base, "capture_id": "scopes:fixture", "last_launches": []},
        }
        point = {"pair": "0.1:42", "profile_off_samples_ms": [1.2], "memory": {"peak_bytes": 123},
                 "shape": {}, "captures": captures, "task_profiles": {"forward": profile},
                 "task_contexts": {"forward": task}, "cycle_profiles": {},
                 "scoped_events": [{"scope": "forward", "context": captures["scopes"], "stream_us": 1500}]}
        document = {"profile_schema_version": 2, "identity": _SOURCE, "rank": 0, "results": [point]}
        result = import_profiles([document], expected_identity=_SOURCE)
        self.assertEqual(len(result["executions"]), 2)
        self.assertEqual(result["measurements"][0]["latency_ms"], [1.2])
        self.assertEqual(result["measurements"][0]["memory"]["peak_bytes"], 123)
        self.assertEqual(result["executions"][0]["missing_ranks"], [1])
        point["task_contexts"]["forward"] = task | {"capture_id": "scopes:fixture"}
        with self.assertRaisesRegex(ValueError, "last launch"):
            import_profiles([document], expected_identity=_SOURCE)
