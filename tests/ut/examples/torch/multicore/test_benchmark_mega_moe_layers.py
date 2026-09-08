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
"""CPU contracts for the checkpointed multi-layer benchmark protocol."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from examples.torch.multicore import benchmark_mega_moe_layers as benchmark
from examples.torch.multicore import mega_moe_layer_profile
from examples.torch.multicore.benchmark_mega_moe_layers import (
    ADAMW_OPTIONS,
    _configuration,
    _execution_route_summary,
    _make_optimizer,
    _optimizer_state_ledger,
    _parse_args,
    _python_source_identity,
    _route_pairs,
)
from hyper_parallel.core.dtensor.dtensor import DTensor


class TestBenchmarkMegaMoeLayers(unittest.TestCase):
    """Keep AdamW execution, state accounting and reported metadata consistent."""

    def test_optimizer_is_adamw_with_reported_options(self):
        """The benchmark must execute AdamW, not SGD or coupled-decay Adam."""
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        args = argparse.Namespace(variant="native", layers=16, warmup=3, measured=7, shuffle_seed=1)
        config = _configuration(args)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(config["optimizer"], {"name": "AdamW", **ADAMW_OPTIONS})
        for name, expected in ADAMW_OPTIONS.items():
            with self.subTest(option=name):
                self.assertEqual(optimizer.param_groups[0][name], expected)

    def test_state_ledger_counts_actual_bfloat16_moments(self):
        """Count both moments and the CPU step scalar, without assuming FP32 moments."""
        parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        self.assertEqual(_optimizer_state_ledger(torch, optimizer)["tensor_bytes"], 0)
        parameter.square().sum().backward()
        optimizer.step()
        ledger = _optimizer_state_ledger(torch, optimizer)
        moment_bytes = 2 * parameter.numel() * parameter.element_size()
        step_bytes = optimizer.state[parameter]["step"].element_size()
        self.assertEqual(ledger["parameter_state_count"], 1)
        self.assertEqual(ledger["tensor_bytes"], moment_bytes + step_bytes)
        self.assertEqual(ledger["tensor_bytes_by_device"], {"cpu": moment_bytes + step_bytes})
        self.assertEqual(ledger["tensor_bytes_by_dtype"], {"torch.bfloat16": moment_bytes, "torch.float32": step_bytes})

    def test_nonfinite_adamw_state_is_rejected(self):
        """A finite forward alone must not hide a corrupted optimizer state."""
        parameter = torch.nn.Parameter(torch.ones(2, device="cpu"))
        optimizer = _make_optimizer(torch, [parameter])
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.state[parameter]["exp_avg_sq"].fill_(float("inf"))
        with self.assertRaisesRegex(AssertionError, "nonfinite AdamW state: exp_avg_sq"):
            _optimizer_state_ledger(torch, optimizer)

    def test_state_ledger_counts_local_dtensor_shard(self):
        """EP parameters and AdamW moments must not count global logical bytes."""
        shard = torch.ones(2, dtype=torch.bfloat16, device="cpu")
        distributed = MagicMock(spec=DTensor)
        distributed.to_local.return_value = shard
        optimizer = SimpleNamespace(state={"parameter": {"exp_avg": distributed}})
        ledger = _optimizer_state_ledger(torch, optimizer)
        self.assertEqual(ledger["tensor_bytes"], shard.numel() * shard.element_size())
        distributed.to_local.assert_called_once_with()
        distributed.numel.assert_not_called()

    def test_layer_factories_keep_variant_weights_aligned(self):
        """Both backends use standard EP initialization with a distinct seed per layer."""
        for variant in ("native", "b0", "b1"):
            with self.subTest(variant=variant):
                worker = MagicMock()
                worker.torch = torch
                layers = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
                worker.new_common_moe.side_effect = layers
                worker.new_common_layers.side_effect = [(layer, object()) for layer in layers]
                sweep = benchmark._LayerSweep.__new__(benchmark._LayerSweep)
                sweep.worker = worker
                sweep.shape = object()
                sweep.args = argparse.Namespace(variant=variant, layers=2)
                sweep.slots = int(variant == "b1")
                result = sweep._make_layers()
                self.assertEqual(list(result), layers)
                if variant == "native":
                    self.assertEqual(worker.new_common_moe.call_args_list,
                                     [call(sweep.shape, seed=2026), call(sweep.shape, seed=2126)])
                    worker.new_common_layers.assert_not_called()
                    worker.MegaMoeExperts.share_execution_resources.assert_not_called()
                else:
                    self.assertEqual(worker.new_common_layers.call_args_list, [
                        call(sweep.shape, seed=seed, expert_capacity_factor=1.5,
                             replica_slots_per_rank=sweep.slots, shared_physical_weights=bool(sweep.slots))
                        for seed in (2026, 2126)
                    ])
                    worker.new_common_moe.assert_not_called()
                    worker.MegaMoeExperts.share_execution_resources.assert_called_once_with(layers)

    def test_limit_preserves_full_sweep_prefix(self):
        """Stability smoke must replay the full sweep's route order exactly."""
        args = argparse.Namespace(pairs=None, shuffle_seed=20260903, limit=None)
        full = _route_pairs(args)
        args.limit = 4
        self.assertEqual(len(full), 55)
        self.assertEqual(_route_pairs(args), full[:4])

    def test_one_wave_summary_matches_actual_home_bypass(self):
        """Do not report hypothetical mean-target replicas when no planner runs."""
        proposal = {
            "replica_slots_per_rank": 1, "baseline_wave_count": 1, "expected_wave_count": 1,
            "destination_loads": [5, 3], "realized_destination_skew": 1.25,
            "physical_destination_loads": [4, 4], "physical_destination_skew": 1.0,
            "receive_capacity": 6, "planner_target_capacity": 4, "replica_slots_used": 1,
            "replica_slot_bindings": [[-1], [0]], "execution_mode": "replica-on-one-wave",
        }
        summary = _execution_route_summary(proposal)
        self.assertEqual(summary["physical_destination_loads"], [5, 3])
        self.assertEqual(summary["physical_destination_skew"], 1.25)
        self.assertEqual(summary["replica_slot_bindings"], [[-1], [-1]])
        self.assertEqual(summary["replica_slots_used"], 0)
        self.assertEqual(summary["planner_target_capacity"], 6)
        self.assertEqual(summary["execution_mode"], "home-only")
        self.assertEqual(proposal["replica_slots_used"], 1)
        proposal["baseline_wave_count"] = 2
        self.assertIs(_execution_route_summary(proposal), proposal)
        proposal["baseline_wave_count"] = 1
        proposal["replica_slots_per_rank"] = 0
        self.assertIs(_execution_route_summary(proposal), proposal)

    def test_source_identity_detects_dirty_runtime_changes(self):
        """Include uncommitted Python implementation edits in benchmark provenance."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "hyper_parallel/core/multicore/runtime.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first = _python_source_identity(repo)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = _python_source_identity(repo)
        self.assertEqual(first["file_count"], 1)
        self.assertEqual(second["file_count"], 1)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_profile_path_does_not_overwrite_benchmark(self):
        """The optional diagnostic trace must not overwrite timing evidence."""
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "timing.json")
            with self.assertRaises(SystemExit):
                _parse_args(["--variant", "b1", "--output", output, "--profile-output", output])
            args = _parse_args(["--variant", "b1", "--output", output])
        self.assertIsNone(args.profile_output)

    def test_all_rank_profile_requires_output_and_rejects_rank_path_collisions(self) -> None:
        """Derived rank filenames must neither overwrite results nor existing trace artifacts."""
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            root = Path(directory)
            base = ["--variant", "b1", "--output", str(root / "timing.json")]
            self.assertFalse(_parse_args(base).profile_all_ranks)
            with self.assertRaises(SystemExit):
                _parse_args([*base, "--profile-all-ranks"])
            profile = ["--profile-output", str(root / "trace.json"), "--profile-all-ranks"]
            self.assertTrue(_parse_args([*base, *profile]).profile_all_ranks)
            with self.assertRaises(SystemExit):
                _parse_args(["--variant", "b1", "--output", str(root / "trace.rank1.json"), *profile])
            (root / "trace.rank1.json").write_text("keep\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                _parse_args([*base, *profile])
            self.assertEqual((root / "trace.rank1.json").read_text(encoding="utf-8"), "keep\n")

    def test_runtime_identity_excludes_generated_native_build_sources(self):
        """A built checkout and a clean worktree should hash the same runtime code."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            root = repo / "hyper_parallel/core/multicore"
            root.mkdir(parents=True)
            (root / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
            before = _python_source_identity(repo)
            for name in ("build", "prebuild"):
                (root / name).mkdir()
                (root / name / "generated.py").write_text("BUILD = True\n", encoding="utf-8")
            after = _python_source_identity(repo)
        self.assertEqual(before, after)
        self.assertEqual(after["file_count"], 1)
        self.assertEqual(after["scope"], "torch_runtime_with_standard_ep_without_generated_build_trees_v3")

    def test_profile_annotations_preserve_call_and_restore_after_error(self):
        """Diagnostic wrappers must preserve arguments and never leak into timing."""
        original = MagicMock(return_value=17)
        target = SimpleNamespace(launch=original)
        record = MagicMock()
        with (
            patch.dict(mega_moe_layer_profile._SPANS, {"execution": ("launch",)}, clear=True),
            patch.object(mega_moe_layer_profile.importlib, "import_module", return_value=target),
            self.assertRaisesRegex(RuntimeError, "diagnostic failed"),
            mega_moe_layer_profile._profile_spans(record),
        ):
            self.assertEqual(target.launch(3, value=4), 17)
            raise RuntimeError("diagnostic failed")
        self.assertIs(target.launch, original)
        original.assert_called_once_with(3, value=4)
        record.assert_called_once_with("megamoe::execution.launch")

    def test_profile_annotations_restore_after_partial_setup(self):
        """An invalid later target must not leave earlier methods instrumented."""
        original = MagicMock()
        target = SimpleNamespace(launch=original)
        with (
            patch.dict(mega_moe_layer_profile._SPANS, {"execution": ("launch", "missing")}, clear=True),
            patch.object(mega_moe_layer_profile.importlib, "import_module", return_value=target),
            self.assertRaises(AttributeError),
            mega_moe_layer_profile._profile_spans(MagicMock()),
        ):
            self.fail("missing profiling target unexpectedly resolved")
        self.assertIs(target.launch, original)

    def test_non_recording_rank_trains_without_constructing_profiler(self):
        """Every rank participates, but only rank zero may create profiler artifacts."""
        sweep = MagicMock()
        sweep.worker._RANK = 1
        route = (object(), object(), object())
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(mega_moe_layer_profile.importlib, "import_module") as import_module:
            mega_moe_layer_profile.profile_layer_step(sweep, route, Path(directory) / "trace.json")
        import_module.assert_not_called()
        self.assertEqual(sweep._train_step.call_count, 2)
        self.assertEqual(sweep.optimizer.step.call_count, 2)

    def test_cycle_profile_rejects_native_and_instrumented_timing(self):
        """Diagnostic cycles must not silently contaminate ordinary measurements."""
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "timing.json")
            with self.assertRaises(SystemExit):
                _parse_args(["--variant", "native", "--output", output, "--profile-kernel-cycles"])
            with patch.dict(os.environ, {"HP_MEGA_MOE_KERNEL_CYCLE_PROFILE": "1"}), \
                    self.assertRaises(SystemExit):
                _parse_args(["--variant", "b1", "--output", output])
            with patch.dict(os.environ, {"HP_MEGA_MOE_KERNEL_CYCLE_PROFILE": "0"}):
                args = _parse_args(["--variant", "b1", "--output", output, "--profile-kernel-cycles"])
        self.assertTrue(args.profile_kernel_cycles)

    def test_cycle_profile_warms_updates_collects_and_restores_environment(self):
        """Enable counters only for the extra post-update step, including nonzero ranks."""
        name = "HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"
        for previous in (None, "0"):
            with self.subTest(previous=previous), patch.dict(os.environ):
                os.environ.pop(name, None)
                if previous is not None:
                    os.environ[name] = previous
                observed = []
                sweep = MagicMock()
                sweep.worker._RANK = 1
                sweep._train_step.side_effect = lambda route, observed=observed: observed.append(os.environ.get(name))
                counters = {"unit": "system_cycle", "per_rank": [{"rank": 0}, {"rank": 1}]}
                forward_counters = {"unit": "system_cycle", "categories": ["up_proj", "down_proj"]}
                sweep.worker._collect_kernel_cycle_profile.side_effect = [counters, forward_counters]
                result = mega_moe_layer_profile.profile_layer_cycles(sweep, (1, 2, 3))
                self.assertEqual(observed, [previous, "1"])
                self.assertEqual(os.environ.get(name), previous)
                self.assertIs(result["counters"], counters)
                self.assertIs(result["forward_counters"], forward_counters)
                self.assertEqual(result["layer_index"], 0)
                self.assertEqual(sweep.optimizer.step.call_count, 2)
                self.assertEqual(sweep.worker._collect_kernel_cycle_profile.call_args_list,
                                 [call(sweep.layers[0]), call(sweep.layers[0], forward=True)])

    def test_cycle_profile_restores_environment_after_failed_collection(self):
        """A diagnostic failure must not leave future kernels instrumented."""
        name = "HP_MEGA_MOE_KERNEL_CYCLE_PROFILE"
        for previous in (None, "0"):
            with self.subTest(previous=previous), patch.dict(os.environ):
                os.environ.pop(name, None)
                if previous is not None:
                    os.environ[name] = previous
                sweep = MagicMock()
                sweep.worker._collect_kernel_cycle_profile.side_effect = RuntimeError("collection failed")
                with self.assertRaisesRegex(RuntimeError, "collection failed"):
                    mega_moe_layer_profile.profile_layer_cycles(sweep, (1, 2, 3))
                self.assertEqual(os.environ.get(name), previous)

    def test_main_collects_cycles_only_after_all_measurements(self):
        """Keep diagnostics outside every route's latency and memory window."""
        events = []
        worker = MagicMock()
        worker._RANK = 0
        worker.dist.get_rank.return_value = 0
        worker.dist.get_world_size.return_value = 8
        worker.dist.is_initialized.return_value = False
        sweep = MagicMock()
        sweep.measure.side_effect = lambda alpha, seed: events.append("measure") or {"seed": seed}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            output = Path(directory) / "timing.json"
            args = argparse.Namespace(variant="b1", layers=16, warmup=3, measured=7, shuffle_seed=1,
                                      output=output, profile_output=None, profile_kernel_cycles=True, profile_p2p=False)
            with patch.object(benchmark, "_parse_args", return_value=args), \
                    patch.object(benchmark, "_route_pairs", return_value=[(1.0, 7), (0.1, 42)]), \
                    patch.object(benchmark, "_source_identity", return_value={}), \
                    patch.object(benchmark.importlib, "import_module", return_value=worker), \
                    patch.object(benchmark, "_LayerSweep", return_value=sweep), \
                    patch.object(benchmark, "profile_layer_p2p",
                                 side_effect=AssertionError("unexpected P2P capture")), \
                    patch.object(benchmark, "profile_layer_cycles",
                                 side_effect=lambda *_args, **_kwargs: events.append("cycles") or {"layer_index": 0}):
                benchmark.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(events, ["measure", "measure", "cycles"])
        self.assertEqual(payload["points"], [{"seed": 7}, {"seed": 42}])
        self.assertEqual(payload["diagnostic_kernel_cycles"], {"alpha": 1.0, "seed": 7, "layer_index": 0})
        self.assertNotIn("diagnostic_p2p", payload)

    def test_p2p_profile_is_opt_in_and_rejects_native(self) -> None:
        """P2P diagnostics must not activate accidentally or accept an unrelated backend."""
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "timing.json")
            self.assertFalse(_parse_args(["--variant", "b1", "--output", output]).profile_p2p)
            self.assertTrue(_parse_args(["--variant", "b1", "--output", output, "--profile-p2p"]).profile_p2p)
            with self.assertRaises(SystemExit):
                _parse_args(["--variant", "native", "--output", output, "--profile-p2p"])

    def test_main_collects_p2p_only_after_all_measurements(self) -> None:
        """P2P timestamps and aggregation must remain outside every timed route."""
        events = []
        worker = MagicMock()
        worker._RANK = 0
        worker.dist.get_rank.return_value = 0
        worker.dist.get_world_size.return_value = 8
        worker.dist.is_initialized.return_value = False
        sweep = MagicMock()
        sweep.measure.side_effect = lambda alpha, seed: events.append("measure") or {"seed": seed}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            output = Path(directory) / "timing.json"
            args = argparse.Namespace(variant="b1", layers=16, warmup=3, measured=7, shuffle_seed=1,
                                      output=output, profile_output=None, profile_kernel_cycles=False, profile_p2p=True)
            with patch.object(benchmark, "_parse_args", return_value=args), \
                    patch.object(benchmark, "_route_pairs", return_value=[(1.0, 7), (0.1, 42)]), \
                    patch.object(benchmark, "_source_identity", return_value={}), \
                    patch.object(benchmark.importlib, "import_module", return_value=worker), \
                    patch.object(benchmark, "_LayerSweep", return_value=sweep), \
                    patch.object(benchmark, "profile_layer_p2p",
                                 side_effect=lambda *_args: events.append("p2p") or {"per_rank": []}):
                benchmark.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(events, ["measure", "measure", "p2p"])
        self.assertEqual(payload["diagnostic_p2p"], {"alpha": 1.0, "seed": 7, "per_rank": []})

    def test_main_records_all_rank_trace_paths_only_after_measurements(self) -> None:
        """All-rank traces are separate diagnostics with a complete per-rank manifest."""
        events = []
        worker = MagicMock()
        worker._RANK = 0
        worker.dist.get_rank.return_value = 0
        worker.dist.get_world_size.return_value = 8
        worker.dist.get_world_size.return_value = 8
        worker.dist.is_initialized.return_value = False
        sweep = MagicMock()
        sweep.measure.side_effect = lambda alpha, seed: events.append("measure") or {"seed": seed}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            output = Path(directory) / "timing.json"
            trace = Path(directory) / "trace.json"
            args = _parse_args(["--variant", "b1", "--output", str(output),
                                "--profile-output", str(trace), "--profile-all-ranks"])
            with patch.object(benchmark, "_parse_args", return_value=args), \
                    patch.object(benchmark, "_route_pairs", return_value=[(0.1, 42), (10000.0, 42)]), \
                    patch.object(benchmark, "_source_identity", return_value={}), \
                    patch.object(benchmark.importlib, "import_module", return_value=worker), \
                    patch.object(benchmark, "_LayerSweep", return_value=sweep), \
                    patch.object(benchmark, "profile_layer_step",
                                 side_effect=lambda *_a, **_k: events.append("trace")) as capture:
                benchmark.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(events, ["measure", "measure", "trace"])
        self.assertTrue(capture.call_args.kwargs["all_ranks"])
        self.assertEqual(capture.call_args.kwargs["context"].base["world_size"], 8)
        self.assertEqual(payload["diagnostic_trace"]["per_rank"], [
            {"rank": rank, "path": str(trace.with_name(f"trace.rank{rank}.json"))} for rank in range(8)
        ])
        self.assertEqual(payload["diagnostic_trace"]["path"], str(trace.with_name("trace.rank0.json")))
