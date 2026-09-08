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
"""CPU contracts for opt-in all-rank layer traces."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from examples.torch.multicore import mega_moe_layer_profile as profiling


class TestMegaMoeLayerProfile(unittest.TestCase):
    """Keep trace capture rank-local and outside all measured work."""

    def test_trace_paths_preserve_default_and_isolate_all_ranks(self) -> None:
        """Default capture keeps its exact filename; all-rank capture adds rank suffixes."""
        output = Path("results/trace.json")
        self.assertEqual(profiling.profile_trace_paths(output, 3), (output,))
        self.assertEqual(profiling.profile_trace_paths(output, 3, all_ranks=True), tuple(
            Path(f"results/trace.rank{rank}.json") for rank in range(3)
        ))
        for world_size in (0, -1):
            with self.subTest(world_size=world_size), self.assertRaisesRegex(ValueError, "positive"):
                profiling.profile_trace_paths(output, world_size, all_ranks=True)
        with self.assertRaisesRegex(ValueError, "json"):
            profiling.profile_trace_paths(Path("trace.txt"), 2, all_ranks=True)

    def test_all_ranks_capture_and_default_rank_zero_export_unique_files(self) -> None:
        """Each selected rank owns one profiler, two updates and its own output filename."""
        for rank, all_ranks in ((0, False), (0, True), (1, True)):
            with self.subTest(rank=rank, all_ranks=all_ranks), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "trace.json"
                expected = output.with_name(f"trace.rank{rank}.json") if all_ranks else output
                sweep = MagicMock()
                sweep.worker._RANK = rank
                sweep.worker.dist.get_world_size.return_value = 2
                profiler = MagicMock()

                def export(path: str) -> None:
                    """Materialize only the profiler's assigned artifact."""
                    Path(path).write_text("[]\n", encoding="utf-8")

                profiler.profile.return_value.export_chrome_trace.side_effect = export
                with patch.object(profiling.importlib, "import_module", return_value=profiler) as load, \
                        patch.object(profiling, "_profile_spans", return_value=nullcontext()):
                    profiling.profile_layer_step(sweep, (1, 2, 3), output, all_ranks=all_ranks)

                load.assert_called_once_with("torch_npu.profiler")
                profiler.profile.assert_called_once()
                profiler.profile.return_value.export_chrome_trace.assert_called_once_with(str(expected))
                self.assertEqual(profiler.tensorboard_trace_handler.call_args.kwargs["worker_name"], expected.stem)
                self.assertEqual(sorted(Path(directory).iterdir()), [expected])
                self.assertEqual(sweep._train_step.call_count, 2)
                self.assertEqual(sweep.optimizer.step.call_count, 2)

    def test_existing_foreign_rank_output_is_rejected_before_work(self) -> None:
        """Every rank preflights all destinations to catch a shared-filesystem collision early."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            existing = output.with_name("trace.rank0.json")
            existing.write_text("preserve\n", encoding="utf-8")
            sweep = MagicMock()
            sweep.worker._RANK = 1
            sweep.worker.dist.get_world_size.return_value = 2
            with self.assertRaisesRegex(FileExistsError, "rank0"), \
                    patch.object(profiling.importlib, "import_module") as load:
                profiling.profile_layer_step(sweep, (1, 2, 3), output, all_ranks=True)
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve\n")
            load.assert_not_called()
            sweep._train_step.assert_not_called()

    def test_failed_all_rank_capture_restores_wrapped_functions(self) -> None:
        """A diagnostic-step exception must unwind annotation patches and the profiler context."""
        with tempfile.TemporaryDirectory() as directory:
            sweep = MagicMock()
            sweep.worker._RANK = 1
            sweep.worker.dist.get_world_size.return_value = 2
            sweep._train_step.side_effect = (None, RuntimeError("capture failed"))
            profiler = MagicMock()
            original = MagicMock()
            target = SimpleNamespace(launch=original)

            def load(name: str) -> object:
                """Resolve the optional profiler and one isolated annotation target."""
                return profiler if name == "torch_npu.profiler" else target

            with patch.dict(profiling._SPANS, {"execution": ("launch",)}, clear=True), \
                    patch.object(profiling.importlib, "import_module", side_effect=load), \
                    self.assertRaisesRegex(RuntimeError, "capture failed"):
                profiling.profile_layer_step(sweep, (1, 2, 3), Path(directory) / "trace.json", all_ranks=True)

            self.assertIs(target.launch, original)
            self.assertIs(profiler.profile.return_value.__exit__.call_args.args[0], RuntimeError)
            profiler.profile.return_value.export_chrome_trace.assert_not_called()

    def test_export_rejects_truncated_json_even_when_vendor_does_not_raise(self) -> None:
        """A CANN exporter failure must not turn a partial file into successful diagnostic evidence."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            capture = MagicMock()
            capture.export_chrome_trace.side_effect = (
                lambda path: Path(path).write_text('[{"ph": "X"}', encoding="utf-8")
            )
            with self.assertRaisesRegex(RuntimeError, "Incomplete NPU trace export"):
                profiling._export_trace(capture, output)
