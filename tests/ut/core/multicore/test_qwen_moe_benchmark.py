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
"""CPU tests for process-group ownership in the standalone NPU benchmark."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class TestQwenBenchmarkRuntime(unittest.TestCase):
    """Run the actual entry functions with device and workload dependencies mocked."""

    def test_process_group_ownership(self) -> None:
        """Only release self-created groups on validation failure or normal exit."""
        path = Path(__file__).resolve().parents[4] / (
            "hyper_parallel/core/multicore/examples/mega_moe/qwen_moe_benchmark.py"
        )
        # Extract entry functions to avoid importing the standalone NPU model stack.
        tree = ast.parse("from __future__ import annotations\n")
        tree.body.extend(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                         if isinstance(node, ast.FunctionDef) and node.name in ("_init_runtime", "main"))
        code = compile(tree, str(path), "exec")
        for existing in (False, True):
            for valid_world in (False, True):
                with self.subTest(existing=existing, valid_world=valid_world):
                    dist = Mock()
                    dist.is_initialized.side_effect = [existing, True]
                    dist.get_world_size.return_value = 8 if valid_world else 1
                    namespace = {name: Mock() for name in (
                        "torch", "multicore", "parse_args", "replace", "QwenMoeConfig",
                        "_build_backend_model", "_build_workload", "_measure_backend", "_write_result",
                    )}
                    namespace.update(
                        os=SimpleNamespace(environ={}), sys=sys, dist=dist, _WORLD_SIZE=8,
                        _build_batch=Mock(return_value=(None, None)),
                        _compare_first_step_accuracy=Mock(return_value=(
                            {}, {"common": ({}, 0), "mega_moe": ({}, 0)},
                        )),
                    )
                    exec(code, namespace)  # pylint: disable=exec-used
                    if valid_world:
                        self.assertEqual(namespace["main"]([]), 0)
                        namespace["multicore"].shutdown.assert_called_once_with()
                    else:
                        with self.assertRaisesRegex(ValueError, "requires 8 ranks, got 1"):
                            namespace["main"]([])
                        namespace["multicore"].shutdown.assert_not_called()
                    self.assertEqual(dist.init_process_group.call_count, int(not existing))
                    self.assertEqual(dist.destroy_process_group.call_count, int(not existing))
