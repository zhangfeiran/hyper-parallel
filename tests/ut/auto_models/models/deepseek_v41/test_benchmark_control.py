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
"""Keep runtime CPU-control receipts and final benchmark metadata consistent."""

import ast
import inspect
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from examples.training_demo.deepseek_v41 import deepseek_v41_benchmark_control as control_module


class TestBenchmarkControlMetadata(unittest.TestCase):
    """Exercise the real output expression without starting a model or an NPU."""

    def test_result_backend_matches_the_runtime_receipt(self) -> None:
        """The serialized backend must follow the same object as the live receipt."""
        tree = ast.parse(inspect.getsource(benchmark.main))
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'control_backend'
                                  for target in node.targets))
        expression = compile(ast.Expression(assignment.value), '<benchmark-control-metadata>', 'eval')
        with TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '28600'}), patch.object(control_module, 'dist') as dist:
            dist.is_initialized.return_value = True
            dist.get_world_size.return_value = 128
            dist.get_rank.return_value = 0
            directory = Path(temporary)
            control = control_module.BenchmarkControl(directory)
            receipt = json.loads((directory / 'benchmark-control.json').read_text())
            actual = eval(expression, {'diagnostics': SimpleNamespace(control=control)})  # pylint: disable=eval-used
            self.assertEqual(actual, receipt['backend'])
            self.assertEqual(actual, 'tcpstore')
            ordinary = eval(expression, {'diagnostics': SimpleNamespace(control=None)})  # pylint: disable=eval-used
            self.assertEqual(ordinary, 'hccl_world')
            control.close()
