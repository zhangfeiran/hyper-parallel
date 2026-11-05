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
"""Exercise the real preflight checker without importing its NPU-only launcher dependencies."""

import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import hyper_parallel


class TestA5Preflight(unittest.TestCase):
    """Registered Python names cannot replace checks for both CANN entry points."""

    def setUp(self) -> None:
        """Execute the checker definition with CPU-only dependency replacements."""
        root = Path(hyper_parallel.__file__).resolve().parents[1]
        source = root / 'examples/training_demo/deepseek_v41/preflight_mhc_post_a5.py'
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_validate_mhc_post_runtime')
        self.pta = SimpleNamespace(npu_mhc_post=SimpleNamespace(_schemas={'': 'forward'}),
                                   npu_mhc_post_backward=SimpleNamespace(_schemas={'': 'backward'}))
        self.symbols = ('aclnnMhcPost', 'aclnnMhcPostGetWorkspaceSize',
                        'aclnnMhcPostBackward', 'aclnnMhcPostBackwardGetWorkspaceSize')
        self.library = SimpleNamespace(**dict.fromkeys(self.symbols, object()))
        self.loader = Mock(return_value=self.library)
        namespace = {'torch': SimpleNamespace(ops=SimpleNamespace(npu=self.pta)),
                     'ctypes': SimpleNamespace(CDLL=self.loader), 'hashlib': hashlib, 'Path': Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        self.check = namespace['_validate_mhc_post_runtime']

    def test_records_library_identity_and_both_schemas(self) -> None:
        """The evidence contains the actual checked bytes and required symbol set."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'lib64/libopapi.so'
            path.parent.mkdir()
            path.write_bytes(b'qualified operator library')
            result = self.check(root)
        self.assertEqual(result['symbols'], self.symbols)
        self.assertEqual(set(result['pta_schemas']), {'npu_mhc_post', 'npu_mhc_post_backward'})
        self.assertEqual(result['library_sha256'], hashlib.sha256(b'qualified operator library').hexdigest())

    def test_missing_backward_registration_fails_before_library_load(self) -> None:
        """A forward-only Python registration must not pass training preflight."""
        del self.pta.npu_mhc_post_backward
        with self.assertRaisesRegex(RuntimeError, 'Missing A5 MHC post PTA operator.*backward'):
            self.check(Path('/unused'))
        self.loader.assert_not_called()

    def test_missing_each_cann_entry_point_is_reported(self) -> None:
        """The original failure cannot be hidden behind a successfully imported Python package."""
        for missing in self.symbols:
            with self.subTest(symbol=missing):
                available = {name: object() for name in self.symbols if name != missing}
                self.loader.return_value = SimpleNamespace(**available)
                with self.assertRaisesRegex(RuntimeError, f'Missing A5 MHC post CANN APIs.*{missing}'):
                    self.check(Path('/unused'))

    def test_unloadable_library_preserves_the_loader_error(self) -> None:
        """The selected CANN path remains visible when its library cannot be loaded."""
        self.loader.side_effect = OSError('Missing runtime dependency')
        with self.assertRaisesRegex(RuntimeError, 'Cannot load the selected CANN operator API library') as raised:
            self.check(Path('/unused'))
        self.assertIsInstance(raised.exception.__cause__, OSError)
