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
"""Keep unrelated optional kernels out of the public module import path."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestModuleExports(unittest.TestCase):
    """Test the package namespace without importing any framework or native extension."""

    def setUp(self) -> None:
        """Load an isolated namespace so cached exports cannot leak between tests."""
        path = Path(__file__).resolve().parents[4] / 'hyper_parallel/components/modules/__init__.py'
        spec = importlib.util.spec_from_file_location('_module_exports_contract', path)
        self.exports = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.exports)

    def test_export_is_imported_on_demand_and_cached(self) -> None:
        """Requesting RMSNorm must not touch any DeepSeek-only dependency."""
        expected = object()
        with patch.object(self.exports.importlib, 'import_module',
                          return_value=SimpleNamespace(RMSNorm=expected)) as load:
            self.assertIs(self.exports.RMSNorm, expected)
            self.assertIs(self.exports.RMSNorm, expected)
        load.assert_called_once_with('_module_exports_contract.rms_norm')
        self.assertIn('RMSNorm', dir(self.exports))
        self.assertEqual(set(self.exports.__all__), set(self.exports._EXPORT_TO_MODULE))

    def test_optional_dependency_errors_are_not_hidden_or_cached(self) -> None:
        """A caller explicitly requesting an unavailable optional module still receives its error."""
        error = ModuleNotFoundError('missing custom operators', name='omni_training_custom_ops')
        with patch.object(self.exports.importlib, 'import_module', side_effect=error) as load:
            for _ in range(2):
                with self.assertRaises(ModuleNotFoundError) as caught:
                    _ = self.exports.DSAAttention
                self.assertIs(caught.exception, error)
            self.assertEqual(load.call_count, 2)
        self.assertNotIn('DSAAttention', self.exports.__dict__)

    def test_unknown_export_does_not_import_anything(self) -> None:
        """Typos remain ordinary AttributeError, not unrelated backend import failures."""
        with patch.object(self.exports.importlib, 'import_module') as load, self.assertRaises(AttributeError):
            _ = self.exports.unknown_export
        load.assert_not_called()
