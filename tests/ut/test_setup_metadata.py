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
"""Protect installed editable metadata during pip's temporary metadata generation."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tests.common.mark_utils import arg_mark


class TestSetupMetadata(unittest.TestCase):
    """Exercise real metadata cleanup inside an isolated temporary source tree."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_temporary_egg_base_preserves_installed_metadata(self) -> None:
        """Feature: Repeated editable installation.
        Description: Generate temporary egg metadata while root metadata represents the installed distribution.
        Expectation: Only the selected egg-info target is replaced and installed metadata remains readable.
        """
        setup_path = Path(__file__).resolve().parents[2] / "setup.py"
        spec = importlib.util.spec_from_file_location("hyper_setup_under_test", setup_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            installed = root / "hyper_parallel.egg-info"
            installed.mkdir()
            metadata = "Name: hyper_parallel\nVersion: 1.2.0.dev0\n"
            (installed / "PKG-INFO").write_text(metadata)
            target = root / "pip-egg-base" / "hyper_parallel.egg-info"
            target.mkdir(parents=True)
            (target / "obsolete").write_text("stale metadata")
            command = module.EggInfo(module.Distribution())
            command.egg_info = str(target)

            def regenerate() -> None:
                """Represent setuptools writing the selected target, without repository discovery."""
                target.mkdir(parents=True, exist_ok=True)
                (target / "PKG-INFO").write_text(metadata)

            with mock.patch.object(module, "__file__", str(root / "setup.py")), \
                    mock.patch.object(module.egg_info, "run", side_effect=regenerate):
                command.run()
            self.assertEqual((installed / "PKG-INFO").read_text(), metadata)
            self.assertEqual((target / "PKG-INFO").read_text(), metadata)
            self.assertFalse((target / "obsolete").exists())
