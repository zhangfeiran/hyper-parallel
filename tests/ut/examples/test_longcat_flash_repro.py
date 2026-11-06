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
"""CPU checks for portable LongCat reproduction publication and device assignment."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import yaml

from tests.common.mark_utils import arg_mark

REPO = Path(__file__).resolve().parents[3]
REPRO = REPO / "examples/longcat_flash_repro"


def load_module(name: str, filename: str) -> ModuleType:
    """Load a standalone entrypoint without importing accelerator frameworks.

    Args:
        name: Isolated module name.
        filename: Entry file under the reproduction directory.
    """
    spec = importlib.util.spec_from_file_location(name, REPRO / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLongcatReproduction(unittest.TestCase):
    """Exercise source, device, relocation and publication guards on CPU."""

    def setUp(self) -> None:
        """Create an automatically cleaned CPU-only fixture tree."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.launch = load_module("longcat_repro_launch", "launch.py")
        self.audit = load_module("longcat_repro_publication", "audit_publication.py")

    def make_repository(self) -> tuple[Path, str]:
        """Create a minimal model baseline and a later independent commit."""
        repo = self.root / "source checkout"
        repo.mkdir()
        self.git(repo, "init", "--quiet")
        (repo / "model.py").write_text("MODEL = 1\n")
        self.git(repo, "add", "model.py")
        self.git(repo, "commit", "--quiet", "-m", "test: baseline model")
        baseline = self.git(repo, "rev-parse", "HEAD").strip()
        (repo / "README.md").write_text("Reproduction material\n")
        self.git(repo, "add", "README.md")
        self.git(repo, "commit", "--quiet", "-m", "test: add reproduction material")
        return repo, baseline

    @staticmethod
    def git(repo: Path, *args: str) -> str:
        """Execute isolated local Git commands without global configuration."""
        command = ["git", "-c", "user.name=Reproduction Test", "-c", "user.email=test@example.invalid", *args]
        return subprocess.check_output(command, cwd=repo, text=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_committed_tree_has_no_publication_findings(self) -> None:
        """Feature: Portable LongCat reproduction.

        Description: The complete checked-in example must pass the publication scan.
        Expectation: Publication and source guards retain the declared behavior without device execution.
        """
        self.assertEqual(self.audit.audit(REPRO), [])
        self.launch.check_files()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_baseline_descendant_keeps_model_hash_guard(self) -> None:
        """Feature: Portable LongCat reproduction.

        Description: A reproduction commit is accepted while model edits remain rejected.
        Expectation: Publication and source guards retain the declared behavior without device execution.
        """
        repo, baseline = self.make_repository()
        manifest = {
            "required_commit": baseline,
            "source_sha256": {"model.py": hashlib.sha256((repo / "model.py").read_bytes()).hexdigest()},
        }
        self.launch.check_source(repo, manifest)
        (repo / "model.py").write_text("MODEL = 2\n")
        with self.assertRaisesRegex(RuntimeError, "Source files differ"):
            self.launch.check_source(repo, manifest)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_device_assignment_is_explicit_and_unique(self) -> None:
        """Feature: Portable LongCat reproduction.

        Description: Device requirements cannot silently select this host's cards.
        Expectation: Publication and source guards retain the declared behavior without device execution.
        """
        valid = ", ".join(str(index * 3) for index in range(4))
        expected = ",".join(str(index * 3) for index in range(4))
        self.assertEqual(self.launch.validate_devices(valid, 4, "component"), expected)
        for invalid in (None, "", "1,1,2,3", "-1,2,3,4", "1,2,3", "1,2,3,a"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.launch.validate_devices(invalid, 4, "component")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_paths_and_devices_render_without_npu_execution(self) -> None:
        """Feature: Portable LongCat reproduction.

        Description: Spaces and quotes survive Python, YAML and shell rendering.
        Expectation: Publication and source guards retain the declared behavior without device execution.
        """
        repo, _ = self.make_repository()
        native = repo / "native"
        native.mkdir()
        (native / "receipt.json").write_text("{}")
        output = self.root / "replay space's"
        component = ",".join(str(index * 2) for index in range(4))
        trainer = ",".join(str(index * 2) for index in range(16))
        args = argparse.Namespace(
            out=output, phase="all", prepare_only=True, component_devices=component, trainer_devices=trainer
        )
        source_copy = self.root / "reproduction sources"
        shutil.copytree(REPRO / "templates", source_copy / "templates", ignore=shutil.ignore_patterns("__pycache__"))
        cache = source_copy / "templates/__pycache__"
        cache.mkdir()
        (cache / "unrelated.pyc").write_bytes(bytes([255, 0, 128]))
        (source_copy / "templates/not-a-template.pyc").write_bytes(bytes([255, 0, 128]))
        with patch.object(self.launch, "ROOT", source_copy):
            self.launch.prepare(args, repo, native)
        self.assertFalse((output / "not-a-template.pyc").exists())
        self.assertFalse((output / "__pycache__/unrelated.pyc").exists())
        receipt = json.loads((output / "portable-launch.json").read_text())
        self.assertEqual(receipt["component_devices"], component)
        self.assertEqual(receipt["trainer_devices"], trainer)
        self.assertIn(f"ASCEND_RT_VISIBLE_DEVICES={component}", (output / "run_trial.sh").read_text())
        self.assertIn(f"ASCEND_RT_VISIBLE_DEVICES={trainer}", (output / "run_formal_trainer.sh").read_text())
        for script in output.glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)
        config = yaml.safe_load((output / "trainer/initialize.yaml").read_text())
        self.assertEqual(config["checkpoint"]["checkpoint_dir"], str(output / "warmstart"))
        self.assertEqual(config["model"]["config"]["config_dict"]["n_routed_experts"], 128)
        sentinel = output / "preserve.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.launch.prepare(args, repo, native)
        self.assertEqual(sentinel.read_text(), "keep")
        args.out = repo / "runtime-reports"
        with self.assertRaisesRegex(RuntimeError, "outside the source"):
            self.launch.prepare(args, repo, native)
        self.assertFalse(args.out.exists())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_publication_audit_rejects_identity_without_echoing_value(self) -> None:
        """Feature: Portable LongCat reproduction.

        Description: A private reference path or concrete process identity is rejected.
        Expectation: Publication and source guards retain the declared behavior without device execution.
        """
        references = self.root / "reference"
        references.mkdir()
        private_path = "/" + "home" + "/example-user/experiment"
        path = references / "identity.json"
        path.write_text(json.dumps({"source": private_path, "pid": 12345}))
        findings = self.audit.audit(self.root)
        self.assertEqual(
            {entry["reason"] for entry in findings}, {"workstation_home_or_cache_path", "concrete_identity"}
        )
        self.assertNotIn(private_path, json.dumps(findings))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_branch_contains_verified_sources_and_data(self) -> None:
        """Feature: Direct Git branch delivery.

        Description: Verify committed source/data checksums without producing an archive.
        Expectation: Model inputs are present and generated artifacts have no delivery entrypoint.
        """
        subprocess.run([sys.executable, str(REPRO / "checksums.py")], check=True)
        tracked = self.audit.source_files(REPRO)
        self.assertIn(REPRO / "assets/train.jsonl", tracked)
        self.assertTrue((REPRO / "templates/trainer/initialize.yaml").is_file())
        self.assertFalse((REPRO / "package.py").exists())
        self.assertFalse((REPRO / "optional-wheels.json").exists())
        self.assertFalse(any("runtime-deps" in path.parts or "wheels" in path.parts for path in tracked))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_supported_host_architectures_match_native_elf(self) -> None:
        """Feature: Host architecture portability.

        Description: Validate x86 and ARM host-library headers without accelerator execution.
        Expectation: Matching ELF64 libraries pass and cross-architecture libraries fail.
        """
        self.assertEqual(self.launch.host_architecture("AMD64"), "x86_64")
        self.assertEqual(self.launch.host_architecture("arm64"), "aarch64")
        with self.assertRaisesRegex(RuntimeError, "Unsupported host"):
            self.launch.host_architecture("riscv64")
        for machine, architecture, other in ((62, "x86_64", "aarch64"), (183, "aarch64", "x86_64")):
            with self.subTest(architecture=architecture):
                header = bytearray(64)
                header[:4] = b"\x7fELF"
                header[4:6] = bytes((2, 1))
                header[18:20] = machine.to_bytes(2, "little")
                library = self.root / f"{architecture}.so"
                library.write_bytes(header)
                self.launch.verify_elf_architecture(library, architecture)
                with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
                    self.launch.verify_elf_architecture(library, other)
        invalid = self.root / "invalid.so"
        invalid.write_bytes(b"not an ELF library")
        with self.assertRaisesRegex(RuntimeError, "Expected an ELF64"):
            self.launch.verify_elf_architecture(invalid, "x86_64")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_trainer_inventory_rejects_an_eight_device_host(self) -> None:
        """Feature: Fixed-topology performance replay.

        Description: Inspect synthetic device inventories before creating process groups.
        Expectation: Sixteen-device replay passes only on a sufficient inventory; four-device components remain usable.
        """
        args = argparse.Namespace(
            component_devices=",".join(map(str, range(4))), trainer_devices=",".join(map(str, range(16)))
        )
        inventory = "\n".join(f"{index // 2} {index % 2} {index} {index} Ascend910C" for index in range(16))
        with patch.object(self.launch.subprocess, "check_output", return_value=inventory):
            self.launch.check_device_inventory(args)
        smaller = "\n".join(inventory.splitlines()[:8])
        with patch.object(self.launch.subprocess, "check_output", return_value=smaller):
            with self.assertRaisesRegex(RuntimeError, "requires 16"):
                self.launch.check_device_inventory(args)
            args.trainer_devices = None
            self.launch.check_device_inventory(args)
