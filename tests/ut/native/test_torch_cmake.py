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
"""Configure real Torch bridge CMake files against hardware-free package layouts."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "CMake and a host C++ compiler are required")
class TestTorchCmake(unittest.TestCase):
    """Public Torch include discovery must accept both wheel and Conda headers."""

    def setUp(self) -> None:
        """Supply inert build inputs; no Torch, NPU library or device is needed."""
        self.root = Path(tempfile.mkdtemp(prefix="hp torch cmake "))
        self.addCleanup(shutil.rmtree, self.root)
        self.packages = self.root / "packages"
        self.repository = Path(__file__).resolve().parents[3]
        self._write(
            "packages/torch/__init__.py",
            "import os\n"
            "if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':\n"
            "    raise RuntimeError('backend autoload must be disabled during discovery')\n"
            "def compiled_with_cxx11_abi():\n"
            "    return int(os.environ['HP_TEST_TORCH_ABI'])\n",
        )
        self._write("packages/torch/utils/__init__.py")
        self._write(
            "packages/torch/utils/cpp_extension.py",
            "import json, os\n"
            "def include_paths():\n"
            "    return json.loads(os.environ['HP_TEST_TORCH_INCLUDE_DIRS'])\n",
        )
        self._write("packages/torch_npu/__init__.py", "raise RuntimeError('do not import the device backend')\n")
        for name in ("c10", "torch", "torch_cpu"):
            self._write(f"packages/torch/lib/lib{name}.so")
        self._write("packages/torch_npu/lib/libtorch_npu.so")
        self._write("packages/torch_npu/include/third_party/op-plugin/op_plugin/include/npu_cpp_extension.h")
        self._write("packages/torch_npu/include/torch_npu/csrc/core/npu/NPUStream.h")
        self._write("vendor/op_api/include/placeholder.h")
        self._write("cann/include/placeholder.h")
        self._write("sdk/shmem/include/placeholder.h")
        self._write("sdk/shmem/lib/libhyper_parallel_shmem.so")
        self._write("sdk/shmem/lib/libhyper_parallel_shmem_utils.so")
        self._write("libaclshmem_symmetric_memory_kernel.so")

    def _write(self, relative: str, content: str = "") -> None:
        """Create one disposable fixture input, including paths containing spaces."""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _configure(self, component: str, includes: list[Path], abi: int) -> tuple[subprocess.CompletedProcess, Path]:
        """Run the repository CMake configuration without compiling fake libraries."""
        source = self.repository / "hyper_parallel/core" / component / "platform/torch"
        build = self.root / f"build {component}"
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": str(self.packages), "ASCEND_HOME_PATH": str(self.root / "cann"),
            "SHMEM_HOME_PATH": str(self.root / "sdk"), "HP_TEST_TORCH_ABI": str(abi),
            "HP_TEST_TORCH_INCLUDE_DIRS": json.dumps([str(path) for path in includes]),
        })
        result = subprocess.run(
            [
                "cmake", "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
                f"-DPython3_EXECUTABLE={sys.executable}", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                f"-DHP_MULTICORE_VENDOR_ROOT={self.root / 'vendor'}", "-DBUILD_TORCH_LIB=ON",
                f"-DACLSHMEM_SYMMETRIC_MEMORY_KERNEL_LIB={self.root / 'libaclshmem_symmetric_memory_kernel.so'}",
            ],
            env=environment, capture_output=True, text=True, check=False, timeout=60,
        )
        return result, build

    def _check_layout(self, *, conda: bool, missing_header: bool = False) -> None:
        """Require usable include roots and actionable failure for an incomplete install."""
        package_include = self.packages / "torch/include"
        package_include.mkdir()
        core = self.root / "conda prefix/include" if conda else package_include
        frontend = core / "torch/csrc/api/include"
        frontend.mkdir(parents=True)
        for path in (core / "ATen/ATen.h", core / "torch/extension.h"):
            self._write(str(path.relative_to(self.root)))
        if not missing_header:
            self._write(str((frontend / "torch/torch.h").relative_to(self.root)))
        reported = [package_include, package_include / "torch/csrc/api/include"]
        if conda:
            reported.extend([core, frontend])
        abi = 0 if conda else 1
        for component in ("multicore", "symmetric_memory"):
            with self.subTest(component=component):
                result, build = self._configure(component, reported, abi)
                diagnostic = result.stdout + result.stderr
                if missing_header:
                    self.assertNotEqual(result.returncode, 0, diagnostic)
                    self.assertIn("Missing Torch C++ headers: torch/torch.h", " ".join(diagnostic.split()))
                    continue
                self.assertEqual(result.returncode, 0, diagnostic)
                commands = json.loads((build / "compile_commands.json").read_text(encoding="utf-8"))
                self.assertTrue(commands, f"No compile commands generated for {component}.")
                for entry in commands:
                    arguments = shlex.split(entry["command"])
                    self.assertIn(f"-I{core}", arguments)
                    self.assertIn(f"-I{frontend}", arguments)
                    self.assertIn(f"-D_GLIBCXX_USE_CXX11_ABI={abi}", arguments)
                    if conda:
                        self.assertNotIn(f"-I{reported[1]}", arguments)

    def test_wheel_include_layout(self) -> None:
        """
        Feature: native Torch build discovery
        Description: configure both bridges with wheel-local headers and C++11 ABI
        Expectation: generated compiler arguments use the public include paths and ABI
        """
        self._check_layout(conda=False)

    def test_conda_include_layout(self) -> None:
        """
        Feature: native Torch build discovery
        Description: public discovery mixes valid prefix headers with a nonexistent wheel frontend
        Expectation: both bridges accept Conda roots, filter missing directories and retain the ABI
        """
        self._check_layout(conda=True)

    def test_missing_frontend_header_is_rejected(self) -> None:
        """
        Feature: native Torch build diagnostics
        Description: include directories exist but the required C++ frontend header is absent
        Expectation: configuration fails with the missing header before starting compilation
        """
        self._check_layout(conda=False, missing_header=True)
