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
"""Compile the real framework-independent C++ configuration parser on the CPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestNativeUdmaConfiguration(unittest.TestCase):
    """Keep native admission consistent with Python without loading CANN."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the production parser once with an ordinary host compiler."""
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('A host C++ compiler is required for the configuration contract tests')
        temporary = Path(tempfile.mkdtemp(prefix='shmem-config-'))
        cls.addClassCleanup(shutil.rmtree, temporary)
        cls.probe = temporary / 'config-probe'
        root = Path(__file__).resolve().parents[5]
        source = root / 'hyper_parallel/core/multicore/shmem/ccsrc'
        subprocess.run([compiler, '-std=c++17', '-I', str(source), str(Path(__file__).with_name('config_probe.cpp')),
                        str(source / 'runtime/config.cpp'), '-o', str(cls.probe)],
                       check=True, capture_output=True, text=True)

    def run_probe(self, values: dict[str, str]) -> subprocess.CompletedProcess:
        """Run one isolated configuration without inheriting active SHMEM settings."""
        environment = {key: value for key, value in os.environ.items() if not key.startswith('HYPER_PARALLEL_SHMEM_')}
        return subprocess.run([str(self.probe)], env={**environment, **values},
                              check=False, capture_output=True, text=True)

    def test_default_and_explicit_pools(self) -> None:
        """Preserve default32 and admit both qualified hybrid pool sizes."""
        self.assertEqual(self.run_probe({}).stdout.strip(), '32')
        for count in (1, 8, 16, 32):
            with self.subTest(count=count):
                result = self.run_probe({'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                         'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': str(count)})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), str(count))

    def test_invalid_pools_are_rejected(self) -> None:
        """Reject malformed settings in native code even without the Python consumer."""
        for value in ('', '0', '2', '17', ' 16', '16 ', '+16', '16.0'):
            with self.subTest(value=value):
                result = self.run_probe({'HYPER_PARALLEL_SHMEM_DATA_ENGINE': 'mte_udma',
                                         'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': value})
                self.assertEqual(result.returncode, 2)
                self.assertIn('1, 8, 16 or 32', result.stderr)

    def test_reduced_pool_requires_hybrid_engine(self) -> None:
        """Do not silently ignore a resource request on the MTE-only path."""
        result = self.run_probe({'HYPER_PARALLEL_SHMEM_UDMA_QP_NUM': '16'})
        self.assertEqual(result.returncode, 2)
        self.assertIn('mte_udma', result.stderr)


if __name__ == '__main__':
    unittest.main()
