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
"""Exercise fatal-handler composition in isolated CPU interpreters."""

import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from examples.training_demo.deepseek_v41.deepseek_v41_capture_worker import preflight


class TestNativeStackCapture(unittest.TestCase):
    """Native capture must retain a backtrace and terminate with the fatal signal."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile the diagnostic library without loading Torch or initializing devices."""
        compiler = shutil.which('cc')
        if sys.platform != 'linux' or compiler is None:
            raise unittest.SkipTest('Native capture requires Linux and a C compiler')
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.library = cls.root/'libdsv41_capture.so'
        source = Path(__file__).resolve().parents[5]/'examples/training_demo/deepseek_v41/native_stack_capture.c'
        subprocess.run([compiler, '-std=c11', '-shared', '-fPIC', '-O2', '-g', '-fno-omit-frame-pointer',
                        '-Wall', '-Wextra', '-Werror', str(source), '-ldl', '-o', str(cls.library)],
                       check=True, timeout=30, capture_output=True)

    def run_probe(self, trigger: str, expected_signal: int, *, refresh: bool = True,
                  faulthandler: bool = True, threaded: bool = False, reenable: bool = False) -> str:
        """Check a child without allowing recursive fatal logs or core dumps to grow."""
        with TemporaryDirectory(dir=self.root) as directory:
            folder = Path(directory)
            env = dict(os.environ, LD_PRELOAD=str(self.library),
                       DSV41_NATIVE_CAPTURE_LOG=str(folder/'native.log'))
            env.pop('PYTHONFAULTHANDLER', None)
            if faulthandler:
                env['PYTHONFAULTHANDLER'] = '1'
            code = (
                'import ctypes,faulthandler,os,resource,signal,threading; '
                'resource.setrlimit(resource.RLIMIT_CORE,(0,0)); '
                'resource.setrlimit(resource.RLIMIT_FSIZE,(131072,131072)); '
                f'lib=ctypes.CDLL({str(self.library)!r}); '
                'lib.__mulvdi3.argtypes=[ctypes.c_int64,ctypes.c_int64]; '
                'lib.__mulvdi3.restype=ctypes.c_int64; '
                'lib.dsv41_capture_set_step(87); '
            )
            if reenable:
                code += 'faulthandler.disable(); faulthandler.enable(); '
            if refresh:
                code += '[lib.dsv41_capture_install() for _ in range(3)]; '
            code += f'threading.Thread(target=lambda: {trigger}).start()' if threaded else trigger
            with (folder/'stderr.log').open('w') as stream:
                result = subprocess.run([sys.executable, '-c', code], env=env, stdout=subprocess.DEVNULL,
                                        stderr=stream, timeout=5, check=False)
            native = (folder/'native.log').read_text()
            stderr = (folder/'stderr.log').read_text()
            self.assertEqual(result.returncode, -expected_signal)
            self.assertEqual(native.count('DSV41_NATIVE_STACK_BEGIN'), 1)
            self.assertEqual(native.count('DSV41_NATIVE_STACK_END'), 1)
            self.assertIn('optimizer_step=87', native)
            self.assertLessEqual(stderr.count('Fatal Python error:'), 1)
            return native

    def test_refreshed_handler_with_python_faulthandler(self) -> None:
        """Match the production constructor, Python initialization and step refresh order."""
        self.run_probe('os.abort()', signal.SIGABRT)

    def test_fatal_signals_and_worker_threads(self) -> None:
        """Preserve signals for both abort interception and direct fatal delivery."""
        for trigger, signum in [('os.abort()', signal.SIGABRT),
                                ('signal.raise_signal(signal.SIGABRT)', signal.SIGABRT),
                                ('ctypes.string_at(0)', signal.SIGSEGV)]:
            for threaded in (False, True):
                with self.subTest(trigger=trigger, threaded=threaded):
                    self.run_probe(trigger, signum, threaded=threaded)

    def test_original_handler_orders(self) -> None:
        """Keep preflight and non-faulthandler abort paths working after the repair."""
        for refresh, faulthandler in [(False, True), (True, False), (False, False)]:
            with self.subTest(refresh=refresh, faulthandler=faulthandler):
                self.run_probe('os.abort()', signal.SIGABRT, refresh=refresh, faulthandler=faulthandler)

    def test_reenabled_python_faulthandler(self) -> None:
        """Handle a later runtime replacing Python's fatal handlers before a refresh."""
        self.run_probe('os.abort()', signal.SIGABRT, reenable=True)

    def test_integer_overflow_retains_operands(self) -> None:
        """Preserve checked multiplication semantics while recording the failing CANN operands."""
        for left, right in [(1 << 62, 4), (-(1 << 63), -1)]:
            with self.subTest(left=left, right=right):
                native = self.run_probe(f'lib.__mulvdi3({left},{right})', signal.SIGABRT)
                self.assertIn(f'operation=__mulvdi3 lhs={left} rhs={right}', native)
        library = ctypes.CDLL(str(self.library))
        multiply = getattr(library, '__mulvdi3')
        multiply.argtypes = [ctypes.c_int64, ctypes.c_int64]
        multiply.restype = ctypes.c_int64
        for left, right in [(0, -(1 << 63)), (-(1 << 63), 1), ((1 << 63)-1, 1), (-7, 9)]:
            with self.subTest(left=left, right=right):
                self.assertEqual(multiply(left, right), left*right)

    def test_production_preflight_refreshes_handlers(self) -> None:
        """Check the actual node startup gate with Python fatal handling enabled."""
        with TemporaryDirectory(dir=self.root) as directory:
            folder = Path(directory)
            with patch.dict(os.environ, {'DSV41_NATIVE_CAPTURE_LIB': str(self.library)}):
                preflight(folder)
            result = json.loads((folder/'preflight.json').read_text())
            self.assertEqual(result['exit_status'], -signal.SIGABRT)
            self.assertTrue(result['native_stack_complete'])
            self.assertLessEqual((folder/'probe.log').read_text().count('Fatal Python error:'), 1)
