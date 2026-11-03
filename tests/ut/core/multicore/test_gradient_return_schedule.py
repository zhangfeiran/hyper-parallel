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
"""Compile the production gradient traversal with host-only device primitives."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def _definition(source: str, start: str) -> str:
    """Read one brace-delimited production definition without copying its loops."""
    begin = source.index(start)
    opening = source.index('{', begin)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == '{') - (source[cursor] == '}')
        cursor += 1
    return source[begin:cursor]


class TestGradientReturnSchedule(unittest.TestCase):
    """Prove coverage and nonassociative FP32 order using the real native traversal."""

    def test_static_band_and_peer_order_matches_with_and_without_observations(self) -> None:
        """Device primitives are stubs; distributed ST checks readiness and ACK behavior."""
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('The host native traversal test requires g++')
        repository = Path(__file__).resolve().parents[4]
        runtime = repository / 'hyper_parallel/core/multicore/ops/runtime'
        bands = (runtime / 'gradient_band.h').read_text()
        gradient = (runtime / 'replica_gradient.h').read_text()
        worker = (runtime / 'worker_kernel.h').read_text()
        observation = (runtime / 'gradient_band_observer.h').read_text()
        encoding = (runtime / 'replica_trace.h').read_text()
        partition = _definition(bands, 'struct GradientBandPartition') + ';\n'
        partition += _definition(bands, '__aicore__ inline GradientBandPartition PartitionGradientBands')
        partition += '\n' + _definition(gradient, '__aicore__ inline bool AreBandProducersReady')
        partition += '\n' + _definition(encoding, 'enum class ReplicaTraceEvent') + ';\n'
        partition += _definition(encoding, '__aicore__ inline void RecordReplica') + '\n'
        partition += _definition(observation, 'class GradientBandObserver') + ';\n'
        traversal = _definition(worker, '__aicore__ inline void ProcessGradientBands(')
        traversal += '\n' + _definition(worker, '__aicore__ inline void ProcessGradientBandsProfiled')
        header = partition + '\nstruct NativeSchedule {\nint32_t worker_id_, core_num, home_experts_;\n'
        header += traversal + '\n};\n'
        fixture = Path(__file__).with_name('gradient_return_schedule.cpp')
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'native_schedule.h').write_text(header)
            binary = work / 'schedule'
            build = subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(work), str(fixture), '-o', str(binary)],
                                   capture_output=True, text=True, check=False)
            self.assertEqual(build.returncode, 0, f'Compilation failed: {build.stdout}\n{build.stderr}')
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f'Native schedule failed: {result.stdout}\n{result.stderr}')
            self.assertIn('Verified 74 static native schedules', result.stdout)
