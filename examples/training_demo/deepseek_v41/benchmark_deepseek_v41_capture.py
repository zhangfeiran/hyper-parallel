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
"""Refresh chained fatal handlers after runtime initialization and mark optimizer steps."""

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark


ORIGINAL_STEP = benchmark._step
CAPTURE = None


def write_runtime_snapshot(work: Path) -> None:
    """Retain loaded paths and binary identities after the first warmup step."""
    maps = Path('/proc/self/maps').read_text()
    (work/'runtime-maps.txt').write_text(maps)
    names = ('libtorch_npu.so', 'libgcc_s.so', 'libopapi', 'libnnopbase.so', 'libcust_opapi.so',
             'libascendcl.so', 'libruntime.so', 'libgraph.so', 'libcust_opsproto_rt2.0.so',
             'libcust_opmaster_rt2.0.so', 'liboptiling.so', 'libhyper_parallel', 'omni_training_custom_ops',
             'lightning_indexer', 'sparse_flash_mla')
    paths = sorted({line.split(maxsplit=5)[-1] for line in maps.splitlines()
                    if '/' in line and any(name in line for name in names)})
    libraries = []
    for value in paths:
        path = Path(value)
        try:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            libraries.append({'path': value, 'sha256': digest})
        except OSError as error:
            libraries.append({'path': value, 'error': str(error)})
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'torch-npu', 'omni-training-custom-ops')}
    if importlib.util.find_spec('cann_ops_transformer') is not None:
        versions['cann-ops-transformer'] = importlib.metadata.version('cann-ops-transformer')
    record = {'versions': versions, 'libraries': libraries}
    (work/'runtime-libraries.json').write_text(json.dumps(record, indent=2)+'\n')


def step(trainer: Any, iterator: Any, index: int, work: Path, *args: Any) -> dict[str, Any]:
    """Preserve training while retaining handlers installed later by Torch or CANN."""
    if CAPTURE.dsv41_capture_install() != 0:
        raise RuntimeError(f'Cannot refresh native capture handlers before optimizer step {index+1}')
    CAPTURE.dsv41_capture_set_step(index+1)
    result = ORIGINAL_STEP(trainer, iterator, index, work, *args)
    if index == 0:
        write_runtime_snapshot(work)
    return result


def main() -> None:
    """Enable in-process native backtraces without a profiler or GC changes."""
    global CAPTURE  # pylint: disable=global-statement
    CAPTURE = ctypes.CDLL(os.environ['DSV41_NATIVE_CAPTURE_LIB'])
    CAPTURE.dsv41_capture_install.argtypes = []
    CAPTURE.dsv41_capture_install.restype = ctypes.c_int
    CAPTURE.dsv41_capture_set_step.argtypes = [ctypes.c_long]
    CAPTURE.dsv41_capture_set_step.restype = None
    benchmark._step = step
    try:
        benchmark.main()
    finally:
        benchmark._step = ORIGINAL_STEP


if __name__ == '__main__':
    main()
