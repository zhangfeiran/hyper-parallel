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
"""Preload native stack capture before the unchanged rank CPU-placement worker."""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Mapping


def capture_environment(folder: Path, environment: Mapping[str, str]) -> dict[str, str]:
    """Scope native capture to one child process and its rank artifact directory."""
    library = Path(environment['DSV41_NATIVE_CAPTURE_LIB']).resolve(strict=True)
    folder.mkdir(parents=True, exist_ok=True)
    preload = environment.get('LD_PRELOAD', '')
    return {**environment, 'LD_PRELOAD': str(library) + (':' + preload if preload else ''),
            'DSV41_NATIVE_CAPTURE_LOG': str((folder/'native-stacks.log').resolve())}


def preflight(folder: Path) -> None:
    """Verify real CPU abort interception before spending time installing the runtime."""
    environment = capture_environment(folder, os.environ)
    # Match training's handler order, including the refresh after Python initializes faulthandler.
    code = ('import ctypes, faulthandler, os, resource; '
            'resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); '
            'resource.setrlimit(resource.RLIMIT_FSIZE, (131072, 131072)); '
            'faulthandler.enable(); '
            'library = ctypes.CDLL(os.environ["DSV41_NATIVE_CAPTURE_LIB"]); '
            'result = library.dsv41_capture_install(); '
            'result == 0 or os._exit(190); '
            'os.abort()')
    command = [sys.executable, '-c', code]
    with (folder/'probe.log').open('w') as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=30, check=False)
    path = folder/'native-stacks.log'
    text = path.read_text() if path.exists() else ''
    match = re.search(r'DSV41_NATIVE_STACK_END frames=(\d+)', text)
    frames = int(match.group(1)) if match else 0
    complete = 'DSV41_NATIVE_STACK_BEGIN' in text and frames >= 4
    evidence = {'exit_status': result.returncode, 'expected_signal': signal.SIGABRT,
                'native_stack_complete': complete, 'native_frames': frames,
                'ptrace_required': False, 'npu_initialized': False}
    (folder/'preflight.json').write_text(json.dumps(evidence, indent=2)+'\n')
    if result.returncode != -signal.SIGABRT or not complete:
        raise RuntimeError(f'Native capture preflight failed: {evidence}; inspect {folder}')


def main() -> None:
    """Exec a fresh training interpreter so capture is loaded before Torch and CANN."""
    if sys.argv[1:2] == ['--preflight']:
        preflight(Path(sys.argv[sys.argv.index('--output')+1]))
        return
    rank = int(os.environ['RANK'])
    folder = Path(sys.argv[sys.argv.index('--output')+1])/f'rank{rank}'
    environment = capture_environment(folder, os.environ)
    command = [sys.executable, '-m', 'examples.training_demo.deepseek_v41.deepseek_v41_cpu_worker', *sys.argv[1:]]
    os.execve(sys.executable, command, environment)


if __name__ == '__main__':
    main()
