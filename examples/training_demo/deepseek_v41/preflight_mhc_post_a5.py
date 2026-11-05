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
"""Check Ascend950 MHC post registrations and CANN symbols without allocating an NPU."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import torch
import torch_npu


def _validate_mhc_post_runtime(cann: Path) -> dict[str, object]:
    """Resolve the A5 forward and backward APIs before allocating training devices."""
    schemas = {}
    for name in ('npu_mhc_post', 'npu_mhc_post_backward'):
        operation = getattr(torch.ops.npu, name, None)
        if operation is None:
            raise RuntimeError(f'Missing A5 MHC post PTA operator: npu::{name}')
        schemas[name] = str(operation._schemas)
    path = (cann / 'lib64/libopapi.so').resolve()
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise RuntimeError(f'Cannot load the selected CANN operator API library: {path}') from error
    symbols = ('aclnnMhcPost', 'aclnnMhcPostGetWorkspaceSize',
               'aclnnMhcPostBackward', 'aclnnMhcPostBackwardGetWorkspaceSize')
    missing = [name for name in symbols if not hasattr(library, name)]
    if missing:
        raise RuntimeError(f'Missing A5 MHC post CANN APIs in {path}: {missing}')
    return {'pta_schemas': schemas, 'library': str(path), 'symbols': symbols,
            'library_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    """Record a cold API check separately from numerical or training qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cann-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch_npu.npu.is_initialized():
        raise RuntimeError('MHC preflight must start before NPU initialization')
    result = _validate_mhc_post_runtime(args.cann_root.resolve())
    if torch_npu.npu.is_initialized():
        raise RuntimeError('MHC preflight unexpectedly initialized an NPU')
    record = dict(status='passed', scope='PTA and CANN API availability only; no NPU execution',
                  torch=torch.__version__, torch_npu=torch_npu.__version__, npu_initialized=False,
                  mhc_post_runtime=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
