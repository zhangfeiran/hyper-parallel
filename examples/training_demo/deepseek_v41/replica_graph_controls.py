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
"""Checksumed graph controls shared by component and full Trainer comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


GRAPH_FILES = tuple('hyper_parallel/core/multicore/modules/mega_moe/' + name
                    for name in ('module.py', 'plan.py', 'function.py', 'workspace.py', 'inactive_tasks.py'))


def install_graph(source: Path, bundle: Path, variant: str,
                  control_source: Path | None = None) -> dict[str, Any]:
    """Change only the node-local clone between completed fresh-process stages."""
    if (source.resolve().is_relative_to(bundle.resolve()) or (source / '.git').exists()
            or control_source is not None and source.resolve().is_relative_to(control_source.resolve())
            or variant not in ('original', 'suffix')):
        raise ValueError('Graph controls require a node-local source clone and a declared variant')
    manifest = json.loads((bundle / 'schedule-controls/manifest.json').read_text())[variant]
    if set(manifest['files']) != set(GRAPH_FILES):
        raise ValueError('Graph source manifest does not match the bounded control file set')
    origin = control_source if variant == 'original' and control_source is not None else (
        bundle / 'schedule-controls' / variant)
    payloads = {relative: (origin / relative).read_bytes() for relative in GRAPH_FILES}
    for relative, raw in payloads.items():
        if hashlib.sha256(raw).hexdigest() != manifest['files'][relative]:
            raise ValueError(f'Graph control checksum mismatch: {variant}/{relative}')
    for relative, raw in payloads.items():
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    return manifest | dict(control_source=str(origin))
