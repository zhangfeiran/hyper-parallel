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
"""前向和反向共用的密封原生产物加载与命名空间所有权。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from hyper_parallel.core.multicore._build.build_dense_tile import verify_dense_tile_payload

_LOADED: dict[str, str] = {}


def load_dense_payload(manifest: Path, identity: dict[str, object]) -> tuple[dict[str, object], object]:
    """先验证真实源码、SDK 和产物字节，再加载同一进程内的原生算子。

    Args:
        manifest: 调用方缓存中的密封产物清单。
        identity: 根据实际源码和工具链重新计算的身份。
    """
    data = verify_dense_tile_payload(manifest, identity)
    library = (manifest.parent / data["library"]).resolve()
    fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
    namespace = data["namespace"]
    if namespace in _LOADED and _LOADED[namespace] != fingerprint:
        raise ValueError("A different resident dense payload owns this operator namespace")
    if (namespace not in _LOADED and getattr(getattr(torch.ops, namespace), "launch", None) is not None
            and str(library) not in torch.ops.loaded_libraries):
        raise ValueError("An unowned resident dense operator namespace is already registered")
    torch.ops.load_library(str(library))
    _LOADED[namespace] = fingerprint
    return data, getattr(torch.ops, namespace)
