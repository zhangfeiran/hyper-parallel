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
"""Internal PyTorch adapter for the native MoE-FFN operators.

The adapter registers the out-of-tree operators in the ``hyper_parallel``
PyTorch namespace without publishing their positional ABI from
``hyper_parallel.core.multicore``.
"""

__all__ = []

import importlib
from functools import lru_cache
from typing import Any

from hyper_parallel.core.multicore._loader import (
    NativeComponentUnavailableError,
    get_multicore_paths,
    preload_vendor_library,
)


@lru_cache(maxsize=1)
def _get_torch() -> Any:
    """Load Torch and register the ABI-specific multicore adapter once."""
    vendor_root, adapter_path = get_multicore_paths("torch")
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    preload_vendor_library(vendor_root)
    try:
        torch.ops.load_library(str(adapter_path))
    except (OSError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] "
            "component=multicore framework=torch "
            f"library={adapter_path} error={error}. "
            "Check the Python/Torch/torch_npu/CANN version combination and "
            "rebuild the Torch adapter."
        ) from error
    return torch


def run_mega_moe(*args: Any) -> None:
    """Invoke the internal Torch forward operator with its positional ABI."""
    _get_torch().ops.hyper_parallel.mega_moe(*args)


def run_mega_moe_grad(*args: Any) -> None:
    """Invoke the internal Torch backward operator with its positional ABI."""
    _get_torch().ops.hyper_parallel.mega_moe_grad(*args)
