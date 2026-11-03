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
"""Restore only LongCat's deterministic non-persistent rotary buffers."""

from typing import Any

from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts


def _share_runtime(model: Any, _args: Any) -> None:
    """Share compatible native storage after EP binding and before the first layer.

    Args:
        model: Complete model whose state is managed.
        _args: Declared  args value.
    """
    if getattr(model, "_longcat_resources_shared", False):
        return
    executors = [module._executor for module in model.modules()  # pylint: disable=protected-access
                 if isinstance(module, LongcatFlashMegaMoeExperts) and module._executor is not None]
    if executors:
        from hyper_parallel.core.multicore import MegaMoeExperts  # pylint: disable=C0415

        MegaMoeExperts.share_execution_resources(executors)
    model._longcat_resources_shared = True  # pylint: disable=protected-access


def register_materialization(model: Any) -> None:
    """Register RoPE state without changing persistent router correction biases.

    Args:
        model: Complete model whose state is managed.
    """
    for module in model.modules():
        if isinstance(module, LongcatFlashMegaMoeExperts):
            module.initializer_range = model.config.initializer_range
        for name in ("inv_freq", "original_inv_freq"):
            if name in getattr(module, "_non_persistent_buffers_set", ()):
                register_rebuildable_buffer(module, name)
    if not getattr(model, "_longcat_runtime_hook_registered", False):
        model.register_forward_pre_hook(_share_runtime)
        model._longcat_runtime_hook_registered = True  # pylint: disable=protected-access
