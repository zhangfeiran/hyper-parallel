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
"""Diagnostic subintervals of the unchanged route validation/count operations."""

from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from typing import Any
from unittest.mock import patch

import torch


class CountMeasurements:
    """Observe only operations inside the existing validation/count span."""

    def __init__(self, host: Any) -> None:
        """Retain the observer's nesting and event policy.

        Args:
            host: Observer exposing a nesting stack and span context manager.
        """
        self.host = host

    def _scoped(self, original: Callable, name: str) -> Callable:
        @wraps(original)
        def _call(*args, **kwargs):
            if "validation_and_counts" not in self.host.stack:
                return original(*args, **kwargs)
            with self.host.span(name):
                return original(*args, **kwargs)
        return _call

    def install(self, patches: ExitStack) -> None:
        """Restore all patched operations when the diagnostic scope closes.

        Args:
            patches: Scope retaining the existing callable and its restoration.
        """
        operations = ((torch.Tensor, "float", "count_sort_key_cast"),
                      (torch.Tensor, "sort", "count_topk_sort"),
                      (torch.Tensor, "__lt__", "count_range_lower"),
                      (torch.Tensor, "__ge__", "count_range_upper"),
                      (torch.Tensor, "__eq__", "count_duplicate_compare"),
                      (torch.Tensor, "__or__", "count_flag_combine"),
                      (torch.Tensor, "any", "count_flag_reduce"),
                      (torch.Tensor, "clamp", "count_histogram_clamp"),
                      (torch, "zeros", "count_histogram_allocate"),
                      (torch, "ones_like", "count_histogram_source"),
                      (torch.Tensor, "scatter_add_", "count_histogram_scatter"),
                      (torch.Tensor, "to", "count_flag_cast"),
                      (torch.Tensor, "copy_", "count_flag_copy"),
                      (torch, "cat", "count_payload_pack"))
        for owner, attribute, stage in operations:
            patches.enter_context(patch.object(owner, attribute, self._scoped(getattr(owner, attribute), stage)))
