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
"""Task-scoped or explicit multicore cleanup, without per-step polling."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from .modules import module as module_api

_P = ParamSpec("_P")
_R = TypeVar("_R")


def shutdown() -> None:
    """Close all multicore resources once, without destroying process groups.

    All WORLD ranks must call in the same order after completing backward and
    before destroying communication. Failed cleanup retains remaining handles.
    Do not call from arbitrary exception handlers, signal handlers, or finalizers.
    """
    module_api._RESOURCE_MANAGER.close_groups()  # pylint: disable=protected-access


def managed_run(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Close resources created inside a synchronous task when it returns normally.

    Place on a function covering model construction through its final backward.
    Existing outer resources remain open; models returned from this scope are
    closed and cannot be reused. Nested scopes are supported, concurrent scopes
    are not. All WORLD ranks must enter and leave scopes in the same order.
    Signals and process groups remain owned by the application. On exceptions,
    no cleanup collectives are attempted; the launcher must handle failed jobs.

    Args:
        function: Synchronous training or evaluation task.

    Returns:
        Wrapped function preserving its arguments and return value.
    """
    if (inspect.iscoroutinefunction(function) or inspect.isgeneratorfunction(function)
            or inspect.isasyncgenfunction(function)):
        raise TypeError("managed_run requires a synchronous non-generator function")

    @functools.wraps(function)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Restore scope ownership even when the task or cleanup fails."""
        manager = module_api._RESOURCE_MANAGER  # pylint: disable=protected-access
        previous = manager.owner
        owner = object()
        manager.owner = owner
        try:
            result = function(*args, **kwargs)
            manager.close_groups(owner=owner)
            return result
        finally:
            manager.owner = previous

    return run
