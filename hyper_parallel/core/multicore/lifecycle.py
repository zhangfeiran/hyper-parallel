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
"""Explicit safe points and entry-point ownership for multicore resources."""

from __future__ import annotations

import functools
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import ParamSpec, TypeVar

import torch.distributed as dist

from .modules import module as module_api

_P = ParamSpec("_P")
_R = TypeVar("_R")
_requested_signal = 0
_agreed_signal = 0
_managed_run_active = False


class _StopRequested(SystemExit):
    """Exit only after all ranks have agreed to stop at a safe point."""


def _record_signal(signum: int, _frame: FrameType | None) -> None:
    """Defer device operations and collective teardown to the main execution path."""
    global _requested_signal  # pylint: disable=global-statement
    _requested_signal = signum


def _reclaim(operation: str) -> tuple[int, int]:
    """Agree on stop requests while draining the selected process resource groups."""
    global _agreed_signal  # pylint: disable=global-statement
    count, agreed_signal = module_api._RESOURCE_MANAGER.reclaim(  # pylint: disable=protected-access
        operation=operation, stop_signal=_requested_signal,
    )
    # Only the signal handler writes the local request during a run.
    _agreed_signal = max(_agreed_signal, agreed_signal)
    return count, agreed_signal


def collect_resources() -> int:
    """Collect orphaned groups whose modules and backward work are gone on all ranks.

    Every rank must call this at the same idle model boundary, even when its local
    pending set is empty. Collection keeps groups with pending backward graphs.
    Communication failures require the job launcher to terminate the workers.

    Returns:
        Number of resource groups released on this rank.
    """
    count, _ = _reclaim("collect")
    return count


def lifecycle_checkpoint() -> None:
    """Collect orphans and cooperatively observe termination between complete steps.

    Call on every rank in the same order, outside timed forward/backward work.
    With :func:`managed_run`, a SIGINT or SIGTERM on any rank causes all ranks to
    leave through the managed shutdown path. No polling is added to model layers.

    Raises:
        SystemExit: Every rank has agreed to a termination request.
        RuntimeError: Lifecycle manifests differ or resource cleanup fails.
    """
    _, agreed_signal = _reclaim("checkpoint")
    if agreed_signal:
        raise _StopRequested(128 + agreed_signal)


def shutdown(*, destroy_process_group: bool = False) -> None:
    """Close all multicore groups before optionally destroying the process group.

    Call on every rank after all backward work, while communication is healthy.
    Live modules become unusable after their resources close. Failed cleanup
    retains the remaining handles and does not destroy the process group.
    Never call from a signal handler, GC finalizer, or arbitrary exception path.

    Args:
        destroy_process_group: Destroy the default distributed world after cleanup.

    Raises:
        RuntimeError: Backward is pending, ranks disagree, or native cleanup fails.
    """
    _reclaim("shutdown")
    if destroy_process_group and dist.is_initialized():
        dist.destroy_process_group()


def _install_signal_handlers() -> dict[int, signal.Handlers | Callable]:
    """Install opt-in handlers without replacing a host framework's custom policy."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("managed_run must execute on the main thread")
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    defaults = (signal.SIG_DFL, signal.default_int_handler)
    if any(handler not in defaults for handler in previous.values()):
        raise RuntimeError("managed_run cannot replace custom signal handlers; use framework lifecycle callbacks")
    installed = []
    try:
        for signum in previous:
            signal.signal(signum, _record_signal)
            installed.append(signum)
    except BaseException:
        for signum in installed:
            signal.signal(signum, previous[signum])
        raise
    return previous


def managed_run(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Own cleanup and cooperative signals around an existing worker entry point.

    The decorated function must return before destroying its process group, or
    replace its normal destroy call with ``shutdown(destroy_process_group=True)``.
    Use ``lifecycle_checkpoint()`` between complete steps to observe signals.
    Unexpected exceptions propagate without starting unsafe cleanup collectives;
    an external launcher must terminate failed or unresponsive jobs. SIGKILL is
    uncatchable. This decorator does not install atexit cleanup or a watchdog.

    Args:
        function: Synchronous worker entry point called on every rank.

    Returns:
        Wrapped entry point with identical arguments and successful return value.
    """
    @functools.wraps(function)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Restore the caller's signal policy on every exit path."""
        global _managed_run_active, _requested_signal, _agreed_signal  # pylint: disable=global-statement
        if _managed_run_active:
            raise RuntimeError("managed_run cannot be nested")
        _requested_signal = 0
        _agreed_signal = 0
        previous = _install_signal_handlers()
        _managed_run_active = True
        try:
            try:
                result = function(*args, **kwargs)
            except _StopRequested:
                shutdown(destroy_process_group=True)
                raise
            shutdown(destroy_process_group=True)
            if _requested_signal or _agreed_signal:
                raise SystemExit(128 + max(_requested_signal, _agreed_signal))
            return result
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            _managed_run_active = False
            _requested_signal = 0
            _agreed_signal = 0

    return run
