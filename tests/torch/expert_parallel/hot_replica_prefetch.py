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
"""Untimed prefetch subspans without changing stream or transport behavior."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from functools import wraps
from typing import Any
from collections.abc import Callable, Iterator
import sys
from unittest.mock import patch

import torch

from hyper_parallel.core.expert_parallel.hot_replica.pool import ReplicaPool
from hyper_parallel.core.expert_parallel.hot_replica.signal_transport import SignalReplicaTransport
from hyper_parallel.core.multicore import shmem


class PrefetchMeasurements:
    """Patch only diagnostic calls inside the existing prefetch and release spans."""

    def __init__(self, host: Any) -> None:
        """Use an existing host observer, retaining its nesting and event policy.

        Args:
            host: Observer with records, nesting stack and a span context manager.
        """
        self.host = host

    def _active(self) -> bool:
        return any(name in self.host.stack for name in ("weight_prefetch_enqueue", "weight_lease_release"))

    def _scoped(self, original: Callable, name: str) -> Callable:
        @wraps(original)
        def _call(*args, **kwargs):
            if not self._active():
                return original(*args, **kwargs)
            with self.host.span(name):
                return original(*args, **kwargs)
        return _call

    def _signal(self, original: Callable, operation: str) -> Callable:
        names = ("credit", "weight_ready", "weight_ack", "gradient_ready", "gradient_ack")

        @wraps(original)
        def _call(provider, channel, peer, slot, *args, **kwargs):
            if not self._active():
                return original(provider, channel, peer, slot, *args, **kwargs)
            with self.host.span(f"prefetch_{names[channel]}_{operation}"):
                result = original(provider, channel, peer, slot, *args, **kwargs)
            self.host.records[-1].update(peer=peer, slot=slot)
            return result
        return _call

    def _put(self, original: Callable) -> Callable:
        @wraps(original)
        def _call(destination, source, target, **kwargs):
            if not self._active():
                return original(destination, source, target, **kwargs)
            kind = "ready" if source.dtype == torch.int32 and source.numel() == 1 else "weight"
            with self.host.span(f"prefetch_{kind}_put_enqueue"):
                result = original(destination, source, target, **kwargs)
            self.host.records[-1].update(payload_bytes=source.numel() * source.element_size(), peer=target)
            return result
        return _call

    def _context(self, original: Callable, name: str) -> Callable:
        @contextmanager
        @wraps(original)
        def _call(*args, **kwargs) -> Iterator[Any]:
            if not self._active():
                with original(*args, **kwargs) as value:
                    yield value
                return
            with ExitStack() as cleanup:
                with self.host.span(f"prefetch_{name}_enter"):
                    value = cleanup.enter_context(original(*args, **kwargs))
                try:
                    yield value
                finally:
                    with self.host.span(f"prefetch_{name}_exit"):
                        cleanup.__exit__(*sys.exc_info())
        return _call

    def install(self, patches: ExitStack) -> None:
        """Install subspans in a diagnostic scope without additional synchronization.

        Args:
            patches: Scope that restores every original callable on exit.
        """
        ordinary = ((torch, "tensor", "prefetch_epoch_tensor"),
                    (torch.Tensor, "contiguous", "prefetch_source_contiguous"),
                    (torch.Tensor, "zero_", "prefetch_zero_enqueue"),
                    (torch.Tensor, "fill_", "prefetch_fill_enqueue"),
                    (SignalReplicaTransport, "_advance", "prefetch_epoch_advance"))
        for target, method, stage in ordinary:
            patches.enter_context(patch.object(target, method, self._scoped(getattr(target, method), stage)))
        for method, operation in (("_publish", "publish_enqueue"), ("_wait", "wait_enqueue")):
            patches.enter_context(patch.object(
                SignalReplicaTransport, method, self._signal(getattr(SignalReplicaTransport, method), operation)))
        for target, method, stage in ((ReplicaPool, "lease", "pool_lease"),
                                      (SignalReplicaTransport, "_copy_streams", "copy_streams")):
            patches.enter_context(patch.object(target, method, self._context(getattr(target, method), stage)))
        patches.enter_context(patch.object(shmem, "put", self._put(shmem.put)))
