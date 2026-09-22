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
"""Process-group shared guest storage with stream-ordered exclusive leases."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist


class ReplicaPool:
    """Reuse B guest slots across layers without retaining parameter snapshots."""

    def __init__(self, weights: tuple[torch.Tensor, ...], slots: int) -> None:
        """Allocate only guest matrices; parameter storage remains caller-owned."""
        self.weights = tuple(weight.new_empty((slots, *weight.shape[1:])) for weight in weights)
        self.gradients = None
        self._lock = threading.Lock()
        self._event = None

    @contextmanager
    def lease(self, *, backward: bool = False) -> Iterator[ReplicaPool]:
        """Order this stream after the previous consumer, rejecting host overlap."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent expert replica pool leases are unsupported")
        backend, stream = None, None
        try:
            device = self.weights[0].device
            backend = None if device.type == "cpu" else getattr(torch, device.type)
            stream = None if backend is None else backend.current_stream(device)
            if self._event is not None:
                stream.wait_event(self._event)
            if backward:
                if self.gradients is None:
                    self.gradients = tuple(torch.empty_like(weight, dtype=torch.float32) for weight in self.weights)
                for gradient in self.gradients:
                    gradient.zero_()
            yield self
        finally:
            try:
                if stream is not None:
                    self._event = backend.Event()
                    self._event.record(stream)
                    for tensor in self.weights + (self.gradients or ()):
                        tensor.record_stream(stream)
            finally:
                self._lock.release()


_POOLS = WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def replica_pool(weights: tuple[torch.Tensor, ...], slots: int, group: object) -> ReplicaPool:
    """Get one pool per group, device, dtype and expert matrix layout.

    The registry holds no process-group references in its values, so destroying
    the group also releases its pools. Pools never retain owner parameters.
    """
    group = dist.group.WORLD if group is None else group
    if group is None:
        raise ValueError("Replica pools require an initialized process group")
    key = (slots, tuple((weight.device, weight.dtype, tuple(weight.shape[1:])) for weight in weights))
    with _REGISTRY_LOCK:
        pools = _POOLS.setdefault(group, {})
        if key not in pools:
            pools[key] = ReplicaPool(weights, slots)
        return pools[key]
