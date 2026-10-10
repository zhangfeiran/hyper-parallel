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
"""按真实存储别名生命周期复用稠密 SSA 的内部保存值。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from threading import Lock
from weakref import ReferenceType, WeakSet, WeakValueDictionary, ref

import torch

_POOLS: WeakValueDictionary[torch.device, DenseSavedPool] = WeakValueDictionary()
_POOLS_LOCK = Lock()


@dataclass
class _SavedSlot:
    base: torch.Tensor
    stream: object
    generation: int = 0


@dataclass(frozen=True)
class DenseSavedUse:
    """保留调用代次，不额外持有逻辑张量或缓存基准分配。"""

    pool: ReferenceType[DenseSavedPool]
    slot: ReferenceType[_SavedSlot]
    generation: int

    def consume(self, tensors: tuple[torch.Tensor, ...], stream: object) -> None:
        """在实际消费流上建立依赖并登记解包后的保存值。

        Args:
            tensors: 当前 autograd 解包的张量，可能来自 CPU 保存钩子。
            stream: 实际反向执行流。
        """
        pool, slot = self.pool(), self.slot()
        if pool is not None and slot is not None:
            pool.record_use(slot, self.generation, stream)
        for tensor in tensors:
            if tensor.device.type != "cpu":
                tensor.record_stream(stream)


class DenseSavedPool:
    """保存别名释放后才复用的有界缓存，支持重叠调用和流交接。"""

    def __init__(self, device: torch.device, exclusive: Callable[[torch.Tensor], bool],
                 max_cached_invocations: int = 2) -> None:
        """配置缓存；引用查询只读主机元数据。

        Args:
            device: 已规范化的设备；CPU 用于协议测试。
            exclusive: 经验证的原生存储独占查询。
            max_cached_invocations: 缓存分配数量上限；零表示每次使用私有分配。
        """
        if type(max_cached_invocations) not in (int,) or max_cached_invocations < 0 or not callable(exclusive):
            raise ValueError("Dense saved cache requires a callable query and a nonnegative slot bound")
        self.device, self.exclusive = device, exclusive
        self.max_cached_invocations = max_cached_invocations
        self._slots: list[_SavedSlot] = []
        self._owners: WeakSet[object] = WeakSet()
        self._lock = Lock()
        self._closed = False
        self._allocations = self._reuses = self._fallbacks = self._transfers = 0

    def register_owner(self, owner: object) -> bool:
        """登记弱所有者；关闭后拒绝新所有者。

        Args:
            owner: 持有缓存的执行对象。
        """
        with self._lock:
            if self._closed:
                return False
            self._owners.add(owner)
            return True

    def release_owner(self, owner: object) -> None:
        """最后一个所有者关闭时释放缓存，已保存的张量继续持有存储。

        Args:
            owner: 已关闭的执行对象。
        """
        with self._lock:
            self._owners.discard(owner)
            if not self._owners:
                self._closed = True
                self._slots.clear()

    def _handoff(self, slot, stream):
        if slot.stream != stream:
            event = torch.get_device_module(self.device).Event()
            event.record(slot.stream)
            stream.wait_event(event)
            self._transfers += 1
            slot.stream = stream
        if self.device.type != "cpu":
            slot.base.record_stream(stream)

    def record_use(self, slot: _SavedSlot, generation: int, stream: object) -> None:
        """消费前建立流依赖，忽略已被 CPU 保存钩子释放的旧代次。

        Args:
            slot: 本次调用的缓存槽位。
            generation: 借用时的代次。
            stream: 实际消费流。
        """
        with self._lock:
            if not self._closed and slot.generation == generation:
                self._handoff(slot, stream)

    @staticmethod
    def _layout(shapes):
        offsets, elements = {}, 0
        for value_id, shape in shapes.items():
            if type(value_id) not in (int,) or any(type(size) not in (int,) or size < 0 for size in shape):
                raise ValueError("Dense saved shapes require integer IDs and nonnegative dimensions")
            offsets[value_id] = elements
            elements += (prod(shape) + 31) // 32 * 32
        return offsets, elements

    def acquire(self, shapes: Mapping[int, tuple[int, ...]], stream: object
                ) -> tuple[dict[int, torch.Tensor], DenseSavedUse]:
        """借出不重叠的 BF16 内部值；活跃别名阻止复用。

        Args:
            shapes: SSA 中需要保存且未返回给调用方的值及形状。
            stream: 当前分配和执行流。
        """
        offsets, elements = self._layout(shapes)
        with self._lock:
            if self._closed:
                raise RuntimeError("Dense saved pool is closed")
            slot = next((item for item in self._slots if self.exclusive(item.base)), None)
            if slot is None:
                slot = _SavedSlot(torch.empty(elements, dtype=torch.bfloat16, device=self.device), stream)
                self._allocations += 1
                if len(self._slots) < self.max_cached_invocations:
                    self._slots.append(slot)
                else:
                    self._fallbacks += 1
            else:
                self._handoff(slot, stream)
                if slot.base.numel() < elements:
                    slot.base = torch.empty(elements, dtype=torch.bfloat16, device=self.device)
                    self._allocations += 1
                else:
                    self._reuses += 1
            slot.generation += 1
            # 独立叶张量不通过 _base 暴露缓存分配，所有可访问的别名均增加存储引用。
            values = {value_id: torch.empty(0, dtype=torch.bfloat16, device=self.device).set_(
                slot.base.untyped_storage(), offsets[value_id], shape) for value_id, shape in shapes.items()}
            return values, DenseSavedUse(ref(self), ref(slot), slot.generation)

    def statistics(self) -> dict[str, int]:
        """读取缓存高水位与复用统计，不同步设备。"""
        with self._lock:
            return {"cached_invocations": len(self._slots), "max_cached_invocations": self.max_cached_invocations,
                    "cached_bytes": sum(slot.base.numel() * slot.base.element_size() for slot in self._slots),
                    "allocations": self._allocations, "reuses": self._reuses, "private_fallbacks": self._fallbacks,
                    "stream_transfers": self._transfers}

    def close(self) -> None:
        """关闭借用入口，保留已有张量自身的存储生命周期。"""
        with self._lock:
            self._closed = True
            self._owners.clear()
            self._slots.clear()


def shared_dense_saved(device: torch.device, exclusive: Callable[[torch.Tensor], bool],
                       owner: object) -> DenseSavedPool:
    """在同一设备的执行对象之间共享有界保存缓存。

    Args:
        device: 已规范化的设备。
        exclusive: 已加载原生适配器的存储查询。
        owner: 持有缓存直到关闭或回收的执行对象。
    """
    with _POOLS_LOCK:
        pool = _POOLS.get(device)
        if pool is None or not pool.register_owner(owner):
            pool = DenseSavedPool(device, exclusive)
            pool.register_owner(owner)
            _POOLS[device] = pool
        return pool
