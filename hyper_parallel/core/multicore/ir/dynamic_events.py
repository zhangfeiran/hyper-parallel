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
"""MoE 前缀计数驱动的受限有效任务域；预留任务仍完整发布事件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MoeCountSourceIR:
    """借用原生 group_list 的 int64 专家前缀计数，不接收任意运行时索引函数。"""

    input_positions: tuple[int, ...]
    expert_count: int
    maximum_expert_rows: int
    maximum_total_rows: int

    def __post_init__(self) -> None:
        sizes = (self.expert_count, self.maximum_expert_rows, self.maximum_total_rows)
        if any(type(size) not in (int,) or size <= 0 for size in sizes):
            raise ValueError("MoE dynamic count bounds must be positive integers")
        if (not isinstance(self.input_positions, tuple) or not self.input_positions
                or any(type(slot) not in (int,) or slot < 0 for slot in self.input_positions)
                or len(set(self.input_positions)) != len(self.input_positions)):
            raise ValueError("MoE count input positions must be unique immutable nonnegative integers")

    def counts(self, prefix: tuple[int, ...]) -> tuple[int, ...]:
        """校验显式主机快照并返回每个专家的有效行数；不读取设备张量。

        Args:
            prefix: 诊断调用显式提供的专家累计计数。
        """
        if not isinstance(prefix, tuple) or len(prefix) != self.expert_count:
            raise ValueError("MoE prefix counts differ from the local expert domain")
        if any(type(value) not in (int,) or value < 0 for value in prefix):
            raise ValueError("MoE prefix counts require nonnegative integers")
        counts = tuple(end - start for start, end in zip((0, *prefix[:-1]), prefix))
        if any(not 0 <= count <= self.maximum_expert_rows for count in counts):
            raise ValueError("MoE prefix counts are nonmonotone or exceed the per-expert row bound")
        if prefix[-1] > self.maximum_total_rows:
            raise ValueError("MoE prefix counts exceed the total receive row bound")
        return counts

    def validate_device_binding(self, tensor: torch.Tensor, device: Any) -> None:
        """仅校验借用张量的元数据，保持设备计数在原生路由与 worker 中消费。

        Args:
            tensor: 已有路由生产的设备 group_list；内容边界归路由容量协议。
            device: 本次 kernel 的输入设备。
        """
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int64:
            raise ValueError("MoE device group_list must be an int64 Tensor")
        if tuple(tensor.shape) != (self.expert_count,) or not tensor.is_contiguous():
            raise ValueError("MoE device group_list must be contiguous with one prefix per local expert")
        if tensor.device != torch.device(device):
            raise ValueError("MoE device group_list must share the execution device")


@dataclass(frozen=True)
class MoeTaskDomainIR:
    """原生专家 × Cube lane 或专家 × Vector tile 域，空任务也通知原有事件。"""

    kind: str
    shape: tuple[int, int]
    rows_per_task: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.shape, tuple) or len(self.shape) != 2
                or any(type(size) not in (int,) or size <= 0 for size in self.shape)):
            raise ValueError("MoE dynamic task domain requires positive expert/task dimensions")
        if self.kind not in ("grouped_matmul", "swiglu") or type(self.rows_per_task) not in (int,):
            raise ValueError("Unsupported MoE dynamic task operation")
        if (self.kind == "grouped_matmul" and self.rows_per_task != 0
                or self.kind == "swiglu" and self.rows_per_task <= 0):
            raise ValueError("MoE dynamic row tile differs from the native task operation")

    def row_intervals(self, source: MoeCountSourceIR, prefix: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """解析受限前缀差和 tile 截断；GMM lane 的核内分工仍由原生 tiling 决定。

        Args:
            source: 同一事件图的计数来源和容量边界。
            prefix: 诊断用主机快照，不支持设备张量自动拷回。
        """
        counts = source.counts(prefix)
        if self.shape[0] != source.expert_count:
            raise ValueError("MoE task domain differs from its expert count source")
        if self.kind == "swiglu" and self.shape[1] * self.rows_per_task < source.maximum_expert_rows:
            raise ValueError("MoE reserved Vector tiles do not cover the per-expert row bound")
        intervals = []
        for expert, count in enumerate(counts):
            begin = 0 if expert == 0 else prefix[expert - 1]
            for task in range(self.shape[1]):
                offset = task * self.rows_per_task
                end = count if self.kind == "grouped_matmul" else min(count, offset + self.rows_per_task)
                intervals.append((begin + min(offset, count), begin + end))
        return tuple(intervals)
