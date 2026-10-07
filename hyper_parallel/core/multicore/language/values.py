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
"""CPU reference values for capacity-shaped tensors and local routed expert groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hyper_parallel.core.multicore.language.types import DType, RaggedTensorType


@dataclass(frozen=True)
class RaggedTensor:
    """CPU debug storage and effective rows; native execution uses existing route/workspace objects."""

    storage: torch.Tensor
    valid_rows: int

    def __post_init__(self):
        if not isinstance(self.storage, torch.Tensor) or self.storage.dim() != 2:
            raise ValueError("RaggedTensor storage must be a matrix")
        if type(self.valid_rows) not in (int,) or not 0 <= self.valid_rows <= self.storage.shape[0]:
            raise ValueError("RaggedTensor valid_rows must fit its storage capacity")

    def __class_getitem__(cls, arguments: tuple[DType, tuple]) -> RaggedTensorType:
        return RaggedTensorType(*arguments)


@dataclass(frozen=True)
class RouteMetadata:
    """CPU local expert-major reference metadata; group_list is cumulative INT64 row counts."""

    group_list: torch.Tensor

    def __post_init__(self):
        if not isinstance(self.group_list, torch.Tensor) or self.group_list.dtype != torch.int64:
            raise ValueError("RouteMetadata group_list must be INT64")
        if self.group_list.dim() != 1 or self.group_list.numel() == 0 or not self.group_list.is_contiguous():
            raise ValueError("RouteMetadata requires a nonempty contiguous group-list vector")
