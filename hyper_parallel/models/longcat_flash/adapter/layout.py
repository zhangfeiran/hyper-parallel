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
"""Reversible expert ownership packing retaining HF's unused Gate/Up rows."""

from __future__ import annotations

from typing import Any

# Native tensor APIs are the project backend contract.
# pylint: disable=forbidden-backend-import
import torch

from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps


class ExpertOwnershipPacking(ConversionOps):
    """Put each EP owner's real and unused rows in one contiguous shard."""

    def __init__(self, real_experts: int, zero_experts: int, ep_size: int, reverse: bool = False) -> None:
        """Configure real/zero cardinalities and the checkpoint's EP packing.

        Args:
            real_experts: Number of parameterized experts.
            zero_experts: Number of identity experts and retained unused rows.
            ep_size: Number of ranks in the expert group.
            reverse: Restore the original checkpoint order when true.
        """
        if real_experts <= 0 or zero_experts < 0 or ep_size <= 0:
            raise ValueError("invalid expert ownership dimensions")
        if real_experts % ep_size or zero_experts % ep_size:
            raise ValueError("real and zero expert rows must both be divisible by EP size")
        self.real_experts = real_experts
        self.zero_experts = zero_experts
        self.ep_size = ep_size
        self.reverse = reverse

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack HF [all real, all unused] rows or restore their original order.

        Args:
            tensor: Complete expert parameter tensor.
        """
        real, zero, ep = self.real_experts, self.zero_experts, self.ep_size
        if tensor.shape[0] != real + zero:
            raise ValueError("Gate/Up rows do not match the declared expert cardinalities")
        if ep == 1 or zero == 0:
            return tensor
        tail = tensor.shape[1:]
        if self.reverse:
            owned = tensor.reshape(ep, (real + zero) // ep, *tail)
            return torch.cat((owned[:, :real // ep].reshape(real, *tail),
                              owned[:, real // ep:].reshape(zero, *tail)), dim=0).contiguous()
        return torch.cat((tensor[:real].reshape(ep, real // ep, *tail),
                          tensor[real:].reshape(ep, zero // ep, *tail)), dim=1).reshape_as(tensor).contiguous()

    def convert(self, input_dict: dict, source_patterns: list[str], target_patterns: list[str],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Apply the packing through the framework checkpoint conversion API.

        Args:
            input_dict: Checkpoint tensors collected by source pattern.
            source_patterns: Checkpoint source tensor names.
            target_patterns: Destination tensor names.
        """
        del kwargs
        if self.reverse and len(source_patterns) == 2:
            marker = input_dict[source_patterns[1]]
            marker = marker[0] if isinstance(marker, list) else marker
            if int(marker.item()) != self.ep_size:
                raise ValueError("checkpoint EP packing differs; export to HF before changing EP size")
        elif len(source_patterns) != 1:
            raise ValueError("expert packing requires one source tensor")
        if len(target_patterns) not in (1, 2) or self.reverse and len(target_patterns) != 1:
            raise ValueError("invalid expert packing destination tensors")
        value = input_dict[source_patterns[0]]
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("expert packing requires exactly one tensor")
            value = value[0]
        result = {target_patterns[0]: self.apply(value)}
        if len(target_patterns) == 2:
            result[target_patterns[1]] = torch.tensor([self.ep_size], dtype=torch.int64, device=value.device)
        return result

    @property
    def reverse_op(self) -> ExpertOwnershipPacking:
        """Return the exact inverse layout conversion."""
        return ExpertOwnershipPacking(self.real_experts, self.zero_experts, self.ep_size, not self.reverse)
