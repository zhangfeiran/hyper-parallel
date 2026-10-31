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
"""Native Torch-compatible BF16 combine kernels with shape-independent token metadata."""

from __future__ import annotations

from functools import lru_cache

import torch

from hyper_parallel.core.multicore.scheduler.config import TaskAiCoreType, TaskDescC, TaskSplitValue, TaskType
from hyper_parallel.core.multicore.scheduler.config import validate_runtime_config
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config
from hyper_parallel.core.multicore.torch import ops as multicore_ops


def _tensor_from_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Keep serialization on CPU before copying immutable metadata to its owner device."""
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device)


def _inverse_order(source_order: torch.Tensor) -> torch.Tensor:
    """Build invocation-owned inverse indices for both combine and its score adjoint."""
    inverse = torch.empty(source_order.numel(), dtype=torch.int64, device=source_order.device)
    inverse.scatter_(0, source_order, torch.arange(source_order.numel(), device=source_order.device))
    return inverse


@lru_cache(maxsize=1)
def _require_payload() -> None:
    """Reject a Python/native task-layout mismatch before submitting device work."""
    multicore_ops._load_native()  # The component owns activation of its private adapter.
    try:
        version = torch.ops.hyper_parallel.mega_moe_native_combine_version()
    except AttributeError as error:
        raise RuntimeError("Native-bit combine requires payload ABI 8; rebuild the multicore component.") from error
    if version != 8:
        raise RuntimeError(f"Native-bit combine requires payload ABI 8, got {version}; rebuild the component.")


def _runtime_image(hidden: int, top_k: int, backward: bool = False, need_weight_grad: bool = True,
                   *, sum_only: bool = False, reduce_scores: bool = False, restore_scores: bool = False) -> bytes:
    """Reject unsupported rows before the private native entry can consume them."""
    if not 0 < top_k <= 8 or not 16 <= hidden <= 8192 or hidden % 16:
        raise ValueError('Native combine requires top_k in [1,8] and aligned hidden in [16,8192].')
    if reduce_scores and (not backward or (hidden, top_k) != (5120, 6)):
        raise ValueError("Fused score reduction requires backward H5120/K6.")
    if restore_scores and (not reduce_scores or not need_weight_grad):
        raise ValueError("Source-ordered scores require fused score reduction and weight gradients.")
    config = allocate_runtime_config(1, 1024, 1)
    config.num_workers = 48
    config.task_num = 1
    config.task_index_num[1] = 1
    config.vector_task_indices[0] = 0
    config.dynamic_data.dynamic_group_size = 1
    task = TaskDescC()
    task.task_type = TaskType.TASK_NATIVE_BIT_COMBINE_GRAD if backward else TaskType.TASK_NATIVE_BIT_COMBINE
    task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
    task.dependent_event = 0xFFFFFFFF
    task.extra_value_0 = top_k
    task.extra_value_1 = int(need_weight_grad)
    task.extra_value_2 = int(reduce_scores if backward else sum_only)
    if restore_scores:
        task.extra_value_2 = 2
    task.outputs[0].dim[0] = 1
    task.outputs[0].dim[1] = hidden
    config.all_tasks[0] = task
    validate_runtime_config(config, TaskSplitValue(tp=1, ep=1, seq_size=128, all_expert_num=1, top_k=1), 24)
    return serialize_runtime_config(config)


class _NativeCombine:
    """Keep static metadata separate from per-invocation maps and owned outputs."""

    def __init__(self, hidden: int, top_k: int, device_index: int) -> None:
        _require_payload()
        device = torch.device('npu', device_index)
        self.config = _tensor_from_bytes(_runtime_image(hidden, top_k), device)
        self.sum_config = _tensor_from_bytes(_runtime_image(hidden, top_k, sum_only=True), device)
        self.grad_configs = tuple(
            _tensor_from_bytes(_runtime_image(
                hidden, top_k, True, need, reduce_scores=(hidden, top_k) == (5120, 6) and need), device)
            for need in (False, True)
        )
        self.restore_config = (
            _tensor_from_bytes(_runtime_image(
                hidden, top_k, True, True, reduce_scores=True, restore_scores=True), device)
            if (hidden, top_k) == (5120, 6) else self.grad_configs[1]
        )
        self.hidden, self.top_k = hidden, top_k
        self.dummy = torch.empty((1, 1), device=device, dtype=torch.bfloat16)
        self.dummy_long = torch.zeros(1, device=device, dtype=torch.int64)
        self.dummy_int = torch.zeros(1, device=device, dtype=torch.int32)
        self.dummy_byte = torch.empty(4096, device=device, dtype=torch.uint8)
        initial_stream = torch.npu.current_stream(device)
        self.ready_stream = initial_stream.npu_stream
        self.ready = torch.npu.Event()
        self.ready.record(initial_stream)

    def __call__(self, expert: torch.Tensor, scores: torch.Tensor, source_order: torch.Tensor,
                 *, sum_only: bool = False, inverse_order: torch.Tensor | None = None) -> torch.Tensor:
        """Build a fresh inverse map and fuse BF16 multiplication with literal native summation."""
        rows = expert.shape[0]
        tokens = rows // self.top_k
        if expert.dtype != torch.bfloat16 or expert.shape != (rows, self.hidden) or not expert.is_contiguous():
            raise ValueError('Native combine requires contiguous BF16 expert rows.')
        if not sum_only and (scores.dtype != torch.bfloat16 or scores.numel() != rows or not scores.is_contiguous()):
            raise ValueError('The source-ordered scores must be a contiguous BF16 vector.')
        if source_order.dtype != torch.int64 or source_order.shape != (rows,) or not source_order.is_contiguous():
            raise ValueError('The source order must be a contiguous INT64 permutation.')
        if rows == 0:
            return expert.new_empty((0, self.hidden))
        inverse = _inverse_order(source_order) if inverse_order is None else inverse_order
        if (inverse.dtype != torch.int64 or inverse.shape != source_order.shape
                or inverse.device != source_order.device or not inverse.is_contiguous()):
            raise ValueError("The inverse order must be a contiguous INT64 permutation on the source device.")
        result = torch.empty((tokens, self.hidden), device=expert.device, dtype=expert.dtype)
        config = self.sum_config if sum_only else self.config
        self._record(config, expert.device)
        multicore_ops.mega_moe_grad(
            self.dummy, inverse, expert, self.dummy_long, self.dummy_int,
            self.dummy if sum_only else scores.reshape(rows, 1), self.dummy, self.dummy, self.dummy, self.dummy, self.dummy,
            self.dummy, self.dummy, result, self.dummy_long, self.dummy_long, self.dummy_int,
            self.dummy, self.dummy, self.dummy_long,
            self.dummy_byte, self.dummy_byte, self.dummy_byte, self.dummy_byte, self.dummy_byte,
            self.dummy_byte, self.dummy_byte, config, self.dummy_byte,
            0, 1, 1, self.hidden, tokens,
        )
        return result

    def sum(self, expert: torch.Tensor, source_order: torch.Tensor, *,
            inverse_order: torch.Tensor | None = None) -> torch.Tensor:
        """Sum routed BF16 input gradients in source order without product rounding."""
        if (self.hidden, self.top_k) != (5120, 6):
            raise ValueError("Native input-gradient sum is validated for H5120/K6.")
        return self(expert, self.dummy, source_order, sum_only=True, inverse_order=inverse_order)

    def _record(self, config: torch.Tensor, device: torch.device) -> None:
        """Keep cached storage live if another current stream consumes the plan."""
        current = torch.npu.current_stream(device)
        if current.npu_stream != self.ready_stream:
            current.wait_event(self.ready)
        for tensor in (config, self.dummy, self.dummy_long, self.dummy_int, self.dummy_byte):
            tensor.record_stream(current)

    def gradients(self, expert: torch.Tensor, scores: torch.Tensor, source_order: torch.Tensor,
                  grad_output: torch.Tensor, out: torch.Tensor, need_weight_grad: bool,
                  *, source_ordered: bool = False) -> torch.Tensor:
        """Return BF16 row scores or compact FP32 source scores after native BF16 rounding."""
        tokens = grad_output.shape[0]
        if expert.shape != out.shape or not out.is_contiguous():
            raise ValueError('Native combine gradient output must match the contiguous expert rows.')
        if any(out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
               for value in (expert, scores, grad_output)):
            raise ValueError('Native combine gradient output must not alias a source operand.')
        reduce_scores = (self.hidden, self.top_k) == (5120, 6) and need_weight_grad
        if source_ordered and not reduce_scores:
            raise ValueError("Source-ordered scores require weight gradients at H5120/K6.")
        # The private operator uses a BF16 carrier; mode 2 stores one FP32 value per 32-byte block.
        product = (torch.empty((expert.shape[0], 16), device=expert.device, dtype=expert.dtype) if reduce_scores
                   else torch.empty_like(expert) if need_weight_grad else self.dummy)
        config = self.restore_config if source_ordered else self.grad_configs[int(need_weight_grad)]
        self._record(config, expert.device)
        multicore_ops.mega_moe_grad(
            self.dummy, source_order, expert, self.dummy_long, self.dummy_int,
            scores.reshape(-1, 1), self.dummy, self.dummy, self.dummy, grad_output.contiguous(), self.dummy,
            self.dummy, out, product, self.dummy_long, self.dummy_long, self.dummy_int,
            self.dummy, self.dummy, self.dummy_long,
            self.dummy_byte, self.dummy_byte, self.dummy_byte, self.dummy_byte, self.dummy_byte,
            self.dummy_byte, self.dummy_byte, config, self.dummy_byte,
            0, 1, 1, self.hidden, tokens,
        )
        if not need_weight_grad:
            return self.dummy
        if source_ordered:
            return product.view(torch.float32)[:, 0].contiguous()
        return product[:, 0] if reduce_scores else product.sum(dim=-1)


@lru_cache(maxsize=8)
def _kernel(hidden: int, top_k: int, device_index: int) -> _NativeCombine:
    """Reuse immutable plans across token counts, without caching routing-dependent maps."""
    return _NativeCombine(hidden, top_k, device_index)


@lru_cache(maxsize=8)
def _device_supported(device_index: int) -> bool:
    """Restrict this kernel's 48-vector schedule to the validated 24-Cube topology."""
    return torch.npu.get_device_limit(device_index).get("cube_core_num") == 24


def _is_supported(expert: torch.Tensor, source_order: torch.Tensor, weights: torch.Tensor) -> bool:
    """Keep the existing literal path outside the validated BF16 geometry."""
    return (expert.device.type == "npu" and expert.dtype == torch.bfloat16
            and 0 < weights.shape[0] * weights.shape[1] < 2 ** 31
            and 0 < weights.shape[1] <= 8 and 16 <= expert.shape[-1] <= 8192
            and expert.shape[0] == weights.numel() and source_order.shape == (expert.shape[0],)
            and expert.shape[-1] % 16 == 0 and expert.is_contiguous()
            and source_order.dtype == torch.int64 and source_order.is_contiguous()
            and _device_supported(expert.device.index))
