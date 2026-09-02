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
"""Symmetric memory module for hyper-parallel."""
from __future__ import annotations

__all__ = [
    "is_shmem_available",
    "initialize",
    "empty",
    "aligned_empty",
    "free",
    "rendezvous",
    "set_signal_pad_size",
    "get_signal_pad_size",
    "barrier",
    "shmem_put",
    "shmem_get",
    "shmem_wait_for_signal",
    "shmem_put_with_signal",
    "shmem_signal_op",
    "shmem_allgather",
    "shmem_alltoall",
    "fused_all_gather_matmul",
    "fused_matmul_reduce_scatter",
    # "overlap_launch_all_to_all_v",
]

from typing import Any

from hyper_parallel.platform import get_platform

platform = get_platform()


class _LazySymmetricMemoryHandler:
    """Resolve the optional platform handler on first operation."""

    def __init__(self) -> None:
        """Initialize without importing an optional native backend."""
        self._handler = None

    def __getattr__(self, name: str) -> Any:
        if self._handler is None:
            self._handler = platform.get_symmetric_memory_handler()
        return getattr(self._handler, name)


_symm_handler = _LazySymmetricMemoryHandler()


def is_shmem_available() -> bool:
    """Check whether symmetric memory (shmem) is available on the current platform."""
    return _symm_handler.is_shmem_available()


def initialize(group: Any | None = None) -> None:
    """Initialize one symmetric-memory world for ``group``.

    Args:
        group: Optional platform process group.
    """
    initializer = getattr(_symm_handler, "initialize", None)
    if not callable(initializer):
        if group is not None:
            raise NotImplementedError(
                "group-scoped symmetric memory is unavailable on the active platform."
            )
        return
    initializer(group)


def empty(shape: Any, dtype: Any) -> Any:
    r"""
    Similar to :func:`empty()`. The returned tensor will malloc a
    symmetric memory among participating processes. This output tensor can be directly used
    for one-sided communication, normal computation, or can be used by rendezvous()

    Args:
        shape (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`mindspore.dtype` or :class:`torch.dtype`): the desired data type of returned tensor.

    example::
        >>> # doctest: +SKIP
        >>> # Create a symmetric memory tensor of shape (2, 3) with float32 data type
        >>> symm_tensor = symmetric_memory.empty((2, 3), dtype=mindspore.float32)
    """
    return _symm_handler.empty(shape, dtype)


def aligned_empty(shape: Any, dtype: Any, alignment: int) -> Any:
    """Allocate symmetric storage with an aligned base address.

    Args:
        shape: Output tensor shape.
        dtype: Output tensor dtype.
        alignment: Required power-of-two byte alignment.

    Returns:
        A tensor backed by aligned symmetric storage.
    """
    allocator = getattr(_symm_handler, "aligned_empty", None)
    if not callable(allocator):
        raise NotImplementedError(
            "aligned symmetric allocation is unavailable on the active platform."
        )
    return allocator(shape, dtype, alignment)


def free(tensor: Any) -> None:
    """Release storage allocated by :func:`empty` or :func:`aligned_empty`."""
    releaser = getattr(_symm_handler, "free", None)
    if not callable(releaser):
        raise NotImplementedError("symmetric-memory free is unavailable on the active platform.")
    releaser(tensor)


def barrier() -> None:
    r"""
    A synchronization barrier for all processes in the default process group.
    This function blocks until all processes have reached this method, ensuring
    that all processes are synchronized at this point in the code.

    Example::
        >>> # doctest: +SKIP
        >>> # Synchronize all processes before starting a new phase of computation
        >>> symmetric_memory.barrier()
    """
    return _symm_handler.barrier()


def rendezvous(tensor: Any, group: Any) -> Any:
    r"""
    This interface is not compatible yet, cause shmem rendezvous is still in development.

    rendezvous(tensor, group) -> _SymmetricMemory

    Establish a symmetric memory tensor among participating processes. This is
    a collective operation. It will malloc a signal symmetric memory coupled with tensor
    and get all corresponding ptrs for buffer and signal.

    Args:
        tensor: the local tensor used to establish the symmetric memory tensor.
            It must be allocated via :func:`symmetric_memory.empty()`. The shape,
            dtype, and device type must be identical across all participating processes.
        group: The group identifying the
            participating processes. This can be either a group name or a process group object.
    """
    return _symm_handler.rendezvous(tensor, group)


def set_signal_pad_size(size: int) -> None:
    r"""
    This interface is not compatible yet, you can alloc the signal tensor in your program instead.
    Set the signal pad size for future symmetric memory allocations.

    Signal pads are P2P-accessible memory regions used for synchronization in
    symmetric memory. This function allows users to configure
    the signal pad size to be proportional to their workload requirements.

    .. warning::
        This must be called before any symmetric memory allocations are made.
        The size cannot be changed after allocations have been performed.

    Args:
        size (int): the signal pad size in bytes. The size should be
            proportional to the number of blocks launched and the world size.
    """
    return _symm_handler.set_signal_pad_size(size)


def get_signal_pad_size() -> int:
    r"""
    This interface is not compatible yet, you can alloc the signal tensor in your program instead.
    Get the current signal pad size for symmetric memory allocations.

    Returns the user-configured size if set via :func:`set_signal_pad_size`,
    otherwise returns the default size.

    Returns:
        int: the signal pad size in bytes.
    """
    return _symm_handler.get_signal_pad_size()


def shmem_put(
    target: Any,
    target_offset: Any,
    src: Any,
    src_offset: Any,
    size: Any,
    target_rank: int,
) -> None:
    r"""
    Perform a one-sided send operation to write data from the local source tensor to a target tensor.
    """
    _symm_handler.shmem_put(target, target_offset, src, src_offset, size, target_rank)


def shmem_get(
    target: Any,
    target_offset: Any,
    src: Any,
    src_offset: Any,
    size: Any,
    target_rank: int,
) -> None:
    r"""
    Perform a one-sided receive operation to read data from a target tensor into the local source tensor.
    """
    _symm_handler.shmem_get(target, target_offset, src, src_offset, size, target_rank)


def shmem_signal_op(
    signal: Any,
    signal_offset: Any,
    signal_value: Any,
    signal_op: int,
    target_rank: int,
) -> None:
    r"""
    Perform an atomic operation on a signal in the symmetric memory.

    This function allows for atomic updates to a signal value at a specified offset within a symmetric memory tensor.
    The operation is performed on the target rank's memory, enabling efficient synchronization between processes.

    Args:
        signal (tensor, int32): The symmetric memory tensor that contains the signal to be updated. (Data type: int32)
        signal_offset (tensor): The byte offset within the signal tensor
            where the signal value is located.
        signal_value (tensor, int32): The value to update the signal with. (Data type: int32)
        signal_op (int64, optional): The operation to perform on the signal value, 0:set, 1:add. Defaults to 0.
        target_rank (int64, optional): The rank of the target process that owns the signal tensor. Defaults to 0.
    """
    _symm_handler.shmem_signal_op(signal, signal_offset, signal_value, signal_op, target_rank)


def shmem_wait_for_signal(
    depend_tensor: Any,
    signal: Any,
    signal_offset: Any,
    compare_value: Any,
    compare_op: int,
) -> None:
    r"""
    Wait for a signal to satisfy a specified condition before proceeding.
    This function blocks the calling process until the value at the specified signal offset
    meets the condition defined by compare_value and compare_op.
    Args:
        depend_tensor (tensor): A tensor that the wait operation depends on.
            It is used to ensure proper ordering of operations.
        signal (tensor, int32): The symmetric memory tensor that contains the signal to wait on. (Data type: int32)
        signal_offset (tensor): The byte offset within the signal tensor where the signal value is located.
        compare_value (tensor, int32): The value to compare against the
            signal value at the specified offset. (Data type: int32)
        compare_op (int64, optional): The comparison operator to use.
            0: equal, 1: greater than, 2: less than. Defaults to 0.
    """
    _symm_handler.shmem_wait_for_signal(depend_tensor, signal, signal_offset, compare_value, compare_op)


def shmem_put_with_signal(
    target: Any,
    target_offset: Any,
    src: Any,
    src_offset: Any,
    size: Any,
    signal: Any,
    signal_offset: Any,
    signal_value: Any,
    signal_op: int,
    target_rank: int,
) -> None:
    r"""
    Perform a one-sided send operation to write data from the local source tensor to a target tensor,
    then update the signal value at signal_offset with signal_op.
    This function combines the data transfer of shmem_put with an atomic update to a signal,
    allowing for efficient synchronization after the put operation.
    Args:
        target (tensor): The target symmetric memory tensor to write to.
        target_offset (tensor): The byte offset within the target tensor where the data should be written.
        src (tensor): The local source tensor containing the data to be sent.
            Its dtype must match the dtype of the target tensor.
        src_offset (tensor): The byte offset within the source tensor where the data to be sent is located.
        size (tensor): The size of the data to be sent in bytes.
        signal (tensor, int32): The symmetric memory tensor that contains
            the signal to be updated after the put operation. (Data type: int32)
        signal_offset (tensor): The byte offset within the signal tensor where the signal value is located.
        signal_value (tensor, int32): The value to update the signal with after the put operation. (Data type: int32)
        signal_op (int64, optional): The operation to perform on the signal value, 0:set, 1:add. Defaults to 0.
        target_rank (int64, optional): The rank of the target process that
            owns the target tensor and signal. Defaults to 0.
    """
    _symm_handler.shmem_put_with_signal(target, target_offset, src, src_offset,
                                        size, signal, signal_offset, signal_value, signal_op, target_rank)


def shmem_allgather(output_tensor: Any, input_tensor: Any) -> None:
    """
    This interface supports both torch and MindSpore.
    This function gathers the input tensor from all ranks and concatenates them into the output tensor.
    The resulting output tensor will contain the gathered data from all ranks,
    and the order of the gathered data will correspond to the order of the ranks.
    All ranks must provide an input tensor of the same shape and dtype,
    and the output tensor must be appropriately sized to hold the gathered data from all ranks.
    Args:
        output_tensor (tensor): The symmetric memory tensor that will hold the gathered data from all ranks.
            Its shape should be (world_size * local_shape) where local_shape is the shape of input_tensor.
        input_tensor (tensor): The local tensor to be gathered from each rank.
    """
    _symm_handler.shmem_allgather(output_tensor, input_tensor)


def shmem_alltoall(
    send_tensor_list: Any,
    receive_tensor: Any,
    receive_list: Any,
) -> None:
    """
    This interface only supports torch for now, the mindspore version is still in development.
    This function performs an all-to-all communication pattern where each rank sends a tensor
    to every other rank and receives a tensor from every other rank.
    the send_tensor_list is a list of tensors to be sent to each rank,
    and the receive_tensor is the tensor that will hold the received data from all ranks.
    The receive_list is a list of tensors that will hold the received data from each rank.
    The order of the received data in the receive_tensor and receive_list will correspond to the order of the ranks.
    Args:
        send_tensor_list (list of tensors): A list of tensors to be sent to each rank,
            where send_tensor_list[i] is the tensor to be sent to rank i.
        receive_tensor (tensor): The symmetric memory tensor that will hold the received data from all ranks.
        receive_list (list of int): A list of int that specifies the size of the data to be received from each rank,
            where receive_list[i] is the size of the data to be received from rank i.
    """
    _symm_handler.shmem_alltoall(send_tensor_list, receive_tensor, receive_list)


def fused_all_gather_matmul(
    a: Any,
    b: Any,
    c: Any,
    gather_out: Any,
    signal: Any,
    block_size: int | None,
) -> Any:
    """Fuse all-gather with matrix multiplication on supported backends.

    Args:
        a: Rank-local matrix input.
        b: Matrix multiplication weight.
        c: Preallocated matrix multiplication output.
        gather_out: Preallocated symmetric all-gather output.
        signal: Symmetric synchronization tensor.
        block_size: Optional tiled row count.

    Returns:
        Backend-specific fused outputs.
    """
    return _symm_handler.fused_all_gather_matmul(a, b, c, gather_out, signal, block_size)


def fused_matmul_reduce_scatter(
    x1: Any,
    x2: Any,
    symm_tensor: Any,
    signal: Any,
    reduce_op: str,
) -> Any:
    """Fuse matrix multiplication with reduce-scatter.

    Args:
        x1: Left matrix input.
        x2: Right matrix input.
        symm_tensor: Symmetric communication workspace.
        signal: Symmetric synchronization tensor.
        reduce_op: Reduction name, either ``sum`` or ``avg``.

    Returns:
        Rank-local reduced matrix.
    """
    return _symm_handler.fused_matmul_reduce_scatter(x1, x2, symm_tensor, signal, reduce_op)
