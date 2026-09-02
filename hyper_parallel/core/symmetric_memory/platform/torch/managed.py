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
"""Group-scoped lifecycle layered over the legacy Torch SHMEM operations."""

from __future__ import annotations

import importlib
import os
import socket
from types import ModuleType
from typing import Any

from hyper_parallel.platform.torch.symmetric_memory import symmetric_memory as _legacy


def _get_torch() -> ModuleType:
    """Return Torch without importing it during module initialization."""
    return importlib.import_module("torch")


def _get_torch_distributed() -> ModuleType:
    """Return Torch distributed without a static backend import."""
    return importlib.import_module("torch.distributed")


def _get_torch_distributed_c10d() -> ModuleType:
    """Return Torch distributed internals lazily."""
    return importlib.import_module("torch.distributed.distributed_c10d")


class TorchSymmetricMemoryHandler(_legacy.TorchSymmetricMemoryHandler):
    """Bind the process-wide ACLSHMEM runtime to one explicit rank group."""

    _is_init = False
    _group = None
    _group_ranks: tuple[int, ...] | None = None
    _rank: int | None = None
    _world_size: int | None = None
    _endpoint: str | None = None
    _init_generation = 0

    @classmethod
    def _resolve_group(cls, group: Any | None) -> tuple[int, int, tuple[int, ...]]:
        """Resolve group-local rank, size and ordered global ranks."""
        dist = _get_torch_distributed()

        if not dist.is_initialized():
            if group is not None:
                raise RuntimeError(
                    "a SHMEM process group cannot be selected before torch.distributed initialization."
                )
            return 0, 1, (0,)
        if group is None:
            world_size = dist.get_world_size()
            return dist.get_rank(), world_size, tuple(range(world_size))
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        get_group_ranks = getattr(dist, "get_process_group_ranks", None)
        if get_group_ranks is not None:
            group_ranks = tuple(get_group_ranks(group))
        else:
            group_ranks = tuple(
                dist.get_global_rank(group, group_rank)
                for group_rank in range(world_size)
            )
        return rank, world_size, group_ranks

    @staticmethod
    def _resolve_endpoint_host(host: str) -> tuple[str, str]:
        """Resolve a configured host into an explicit IP endpoint."""
        unbracketed = (
            host[1:-1] if host.startswith("[") and host.endswith("]") else host
        )
        try:
            socket.inet_pton(socket.AF_INET, unbracketed)
            return "tcp", unbracketed
        except OSError:
            pass
        try:
            socket.inet_pton(socket.AF_INET6, unbracketed)
            return "tcp6", f"[{unbracketed}]"
        except OSError:
            pass
        try:
            addresses = socket.getaddrinfo(
                unbracketed,
                None,
                family=socket.AF_INET,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as error:
            raise ValueError(
                f"cannot resolve SHMEM endpoint host {host!r} to IPv4."
            ) from error
        if not addresses:
            raise ValueError(f"cannot resolve SHMEM endpoint host {host!r} to IPv4.")
        return "tcp", addresses[0][4][0]

    @staticmethod
    def _parse_endpoint(
        base_endpoint: str,
        group_ranks: tuple[int, ...],
        global_world_size: int,
    ) -> tuple[str, int]:
        """Parse the base endpoint and isolate subgroups by port."""
        prefix, separator, port_text = base_endpoint.rpartition(":")
        if not separator:
            raise ValueError(
                f"SHMEM_IP_PORT must end in a TCP port, got {base_endpoint!r}."
            )
        try:
            port = int(port_text)
        except ValueError as error:
            raise ValueError(
                f"SHMEM_IP_PORT must end in a numeric TCP port, got {base_endpoint!r}."
            ) from error
        isolated_port = (
            port if len(group_ranks) == global_world_size else port + min(group_ranks)
        )
        if isolated_port <= 0 or isolated_port > 65535:
            raise ValueError(
                f"derived SHMEM subgroup port is invalid: {isolated_port}."
            )
        scheme, scheme_separator, host = prefix.partition("://")
        if not scheme_separator or scheme not in {"tcp", "tcp6"} or not host:
            raise ValueError(
                f"SHMEM_IP_PORT must use tcp:// or tcp6://, got {base_endpoint!r}."
            )
        return host, isolated_port

    @staticmethod
    def _worker_hosts() -> tuple[str, ...]:
        """Return configured worker hosts, if the launcher provides them."""
        return tuple(
            host.strip()
            for host in os.getenv("VC_WORKER_HOSTS", "").split(",")
            if host.strip()
        )

    @staticmethod
    def _local_world_size(worker_hosts: tuple[str, ...], global_world_size: int) -> int:
        """Resolve and validate ranks per worker host."""
        configured = (
            os.getenv("LOCAL_WORLD_SIZE", "").strip()
            or os.getenv("GPUS_PER_NODE", "").strip()
        )
        if configured:
            try:
                local_world_size = int(configured)
            except ValueError as error:
                raise ValueError(
                    f"LOCAL_WORLD_SIZE/GPUS_PER_NODE must be positive, got {configured!r}."
                ) from error
        else:
            if global_world_size % len(worker_hosts):
                raise ValueError(
                    f"world size {global_world_size} is not divisible by worker host count "
                    f"{len(worker_hosts)}."
                )
            local_world_size = global_world_size // len(worker_hosts)
        if (
            local_world_size <= 0
            or local_world_size * len(worker_hosts) != global_world_size
        ):
            raise ValueError(
                "LOCAL_WORLD_SIZE and VC_WORKER_HOSTS do not match global world size."
            )
        return local_world_size

    @classmethod
    def _group_endpoint(
        cls,
        base_endpoint: str,
        group_ranks: tuple[int, ...],
        global_world_size: int,
    ) -> str:
        """Derive a deterministic endpoint for this process subgroup."""
        host, port = cls._parse_endpoint(base_endpoint, group_ranks, global_world_size)
        worker_hosts = cls._worker_hosts()
        if len(group_ranks) != global_world_size and worker_hosts:
            local_world_size = cls._local_world_size(worker_hosts, global_world_size)
            root_node = min(group_ranks) // local_world_size
            if root_node >= len(worker_hosts):
                raise ValueError(
                    "SHMEM subgroup root does not map to a configured worker host."
                )
            host = worker_hosts[root_node]
        scheme, resolved_host = cls._resolve_endpoint_host(host)
        endpoint = f"{scheme}://{resolved_host}:{port}"
        if len(endpoint.encode("utf-8")) >= 64:
            raise ValueError(f"resolved SHMEM endpoint exceeds 63 bytes: {endpoint!r}.")
        return endpoint

    @classmethod
    def _default_store(cls) -> Any:
        """Return Torch's default TCPStore for bootstrap-only rendezvous."""
        try:
            c10d = _get_torch_distributed_c10d()
            get_default_store = vars(c10d)["_get_default_store"]
        except (ImportError, KeyError) as error:
            raise RuntimeError(
                "PyTorch does not expose the default store required for ACLSHMEM bootstrap."
            ) from error
        return get_default_store()

    @classmethod
    def _store_barrier(
        cls,
        rank: int,
        world_size: int,
        group_ranks: tuple[int, ...],
        phase: str,
    ) -> None:
        """Rendezvous without launching HCCL while ACLSHMEM bootstraps."""
        prefix = (
            "hyper_parallel/aclshmem_init/"
            f"{min(group_ranks)}_{world_size}/{cls._init_generation}/{phase}"
        )
        store = cls._default_store()
        store.set(f"{prefix}/{rank}", b"1")
        store.wait([f"{prefix}/{peer}" for peer in range(world_size)])

    @classmethod
    def _serialize_subgroups(
        cls,
        rank: int,
        world_size: int,
        group_ranks: tuple[int, ...],
        global_world_size: int,
        phase: str,
    ) -> None:
        """Serialize disjoint subgroup communicator construction."""
        group_base = min(group_ranks)
        prefix = (
            "hyper_parallel/aclshmem_init/ordered_subgroups/"
            f"{global_world_size}_{world_size}/{cls._init_generation}"
        )
        store = cls._default_store()
        if phase == "wait_turn":
            if group_base:
                store.wait([f"{prefix}/group_complete/{group_base - world_size}"])
        elif phase == "mark_complete":
            if rank == 0:
                store.set(f"{prefix}/group_complete/{group_base}", b"1")
        elif phase == "wait_all":
            store.wait([f"{prefix}/group_complete/{global_world_size - world_size}"])
        else:
            raise ValueError(f"unsupported subgroup bootstrap phase {phase!r}.")

    @classmethod
    def initialize(cls, group: Any | None = None) -> None:
        """Initialize ACLSHMEM for exactly the ranks in ``group``."""
        # pylint: disable=protected-access
        torch = _get_torch()
        dist = _get_torch_distributed()

        rank, world_size, group_ranks = cls._resolve_group(group)
        if cls._is_init:
            if cls._group_ranks != group_ranks:
                raise RuntimeError(
                    "symmetric memory is already initialized for ranks "
                    f"{cls._group_ranks}, cannot bind ranks {group_ranks}."
                )
            return
        group_base = min(group_ranks)
        expected_ranks = tuple(range(group_base, group_base + world_size))
        if group_ranks != expected_ranks or group_base % world_size:
            raise ValueError(
                "MegaMoe SHMEM isolation requires contiguous, EP-aligned global ranks; "
                f"got {group_ranks}."
            )
        global_world_size = (
            dist.get_world_size() if dist.is_initialized() else world_size
        )
        if global_world_size % world_size:
            raise ValueError(
                f"global world size {global_world_size} is not divisible by SHMEM size {world_size}."
            )
        endpoint = cls._group_endpoint(
            os.getenv("SHMEM_IP_PORT", "tcp://127.0.0.1:8662"),
            group_ranks,
            global_world_size,
        )
        heap_size = int(os.getenv("SYMMETRIC_MEMORY_HEAP_SIZE", str(1024**3)))
        if dist.is_initialized():
            cls._store_barrier(rank, world_size, group_ranks, "before")
            if world_size != global_world_size:
                cls._serialize_subgroups(
                    rank, world_size, group_ranks, global_world_size, "wait_turn"
                )
        status = _legacy._manager.attr_init(rank, world_size, heap_size, endpoint)
        if status != 0:
            raise RuntimeError(f"aclshmem initialization failed with status {status}.")
        if dist.is_initialized():
            cls._store_barrier(rank, world_size, group_ranks, "after")
            if world_size != global_world_size:
                cls._serialize_subgroups(
                    rank, world_size, group_ranks, global_world_size, "mark_complete"
                )
                cls._serialize_subgroups(
                    rank, world_size, group_ranks, global_world_size, "wait_all"
                )
        cls.comm_streams = [torch.npu.Stream() for _ in range(min(world_size, 16))]
        cls.compute_streams = [torch.npu.Stream() for _ in range(world_size)]
        cls._group = group
        cls._group_ranks = group_ranks
        cls._rank = rank
        cls._world_size = world_size
        cls._endpoint = endpoint
        cls._is_init = True
        cls._init_generation += 1
        _legacy.TorchSymmetricMemoryHandler._is_init = True
        _legacy.TorchSymmetricMemoryHandler.comm_streams = cls.comm_streams
        _legacy.TorchSymmetricMemoryHandler.compute_streams = cls.compute_streams

    @classmethod
    def _init_shmem(cls) -> None:
        """Initialize the default process group for legacy callers."""
        cls.initialize(None)

    @classmethod
    def empty(cls, shape: Any, dtype: Any) -> Any:
        """Allocate a tensor from the initialized symmetric heap."""
        if not cls._is_init:
            cls.initialize(None)
        normalized_shape = [shape] if isinstance(shape, int) else list(shape)
        return _legacy._manager.malloc(normalized_shape, dtype)  # pylint: disable=protected-access

    @classmethod
    def aligned_empty(cls, shape: Any, dtype: Any, alignment: int) -> Any:
        """Allocate a symmetric tensor with an aligned base address."""
        if (
            not isinstance(alignment, int)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise ValueError(
                f"alignment must be a positive power of two, got {alignment}."
            )
        if not cls._is_init:
            cls.initialize(None)
        normalized_shape = [shape] if isinstance(shape, int) else list(shape)
        allocator = getattr(_legacy._manager, "aligned_malloc", None)  # pylint: disable=protected-access
        if not callable(allocator):
            raise RuntimeError(  # noqa: TRY004
                "the loaded SHMEM adapter does not provide aligned allocation."
            )
        return allocator(normalized_shape, dtype, alignment)  # pylint: disable=not-callable

    @staticmethod
    def free(tensor: Any) -> None:
        """Release one tensor allocated from the symmetric heap."""
        _legacy._manager.free(tensor)  # pylint: disable=protected-access

    @classmethod
    def barrier(cls) -> None:
        """Synchronize ranks in the selected SHMEM group."""
        dist = _get_torch_distributed()

        if dist.is_initialized() and cls._world_size != 1:
            dist.barrier(group=cls._group)

    @staticmethod
    def rendezvous(tensor: Any, group: Any) -> Any:
        """Preserve the legacy unsupported rendezvous contract.

        Args:
            tensor: Tensor that would participate in rendezvous.
            group: Process group that would own the allocation.

        Returns:
            The legacy rendezvous result when the backend supports it.
        """
        return _legacy.TorchSymmetricMemoryHandler.rendezvous(tensor, group)

    @staticmethod
    def set_signal_pad_size(size: int) -> None:
        """Preserve the legacy unsupported signal-pad setter contract.

        Args:
            size: Requested signal pad size in bytes.
        """
        _legacy.TorchSymmetricMemoryHandler.set_signal_pad_size(size)

    @staticmethod
    def get_signal_pad_size() -> int:
        """Preserve the legacy unsupported signal-pad getter contract.

        Returns:
            The configured signal pad size when the backend supports it.
        """
        return _legacy.TorchSymmetricMemoryHandler.get_signal_pad_size()
