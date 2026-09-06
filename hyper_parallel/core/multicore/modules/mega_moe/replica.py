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
"""Bounded Torch weight replicas and owner-directed gradient reduction."""

from __future__ import annotations

from importlib import import_module

import struct
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

from hyper_parallel.core.multicore.modules.mega_moe.placement import (
    ExpertPlacementPlan,
)

from .spec import MegaMoeSpec

torch = import_module("torch")
dist = import_module("torch.distributed")


_PHYSICAL_WEIGHT_FUNCTION = None


@dataclass(frozen=True)
class _P2PTransfer:
    """One rank-local tensor send or receive in a deterministic batch."""

    operation: Literal["send", "receive"]
    tensor: Any
    peer_rank: int


@dataclass(frozen=True)
class _PhysicalWeightCacheKey:
    """Binding and parameter state represented by one physical pool."""

    slot_map: tuple[tuple[int, ...], ...]
    weight1_identity: int
    weight1_data_ptr: int
    weight1_version: int
    weight2_identity: int
    weight2_data_ptr: int
    weight2_version: int


@dataclass
class _PhysicalWeightPool:
    """Persistent physical weight storage populated for one cache key."""

    key: _PhysicalWeightCacheKey
    cache_id: tuple[int, int]
    home_weight1_ref: Any
    home_weight2_ref: Any
    weight1: Any
    weight2: Any
    aliases_home: bool = False
    ready: bool = False
    materializations: tuple[Any, ...] = ()


def _global_peer_rank(dist: Any, group: Any | None, group_rank: int) -> int:
    """Translate an EP-group rank to the global rank expected by P2POp."""
    return group_rank if group is None else dist.get_global_rank(group, group_rank)


def _launch_transfers(
    transfers: tuple[_P2PTransfer, ...],
    group: Any | None,
) -> tuple[Any, ...]:
    """Launch a deterministic P2P batch and return its completion handles."""
    if not transfers:
        return ()
    operations = []
    for transfer in transfers:
        operation = dist.isend if transfer.operation == "send" else dist.irecv
        operations.append(
            dist.P2POp(
                operation,
                transfer.tensor,
                _global_peer_rank(dist, group, transfer.peer_rank),
                group,
            )
        )
    return tuple(dist.batch_isend_irecv(operations) or ())


def _wait_handles(handles: tuple[Any, ...]) -> None:
    """Wait before a received tensor is consumed or a send buffer is released."""
    for handle in handles:
        handle.wait()


@dataclass
class PendingExpertWeights:
    """Physical expert weights whose remote replica transfers may be in flight."""

    manager: ExpertReplicaManager
    plan: ExpertPlacementPlan
    home_weight1: Any
    home_weight2: Any
    physical_weight1: Any
    physical_weight2: Any
    generation: int
    _handles: tuple[Any, ...]
    _transfers: tuple[_P2PTransfer, ...]
    _pool: _PhysicalWeightPool
    _materialized: bool = False

    def _complete_transfers(self) -> None:
        """Wait for this pool and publish it as reusable on every rank."""
        try:
            _wait_handles(self._handles)
        except Exception:
            self.manager._invalidate_pool(self._pool)  # pylint: disable=W0212
            raise
        self._handles = ()
        self._transfers = ()
        self.manager._mark_pool_ready(self._pool)  # pylint: disable=W0212

    def wait(self) -> tuple[Any, Any]:
        """Wait for prefetch and bind owner-directed autograd semantics."""
        if self._materialized:
            raise RuntimeError("MegaMoe replica weights can be materialized only once.")
        self._complete_transfers()
        self._materialized = True
        function = _get_physical_weight_function()
        physical_weights = function.apply(
            self.home_weight1,
            self.home_weight2,
            self,
        )
        self.manager._track_materialization(  # pylint: disable=W0212
            self._pool,
            physical_weights,
        )
        return physical_weights

    def drain(self) -> None:
        """Complete outstanding sends and receives without creating autograd state."""
        self._complete_transfers()


@dataclass
class SharedExpertWeightLease:
    """Exclusive, event-ordered use of replayable B-slot weights and gradients.

    Only the manager and immutable plan may be retained for backward. The
    returned weights are scratch and must never be saved by autograd.
    """

    manager: ExpertReplicaManager
    plan: ExpertPlacementPlan
    weights: tuple[Any, Any]
    handles: tuple[Any, ...]
    transfers: tuple[Any, ...]
    released: bool = False
    ready: bool = False

    def wait(self) -> tuple[Any, Any]:
        """Order expert reads after all prefetch transfers."""
        if self.released:
            raise RuntimeError("shared expert weight lease has been released.")
        if self.ready:
            return self.weights
        _wait_handles(self.handles)
        self.handles = ()
        self.transfers = ()
        # Replica pointers are carried in an opaque plan descriptor, not
        # ordinary operator tensor inputs. Protect their allocator lifetime on
        # the consuming stream even if the resource group is subsequently freed.
        for weight in self.weights:
            if weight.device.type != "cpu":
                weight.record_stream(torch.npu.current_stream(weight.device))
        self.ready = True
        return self.weights

    def drain(self) -> None:
        """Finish abandoned route prefetch and release its staging lease."""
        self.release()

    def runtime_config(self, base_config: Any, *, backward: bool = False) -> Any:
        """Bind the stable suffix addresses to one leased plan descriptor."""
        self.wait()
        return self.manager.shared_runtime_config(base_config, backward=backward)

    def prepare_backward(self, *, multi_wave: bool) -> tuple[Any, Any]:
        """Prepare reusable replica-gradient outputs without home storage."""
        self.wait()
        return self.manager.prepare_shared_gradients(multi_wave=multi_wave)

    def release(self) -> None:
        """Record completion after the last consumer before allowing reuse."""
        if self.released:
            return
        self.wait()
        self.manager.release_shared_pool()
        self.released = True


class ExpertReplicaManager:
    """Build physical weight pools and return replica gradients to home ranks."""

    def __init__(
        self,
        spec: MegaMoeSpec,
        pool_cache_size: int = 1,
        *,
        compact_physical_weights: bool = False,
    ) -> None:
        """Initialize empty placement-generation state."""
        if spec.replica_slots_per_rank <= 0:
            raise ValueError("ExpertReplicaManager requires at least one replica slot.")
        if (
            not isinstance(pool_cache_size, int)
            or isinstance(pool_cache_size, bool)
            or pool_cache_size <= 0
        ):
            raise ValueError(
                f"pool_cache_size must be a positive integer, got {pool_cache_size!r}."
            )
        if not isinstance(compact_physical_weights, bool):
            raise ValueError(
                "compact_physical_weights must be a bool, got "
                f"{compact_physical_weights!r}."
            )
        self.spec = spec
        self._generation = 0
        self._slot_map: tuple[tuple[int, ...], ...] | None = None
        self._pool_cache_size = pool_cache_size
        self._compact_physical_weights = compact_physical_weights
        self._physical_pools: OrderedDict[
            tuple[int, int], _PhysicalWeightPool
        ] = OrderedDict()
        self._shared_weights: tuple[Any, Any] | None = None
        self._shared_gradients: tuple[Any, Any] | None = None
        self._shared_runtime_configs: dict[tuple[int, bool], tuple[Any, Any]] = {}
        self._shared_lock = threading.Lock()
        self._shared_in_use = False
        self._shared_event: Any | None = None
        self._shared_failed = False

    def start_shared_prefetch(
        self,
        weight1: Any,
        weight2: Any,
        plan: ExpertPlacementPlan,
    ) -> SharedExpertWeightLease:
        """Prefetch only B replicas without copying or rebinding home weights.

        Callers must release the returned lease after all kernel consumers
        have been enqueued. Backward must acquire a fresh lease with the
        saved plan and unchanged home parameters, not save these scratch views.
        """
        self._validate_plan(plan)
        self._validate_home_weights(weight1, weight2)
        with self._shared_lock:
            if self._shared_failed:
                raise RuntimeError("shared expert weight pool failed and cannot be reused.")
            if self._shared_in_use:
                raise RuntimeError("shared expert weight pool does not support concurrent calls.")
            self._shared_in_use = True
        try:
            if self._shared_weights is None:
                self._shared_weights = tuple(
                    torch.empty(
                        (self.spec.replica_slots_per_rank, *weight.shape[1:]),
                        dtype=weight.dtype,
                        device=weight.device,
                    )
                    for weight in (weight1, weight2)
                )
            replica_weight1, replica_weight2 = self._shared_weights
            if replica_weight1.device != weight1.device or replica_weight1.dtype != weight1.dtype:
                raise ValueError("shared expert weight pool cannot change device or dtype.")
            if any(
                tuple(pool.shape[1:]) != tuple(home.shape[1:])
                for pool, home in zip(self._shared_weights, (weight1, weight2))
            ):
                raise ValueError("shared expert weight pool cannot change matrix shapes.")
            if self._shared_event is not None:
                torch.npu.current_stream(weight1.device).wait_event(self._shared_event)
            transfers = self._prefetch_transfers(
                plan, weight1, weight2, replica_weight1, replica_weight2,
                pool_start=self.spec.local_experts,
            )
            handles = _launch_transfers(transfers, self.spec.ep_group)
            return SharedExpertWeightLease(self, plan, self._shared_weights, handles, transfers)
        except Exception:
            # A failed P2P launch may have enqueued only part of the batch.
            # Reject further use instead of overwriting potentially live DMA.
            with self._shared_lock:
                self._shared_failed = True
                self._shared_in_use = False
            raise

    def prepare_shared_gradients(self, *, multi_wave: bool) -> tuple[Any, Any]:
        """Return FP32 B-slot partials, cleared before multi-wave accumulation."""
        if not self._shared_in_use or self._shared_weights is None:
            raise RuntimeError("shared expert weight pool is not claimed.")
        if self._shared_gradients is None:
            self._shared_gradients = tuple(
                torch.empty_like(weight, dtype=torch.float32) for weight in self._shared_weights
            )
        if multi_wave:
            for gradient in self._shared_gradients:
                gradient.zero_()
        for gradient in self._shared_gradients:
            if gradient.device.type != "cpu":
                gradient.record_stream(torch.npu.current_stream(gradient.device))
        return self._shared_gradients

    def shared_runtime_config(self, base_config: Any, *, backward: bool) -> Any:
        """Cache a versioned suffix-address extension for the native scheduler.

        Home addresses remain ordinary operator inputs. Only resource-group
        storage appears in this table, so it is immutable across layer changes.
        Its first device copy and all consumers are ordered by the pool lease.
        """
        if not self._shared_in_use or self._shared_weights is None:
            raise RuntimeError("shared expert weight pool is not claimed.")
        if backward and self._shared_gradients is None:
            raise RuntimeError("shared expert gradients are not initialized.")
        key = (id(base_config), backward)
        if key not in self._shared_runtime_configs:
            gradients = self._shared_gradients if backward else None
            # The compact RuntimeConfig wire image is 8-byte aligned. The
            # native reader locates this extension after ReadyHandshakeMeta.
            if (
                base_config.dtype != torch.uint8 or base_config.ndim != 1
                or not base_config.is_contiguous() or base_config.numel() % 8
                or base_config.device != self._shared_weights[0].device
            ):
                raise ValueError("split-weight plan descriptor must be aligned device uint8 storage.")
            gradient_ptrs = (0, 0) if gradients is None else tuple(gradient.data_ptr() for gradient in gradients)
            data = struct.pack(
                "<II4Q", 0x53505754, 1,
                *(weight.data_ptr() for weight in self._shared_weights),
                *gradient_ptrs,
            )
            metadata = torch.tensor(list(data), dtype=torch.uint8, device=base_config.device)
            # Retain the base object as well, preventing Python id reuse from
            # ever matching an unrelated descriptor after plan replacement.
            self._shared_runtime_configs[key] = (base_config, torch.cat((base_config, metadata)))
        runtime_config = self._shared_runtime_configs[key][1]
        if runtime_config.device.type != "cpu":
            runtime_config.record_stream(torch.npu.current_stream(runtime_config.device))
        return runtime_config

    def release_shared_pool(self) -> None:
        """Order the next lease after this stream's last staging consumer."""
        with self._shared_lock:
            if self._shared_weights is None or not self._shared_in_use:
                raise RuntimeError("shared expert weight pool is not claimed.")
            device = self._shared_weights[0].device
            if device.type != "cpu":
                if self._shared_event is None:
                    self._shared_event = torch.npu.Event()
                self._shared_event.record(torch.npu.current_stream(device))
            self._shared_in_use = False

    @property
    def generation(self) -> int:
        """Return the latest physical slot-binding generation."""
        return self._generation

    def _validate_plan(self, plan: ExpertPlacementPlan) -> None:
        """Require a placement built for this plan's fixed physical shape."""
        expected = (
            self.spec.ep_size,
            self.spec.num_experts,
            self.spec.replica_slots_per_rank,
        )
        actual = (
            plan.ep_size,
            plan.logical_num_experts,
            plan.replica_slots_per_rank,
        )
        if actual != expected:
            raise ValueError(
                "replica placement topology does not match the MegaMoe plan: "
                f"expected {expected}, got {actual}."
            )

    def _validate_home_weights(self, weight1: Any, weight2: Any) -> None:
        """Require equally sized rank-local home expert tensors."""
        if weight1.ndim != 3 or weight2.ndim != 3:
            raise ValueError("MegaMoe home expert weights must be three-dimensional.")
        if (
            weight1.shape[0] != self.spec.local_experts
            or weight2.shape[0] != self.spec.local_experts
        ):
            raise ValueError(
                "MegaMoe home expert weight count must match the plan's "
                f"{self.spec.local_experts} local experts."
            )
        if weight1.device != weight2.device or weight1.dtype != weight2.dtype:
            raise ValueError("MegaMoe home expert weights must share device and dtype.")

    def _update_generation(self, plan: ExpertPlacementPlan) -> int:
        """Advance the generation only when a physical slot changes binding."""
        if self._slot_map != plan.slot_to_logical_expert:
            self._slot_map = plan.slot_to_logical_expert
            self._generation += 1
        return self._generation

    @staticmethod
    def _cache_key(
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
    ) -> _PhysicalWeightCacheKey:
        """Describe weights whose immutable physical materialization can be shared."""
        return _PhysicalWeightCacheKey(
            slot_map=plan.slot_to_logical_expert,
            weight1_identity=id(weight1),
            weight1_data_ptr=weight1.data_ptr(),
            weight1_version=int(getattr(weight1, "_version")),
            weight2_identity=id(weight2),
            weight2_data_ptr=weight2.data_ptr(),
            weight2_version=int(getattr(weight2, "_version")),
        )

    def _all_ranks_can_reuse(self, locally_reusable: bool, weight: Any) -> bool:
        """Require every EP rank to make the same cache/P2P protocol decision."""
        if self.spec.ep_size == 1:
            return locally_reusable
        if not dist.is_available() or not dist.is_initialized():
            return locally_reusable
        reuse_flag = torch.tensor(
            int(locally_reusable),
            dtype=torch.int32,
            device=weight.device,
        )
        dist.all_reduce(
            reuse_flag,
            op=dist.ReduceOp.MIN,
            group=self.spec.ep_group,
        )
        return bool(reuse_flag.item())

    @staticmethod
    def _pool_cache_id(weight1: Any, weight2: Any) -> tuple[int, int]:
        """Identify one optimizer-owned expert parameter pair."""
        return id(weight1), id(weight2)

    def _cached_pool(
        self,
        weight1: Any,
        weight2: Any,
    ) -> _PhysicalWeightPool | None:
        """Return the most recent pool owned by this exact parameter pair."""
        cache_id = self._pool_cache_id(weight1, weight2)
        pool = self._physical_pools.get(cache_id)
        if pool is None:
            return None
        if (
            pool.home_weight1_ref() is not weight1
            or pool.home_weight2_ref() is not weight2
        ):
            self._physical_pools.pop(cache_id, None)
            return None
        self._physical_pools.move_to_end(cache_id)
        return pool

    def _store_pool(self, pool: _PhysicalWeightPool) -> None:
        """Publish one pool and enforce the shared-layer cache bound."""
        self._physical_pools[pool.cache_id] = pool
        self._physical_pools.move_to_end(pool.cache_id)
        while len(self._physical_pools) > self._pool_cache_size:
            self._physical_pools.popitem(last=False)

    def _is_current_pool(self, pool: _PhysicalWeightPool) -> bool:
        """Return whether the cache still maps this parameter pair to ``pool``."""
        return self._physical_pools.get(pool.cache_id) is pool

    def _mark_pool_ready(self, pool: _PhysicalWeightPool) -> None:
        """Publish a completed pool only if it is still the current candidate."""
        if self._is_current_pool(pool):
            if pool.aliases_home:
                weight1 = pool.home_weight1_ref()
                weight2 = pool.home_weight2_ref()
                if weight1 is not None and weight2 is not None:
                    pool.key = _PhysicalWeightCacheKey(
                        slot_map=pool.key.slot_map,
                        weight1_identity=id(weight1),
                        weight1_data_ptr=weight1.data_ptr(),
                        weight1_version=int(getattr(weight1, "_version")),
                        weight2_identity=id(weight2),
                        weight2_data_ptr=weight2.data_ptr(),
                        weight2_version=int(getattr(weight2, "_version")),
                    )
            pool.ready = True

    def _invalidate_pool(self, pool: _PhysicalWeightPool) -> None:
        """Drop a failed current pool without disturbing an intervening generation."""
        if self._is_current_pool(pool):
            self._physical_pools.pop(pool.cache_id)

    @staticmethod
    def _track_materialization(
        pool: _PhysicalWeightPool,
        physical_weights: tuple[Any, Any],
    ) -> None:
        """Retain weak leases while an autograd graph can read this storage."""
        live_materializations = tuple(
            reference
            for reference in pool.materializations
            if reference() is not None
        )
        pool.materializations = live_materializations + tuple(
            weakref.ref(weight) for weight in physical_weights
        )

    @staticmethod
    def _aliases_home_parameter_storage(
        pool: _PhysicalWeightPool,
        weight1: Any,
        weight2: Any,
    ) -> bool:
        """Return whether a compact pool still backs both home parameters."""
        return (
            pool.aliases_home
            and pool.home_weight1_ref() is weight1
            and pool.home_weight2_ref() is weight2
            and pool.weight1.data_ptr() == weight1.data_ptr()
            and pool.weight2.data_ptr() == weight2.data_ptr()
        )

    @staticmethod
    def _can_refresh_pool(
        pool: _PhysicalWeightPool,
        weight1: Any,
        weight2: Any,
    ) -> bool:
        """Return whether an inactive aliased pool can accept new replicas."""
        pool.materializations = tuple(
            reference
            for reference in pool.materializations
            if reference() is not None
        )
        return (
            pool.ready
            and not pool.materializations
            and pool.home_weight1_ref() is weight1
            and pool.home_weight2_ref() is weight2
            and (
                not pool.aliases_home
                or ExpertReplicaManager._aliases_home_parameter_storage(
                    pool,
                    weight1,
                    weight2,
                )
            )
        )

    def _allocate_compact_pool(
        self,
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
    ) -> _PhysicalWeightPool:
        """Attach replica suffix storage directly behind optimizer-owned rows."""
        shape1 = (self.spec.physical_local_experts, *weight1.shape[1:])
        shape2 = (self.spec.physical_local_experts, *weight2.shape[1:])
        backing_weight1 = torch.empty(shape1, dtype=weight1.dtype, device=weight1.device)
        backing_weight2 = torch.empty(shape2, dtype=weight2.dtype, device=weight2.device)
        with torch.no_grad():
            backing_weight1[: self.spec.local_experts].copy_(weight1)
            backing_weight2[: self.spec.local_experts].copy_(weight2)
            weight1.set_(
                backing_weight1.untyped_storage(),
                0,
                tuple(weight1.shape),
                tuple(backing_weight1.stride()),
            )
            weight2.set_(
                backing_weight2.untyped_storage(),
                0,
                tuple(weight2.shape),
                tuple(backing_weight2.stride()),
            )
        # Build the full views from the parameters after ``set_`` so optimizer
        # updates and saved physical tensors share one version counter.
        physical_weight1 = weight1.detach().as_strided(shape1, backing_weight1.stride())
        physical_weight2 = weight2.detach().as_strided(shape2, backing_weight2.stride())
        return _PhysicalWeightPool(
            key=self._cache_key(plan, weight1, weight2),
            cache_id=self._pool_cache_id(weight1, weight2),
            home_weight1_ref=weakref.ref(weight1),
            home_weight2_ref=weakref.ref(weight2),
            weight1=physical_weight1,
            weight2=physical_weight2,
            aliases_home=True,
        )

    def _allocate_snapshot_pool(
        self,
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
    ) -> _PhysicalWeightPool:
        """Allocate a pool isolated from optimizer-owned home parameters."""
        physical_weight1 = torch.empty(
            (self.spec.physical_local_experts, *weight1.shape[1:]),
            dtype=weight1.dtype,
            device=weight1.device,
        )
        physical_weight2 = torch.empty(
            (self.spec.physical_local_experts, *weight2.shape[1:]),
            dtype=weight2.dtype,
            device=weight2.device,
        )
        # Home rows are overwritten below and every active replica row is an
        # asynchronous receive target. Unbound suffix rows have zero route
        # counts, so initializing the complete pool would only add bandwidth.
        with torch.no_grad():
            physical_weight1[: self.spec.local_experts].copy_(weight1)
            physical_weight2[: self.spec.local_experts].copy_(weight2)
        return _PhysicalWeightPool(
            key=self._cache_key(plan, weight1, weight2),
            cache_id=self._pool_cache_id(weight1, weight2),
            home_weight1_ref=weakref.ref(weight1),
            home_weight2_ref=weakref.ref(weight2),
            weight1=physical_weight1,
            weight2=physical_weight2,
        )

    def _allocate_pool(
        self,
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
    ) -> _PhysicalWeightPool:
        """Allocate the configured persistent physical-weight pool."""
        if self._compact_physical_weights:
            return self._allocate_compact_pool(plan, weight1, weight2)
        return self._allocate_snapshot_pool(plan, weight1, weight2)

    def _refresh_pool(
        self,
        pool: _PhysicalWeightPool,
        key: _PhysicalWeightCacheKey,
        weight1: Any,
        weight2: Any,
    ) -> None:
        """Refresh an inactive pool after an optimizer-owned weight update."""
        if not pool.aliases_home:
            with torch.no_grad():
                pool.weight1[: self.spec.local_experts].copy_(weight1)
                pool.weight2[: self.spec.local_experts].copy_(weight2)
        pool.key = key
        pool.ready = False

    def _prefetch_transfers(
        self,
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
        physical_weight1: Any,
        physical_weight2: Any,
        *,
        pool_start: int = 0,
    ) -> tuple[_P2PTransfer, ...]:
        """Build this rank's matching weight send and receive descriptors."""
        rank_id = self.spec.rank_id
        transfers = []
        for transfer in plan.replica_transfers:
            if transfer.owner_rank == rank_id:
                tensors = (
                    weight1[transfer.owner_slot].contiguous(),
                    weight2[transfer.owner_slot].contiguous(),
                )
                transfers.extend(
                    _P2PTransfer("send", tensor, transfer.target_rank)
                    for tensor in tensors
                )
            elif transfer.target_rank == rank_id:
                tensors = (
                    physical_weight1[transfer.target_slot - pool_start],
                    physical_weight2[transfer.target_slot - pool_start],
                )
                transfers.extend(
                    _P2PTransfer("receive", tensor, transfer.owner_rank)
                    for tensor in tensors
                )
        return tuple(transfers)

    def start_prefetch(
        self,
        weight1: Any,
        weight2: Any,
        plan: ExpertPlacementPlan,
    ) -> PendingExpertWeights:
        """Launch bounded remote-weight transfers into a fixed-shaped pool.

        Args:
            weight1: Rank-local optimizer-owned gate/up weights.
            weight2: Rank-local optimizer-owned down-projection weights.
            plan: Logical-to-physical placement for this routing step.

        Returns:
            Pending physical weights. Call :meth:`PendingExpertWeights.wait`
            before launching the expert kernels.
        """
        self._validate_plan(plan)
        self._validate_home_weights(weight1, weight2)
        key = self._cache_key(plan, weight1, weight2)
        pool = self._cached_pool(weight1, weight2)
        locally_reusable = pool is not None and pool.ready and pool.key == key
        generation = self._update_generation(plan)
        if self._all_ranks_can_reuse(locally_reusable, weight1):
            return PendingExpertWeights(
                manager=self,
                plan=plan,
                home_weight1=weight1,
                home_weight2=weight2,
                physical_weight1=pool.weight1,
                physical_weight2=pool.weight2,
                generation=generation,
                _handles=(),
                _transfers=(),
                _pool=pool,
            )

        cache_pool = pool
        store_pool = True
        if pool is not None and self._can_refresh_pool(pool, weight1, weight2):
            self._refresh_pool(pool, key, weight1, weight2)
        elif pool is not None and self._aliases_home_parameter_storage(
            pool,
            weight1,
            weight2,
        ):
            # A concurrent graph or in-flight transfer still owns the aliased
            # suffix. Use isolated transient storage for this generation so an
            # internal ``set_`` cannot invalidate that graph's version counter.
            pool = self._allocate_snapshot_pool(plan, weight1, weight2)
            store_pool = False
        else:
            # A live graph retains its isolated old pool; the new generation
            # must not overwrite weights that its backward may still read.
            pool = self._allocate_pool(plan, weight1, weight2)
        if store_pool:
            self._store_pool(pool)
        transfers = self._prefetch_transfers(
            plan,
            weight1,
            weight2,
            pool.weight1,
            pool.weight2,
        )
        try:
            handles = _launch_transfers(transfers, self.spec.ep_group)
        except Exception:
            if store_pool or pool is cache_pool:
                self._invalidate_pool(pool)
            raise
        return PendingExpertWeights(
            manager=self,
            plan=plan,
            home_weight1=weight1,
            home_weight2=weight2,
            physical_weight1=pool.weight1,
            physical_weight2=pool.weight2,
            generation=generation,
            _handles=handles,
            _transfers=transfers,
            _pool=pool,
        )

    def _gradient_transfers(
        self,
        plan: ExpertPlacementPlan,
        grad_weight1: Any,
        grad_weight2: Any,
        *,
        replica_gradients: tuple[Any, Any] | None = None,
    ) -> tuple[tuple[_P2PTransfer, ...], tuple[tuple[int, Any, Any], ...]]:
        """Build replica-gradient transfers and owner accumulation records."""
        rank_id = self.spec.rank_id
        transfers = []
        received = []
        source_grad1, source_grad2 = (
            (grad_weight1, grad_weight2) if replica_gradients is None else replica_gradients
        )
        pool_start = 0 if replica_gradients is None else self.spec.local_experts
        for transfer in plan.replica_transfers:
            if transfer.target_rank == rank_id:
                tensors = (
                    source_grad1[transfer.target_slot - pool_start].contiguous(),
                    source_grad2[transfer.target_slot - pool_start].contiguous(),
                )
                transfers.extend(
                    _P2PTransfer("send", tensor, transfer.owner_rank)
                    for tensor in tensors
                )
            elif transfer.owner_rank == rank_id:
                received_weight1 = torch.empty(
                    tuple(grad_weight1.shape[1:]),
                    dtype=grad_weight1.dtype,
                    device=grad_weight1.device,
                )
                received_weight2 = torch.empty(
                    tuple(grad_weight2.shape[1:]),
                    dtype=grad_weight2.dtype,
                    device=grad_weight2.device,
                )
                transfers.extend(
                    (
                        _P2PTransfer(
                            "receive",
                            received_weight1,
                            transfer.target_rank,
                        ),
                        _P2PTransfer(
                            "receive",
                            received_weight2,
                            transfer.target_rank,
                        ),
                    )
                )
                received.append(
                    (
                        transfer.owner_slot,
                        received_weight1,
                        received_weight2,
                    )
                )
        return tuple(transfers), tuple(received)

    def reduce_replica_gradients(
        self,
        plan: ExpertPlacementPlan,
        grad_weight1: Any,
        grad_weight2: Any,
        *,
        replica_gradients: tuple[Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Return gradients for only the optimizer-owned home parameters.

        Args:
            plan: Placement that produced the physical expert gradients.
            grad_weight1: Gate/up gradients for every local physical slot.
            grad_weight2: Down-projection gradients for every local physical slot.
            replica_gradients: Optional separate B-slot outputs. When supplied,
                the first two gradient arguments contain home experts only.

        Returns:
            Owner-local gate/up and down-projection gradients.
        """
        self._validate_plan(plan)
        expected_experts = (
            self.spec.physical_local_experts if replica_gradients is None else self.spec.local_experts
        )
        if (
            grad_weight1.ndim != 3
            or grad_weight2.ndim != 3
            or grad_weight1.shape[0] != expected_experts
            or grad_weight2.shape[0] != expected_experts
        ):
            raise ValueError(
                "physical expert gradients must contain "
                f"{expected_experts} groups on each rank."
            )
        if replica_gradients is not None:
            if len(replica_gradients) != 2:
                raise ValueError("replica gradients must contain exactly W1 and W2.")
            for replica, home in zip(replica_gradients, (grad_weight1, grad_weight2)):
                if (
                    tuple(replica.shape) != (self.spec.replica_slots_per_rank, *home.shape[1:])
                    or replica.dtype != home.dtype
                    or replica.device != home.device
                ):
                    raise ValueError("replica gradients must match the B-slot shape, dtype and device.")
        local_experts = self.spec.local_experts
        # Dense physical gradients can expose their home-prefix views directly.
        # ``contiguous`` still materializes broadcast/autograd-expanded inputs
        # before the owner receives are accumulated in place.
        home_grad_weight1 = grad_weight1[:local_experts].contiguous()
        home_grad_weight2 = grad_weight2[:local_experts].contiguous()
        transfers, received = self._gradient_transfers(
            plan,
            grad_weight1,
            grad_weight2,
            replica_gradients=replica_gradients,
        )
        _wait_handles(_launch_transfers(transfers, self.spec.ep_group))
        received_by_owner: dict[int, list[tuple[Any, Any]]] = {}
        for owner_slot, received_weight1, received_weight2 in received:
            received_by_owner.setdefault(owner_slot, []).append(
                (received_weight1, received_weight2)
            )
        for owner_slot, contributions in received_by_owner.items():
            # Round only after all copies have contributed. Repeated BF16/FP16
            # in-place adds lose small gradients when replica sums cancel.
            # Process one owner at a time to bound the FP32 scratch footprint.
            for weight_index, home_grad in enumerate((home_grad_weight1, home_grad_weight2)):
                home = home_grad[owner_slot]
                accumulator = (
                    home.float()
                    if home.dtype in (torch.bfloat16, torch.float16)
                    else home
                )
                for contribution in contributions:
                    accumulator.add_(contribution[weight_index])
                home.copy_(accumulator)
        return home_grad_weight1, home_grad_weight2


def _get_physical_weight_function() -> Any:
    """Create the owner-directed physical-weight autograd bridge lazily."""
    # pylint: disable=global-statement
    global _PHYSICAL_WEIGHT_FUNCTION
    if _PHYSICAL_WEIGHT_FUNCTION is not None:
        return _PHYSICAL_WEIGHT_FUNCTION

    class _PhysicalExpertWeightFunction(  # pylint: disable=abstract-method
        torch.autograd.Function
    ):
        """Expose prefetched physical weights while retaining home ownership."""

        @staticmethod
        def forward(
            ctx: Any,
            home_weight1: Any,
            home_weight2: Any,
            pending: PendingExpertWeights,
        ) -> tuple[Any, Any]:
            """Expose the completed fixed physical pool to autograd.

            Args:
                ctx: Autograd context retaining the placement contract.
                home_weight1: Optimizer-owned gate/up parameter.
                home_weight2: Optimizer-owned down-projection parameter.
                pending: Completed physical pool and placement metadata.

            Returns:
                Autograd-bound views of the two physical weight pools.
            """
            del home_weight1, home_weight2
            ctx.manager = pending.manager
            ctx.plan = pending.plan
            # Return distinct Tensor wrappers over the persistent storage.
            # Otherwise ``Function.apply`` attaches its grad_fn directly to
            # the manager-owned cache tensor and keeps the previous graph live.
            return (
                pending.physical_weight1.detach(),
                pending.physical_weight2.detach(),
            )

        @staticmethod
        def backward(
            ctx: Any,
            grad_weight1: Any,
            grad_weight2: Any,
        ) -> tuple[Any, Any, None]:
            """Reduce replica gradients to their optimizer-owned home rows.

            Args:
                ctx: Autograd context retaining the placement contract.
                grad_weight1: Gate/up gradients for physical slots.
                grad_weight2: Down-projection gradients for physical slots.

            Returns:
                Home gate/up gradient, home down gradient, and no pending gradient.
            """
            home_grad_weight1, home_grad_weight2 = (
                ctx.manager.reduce_replica_gradients(
                    ctx.plan,
                    grad_weight1,
                    grad_weight2,
                )
            )
            return home_grad_weight1, home_grad_weight2, None

    _PHYSICAL_WEIGHT_FUNCTION = _PhysicalExpertWeightFunction
    return _PHYSICAL_WEIGHT_FUNCTION
