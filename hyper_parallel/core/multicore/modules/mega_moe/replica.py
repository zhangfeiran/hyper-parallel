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
    _materialized: bool = False

    def wait(self) -> tuple[Any, Any]:
        """Wait for prefetch and bind owner-directed autograd semantics."""
        if self._materialized:
            raise RuntimeError("MegaMoe replica weights can be materialized only once.")
        _wait_handles(self._handles)
        self._handles = ()
        self._transfers = ()
        self._materialized = True
        function = _get_physical_weight_function()
        return function.apply(self.home_weight1, self.home_weight2, self)

    def drain(self) -> None:
        """Complete outstanding sends and receives without creating autograd state."""
        _wait_handles(self._handles)
        self._handles = ()
        self._transfers = ()


class ExpertReplicaManager:
    """Build physical weight pools and return replica gradients to home ranks."""

    def __init__(self, spec: MegaMoeSpec) -> None:
        """Initialize empty placement-generation state."""
        if spec.replica_slots_per_rank <= 0:
            raise ValueError("ExpertReplicaManager requires at least one replica slot.")
        self.spec = spec
        self._generation = 0
        self._slot_map: tuple[tuple[int, ...], ...] | None = None

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

    def _prefetch_transfers(
        self,
        plan: ExpertPlacementPlan,
        weight1: Any,
        weight2: Any,
        physical_weight1: Any,
        physical_weight2: Any,
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
                    physical_weight1[transfer.target_slot],
                    physical_weight2[transfer.target_slot],
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
        physical_shape1 = (
            self.spec.physical_local_experts,
            *weight1.shape[1:],
        )
        physical_shape2 = (
            self.spec.physical_local_experts,
            *weight2.shape[1:],
        )
        physical_weight1 = torch.zeros(
            physical_shape1,
            dtype=weight1.dtype,
            device=weight1.device,
        )
        physical_weight2 = torch.zeros(
            physical_shape2,
            dtype=weight2.dtype,
            device=weight2.device,
        )
        with torch.no_grad():
            physical_weight1[: self.spec.local_experts].copy_(weight1)
            physical_weight2[: self.spec.local_experts].copy_(weight2)
        transfers = self._prefetch_transfers(
            plan,
            weight1,
            weight2,
            physical_weight1,
            physical_weight2,
        )
        return PendingExpertWeights(
            manager=self,
            plan=plan,
            home_weight1=weight1,
            home_weight2=weight2,
            physical_weight1=physical_weight1,
            physical_weight2=physical_weight2,
            generation=self._update_generation(plan),
            _handles=_launch_transfers(transfers, self.spec.ep_group),
            _transfers=transfers,
        )

    def _gradient_transfers(
        self,
        plan: ExpertPlacementPlan,
        grad_weight1: Any,
        grad_weight2: Any,
    ) -> tuple[tuple[_P2PTransfer, ...], tuple[tuple[int, Any, Any], ...]]:
        """Build replica-gradient transfers and owner accumulation records."""
        rank_id = self.spec.rank_id
        transfers = []
        received = []
        for transfer in plan.replica_transfers:
            if transfer.target_rank == rank_id:
                tensors = (
                    grad_weight1[transfer.target_slot].contiguous(),
                    grad_weight2[transfer.target_slot].contiguous(),
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
    ) -> tuple[Any, Any]:
        """Return gradients for only the optimizer-owned home parameters."""
        self._validate_plan(plan)
        expected_experts = self.spec.physical_local_experts
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
        local_experts = self.spec.local_experts
        home_grad_weight1 = grad_weight1[:local_experts].clone()
        home_grad_weight2 = grad_weight2[:local_experts].clone()
        transfers, received = self._gradient_transfers(
            plan,
            grad_weight1,
            grad_weight2,
        )
        _wait_handles(_launch_transfers(transfers, self.spec.ep_group))
        for owner_slot, received_weight1, received_weight2 in received:
            home_grad_weight1[owner_slot].add_(received_weight1)
            home_grad_weight2[owner_slot].add_(received_weight2)
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
            """Expose the completed fixed physical pool to autograd."""
            del home_weight1, home_weight2
            ctx.manager = pending.manager
            ctx.plan = pending.plan
            return pending.physical_weight1, pending.physical_weight2

        @staticmethod
        def backward(
            ctx: Any,
            grad_weight1: Any,
            grad_weight2: Any,
        ) -> tuple[Any, Any, None]:
            """Reduce replica gradients to their optimizer-owned home rows."""
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
