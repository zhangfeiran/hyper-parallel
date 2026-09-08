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
"""Shared scalar correlation context for the existing diagnostic collectors."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import socket
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any

PREFIX = "megamoe::profile|"
PACKAGE = "hyper_parallel.core.multicore.modules.mega_moe"


def new_capture_id(distributed: Any, channel: str) -> str:
    """Broadcast a fresh execution id; different diagnostic steps never share one."""
    identifiers = [f"{channel}:{uuid.uuid4().hex}" if distributed.get_rank() == 0 else None]
    distributed.broadcast_object_list(identifiers, src=0)
    return identifiers[0]


def source_id(identity: dict[str, Any]) -> str:
    """Hash the complete source/native identity recorded by the owning benchmark."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scope_name(role: str, metadata: dict[str, Any]) -> str:
    """Encode a JSON identity in a Chrome CPU scope without process addresses."""
    if not isinstance(role, str) or not role or "role" in metadata:
        raise ValueError("profile scopes require a role separate from metadata")
    if not isinstance(metadata.get("capture_id"), str) or not metadata["capture_id"]:
        raise ValueError("profile scopes require a nonempty capture_id")
    for name in ("rank", "world_size", "step"):
        if type(metadata.get(name)) is not int or metadata[name] < 0:  # pylint: disable=unidiomatic-typecheck
            raise ValueError(f"profile scope has an invalid {name}")
    if metadata["rank"] >= metadata["world_size"]:
        raise ValueError("profile scope rank must be smaller than world_size")
    return PREFIX + json.dumps({"role": role, **metadata}, sort_keys=True, separators=(",", ":"))


def parse_scope_name(name: str) -> dict[str, Any] | None:
    """Decode tagged scopes, rejecting malformed data and preserving old names."""
    if not name.startswith(PREFIX):
        return None
    metadata = json.loads(name[len(PREFIX):])
    role = metadata.pop("role")
    scope_name(role, metadata)
    return {"role": role, **metadata}


def placement_metadata(manager: Any, plan: Any) -> dict[str, Any]:
    """Identify the actual immutable placement, not a mutable tensor address."""
    payload = {"slots": plan.slot_to_logical_expert, "counts": plan.dispatch_counts}
    return {
        "placement_id": hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest(),
        "placement_generation": manager.generation,
        "logical_experts": list(plan.slot_to_logical_expert[manager.spec.rank_id]),
    }


class ProfileContext:
    """Correlate one extra execution while keeping device tensors out of metadata."""

    def __init__(
        self, *, rank: int, world_size: int, capture_id: str, record_function: Any = None, step: int = 0,
        source: dict[str, Any] | None = None,
    ) -> None:
        """Create a tensor-free context for one explicitly requested diagnostic execution."""
        self.base = {"capture_id": capture_id, "rank": rank, "world_size": world_size, "step": step}
        if source is not None:
            self.base["source_id"] = source_id(source)
        scope_name("capture", self.base)
        self.record_function = record_function
        self.layer_registry: list[dict[str, int]] = []
        self.records: list[dict[str, Any]] = []
        self.last_launch: dict[tuple[int, str], dict[str, Any]] = {}
        self._state: ContextVar[dict[str, Any]] = ContextVar("megamoe_profile_context", default=self.base)
        self._layers: dict[int, dict[str, int]] = {}
        self._counts: Counter = Counter()

    def metadata(self) -> dict[str, Any]:
        """Return a copy so callers cannot alter an enclosing operation."""
        return dict(self._state.get())

    def next_index(self, *key: Any) -> int:
        """Number invocations within this capture, including locally empty batches."""
        value = self._counts[key]
        self._counts[key] += 1
        return value

    def bind_layers(self, layers: Any) -> None:
        """Assign resource group ids by first layer occurrence, equally on each rank."""
        if self.layer_registry:
            raise ValueError("profile layers are already registered")
        groups = {}
        for index, layer in enumerate(layers):
            group = id(getattr(layer, "_resource_group", layer))
            if group not in groups:
                groups[group] = len(groups)
            entry = {"layer": index, "resource_group": groups[group]}
            self.layer_registry.append(entry)
            self._layers[id(layer)] = entry

    @contextmanager
    def span(self, role: str, **fields: Any) -> Iterator[None]:
        """Annotate a nested operation and restore context even after an exception."""
        metadata = self.metadata() | fields
        name = scope_name(role, metadata)
        token = self._state.set(metadata)
        self.records.append({"role": role, **metadata})
        try:
            with self.record_function(name) if self.record_function else nullcontext():
                yield
        finally:
            self._state.reset(token)

    def manifest(self) -> dict[str, Any]:
        """Describe host identities and task coverage without claiming clock alignment."""
        return {
            "schema_version": 2, **self.base, "host": socket.gethostname(),
            "layers": self.layer_registry, "operations": self.records,
            "last_launches": list(self.last_launch.values()),
            "clock_domains": {
                "trace": "CANN host-aligned monotonic_raw_ns; comparable only on the same host",
                "task": "rank-local device system_cycle; no cross-rank or trace alignment",
                "p2p_events": "elapsed_ms on the recorded stream; no absolute origin",
            },
        }

    def _module_forward(self, original: Any, torch_module: Any, checkpoint: str) -> Any:
        @wraps(original)
        def forward(layer: Any, *args: Any, **kwargs: Any) -> Any:
            """Identify the registered layer and checkpoint invocation."""
            if id(layer) not in self._layers:
                raise ValueError("unregistered MegaMoe layer in diagnostic capture")
            entry = self._layers[id(layer)]
            index = self.next_index("forward", entry["layer"])
            direction = "recompute" if checkpoint == "reentrant" and torch_module.is_grad_enabled() else "forward"
            with self.span("call", **entry, direction=direction, call_id=index, forward_call=index):
                return original(layer, *args, **kwargs)
        return forward

    def _autograd_forward(self, original: Any) -> Any:
        @wraps(original)
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Save immutable placement metadata on the autograd context."""
            route = args[6]
            pending = route.placement_payload
            metadata = placement_metadata(pending.manager, pending.plan) if pending is not None else {
                "placement_id": "home", "placement_generation": None,
            }
            ctx.megamoe_profile_identity = self.metadata() | metadata
            with self.span("autograd_forward", **metadata):
                return original(ctx, *args, **kwargs)
        return forward

    def _autograd_backward(self, original: Any) -> Any:
        @wraps(original)
        def backward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Restore the forward identity on the autograd execution thread."""
            metadata = getattr(ctx, "megamoe_profile_identity", None)
            if metadata is None or metadata["capture_id"] != self.base["capture_id"]:
                raise ValueError("backward has no forward identity in this diagnostic capture")
            metadata = metadata | {
                "direction": "backward", "call_id": self.next_index("backward", metadata["layer"]),
            }
            with self.span("call", **metadata):
                return original(ctx, *args, **kwargs)
        return backward

    def _reduction(self, original: Any) -> Any:
        @wraps(original)
        def reduce(manager: Any, plan: Any, *args: Any, **kwargs: Any) -> Any:
            """Identify one reduction with its actual placement generation."""
            metadata = self.metadata()
            if "layer" not in metadata:
                raise ValueError("gradient reduction lacks its layer identity")
            key = (metadata["layer"], metadata["direction"], metadata["call_id"])
            with self.span("reduction", reduction_index=self.next_index("reduction", *key),
                           **placement_metadata(manager, plan)):
                return original(manager, plan, *args, **kwargs)
        return reduce

    def _batch(self, original: Any) -> Any:
        @wraps(original)
        def complete(manager: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
            """Count every globally ordered batch, including empty local batches."""
            metadata = self.metadata()
            key = tuple(metadata[name] for name in ("layer", "direction", "call_id", "reduction_index"))
            with self.span("batch", batch_index=self.next_index("batch", *key),
                           receive_only=kwargs.get("receive_control") is not None):
                return original(manager, batch, *args, **kwargs)
        return complete

    def _gradient_group(self, original: Any) -> Any:
        @wraps(original)
        def reduce(manager: Any, owner_slot: Any, *args: Any, **kwargs: Any) -> Any:
            """Bound the owner accumulator and receive-window reuse interval."""
            metadata = self.metadata()
            key = tuple(metadata[name] for name in ("layer", "direction", "call_id", "reduction_index"))
            with self.span("gradient_group", group_index=self.next_index("gradient_group", *key),
                           owner_slot=owner_slot):
                return original(manager, owner_slot, *args, **kwargs)
        return reduce

    def _kernel(self, original: Any, direction: str) -> Any:
        @wraps(original)
        def launch(*args: Any, **kwargs: Any) -> Any:
            """Remember which launch last wrote each shared task arena."""
            plan = args[0] if direction == "forward" else args[0].plan
            wave_count = args[1].wave_count if direction == "forward" else args[0].wave_count
            spec = plan.spec
            metadata = self.metadata()
            logical = metadata.get("logical_experts", [
                spec.rank_id * spec.local_experts + index for index in range(spec.local_experts)
            ])
            with self.span("kernel", kernel_direction=direction, wave_count=wave_count,
                           num_cube_cores=spec.num_cube_cores, logical_experts=logical):
                result = original(*args, **kwargs)
                self.last_launch[(metadata["resource_group"], direction)] = self.metadata()
                return result
        return launch

    @contextmanager
    def observe(self, torch_module: Any, layers: Any, *, checkpoint: str = "none") -> Iterator[None]:
        """Install scalar identities only inside an explicitly requested extra step."""
        self.bind_layers(layers)
        module = importlib.import_module(f"{PACKAGE}.module")
        function = importlib.import_module(f"{PACKAGE}.function")
        replica = importlib.import_module(f"{PACKAGE}.replica")
        autograd = function._get_mega_moe_function()  # pylint: disable=protected-access
        targets = (
            (module.MegaMoeExperts, "forward", lambda fn: self._module_forward(fn, torch_module, checkpoint)),
            (autograd, "forward", self._autograd_forward),
            (autograd, "backward", self._autograd_backward),
            (replica.ExpertReplicaManager, "reduce_replica_gradients", self._reduction),
            (replica.ExpertReplicaManager, "_complete_gradient_transfers", self._batch),
            (replica.ExpertReplicaManager, "_reduce_gradient_group", self._gradient_group),
            (function, "_launch_forward", lambda fn: self._kernel(fn, "forward")),
            (function, "_launch_backward", lambda fn: self._kernel(fn, "backward")),
        )
        with ExitStack() as stack:
            for target, name, decorate in targets:
                descriptor = inspect.getattr_static(target, name)
                original = getattr(target, name)
                stack.callback(setattr, target, name, descriptor)
                decorated = decorate(original)
                setattr(target, name, staticmethod(decorated) if isinstance(descriptor, staticmethod) else decorated)
            yield
