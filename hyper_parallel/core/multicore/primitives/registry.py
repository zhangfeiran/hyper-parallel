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
"""Identity-based, versioned primitive registration shared by frontend and interpreter."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from hyper_parallel.core.multicore.language.types import TensorType


def _read_effects(arguments: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(("read", key) for key, value in arguments.items() if isinstance(value, TensorType))


@dataclass(frozen=True)
class OpSchema:
    """Trusted type/effect/reference hooks for a namespaced numerical primitive.

    Type inference receives normalized arguments with TensorType in place of values.
    Reference hooks are called only by the debug interpreter. Native implementations
    and backward recipes will be added when their family contracts are integrated.
    """

    logical_name: str
    version: int
    signature: inspect.Signature
    infer_types_and_shapes: Callable[[Mapping[str, object]], tuple[TensorType, ...]]
    reference: Callable | None
    infer_effects: Callable[[Mapping[str, object]], tuple[tuple[str, str], ...]] = _read_effects


class Primitive:
    """A registered DSL symbol; compiling a call never executes this object."""

    def __init__(self, schema: OpSchema) -> None:
        """Retain the immutable registered schema."""
        self.schema = schema

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Reject accidental eager evaluation of DSL code."""
        raise RuntimeError("DSL primitives can only be used in a program; use Program.interpret for debugging")


class PrimitiveRegistry:
    """Resolve exact symbol identity and reject conflicting schema registrations."""

    def __init__(self) -> None:
        """Create a compilation-independent schema and symbol registry."""
        self._schemas = {}
        self._symbols = {}

    def register(self, schema: OpSchema) -> Primitive:
        """Register a unique namespaced schema and return its callable symbol.

        Args:
            schema: Trusted numerical and access contract.
        """
        if (
            type(schema.logical_name) not in (str,)
            or "." not in schema.logical_name
            or any(not part.isidentifier() for part in schema.logical_name.split("."))
            or type(schema.version) not in (int,)
            or schema.version < 1
        ):
            raise ValueError("Primitive schemas require a namespaced name and positive version")
        key = (schema.logical_name, schema.version)
        if key in self._schemas:
            raise ValueError(f"Primitive schema already registered: {key}")
        symbol = Primitive(schema)
        self._schemas[key] = schema
        self._symbols[id(symbol)] = symbol
        return symbol

    def resolve(self, symbol: object) -> OpSchema:
        """Resolve identity, never a callable name or user-defined equality.

        Args:
            symbol: Exact registered primitive symbol.
        """
        registered = self._symbols.get(id(symbol))
        if registered is not symbol:
            raise ValueError("Call target is not a registered primitive")
        return registered.schema

    def schema(self, logical_name: str, version: int) -> OpSchema:
        """Resolve a versioned IR operation for interpretation/lowering.

        Args:
            logical_name: Namespaced primitive schema name.
            version: Positive schema version.
        """
        try:
            return self._schemas[(logical_name, version)]
        except KeyError as exc:
            raise ValueError(f"Unknown primitive schema: {logical_name}.v{version}") from exc


REGISTRY = PrimitiveRegistry()
