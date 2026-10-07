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
"""Immutable semantic IR; no device addresses, task IDs or ABI slots."""

from __future__ import annotations

import json
from dataclasses import dataclass

from hyper_parallel.core.multicore.language.types import DType, TensorType


@dataclass(frozen=True)
class SourceSpan:
    """Original Python location, including helper call sites."""

    filename: str
    line: int
    column: int
    end_line: int
    end_column: int
    call_chain: tuple[str, ...] = ()

    def __str__(self) -> str:
        """Format a standard file/line/column diagnostic location."""
        return f"{self.filename}:{self.line}:{self.column + 1}"


@dataclass(frozen=True)
class Value:
    """SSA value with a complete logical tensor type."""

    id: int
    name: str
    type: TensorType


@dataclass(frozen=True)
class Effect:
    """Logical access; physical buffer effects belong to buffer lowering."""

    kind: str
    value_id: int


@dataclass(frozen=True)
class Operation:
    """Normalized primitive call with frozen static attributes."""

    logical_name: str
    version: int
    arguments: tuple[tuple[str, object], ...]
    outputs: tuple[Value, ...]
    effects: tuple[Effect, ...]
    source: SourceSpan


@dataclass(frozen=True)
class ProgramIR:
    """Typed computation in source order with explicit outputs."""

    name: str
    inputs: tuple[Value, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[Value, ...]
    constants: tuple[tuple[str, object], ...]
    returns_tuple: bool
    numeric_policy: str = "preserve_numeric_order"

    def dump(self) -> str:
        """Return a deterministic, JSON-serializable semantic/source dump."""

        def _encode(item: object) -> object:
            if isinstance(item, Value):
                return {
                    "value": item.id,
                    "name": item.name,
                    "dtype": item.type.dtype.value,
                    "shape": item.type.shape,
                    "layout": item.type.layout,
                }
            if isinstance(item, DType):
                return item.value
            return item

        data = {
            "name": self.name,
            "numeric_policy": self.numeric_policy,
            "inputs": [_encode(value) for value in self.inputs],
            "constants": dict(self.constants),
            "operations": [
                {
                    "op": f"{op.logical_name}.v{op.version}",
                    "arguments": {key: _encode(value) for key, value in op.arguments},
                    "outputs": [_encode(value) for value in op.outputs],
                    "effects": [{"kind": effect.kind, "value": effect.value_id} for effect in op.effects],
                    "source": str(op.source),
                    "call_chain": op.source.call_chain,
                }
                for op in self.operations
            ],
            "outputs": [_encode(value) for value in self.outputs],
            "returns_tuple": self.returns_tuple,
        }
        return json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
