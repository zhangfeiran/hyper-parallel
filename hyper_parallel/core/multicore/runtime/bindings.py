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
"""Shared native argument contracts and one-time dispatcher ABI admission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class NativeArgument:
    """One native argument's logical name, wire order, type and mutation contract."""

    name: str
    type: str
    write: bool


@dataclass(frozen=True)
class NativeEntry:
    """Family-specific native entry schema shared by generators and runtime admission."""

    name: str
    family: str
    schema: str
    arguments: tuple[NativeArgument, ...]

    def pack(self, values: dict[str, object]) -> tuple[object, ...]:
        """Bind named logical resources without allowing missing or extra arguments.

        Args:
            values: Per-invocation tensors and scalar attributes, never cached pointers.
        """
        if set(values) != {argument.name for argument in self.arguments}:
            raise ValueError(f"Native binding names do not match {self.name}")
        return tuple(values[argument.name] for argument in self.arguments)

    def verify(self, schema: Any) -> None:
        """Verify types, names, aliases, mutation and result order against the loaded dispatcher.

        Args:
            schema: Loaded default overload schema.
        """
        expected = torch._C.parse_schema("hyper_parallel::" + self.schema)
        if str(schema) != str(expected):
            raise ValueError(f"Native dispatcher schema mismatch: {self.name}")


@lru_cache(maxsize=6)
def native_entry(name: str) -> NativeEntry:
    """Resolve one of the six explicitly supported forward/backward entries.

    Args:
        name: Exact registered operator name without a namespace.
    """
    entries = json.loads(Path(__file__).with_name("native_calls.json").read_text(encoding="utf-8"))["entries"]
    if name not in entries:
        raise ValueError(f"Unknown native entry: {name}")
    entry = entries[name]
    return NativeEntry(name, entry["family"], entry["schema"],
                       tuple(NativeArgument(**argument) for argument in entry["arguments"]))


@lru_cache(maxsize=6)
def resolve_native_call(name: str) -> Any:
    """Validate the activated entry once and return its default dispatch overload.

    Args:
        name: Exact registered operator name from the shared schema.
    """
    entry = native_entry(name)
    operation = getattr(torch.ops.hyper_parallel, name).default
    entry.verify(operation._schema)
    return operation
