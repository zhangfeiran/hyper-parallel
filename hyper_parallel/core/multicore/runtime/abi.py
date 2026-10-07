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
"""Versioned host family contracts; legacy wire headers remain family-specific."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ABIField:
    """One fixed-revision native structure field."""

    name: str
    type: str
    count: int
    offset: int


@dataclass(frozen=True)
class StructABI:
    """Validated structure size and ordered field offsets."""

    name: str
    size: int
    fields: tuple[ABIField, ...]

    def offset(self, name: str) -> int:
        """Return the byte offset of a declared ABI field.

        Args:
            name: Exact legacy field name.
        """
        for field in self.fields:
            if field.name == name:
                return field.offset
        raise ValueError(f"Unknown {self.name} field: {name}")


@dataclass(frozen=True)
class TaskBinding:
    """Namespaced logical identity and family-local native task number."""

    logical_name: str
    native_name: str
    numeric_id: int


@dataclass(frozen=True)
class PipelineTemplate:
    """Ordered stage metadata extracted from the original family builder."""

    name: str
    kernel_name: str
    logical_stages: tuple[str, ...]
    stage_names: tuple[str, ...]


@dataclass(frozen=True)
class NativeManifest:
    """Metadata to be supplied by a matching native build, never inferred from a filename."""

    family: str
    runtime_abi_version: int
    schema_hash: str
    source_fingerprint: str
    build_fingerprint: str


@dataclass(frozen=True)
class FamilyABI:
    """Frozen source/dependency identity and family-specific ABI schema."""

    family: str
    source_revision: str
    runtime_abi_version: int
    schema_hash: str
    source_fingerprint: str
    task_types: tuple[TaskBinding, ...]
    structures: tuple[StructABI, ...]
    pipelines: tuple[PipelineTemplate, ...]
    scratch_policy: str

    def structure(self, name: str) -> StructABI:
        """Resolve a declared legacy structure.

        Args:
            name: Structure class name in the pinned scheduler/config.py.
        """
        for structure in self.structures:
            if structure.name == name:
                return structure
        raise ValueError(f"Unknown {self.family} structure: {name}")

    def task_id(self, logical_name: str) -> int:
        """Resolve a logical task within this native family.

        Args:
            logical_name: Versioned namespaced primitive identity.
        """
        for binding in self.task_types:
            if binding.logical_name == logical_name:
                return binding.numeric_id
        raise ValueError(f"Unknown {self.family} task: {logical_name}")

    def pipeline(self, name: str) -> PipelineTemplate:
        """Resolve a pinned worker descriptor sequence.

        Args:
            name: Forward or backward template variant.
        """
        for pipeline in self.pipelines:
            if pipeline.name == name:
                return pipeline
        raise ValueError(f"Unknown {self.family} pipeline: {name}")

    def verify_native(self, manifest: NativeManifest) -> None:
        """Reject mismatched metadata before any future native launch.

        This checks build-provided metadata, not ELF contents. A native artifact
        must emit this metadata as part of its build before it can be bound.

        Args:
            manifest: Metadata provided by the native artifact's build.
        """
        for name in ("family", "runtime_abi_version", "schema_hash", "source_fingerprint"):
            actual, expected = getattr(manifest, name), getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"Native manifest mismatch: {name}")
        if (
            type(manifest.build_fingerprint) not in (str,)
            or len(manifest.build_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in manifest.build_fingerprint)
        ):
            raise ValueError("Native build fingerprint must be a lowercase SHA256 digest")

    def export_manifest(self) -> dict[str, object]:
        """Export host compatibility identity without claiming a bound native build."""
        return {
            "family": self.family,
            "source_revision": self.source_revision,
            "runtime_abi_version": self.runtime_abi_version,
            "schema_hash": self.schema_hash,
            "source_fingerprint": self.source_fingerprint,
            "native_build_fingerprint": None,
        }


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=3)
def family_abi(family: str) -> FamilyABI:
    """Load an immutable family ABI extracted from the design's pinned sources.

    Args:
        family: moe, mhc or gate. Native numeric IDs are local to this family.
    """
    baseline = json.loads((Path(__file__).parent / "baselines/families.json").read_text(encoding="utf-8"))
    try:
        data = baseline["families"][family]
    except KeyError as exc:
        raise ValueError(f"Unknown native family: {family}") from exc
    tasks = tuple(TaskBinding(**item) for item in data["task_types"])
    structures = tuple(
        StructABI(item["name"], item["size"], tuple(ABIField(**field) for field in item["fields"]))
        for item in data["structures"]
    )
    native_names = {item.native_name: item.logical_name for item in tasks}
    pipelines = tuple(
        PipelineTemplate(
            name,
            item["kernel_name"],
            tuple(native_names[task] for task in item["task_types"]),
            tuple(item["stage_names"]),
        )
        for name, item in data.get("pipelines", {}).items()
    )
    schema = {
        key: data[key]
        for key in ("family", "runtime_abi_version", "task_types", "structures", "constants", "scratch_policy")
    }
    sources = {key: data[key] for key in ("source_revision", "dependencies", "source_hashes")}
    return FamilyABI(
        family,
        data["source_revision"],
        data["runtime_abi_version"],
        _digest(schema),
        _digest(sources),
        tasks,
        structures,
        pipelines,
        data["scratch_policy"],
    )
