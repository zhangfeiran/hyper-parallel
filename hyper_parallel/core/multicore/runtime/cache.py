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
"""Integrity-checked disk storage for source-only frontend emission artifacts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.backends.ascend import BackendEmission


class EmissionCache:
    """Content-addressed generated source cache; device bindings are invocation-owned."""

    def __init__(self, root: Path) -> None:
        """Select the user-owned cache root without probing any devices.

        Args:
            root: Directory receiving immutable per-emission artifact bundles.
        """
        self.root = Path(root).resolve()

    def store(self, emission: BackendEmission) -> Path:
        """Atomically store an emission or verify and reuse its intact existing entry.

        Args:
            emission: Generated sources, descriptors and source maps with separate static keys.
        """
        target = self.root / emission.artifact_key
        if target.exists():
            self.load(emission.artifact_key)
            return target
        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".frontend-stage-", dir=self.root) as directory:
            stage = Path(directory) / "entry"
            stage.mkdir()
            for file in emission.files:
                path = _member(stage, file.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.content)
            (stage / "manifest.json").write_text(
                json.dumps(emission.export_manifest(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
            try:
                os.rename(stage, target)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                self.load(emission.artifact_key)
        self.load(emission.artifact_key)
        return target

    def load(self, key: str) -> dict[str, bytes]:
        """Reject missing, escaping, corrupted or unrecorded files before cache reuse.

        Args:
            key: Artifact identity from BackendEmission.export_manifest.
        """
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("Emission cache key must be a lowercase SHA256 digest")
        directory = self.root / key
        if not directory.resolve().is_relative_to(self.root):
            raise ValueError("Emission cache entry escapes its root")
        data = json.loads((directory / "manifest.json").read_text())
        digests = data["artifacts"]
        identity = {name: data[name] for name in ("definition_key", "plan_key", "family", "artifacts")}
        expected = hashlib.sha256((json.dumps(identity, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        if (data["format_version"] != 1 or data["status"] != "source_only"
                or data["artifact_key"] != key or expected != key):
            raise ValueError("Emission cache manifest identity mismatch")
        result = {}
        for name, digest in digests.items():
            content = _member(directory, name).read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Emission cache artifact corrupted: {name}")
            result[name] = content
        actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
        if actual != set(digests) | {"manifest.json"}:
            raise ValueError("Emission cache contains unrecorded artifacts")
        return result


def _member(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or name == "manifest.json":
        raise ValueError(f"Invalid emission artifact path: {name}")
    member = (root / path).resolve()
    if not member.is_relative_to(root.resolve()):
        raise ValueError(f"Emission artifact escapes its root: {name}")
    return member
