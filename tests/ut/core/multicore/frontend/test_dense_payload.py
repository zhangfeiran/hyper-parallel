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
"""Dense native metadata admission using fake libraries without compiling or loading code."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore._build import build_dense
from hyper_parallel.core.multicore.backends.dense import dense_artifacts
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime import dense_compiled
from hyper_parallel.core.multicore.runtime.dense import DenseSpec
from tests.common.mark_utils import arg_mark


def _payload(root):
    plan = dense_ffn.plan(DenseSpec({"T": 7, "H": 8, "PackedI": 12, "I": 6}), intermediate_size=6)
    definition, _, files = dense_artifacts(plan)
    source = files["native/dense.cpp"]
    identity = {"definition_key": definition, "source_sha256": hashlib.sha256(source).hexdigest(),
                "build": build_dense.dense_build_identity()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = root / key
    directory.mkdir()
    (directory / "dense.cpp").write_bytes(source)
    library = directory / "fixture.so"
    library.write_bytes(b"metadata fixture only; never load this file")
    record = {"format_version": 1, "namespace": "hp_dense_" + definition[:24], "identity": identity,
              "execution_mode": "native_host_stream_adapter", "library": library.name,
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return plan, manifest, record


class TestDensePayload(unittest.TestCase):
    """Validate integrity before permitting native operator registration."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_cached_payload_reuse_does_not_build_or_load(self):
        """Feature: Sealed native cache.
        Description: Reuse matching source, ABI and library metadata with a fake sealed binary.
        Expectation: The build API returns the cache manifest without executing native code.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, manifest, _ = _payload(root)
            with patch.object(build_dense, "load") as loader:
                self.assertEqual(build_dense.build_dense_payload(plan, root), manifest)
                loader.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_tampered_cache_source_library_and_escaped_path_reject_before_load(self):
        """Feature: Native source and payload integrity.
        Description: Corrupt source/library bytes or make the library path escape its owning cache.
        Expectation: Both build-cache admission and executable admission reject before loading.
        """
        for failure in ("source", "library", "escape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan, manifest, record = _payload(root)
                if failure == "source":
                    (manifest.parent / "dense.cpp").write_bytes(b"modified source")
                elif failure == "library":
                    (manifest.parent / record["library"]).write_bytes(b"modified library")
                else:
                    record["library"] = "../outside.so"
                    manifest.write_text(json.dumps(record), encoding="utf-8")
                with patch.object(torch.ops, "load_library") as loader:
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        build_dense.build_dense_payload(plan, root)
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        dense_compiled.CompiledDenseExecutable(plan, "cpu", manifest)
                    loader.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_wrong_compiler_identity_rejects_before_load(self):
        """Feature: Exact compiler and framework identity.
        Description: Change recorded compiler provenance while retaining source and fake library hashes.
        Expectation: The executable rejects mismatched provenance without native loading.
        """
        with tempfile.TemporaryDirectory() as directory:
            plan, manifest, record = _payload(Path(directory))
            record["identity"]["build"]["compiler"] = "different compiler"
            manifest.write_text(json.dumps(record), encoding="utf-8")
            with patch.object(torch.ops, "load_library") as loader:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    dense_compiled.CompiledDenseExecutable(plan, "cpu", manifest)
                loader.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_different_loaded_payload_cannot_own_the_same_namespace(self):
        """Feature: Process-local native namespace ownership.
        Description: Present a valid metadata fixture after another fingerprint owns its namespace.
        Expectation: A descriptive error occurs before native library registration.
        """
        with tempfile.TemporaryDirectory() as directory:
            plan, manifest, record = _payload(Path(directory))
            with (patch.dict(dense_compiled._LOADED, {record["namespace"]: "other payload"}, clear=True),
                  patch.object(torch.ops, "load_library") as loader):
                with self.assertRaisesRegex(ValueError, "already owns"):
                    dense_compiled.CompiledDenseExecutable(plan, "cpu", manifest)
                loader.assert_not_called()
