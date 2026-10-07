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
"""CPU rejection tests for native Gate artifact and stream-profile contracts."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.gate import (
    GateExecutable,
    GateProfile,
    verify_gate_payload,
)
from tests.common.mark_utils import arg_mark


def _manifest(root):
    artifacts = {}
    for name in ("framework/torch/libhyper_parallel_mega_gate_torch.so", "set_env.bash",
                 "vendors/hyper_parallel_multicore_gate_v1/op_api/lib/libcust_opapi.so"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        artifacts[name] = hashlib.sha256(b"fixture").hexdigest()
    data = {
        **family_abi("gate").export_manifest(), "vendor": "hyper_parallel_multicore_gate_v1",
        "build": {"soc": "ascend910b"}, "artifacts": artifacts,
    }
    data.pop("native_build_fingerprint")
    _save(root, data)
    return data


def _save(root, data):
    data["build_fingerprint"] = hashlib.sha256(json.dumps(
        {"build": data["build"], "artifacts": data["artifacts"]}, sort_keys=True,
    ).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(data))


class TestGatePayload(unittest.TestCase):
    """A metadata match alone cannot authorize loading stale native bytes."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_artifact_corruption_and_missing_files(self):
        """
        Feature: Native binding.
        Description: Tamper with bytes.
        Expectation: Reject before loading.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _manifest(root)
            self.assertEqual(verify_gate_payload(root)["build_fingerprint"], data["build_fingerprint"])
            library = root / "framework/torch/libhyper_parallel_mega_gate_torch.so"
            library.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                verify_gate_payload(root)
            library.unlink()
            with self.assertRaisesRegex(ValueError, "Missing or escaping"):
                verify_gate_payload(root)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_foreign_family_and_build_identity(self):
        """
        Feature: Native binding.
        Description: Replace identities.
        Expectation: Reject incompatible builds.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in (("family", "moe"), ("schema_hash", "0" * 64),
                               ("source_fingerprint", "0" * 64), ("runtime_abi_version", True)):
                with self.subTest(key=key):
                    data = _manifest(root)
                    data[key] = value
                    _save(root, data)
                    with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                        verify_gate_payload(root)
            data = _manifest(root)
            data["build"]["soc"] = "ascend910_93"
            (root / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "build fingerprint mismatch"):
                verify_gate_payload(root)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unrecorded_and_escaping_artifacts(self):
        """
        Feature: Native binding.
        Description: Add foreign files.
        Expectation: Reject ambiguous payloads.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _manifest(root)
            extra = root / "foreign.so"
            extra.write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "unrecorded artifacts"):
                verify_gate_payload(root)
            extra.unlink()
            data["artifacts"]["../foreign.so"] = "0" * 64
            _save(root, data)
            with self.assertRaisesRegex(ValueError, "Missing or escaping"):
                verify_gate_payload(root)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_materialization_rejects_foreign_topology_before_allocating(self):
        """
        Feature: Device binding.
        Description: Compare actual AIV count.
        Expectation: Reject stale partitions.
        """
        plan = _route.plan({"T": 49, "E": 64}, k=3, scale=2.5)
        npu = Mock()
        npu.get_device_name.return_value = "Ascend910B3"
        npu.get_device_properties.return_value = SimpleNamespace(vector_core_num=40)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _manifest(root)
            with patch("hyper_parallel.core.multicore.runtime.gate._load_payload"), \
                 patch.object(torch, "npu", npu, create=True), \
                 patch("hyper_parallel.core.multicore.runtime.gate.torch.device") as device, \
                 patch("hyper_parallel.core.multicore.runtime.gate.torch.tensor") as allocate:
                device.return_value = SimpleNamespace(type="npu", index=0)
                with self.assertRaisesRegex(ValueError, "device has 40 AIV"):
                    GateExecutable(plan, "npu:0", root)
                allocate.assert_not_called()


class TestGateProfile(unittest.TestCase):
    """Wire records map to the compiled stages and reject dropped or foreign tasks."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_source_mapping_and_private_completion(self):
        """
        Feature: Profiling.
        Description: Decode native wire records.
        Expectation: Keep worker and source identity.
        """
        plan = _route.plan({"T": 1, "E": 8}, k=3, scale=2.5)
        wire = bytearray(24 * 64 + 48 * (64 + 16 * 32))
        offset = 24 * 64
        struct.pack_into("<QIIIIII32x", wire, offset, 100, 10, 0, 2, 0, 16, 0)
        for index in range(10):
            struct.pack_into("<QQIIII", wire, offset + 64 + index * 32,
                             100 + index * 10, 105 + index * 10, 0x30000 + index, index, 0, 0)
        event = Mock()
        profile = GateProfile("forward", plan.forward, torch.tensor(list(wire), dtype=torch.uint8), event)
        records = profile.records()
        event.synchronize.assert_called_once_with()
        self.assertEqual(len(records), 10)
        self.assertEqual(records[0]["stage"], "gate.softplus.v1")
        self.assertTrue(records[0]["source"])
        wire[offset + 12] = 1
        profile.buffer = torch.tensor(list(wire), dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "Invalid or dropped"):
            profile.records()


if __name__ == "__main__":
    unittest.main()
