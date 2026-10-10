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
"""Resident payload integrity and autograd invocation protocols without NPU execution."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional

from hyper_parallel.core.multicore._build.build_dense_tile import verify_dense_tile_payload
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseSpec
from hyper_parallel.core.multicore.runtime.dense_scratch import DenseScratchPool
from hyper_parallel.core.multicore.runtime.dense_saved import DenseSavedPool
from hyper_parallel.core.multicore.runtime.dense_tile import ResidentDenseExecutable
from tests.common.mark_utils import arg_mark


def _record(root):
    identity = {"sources": "synthetic-source", "soc": "synthetic-target"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    files = {name: b"synthetic-artifact-" + name.encode()
             for name in ("launch.so", "tiling.so", "reference.so", "dense.bin")}
    for name, value in files.items():
        (root / name).write_bytes(value)
    record = {"format_version": 1, "execution_mode": "resident_dense_tile_candidate", "identity": identity,
              "namespace": "hp_dense_tile_" + key[:24], "library": "launch.so", "tiling_library": "tiling.so",
              "binary": "dense.bin", "reference_library": "reference.so",
              "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    return manifest, identity, record


def _cpu_protocol(rows):
    spec = DenseSpec({"T": rows, "H": 8, "PackedI": 12, "I": 6})
    plan = dense_ffn.plan(spec, intermediate_size=6)
    executable = ResidentDenseExecutable.__new__(ResidentDenseExecutable)
    DenseExecutable.__init__(executable, plan, torch.device("cpu"))
    executable.tile_plan = compile_dense_tiles(plan, DenseTilePolicy(cube_workers=1))
    executable.saved_ids = saved_value_ids(plan)
    executable.saved_shapes = {value_id: dict(plan.value_types)[value_id].shape for value_id in executable.saved_ids}
    executable.saved_pool = DenseSavedPool(torch.device("cpu"), lambda _base: False, max_cached_invocations=0)
    executable.scratch = DenseScratchPool(torch.device("cpu"), len(plan.value_types),
                                         executable.tile_plan.event_count * 32, 8)

    def _vjp(inputs, saved, gradients):
        x, gate_up, down = inputs
        packed, hidden = saved
        dy, = gradients
        if dy is None:
            return [None, None, None]
        dh = dy @ down.t()
        gate, up = packed.float().chunk(2, dim=-1)
        sigmoid = gate.sigmoid()
        gate_grad = dh.float() * up * sigmoid * (1 + gate * (1 - sigmoid))
        up_grad = dh.float() * functional.silu(gate)
        dp = torch.cat((gate_grad, up_grad), dim=-1).to(torch.bfloat16)
        return [dp @ gate_up.t(), x.t() @ dp, hidden.t() @ dy]

    executable.vjp = SimpleNamespace(native_ops=SimpleNamespace(backward=SimpleNamespace(default=_vjp)))
    return executable


def _cpu_launch(values):
    x, gate_up, down = (values[index] for index in range(3))
    values[3].copy_(x @ gate_up)
    gate, up = values[3].float().chunk(2, dim=-1)
    values[4].copy_((functional.silu(gate) * up).to(torch.bfloat16))
    values[5].copy_(values[4] @ down)


class TestDenseTilePayload(unittest.TestCase):
    """Integrity checks and CPU protocol doubles are separate from real native validation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sealed_artifact_and_source_identity_reject_tampering(self):
        """Feature: Resident native payload admission.
        Description: Change source identity, artifact bytes, namespace or mode in a synthetic manifest.
        Expectation: Only an intact, fully sealed payload is accepted, without loading a library.
        """
        for failure in ("source", "library", "namespace", "mode", "omitted", "reference"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, identity, record = _record(root)
                self.assertEqual(verify_dense_tile_payload(manifest, identity), record)
                if failure == "source":
                    identity = {**identity, "sources": "different"}
                elif failure == "library":
                    (root / "launch.so").write_bytes(b"different")
                elif failure == "namespace":
                    record["namespace"] = "unowned"
                elif failure == "mode":
                    record["execution_mode"] = "native_host_stream_adapter"
                elif failure == "omitted":
                    record["files"].pop("dense.bin")
                else:
                    record["files"].pop("reference.so")
                manifest.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity|integrity|required"):
                    verify_dense_tile_payload(manifest, identity)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_library_path_cannot_escape_cache(self):
        """Feature: Caller-owned native cache boundary.
        Description: Replace a sealed library by a symlink to another directory with equal bytes.
        Expectation: Matching hashes cannot bypass path containment.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            manifest, identity, _ = _record(cache)
            outside = root / "outside.so"
            outside.write_bytes((cache / "launch.so").read_bytes())
            (cache / "launch.so").unlink()
            (cache / "launch.so").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "path"):
                verify_dense_tile_payload(manifest, identity)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_private_saved_invocations_survive_other_weights_and_close(self):
        """Feature: Resident autograd state ownership.
        Description: Use a CPU launcher/VJP double for two in-flight calls with different weights, then close.
        Expectation: Each backward uses its original buffers and close only rejects future forwards.
        """
        torch.manual_seed(19)
        executable = _cpu_protocol(7)
        cases = [tuple((torch.randn(shape, dtype=torch.bfloat16) * 0.1).requires_grad_()
                       for shape in ((7, 8), (8, 12), (6, 8))) for _ in range(2)]
        versions = [tuple(value._version for value in inputs) for inputs in cases]
        with patch.object(executable, "_launch", side_effect=_cpu_launch):
            outputs = [executable(*inputs) for inputs in cases]
        self.assertEqual([tuple(value._version for value in inputs) for inputs in cases], versions)
        executable.close()
        for inputs, output in zip(cases, outputs):
            reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
            reference = executable.plan.materialize("cpu")(*reference_inputs)
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            actual_grads = torch.autograd.grad(output.float().square().sum(), inputs)
            expected_grads = torch.autograd.grad(reference.float().square().sum(), reference_inputs)
            for actual, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            executable(*cases[0])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_zero_tokens_skip_launch_and_keep_zero_weight_gradients(self):
        """Feature: Zero-token resident invocation.
        Description: Execute empty tensors with a CPU VJP double and a monitored launch boundary.
        Expectation: No kernel launches and the backward retains correctly shaped zero gradients.
        """
        executable = _cpu_protocol(0)
        inputs = tuple(torch.empty(shape, dtype=torch.bfloat16, requires_grad=True)
                       for shape in ((0, 8), (8, 12), (6, 8)))
        with patch.object(executable, "_launch") as launch:
            executable(*inputs).float().sum().backward()
        launch.assert_not_called()
        for tensor in inputs:
            self.assertEqual(tuple(tensor.grad.shape), tuple(tensor.shape))
            self.assertEqual(torch.count_nonzero(tensor.grad).item(), 0)
