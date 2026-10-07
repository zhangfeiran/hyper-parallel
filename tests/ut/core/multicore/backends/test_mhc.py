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
"""CPU shifted MHC semantic, complete descriptor and admission tests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import hyper_parallel.core.multicore.frontend as mc
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.modules.mega_mhc.module import HyperMegaMhc
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.mhc_native import verify_mhc_payload
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.common.mark_utils import arg_mark

FIXTURES = Path(__file__).with_name("fixtures")


def _variant(old, new):
    return mc.from_source(mhc_boundary.source.text.replace(old, new), symbols=mhc_boundary.source.symbols,
                          schedule=mc.TaskDAG("shifted_mhc_v1"))


class TestMhcCompatibility(unittest.TestCase):
    """Compare full wire images and exercise numerical/type/lifetime contracts."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_artifact_seal_rejects_drift(self):
        """Feature: Isolated MHC native artifact admission.

        Description: Seal mock build artifacts, then corrupt their bytes, schema or vendor identity.
        Expectation: Only the intact family-specific payload passes without loading native code.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = ("vendors/hyper_parallel_multicore_mhc_v1/op_api/lib/libcust_opapi.so",
                     "framework/torch/libhyper_parallel_mega_mhc_torch.so", "set_env.bash")
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture artifact")
            data = family_abi("mhc").export_manifest()
            data.pop("native_build_fingerprint")
            data.update(vendor="hyper_parallel_multicore_mhc_v1", build={"identity": "fixture"},
                        artifacts={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})
            data["build_fingerprint"] = hashlib.sha256(json.dumps(
                {"build": data["build"], "artifacts": data["artifacts"]}, sort_keys=True).encode()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(data))
            self.assertEqual(verify_mhc_payload(root)["family"], "mhc")
            (root / files[0]).write_bytes(b"corruption")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                verify_mhc_payload(root)
            (root / files[0]).write_bytes(b"fixture artifact")
            for field in ("schema_hash", "vendor"):
                manifest.write_text(json.dumps({**data, field: "wrong"}))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    verify_mhc_payload(root)
            manifest.write_text(json.dumps(data))
            (root / "unexpected.so").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "unrecorded"):
                verify_mhc_payload(root)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_complete_fixed_revision_images(self):
        """Feature: Original MHC family compatibility.

        Description: Compare all forward/backward normal/profiled bytes against pinned builders.
        Expectation: Sixteen images match, FIFO progresses and ring/macro contracts are retained.
        """
        fixture = json.loads((FIXTURES / "mhc_snapshots.json").read_text())
        self.assertEqual(fixture["revision"], "979e2a9ac913413e361f4fc2dd9987766af8ddb4")
        for case in fixture["cases"]:
            with self.subTest(tokens=case["T"], tile=case["tile"]):
                plan = mhc_boundary.plan(MhcSpec(case["T"], case["H"], case["C"], case["tile"], case["tile"]))
                for name, record in case["images"].items():
                    direction, profiled = name.split("_")
                    image = getattr(plan, direction)
                    actual = image.profiled if int(profiled) else image.normal
                    expected = gzip.decompress((FIXTURES / record["file"]).read_bytes())
                    self.assertEqual(hashlib.sha256(expected).hexdigest(), record["sha256"])
                    self.assertEqual(len(actual), record["bytes"])
                    self.assertEqual(actual, expected)
                    self.assertEqual(len(image.simulate()["visited"]), len(image.tasks))
                self.assertIn("NormCast(i) waits", plan.export_manifest()["ring_reuse"])
                self.assertEqual(sorted(index for join in plan.macro_join for index in join),
                                 list(range((case["T"] + plan.grad_token_tile - 1) // plan.grad_token_tile)))
                if case["T"] == 2593:
                    self.assertEqual(plan.ring_slots, 80)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shifted_semantics_and_all_input_gradients(self):
        """Feature: Shifted five-output reference.

        Description: Interpret all four primitives and differentiate all nine input tensors.
        Expectation: InputMix uses previous pre; all outputs/dtypes and finite gradients survive.
        """
        torch.manual_seed(24)
        shapes = ((48, 4, 128), (48, 128), (48, 4), (48, 4), (48, 4, 4), (24, 512), (3,), (24,), (128,))
        values = [torch.randn(shape, dtype=torch.bfloat16 if index in (0, 1, 8) else torch.float32,
                              requires_grad=True) for index, shape in enumerate(shapes)]
        outputs = mhc_boundary.interpret(*values)
        self.assertEqual(tuple(tuple(output.shape) for output in outputs),
                         (shapes[0], shapes[2], shapes[3], shapes[4], shapes[1]))
        mixed = (outputs[0].float() * values[2].float().unsqueeze(-1)).sum(-2).bfloat16().float()
        expected = (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
                    * values[8].float()).bfloat16()
        torch.testing.assert_close(outputs[4], expected, rtol=0, atol=0)
        sum(output.float().square().mean() for output in outputs).backward()
        for value in values:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertEqual(outputs[1].dtype, torch.float32)
        self.assertEqual(outputs[4].dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_unsupported_semantics_and_native_boundaries(self):
        """Feature: Native contract admission.

        Description: Change the shifted mix, RMS epsilon, dtypes and native capacity boundaries.
        Expectation: Every unsupported contract fails on CPU before allocating native state.
        """
        for old, new in (("updated, previous_pre)", "updated, next_pre)"),
                         ("eps=norm_eps)", "eps=hc_eps)"),
                         ("phi: ml.Tensor[ml.fp32", "phi: ml.Tensor[ml.bf16")):
            with self.subTest(change=new), self.assertRaises((ValueError, TypeError)):
                _variant(old, new).plan(MhcSpec(128, 128, 20), norm_eps=1e-5)
        for constants in ({"num_iters": 1}, {"hc_eps": 0.0}, {"norm_eps": float("nan")}):
            with self.subTest(constants=constants), self.assertRaises((ValueError, TypeError)):
                mhc_boundary.plan(MhcSpec(128, 128, 20), **constants)
        for values in ((39, 128, 20), (48, 127, 24), (48, 5888, 24), (True, 128, 20)):
            with self.subTest(spec=values), self.assertRaises(ValueError):
                MhcSpec(*values)
        with self.assertRaisesRegex(ValueError, "padding"):
            mhc_boundary.plan(MhcSpec(10817, 128, 20))
        self.assertIsNone(mhc_boundary.plan(MhcSpec(1, 128, 20, need_backward=False)).backward)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_module_parameter_contract_and_cpu_only_compilation(self):
        """Feature: Model entry point and deferred native loading.

        Description: Construct the module on CPU and compile while torch_npu imports are blocked.
        Expectation: Original parameter names/dtypes survive; no device libraries load.
        """
        module = HyperMegaMhc(128)
        self.assertEqual(tuple(module.state_dict()), ("phi", "alpha", "bias", "norm_weight"))
        self.assertEqual(module.phi.shape, (24, 512))
        self.assertEqual(module.norm_weight.dtype, torch.bfloat16)
        self.assertFalse(module._executables)
        module.close()
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise AssertionError('torch_npu imported by host MHC compiler')
sys.meta_path.insert(0, Block())
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.mhc_native import verify_mhc_payload
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
plan = mhc_boundary.plan(MhcSpec(2593,128,20))
assert plan.forward.normal and plan.backward.normal
assert 'torch_npu' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
