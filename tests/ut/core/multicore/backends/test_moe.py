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
"""CPU MoE AST, local references and complete legacy schedule parity."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.compiler.moe import match_moe_region
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.primitives.registry import (
    PrimitiveRegistry,
)
from hyper_parallel.core.multicore.runtime import moe_native
from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi
from hyper_parallel.core.multicore.runtime.moe import _image
from tests.common.mark_utils import arg_mark


def _spec(mode="push", ep=2, rank=0, limit=None, replicas=0):
    return MegaMoeSpec(local_num_tokens=128, hidden_size=512, intermediate_size=128,
                       num_experts=(2 + replicas) * ep, top_k=2,
                       initial_capacity_factor=1.25 if mode == "push" else None,
                       receive_capacity=384, ep_size=ep, ep_group=None, rank_id=rank,
                       num_cube_cores=20, dispatch_mode=mode,
                       capacity_growth_factor=1.25 if mode == "push" else None,
                       swiglu_limit=limit, replica_slots_per_rank=replicas,
                       logical_num_experts=2 * ep, replica_transport="shmem_signal_kernel_gradient")


def _variant(old, new):
    return mc.from_source(moe_region.source.text.replace(old, new), symbols=moe_region.source.symbols,
                          schedule=mc.TaskDAG("moe_ratr_v1"))


class TestMoeCompatibility(unittest.TestCase):
    """Native DAG compilation retains routing protocols and numeric contracts."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_module_program_sharing_identity(self):
        """Feature: AST module resource ownership.

        Description: Construct native and AST modules without allocating NPU resources.
        Expectation: Equal AST recipes share resources; native/AST mixing is rejected.
        """
        options = {"local_num_tokens": 128, "hidden_size": 512, "intermediate_size": 128,
                   "num_experts": 4, "top_k": 2, "ep_size": 2, "create_parameters": False}
        native = MegaMoeExperts(**options)
        first = MegaMoeExperts(program=moe_region, **options)
        second = MegaMoeExperts(program=moe_region, **options)
        try:
            self.assertNotIn("program_fingerprint", native._resource_group.specification)
            self.assertEqual(first._resource_group.specification["program_fingerprint"],
                             moe_region.plan(_spec()).recipe.fingerprint)
            with self.assertRaisesRegex(ValueError, "identical"):
                MegaMoeExperts.share_execution_resources([native, first])
            MegaMoeExperts.share_execution_resources([first, second])
            self.assertIs(first._resource_group, second._resource_group)
            with self.assertRaisesRegex(TypeError, "TaskDAG"):
                MegaMoeExperts(program=object(), **options)
        finally:
            for module in (native, first, second):
                module.close()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_payload_seal_rejects_drift(self):
        """Feature: Native artifact admission.

        Description: Corrupt or add artifacts, swap ABI or vendor priority before binding.
        Expectation: CPU admission rejects each mismatch without loading native libraries.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            multicore, shmem, cann = root / "lib", root / "shmem/lib", root / "cann"
            vendor = multicore / "vendors/hyper_parallel_multicore_nn"
            vendor.mkdir(parents=True)
            shmem.mkdir(parents=True)
            (cann / "opp").mkdir(parents=True)
            (cann / "opp/version.info").write_text("CANN fixture")
            files = {"multicore": multicore / "payload.so", "shmem": shmem / "payload.so"}
            for path in files.values():
                path.write_bytes(b"sealed artifact")
            abi = family_abi("moe")
            data = {field.name: getattr(abi, field.name) for field in fields(NativeManifest)
                    if field.name != "build_fingerprint"}
            data.update(build={"cann_root": str(cann), "cann_version": "CANN fixture", "torch": torch.__version__,
                               "torch_npu": "fixture", "torch_cxx11_abi": torch.compiled_with_cxx11_abi()},
                        artifacts={name: {"payload.so": hashlib.sha256(path.read_bytes()).hexdigest()}
                                   for name, path in files.items()})
            data["build_fingerprint"] = hashlib.sha256(json.dumps(
                {"build": data["build"], "artifacts": data["artifacts"]}, sort_keys=True).encode()).hexdigest()
            manifest = multicore / "frontend_manifest.json"
            manifest.write_text(json.dumps(data))
            with patch.object(moe_native, "get_multicore_paths", return_value=(vendor, multicore / "adapter.so")), \
                    patch.object(moe_native, "version", return_value="fixture"), \
                    patch.dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(vendor), ASCEND_HOME_PATH=str(cann)):
                self.assertEqual(moe_native.verify_moe_native()["family"], "moe")
                files["shmem"].write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    moe_native.verify_moe_native()
                files["shmem"].write_bytes(b"sealed artifact")
                extra = shmem / "foreign.so"
                extra.write_bytes(b"unrecorded")
                with self.assertRaisesRegex(ValueError, "unrecorded"):
                    moe_native.verify_moe_native()
                extra.unlink()
                with patch.dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(root) + os.pathsep + str(vendor)), \
                        self.assertRaisesRegex(ValueError, "first"):
                    moe_native.verify_moe_native()
                manifest.write_text(json.dumps({**data, "family": "gate"}))
                with self.assertRaisesRegex(ValueError, "family"):
                    moe_native.verify_moe_native()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_full_forward_backward_images_match_native_builder(self):
        """Feature: MoE TaskDAG lowering.

        Description: Compare both transports, clamp specializations and EP/rank variants.
        Expectation: Complete normal/profiled descriptors and physical queues match.
        """
        for mode, ep, limit in itertools.product(("push", "pull"), (1, 2, 4), (None, 10.0)):
            for rank in range(ep):
                with self.subTest(mode=mode, ep=ep, rank=rank, limit=limit):
                    spec = _spec(mode, ep, rank, limit)
                    plan = moe_region.plan(spec, limit=limit)
                    fgraph, fconfig, bgraph, bconfig = _build_runtime_artifacts(spec, overlap_w13=False)
                    for actual, config, graph, backward in ((plan.forward, fconfig, fgraph, False),
                                                           (plan.backward, bconfig, bgraph, True)):
                        expected = _image(config, graph, plan.recipe, backward, spec)
                        self.assertEqual(actual.normal, expected.normal)
                        self.assertEqual(actual.profiled, expected.profiled)
                        self.assertEqual(actual.queues, expected.queues)
                        self.assertEqual(actual.completion_event, expected.completion_event)
                        self.assertEqual(actual.protocol_version, expected.protocol_version)
                        for queue, worker_queues in zip(actual.queues, actual.worker_queues()):
                            self.assertCountEqual(queue, [task for worker in worker_queues for task in worker])
                    self.assertEqual([stage[0] for stage in plan.forward.stages],
                                     ["dispatch", "up_proj", "swiglu", "down_proj", "combine"])
                    sources = {name: span for name, _, _, span in plan.backward.stages}
                    self.assertEqual(sources["dispatch"], plan.recipe.ir.operations[-1].source)
                    self.assertEqual(sources["combine"], plan.recipe.ir.operations[0].source)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_replica_variants_and_dynamic_group_scratch(self):
        """Feature: Replica backward and dynamic group-list scratch.

        Description: Lower home/guest and large local expert configurations.
        Expectation: Preserve overlap/fallback plans and support more than sixteen groups.
        """
        for mode in ("push", "pull"):
            spec = _spec(mode, replicas=1)
            plan = moe_region.plan(spec)
            self.assertIsNotNone(plan.backward_no_replica)
            _, _, graph, config = _build_runtime_artifacts(spec, overlap_w13=False)
            expected = _image(config, graph, plan.recipe, True, spec)
            self.assertEqual(plan.backward_no_replica.normal, expected.normal)
            self.assertNotEqual(plan.backward.normal, expected.normal)
        spec = replace(_spec(), num_experts=34, logical_num_experts=34)
        plan = moe_region.plan(spec)
        self.assertEqual(plan.spec.local_experts, 17)
        self.assertGreater(len(plan.forward.normal), 0)
        self.assertEqual(json.loads(plan.explain())["manifest"]["runtime_counts"], "borrowed_device_group_list")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_rejects_non_native_numerics_and_bindings(self):
        """Feature: Native recipe validation.

        Description: Change operand identity, matrix shapes, policy and clamp.
        Expectation: Reject incompatible regions before any native allocation.
        """
        for old, new in (("received, w13, groups", "received, w2, groups"),
                         ("layout=\"gate_up\"", "layout=\"up_gate\""),
                         ("ml.combine(projected, route_meta)", "projected")):
            with self.subTest(change=new), self.assertRaises((ValueError, mc.FrontendError)):
                _variant(old, new).plan(_spec())
        with self.assertRaisesRegex(ValueError, "limit"):
            moe_region.plan(_spec(limit=10.0), limit=None)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            _variant("UP_SHAPE", "(512, 258)").plan(_spec())
        ir = replace(moe_region.lower(), numeric_policy="fast_math")
        with self.assertRaisesRegex(ValueError, "five-stage"):
            match_moe_region(ir)
        with self.assertRaisesRegex(TypeError, "MegaMoeSpec"):
            moe_region.plan({"Rows": 256, "H": 512})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_canonical_schema_identity(self):
        """Feature: Registry isolation.

        Description: Impersonate the dispatch schema in an independent registry.
        Expectation: Native lowering rejects the replacement schema.
        """
        registry = PrimitiveRegistry()
        original = ml.dispatch.schema
        fake = registry.register(replace(original))
        symbols = dict(moe_region.source.symbols)
        symbols["fake"] = fake
        program = mc.from_source(moe_region.source.text.replace("ml.dispatch", "fake"), symbols=symbols,
                                 registry=registry, schedule=mc.TaskDAG("moe_ratr_v1"))
        # Other primitives retain their actual identities in this registry.
        for symbol in (ml.grouped_matmul, ml.swiglu_packed, ml.combine):
            registry._schemas[(symbol.schema.logical_name, symbol.schema.version)] = symbol.schema
            registry._symbols[id(symbol)] = symbol
        with self.assertRaisesRegex(ValueError, "canonical"):
            program.plan(_spec())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_cpu_reference_handles_ragged_rows_empty_experts_and_gradients(self):
        """Feature: Local MoE reference execution.

        Description: Interpret padded expert-major BF16 input with an empty expert.
        Expectation: Effective rows and gradients are preserved; padded rows receive zero gradient.
        """
        torch.manual_seed(12)
        for limit in (None, 0.5):
            source = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
            w13 = torch.randn(3, 4, 6, dtype=torch.bfloat16, requires_grad=True)
            w2 = torch.randn(3, 3, 4, dtype=torch.bfloat16, requires_grad=True)
            metadata = ml.RouteMetadata(torch.tensor([2, 2, 5], dtype=torch.int64))
            output = moe_region.interpret(ml.RaggedTensor(source, 5), w13, w2, metadata, limit)
            self.assertEqual(output.valid_rows, 5)
            self.assertEqual(tuple(output.storage.shape), (8, 4))
            output.storage.float().square().sum().backward()
            self.assertEqual(torch.count_nonzero(source.grad[5:]).item(), 0)
            self.assertEqual(torch.count_nonzero(w13.grad[1]).item(), 0)
            self.assertEqual(torch.count_nonzero(w2.grad[1]).item(), 0)
            self.assertTrue(torch.isfinite(source.grad).all())
            bad = ml.RouteMetadata(torch.tensor([2, 1, 5], dtype=torch.int64))
            with self.assertRaisesRegex(ValueError, "monotonic"):
                moe_region.interpret(ml.RaggedTensor(source, 5), w13, w2, bad, limit)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_cpu_compilation_does_not_import_npu(self):
        """Feature: CPU compilation boundary.

        Description: Block torch_npu imports in a fresh process and produce a complete MoE plan.
        Expectation: AST, graph filling, finalize and serialization remain CPU-only.
        """
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise AssertionError('torch_npu imported by host compiler')
sys.meta_path.insert(0, Block())
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
spec = MegaMoeSpec(128,512,128,4,2,1.25,384,2,None,0,20)
plan = moe_region.plan(spec)
assert plan.forward.normal and plan.backward.normal
assert 'torch_npu' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_clamp_boundary_derivatives_and_all_empty_groups(self):
        """Feature: Packed SwiGLU reference boundary.

        Description: Differentiate exact clamp endpoints and execute all-empty groups.
        Expectation: Endpoint derivatives and every empty expert gradient are zero.
        """
        packed = torch.tensor([[1.0, 2.0, -1.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)
        result = ml.swiglu_packed.schema.reference(ml.RaggedTensor(packed, 1), limit=1.0)
        result.storage.float().sum().backward()
        self.assertEqual(torch.count_nonzero(packed.grad).item(), 0)
        storage = torch.randn(4, 8, dtype=torch.bfloat16, requires_grad=True)
        w13 = torch.randn(3, 8, 12, dtype=torch.bfloat16, requires_grad=True)
        w2 = torch.randn(3, 6, 8, dtype=torch.bfloat16, requires_grad=True)
        metadata = ml.RouteMetadata(torch.zeros(3, dtype=torch.int64))
        output = moe_region.interpret(ml.RaggedTensor(storage, 0), w13, w2, metadata)
        output.storage.float().sum().backward()
        self.assertEqual(output.valid_rows, 0)
        for value in (output.storage, storage.grad, w13.grad, w2.grad):
            self.assertEqual(torch.count_nonzero(value).item(), 0)


if __name__ == "__main__":
    unittest.main()
