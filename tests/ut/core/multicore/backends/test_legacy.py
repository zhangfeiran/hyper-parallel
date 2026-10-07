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
"""Fixed-source ABI, legacy plan parity and illegal-lowering CPU tests."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.legacy import gate_runtime_image
from hyper_parallel.core.multicore.compiler.pipeline import compile_worker_pipeline
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.ir.program import Effect
from hyper_parallel.core.multicore.primitives.registry import (
    REGISTRY,
    PrimitiveRegistry,
)
from hyper_parallel.core.multicore.runtime.abi import NativeManifest, family_abi
from hyper_parallel.core.multicore.scheduler import config as current_config
from tests.common.mark_utils import arg_mark

FIXTURES = Path(__file__).parent / "fixtures"


def _variant(*replacements):
    source = _route.source.text
    for old, new in replacements:
        source = source.replace(old, new)
    return mc.from_source(source, symbols=_route.source.symbols, schedule=mc.WorkerPipeline())


class TestFamilyABI(unittest.TestCase):
    """Cross-family identity and actual legacy C++ field layouts."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_family_task_ids_are_namespaced(self):
        """Numeric collisions are confined to family mappings, while logical names differ.

        Feature: Legacy Gate compatibility lowering.
        Description: Numeric collisions are confined to family mappings, while logical names differ.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        tasks = [
            next(task for task in family_abi(family).task_types if task.numeric_id == 107)
            for family in ("moe", "mhc", "gate")
        ]
        self.assertEqual(len({task.logical_name for task in tasks}), 3)
        self.assertEqual(
            [task.native_name for task in tasks], ["TASK_SHMEM_GET_MEM", "TASK_MHC_POST", "TASK_MEGA_GATE_ROUTE"]
        )
        with self.assertRaisesRegex(ValueError, "Unknown moe task"):
            family_abi("moe").task_id("gate.softplus.v1")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_moe_schema_matches_current_python_layout(self):
        """The pinned MoE header preserves completion/protocol fields and existing ctypes offsets.

        Feature: Legacy Gate compatibility lowering.
        Description: The pinned MoE header preserves completion/protocol fields and existing ctypes offsets.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        for structure in family_abi("moe").structures:
            actual = getattr(current_config, structure.name)
            self.assertEqual(ctypes.sizeof(actual), structure.size)
            for field in structure.fields:
                self.assertEqual(getattr(actual, field.name).offset, field.offset)
        self.assertEqual(family_abi("moe").structure("RuntimeConfigC").offset("protocol_version"), 36)
        self.assertEqual(family_abi("gate").structure("RuntimeConfigC").offset("_padding"), 32)
        self.assertEqual(family_abi("mhc").scratch_policy, "fixed_512_int64")
        self.assertEqual(family_abi("moe").scratch_policy, "dynamic_local_experts")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_python_schema_matches_original_cpp_layouts(self):
        """Compile extracted original C++ declarations and check all sizes/offsets against Python schema.

        Feature: Legacy Gate compatibility lowering.
        Description: Compile extracted original C++ declarations and check all sizes/offsets against Python schema.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        for family in ("moe", "mhc", "gate"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                declarations = (FIXTURES / f"{family}_runtime_layout.txt").read_text(encoding="utf-8")
                lines = [declarations, "using namespace MulticoreRuntime;"]
                for structure in family_abi(family).structures:
                    name = structure.name.removesuffix("C")
                    if name == "RuntimeConfig":
                        name = "RuntimeHeader"
                    lines.append(f'static_assert(sizeof({name}) == {structure.size}, "size");')
                    for field in structure.fields:
                        field_name = "padding" if field.name == "_padding" else field.name
                        lines.append(f'static_assert(offsetof({name}, {field_name}) == {field.offset}, "offset");')
                lines.append("int main() { return 0; }")
                source = root / "layout.cpp"
                source.write_text("\n".join(lines), encoding="utf-8")
                result = subprocess.run(
                    ["c++", "-std=c++17", str(source), "-o", str(root / "layout")],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, f"returncode={result.returncode}, errors={result.stderr}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_manifest_requires_exact_family_schema_and_source(self):
        """Reject wrong family, schema and dependency/source fingerprints before any future launch.

        Feature: Legacy Gate compatibility lowering.
        Description: Reject wrong family, schema and dependency/source fingerprints before any future launch.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        abi = family_abi("gate")
        manifest = NativeManifest(
            abi.family, abi.runtime_abi_version, abi.schema_hash, abi.source_fingerprint, "f" * 64
        )
        abi.verify_native(manifest)
        wrong = [
            replace(manifest, family="moe"),
            replace(manifest, runtime_abi_version=2),
            replace(manifest, schema_hash="e" * 64),
            replace(manifest, source_fingerprint="e" * 64),
            replace(manifest, build_fingerprint="missing"),
        ]
        for candidate in wrong:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                abi.verify_native(candidate)
        self.assertNotEqual(family_abi("moe").schema_hash, abi.schema_hash)
        self.assertEqual(len(abi.source_fingerprint), 64)
        self.assertIsNone(abi.export_manifest()["native_build_fingerprint"])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_gate_serializer_rejects_foreign_family(self):
        """Gate wire output never silently reuses a MoE or MHC layout.

        Feature: Legacy Gate compatibility lowering.
        Description: Gate wire output never silently reuses a MoE or MHC layout.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        for family in ("moe", "mhc"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "matching Gate"):
                gate_runtime_image(family_abi(family), family_abi("gate").pipeline("forward"))


class TestGatePlan(unittest.TestCase):
    """Native descriptor parity with strict numerical-recipe matching."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_normal_and_profiled_images_match_original_builder(self):
        """Both specializations reproduce all six captured wire images exactly.

        Feature: Legacy Gate compatibility lowering.
        Description: Both specializations reproduce all six captured wire images exactly.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        snapshots = json.loads((FIXTURES / "gate_snapshots.json").read_text(encoding="utf-8"))
        for count in (1, 3):
            plan = _route.plan({"T": 33, "E": 4}, k=count, scale=2.5)
            for direction, runtime in (
                ("forward", plan.forward),
                ("backward_k1" if count == 1 else "backward", plan.backward),
            ):
                for suffix, actual in (("normal", runtime.normal), ("profiled", runtime.profiled)):
                    with self.subTest(count=count, direction=direction, suffix=suffix):
                        expected = (FIXTURES / f"gate_{direction}_{suffix}.bin").read_bytes()
                        self.assertEqual(actual, expected)
                        self.assertEqual(
                            hashlib.sha256(actual).hexdigest(),
                            snapshots["sequences"][direction]["images"][suffix]["sha256"],
                        )
                self.assertEqual(
                    [stage.display_name for stage in runtime.stages], snapshots["sequences"][direction]["stage_names"]
                )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_k1_retains_native_normalization_descriptors(self):
        """Keep the legacy forward sequence even when static specialization removes IR normalization.

        Feature: Legacy Gate compatibility lowering.
        Description: Keep the legacy forward sequence even when static specialization removes IR normalization.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        self.assertEqual(len(_route.lower(k=1, scale=2.0).operations), 8)
        plan = _route.plan({"T": 3, "E": 4}, k=1, scale=2.0)
        self.assertEqual(len(plan.schedule.stages), 10)
        self.assertEqual(len(plan.backward.stages), 2)
        self.assertEqual([stage.reason for stage in plan.schedule.stages[5:8]], ["retained_legacy_k1_stage"] * 3)
        self.assertTrue(all(stage.source_spans for stage in plan.schedule.stages))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_row_ownership_matches_native_launch_and_empty_tail(self):
        """Preserve launched workers, including zero-row tails, while covering each row exactly once.

        Feature: Legacy Gate compatibility lowering.
        Description: Preserve launched workers, including zero-row tails, while covering each row exactly once.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        for tokens, workers in ((1, 48), (3, 48), (10, 6), (33, 8), (101, 96)):
            with self.subTest(tokens=tokens, workers=workers):
                plan = _route.plan({"T": tokens, "E": 4}, mc.HardwareSpec(workers), k=3, scale=1.0)
                partitions = plan.schedule.partitions
                self.assertEqual(len(partitions), min(tokens, workers, 48))
                rows = [row for part in partitions for row in range(part.first_row, part.first_row + part.row_count)]
                self.assertEqual(rows, list(range(tokens)))
                simulated = plan.schedule.simulate()
                self.assertEqual(len(simulated), len(partitions) * 10)
                for part in partitions:
                    self.assertEqual(
                        [stage for worker, stage, _, _ in simulated if worker == part.worker_id], list(range(10))
                    )
                if (tokens, workers) == (10, 6):
                    self.assertEqual(partitions[-1].row_count, 0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_wire_images_are_independent_of_shape_and_available_workers(self):
        """Shape/topology affect host row ownership, not broadcast descriptor count or family slot capacity.

        Feature: Legacy Gate compatibility lowering.
        Description: Shape/topology affect host row ownership, not broadcast descriptor count or family slot capacity.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        first = _route.plan({"T": 2, "E": 4}, mc.HardwareSpec(3), k=3, scale=1.0)
        second = _route.plan({"T": 33, "E": 8}, mc.HardwareSpec(48), k=3, scale=1.0)
        self.assertEqual(first.forward.normal, second.forward.normal)
        self.assertEqual(first.backward.normal, second.backward.normal)
        self.assertNotEqual(first.schedule.partitions, second.schedule.partitions)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_bindings_and_backward_external_boundary(self):
        """Preserve owned caches, detached text/vision alias and separate CANN/direct-logits backward calls.

        Feature: Legacy Gate compatibility lowering.
        Description: Preserve owned caches, detached text/vision alias and separate CANN/direct-logits backward calls.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        plan = _route.plan({"T": 2, "E": 4}, k=3, scale=1.0)
        names = [binding.name for binding in plan.bindings]
        self.assertEqual(names[0:4], ["logits", "text_bias", "vision_bias(text_alias)", "image_mask_placeholder"])
        self.assertEqual(plan.bindings[1].value_id, plan.bindings[2].value_id)
        self.assertEqual(len(plan.bindings), 13)
        self.assertEqual(
            [binding.name for binding in plan.backward_bindings],
            [
                "selected_scores",
                "normalization_denominator",
                "grad_routing_weights",
                "route_scores",
                "expert_indices",
                "runtime_config",
                "profile_buffer",
                "selected_score_grad",
                "zero_score_grad",
                "workspace",
                "tiling",
            ],
        )
        self.assertEqual(
            plan.saved_state,
            ("logits", "expert_indices", "route_scores", "selected_scores", "normalization_denominator"),
        )
        self.assertIn("HyperMegaGateSoftplusV2Grad", plan.external_backward_calls)
        self.assertEqual(plan.external_backward_calls[-1], "HyperMegaGateAddGrad(optional_direct)")
        manifest = plan.export_manifest()
        self.assertEqual(manifest["native_status"], "unbound")
        self.assertEqual(manifest["device_tiling"], "required_from_legacy_host")
        self.assertIsNone(manifest["native_build_fingerprint"])
        self.assertEqual(json.loads(plan.explain())["manifest"]["family"], "gate")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_unsupported_numeric_variants_are_rejected(self):
        """Reject changes to normalization, detached bias, selected scores, indices order and dtype.

        Feature: Legacy Gate compatibility lowering.
        Description: Reject changes to normalization, detached bias, selected scores, indices order and dtype.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        changes = [
            ("1.0e-20", "1.0e-10"),
            ("ml.stop_gradient(bias)", "bias"),
            ("ml.gather(scores, indices", "ml.gather(selection, indices"),
            ("sorted=False", "sorted=True"),
            ("ml.int64)", "ml.int32)"),
            ("ml.multiply(selected, scale)", "ml.multiply(selected, ml.stop_gradient(bias))"),
        ]
        for old, new in changes:
            with self.subTest(change=(old, new)), self.assertRaises(ValueError):
                _variant((old, new)).plan({"T": 2, "E": 4}, k=3, scale=1.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_k1_normalization_is_not_silently_removed(self):
        """Native k=1 behavior rejects a semantic program that still normalizes selected scores.

        Feature: Legacy Gate compatibility lowering.
        Description: Native k=1 behavior rejects a semantic program that still normalizes selected scores.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        with self.assertRaisesRegex(ValueError, "gate.gather"):
            _variant(("if k > 1:", "if k >= 1:")).plan({"T": 2, "E": 4}, k=1, scale=1.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_invalid_shape_and_topology_boundaries(self):
        """Reject missing/extra dimensions, uint32 overflow, zero counts and impossible top-k.

        Feature: Legacy Gate compatibility lowering.
        Description: Reject missing/extra dimensions, uint32 overflow, zero counts and impossible top-k.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        shapes = [
            {},
            {"T": 2},
            {"T": 2, "E": 4, "unused": 1},
            {"T": 0, "E": 4},
            {"T": True, "E": 4},
            {"T": 2**32, "E": 4},
            {"T": 2, "E": 2},
        ]
        for signature in shapes:
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                _route.plan(signature, k=3, scale=1.0)
        for workers in (0, -1, True, 2**32):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                mc.HardwareSpec(workers)
        with self.assertRaisesRegex(ValueError, "representable as FP32"):
            _route.plan({"T": 2, "E": 4}, k=3, scale=1.0e100)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_additional_ops_or_mutating_effects_are_rejected(self):
        """A fixed template cannot discard extra computation or a changed primitive access contract.

        Feature: Legacy Gate compatibility lowering.
        Description: A fixed template cannot discard extra computation or a changed primitive access contract.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        candidate = _variant(("return weights,", "unused = ml.softplus(logits)\n    return weights,"))
        with self.assertRaisesRegex(ValueError, "additional operations"):
            candidate.plan({"T": 2, "E": 4}, k=3, scale=1.0)
        ir = _route.lower(k=3, scale=1.0)
        modified = replace(ir.operations[0], effects=(Effect("write", ir.inputs[0].id),))
        altered = replace(ir, operations=(modified,) + ir.operations[1:])
        with self.assertRaisesRegex(ValueError, "pure read"):
            compile_worker_pipeline(altered, mc.WorkerPipeline(), {"T": 2, "E": 4}, mc.HardwareSpec())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_explicit_static_shape_and_aliases(self):
        """Resolve annotation-free source by explicit tensor types and preserve registered aliases.

        Feature: Legacy Gate compatibility lowering.
        Description: Resolve annotation-free source by explicit tensor types and preserve registered aliases.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        variant = _variant(("ml.sqrt", "sqrt_alias"))
        variant.source.symbols["sqrt_alias"] = ml.sqrt
        plan = variant.plan({"T": 2, "E": 4}, k=3, scale=1.0)
        self.assertEqual(plan.forward.normal, _route.plan({"T": 2, "E": 4}, k=3, scale=1.0).forward.normal)
        static = _variant(("TOKEN_EXPERT_SHAPE", "(2, 4)"), ("EXPERT_SHAPE", "(4,)"))
        self.assertEqual(static.plan(k=3, scale=1.0).token_count, 2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_custom_schema_cannot_impersonate_canonical_gate(self):
        """A same-name schema in another registry cannot claim the canonical native Route recipe.

        Feature: Legacy Gate compatibility lowering.
        Description: A same-name schema in another registry cannot claim the canonical native Route recipe.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        registry = PrimitiveRegistry()
        language = ModuleType("hyper_parallel.core.multicore.language")
        for name in ("Tensor", "Constexpr", "fp32", "int64"):
            setattr(language, name, getattr(ml, name))
        for name in (
            "softplus",
            "sqrt",
            "stop_gradient",
            "add",
            "topk_indices",
            "gather",
            "reduce_sum",
            "divide",
            "multiply",
            "cast",
        ):
            schema = REGISTRY.resolve(getattr(ml, name))
            if name == "sqrt":
                schema = replace(schema, reference=lambda value: value)
            setattr(language, name, registry.register(schema))
        candidate = mc.from_source(
            _route.source.text,
            symbols={**_route.source.symbols, "ml": language},
            registry=registry,
            schedule=mc.WorkerPipeline(),
        )
        with self.assertRaisesRegex(ValueError, "canonical registered primitive"):
            candidate.plan({"T": 2, "E": 4}, k=3, scale=1.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_plan_import_and_generation_do_not_load_npu(self):
        """Block all NPU imports in a fresh process and compile actual legacy images.

        Feature: Legacy Gate compatibility lowering.
        Description: Block all NPU imports in a fresh process and compile actual legacy images.
        Expectation: The specified ABI, parity and rejection contracts hold.
        """
        script = """
import importlib.abc
import sys
class BlockNpu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise RuntimeError('unexpected NPU import')
sys.meta_path.insert(0, BlockNpu())
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
plan = _route.plan({'T': 33, 'E': 4}, k=3, scale=1.0)
assert len(plan.forward.stages) == 10
assert len(plan.backward.stages) == 11
assert 'torch_npu' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0"),
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode}, stderr={result.stderr}")
