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
"""Dense type, provider, SSA scheduling and source-emission contracts."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.providers import DENSE_PROVIDERS, ProviderRegistry
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.cache import EmissionCache
from tests.common.mark_utils import arg_mark


class TestDense(unittest.TestCase):
    """Verify real shape constraints and generic graph execution without devices."""

    def _spec(self, rows=7, packed=12):
        return mc.DenseSpec({"T": rows, "H": 8, "PackedI": packed, "I": 6})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_ffn_plan_has_ssa_dependencies_and_backward_lifetimes(self):
        """Feature: Dense AST lowering.
        Description: Derive dependencies and retained intermediates from the three primitive SSA graph.
        Expectation: The plan contains no expert fields and retains both backward inputs.
        """
        plan = dense_ffn.plan(self._spec(), intermediate_size=6)
        self.assertEqual([task.dependencies for task in plan.tasks], [(), (0,), (1,)])
        self.assertEqual([task.provider.worker for task in plan.tasks], ["cube", "vector", "cube"])
        self.assertEqual([buffer.size_bytes for buffer in plan.buffers], [168, 84, 112])
        self.assertEqual([buffer.saved_for_backward for buffer in plan.buffers], [True, True, False])
        self.assertNotIn("ep_size", plan.export_manifest())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_binding_enforces_symbolic_packed_width_relation(self):
        """Feature: Dense shape binding.
        Description: Bind a packed dimension inconsistent with the constexpr activation width.
        Expectation: The actual bound primitive contract rejects the plan before execution.
        """
        with self.assertRaisesRegex(ValueError, "twice intermediate_size"):
            dense_ffn.plan(self._spec(packed=14), intermediate_size=6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_generic_dense_dag_supports_fanout_and_tuple_returns(self):
        """Feature: Generic dense TaskDAG.
        Description: Lower two parallel matrix products sharing one input without an FFN matcher.
        Expectation: Independent tasks return both products and sum their input gradients.
        """
        program = mc.from_source(
            "def fork(x, a, b):\n    first = ml.matmul(x, a)\n    second = ml.matmul(x, b)\n    return first, second\n",
            signature={"x": ml.Tensor[ml.fp32, (3, 4)], "a": ml.Tensor[ml.fp32, (4, 2)],
                       "b": ml.Tensor[ml.fp32, (4, 5)]},
            symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"),
        )
        plan = program.plan(mc.DenseSpec({}))
        self.assertEqual([task.dependencies for task in plan.tasks], [(), ()])
        x = torch.randn(3, 4, requires_grad=True)
        a, b = torch.randn(4, 2), torch.randn(4, 5)
        first, second = plan.materialize("cpu")(x, a, b)
        torch.testing.assert_close(first, x @ a)
        torch.testing.assert_close(second, x @ b)
        (first.sum() + second.sum()).backward()
        torch.testing.assert_close(x.grad, (a.sum(1) + b.sum(1)).expand_as(x))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_transpose_is_a_primitive_attribute_and_preserves_gradients(self):
        """Feature: Dense transpose contract.
        Description: Execute an ordinary Linear-layout weight through a transpose attribute.
        Expectation: Shape and gradients match the native Torch matrix product.
        """
        program = mc.from_source(
            "def linear(x, weight):\n    return ml.matmul(x, weight, transpose_right=True)\n",
            signature={"x": ml.Tensor[ml.fp32, (3, 4)], "weight": ml.Tensor[ml.fp32, (2, 4)]},
            symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"),
        )
        x, weight = torch.randn(3, 4, requires_grad=True), torch.randn(2, 4, requires_grad=True)
        actual = program.plan(mc.DenseSpec({})).materialize("cpu")(x, weight)
        expected = x @ weight.t()
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), (x, weight), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (x, weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_provider_registry_rejects_schema_substitution_and_duplicates(self):
        """Feature: Trusted native providers.
        Description: Attempt a same-fields replacement schema and duplicate canonical registration.
        Expectation: Both fail without executing a supplied callable.
        """
        registry = ProviderRegistry()
        provider = DENSE_PROVIDERS.resolve(ml.matmul.schema)
        with self.assertRaisesRegex(ValueError, "canonical"):
            registry.register(replace(ml.matmul.schema), provider)
        registry.register(ml.matmul.schema, provider)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            registry.register(ml.matmul.schema, provider)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_spec_copies_dimensions_and_rejects_missing_bindings(self):
        """Feature: Dense static specification.
        Description: Mutate the caller mapping after construction and omit a required symbol.
        Expectation: The specification is immutable and incomplete binding is rejected.
        """
        dimensions = {"T": 7, "H": 8, "PackedI": 12, "I": 6}
        spec = mc.DenseSpec(dimensions)
        dimensions["T"] = 99
        self.assertEqual(spec.dimensions["T"], 7)
        with self.assertRaises(TypeError):
            spec.dimensions["T"] = 2
        with self.assertRaisesRegex(ValueError, "Unbound"):
            dense_ffn.plan(mc.DenseSpec({"T": 7}), intermediate_size=6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_source_cache_identity_separates_bound_shapes(self):
        """Feature: Dense source artifact identity.
        Description: Emit the same semantic FFN at two token counts and store both bundles.
        Expectation: Definition identity is shared while plans and artifacts differ and verify.
        """
        first = dense_ffn.compile(self._spec(7), intermediate_size=6)
        second = dense_ffn.compile(self._spec(9), intermediate_size=6)
        self.assertEqual(first.definition_key, second.definition_key)
        self.assertNotEqual(first.plan_key, second.plan_key)
        self.assertNotEqual(first.artifact_key, second.artifact_key)
        self.assertEqual(first.export_manifest()["status"], "source_only")
        with tempfile.TemporaryDirectory() as directory:
            cache = EmissionCache(Path(directory))
            cache.store(first)
            cache.store(second)
            self.assertIn("bindings/providers.json", cache.load(first.artifact_key))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_dense_primitives_reject_invalid_storage_and_static_flags(self):
        """Feature: Dense primitive admission.
        Description: Pass mismatched dimensions and a nonboolean transpose attribute.
        Expectation: Semantic parsing rejects invalid arithmetic before planning.
        """
        for right, transpose in ((ml.Tensor[ml.fp32, (5, 2)], False), (ml.Tensor[ml.fp32, (4, 2)], 1)):
            with self.subTest(right=right, transpose=transpose):
                program = mc.from_source(
                    "def invalid(x, weight):\n    return ml.matmul(x, weight, transpose_right=flag)\n",
                    signature={"x": ml.Tensor[ml.fp32, (3, 4)], "weight": right},
                    symbols={"ml": ml, "flag": transpose},
                )
                with self.assertRaises(mc.FrontendError):
                    program.lower()
