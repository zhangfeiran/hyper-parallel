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
"""CPU semantics, source diagnostics and symbol boundaries for the AST frontend."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.ir.program import Value
from hyper_parallel.core.multicore.primitives.registry import (
    OpSchema,
    PrimitiveRegistry,
)
from tests.common.mark_utils import arg_mark

ROUTE_SHAPE = ("T", "E")
EXPERT_SHAPE = ("E",)


@mc.program(target="ascend", schedule=mc.WorkerPipeline())
def _route(
    logits: ml.Tensor[ml.fp32, ROUTE_SHAPE],
    bias: ml.Tensor[ml.fp32, EXPERT_SHAPE],
    k: ml.Constexpr[int],
    scale: ml.Constexpr[float],
):
    """Design-document route semantics, including static k=1 and bias detach."""
    scores = ml.sqrt(ml.softplus(logits))
    selection_scores = ml.add(scores, ml.stop_gradient(bias))
    indices = ml.topk_indices(selection_scores, k, axis=-1, sorted=False)
    selected = ml.gather(scores, indices, axis=-1)
    if k > 1:
        denominator = ml.add(ml.reduce_sum(selected, axis=-1, keepdim=True), 1.0e-20)
        selected = ml.divide(selected, denominator)
    weights = ml.multiply(selected, scale)
    return weights, ml.cast(indices, ml.int64)


@mc.helper
def activate(value: ml.Tensor[ml.fp32, ROUTE_SHAPE]) -> ml.Tensor[ml.fp32, ROUTE_SHAPE]:
    """Typed helper for two reference primitives."""
    return ml.sqrt(ml.softplus(value))


class TestProgram(unittest.TestCase):
    """Verify actual semantic behavior and reject unsupported Python execution."""

    def _source(self, body, parameters="value: ml.Tensor[ml.fp32, ROUTE_SHAPE]", **symbols):
        source = "def example(" + parameters + "):\n" + textwrap.indent(textwrap.dedent(body), "    ")
        return mc.from_source(
            source,
            symbols={
                "ml": ml,
                "ROUTE_SHAPE": ROUTE_SHAPE,
                "EXPERT_SHAPE": EXPERT_SHAPE,
                **symbols,
            },
            filename="example.py",
            first_line=40,
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_gate_route_matches_reference_and_gradients(self):
        """Compare both static branches and logits gradients against direct Torch.

        Feature: Typed AST frontend.
        Description: Compare both static branches and logits gradients against direct Torch.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        for k in (1, 3):
            with self.subTest(k=k):
                logits = torch.tensor([[0.1, -0.7, 1.2, 0.6], [0.8, 0.3, -0.4, 1.7]], requires_grad=True)
                bias = torch.tensor([0.0, 1.0, -1.0, 0.2], requires_grad=True)
                weights, indices = _route.interpret(logits, bias, k, 2.5)
                expected_input = logits.detach().clone().requires_grad_()
                scores = torch.sqrt(torch.nn.Softplus()(expected_input))
                expected_indices = torch.topk(scores + bias.detach(), k, dim=-1, sorted=False).indices
                selected = torch.gather(scores, -1, expected_indices)
                if k > 1:
                    selected = selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
                expected_weights = selected * 2.5
                torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
                coefficients = torch.arange(weights.numel(), dtype=weights.dtype).reshape(weights.shape)
                (weights * coefficients).sum().backward()
                (expected_weights * coefficients).sum().backward()
                torch.testing.assert_close(logits.grad, expected_input.grad, rtol=0, atol=0)
                self.assertIsNone(bias.grad)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_ir_specialization_types_and_dump(self):
        """Preserve source order, logical dtype and static attributes in JSON IR.

        Feature: Typed AST frontend.
        Description: Preserve source order, logical dtype and static attributes in JSON IR.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        ir = _route.lower(k=3, scale=2.5)
        names = [operation.logical_name for operation in ir.operations]
        self.assertEqual(
            names,
            [
                "gate.softplus",
                "gate.sqrt",
                "gate.stop_gradient",
                "gate.add",
                "gate.topk_indices",
                "gate.gather",
                "gate.reduce_sum",
                "gate.add",
                "gate.divide",
                "gate.multiply",
                "gate.cast",
            ],
        )
        self.assertEqual(ir.outputs[0].type.shape, ("T", 3))
        self.assertEqual(ir.outputs[1].type.dtype, ml.int64)
        self.assertEqual(ir.numeric_policy, "preserve_numeric_order")
        self.assertEqual(len(_route.lower(k=1, scale=2.5).operations), 8)
        self.assertEqual(json.loads(ir.dump())["constants"], {"k": 3, "scale": 2.5})
        self.assertEqual(ir.dump(), _route.lower(k=3, scale=2.5).dump())
        for operation in ir.operations:
            self.assertGreaterEqual(operation.source.line, 1)
            self.assertTrue(operation.effects)
        self.assertIn("backend=semantic-only", _route.explain(k=1, scale=2.5))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_alias_tuple_assignment_and_annotated_assignment(self):
        """Aliases resolve by identity and tuple unpacking preserves SSA values.

        Feature: Typed AST frontend.
        Description: Aliases resolve by identity and tuple unpacking preserves SSA values.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source(
            """
            activation = alias
            first: ml.Tensor[ml.fp32, ROUTE_SHAPE] = activation(value)
            second, third = (ml.sqrt(first), first)
            return second, third
        """,
            alias=ml.softplus,
        )
        ir = candidate.lower()
        self.assertEqual([op.logical_name for op in ir.operations], ["gate.softplus", "gate.sqrt"])
        self.assertIs(ir.outputs[1], ir.operations[0].outputs[0])
        actual = candidate.interpret(torch.ones(2, 4))
        torch.testing.assert_close(actual[0], torch.sqrt(torch.nn.Softplus()(torch.ones(2, 4))))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_helper_inline_preserves_definition_and_callsite(self):
        """Inline helper calls while preserving both definition and caller locations.

        Feature: Typed AST frontend.
        Description: Inline helper calls while preserving both definition and caller locations.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source("return activate(value)", activate=activate)
        ir = candidate.lower()
        self.assertEqual(len(ir.operations), 2)
        self.assertTrue(ir.operations[0].source.filename.endswith("test_program.py"))
        self.assertEqual(ir.operations[0].source.call_chain, ("example.py:41:12",))
        actual = candidate.interpret(torch.ones(2, 4))
        torch.testing.assert_close(actual, torch.sqrt(torch.nn.Softplus()(torch.ones(2, 4))))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_static_loop_unrolls_and_bounds_compilation(self):
        """Bounded loops compile without executing marker calls or runtime tensors.

        Feature: Typed AST frontend.
        Description: Bounded loops compile without executing marker calls or runtime tensors.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source("""
            for index in static_range(3):
                value = ml.add(value, index * 0.5)
            return value
        """)
        self.assertEqual(len(candidate.lower().operations), 3)
        torch.testing.assert_close(candidate.interpret(torch.zeros(2, 4)), torch.full((2, 4), 1.5))
        oversized = self._source("for index in static_range(1025):\n    value = ml.sqrt(value)\nreturn value")
        with self.assertRaisesRegex(mc.FrontendError, "bound of 1024"):
            oversized.lower()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_nested_capture_closure_and_original_line(self):
        """Capture nested functions and immutable closure scalars without executing bodies.

        Feature: Typed AST frontend.
        Description: Capture nested functions and immutable closure scalars without executing bodies.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        threshold = 2.0

        @mc.program
        def nested(
            value: ml.Tensor[ml.fp32, ROUTE_SHAPE],
        ) -> ml.Tensor[ml.fp32, ROUTE_SHAPE]:
            """Use a captured scalar as a static primitive argument."""
            return ml.multiply(value, threshold)

        ir = nested.lower()
        self.assertEqual(dict(ir.operations[0].arguments)["right"], threshold)
        span = ir.operations[0].source
        line = Path(span.filename).read_text(encoding="utf-8").splitlines()[span.line - 1]
        self.assertEqual(line[span.column : span.end_column], "ml.multiply(value, threshold)")
        torch.testing.assert_close(nested.interpret(torch.ones(2, 4)), torch.full((2, 4), threshold))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_explicit_signature_and_string_annotation(self):
        """from_source supports unavailable inspect source and future/string annotations.

        Feature: Typed AST frontend.
        Description: from_source supports unavailable inspect source and future/string annotations.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = mc.from_source(
            "def identity(value):\n    return value",
            {"value": ml.Tensor[ml.fp32, (2, 4)]},
        )
        self.assertEqual(candidate.lower().outputs[0], candidate.lower().inputs[0])
        quoted = self._source("return value", parameters="value: 'ml.Tensor[ml.fp32, (2, 4)]'")
        self.assertEqual(quoted.lower().inputs[0].type.shape, (2, 4))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_constexpr_default_and_optional_float(self):
        """Static defaults and None/float unions specialize without eval.

        Feature: Typed AST frontend.
        Description: Static defaults and None/float unions specialize without eval.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source(
            """
            if scale is None:
                return value
            return ml.multiply(value, scale)
        """,
            parameters="value: ml.Tensor[ml.fp32, ROUTE_SHAPE], scale: ml.Constexpr[float | None] = None",
        )
        self.assertEqual(len(candidate.lower().operations), 0)
        torch.testing.assert_close(candidate.interpret(torch.ones(2, 4)), torch.ones(2, 4))
        torch.testing.assert_close(candidate.interpret(torch.ones(2, 4), 3.0), torch.full((2, 4), 3.0))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_arbitrary_callable_is_not_executed(self):
        """Reject same-name callables and properties without running user hooks.

        Feature: Typed AST frontend.
        Description: Reject same-name callables and properties without running user hooks.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        impostor = Mock(name="sqrt")
        candidate = self._source("return sqrt(value)", sqrt=impostor)
        with self.assertRaisesRegex(mc.FrontendError, "not a registered primitive"):
            candidate.lower()
        impostor.assert_not_called()
        object_with_properties = Mock()
        with self.assertRaisesRegex(mc.FrontendError, "registered DSL language module"):
            self._source("return unsafe.sqrt(value)", unsafe=object_with_properties).lower()
        object_with_properties.assert_not_called()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_shadowing_and_unbound_locals(self):
        """Local names shadow global primitive aliases under Python lexical rules.

        Feature: Typed AST frontend.
        Description: Local names shadow global primitive aliases under Python lexical rules.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        cases = [
            ("sqrt = value\nreturn sqrt(value)", "not a registered primitive"),
            ("result = sqrt(value)\nsqrt = value\nreturn result", "before assignment"),
        ]
        for body, message in cases:
            with (
                self.subTest(body=body),
                self.assertRaisesRegex(mc.FrontendError, message),
            ):
                self._source(body, sqrt=ml.sqrt).lower()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_unsupported_syntax_and_runtime_branch_locations(self):
        """Reject dynamic control flow, in-place writes, arbitrary indexing and comprehensions.

        Feature: Typed AST frontend.
        Description: Reject dynamic control flow, in-place writes, arbitrary indexing and comprehensions.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        bodies = [
            "while True:\n    value = ml.sqrt(value)\nreturn value",
            "if value:\n    return value\nreturn value",
            "value[0] = 1\nreturn value",
            "value += 1\nreturn value",
            "return [ml.sqrt(value) for item in (1, 2)]",
            "return value[0]",
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(mc.FrontendError) as caught:
                self._source(body).lower()
            self.assertEqual(caught.exception.source.filename, "example.py")
            self.assertGreaterEqual(caught.exception.source.line, 41)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_shape_dtype_and_attribute_errors(self):
        """Validate type/shape/static primitive contracts before reference execution.

        Feature: Typed AST frontend.
        Description: Validate type/shape/static primitive contracts before reference execution.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        cases = [
            ("return ml.topk_indices(value, 0)", "positive static k"),
            ("return ml.topk_indices(value, 5)", "exceeds"),
            ("return ml.reduce_sum(value, axis=3)", "within tensor rank"),
            ("return ml.add(value, indices)", "dtypes must match"),
            ("return ml.add(value, other)", "broadcast compatibility"),
            ("return ml.gather(value, value)", "int64 indices"),
            ("return ml.sqrt(value, bogus=True)", "unexpected keyword"),
        ]
        parameters = "value: ml.Tensor[ml.fp32, (2, 4)], indices: ml.Tensor[ml.int32, (2, 4)], "
        parameters += "other: ml.Tensor[ml.fp32, (3,)]"
        for body, message in cases:
            with (
                self.subTest(body=body),
                self.assertRaisesRegex(mc.FrontendError, message),
            ):
                self._source(body, parameters=parameters).lower()
        with self.assertRaisesRegex(mc.FrontendError, "Invalid constexpr type"):
            _route.lower(k=True, scale=2.0)
        with self.assertRaisesRegex(mc.FrontendError, "Missing constexpr"):
            _route.lower(scale=2.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_interpreter_unifies_symbolic_shapes_and_dtype(self):
        """Reject inconsistent symbolic sizes and dtype without touching devices.

        Feature: Typed AST frontend.
        Description: Reject inconsistent symbolic sizes and dtype without touching devices.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        for logits, bias in [
            (torch.ones(2, 4), torch.ones(3)),
            (torch.ones(2, 4, dtype=torch.float16), torch.ones(4)),
            (torch.ones(4, 2).t(), torch.ones(4)),
        ]:
            with self.subTest(logits=logits.shape), self.assertRaises(ValueError):
                _route.interpret(logits, bias, 1, 1.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_mutable_and_runtime_constants_are_rejected(self):
        """Do not freeze tensors, arbitrary containers or oversized scalar values.

        Feature: Typed AST frontend.
        Description: Do not freeze tensors, arbitrary containers or oversized scalar values.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        for value in ([1], {"x": 1}, torch.ones(1), float("inf"), 2**100):
            with self.subTest(value=type(value)), self.assertRaises(mc.FrontendError):
                _route.lower(k=value, scale=2.0)
        with self.assertRaisesRegex(mc.FrontendError, "cannot be frozen"):
            _route.lower(logits=torch.ones(2, 4), k=1, scale=1.0)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_type_annotations_do_not_execute_calls(self):
        """Reject calls inside ordinary and string annotations before building any operations.

        Feature: Typed AST frontend.
        Description: Reject calls inside ordinary and string annotations before building any operations.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        for declaration in ("ml.Constexpr[unsafe()]", "'ml.Constexpr[unsafe()]'"):
            unsafe = Mock()
            candidate = self._source("return value", parameters=f"value: {declaration}", unsafe=unsafe)
            with (
                self.subTest(declaration=declaration),
                self.assertRaisesRegex(mc.FrontendError, "Calls are unsupported"),
            ):
                candidate.lower()
            unsafe.assert_not_called()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_return_annotations_are_checked(self):
        """Check complete single and tuple return types against inferred outputs.

        Feature: Typed AST frontend.
        Description: Check complete single and tuple return types against inferred outputs.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        for dtype in ("fp32", "int32"):
            candidate = mc.from_source(
                f"def identity(value: ml.Tensor[ml.fp32, (2,)]) -> ml.Tensor[ml.{dtype}, (2,)]:\n    return value",
                symbols={"ml": ml},
            )
            if dtype == "fp32":
                self.assertEqual(candidate.lower().outputs[0].type.dtype, ml.fp32)
            else:
                with self.assertRaisesRegex(mc.FrontendError, "Return annotation disagrees"):
                    candidate.lower()
        pair = mc.from_source(
            "def pair(value: ml.Tensor[ml.fp32, (2,)]) -> (ml.Tensor[ml.fp32, (2,)], ml.Tensor[ml.fp32, (2,)]):"
            "\n    return value, value",
            symbols={"ml": ml},
        )
        self.assertEqual(len(pair.lower().outputs), 2)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_static_specialization_validates_unselected_syntax(self):
        """Unsupported syntax is diagnosed even when constexpr would remove a branch.

        Feature: Typed AST frontend.
        Description: Unsupported syntax is diagnosed even when constexpr would remove a branch.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source("if True:\n    return value\nelse:\n    return [value for index in (1,)]")
        with self.assertRaisesRegex(mc.FrontendError, "Unsupported expression: ListComp"):
            candidate.lower()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_defaults_use_definition_scope_and_missing_inputs_fail(self):
        """Default values resolve in definition scope, preserving ordinary argument binding.

        Feature: Typed AST frontend.
        Description: Default values resolve in definition scope, preserving ordinary argument binding.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        candidate = self._source(
            "return ml.multiply(value, scale)",
            parameters="value: ml.Tensor[ml.fp32, ROUTE_SHAPE], scale: ml.Constexpr[float] = scale",
            scale=2.0,
        )
        torch.testing.assert_close(candidate.interpret(torch.ones(2, 4)), torch.full((2, 4), 2.0))
        with self.assertRaisesRegex(TypeError, "Missing argument"):
            candidate.interpret()
        earlier_static = mc.from_source(
            "def multiply(scale: ml.Constexpr[float], value: ml.Tensor[ml.fp32, (2,)]):"
            "\n    return ml.multiply(value, scale)",
            constants={"scale": 3.0},
            symbols={"ml": ml},
        )
        torch.testing.assert_close(earlier_static.interpret(value=torch.ones(2)), torch.full((2,), 3.0))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_helper_default_and_error_call_chain(self):
        """Helpers normalize keyword/default arguments and diagnose their original source.

        Feature: Typed AST frontend.
        Description: Helpers normalize keyword/default arguments and diagnose their original source.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """

        @mc.helper
        def scaled(
            value: ml.Tensor[ml.fp32, ROUTE_SHAPE], scale: ml.Constexpr[float] = 2.0
        ) -> ml.Tensor[ml.fp32, ROUTE_SHAPE]:
            """Use a typed helper default."""
            return ml.multiply(value, scale)

        candidate = self._source("return scaled(value=value)", scaled=scaled)
        torch.testing.assert_close(candidate.interpret(torch.ones(2, 4)), torch.full((2, 4), 2.0))
        incorrect = self._source("return scaled(value, 1)", scaled=scaled)
        with self.assertRaisesRegex(mc.FrontendError, "Helper constexpr argument type mismatch") as caught:
            incorrect.lower()
        self.assertEqual(len(caught.exception.source.call_chain), 1)
        self.assertTrue(caught.exception.source.filename.endswith("test_program.py"))

        @mc.helper
        def recursive(value: ml.Tensor[ml.fp32, ROUTE_SHAPE]) -> ml.Tensor[ml.fp32, ROUTE_SHAPE]:
            """Exercise recursive helper rejection without executing Python code."""
            return recursive(value)

        with self.assertRaisesRegex(mc.FrontendError, "Recursive helpers"):
            self._source("return recursive(value)", recursive=recursive).lower()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_frontend_import_does_not_require_npu(self):
        """Block torch_npu imports in a fresh process and import the frontend.

        Feature: Typed AST frontend.
        Description: Block torch_npu imports in a fresh process and import the frontend.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        script = """
import importlib.abc
import sys
class BlockNpu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise RuntimeError('unexpected torch_npu import')
sys.meta_path.insert(0, BlockNpu())
import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
program = mc.from_source('def identity(value):\\n    return value', {'value': ml.Tensor[ml.fp32, (2,)]})
program.lower()
assert 'torch_npu' not in sys.modules
assert 'hyper_parallel.core.multicore.modules.mega_moe.function' not in sys.modules
"""
        environment = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"returncode={result.returncode}, stderr={result.stderr}",
        )


class TestRegistry(unittest.TestCase):
    """Version/schema identity tests, including extensible multi-result primitives."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_unique_registration_and_identity(self):
        """Reject logical collisions and same-name impostors, preserving dtype distinctions.

        Feature: Typed AST frontend.
        Description: Reject logical collisions and same-name impostors, preserving dtype distinctions.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        registry = PrimitiveRegistry()
        schema = OpSchema(
            "test.identity",
            1,
            inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)]),
            lambda args: (args["value"],),
            lambda value: value,
        )
        symbol = registry.register(schema)
        self.assertIs(registry.resolve(symbol), schema)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(schema)
        with self.assertRaisesRegex(ValueError, "not a registered"):
            registry.resolve(Mock())
        self.assertNotEqual(ml.Tensor[ml.fp32, (2,)], ml.Tensor[ml.int32, (2,)])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_registered_multi_result_primitive_and_reference(self):
        """Normalize tuple-result custom schemas and interpret them by registry identity.

        Feature: Typed AST frontend.
        Description: Normalize tuple-result custom schemas and interpret them by registry identity.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        registry = PrimitiveRegistry()
        schema = OpSchema(
            "test.pair",
            1,
            inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)]),
            lambda args: (args["value"], args["value"]),
            lambda value: (value, value.clone()),
        )
        pair = registry.register(schema)
        candidate = mc.from_source(
            "def pair_program(value):\n    first, second = pair(value)\n    return first, second",
            {"value": ml.Tensor[ml.fp32, (2,)]},
            symbols={"pair": pair},
            registry=registry,
        )
        outputs = candidate.interpret(torch.ones(2))
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(isinstance(value, Value) for value in candidate.lower().outputs))
        torch.testing.assert_close(outputs[0], outputs[1])


class TestPrimitiveEffects(unittest.TestCase):
    """Logical accesses are declared by schemas and checked against tensor arguments."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_declared_effects_survive_lowering(self):
        """Preserve registered logical reductions instead of replacing them with reads.

        Feature: Typed AST frontend.
        Description: Preserve registered logical reductions instead of replacing them with reads.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        registry = PrimitiveRegistry()
        signature = inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        symbol = registry.register(
            OpSchema(
                "test.reduce",
                1,
                signature,
                lambda args: (args["value"],),
                lambda value: value.clone(),
                lambda args: (("reduce", "value"),),
            )
        )
        candidate = mc.from_source(
            "def reduction(value):\n    return reduce(value)",
            {"value": ml.Tensor[ml.fp32, (2,)]},
            symbols={"reduce": symbol},
            registry=registry,
        )
        effects = candidate.lower().operations[0].effects
        self.assertEqual([effect.kind for effect in effects], ["reduce", "write"])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_effects_reject_unknown_arguments(self):
        """An invalid schema cannot silently emit accesses to nonexistent tensor inputs.

        Feature: Typed AST frontend.
        Description: An invalid schema cannot silently emit accesses to nonexistent tensor inputs.
        Expectation: All stated semantic and diagnostic contracts are enforced.
        """
        registry = PrimitiveRegistry()
        signature = inspect.Signature([inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        symbol = registry.register(
            OpSchema(
                "test.invalid_effect",
                1,
                signature,
                lambda args: (args["value"],),
                None,
                lambda args: (("read", "missing"),),
            )
        )
        candidate = mc.from_source(
            "def broken(value):\n    return op(value)",
            {"value": ml.Tensor[ml.fp32, (2,)]},
            symbols={"op": symbol},
            registry=registry,
        )
        with self.assertRaisesRegex(mc.FrontendError, "effects must name tensor arguments"):
            candidate.lower()
