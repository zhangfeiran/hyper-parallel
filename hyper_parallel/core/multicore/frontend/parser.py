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
"""Restricted AST lowering with source locations, specialization and helper inlining."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from hyper_parallel.core.multicore.frontend.diagnostics import FrontendError
from hyper_parallel.core.multicore.frontend.symbols import (
    annotation,
    attribute,
    frozen,
    static_operation,
)
from hyper_parallel.core.multicore.ir.program import (
    Effect,
    Operation,
    ProgramIR,
    SourceSpan,
    Value,
)
from hyper_parallel.core.multicore.language.types import (
    LOGICAL_TYPES,
    ConstexprType,
    DType,
    LogicalType,
)
from hyper_parallel.core.multicore.primitives.registry import (
    Primitive,
    PrimitiveRegistry,
)


@dataclass(frozen=True)
class Source:
    """Captured source and explicit symbolic environment."""

    text: str
    filename: str
    first_line: int
    symbols: dict[str, object]
    column_offsets: tuple[int, ...] = ()


@dataclass
class IRBuilder:
    """Compilation-local SSA/effect state shared by inlined helpers."""

    operations: list[Operation] = field(default_factory=list)
    next_id: int = 0
    unrolled_iterations: int = 0

    def value(self, name: str, tensor_type: LogicalType) -> Value:
        """Allocate a new SSA result.

        Args:
            name: Symbol or logical value name.
            tensor_type: Complete logical result type.
        """
        result = Value(self.next_id, name, tensor_type)
        self.next_id += 1
        return result


class Parser:
    """Compile one function without executing Python user code."""

    def __init__(
        self,
        source: Source,
        registry: PrimitiveRegistry,
        builder: IRBuilder | None = None,
    ) -> None:
        """Capture one function and validate statement syntax."""
        self.source = source
        self.registry = registry
        self.builder = builder if builder is not None else IRBuilder()
        self.env = {}
        self.call_chain = ()
        self.active_helpers = ()
        self.outputs = None
        self.returns_tuple = False
        self.function = self._parse_function()
        self.local_names = {
            node.id
            for node in ast.walk(self.function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.local_names.update(arg.arg for arg in self.function.args.args)
        self._validate_subset(self.function.body)

    def _parse_function(self):
        source = self.source
        try:
            tree = ast.parse(source.text, filename=source.filename)
        except SyntaxError as exc:
            span = SourceSpan(
                source.filename,
                source.first_line + (exc.lineno or 1) - 1,
                (exc.offset or 1) - 1,
                source.first_line + (exc.lineno or 1) - 1,
                0,
            )
            raise FrontendError(exc.msg, span) from exc
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or len(tree.body) != 1:
            raise FrontendError("Source must contain exactly one synchronous function", self.span(tree))
        function = functions[0]
        arguments = function.args
        if arguments.posonlyargs or arguments.kwonlyargs or arguments.vararg or arguments.kwarg:
            raise self.error(function, "V0 supports ordinary named parameters only")
        return function

    def _validate_subset(self, statements):
        for node in statements:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise self.error(node, "Chained assignments are unsupported")
                self._validate_target(node.targets[0])
            elif isinstance(node, ast.AnnAssign):
                self._validate_target(node.target)
            elif isinstance(node, ast.For) and not isinstance(node.target, ast.Name):
                raise self.error(node.target, "Static loops require a local name")
            for name in ("value", "test", "iter"):
                expression = getattr(node, name, None)
                if expression is not None:
                    self._validate_expression(expression)
            if isinstance(node, (ast.If, ast.For)):
                self._validate_subset(node.body)
                self._validate_subset(node.orelse)
            elif not isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return)):
                if (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    continue
                raise self.error(node, f"Unsupported statement: {type(node).__name__}")

    def _validate_target(self, target):
        for node in ast.walk(target):
            if isinstance(node, ast.expr) and not isinstance(node, (ast.Name, ast.Tuple)):
                raise self.error(node, "Assignment targets require local names or tuple unpacking")

    def _validate_expression(self, root):
        supported = (
            ast.Constant,
            ast.Name,
            ast.Attribute,
            ast.Tuple,
            ast.Call,
            ast.UnaryOp,
            ast.BinOp,
            ast.Compare,
            ast.BoolOp,
        )
        for node in ast.walk(root):
            if isinstance(node, ast.expr) and not isinstance(node, supported):
                raise self.error(node, f"Unsupported expression: {type(node).__name__}")

    def span(self, node: ast.AST) -> SourceSpan:
        """Map dedented AST coordinates back to the captured file.

        Args:
            node: Python AST node to resolve.
        """
        line_index = getattr(node, "lineno", 1) - 1
        end_index = getattr(node, "end_lineno", 1) - 1
        start_offset = self.source.column_offsets[line_index] if self.source.column_offsets else 0
        end_offset = self.source.column_offsets[end_index] if self.source.column_offsets else 0
        return SourceSpan(
            self.source.filename,
            self.source.first_line + line_index,
            getattr(node, "col_offset", 0) + start_offset,
            self.source.first_line + getattr(node, "end_lineno", 1) - 1,
            getattr(node, "end_col_offset", 0) + end_offset,
            self.call_chain,
        )

    def error(self, node: ast.AST, message: str) -> FrontendError:
        """Report a source-located compile failure.

        Args:
            node: Python AST node to resolve.
            message: Diagnostic description.
        """
        return FrontendError(message, self.span(node))

    def lower(
        self,
        constants: dict[str, object],
        signature: dict[str, LogicalType] | None = None,
    ) -> ProgramIR:
        """Bind input types/static values and produce semantic IR.

        Args:
            constants: Explicit constexpr specialization values.
            signature: Optional explicit parameter types.
        """
        signature = {} if signature is None else signature
        parameters = {arg.arg for arg in self.function.args.args}
        if set(constants) - parameters or set(signature) - parameters:
            raise self.error(self.function, "Specialization contains unknown parameters")
        inputs = []
        specializations = []
        defaults = (
            dict(
                zip(
                    [arg.arg for arg in self.function.args.args][-len(self.function.args.defaults) :],
                    self.function.args.defaults,
                )
            )
            if self.function.args.defaults
            else {}
        )
        for arg in self.function.args.args:
            declared = self._parameter_type(arg, signature)
            value = self._bind_parameter(arg, declared, constants, defaults)
            self.env[arg.arg] = value
            if isinstance(value, Value):
                inputs.append(value)
            else:
                specializations.append((arg.arg, value))
        self.block(self.function.body)
        if self.outputs is None:
            raise self.error(self.function, "Program must return tensor values")
        self._check_return_annotation()
        return ProgramIR(
            self.function.name,
            tuple(inputs),
            tuple(self.builder.operations),
            self.outputs,
            tuple(specializations),
            self.returns_tuple,
        )

    def _parameter_type(self, arg, signature):
        declared = signature.get(arg.arg)
        if arg.annotation is not None:
            annotated = self.resolve_annotation(arg.annotation)
            if declared is not None and declared != annotated:
                raise self.error(arg, "Explicit signature disagrees with the declared annotation")
            declared = annotated
        return declared

    def _bind_parameter(self, arg, declared, constants, defaults):
        if isinstance(declared, LOGICAL_TYPES):
            if arg.arg in constants:
                raise self.error(arg, "Runtime tensor inputs cannot be frozen as constants")
            value = self.builder.value(arg.arg, declared)
            return value
        if isinstance(declared, ConstexprType):
            if arg.arg in constants:
                value = constants[arg.arg]
            elif arg.arg in defaults:
                value = self.default_value(defaults[arg.arg])
            else:
                raise self.error(arg, f"Missing constexpr specialization: {arg.arg}")
            self._freeze(arg, value)
            if not declared.accepts(value):
                raise self.error(arg, f"Invalid constexpr type for {arg.arg}")
            return value
        raise self.error(
            arg,
            f"Parameter {arg.arg} requires a Tensor or Constexpr annotation/signature",
        )

    def resolve_annotation(self, node: ast.AST) -> object:
        """Resolve an annotation without executing arbitrary Python.

        Args:
            node: Python AST node to resolve.
        """
        try:
            if any(isinstance(item, ast.Call) for item in ast.walk(node)):
                raise self.error(node, "Calls are unsupported in type annotations")
            return annotation(node, self.expression)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise self.error(node, str(exc))

    def _check_return_annotation(self):
        if self.function.returns is not None:
            declared = self.resolve_annotation(self.function.returns)
            expected = declared if isinstance(declared, tuple) else (declared,)
            if any(not isinstance(item, LOGICAL_TYPES) for item in expected):
                raise self.error(self.function.returns, "Return annotations require complete tensor types")
            if expected != tuple(value.type for value in self.outputs):
                raise self.error(
                    self.function.returns,
                    "Return annotation disagrees with inferred outputs",
                )

    def default_value(self, node: ast.AST) -> object:
        """Evaluate scalar defaults in definition scope without invoking calls.

        Args:
            node: Python AST node to resolve.
        """
        if any(isinstance(item, ast.Call) for item in ast.walk(node)):
            raise self.error(node, "Calls are unsupported in parameter defaults")
        saved_env, saved_names = self.env, self.local_names
        try:
            self.env, self.local_names = {}, set()
            return self._freeze(node, self.expression(node))
        finally:
            self.env, self.local_names = saved_env, saved_names

    def _freeze(self, node, value):
        try:
            return frozen(value)
        except ValueError as exc:
            raise self.error(node, str(exc))

    def block(self, statements: list[ast.stmt]) -> None:
        """Lower supported statements in source order.

        Args:
            statements: Ordered source statements.
        """
        for index, node in enumerate(statements):
            if self.outputs is not None:
                break
            if (
                isinstance(node, ast.Expr)
                and index == 0
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            self._statement(node)

    def _statement(self, node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            self.assign(node.targets[0], self.expression(node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = self.expression(node.value)
            declared = self.resolve_annotation(node.annotation)
            if not isinstance(value, Value) or value.type != declared:
                raise self.error(node, "Annotated assignment must match the inferred tensor type")
            self.assign(node.target, value)
        elif isinstance(node, ast.Return):
            self._return(node)
        elif isinstance(node, ast.If):
            condition = self._freeze(node.test, self.expression(node.test))
            if type(condition) not in (bool,):
                raise self.error(node.test, "Static if requires a boolean constexpr condition")
            self.block(node.body if condition else node.orelse)
        elif isinstance(node, ast.For):
            self._loop(node)
        else:
            raise self.error(node, f"Unsupported statement: {type(node).__name__}")

    def _return(self, node):
        value = self.expression(node.value)
        self.returns_tuple = isinstance(value, tuple)
        values = value if self.returns_tuple else (value,)
        if not values or any(not isinstance(item, Value) for item in values):
            raise self.error(node, "Return values must be tensors")
        self.outputs = values

    def assign(self, target: ast.AST, value: object) -> None:
        """Bind SSA aliases and destructure tuples without mutating tensors.

        Args:
            target: Assignment target or compilation target.
            value: Logical value, scalar or captured symbol to validate.
        """
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple) and len(target.elts) == len(value):
            for item, element in zip(target.elts, value):
                self.assign(item, element)
        else:
            raise self.error(target, "Only local names and matching tuple unpacking can be assigned")

    def expression(self, node: ast.AST) -> object:
        """Resolve values/symbols and evaluate only the whitelisted AST subset.

        Args:
            node: Python AST node to resolve.
        """
        try:
            return self._expression(node)
        except FrontendError:
            raise
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise self.error(node, str(exc))

    def _expression(self, node):
        if isinstance(node, ast.Constant):
            return frozen(node.value)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            if node.id in self.local_names:
                raise self.error(node, f"Local symbol used before assignment: {node.id}")
            if node.id in self.source.symbols:
                return self.source.symbols[node.id]
            builtins = {"int": int, "float": float, "bool": bool, "str": str}
            if node.id in builtins:
                return builtins[node.id]
            raise self.error(node, f"Unknown symbol: {node.id}")
        if isinstance(node, ast.Attribute):
            return attribute(self.expression(node.value), node.attr)
        if isinstance(node, ast.Tuple):
            return tuple(self.expression(item) for item in node.elts)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            return static_operation(node, self.expression)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise self.error(node, f"Unsupported expression: {type(node).__name__}")

    def _call(self, node):
        target = self.expression(node.func)
        args, kwargs = self._call_arguments(node)
        if isinstance(target, Helper):
            return self._inline(node, target, args, kwargs)
        if not isinstance(target, Primitive):
            raise self.error(node.func, "Call target is not a registered primitive or helper")
        schema = self.registry.resolve(target)
        bound = schema.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self._emit_primitive(node, schema, bound)

    def _call_arguments(self, node):
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(key.arg is None for key in node.keywords):
            raise self.error(node, "Expanded positional/keyword calls are unsupported")
        args = [self.expression(arg) for arg in node.args]
        kwargs = {key.arg: self.expression(key.value) for key in node.keywords}
        if len(kwargs) != len(node.keywords):
            raise self.error(node, "Duplicate keyword argument")
        return args, kwargs

    def _emit_primitive(self, node, schema, bound):
        for value in bound.arguments.values():
            if not isinstance(value, (Value, DType)):
                frozen(value)
        typed = {key: value.type if isinstance(value, Value) else value for key, value in bound.arguments.items()}
        output_types = schema.infer_types_and_shapes(typed)
        if not output_types or any(not isinstance(item, LOGICAL_TYPES) for item in output_types):
            raise self.error(node, "Primitive inference must return tensor types")
        outputs = tuple(self.builder.value(f"result_{self.builder.next_id}", item) for item in output_types)
        effects = []
        for kind, name in schema.infer_effects(typed):
            value = bound.arguments.get(name)
            if kind not in ("read", "write", "reduce", "atomic") or not isinstance(value, Value):
                raise self.error(node, "Primitive effects must name tensor arguments and valid logical accesses")
            effects.append(Effect(kind, value.id))
        effects = tuple(effects)
        effects += tuple(Effect("write", value.id) for value in outputs)
        self.builder.operations.append(
            Operation(
                schema.logical_name,
                schema.version,
                tuple(bound.arguments.items()),
                outputs,
                effects,
                self.span(node),
            )
        )
        return outputs[0] if len(outputs) == 1 else outputs

    def _inline(self, node, helper, args, kwargs):
        if id(helper) in self.active_helpers or len(self.active_helpers) >= 32:
            raise self.error(node, "Recursive helpers or helper nesting beyond 32 are unsupported")
        child = Parser(helper.source, self.registry, self.builder)
        child.call_chain = self.call_chain + (str(self.span(node)),)
        child.active_helpers = self.active_helpers + (id(helper),)
        child._bind_helper(node, self, args, kwargs)
        child.block(child.function.body)
        if child.outputs is None:
            raise child.error(child.function, "Helper must return tensor values")
        child._check_return_annotation()
        return child.outputs if child.returns_tuple else child.outputs[0]

    def _bind_helper(self, callsite, caller, args, kwargs):
        child = self
        parameters = child.function.args.args
        if len(args) > len(parameters):
            raise caller.error(callsite, "Too many helper arguments")
        child.env = dict(zip([arg.arg for arg in parameters], args))
        for key, value in kwargs.items():
            if key in child.env or key not in {arg.arg for arg in parameters}:
                raise caller.error(callsite, f"Unknown or duplicate helper argument: {key}")
            child.env[key] = value
        defaults = (
            dict(
                zip(
                    [arg.arg for arg in parameters][-len(child.function.args.defaults) :],
                    child.function.args.defaults,
                )
            )
            if child.function.args.defaults
            else {}
        )
        for arg in parameters:
            if arg.arg not in child.env:
                if arg.arg not in defaults:
                    raise caller.error(callsite, f"Missing helper argument: {arg.arg}")
                child.env[arg.arg] = child.default_value(defaults[arg.arg])
            child._check_helper_type(arg)

    def _check_helper_type(self, arg):
        if arg.annotation is None:
            raise self.error(arg, "Helper parameters require annotations")
        declared = self.resolve_annotation(arg.annotation)
        value = self.env[arg.arg]
        if isinstance(declared, LOGICAL_TYPES):
            if not isinstance(value, Value) or value.type != declared:
                raise self.error(arg, "Helper tensor argument type mismatch")
        elif not isinstance(declared, ConstexprType) or not declared.accepts(value):
            raise self.error(arg, "Helper constexpr argument type mismatch")

    def _loop(self, node):
        if node.orelse or not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call):
            raise self.error(node, "Loops require a local name and bounded static_range")
        if self.expression(node.iter.func) is not static_range or node.iter.keywords:
            raise self.error(node, "Loops require the registered static_range symbol")
        values = [self._freeze(arg, self.expression(arg)) for arg in node.iter.args]
        if not 1 <= len(values) <= 3 or any(type(value) not in (int,) for value in values):
            raise self.error(node, "static_range requires one to three integer constants")
        try:
            iterations = range(*values)
            count = len(iterations)
        except (ValueError, OverflowError) as exc:
            raise self.error(node, str(exc))
        if count + self.builder.unrolled_iterations > 1024:
            raise self.error(node, "Static loop expansion exceeds the compilation bound of 1024")
        self.builder.unrolled_iterations += len(iterations)
        for value in iterations:
            self.env[node.target.id] = value
            self.block(node.body)
            if self.outputs is not None:
                raise self.error(node, "Return inside static_range is unsupported")


@dataclass(frozen=True)
class Helper:
    """An explicitly registered function captured for AST inlining."""

    source: Source


def static_range(*bounds: int) -> range:
    """Compile-time loop marker; compiled programs never execute it."""
    return range(*bounds)
