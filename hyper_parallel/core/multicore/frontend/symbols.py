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
"""Safe static scalar evaluation and restricted DSL symbol access."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from types import ModuleType

from hyper_parallel.core.multicore.language.types import Constexpr, DType, Tensor

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def frozen(value: object) -> object:
    """Validate bounded immutable JSON scalars/tuples without invoking user hooks.

    Args:
        value: Logical value, scalar or captured symbol to validate.
    """
    if type(value) not in (int, float, bool, str, tuple, type(None)):
        raise ValueError("Expected a frozen scalar or tuple, not runtime data")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Static floats must be finite")
    if isinstance(value, int) and value.bit_length() > 64:
        raise ValueError("Static integers must fit within 64 bits")
    if isinstance(value, (str, tuple)) and len(value) > 1024:
        raise ValueError("Static strings/tuples exceed the V0 bound of 1024")
    if isinstance(value, tuple):
        for item in value:
            frozen(item)
    return value


def attribute(base: object, name: str) -> object:
    """Read only public names from the DSL module; never invoke arbitrary properties.

    Args:
        base: Resolved DSL module symbol.
        name: Symbol or logical value name.
    """
    if type(base) not in (ModuleType,) or base.__name__ != "hyper_parallel.core.multicore.language":
        raise ValueError("Attribute reads require the registered DSL language module")
    if name.startswith("_") or name not in vars(base):
        raise ValueError(f"Unknown DSL attribute: {name}")
    return vars(base)[name]


def static_operation(node: ast.AST, evaluate: Callable[[ast.AST], object]) -> object:
    """Evaluate whitelisted operators over validated scalar operands.

    Args:
        node: Python AST node to resolve.
        evaluate: Restricted recursive expression resolver.
    """
    if isinstance(node, ast.UnaryOp):
        return _static_unary(node, evaluate)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = frozen(evaluate(node.left)), frozen(evaluate(node.right))
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("Static arithmetic requires scalar numbers")
        return frozen(_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.Compare):
        return _static_comparison(node, evaluate)
    if isinstance(node, ast.BoolOp):
        for operand in node.values:
            result = frozen(evaluate(operand))
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result
    raise ValueError(f"Unsupported static operator: {type(node).__name__}")


def annotation(node: ast.AST, evaluate: Callable[[ast.AST], object]) -> object:
    """Resolve Tensor/Constexpr annotations with no eval or user subscripting.

    Args:
        node: Python AST node to resolve.
        evaluate: Restricted recursive expression resolver.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return annotation(ast.parse(node.value, mode="eval").body, evaluate)
    if any(isinstance(item, ast.Call) for item in ast.walk(node)):
        raise ValueError("Calls are unsupported in type annotations")
    if isinstance(node, ast.Tuple):
        return tuple(annotation(item, evaluate) for item in node.elts)
    if isinstance(node, ast.Subscript):
        factory = evaluate(node.value)
        if factory is Tensor:
            arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else ()
            if len(arguments) != 2:
                raise ValueError("Tensor annotation requires dtype and shape")
            dtype, shape = (evaluate(item) for item in arguments)
            return Tensor[dtype, shape]
        if factory is Constexpr:
            return Constexpr[_scalar_annotation(node.slice, evaluate)]
    result = evaluate(node)
    if isinstance(result, DType):
        raise TypeError("Use Tensor[dtype, shape] for tensor arguments")
    return result


def _scalar_annotation(node, evaluate):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _scalar_annotation(node.left, evaluate)
        right = _scalar_annotation(node.right, evaluate)
        return left | right
    if isinstance(node, ast.Constant) and node.value is None:
        return type(None)
    result = evaluate(node)
    if not any(result is item for item in (int, float, bool, str, type(None))):
        raise TypeError("Constexpr annotations require scalar types")
    return result


def _static_comparison(node, evaluate):
    left = frozen(evaluate(node.left))
    for operation, comparator in zip(node.ops, node.comparators):
        right = frozen(evaluate(comparator))
        compare = _COMPARE.get(type(operation))
        if compare is None:
            raise ValueError("Unsupported static comparison")
        if not compare(left, right):
            return False
        left = right
    return True


def _static_unary(node, evaluate):
    value = frozen(evaluate(node.operand))
    if isinstance(node.op, ast.Not):
        return not value
    if type(value) not in (int, float):
        raise ValueError("Unary arithmetic requires static numbers")
    if isinstance(node.op, ast.USub):
        return frozen(-value)
    if isinstance(node.op, ast.UAdd):
        return value
    raise ValueError("Unsupported unary operator")
