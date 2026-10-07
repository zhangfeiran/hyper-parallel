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
"""Source capture, semantic lowering and a validated CPU reference interpreter."""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable, Mapping

import torch

from hyper_parallel.core.multicore.compiler.moe import match_moe_region
from hyper_parallel.core.multicore.compiler.pipeline import compile_worker_pipeline
from hyper_parallel.core.multicore.frontend.parser import (
    Helper,
    Parser,
    Source,
    static_range,
)
from hyper_parallel.core.multicore.frontend.symbols import frozen
from hyper_parallel.core.multicore.ir.program import ProgramIR, Value
from hyper_parallel.core.multicore.ir.schedule import (
    HardwareSpec,
    TaskDAG,
    WorkerPipeline,
)
from hyper_parallel.core.multicore.language.types import (
    LOGICAL_TYPES,
    LogicalType,
    RaggedTensorType,
    RouteMetadataType,
    TensorListType,
    TensorType,
)
from hyper_parallel.core.multicore.language.values import RaggedTensor, RouteMetadata
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.primitives.gate import TORCH_DTYPES
from hyper_parallel.core.multicore.primitives.registry import (
    REGISTRY,
    PrimitiveRegistry,
)
from hyper_parallel.core.multicore.runtime.moe import MoeKernelPlan, compile_moe_plan
from hyper_parallel.core.multicore.runtime.plan import KernelPlan


def _capture(function: Callable) -> Source:
    try:
        lines, first_line = inspect.getsourcelines(function)
    except (OSError, TypeError) as exc:
        raise ValueError("Cannot capture function source; use frontend.from_source instead") from exc
    symbols = dict(function.__globals__)
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
        try:
            symbols[name] = cell.cell_contents
        except ValueError:
            # Decorated nested helpers can capture their own not-yet-bound name.
            # Unbound captures otherwise receive a source-located resolution error.
            continue
    return _source(
        "".join(lines),
        inspect.getsourcefile(function) or "<unknown>",
        first_line,
        symbols,
    )


def _source(text, filename, first_line, symbols):
    dedented = textwrap.dedent(text)
    offsets = tuple(
        len(original) - len(normalized) for original, normalized in zip(text.splitlines(), dedented.splitlines())
    )
    return Source(dedented, filename, first_line, symbols, offsets)


class Program:
    """A captured DSL function, lowered without executing its Python body.

    The Gate WorkerPipeline backend produces isolated legacy runtime images.
    Native tiling, materialization and execution require a matching family payload.
    """

    def __init__(
        self,
        source: Source,
        *,
        target: str = "ascend",
        schedule: WorkerPipeline | TaskDAG | None = None,
        signature: Mapping[str, LogicalType] | None = None,
        constants: Mapping[str, object] | None = None,
        registry: PrimitiveRegistry = REGISTRY,
    ) -> None:
        """Store compilation options and validate frozen specialization defaults."""
        if target != "ascend":
            raise ValueError("The multicore frontend currently targets ascend")
        if schedule is not None and not isinstance(schedule, (WorkerPipeline, TaskDAG)):
            raise ValueError("schedule must be WorkerPipeline or TaskDAG metadata")
        self.source = source
        self.target = target
        self.schedule = schedule
        self.signature = dict(signature or {})
        self.constants = {key: frozen(value) for key, value in (constants or {}).items()}
        self.registry = registry

    def lower(self, *, signature: Mapping[str, LogicalType] | None = None, **constants: object) -> ProgramIR:
        """Specialize constexpr parameters and lower to immutable ProgramIR.

        Args:
            signature: Optional explicit input types for annotation-free source.
            **constants: Static values for declared Constexpr parameters.

        Returns:
            Semantic IR in preserved source/numerical order.
        """
        static = {**self.constants, **constants}
        declared = self.signature if signature is None else dict(signature)
        return Parser(self.source, self.registry).lower(static, declared)

    def plan(
        self,
        signature: Mapping[str, int] | MegaMoeSpec | None = None,
        topology: HardwareSpec | None = None,
        **constants: object,
    ) -> KernelPlan | MoeKernelPlan:
        """Compile a supported Gate Route or MoE region into a host compatibility plan.

        Args:
            signature: Symbolic Gate dimensions or a bound native MegaMoeSpec for TaskDAG.
            topology: Physical worker availability, independent of computation.
            **constants: Declared constexpr specializations, such as k, scale or limit.

        Returns:
            A source-mapped host plan with normal/profiled native descriptor images.
        """
        if self.schedule is None:
            raise ValueError("An explicit WorkerPipeline or TaskDAG schedule is required")
        ir = self.lower(**constants)
        for operation in ir.operations:
            canonical = REGISTRY.schema(operation.logical_name, operation.version)
            if self.registry.schema(operation.logical_name, operation.version) is not canonical:
                raise ValueError("Compatibility plans require canonical registered primitive schemas")
        if isinstance(self.schedule, TaskDAG):
            if self.schedule.policy != "moe_ratr_v1":
                raise ValueError("Only the moe_ratr_v1 TaskDAG policy is implemented")
            if not isinstance(signature, MegaMoeSpec):
                raise TypeError("MoE TaskDAG plans require a bound MegaMoeSpec")
            if topology is not None and topology.available_aiv_workers != 2 * signature.num_cube_cores:
                raise ValueError("MoE topology must match the native spec's Cube/AIV worker ratio")
            return compile_moe_plan(match_moe_region(ir), signature)
        return compile_worker_pipeline(ir, self.schedule, dict(signature or {}), topology or HardwareSpec())

    def explain(self, **constants: object) -> str:
        """Describe schedule intent and dump the source-mapped semantic IR."""
        return (
            f"target={self.target}, schedule={self.schedule!r}, backend=semantic-only\n"
            + self.lower(**constants).dump()
        )

    def interpret(self, *args: object, **kwargs: object) -> object:
        """Execute registered references on validated CPU tensors.

        Symbolic dimensions are unified across all inputs and outputs. This debug
        path preserves reference autograd through reference operations but does not
        implement a native backward recipe or device execution.
        """
        parser = Parser(self.source, self.registry)
        bound, constants = self._bind_interpreter(parser, args, kwargs)
        ir = self.lower(**constants)
        return self._execute_ir(ir, bound)

    def _bind_interpreter(self, parser, args, kwargs):
        defaults = parser.function.args.defaults
        first_default = len(parser.function.args.args) - len(defaults)
        parameters = []
        for index, arg in enumerate(parser.function.args.args):
            default = inspect.Parameter.empty
            if index >= first_default:
                default = parser.default_value(defaults[index - first_default])
            parameters.append(inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))
        bound = inspect.Signature(parameters).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for parameter in parameters:
            if parameter.name not in bound.arguments:
                if parameter.name not in self.constants:
                    raise TypeError(f"Missing argument: {parameter.name}")
                bound.arguments[parameter.name] = self.constants[parameter.name]
        declared = {}
        for arg in parser.function.args.args:
            declared[arg.arg] = self.signature.get(arg.arg)
            if arg.annotation is not None:
                declared[arg.arg] = parser.resolve_annotation(arg.annotation)
        constants = {key: value for key, value in bound.arguments.items()
                     if not isinstance(declared[key], LOGICAL_TYPES)}
        return bound, constants

    def _execute_ir(self, ir, bound):
        values = {}
        dimensions = {}
        for value in ir.inputs:
            tensor = bound.arguments[value.name]
            _validate_value(tensor, value.type, dimensions, value.name)
            values[value.id] = tensor
        for operation in ir.operations:
            schema = self.registry.schema(operation.logical_name, operation.version)
            if schema.reference is None:
                raise ValueError(f"No reference implementation for {operation.logical_name}.v{operation.version}")
            arguments = {key: values[item.id] if isinstance(item, Value) else item for key, item in operation.arguments}
            rebound = schema.signature.bind(**arguments)
            result = schema.reference(*rebound.args, **rebound.kwargs)
            results = (result,) if len(operation.outputs) == 1 else result
            if not isinstance(results, tuple) or len(results) != len(operation.outputs):
                raise ValueError(f"Reference output arity mismatch for {operation.logical_name}")
            for value, tensor in zip(operation.outputs, results):
                _validate_value(tensor, value.type, dimensions, value.name)
                values[value.id] = tensor
        outputs = tuple(values[value.id] for value in ir.outputs)
        return outputs if ir.returns_tuple else outputs[0]


def _validate_value(value, logical_type, dimensions, name):
    if isinstance(logical_type, RaggedTensorType):
        if not isinstance(value, RaggedTensor):
            raise TypeError(f"Reference {name} requires a RaggedTensor storage/valid_rows pair")
        _validate_tensor(value.storage, logical_type, dimensions, name)
    elif isinstance(logical_type, RouteMetadataType):
        if not isinstance(value, RouteMetadata):
            raise TypeError(f"Reference {name} requires RouteMetadata")
        _validate_tensor(value.group_list, TensorType(logical_type.dtype, logical_type.shape), dimensions, name)
    elif isinstance(logical_type, TensorListType):
        matrices = tuple(value.unbind(0)) if isinstance(value, torch.Tensor) and value.dim() == 3 else value
        if not isinstance(matrices, (list, tuple)) or not matrices:
            raise ValueError(f"Reference {name} requires a nonempty expert matrix list or stacked 3D tensor")
        shape = logical_type.shape or tuple(matrices[0].shape)
        if len(shape) != 2:
            raise ValueError("TensorList elements must be matrices")
        for matrix in matrices:
            _validate_tensor(matrix, TensorType(logical_type.dtype, shape), dimensions, name)
        if dimensions.setdefault("LocalExperts", len(matrices)) != len(matrices):
            raise ValueError("Expert matrix lists must match the route group-list size")
    else:
        _validate_tensor(value, logical_type, dimensions, name)


def _validate_tensor(tensor, tensor_type, dimensions, name):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise ValueError(f"Reference input/output {name} must be a CPU Torch tensor")
    if tensor.dtype != TORCH_DTYPES[tensor_type.dtype] or len(tensor.shape) != len(tensor_type.shape):
        raise ValueError(f"Tensor {name} dtype/rank does not match its logical type")
    if not tensor.is_contiguous():
        raise ValueError(f"Tensor {name} must be contiguous for the declared V0 layout")
    for expected, actual in zip(tensor_type.shape, tensor.shape):
        if expected is None:
            continue
        if isinstance(expected, str):
            if dimensions.setdefault(expected, actual) != actual:
                raise ValueError(f"Symbolic dimension {expected} is inconsistent at {name}")
        elif expected != actual:
            raise ValueError(f"Tensor {name} dimension mismatch: expected {expected}, got {actual}")


def program(
    function: Callable | None = None,
    *,
    target: str = "ascend",
    schedule: WorkerPipeline | TaskDAG | None = None,
) -> Program | Callable:
    """Capture a typed DSL function, supporting bare/configured decorators.

    Args:
        function: Typed Python function to capture.
    """

    def _decorate(candidate: Callable) -> Program:
        return Program(_capture(candidate), target=target, schedule=schedule)

    return _decorate if function is None else _decorate(function)


def helper(function: Callable) -> Helper:
    """Register a typed helper for source-mapped AST inlining.

    Args:
        function: Typed Python function to capture.
    """
    captured = _capture(function)
    registered = Helper(captured)
    captured.symbols[function.__name__] = registered
    return registered


def from_source(
    source: str,
    signature: Mapping[str, LogicalType] | None = None,
    constants: Mapping[str, object] | None = None,
    *,
    symbols: Mapping[str, object] | None = None,
    filename: str = "<source>",
    first_line: int = 1,
    registry: PrimitiveRegistry = REGISTRY,
    schedule: WorkerPipeline | TaskDAG | None = None,
) -> Program:
    """Capture explicit source for notebooks or generated functions without eval.

    Args:
        source: Exactly one synchronous Python function definition.
        signature: Explicit TensorType inputs when annotations are absent.
        constants: Default constexpr specializations, copied at capture time.
        symbols: Explicit primitive/helper/DSL symbols and frozen constants.
        filename: Source identity for diagnostics.
        first_line: Original 1-based starting line.
        registry: Trusted primitive registry used for lowering and interpretation.
        schedule: Optional execution mode for compatibility plan generation.

    Returns:
        A program ready for semantic lowering or CPU interpretation.
    """
    if type(first_line) not in (int,) or first_line < 1:
        raise ValueError("first_line must be a positive integer")
    environment = dict(symbols or {})
    environment.setdefault("static_range", static_range)
    captured = _source(source, filename, first_line, environment)
    return Program(captured, signature=signature, constants=constants, registry=registry, schedule=schedule)
