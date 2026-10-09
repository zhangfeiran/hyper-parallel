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
"""Lower dense SSA operations directly through trusted primitive providers."""

from __future__ import annotations

from hyper_parallel.core.multicore.backends.providers import DENSE_PROVIDERS, ProviderRegistry
from hyper_parallel.core.multicore.ir.program import ProgramIR, Value
from hyper_parallel.core.multicore.language.types import TensorType
from hyper_parallel.core.multicore.primitives.gate import TORCH_DTYPES
from hyper_parallel.core.multicore.primitives.registry import REGISTRY
from hyper_parallel.core.multicore.runtime.dense import DenseBuffer, DenseKernelPlan, DenseSpec, DenseTask


def _arguments(operation, types, producers):
    arguments, reads, dependencies = {}, [], set()
    for name, value in operation.arguments:
        if not isinstance(value, Value):
            arguments[name] = value
            continue
        if value.id not in types:
            raise ValueError("Dense IR reads a value before its definition")
        arguments[name] = types[value.id]
        reads.append(value.id)
        if value.id in producers:
            dependencies.add(producers[value.id])
    return arguments, reads, dependencies


def _outputs(operation, arguments, provider, spec):
    schema = REGISTRY.schema(operation.logical_name, operation.version)
    inferred = schema.infer_types_and_shapes(arguments)
    if len(inferred) != len(operation.outputs):
        raise ValueError("Dense native provider output arity does not match the IR")
    for output, inferred_type in zip(operation.outputs, inferred):
        if inferred_type != spec.bind(output.type) or inferred_type.dtype not in provider.dtypes:
            raise ValueError(f"Dense bound shape/dtype mismatch at {operation.logical_name}")
    return inferred


def _buffers(types, producers, last_uses, saved):
    buffers = []
    for value_id, first_task in producers.items():
        tensor_type = types[value_id]
        size = TORCH_DTYPES[tensor_type.dtype].itemsize
        for dimension in tensor_type.shape:
            size *= dimension
        buffers.append(DenseBuffer(value_id, first_task, last_uses[value_id], size, value_id in saved))
    return tuple(buffers)


def _input_types(ir, spec):
    types = {}
    for value in ir.inputs:
        if not isinstance(value.type, TensorType):
            raise ValueError("Dense lowering requires ordinary Tensor inputs")
        types[value.id] = spec.bind(value.type)
    return types


def _return_lifetimes(outputs, types, last_uses, task_count):
    for output in outputs:
        if output.id not in types:
            raise ValueError("Dense program returns an undefined value")
        last_uses[output.id] = task_count


def compile_dense_plan(ir: ProgramIR, spec: DenseSpec,
                       providers: ProviderRegistry = DENSE_PROVIDERS) -> DenseKernelPlan:
    """Derive task dependencies, bindings and live intervals from semantic SSA.

    Args:
        ir: Canonical registered dense primitive computation.
        spec: Static symbolic dimensions independent of EP or route metadata.
        providers: Trusted implementation contracts.
    """
    if ir.numeric_policy != "preserve_numeric_order":
        raise ValueError("Dense lowering requires explicit preservation of numerical order")
    types, producers, last_uses, saved = _input_types(ir, spec), {}, {}, set()
    tasks = []
    for index, operation in enumerate(ir.operations):
        schema = REGISTRY.schema(operation.logical_name, operation.version)
        provider = providers.resolve(schema)
        arguments, reads, dependencies = _arguments(operation, types, producers)
        last_uses.update((value_id, index) for value_id in reads)
        saved.update(value.id for name, value in operation.arguments
                     if isinstance(value, Value) and name in provider.saved_inputs)
        expected_effects = {("read", value_id) for value_id in reads}
        expected_effects.update(("write", output.id) for output in operation.outputs)
        if {(effect.kind, effect.value_id) for effect in operation.effects} != expected_effects:
            raise ValueError("Dense lowering requires explicit out-of-place read-only primitive effects")
        for output, inferred_type in zip(operation.outputs, _outputs(operation, arguments, provider, spec)):
            if output.id in types:
                raise ValueError("Dense SSA values cannot be overwritten")
            types[output.id], producers[output.id], last_uses[output.id] = inferred_type, index, index
        tasks.append(DenseTask(index, provider, tuple(sorted(dependencies)), tuple(reads),
                               tuple(output.id for output in operation.outputs)))
    _return_lifetimes(ir.outputs, types, last_uses, len(tasks))
    return DenseKernelPlan(ir, spec, tuple(tasks), tuple(sorted(types.items())),
                           _buffers(types, producers, last_uses, saved))
