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
"""Generate native host-stream forward/VJP adapters from trusted dense SSA providers."""

from __future__ import annotations

from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.ir.program import Value
from hyper_parallel.core.multicore.runtime.dense import DenseKernelPlan

_HELPERS = r'''
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <optional>
#include <vector>

namespace {
bool storage_exclusive(const at::Tensor& base) {
  return base.storage().use_count() == 1;
}

std::vector<at::Tensor> saved_views(const at::Tensor& base, const std::vector<int64_t>& offsets,
                                  const std::vector<std::vector<int64_t>>& shapes) {
  TORCH_CHECK(base.scalar_type() == at::kBFloat16 && base.dim() == 1 && offsets.size() == shapes.size(),
              "Dense saved views require a BF16 base and matching offsets/shapes");
  if (!storage_exclusive(base)) {
    return {};
  }
  std::vector<at::Tensor> values;
  values.reserve(shapes.size());
  for (size_t index = 0; index < shapes.size(); ++index) {
    int64_t elements = 1;
    for (int64_t dimension : shapes[index]) {
      TORCH_CHECK(dimension >= 0, "Dense saved view dimensions must be nonnegative");
      if (dimension == 0) {
        elements = 0;
      }
    }
    if (elements != 0) {
      for (int64_t dimension : shapes[index]) {
        TORCH_CHECK(elements <= base.numel() / dimension,
                    "Dense saved view dimensions exceed their base allocation");
        elements *= dimension;
      }
    }
    TORCH_CHECK(offsets[index] >= 0 && offsets[index] <= base.numel() &&
                elements <= base.numel() - offsets[index], "Dense saved view exceeds its base allocation");
    values.push_back(at::empty({0}, base.options()).set_(base.storage(), offsets[index], shapes[index]));
  }
  return values;
}

at::Tensor dense_matmul(const at::Tensor& left, const at::Tensor& right) {
  if (left.size(0) == 0 || right.size(1) == 0) {
    return at::empty({left.size(0), right.size(1)}, left.options());
  }
  if (left.size(1) == 0) {
    return at::zeros({left.size(0), right.size(1)}, left.options());
  }
  return at::mm(left, right);
}

at::Tensor dense_linear(const at::Tensor& left, const at::Tensor& right, bool transpose_right) {
  TORCH_CHECK(transpose_right || left.device().type() != c10::DeviceType::PrivateUse1,
              "Native Linear requires borrowed NK weights without a layout copy");
  const auto weight = transpose_right ? right : right.t().contiguous();
  return at::linear(left, weight);
}

at::Tensor dense_swiglu(const at::Tensor& packed) {
  if (packed.numel() == 0 || packed.device().type() == c10::DeviceType::Meta) {
    return at::empty({packed.size(0), packed.size(1) / 2}, packed.options());
  }
  if (packed.device().type() == c10::DeviceType::PrivateUse1) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_swiglu", "")
        .typed<at::Tensor(const at::Tensor&, int64_t)>();
    return op.call(packed, -1);
  }
  auto halves = packed.to(at::kFloat).chunk(2, -1);
  return (at::silu(halves[0]) * halves[1]).to(packed.scalar_type());
}

at::Tensor dense_swiglu_grad(const at::Tensor& grad, const at::Tensor& packed) {
  if (packed.numel() == 0 || packed.device().type() == c10::DeviceType::Meta) {
    return at::empty_like(packed);
  }
  if (packed.device().type() == c10::DeviceType::PrivateUse1) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_swiglu_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>();
    return op.call(grad.contiguous(), packed, -1);
  }
  auto halves = packed.to(at::kFloat).chunk(2, -1);
  auto gate = halves[0];
  auto sigmoid = gate.sigmoid();
  auto delta = grad.to(at::kFloat);
  auto gate_grad = delta * halves[1] * sigmoid * (1 + gate * (1 - sigmoid));
  auto up_grad = delta * at::silu(gate);
  return at::cat({gate_grad, up_grad}, -1).to(packed.scalar_type());
}

void accumulate(std::vector<at::Tensor>& gradients, int64_t index, const at::Tensor& value) {
  gradients[index] = gradients[index].defined() ? gradients[index] + value : value;
}
'''


def saved_value_ids(plan: DenseKernelPlan) -> tuple[int, ...]:
    """Select internal values required by explicit native backward recipes.

    Args:
        plan: Typed dense primitive plan.
    """
    return tuple(buffer.value_id for buffer in plan.buffers if buffer.saved_for_backward)


def _operand(value: Value, transpose: bool) -> str:
    return f"v{value.id}" + (".t()" if transpose else "")


def _forward_expression(provider, arguments):
    if provider.implementation == "aten.mm.default":
        left = _operand(arguments["left"], arguments["transpose_left"])
        right = _operand(arguments["right"], arguments["transpose_right"])
        return f"dense_matmul({left}, {right})"
    if provider.implementation == "aten.linear.default":
        left = _operand(arguments["left"], arguments["transpose_left"])
        return f"dense_linear({left}, v{arguments['right'].id}, {str(arguments['transpose_right']).lower()})"
    if provider.implementation == "npu.npu_swiglu.default":
        return f"dense_swiglu(v{arguments['packed'].id})"
    if provider.implementation == "aten.silu.default":
        return f"at::silu(v{arguments['value'].id})"
    if provider.implementation == "aten.mul.Tensor":
        return f"(v{arguments['left'].id} * v{arguments['right'].id})"
    raise ValueError(f"No native dense source template for {provider.implementation}")


def _backward_statement(node, contribution):
    primal, = node.primal_inputs
    gradient = f"gradients[{contribution.source_target}]"
    if node.kind == "swiglu_input_grad":
        expression = f"dense_swiglu_grad({gradient}, v{primal})"
    elif node.kind == "silu_input_grad":
        expression = f"at::silu_backward({gradient}, v{primal})"
    elif node.kind in ("multiply_left_grad", "multiply_right_grad"):
        expression = f"({gradient} * v{primal})"
    elif node.kind == "linear_weight_grad":
        operand = f"v{primal}" + (".t()" if node.operand_transposes[1][0] else "")
        expression = f"(dense_matmul(({operand}).t(), {gradient})).t()"
        if node.transpose_output:
            expression = f"({expression}).t()"
    else:
        side = 1 if node.kind == "matmul_left_grad" else 0
        operand = f"v{primal}" + (".t()" if node.operand_transposes[side][0] else "")
        arguments = f"{gradient}, ({operand}).t()" if side else f"({operand}).t(), {gradient}"
        expression = f"dense_matmul({arguments})"
        if node.transpose_output:
            expression = f"({expression}).t()"
    return f"    accumulate(gradients, {contribution.target}, {expression});"


def generate_dense_cpp(plan: DenseKernelPlan, namespace: str) -> str:
    """Emit provider calls and reverse SSA accumulation using fixed trusted templates.

    Args:
        plan: Canonical dense semantic program and native provider assignments.
        namespace: Unique identifier derived from sealed semantic/source identity.
    """
    if not namespace.isidentifier() or not namespace.startswith("hp_dense_"):
        raise ValueError("Dense native namespace must be a sealed hp_dense identifier")
    operations = plan.ir.operations
    adjoint = compile_dense_backward(plan)
    saved = saved_value_ids(plan)
    forward = ["std::vector<at::Tensor> forward(const std::vector<at::Tensor>& inputs) {",
               f'  TORCH_CHECK(inputs.size() == {len(plan.ir.inputs)}, "Dense native input count mismatch");']
    declarations = [f"  const auto& v{value.id} = inputs[{index}];"
                    for index, value in enumerate(plan.ir.inputs)]
    forward.extend(declarations)
    backward = ["std::vector<std::optional<at::Tensor>> backward(const std::vector<at::Tensor>& inputs,",
                "    const std::vector<at::Tensor>& saved,",
                "    const std::vector<std::optional<at::Tensor>>& output_grads) {",
                f'  TORCH_CHECK(inputs.size() == {len(plan.ir.inputs)} && saved.size() == {len(saved)} &&',
                f'      output_grads.size() == {len(plan.ir.outputs)}, "Dense native backward count mismatch");',
                *declarations, *[f"  const auto& v{value_id} = saved[{index}];"
                                 for index, value_id in enumerate(saved)],
                f"  std::vector<at::Tensor> gradients({max(dict(plan.value_types), default=-1) + 1});"]
    for index, output in enumerate(plan.ir.outputs):
        backward.extend([f"  if (output_grads[{index}].has_value()) {{",
                         f"    accumulate(gradients, {output.id}, *output_grads[{index}]);", "  }"])
    for task, operation in zip(plan.tasks, operations):
        out, = operation.outputs
        forward.append(f"  auto v{out.id} = {_forward_expression(task.provider, dict(operation.arguments))};")
    returns = [f"v{value.id}" for value in plan.ir.outputs] + [f"v{value_id}" for value_id in saved]
    forward.extend(["  return {" + ", ".join(returns) + "};", "}"])
    for index in reversed(range(len(operations))):
        operation = operations[index]
        out, = operation.outputs
        backward.append(f"  if (gradients[{out.id}].defined()) {{")
        backward.extend(_backward_statement(adjoint.nodes[item.node], item) for item in adjoint.contributions
                        if item.operation == index)
        backward.append("  }")
    input_grads = [f"gradients[{value.id}].defined() ? std::make_optional(gradients[{value.id}]) : std::nullopt"
                   for value in plan.ir.inputs]
    backward.extend(["  return {" + ", ".join(input_grads) + "};", "}"])
    registration = f'''
}}  // namespace
TORCH_LIBRARY({namespace}, m) {{
  m.def("forward(Tensor[] inputs) -> Tensor[]");
  m.def("backward(Tensor[] inputs, Tensor[] saved, Tensor?[] output_grads) -> Tensor?[]");
  m.def("storage_exclusive(Tensor base) -> bool");
  m.def("saved_views(Tensor base, int[] offsets, int[][] shapes) -> Tensor[]");
}}
TORCH_LIBRARY_IMPL({namespace}, CompositeExplicitAutograd, m) {{
  m.impl("forward", &forward);
  m.impl("backward", &backward);
  m.impl("storage_exclusive", &storage_exclusive);
  m.impl("saved_views", &saved_views);
}}
'''
    header = "// Copyright 2026 Huawei Technologies Co., Ltd.\n// SPDX-License-Identifier: Apache-2.0\n"
    return header + _HELPERS + "\n".join(forward + backward) + registration
