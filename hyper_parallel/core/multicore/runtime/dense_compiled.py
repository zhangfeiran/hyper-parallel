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
"""Native dense adapter loading and per-invocation explicit reverse-mode execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore._build.build_dense import dense_build_identity
from hyper_parallel.core.multicore.backends.dense import dense_artifacts
from hyper_parallel.core.multicore.backends.dense_codegen import saved_value_ids
from hyper_parallel.core.multicore.runtime.dense import DenseExecutable, DenseKernelPlan
from hyper_parallel.core.multicore.runtime.dense_native import verify_dense_dispatch

_LOADED: dict[str, str] = {}


class CompiledDenseExecutable(DenseExecutable):
    """One native call per forward/backward, using generated primitive provider calls.

    Provider kernels still execute on the current host stream. This adapter does
    not claim a resident AIC/AIV worker or tile pipeline.
    """

    def __init__(self, plan: DenseKernelPlan, device: torch.device | str, manifest: Path) -> None:
        """Verify exact native source and ABI before loading any operator definitions.

        Args:
            plan: Canonical dense task graph.
            device: CPU references or NPU native providers.
            manifest: Sealed payload produced by build_dense_payload.
        """
        super().__init__(plan, torch.device(device))
        manifest = Path(manifest).resolve()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        definition_key, _, files = dense_artifacts(plan)
        namespace = "hp_dense_" + definition_key[:24]
        source_hash = hashlib.sha256(files["native/dense.cpp"]).hexdigest()
        if (data["format_version"] != 1 or data["namespace"] != namespace
                or data["execution_mode"] != "native_host_stream_adapter"
                or data["identity"] != {"definition_key": definition_key, "source_sha256": source_hash,
                                       "build": dense_build_identity()}):
            raise ValueError("Compiled dense definition/source/framework identity mismatch")
        library = (manifest.parent / data["library"]).resolve()
        if (not library.is_relative_to(manifest.parent)
                or hashlib.sha256(library.read_bytes()).hexdigest() != data["library_sha256"]
                or hashlib.sha256((manifest.parent / "dense.cpp").read_bytes()).hexdigest() != source_hash):
            raise ValueError("Compiled dense native source/library integrity mismatch")
        if self.device.type == "npu":
            verify_dense_dispatch()
        fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if namespace in _LOADED:
            if _LOADED[namespace] != fingerprint:
                raise ValueError("A different native dense payload already owns this operator namespace")
        else:
            torch.ops.load_library(str(library))
            _LOADED[namespace] = fingerprint
        self.native_ops = getattr(torch.ops, namespace)
        self.native_identity = data
        self.saved_count = len(saved_value_ids(plan))

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Invoke generated native provider calls with private autograd state.

        Args:
            *inputs: Caller-owned dense inputs and weights in program parameter order.
        """
        self._bind_inputs(inputs)
        return _CompiledDenseFunction.apply(self, *inputs)


class _CompiledDenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, executable: CompiledDenseExecutable,
                *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Retain exactly the declared per-invocation native backward cache.

        Args:
            ctx: Private autograd invocation state.
            executable: Sealed native forward/backward operator binding.
            *inputs: Caller-owned program inputs and parameters.
        """
        ctx.executable = executable
        ctx.set_materialize_grads(False)
        results = executable.native_ops.forward.default(list(inputs))
        count = len(executable.plan.ir.outputs)
        if len(results) != count + executable.saved_count:
            raise RuntimeError("Compiled dense forward returned an invalid saved-state contract")
        ctx.input_count = len(inputs)
        ctx.save_for_backward(*inputs, *results[count:])
        outputs = tuple(results[:count])
        return outputs if executable.plan.ir.returns_tuple else outputs[0]

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, *output_grads: torch.Tensor | None) -> tuple[object, ...]:
        """Differentiate the saved invocation through its generated native VJP.

        Args:
            ctx: Exact forward state retained by this graph.
            *output_grads: Cotangents of the public semantic outputs.
        """
        tensors = ctx.saved_tensors
        inputs, saved = tensors[:ctx.input_count], tensors[ctx.input_count:]
        gradients = ctx.executable.native_ops.backward.default(list(inputs), list(saved), list(output_grads))
        if len(gradients) != ctx.input_count:
            raise RuntimeError("Compiled dense backward returned an invalid input-gradient contract")
        return (None, *(gradient if needed else None for gradient, needed in
                        zip(gradients, ctx.needs_input_grad[1:])))
