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
"""Trusted primitive providers independent of family-specific task numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from hyper_parallel.core.multicore.language.types import DType
from hyper_parallel.core.multicore.primitives.dense import matmul, swiglu_dense
from hyper_parallel.core.multicore.primitives.registry import REGISTRY, OpSchema


@dataclass(frozen=True)
class NativeProvider:
    """Static implementation, ownership and differentiation contract for one primitive.

    The backend resolves implementation names through trusted code. Descriptors
    never contain importable callables or invocation-owned addresses.
    """

    name: str
    logical_name: str
    version: int
    implementation: str
    dtypes: tuple[DType, ...]
    worker: str
    backward: tuple[str, ...]
    saved_inputs: tuple[str, ...]
    workspace: str = "operator_owned"
    context: str = "current_stream_invocation"
    tiling: str = "native_operator"

    def export_manifest(self) -> dict[str, object]:
        """Describe the implementation without materializing any resources."""
        data = asdict(self)
        data["dtypes"] = [dtype.value for dtype in self.dtypes]
        return data


class ProviderRegistry:
    """Admit only providers bound to exact canonical semantic schema identities."""

    def __init__(self) -> None:
        """Create a registry without loading native libraries."""
        self._providers: dict[tuple[str, int], NativeProvider] = {}

    def register(self, schema: OpSchema, provider: NativeProvider) -> None:
        """Bind one trusted implementation to its canonical semantic schema.

        Args:
            schema: Exact object registered in the semantic registry.
            provider: Immutable native capability and backward contract.
        """
        key = (schema.logical_name, schema.version)
        if REGISTRY.schema(*key) is not schema:
            raise ValueError("Native providers require canonical semantic schema identity")
        if (provider.logical_name, provider.version) != key:
            raise ValueError("Native provider identity does not match the semantic schema")
        if (not provider.name or not provider.implementation or not provider.backward
                or not provider.dtypes or any(not isinstance(dtype, DType) for dtype in provider.dtypes)):
            raise ValueError("Native providers require explicit implementation, dtype and backward contracts")
        if key in self._providers:
            raise ValueError(f"Duplicate native provider for {key}")
        self._providers[key] = provider

    def resolve(self, schema: OpSchema) -> NativeProvider:
        """Reject unknown or substituted schemas before native lowering.

        Args:
            schema: Canonical semantic schema of the IR operation.
        """
        if REGISTRY.schema(schema.logical_name, schema.version) is not schema:
            raise ValueError("Native provider resolution requires canonical schema identity")
        try:
            return self._providers[(schema.logical_name, schema.version)]
        except KeyError as error:
            raise ValueError(f"No native provider for {schema.logical_name}.v{schema.version}") from error


DENSE_PROVIDERS = ProviderRegistry()
DENSE_PROVIDERS.register(matmul.schema, NativeProvider(
    "torch_mm_v1", "dense.matmul", 1, "aten.mm.default", (DType.BF16, DType.FP16, DType.FP32),
    "cube", ("matmul_left_grad", "matmul_right_grad"), ("left", "right"),
))
DENSE_PROVIDERS.register(swiglu_dense.schema, NativeProvider(
    "npu_swiglu_v1", "dense.swiglu", 1, "npu.npu_swiglu.default", (DType.BF16, DType.FP16, DType.FP32),
    "vector", ("swiglu_input_grad",), ("packed",),
))
