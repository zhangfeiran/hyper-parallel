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
"""Parameter-owning shifted MHC entry point using the validated AST recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.frontend.program import Program
from hyper_parallel.core.multicore.runtime.mhc_native import MhcExecutable, MhcProfile
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec


class HyperMegaMhc(torch.nn.Module):
    """Own mapping parameters and cache immutable descriptors per shape/device."""

    def __init__(self, hidden_size: int, *, hc_eps: float = 1e-6, norm_eps: float = 1e-6,
                 num_iters: int = 20, token_tile: int = 32, device: Any = None,
                 program: Program = mhc_boundary, payload_root: Path | None = None) -> None:
        """Initialize the original four-stream parameter shapes and dtypes.

        Args:
            hidden_size: Feature count divisible by 128.
            hc_eps: Sinkhorn stability epsilon.
            norm_eps: NormCast and RMSNorm stability epsilon.
            num_iters: Fixed native Sinkhorn count of twenty.
            token_tile: Preferred outer token tile, increased for event capacity if needed.
            device: Initial parameter device; CPU construction does not load native code.
            program: Compatible shifted five-output AST region.
            payload_root: Isolated MHC payload, or the activated environment default.
        """
        super().__init__()
        MhcSpec(48, hidden_size, token_tile=token_tile)
        self.program = program
        self.hidden_size, self.token_tile = hidden_size, token_tile
        self.hc_eps, self.norm_eps, self.num_iters = hc_eps, norm_eps, num_iters
        self.payload_root = payload_root
        self._constexprs = {"hc_eps": hc_eps, "norm_eps": norm_eps, "num_iters": num_iters}
        program.plan(MhcSpec(48, hidden_size, token_tile=token_tile), **self._constexprs)
        options = {"device": device, "dtype": torch.float32}
        self.phi = torch.nn.Parameter(torch.empty((24, 4 * hidden_size), **options))
        self.alpha = torch.nn.Parameter(torch.ones(3, **options))
        self.bias = torch.nn.Parameter(torch.zeros(24, **options))
        self.norm_weight = torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=torch.bfloat16))
        torch.nn.init.normal_(self.phi, mean=0.0, std=(4 * hidden_size) ** -0.5)
        self._executables: dict[tuple[int, torch.device, bool], MhcExecutable] = {}
        self._closed = False

    def forward(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor, *, profile: bool = False) -> tuple[torch.Tensor, ...]:
        """Advance the shifted boundary with the previous layer's input mix.

        Args:
            previous_output: Previous block BF16 output.
            residual: BF16 streams ending in [4,H]; leading dimensions are flattened.
            previous_pre_mix: Previous input-mix coefficients in FP32.
            previous_post_mix: Previous output-mix coefficients in FP32.
            previous_residual_mix: Previous residual-mix matrix in FP32.
            profile: Collect real forward and backward task-cycle records.
        """
        if self._closed:
            raise RuntimeError("HyperMegaMhc is closed")
        if residual.device.type != "npu":
            raise ValueError("HyperMegaMhc requires NPU tensors; use program.interpret for the CPU reference")
        values = (residual, previous_output, previous_pre_mix, previous_post_mix,
                  previous_residual_mix, self.phi, self.alpha, self.bias, self.norm_weight)
        rows = residual.numel() // (4 * self.hidden_size)
        need_backward = torch.is_grad_enabled() and any(value.requires_grad for value in values)
        key = rows, residual.device, need_backward
        if key not in self._executables:
            cores = torch.npu.get_device_limit(residual.device.index)["cube_core_num"]
            spec = MhcSpec(rows, self.hidden_size, cores, self.token_tile, self.token_tile, need_backward)
            plan = self.program.plan(spec, **self._constexprs)
            self._executables[key] = plan.materialize(str(residual.device), payload_root=self.payload_root)
        return self._executables[key](*values, profile=profile)

    def take_profiles(self) -> tuple[MhcProfile, ...]:
        """Transfer invocation-owned profile buffers without synchronizing the hot path."""
        return tuple(profile for executable in self._executables.values() for profile in executable.take_profiles())

    def close(self) -> None:
        """Release cached descriptor ownership; pending autograd calls retain their resources."""
        self._closed = True
        for executable in self._executables.values():
            executable.close()
        self._executables.clear()
