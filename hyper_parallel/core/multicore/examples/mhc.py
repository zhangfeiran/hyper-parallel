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

"""Ordinary and Mega four-stream mHC boundaries with the same state and parameters."""

from __future__ import annotations

import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.sinkhorn import sinkhorn, sinkhorn_knopps
from hyper_parallel.core.multicore import HyperMegaMhc


# State order: previous block output, residual streams, pre, post, residual mix.
MhcState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def initial_mhc_state(hidden_states: torch.Tensor) -> MhcState:
    """Expand embeddings and bootstrap the first boundary with identity mappings."""
    leading_shape = hidden_states.shape[:-1]
    residual = hidden_states.unsqueeze(-2).expand(*leading_shape, 4, hidden_states.shape[-1]).contiguous()
    pre = torch.zeros((*leading_shape, 4), device=hidden_states.device, dtype=torch.float32)
    pre[..., 0] = 1.0
    post = torch.zeros_like(pre)
    mix = torch.eye(4, device=hidden_states.device, dtype=torch.float32).expand(*leading_shape, 4, 4).contiguous()
    return torch.zeros_like(hidden_states), residual, pre, post, mix


def require_omni_mhc() -> None:
    """Require the complete Omni training baseline instead of silently using Torch."""
    # Omni is optional for CPU semantic checks, but required by the NPU common path.
    try:
        import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError as exc:
        raise ImportError("NPU mHC common requires omni_training_custom_ops and its matching vendor OPP") from exc
    required = (
        "npu_ai_infra_manifold_constrained_hyper_connection_post",
        "npu_ai_infra_mhc_post_grad",
        "npu_sinkhorn",
        "npu_sinkhorn_grad",
    )
    missing = [name for name in required if not hasattr(torch.ops.custom, name)]
    if missing:
        raise RuntimeError(f"NPU mHC common is missing Omni training operators: {', '.join(missing)}")


def _advance_residual(state: MhcState) -> torch.Tensor:
    """Apply the shared Omni post, including the final pending block output."""
    output, residual, _, post, mix = state
    if residual.device.type != "npu":
        return _reference_residual(state)
    require_omni_mhc()
    hidden = residual.shape[-1]
    result = mhc_post(
        output.reshape(1, -1, hidden), residual.reshape(1, -1, 4 * hidden),
        post.reshape(1, -1, 4), mix.reshape(1, -1, 4, 4), 4,
    )
    return result.reshape_as(residual)


def _reference_residual(state: MhcState) -> torch.Tensor:
    """Evaluate the post formula without Omni for independent precision checks."""
    output, residual, _, post, mix = state
    if residual.device.type == "npu":
        # Match CANN MhcPost: start with post, then accumulate four Axpy operations.
        result = post.float().unsqueeze(-1) * output.float().unsqueeze(-2)
        for stream in range(4):
            result = torch.addcmul(
                result, residual[..., stream:stream + 1, :].float(), mix[..., stream, :].float().unsqueeze(-1),
            )
        return result.to(residual.dtype)
    return (
        torch.einsum("...ji,...jh->...ih", mix.float(), residual.float())
        + post.float().unsqueeze(-1) * output.float().unsqueeze(-2)
    ).to(residual.dtype)


def _mapping_logits(residual: torch.Tensor, phi: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """Use semantic CPU math and the matching CANN projection order on NPU."""
    residual_fp32 = residual.float()
    variance = residual_fp32.square().mean(dim=(-2, -1), keepdim=True) + norm_eps
    if residual.device.type != "npu":
        normalized = residual_fp32 * torch.rsqrt(variance)
        return torch.matmul(normalized.flatten(-2), phi.float().transpose(-1, -2))
    # CANN projects the unnormalized BF16 values with HF32, then scales the result.
    previous_hf32 = torch.npu.matmul.allow_hf32
    try:
        torch.npu.matmul.allow_hf32 = True
        logits = torch.matmul(residual_fp32.flatten(-2), phi.float().transpose(-1, -2))
    finally:
        torch.npu.matmul.allow_hf32 = previous_hf32
    return logits * (1.0 / torch.sqrt(variance)).squeeze(-1)


def finish_mhc_state(state: MhcState) -> torch.Tensor:
    """Flush the final block output and collapse streams using the pending pre mapping."""
    residual = _advance_residual(state)
    return (residual.float() * state[2].float().unsqueeze(-1)).sum(dim=-2).to(residual.dtype)


class MhcBoundary(HyperMegaMhc):
    """Compare ordinary Omni/Torch mHC with Mega using identical parameters."""

    def __init__(self, hidden_size: int, norm_eps: float, *, use_mega_mhc: bool) -> None:
        """Initialize FP32 predictors and the BF16 input RMSNorm weight."""
        super().__init__(hidden_size, norm_eps=norm_eps)
        self.use_mega_mhc = use_mega_mhc

    def forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MhcState:
        """Return updated residual, next mappings, and the current normalized block input."""
        state = (previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix)
        if self.use_mega_mhc:
            return self.mega_forward(*state)
        return self.native_forward(*state)

    def mega_forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MhcState:
        """Replace the boundary calculations with one managed Mega mHC call."""
        return super().forward(
            previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix,
        )

    def _mappings(self, logits: torch.Tensor, *, use_omni: bool) -> tuple[torch.Tensor, ...]:
        """Share coefficient transforms while keeping the Sinkhorn backend explicit."""
        pre_logits, post_logits, mix_logits = logits.split((4, 4, 16), dim=-1)
        pre_bias, post_bias, mix_bias = self.bias.float().split((4, 4, 16))
        pre = torch.sigmoid(pre_logits * self.alpha[0] + pre_bias) + self.hc_eps
        post = 2.0 * torch.sigmoid(post_logits * self.alpha[1] + post_bias)
        mix = (mix_logits * self.alpha[2] + mix_bias).reshape(*logits.shape[:-1], 4, 4)
        normalize = sinkhorn if use_omni else sinkhorn_knopps
        return pre, post, normalize(mix, self.num_iters, self.hc_eps)

    def native_forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MhcState:
        """Run ordinary Post, projection, Sinkhorn, mixing, and RMSNorm operations."""
        state = (previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix)
        if residual.device.type != "npu":
            return self.reference_forward(*state)
        new_residual = _advance_residual(state)
        flattened = new_residual.flatten(-2).float()
        normalized = flattened * torch.rsqrt(flattened.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        logits = F.linear(normalized, self.phi.float())  # pylint: disable=not-callable
        pre, post, mix = self._mappings(logits, use_omni=True)
        mixed = (new_residual.float() * state[2].float().unsqueeze(-1)).sum(dim=-2).to(new_residual.dtype)
        normalized = mixed.float() * torch.rsqrt(mixed.float().square().mean(dim=-1, keepdim=True) + self.norm_eps)
        block_input = normalized.to(mixed.dtype) * self.norm_weight
        return new_residual, pre, post, mix, block_input

    def reference_forward(
        self,
        previous_output: torch.Tensor,
        residual: torch.Tensor,
        previous_pre_mix: torch.Tensor,
        previous_post_mix: torch.Tensor,
        previous_residual_mix: torch.Tensor,
    ) -> MhcState:
        """Evaluate the Torch precision reference without selecting the timing baseline."""
        state = (previous_output, residual, previous_pre_mix, previous_post_mix, previous_residual_mix)
        new_residual = _reference_residual(state)
        logits = _mapping_logits(new_residual, self.phi, self.norm_eps)
        pre, post, mix = self._mappings(logits, use_omni=False)
        if residual.device.type == "npu":
            # CANN reduces streams left to right and stores BF16 before ordinary RMSNorm.
            weighted = new_residual.float() * previous_pre_mix.float().unsqueeze(-1)
            mixed_input = weighted[..., 0, :]
            for stream in range(1, 4):
                mixed_input = mixed_input + weighted[..., stream, :]
            block_input, _ = torch.ops.npu.npu_rms_norm(
                mixed_input.to(residual.dtype), self.norm_weight, self.norm_eps,
            )
            return new_residual, pre, post, mix, block_input
        mixed_input = (new_residual.float() * previous_pre_mix.float().unsqueeze(-1)).sum(dim=-2)
        block_input = mixed_input * torch.rsqrt(mixed_input.square().mean(dim=-1, keepdim=True) + self.norm_eps)
        block_input = (block_input * self.norm_weight.float()).to(residual.dtype)
        return new_residual, pre, post, mix, block_input
