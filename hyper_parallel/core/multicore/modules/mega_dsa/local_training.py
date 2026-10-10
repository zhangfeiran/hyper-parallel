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
"""First-order local LM and selected-KL training with invocation-owned saved gradients."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedCpSavedAttention,
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_kl import (
    LocalKlResult,
    local_kl_gradients,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


def _save_training(ctx, probe, states, index, result, gradients):
    saved = result.saved
    ctx.save_for_backward(*states, *index, *saved.states, saved.indices, *saved.forward, *gradients)
    ctx.probe = probe
    ctx.batch_meta = saved.batch_meta
    ctx.versions = saved.versions
    ctx.geometry = saved.geometry
    ctx.token = saved._token
    ctx.output_shape = tuple(result.output.shape)
    ctx.local_query_layout = saved.local_query_layout
    ctx.index_dtypes = tuple(tensor.dtype for tensor in index)
    ctx.main_enabled = probe.mode != "kl_only"
    ctx.index_enabled = probe.mode != "lm_only"
    ctx.has_index_gradients = probe.loss_coeff != 0


def _saved_attention(ctx, tensors):
    return FusedCpSavedAttention(ctx.batch_meta, tuple(tensors[7:11]), tensors[11], tuple(tensors[12:15]),
                                 ctx.versions, ctx.probe.forward_backend, ctx.geometry, ctx.token,
                                 "local", ctx.local_query_layout)


class _LocalCpTraining(torch.autograd.Function):
    """Keep teacher and hard-selection gradients isolated from the main attention VJP."""

    @staticmethod
    def forward(ctx: Any, probe: FusedDsaLocalCpTrainingProbe, invocation: DsaWorkspaceInvocation,
                query_layout: DsaLocalQueryLayout, query: torch.Tensor, compressed: torch.Tensor,
                query_rope: torch.Tensor, key_rope: torch.Tensor, index_query: torch.Tensor,
                index_key: torch.Tensor, merge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute raw index VJPs, then save scaled independent FP32 gradients through tensor hooks."""
        ctx.set_materialize_grads(False)
        states = (query, compressed, query_rope, key_rope)
        index = (index_query, index_key, merge_weight)
        probe.check_native_support()
        fused = probe.fused_kl and probe.mode != "lm_only" and probe.loss_coeff != 0
        extra = {"selected_kl": True} if fused else {}
        result = probe.forward_backend.forward_local(invocation, tuple(tensor.detach() for tensor in states),
                                                      tuple(tensor.detach() for tensor in index),
                                                      query_layout=query_layout, **extra)
        scale = probe.loss_coeff * probe.normalization.local_sum_scale
        if probe.mode != "lm_only" and scale:
            raw = (LocalKlResult(result.kl_gradients, result.kl_key_partials, result.kl_loss,
                                  result.phase_traces[-3:], result.kl_retained) if fused
                   else local_kl_gradients(result))
            loss = raw.loss * scale
            gradients = (raw.gradients[0].float()*scale, raw.key_partials*scale, raw.gradients[2].float()*scale)
        else:
            loss = torch.zeros((), dtype=torch.float32, device=query.device)
            gradients = tuple(torch.empty(0, dtype=torch.float32, device=query.device) for _ in range(3))
        _save_training(ctx, probe, states, index, result, gradients)
        output = result.output.detach()
        if probe.mode == "kl_only":
            ctx.mark_non_differentiable(output)
        if probe.mode == "lm_only":
            ctx.mark_non_differentiable(loss)
        return output, loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor | None, grad_loss: torch.Tensor | None) -> tuple:
        """Apply each upstream cotangent once, then reduce only global index-K contributions."""
        if torch.is_grad_enabled():
            raise ValueError("local CP training supports first-order backward only")
        tensors = ctx.saved_tensors
        if len(tensors) != 18:
            raise RuntimeError("local CP training saved-state contract is incomplete")
        saved = _saved_attention(ctx, tensors)
        saved.validate_versions()
        backend = ctx.probe.forward_backend
        if ctx.geometry != (backend.heads, backend.scale, backend.schedule):
            raise ValueError("local CP training requires unchanged forward geometry/scale/schedule")
        main = (None,)*4
        if ctx.main_enabled:
            cotangent = (torch.zeros_like(tensors[12]) if grad_output is None
                         else grad_output.detach().contiguous())
            main = ctx.probe.backward_backend._backward_saved(
                saved, ctx.output_shape, cotangent, local_queries=True).gradients
        index = (None,)*3
        if ctx.index_enabled and ctx.has_index_gradients:
            upstream = (tensors[15].new_zeros(()) if grad_loss is None else grad_loss.detach())
            gradients = tuple(tensor * upstream for tensor in tensors[15:])
            index = backend.layout.reduce_local_index_gradients(gradients, ctx.batch_meta, ctx.index_dtypes)
        elif ctx.index_enabled:
            # The collectively declared zero coefficient has no contribution on any member.
            index = tuple(torch.zeros_like(tensor) for tensor in tensors[4:7])
        return (None, None, None, *main, *index)


class FusedDsaLocalCpTrainingProbe:
    """Local LI/pull/SFA training with raw native selected KL and independent gradient consumers.

    Selected KL can run in the forward graph with ``fused_kl=True`` or through
    separately submitted phases. Both retain complete index-K owner reduction.
    ``sparse_main_return=True`` omits zero merged main-gradient rows; index-K
    still uses complete owner reduction. Mode, coefficient, normalization,
    fusion and main-gradient return are collectively declared at
    construction. Every member must enter the same forward/backward schedule,
    including empty owners, with compatible requires-grad policies. Projection
    callers detach hidden states before index projection so auxiliary gradients
    cannot update the main hidden-state path.
    """

    def __init__(self, forward: FusedDsaCpForwardProbe, *, normalization: DsaLossNormalization,
                 loss_coeff: float = 1.0, mode: str = "lm_plus_kl", fused_kl: bool = False,
                 sparse_main_return: bool = False) -> None:
        """Cold prepare a common objective contract before any native transport or training step."""
        if mode not in ("lm_only", "kl_only", "lm_plus_kl"):
            raise ValueError("local CP training mode must be lm_only, kl_only or lm_plus_kl")
        if not isinstance(normalization, DsaLossNormalization):
            raise ValueError("local CP training requires explicit DsaLossNormalization")
        if normalization.global_valid_queries != forward.layout.batch_meta.global_valid_queries:
            raise ValueError("local CP training normalization must use complete global query count")
        if not isinstance(loss_coeff, (int, float)) or isinstance(loss_coeff, bool) or not math.isfinite(loss_coeff):
            raise ValueError("local CP training coefficient must be finite and nonnegative")
        if loss_coeff < 0:
            raise ValueError("local CP training coefficient must be finite and nonnegative")
        if not isinstance(fused_kl, bool):
            raise ValueError("local fused KL selection must be a boolean")
        if not isinstance(sparse_main_return, bool):
            raise ValueError("local sparse main-gradient return must be a boolean")
        declaration = (mode, float(loss_coeff), normalization, fused_kl, sparse_main_return)
        peers = [declaration]
        if forward.layout.cp_size > 1:
            peers = [None]*forward.layout.cp_size
            dist.all_gather_object(peers, declaration, group=forward.layout.group)
        if any(peer != declaration for peer in peers):
            raise ValueError("CP members disagree on the prepared local training objective")
        self.forward_backend = forward
        self.backward_backend = FusedDsaCpBackwardProbe(forward, sparse_return=sparse_main_return)
        self.normalization = normalization
        self.loss_coeff = float(loss_coeff)
        self.mode = mode
        self.fused_kl = fused_kl
        self.sparse_main_return = sparse_main_return
        self._declaration = declaration

    def check_native_support(self) -> None:
        """Admit all needed native ABIs before acquiring the forward workspace lease."""
        if self._declaration != (self.mode, self.loss_coeff, self.normalization, self.fused_kl,
                                 self.sparse_main_return):
            raise ValueError("local training objective was modified; prepare a new probe")
        if (not isinstance(self.sparse_main_return, bool)
                or self.backward_backend.sparse_return != self.sparse_main_return):
            raise ValueError("local training main-gradient return was modified; prepare a new probe")
        if self.mode != "kl_only":
            self.backward_backend.check_local_native_support()
            if self.sparse_main_return:
                _load_native()
                version = getattr(torch.ops.hyper_parallel, "dsa_sparse_cp_grad_version", None)
                if version is None or version() != 1:
                    raise RuntimeError("sparse main-gradient return requires adapter ABI 1; rebuild this payload")
        if self.mode != "lm_only" and self.loss_coeff:
            _load_native()
            name = "dsa_local_cp_training_version" if self.fused_kl else "dsa_local_mixed_kl_version"
            version = getattr(torch.ops.hyper_parallel, name, None)
            if version is None or version() != 1:
                raise RuntimeError("local CP training requires local KL adapter ABI 1; rebuild this checkout's payload")

    def forward(self, invocation: DsaWorkspaceInvocation, main_states: tuple[torch.Tensor, ...],
                index_states: tuple[torch.Tensor, ...], *,
                query_layout: DsaLocalQueryLayout) -> tuple[torch.Tensor, torch.Tensor]:
        """Return caller-ordered compressed output and this tile's normalized auxiliary loss.

        Args:
            invocation: Workspace invocation with unchanged owner geometry.
            main_states: Caller-local Q/Q-RoPE and complete owner-local C/K-RoPE.
            index_states: Caller-local IQ/weights and complete owner-local IK.
                Projection outputs retain autograd; their hidden inputs must be
                detached by the caller before index projection.
            query_layout: Explicit cold layout captured for delayed execution
                or checkpoint replay, even after a later tile is prepared.

        Returns:
            Compressed output and the scaled local KL contribution. Main loss
            gradients reach only four main inputs. Auxiliary loss gradients
            reach only three index inputs; global key contributions are summed
            in FP32 before one cast, with no additional CP division.
        """
        if len(main_states) != 4 or len(index_states) != 3:
            raise ValueError("local CP training requires four main and three index states")
        if not isinstance(query_layout, DsaLocalQueryLayout):
            raise ValueError("local CP training requires an explicit cold query layout")
        return _LocalCpTraining.apply(self, invocation, query_layout, *main_states, *index_states)
