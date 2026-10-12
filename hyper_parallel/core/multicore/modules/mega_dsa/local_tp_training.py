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
"""First-order fused CP/TP training with FP32 owner returns before final dtype conversion."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.mega_dsa.cp_tp_return import (
    CpTpSharedGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fp32_fused_kl import (
    Fp32FusedSelectedKlProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
    FusedDsaCpTpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.index_return import (
    SparseIndexGradReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_training import (
    _save_training,
    _saved_attention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.query_workspace import (
    scaled_query_workspace_vjps,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_owner_return import (
    TpOwnerGradientReturnProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import TpTileSumProbe
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)
from hyper_parallel.core.multicore.torch.ops import _load_native


class _LocalCpTpTraining(torch.autograd.Function):
    """Separate local-head main/index VJPs and return each shared field through CP then TP once."""

    @staticmethod
    def forward(ctx: Any, probe: FusedDsaLocalCpTpTrainingProbe, invocation: FusedCpTpInvocation,
                query: torch.Tensor, compressed: torch.Tensor, query_rope: torch.Tensor, key_rope: torch.Tensor,
                index_query: torch.Tensor, index_key: torch.Tensor,
                merge_weight: torch.Tensor,
                li_index: tuple[torch.Tensor, ...] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Save activations and scaled local-head KL VJPs through autograd tensor hooks."""
        ctx.set_materialize_grads(False)
        probe.check_native_support()
        main = (query, compressed, query_rope, key_rope)
        native_main = tuple(tensor.detach().bfloat16() for tensor in main) if probe.fp32_compressed_vjp else main
        index = (index_query, index_key, merge_weight)
        selected = index if li_index is None else li_index
        result = probe.tp_forward.forward(tuple(tensor.detach() for tensor in native_main),
                                          tuple(tensor.detach() for tensor in selected), invocation)
        scale = probe.loss_coeff*probe.normalization.local_sum_scale
        if probe.index_active and probe.fp32_index_kl:
            fp32 = probe.fp32_kl_backend.gradients(result, tuple(tensor.detach() for tensor in native_main),
                                                  tuple(tensor.detach() for tensor in index))
            loss = fp32.vjp.loss.sum()*scale
            gradients = tuple(tensor*scale for tensor in fp32.vjp.gradients)
        elif probe.index_active:
            loss = result.kl_loss*scale
            gradients = (result.kl_gradients[0].float()*scale, result.kl_key_partials*scale,
                         result.kl_gradients[2].float()*scale)
        else:
            loss = torch.zeros((), dtype=torch.float32, device=query.device)
            gradients = tuple(torch.empty(0, dtype=torch.float32, device=query.device) for _ in range(3))
        _save_training(ctx, probe, main, index, result, gradients)
        output = result.output.detach()
        if probe.mode == 'kl_only':
            ctx.mark_non_differentiable(output)
        if probe.mode == 'lm_only':
            ctx.mark_non_differentiable(loss)
        return output, loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor | None, grad_loss: torch.Tensor | None) -> tuple:
        """Preserve owner FP32 precision across both reductions and cast at the projection boundary."""
        if torch.is_grad_enabled():
            raise ValueError('local CP TP training supports first-order backward only')
        tensors = ctx.saved_tensors
        if len(tensors) != 18:
            raise RuntimeError('local CP TP training saved-state contract is incomplete')
        probe = ctx.probe
        probe.check_declaration()
        saved = _saved_attention(ctx, tensors)
        saved.validate_versions()
        backend = probe.forward_backend
        if ctx.geometry != (backend.heads, backend.scale, backend.schedule):
            raise ValueError('local CP TP training requires unchanged forward geometry')
        main, owner = (None,)*4, (None,)*2
        if ctx.main_enabled:
            cotangent = torch.zeros_like(tensors[12]) if grad_output is None else grad_output.detach().contiguous()
            result = probe.backward_backend._backward_saved(saved, ctx.output_shape, cotangent,
                                                            local_queries=True, defer_owner_cast=True)
            main = (result.gradients[0], None, result.gradients[2], None)
            if probe.fp32_native_query_vjp:
                query_gradient, rope_gradient = scaled_query_workspace_vjps(
                    result.retained, query_tokens=saved.states[0].shape[0], key_tokens=saved.states[1].shape[0],
                    heads=backend.heads, attention_scale=backend.scale)
                query_rope_gradient = (saved.local_query_layout.restore(rope_gradient)
                                       if probe.fp32_rope_vjp else main[2])
                main = (saved.local_query_layout.restore(query_gradient), None, query_rope_gradient, None)
            owner = result.owner_fp32
        index, index_partial = (None,)*3, None
        if ctx.index_enabled and ctx.has_index_gradients:
            upstream = tensors[15].new_zeros(()) if grad_loss is None else grad_loss.detach()
            gradients = tuple(tensor*upstream for tensor in tensors[15:])
            index = (gradients[0].to(ctx.index_dtypes[0]), None, gradients[2].to(ctx.index_dtypes[2]))
            index_partial = gradients[1]
        elif ctx.index_enabled:
            index = tuple(torch.zeros_like(tensor) for tensor in tensors[4:7])
        presence = (ctx.main_enabled, ctx.main_enabled, ctx.index_enabled and ctx.has_index_gradients)
        cp = backend.workspace.prepare(ctx.batch_meta)
        epoch = probe.shared_return.tp_return.transport.last_epoch+1
        invocation = probe.shared_return.prepare(cp, epoch=epoch, presence=presence)
        shared = probe.shared_return.return_owner(index_partial, owner, invocation,
            input_dtypes=(tensors[1].dtype, tensors[3].dtype, tensors[5].dtype)).shared_tp.gradients
        if ctx.main_enabled:
            main = (main[0], shared[0], main[2], shared[1])
        if presence[2]:
            index = (index[0], shared[2], index[2])
        return (None, None, *main, *index, None)


class FusedDsaLocalCpTpTrainingProbe:
    """Fused detached forward plus native CP main/index return and one shared TP SUM.

    Caller projections provide local main/index heads and shared owner-local
    keys. Main loss updates only Q/C/Q-RoPE/K-RoPE. Selected KL updates only
    IQ/IK/weights; detach its projection inputs from main hidden states. Loss
    remains TP replicated with identity predictor-SUM VJP. Coefficient, CP
    normalization and upstream cotangent apply once, with no extra TP divisor.
    Reserve the owner TP transport before any root consumer binds.
    """

    def __init__(self, forward: FusedDsaCpTpForwardProbe, owner_transport: TpTileSumProbe, *,
                 normalization: DsaLossNormalization, loss_coeff: float = 1., mode: str = 'lm_plus_kl',
                 sparse_main_return: bool = True, backward_schedule: MixedSfaSchedule | None = None,
                 group_private_partials: bool = False, fp32_index_kl: bool = False,
                 fp32_compressed_vjp: bool = False, fp32_native_query_vjp: bool = False,
                 fp32_rope_vjp: bool = False) -> None:
        """Collectively declare the objective and optional independent main backward schedule.

        The default backward schedule follows forward. An explicit schedule
        changes main backward compute groups while preserving forward/LI/KL
        geometry, loss scaling and the shared CP/TP return protocol.
        FP32 compressed VJP declares a compute-dtype C autograd input with
        unchanged BF16 native forward values and no cast after the CP/TP SUM.
        FP32 native query VJP also retains dQ before native BF16 post conversion;
        it requires the compressed VJP path and a compute-dtype Q input.
        FP32 RoPE VJP additionally retains dQ-RoPE from the native workspace
        and returns owner K-RoPE in FP32 after the CP/TP SUM.
        """
        if mode not in ('lm_only', 'kl_only', 'lm_plus_kl'):
            raise ValueError('CP TP training requires lm_only, kl_only or lm_plus_kl')
        if (not isinstance(normalization, DsaLossNormalization)
                or normalization.global_valid_queries != forward.cp.layout.batch_meta.global_valid_queries):
            raise ValueError('CP TP normalization must describe complete global queries')
        if (isinstance(loss_coeff, bool) or not isinstance(loss_coeff, (int, float))
                or not math.isfinite(loss_coeff) or loss_coeff < 0):
            raise ValueError('CP TP coefficient must be finite and nonnegative')
        if (owner_transport is forward.li or owner_transport is forward.kl
                or owner_transport.root is not forward.li.root or owner_transport.members != forward.li.members
                or owner_transport.arena is None):
            raise ValueError('shared owner return requires a separate bound transport on the same TP group and root')
        chosen = forward.cp.schedule if backward_schedule is None else backward_schedule
        if not isinstance(chosen, MixedSfaSchedule) or chosen.rounds != 1:
            raise ValueError('CP TP training backward schedule requires MixedSfaSchedule with rounds=1')
        if not isinstance(group_private_partials, bool):
            raise TypeError('CP TP group-private partials must be a boolean')
        if not isinstance(fp32_index_kl, bool):
            raise TypeError('CP TP FP32 index KL must be a boolean')
        if not isinstance(fp32_compressed_vjp, bool):
            raise TypeError('CP TP FP32 compressed VJP must be a boolean')
        if not isinstance(fp32_native_query_vjp, bool):
            raise TypeError('CP TP FP32 native query VJP must be a boolean')
        if fp32_native_query_vjp and not fp32_compressed_vjp:
            raise ValueError('FP32 native query VJP requires FP32 compressed VJP')
        if not isinstance(fp32_rope_vjp, bool):
            raise TypeError('CP TP FP32 RoPE VJP must be a boolean')
        if fp32_rope_vjp and not fp32_native_query_vjp:
            raise ValueError('FP32 RoPE VJP requires FP32 native query VJP')
        self.fp32_rope_vjp = fp32_rope_vjp
        self.fp32_index_kl = fp32_index_kl
        self.fp32_compressed_vjp = fp32_compressed_vjp
        self.fp32_native_query_vjp = fp32_native_query_vjp
        self.group_private_partials = group_private_partials
        self.tp_forward = forward
        self.forward_backend = forward.cp
        self.normalization = normalization
        self.loss_coeff = float(loss_coeff)
        self.mode = mode
        self.index_active = mode != 'lm_only' and self.loss_coeff != 0
        if self.index_active and not fp32_index_kl and forward.kl is None:
            raise ValueError('active CP TP selected KL requires a forward KL transport')
        self.sparse_main_return = sparse_main_return
        self.backward_schedule = chosen
        self._declaration = (mode, self.loss_coeff, normalization, self.index_active, sparse_main_return, chosen,
                             group_private_partials, fp32_index_kl, fp32_compressed_vjp,
                             fp32_native_query_vjp, fp32_rope_vjp)
        for group in (forward.cp.layout.group, forward.li.group):
            declarations = [None]*dist.get_world_size(group)
            dist.all_gather_object(declarations, self._declaration, group=group)
            if any(value != self._declaration for value in declarations):
                raise ValueError('CP TP members disagree on the training objective')
        self.backward_backend = FusedDsaCpBackwardProbe(
            forward.cp, sparse_return=sparse_main_return, schedule=chosen,
            group_private_partials=group_private_partials)
        self.shared_return = CpTpSharedGradientReturnProbe(SparseIndexGradReturnProbe(forward.cp),
                                                           TpOwnerGradientReturnProbe(owner_transport))
        self.fp32_kl_backend = (Fp32FusedSelectedKlProbe(forward.cp.layout.batch_meta, forward.cp.layout.group,
                                                       forward.li.group, forward.cp.layout.device,
                                                       scale=forward.cp.scale)
                                if fp32_index_kl and self.index_active else None)

    def check_declaration(self) -> None:
        """Reject objective mutation before retained or delayed backward can enter collectives."""
        if self._declaration != (self.mode, self.loss_coeff, self.normalization,
                                 self.index_active, self.sparse_main_return, self.backward_schedule,
                                 self.group_private_partials, self.fp32_index_kl, self.fp32_compressed_vjp,
                                 self.fp32_native_query_vjp, self.fp32_rope_vjp):
            raise ValueError('CP TP training objective was modified; prepare a new probe')

    def check_native_support(self) -> None:
        """Require every active forward/backward/return ABI before publication."""
        self.check_declaration()
        self.forward_backend.check_mapping_support()
        self.shared_return.tp_return.check_native_support()
        if self.mode != 'kl_only':
            self.backward_backend.check_local_native_support()
            if self.group_private_partials:
                self.backward_backend.check_group_partials_native_support()
        if self.index_active:
            self.shared_return.index_return.check_native_support()
        _load_native()
        if self.index_active and self.fp32_index_kl:
            fp32 = getattr(torch.ops.hyper_parallel, 'dsa_fp32_index_kl_version', None)
            if fp32 is None or fp32() != 1:
                raise RuntimeError('CP TP FP32 index KL requires native ABI 1')
        if self.mode != 'kl_only' and self.sparse_main_return:
            sparse = getattr(torch.ops.hyper_parallel, 'dsa_sparse_cp_grad_version', None)
            if sparse is None or sparse() != 1:
                raise RuntimeError('CP TP training requires sparse CP gradient adapter ABI 1')
        version = getattr(torch.ops.hyper_parallel, 'dsa_tp_local_cp_training_version', None)
        if version is None or version() != 1:
            raise RuntimeError('CP TP training requires fused TP adapter ABI 1')

    def prepare(self, invocation: DsaWorkspaceInvocation, *, epoch: int,
                query_ids: tuple[int, ...] | None = None) -> FusedCpTpInvocation:
        """Prepare fresh forward TP epochs; checkpoint replay must prepare a fresh invocation."""
        self.check_declaration()
        return self.tp_forward.prepare(invocation, epoch=epoch, query_ids=query_ids,
                                       selected_kl=self.index_active and not self.fp32_index_kl)

    def forward(self, invocation: FusedCpTpInvocation, main: tuple[torch.Tensor, ...],
                index: tuple[torch.Tensor, ...], *,
                li_index: tuple[torch.Tensor, ...] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local compressed output and scaled TP-replicated auxiliary loss with first-order VJPs."""
        self.check_declaration()
        if (invocation.backend is not self.tp_forward
                or invocation.selected_kl != (self.index_active and not self.fp32_index_kl)
                or len(main) != 4 or len(index) != 3):
            raise ValueError('CP TP training requires matching prepared branches, four main and three index states')
        if self.fp32_compressed_vjp and tuple(tensor.dtype for tensor in main) != (
                torch.float32 if self.fp32_native_query_vjp else torch.bfloat16,
                torch.float32, *(torch.float32 if self.fp32_rope_vjp else torch.bfloat16,)*2):
            raise ValueError('FP32 compressed VJP requires declared Q/RoPE precision and FP32 source-rounded C')
        if self.fp32_index_kl and self.index_active:
            if any(tensor.dtype != torch.float32 for tensor in index) or li_index is None or len(li_index) != 3:
                raise ValueError('CP TP FP32 KL requires FP32 index states and separate original BF16 LI states')
            if any(tensor.shape != original.shape or tensor.device != original.device
                   for tensor, original in zip(index, li_index)):
                raise ValueError('CP TP FP32 KL and original LI states must describe the same local rows and heads')
        elif li_index is not None:
            raise ValueError('separate LI states require an active FP32 KL branch')
        if self.mode == 'lm_only':
            index = tuple(tensor.detach() for tensor in index)
        elif self.mode == 'kl_only':
            main = tuple(tensor.detach() for tensor in main)
        return _LocalCpTpTraining.apply(self, invocation, *main, *index, li_index)
