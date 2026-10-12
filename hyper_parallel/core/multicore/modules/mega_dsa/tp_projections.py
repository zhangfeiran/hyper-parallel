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
"""Cold DSA projection sharding, checkpoint conversion and explicit TP projection VJPs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.core.multicore.modules.mega_dsa.parameter_sum import (
    ordered_parameter_sum,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_collectives import (
    DsaTpHeadCollectives,
)


@dataclass(frozen=True)
class DsaTpProjectionLayout:
    """Global head geometry and ordered plain-Torch parameter shards for one TP group."""

    group: dist.ProcessGroup
    members: tuple[int, ...]
    rank: int
    global_heads: int
    global_index_heads: int
    shards: tuple[tuple[str, int, tuple[int, ...]], ...]

    @property
    def local_heads(self) -> int:
        """Return this evenly sized main head shard."""
        return self.global_heads//len(self.members)

    @property
    def local_index_heads(self) -> int:
        """Return this evenly sized shard of the original index heads."""
        return self.global_index_heads//len(self.members)


def shard_dsa_tp_projections(module: DeepseekV32DSAAttention, group: dist.ProcessGroup) -> DsaTpProjectionLayout:
    """Shard projection storage in place before optimizer/FSDP construction, preserving Parameter identity.

    Q-B, KV-B, IQ and merge-weight projections use row shards; O uses a column
    shard. QKV latent projection, norms, index-K projection and O bias remain
    replicated. The source becomes a local projection container and must be
    consumed by the CP/TP attention adapter. Do not call its unsharded forward.
    This cold operation accepts plain Torch Parameters without existing grads.
    The first TP member supplies the initial global parameter values.
    """
    if group is None or group == dist.GroupMember.NON_GROUP_MEMBER or not dist.is_initialized():
        raise ValueError('DSA TP projection sharding requires an initialized explicit group')
    if getattr(module, 'dsa_tp_projection_layout', None) is not None:
        raise ValueError('DSA projections have already been sharded')
    members = tuple(dist.get_process_group_ranks(group))
    rank, size = dist.get_rank(group), len(members)
    if (size not in (1, 2, 4, 8) or rank < 0 or module.num_heads % size
            or module.num_index_heads != 64 or module.num_heads//size not in (32, 64)):
        raise ValueError('DSA projection TP requires local H32/H64, global Hi64 and evenly sized TP1/2/4/8')
    if module.kv_b_proj.bias is not None:
        raise ValueError('absorbed DSA TP requires a bias-free KV-B projection')
    row_layers = ('q_b_proj', 'kv_b_proj', 'indexer.wq_b', 'indexer.weights_proj')
    shapes = {
        'q_b_proj': (module.num_heads*module.qk_head_dim, module.q_lora_rank),
        'kv_b_proj': (module.num_heads*(module.qk_nope_head_dim+module.v_head_dim), module.kv_lora_rank),
        'o_proj': (module.linear_qkv.in_features, module.num_heads*module.v_head_dim),
        'indexer.wq_b': (64*module.index_head_dim, module.q_lora_rank),
        'indexer.weights_proj': (64, module.linear_qkv.in_features),
    }
    shards = []
    layers = []
    for name in (*row_layers, 'o_proj'):
        layer = module.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise TypeError('DSA TP projections require plain Torch Linear layers')
        if tuple(layer.weight.shape) != shapes[name]:
            raise ValueError('DSA projection weights must have the complete declared global geometry')
        axis = 1 if name == 'o_proj' else 0
        if layer.weight.shape[axis] % size:
            raise ValueError('DSA projection dimension must divide evenly over TP')
        shards.append((name+'.weight', axis, tuple(layer.weight.shape)))
        if axis == 0 and layer.bias is not None:
            shards.append((name+'.bias', 0, tuple(layer.bias.shape)))
        layers.append((layer, axis))
    shard_names = {name for name, _axis, _shape in shards}
    seen = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if type(parameter) is not nn.Parameter:  # pylint: disable=unidiomatic-typecheck
            raise TypeError('DSA TP sharding must precede DTensor/FSDP parameter wrapping')
        if parameter.grad is not None or parameter._backward_hooks:
            raise ValueError('DSA TP sharding must precede gradient hooks and optimizer state')
        previous = seen.get(id(parameter))
        if previous is not None and (name in shard_names or previous in shard_names):
            raise ValueError('DSA TP head projection parameters must not be tied to other fields')
        seen[id(parameter)] = name
    declaration = (module.num_heads, module.num_index_heads, tuple(shards),
                   tuple((name, str(parameter.dtype), parameter.requires_grad)
                         for name, parameter in module.named_parameters()))
    peers = [None]*size
    dist.all_gather_object(peers, declaration, group=group)
    if any(value != declaration for value in peers):
        raise ValueError('TP peers disagree on projection geometry, dtype or trainable fields')
    if size > 1:
        with torch.no_grad():
            for parameter in module.parameters():
                dist.broadcast(parameter.detach(), src=members[0], group=group)
    layout = DsaTpProjectionLayout(group, members, rank, module.num_heads, module.num_index_heads, tuple(shards))
    if size > 1:
        with torch.no_grad():
            for name, axis, _shape in layout.shards:
                parameter = module.get_parameter(name)
                local = parameter.detach().chunk(size, dim=axis)[rank].clone(memory_format=torch.contiguous_format)
                parameter.set_(local)
        for layer, axis in layers:
            if axis == 0:
                layer.out_features //= size
            else:
                layer.in_features //= size
    module.dsa_tp_projection_layout = layout
    return layout


def load_dsa_tp_state_dict(module: nn.Module, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> Any:
    """Load global HP-format tensors into existing local shards without replacing Parameters.

    This accepts the already converted HP keys, including linear_qkv. Original
    Transformers-to-HP Q/KV concatenation remains the separate source converter.
    All TP peers must load the same shared weights and their corresponding shards.
    """
    layout = getattr(module, 'dsa_tp_projection_layout', None)
    if not isinstance(layout, DsaTpProjectionLayout):
        raise TypeError('checkpoint loading requires prepared DSA TP projections')
    local = dict(state)
    for name, axis, shape in layout.shards:
        if name not in local:
            continue
        if tuple(local[name].shape) != shape:
            raise ValueError(f'global checkpoint tensor {name} has incompatible shape')
        local[name] = local[name].chunk(len(layout.members), dim=axis)[layout.rank].contiguous()
    return module.load_state_dict(local, strict=strict)


def gather_dsa_tp_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Collectively reconstruct global HP-format weights on every TP member outside autograd."""
    layout = getattr(module, 'dsa_tp_projection_layout', None)
    if not isinstance(layout, DsaTpProjectionLayout):
        raise TypeError('checkpoint gathering requires prepared DSA TP projections')
    state = dict(module.state_dict())
    for name, axis, shape in layout.shards:
        source = state[name].contiguous()
        peers = [torch.empty_like(source) for _ in layout.members]
        dist.all_gather(peers, source, group=layout.group)
        value = torch.cat(peers, dim=axis)
        if tuple(value.shape) != shape:
            raise ValueError('gathered DSA TP checkpoint has incompatible global geometry')
        state[name] = value
    return state


class _CopyProjectionInput(torch.autograd.Function):
    """A replicated query latent has one contribution from each local main head shard."""

    @staticmethod
    def forward(ctx: Any, boundary: DsaTpProjectionCollectives, tensor: torch.Tensor) -> torch.Tensor:
        """Keep the projection input replicated without gathering heads."""
        ctx.set_materialize_grads(False)
        ctx.boundary = boundary
        return tensor

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """SUM head-local input VJPs before reconnecting the shared Q latent projection."""
        if torch.is_grad_enabled():
            raise ValueError('DSA TP projections support first-order backward only')
        summed = None if gradient is None else ctx.boundary.sum_detached(gradient.detach()).to(gradient.dtype)
        return None, summed


class _SumProjectionOutput(torch.autograd.Function):
    """Row-parallel O projection replicates its output and preserves one cotangent per shard."""

    @staticmethod
    def forward(ctx: Any, boundary: DsaTpProjectionCollectives, partial: torch.Tensor) -> torch.Tensor:
        """SUM local O projection values in FP32, then convert once to the activation dtype."""
        ctx.set_materialize_grads(False)
        return boundary.sum_detached(partial.detach()).to(partial.dtype)

    @staticmethod
    def backward(_ctx: Any, gradient: torch.Tensor | None) -> tuple:
        """Return the replicated output cotangent to this local weight shard."""
        return None, gradient


class DsaTpProjectionCollectives(DsaTpHeadCollectives):
    """Use distinct input-SUM and output-identity VJPs for plain-Torch projection shards."""

    def __init__(self, group: dist.ProcessGroup, *, row_signature: tuple,
                 reduction_order: str = 'collective') -> None:
        """Agree on projection SUM policy before executing any projection.

        Args:
            group: Explicit participating TP group.
            row_signature: Immutable declaration of the same projection rows.
            reduction_order: Default collective SUM or experimental group-rank
                ordered SUM for Q-B input VJPs and row-parallel O values.
        """
        if reduction_order not in ('collective', 'group_rank'):
            raise ValueError('TP projection reduction order must be collective or group_rank')
        if not isinstance(row_signature, tuple):
            raise TypeError('TP row signature must be a host tuple')
        super().__init__(group, row_signature=(row_signature, reduction_order))
        self.reduction_order = reduction_order

    def sum_detached(self, partial: torch.Tensor) -> torch.Tensor:
        """Apply the agreed FP32/FP64 SUM without changing projection autograd semantics."""
        if self.reduction_order == 'collective':
            return super().sum_detached(partial)
        if partial.requires_grad:
            raise ValueError('detached TP SUM must not discard an active autograd edge')
        if not partial.is_floating_point():
            raise ValueError('TP fields must be floating-point tensors')
        dtype = torch.float64 if partial.dtype == torch.float64 else torch.float32
        return ordered_parameter_sum(partial.to(dtype), self.group)

    def copy_query_latent(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replicate normalized main Q latent; SUM its local-head VJPs in backward."""
        return _CopyProjectionInput.apply(self, tensor)

    def sum_output(self, partial: torch.Tensor) -> torch.Tensor:
        """Replicate the row-parallel O output; use identity VJP for the replicated loss."""
        return _SumProjectionOutput.apply(self, partial)
