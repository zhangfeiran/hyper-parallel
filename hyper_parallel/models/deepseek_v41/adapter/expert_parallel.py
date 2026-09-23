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
"""Expert-parallel forward replacement for DeepSeek-V4.1 MoE blocks."""

from __future__ import annotations

import inspect
from typing import Any, Callable, TYPE_CHECKING

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
    ep_routed_forward,
    require_attrs,
)
from hyper_parallel.distributed.recipe_spec import local_compute

if TYPE_CHECKING:
    from hyper_parallel.trainer.config import TrainerConfig


def _router(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the V4.1 learned router's selected experts and weights."""
    output = (
        module.gate(hidden_states)
        if image_mask is None
        else module.gate(hidden_states, image_mask=image_mask)
    )
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise TypeError("DeepSeek-V4.1 gate must return logits, weights, and indices")
    _, weights, indices = output
    return indices, weights


def _routed_and_shared_forward(
        module: Any,
        hidden_states: torch.Tensor,
        image_mask: torch.Tensor | None,
        ep_group: Any,
        megamoe: bool = False,
) -> torch.Tensor:
    """Dispatch routed experts and add the V4.1 shared expert branch."""
    if megamoe:
        indices, weights = _router(module, hidden_states, image_mask)
        routed = module.experts(hidden_states, indices, weights)
    else:
        routed = ep_routed_forward(
            module,
            hidden_states,
            router_fn=lambda target_module, target_states: _router(target_module, target_states, image_mask),
            ep_group=ep_group,
        )
    return routed + module.shared_experts(hidden_states)


@local_compute
def deepseek_v41_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
        megamoe: bool = False,
        max_local_num_tokens: int = 128,
        expert_capacity_factor: float | None = 2.0,
) -> Callable:
    """Build the complete V4.1 routed-plus-shared expert forward."""
    del mesh
    require_attrs(module, "gate", "experts", "shared_experts", owner="DeepSeek-V4.1 EP")
    if ep_mesh is None and not megamoe:
        raise ValueError("DeepSeek-V4.1 EP forward requires an active ep_mesh")
    if module.is_hash:
        raise ValueError("DeepSeek-V4.1 validation layers must use learned routing")
    if use_grouped_gemm:
        raise ValueError("DeepSeek-V4.1 clamp semantics currently require use_grouped_gemm=false")
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    if megamoe:
        # Native EP must remain usable without the optional multicore component.
        from hyper_parallel.models.deepseek_v41.adapter.replacements import (  # pylint: disable=C0415
            DeepseekV41TrainingExperts, _swiglu_limit,
        )

        if any(axis is not None and axis.size() > 1 for axis in (tp_mesh, cp_mesh)):
            raise ValueError("DSV4.1 MegaMoe Trainer currently requires TP=CP=1")
        if not isinstance(module.experts, DeepseekV41TrainingExperts):
            raise TypeError("Apply DeepseekV41TrainingExperts replacement before enabling MegaMoe")
        if _swiglu_limit(module.shared_experts.limit) != module.experts.swiglu_limit:
            raise ValueError("DSV4.1 routed and shared experts must use the same swiglu_limit")
        module.experts.configure(ep_group, 1 if ep_mesh is None else ep_mesh["ep"].size(), max_local_num_tokens,
                                 module.gate.top_k, expert_capacity_factor)
    else:
        bind_local_expert_forward(
            module,
            ep_mesh["ep"].size(),
            apply_gate=module.experts._apply_gate,  # pylint: disable=protected-access
        )

    if "image_mask" not in inspect.signature(module.forward).parameters:
        def text_compute_fn(
                module: Any,
                hidden_states: torch.Tensor,
                input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the text-only source contract without visual routing state."""
            del input_ids
            return _routed_and_shared_forward(module, hidden_states, None, ep_group, megamoe)

        return text_compute_fn

    def multimodal_compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the multimodal source contract with optional visual routing state."""
        del input_ids
        return _routed_and_shared_forward(module, hidden_states, image_mask, ep_group, megamoe)

    return multimodal_compute_fn


@local_compute
def deepseek_v41_engram_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
) -> Callable:
    """Build the sparse EP-row lookup with CP/TP sequence alignment."""
    del mesh
    require_attrs(
        module,
        "hash_mapping",
        "embed",
        "parallel_forward",
        owner="DeepSeek-V4.1 Engram EP",
    )
    if ep_mesh is None:
        raise ValueError("DeepSeek-V4.1 Engram EP requires an active ep_mesh")
    ep_group = ep_mesh.get_group("ep")
    ep_size = ep_mesh["ep"].size()
    ep_rank = ep_mesh.get_local_rank("ep")
    cp_group = None if cp_mesh is None else cp_mesh.get_group()
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    cp_rank = 0 if cp_mesh is None else cp_mesh.get_local_rank()
    tp_size = 1 if tp_mesh is None else tp_mesh.size()
    tp_rank = 0 if tp_mesh is None else tp_mesh.get_local_rank()

    def compute_fn(
            module: Any,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            segment_starts: torch.Tensor | None = None,
            token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route hash-row requests and run the exact V4.1 fusion."""
        return module.parallel_forward(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    return compute_fn


def configure_megamoe(config: TrainerConfig) -> None:
    """Enable DSV4.1 MegaMoe in the existing recipe, leaving disabled recipes intact.

    Args:
        config: Resolved training recipe, including CLI sequence-length and packing-budget overrides.
    """
    if not config.megamoe:
        return
    # Trainer configuration and multicore replacements are optional for native EP.
    from hyper_parallel.trainer.config import PlanOverride, Target  # pylint: disable=C0415
    from hyper_parallel.models.deepseek_v41.adapter.replacements import (  # pylint: disable=C0415
        DeepseekV41TrainingExperts,
    )

    if any(getattr(config.accelerator, axis) != 1 for axis in ("tp_size", "cp_size", "pp_size")):
        raise ValueError("DSV4.1 MegaMoe training requires TP=CP=PP=1")
    entries = [entry for entry in config.plan_overrides
               if entry.local_compute_fn is not None
               and entry.local_compute_fn._target_ is deepseek_v41_ep_compute_fn]  # pylint: disable=protected-access
    if not entries:
        raise ValueError("megamoe=true requires a DSV4.1 EP compute rule")
    max_length = config.dataset.data_transform.max_seq_len
    token_budget = getattr(config.dataloader, "token_budget", None)
    if token_budget is None:
        token_budget = config.training.micro_batch_size * max_length
    if max_length <= 0 or token_budget <= 0:
        raise ValueError("MegaMoe sequence length and packing budget must be positive")
    # Omni packing may emit one full sample even when it exceeds the selection budget.
    tokens = max(max_length, token_budget)
    for entry in entries:
        entry.when = None
        entry.local_compute_fn = Target(
            deepseek_v41_ep_compute_fn,
            target_path="hyper_parallel.models.deepseek_v41.adapter.expert_parallel.deepseek_v41_ep_compute_fn",
            use_grouped_gemm=getattr(entry.local_compute_fn, "use_grouped_gemm", False),
            expert_capacity_factor=getattr(entry.local_compute_fn, "expert_capacity_factor", 2.0),
            megamoe=True, max_local_num_tokens=tokens,
        )
    config.plan_overrides.insert(0, PlanOverride(
        match="*.mlp.experts",
        module_type="transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts",
        replace_module=Target(
            DeepseekV41TrainingExperts,
            target_path="hyper_parallel.models.deepseek_v41.adapter.replacements.DeepseekV41TrainingExperts",
        ),
    ))


__all__ = ["configure_megamoe", "deepseek_v41_engram_compute_fn", "deepseek_v41_ep_compute_fn"]
