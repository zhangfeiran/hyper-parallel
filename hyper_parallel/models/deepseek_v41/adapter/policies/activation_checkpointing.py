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
"""Declare the DeepSeek-V4.1 recompute regions used by normal training."""

from functools import partial
from typing import Any

from hyper_parallel.models.adapter_spec import RecomputePolicy
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import (
    attention_role,
    configure_shared_state,
)

#: The CSA2 attention roles, producers first.
_ATTENTION_ROLES = ("full", "reindex", "reuse")

#: The attention roles that recompute with their layer, per tier.
_RECOMPUTED_ROLES = {
    "none": (),
    "reuse": ("reuse",),
    "reuse_reindex": ("reuse", "reindex"),
    "all": _ATTENTION_ROLES,
}

#: Recompute every attention role; discrete source selections remain pinned.
ATTENTION_RECOMPUTE_TIER = "all"


def _attention_stays_resident(module_fqn: str, module: Any, *, resident: tuple[str, ...]) -> bool:
    """Keep the attention of a role that does not recompute out of the region."""
    del module_fqn
    return attention_role(module) in resident


def build_recompute_policy() -> RecomputePolicy:
    """Return the V4.1 recompute regions used by normal training.

    Every decoder layer is one region. The attention roles listed for
    :data:`ATTENTION_RECOMPUTE_TIER` recompute with their layer; the remaining
    roles stay outside the region, which is what the tiers differ in. The
    model-owned hook then pins the discrete selection of the recomputed index
    sources, so recomputing one neither chooses a different key set nor
    publishes its state a second time; published cross-layer entries stay alive
    until the graph drops the state in backward.
    """
    recomputed = _RECOMPUTED_ROLES[ATTENTION_RECOMPUTE_TIER]
    resident = tuple(role for role in _ATTENTION_ROLES if role not in recomputed)
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        exclude_predicate=(
            partial(_attention_stays_resident, resident=resident) if resident else None
        ),
        on_applied=partial(configure_shared_state, roles=recomputed),
    )


__all__ = ["build_recompute_policy"]
