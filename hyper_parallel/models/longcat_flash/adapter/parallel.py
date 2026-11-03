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
"""LongCat parameter roles and explicit EP binding for real and identity experts."""

from typing import Any, Callable

from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Declare MLA latent projections and the router classifier as replicated."""
    return [(["q_a_proj", "kv_a_proj_with_mqa", "linear_qkv", "router.classifier"], ParamRole.REPLICATED),
            (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
            (["linear_fc1", "linear_fc2"], ParamRole.REPLICATED)]


@local_compute
def longcat_flash_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any,
                               cp_mesh: Any, ep_mesh: Any) -> Callable:
    """Bind a replaced expert holder while leaving the authoritative router intact."""
    del mesh
    if cp_mesh is not None and cp_mesh.size() > 1:
        raise ValueError("LongCat Flash EP currently requires CP=1")
    if not isinstance(module.experts, LongcatFlashMegaMoeExperts):
        raise TypeError("LongCat EP requires the explicit expert replacement")
    ep_size = 1 if ep_mesh is None else ep_mesh["ep"].size()
    if tp_mesh is not None and tp_mesh.size() > 1 and ep_size == 1:
        raise ValueError("LongCat expert tensor parallelism requires EP>1")
    ep_group = None if ep_mesh is None else ep_mesh.get_group("ep")
    module.experts.configure(ep_group, ep_size, module.router.top_k)

    def compute_fn(module: Any, hidden_states: Any) -> Any:
        """Execute the original router and the replaced expert module.

        Args:
            module: Source module supplied by the replacement executor.
            hidden_states: Token states in original source order.
        """
        shape = hidden_states.shape
        routing = module.router(hidden_states)
        if not isinstance(routing, tuple) or len(routing) not in (2, 3):
            raise TypeError("LongCat router must return weights/IDs, with optional leading logits")
        weights, ids = routing[-2:]
        return module.experts(hidden_states.reshape(-1, shape[-1]), ids, weights).reshape(shape)

    return compute_fn
