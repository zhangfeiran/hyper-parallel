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
"""Fixed-capacity MegaMoe adapter for the shared consumer registry."""

from collections.abc import Mapping
from typing import Any

import torch

from hyper_parallel.core.expert_parallel.hot_replica.capacity import ExpertReplicaConfig
from hyper_parallel.core.multicore.shmem.consumer import (
    SharedShmemRoot,
    ShmemConsumer,
    ShmemConsumerSpec,
)

from .heap_manager import root_members
from .spec import _align_capacity, initial_receive_capacity
from .workspace import _spec_workspace_bytes


def reserve_mega_moe_consumer(
    root: SharedShmemRoot, name: str, specification: Mapping[str, Any], dtype: torch.dtype,
) -> ShmemConsumer:
    """Reserve legacy MoE layout bytes after proving that capacity cannot require rebuilding."""
    if root.members != root_members(specification.get("ep_group")) or len(root.members) != specification["ep_size"]:
        raise ValueError("shared MoE EP must have the same ordered membership as the SHMEM root")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("shared MoE resources require FP16 or BF16")
    rows = specification["local_num_tokens"] * specification["top_k"]
    maximum = _align_capacity(rows * specification["ep_size"])
    if specification.get("replica_slots_per_rank", 0):
        maximum = ExpertReplicaConfig(specification["logical_num_experts"], specification["ep_size"],
                                      specification["replica_slots_per_rank"]).maximum_receive_rows(
                                          specification["local_num_tokens"], specification["top_k"])
    if specification["dispatch_mode"] == "push" and initial_receive_capacity(specification) < maximum:
        raise ValueError("shared MoE push requires the full lossless receive capacity before binding; "
                         "use pull or preallocate")
    layout = (str(dtype), tuple(sorted((key, value) for key, value in specification.items() if key != "ep_group")))
    element_size = torch.empty((), dtype=dtype).element_size()
    # The legacy layout includes alignment slack; add vendor rounding for every allocation.
    count = 5 if specification.get("replica_slots_per_rank", 0) and specification["replica_transport"] != "p2p" else 4
    budget = _spec_workspace_bytes(specification, element_size) + count * 15
    return root.reserve(ShmemConsumerSpec(name, "mega_moe", budget, layout))


class FixedMegaMoeHeap:
    """Preserve MoE workspace allocation while delegating ownership to the frozen root."""

    def __init__(self, consumer: ShmemConsumer) -> None:
        """Bind the shared root consumer without acquiring a second native reference."""
        self.consumer = consumer
        self.consumer.bind()

    def access(self) -> Any:
        """Exclude another consumer during MoE setup or an active execution lease."""
        return self.consumer.access()

    @staticmethod
    def ensure_capacity(resource: Any, maximum_received_slots: int) -> None:
        """Reject unexpected capacity overflow without any heap teardown or address changes."""
        if maximum_received_slots > resource.workspace.capacity_floor:
            raise RuntimeError("shared MoE receive capacity is fixed; cross-consumer heap rebuild is disabled")
