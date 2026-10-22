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
"""Torch-only Multicore APIs, separate from the HyperParallel root exports."""

from typing import TYPE_CHECKING, Any

from hyper_parallel.core.multicore import profiler

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.modules.routed_ffn import routed_ffn
    from hyper_parallel.core.multicore.modules.mega_gate.module import MegaGate
    from hyper_parallel.core.multicore.modules.mega_mhc.module import HyperMegaMhc
    from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts

__all__ = ["HyperMegaMhc", "MegaGate", "MegaMoeExperts", "profiler", "routed_ffn"]


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
    """Load optional multicore modules only when their public API is requested."""
    if name == "routed_ffn":
        from hyper_parallel.core.multicore.modules.routed_ffn import (  # pylint: disable=import-outside-toplevel
            routed_ffn as local_ffn,
        )

        globals()[name] = local_ffn
        return local_ffn
    if name == "HyperMegaMhc":
        from hyper_parallel.core.multicore.modules.mega_mhc.module import (  # pylint: disable=import-outside-toplevel
            HyperMegaMhc as hyper_mega_mhc,
        )

        globals()[name] = hyper_mega_mhc
        return hyper_mega_mhc
    if name == "MegaGate":
        from hyper_parallel.core.multicore.modules.mega_gate.module import (  # pylint: disable=import-outside-toplevel
            MegaGate as mega_gate,
        )

        globals()[name] = mega_gate
        return mega_gate
    if name == "MegaMoeExperts":
        from hyper_parallel.core.multicore.modules.mega_moe.module import (  # pylint: disable=import-outside-toplevel
            MegaMoeExperts as mega_moe_experts,
        )

        globals()[name] = mega_moe_experts
        return mega_moe_experts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
