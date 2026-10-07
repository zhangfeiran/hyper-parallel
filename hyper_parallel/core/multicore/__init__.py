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
# pylint: disable=undefined-all-variable
"""Torch-only Multicore APIs, separate from the HyperParallel root exports."""

__all__ = ["MegaMoeExperts", "profiler"]

from importlib import import_module


def __getattr__(name: str) -> object:
    """Load NPU business modules when requested, allowing CPU frontend imports."""
    if name == "MegaMoeExperts":
        module = import_module("hyper_parallel.core.multicore.modules.mega_moe.module")
        value = module.MegaMoeExperts
        globals()[name] = value
        return value
    if name == "profiler":
        value = import_module("hyper_parallel.core.multicore.profiler")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
