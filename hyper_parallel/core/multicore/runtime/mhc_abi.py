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
"""Family-local MHC task kinds; the current MoE scheduler enums remain independent."""

from enum import Enum, IntEnum

from hyper_parallel.core.multicore.runtime.abi import family_abi

MhcTaskType = IntEnum("MhcTaskType", {task.native_name: task.numeric_id for task in family_abi("mhc").task_types})
FAST_DEPENDENCY_POLL_INTERVAL_US = 5


class MhcOpType(Enum):
    """Legacy MHC graph categories, separate from global scheduler operation kinds."""

    MHC_POST = "mhc_post"
    MHC_NORM_CAST = "mhc_norm_cast"
    MHC_PROJECTION = "mhc_projection"
    MHC_MAPPING = "mhc_mapping"
    MHC_INPUT_MIX = "mhc_input_mix"
    MHC_PIPELINE = "mhc_pipeline"
    RMS_NORM = "rms_norm"
