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
"""原始 NK 稠密 FFN 的双卡 FSDP 验收启动器。"""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dense_source_packed_fully_shard_adamw() -> None:
    """Feature: NK packed FFN 的 FSDP 接入。
    Description: 比较完整设备反向、FP32 主参数 AdamW、重计算和磁盘恢复后的独立轨迹。
    Expectation: 与原始 packed 一致，全部输入执行设备反向，恢复状态逐值一致。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_fsdp_adamw_impl.py")),
                  case_name="test_dense_source_packed_fully_shard_adamw", master_port=29625, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dense_source_common_fully_shard_adamw() -> None:
    """Feature: NK common FFN 的 FSDP 接入。
    Description: 比较三个独立投影的设备反向、FP32 主参数 AdamW 与检查点恢复后的独立轨迹。
    Expectation: 与原始 common 一致，全部四输入执行设备反向，恢复状态逐值一致。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_fsdp_adamw_impl.py")),
                  case_name="test_dense_source_common_fully_shard_adamw", master_port=29626, num_proc=2)
