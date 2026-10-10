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
"""不导入框架的 Ascend 稠密 FFN 训练验证启动器。"""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_ffn_training() -> None:
    """Feature: 原生稠密 FFN 训练。
    Description: 在单卡 Ascend NPU 上比较输出、梯度和 checkpoint/close 生命周期。
    Expectation: 所有组件比较及关闭后待执行反向均通过。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_impl.py")),
                  case_name="test_dense_ffn_training", master_port=29617, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_ffn_saved_buffers() -> None:
    """Feature: resident 保存缓存。
    Description: 验证重叠调用、真实地址复用、跨流、retain_graph、保存钩子和 CPU 卸载。
    Expectation: 活跃存储别名阻止复用；关闭后待执行反向仍然正确。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_saved_impl.py")),
                  case_name="test_dense_ffn_saved_buffers", master_port=29618, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_backward_components() -> None:
    """Feature: 原生反向 worker。
    Description: 核对完整 K 权重梯度、SDK SwiGLUGrad、梯度累加和跨流队列。
    Expectation: 与独立原生 autograd 一致，完整 FFN 的全部输入梯度使用设备反向。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_backward_impl.py")),
                  case_name="test_dense_backward_components", master_port=29620, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_backward_autograd() -> None:
    """Feature: 常驻反向 autograd。
    Description: 验证真实梯度需求、保存值、retain_graph、保存钩子、跨流和关闭后反向。
    Expectation: 已准入的完整梯度图执行设备反向，通用图的未绑定能力保留原有完整 VJP。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_backward_autograd_impl.py")),
                  case_name="test_dense_backward_autograd", master_port=29621, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_linear_ffn_training() -> None:
    """Feature: 原始 Linear 数值契约。
    Description: 核对 NK 权重的配套常驻前后向、尾部、重计算和关闭后反向。
    Expectation: 与原始打包 FFN 逐值一致，完整设备反向没有主机补算。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_linear_ffn_impl.py")),
                  case_name="test_dense_linear_ffn_training", master_port=29623, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_dense_common_ffn_training() -> None:
    """Feature: 原始 common 数值契约。
    Description: 核对三个独立 NK 投影、BF16 SiLU 和乘法，覆盖可选梯度及重计算。
    Expectation: 与原始 common FFN 逐值一致，完整设备反向没有主机补算。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_common_ffn_impl.py")),
                  case_name="test_dense_common_ffn_training", master_port=29624, num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dense_ffn_fully_shard() -> None:
    """Feature: 稠密 FFN FSDP 接入。
    Description: 在双卡上验证梯度、分片权重更新、重计算和检查点恢复。
    Expectation: 独立 SGD 组件在反复 unshard/reshard 后保持一致。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_fsdp_impl.py")),
                  case_name="test_dense_ffn_fully_shard", master_port=29619, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dense_ffn_fully_shard_adamw() -> None:
    """Feature: FSDP FP32 主参数 AdamW。
    Description: 比较输出、梯度、BF16 参数、FP32 主参数、优化器矩及磁盘恢复后的独立训练。
    Expectation: 完整设备反向无主机补算，恢复轨迹与未中断轨迹逐值一致。
    """
    torchrun_case(file_name=str(Path(__file__).with_name("_dense_ffn_fsdp_adamw_impl.py")),
                  case_name="test_dense_ffn_fully_shard_adamw", master_port=29622, num_proc=2)
