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
"""Thin launchers for bounded expert replication on Ascend."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_WORKER = str(Path(__file__).with_name("_test_hot_replica.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_replica_count_payload() -> None:
    """
    Feature: Fixed-size route counting
    Description: Exercise empty and invalid routes and a count exceeding FP32's exact range.
    Expectation: Payloads retain exact int64 counts and collective error flags without readback.
    """
    torchrun_case(str(Path(__file__).with_name("_test_replica_boundaries.py")),
                  "test_replica_count_payload_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_native_hot_replica() -> None:
    """Check native training with bounded expert replicas."""
    torchrun_case(_WORKER, "test_native_hot_replica_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_hot_replica() -> None:
    """Check multicore push training and receive-buffer growth."""
    torchrun_case(_WORKER, "test_push_hot_replica_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_hot_replica() -> None:
    """Check multicore pull training with the shared placement planner."""
    torchrun_case(_WORKER, "test_pull_hot_replica_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_native_device_hot_replica() -> None:
    """Native AIV planner with P2P."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_native_device_hot_replica_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_default_group() -> None:
    """Push default-group P2P."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_push_default_group_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_default_group() -> None:
    """Pull default-group P2P."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_pull_default_group_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_replica_groups() -> None:
    """Default and noncontiguous group round trips."""
    torchrun_case(str(Path(__file__).with_name("_test_replica_boundaries.py")),
                  "test_replica_groups_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_device_replica_boundaries() -> None:
    """AIV parity and retained cross-stream plans."""
    torchrun_case(str(Path(__file__).with_name("_test_replica_boundaries.py")),
                  "test_device_replica_boundaries_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_device_projection() -> None:
    """push AIV planner and projection."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_push_device_projection_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_device_kernel_gradient() -> None:
    """push AIV planner and kernel_gradient."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_push_device_kernel_gradient_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_device_projection() -> None:
    """pull AIV planner and projection."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_pull_device_projection_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_device_kernel_gradient() -> None:
    """pull AIV planner and kernel_gradient."""
    torchrun_case(str(Path(__file__).with_name("_test_hot_replica.py")),
                  "test_pull_device_kernel_gradient_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_native_deferred_topk() -> None:
    """Check legal routes and reverse backward across K=1/2/3/4/5/6/8."""
    torchrun_case(_WORKER, "test_native_deferred_topk_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_static_runtime_streams() -> None:
    """
    Feature: Push static invocation reuse
    Description: Run two streams with retained plans and reversed backward.
    Expectation: Match FP32 reference and reuse one image per direction.
    """
    torchrun_case(_WORKER, "test_push_static_runtime_streams_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_static_runtime_streams() -> None:
    """
    Feature: Pull static invocation reuse
    Description: Run two streams with retained plans and reversed backward.
    Expectation: Match FP32 reference and reuse one image per direction.
    """
    torchrun_case(_WORKER, "test_pull_static_runtime_streams_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_device_route_validation() -> None:
    """
    Feature: Collective route validation
    Description: Inject one rank's invalid or duplicate IDs with CPU and device planners.
    Expectation: All ranks reject bad routes and recover for valid routes using None and WORLD groups.
    """
    torchrun_case(str(Path(__file__).with_name("_test_replica_boundaries.py")),
                  "test_device_route_validation_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_push_projection_runtime_lifetime() -> None:
    """
    Feature: Projection invocation lifetime
    Description: Retain v4 images on two streams with reversed backward.
    Expectation: Each launch observes its own epoch and matches the FP32 native reference.
    """
    torchrun_case(_WORKER, "test_push_projection_runtime_lifetime_npu", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_pull_projection_runtime_lifetime() -> None:
    """
    Feature: Projection invocation lifetime
    Description: Retain v4 images on two streams with reversed backward.
    Expectation: Each launch observes its own epoch and matches the FP32 native reference.
    """
    torchrun_case(_WORKER, "test_pull_projection_runtime_lifetime_npu", num_proc=4)
