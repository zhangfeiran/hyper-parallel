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
"""The independent EP benchmark updates actual local leaves with ordinary AdamW."""

import unittest
from unittest.mock import patch

import torch

from examples.torch.multicore.benchmark_mega_moe_layers import (
    ADAMW_OPTIONS,
    _make_optimizer,
)
from hyper_parallel.core.dtensor import device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout


class TestLocalOptimizer(unittest.TestCase):
    """Optimizer state, gradients and parameter updates stay on the owned EP shard."""

    def test_dtensor_leaf_updates_match_plain_adamw_over_multiple_steps(self) -> None:
        """Use real autograd and AdamW while mocking only the distributed mesh rank."""
        local = torch.nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        reference = torch.nn.Parameter(torch.clone(torch.detach(local)))
        with (
            patch.object(device_mesh.platform, "get_rank", return_value=0),
            patch.dict(device_mesh._DEVICE_MESH_MAP, {}, clear=True),
            patch.dict(device_mesh.EXISTING_COMM_GROUPS, {}, clear=True),
        ):
            mesh = Layout((2,), ("ep",), init_backend=False).mesh
            distributed = DTensor.from_local(local, mesh, ("ep", "None"))
            actual_optimizer = _make_optimizer(torch, [distributed])
            reference_optimizer = torch.optim.AdamW([reference], **ADAMW_OPTIONS)
            self.assertIs(actual_optimizer.param_groups[0]["params"][0], local)
            for step in range(3):
                actual_optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                self.assertIsNone(distributed.grad)
                ((local + step).square().sum()).backward()
                ((reference + step).square().sum()).backward()
                self.assertIs(distributed.grad, local.grad)
                actual_optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(distributed.to_local(), reference, rtol=0, atol=0)
                for name, value in actual_optimizer.state[local].items():
                    torch.testing.assert_close(value, reference_optimizer.state[reference][name], rtol=0, atol=0)
