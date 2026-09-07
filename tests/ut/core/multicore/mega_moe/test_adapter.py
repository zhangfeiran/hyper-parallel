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
"""CPU contracts for managed and legacy Python-to-native MegaMoe arguments."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.platform import mindspore as ms_adapter
from hyper_parallel.core.multicore.platform import torch as torch_adapter


class TestMegaMoeAdapter(unittest.TestCase):
    """Preserve the legacy entry point while passing every managed ABI field."""

    def test_managed_forward_passes_fusion_wave_shared_and_profile_fields(self):
        """New keyword inputs occupy their exact native forward ABI positions."""
        tensors = tuple(object() for _ in range(22))
        route = tuple(object() for _ in range(6))
        topology = (0, 8, 56, 5120, 4096)
        native = Mock()
        framework = SimpleNamespace(ops=SimpleNamespace(hyper_parallel=native))
        with patch.object(torch_adapter, "_get_torch", return_value=framework):
            torch_adapter.mega_moe(
                *tensors, *topology,
                original_x=route[0], permute_src_index=route[1], reduce_route_index=route[2],
                topk_weights=route[3], final_out=route[4], cycle_profile_workspace=route[5],
                receive_capacity=49152, wave_count=3, home_experts=6, cycle_profile=1,
            )
        native.mega_moe.assert_called_once_with(*tensors, *route, *topology, 49152, 3, 6, 1)

    def test_managed_backward_passes_fusion_wave_shared_and_profile_fields(self):
        """Backward keeps its distinct profile/home-experts attribute order."""
        tensors = tuple(object() for _ in range(29))
        route = tuple(object() for _ in range(3))
        topology = (0, 8, 56, 5120, 4096)
        native = Mock()
        framework = SimpleNamespace(ops=SimpleNamespace(hyper_parallel=native))
        with patch.object(torch_adapter, "_get_torch", return_value=framework):
            torch_adapter.mega_moe_grad(
                *tensors, *topology,
                permute_src_index=route[0], route_weight=route[1], reduce_route_index=route[2],
                receive_capacity=49152, wave_count=4, home_experts=6, cycle_profile=1,
            )
        native.mega_moe_grad.assert_called_once_with(*tensors, *route, *topology, 49152, 4, 1, 6)

    def test_torch_legacy_entry_points_keep_one_wave_and_typed_unused_inputs(self):
        """Unfused callers retain their positional API without interpreting IDs as weights."""
        topology = (0, 2, 4, 8, 128)
        for name, count, weight_index in (("mega_moe", 22, 25), ("mega_moe_grad", 29, 30)):
            with self.subTest(operator=name):
                tensors = [object() for _ in range(count)]
                tensors[0] = torch.empty((256, 8), dtype=torch.bfloat16, device="cpu")
                tensors[4] = torch.ones(4, dtype=torch.int32, device="cpu")
                native = Mock()
                framework = SimpleNamespace(float32=torch.float32, ops=SimpleNamespace(hyper_parallel=native))
                with patch.object(torch_adapter, "_get_torch", return_value=framework):
                    getattr(torch_adapter, name)(*tensors, *topology)
                arguments = getattr(native, name).call_args.args
                self.assertEqual(arguments[:count], tuple(tensors))
                self.assertEqual(arguments[weight_index].dtype, torch.float32)
                self.assertEqual(arguments[-9:], (*topology, 256, 1, 0, 0))

    def test_mindspore_legacy_entry_points_match_pynative_abi(self):
        """The retained PyNative interface supplies unfused placeholders and one-wave capacity."""
        topology = (0, 2, 4, 8, 128)
        for name, count, weight_index in (("mega_moe", 22, 25), ("mega_moe_grad", 29, 30)):
            with self.subTest(operator=name):
                tensors = [Mock() for _ in range(count)]
                tensors[0].shape = (256, 8)
                counts, converted = Mock(), object()
                counts.astype.return_value = converted
                tensors[4] = Mock()
                tensors[4].__getitem__ = Mock(return_value=counts)
                native = Mock()
                framework = SimpleNamespace(float32=object())
                with patch.dict(sys.modules, {"mindspore": framework}), \
                        patch.object(ms_adapter, "_check_pynative_mode") as check_mode, \
                        patch.object(ms_adapter, "_get_ms_ops", return_value=native):
                    getattr(ms_adapter, name)(*tensors, *topology)
                check_mode.assert_called_once_with(name)
                counts.astype.assert_called_once_with(framework.float32)
                arguments = getattr(native, name).call_args.args
                self.assertEqual(arguments[:count], tuple(tensors))
                self.assertIs(arguments[weight_index], converted)
                self.assertEqual(arguments[-7:], (*topology, 256, 1))
