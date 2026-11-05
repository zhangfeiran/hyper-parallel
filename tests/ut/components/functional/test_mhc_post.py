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
"""Check MHC post device dispatch and the four-input autograd contract on CPU."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


mhc = importlib.import_module("hyper_parallel.components.functional.mhc_post")


class TestMhcPost(unittest.TestCase):
    """An installed legacy Python wrapper must not select missing A5 ACLNN symbols."""

    def tearDown(self) -> None:
        """Do not retain device selections between hardware-independent cases."""
        if hasattr(mhc, "_uses_native_mhc"):
            mhc._uses_native_mhc.cache_clear()

    def test_a5_names_select_native_and_cache_device_query(self) -> None:
        """Both released A5 device names use built-in PTA operators."""
        for name in ("Ascend950DT_9582", "Ascend910_950", "Ascend910_9391"):
            with self.subTest(name=name):
                device = Mock(type="npu")
                backend = SimpleNamespace(get_device_name=Mock(return_value=name))
                with patch.object(torch, "npu", backend, create=True):
                    expected = "950" in name
                    self.assertEqual(mhc._uses_native_mhc(device), expected)
                    self.assertEqual(mhc._uses_native_mhc(device), expected)
                backend.get_device_name.assert_called_once_with(device)

    def test_cpu_device_does_not_query_accelerator(self) -> None:
        """CPU use remains independent of torch_npu and device initialization."""
        backend = SimpleNamespace(get_device_name=Mock(side_effect=AssertionError("Unexpected device query")))
        with patch.object(torch, "npu", backend, create=True):
            self.assertFalse(mhc._uses_native_mhc(torch.device("cpu")))
        backend.get_device_name.assert_not_called()

    @staticmethod
    def _operands() -> tuple[torch.Tensor, ...]:
        """Create actual mixed-dtype input roles without allocating an accelerator."""
        return tuple(torch.randn(shape, dtype=dtype, requires_grad=True) for shape, dtype in (
            ((2, 3, 4, 5), torch.bfloat16), ((2, 3, 4, 4), torch.float32),
            ((2, 3, 5), torch.bfloat16), ((2, 3, 4), torch.float32)))

    def _check_bridge(self, native: bool) -> None:
        """Check provider identity, contiguous upstream gradients and all gradient positions."""
        values = self._operands()
        forward = Mock(return_value=torch.zeros_like(values[0]))
        backward = Mock(return_value=tuple(torch.full_like(value, index + 1) for index, value in enumerate(values)))
        builtin = SimpleNamespace(npu_mhc_post=forward, npu_mhc_post_backward=backward)
        legacy = SimpleNamespace(npu_ai_infra_manifold_constrained_hyper_connection_post=forward,
                                 npu_ai_infra_mhc_post_grad=backward)
        other_forward = Mock(side_effect=AssertionError("Wrong MHC provider"))
        if native:
            legacy.npu_ai_infra_manifold_constrained_hyper_connection_post = other_forward
        else:
            builtin.npu_mhc_post = other_forward
        with (patch.object(mhc, "_uses_native_mhc", return_value=native, create=True),
              patch.object(mhc, "omni_training_custom_ops", None if native else object()),
              patch.object(torch.ops, "npu", builtin), patch.object(torch.ops, "custom", legacy)):
            output = mhc._MhcPost.apply(*values)
            upstream = torch.ones(2, 3, 5, 4, dtype=output.dtype).transpose(-1, -2)
            with patch.object(mhc, "_uses_native_mhc", side_effect=AssertionError("Provider changed in backward")):
                gradients = torch.autograd.grad(output, values, upstream)
        forward.assert_called_once()
        for observed, expected in zip(forward.call_args.args, values):
            self.assertIs(observed, expected)
        backward.assert_called_once()
        self.assertTrue(backward.call_args.args[0].is_contiguous())
        for observed, expected in zip(backward.call_args.args[1:], values):
            self.assertIs(observed, expected)
        for index, gradient in enumerate(gradients):
            torch.testing.assert_close(gradient, torch.full_like(values[index], index + 1), rtol=0, atol=0)
        other_forward.assert_not_called()

    def test_native_bridge_does_not_require_legacy_package(self) -> None:
        """A5 forward and backward use matching built-ins even without Omni registration."""
        self._check_bridge(native=True)

    def test_legacy_bridge_preserves_a3_operator_order(self) -> None:
        """A3 keeps the original custom operators and gradient return order."""
        self._check_bridge(native=False)

    def test_cpu_public_path_preserves_values_and_gradients(self) -> None:
        """The portable path preserves its formula and does not query device identity."""
        residual, combine, output, post = self._operands()
        with patch.object(mhc, "_uses_native_mhc", side_effect=AssertionError("Unexpected device query"), create=True):
            actual = mhc.mhc_post(output, residual.flatten(2), post, combine, 4)
        reference = (torch.einsum("bsij,bsih->bsjh", combine, residual.float())
                     + post.unsqueeze(-1) * output.float().unsqueeze(-2)).flatten(2).to(output.dtype)
        torch.testing.assert_close(actual, reference)
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (residual, combine, output, post), upstream, retain_graph=True)
        expected_gradients = torch.autograd.grad(reference, (residual, combine, output, post), upstream)
        for left, right in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(left, right)
