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
"""Unit tests for hyper_parallel.core.dtensor.init_weights."""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch
from torch.utils import deterministic

from hyper_parallel.core.dtensor import init_weights as init_weights_module
from hyper_parallel.core.dtensor.init_weights import init_empty_weights, init_on_device

_INIT_ON_DEVICE_TARGET = "hyper_parallel.core.dtensor.init_weights._init_on_device"


@contextmanager
def _fake_init_on_device(device, include_buffers=False):
    """Stand-in for the Torch ``init_on_device`` context manager."""
    del device, include_buffers
    yield


class TestInitEmptyWeights(unittest.TestCase):
    """Tests for InitEmptyWeights."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default_include_buffers_false(self):
        """Test default include buffers false."""
        with init_empty_weights():
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        with init_empty_weights(include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=True
        )


class TestInitOnDevice(unittest.TestCase):
    """Tests for InitOnDevice."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default(self):
        """Test default."""
        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=True
        )


class TestDeterministicMetaInitialization(unittest.TestCase):
    """Meta construction must not fill temporary full-size CPU weights."""

    def setUp(self) -> None:
        """Restore process-wide determinism controls after every test."""
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.addCleanup(setattr, deterministic, "fill_uninitialized_memory", deterministic.fill_uninitialized_memory)
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic.fill_uninitialized_memory = True

    def test_parameter_factories_do_not_fill_temporary_cpu_storage(self) -> None:
        """Inspect factories before parameter registration can hide CPU storage."""
        original_empty = torch.empty
        fill_requests = []

        def _tracked_empty(*args, **kwargs) -> torch.Tensor:
            fill_requests.append(
                torch.are_deterministic_algorithms_enabled() and deterministic.fill_uninitialized_memory
            )
            return original_empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=_tracked_empty), init_empty_weights():
            model = torch.nn.Embedding(16, 8)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertTrue(fill_requests)
        self.assertFalse(any(fill_requests))
        self.assertTrue(model.weight.is_meta)
        self.assertTrue(deterministic.fill_uninitialized_memory)
        self.assertTrue(torch.empty(8).isnan().all().item())

    def test_nested_meta_context_restores_outer_policy(self) -> None:
        """Nested exits must not re-enable filling inside the outer context."""
        for original_fill in (False, True):
            with self.subTest(original_fill=original_fill):
                deterministic.fill_uninitialized_memory = original_fill
                with init_empty_weights():
                    with init_on_device(torch.device("meta"), include_buffers=True):
                        model = torch.nn.Linear(4, 4)
                    self.assertFalse(deterministic.fill_uninitialized_memory)
                self.assertTrue(model.weight.is_meta)
                self.assertEqual(deterministic.fill_uninitialized_memory, original_fill)

    def test_exception_restores_fill_and_registration(self) -> None:
        """A failed constructor must not leak altered globals to later work."""
        register_parameter = torch.nn.Module.register_parameter
        register_buffer = torch.nn.Module.register_buffer
        with self.assertRaisesRegex(RuntimeError, "constructor failed"):
            with init_empty_weights(include_buffers=True):
                raise RuntimeError("constructor failed")
        self.assertTrue(deterministic.fill_uninitialized_memory)
        self.assertIs(torch.nn.Module.register_parameter, register_parameter)
        self.assertIs(torch.nn.Module.register_buffer, register_buffer)
        self.assertTrue(torch.empty(8).isnan().all().item())

    def test_cpu_initialization_keeps_deterministic_fill(self) -> None:
        """Real-device initialization must retain the caller's fill policy."""
        with init_on_device(torch.device("cpu")):
            self.assertTrue(deterministic.fill_uninitialized_memory)
            self.assertTrue(torch.empty(8).isnan().all().item())

    def test_meta_initialization_preserves_cpu_metadata(self) -> None:
        """Hash-style CPU buffers must remain readable during construction."""
        with init_empty_weights():
            model = torch.nn.Linear(4, 4)
            model.register_buffer("offsets", torch.tensor([3, 5]).cumsum(0))
            total = model.offsets[-1].item()
        self.assertTrue(model.weight.is_meta)
        self.assertEqual(model.offsets.device.type, "cpu")
        self.assertEqual(total, 8)


if __name__ == "__main__":
    unittest.main()
