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
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights():
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

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
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=True
        )


class TestMetaDeterministicFill(unittest.TestCase):
    """Check temporary storage policy under deterministic model construction."""

    def setUp(self) -> None:
        """Restore all process-wide deterministic settings after each test."""
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.addCleanup(setattr, deterministic, "fill_uninitialized_memory", deterministic.fill_uninitialized_memory)
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic.fill_uninitialized_memory = True

    def test_factory_before_parameter_registration(self) -> None:
        """Avoid touching temporary CPU storage while retaining deterministic algorithms."""
        empty = torch.empty
        observed = []

        def _record_factory(*args, **kwargs) -> torch.Tensor:
            observed.append(deterministic.fill_uninitialized_memory)
            return empty(*args, **kwargs)

        with patch("torch.empty", side_effect=_record_factory), init_weights_module.init_empty_weights():
            module = torch.nn.Embedding(32, 8)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())

        self.assertEqual(observed, [False])
        self.assertEqual(module.weight.device.type, "meta")
        self.assertTrue(deterministic.fill_uninitialized_memory)

    def test_exception_restores_global_state(self) -> None:
        """Restore fill policy and registration functions when construction fails."""
        register_parameter = torch.nn.Module.register_parameter
        register_buffer = torch.nn.Module.register_buffer
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            with init_weights_module.init_empty_weights(include_buffers=True):
                self.assertFalse(deterministic.fill_uninitialized_memory)
                raise RuntimeError("construction failed")

        self.assertTrue(deterministic.fill_uninitialized_memory)
        self.assertIs(torch.nn.Module.register_parameter, register_parameter)
        self.assertIs(torch.nn.Module.register_buffer, register_buffer)

    def test_nested_meta_contexts(self) -> None:
        """Keep filling disabled until the outermost meta context exits."""
        with init_weights_module.init_empty_weights():
            self.assertFalse(deterministic.fill_uninitialized_memory)
            with init_weights_module.init_empty_weights():
                self.assertFalse(deterministic.fill_uninitialized_memory)
            self.assertFalse(deterministic.fill_uninitialized_memory)
        self.assertTrue(deterministic.fill_uninitialized_memory)

    def test_original_disabled_fill_is_preserved(self) -> None:
        """Preserve callers that already disabled deterministic filling."""
        deterministic.fill_uninitialized_memory = False
        with init_weights_module.init_empty_weights():
            self.assertFalse(deterministic.fill_uninitialized_memory)
        self.assertFalse(deterministic.fill_uninitialized_memory)

    def test_cpu_context_preserves_fill(self) -> None:
        """Leave deterministic CPU allocation policy unchanged."""
        with init_weights_module.init_on_device(torch.device("cpu")):
            tensor = torch.empty(8)
            self.assertTrue(deterministic.fill_uninitialized_memory)
            self.assertTrue(torch.isnan(tensor).all().item())
        self.assertTrue(deterministic.fill_uninitialized_memory)


if __name__ == "__main__":
    unittest.main()
