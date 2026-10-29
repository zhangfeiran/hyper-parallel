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

"""Unit tests for native Torch adapter registration diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


from tests.common.mark_utils import arg_mark

from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.torch import ops
from tests.common.mark_utils import arg_mark


# Registration-cache and alias-ABI checks require Torch's private operator schema and loader.
class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Isolate backend registration from the native adapter schema checks."""
        patcher = patch.object(ops, "load_torch_npu")
        self.backend = patcher.start()
        self.addCleanup(patcher.stop)
    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_route_diagnostics_bind_logical_ports_once(self):
        """Feature: MegaMoe route diagnostics.
        Description: Profile forward and backward through their logical port tuples.
        Expectation: Diagnostic binding preserves counts before native operand packing.
        """
        for forward, count in ((True, 27), (False, 33)):
            with self.subTest(forward=forward):
                parameters = tuple(ops.torch.ones(2, dtype=ops.torch.bfloat16) for _ in range(count))
                call = SimpleNamespace(runtime_config=parameters[0], event_counters=parameters[1],
                                       diagnostic_buffer=parameters[2])
                context = MagicMock()
                context.__enter__.return_value = call
                diagnostics = SimpleNamespace(is_active=True, launch=Mock(return_value=context))
                name = "mega_moe" if forward else "mega_moe_grad"
                with (patch.object(ops, "_load_native"),
                      patch.object(ops.torch.ops.hyper_parallel, name, create=True) as native):
                    ops.launch_route_fused(parameters, (0, 2, 8, 512, 128), diagnostics, forward=forward)
                self.assertIs(diagnostics.launch.call_args.args[0], parameters)
                self.assertEqual(len(native.call_args.args), (27 if forward else 31) + 6)
                self.assertIs(native.call_args.args[-6], call.diagnostic_buffer)

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        ops._load_native.cache_clear()

    @staticmethod
    def _backward_op(alias: str = "a", route_outputs: bool = True) -> SimpleNamespace:
        """Build a real schema for the mutable receive and expert input-gradient slots."""
        middle = ", ".join(f"Tensor input_{index}" for index in range(1, 12))
        tail = ", ".join(f"Tensor input_{index}" for index in range(13, 25))
        combine = "Tensor(j!) combine" if route_outputs else "Tensor combine"
        weights = "Tensor(k!) weights" if route_outputs else "Tensor weights"
        schema = ops.torch._C.parse_schema(
            f"mega_moe_grad(Tensor(a!) dispatch, {middle}, Tensor({alias}!) gate_dx, {tail}, "
            f"{combine}, Tensor runtime, Tensor events, Tensor indices, {weights}) -> ()"
        )
        return SimpleNamespace(default=SimpleNamespace(_schema=schema))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adapter_failure_reports_original_cause(self):
        """Feature: Native adapter load diagnostics.
        Description: Register the backend and fail the actual adapter library load.
        Expectation: Preserve the failed library and original ABI error.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_successful_registration_is_cached(self):
        """Feature: Cached native registration.
        Description: Load the backend and adapter twice, then reject mismatched native schemas.
        Expectation: Register once per cache lifetime and retain transport and alias validation.
        """
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_transport_version",
                         return_value=5, create=True) as version,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_grad", self._backward_op(), create=True) as backward,
        ):
            ops._load_native()
            ops._load_native()
            self.backend.assert_called_once_with()
            preload.assert_called_once_with(Path("vendor"))
            load.assert_called_once_with("good.so")
            for protocol, alias, mutable, message in (
                    (4, "a", True, "transport ABI"), (5, "e", True, "storage reuse"),
                    (5, "a", False, "route output mutation")):
                ops._load_native.cache_clear()
                version.return_value = protocol
                backward.default._schema = self._backward_op(alias, mutable).default._schema
                with self.assertRaisesRegex(NativeComponentUnavailableError, message):
                    ops._load_native()
