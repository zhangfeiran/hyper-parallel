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

import ast
from collections.abc import Callable
import inspect
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch


from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.torch import ops


# Registration-cache and alias-ABI checks require Torch's private operator schema and loader.
class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    # pylint: disable=protected-access

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        ops._load_native.cache_clear()

    @staticmethod
    def _backward_op(alias: str = "a") -> SimpleNamespace:
        """Build a real schema for the mutable receive and expert input-gradient slots."""
        middle = ", ".join(f"Tensor input_{index}" for index in range(1, 12))
        schema = ops.torch._C.parse_schema(
            f"mega_moe_grad(Tensor(a!) dispatch, {middle}, Tensor({alias}!) gate_dx) -> ()"
        )
        return SimpleNamespace(default=SimpleNamespace(_schema=schema))

    def test_adapter_failure_reports_original_cause(self):
        """Preserve the failed library and ABI error in the native diagnostic."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    def test_successful_registration_is_cached(self) -> None:
        """Load the adapter only once across repeated operation calls."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_transport_version",
                         return_value=7, create=True) as version,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_shared_version",
                         return_value=4, create=True),
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_grad", self._backward_op(), create=True) as backward,
        ):
            ops._load_native()
            ops._load_native()
            preload.assert_called_once_with(Path("vendor"))
            load.assert_called_once_with("good.so")
            for protocol, alias, message in ((0, "a", "transport ABI"), (7, "e", "storage reuse")):
                ops._load_native.cache_clear()
                version.return_value = protocol
                backward.default._schema = self._backward_op(alias).default._schema
                with self.assertRaisesRegex(NativeComponentUnavailableError, message):
                    ops._load_native()

    def test_shared_wrappers_match_registered_native_schemas(self) -> None:
        """Exercise normal and diagnostic calls through Torch's actual schema dispatcher."""
        registration = Path(ops.__file__).parent / "csrc/registration.cpp"
        source = registration.read_text()
        library = ops.torch.library.Library("hp_shared_abi_test", "DEF")
        captures = {}
        for name, indices in (("mega_moe", (0, 7, 8, 11, 12)),
                              ("mega_moe_grad", (0, 6, 8, 10, 12, 13, 17, 18))):
            start = source.index(f'"{name}("')
            declaration = source[start:source.index('");', start) + 1]
            schema = "".join(ast.literal_eval(token) for token in re.findall(r'"(?:[^"\\]|\\.)*"', declaration))
            library.define(schema)
            library.impl(name, self._capture_native(captures, name, indices), "CPU")
        proxy = SimpleNamespace(mega_moe=ops.torch.ops.hp_shared_abi_test.mega_moe,
                                mega_moe_grad=ops.torch.ops.hp_shared_abi_test.mega_moe_grad)
        attributes = {"rank_id", "ep", "expert_num", "hidden_size", "seq_size"}
        with patch.object(ops, "_load_native"), patch.object(ops.torch.ops, "hyper_parallel", proxy):
            for name in ("mega_moe", "mega_moe_grad", "mega_moe_with_diagnostic_buffer",
                         "mega_moe_grad_with_diagnostic_buffer"):
                for shared in (False, True):
                    with self.subTest(wrapper=name, shared=shared):
                        function = getattr(ops, name)
                        arguments = {key: 1 if key in attributes else ops.torch.empty((2, 2))
                                     for key in inspect.signature(function).parameters}
                        if not shared:
                            arguments.update(shared_ptrs=None, shared_runtime=None)
                        function(**arguments)
                        actual = captures["mega_moe_grad" if "grad" in name else "mega_moe"]
                        for index, key in ((-7, "shared_ptrs"), (-6, "shared_runtime"),
                                           (-8, "diagnostic_buffer")):
                            expected = arguments.get(key)
                            if expected is None:
                                expected = arguments["all_event_counters"]
                            self.assertEqual(actual[index].data_ptr(), expected.data_ptr())

    @staticmethod
    def _capture_native(captures: dict, name: str, indices: tuple[int, ...]) -> Callable[..., tuple]:
        """Return a CPU implementation behind the independently parsed native schema."""
        def capture(*arguments: object) -> tuple:
            """Capture dispatcher-validated arguments and return declared output aliases."""
            captures[name] = arguments
            return tuple(arguments[index] for index in indices)
        return capture
