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
"""Shared schema generation, typed binding, source identity and emission cache admission."""

from __future__ import annotations

import ctypes
import importlib
import inspect
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import hyper_parallel.core.multicore.frontend as mc
from hyper_parallel.core.multicore.backends import schema
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.runtime.abi import family_abi
from hyper_parallel.core.multicore.runtime.bindings import native_entry
from hyper_parallel.core.multicore.runtime.cache import EmissionCache
from hyper_parallel.core.multicore.runtime.generated import native_calls
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.common.mark_utils import arg_mark

FIXTURES = Path(__file__).with_name("fixtures")


def _emissions():
    return (_route.compile({"T": 41, "E": 16}, k=3, scale=2.5),
            moe_region.compile(MegaMoeSpec(128, 512, 128, 4, 2, 1.25, 384, 2, None, 0, 20)),
            mhc_boundary.compile(MhcSpec(2593, 128, 20)))


def _cpp_call_probe(entries):
    source = schema.cpp_calls() + "\n#include <array>\n#include <cstdlib>\n#include <type_traits>\n"
    source += "struct Tensor { int id; };\nint main() {\n"
    for name, entry in entries.items():
        arguments = entry["arguments"]
        writes = ",".join("true" if arg["write"] else "false" for arg in arguments)
        source += "{ Tensor tensor[64]; for (int i=0;i<64;++i) tensor[i].id=i;\n"
        source += f"std::array<bool,{len(arguments)}> writes{{{{{writes}}}}};\n"
        source += """
auto dispatch = [&](auto&&... values) {
    int position = 0;
    auto visit = [&](auto&& value) {
        using Kind = std::remove_reference_t<decltype(value)>;
        if constexpr (std::is_same_v<std::remove_cv_t<Kind>, Tensor>) {
            if (value.id != position || std::is_const_v<Kind> == writes[position]) std::abort();
        } else if constexpr (!std::is_same_v<Kind, bool>) {
            if (static_cast<double>(value) != position) std::abort();
        }
        ++position;
    };
    (visit(values), ...);
    return position;
};
"""
        actual = []
        for index, arg in enumerate(arguments):
            expression = f"tensor[{index}]" if arg["type"] == "Tensor" else (
                "true" if arg["type"] == "bool" else f"{index}")
            actual.append(expression)
        source += (f"if (hyper_parallel::multicore::generated::{name}(dispatch, {','.join(actual)}) "
                   f"!= {len(arguments)}) return 1; }}\n")
    return source + "return 0; }\n"


class TestCodegen(unittest.TestCase):
    """Compare independent wire definitions and native call order before device allocation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_reader_preserves_every_descriptor_word(self):
        """Feature: Native descriptor reader compatibility.

        Description: Compile the final reader and independent original reader against random complete wire records.
        Expectation: All 144 words per descriptor are decoded identically across event capacities and task indices.
        """
        header = (schema.CORE / "ops/runtime/runtime_config.hpp").read_text()
        start = header.index("__aicore__ inline void getTaskDesc")
        end = header.index("__aicore__", start + len("__aicore__"))
        current = header[start:end]
        original = (FIXTURES / "original_task_reader.txt").read_text()
        declarations = (FIXTURES / "moe_runtime_layout.txt").read_text().replace(
            "namespace MulticoreRuntime", "namespace OriginalRuntime")
        source = "#include <cstring>\n#include <vector>\n#include <random>\n#define __aicore__\n#define __gm__\n"
        source += schema.cpp_schema("moe") + declarations
        for namespace, reader in (("OriginalRuntime", original), ("MulticoreRuntime", current)):
            source += f"namespace {namespace} {{\n"
            if namespace == "MulticoreRuntime":
                source += "constexpr uint32_t UINT32_T_SIZE = 4;\n"
                source += "constexpr uint32_t MAX_TENSOR_DIMS = 4, MAX_INPUTS_PER_TASK = 4, MAX_OUTPUTS_PER_TASK = 4;\n"
            source += ("inline uint32_t getAllTasksOffset(uint8_t* data) { "
                       "return 64 + reinterpret_cast<uint32_t*>(data)[3] * 4; }\n")
            source += reader + "}\n"
        source += """
int main() {
    std::mt19937 random(2026);
    for (uint32_t capacity : {16u, 1024u, 2048u}) {
        std::vector<uint32_t> wire(16 + capacity + 144 * 32);
        for (auto& word : wire) word = random();
        wire[3] = capacity;
        for (uint32_t index = 0; index < 32; ++index) {
            OriginalRuntime::TaskDesc expected;
            MulticoreRuntime::TaskDesc actual;
            std::memset(&expected, 0xCD, sizeof(expected));
            std::memset(&actual, 0xCD, sizeof(actual));
            auto* bytes = reinterpret_cast<uint8_t*>(wire.data());
            OriginalRuntime::getTaskDesc(bytes, &expected, index);
            MulticoreRuntime::getTaskDesc(bytes, &actual, index);
            if (std::memcmp(&expected, &actual, sizeof(actual)) != 0) return 1;
        }
    }
    return 0;
}
"""
        with TemporaryDirectory() as directory:
            path, binary = Path(directory) / "reader.cpp", Path(directory) / "reader"
            path.write_text(source)
            subprocess.run(["c++", "-std=c++17", "-O2", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_generated_files_and_ctypes_match_schemas(self):
        """Feature: Same-source ABI generation.

        Description: Verify checked-in generated outputs and every ctypes size/field/enum.
        Expectation: Generated files are reproducible and all family schemas retain exact wire layout.
        """
        for relative, content in schema.generated_files().items():
            with self.subTest(path=relative):
                self.assertEqual((schema.CORE / relative).read_text(), content)
        for family in ("gate", "moe", "mhc"):
            module = importlib.import_module(f"hyper_parallel.core.multicore.runtime.generated.{family}_abi")
            for structure in family_abi(family).structures:
                kind = getattr(module, structure.name)
                self.assertEqual(ctypes.sizeof(kind), structure.size)
                for field in structure.fields:
                    self.assertEqual(getattr(kind, field.name).offset, field.offset)
            for binding in family_abi(family).task_types:
                self.assertEqual(getattr(module.TaskType, binding.native_name), binding.numeric_id)
            self.assertEqual(module.TaskDescC().profile_desc_id, 0xFFFFFFFF)
            self.assertEqual(module.TaskDescC(profile_owner_id=8).profile_owner_id, 8)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_generated_cpp_matches_independent_original_layout(self):
        """Feature: Native typed wire compatibility.

        Description: Compile generated declarations alongside independent fixed-revision native snapshots.
        Expectation: Every native sizeof/offsetof agrees for all three families.
        """
        names = {"TensorDescC": "TensorDesc", "TaskDescC": "TaskDesc", "EventDescC": "EventDesc",
                 "DynamicDataC": "DynamicData", "RuntimeConfigC": "RuntimeHeader"}
        for family in ("gate", "moe", "mhc"):
            with self.subTest(family=family), TemporaryDirectory() as directory:
                source = schema.cpp_schema(family)
                source += (FIXTURES / f"{family}_runtime_layout.txt").read_text().replace(
                    "namespace MulticoreRuntime", "namespace OriginalRuntime")
                for structure in family_abi(family).structures:
                    name = names[structure.name]
                    source += f"static_assert(sizeof(MulticoreRuntime::{name}) == sizeof(OriginalRuntime::{name}));\n"
                    for field in structure.fields:
                        member = field.name.removeprefix("_")
                        source += (f"static_assert(offsetof(MulticoreRuntime::{name}, {member}) == "
                                   f"offsetof(OriginalRuntime::{name}, {member}));\n")
                for binding in family_abi(family).task_types:
                    name = binding.native_name
                    source += (f"static_assert(static_cast<uint32_t>(MulticoreRuntime::TaskType::{name}) == "
                               f"static_cast<uint32_t>(OriginalRuntime::TaskType::{name}));\n")
                path = Path(directory) / "schema.cpp"
                path.write_text(source + "int main() { return 0; }\n")
                subprocess.run(["c++", "-std=c++17", str(path), "-o", str(Path(directory) / "schema")], check=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_typed_wrappers_and_loaded_schema_admission(self):
        """Feature: Shared typed native launch order.

        Description: Bind reversed named resources and impersonate native schema/alias contracts.
        Expectation: All six wrappers preserve argument order; schema and binding drift is rejected.
        """
        entries = json.loads((schema.CORE / "runtime/native_calls.json").read_text())["entries"]
        for name in entries:
            entry = native_entry(name)
            values = {arg.name: object() for arg in reversed(entry.arguments)}
            packed = entry.pack(values)
            wrapper = getattr(native_calls, name)
            self.assertEqual(tuple(inspect.signature(wrapper).parameters), tuple(arg.name for arg in entry.arguments))
            with patch.object(native_calls, "resolve_native_call") as resolver:
                wrapper(**values)
                resolver.assert_called_once_with(name)
                resolver.return_value.assert_called_once_with(*packed)
            expected = torch._C.parse_schema("hyper_parallel::" + entry.schema)
            entry.verify(expected)
            if "Tensor(a!)" in entry.schema:
                with self.assertRaisesRegex(ValueError, "schema mismatch"):
                    entry.verify(torch._C.parse_schema("hyper_parallel::" + entry.schema.replace(
                        "Tensor(a!)", "Tensor(a)")))
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                entry.verify(torch._C.parse_schema("hyper_parallel::" + entry.schema.replace("runtime_config", "wrong")))
            with self.assertRaisesRegex(ValueError, "binding names"):
                entry.pack({**values, "unexpected": 1})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calls.cpp"
            path.write_text(_cpp_call_probe(entries))
            binary = Path(directory) / "calls"
            subprocess.run(["c++", "-std=c++17", str(path), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_common_emission_cache_and_corruption_admission(self):
        """Feature: Unified source emission cache.

        Description: Generate all family plans and corrupt a stored descriptor or artifact set.
        Expectation: Normal/profiled images survive emission; cache reuse rejects every drift.
        """
        for emission in _emissions():
            with self.subTest(family=emission.export_manifest()["family"]), TemporaryDirectory() as directory:
                cache = EmissionCache(Path(directory))
                target = cache.store(emission)
                self.assertEqual(cache.store(emission), target)
                files = cache.load(emission.artifact_key)
                self.assertEqual(files["schedule/forward.bin"], emission.plan.forward.normal)
                self.assertEqual(files["schedule/forward.profiled.bin"], emission.plan.forward.profiled)
                self.assertTrue(json.loads(files["source_map.json"])["manifest"])
                path = target / "schedule/forward.bin"
                original = path.read_bytes()
                path.write_bytes(b"corruption")
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    cache.load(emission.artifact_key)
                path.write_bytes(original)
                manifest = target / "manifest.json"
                intact = manifest.read_text()
                data = json.loads(intact)
                data["plan_key"] = "0" * 64
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "identity"):
                    cache.load(emission.artifact_key)
                manifest.write_text(intact)
                (target / "unexpected.bin").write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "unrecorded"):
                    cache.load(emission.artifact_key)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_definition_plan_and_source_map_keys_are_separate(self):
        """Feature: Static key and source provenance separation.

        Description: Change rows or source locations without changing primitives/constexpr attributes.
        Expectation: Definition identity is reused; plan and source-mapped artifact keys remain distinct.
        """
        first = _route.compile({"T": 41, "E": 16}, k=3, scale=2.5)
        second = _route.compile({"T": 65, "E": 16}, k=3, scale=2.5)
        self.assertEqual(first.definition_key, second.definition_key)
        self.assertNotEqual(first.plan_key, second.plan_key)
        program = mc.from_source("\n" + _route.source.text, symbols=_route.source.symbols, schedule=mc.WorkerPipeline())
        third = program.compile({"T": 41, "E": 16}, k=3, scale=2.5)
        self.assertEqual(first.definition_key, third.definition_key)
        self.assertEqual(first.plan_key, third.plan_key)
        self.assertNotEqual(first.artifact_key, third.artifact_key)
        code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu' or fullname.startswith('torch_npu.'):
            raise AssertionError('device backend imported by emitter')
sys.meta_path.insert(0, Block())
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
assert mhc_boundary.compile(MhcSpec(41,128,20)).files
"""
        result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
