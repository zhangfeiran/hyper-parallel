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
"""Independent fixed-revision worker dispatch and native address-table admission."""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hyper_parallel.core.multicore.backends import schema, workers
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.common.mark_utils import arg_mark

_FIXTURES = Path(__file__).with_name("fixtures") / "original_worker_glue.json"


def _stub(original):
    return (
        original["copyright"]
        + "class Worker { public:\n"
        + original["constants"]
        + "\n"
        + original["method"]
        + "\nprivate:\n"
        + original["members"]
        + "\n};\n"
    )


def _expand(text, generated):
    for path, content in generated.items():
        text = text.replace(f'#include "runtime/generated/{path.name}"', content)
    return text


def _compile(source):
    with TemporaryDirectory() as directory:
        path, binary = Path(directory) / "worker.cpp", Path(directory) / "worker"
        path.write_text(source)
        subprocess.run(
            ["c++", "-std=c++17", "-O2", str(path), "-o", str(binary)], check=True
        )
        subprocess.run([str(binary)], check=True)


def _dispatch_probe(name, original, generated):
    method = original["method"]
    namespace = "HyperMegaGate" if original["family"] == "gate" else ""
    calls = (
        re.findall(r"(?:route_|pipeline_)\.(\w+)\(\)", method)
        if namespace
        else re.findall(r"(Execute\w+)(?:<[^>]+>)?\((taskDesc|task_desc)\)", method)
    )
    code = (
        schema.cpp_schema(original["family"])
        + "\n#include <string>\n#include <utility>\n"
        "#define __aicore__\n#define DTYPE_DISPATCH_TARGET int\n"
        "using MulticoreRuntime::TaskType;\nstruct TaskDesc { TaskType task_type; uint32_t marker; };\n"
        "std::pair<std::string,uint32_t> observed;\nvoid Trap() { throw 1; }\n"
    )
    if namespace:
        context = (
            "RoutePipeline"
            if original["direction"] == "forward"
            else "RouteGradPipeline"
        )
        code += f"namespace HyperMegaGate {{ struct {context} {{\n"
        code += "".join(
            f'void {call}() {{ observed = {{"{call}",0}}; }}\n' for call in calls
        )
        code += "}; }\n"
        callbacks = ""
    else:
        callbacks = "".join(
            ("template <typename T> " if call == "ExecuteShmemGetMem" else "")
            + f'void {call}(TaskDesc task) {{ observed = {{"{call}",task.marker}}; }}\n'
            for call in dict.fromkeys(call for call, _ in calls)
        )
    for class_name, stub in (
        ("Expected", _stub(original)),
        ("Actual", workers.adapt_native_worker(_stub(original), name)),
    ):
        stub = _expand(stub, generated).replace("class Worker", "class " + class_name)
        stub = stub.replace("private:", callbacks + "private:")
        code += f"namespace {class_name}Space {{\n{original['preamble']}\n{stub}\n}}\n"
    code += (
        "int main() { ExpectedSpace::Expected expected; ActualSpace::Actual actual;\n"
    )
    code += "for (uint32_t task=0;task<160;++task) {\n"
    if namespace:
        code += "for (uint32_t stage=0;stage<12;++stage) {\n"
        invocation = "static_cast<TaskType>(task), stage"
    else:
        invocation = "TaskDesc{static_cast<TaskType>(task), task + 2026}"
    for instance, result in (("expected", "reference"), ("actual", "result")):
        code += 'observed = {"unchanged",0};\n'
        code += f'try {{ {instance}.ExecuteComputeKernel({invocation}); }} catch (int) {{ observed = {{"trap",0}}; }}\n'
        code += f"auto {result} = observed;\n"
    code += (
        "if (reference != result) return 1;\n"
        + ("}\n" if namespace else "")
        + "} return 0; }\n"
    )
    return code


class TestWorkers(unittest.TestCase):
    """Compile independent original worker switches and reject binding drift before allocation."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_generated_switches_match_original_callbacks_and_fallbacks(self):
        """Feature: Generated native worker dispatch.

        Description: Execute generated and independent pinned switches over known/unknown tasks and stage IDs.
        Expectation: Callback selection, descriptor arguments and traps match for every family and direction.
        """
        generated = workers.generated_files()
        for name, original in json.loads(_FIXTURES.read_text()).items():
            with self.subTest(worker=name):
                _compile(_dispatch_probe(name, original, generated))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_generated_address_tables_match_original_slots(self):
        """Feature: Native parameter-to-address mapping.

        Description: Compile independent original and generated input arrays with distinct non-null addresses.
        Expectation: Every input/cache/workspace address stays in its original slot across both TaskDAG families.
        """
        generated = workers.generated_files()
        for name, original in json.loads(_FIXTURES.read_text()).items():
            if "inputs" not in original:
                continue
            with self.subTest(worker=name):
                original_inputs = original["inputs"]
                clean = re.sub(r"//[^\n]*", "", original_inputs)
                array = re.search(r"(input_list|inputList)\[", clean)[1]
                addresses = re.findall(r"\w+", clean.split("{", 1)[1].split("}")[0])
                identifiers = list(
                    dict.fromkeys(
                        address for address in addresses if address != "nullptr"
                    )
                )
                code = "#include <cstdint>\nusing GM_ADDR = char*;\nchar storage[128];\nint main() {\n"
                code += "".join(
                    f"GM_ADDR {address} = storage + {index};\n"
                    for index, address in enumerate(identifiers)
                )
                code += original_inputs.replace(array, "expected") + "\n"
                code += _expand(
                    workers.adapt_native_entry(original_inputs, name), generated
                ).replace(array, "actual")
                code += "\nfor (unsigned i=0;i<sizeof(expected)/sizeof(GM_ADDR);++i) if (expected[i]!=actual[i]) return 1;\n"
                code += "return 0; }\n"
                _compile(code)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_adapters_preserve_context_and_reject_native_drift(self):
        """Feature: Verified worker glue adaptation.

        Description: Expand installed fragments back into independent sources and mutate callback/address/member contracts.
        Expectation: Native context tokens are preserved; altered callbacks, slots and members are rejected.
        """
        for name, original in json.loads(_FIXTURES.read_text()).items():
            with self.subTest(worker=name):
                text = _stub(original)
                adapted = workers.adapt_native_worker(text, name)
                expanded = _expand(adapted, workers.generated_files())
                self.assertEqual(workers._tokens(text), workers._tokens(expanded))
                mutated = text.replace("break;", "return;", 1)
                with self.assertRaisesRegex(ValueError, "contract drift"):
                    workers.adapt_native_worker(mutated, name)
                if original["constants"]:
                    mutated = re.sub(r"(_IDX = )\d+", r"\g<1>999", text, count=1)
                    with self.assertRaisesRegex(ValueError, "contract drift"):
                        workers.adapt_native_worker(mutated, name)
                    with self.assertRaisesRegex(ValueError, "contract drift"):
                        workers.adapt_native_entry(
                            original["inputs"].replace("tiling", "wrong_tiling"), name
                        )
                else:
                    with self.assertRaisesRegex(ValueError, "contract drift"):
                        workers.adapt_native_worker(
                            text.replace("Pipeline ", "WrongPipeline "), name
                        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_all_family_emissions_include_worker_contracts_and_sources(self):
        """Feature: Common worker source emission.

        Description: Compile all families and independently reject unknown/duplicate tasks and out-of-bounds slots.
        Expectation: Both worker directions and their original context ownership accompany each source-only bundle.
        """
        emissions = [
            _route.compile({"T": 41, "E": 16}, k=3, scale=2.5),
            moe_region.compile(
                MegaMoeSpec(128, 512, 128, 4, 2, 1.25, 384, 2, None, 0, 20)
            ),
            mhc_boundary.compile(MhcSpec(2593, 128, 20)),
        ]
        for emission in emissions:
            files = {file.path: file.content for file in emission.files}
            manifest = json.loads(files["workers/manifest.json"])
            family = manifest["family"]
            self.assertEqual(
                set(manifest["workers"]), {family + "_forward", family + "_backward"}
            )
            for name, worker in manifest["workers"].items():
                self.assertIn(f"workers/{name}_dispatch.inc", files)
                self.assertEqual(
                    worker["context_owner"],
                    "worker" if family == "gate" else "native_callback",
                )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            original = json.loads(workers._CONTRACTS.read_text())
            for invalid in ("unknown", "duplicate", "slot"):
                data = json.loads(json.dumps(original))
                worker = data["workers"]["moe_forward"]
                if invalid == "slot":
                    worker["constants"]["TILING_IDX"] = 999
                else:
                    task = (
                        "UNKNOWN_TASK"
                        if invalid == "unknown"
                        else worker["cases"][0]["tasks"][0]
                    )
                    worker["cases"].append({"tasks": [task], "call": ""})
                path.write_text(json.dumps(data))
                with (
                    patch.object(workers, "_CONTRACTS", path),
                    self.assertRaises(ValueError),
                ):
                    workers.worker_manifest("moe")
