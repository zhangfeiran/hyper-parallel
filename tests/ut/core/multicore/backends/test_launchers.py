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
"""Independent native launcher admission, real Meta dispatch and context order checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.backends import contexts, launchers, schema, workers
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from tests.common.mark_utils import arg_mark

_FIXTURE = Path(__file__).with_name("fixtures") / "original_launchers_contexts.json"


def _meta_library(directory, fixtures):
    root = Path(directory)
    header = root / "op_plugin/include/npu_cpp_extension.h"
    header.parent.mkdir(parents=True)
    header.write_text('#pragma once\n#include <ATen/ATen.h>\n#define EXEC_NPU_CMD_EXT(op, ...) do { ++launch_count; } while (0)\n')
    header = root / "csrc/cached_op_api.h"
    header.parent.mkdir(parents=True)
    header.write_text('''#pragma once
namespace hyper_parallel::multicore {
struct CachedOpApi { CachedOpApi(const char*, const char*) {} };
template <typename... Args> void execute_cached_op(const CachedOpApi&, Args&&...) { ++launch_count; }
}
''')
    code = 'int launch_count = 0;\nextern "C" int test_launch_count() { return launch_count; }\n'
    for family in ("gate", "mhc", "moe"):
        for filename, original in fixtures[family].items():
            if "registration" in filename:
                code += launchers.adapt_registration_source(original, family)
            else:
                code += launchers.adapt_launcher_source(original, family, filename)
            code += "\n"
    for path, content in launchers.generated_files().items():
        if path.name == "launcher_types.hpp":
            (root / "generated").mkdir(exist_ok=True)
            (root / "generated/launcher_types.hpp").write_text(content)
            continue
        code = code.replace(f'#include "generated/{path.name}"', content)
    source, library = root / "launchers.cpp", root / "launchers.so"
    source.write_text(code)
    torch_root = Path(torch.__file__).parent
    command = ["c++", "-std=c++17", "-O0", "-g0", "-shared", "-fPIC", str(source),
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
               "-I" + str(root), "-I" + str(torch_root / "include"),
               "-I" + str(torch_root / "include/torch/csrc/api/include"),
               "-L" + str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch", "-ltorch_cpu", "-lc10", "-o", str(library)]
    subprocess.run(command, check=True)
    return library


class TestLaunchers(unittest.TestCase):
    """Preserve loaded dispatcher aliases and numerical recipe context boundaries."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_native_admission_and_source_emission(self):
        """Feature: Same-schema native launch generation.

        Description: Compare independent bridge/schema snapshots and mutate their argument and return order.
        Expectation: All six native contracts are admitted; schema/alias/call drift is rejected before building.
        """
        originals = json.loads(_FIXTURE.read_text())
        for family in ("gate", "mhc", "moe"):
            for name, entry in launchers._entries(family).items():
                original = originals[family][entry["launcher"]["source"]]
                launchers.adapt_launcher_source(original, family, entry["launcher"]["source"])
                symbol = entry["launcher"]["symbol"]
                with self.assertRaisesRegex(ValueError, "drift"):
                    launchers.adapt_launcher_source(original.replace(symbol, "wrong_symbol"), family,
                                                   entry["launcher"]["source"])
                registration = originals[family][entry["launcher"]["registration"]]
                with self.assertRaises(ValueError):
                    launchers.adapt_registration_source(registration.replace('"' + name + '(', '"' + name + '_broken(', 1), family)
        emission = _route.compile({"T": 41, "E": 16}, k=3, scale=2.5)
        artifacts = {file.path: file.content for file in emission.files}
        manifest = json.loads(artifacts["launchers/manifest.json"])
        self.assertEqual(set(manifest["entries"]), {"mega_gate_route", "mega_gate_route_grad"})
        self.assertIn("launchers/mega_gate_route_launcher.inc", artifacts)
        for path, content in schema.generated_files().items():
            self.assertEqual((schema.CORE / path).read_text(), content)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            data = json.loads(launchers._CALLS.read_text())
            data["entries"]["mega_moe"]["launcher"]["return_arguments"].reverse()
            path.write_text(json.dumps(data))
            with patch.object(launchers, "_CALLS", path), self.assertRaisesRegex(ValueError, "alias contract"):
                launchers.launcher_manifest("moe")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_compiled_meta_dispatch_preserves_outputs_and_allocations(self):
        """Feature: Generated dispatcher and Meta implementations.

        Description: Compile generated native/Meta entries with real framework headers and stub only the device bridge.
        Expectation: Real Meta dispatch preserves caller-owned aliases and Gate-owned shapes/dtypes without native enqueue.
        """
        with TemporaryDirectory() as directory:
            library = _meta_library(directory, json.loads(_FIXTURE.read_text()))
            child = '''
import ctypes, json, sys, torch
from hyper_parallel.core.multicore.runtime.bindings import native_entry
entries=json.loads(open(sys.argv[2]).read())["entries"]
torch.ops.load_library(sys.argv[1])
for name,entry in entries.items():
    args={arg["name"]:torch.empty((1,),device="meta") if arg["type"]=="Tensor" else
          {"int":1,"float":1.25,"bool":False}[arg["type"]] for arg in entry["arguments"]}
    if entry["family"]=="gate":
        args.update(logits=torch.empty((41,16),device="meta"), top_k=3,
                    runtime_config=torch.empty((64,),dtype=torch.uint8,device="meta"),
                    profile_buffer=torch.empty((0,),dtype=torch.uint8,device="meta"))
        if name=="mega_gate_route":
            args.update(text_bias=torch.empty((16,),device="meta"),vision_bias=torch.empty((16,),device="meta"),
                        image_mask=torch.empty((1,),dtype=torch.bool,device="meta"))
        else:
            args.update(route_scores=torch.empty((41,16),device="meta"),
                        selected_scores=torch.empty((41,3),device="meta"),
                        normalization_denominator=torch.empty((41,1),device="meta"),
                        expert_indices=torch.empty((41,3),dtype=torch.int64,device="meta"),
                        grad_routing_weights=torch.empty((41,3),device="meta"),
                        direct_grad_logits=torch.empty((41,16),device="meta"))
    op=getattr(torch.ops.hyper_parallel,name).default
    native_entry(name).verify(op._schema)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("hyper_parallel::"+name,"PrivateUse1")
    values=op(**args)
    outputs=values if isinstance(values,tuple) else (values,)
    if entry["family"]!="gate":
        for value,arg in zip(outputs,entry["launcher"]["return_arguments"]):
            assert value._cdata==args[arg]._cdata, (name,arg)
    elif name=="mega_gate_route":
        assert [tuple(v.shape) for v in outputs]==[(41,3),(41,3),(41,16),(41,3),(41,1)]
        assert outputs[1].dtype==torch.int64
        assert len({v._cdata for v in outputs})==5
        assert all(v._cdata!=args["logits"]._cdata for v in outputs)
        try: op(**{**args,"top_k":17})
        except RuntimeError: pass
        else: raise AssertionError("invalid top_k admitted")
    else:
        assert tuple(outputs[0].shape)==(41,16)
        assert outputs[0]._cdata!=args["logits"]._cdata
bridge=ctypes.CDLL(sys.argv[1])
assert bridge.test_launch_count()==0
'''
            subprocess.run([sys.executable, "-c", child, str(library), str(launchers._CALLS)],
                           env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}, check=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_context_init_cleanup_and_reset_order(self):
        """Feature: Generated native context initialization and cleanup.

        Description: Expand generated setup/finish fragments in independent original forward/backward MHC workers.
        Expectation: Every constructor, Init, Process, sync, Reset, Destroy and early return retains its token order.
        """
        originals = json.loads(_FIXTURE.read_text())["contexts"]
        generated = contexts.generated_files()
        for direction, original in originals.items():
            with self.subTest(direction=direction):
                adapted = contexts.adapt_contexts(original, "mhc_" + direction)
                self.assertIn("_init_context.inc", adapted)
                self.assertIn("_finish_context.inc", adapted)
                expanded = adapted
                for path, content in generated.items():
                    expanded = expanded.replace(f'#include "runtime/generated/{path.name}"', content)
                self.assertEqual(workers._tokens(original), workers._tokens(expanded))
                with self.assertRaisesRegex(ValueError, "contract drift"):
                    contexts.adapt_contexts(original.replace("pipe.Destroy();", "pipe.Reset();"), "mhc_" + direction)
