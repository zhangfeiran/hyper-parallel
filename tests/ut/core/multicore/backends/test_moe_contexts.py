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
"""Independent fixed MoE primitive context factory and scope-lifetime probes."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hyper_parallel.core.multicore._build import write_moe_manifest
from hyper_parallel.core.multicore.backends import contexts, workers
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark

_FIXTURE = Path(__file__).with_name("fixtures") / "original_moe_contexts.json"
_COMMON = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
using GM_ADDR = unsigned char*;
struct bfloat16_t {};
struct half {};
namespace AscendC {}
std::vector<std::string> trace;
struct TPipe {
  TPipe() { trace.push_back("pipe_create"); }
  ~TPipe() { trace.push_back("pipe_destroy"); }
};
void Emit() { for (const auto& event : trace) std::cout << event << " "; std::cout << "\n"; }
'''
_GLU = r'''
struct SwiGluTilingData { int isDoubleBuffer; };
#define GET_TILING_DATA_WITH_STRUCT(Type, Name, Address) Type Name = *reinterpret_cast<Type*>(Address)
unsigned char storage[32];
SwiGluTilingData shape;
template <typename Input, typename Accum, typename Output, int Buffers> struct SwigluVectorBF16 {
  TPipe pipe;
  SwigluVectorBF16() { trace.push_back(std::string("vector:") +
      (std::is_same_v<Input, bfloat16_t> ? "bf16:" : "half:") + std::to_string(Buffers)); }
  ~SwigluVectorBF16() { trace.push_back("vector_destroy"); }
};
template <typename Input, typename Accum, typename Output, int Buffers>
using SwiGluGradBF16 = SwigluVectorBF16<Input, Accum, Output, Buffers>;
template <typename Vector, typename Input, typename Output> struct SwigluSingle {
  Vector vector;
  SwigluSingle() { trace.push_back("instance_create"); }
  ~SwigluSingle() { trace.push_back("instance_destroy"); }
  void Init(GM_ADDR first, GM_ADDR second, GM_ADDR output, GM_ADDR tiling, float clamp) {
    assert(first == storage + 1 && output == storage + 3 && tiling == reinterpret_cast<GM_ADDR>(&shape));
    assert(second == nullptr || second == storage + 2);
    trace.push_back(std::string("init:") + (second == nullptr ? "forward:" : "backward:") + std::to_string(clamp));
  }
  void Process() { trace.push_back("process"); }
};
template <typename Vector, typename Input, typename Output>
using SwiGluGradSingle = SwigluSingle<Vector, Input, Output>;
'''
_GMM = r'''
bool is_aiv = false;
#define ASCEND_IS_AIV (is_aiv)
#define KERNEL_TYPE_AIC_ONLY 1
#define KERNEL_TASK_TYPE_DEFAULT(Type)
namespace AscendCUtils { void SetOverflow(int value) { assert(value == 1); trace.push_back("overflow"); } }
unsigned char storage[32];
GM_ADDR GetUserWorkspace(GM_ADDR address) { assert(address == storage + 11); trace.push_back("workspace"); return address + 1; }
struct Params { int marker = 31; };
struct TCubeTiling { int marker = 32; };
#define GET_TILING_DATA_MEMBER(Type, Field, Name, Address) decltype(Field) Name = Field
#define GET_TILING_DATA_MEMBER_ADDR(Type, Field, Name, Address) GM_ADDR Name = storage + 13
Params gmmBaseParams;
TCubeTiling mmTilingData;
template <bool Transpose> struct xType { static constexpr bool trans = Transpose; };
template <bool Transpose> using weightType = xType<Transpose>;
struct yType { static constexpr bool fp32 = false; };
struct biasType {};
namespace AscendC { enum class TPosition { GM }; }
enum class CubeFormat { ND };
template <AscendC::TPosition P, CubeFormat F, typename T>
struct MatmulType { static constexpr bool fp32 = std::is_same_v<T, float>; };
constexpr int matmulCFG = 1, matmulCFGUnitFlag = 2;
template <typename X, typename W, typename Y, typename Bias, int Config> struct MMImplType {
  struct MT {
    MT() { trace.push_back("mm:" + std::to_string(X::trans) + ":" + std::to_string(W::trans) + ":" +
                           std::to_string(Y::fp32) + ":" + std::to_string(Config)); }
    ~MT() { trace.push_back("mm_destroy"); }
    void SetSubBlockIdx(int index) { assert(index == 0); trace.push_back("subblock"); }
    void Init(TCubeTiling* tiling, TPipe* pipe) { assert(tiling->marker == 32 && pipe); trace.push_back("mm_init"); }
  };
};
template <typename M, bool Sync> struct GMMCompute {
  GMMCompute(typename M::MT&) { assert(!Sync); trace.push_back("compute_create"); }
  ~GMMCompute() { trace.push_back("compute_destroy"); }
  void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset, GM_ADDR antiScale,
            GM_ADDR antiOffset, GM_ADDR groupList, GM_ADDR tokenScale, GM_ADDR y, GM_ADDR user,
            Params* params, TCubeTiling* tiling, TPipe* pipe) {
    assert(x == storage + 1 && weight == storage + 2 && bias == storage + 3 && scale == storage + 4);
    assert(offset == storage + 5 && antiScale == storage + 6 && antiOffset == storage + 7);
    assert(groupList == storage + 8 && tokenScale == storage + 9 && y == storage + 10 && user == storage + 12);
    assert(params->marker == 31 && tiling->marker == 32 && pipe); trace.push_back("compute_init");
  }
};
template <typename Compute> struct GMMProcess {
  GMMProcess(Compute&) { trace.push_back("process_create"); }
  ~GMMProcess() { trace.push_back("process_destroy"); }
  void Init(Params* params, TCubeTiling* tiling, GM_ADDR array, GM_ADDR groups, GM_ADDR data) {
    assert(params->marker == 31 && tiling->marker == 32 && array == storage + 13);
    assert(groups == storage + 8 && data == storage + 14); trace.push_back("process_init");
  }
  void Process() { trace.push_back("process"); }
};
'''


def _native_tokens(text):
    return workers._tokens(text.replace("\\\n", " "))


def _expand(text, source):
    generated = contexts.generated_files()
    prefix = "../" * len(Path(source).parent.parts)
    for path, content in generated.items():
        text = text.replace(f'#include "{prefix}runtime/generated/{path.name}"', content)
    return text


def _probe(code):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for header in ("swi_glu_impl.hpp", "swi_glu_bf16.hpp", "swi_glu_single.hpp",
                       "swi_glu_grad_float.hpp", "swi_glu_grad_bf16.hpp", "swi_glu_grad_single.hpp"):
            (root / header).write_text("")
        source, executable = root / "contexts.cpp", root / "contexts"
        source.write_text(code)
        subprocess.run(["c++", "-std=c++17", "-O0", "-I" + str(root), str(source), "-o", str(executable)], check=True)
        return subprocess.check_output([str(executable)], text=True).splitlines()


def _glu_probe(originals, dtype):
    code = _COMMON + _GLU + f"\n#define DTYPE_Y {dtype}\n#define DTYPE_DY {dtype}\n"
    for namespace in ("Expected", "Actual"):
        code += f"\n#undef OPP_MEGA_MOE_SWI_GLU_CPP\nnamespace {namespace} {{\n"
        for name, source in (("moe_forward", "swi_glu/swi_glu.cpp"),
                             ("moe_backward", "swi_glu_grad/swi_glu_grad.cpp")):
            text = originals[name][source]
            emitted = text if namespace == "Expected" else _expand(contexts.adapt_contexts(text, name, source), source)
            # Side-by-side probe namespaces need independent linkage for the identical C entry names.
            code += emitted.replace('extern "C" inline', 'inline')
        code += "\n}\n"
    code += 'int main() { std::vector<std::string> expected;\n'
    for buffered in (0, 1):
        for clamp in (0., 10., .25):
            for method in ("swi_glu", "swi_glu_grad"):
                arguments = ("storage + 1, storage + 3, storage + 4, reinterpret_cast<GM_ADDR>(&shape)"
                             if method == "swi_glu" else
                             "storage + 1, storage + 2, storage + 3, storage + 4, reinterpret_cast<GM_ADDR>(&shape)")
                code += f'''shape.isDoubleBuffer = {buffered}; trace.clear();
                    Expected::{method}({arguments}, {clamp}f); expected = trace;
                    trace.clear(); Actual::{method}({arguments}, {clamp}f);
                    assert(trace == expected); Emit();
                    '''
    return _probe(code + '}\n')


def _gmm_probe(originals, name):
    source = "grouped_matmul/grouped_matmul.cpp"
    original = originals[name][source]
    generated = _expand(contexts.adapt_contexts(original, name, source), source)
    code = _COMMON + _GMM
    for namespace, text in (("Expected", original), ("Actual", generated)):
        code += '#undef GMM_CUBE_IMP\n' + contexts._macro(text, "GMM_CUBE_IMP") + "\n"
        code += f"namespace {namespace} {{\n" + contexts._method(text, "grouped_matmul") + "\n}\n"
    code += 'int main() { std::vector<std::string> expected;\n'
    for aiv in (False, True):
        for trans_a, trans_b in ((False, False), (False, True), (True, False), (True, True)):
            for fp32 in (False, True):
                arguments = ", ".join([f"storage + {index}" for index in range(1, 12)] +
                                      ["storage + 14", str(trans_a).lower(), str(trans_b).lower(), str(fp32).lower()])
                code += f'''is_aiv = {str(aiv).lower()}; trace.clear();
                    Expected::grouped_matmul({arguments}); expected = trace;
                    trace.clear(); Actual::grouped_matmul({arguments}); assert(trace == expected); Emit();
                    '''
    return _probe(code + '}\n')


class TestMoeContexts(unittest.TestCase):
    """Preserve selected primitive construction, borrowed pipes and implicit scope exit."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_original_dependency_tokens_and_drift(self):
        """Feature: Fixed MoE context factories inside locked adapted dependency sources.

        Description: Expand nested generated includes and corrupt constructor count, clamps and GMM ownership.
        Expectation: Full original tokens match; changed setup and duplicate/missing occurrences fail admission.
        """
        originals = json.loads(_FIXTURE.read_text())["sources"]
        for name, sources in originals.items():
            for source, original in sources.items():
                with self.subTest(worker=name, source=source):
                    generated = contexts.adapt_contexts(original, name, source)
                    self.assertIn('#include "../runtime/generated/', generated)
                    self.assertEqual(_native_tokens(original), _native_tokens(_expand(generated, source)))
                    broken = (original.replace("mm.Init(&mmTilingData_, &tPipe)", "mm.Init(nullptr, &tPipe)")
                              if "grouped_matmul" in source else original.replace(
                                  "output_gm, tiling, clamp_limit", "output_gm, tiling, 0.0f"))
                    with self.assertRaisesRegex(ValueError, "contract drift"):
                        contexts.adapt_contexts(broken, name, source)
        original = originals["moe_forward"]["swi_glu/swi_glu.cpp"]
        with self.assertRaisesRegex(ValueError, "contract drift"):
            contexts.adapt_contexts(original.replace("bfloat16_t, float, bfloat16_t, 2", "bfloat16_t, float, bfloat16_t, 1", 1),
                                    "moe_forward", "swi_glu/swi_glu.cpp")
        with TemporaryDirectory() as directory:
            kernel = Path(directory)
            for source, original in originals["moe_backward"].items():
                path = kernel / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(original)
            contexts.install_context_glue(kernel, "moe_backward")
            self.assertTrue(all("_context.inc" in path.read_text() for path in kernel.rglob("*.cpp")))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_compiled_glu_and_gmm_factory_lifetimes(self):
        """Feature: Original-scope primitive instances and borrowed GMM context.

        Description: Compile original/generated wrappers across dtype/buffering/clamp and transpose/FP32/AIV cases.
        Expectation: Init argument order, instance selection, processing, early returns and reverse destruction match.
        """
        originals = json.loads(_FIXTURE.read_text())["sources"]
        for dtype in ("bfloat16_t", "half"):
            lines = _glu_probe(originals, dtype)
            self.assertEqual(len(lines), 12)
            for index, line in enumerate(lines):
                events = line.split()
                buffered = index // 6
                forward = index % 2 == 0
                count = 2 if buffered or (forward and dtype == "bfloat16_t") else 1
                kind = "bf16" if dtype == "bfloat16_t" else "half"
                clamp = (0., 10., .25)[(index % 6) // 2]
                self.assertEqual(events, ["pipe_create", f"vector:{kind}:{count}", "instance_create",
                                          f'init:{"forward" if forward else "backward"}:{clamp:.6f}', "process",
                                          "instance_destroy", "vector_destroy", "pipe_destroy"])
        for name in originals:
            lines = _gmm_probe(originals, name)
            self.assertEqual(len(lines), 16)
            for index, line in enumerate(lines):
                prefix = ["pipe_create", "overflow", "workspace"]
                if index >= 8:
                    expected = [*prefix, "pipe_destroy"]
                else:
                    trans_a = index >= 4
                    trans_b = index in (2, 3)
                    fp32 = trans_a and index % 2 == 1
                    expected = [*prefix, f"mm:{int(trans_a)}:{int(trans_b)}:{int(fp32)}:{1 if trans_a else 2}",
                                "subblock", "mm_init", "compute_create", "compute_init", "process_create",
                                "process_init", "process", "process_destroy", "compute_destroy", "mm_destroy", "pipe_destroy"]
                self.assertEqual(line.split(), expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_bundle_contract_validation_and_native_source_sealing(self):
        """Feature: Shared context bundle and sealed actual native source closure.

        Description: Inspect emitted factories, reject invalid scope/count/path, and mutate a staged dependency source.
        Expectation: Source bundles include both directions; source hash changes and incomplete native closures fail.
        """
        emission = moe_region.compile(MegaMoeSpec(128, 512, 128, 4, 2, 1.25, 384, 2, None, 0, 20))
        artifacts = {file.path: file.content for file in emission.files}
        self.assertIn("workers/moe_backward_gmm_cube_factory_definition_context.inc", artifacts)
        data = json.loads(contexts._CONTRACTS.read_text())
        for field, value in (("source", "../escape.cpp"), ("source", "a//b.cpp"),
                             ("scope", "unknown"), ("occurrences", 0), ("occurrences", True)):
            with TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                invalid = json.loads(json.dumps(data))
                invalid["workers"]["moe_forward"]["contexts"]["swiglu_bf16_double"][field] = value
                path.write_text(json.dumps(invalid))
                with patch.object(contexts, "_CONTRACTS", path), self.assertRaisesRegex(ValueError, "Invalid native context"):
                    contexts.generated_files()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            component = root / "hyper_parallel/core/multicore"
            baseline = component / "runtime/baselines/families.json"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps({"families": {"moe": {"source_hashes": {}}}}))
            for relative in ("build.sh", "_build/dependencies.lock.json", "_build/write_moe_manifest.py",
                             "backends/schema.py", "backends/workers.py", "backends/contexts.py", "backends/launchers.py",
                             "runtime/native_calls.json", "runtime/worker_calls.json", "_build/assemble_multicore_source.py"):
                path = component / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture input")
            import_tag = f"cp{write_moe_manifest.sys.version_info.major}{write_moe_manifest.sys.version_info.minor}"
            header = root / "build/native/work/multicore/framework" / import_tag / "torch-stage/torch/csrc/generated/launcher_types.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("fixture header")
            native = root / "build/native/work/multicore/source-assembly/ascend910b/source"
            worker = native / "hyper_mega_moe/op_kernel/worker_kernel.cpp"
            worker.parent.mkdir(parents=True)
            worker.write_text("original generated worker")
            with patch.object(write_moe_manifest, "_REPO", root), patch.object(write_moe_manifest, "_COMPONENT", component):
                first = write_moe_manifest._sources("ascend910b")
                worker.write_text("changed generated worker")
                second = write_moe_manifest._sources("ascend910b")
                key = "assembled_native/ascend910b/hyper_mega_moe/op_kernel/worker_kernel.cpp"
                self.assertNotEqual(first[key], second[key])
                with self.assertRaisesRegex(ValueError, "must be assembled"):
                    write_moe_manifest._sources("ascend910b,ascend910_93")
