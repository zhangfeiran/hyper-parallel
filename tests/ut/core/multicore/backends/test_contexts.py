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
"""Independent Gate context admission and compiled UB/lifecycle contract tests."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from hyper_parallel.core.multicore.backends import contexts, workers
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from tests.common.mark_utils import arg_mark

_FIXTURE = Path(__file__).with_name("fixtures") / "original_gate_contexts.json"
_HARNESS = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECCALC };
enum class HardEvent { MTE3_S };
using event_t = int;
std::vector<std::string> trace;
template <QuePosition Position, int Depth> struct TQue {};
template <TPosition Position> struct TBuf {};
struct TPipe {
  bool destroyed = false;
  TPipe() { trace.push_back("create"); }
  ~TPipe() { assert(destroyed); trace.push_back("destruct"); }
  template <QuePosition P, int D>
  void InitBuffer(TQue<P, D>&, int depth, uint32_t bytes) {
    assert(!destroyed && depth == D);
    trace.push_back(std::string(P == QuePosition::VECIN ? "in:" : "out:") + std::to_string(bytes));
  }
  template <TPosition P> void InitBuffer(TBuf<P>&, uint32_t bytes) {
    assert(!destroyed); trace.push_back("buf:" + std::to_string(bytes));
  }
  int FetchEventID(HardEvent) { assert(!destroyed); trace.push_back("fetch"); return 7; }
  void Destroy() { assert(!destroyed); destroyed = true; trace.push_back("destroy"); }
};
template <HardEvent E> void SetFlag(event_t event) { assert(event == 7); trace.push_back("set"); }
template <HardEvent E> void WaitFlag(event_t event) { assert(event == 7); trace.push_back("wait"); }
'''


def _expected_buffers(batch, experts, selected, shared):
    score, topk, row = batch * experts * 4, batch * selected * 4, ((batch + 7) // 8) * 8 * 4
    return {
        "gate_forward_softplus": ["in:" + str(score), "out:" + str(score), "buf:" + str(score),
                                  "buf:" + str(score), "buf:" + str(((batch * experts + 255) // 256) * 32),
                                  "buf:" + str(((batch * experts + 255) // 256) * 32)],
        "gate_forward_sqrt_score": [f"in:{score}", f"out:{score}"],
        "gate_forward_add_text_bias": [f"in:{score}", f"out:{score}", f"buf:{experts * 4}"],
        "gate_forward_add_vision_bias": [f"in:{score}", f"in:{experts * 4}", f"in:{experts * 4}",
                                         f"in:{((batch + 31) // 32) * 32}", f"out:{score}"],
        "gate_forward_top_k_score": [f"in:{score}", f"buf:{experts * 4}", f"out:{topk}", f"out:{topk}"],
        "gate_forward_gather_score": [f"in:{score}", f"in:{topk}", f"buf:{row}", f"buf:{topk}",
                                      f"buf:{topk}", f"buf:{topk}", f"out:{topk}", f"buf:{selected * 4}",
                                      f"buf:{topk}", f"buf:{shared}"],
        "gate_forward_reduce_selected": [f"in:{topk}", f"out:{row}", f"buf:{shared}"],
        "gate_forward_add_epsilon": [f"in:{batch * 32}", f"out:{batch * 32}"],
        "gate_forward_divide_selected": [f"in:{topk}", f"in:{row}", f"out:{topk}", f"buf:{topk}", f"buf:{shared}"],
        "gate_forward_scale_selected": [f"in:{topk}", f"out:{topk}"],
        "gate_forward_cast_indices": [f"in:{topk}", f"out:{topk * 2}"],
        "gate_backward_reduce_cross_term": [f"in:{topk}", f"out:{batch * 32}", f"buf:{shared}"],
        "gate_backward_zeros_like": [f"out:{score}"],
        "gate_backward_run_unary_muls": [f"in:{topk}", f"out:{topk}"],
        "gate_backward_run_unary_neg": [f"in:{topk}", f"out:{topk}"],
        "gate_backward_run_broadcast": [f"in:{batch * 32}", f"out:{topk}", f"buf:{shared}"],
        "gate_backward_run_binary": [f"in:{topk}", f"in:{topk}", f"out:{topk}"],
        "gate_backward_run_add": [f"in:{topk}", f"in:{topk}", f"out:{topk}"],
    }


def _compile_stages(root):
    for path, content in contexts.generated_files().items():
        (root / path.name).write_text(content)
    code = _HARNESS
    stages = []
    for name, worker in contexts._contracts().items():
        if worker["family"] != "gate":
            continue
        code += f'void {name}_finish(TPipe& pipe) {{\n#include "{name}_write_completion_finish_context.inc"\n}}\n'
        for key, context in worker["contexts"].items():
            if context["ownership"] != "stage_local":
                continue
            stage = name + "_" + key
            stages.append(stage)
            code += f'''void {stage}(uint32_t batch_rows_, uint32_t expert_align_,
                                     uint32_t k_align_, uint32_t shared_tmp_bytes_) {{
                auto FinishStage = {name}_finish;
                #include "{stage}_init_context.inc"
                trace.push_back("body");
                #include "{stage}_finish_context.inc"
            }}
            '''
    code += 'int main() {\n'
    for batch, experts, selected, shared in ((3, 64, 8, 128), (9, 128, 16, 512), (1, 64, 8, 64)):
        for stage in stages:
            code += f'''trace.clear(); {stage}({batch}, {experts}, {selected}, {shared});
                std::cout << "{batch}:{stage}";
                for (const auto& item : trace) std::cout << " " << item;
                std::cout << "\\n";
                '''
    code += '}\n'
    source, executable = root / "contexts.cpp", root / "contexts"
    source.write_text(code)
    subprocess.run(["c++", "-std=c++17", "-O0", "-I" + str(root), str(source), "-o", str(executable)], check=True)
    return subprocess.check_output([str(executable)], text=True)


class TestContexts(unittest.TestCase):
    """Preserve worker-owned bindings and per-stage pipe/UB lifetime."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_independent_sibling_sources_and_drift(self):
        """Feature: Context generation across worker and pipeline sources.

        Description: Expand generated includes in independent pinned originals and corrupt UB/cleanup contracts.
        Expectation: Full source tokens match, drift fails before installation, and original siblings remain intact.
        """
        originals = json.loads(_FIXTURE.read_text())["sources"]
        generated = contexts.generated_files()
        for name, sources in originals.items():
            for source, original in sources.items():
                with self.subTest(worker=name, source=source):
                    adapted = contexts.adapt_contexts(original, name, source)
                    self.assertIn("_context.inc", adapted)
                    for path, content in generated.items():
                        adapted = adapted.replace(f'#include "runtime/generated/{path.name}"', content)
                    self.assertEqual(workers._tokens(original), workers._tokens(adapted))
                    if source.endswith(".h"):
                        for broken in (original.replace("sizeof(float)", "sizeof(double)"),
                                       original.replace("WaitFlag<HardEvent::MTE3_S>", "SetFlag<HardEvent::MTE3_S>")):
                            with self.assertRaisesRegex(ValueError, "contract drift"):
                                contexts.adapt_contexts(broken, name, source)
            with TemporaryDirectory() as directory:
                kernel = Path(directory)
                for source, original in sources.items():
                    (kernel / source).write_text(original)
                header = next(source for source in sources if source.endswith(".h"))
                (kernel / header).write_text(sources[header].replace("FinishStage(pipe);", "pipe.Destroy();"))
                before = {path: path.read_text() for path in kernel.iterdir()}
                with self.assertRaisesRegex(ValueError, "contract drift"):
                    contexts.install_context_glue(kernel, name)
                self.assertEqual(before, {path: path.read_text() for path in kernel.iterdir()})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_compiled_ub_allocations_and_write_completion(self):
        """Feature: Per-stage pipe and queue/UB context generation.

        Description: Compile all 18 Gate setup/cleanup recipes against a recording Ascend interface on CPU.
        Expectation: Three alignments yield exact buffer bytes and create/body/fetch/set/wait/destroy/destruct order.
        """
        with TemporaryDirectory() as directory:
            lines = _compile_stages(Path(directory)).splitlines()
        self.assertEqual(len(lines), 54)
        shapes = {3: (64, 8, 128), 9: (128, 16, 512), 1: (64, 8, 64)}
        for line in lines:
            label, *events = line.split()
            batch, stage = label.split(":", 1)
            expected = _expected_buffers(int(batch), *shapes[int(batch)])[stage]
            self.assertEqual(events, ["create", *expected, "body", "fetch", "set", "wait", "destroy", "destruct"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_bundle_installation_and_contract_rejection(self):
        """Feature: Shared native context source ownership.

        Description: Install admitted worker/header contexts and inspect bundle ownership; corrupt schema boundaries.
        Expectation: All Gate sources include generated contexts; invalid path, owner and phase contracts fail early.
        """
        originals = json.loads(_FIXTURE.read_text())["sources"]
        for name, sources in originals.items():
            with TemporaryDirectory() as directory:
                operator = Path(directory)
                kernel = operator / "op_kernel"
                kernel.mkdir()
                for source, original in sources.items():
                    (kernel / source).write_text(original)
                workers.install_worker_glue(operator, "gate", name.split("_")[1])
                for source in sources:
                    self.assertIn("_context.inc", (kernel / source).read_text())
        artifacts = {file.path: file.content for file in _route.compile({"T": 41, "E": 16}, k=3, scale=2.5).files}
        self.assertIn("workers/gate_forward_softplus_init_context.inc", artifacts)
        manifest = json.loads(artifacts["workers/manifest.json"])
        self.assertEqual(manifest["workers"]["gate_forward"]["contexts"]["softplus"]["ownership"], "stage_local")
        data = json.loads(contexts._CONTRACTS.read_text())
        for field, value in (("source", "../escape.h"), ("ownership", "unknown"), ("phases", {"run": "foo();"})):
            with TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.json"
                invalid = json.loads(json.dumps(data))
                invalid["workers"]["gate_forward"]["contexts"]["softplus"][field] = value
                path.write_text(json.dumps(invalid))
                with patch.object(contexts, "_CONTRACTS", path), self.assertRaisesRegex(ValueError, "Invalid native context"):
                    contexts.generated_files()
