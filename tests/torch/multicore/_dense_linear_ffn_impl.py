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
"""原始 NK Linear 常驻 FFN 与独立原始打包实现的设备组件验收。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig


CASES = ((0, 64, 128, None, False), (1, 64, 128, None, False), (7, 64, 128, None, False),
         (129, 64, 128, None, True), (7, 17, 19, None, False), (1024, 1024, 4096, None, False),
         (1024, 1024, 4096, DenseTilePolicy(cube_workers=3, column_workers=4), False))


def _source(hidden, intermediate):
    source = nn.Module()
    source.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
    source.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
    source.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)
    return source


def _execution(policy):
    return DenseExecutionConfig(backend="resident_tiles", soc="Ascend910B3", tile_policy=policy,
                                cache_root=Path(os.environ["HP_FFN_CACHE_ROOT"]) if "HP_FFN_CACHE_ROOT" in os.environ
                                else None)


def _difference(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return float((actual.float() - expected.float()).abs().max()) if actual.numel() else 0.0


def _case(rows, hidden, intermediate, policy, recompute):
    torch.manual_seed(37)
    source = _source(hidden, intermediate)
    baseline = SwiGLUMLP(module=source)
    candidate = MegaFFNAdapter(module=source, context={"mega_ffn_program": linear_ffn,
                              "mega_ffn_execution": _execution(policy)})
    baseline, candidate = baseline.to("npu:0"), candidate.to("npu:0")
    generator = torch.Generator(device="cpu").manual_seed(113)
    inputs = torch.randn(rows, hidden, generator=generator, dtype=torch.bfloat16).to("npu:0")
    actual_x, expected_x = inputs.detach().clone().requires_grad_(), inputs.detach().clone().requires_grad_()
    weights = tuple(candidate.parameters())
    versions = tuple(value._version for value in (actual_x, *weights))
    if recompute:
        actual = checkpoint(candidate, actual_x, use_reentrant=False)
        expected = checkpoint(baseline, expected_x, use_reentrant=False)
    else:
        actual, expected = candidate(actual_x), baseline(expected_x)
    held = tuple(candidate._executables.values())
    close_before_backward = rows == 7 and hidden == 64
    if close_before_backward:
        candidate.close()
    gradient = torch.randn(rows, hidden, generator=generator, dtype=torch.bfloat16).to("npu:0")
    actual.backward(gradient)
    expected.backward(gradient)
    torch.npu.synchronize()
    comparisons = ((actual, expected), (actual_x.grad, expected_x.grad),
                   (candidate.gate_up.grad, baseline.linear_fc1.weight.grad),
                   (candidate.down.grad, baseline.linear_fc2.weight.grad))
    differences = [_difference(value, reference) for value, reference in comparisons]
    if versions != tuple(value._version for value in (actual_x, *weights)):
        raise AssertionError("Linear forward/backward changed borrowed input or weight versions")
    native = [executable.native_identity for executable in held]
    for identity in native:
        runtime = identity["backward_runtime"]
        if runtime["native_calls"] != 1 or runtime["host_calls"]:
            raise AssertionError("Linear FFN must execute one complete device backward without host repair")
        if identity["tiling"]["direction"] != "forward":
            raise AssertionError("Shared matrix launch lost the semantic forward direction")
    candidate.close()
    return {"tokens": rows, "hidden": hidden, "intermediate": intermediate, "recompute": recompute,
            "close_before_backward": close_before_backward, "max_abs": differences,
            "weight_layout": "nk", "input_versions_preserved": True, "native_payloads": native}


def test_dense_linear_ffn_training() -> None:
    """Feature: 配套原始 Linear 前后向。
    Description: 比较完整设备 FFN 的输出、dX、两组 dW，覆盖重计算与关闭后反向。
    Expectation: 与独立原始 SwiGLUMLP 逐值一致，实际反向无主机补算。
    """
    torch.npu.set_device(0)
    records = []
    directory = Path(os.environ.get("HP_FFN_EVIDENCE_DIR", "build/native/dense/linear-acceptance"))
    directory.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        records.append(_case(*case))
        record = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "cases": records,
                  "complete": len(records) == len(CASES), "formal_original_training_accepted": False,
                  "performance_accepted": False}
        (directory / "linear-component.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_dense_linear_ffn_training()
