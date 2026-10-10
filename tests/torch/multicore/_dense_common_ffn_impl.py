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
"""三投影 common FFN 与独立原始算子的设备组件验收。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu
from torch import nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig

CASES = ((0, 64, 128, None, False, False), (1, 64, 128, None, False, False),
         (7, 64, 128, None, False, False), (129, 64, 128, None, True, False),
         (7, 17, 19, None, False, False), (7, 64, 128, None, False, True),
         (1024, 1024, 4096, None, False, False),
         (1024, 1024, 4096, DenseTilePolicy(cube_workers=3, column_workers=4), False, False))


class _SourceFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        """创建三个独立 BF16 投影，作为不修改的原始组件基线。"""
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """执行原始 F.linear、BF16 SiLU、BF16 乘法和下投影。"""
        return self.down_proj(functional.silu(self.gate_proj(value)) * self.up_proj(value))


def _execution(policy):
    return DenseExecutionConfig(backend="resident_tiles", soc="Ascend910B3", tile_policy=policy,
                                cache_root=Path(os.environ["HP_FFN_CACHE_ROOT"]) if "HP_FFN_CACHE_ROOT" in os.environ
                                else None)


def _case(rows, hidden, intermediate, policy, recompute, freeze_up):
    torch.manual_seed(37)
    source = _SourceFFN(hidden, intermediate)
    source.up_proj.weight.requires_grad_(not freeze_up)
    candidate = MegaFFNAdapter(module=source, context={"mega_ffn_program": common_ffn,
                              "mega_ffn_execution": _execution(policy)})
    source, candidate = source.to("npu:0"), candidate.to("npu:0")
    generator = torch.Generator(device="cpu").manual_seed(113)
    inputs = torch.randn(rows, hidden, generator=generator, dtype=torch.bfloat16).to("npu:0")
    actual_x, expected_x = inputs.detach().clone().requires_grad_(), inputs.detach().clone().requires_grad_()
    values = (actual_x, *candidate.parameters())
    versions = tuple(value._version for value in values)
    if recompute:
        actual = checkpoint(candidate, actual_x, use_reentrant=False)
        expected = checkpoint(source, expected_x, use_reentrant=False)
    else:
        actual, expected = candidate(actual_x), source(expected_x)
    held = tuple(candidate._executables.values())
    close_before_backward = rows == 7 and hidden == 64 and not freeze_up
    if close_before_backward:
        candidate.close()
    gradient = torch.randn(rows, hidden, generator=generator, dtype=torch.bfloat16).to("npu:0")
    actual.backward(gradient)
    expected.backward(gradient)
    torch.npu.synchronize()
    comparisons = (("output", actual, expected), ("input_gradient", actual_x.grad, expected_x.grad),
                   *((name, value.grad, reference.grad)
                     for name, value, reference in zip(candidate.weight_names, candidate.parameters(),
                                                       source.parameters())))
    differences = {}
    for name, value, reference in comparisons:
        if value is None or reference is None:
            if value is not reference:
                raise AssertionError(f"Common FFN changed the nullable gradient for {name}")
            differences[name] = None
        else:
            differences[name] = {"max_abs": float((value.float() - reference.float()).abs().max())
                                 if value.numel() else 0.0,
                                 "different_values": int(torch.count_nonzero(value != reference))}
    native = [executable.native_identity for executable in held]
    record = {"tokens": rows, "hidden": hidden, "intermediate": intermediate, "recompute": recompute,
              "freeze_up": freeze_up, "close_before_backward": close_before_backward,
              "differences": differences, "weight_layout": "nk", "native_payloads": native}
    if any(value is not None and value["different_values"] for value in differences.values()):
        directory = Path(os.environ.get("HP_FFN_EVIDENCE_DIR", "build/native/dense/common-acceptance"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "component-mismatch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise AssertionError(f"Common FFN differs from original operators: {differences}")
    if versions != tuple(value._version for value in values):
        raise AssertionError("Common forward/backward changed borrowed input or weight versions")
    for identity in native:
        runtime = identity["backward_runtime"]
        if runtime["native_calls"] != 1 or runtime["host_calls"]:
            raise AssertionError("Common FFN must execute one complete device backward without host repair")
    candidate.close()
    return record


def test_dense_common_ffn_training() -> None:
    """Feature: 原始 common 三投影 FFN。
    Description: 比较输出、dX、三个 dW，覆盖重计算、冻结 up 及关闭后反向。
    Expectation: 与独立原始算子逐值一致，实际反向无主机补算。
    """
    torch.npu.set_device(0)
    directory = Path(os.environ.get("HP_FFN_EVIDENCE_DIR", "build/native/dense/common-acceptance"))
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for case in CASES:
        records.append(_case(*case))
        record = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "cases": records,
                  "complete": len(records) == len(CASES), "formal_original_training_accepted": False,
                  "performance_accepted": False}
        (directory / "common-component.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_dense_common_ffn_training()
