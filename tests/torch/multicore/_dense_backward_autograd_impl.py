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
"""标准 resident autograd 的完整梯度图、真实保存状态及关闭生命周期。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.runtime.dense_tile import ResidentDenseExecutable
from tests.torch.multicore._dense_backward_impl import _program, _rand, _reference


def _case(kind, input_mask, output_mask, seed, cache_root, close=False, retain=False, stream=False, hooks=False,
          geometry=None):
    device = torch.device("npu:0")
    tokens, hidden, columns = geometry or (129, 76 if kind in ("glu", "chain", "duplicate") else 513,
                                          74 if kind == "glu_linear" else 37)
    plan = _program(kind, tokens, hidden, columns)
    executable = ResidentDenseExecutable(plan, device, cann_root=Path(os.environ["ASCEND_HOME_PATH"]),
                                        cache_root=cache_root, soc=torch.npu.get_device_name(device),
                                        policy=DenseTilePolicy(rows_per_tile=32, cube_workers=3,
                                                               column_workers=4, prefetch_tiles=1))
    types = dict(plan.value_types)
    values = tuple(_rand(types[value.id].shape, seed + index, device).requires_grad_(needed)
                   for index, (value, needed) in enumerate(zip(plan.ir.inputs, input_mask)))
    independent = tuple(value.detach().clone().requires_grad_(needed)
                        for value, needed in zip(values, input_mask))
    lane = torch.npu.Stream(device=device) if stream else torch.npu.current_stream(device)
    initial = torch.npu.current_stream(device)
    lane.wait_stream(initial)
    captured = []

    def _pack(value):
        captured.append(value.detach())
        return captured[-1]

    try:
        with torch.npu.stream(lane):
            if hooks:
                with torch.autograd.graph.saved_tensors_hooks(_pack, lambda value: value):
                    output = executable(*values)
            else:
                output = executable(*values)
            outputs = output if isinstance(output, tuple) else (output,)
            references, _saved = _reference(kind, independent)
            gradients = tuple(_rand(value.shape, seed + index + 100, device) for index, value in enumerate(outputs))
            selected = tuple(value for value, active in zip(outputs, output_mask) if active)
            reference_selected = tuple(value for value, active in zip(references, output_mask) if active)
            cotangents = tuple(value for value, active in zip(gradients, output_mask) if active)
            requested = tuple(value for value in values if value.requires_grad)
            reference_requested = tuple(value for value in independent if value.requires_grad)
            expected = torch.autograd.grad(reference_selected, reference_requested, cotangents)
            if close:
                executable.close()
            actual = torch.autograd.grad(selected, requested, cotangents, retain_graph=retain)
            if retain:
                repeated = torch.autograd.grad(selected, requested, cotangents)
                for first, second in zip(actual, repeated):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
        initial.wait_stream(lane)
        initial.synchronize()
        for value, reference in zip(outputs, references):
            torch.testing.assert_close(value, reference, rtol=0.02, atol=0.002)
        comparisons = []
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0.02, atol=0.002)
            comparisons.append(float((value.float() - reference.float()).abs().max()) if value.numel() else 0.0)
        runtime = executable.backward_router.statistics()
        if runtime["execution_mode"] != "resident_dense_backward" or runtime["host_calls"]:
            raise AssertionError("标准 autograd 用例必须使用完整设备反向")
        return {"kind": kind, "tokens": tokens, "hidden": hidden, "columns": columns,
                "input_mask": input_mask, "output_mask": output_mask,
                "close_before_backward": close, "retain_graph": retain, "nondefault_stream": stream,
                "saved_hooks": hooks, "gradient_max_abs": comparisons,
                "backward_mode": runtime["execution_mode"], "native_calls": runtime["native_calls"],
                "host_calls": runtime["host_calls"], "cached_variants": runtime["cached_variants"],
                "native": executable.native_identity}
    finally:
        executable.close()
        initial.wait_stream(lane)
        initial.synchronize()


def test_dense_backward_autograd() -> None:
    """核对真实 autograd，包括完整 FFN 的三组实际设备梯度。"""
    if not torch_npu.npu.is_available():
        raise RuntimeError("常驻反向 autograd 验收需要真实 Ascend NPU")
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True)
    cache_root = Path(os.environ.get("HP_FFN_CACHE_ROOT", "build/native/dense"))
    rows = [
        _case("glu", (True,), (True,), 8001, cache_root),
        _case("chain", (True,), (True,), 8002, cache_root, retain=True, hooks=True),
        _case("chain", (True,), (True,), 8003, cache_root, close=True, stream=True),
        _case("duplicate", (True,), (True, False), 8004, cache_root),
        _case("duplicate", (True,), (False, True), 8005, cache_root, close=True),
        _case("glu_linear", (False, True), (True,), 8006, cache_root, retain=True, stream=True),
        _case("shared", (False, True), (True, True, True), 8007, cache_root),
        _case("ffn", (True, True, True), (True,), 8011, cache_root, geometry=(0, 17, 38)),
        _case("ffn", (True, True, True), (True,), 8012, cache_root, geometry=(1, 17, 38)),
        _case("ffn", (True, True, True), (True,), 8013, cache_root, retain=True, stream=True, hooks=True,
              geometry=(129, 513, 1030)),
        _case("ffn", (True, True, True), (True,), 8014, cache_root, close=True, stream=True,
              geometry=(1024, 1024, 8192)),
    ]
    destination = os.environ.get("HP_FFN_EVIDENCE_DIR")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / "backward_autograd.json").write_text(json.dumps({"complete": True, "cases": rows,
            "full_ffn_accepted": False, "original_training_accepted": False, "performance_accepted": False},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_dense_backward_autograd()
