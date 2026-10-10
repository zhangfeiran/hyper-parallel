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
"""真实设备上的完整 K 权重梯度、原始 SDK SwiGLUGrad 和事件汇合。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy
from hyper_parallel.core.multicore.runtime.dense_backward import ResidentDenseBackward
from hyper_parallel.core.multicore.runtime.dense_native import NativeSwiGLU


def _program(kind, rows, hidden, columns):
    signature = {"x": ml.Tensor[ml.bf16, (rows, hidden)]}
    if kind in ("linear", "linear_transpose", "glu_linear", "ffn"):
        weight_shape = (columns, hidden) if kind == "linear_transpose" else (hidden, columns)
        signature["weight"] = ml.Tensor[ml.bf16, weight_shape]
    if kind == "ffn":
        signature["down"] = ml.Tensor[ml.bf16, (columns // 2, hidden)]
    sources = {
        "linear": "def program(x, weight):\n    return ml.matmul(x, weight)\n",
        "linear_transpose": ("def program(x, weight):\n"
                             "    return ml.matmul(x, weight, transpose_right=True)\n"),
        "glu_linear": ("def program(x, weight):\n    packed = ml.matmul(x, weight)\n"
                       f"    return ml.swiglu_dense(packed, intermediate_size={columns // 2})\n"),
        "ffn": ("def program(x, weight, down):\n    packed = ml.matmul(x, weight)\n"
                f"    active = ml.swiglu_dense(packed, intermediate_size={columns // 2})\n"
                "    return ml.matmul(active, down)\n"),
        "glu": f"def program(x):\n    return ml.swiglu_dense(x, intermediate_size={hidden // 2})\n",
        "chain": (f"def program(x):\n    active = ml.swiglu_dense(x, intermediate_size={hidden // 2})\n"
                  f"    return ml.swiglu_dense(active, intermediate_size={hidden // 4})\n"),
        "duplicate": (f"def program(x):\n    active = ml.swiglu_dense(x, intermediate_size={hidden // 2})\n"
                      "    return active, active\n"),
        "fanout": (f"def program(x):\n    first = ml.swiglu_dense(x, intermediate_size={hidden // 2})\n"
                   f"    second = ml.swiglu_dense(x, intermediate_size={hidden // 2})\n"
                   "    return first, second\n"),
    }
    return mc.from_source(sources[kind], signature=signature, symbols={"ml": ml},
                          schedule=mc.TaskDAG("dense_v1")).plan(mc.DenseSpec({}))


def _reference(kind, inputs):
    value = inputs[0]
    if kind == "linear":
        return (value @ inputs[1],), ()
    if kind == "linear_transpose":
        return (value @ inputs[1].t(),), ()
    if kind in ("glu_linear", "ffn"):
        packed = value @ inputs[1]
        active = NativeSwiGLU.apply(packed)
        return ((active @ inputs[2],), (packed, active)) if kind == "ffn" else ((active,), (packed,))
    active = NativeSwiGLU.apply(value)
    if kind == "chain":
        return (NativeSwiGLU.apply(active),), (active,)
    if kind == "duplicate":
        return (active, active), ()
    if kind == "fanout":
        return (active, NativeSwiGLU.apply(value)), ()
    return (active,), ()


def _rand(shape, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.08).to(device)


def _invocation(candidate, kind, seed, noncontiguous):
    types = dict(candidate.plan.value_types)
    inputs = tuple(_rand(types[value.id].shape, seed + index, candidate.device)
                   for index, value in enumerate(candidate.plan.ir.inputs))
    if kind in ("glu", "chain"):
        inputs = (inputs[0] * 64,)
    independent = tuple(value.detach().clone().requires_grad_(needed)
                        for value, needed in zip(inputs, candidate.tile_plan.input_mask))
    with torch.no_grad():
        outputs, saved = _reference(kind, inputs)
    references, _ = _reference(kind, independent)
    cotangents = []
    for index, (output, active) in enumerate(zip(outputs, candidate.tile_plan.output_mask)):
        if not active:
            cotangents.append(None)
        elif noncontiguous:
            cotangents.append(_rand((output.shape[0], output.shape[1] * 2), seed + index + 100,
                                    candidate.device)[:, ::2])
        else:
            cotangents.append(_rand(output.shape, seed + index + 100, candidate.device))
    active_outputs = [output for output, active in zip(references, candidate.tile_plan.output_mask) if active]
    active_grads = [gradient for gradient in cotangents if gradient is not None]
    requested = [value for value in independent if value.requires_grad]
    expected_values = torch.autograd.grad(active_outputs, requested, active_grads) if active_outputs else (
        (None,) * len(requested))
    iterator = iter(expected_values)
    expected = tuple(next(iterator) if value.requires_grad else None for value in independent)
    return inputs, saved, tuple(cotangents), expected


def _compare(actual, expected, exact):
    comparisons = []
    for value, reference in zip(actual, expected):
        if reference is None:
            if value is not None:
                raise AssertionError("无需求或空余切的输入梯度必须保持 None")
            comparisons.append(None)
            continue
        torch.testing.assert_close(value, reference, rtol=0.0 if exact else 0.02, atol=0.0 if exact else 0.002)
        error = float((value.float() - reference.float()).abs().max()) if value.numel() else 0.0
        comparisons.append({"max_abs": error, "bitwise": bool(torch.equal(value, reference))})
    return comparisons


def _case(kind, rows, hidden, columns, input_mask, output_mask, policy, cache_root, seed, streams=False):
    device = torch.device("npu:0")
    candidate = ResidentDenseBackward(
        _program(kind, rows, hidden, columns), device,
        cann_root=Path(os.environ["ASCEND_HOME_PATH"]), cache_root=cache_root,
        soc=torch.npu.get_device_name(device), policy=policy, input_mask=input_mask, output_mask=output_mask)
    invocations = [_invocation(candidate, kind, seed + index * 1000, index % 2 == 1)
                   for index in range(6 if streams else 1)]
    initial = torch.npu.current_stream(device)
    lanes = [torch.npu.Stream(device=device) for _ in range(3)] if streams else [initial]
    for lane in lanes:
        lane.wait_stream(initial)
    actuals, versions = [], []
    for index, (inputs, saved, cotangents, _expected) in enumerate(invocations):
        borrowed = (*inputs, *saved, *(gradient for gradient in cotangents if gradient is not None))
        before = tuple(value._version for value in borrowed)
        with torch.npu.stream(lanes[index % len(lanes)]):
            actuals.append(candidate(inputs, saved, cotangents))
        versions.append(before == tuple(value._version for value in borrowed))
    for lane in lanes:
        initial.wait_stream(lane)
    initial.synchronize()
    if not all(versions):
        raise AssertionError("反向设备 worker 改写了借用输入、实际保存值或余切")
    exact = kind in ("glu", "chain", "duplicate", "fanout")
    comparisons = [_compare(actual, invocation[3], exact) for actual, invocation in zip(actuals, invocations)]
    return {"kind": kind, "tokens": rows, "hidden": hidden, "columns": columns,
            "input_mask": input_mask, "output_mask": output_mask, "invocations": len(invocations),
            "three_streams": streams, "borrowed_versions_preserved": True, "comparisons": comparisons,
            "scratch": candidate.scratch.statistics(), "native": candidate.native_identity}


def _write_record(rows, destination, complete):
    if not destination:
        return
    payloads = {}
    results = []
    for row in rows:
        native = dict(row["native"])
        payload = native.pop("payload")
        payloads[payload["namespace"]] = payload
        native["payload_namespace"] = payload["namespace"]
        results.append({**row, "native": native})
    record = {"complete": complete, "scope": "native_backward_components", "cases": results, "payloads": payloads,
              "resident_full_ffn_backward_accepted": False, "original_training_accepted": False,
              "performance_accepted": False}
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "backward_components.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")


def _append_case(rows, destination, *args, **kwargs):
    rows.append(_case(*args, **kwargs))
    _write_record(rows, destination, False)


def test_dense_backward_components() -> None:
    """核对独立原生 autograd，覆盖阶段汇合、别名余切、零填充和跨流生命周期。"""
    if not torch_npu.npu.is_available():
        raise RuntimeError("稠密反向组件验收需要真实 Ascend NPU")
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True)
    destination = os.environ.get("HP_FFN_EVIDENCE_DIR")
    cache_root = Path(os.environ.get("HP_FFN_CACHE_ROOT", "build/native/dense"))
    policies = (DenseTilePolicy(rows_per_tile=32, cube_workers=3, column_workers=4, prefetch_tiles=1),
                DenseTilePolicy(rows_per_tile=224, cube_workers=20, column_workers=4))
    rows = []
    for index, tokens in enumerate((0, 1, 7, 65, 129, 1024)):
        _append_case(rows, destination, "linear", tokens, 17 if tokens < 1024 else 1024,
                     37 if tokens < 1024 else 8192, (False, True), (True,),
                     policies[index % 2], cache_root, 991 + index)
    _append_case(rows, destination, "linear", 129, 17, 37, (False, True), (True,), None, cache_root, 1191)
    for index, tokens in enumerate((1, 65, 1024)):
        _append_case(rows, destination, "linear_transpose", tokens, 17 if tokens < 1024 else 1024,
                     37 if tokens < 1024 else 8192, (True, False), (True,),
                     policies[index % 2], cache_root, 1491 + index)
    for index, tokens in enumerate((0, 1, 65, 129, 1024)):
        _append_case(rows, destination, "glu_linear", tokens, 17 if tokens < 1024 else 1024,
                     38 if tokens < 1024 else 8192, (False, True), (True,),
                     policies[index % 2], cache_root, 1991 + index, streams=tokens == 129)
    for index, kind in enumerate(("glu", "chain", "duplicate", "fanout")):
        masks = ((True,),) if kind in ("glu", "chain") else (
            (True, True), (True, False), (False, True), (False, False))
        for mask in masks:
            _append_case(rows, destination, kind, 129, 76, 0, (True,), mask, policies[0], cache_root, 2991 + index)
    _append_case(rows, destination, "ffn", 129, 17, 38, (False, False, True), (True,), policies[0], cache_root, 3991)
    _write_record(rows, destination, True)


if __name__ == "__main__":
    test_dense_backward_components()
