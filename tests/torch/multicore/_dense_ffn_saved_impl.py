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
"""在真实 NPU 上验证正式 resident autograd 的保存缓存生命周期。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.runtime.dense import DenseSpec
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig, materialize_dense


def _pointers(output):
    saved = output.grad_fn.saved_tensors[3:]
    if len(saved) != 2 or any(value._base is not None for value in saved):  # pylint: disable=protected-access
        raise AssertionError("保存值必须为两个不暴露缓存基准的独立叶张量")
    return tuple(value.data_ptr() for value in saved)


def _checked_gradients(output, inputs, cotangent, *, retain_graph=False):
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    value, gate_up, down = reference_inputs
    expected = torch.mm(torch.ops.npu.npu_swiglu.default(torch.mm(value, gate_up), -1), down)
    actual = torch.autograd.grad(output, inputs, cotangent, retain_graph=retain_graph)
    reference = torch.autograd.grad(expected, reference_inputs, cotangent)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
    for gradient, target in zip(actual, reference):
        torch.testing.assert_close(gradient, target, rtol=0.02, atol=0.002)
    return actual


def _overlap_and_reuse(executable, inputs, cotangent, streams):
    first, second = streams
    with torch.npu.stream(first):
        outputs = [executable(*inputs) for _ in range(3)]
        pointers = [_pointers(output) for output in outputs]
        if len({pointer[0] for pointer in pointers}) != 3:
            raise AssertionError("重叠前向覆盖了仍需用于反向的保存值")
        statistics = executable.saved_pool.statistics()
        if statistics["cached_invocations"] != 2 or statistics["private_fallbacks"] != 1:
            raise AssertionError(f"保存缓存未遵循双槽位边界和私有回退: {statistics}")
        for output in outputs:
            _checked_gradients(output, inputs, cotangent)
        del outputs
    first.synchronize()
    before = executable.saved_pool.statistics()["stream_transfers"]
    with torch.npu.stream(second):
        reused = executable(*inputs)
        if _pointers(reused) != pointers[0]:
            raise AssertionError("最后存储别名释放后未复用原始保存地址")
        _checked_gradients(reused, inputs, cotangent)
    second.synchronize()
    if executable.saved_pool.statistics()["stream_transfers"] <= before:
        raise AssertionError("保存缓存跨流复用未建立事件交接")
    return {"overlapping_forwards": 3, "actual_pointer_reuse": True, "cross_stream_handoff": True,
            "bounded_cache": statistics}


def _retain_and_alias(executable, inputs, cotangent):
    output = executable(*inputs)
    pointers = _pointers(output)
    first = _checked_gradients(output, inputs, cotangent, retain_graph=True)
    blocked = executable(*inputs)
    if _pointers(blocked)[0] == pointers[0]:
        raise AssertionError("retain_graph 保存值存活时发生缓存复用")
    second = _checked_gradients(output, inputs, cotangent)
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _checked_gradients(blocked, inputs, cotangent)
    del output, blocked
    output = executable(*inputs)
    alias = output.grad_fn.saved_tensors[3].detach()
    snapshot = alias.clone()
    _checked_gradients(output, inputs, cotangent)
    blocked = executable(*inputs)
    if _pointers(blocked)[0] == alias.data_ptr():
        raise AssertionError("外部 detach 别名未阻止缓存复用")
    torch.testing.assert_close(alias, snapshot, rtol=0, atol=0)
    _checked_gradients(blocked, inputs, cotangent)
    return {"retain_graph_guard": True, "external_alias_guard": True}


def _saved_hooks(executable, inputs, cotangent):
    held = []
    with torch.autograd.graph.saved_tensors_hooks(lambda value: held.append(value.detach()) or held[-1],
                                                lambda value: value):
        output = executable(*inputs)
    _checked_gradients(output, inputs, cotangent)
    blocked = executable(*inputs)
    if _pointers(blocked)[0] == held[3].data_ptr():
        raise AssertionError("保存钩子持有的外部存储别名未阻止复用")
    _checked_gradients(blocked, inputs, cotangent)
    return {"saved_hooks_guard": True}


def _offload_generation(executable, inputs, cotangent, streams):
    first, second = streams
    pointers = []

    def _pack(value):
        if tuple(value.shape) == (129, 262):
            pointers.append(value.data_ptr())
        return value.detach().cpu()

    with torch.npu.stream(first), torch.autograd.graph.saved_tensors_hooks(
            _pack, lambda value: value.to(inputs[0].device)):
        offloaded = executable(*inputs)
    first.synchronize()
    if len(pointers) != 1:
        raise AssertionError("CPU 保存钩子未捕获唯一打包保存值")
    with torch.npu.stream(second):
        newer = executable(*inputs)
        if _pointers(newer)[0] != pointers[0]:
            raise AssertionError("CPU 卸载释放最后别名后未复用缓存槽位")
    before = executable.saved_pool.statistics()["stream_transfers"]
    with torch.npu.stream(first):
        _checked_gradients(offloaded, inputs, cotangent)
    if executable.saved_pool.statistics()["stream_transfers"] != before:
        raise AssertionError("旧 CPU 保存代次修改了新调用的缓存消费流")
    with torch.npu.stream(second):
        _checked_gradients(newer, inputs, cotangent)
    first.synchronize()
    second.synchronize()
    return {"cpu_offload_guard": True, "expired_generation_guard": True}


def test_dense_ffn_saved_buffers() -> None:
    """验证 resident 保存值的真实地址、跨流、钩子和关闭后反向。"""
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    generator = torch.Generator(device="cpu").manual_seed(802)
    inputs = tuple((torch.randn(shape, dtype=torch.bfloat16, generator=generator) * 0.05).to(device).requires_grad_()
                   for shape in ((129, 80), (80, 262), (131, 80)))
    cotangent = torch.randn((129, 80), dtype=torch.bfloat16, generator=generator).to(device)
    plan = dense_ffn.plan(DenseSpec({"T": 129, "H": 80, "PackedI": 262, "I": 131}), intermediate_size=131)
    executable = materialize_dense(plan, device, execution=DenseExecutionConfig(backend="resident_tiles"))
    initial = torch.npu.current_stream(device)
    streams = (torch.npu.Stream(device=device), torch.npu.Stream(device=device))
    for stream in streams:
        stream.wait_stream(initial)
    try:
        record = _overlap_and_reuse(executable, inputs, cotangent, streams)
        for stream in streams:
            initial.wait_stream(stream)
        record.update(_retain_and_alias(executable, inputs, cotangent))
        record.update(_saved_hooks(executable, inputs, cotangent))
        initial.synchronize()
        for stream in streams:
            stream.wait_stream(initial)
        record.update(_offload_generation(executable, inputs, cotangent, streams))
        with torch.npu.stream(streams[1]):
            pending = executable(*inputs)
        record["statistics_before_close"] = executable.saved_pool.statistics()
        executable.close()
        if executable.saved_pool.statistics()["cached_bytes"] != 0:
            raise AssertionError("关闭后仍持有保存缓存根分配")
        with torch.npu.stream(streams[1]):
            _checked_gradients(pending, inputs, cotangent)
        record.update(complete=True, close_before_backward=True, native=executable.native_identity,
                      torch=torch.__version__, torch_npu=torch_npu.__version__)
        destination = os.environ.get("HP_FFN_EVIDENCE_DIR")
        if destination:
            directory = Path(destination)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "saved_buffers.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8")
    finally:
        executable.close()
        for stream in streams:
            initial.wait_stream(stream)
        initial.synchronize()


if __name__ == "__main__":
    test_dense_ffn_saved_buffers()
