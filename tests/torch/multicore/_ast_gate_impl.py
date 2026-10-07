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
"""Single-card native acceptance against independent CPU and NPU Torch references."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu  # pylint: disable=unused-import

import hyper_parallel
from hyper_parallel.core.multicore.frontend.examples.gate_route import _route
from hyper_parallel.core.multicore.ir.schedule import HardwareSpec

_RTOL = 2.0e-4
_ATOL = 2.0e-5
_CASES = (
    (1, 4, 1, 2.5), (7, 16, 3, 2.5), (49, 33, 1, 1.0), (49, 64, 8, 2.5),
    (65, 128, 2, -1.5), (257, 256, 8, 2.5), (8193, 64, 3, 2.5), (32, 16, 3, 0.0),
)


def _reference(logits, bias, count, scale):
    scores = torch.sqrt(torch.nn.Softplus()(logits))
    indices = torch.topk(scores + bias.detach(), count, dim=-1, sorted=False).indices
    selected = torch.gather(scores, -1, indices)
    if count > 1:
        selected = selected / (selected.sum(-1, keepdim=True) + 1.0e-20)
    return selected * scale, indices


def _loss(logits, weights, indices):
    coefficients = torch.sin(torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32))
    return (weights * coefficients[indices]).sum() + 0.03125 * logits.square().sum()


def _check(actual, expected):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=_RTOL, atol=_ATOL)
    return float((actual.detach().cpu() - expected.detach().cpu()).abs().max())


def _plan(tokens, experts, count, scale):
    topology = HardwareSpec(torch.npu.get_device_properties(0).vector_core_num)
    return _route.plan({"T": tokens, "E": experts}, topology, k=count, scale=scale)


def _case(tokens, experts, count, scale, seed):
    generator = torch.Generator().manual_seed(seed)
    logits_cpu = (torch.randn(tokens, experts, generator=generator) * 2).requires_grad_()
    bias_cpu = torch.randn(experts, generator=generator) * 0.3
    expected_weights, expected_indices = _reference(logits_cpu, bias_cpu, count, scale)
    expected_loss = _loss(logits_cpu, expected_weights, expected_indices)
    expected_loss.backward()
    logits = logits_cpu.detach().npu().requires_grad_()
    bias = bias_cpu.npu().requires_grad_()
    plan = _plan(tokens, experts, count, scale)
    executable = plan.materialize()
    weights, indices = executable(logits, bias, profile=True)
    torch.testing.assert_close(indices.cpu().sort(-1).values, expected_indices.sort(-1).values, rtol=0, atol=0)
    cpu_in_native_order = torch.sqrt(torch.nn.Softplus()(logits_cpu.detach())).gather(-1, indices.cpu())
    if count > 1:
        cpu_in_native_order = cpu_in_native_order / (cpu_in_native_order.sum(-1, keepdim=True) + 1.0e-20)
    weight_error = _check(weights, cpu_in_native_order * scale)
    _loss(logits, weights, indices).backward()
    gradient_error = _check(logits.grad, logits_cpu.grad)
    if bias.grad is not None or indices.requires_grad:
        raise AssertionError("Selection bias and expert indices must not receive gradients")
    logits_ref = logits.detach().clone().requires_grad_()
    reference_weights, reference_indices = _reference(logits_ref, bias, count, scale)
    _loss(logits_ref, reference_weights, reference_indices).backward()
    npu_reference_error = _check(logits.grad, logits_ref.grad)
    profiles = executable.take_profiles()
    record_counts = [len(profile.records()) for profile in profiles]
    rows_per_worker = plan.schedule.rows_per_worker
    expected_counts = [
        len(plan.schedule.partitions) * 10,
        ((tokens + rows_per_worker - 1) // rows_per_worker) * (2 if count == 1 else 11),
    ]
    if record_counts != expected_counts:
        raise AssertionError(f"Profile coverage mismatch: {record_counts}, expected {expected_counts}")
    return {
        "tokens": tokens, "experts": experts, "top_k": count, "scale": scale,
        "max_abs_weights_cpu": weight_error, "max_abs_grad_cpu": gradient_error,
        "max_abs_grad_npu_torch": npu_reference_error, "profile_records": record_counts,
        "build_fingerprint": executable.manifest["build_fingerprint"],
    }


def _reuse_and_streams():
    plan = _plan(49, 64, 3, 2.5)
    executable = plan.materialize()
    bias = torch.linspace(-0.2, 0.2, 64, device="npu:0", requires_grad=True)
    streams = (torch.npu.Stream(), torch.npu.Stream())
    inputs = tuple(torch.randn(49, 64, device="npu:0", requires_grad=True) for _ in streams)
    outputs = []
    for stream, logits in zip(streams, inputs):
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            outputs.append(executable(logits, bias, profile=True))
    for stream in streams:
        torch.npu.current_stream().wait_stream(stream)
    sum(_loss(logits, *output) for logits, output in zip(inputs, outputs)).backward()
    for logits in inputs:
        cpu = logits.detach().cpu().requires_grad_()
        weights, indices = _reference(cpu, bias.detach().cpu(), 3, 2.5)
        _loss(cpu, weights, indices).backward()
        _check(logits.grad, cpu.grad)
    profiles = executable.take_profiles()
    if len(profiles) != 4 or len({profile.buffer.data_ptr() for profile in profiles}) != 4:
        raise AssertionError("Overlapping calls must own separate profiling buffers")
    for profile in profiles:
        if not profile.records():
            raise AssertionError("Each stream invocation must produce stage records")
    if bias.grad is not None:
        raise AssertionError("Selection bias must remain detached across overlapping calls")
    _boundary_checks(executable, bias)


def _boundary_checks(executable, bias):
    logits = torch.randn(49, 64, device="npu:0", requires_grad=True)
    weights, _ = executable(logits, bias)
    incoming = torch.ones(3, 49, device="npu:0").t()
    actual, = torch.autograd.grad(weights, logits, incoming)
    cpu = logits.detach().cpu().requires_grad_()
    expected, _ = _reference(cpu, bias.detach().cpu(), 3, 2.5)
    expected.sum().backward()
    _check(actual, cpu.grad)
    logits = torch.randn(49, 64, device="npu:0", requires_grad=True)
    executable(logits, bias, profile=True)
    (logits * 0.37).sum().backward()
    _check(logits.grad, torch.full_like(logits, 0.37))
    if len(executable.take_profiles()) != 1:
        raise AssertionError("Unused route weights must not launch backward")
    with torch.no_grad():
        executable(logits, bias)
    try:
        executable(logits.t(), bias)
    except ValueError:
        pass
    else:
        raise AssertionError("A shape mismatch must be rejected before the native call")
    if torch.npu.get_device_properties(0).vector_core_num != 48:
        try:
            _route.plan({"T": 49, "E": 64}, HardwareSpec(48), k=3, scale=2.5).materialize()
        except ValueError as error:
            if "topology" not in str(error):
                raise
        else:
            raise AssertionError("Host-only slot capacity must not impersonate the device topology")


def _projection():
    generator = torch.Generator().manual_seed(9000)
    hidden_cpu = torch.randn(49, 32, generator=generator).requires_grad_()
    weight_cpu = (torch.randn(64, 32, generator=generator) * 0.1).requires_grad_()
    bias_cpu = torch.linspace(-0.2, 0.2, 64)
    hidden = hidden_cpu.detach().npu().requires_grad_()
    weight = weight_cpu.detach().npu().requires_grad_()
    logits = hidden @ weight.t()
    executable = _plan(49, 64, 3, 2.5).materialize()
    routed = executable(logits, bias_cpu.npu())
    _loss(logits, *routed).backward()
    logits_cpu = hidden_cpu @ weight_cpu.t()
    expected = _reference(logits_cpu, bias_cpu, 3, 2.5)
    _loss(logits_cpu, *expected).backward()
    return {"hidden_grad_max_abs": _check(hidden.grad, hidden_cpu.grad),
            "projection_grad_max_abs": _check(weight.grad, weight_cpu.grad)}


def run_acceptance() -> None:
    """Check P2 forward/backward, direct logits loss, tail workers and overlapping resources."""
    torch.npu.set_device(0)
    print(f"checkout={hyper_parallel.__file__}, torch={torch.__version__}, device={torch.npu.get_device_name(0)}")
    records = []
    for index, shape in enumerate(_CASES):
        record = _case(*shape, 1000 + index)
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)
    _reuse_and_streams()
    projection = _projection()
    torch.npu.synchronize()
    result = {
        "status": "passed", "rtol": _RTOL, "atol": _ATOL, "cases": records,
        "checkout": str(Path(hyper_parallel.__file__).resolve()), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "device": torch.npu.get_device_name(0),
        "available_aiv_workers": torch.npu.get_device_properties(0).vector_core_num,
        "projection": projection,
        "resource_checks": ["overlapping_forward", "multiple_streams", "private_profiles",
                            "noncontiguous_grad", "unused_route_grad", "no_grad", "input_rejection",
                            "topology_rejection", "projection_autograd"],
    }
    path = Path(os.environ.get("HP_AST_GATE_RESULT", "build/native/gate/acceptance.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
