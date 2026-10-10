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
"""Actual-device dense FFN outputs, gradients and invocation lifecycle validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import torch
import torch_npu
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig


class _SourceFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        """Initialize a conventional bias-free source FFN.

        Args:
            hidden: Input/output channels.
            intermediate: Unpacked intermediate channels.
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


def _empty_tokens(candidate, hidden, device):
    candidate.zero_grad(set_to_none=True)
    value = torch.empty(0, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = candidate(value)
    if tuple(output.shape) != (0, hidden) or output.dtype != torch.bfloat16:
        raise AssertionError("Dense empty-token output must preserve shape and storage")
    output.float().sum().backward()
    if value.grad is None or tuple(value.grad.shape) != (0, hidden):
        raise AssertionError("Dense empty-token input gradient is missing or malformed")
    for parameter in candidate.parameters():
        if parameter.grad is None or bool(torch.count_nonzero(parameter.grad)):
            raise AssertionError("Dense empty-token weight gradients must be allocated zeros")


class _StreamCase(NamedTuple):
    """Independent caller-owned weights and packed reference state for one stream."""

    value: torch.Tensor
    reference_value: torch.Tensor
    weights: tuple[torch.Tensor, torch.Tensor]
    baseline: SwiGLUMLP
    cotangent: torch.Tensor


def _stream_case(hidden, intermediate, seed, device):
    torch.manual_seed(seed)
    source = _SourceFFN(hidden, intermediate).to(dtype=torch.bfloat16)
    baseline = SwiGLUMLP(module=source).to(device)
    weights = tuple(weight.t().contiguous().detach().requires_grad_()
                    for weight in (baseline.linear_fc1.weight, baseline.linear_fc2.weight))
    generator = torch.Generator(device="cpu").manual_seed(seed + 100)
    value = torch.randn(129, hidden, generator=generator, dtype=torch.bfloat16).to(device)
    cotangent = torch.randn(129, hidden, generator=generator, dtype=torch.bfloat16).to(device)
    return _StreamCase(value.detach().clone().requires_grad_(), value.detach().clone().requires_grad_(),
                       weights, baseline, cotangent)


def _stream_rows(cases, assignments, outputs, references, scratch_reuse=False):
    rows = []
    for case, lane, output, reference in zip(cases, assignments, outputs, references):
        torch.testing.assert_close(output, reference, rtol=0.02, atol=0.002)
        pairs = ((case.value.grad, case.reference_value.grad),
                 (case.weights[0].grad, case.baseline.linear_fc1.weight.grad.t()),
                 (case.weights[1].grad, case.baseline.linear_fc2.weight.grad.t()))
        for actual, expected in pairs:
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
        row = {"tokens": case.value.shape[0], "hidden": case.value.shape[1],
               "intermediate": case.weights[1].shape[0], "stream_lane": lane,
               "nondefault_stream": True, "borrowed_weights": True, "close_before_backward": True,
               "input_versions_preserved": True,
               "output_max_abs": float((output.float() - reference.float()).abs().max()),
               "gradient_max_abs": [float((actual.float() - expected.float()).abs().max())
                                    for actual, expected in pairs]}
        if scratch_reuse:
            row.update(scratch_reuse=True, private_scratch_fallback=lane == 2)
        rows.append(row)
    return rows


def _stream_backward(cases, assignments, streams, outputs, references, initial):
    for case, lane, output, reference in zip(cases, assignments, outputs, references):
        with torch.npu.stream(streams[lane]):
            output.backward(case.cotangent)
            reference.backward(case.cotangent)
    for stream in streams:
        initial.wait_stream(stream)
    initial.synchronize()


def _alternate_streams(hidden, intermediate, device):
    candidate = MegaFFN(hidden, intermediate, create_parameters=False,
                       execution=DenseExecutionConfig(backend="resident_tiles"))
    cases = [_stream_case(hidden, intermediate, seed, device) for seed in (701, 702)]
    initial = torch.npu.current_stream(device)
    streams = [torch.npu.Stream(device=device) for _ in cases]
    outputs, references = [], []
    versions = [tuple(value._version for value in (case.value, *case.weights)) for case in cases]
    try:
        for case, stream in zip(cases, streams):
            stream.wait_stream(initial)
            with torch.npu.stream(stream):
                outputs.append(candidate(case.value, weights=case.weights))
                references.append(case.baseline(case.reference_value))
        if [tuple(value._version for value in (case.value, *case.weights)) for case in cases] != versions:
            raise AssertionError("Dense resident forward must preserve caller input/weight versions")
        payloads = candidate.execution_manifest()
        candidate.close()
        assignments = tuple(range(len(streams)))
        _stream_backward(cases, assignments, streams, outputs, references, initial)
        rows = _stream_rows(cases, assignments, outputs, references)
        return rows, payloads
    finally:
        candidate.close()
        for stream in streams:
            initial.wait_stream(stream)
        initial.synchronize()


def _scratch_reuse(hidden, intermediate, device):
    candidate = MegaFFN(hidden, intermediate, create_parameters=False,
                       execution=DenseExecutionConfig(backend="resident_tiles"))
    cases = [_stream_case(hidden, intermediate, seed, device) for seed in range(801, 807)]
    initial = torch.npu.current_stream(device)
    streams = [torch.npu.Stream(device=device) for _ in range(3)]
    assignments = (0, 1, 0, 2, 1, 0)
    outputs, references = [], []
    versions = [tuple(value._version for value in (case.value, *case.weights)) for case in cases]
    try:
        for case, lane in zip(cases, assignments):
            stream = streams[lane]
            stream.wait_stream(initial)
            with torch.npu.stream(stream):
                outputs.append(candidate(case.value, weights=case.weights))
                references.append(case.baseline(case.reference_value))
        # Inspect actual storage ownership independently of numerical agreement.
        executable = candidate._executables[(129, device)]  # pylint: disable=protected-access
        statistics = executable.scratch.statistics()
        if (statistics["cached_streams"], statistics["allocations"], statistics["reuses"]) != (2, 3, 3):
            raise AssertionError(f"Scratch cache did not stay bounded while reusing streams: {statistics}")
        if [tuple(value._version for value in (case.value, *case.weights)) for case in cases] != versions:
            raise AssertionError("Scratch reuse changed caller input/weight versions")
        payloads = candidate.execution_manifest()
        candidate.close()
        if executable.scratch.statistics()["cached_bytes"] != 0:
            raise AssertionError("Dense close retained cached scratch storage")
        _stream_backward(cases, assignments, streams, outputs, references, initial)
        rows = _stream_rows(cases, assignments, outputs, references, scratch_reuse=True)
        return rows, payloads, statistics
    finally:
        candidate.close()
        for stream in streams:
            initial.wait_stream(stream)
        initial.synchronize()


def test_dense_ffn_training() -> None:
    """Compare the native AST module to the existing packed training implementation."""
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    rows, payloads = [], []
    for hidden, intermediate in ((64, 128), (128, 256), (80, 131)):
        torch.manual_seed(17)
        source = _SourceFFN(hidden, intermediate).to(dtype=torch.bfloat16)
        candidate = MegaFFNAdapter(module=source,
                                   context={"mega_ffn_execution": DenseExecutionConfig(backend="resident_tiles")})
        candidate = candidate.to(device)
        baseline = SwiGLUMLP(module=source).to(device)
        _empty_tokens(candidate, hidden, device)
        rows.append({"tokens": 0, "hidden": hidden, "intermediate": intermediate,
                     "output_max_abs": 0.0, "gradient_max_abs": [0.0, 0.0, 0.0]})
        for tokens in (1, 7, 129):
            generator = torch.Generator(device="cpu").manual_seed(100 + tokens)
            value = torch.randn(tokens, hidden, generator=generator, dtype=torch.bfloat16).to(device)
            x = value.detach().clone().requires_grad_()
            reference_x = value.detach().clone().requires_grad_()
            candidate.zero_grad(set_to_none=True)
            baseline.zero_grad(set_to_none=True)
            output, expected = candidate(x), baseline(reference_x)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.002)
            cotangent = torch.randn(tokens, hidden, generator=generator, dtype=torch.bfloat16).to(device)
            output.backward(cotangent)
            expected.backward(cotangent)
            pairs = ((x.grad, reference_x.grad),
                     (candidate.gate_up.grad, baseline.linear_fc1.weight.grad.t()),
                     (candidate.down.grad, baseline.linear_fc2.weight.grad.t()))
            for actual_grad, expected_grad in pairs:
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0.02, atol=0.002)
            rows.append({"tokens": tokens, "hidden": hidden, "intermediate": intermediate,
                         "output_max_abs": float((output.float() - expected.float()).abs().max()),
                         "gradient_max_abs": [float((a.float() - b.float()).abs().max()) for a, b in pairs]})
        candidate.zero_grad(set_to_none=True)
        x = torch.randn(7, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
        first = candidate(x)
        second = checkpoint(candidate, x, use_reentrant=False)
        second.float().square().mean().backward(retain_graph=True)
        payloads.extend(candidate.execution_manifest())
        candidate.close()
        first.float().square().mean().backward()
        torch.npu.synchronize()
        if not bool(torch.isfinite(x.grad).all()):
            raise AssertionError("Dense FFN repeated-forward/checkpoint/close lifecycle produced nonfinite gradients")
        stream_rows, stream_payloads = _alternate_streams(hidden, intermediate, device)
        rows.extend(stream_rows)
        payloads.extend(stream_payloads)
    scratch_rows, scratch_payloads, scratch_statistics = _scratch_reuse(64, 128, device)
    rows.extend(scratch_rows)
    payloads.extend(scratch_payloads)
    destination = os.environ.get("HP_FFN_EVIDENCE_DIR")
    if destination:
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "component.json").write_text(json.dumps({"torch": torch.__version__,
                                                            "torch_npu": torch_npu.__version__,
                                                            "cases": rows, "native_payloads": payloads,
                                                            "scratch_reuse": scratch_statistics},
                                                           indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_dense_ffn_training()
