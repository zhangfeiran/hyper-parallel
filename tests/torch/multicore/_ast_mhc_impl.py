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
"""Single-card MHC acceptance with the independent pinned Torch semantic oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu  # pylint: disable=unused-import
from torch.utils.checkpoint import checkpoint

import hyper_parallel
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.modules.mega_mhc.module import HyperMegaMhc
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.torch.multicore._ast_mhc_reference import torch_mega_mhc

_INPUT_NAMES = ("residual", "previous_output", "previous_pre", "previous_post", "previous_matrix",
                "phi", "alpha", "bias", "norm_weight")
_OUTPUT_NAMES = ("new_residual", "next_pre", "next_post", "next_matrix", "block_input")


def _inputs(tokens, hidden, seed):
    torch.manual_seed(seed)
    values = (torch.randn(tokens, 4, hidden, dtype=torch.bfloat16),
              torch.randn(tokens, hidden, dtype=torch.bfloat16),
              torch.sigmoid(torch.randn(tokens, 4)), 2 * torch.sigmoid(torch.randn(tokens, 4)),
              torch.softmax(torch.randn(tokens, 4, 4), dim=-1),
              torch.randn(24, 4 * hidden) * (4 * hidden) ** -0.5,
              torch.ones(3), torch.zeros(24), torch.ones(hidden, dtype=torch.bfloat16))
    return tuple(value.requires_grad_() for value in values)


def _oracle(values):
    return torch_mega_mhc(values[1], values[0], *values[2:])


def _metric(actual, expected):
    actual, expected = actual.detach().cpu().float().flatten(), expected.detach().cpu().float().flatten()
    if not torch.isfinite(actual).all():
        raise AssertionError("MHC produced nonfinite values")
    error = (actual - expected).norm().item() / max(expected.norm().item(), 1e-12)
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()  # pylint: disable=not-callable
    if expected.norm().item() == 0:
        cosine = 1.0 if actual.norm().item() == 0 else 0.0
    metrics = {"max_abs": (actual - expected).abs().max().item(), "relative_l2": error, "cosine": cosine}
    if error > 0.02 or cosine < 0.999:
        raise AssertionError(f"MHC original semantic precision gate failed: {metrics}")
    return metrics


def _profiles(executable, root, label):
    records = tuple(record for profile in executable.take_profiles() for record in profile.records())
    directions = {record["direction"] for record in records}
    if directions != {"forward", "backward"}:
        raise AssertionError(f"Incomplete MHC profile directions: {directions}")
    for direction, image in (("forward", executable.plan.forward), ("backward", executable.plan.backward)):
        seen = {record["stage"] for record in records if record["direction"] == direction}
        if seen != {name for name, _, _ in image.stages}:
            raise AssertionError(f"Incomplete MHC stages: {seen}")
    (root / f"{label}-cycles.json").write_text(json.dumps(records, indent=2) + "\n")
    return {"records": len(records), "dropped": 0, "directions": sorted(directions)}


def _case(tokens, hidden, tile, seed, root):
    reference = _inputs(tokens, hidden, seed)
    values = tuple(value.detach().npu().requires_grad_() for value in reference)
    expected = _oracle(reference)
    gradients = tuple(torch.randn_like(value) * 0.1 for value in expected)
    cores = torch.npu.get_device_limit(0)["cube_core_num"]
    executable = mhc_boundary.plan(MhcSpec(tokens, hidden, cores, tile, tile)).materialize()
    actual = executable(*values, profile=True)
    forward = {name: _metric(output, oracle) for name, output, oracle in zip(_OUTPUT_NAMES, actual, expected)}
    torch.autograd.backward(expected, gradients)
    torch.autograd.backward(actual, tuple(gradient.npu() for gradient in gradients))
    backward = {name: _metric(value.grad, oracle.grad) for name, value, oracle in zip(_INPUT_NAMES, values, reference)}
    profiling = _profiles(executable, root, f"t{tokens}-h{hidden}-tile{tile}")
    executable.close()
    return {"tokens": tokens, "hidden": hidden, "tile": tile, "forward": forward,
            "backward": backward, "profile": profiling}


def _lifecycle(root):
    reference = _inputs(128, 128, 72)
    cores = torch.npu.get_device_limit(0)["cube_core_num"]
    plan = mhc_boundary.plan(MhcSpec(128, 128, cores))
    executable = plan.materialize()
    streams = (torch.npu.Stream(), torch.npu.Stream())
    initial = torch.npu.current_stream()
    native_values = [tuple(value.detach().npu().requires_grad_() for value in reference) for _ in streams]
    outputs = []
    for stream, values in zip(streams, native_values):
        stream.wait_stream(initial)
        with torch.npu.stream(stream):
            outputs.append(executable(*values))
    executable.close()
    for stream in streams:
        initial.wait_stream(stream)
    upstream = torch.randn_like(reference[1]) * 0.1
    for values, output in zip(native_values, outputs):
        torch.autograd.backward(output[-1], upstream.npu())
        oracle_values = tuple(value.detach().clone().requires_grad_() for value in reference)
        torch.autograd.backward(_oracle(oracle_values)[-1], upstream)
        for native, oracle in zip(values, oracle_values):
            _metric(native.grad, torch.zeros_like(oracle) if oracle.grad is None else oracle.grad)
    with torch.no_grad():
        try:
            executable(*native_values[0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("Closed MHC accepted a new forward")
    model = HyperMegaMhc(128, device="npu:0")
    values = tuple(value.detach().npu().requires_grad_() for value in reference[:5])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    before = model.phi.detach().clone()  # pylint: disable=not-callable
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = checkpoint(lambda *args: model(args[1], args[0], *args[2:]), *values, use_reentrant=False)
        sum(value.float().square().mean() for value in output).backward()
        optimizer.step()
    if torch.equal(before, model.phi):
        raise AssertionError("MHC optimizer did not update caller-owned parameters")
    if model.take_profiles():
        raise AssertionError("Disabled profiling created buffers")
    model.close()
    data = {"pending_forwards": 2, "streams": 2, "closed_pending_backward": True,
            "missing_output_gradients": True, "checkpoint_steps": 2, "optimizer_updated": True,
            "disabled_profile_buffers": 0}
    (root / "lifecycle.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def run_acceptance() -> None:
    """Validate the fixed native MHC numerical and ownership contracts.

    Evidence is written to HP_AST_MHC_RESULTS when specified.
    """
    torch.npu.set_device(0)
    root = Path(os.environ.get("HP_AST_MHC_RESULTS", "/tmp/hp-ast-mhc-results"))
    root.mkdir(parents=True, exist_ok=True)
    cases = ((48, 128, 32),) if os.environ.get("HP_AST_MHC_SMOKE") else (
        (40, 128, 32), (41, 128, 32), (129, 512, 64), (2593, 128, 32), (4097, 128, 64),
        (40, 5760, 32), (12000, 128, 96))
    records = []
    for index, (tokens, hidden, tile) in enumerate(cases):
        record = _case(tokens, hidden, tile, 17 + index, root)
        records.append(record)
        print(json.dumps(record), flush=True)
    data = {"checkout": hyper_parallel.__file__, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "precision_gate": {"relative_l2_max": 0.02, "cosine_min": 0.999}, "cases": records}
    if not os.environ.get("HP_AST_MHC_SMOKE"):
        data["lifecycle"] = _lifecycle(root)
    (root / "acceptance.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    run_acceptance()
