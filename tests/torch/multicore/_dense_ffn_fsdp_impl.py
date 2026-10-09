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
"""Two-card FSDP dense FFN gradient, changing-weight and checkpoint lifecycle validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, wait_grad_handle
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig


class _SourceFFN(nn.Module):
    def __init__(self) -> None:
        """Create the conventional weights before module replacement and sharding."""
        super().__init__()
        self.gate_proj = nn.Linear(64, 128, bias=False)
        self.up_proj = nn.Linear(64, 128, bias=False)
        self.down_proj = nn.Linear(128, 64, bias=False)


def _full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _compare_weights(candidate, baseline, gradients=False):
    for actual, expected in ((candidate.gate_up, baseline.linear_fc1.weight),
                             (candidate.down, baseline.linear_fc2.weight)):
        if gradients:
            actual, expected = actual.grad, expected.grad
        if actual is None or expected is None:
            raise AssertionError("FSDP dense FFN gradient is missing")
        torch.testing.assert_close(_full(actual), _full(expected).t(), rtol=0.02, atol=0.002)


def _train_steps(candidate, baseline, rank, device):
    optimizers = [torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False) for model in (candidate, baseline)]
    rows = []
    for step, tokens in enumerate((7, 129, 7, 1)):
        generator = torch.Generator(device="cpu").manual_seed(1000 + rank * 10 + step)
        value = torch.randn(tokens, 64, generator=generator, dtype=torch.bfloat16).to(device)
        cotangent = torch.randn(tokens, 64, generator=generator, dtype=torch.bfloat16).to(device)
        inputs = [value.detach().clone().requires_grad_() for _ in range(2)]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        outputs = [checkpoint(model, x, use_reentrant=False) if step == 2 else model(x)
                   for model, x in zip((candidate, baseline), inputs)]
        torch.testing.assert_close(*outputs, rtol=0.02, atol=0.002)
        for output in outputs:
            output.backward(cotangent)
            wait_grad_handle()
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, rtol=0.02, atol=0.002)
        _compare_weights(candidate, baseline, gradients=True)
        with SkipDTensorDispatch():
            for optimizer in optimizers:
                optimizer.step()
        _compare_weights(candidate, baseline)
        rows.append({"step": step + 1, "tokens": tokens, "recompute": step == 2, "passed": True})
    return rows


def _checkpoint_roundtrip(candidate, baseline, device):
    saved = {name: _full(parameter).detach().clone() for name, parameter in candidate.named_parameters()}
    with torch.no_grad():
        for parameter in candidate.parameters():
            parameter.to_local().zero_()
    candidate.load_state_dict(saved)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(_full(parameter), saved[name], rtol=0, atol=0)
    value = torch.randn(7, 64, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        torch.testing.assert_close(candidate(value), baseline(value), rtol=0.02, atol=0.002)


def test_dense_ffn_fully_shard() -> None:
    """Exercise normal replacement-before-FSDP with independently updated packed parameters."""
    rank, world_size = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError("Dense FSDP acceptance requires exactly two worker ranks")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group("hccl")
    candidate = None
    try:
        mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("dp",))
        torch.manual_seed(17)
        source = _SourceFFN().to(dtype=torch.bfloat16)
        candidate = MegaFFNAdapter(module=source,
                                   context={"mega_ffn_execution": DenseExecutionConfig(backend="resident_tiles")})
        candidate, baseline = candidate.to(device), SwiGLUMLP(module=source).to(device)
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      output_dtype=torch.bfloat16, cast_forward_inputs=False)
        for model in (candidate, baseline):
            fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        _compare_weights(candidate, baseline)
        rows = _train_steps(candidate, baseline, rank, device)
        _checkpoint_roundtrip(candidate, baseline, device)
        torch.npu.synchronize()
        destination = os.environ.get("HP_FFN_EVIDENCE_DIR")
        if destination:
            directory = Path(destination)
            directory.mkdir(parents=True, exist_ok=True)
            record = {"rank": rank, "world_size": world_size, "torch": torch.__version__,
                      "torch_npu": torch_npu.__version__, "scope": "BF16 FSDP SGD component lifecycle",
                      "checkpoint_roundtrip": True, "steps": rows,
                      "native_payloads": candidate.execution_manifest()}
            (directory / f"fsdp-rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finally:
        if candidate is not None:
            candidate.close()
        dist.destroy_process_group()
