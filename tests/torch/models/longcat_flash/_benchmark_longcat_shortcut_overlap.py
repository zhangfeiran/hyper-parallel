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
"""Original-width two-attention decoder diagnostic; excludes FSDP and Trainer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashDecoderLayer, LongcatFlashRotaryEmbedding,
)

from hyper_parallel.core.multicore import shmem
from hyper_parallel.models.longcat_flash.adapter.attention import LongcatFlashMLAAttention
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from tests.torch.models.longcat_flash._benchmark_longcat_flash import _hash
from tests.torch.models.longcat_flash._compare_longcat_flash import _compare
from tests.torch.multicore._mega_moe_utils import environment_identity, start_shmem_lifetime


class _OrderedFork(torch.autograd.Function):
    """Keep the serial dense, expert, router BF16 input-gradient fold."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor) -> tuple:
        """Expose separate graph edges without changing the normalized input.

        Args:
            ctx: Autograd context, unused because the fold saves no tensors.
            hidden: Normalized token states shared by the three branches.
        """
        del ctx
        return hidden.view_as(hidden), hidden.view_as(hidden), hidden.view_as(hidden)

    @staticmethod
    def backward(ctx: Any, dense: torch.Tensor, expert: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        """Apply the serial BF16 fold after all three streams return gradients.

        Args:
            ctx: Autograd context, unused because the fold saves no tensors.
            dense: Gradient from the dense continuation.
            expert: Gradient from the shortcut expert computation.
            router: Gradient from the shortcut router computation.
        """
        del ctx
        return (dense + expert) + router


def _build_layer(args: argparse.Namespace, rank: int) -> tuple:
    """Materialize one complete original-width logical layer with eight real experts per rank."""
    config = LongcatFlashConfig.from_dict({
        "num_layers": 1, "hidden_size": 6144, "ffn_hidden_size": 12288,
        "expert_ffn_hidden_size": 2048, "n_routed_experts": 8 * args.ep,
        "zero_expert_num": 4 * args.ep, "moe_topk": 12, "attention_dropout": 0.0,
    })
    config._attn_implementation = "flash_attention_2"
    with torch.device("meta"):
        layer = LongcatFlashDecoderLayer(config, 0)
        layer.mlp.experts = LongcatFlashMegaMoeExperts(
            module=layer.mlp.experts, context={"ep_size": args.ep, "ep": args.ep > 1},
            backend="native" if args.backend == "native" else "megamoe",
            dispatch_mode="pull" if args.backend == "pull" else "push",
            routing_backend="triton", swiglu_precision="fused", initial_token_capacity=args.tokens * 12,
        )
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.empty((12, 4096, 6144)))
        layer.mlp.experts.down_proj = torch.nn.Parameter(torch.empty((8, 6144, 2048)))
        for index in (0, 1):
            layer.self_attn[index] = LongcatFlashMLAAttention(module=layer.self_attn[index], context={})
    layer.to_empty(device="npu").to(dtype=torch.bfloat16)
    torch.npu.manual_seed(55100)
    with torch.no_grad():
        for name, value in layer.named_parameters():
            if "norm" in name:
                value.fill_(1)
            elif name.endswith("bias"):
                value.zero_()
            else:
                value.normal_(std=.02)
        for name, value in layer.named_buffers():
            value.fill_(args.ep if name.endswith("_checkpoint_ep_size") else 0)
        torch.npu.manual_seed(22000 + rank)
        for value in layer.mlp.experts.parameters():
            value.normal_(std=.02)
    layer.mlp.experts.configure(dist.group.WORLD, args.ep, 12)
    torch.npu.manual_seed(32000 + rank)
    hidden = torch.randn((1, args.tokens, 6144), device="npu", dtype=torch.bfloat16).requires_grad_()
    upstream = torch.randn_like(hidden)
    rotary = LongcatFlashRotaryEmbedding(config).to("npu")
    positions = torch.arange(args.tokens, device="npu").unsqueeze(0)
    return layer, hidden, upstream, rotary(hidden, positions)


def _overlapped_layer(layer: torch.nn.Module, hidden: torch.Tensor, embeddings: tuple,
                      stream: torch.npu.Stream) -> torch.Tensor:
    """Run the shortcut on a side stream and join it at the final residual add."""
    residual = hidden
    states = layer.input_layernorm[0](hidden)
    states, _ = layer.self_attn[0](hidden_states=states, position_embeddings=embeddings)
    states = residual + states
    residual = states
    normalized = layer.post_attention_layernorm[0](states)
    dense_input, expert_input, router_input = _OrderedFork.apply(normalized)
    default = torch.npu.current_stream()
    stream.wait_stream(default)
    with torch.npu.stream(stream):
        router_input.record_stream(stream)
        probability, indices = layer.mlp.router(router_input)[-2:]
    states = residual + layer.mlps[0](dense_input)
    with torch.npu.stream(stream):
        expert_input.record_stream(stream)
        shortcut = layer.mlp.experts(expert_input.reshape(-1, 6144), indices, probability).view_as(normalized)
    residual = states
    states = layer.input_layernorm[1](states)
    states, _ = layer.self_attn[1](hidden_states=states, position_embeddings=embeddings)
    states = residual + states
    residual = states
    states = layer.mlps[1](layer.post_attention_layernorm[1](states))
    states = residual + states
    default.wait_stream(stream)
    shortcut.record_stream(default)
    return states + shortcut


def _step(fixture: tuple, stream: torch.npu.Stream, mode: str) -> torch.Tensor:
    """Run one complete-layer forward and backward without an optimizer."""
    layer, hidden, upstream, embeddings = fixture
    layer.zero_grad(set_to_none=True)
    hidden.grad = None
    output = (layer(hidden, position_embeddings=embeddings) if mode == "serial"
              else _overlapped_layer(layer, hidden, embeddings, stream))
    output.backward(upstream)
    return output


def _values(fixture: tuple, output: torch.Tensor) -> dict:
    """Expose the same named tensors for every decoder variant."""
    layer, hidden, _, _ = fixture
    return {"output": output, "input_gradient": hidden.grad,
            **{name: parameter.grad for name, parameter in layer.named_parameters()}}


def _validate(fixture: tuple, stream: torch.npu.Stream, mode: str) -> dict:
    """Fail closed if any output or gradient differs from a fresh serial graph."""
    expected = {name: None if value is None else value.detach().cpu()
                for name, value in _values(fixture, _step(fixture, stream, "serial")).items()}
    observed = _values(fixture, _step(fixture, stream, mode))
    comparison = {}
    for name, value in observed.items():
        reference = expected[name]
        comparison[name] = ({"strict_pass": value is None and reference is None, "max_abs": 0.0}
                            if value is None or reference is None else _compare(value, reference))
    status = torch.tensor([all(item["strict_pass"] and item["max_abs"] == 0
                              for item in comparison.values())], device="npu", dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    return {"all_ranks_exact": bool(status.item()), "comparison": comparison}


def _measure(args: argparse.Namespace, fixture: tuple, stream: torch.npu.Stream) -> list[float]:
    """Keep all synchronized rank-maximum samples after fixed warmup."""
    for _ in range(args.warmup):
        _step(fixture, stream, args.mode)
    torch.npu.synchronize()
    dist.barrier()
    samples = []
    for _ in range(args.samples):
        start = time.perf_counter()
        _step(fixture, stream, args.mode)
        torch.npu.synchronize()
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000], device="npu")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))
    return samples


def _provenance() -> dict:
    root = Path(__file__).resolve().parents[4]
    loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                     if "/hyper_parallel/core/multicore/" in line and line.split()[-1].startswith("/")})
    if not loaded or any(not Path(path).resolve().is_relative_to(root) for path in loaded):
        raise RuntimeError(f"Foreign or missing native libraries: {loaded}")
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "hyper_parallel/core/multicore",
                                       "hyper_parallel/core/expert_parallel/hot_replica",
                                       "hyper_parallel/models/longcat_flash"]).decode().splitlines()
    return {"git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip(),
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked},
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loaded_native_libraries": loaded}


def main() -> None:
    """Validate, measure and close the explicitly selected decoder diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serial", "ordered"), required=True)
    parser.add_argument("--backend", choices=("push", "pull", "native"), default="push")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.ep, args.tokens, args.samples) <= 0 or args.warmup < 0:
        parser.error("EP, tokens and samples must be positive; warmup must be nonnegative")
    torch.use_deterministic_algorithms(True)
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("This diagnostic requires world size equal to EP")
    start_shmem_lifetime()
    fixture = _build_layer(args, rank)
    identities = {name: _hash(value) for name, value in fixture[0].named_parameters()}
    identities.update(hidden=_hash(fixture[1]), upstream=_hash(fixture[2]))
    stream = torch.npu.Stream()
    validation = _validate(fixture, stream, args.mode)
    samples = _measure(args, fixture, stream) if validation["all_ranks_exact"] else []
    provenance = _provenance()
    torch.npu.synchronize()
    fixture[0].mlp.experts.close_runtime()
    report = {"rank": rank, "mode": args.mode, "backend": args.backend, "identities": identities,
              "shape": {"ep": args.ep, "tokens": args.tokens, "hidden": 6144, "intermediate": 2048,
                        "dense_intermediate": 12288, "real": 8 * args.ep, "zero": 4 * args.ep, "top_k": 12},
              "scope": "One complete logical decoder: two MLA, two dense MLP, one shortcut MoE; no FSDP/Trainer",
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "validation": validation, "rank_max_ms": samples, "environment": environment_identity(),
              "provenance": provenance, "shmem_after_close": shmem.debug_state()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if not validation["all_ranks_exact"]:
        raise RuntimeError("Shortcut overlap changed at least one output or gradient")


if __name__ == "__main__":
    main()
