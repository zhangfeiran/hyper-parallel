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

"""Fresh-process fixed, dynamic and ordinary EP benchmarks for real token workloads."""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime


def _layer(backend: str, shape: baseline.MoeShape) -> torch.nn.Module:
    """Use identical seeded local expert weights for every backend."""
    if backend == "common":
        layer = baseline.new_common_moe(shape)
    else:
        option = "local_num_tokens" if backend == "fixed" else "max_local_num_tokens"
        layer = MegaMoeExperts(**{option: shape.local_num_tokens}, hidden_size=shape.hidden_size,
                               intermediate_size=shape.intermediate_size, num_experts=shape.num_experts,
                               top_k=shape.top_k, ep_size=shape.ep_size,
                               ep_group=dist.group.WORLD).to(device=baseline.DEVICE, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(14000 + baseline.RANK)
    first = (torch.randn(shape.local_experts, shape.hidden_size, 2 * shape.intermediate_size,
                         generator=generator) / shape.hidden_size**0.5).bfloat16().to(baseline.DEVICE)
    second = (torch.randn(shape.local_experts, shape.intermediate_size, shape.hidden_size,
                          generator=generator) / shape.intermediate_size**0.5).bfloat16().to(baseline.DEVICE)
    with torch.no_grad():
        if isinstance(layer, MegaMoeExperts):
            layer.gate_up_weight.copy_(first)
            layer.down_weight.copy_(second)
        else:
            layer.w1.to_local().copy_(first[..., :shape.intermediate_size].transpose(1, 2))
            layer.w3.to_local().copy_(first[..., shape.intermediate_size:].transpose(1, 2))
            layer.w2.to_local().copy_(second.transpose(1, 2))
    return layer


def _batch(shape: baseline.MoeShape, lengths: list[int], backend: str) -> tuple:
    """Keep padded routes balanced and mask their input, probability and upstream gradient."""
    actual = lengths[baseline.RANK]
    rows = shape.local_num_tokens if backend == "fixed" else actual
    generator = torch.Generator().manual_seed(17000 + baseline.RANK + actual)
    hidden = torch.zeros(rows, shape.hidden_size, dtype=torch.bfloat16)
    dy = torch.zeros_like(hidden)
    hidden[:actual] = torch.randn(actual, shape.hidden_size, generator=generator).bfloat16() * 0.5
    dy[:actual] = torch.randn(actual, shape.hidden_size, generator=generator).bfloat16() * 0.01
    ids = (torch.arange(rows * shape.top_k).reshape(rows, shape.top_k) + baseline.RANK * shape.top_k)
    ids = (ids % shape.num_experts).int()
    probs = torch.full((rows, shape.top_k), 1.0 / shape.top_k)
    probs[actual:] = 0
    counts = torch.bincount(ids.flatten().long(), minlength=shape.num_experts).int()
    return tuple(value.to(baseline.DEVICE) for value in (hidden, ids, probs, counts, dy))


def check_fixed_precision(args: argparse.Namespace) -> None:
    """Validate the benchmark dimensions against the fixed path before comparing timings."""
    shape = baseline.MoeShape(args.capacity, args.hidden, args.intermediate,
                              dist.get_world_size() * 6, 6, dist.get_world_size())
    start_shmem_lifetime()
    layers = [_layer(backend, shape) for backend in ("dynamic", "fixed")]
    reports = []
    for lengths in ([224 + 16 * (rank % 2) for rank in range(shape.ep_size)],
                    [args.capacity] * shape.ep_size):
        snapshots = []
        for layer, backend in zip(layers, ("dynamic", "fixed")):
            layer.zero_grad(set_to_none=True)
            hidden, ids, probs, counts, dy = _batch(shape, lengths, backend)
            hidden.requires_grad_()
            probs.requires_grad_()
            output = layer(hidden, ids, probs, tokens_per_expert=counts)
            output.backward(dy)
            gradients = baseline.expert_weight_gradients(layer)
            snapshots.append([value.detach().cpu().clone() for value in
                              (output[:lengths[baseline.RANK]], hidden.grad[:lengths[baseline.RANK]],
                               probs.grad[:lengths[baseline.RANK]], *gradients)])
        maxima = []
        for index, (actual, expected) in enumerate(zip(*snapshots)):
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3,
                                       msg=lambda message: f"benchmark tensor {index}: {message}")
            maxima.append(float((actual.float() - expected.float()).abs().max()))
        reports.append({"lengths": lengths, "max_abs_error": maxima})
    for layer in layers:
        layer.close()
    gathered = [None] * shape.ep_size
    dist.all_gather_object(gathered, reports)
    if baseline.RANK == 0:
        Path(args.output).write_text(json.dumps({"status": "passed", "shape": vars(shape),
                                                "rank_results": gathered}, indent=2) + "\n", encoding="utf-8")
    dist.destroy_process_group()


def benchmark(args: argparse.Namespace) -> None:
    """Report rank-max host F+B, cold start, allocator peaks and real token throughput."""
    shape = baseline.MoeShape(args.capacity, args.hidden, args.intermediate,
                              dist.get_world_size() * 6, 6, dist.get_world_size())
    if args.backend != "common":
        start_shmem_lifetime()
    layer = _layer(args.backend, shape)
    patterns = {
        "short_ragged": [[224 + 16 * (rank % 2) for rank in range(shape.ep_size)]],
        "full": [[args.capacity] * shape.ep_size],
        "mixed": [[size + rank % 2 for rank in range(shape.ep_size)] for size in (129, 224, 513)]
                 + [[args.capacity] * shape.ep_size],
    }
    results = []
    for name, lengths in patterns.items():
        batches = [_batch(shape, vector, args.backend) for vector in lengths]
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        cold = baseline._timed_fwd_bwd(layer, *batches[0], validate_gradients=True)
        for step in range(args.warmup):
            baseline._timed_fwd_bwd(layer, *batches[step % len(batches)])
        times = [baseline._timed_fwd_bwd(layer, *batches[step % len(batches)])
                 for step in range(args.steps)]
        memory = torch.tensor([torch.npu.max_memory_allocated(), torch.npu.max_memory_reserved()],
                              dtype=torch.int64, device=baseline.DEVICE)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        entry = {"case": name, "length_vectors": lengths, "first_fwd_bwd_ms": cold,
                 "median_fwd_bwd_ms": statistics.median(times), "samples_ms": times,
                 "peak_allocated_reserved_bytes": memory.cpu().tolist(),
                 "shmem_heap_bytes": int(os.getenv("HYPER_PARALLEL_SHMEM_HEAP_SIZE", "0")),
                 "effective_tokens_per_second": sum(sum(lengths[index % len(lengths)])
                                                    for index in range(args.steps)) / (sum(times) / 1000)}
        results.append(entry)
        if baseline.RANK == 0:
            print(json.dumps({"backend": args.backend, **entry}), flush=True)
    if isinstance(layer, MegaMoeExperts):
        layer.close()
    if baseline.RANK == 0:
        Path(args.output).write_text(json.dumps({"backend": args.backend, "shape": vars(shape),
                                                "warmup": args.warmup, "results": results}, indent=2) + "\n",
                                    encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fixed", "dynamic", "common"), required=True)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--check-fixed", action="store_true")
    parser.add_argument("--output", required=True)
    parsed_args = parser.parse_args()
    if parsed_args.check_fixed:
        check_fixed_precision(parsed_args)
    else:
        benchmark(parsed_args)
