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
"""Original-width hot-replica parity across refreshed weights and changing routes."""

import argparse
import json
import os
from dataclasses import asdict
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from tests.torch.models.longcat_flash._benchmark_longcat_flash import (
    _arguments, _build, _hash, _step)
from tests.torch.models.longcat_flash._compare_longcat_flash import (_compare,
                                                                     _values)
from tests.torch.multicore._mega_moe_utils import (environment_identity,
                                                   start_shmem_lifetime)


def _routes(base: tuple, step: int, args: argparse.Namespace, rank: int) -> tuple[str, torch.Tensor]:
    """Exercise hot owners, a source without real rows, balanced routes and global identity steps."""
    pattern = ("hot", "hot", "normal", "hot", "empty_source", "identity", "hot", "normal", "hot", "hot")
    kind = pattern[step % len(pattern)]
    ids = base[2].clone()
    if kind != "normal":
        owner = (step // 3) % args.ep
        real = torch.arange(8, device=ids.device) + owner * 8
        identity = torch.arange(4, device=ids.device) + args.real
        ids = torch.cat((real, identity)).expand(args.tokens, -1).clone()
    if kind == "identity" or (kind == "empty_source" and rank == 0):
        ids = (torch.arange(args.top_k, device=ids.device) + args.real).expand(args.tokens, -1).clone()
    return kind, ids.long()


def _snapshot_optional(tensors: tuple, output: torch.Tensor) -> dict:
    """Keep native values on CPU, including unused expert-gradient None states."""
    return {name: None if value is None else value.detach().cpu()
            for name, value in _values(tensors, output).items()}


def _comparison(tensors: tuple, output: torch.Tensor, expected: dict) -> dict:
    """Compare all elements with the frozen strict gate; preserve unused-gradient semantics."""
    result = {}
    for name, value in _values(tensors, output).items():
        reference = expected[name]
        if value is None or reference is None:
            result[name] = {"strict_pass": value is None and reference is None,
                            "both_unused": value is None and reference is None}
        else:
            result[name] = _compare(value, reference)
    return result


def _update(tensors: tuple, masters: dict) -> None:
    """Apply FP32 SGD master updates, skipping expert parameters unused by all ranks."""
    with torch.no_grad():
        for name, value in tensors[0].named_parameters():
            if value.grad is not None:
                masters[name].add_(value.grad.float(), alpha=-1e-5)
                value.copy_(masters[name])


def _observe_plan(engine: object, reports: list) -> object:
    """Wrap the real runtime decision without replacing the planner or transport."""
    original = engine._prepare_replica_route

    def observed(*args: object, **kwargs: object) -> object:
        """Record the materialized plan returned by the production routing path."""
        route = original(*args, **kwargs)
        if route is not None:
            reports.append({"home_only": route.home_only,
                            "transfers": [asdict(transfer) for transfer in route.plan.transfers],
                            "destination_loads": list(route.plan.destination_loads)})
        return route

    return patch.object(engine, "_prepare_replica_route", side_effect=observed)


def _build_pair(args: argparse.Namespace, rank: int, options: argparse.Namespace) -> tuple:
    """Create two independent parameter sets with identical seeds and route operands."""
    args.backend = "native"
    reference = _build(args, rank)
    candidate = _build(args, rank)
    experts = candidate[0]
    experts.close_runtime()
    experts.backend = "megamoe"
    experts.dispatch_mode = options.dispatch_mode
    experts.expert_weight_layout = options.layout
    experts.replica_slots_per_rank = options.budget
    experts.replica_granularity = "expert"
    experts.replica_transport = options.transport
    experts.configure(dist.group.WORLD, args.ep, args.top_k)
    for left, right in zip(reference[0].parameters(), experts.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    return reference, candidate


def _one_update(reference: tuple, candidate: tuple, masters: tuple, step: int,
                args: argparse.Namespace, rank: int) -> dict:
    """Check one changed route and both independent optimizer updates against native GMM."""
    kind, ids = _routes(reference, step, args, rank)
    reference = reference[:2] + (ids,) + reference[3:]
    candidate = candidate[:2] + (ids,) + candidate[3:]
    expected = _snapshot_optional(reference, _step(*reference, True))
    plans = []
    with _observe_plan(candidate[0]._executor, plans):
        actual = _step(*candidate, True)
    comparison = _comparison(candidate, actual, expected)
    _update(reference, masters[0])
    _update(candidate, masters[1])
    weights = {name: _hash(value) for name, value in candidate[0].named_parameters()}
    matched = weights == {name: _hash(value) for name, value in reference[0].named_parameters()}
    active = any(plan["transfers"] and not plan["home_only"] for plan in plans)
    passed = all(row["strict_pass"] for row in comparison.values()) and matched
    if kind in ("hot", "empty_source") and not active:
        passed = False
    return {"step": step, "fixture": kind, "replicas_active": active, "plans": plans,
            "comparison": comparison, "parameter_hashes": weights,
            "parameters_match_native": matched, "strict_pass": passed}


def validate(args: argparse.Namespace, options: argparse.Namespace) -> None:
    """Keep one replica executor alive across ten updates and save every rank's terminal state.

    Args:
        args: Original-width component configuration and external output directory.
        options: Replica transport, weight layout, guest budget and update count.
    """
    if options.updates < 1 or options.budget < 1:
        raise ValueError("replica acceptance requires positive updates and guest budget")
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep or args.real != args.ep * 8 or args.zero != args.ep * 4:
        raise ValueError("replica acceptance must retain eight real and four identity experts per EP rank")
    start_shmem_lifetime()
    reference, candidate = _build_pair(args, rank, options)
    masters = tuple({name: value.detach().float().clone() for name, value in tensors[0].named_parameters()}
                    for tensors in (reference, candidate))
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(options.updates):
        records.append(_one_update(reference, candidate, masters, step, args, rank))
        status = torch.tensor([records[-1]["strict_pass"]], device="npu", dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        records[-1]["all_ranks_strict_pass"] = bool(status.item())
        (args.output / f"progress-rank{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
        if not records[-1]["all_ranks_strict_pass"]:
            break
    resources = candidate[0]._executor._resource_group.resources
    resolved_queue = resources.last_execution_plan.spec._gradient_queue_order
    candidate[0].close_runtime()
    reference[0].close_runtime()
    report = {"rank": rank, "pid": os.getpid(), "dispatch_mode": options.dispatch_mode,
              "layout": options.layout, "budget": options.budget, "granularity": "expert",
              "resolved_gradient_queue": resolved_queue,
              "transport": options.transport, "records": records,
              "shape": {key: getattr(args, key) for key in
                        ("ep", "tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "strict_pass": len(records) == options.updates
              and all(row["all_ranks_strict_pass"] for row in records),
              "environment": environment_identity(), "shmem_after_close": shmem.debug_state(),
              "scope": "Expert component, FP32 SGD masters and BF16 compute; no full-model Trainer acceptance"}
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if not report["strict_pass"]:
        raise RuntimeError("LongCat whole-expert replicas failed strict multi-update native GMM parity")


def main() -> None:
    """Parse replica controls before the shared original-width benchmark arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--layout", choices=("EOI", "EIO"), default="EOI")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--transport", default="shmem_signal_kernel_gradient_bands")
    options, remaining = parser.parse_known_args()
    # Reuse the benchmark parser without changing the launcher's process arguments.
    with patch("sys.argv", ["longcat_replica_validation", *remaining]):
        args = _arguments()
    validate(args, options)


if __name__ == "__main__":
    main()
