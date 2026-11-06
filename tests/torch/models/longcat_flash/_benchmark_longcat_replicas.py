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
"""Fresh-process original-width LongCat B0/B2 and native component measurements."""

import argparse
import json
import os
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import \
    LongcatFlashTopkRouter

from hyper_parallel.core.multicore import shmem
from tests.torch.models.longcat_flash._benchmark_longcat_flash import (
    _arguments, _build, _hash, _operator_census, _phase, _step)
from tests.torch.models.longcat_flash._validate_longcat_replicas import \
    _observe_plan
from tests.torch.multicore._mega_moe_utils import (environment_identity,
                                                   start_shmem_lifetime)


def _controlled_routes(tensors: tuple, args: argparse.Namespace) -> tuple:
    """Bias only expert selection while retaining the router's original mixture weights."""
    experts, hidden, _, _, upstream = tensors
    config = LongcatFlashConfig.from_dict({"hidden_size": args.hidden,
                                         "expert_ffn_hidden_size": args.intermediate,
                                         "n_routed_experts": args.real, "zero_expert_num": args.zero,
                                         "moe_topk": args.top_k})
    torch.manual_seed(12000)
    router = LongcatFlashTopkRouter(config).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        router.e_score_correction_bias[:args.real // args.ep].fill_(100)
        router.e_score_correction_bias[args.real:args.real + 4].fill_(100)
        probabilities, ids = router(hidden)[-2:]
    return experts, hidden, ids, probabilities.detach().requires_grad_(), upstream


def _prepare(args: argparse.Namespace, options: argparse.Namespace, rank: int) -> tuple:
    """Build identical owner weights before selecting the implementation under test."""
    backend = args.backend
    args.backend = "native"
    tensors = _build(args, rank)
    args.backend = backend
    if options.fixture == "hot":
        tensors = _controlled_routes(tensors, args)
    if backend != "native":
        experts = tensors[0]
        experts.close_runtime()
        experts.backend = "megamoe"
        experts.dispatch_mode = backend
        experts.replica_slots_per_rank = options.budget
        experts.replica_granularity = "expert"
        experts.configure(dist.group.WORLD, args.ep, args.top_k)
    return tensors


def benchmark(args: argparse.Namespace, options: argparse.Namespace) -> None:
    """Retain all local and rank-max samples and every rank's final SHMEM state.

    Args:
        args: Deterministic original-width component and measurement configuration.
        options: Guest budget and learned or controlled-hot route fixture.
    """
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep or args.real != 8 * args.ep or args.zero != 4 * args.ep:
        raise ValueError("retain eight real and four identity experts per EP rank")
    if options.budget and args.backend == "native":
        raise ValueError("native control uses B0")
    start_shmem_lifetime()
    tensors = _prepare(args, options, rank)
    experts = tensors[0]
    identities = {name: _hash(value) for name, value in experts.named_parameters()}
    identities.update({name: _hash(value) for name, value in
                       zip(("hidden", "ids", "probabilities", "upstream"), tensors[1:])})
    plans = []
    observer = _observe_plan(experts._executor, plans) if options.budget else nullcontext()
    with observer, _operator_census(args.backend == "native") as operators:
        _step(*tensors, True)
    phases = {name: _phase(args, tensors, backward) for name, backward in
              (("forward", False), ("forward_backward", True))}
    experts.close_runtime()
    report = {"rank": rank, "pid": os.getpid(), "backend": args.backend, "budget": options.budget,
              "granularity": "expert", "fixture": options.fixture, "identities": identities,
              "plans": plans, "native_operator_census": operators, "phases": phases,
              "shape": {key: getattr(args, key) for key in
                        ("ep", "tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "environment": environment_identity(), "shmem_after_close": shmem.debug_state(),
              "scope": "Expert component; fixed learned or controlled-hot routes; no optimizer"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if options.budget and options.fixture == "hot" and not any(plan["transfers"] for plan in plans):
        raise RuntimeError("controlled hot replicas did not activate")


def main() -> None:
    """Parse replica controls before the common deterministic benchmark arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--fixture", choices=("normal", "hot"), default="normal")
    options, remaining = parser.parse_known_args()
    with patch("sys.argv", ["longcat_replica_benchmark", *remaining]):
        args = _arguments()
    benchmark(args, options)


if __name__ == "__main__":
    main()
