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
"""Distributed AST MoE acceptance using existing numerical and replica oracles."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.core import multicore
from hyper_parallel.core.multicore.frontend.examples.moe_region import moe_region
from hyper_parallel.core.multicore.runtime.moe_native import verify_moe_native
from tests.torch.expert_parallel import _test_hot_replica as replica
from tests.torch.multicore import _test_mega_moe as legacy


class _AstExperts(multicore.MegaMoeExperts):
    """Apply the AST entry to the existing acceptance harness's executor factories."""

    def __init__(self, **options: object) -> None:
        """Select the AST recipe for the harness's original constructor options."""
        super().__init__(program=moe_region, **options)


def _compare(actual, expected):
    errors = {}
    for field in fields(actual):
        value, reference = getattr(actual, field.name), getattr(expected, field.name)
        torch.testing.assert_close(value, reference, rtol=2e-2, atol=2e-3)
        errors[field.name] = float((value.float() - reference.float()).abs().max())
    return errors


def _region_case(mode, limit):
    size = dist.get_world_size()
    shape = legacy.MoeShape(ep_size=size, num_experts=2 * size)
    native, common = legacy.new_layers(shape, dispatch_mode=mode)
    ast = _AstExperts(local_num_tokens=shape.local_num_tokens, hidden_size=shape.hidden_size,
                      intermediate_size=shape.intermediate_size, num_experts=shape.num_experts,
                      top_k=shape.top_k, ep_size=size, ep_group=dist.group.WORLD,
                      dispatch_mode=mode, swiglu_limit=limit).to(device=legacy.DEVICE, dtype=torch.bfloat16)
    if limit is not None:
        native.close()
        native = multicore.MegaMoeExperts(
            local_num_tokens=shape.local_num_tokens, hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size, num_experts=shape.num_experts,
            top_k=shape.top_k, ep_size=size, ep_group=dist.group.WORLD,
            dispatch_mode=mode, swiglu_limit=limit).to(device=legacy.DEVICE, dtype=torch.bfloat16)
        legacy._copy_common_weights_to_mega(common, native)
    legacy._copy_common_weights_to_mega(common, ast)
    hidden, gradient = legacy.make_data(shape)
    records, fingerprint = [], None
    try:
        for pattern in ("balanced", "zero_token_experts", "skew", "single_destination", "balanced"):
            ids, probabilities, counts = legacy._make_fixed_route(shape, pattern)
            expected = legacy.run_layer(native, hidden, ids, probabilities, counts, gradient)
            actual = legacy.run_layer(ast, hidden, ids, probabilities, counts, gradient)
            errors = _compare(actual, expected)
            if limit is None:
                oracle = legacy.run_layer(common, hidden, ids, probabilities, counts, gradient)
                legacy.assert_results_close(oracle, actual)
            resources = ast._get_execution_resources(hidden)
            plan = resources.frontend_plan
            current = plan.recipe.fingerprint
            if fingerprint is not None and current != fingerprint:
                raise AssertionError("Runtime counts recompiled the semantic program")
            fingerprint = current
            records.append({"pattern": pattern, "errors_against_native": errors,
                            "capacity": resources.workspace.capacity_floor,
                            "heap_epoch": resources.heap_manager.epoch,
                            "program_fingerprint": current})
    finally:
        ast.close()
        native.close()
    return {"dispatch_mode": mode, "limit": limit, "steps": records}


def run_acceptance(mode: str, result_dir: str, *, budget: int | None = None,
                   transport: str = "p2p") -> None:
    """Validate AST numerics, dynamic counts and existing replica protocols.

    Args:
        mode: Push or pull dispatch.
        result_dir: Per-rank evidence directory.
        budget: Replica slots; None selects the five-stage numerical matrix.
        transport: Existing replica weight/gradient transport.
    """
    manifest = verify_moe_native()
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    if budget is None:
        cases = [_region_case(mode, limit) for limit in (None, 10.0, 0.25)]
    else:
        with patch.object(multicore, "MegaMoeExperts", _AstExperts):
            replica.run(mode, budget, result_dir, fp32_reference=True, replica_transport=transport,
                        top_k=8 if transport == "shmem_signal_kernel_gradient" else 2)
        cases = [{"budget": budget, "transport": transport, "reference": "native_fp32"}]
    data = {"rank": dist.get_rank(), "ep": dist.get_world_size(), "cases": cases,
            "build_fingerprint": manifest["build_fingerprint"], "program": "moe_region"}
    (root / f"ast-{mode}-b{budget}-rank{dist.get_rank()}.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if dist.get_rank() == 0:
        print(json.dumps(data), flush=True)


def test_ast_moe_device() -> None:
    """Run the region matrix selected by the thin launcher."""
    run_acceptance(os.getenv("HP_AST_MOE_MODE", "push"), os.environ["HP_AST_MOE_RESULTS"])


def main() -> None:
    """Select an acceptance case in an already initialized worker group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("push", "pull"), required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--transport", default="p2p")
    options = parser.parse_args()
    try:
        run_acceptance(options.mode, options.result_dir, budget=options.budget, transport=options.transport)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
