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
"""Local subset/zero-coefficient projection VJPs against independent masked full-query native primitives."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples import (
    mega_dsa_local_training_validate as validation,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    ProjectionGeometry,
    projection_fixture,
    projection_vjp_metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_projection_reference_calibrate import (
    _FIELDS,
    _project_gradients,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaStats,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    mixed_sfa_backward_probe,
    mixed_sfa_gradient_partials,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_kl import _submit_mixed_kl
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_TRAINING_STEP = validation._step


def _global_snapshot(peers, geometry, total, device):
    states = tuple(torch.zeros((total, *tail), dtype=peers[0]["states"][field].dtype)
                   for field, tail in enumerate(geometry.tails))
    output = torch.zeros(total, geometry.heads, geometry.compressed, dtype=torch.bfloat16)
    indices = torch.full((total, 2048), -1, dtype=torch.int32)
    queries, keys = [], []
    for peer in peers:
        rows, owned = list(peer["queries"]), list(peer["keys"])
        queries.extend(rows)
        keys.extend(owned)
        for field, value in enumerate(peer["states"]):
            states[field][owned if field in (1, 3, 5) else rows] = value
        output[rows] = peer["output"]
        indices[rows] = peer["indices"]
    if len(queries) != len(set(queries)) or sorted(keys) != list(range(total)):
        raise RuntimeError("masked native comparison requires disjoint queries and unique complete key ownership")
    return tuple(value.to(device) for value in states), output.to(device), indices.to(device), sorted(queries)


def _check_native(peers, cpu_fixture, geometry, lengths, mode, coefficient, auxiliary, divisor, *,
                  precision_budgets=None):
    if precision_budgets is not None:
        raise ValueError("masked native comparisons do not consume the complete-query FP64 reference profile")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    total = sum(lengths)
    states, observed, global_indices, queries = _global_snapshot(peers, geometry, total, device)
    layout = CannDsaLayout(DsaBatchMeta.packed(lengths), device)
    config = MixedSfaSchedule(7).runtime_config(device)
    selection = torch.ops.npu.npu_lightning_indexer(
        states[4], states[5][:, None], states[6], actual_seq_lengths_query=layout.length_tensor,
        actual_seq_lengths_key=layout.length_tensor, layout_query="TND", layout_key="TND", sparse_count=2048,
        sparse_mode=3, return_value=True)[0]
    native = layout.global_to_sequence_indices(global_indices)
    selection[queries] = native[queries]
    stock = torch.ops.npu.npu_sparse_flash_attention(
        states[0], states[1][:, None], states[1][:, None], selection, validation._SCALE,
        actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_kv=layout.length_tensor,
        query_rope=states[2], key_rope=states[3][:, None], layout_query="TND", layout_kv="TND",
        sparse_block_size=1, sparse_mode=3, attention_mode=2, return_softmax_lse=True)
    torch.testing.assert_close(stock[0][queries].cpu(), observed[queries].cpu(), rtol=.02, atol=2e-5)
    cpu_hidden, cpu_weights, cpu_targets = cpu_fixture
    hidden = cpu_hidden.to(device)
    weights = tuple(value.to(device) for value in cpu_weights)
    output_leaf = observed.detach().requires_grad_()
    output_weight = weights[7].detach().requires_grad_()
    targets = cpu_targets.clone()
    absent = sorted(set(range(total))-set(queries))
    targets[absent] = 0
    prediction = output_leaf.float().flatten(1) @ output_weight.T
    main_loss = (prediction-targets.to(device)).square().sum()/total
    cotangent, output_gradient = torch.autograd.grad(main_loss, (output_leaf, output_weight))
    raw, _, retained = mixed_sfa_backward_probe(
        states[:4], selection, layout.length_tensor, config, cotangent, stock, validation._SCALE,
        MixedSfaSchedule(7))
    key, value = mixed_sfa_gradient_partials(retained, query_tokens=total, key_tokens=total, heads=geometry.heads)
    main_gradients = (raw[0], (key[:, :512]+value).bfloat16(), raw[3], key[:, 512:].bfloat16())
    index = tuple(torch.zeros_like(value) for value in states[4:])
    auxiliary_loss = main_loss.new_zeros(())
    if mode != "lm_only" and coefficient and queries:
        kl = _submit_mixed_kl(states[:4], states[4:], selection, layout.cumulative_lengths,
                              CannDsaStats(stock[1], stock[2]), config, validation._SCALE)
        scale = coefficient/total
        index = ((kl.gradients[0].float()*scale*auxiliary).bfloat16(),
                 (kl.key_partials*scale*auxiliary).bfloat16(),
                 (kl.gradients[2].float()*scale*auxiliary).to(states[6].dtype))
        auxiliary_loss = kl.loss*scale
    expected = _project_gradients(main_gradients, index, hidden, weights, output_gradient, mode)
    expected_objective = auxiliary_loss*auxiliary if mode == "kl_only" else main_loss.detach()
    if mode == "lm_plus_kl":
        expected_objective = expected_objective+auxiliary_loss*auxiliary
    torch.npu.synchronize()
    metrics = {
        "objective": projection_vjp_metrics(sum(peer["objective"] for peer in peers)/divisor, expected_objective),
        "auxiliary_loss": projection_vjp_metrics(sum(peer["loss"] for peer in peers)/divisor, auxiliary_loss),
    }
    for field, (name, reference) in enumerate(zip(_FIELDS, expected)):
        actual = [peer["gradients"][field] for peer in peers if peer["gradients"][field] is not None]
        if reference is None:
            metrics[name] = {"passed": not actual, "absent": not actual}
        elif not actual:
            metrics[name] = {"passed": False, "missing_gradient": True}
        else:
            combined = sum(actual)/divisor
            metrics[name] = projection_vjp_metrics(combined, reference)
            if coefficient == 0 and field in (5, 6, 7):
                metrics[name]["disabled_aux_exact_zero"] = not bool(combined.count_nonzero())
    passed = all(item["passed"] and item.get("disabled_aux_exact_zero", True) for item in metrics.values())
    if coefficient == 0 and bool(sum(peer["loss"] for peer in peers).count_nonzero()):
        raise RuntimeError("disabled auxiliary loss must be exactly zero")
    return {"passed": passed, "masked_global_native_vjp": metrics, "query_rows": len(queries),
            "global_normalization_queries": total, "native_activation_and_output_values_fixed": True,
            "bf16_projection_composition_acceptance": False, "fp64_oracle_acceptance": False,
            "reference": "global mixed SFA/KL with zero-contribution unselected query rows"}


def _guarded_check(*args, **kwargs):
    return collective_validation_check(partial(_check_native, *args, **kwargs))


def _fixture(report, heads, dtype, pattern, smoke, sparse_main_return=False, sparse_index_return=False):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    size, rank = dist.get_world_size(), dist.get_rank()
    meta = validation._metadata((3, 10), pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "subset-native-vjp", DsaWorkspaceSpec(capacity, 13, capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                     attention_scale=validation._SCALE, schedule=MixedSfaSchedule(7))
    geometry = ProjectionGeometry(heads=heads)
    fixture = projection_fixture(13, geometry)
    cases = [("lm_plus_kl", .3, 1, "interior", "plain"),
             ("kl_only", .3, 1, "first_sequence", "plain"),
             ("lm_plus_kl", 0., 1, "owner", "plain"),
             ("kl_only", 0., 1, "interior", "plain"),
             ("lm_plus_kl", .3, 1, "empty", "checkpoint")]
    if not smoke:
        cases = product(("lm_only", "kl_only", "lm_plus_kl"), (0., .3), sorted({1, size}),
                        ("owner", "interior", "first_sequence", "empty"), ("plain", "retained", "checkpoint"))
    try:
        for mode, coefficient, divisor, tile, replay in cases:
            stage = {"heads": heads, "weight_dtype": str(dtype), "pattern": pattern, "mode": mode,
                     "coefficient": coefficient, "auxiliary": -2.5, "reducer_divisor": divisor,
                     "tile": tile, "replay": replay}
            report["stage"] = stage
            evidence = _TRAINING_STEP(workspace, backend, meta, geometry, fixture, tile_mode=tile,
                                      weight_dtype=dtype, mode=mode, coefficient=coefficient, auxiliary=-2.5,
                                      divisor=divisor, replay=replay, fused_kl=True,
                                      sparse_main_return=sparse_main_return, sparse_index_return=sparse_index_return)
            report["cases"].append({**stage, **evidence})
            if not evidence["passed"]:
                raise RuntimeError(f"masked global native projection VJP mismatch: {evidence}")
    finally:
        workspace.close()
        root.close()


def main() -> None:
    """Persist native-to-native comparisons separately from calibrated FP64 projection acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sparse-main-return", action="store_true")
    parser.add_argument("--sparse-index-return", action="store_true")
    args = parser.parse_args()
    report = {"status": "running", "scope": "subset/zero-coefficient fixed-value native projection VJP",
              "full_model_acceptance": False, "bf16_projection_composition_acceptance": False}
    try:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
        report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[],
                      sparse_main_return=args.sparse_main_return, sparse_index_return=args.sparse_index_return)
        fixtures = [(32, torch.float32, "strided")] if args.smoke else product(
            (32, 64), (torch.bfloat16, torch.float32), ("contiguous", "strided", "zigzag", "empty"))
        with patch.object(validation, "_check", side_effect=_guarded_check):
            for heads, dtype, pattern in fixtures:
                _fixture(report, heads, dtype, pattern, args.smoke, args.sparse_main_return, args.sparse_index_return)
        report.update(status="passed", loaded_library_sha256=validation._loaded_library_identity())
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
