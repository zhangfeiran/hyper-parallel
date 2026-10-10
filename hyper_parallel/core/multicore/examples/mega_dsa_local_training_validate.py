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
"""Native local CP projection/autograd validation at fixed observed activation values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch.utils.checkpoint import checkpoint

import hyper_parallel
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import (
    _SCALE,
    _tile,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    ProjectionGeometry,
    project_local_states,
    projection_fixture,
    projection_vjp_metrics,
    projection_vjp_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_training import (
    FusedDsaLocalCpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.projection_precision import (
    coupled_projection_objective_check,
    validate_loaded_projection_reference,
    validate_projection_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    reference_budget_passes,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_FIELDS = ("hidden", "query_projection", "compressed_projection", "query_rope_projection", "key_rope_projection",
           "index_query_projection", "index_key_projection", "merge_projection", "output_projection")


def _snapshots(meta, states, result, output, gradients, objective, loss, *, group=None):
    payload = {"queries": meta.q_global_ids, "keys": meta.kv_global_ids,
               "states": tuple(tensor.detach().cpu() for tensor in states),
               "indices": result.global_indices[:, 0].cpu(), "output": output.detach().cpu(),
               "gradients": tuple(None if tensor is None else tensor.detach().cpu() for tensor in gradients),
               "objective": objective.detach().cpu(), "loss": loss.detach().cpu(),
               "loss_requires_grad": loss.requires_grad}
    peers = [None]*dist.get_world_size(group)
    dist.all_gather_object(peers, payload, group=group)
    return peers


def _check(peers, cpu_fixture, geometry, lengths, mode, coefficient, auxiliary, divisor, *, precision_budgets=None):
    total = sum(lengths)
    hidden, weights, targets = cpu_fixture
    states = tuple(torch.zeros((total, *tail), dtype=peers[0]["states"][field].dtype)
                   for field, tail in enumerate(geometry.tails))
    output = torch.zeros(total, geometry.heads, geometry.compressed, dtype=torch.bfloat16)
    indices = torch.full((total, 2048), -1, dtype=torch.int32)
    contributing = []
    key_cover = []
    for peer in peers:
        rows, keys = list(peer["queries"]), list(peer["keys"])
        contributing.extend(rows)
        key_cover.extend(keys)
        for field, state in enumerate(peer["states"]):
            states[field][keys if field in (1, 3, 5) else rows] = state
        indices[rows] = peer["indices"]
        output[rows] = peer["output"]
    if len(contributing) != len(set(contributing)) or sorted(key_cover) != list(range(total)):
        raise RuntimeError("projection validation requires disjoint query tiles and complete unique key ownership")
    width = max(lengths)
    if bool((indices[:, width:] != -1).any()):
        raise RuntimeError("bounded projection oracle cannot drop valid selected slots")
    targets = targets.clone()
    absent = sorted(set(range(total))-set(contributing))
    targets[absent] = 0
    expected_objective, expected_loss, expected_gradients = projection_vjp_reference(
        hidden, weights, targets, states, output, indices[:, :width], DsaBatchMeta.packed(lengths), geometry,
        mode=mode, coefficient=coefficient, auxiliary=auxiliary, attention_scale=_SCALE)
    objective = sum(peer["objective"] for peer in peers)/divisor
    auxiliary_loss = sum(peer["loss"] for peer in peers)/divisor
    metrics = {"objective": projection_vjp_metrics(objective, expected_objective),
               "auxiliary_loss": projection_vjp_metrics(auxiliary_loss, expected_loss)
               if mode != "lm_only" else {"absent": all(not peer["loss_requires_grad"]
                                                          and not bool(peer["loss"].count_nonzero())
                                                          for peer in peers)}}
    for name, expected, field in zip(_FIELDS, expected_gradients, range(len(_FIELDS))):
        parts = [peer["gradients"][field] for peer in peers]
        if expected is None:
            metrics[name] = {"absent": all(value is None for value in parts)}
        else:
            present = [value for value in parts if value is not None]
            if not present:
                metrics[name] = {"passed": False, "missing_gradient": True}
                continue
            actual = sum(present)/divisor
            metric = projection_vjp_metrics(actual, expected)
            if coefficient == 0 and field in (5, 6, 7):
                metric["disabled_aux_exact_zero"] = not bool(actual.count_nonzero())
            metrics[name] = metric
    passed = all(metric.get("passed", metric.get("absent", False))
                 and metric.get("disabled_aux_exact_zero", True) for metric in metrics.values())
    calibrated = None
    if precision_budgets is not None:
        if (sorted(contributing) != list(range(total)) or tuple(lengths) != (3, 10)
                or geometry != ProjectionGeometry(heads=geometry.heads) or coefficient != .3 or auxiliary != -2.5):
            raise ValueError("projection profile does not cover this query subset, geometry or objective")
        dtype = str(states[6].dtype)
        bucket = f"H{geometry.heads}/{dtype}/{mode}"
        if bucket not in precision_budgets:
            raise ValueError("projection profile does not cover the requested head/weight/mode bucket")
        calibrated = {}
        for name, metric in metrics.items():
            if name in precision_budgets[bucket] and "finite" in metric:
                calibrated[name] = {**metric, "passed": reference_budget_passes(metric,
                                                                                precision_budgets[bucket][name])}
            else:
                calibrated[name] = metric
        if mode == "lm_plus_kl":
            component = coupled_projection_objective_check(
                tuple(peer["objective"] for peer in peers), tuple(peer["loss"] for peer in peers),
                expected_objective, expected_loss, upstream=auxiliary, divisor=divisor,
                main_budget=precision_budgets[f"H{geometry.heads}/{dtype}/lm_only"]["objective"],
                auxiliary_budget=precision_budgets[f"H{geometry.heads}/{dtype}/kl_only"]["auxiliary_loss"])
            calibrated["objective"].update(normalized_reference_passed=calibrated["objective"]["passed"],
                                           component_reference_bound=component, passed=component["passed"],
                                           criterion="independent_components_plus_fp32_rounding")
        calibrated_passed = all(metric.get("passed", metric.get("absent", False))
                                and metric.get("disabled_aux_exact_zero", True) for metric in calibrated.values())
    else:
        calibrated_passed = None
    return {"passed": calibrated_passed if calibrated is not None else passed,
            "original_pointwise_passed": passed, "calibrated_reference": calibrated,
            "independent_fp64_vjp": metrics, "query_rows": len(contributing),
            "global_normalization_queries": total, "native_activation_and_output_values_fixed": True,
            "bf16_projection_composition_acceptance": False}


def _step(workspace, backend, base_meta, geometry, cpu_fixture, *, tile_mode, weight_dtype,
          mode, coefficient, auxiliary, divisor, replay, precision_budgets=None, fused_kl=False,
          sparse_main_return=False, sparse_index_return=False, sgd_step=False):
    device = backend.layout.device
    tile = _tile(base_meta, tile_mode)
    layout = backend.prepare_local_queries(tile.q_global_ids)
    query_rows = torch.tensor(tile.q_global_ids, dtype=torch.long, device=device)
    key_rows = torch.tensor(tile.kv_global_ids, dtype=torch.long, device=device)
    cpu_hidden, cpu_weights, cpu_targets = cpu_fixture
    hidden = cpu_hidden.to(device).requires_grad_()
    weights = tuple(tensor.to(device).requires_grad_() for tensor in cpu_weights)
    states = project_local_states(hidden, weights, geometry, query_rows, key_rows, weight_dtype)
    probe = FusedDsaLocalCpTrainingProbe(backend, normalization=DsaLossNormalization(base_meta.global_valid_queries,
                                                                                  divisor),
                                        loss_coeff=coefficient, mode=mode, fused_kl=fused_kl,
                                        sparse_main_return=sparse_main_return, sparse_index_return=sparse_index_return)
    invocation = workspace.prepare(replace(base_meta, layer=1, microbatch=2, invocation=workspace.transport_epoch+1))
    captured = []
    original = backend.forward_local
    def _capture(*args, **kwargs):
        result = original(*args, **kwargs)
        selected_kl = bool(kwargs.get("selected_kl", False))
        expected_phases = 9 if selected_kl else 6
        if len(result.phase_traces) != expected_phases or (result.kl_loss is not None) != selected_kl:
            raise RuntimeError("local training used an unexpected native phase graph")
        captured.append(result)
        return result
    def _forward(*inputs):
        return probe.forward(invocation, inputs[:4], inputs[4:], query_layout=layout)
    with patch.object(backend, "forward_local", side_effect=_capture):
        output, loss = (checkpoint(_forward, *states, use_reentrant=False) if replay == "checkpoint"
                        else _forward(*states))
        predicted = output.float().flatten(1) @ weights[7].T
        target = cpu_targets.to(device).index_select(0, query_rows)
        main_loss = (predicted-target).square().sum()*probe.normalization.local_sum_scale
        objective = loss*auxiliary if mode == "kl_only" else main_loss
        if mode == "lm_plus_kl":
            objective = objective+loss*auxiliary
        inputs = (hidden, *weights)
        first = torch.autograd.grad(objective, inputs, allow_unused=True, retain_graph=replay == "retained")
        if replay == "retained":
            backend.prepare_local_queries(())
            other = workspace.prepare(replace(base_meta, layer=3, microbatch=4,
                                              invocation=workspace.transport_epoch+1))
            backend.forward_local(other, tuple(tensor.detach() for tensor in states[:4]),
                                  tuple(tensor.detach() for tensor in states[4:]), query_layout=layout)
            second = torch.autograd.grad(objective, inputs, allow_unused=True)
            for actual, expected in zip(second, first):
                if expected is None:
                    if actual is not None:
                        raise RuntimeError("retained replay changed gradient isolation")
                else:
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=2e-5)
    torch.npu.synchronize()
    peers = _snapshots(tile, states, captured[0], output, first, objective, loss, group=backend.layout.group)
    evidence = _check(peers, cpu_fixture, geometry, (3, 10), mode, coefficient, auxiliary, divisor,
                      precision_budgets=precision_budgets)
    if sgd_step:
        evidence["sgd"] = _sgd_check(weights, first[1:], peers, divisor, backend.layout.group)
    evidence.update(captured_forward_invocations=len(captured), replay=replay, fused_kl=fused_kl,
                    sparse_main_return=sparse_main_return, sparse_index_return=sparse_index_return,
                    native_forward_phase_counts=[len(result.phase_traces) for result in captured])
    return evidence


def _sgd_check(weights, gradients, peers, divisor, group):
    expected = []
    updates = 0
    for field, (weight, gradient) in enumerate(zip(weights, gradients)):
        contributions = [peer["gradients"][field+1] for peer in peers if peer["gradients"][field+1] is not None]
        before = weight.detach().cpu().clone()
        if contributions:
            reduced = (torch.zeros_like(weight) if gradient is None
                       else gradient.detach().clone(memory_format=torch.contiguous_format))
            dist.all_reduce(reduced, group=group)
            reduced.div_(divisor)
            reference = sum(contributions)/divisor
            torch.testing.assert_close(reduced.cpu(), reference, rtol=0, atol=0)
            weight.grad = reduced
            before.add_(reference, alpha=-1e-3)
            updates += 1
        expected.append(before)
    torch.optim.SGD(weights, lr=1e-3, foreach=False).step()
    torch.npu.synchronize()
    for weight, reference in zip(weights, expected):
        torch.testing.assert_close(weight.detach().cpu(), reference, rtol=0, atol=torch.finfo(torch.float32).eps)
    return {"passed": True, "reducer_divisor": divisor, "updated_parameters": updates, "learning_rate": 1e-3,
            "reference": "CPU ordered CP gradient aggregation after independent Native VJP validation"}


def _fixture(report, heads, weight_dtype, pattern, *, smoke, precision_budgets=None):
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    rank, size = dist.get_rank(), dist.get_world_size()
    lengths = (3, 10)
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "local-training-vjp", DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                     attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
    geometry = ProjectionGeometry(heads=heads)
    cpu_fixture = projection_fixture(sum(lengths), geometry)
    cases = [("lm_plus_kl", .3, -2.5, 1, "owner", "plain"),
             ("kl_only", .3, -2.5, 1, "owner", "plain")]
    if not smoke:
        cases = [(mode, coefficient, -2.5, divisor, tile, replay)
                 for mode in ("lm_only", "kl_only", "lm_plus_kl") for coefficient in (0., .3)
                 for divisor in sorted({1, size}) for tile in ("owner", "interior", "first_sequence", "empty")
                 for replay in ("plain", "retained", "checkpoint")]
    try:
        for mode, coefficient, auxiliary, divisor, tile, replay in cases:
            report["stage"] = {"heads": heads, "weight_dtype": str(weight_dtype), "pattern": pattern,
                               "mode": mode, "coefficient": coefficient, "auxiliary": auxiliary,
                               "reducer_divisor": divisor, "tile": tile, "replay": replay}
            evidence = _step(workspace, backend, meta, geometry, cpu_fixture, tile_mode=tile,
                             weight_dtype=weight_dtype, mode=mode, coefficient=coefficient,
                             auxiliary=auxiliary, divisor=divisor, replay=replay, precision_budgets=precision_budgets)
            report["cases"].append({**report["stage"], **evidence})
            if not evidence["passed"]:
                raise RuntimeError(f"local training projection VJP did not pass: {evidence}")
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, *, smoke: bool = False, reference_profile: Path | None = None,
                   profile_policy_sha256: str | None = None) -> None:
    """Run actual local CP training and fixed-Native-value FP64 projection VJP checks."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[], package_path=hyper_parallel.__file__)
    budgets, profile = None, None
    fixtures = [(32, torch.float32, "strided")] if smoke else [
        (heads, dtype, pattern) for heads in (32, 64) for dtype in (torch.bfloat16, torch.float32)
        for pattern in ("contiguous", "strided", "zigzag", "empty")]
    try:
        if reference_profile is not None:
            if profile_policy_sha256 is None:
                raise ValueError("projection reference profile requires an independently frozen policy SHA")
            profile = json.loads(reference_profile.read_text())
            libraries = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                         for name in profile["loaded_library_sha256"]}
            budgets = validate_projection_reference_profile(
                profile, expected_versions={"torch_version": torch.__version__,
                                            "torch_npu_version": torch_npu.__version__},
                expected_library_hashes=libraries, expected_policy_sha256=profile_policy_sha256)
            report.update(reference_profile_sha256=hashlib.sha256(reference_profile.read_bytes()).hexdigest(),
                          reference_policy_sha256=profile_policy_sha256)
        for heads, dtype, pattern in fixtures:
            _fixture(report, heads, dtype, pattern, smoke=smoke, precision_budgets=budgets)
        loaded = _loaded_library_identity()
        report["loaded_library_sha256"] = loaded
        if profile is not None:
            validate_loaded_projection_reference(profile, loaded)
        report.update(status="passed", loaded_library_sha256=_loaded_library_identity())
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Persist every failing comparison and keep the full-model acceptance boundary explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reference-profile", type=Path)
    parser.add_argument("--profile-policy-sha256")
    args = parser.parse_args()
    report = {"status": "running", "scope": "local CP training projection VJP at fixed Native values",
              "full_model_acceptance": False, "bf16_projection_composition_acceptance": False}
    try:
        run_validation(report, smoke=args.smoke, reference_profile=args.reference_profile,
                       profile_policy_sha256=args.profile_policy_sha256)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        (args.output/f"rank{rank}.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
