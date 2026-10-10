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
"""Local-Q main backward with independent FP32 math and complete owner-return evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  # Registers the NPU backend.
from torch.utils.checkpoint import checkpoint

import hyper_parallel
from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_backward_validate import _cotangent, _owner_oracle
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _local_states, _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_fused_forward_validate import _main_states
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_validate import _SCALE, _tile
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_indexer_validate import _fixture
from hyper_parallel.core.multicore.examples.mega_dsa_mixed_tile_validate import _reference
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpAttentionProbe, FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import mixed_sfa_backward_probe
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference import sparse_attention_reference
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics, ordered_fp32_owner_bound, reference_absolute_limit,
    reference_budget_passes, validate_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import DsaWorkspaceSpec, MegaDsaWorkspace
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _loaded_library_identity():
    loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
              if any(name in line for name in
                     ("libopapi.so", "libtorch_npu.so", "libcust_opapi.so", "libcust_opmaster_rt2.0.so",
                      "libhyper_parallel_mega_moe_torch.so",
                      "libhyper_parallel_shmem.so", "libhyper_parallel_shmem_ops.so"))}
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in loaded}


def _difference(actual, expected, *, require_close=True):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    close = torch.isclose(actual, expected, rtol=.02, atol=2e-5)
    if require_close:
        torch.testing.assert_close(actual, expected, rtol=.02, atol=2e-5)
    delta = (actual.double() - expected.double()).flatten()
    actual_norm, expected_norm = actual.double().norm(), expected.double().norm()
    cosine = (float((actual.double() * expected.double()).sum() / (actual_norm * expected_norm).clamp_min(1e-30))
              if float(actual_norm + expected_norm) else 1.)
    return {"passed": bool(close.all()), "mismatched_elements": int((~close).sum()),
            "elements": actual.numel(), "max_abs": float(delta.abs().max()) if delta.numel() else 0.,
            "relative_l2": float(delta.norm() / expected_norm.clamp_min(1e-30)), "cosine": cosine,
            "expected_norm": float(expected_norm), "relative_l2_applicable": bool(expected_norm > 0),
            "rtol": .02, "atol": 2e-5}


def _cpu_oracle(states, indices, meta, cotangent, *, dtype=torch.float32):
    rows = list(meta.q_global_ids)
    cpu = tuple(tensor.cpu().to(dtype) for tensor in states)
    if not rows:
        return tuple(torch.zeros_like(tensor) for tensor in (cpu[0][rows], cpu[1], cpu[2][rows], cpu[3]))
    inputs = tuple(tensor.detach().requires_grad_() for tensor in (cpu[0][rows], cpu[1], cpu[2][rows], cpu[3]))
    full_keys = replace(meta, kv_global_ids=tuple(range(meta.global_valid_queries)))
    selected = indices[:, 0].cpu()
    if dtype == torch.float64:
        width = max(end - start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
        selected = selected[:, :width]
    output, _ = sparse_attention_reference(*inputs, selected, full_keys, attention_scale=_SCALE)
    return torch.autograd.grad(output, inputs, grad_outputs=cotangent.cpu().to(dtype))


def _calibrated_owner_check(result, backend, oracle, budgets):
    partials = result.partials.cpu()
    actual = (partials[:, :512] + partials[:, 576:], partials[:, 512:576])
    expected = (oracle[1], oracle[3])
    names = ("compressed_kv", "key_rope")
    limits = tuple(reference_absolute_limit(value, budgets[name]) for name, value in zip(names, expected))
    peers = [None] * backend.backend.layout.cp_size
    dist.all_gather_object(peers, {"actual": actual, "expected": expected, "limits": limits})
    rows = list(backend.backend.layout.batch_meta.kv_global_ids)
    proof = {}
    for index, name in enumerate(names):
        target = sum(peer["expected"][index] for peer in peers)[rows]
        bound = ordered_fp32_owner_bound((peer["actual"][index] for peer in peers),
                                         (peer["limits"][index] for peer in peers))[rows]
        delta = (result.owner_fp32[index].cpu().double() - target).abs()
        proof[name] = {"passed": bool(torch.isfinite(delta).all() and (delta <= bound).all()),
                       "max_abs": float(delta.max()) if delta.numel() else 0.,
                       "max_allowed_abs": float(bound.max()) if bound.numel() else 0.,
                       "fp64_global_objective_sum": True, "fp32_reduction_rounding_bound": True}
    return proof


def _check(result, forward, backward, states, meta, cotangent, layout, stock, *, precision_budgets=None):
    local = forward.saved.local_query_layout
    rows = list(local.global_ids)
    native_indices = stock.clone()
    native_indices[rows] = forward.saved.indices
    baseline = _reference(states, native_indices, layout)
    full_grad = torch.zeros_like(states[0])
    full_grad[list(meta.q_global_ids)] = cotangent
    reference, _, _ = mixed_sfa_backward_probe(
        states, native_indices, layout.length_tensor, backward.backend.config,
        full_grad, baseline, _SCALE, backward.backend.schedule)
    native_metrics = []
    for index, (actual, expected) in enumerate(zip(result.native_gradients, reference)):
        if index in (0, 3):
            expected = expected[rows]
        native_metrics.append(_difference(actual, expected))
    oracle = _cpu_oracle(states, forward.global_indices, meta, cotangent,
                         dtype=torch.float64 if precision_budgets is not None else torch.float32)
    local_math = (result.gradients[0], result.partials[:, :512] + result.partials[:, 576:],
                  result.gradients[2], result.partials[:, 512:576])
    oracle_metrics = [_difference(actual, expected, require_close=False)
                      for actual, expected in zip(local_math, oracle)]
    global_math = (reference[0][list(meta.q_global_ids)], reference[1][:, 0].float() + reference[2][:, 0].float(),
                   reference[3][list(meta.q_global_ids)], reference[4][:, 0])
    baseline_oracle = [_difference(actual, expected, require_close=False)
                       for actual, expected in zip(global_math, oracle)]
    evidence = backward.validate_trace(result, tuple(trace.cpu() for trace in result.phase_traces),
                                      result.transport_trace.cpu())
    owner = _owner_oracle(result, backward)
    for actual, expected in zip((result.gradients[1], result.gradients[3]), result.owner_fp32):
        torch.testing.assert_close(actual.cpu(), expected.bfloat16().cpu(), rtol=0, atol=0)
    if not rows:
        torch.testing.assert_close(result.partials.cpu(), torch.zeros_like(result.partials.cpu()), rtol=0, atol=0)
    forward.saved.validate_versions()
    calibrated = None
    calibrated_baseline = None
    baseline_bits_exact = None
    calibrated_owner = None
    if precision_budgets is not None:
        names = ("query", "compressed_kv", "query_rope", "key_rope")
        calibrated = {name: gradient_error_metrics(actual.cpu(), expected) for name, actual, expected in
                      zip(names, local_math, oracle)}
        for name, metrics in calibrated.items():
            metrics["passed"] = reference_budget_passes(metrics, precision_budgets[name])
        calibrated_baseline = {name: gradient_error_metrics(actual.cpu(), expected)
                               for name, actual, expected in zip(names, global_math, oracle)}
        for name, metrics in calibrated_baseline.items():
            metrics["passed"] = reference_budget_passes(metrics, precision_budgets[name])
        baseline_bits_exact = {name: torch.equal(actual.cpu(), baseline.cpu())
                               for name, actual, baseline in zip(names, local_math, global_math)}
        calibrated_owner = _calibrated_owner_check(result, backward, oracle, precision_budgets)
    return {"trace": evidence, **owner, "native_reference": native_metrics, "fp32_oracle": oracle_metrics,
            "calibrated_oracle": calibrated,
            "calibrated_owner": calibrated_owner,
            "global_query_baseline_calibrated_oracle": calibrated_baseline,
            "global_query_baseline_bits_exact": baseline_bits_exact,
            "global_query_baseline_fp32_oracle": baseline_oracle,
            "query_rows": len(rows), "key_rows": len(states[1]), "empty_query_partials_zero": not rows,
            "owner_cast_after_fp32_reduction": True, "query_collectives": 0}


def _run_fixture(report, lengths, heads, pattern, device, *, smoke, reference_budgets=None):
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-local-cp-backward",
                                 DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    _, layout, _, index = _fixture(lengths, "random_signed", device)
    states = _main_states(sum(lengths), heads, device)
    stock, _ = torch.ops.npu.npu_lightning_indexer(
        *index, actual_seq_lengths_query=layout.length_tensor, actual_seq_lengths_key=layout.length_tensor,
        layout_query="TND", layout_key="TND", sparse_count=2048, sparse_mode=3, return_value=True)
    streams = (torch.npu.current_stream(), torch.npu.Stream())
    try:
        for groups in ((7,) if smoke else (1, 7, 19)):
            backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                             attention_scale=_SCALE, schedule=MixedSfaSchedule(groups))
            backward = FusedDsaCpBackwardProbe(backend)
            for mode in (("owner", "empty") if smoke else ("owner", "interior", "first_sequence", "empty")):
                bucket = f"H{heads}/CP{size}/{pattern}/rank{rank}/{mode}"
                if reference_budgets is not None and bucket not in reference_budgets:
                    raise ValueError("independent reference profile does not cover this query scope")
                tile = _tile(meta, mode)
                backend.prepare_local_queries(tile.q_global_ids)
                local_main, local_index = _local_states(tile, states, index)
                invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1))
                report["stage"] = {"lengths": lengths, "heads": heads, "pattern": pattern,
                                   "groups": groups, "tile": mode, "step": "forward"}
                forward = backend.forward_local(invocation, local_main, local_index)
                cotangent = _cotangent(tile, heads, 0, device)
                ready = torch.npu.Event()
                ready.record(torch.npu.current_stream())
                # A later publication and layout must not replace the earlier saved local positions.
                later_meta = _tile(meta, "empty")
                backend.prepare_local_queries(later_meta.q_global_ids)
                changed = tuple((tensor + .015625).bfloat16() for tensor in states)
                later_main, later_index = _local_states(later_meta, changed, index)
                later = workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1))
                backend.forward_local(later, later_main, later_index)
                pending = []
                for repeat, stream in enumerate(streams):
                    report["stage"].update(step="backward", repeat=repeat)
                    with torch.npu.stream(stream):
                        ready.wait()
                        pending.append(backward.backward_local(forward, cotangent))
                torch.npu.synchronize()
                for repeat, result in enumerate(pending):
                    report["stage"].update(step="check", repeat=repeat)
                    proof = _check(result, forward, backward, states, tile, cotangent, layout, stock,
                                   precision_budgets=(reference_budgets[bucket]
                                                      if reference_budgets is not None else None))
                    report["cases"].append({**report["stage"], **proof,
                                            "delayed_after_layout_and_arena_republication": True,
                                            "cross_stream_backward_before_observation": True})
                for actual, expected in zip(pending[1].gradients, pending[0].gradients):
                    _difference(actual, expected)
    finally:
        workspace.close()
        root.close()
    if shmem.debug_state()["reference_count"] != 0:
        raise RuntimeError("local fused backward root did not close cleanly")


def run_validation(report: dict, output_dir: Path, *, smoke: bool = False,
                   reference_profile: Path | None = None, profile_policy_sha256: str | None = None) -> None:
    """Validate CP main backward for actual local Q and complete-key owner return.

    Args:
        report: Mutable per-rank evidence record.
        output_dir: Directory for complete or partial observations.
        smoke: Run the bounded H32 owner/empty matrix.
        reference_profile: Optional independently validated short-fixture BF16 profile.
            Omitting it retains the existing pointwise numerical gate.
        profile_policy_sha256: Separately frozen policy commitment; required with a profile.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    torch.npu.set_op_timeout_ms(120000)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    report.update(rank=rank, cp_size=dist.get_world_size(), cases=[], package_path=hyper_parallel.__file__,
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    budgets = None
    if reference_profile is not None:
        profile = json.loads(reference_profile.read_text(encoding="utf-8"))
        if (tuple(profile["policy"]["lengths"]) != (3, 10) or profile["policy"]["sparse_count"] != 2048
                or profile["policy"]["attention_scale"] != _SCALE):
            raise ValueError("reference profile has unsupported geometry or attention scale")
        libraries = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in profile["loaded_library_sha256"]}
        budgets = validate_reference_profile(
            profile, expected_versions={"torch_version": torch.__version__, "torch_npu_version": torch_npu.__version__},
            expected_library_hashes=libraries, expected_policy_sha256=profile_policy_sha256)
        report.update(reference_profile_sha256=hashlib.sha256(reference_profile.read_bytes()).hexdigest(),
                      reference_policy_sha256=profile_policy_sha256, oracle_dtype="float64")
    fixtures = [(32, "contiguous")] if smoke else [(heads, pattern) for heads in (32, 64)
                                                  for pattern in ("contiguous", "strided", "empty")]
    output_dir.mkdir(parents=True, exist_ok=True)
    for heads, pattern in fixtures:
        _run_fixture(report, (3, 10), heads, pattern, torch.device(f"npu:{local_rank}"), smoke=smoke,
                     reference_budgets=budgets)
        (output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    numerical = all(metric["passed"] for case in report["cases"] for metric in case["fp32_oracle"])
    calibrated = (all(metric["passed"] for case in report["cases"]
                      for field in ("calibrated_oracle", "calibrated_owner") for metric in case[field].values())
                  if budgets is not None else None)
    loaded = _loaded_library_identity()
    if budgets is not None and any(loaded.get(name) != digest for name, digest in libraries.items()):
        raise ValueError("actual loaded reference libraries do not match the calibrated environment")
    report.update(status="passed" if (calibrated if budgets is not None else numerical) else "failed",
                  stage="complete", case_count=len(report["cases"]), calibrated_oracle_acceptance=calibrated,
                  native_reference_and_transport_passed=True, fp32_oracle_acceptance=numerical,
                  loaded_library_sha256=loaded)
    dist.barrier()
    dist.destroy_process_group()


def _autograd_fixture(report, heads, pattern, device):
    meta = _metadata((3, 10), pattern, dist.get_world_size(), dist.get_rank())
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(dist.get_world_size())))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-local-autograd", DsaWorkspaceSpec(capacity, 13, capacity,
                                                                           dist.get_world_size()))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    _, _, _, index = _fixture((3, 10), "random_signed", device)
    if heads == 64:
        index = (*index[:2], index[2].float() * 1.003)
    states = _main_states(13, heads, device)
    backend = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=heads,
                                     attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
    raw_backward = FusedDsaCpBackwardProbe(backend)
    bridge = FusedDsaCpAttentionProbe(backend)
    stream = torch.npu.Stream()
    try:
        for mode in ("owner", "interior", "first_sequence", "empty"):
            tile = _tile(meta, mode)
            layout = backend.prepare_local_queries(tile.q_global_ids)
            invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1))
            local_main, local_index = _local_states(tile, states, index)
            cotangent = _cotangent(tile, heads, 0, device)
            report["stage"] = {"heads": heads, "pattern": pattern, "tile": mode, "step": "raw_reference"}
            raw = backend.forward_local(invocation, local_main, local_index, query_layout=layout)
            reference = raw_backward.backward_local(raw, cotangent)
            main_inputs = tuple(tensor.clone().requires_grad_() for tensor in local_main)
            index_inputs = tuple(tensor.clone().requires_grad_() for tensor in local_index)
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                output = bridge.attention_local(invocation, main_inputs, index_inputs, query_layout=layout)
            torch.npu.current_stream().wait_stream(stream)
            backend.prepare_local_queries(())
            empty = _tile(meta, "empty")
            changed = tuple((tensor + .015625).bfloat16() for tensor in states)
            later_main, later_index = _local_states(empty, changed, index)
            backend.forward_local(workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1)),
                                  later_main, later_index)
            report["stage"]["step"] = "delayed_retained_backward"
            first = torch.autograd.grad(output, (*main_inputs, *index_inputs), grad_outputs=cotangent,
                                        retain_graph=True, allow_unused=True)
            second = torch.autograd.grad(output, main_inputs, grad_outputs=cotangent)
            checkpoint_main = tuple(tensor.clone().requires_grad_() for tensor in local_main)

            def _attention(*inputs, saved_invocation=invocation, saved_index=local_index, saved_layout=layout):
                return bridge.attention_local(saved_invocation, inputs, saved_index, query_layout=saved_layout)

            report["stage"]["step"] = "checkpoint_forward"
            checkpoint_output = checkpoint(_attention, *checkpoint_main, use_reentrant=False)
            backend.prepare_local_queries(meta.q_global_ids)
            backend.forward_local(workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1)),
                                  *_local_states(meta, changed, index))
            report["stage"]["step"] = "checkpoint_recompute_after_reprepare"
            recomputed = torch.autograd.grad(checkpoint_output, checkpoint_main, grad_outputs=cotangent)
            torch.npu.synchronize()
            if any(gradient is not None for gradient in first[4:]):
                raise RuntimeError("local main attention installed an indexer gradient path")
            torch.testing.assert_close(output.cpu(), raw.output.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(checkpoint_output.cpu(), raw.output.cpu(), rtol=0, atol=0)
            metrics = []
            for actual, replay, checkpoint_gradient, expected in zip(first[:4], second, recomputed,
                                                                     reference.gradients):
                metrics.append({"delayed": _difference(actual, expected), "retained": _difference(replay, expected),
                                "checkpoint": _difference(checkpoint_gradient, expected)})
            proof = _owner_oracle(reference, raw_backward)
            report["cases"].append({**report["stage"], **proof, "query_rows": len(tile.q_global_ids),
                                    "index_weight_dtype": str(index[2].dtype), "main_gradients": metrics,
                                    "indexer_gradients_absent": True, "forward_raw_exact": True,
                                    "checkpoint_after_layout_and_arena_republication": True,
                                    "forward_on_another_stream": True})
    finally:
        workspace.close()
        root.close()
    if shmem.debug_state()["reference_count"] != 0:
        raise RuntimeError("local autograd root did not close cleanly")


def run_autograd_validation(report: dict, output_dir: Path, *, smoke: bool = False) -> None:
    """Certify local autograd lifecycle against raw backward without clearing the numerical oracle gate."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    torch.npu.set_op_timeout_ms(120000)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    report.update(rank=rank, cp_size=dist.get_world_size(), cases=[], scope="local main autograd lifecycle",
                  fp32_oracle_acceptance=False, selected_kl=False, sparse_gradient_return=False,
                  package_path=hyper_parallel.__file__, torch_version=torch.__version__,
                  torch_npu_version=torch_npu.__version__,
                  full_model_acceptance=False, validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fixtures = [(32, "contiguous")] if smoke else [(heads, pattern) for heads in (32, 64)
                                                  for pattern in ("contiguous", "strided", "empty")]
    output_dir.mkdir(parents=True, exist_ok=True)
    for heads, pattern in fixtures:
        _autograd_fixture(report, heads, pattern, torch.device(f"npu:{local_rank}"))
        (output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report.update(status="passed", stage="complete", case_count=len(report["cases"]),
                  backward_calls=len(report["cases"]) * 4, loaded_library_sha256=_loaded_library_identity())
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist raw main-backward proof without claiming sparse return or complete training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--autograd-only", action="store_true")
    parser.add_argument("--reference-profile", type=Path)
    parser.add_argument("--profile-policy-sha256")
    args = parser.parse_args()
    report = {"status": "running", "scope": "local-query fused main backward with complete owner return",
              "backward": True, "sparse_gradient_return": False, "selected_kl": False,
              "full_model_acceptance": False, "performance_measurement": False}
    try:
        if args.autograd_only:
            if args.reference_profile is not None:
                raise ValueError("autograd-only does not certify the independent numerical profile")
            run_autograd_validation(report, args.output_dir, smoke=args.smoke)
        else:
            run_validation(report, args.output_dir, smoke=args.smoke, reference_profile=args.reference_profile,
                           profile_policy_sha256=args.profile_policy_sha256)
        if report["status"] != "passed":
            raise RuntimeError("local CP backward failed the independent FP32 oracle gate")
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{os.environ.get('RANK', '0')}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
