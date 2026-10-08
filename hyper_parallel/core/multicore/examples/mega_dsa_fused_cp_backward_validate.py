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
"""Real CP fused backward, raw FP32 accumulator parity and owner-return certification."""

import argparse
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

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import (
    _SCALE,
    _full_states,
    _local_states,
    _metadata,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_backward import (
    FusedDsaCpAttentionProbe,
    FusedDsaCpBackwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_attention import (
    mixed_sfa_backward_probe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _cotangent(meta, heads, repeat, device):
    generator = torch.Generator().manual_seed(20261009 + heads + repeat)
    value = (torch.randn((meta.global_valid_queries, heads, 512), generator=generator) * 0.1).bfloat16()
    # Unequal rank contributions expose missing reductions and accidental world-size multiplication.
    rows = list(meta.q_global_ids)
    return (value[rows] * (meta.cp_rank + 1)).bfloat16().to(device)


def _owner_oracle(result, backend):
    size = backend.backend.layout.cp_size
    gathered = torch.empty((size * result.partials.shape[0], 1088), dtype=torch.float32,
                           device=result.partials.device)
    dist.all_gather_into_tensor(gathered, result.partials)
    partials = gathered.cpu().reshape(size, result.partials.shape[0], 1088)
    compressed = partials[0, :, :512] + partials[0, :, 576:]
    rope = partials[0, :, 512:576].clone()
    for rank in range(1, size):
        compressed = compressed + (partials[rank, :, :512] + partials[rank, :, 576:])
        rope = rope + partials[rank, :, 512:576]
    rows = list(backend.backend.layout.batch_meta.kv_global_ids)
    for actual, expected in zip(result.owner_fp32, (compressed[rows], rope[rows])):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.cpu().view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
    return {"ordered_fp32_owner_bits_exact": True}


def _reference(result, forward, backend, cotangent):
    saved = forward.saved
    lengths = backend.backend.native_layout.length_tensor
    full = torch.zeros_like(saved.states[0])
    full.index_copy_(0, backend.backend.layout.local_query_ids, cotangent)
    native, _, retained = mixed_sfa_backward_probe(saved.states, saved.indices, lengths, backend.backend.config,
                                                   full, saved.forward, _SCALE, backend.backend.schedule)
    bitwise = []
    for actual, expected in zip(result.native_gradients, native):
        actual_cpu, expected_cpu = actual.cpu(), expected.cpu()
        bitwise.append(bool(torch.equal(actual_cpu.view(torch.int16), expected_cpu.view(torch.int16))))
        torch.testing.assert_close(actual_cpu.float(), expected_cpu.float(), rtol=0.02, atol=2e-5)
    snapshot = retained.cpu()
    offsets = tuple(int(result.transport_trace[index].cpu()) for index in (20, 21))
    total = saved.states[0].shape[0]
    key = snapshot[offsets[0]:offsets[0] + total * 576 * 4].view(torch.float32).reshape(total, 576)
    value = snapshot[offsets[1]:offsets[1] + total * 512 * 4].view(torch.float32).reshape(total, 512)
    reference = torch.cat((key, value), dim=1)
    actual = result.partials.cpu()
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=2e-5)
    delta = (actual.double() - reference.double()).flatten()
    relative_l2 = float(delta.norm() / reference.double().norm().clamp_min(1e-30))
    return {"native_gradients_bitwise": bitwise, "native_gradients_existing_pointwise_threshold": True,
            "pointwise_threshold": {"rtol": 0.02, "atol": 2e-5},
            "fp32_accumulator_reference_relative_l2": relative_l2,
            "fp32_accumulator_reference_max_abs": float(delta.abs().max())}


def _run_fixture(report, lengths, heads, pattern, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata(lengths, pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-fused-cp-gradient",
                                 DsaWorkspaceSpec(capacity, sum(lengths), capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    try:
        for groups in (1, 2, 7, 19):
            prepared = workspace.prepare(meta)
            forward_backend = FusedDsaCpForwardProbe(workspace, prepared, heads=heads,
                                                     attention_scale=_SCALE, schedule=MixedSfaSchedule(groups))
            backward_backend = FusedDsaCpBackwardProbe(forward_backend)
            for repeat in range(3):
                report["stage"] = {"lengths": lengths, "heads": heads, "pattern": pattern,
                                   "groups": groups, "repeat": repeat}
                _, main_states, index, _ = _full_states(lengths, heads, "random_signed_fp32_weights", repeat, device)
                local_main, local_index = _local_states(meta, main_states, index)
                invocation = workspace.prepare(replace(meta, invocation=workspace.transport_epoch + 1,
                                                       layer=repeat % 2, microbatch=repeat))
                forward = forward_backend.forward(invocation, local_main, local_index)
                cotangent = _cotangent(meta, heads, repeat, device)
                result = backward_backend.backward(forward, cotangent)
                torch.npu.synchronize()
                evidence = backward_backend.validate_trace(result, tuple(trace.cpu() for trace in result.phase_traces),
                                                            result.transport_trace.cpu())
                proof = _owner_oracle(result, backward_backend)
                reference = _reference(result, forward, backward_backend, cotangent)
                for local, native in ((result.gradients[0], result.native_gradients[0]),
                                      (result.gradients[2], result.native_gradients[3])):
                    torch.testing.assert_close(local.cpu(), native[forward_backend.layout.local_query_ids].cpu(),
                                               rtol=0, atol=0)
                report["cases"].append({**report["stage"], "trace": evidence, **proof, **reference})
    finally:
        workspace.close()
        root.close()


def _autograd_lifecycle(report: dict, device: torch.device, pattern: str) -> None:
    rank, size = dist.get_rank(), dist.get_world_size()
    meta = _metadata((3, 10), pattern, size, rank)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(size)))
    root = SharedShmemRoot(device, root_group=dist.group.WORLD)
    workspace = MegaDsaWorkspace(root, "dsa-cp-autograd", DsaWorkspaceSpec(capacity, 13, capacity, size))
    workspace.bind()
    meta = replace(meta, heap_generation=root.generation)
    try:
        report["stage"] = {"autograd": pattern, "step": "raw_and_bridge_forward"}
        invocation = workspace.prepare(meta)
        forward_backend = FusedDsaCpForwardProbe(workspace, invocation, heads=32, attention_scale=_SCALE,
                                                 schedule=MixedSfaSchedule(7))
        raw_backward = FusedDsaCpBackwardProbe(forward_backend)
        autograd = FusedDsaCpAttentionProbe(forward_backend)
        _, main_states, index, _ = _full_states((3, 10), 32, "random_signed_fp32_weights", 0, device)
        local_main, local_index = _local_states(meta, main_states, index)
        cotangent = _cotangent(meta, 32, 0, device)
        raw = forward_backend.forward(invocation, local_main, local_index)
        inputs_ready = torch.npu.Event()
        inputs_ready.record(torch.npu.current_stream())
        first_raw = raw_backward.backward(raw, cotangent)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            inputs_ready.wait()
            second_raw = raw_backward.backward(raw, cotangent)
        expected = first_raw.gradients
        main_inputs = tuple(tensor.clone().requires_grad_() for tensor in local_main)
        index_inputs = tuple(tensor.clone().requires_grad_() for tensor in local_index)
        output = autograd.attention(invocation, main_inputs, index_inputs)
        later = workspace.prepare(replace(meta, layer=2, microbatch=3, invocation=workspace.transport_epoch + 1))
        changed = tuple((tensor.detach() + 0.0625).bfloat16() for tensor in local_main)
        forward_backend.forward(later, changed, local_index)
        report["stage"] = {"autograd": pattern, "step": "delayed_and_retained_backward"}
        first = torch.autograd.grad(output, (*main_inputs, *index_inputs), grad_outputs=cotangent,
                                    allow_unused=True, retain_graph=True)
        second = torch.autograd.grad(output, main_inputs, grad_outputs=cotangent)
        torch.npu.synchronize()
        for actual, reference in zip(second_raw.gradients, expected):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0.02, atol=2e-5)
        _owner_oracle(first_raw, raw_backward)
        _owner_oracle(second_raw, raw_backward)
        torch.testing.assert_close(output.cpu(), raw.output.cpu(), rtol=0, atol=0)
        for actual, replay, reference in zip(first[:4], second, expected):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0.02, atol=2e-5)
            torch.testing.assert_close(replay.cpu(), reference.cpu(), rtol=0.02, atol=2e-5)
        if any(gradient is not None for gradient in first[4:]):
            raise RuntimeError("main CP attention installed an indexer/KL gradient path")
        checkpoint_inputs = tuple(tensor.clone().requires_grad_() for tensor in local_main)
        report["stage"] = {"autograd": pattern, "step": "non_reentrant_checkpoint"}

        def _attention(*states):
            return autograd.attention(invocation, states, local_index)

        checkpoint_output = checkpoint(_attention, *checkpoint_inputs, use_reentrant=False)
        checkpoint_gradients = torch.autograd.grad(checkpoint_output, checkpoint_inputs, grad_outputs=cotangent)
        for actual, reference in zip(checkpoint_gradients, expected):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0.02, atol=2e-5)
        report.setdefault("autograd", []).append(
            {"pattern": pattern, "retained_graph": True, "main_gradient_count": 4,
             "indexer_gradients_absent": True, "output_raw_exact": True, "raw_backward_parity": True,
             "delayed_after_arena_republication": True, "non_reentrant_checkpoint": True,
             "cross_stream_raw_backward_before_observation": True})
    finally:
        workspace.close()
        root.close()


def run_validation(report: dict, output_dir: Path, *, smoke: bool, long_history: bool) -> None:
    """Verify raw backward and native owner transport with unequal local Q cotangents."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    if smoke:
        torch.npu.set_op_timeout_ms(30000)
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    report.update(rank=dist.get_rank(), cp_size=dist.get_world_size(), cases=[])
    fixtures = [((3, 10), 32, "contiguous")]
    if not smoke:
        fixtures.extend(((3, 10), heads, pattern) for heads in (32, 64) for pattern in ("strided", "zigzag", "empty"))
    if long_history:
        fixtures.extend(((2176, 2240), heads, "contiguous") for heads in (32, 64))
    output_dir.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        _run_fixture(report, *fixture, device)
        (output_dir / f"rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not smoke:
        for pattern in ("zigzag", "empty"):
            _autograd_lifecycle(report, device, pattern)
    report.update(status="passed", stage="complete", case_count=len(report["cases"]))
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    """Persist complete per-rank functional proof or the exact failing phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--long-history", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {"status": "running", "scope": "single-kernel CP backward and FP32 owner return",
              "autograd": [], "full_model_acceptance": False,
              "fp32_oracle_acceptance": "separate unresolved gate", "selected_kl": False}
    try:
        run_validation(report, args.output_dir, smoke=args.smoke, long_history=args.long_history)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
