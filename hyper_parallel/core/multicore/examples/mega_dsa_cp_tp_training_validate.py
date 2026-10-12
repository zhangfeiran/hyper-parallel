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
"""Native CP/TP primitive training against independent full-head FP32 gradients and SGD."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import (
    _SCALE,
    _inputs,
    _metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_oracle import (
    training_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedDsaCpTpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    validate_selected_training_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot


def _prefix_selection(lengths):
    meta = DsaBatchMeta.packed(lengths)
    selected = torch.full((sum(lengths), 2048), -1, dtype=torch.int32)
    for token in range(sum(lengths)):
        _sequence, position = meta.sequence_position(token)
        selected[token, :position+1] = torch.arange(token-position, token+1)
    return selected


def _case(scenario, probe, meta, full, cotangent, reference, epoch):
    prepared = probe.prepare(probe.forward_backend.workspace.prepare(replace(meta, invocation=epoch)),
                              epoch=epoch, query_ids=meta.q_global_ids)
    rows, owners = list(meta.q_global_ids), list(meta.kv_global_ids)
    rank = dist.get_rank(probe.tp_forward.li.group)
    heads = slice(rank*probe.forward_backend.heads, (rank+1)*probe.forward_backend.heads)
    index_heads = slice(rank*probe.tp_forward.index_heads, (rank+1)*probe.tp_forward.index_heads)
    local = tuple(torch.nn.Parameter(value.clone()) for value in (
        full[0][rows, heads], full[1][owners], full[2][rows, heads], full[3][owners],
        full[4][rows, index_heads], full[5][owners], full[6][rows, index_heads]))
    forwards, backwards, returns = [], [], []
    native_forward = probe.tp_forward.forward
    native_backward = probe.backward_backend._backward_saved
    native_return = probe.shared_return.return_owner
    def _forward(*args, **kwargs):
        value = native_forward(*args, **kwargs)
        forwards.append(value)
        return value
    def _backward(*args, **kwargs):
        value = native_backward(*args, **kwargs)
        backwards.append(value)
        return value
    def _return(*args, **kwargs):
        value = native_return(*args, **kwargs)
        returns.append(value)
        return value
    with patch.object(probe.tp_forward, 'forward', side_effect=_forward), \
            patch.object(probe.backward_backend, '_backward_saved', side_effect=_backward), \
            patch.object(probe.shared_return, 'return_owner', side_effect=_return):
        output, loss = probe.forward(prepared, local[:4], local[4:])
        objective = loss*-1.75 if probe.mode == 'kl_only' else (output*cotangent[rows, heads]).sum()
        if probe.mode == 'lm_plus_kl':
            objective = objective+loss*-1.75
        gradients = torch.autograd.grad(objective, local, allow_unused=True)
    expected = []
    for field, value in enumerate(reference[2]):
        if value is None:
            expected.append(None)
        elif field in (1, 3, 5):
            expected.append(value[owners])
        else:
            expected.append(value[rows, index_heads if field in (4, 6) else heads])
    selected = forwards[0].global_indices[:, 0].cpu()
    torch.testing.assert_close(selected.sort(-1).values, _prefix_selection((3, 10))[rows].sort(-1).values,
                               rtol=0, atol=0)
    phase = validate_selected_training_traces(tuple(trace.cpu() for trace in forwards[0].phase_traces),
                                             probe.forward_backend.schedule, require_ld=False,
                                             with_kl=probe.index_active)
    main_receipts = [probe.backward_backend.validate_trace(value, tuple(t.cpu() for t in value.phase_traces),
                                                           value.transport_trace.cpu()) for value in backwards]
    if len(returns) != 1 or (returns[0].index_cp is not None) != probe.index_active:
        raise RuntimeError('CP TP training has an unexpected shared/index return count')
    index_receipt = None
    if returns[0].index_cp is not None:
        value = returns[0].index_cp
        index_receipt = probe.shared_return.index_return.validate_trace(value, value.transport_trace.cpu())
    transport = probe.shared_return.tp_return.transport
    offset = transport.spec.control_offset
    controls = transport.arena[offset:offset+transport.spec.slots*640].view(torch.int64).reshape(-1, 80).cpu()
    for control in controls:
        sequence = int(control[0])
        if sequence//2**32 != epoch or any(int(control[(2+peer)*8]) != sequence
                                         for peer in range(len(transport.members))):
            raise RuntimeError('CP TP shared return lacks matching final reader ACK')
    metrics = []
    for field, (actual, target) in enumerate(zip(gradients, expected)):
        if target is None:
            if actual is not None:
                raise RuntimeError('inactive CP TP branch received a gradient')
        else:
            if actual is None:
                raise RuntimeError('active CP TP branch lacks a gradient')
            metric = _metrics(actual, target)
            metric['field'] = field
            metrics.append(metric)
    active = tuple(field for field, value in enumerate(gradients) if value is not None)
    initial = tuple(value.detach().clone() for value in local)
    for value, gradient in zip(local, gradients):
        value.grad = gradient
    optimizer = torch.optim.SGD(local, lr=.01)
    optimizer.step()
    updates = []
    for field in active:
        target = initial[field].float().cpu()-.01*expected[field].float()
        metric = _metrics(local[field], target.to(local[field].dtype))
        metric['field'] = field
        updates.append(metric)
    output_metric = _metrics(output, reference[0][rows, heads])
    loss_metric = None
    if probe.index_active:
        replicated = [None]*len(transport.members)
        dist.all_gather_object(replicated, loss.detach().cpu(), group=transport.group)
        for peer in replicated:
            torch.testing.assert_close(peer, replicated[0], rtol=1e-5, atol=1e-7)
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, group=probe.forward_backend.layout.group)
        loss_metric = _metrics(global_loss, reference[1])
    compared = metrics+[output_metric]+([] if loss_metric is None else [loss_metric])
    passed = all(item['relative_l2'] <= .02 or item['max_abs'] <= 2e-5 for item in compared)
    passed = passed and all(item['pointwise_pass'] for item in updates)
    scenario['cases'].append({'passed': passed, 'epoch': epoch, 'queries': len(rows), 'owners': len(owners),
                              'gradients_fp32_oracle': metrics, 'sgd_updates': updates,
                              'output_fp32_oracle': output_metric, 'cp_summed_loss_fp32_oracle': loss_metric,
                              'active_fields': active, 'mode': probe.mode, 'forward_phases': phase,
                              'native_cp_main_receipts': main_receipts, 'native_cp_index_receipt': index_receipt,
                              'shared_return_count': 1,
                              'cp_index_return_active': probe.index_active, 'final_tp_readers_acknowledged': True})
    if not passed:
        raise RuntimeError('CP TP primitive training differs from FP32 gradients or SGD; inspect field metrics')
    return {'passed': True}


def _run_mode(report, topology, groups, pattern, mode, *, case_runner=None):
    cp_members, cp_group, tp_group = groups
    lengths = (3, 10)
    size = dist.get_world_size(tp_group)
    global_heads = 64 if size == 1 else 128
    root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
    meta = _metadata(lengths, pattern, len(cp_members), dist.get_rank(cp_group))
    meta = replace(meta, cp_ranks=cp_members, root_pes=cp_members)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(len(cp_members))))
    workspace = MegaDsaWorkspace(root, 'cp-training',
                                 DsaWorkspaceSpec(capacity, sum(lengths), capacity, len(cp_members)),
                                 cp_group=cp_group)
    li = TpTileSumProbe(root, 'training-li', TpTileSumSpec(14, 512), tp_group=tp_group)
    kl = TpTileSumProbe(root, 'training-kl', TpTileSumSpec(14, 2048), tp_group=tp_group)
    owner = TpTileSumProbe(root, 'training-owner', TpTileSumSpec(14, 512), tp_group=tp_group)
    scenario = {'topology': topology, 'pattern': pattern, 'mode': mode, 'cases': []}
    report['scenarios'].append(scenario)
    try:
        workspace.bind()
        for transport in (li, kl, owner):
            transport.bind()
        meta = replace(meta, heap_generation=root.generation)
        cp = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=global_heads//size,
                                   attention_scale=_SCALE, schedule=MixedSfaSchedule(7))
        forward = FusedDsaCpTpForwardProbe(cp, li, kl)
        normalization = DsaLossNormalization(sum(lengths))
        probe = FusedDsaLocalCpTpTrainingProbe(forward, owner, normalization=normalization,
                                              loss_coeff=.3, mode=mode)
        cpu, cpu_cotangent = _inputs(global_heads, torch.bfloat16, lengths)
        reference = training_reference(cpu, cpu_cotangent, _prefix_selection(lengths), normalization,
                                       .3, mode, -1.75, lengths, attention_scale=_SCALE)
        full = tuple(value.to(root.device) for value in cpu)
        cotangent = cpu_cotangent.to(root.device)
        for epoch in (1, 2):
            runner = _case if case_runner is None else case_runner
            collective_validation_check(partial(runner, scenario, probe, meta, full, cotangent, reference, epoch))
        scenario['status'] = 'passed'
    finally:
        for transport in (owner, kl, li):
            transport.close()
        workspace.close()
        root.close()


def _run(report, topology, groups, pattern):
    for mode in ('lm_only', 'kl_only', 'lm_plus_kl'):
        _run_mode(report, topology, groups, pattern, mode)


def main() -> None:
    """Archive four-card primitive F+B+SGD against FP32 without claiming full-model acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'cp_tp_model_acceptance': False,
              'performance_acceptance': False,
              'scope': 'CP TP primitive F+B+SGD; short packed prefixes; independent full-head FP32 oracle'}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('fused CP TP validator requires four root members')
        rank = dist.get_rank()
        single = tuple(((peer,), dist.new_group([peer])) for peer in range(4))
        cp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 1), (2, 3)))
        tp_pairs = tuple((members, dist.new_group(list(members))) for members in ((0, 2), (1, 3)))
        cp_pair = next(item for item in cp_pairs if rank in item[0])
        tp_pair = next(item for item in tp_pairs if rank in item[0])
        topologies = [('cp4_tp1', (tuple(range(4)), dist.group.WORLD, single[rank][1])),
                      ('cp2_tp2', (*cp_pair, tp_pair[1])), ('cp1_tp4', (*single[rank], dist.group.WORLD))]
        for topology, groups in topologies:
            for pattern in ('strided', 'empty'):
                _run(report, topology, groups, pattern)
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
