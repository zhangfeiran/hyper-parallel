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
"""Native CP/TP delayed backward and BF16 leaf accumulation against a global FP32 objective."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_training_validate import (
    _prefix_selection,
    _run_mode,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import (
    _SCALE,
    _metrics,
)
from hyper_parallel.core.multicore.examples.mega_dsa_training_oracle import (
    training_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    validate_selected_training_traces,
)


def _slice(value, field, rows, owners, heads, index_heads):
    if value is None:
        return None
    return value[owners] if field in (1, 3, 5) else value[rows, index_heads if field in (4, 6) else heads]


def _metric(value, target, field):
    if (value is None) != (target is None):
        raise RuntimeError(f'accumulation active/inactive gradient mismatch for field {field}')
    if value is None:
        return None
    return dict(_metrics(value, target), field=field)


def _acknowledged(transport):
    offset = transport.spec.control_offset
    controls = transport.arena[offset:offset+transport.spec.slots*640].view(torch.int64).reshape(-1, 80).cpu()
    for control in controls:
        sequence = int(control[0])
        if sequence//2**32 != transport.last_epoch or any(
                int(control[(2+peer)*8]) != sequence for peer in range(len(transport.members))):
            raise RuntimeError('accumulation shared return lacks final reader ACK for the current backward epoch')
    return transport.last_epoch


def _case(scenario, probe, meta, full, cotangent, _reference, epoch):
    rows, owners = list(meta.q_global_ids), list(meta.kv_global_ids)
    rank = dist.get_rank(probe.tp_forward.li.group)
    heads = slice(rank*probe.forward_backend.heads, (rank+1)*probe.forward_backend.heads)
    index_heads = slice(rank*probe.tp_forward.index_heads, (rank+1)*probe.tp_forward.index_heads)
    initial_cpu = tuple(value.cpu() for value in full)
    offsets = tuple(value.roll(1, 0)-value for value in initial_cpu)
    cpu_batches = (initial_cpu, tuple(value+offset for value, offset in zip(initial_cpu, offsets)))
    cpu_cotangents = (cotangent.cpu()/2, cotangent.cpu().roll(2, 0)/2)
    references = tuple(training_reference(batch, upstream, _prefix_selection((3, 10)), probe.normalization,
                                          .3, probe.mode, -1.75/2, (3, 10), attention_scale=_SCALE)
                       for batch, upstream in zip(cpu_batches, cpu_cotangents))
    parameters = tuple(torch.nn.Parameter(_slice(value, field, rows, owners, heads, index_heads).clone())
                       for field, value in enumerate(full))
    objectives, inputs, forward_records, backward_records, return_records = [], [], [], [], []
    native_forward = probe.tp_forward.forward
    native_backward = probe.backward_backend._backward_saved
    native_return = probe.shared_return.return_owner

    def _forward(*args, **kwargs):
        value = native_forward(*args, **kwargs)
        selected = value.global_indices[:, 0].cpu()
        torch.testing.assert_close(selected.sort(-1).values, _prefix_selection((3, 10))[rows].sort(-1).values,
                                   rtol=0, atol=0)
        phase = validate_selected_training_traces(tuple(trace.cpu() for trace in value.phase_traces),
                                                 probe.forward_backend.schedule, require_ld=False,
                                                 with_kl=probe.index_active)
        forward_records.append({'phases': phase, 'selected_set_exact': True})
        return value

    def _backward(*args, **kwargs):
        value = native_backward(*args, **kwargs)
        backward_records.append(probe.backward_backend.validate_trace(
            value, tuple(trace.cpu() for trace in value.phase_traces), value.transport_trace.cpu()))
        return value

    def _return(*args, **kwargs):
        value = native_return(*args, **kwargs)
        receipt = None
        if value.index_cp is not None:
            receipt = probe.shared_return.index_return.validate_trace(
                value.index_cp, value.index_cp.transport_trace.cpu())
        return_records.append({'index_receipt': receipt,
                               'tp_epoch': _acknowledged(probe.shared_return.tp_return.transport)})
        return value

    order = (0, 1) if epoch == 1 else (1, 0)
    with patch.object(probe.tp_forward, 'forward', side_effect=_forward), \
            patch.object(probe.backward_backend, '_backward_saved', side_effect=_backward), \
            patch.object(probe.shared_return, 'return_owner', side_effect=_return):
        for microbatch in range(2):
            forward_epoch = (epoch-1)*2+microbatch+1
            batch_meta = replace(meta, microbatch=microbatch, invocation=forward_epoch)
            prepared = probe.prepare(probe.forward_backend.workspace.prepare(batch_meta),
                                     epoch=forward_epoch, query_ids=meta.q_global_ids)
            local_offsets = tuple(_slice(offset, field, rows, owners, heads, index_heads).to(value.device)
                                  for field, (offset, value) in enumerate(zip(offsets, parameters)))
            local = tuple(value+offset*(microbatch == 1) for value, offset in zip(parameters, local_offsets))
            for value in local:
                value.retain_grad()
            inputs.append(local)
            output, loss = probe.forward(prepared, local[:4], local[4:])
            main_loss = (output*cpu_cotangents[microbatch][rows, heads].to(output.device)).sum()
            objective = loss*(-1.75/2) if probe.mode == 'kl_only' else main_loss
            if probe.mode == 'lm_plus_kl':
                objective = objective+loss*(-1.75/2)
            objectives.append(objective)
            forward_records[-1]['output_fp32_oracle'] = _metrics(output, references[microbatch][0][rows, heads])
            if probe.index_active:
                replicated = [None]*dist.get_world_size(probe.tp_forward.li.group)
                dist.all_gather_object(replicated, loss.detach().cpu(), group=probe.tp_forward.li.group)
                for peer in replicated:
                    torch.testing.assert_close(peer, replicated[0], rtol=1e-5, atol=1e-7)
                global_loss = loss.detach().clone()
                dist.all_reduce(global_loss, group=probe.forward_backend.layout.group)
                forward_records[-1]['cp_summed_loss_fp32_oracle'] = _metrics(
                    global_loss, references[microbatch][1])
        # 两个前向均完成后才反向，第二个前向已经覆盖共享 scratch。
        for microbatch in order:
            objectives[microbatch].backward()
    expected = tuple(tuple(_slice(value, field, rows, owners, heads, index_heads)
                           for field, value in enumerate(reference[2])) for reference in references)
    individual = []
    for microbatch in range(2):
        metrics = [_metric(value.grad, target, field)
                   for field, (value, target) in enumerate(zip(inputs[microbatch], expected[microbatch]))]
        individual.append({'microbatch': microbatch, 'gradients': [item for item in metrics if item is not None]})
    accumulated, updates, expected_sum = [], [], []
    initial = tuple(value.detach().clone() for value in parameters)
    for field, value in enumerate(parameters):
        target = None if expected[0][field] is None else expected[0][field]+expected[1][field]
        expected_sum.append(target)
        metric = _metric(value.grad, target, field)
        if metric is not None:
            accumulated.append(metric)
            # BF16 叶子的累积必须与实际返回的两个 BF16 VJP 顺序累积逐位一致。
            rounded = inputs[order[0]][field].grad+inputs[order[1]][field].grad
            torch.testing.assert_close(value.grad, rounded, rtol=0, atol=0)
    optimizer = torch.optim.SGD(parameters, lr=.01)
    optimizer.step()
    for field, target in enumerate(expected_sum):
        if target is not None:
            updated = (initial[field].float().cpu()-.01*target).to(initial[field].dtype)
            updates.append(dict(_metrics(parameters[field], updated), field=field))
    compared = accumulated+[item for batch in individual for item in batch['gradients']]
    for record in forward_records:
        compared.append(record['output_fp32_oracle'])
        if 'cp_summed_loss_fp32_oracle' in record:
            compared.append(record['cp_summed_loss_fp32_oracle'])
    epochs = [record['tp_epoch'] for record in return_records]
    main_backward_count = 2 if probe.mode != 'kl_only' else 0
    protocol = (len(forward_records) == 2 and len(backward_records) == main_backward_count
                and len(return_records) == 2 and epochs == [(epoch-1)*2+1, epoch*2]
                and all((record['index_receipt'] is not None) == probe.index_active for record in return_records))
    passed = (protocol and all(item['relative_l2'] <= .02 or item['max_abs'] <= 2e-5 for item in compared)
              and all(item['pointwise_pass'] for item in updates))
    scenario['cases'].append({'passed': passed, 'epoch': epoch, 'microbatch_count': 2,
                              'backward_order': order, 'mode': probe.mode, 'forward_records': forward_records,
                              'native_main_backward_receipts': backward_records,
                              'owner_return_records': return_records,
                              'per_microbatch_fp32_oracle': individual,
                              'accumulated_gradients_fp32_oracle': accumulated,
                              'sgd_updates': updates, 'bf16_leaf_accumulation_bit_exact': True,
                              'all_forwards_before_backward': True, 'protocol_passed': protocol})
    if not passed:
        raise RuntimeError('CP TP delayed backward or accumulated SGD differs from the global FP32 objective')
    optimizer.zero_grad(set_to_none=True)
    return {'passed': True}


def main() -> None:
    """Validate two distinct microbatches, both delayed-backward orders and actual leaf SGD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'scenarios': [], 'cp_tp_model_acceptance': False, 'performance_acceptance': False,
              'scope': 'seven projection-output parameters; two equal-size packed microbatches; BF16 accumulation',
              'microbatch_objective_scale': .5, 'auxiliary_upstream_per_microbatch': -1.75/2,
              'oracle': 'independent full-head FP32; sum of two half-scaled unsharded objectives',
              'acceptance': {'relative_l2_or_max_abs': [.02, 2e-5], 'sgd_rtol': .02, 'sgd_atol': 2e-5}}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('accumulation validator requires four root members')
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
                for mode in ('lm_only', 'kl_only', 'lm_plus_kl'):
                    _run_mode(report, topology, groups, pattern, mode, case_runner=_case)
        report.update(status='passed', loaded_library_sha256=_loaded_library_identity())
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+'\n')
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
