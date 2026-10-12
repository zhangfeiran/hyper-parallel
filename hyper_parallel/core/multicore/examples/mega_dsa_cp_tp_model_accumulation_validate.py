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
"""Actual CP/TP model shared-weight accumulation, delayed backward and checkpoint acceptance."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_accumulation_validate import (
    _acknowledged,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_global_validate import (
    _topologies,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_validate import (
    _compare,
    _execute,
    _expected_local,
    _reduce_parameters,
    _reference,
    _run,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import (
    _configure,
    _metric,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import model_inputs
from hyper_parallel.core.multicore.modules.mega_dsa.fused_tp_model_boundary import (
    FusedDsaCpTpAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_evaluate import (
    single_sgd_control,
)
from hyper_parallel.core.multicore.modules.mega_dsa.parameter_sum import (
    ordered_parameter_sum,
)
from hyper_parallel.core.multicore.modules.mega_dsa.selected_requests import (
    validate_selected_training_traces,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_projections import (
    gather_dsa_tp_state_dict,
)


def _sum_gradients(batches, *, scale=1.):
    combined = []
    for values in zip(*batches):
        if all(value is None for value in values):
            combined.append(None)
        elif any(value is None for value in values):
            raise ValueError('model accumulation gradient availability changed between microbatches')
        else:
            combined.append(sum(value.detach().float() for value in values)*scale)
    return tuple(combined)


def _exact(actual, expected):
    return {'passed': actual is expected if actual is None or expected is None else bool(torch.equal(actual, expected)),
            'metric': _metric(actual, expected)}


def _delayed(model, probe, meta, batches, embeddings, cotangents, replay):
    records = {'forwards': [], 'main_backwards': [], 'owner_returns': []}
    original_forward = probe.tp_forward.forward
    original_backward = probe.backward_backend._backward_saved
    original_return = probe.shared_return.return_owner
    first_owner_epoch = probe.shared_return.tp_return.transport.last_epoch

    def _capture_forward(*args, **kwargs):
        value = original_forward(*args, **kwargs)
        phase = validate_selected_training_traces(tuple(item.cpu() for item in value.phase_traces),
                                                 probe.forward_backend.schedule, require_ld=False,
                                                 with_kl=probe.index_active)
        records['forwards'].append({'phases': phase, 'indices': value.global_indices[:, 0].cpu().clone(),
                                    'microbatch': value.saved.batch_meta.microbatch,
                                    'invocation': value.saved.batch_meta.invocation,
                                    'raw_kl_loss': (None if not probe.index_active
                                                    else value.kl_loss.detach().cpu().clone())})
        return value

    def _capture_backward(*args, **kwargs):
        value = original_backward(*args, **kwargs)
        records['main_backwards'].append(probe.backward_backend.validate_trace(
            value, tuple(item.cpu() for item in value.phase_traces), value.transport_trace.cpu()))
        return value

    def _capture_return(*args, **kwargs):
        value = original_return(*args, **kwargs)
        receipt = None if value.index_cp is None else probe.shared_return.index_return.validate_trace(
            value.index_cp, value.index_cp.transport_trace.cpu())
        records['owner_returns'].append({'epoch': _acknowledged(probe.shared_return.tp_return.transport),
                                         'index_receipt': receipt})
        return value

    def _forward(value, *, microbatch):
        epoch = probe.tp_forward.li.last_epoch+1
        prepared = probe.forward_backend.workspace.prepare(replace(meta, invocation=epoch, microbatch=microbatch))
        invocation = probe.prepare(prepared, epoch=epoch, query_ids=meta.q_global_ids)
        return model(value, position_embeddings=embeddings, dsa_training=probe, dsa_invocation=invocation)[0]

    outputs, objectives = [], []
    order = (0, 1) if replay == 'ordered' else (1, 0)
    with patch.object(probe.tp_forward, 'forward', side_effect=_capture_forward), \
            patch.object(probe.backward_backend, '_backward_saved', side_effect=_capture_backward), \
            patch.object(probe.shared_return, 'return_owner', side_effect=_capture_return):
        for microbatch, (hidden, cotangent) in enumerate(zip(batches, cotangents)):
            # 固定 microbatch，避免 checkpoint 闭包在反向时读取循环末尾的身份。
            def _bound_forward(value, identity=microbatch):
                return _forward(value, microbatch=identity)
            output = (checkpoint(_bound_forward, hidden, use_reentrant=False)
                      if replay == 'checkpoint' else _bound_forward(hidden))
            outputs.append(output)
            objectives.append((output.float()*cotangent).sum())
        capture = getattr(model, 'parameter_vjp_capture', None)
        if capture is not None:
            capture.begin_backward()
        try:
            for microbatch in order:
                objectives[microbatch].backward()
            if capture is not None:
                capture.finish_backward()
        except Exception:
            if capture is not None:
                capture.discard()
            raise
    records['backward_order'] = order
    records['owner_epoch_start'] = first_owner_epoch
    return outputs, records


def _case(scenario, template, groups, workspace, forward, owner, meta, mode, replay, output_dir,
          *, repeat_isolated=False, capture_parameter_vjps=False, archive_failures=True,
          parameter_reduction_order='collective', projection_reduction_order='collective',
          backward_compute_groups=None, fixture_seed=None, group_private_partials=False):
    _cp_members, cp_group, tp_group = groups
    source = copy.deepcopy(template).to(workspace.root.device)
    _configure(source, mode)
    original_parameters = dict(source.named_parameters())
    model = FusedDsaCpTpAttention(module=source, context={
        'dsa_tp_group': tp_group, 'dsa_capture_output_parameter_vjps': capture_parameter_vjps,
        'dsa_capture_qkv_parameter_vjps': capture_parameter_vjps,
        'dsa_capture_kv_parameter_vjps': capture_parameter_vjps,
        'dsa_capture_query_parameter_vjps': capture_parameter_vjps,
        'dsa_projection_reduction_order': projection_reduction_order})
    capture = model.parameter_vjp_capture
    probe = FusedDsaLocalCpTpTrainingProbe(forward, owner, normalization=DsaLossNormalization(13),
                                          loss_coeff=model.dsa_loss_coeff,
                                          mode='lm_plus_kl' if mode == 'joint' else 'lm_only',
                                          backward_schedule=(None if backward_compute_groups is None else
                                                             MixedSfaSchedule(backward_compute_groups)),
                                          group_private_partials=group_private_partials)
    global_meta = DsaBatchMeta.packed((3, 10))
    hidden, embeddings_cpu = model_inputs(global_meta, dtype=torch.bfloat16,
                                          **({} if fixture_seed is None else {'seed': fixture_seed+2}))
    hidden_cpu = (hidden, hidden.roll(1, 1))
    generator = torch.Generator().manual_seed(20261011 if fixture_seed is None else fixture_seed+3)
    cotangent = torch.randn(hidden.shape, generator=generator)*.01
    cotangents_cpu = (cotangent, cotangent.roll(2, 1))
    ids = list(meta.kv_global_ids)
    device = workspace.root.device
    embeddings = tuple(value[:, ids].to(device) for value in embeddings_cpu)
    stock, oracle = [], []
    set_aux_loss_scale(torch.tensor(-1.75, device=device))
    for hidden_value, upstream in zip(hidden_cpu, cotangents_cpu):
        stock_value, oracle_value = _reference(template, mode, hidden_value, embeddings_cpu,
                                              upstream, global_meta, device)
        stock.append(stock_value)
        oracle.append(oracle_value)
    set_aux_loss_scale(torch.tensor(-1.75/2, device=device))
    isolated, isolated_captures = [], []
    for hidden_value, upstream in zip(hidden_cpu, cotangents_cpu):
        local_hidden = hidden_value[:, ids].to(device).requires_grad_()
        isolated.append(_execute(model, probe, meta, local_hidden, embeddings, upstream[:, ids].to(device)/2, 'plain'))
        if capture is not None:
            isolated_captures.append(capture.take())
    repeat_metrics = []
    if repeat_isolated:
        for microbatch, (hidden_value, upstream) in enumerate(zip(hidden_cpu, cotangents_cpu)):
            local_hidden = hidden_value[:, ids].to(device).requires_grad_()
            repeated = _execute(model, probe, meta, local_hidden, embeddings, upstream[:, ids].to(device)/2, 'plain')
            if capture is not None:
                capture.take()
            metrics = {'output': _exact(repeated[0], isolated[microbatch][0])}
            metrics.update({name: _exact(actual, expected) for name, actual, expected in
                            zip(('hidden', *dict(model.named_parameters())), repeated[1], isolated[microbatch][1])})
            repeat_metrics.append({'microbatch': microbatch, 'fields': metrics})
    batches = tuple(value[:, ids].to(device).requires_grad_() for value in hidden_cpu)
    cotangents = tuple(value[:, ids].to(device)/2 for value in cotangents_cpu)
    model.zero_grad(set_to_none=True)
    outputs, records = _delayed(model, probe, meta, batches, embeddings, cotangents, replay)
    captured = None if capture is None else capture.take()
    exact = {}
    for microbatch in range(2):
        exact[f'output/{microbatch}'] = _exact(outputs[microbatch], isolated[microbatch][0])
        exact[f'hidden/{microbatch}'] = _exact(batches[microbatch].grad, isolated[microbatch][1][0])
    order = records['backward_order']
    for field, (name, parameter) in enumerate(model.named_parameters(), 1):
        values = [isolated[microbatch][1][field] for microbatch in order]
        expected = None if values[0] is None else values[0]+values[1]
        exact[name] = _exact(parameter.grad, expected)
    capture_exact = {}
    if captured is not None:
        for name, gradient in captured.items():
            values = [isolated_captures[microbatch][name] for microbatch in order]
            if any(value is None for value in values) != all(value is None for value in values):
                raise RuntimeError('captured VJP availability changed between microbatches')
            expected = None if values[0] is None else values[0]+values[1]
            capture_exact[name] = _exact(gradient, expected)
            if gradient is not None and gradient.dtype != torch.float32:
                raise RuntimeError('low-precision model accumulation must capture FP32 VJPs')
    local_gradients = (None, *(parameter.grad for parameter in model.parameters()))
    same_source = ({} if captured is None else
                   {name: None if gradient is None else gradient.detach().clone(memory_format=torch.contiguous_format)
                    for name, gradient in captured.items()})
    reduced = _reduce_parameters(model, local_gradients, cp_group, captured_gradients=captured,
                                 reduction_order=parameter_reduction_order)
    reduction_controls, same_source_controls = {}, {}
    if captured is not None:
        for field, (name, _parameter) in enumerate(model.named_parameters(), 1):
            if name in captured and captured[name] is not None:
                expected = sum(isolated_captures[microbatch][name] for microbatch in order)
                if parameter_reduction_order == 'group_rank':
                    expected = ordered_parameter_sum(expected, cp_group)
                else:
                    dist.all_reduce(expected, group=cp_group)
                reduction_controls[name] = _exact(reduced[field], expected.to(reduced[field].dtype))
                repeated = same_source[name]
                if parameter_reduction_order == 'group_rank':
                    repeated = ordered_parameter_sum(repeated, cp_group)
                else:
                    dist.all_reduce(repeated, group=cp_group)
                same_source_controls[name] = _exact(reduced[field], repeated.to(reduced[field].dtype))
    stock_parameters = _sum_gradients([value[2][1:] for value in stock], scale=.5)
    oracle_parameters = _sum_gradients([value[2][1:] for value in oracle], scale=.5)
    names = tuple(dict(model.named_parameters()))
    actual = dict(zip(names, reduced[1:]))
    stock_expected = _expected_local(model, stock[0][1], (stock[0][2][0], *stock_parameters), None, ids)
    oracle_expected = _expected_local(model, oracle[0][1], (oracle[0][2][0], *oracle_parameters), None, ids)
    stock_metrics = _compare(actual, stock_expected)
    oracle_metrics = _compare(actual, oracle_expected)
    per_batch = []
    selected_exact = True
    for microbatch in range(2):
        fields = {'output': outputs[microbatch], 'hidden': batches[microbatch].grad}
        stock_targets = {'output': stock[microbatch][1][:, ids], 'hidden': stock[microbatch][2][0][:, ids]/2}
        oracle_targets = {'output': oracle[microbatch][1][:, ids], 'hidden': oracle[microbatch][2][0][:, ids]/2}
        if probe.index_active:
            loss = records['forwards'][microbatch]['raw_kl_loss'].to(device)
            replicas = [None]*dist.get_world_size(tp_group)
            dist.all_gather_object(replicas, loss.cpu(), group=tp_group)
            for value in replicas:
                torch.testing.assert_close(value, replicas[0], rtol=1e-5, atol=1e-7)
            dist.all_reduce(loss, group=cp_group)
            fields['loss'] = loss*probe.loss_coeff*probe.normalization.local_sum_scale
            stock_targets['loss'] = stock[microbatch][3]
            oracle_targets['loss'] = oracle[microbatch][3]
        per_batch.append({'stock': _compare(fields, stock_targets), 'fp32': _compare(fields, oracle_targets)})
        selected = records['forwards'][microbatch]['indices']
        selected_exact = selected_exact and all(sorted(row[row>=0].tolist()) == sorted(target[target>=0].tolist())
                                                for row, target in zip(selected, stock[microbatch][4][ids]))
    for parameter, gradient in zip(model.parameters(), reduced[1:]):
        parameter.grad = gradient
    step_controls = {name: single_sgd_control(parameter, parameter.grad, learning_rate=.01)
                     for name, parameter in model.named_parameters()}
    for model_reference, gradients in ((stock[0][0], stock_parameters), (oracle[0][0], oracle_parameters)):
        for parameter, gradient in zip(model_reference.parameters(), gradients):
            parameter.grad = None if gradient is None else gradient.to(parameter.dtype)
    for updated in (model, stock[0][0], oracle[0][0]):
        torch.optim.SGD(updated.parameters(), lr=.01, foreach=False).step()
    step_exact = {name: _exact(parameter, step_controls[name]) for name, parameter in model.named_parameters()}
    state = gather_dsa_tp_state_dict(model)
    sgd_stock = _compare(state, stock[0][0].state_dict())
    sgd_oracle = _compare(state, oracle[0][0].state_dict())
    identity = all(value is original_parameters[name] for name, value in model.named_parameters())
    owner_epochs = [item['epoch'] for item in records['owner_returns']]
    expected_forward_order = [0, 1, 1, 0] if replay == 'checkpoint' else [0, 1]
    protocol = (len(records['main_backwards']) == 2 and len(owner_epochs) == 2
                and [item['microbatch'] for item in records['forwards']] == expected_forward_order
                and owner_epochs == [records['owner_epoch_start']+1, records['owner_epoch_start']+2]
                and all((item['index_receipt'] is not None) == probe.index_active for item in records['owner_returns']))
    contract = (identity and selected_exact and protocol and all(value['passed'] for value in exact.values())
                and all(value['passed'] for value in capture_exact.values())
                and all(value['passed'] for value in reduction_controls.values())
                and all(value['passed'] for value in step_exact.values()))
    all_batches_passed = all(item['stock']['passed'] and item['fp32']['passed'] for item in per_batch)
    accepted = (contract and stock_metrics['passed'] and oracle_metrics['passed'] and sgd_stock['passed']
                and sgd_oracle['passed'] and all_batches_passed)
    for item in records['forwards']:
        item.pop('indices')
        item.pop('raw_kl_loss')
    scenario['cases'].append({'passed': accepted, 'accumulation_contract_passed': contract,
                              'mode': mode, 'replay': replay, 'microbatch_count': 2, 'parameter_identity': identity,
                              'fixture_seed': fixture_seed, 'group_private_partials': group_private_partials,
                              'selected_sets_exact': selected_exact, 'protocol_passed': protocol,
                              'isolated_native_bit_exact': exact, 'accumulated_parameters_stock': stock_metrics,
                              'parameter_vjps_captured_before_leaf_cast': capture_parameter_vjps,
                              'isolated_captured_vjp_bit_exact': capture_exact,
                              'captured_cp_reduction_control': reduction_controls,
                              'same_source_cp_reduction_control': same_source_controls,
                              'single_sgd_control': step_exact,
                              'cp_parameter_reduction_count': 1,
                              'parameter_reduction_order': parameter_reduction_order,
                              'projection_reduction_order': projection_reduction_order,
                              'backward_compute_groups': probe.backward_backend.schedule.compute_groups,
                              'accumulated_parameters_fp32': oracle_metrics, 'per_microbatch': per_batch,
                              'sgd_stock': sgd_stock, 'sgd_fp32': sgd_oracle, 'native_records': records})
    if repeat_isolated:
        scenario['cases'][-1]['isolated_native_repeats'] = repeat_metrics
    if not accepted and archive_failures:
        name = f"rank{dist.get_rank()}-{scenario['topology']}-{scenario['pattern']}-{mode}-{replay}.pt"
        torch.save({'parameter_gradients': {name: None if value is None else value.cpu()
                                           for name, value in zip(names, reduced[1:])},
                    'stock_parameters': {name: None if value is None else value.cpu()
                                         for name, value in zip(names, stock_parameters)},
                    'fp32_parameters': dict(zip(names, oracle_parameters)), 'hidden': hidden_cpu,
                    'cotangents': cotangents_cpu, 'initial_state': template.state_dict()}, output_dir/name)
    model.zero_grad(set_to_none=True)
    return {'passed': accepted, 'accumulation_contract_passed': contract}


def main() -> None:
    """Keep actual accumulation contract and full stock/FP32 model acceptance separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, help='model seed; projections/hidden/cotangent use seed+1/+2/+3')
    parser.add_argument('--metrics-only', action='store_true')
    parser.add_argument('--fp32-parameter-gradients', action='store_true')
    parser.add_argument('--group-private-partials', action='store_true')
    parser.add_argument('--parameter-reduction-order', choices=('collective', 'group_rank'), default='collective')
    parser.add_argument('--projection-reduction-order', choices=('collective', 'group_rank'), default='collective')
    parser.add_argument('--main-compute-groups', type=int, choices=range(1, 20), default=7)
    parser.add_argument('--backward-compute-groups', type=int, choices=range(1, 20))
    parser.add_argument('--topology', choices=('cp4_tp1', 'cp2_tp2', 'cp1_tp4', 'cp8_tp1', 'cp4_tp2', 'cp2_tp4'))
    parser.add_argument('--replay', choices=('ordered', 'reverse', 'checkpoint'))
    parser.add_argument('--pattern', choices=('strided', 'empty'))
    args = parser.parse_args()
    if args.seed is not None and not 0 <= args.seed < 2**63-3:
        parser.error('seed must be nonnegative and leave room for the three derived fixture seeds')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'scenarios': [], 'performance_acceptance': False, 'full_lm_acceptance': False,
              'scope': 'actual 11-parameter HP model; two packed microbatches; delayed BF16 leaf accumulation',
              'fixture_seed': args.seed, 'group_private_partials': args.group_private_partials,
              'fixture_seeds': ({'model': 20261007, 'projections': 733, 'hidden': 20261008, 'cotangent': 20261011}
                                if args.seed is None else
                                {'model': args.seed, 'projections': args.seed+1,
                                 'hidden': args.seed+2, 'cotangent': args.seed+3}),
              'microbatch_objective_scale': .5, 'auxiliary_upstream_per_microbatch': -1.75/2,
              'metrics_only': args.metrics_only, 'full_model_acceptance': False,
              'normalized_model_acceptance': False, 'profile_consumed_for_acceptance': False,
              'parameter_reduction_order': args.parameter_reduction_order,
              'projection_reduction_order': args.projection_reduction_order,
              'main_compute_groups': args.main_compute_groups,
              'backward_compute_groups': (args.main_compute_groups if args.backward_compute_groups is None else
                                          args.backward_compute_groups),
              'parameter_vjps_captured_before_leaf_cast': args.fp32_parameter_gradients}
    try:
        torch.set_num_threads(1)
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        world = dist.get_world_size()
        if world not in (4, 8):
            raise ValueError('model accumulation validator requires four or eight root members')
        rank = dist.get_rank()
        topologies = _topologies(world, rank)
        if args.topology is not None:
            topologies = [item for item in topologies if item[0] == args.topology]
            if not topologies:
                raise ValueError('requested accumulation topology does not match the root size')
        report['root_size'] = world
        report['scope'] = ('actual 11-parameter HP model; two packed microbatches; delayed '
                           +('FP32 captured VJP' if args.fp32_parameter_gradients else 'BF16 leaf')+' accumulation')
        matrix = [(mode, replay) for mode in (('joint',) if args.smoke else ('joint', 'freeze', 'zero', 'eval'))
                  for replay in ((args.replay,) if args.replay else ('ordered', 'reverse', 'checkpoint'))]
        for topology, groups in topologies:
            for pattern in ((args.pattern,) if args.pattern else ('strided', 'empty')):
                runner = partial(_case, capture_parameter_vjps=args.fp32_parameter_gradients,
                                 archive_failures=not args.metrics_only,
                                 parameter_reduction_order=args.parameter_reduction_order,
                                 projection_reduction_order=args.projection_reduction_order,
                                 backward_compute_groups=args.backward_compute_groups, fixture_seed=args.seed,
                                 group_private_partials=args.group_private_partials)
                _run(report, topology, groups, pattern, False, args.output, case_runner=runner, case_matrix=matrix,
                     main_compute_groups=args.main_compute_groups, template_seed=args.seed)
        cases = [case for scenario in report['scenarios'] for case in scenario['cases']]
        accepted = all(case['passed'] for case in cases)
        report.update(status=('diagnostic_finished' if args.metrics_only else
                              'passed' if accepted else 'numerical_mismatch'),
                      full_model_acceptance=accepted and not args.metrics_only,
                      accumulation_contract_passed=all(case['accumulation_contract_passed'] for case in cases),
                      loaded_library_sha256=_loaded_library_identity())
        if not accepted and not args.metrics_only:
            raise RuntimeError('actual model accumulated numerical acceptance failed; inspect per-field evidence')
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if report['status'] == 'running':
            report['status'] = 'error'
        raise
    finally:
        (args.output/f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2)+'\n')
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
