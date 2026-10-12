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
"""Real head-sharded BF16 CP/TP attention against stock and independent FP32 model gradients."""

from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from contextlib import nullcontext
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
from hyper_parallel.core.multicore.examples.mega_dsa_fused_cp_validate import _metadata
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import (
    _complete_indices,
    _configure,
    _IndependentStockReference,
    _metric,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.examples.mega_dsa_validation_control import (
    collective_validation_check,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.fp32_index_kl import (
    validate_fp32_index_kl_trace,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import (
    FusedDsaCpForwardProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedDsaCpTpForwardProbe,
)
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
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
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
from hyper_parallel.core.multicore.modules.mega_dsa.tp_tile_sum import (
    TpTileSumProbe,
    TpTileSumSpec,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from hyper_parallel.core.multicore.shmem.consumer import SharedShmemRoot

_AUXILIARY = -1.75


def _snapshot(gradients):
    return tuple(None if value is None else value.detach().cpu().clone() for value in gradients)


def _reduce_parameters(model, gradients, group, *, captured_gradients=None, reduction_order='collective'):
    if reduction_order not in ('collective', 'group_rank'):
        raise ValueError('parameter reduction order must be collective or group_rank')
    capture = getattr(model, 'parameter_vjp_capture', None)
    captured = ({} if capture is None else capture.take()) if captured_gradients is None else captured_gradients
    if captured_gradients is not None and (capture is None or set(captured) != set(capture.parameters)):
        raise ValueError('supplied parameter VJPs must match the model capture declarations')
    presence = tuple(value is not None for value in gradients[1:])
    peers = [None]*dist.get_world_size(group)
    dist.all_gather_object(peers, presence, group=group)
    reduced = []
    for field, ((name, parameter), gradient) in enumerate(zip(model.named_parameters(), gradients[1:])):
        if name in captured:
            value = captured[name]
            if (value is None) != (gradient is None):
                raise RuntimeError('captured parameter VJP availability differs from its original leaf')
            gradient = value
        if not any(value[field] for value in peers):
            reduced.append(None)
            continue
        value = (torch.zeros_like(parameter, dtype=torch.float32) if gradient is None else
                 gradient.detach().contiguous() if name in captured else gradient.detach().float().contiguous())
        if reduction_order == 'group_rank':
            value = ordered_parameter_sum(value, group)
        else:
            dist.all_reduce(value, group=group)
        reduced.append(value.to(parameter.dtype))
    return (gradients[0], *reduced)


def _expected_local(model, output, gradients, loss, ids):
    layout = model.dsa_tp_projection_layout
    axes = {name: axis for name, axis, _shape in layout.shards}
    fields = {'output': output[:, ids], 'hidden': gradients[0][:, ids], 'loss': loss}
    for (name, _parameter), value in zip(model.named_parameters(), gradients[1:]):
        if value is not None and name in axes:
            value = value.chunk(len(layout.members), dim=axes[name])[layout.rank]
        fields[name] = value
    return fields


def _compare(fields, expected):
    metrics = {name: _metric(value, expected[name]) for name, value in fields.items()}
    return {'passed': all(value['passed'] for value in metrics.values()), 'fields': metrics}


def _reference(template, mode, hidden, embeddings, cotangent, meta, device):
    stock = CannDsaReferenceAttention(module=copy.deepcopy(template).to(device))
    oracle = copy.deepcopy(template).float()
    for model in (stock, oracle):
        _configure(model, mode)
    stock_hidden = hidden.to(device).requires_grad_()
    stock_backend = _IndependentStockReference(CannDsaLayout(meta, device), attention_scale=stock.scaling,
                                               indices=_complete_indices(meta))
    stock_output = stock(stock_hidden, position_embeddings=tuple(value.to(device) for value in embeddings),
                         dsa_reference=stock_backend)[0]
    stock_gradients = torch.autograd.grad((stock_output.float()*cotangent.to(device)).sum(),
                                          (stock_hidden, *stock.parameters()), allow_unused=True)
    oracle_hidden = hidden.float().requires_grad_()
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        oracle_output, oracle_loss = unabsorbed_model_reference(
            oracle, oracle_hidden, tuple(value.float() for value in embeddings), meta, _complete_indices(meta))
        objective = (oracle_output*cotangent).sum()
        if oracle_loss is not None:
            objective = objective+oracle_loss*_AUXILIARY
        oracle_gradients = torch.autograd.grad(objective, (oracle_hidden, *oracle.parameters()), allow_unused=True)
    return ((stock, stock_output, stock_gradients, stock_backend.last_loss,
             stock_backend.last_selection.to_global_indices().cpu()),
            (oracle, oracle_output, oracle_gradients, oracle_loss))


def _execute(model, probe, meta, hidden, embeddings, cotangent, replay):
    forwards, receipts, returned = [], [], []
    retained = None
    original_forward = probe.tp_forward.forward
    original_backward = probe.backward_backend._backward_saved
    original_return = probe.shared_return.return_owner
    def _capture_forward(*args, **kwargs):
        value = original_forward(*args, **kwargs)
        trace = validate_selected_training_traces(tuple(item.cpu() for item in value.phase_traces),
                                                  probe.forward_backend.schedule, require_ld=False,
                                                  with_kl=probe.index_active and not probe.fp32_index_kl)
        forwards.append((value, trace, value.global_indices[:, 0].cpu().clone()))
        return value
    def _capture_fp32(*args, **kwargs):
        value = original_fp32(*args, **kwargs)
        rows = value.logits.shape[0]
        forwards[-1][1]['fp32_index_kl'] = {
            'scores': validate_fp32_index_kl_trace(value.score_trace.cpu(), query_tokens=rows, phase=0),
            'vjp': validate_fp32_index_kl_trace(value.vjp.trace.cpu(), query_tokens=rows, phase=1)}
        return value
    def _capture_backward(*args, **kwargs):
        value = original_backward(*args, **kwargs)
        receipts.append(probe.backward_backend.validate_trace(value, tuple(item.cpu() for item in value.phase_traces),
                                                               value.transport_trace.cpu()))
        return value
    def _capture_return(*args, **kwargs):
        value = original_return(*args, **kwargs)
        index_receipt = None
        if value.index_cp is not None:
            index_receipt = probe.shared_return.index_return.validate_trace(value.index_cp,
                                                                            value.index_cp.transport_trace.cpu())
        transport = probe.shared_return.tp_return.transport
        start = transport.spec.control_offset
        controls = transport.arena[start:start+transport.spec.slots*640].view(torch.int64).reshape(-1, 80).cpu()
        for control in controls:
            sequence = int(control[0])
            if sequence//2**32 != transport.last_epoch or any(int(control[(2+peer)*8]) != sequence
                                                             for peer in range(len(transport.members))):
                raise RuntimeError('model shared TP return lacks final reader ACK')
        returned.append({'cp_index_receipt': index_receipt, 'tp_ack': True, 'epoch': transport.last_epoch})
        return value
    def _forward(value):
        epoch = probe.tp_forward.li.last_epoch+1
        cp_invocation = probe.forward_backend.workspace.prepare(replace(meta, invocation=epoch))
        invocation = probe.prepare(cp_invocation, epoch=epoch, query_ids=meta.q_global_ids)
        return model(value, position_embeddings=embeddings, dsa_training=probe, dsa_invocation=invocation)[0]
    original_fp32 = None if probe.fp32_kl_backend is None else probe.fp32_kl_backend.gradients
    fp32_capture = (nullcontext() if original_fp32 is None else
                    patch.object(probe.fp32_kl_backend, 'gradients', side_effect=_capture_fp32))
    with fp32_capture, patch.object(probe.tp_forward, 'forward', side_effect=_capture_forward), \
            patch.object(probe.backward_backend, '_backward_saved', side_effect=_capture_backward), \
            patch.object(probe.shared_return, 'return_owner', side_effect=_capture_return):
        output = checkpoint(_forward, hidden, use_reentrant=False) if replay == 'checkpoint' else _forward(hidden)
        objective = (output.float()*cotangent).sum()
        capture = getattr(model, 'parameter_vjp_capture', None)
        if capture is not None:
            capture.begin_backward()
        gradients = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True,
                                        retain_graph=replay == 'retained')
        if capture is not None:
            capture.finish_backward()
        if replay == 'retained':
            saved = _snapshot(gradients)
            if capture is not None:
                capture.discard()
                capture.begin_backward()
            gradients = torch.autograd.grad(objective, (hidden, *model.parameters()), allow_unused=True)
            if capture is not None:
                capture.finish_backward()
            repeated = _snapshot(gradients)
            retained = {}
            for name, actual, expected in zip(('hidden', *dict(model.named_parameters())), repeated, saved):
                available = actual is not None and expected is not None
                retained[name] = {'passed': bool(torch.equal(actual, expected)) if available else actual is expected,
                                  'availability_only': not available,
                                  'metric': _metric(actual, expected)}
    return output, gradients, forwards, receipts, returned, retained


def _check_loss_replicas(replicas):
    if not replicas or any(value.dtype != torch.float32 or value.ndim != 0
                           or not torch.equal(value, replicas[0]) for value in replicas):
        raise ValueError('local TP loss replicas must be identical FP32 scalars before CP SUM')
    return {'local_tp_replicas_bit_exact': True}


def _reconstruct_selection(packets, meta):
    selection = torch.full((meta.global_valid_queries,2048),-1,dtype=torch.int32)
    seen = set()
    for ids, indices in packets:
        if indices.dtype != torch.int32 or indices.shape != (len(ids),2048):
            raise ValueError('selected reference requires complete caller-ordered global packed selection packets')
        if any(token in seen for token in ids):
            raise ValueError('selected reference query ownership overlaps')
        seen.update(ids)
        selection[list(ids)] = indices
    if seen != set(range(meta.global_valid_queries)):
        raise ValueError('selected reference query ownership is incomplete')
    return selection


def _case(scenario, template, groups, workspace, forward, owner, meta, mode, replay, output_dir, *,
          fixture_seed=None, global_evaluator=None, archive_failures=True, capture_output_vjps=False,
          capture_qkv_vjps=False, capture_kv_vjps=False, capture_query_vjps=False, group_private_partials=False,
          parameter_reduction_order='collective', fp32_index_kl=False, fp32_query_vjp=False,
          fp32_compressed_vjp=False, fp32_native_query_vjp=False, fp32_rope_vjp=False,
          model_input_factory=model_inputs, selected_reference_factory=None):
    _cp_members, cp_group, tp_group = groups
    source = copy.deepcopy(template).to(workspace.root.device)
    _configure(source, mode)
    original_parameters = dict(source.named_parameters())
    model = FusedDsaCpTpAttention(module=source, context={
        'dsa_tp_group': tp_group, 'dsa_capture_output_parameter_vjps': capture_output_vjps,
        'dsa_capture_qkv_parameter_vjps': capture_qkv_vjps, 'dsa_capture_kv_parameter_vjps': capture_kv_vjps,
        'dsa_capture_query_parameter_vjps': capture_query_vjps, 'dsa_fp32_index_kl': fp32_index_kl,
        'dsa_fp32_query_vjp': fp32_query_vjp, 'dsa_fp32_compressed_vjp': fp32_compressed_vjp,
        'dsa_fp32_native_query_vjp': fp32_native_query_vjp,
        'dsa_fp32_rope_vjp': fp32_rope_vjp})
    enabled = mode == 'joint'
    probe = FusedDsaLocalCpTpTrainingProbe(forward, owner,
                                          normalization=DsaLossNormalization(meta.global_valid_queries),
                                          loss_coeff=model.dsa_loss_coeff, mode='lm_plus_kl' if enabled else 'lm_only',
                                          group_private_partials=group_private_partials, fp32_index_kl=fp32_index_kl,
                                          fp32_compressed_vjp=fp32_compressed_vjp,
                                          fp32_native_query_vjp=fp32_native_query_vjp, fp32_rope_vjp=fp32_rope_vjp)
    lengths = tuple(end-start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
    global_meta = DsaBatchMeta.packed(lengths)
    hidden_cpu, embeddings_cpu = model_input_factory(global_meta, dtype=torch.bfloat16,
                                              **({} if fixture_seed is None else {'seed': fixture_seed+2}))
    ids = list(meta.kv_global_ids)
    generator = torch.Generator().manual_seed(20261011 if fixture_seed is None else fixture_seed+3)
    cotangent = torch.randn(hidden_cpu.shape, generator=generator)*.01
    set_aux_loss_scale(torch.tensor(_AUXILIARY, device=workspace.root.device))
    stock, oracle = (None,None) if selected_reference_factory is not None else _reference(
        template, mode, hidden_cpu, embeddings_cpu, cotangent, global_meta, workspace.root.device)
    hidden = hidden_cpu[:, ids].to(workspace.root.device).requires_grad_()
    embeddings = tuple(value[:, ids].to(workspace.root.device) for value in embeddings_cpu)
    output, gradients, forwards, receipts, returned, retained = _execute(
        model, probe, meta, hidden, embeddings, cotangent[:, ids].to(workspace.root.device), replay)
    if selected_reference_factory is not None:
        packets = [None]*dist.get_world_size(cp_group)
        dist.all_gather_object(packets,(ids,forwards[0][2]),group=cp_group)
        selection = collective_validation_check(partial(_reconstruct_selection,packets,global_meta))
        stock, oracle = collective_validation_check(partial(
            selected_reference_factory,template,mode,hidden_cpu,embeddings_cpu,cotangent,
            global_meta,workspace.root.device,selection))
    reduced = _reduce_parameters(model, gradients, cp_group, reduction_order=parameter_reduction_order)
    loss = None
    loss_receipt = None
    if enabled:
        raw_loss = (probe.fp32_kl_backend.last_result.vjp.loss.sum() if fp32_index_kl else forwards[0][0].kl_loss)
        loss = raw_loss.detach()*probe.loss_coeff*probe.normalization.local_sum_scale
        replicas = [None]*dist.get_world_size(tp_group)
        dist.all_gather_object(replicas, loss.cpu(), group=tp_group)
        proof = collective_validation_check(partial(_check_loss_replicas, replicas))
        # 不同 CP 组的 collective 加法顺序可能不同，冷诊断需要相同的 FP32 归约顺序。
        loss = ordered_parameter_sum(loss, cp_group)
        loss_receipt = {'local_tp_replicas': [float(value) for value in replicas], **proof,
                        'cp_reduction_order': 'group_rank', 'cp_reduction_count': 1}
    fields = {'output': output, 'hidden': reduced[0], 'loss': loss}
    fields.update({name: value for (name, _parameter), value in zip(model.named_parameters(), reduced[1:])})
    stock_metrics = _compare(fields, _expected_local(model, stock[1], stock[2], stock[3], ids))
    oracle_metrics = _compare(fields, _expected_local(model, oracle[1], oracle[2], oracle[3], ids))
    selections_equal = all(sorted(row[row>=0].tolist()) == sorted(target[target>=0].tolist())
                           for row, target in zip(forwards[0][2], stock[4][ids]))
    identity = all(parameter is original_parameters[name] for name, parameter in model.named_parameters())
    for parameter, value in zip(model.parameters(), reduced[1:]):
        parameter.grad = value
    for parameter, value in zip(stock[0].parameters(), stock[2][1:]):
        parameter.grad = value
    for parameter, value in zip(oracle[0].parameters(), oracle[2][1:]):
        parameter.grad = value
    for target in (model, stock[0], oracle[0]):
        torch.optim.SGD(target.parameters(), lr=.01, foreach=False).step()
    checkpoint_state = gather_dsa_tp_state_dict(model)
    checkpoint_stock = _compare(checkpoint_state, stock[0].state_dict())
    checkpoint_oracle = _compare(checkpoint_state, oracle[0].state_dict())
    global_reference = (None if global_evaluator is None else
                        global_evaluator(model, fields, oracle, template, groups, checkpoint_state, ids, mode,
                                         loss_receipt))
    expected_backwards = 2 if replay == 'retained' else 1
    if len(receipts) != expected_backwards or len(returned) != expected_backwards:
        raise RuntimeError('model training did not execute the declared native backward/owner return count')
    passed = (stock_metrics['passed'] and oracle_metrics['passed'] and checkpoint_stock['passed']
              and checkpoint_oracle['passed'] and selections_equal and identity
              and (retained is None or all(value['passed'] for value in retained.values())))
    case = {'passed': passed, 'mode': mode, 'replay': replay, 'owner_ids': ids,
            'fp32_index_kl': fp32_index_kl,
            'fp32_query_vjp': fp32_query_vjp,
            'fp32_compressed_vjp': fp32_compressed_vjp,
            'fp32_native_query_vjp': fp32_native_query_vjp, 'fp32_rope_vjp': fp32_rope_vjp,
            'group_private_partials': group_private_partials,
            'oracle_uses_actual_fused_selection': selected_reference_factory is not None,
            'stock_model': stock_metrics, 'independent_fp32_model': oracle_metrics,
            'sgd_checkpoint_stock': checkpoint_stock, 'sgd_checkpoint_fp32': checkpoint_oracle,
            'selected_sets_equal': selections_equal, 'original_parameters_preserved': identity,
            'cp_parameter_reduction': 'FP32 SUM once before SGD',
            'parameter_reduction_order': parameter_reduction_order,
            'main_compressed_vjp_precision': {
                'native_transport': ('FP32 dK+dV, CP owner SUM, TP SUM, FP32 KV norm/QKV VJP'
                                     if fp32_compressed_vjp else
                                     'FP32 dK+dV, CP owner SUM, TP SUM, then one activation dtype cast'),
                'stock_reference': 'stock dK and dV each in activation dtype, then activation dtype add',
                'fp32_oracle': 'unabsorbed FP32 model without intermediate activation dtype casts'},
            'indexer_vjp_precision': {
                'native_query_weight': ('FP32 index projection, raw VJP and coefficient/upstream' if fp32_index_kl else
                                        'activation dtype raw VJP, FP32 coefficient/upstream, '
                                        'then activation dtype cast'),
                'native_key': 'FP32 raw VJP, coefficient/upstream, CP owner SUM, TP SUM, then activation dtype cast',
                'teacher': ('FP32 dots on represented BF16 main states, head probability SUM, normalization'
                            if fp32_index_kl else 'native BF16 selected-KL teacher'),
                'selection': 'original BF16 fused LI states',
                'stock_reference': ('activation dtype raw VJP, activation dtype coefficient multiply '
                                    'and upstream multiply'),
                'fp32_oracle': 'FP32 detached teacher and index states without intermediate activation dtype casts'},
            'output_parameter_vjps_captured_before_leaf_cast': capture_output_vjps,
            'qkv_parameter_vjps_captured_before_leaf_cast': capture_qkv_vjps,
            'kv_parameter_vjps_captured_before_leaf_cast': capture_kv_vjps,
            'query_parameter_vjps_captured_before_leaf_cast': capture_query_vjps,
            'loss_reduction_receipt': loss_receipt,
            'forward_phase_receipts': [item[1] for item in forwards],
            'cp_backward_receipts': receipts, 'shared_owner_returns': returned, 'retained_exact_gradients': retained}
    scenario['cases'].append(case)
    if global_reference is not None:
        case['global_reference'] = global_reference
        case['fixture_seed'] = fixture_seed
    if not passed and archive_failures:
        name = f"{scenario['topology']}_{scenario['pattern']}_{mode}_{replay}_rank{dist.get_rank()}.pt"
        torch.save({'actual': {key: None if value is None else value.detach().cpu() for key, value in fields.items()},
                    'stock': _expected_local(model, stock[1], stock[2], stock[3], ids),
                    'oracle': _expected_local(model, oracle[1], oracle[2], oracle[3], ids),
                    'hidden': hidden_cpu, 'embeddings': embeddings_cpu, 'cotangent': cotangent,
                    'global_indices': forwards[0][2], 'initial_state': template.state_dict()}, output_dir/name)
    return {'passed': passed}


def _run(report, topology, groups, pattern, smoke, output_dir, *, case_runner=None, case_matrix=None,
         template_seed=None, main_compute_groups=7, template_factory=None, lengths=(3, 10)):
    schedule = MixedSfaSchedule(main_compute_groups)
    cp_members, cp_group, tp_group = groups
    tp_size = dist.get_world_size(tp_group)
    template = (tp_model_fixture(heads=64 if tp_size == 1 else 128, seed=template_seed)
                if template_factory is None else template_factory(seed=template_seed))
    root = SharedShmemRoot(torch.device('npu', int(os.environ['LOCAL_RANK'])), root_group=dist.group.WORLD)
    meta = _metadata(lengths, pattern, len(cp_members), dist.get_rank(cp_group))
    meta = replace(meta, cp_ranks=cp_members, root_pes=cp_members, q_global_ids=meta.kv_global_ids)
    capacity = max(1, max(meta.token_owners.count(peer) for peer in range(len(cp_members))))
    workspace = MegaDsaWorkspace(root, 'model-cp', DsaWorkspaceSpec(capacity, sum(lengths), capacity, len(cp_members)),
                                 cp_group=cp_group)
    slots = 2*schedule.compute_groups
    li = TpTileSumProbe(root, 'model-li', TpTileSumSpec(slots, 512), tp_group=tp_group)
    kl = TpTileSumProbe(root, 'model-kl', TpTileSumSpec(slots, 2048), tp_group=tp_group)
    owner = TpTileSumProbe(root, 'model-owner', TpTileSumSpec(14, 512), tp_group=tp_group)
    scenario = {'topology': topology, 'pattern': pattern, 'cases': [],
                'main_compute_groups': schedule.compute_groups, 'li_kl_transport_slots': slots}
    report['scenarios'].append(scenario)
    try:
        workspace.bind()
        for transport in (li, kl, owner):
            transport.bind()
        meta = replace(meta, heap_generation=root.generation)
        cp = FusedDsaCpForwardProbe(workspace, workspace.prepare(meta), heads=template.num_heads//tp_size,
                                   attention_scale=template.scaling, schedule=schedule)
        forward = FusedDsaCpTpForwardProbe(cp, li, kl)
        cases = [('joint', 'plain')] if smoke else [('joint', replay) for replay in ('plain', 'retained', 'checkpoint')]
        if not smoke:
            cases += [(mode, 'plain') for mode in ('freeze', 'zero', 'eval')]
        if case_matrix is not None:
            cases = case_matrix
        runner = _case if case_runner is None else case_runner
        for mode, replay in cases:
            collective_validation_check(partial(runner, scenario, template, groups, workspace, forward, owner,
                                                meta, mode, replay, output_dir))
        scenario['status'] = 'finished'
    except Exception as error:
        scenario.update(status='error', error=repr(error), traceback=traceback.format_exc())
        (output_dir/f"rank{dist.get_rank()}_before_cleanup.json").write_text(json.dumps(report, indent=2)+'\n')
        raise
    finally:
        for transport in (owner, kl, li):
            transport.close()
        workspace.close()
        root.close()


def main() -> None:
    """Record stock and FP32 acceptance separately, including every failed model field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'scenarios': [], 'performance_acceptance': False,
              'scope': 'real HP BF16 attention CPxTP F+B+SGD; H64/H128 Hi64 hidden32 Q-LoRA64 V16 packed13',
              'standard_checkpoint_acceptance': False, 'full_lm_acceptance': False}
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        if dist.get_world_size() != 4:
            raise ValueError('CP TP model validator requires four root members')
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
                _run(report, topology, groups, pattern, args.smoke, args.output)
        accepted = all(case['passed'] for scenario in report['scenarios'] for case in scenario['cases'])
        report.update(status='passed' if accepted else 'numerical_mismatch',
                      loaded_library_sha256=_loaded_library_identity())
        if not accepted:
            raise RuntimeError('CP TP model acceptance failed; inspect archived stock and FP32 fields')
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
