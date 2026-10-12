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
"""Fresh-seed native CP/TP global model evaluation with a validated independent profile."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_validate import (
    _case,
    _run,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_reference_calibrate import (
    _environment,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_global_fields import (
    assemble_global_model_fields,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    policy_sha256,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_evaluate import (
    collect_global_model_diagnostics,
    evaluate_coupled_global_model_reference,
    verify_candidate_reference_environment,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_precision import (
    coupled_model_reference_policy,
    validate_coupled_model_reference_profile,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference_precision import (
    gradient_error_metrics,
)


def _replica_field_diagnostics(peers, field):
    if field not in peers[0]['fields']:
        return None
    reference = peers[0]['fields'][field]
    result = []
    for peer in peers:
        value = peer['fields'][field]
        record = {'cp_rank': peer['cp_rank'], 'tp_rank': peer['tp_rank']}
        if value is None or reference is None:
            record['availability_equal'] = value is reference
        elif value.shape != reference.shape or value.dtype != reference.dtype:
            record.update(shape=list(value.shape), dtype=str(value.dtype), compatible=False)
        else:
            record.update(equal=bool(torch.equal(value, reference)),
                          differing_elements=int(torch.count_nonzero(value != reference)),
                          metrics=gradient_error_metrics(value.float(), reference.float()))
        result.append(record)
    return result


def _evaluate_global(profile, seed, libraries, metrics_only, output_dir, model, fields, oracle, template, groups,
                     checkpoint_state, ids, mode, loss_receipt):
    _members, cp_group, tp_group = groups
    named = {'output': fields['output'], 'hidden': fields['hidden'], 'loss': fields['loss'],
             **{'gradient/'+name: fields[name] for name, _parameter in model.named_parameters()},
             **{'sgd/'+name: parameter for name, parameter in model.named_parameters()}}
    record = {'cp_rank': dist.get_rank(cp_group), 'tp_rank': dist.get_rank(tp_group), 'owner_ids': ids,
              'loss_receipt': loss_receipt,
              'fields': {name: None if value is None else value.detach().cpu() for name, value in named.items()}}
    peers = [None]*dist.get_world_size()
    dist.all_gather_object(peers, record)
    try:
        actual = assemble_global_model_fields(peers, heads=template.num_heads,
            cp_size=dist.get_world_size(cp_group), tp_size=dist.get_world_size(tp_group))
    except ValueError as error:
        if dist.get_rank() == 0:
            replicas = [{'cp_rank': peer['cp_rank'], 'tp_rank': peer['tp_rank'], 'owner_ids': peer['owner_ids'],
                         'loss_receipt': peer['loss_receipt'],
                         'loss': None if peer['fields']['loss'] is None else float(peer['fields']['loss'])}
                        for peer in peers]
            topology = f'cp{dist.get_world_size(cp_group)}_tp{dist.get_world_size(tp_group)}'
            name = f'replica_loss_{topology}_{seed}_{mode}.json'
            (output_dir/name).write_text(json.dumps(replicas, indent=2)+'\n')
            field = str(error).rsplit(': ', 1)[-1]
            diagnostics = {'error': str(error), 'field': field, 'reference_coordinate': [0, 0],
                           'replicas': _replica_field_diagnostics(peers, field)}
            name = f'replica_fields_{topology}_{seed}_{mode}.json'
            (output_dir/name).write_text(json.dumps(diagnostics, indent=2)+'\n')
        raise
    for name, updated in checkpoint_state.items():
        if not torch.equal(actual['sgd/'+name], updated.detach().cpu()):
            raise ValueError('global reconstruction disagrees with the independent TP checkpoint gather')
        actual['sgd/'+name] = updated
    reference, output, gradients, loss = oracle
    expected = {'output': output, 'hidden': gradients[0], 'loss': loss,
                **{'gradient/'+name: value for (name, _parameter), value in
                   zip(reference.named_parameters(), gradients[1:])},
                **{'sgd/'+name: parameter for name, parameter in reference.named_parameters()}}
    environment = _environment(profile['policy'])
    verify_candidate_reference_environment(profile['environment'], environment, candidate_libraries=libraries)
    if metrics_only:
        result = collect_global_model_diagnostics(actual, expected, dict(template.named_parameters()),
            heads=template.num_heads, learning_rate=profile['policy']['learning_rate'])
        result['candidate_seed'] = seed
    else:
        result = evaluate_coupled_global_model_reference(profile, actual, expected, dict(template.named_parameters()),
            heads=template.num_heads, mode=mode, candidate_seed=seed,
            expected_policy_sha256=policy_sha256(profile['policy']),
            expected_environment=profile['environment'])
    result.update(global_reconstruction_verified=True, cp_gradient_reduction_count=1,
                  exact_cp_and_tp_replica_checks=True, independent_tp_checkpoint_gather_verified=True)
    return result


def _topologies(world, rank):
    result = []
    for tp_size in (1, 2, 4):
        cp_size = world//tp_size
        cp_groups = []
        for tp in range(tp_size):
            members = tuple(range(tp*cp_size, (tp+1)*cp_size))
            group = dist.group.WORLD if tp_size == 1 else dist.new_group(list(members))
            cp_groups.append((members, group))
        tp_groups = []
        for cp in range(cp_size):
            members = tuple(range(cp, world, cp_size))
            group = dist.group.WORLD if cp_size == 1 else dist.new_group(list(members))
            tp_groups.append((members, group))
        cp_members, cp_group = next(item for item in cp_groups if rank in item[0])
        tp_group = next(group for members, group in tp_groups if rank in members)
        result.append((f'cp{cp_size}_tp{tp_size}', (cp_members, cp_group, tp_group)))
    return result


def main() -> None:
    """Keep historical pointwise gates alongside a separate complete-global normalized evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--payload-identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--metrics-only', action='store_true',
                        help='collect complete-global numerical diagnostics without consuming or passing a profile')
    parser.add_argument('--fp32-output-parameter-gradients', action='store_true')
    parser.add_argument('--fp32-qkv-parameter-gradients', action='store_true')
    parser.add_argument('--fp32-kv-parameter-gradients', action='store_true')
    parser.add_argument('--fp32-query-parameter-gradients', action='store_true')
    parser.add_argument('--fp32-index-kl', action='store_true')
    parser.add_argument('--fp32-query-vjp', action='store_true')
    parser.add_argument('--fp32-compressed-vjp', action='store_true')
    parser.add_argument('--fp32-native-query-vjp', action='store_true')
    parser.add_argument('--fp32-rope-vjp', action='store_true')
    parser.add_argument('--group-private-partials', action='store_true')
    parser.add_argument('--parameter-reduction-order', choices=('collective', 'group_rank'), default='collective')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    policy = coupled_model_reference_policy()
    payload = json.loads(args.payload_identity.read_text())['payload_sha256']
    libraries = {}
    for name, digest in payload.items():
        filename = Path(name).name
        if filename.endswith('.so'):
            if filename in libraries and libraries[filename] != digest:
                raise ValueError('frozen native payload has conflicting library identities')
            libraries[filename] = digest
    if profile['policy'] != policy:
        raise ValueError('native global evaluation requires the current frozen reference policy')
    torch.set_num_threads(policy['cpu_reference_threads'])
    if not args.metrics_only:
        validate_coupled_model_reference_profile(profile, expected_policy_sha256=policy_sha256(policy),
                                                 expected_environment=profile['environment'])
    elif profile.get('status') not in ('validated', 'held_out_failed') or not profile.get('environment'):
        raise ValueError('metrics-only diagnostics require completed reference sampling and an environment identity')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'scenarios': [], 'policy_sha256': policy_sha256(policy),
              'performance_acceptance': False, 'standard_checkpoint_acceptance': False, 'full_lm_acceptance': False,
              'scope': 'complete-global H64/H128 Hi64 hidden32 Q-LoRA64 V16 packed13 CPxTP BF16 F+B+SGD',
              'historical_gates_preserved': True, 'candidate_measurements_used_for_fitting': False,
              'metrics_only': args.metrics_only, 'profile_consumed_for_acceptance': not args.metrics_only}
    report['output_parameter_vjps_captured_before_leaf_cast'] = args.fp32_output_parameter_gradients
    report['qkv_parameter_vjps_captured_before_leaf_cast'] = args.fp32_qkv_parameter_gradients
    report['kv_parameter_vjps_captured_before_leaf_cast'] = args.fp32_kv_parameter_gradients
    report['query_parameter_vjps_captured_before_leaf_cast'] = args.fp32_query_parameter_gradients
    report['fp32_index_kl'] = args.fp32_index_kl
    report['fp32_query_vjp'] = args.fp32_query_vjp
    report['fp32_compressed_vjp'] = args.fp32_compressed_vjp
    report['fp32_native_query_vjp'] = args.fp32_native_query_vjp
    report['fp32_rope_vjp'] = args.fp32_rope_vjp
    report['group_private_partials'] = args.group_private_partials
    report['loss_reduction_order'] = 'group_rank'
    report['parameter_reduction_order'] = args.parameter_reduction_order
    try:
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('hccl', timeout=timedelta(minutes=10))
        world = dist.get_world_size()
        if world not in (4, 8):
            raise ValueError('native global model validator requires four or eight root members')
        rank = dist.get_rank()
        topologies = _topologies(world, rank)
        report.update(root_size=world, declared_topologies=[name for name, _groups in topologies],
                      unsupported_topologies=({'cp1_tp8': 'H128/TP8 has local H16; native supports local H32/H64'}
                                              if world == 8 else {}))
        seeds = policy['reserved_candidate_seeds'][:1] if args.smoke else policy['reserved_candidate_seeds']
        cases = [('joint', 'plain')] if args.smoke else [(mode, 'plain') for mode in policy['modes']]
        for seed in seeds:
            for topology, groups in topologies:
                for pattern in ('strided', 'empty'):
                    runner = partial(_case, fixture_seed=seed, archive_failures=False,
                                     capture_output_vjps=args.fp32_output_parameter_gradients,
                                     capture_qkv_vjps=args.fp32_qkv_parameter_gradients,
                                     capture_kv_vjps=args.fp32_kv_parameter_gradients,
                                     capture_query_vjps=args.fp32_query_parameter_gradients,
                                     fp32_index_kl=args.fp32_index_kl,
                                     fp32_query_vjp=args.fp32_query_vjp,
                                     fp32_compressed_vjp=args.fp32_compressed_vjp,
                                     fp32_native_query_vjp=args.fp32_native_query_vjp,
                                     fp32_rope_vjp=args.fp32_rope_vjp,
                                     group_private_partials=args.group_private_partials,
                                     parameter_reduction_order=args.parameter_reduction_order,
                                     global_evaluator=partial(_evaluate_global, profile, seed, libraries,
                                                              args.metrics_only, args.output))
                    _run(report, topology, groups, pattern, args.smoke, args.output,
                         case_runner=runner, case_matrix=cases, template_seed=seed)
        accepted = all(case['global_reference']['passed'] and case['selected_sets_equal']
                       and case['original_parameters_preserved']
                       for scenario in report['scenarios'] for case in scenario['cases'])
        report.update(status=('diagnostic_finished' if args.metrics_only else
                              'passed' if accepted else 'numerical_mismatch'),
                      environment=_environment(policy),
                      reference_profile_sha256=profile['reference_cases_sha256'])
        if not accepted and not args.metrics_only:
            raise RuntimeError('native complete-global normalized model acceptance failed; limits unchanged')
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
