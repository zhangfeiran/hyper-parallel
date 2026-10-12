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
"""Standard-dimensional CP/TP F+B+SGD diagnostics without consuming small-model budgets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import timedelta
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore import _loader
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_global_validate import (
    _topologies,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_validate import (
    _AUXILIARY,
    _case,
    _run,
)
from hyper_parallel.core.multicore.examples.mega_dsa_fused_model_cp_validate import (
    _configure,
    _IndependentStockReference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_local_cp_backward_validate import (
    _loaded_library_identity,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
)
from hyper_parallel.core.multicore.examples.mega_dsa_selected_model_reference import (
    selected_unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_standard_model_common import (
    standard_model_fixture,
    standard_model_inputs,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.model_boundary import (
    CannDsaReferenceAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_global_fields import (
    assemble_global_model_fields,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_evaluate import (
    collect_global_model_diagnostics,
)


def _selected_reference(template,mode,hidden,embeddings,cotangent,meta,device,selection, *, output_dir):
    stock = CannDsaReferenceAttention(module=copy.deepcopy(template).to(device))
    oracle = copy.deepcopy(template).float()
    for target in (stock,oracle):
        _configure(target,mode)
    stock_hidden = hidden.to(device).requires_grad_()
    backend = _IndependentStockReference(CannDsaLayout(meta,device),attention_scale=stock.scaling,indices=selection)
    stock_output = stock(stock_hidden,position_embeddings=tuple(value.to(device) for value in embeddings),
                         dsa_reference=backend)[0]
    stock_gradients = torch.autograd.grad((stock_output.float()*cotangent.to(device)).sum(),
                                         (stock_hidden,*stock.parameters()),allow_unused=True)
    oracle_hidden = hidden.float().requires_grad_()
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        output,loss = selected_unabsorbed_model_reference(oracle,oracle_hidden,
                        tuple(value.float() for value in embeddings),meta,selection)
        objective = (output*cotangent).sum()
        if loss is not None:
            objective = objective+loss*_AUXILIARY
        gradients = torch.autograd.grad(objective,(oracle_hidden,*oracle.parameters()),allow_unused=True)
    stock_selection = backend.last_selection.to_global_indices().cpu()
    torch.save({'actual_indices':selection,'stock_indices':stock_selection,
                'global_cu_seqlens':meta.global_cu_seqlens,'oracle_uses_actual_fused_selection':True},
               output_dir/f'selection_rank{dist.get_rank()}.pt')
    return ((stock,stock_output,stock_gradients,backend.last_loss,stock_selection),
            (oracle,output,gradients,loss))


def _evaluate(model, fields, oracle, template, groups, checkpoint_state, ids, mode, loss_receipt):
    del mode, loss_receipt
    _members, cp_group, tp_group = groups
    named = {'output':fields['output'], 'hidden':fields['hidden'], 'loss':fields['loss'],
             **{'gradient/'+name:fields[name] for name, _parameter in model.named_parameters()},
             **{'sgd/'+name:parameter for name, parameter in model.named_parameters()}}
    record = {'cp_rank':dist.get_rank(cp_group), 'tp_rank':dist.get_rank(tp_group), 'owner_ids':ids,
              'fields':{name:None if value is None else value.detach().cpu() for name, value in named.items()}}
    peers = [None]*dist.get_world_size()
    dist.all_gather_object(peers, record)
    reference, output, gradients, loss = oracle
    expected = {'output':output, 'hidden':gradients[0], 'loss':loss,
                **{'gradient/'+name:value for (name, _parameter), value in
                   zip(reference.named_parameters(), gradients[1:])},
                **{'sgd/'+name:parameter for name, parameter in reference.named_parameters()}}
    shapes = {'output':list(output.shape), 'hidden':list(output.shape), 'loss':[],
              **{'gradient/'+name:list(parameter.shape) for name, parameter in template.named_parameters()},
              **{'sgd/'+name:list(parameter.shape) for name, parameter in template.named_parameters()}}
    actual = assemble_global_model_fields(peers, heads=template.num_heads, cp_size=dist.get_world_size(cp_group),
                                         tp_size=dist.get_world_size(tp_group), declared_shapes=shapes)
    for name, updated in checkpoint_state.items():
        if not torch.equal(actual['sgd/'+name], updated.detach().cpu()):
            raise ValueError('standard global reconstruction disagrees with independent TP checkpoint gather')
        actual['sgd/'+name] = updated
    result = collect_global_model_diagnostics(actual, expected, dict(template.named_parameters()),
        heads=template.num_heads, learning_rate=.01, declared_shapes=shapes)
    if not all(value.get('single_axpy_verified', True) for value in result['fields'].values()):
        raise ValueError('standard training violated a complete parameter single-AXPY update')
    result.update(global_reconstruction_verified=True, exact_cp_and_tp_replica_checks=True,
                  independent_tp_checkpoint_gather_verified=True, cp_gradient_reduction_count=1,
                  shapes=shapes)
    return result


def main() -> None:
    """Run explicit standard geometry and packed lengths, recording numerical diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp-size', type=int, choices=(2, 4), default=2)
    parser.add_argument('--lengths', type=int, nargs='+', default=[3, 10])
    parser.add_argument('--cpu-reference-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261258)
    parser.add_argument('--pattern', choices=('strided', 'empty'), default='strided')
    parser.add_argument('--replay', choices=('plain', 'retained', 'checkpoint'), default='plain')
    parser.add_argument('--selected-oracle', action='store_true',
                        help='reconstruct actual fused selection and use the bounded unabsorbed CPU oracle')
    parser.add_argument('--mode', choices=('joint', 'eval'), default='joint')
    args = parser.parse_args()
    if args.cpu_reference_threads < 1:
        raise ValueError('CPU reference thread count must be positive')
    if any(length <= 0 or (length > 2048 and not args.selected_oracle) for length in args.lengths):
        raise ValueError('standard dense short-context oracle requires positive packed lengths no greater than K=2048')
    config = json.loads(args.config.read_text())
    payload = args.payload_root.resolve()
    _loader._component_root = lambda:payload/'lib'
    torch.set_num_threads(args.cpu_reference_threads)
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(minutes=15))
    args.output.mkdir(parents=True, exist_ok=True)
    world, rank = dist.get_world_size(), dist.get_rank()
    if world not in (4, 8) or world % args.tp_size:
        raise ValueError('standard CP TP validator requires a complete four/eight-member root')
    topology = f'cp{world//args.tp_size}_tp{args.tp_size}'
    selected = next((name, groups) for name, groups in _topologies(world, rank) if name == topology)
    report = {'status':'running', 'scenarios':[], 'full_model_acceptance':False, 'performance_acceptance':False,
              'pretrained_checkpoint_loaded':False, 'full_lm_acceptance':False,
              'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(), 'source_config':config,
              'payload_root':str(payload), 'lengths':args.lengths, 'seed':args.seed,
              'geometry':'H128 Hi64 hidden7168 Q-LoRA1536 V128 C512 Dr64 Di128 K2048',
              'precision_path':'BF16 source primal; FP32 index KL/Q/norm/C VJPs; original BF16 rotary VJP',
              'reference_policy_consumed':False, 'selected_oracle':args.selected_oracle,
              'cpu_reference_threads':args.cpu_reference_threads}
    factory = partial(standard_model_fixture, config)
    runner = partial(_case, fixture_seed=args.seed, archive_failures=False, global_evaluator=_evaluate,
                     capture_output_vjps=True, capture_qkv_vjps=True, capture_kv_vjps=True, capture_query_vjps=True,
                     group_private_partials=True, parameter_reduction_order='group_rank', fp32_index_kl=True,
                     fp32_query_vjp=True, fp32_compressed_vjp=True, fp32_native_query_vjp=True,
                     model_input_factory=partial(standard_model_inputs, config=config),
                     selected_reference_factory=(partial(_selected_reference,output_dir=args.output)
                                                 if args.selected_oracle else None))
    try:
        _run(report, selected[0], selected[1], args.pattern, True, args.output,
             case_runner=runner, case_matrix=[(args.mode,args.replay)], template_seed=args.seed,
             template_factory=factory, lengths=tuple(args.lengths))
        report.update(status='diagnostic_finished', loaded_library_sha256=_loaded_library_identity(),
                      hyper_parallel_import_path=str(Path(_loader.__file__).resolve()),
                      torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        (args.output/f'rank{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
