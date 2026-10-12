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
"""Two-process Gloo real TP projections against independent unabsorbed full-model FP64 math."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp_tp import (
    FusedCpTpInvocation,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_tp_model_boundary import (
    FusedDsaCpTpAttention,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import (
    DsaLocalQueryLayout,
)
from hyper_parallel.core.multicore.modules.mega_dsa.local_tp_training import (
    FusedDsaLocalCpTpTrainingProbe,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.tp_projections import (
    gather_dsa_tp_state_dict,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
)


class _OracleTpTraining(FusedDsaLocalCpTpTrainingProbe):
    """Explicit CPU execution double; the candidate's actual model projections are unchanged."""

    def __init__(self, model: FusedDsaCpTpAttention, meta: DsaBatchMeta, *, auxiliary: bool) -> None:
        """Bind CP1 ownership with real Gloo TP score/teacher and shared-input VJPs."""
        workspace = SimpleNamespace(root=SimpleNamespace(generation=meta.heap_generation),
                                     closed=False, participates=True)
        self.forward_backend = SimpleNamespace(workspace=workspace, heads=model.local_heads, scale=model.scaling,
                                               layout=SimpleNamespace(batch_meta=meta, device=torch.device('cpu')))
        layout = model.dsa_tp_projection_layout
        self.tp_forward = SimpleNamespace(li=SimpleNamespace(group=layout.group, members=layout.members),
                                          index_heads=model.local_index_heads)
        self.mode = 'lm_plus_kl' if auxiliary else 'lm_only'
        self.loss_coeff = model.dsa_loss_coeff
        self.boundary = model.tp_collectives
        self.meta = meta
        self.normalization = DsaLossNormalization(meta.global_valid_queries)
        self.losses = []

    def invocation(self) -> FusedCpTpInvocation:
        """Build an explicit CPU-only cold declaration accepted by the real model adapter."""
        cp = DsaWorkspaceInvocation(self.meta, torch.arange(self.meta.global_valid_queries),
                                    self.forward_backend.workspace)
        return FusedCpTpInvocation(self.tp_forward, cp, DsaLocalQueryLayout(self.meta, 'cpu'), (),
                                   self.mode == 'lm_plus_kl')

    def forward(self, invocation: FusedCpTpInvocation, main_states: tuple[torch.Tensor, ...],
                index: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Use complete short causal histories and detached global TP teacher probabilities."""
        del invocation
        query, compressed, query_rope, key_rope = main_states
        compressed = self.boundary.copy_query_latent(compressed)
        key_rope = self.boundary.copy_query_latent(key_rope)
        total = self.meta.global_valid_queries
        positions = torch.arange(total)
        starts = torch.tensor([self.meta.global_cu_seqlens[self.meta.sequence_position(token)[0]]
                               for token in range(total)])
        valid = (positions[None] >= starts[:, None]) & (positions[None] <= positions[:, None])
        logits = (query@compressed.T+query_rope@key_rope.T)*self.forward_backend.scale
        probability = logits.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        output = probability@compressed
        if self.mode == 'lm_only':
            return output, output.new_zeros(())
        iq, ik, weight = index
        ik = self.boundary.copy_query_latent(ik)
        partial = ((iq@ik.T).relu()*weight[..., None]).sum(1)
        scores = self.boundary.sum_selected_scores(partial)
        teacher = self.boundary.teacher(logits.detach(), valid)
        loss = self.boundary.selected_kl(scores, teacher, valid, normalization=self.normalization,
                                          loss_coeff=self.loss_coeff)
        self.losses.append(loss.detach())
        return output, loss


def _selection(meta):
    width = max(end-start for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]))
    selected = torch.full((meta.global_valid_queries, width), -1, dtype=torch.int32)
    for token in range(meta.global_valid_queries):
        start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
        selected[token, :token-start+1] = torch.arange(start, token+1, dtype=torch.int32)
    return selected


def _metrics(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    delta = actual.detach()-expected.detach()
    return {'max_abs': float(delta.abs().max()) if delta.numel() else 0.,
            'relative_l2': float(delta.norm()/expected.detach().norm().clamp_min(1e-30))}


def _case(mode):
    meta = DsaBatchMeta.packed((3, 10))
    reference = tp_model_fixture(dtype=torch.float64, projection_bias=True)
    source = tp_model_fixture(dtype=torch.float64, projection_bias=True)
    original_parameters = dict(source.named_parameters())
    candidate = FusedDsaCpTpAttention(module=source, context={'dsa_tp_group': dist.group.WORLD})
    if mode == 'freeze':
        candidate.freeze_dsa = reference.freeze_dsa = True
    elif mode == 'zero':
        candidate.dsa_loss_coeff = reference.dsa_loss_coeff = 0.
    elif mode == 'eval':
        candidate.eval()
        reference.eval()
    auxiliary = mode == 'joint'
    probe = _OracleTpTraining(candidate, meta, auxiliary=auxiliary)
    invocation = probe.invocation()
    hidden, embeddings = model_inputs(meta, dtype=torch.float64)
    expected_hidden = hidden.clone().requires_grad_()
    actual_hidden = hidden.clone().requires_grad_()
    generator = torch.Generator().manual_seed(20261011)
    cotangent = torch.randn(hidden.shape, dtype=torch.float64, generator=generator)*.01
    set_aux_loss_scale(torch.tensor(-1.75, dtype=torch.float64))
    with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
               side_effect=_rotary_mul_oracle):
        expected, loss = unabsorbed_model_reference(reference, expected_hidden, embeddings, meta, _selection(meta))
        actual, _unused = candidate(actual_hidden, embeddings, dsa_training=probe, dsa_invocation=invocation)
        objective = (expected*cotangent).sum()
        if loss is not None:
            objective = objective+loss*-1.75
        objective.backward()
        (actual*cotangent).sum().backward()
    metrics = {'output': _metrics(actual, expected), 'hidden': _metrics(actual_hidden.grad, expected_hidden.grad)}
    layout = candidate.dsa_tp_projection_layout
    shards = {name: axis for name, axis, _shape in layout.shards}
    expected_parameters = dict(reference.named_parameters())
    updated = []
    for name, parameter in candidate.named_parameters():
        if parameter is not original_parameters[name]:
            raise RuntimeError('TP sharding replaced an original Parameter object')
        expected_gradient = expected_parameters[name].grad
        if name in shards and expected_gradient is not None:
            expected_gradient = expected_gradient.chunk(len(layout.members), dim=shards[name])[layout.rank]
        if expected_gradient is None:
            if parameter.grad is not None:
                raise RuntimeError('inactive source projection received an unexpected gradient')
        else:
            metrics[name] = _metrics(parameter.grad, expected_gradient)
            updated.append(name)
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    torch.optim.SGD(candidate.parameters(), lr=.01).step()
    global_state = gather_dsa_tp_state_dict(candidate)
    checkpoint = {name: _metrics(value, reference.state_dict()[name]) for name, value in global_state.items()}
    return {'passed': True, 'mode': mode, 'metrics': metrics, 'sgd_checkpoint': checkpoint,
            'updated_parameters': updated, 'original_parameters_preserved': True,
            'physical_weight_shards': True, 'native_backend_acceptance': False}


def main() -> None:
    """Validate real TP2 projection weights and full-model VJPs without running NPU kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'running', 'cases': [], 'scope': 'CP1 TP2 actual projections with explicit CPU math double',
              'native_cp_tp_model_acceptance': False, 'performance_acceptance': False}
    try:
        dist.init_process_group('gloo')
        if dist.get_world_size() != 2:
            raise ValueError('CPU TP model validator requires two processes')
        for mode in ('joint', 'freeze', 'zero', 'eval'):
            report['cases'].append(_case(mode))
        report['status'] = 'passed'
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
