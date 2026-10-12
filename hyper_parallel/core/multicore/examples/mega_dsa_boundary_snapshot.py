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
"""CPU-only reconstruction of explicit CP/TP diagnostic snapshots; no process groups or native imports."""

from __future__ import annotations

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_local_training_oracle import (
    projection_vjp_metrics,
)

_HEAD_FIELDS = {'query', 'query_rope', 'index_query', 'merge_weight', 'sparse_output', 'sparse_cotangent'}


def _metric(actual, expected):
    if actual is None or expected is None:
        return {'passed': actual is None and expected is None, 'availability_only': True}
    return projection_vjp_metrics(actual, expected)


def merge_boundary_rows(peers: list[dict], section: str | None, field: str,
                        replicas: list[dict]) -> torch.Tensor:
    """Restore the packed13 diagnostic rows/head shards, recording duplicate replica differences."""
    sample = peers[0]['native'][section][field] if section else peers[0]['native'][field]
    tail = list(sample.shape[1:])
    head = field in _HEAD_FIELDS
    if head:
        tail[0] = 64 if field in ('index_query', 'merge_weight') else peers[0]['global_heads']
    output = torch.zeros((13, *tail), dtype=sample.dtype)
    covered = torch.zeros((13, tail[0]) if head else (13,), dtype=torch.bool)
    for peer in peers:
        value = peer['native'][section][field] if section else peer['native'][field]
        rows = list(peer['owner_ids'])
        if not rows:
            continue
        start = peer['tp_rank']*value.shape[1] if head else 0
        index = (rows, slice(start, start+value.shape[1])) if head else rows
        existing = covered[index]
        if existing.any():
            if not existing.all():
                raise ValueError('capture replicas only partially overlap; ownership is inconsistent')
            replicas.append({'field': f'{section}.{field}', 'exact': bool(torch.equal(output[index], value)),
                             'metric': _metric(output[index], value)})
        else:
            output[index] = value
            covered[index] = True
    if not covered.all():
        raise ValueError(f'captures do not cover every global row/head of {field}')
    return output


def merge_boundary_parameters(peers: list[dict], replicas: list[dict]) -> dict[str, torch.Tensor]:
    """Gather diagnostic weight VJPs once per TP shard and restore owner-local hidden rows."""
    axes = {name: axis for name, axis, _shape in peers[0]['shards']}
    result = {}
    for name, template in peers[0]['initial_state'].items():
        value = torch.zeros_like(template)
        covered = set()
        for peer in peers:
            gradient = peer['parameter_gradients'][name]
            key = peer['tp_rank'] if name in axes else 0
            index = [slice(None)]*template.ndim
            if name in axes:
                axis = axes[name]
                width = gradient.shape[axis]
                index[axis] = slice(key*width, (key+1)*width)
            index = tuple(index)
            if key in covered:
                replicas.append({'field': name, 'exact': bool(torch.equal(value[index], gradient)),
                                 'metric': _metric(value[index], gradient)})
            else:
                value[index] = gradient
                covered.add(key)
        expected = len(peers[0]['tp_members']) if name in axes else 1
        if len(covered) != expected:
            raise ValueError(f'parameter captures do not cover every shard of {name}')
        result[name] = value
    hidden = torch.zeros_like(peers[0]['hidden'])
    visited = set()
    for peer in peers:
        for offset, token in enumerate(peer['owner_ids']):
            if token not in visited:
                hidden[:, token] = peer['hidden_gradient'][:, offset]
                visited.add(token)
    if visited != set(range(13)):
        raise ValueError('hidden VJP captures do not cover every global token')
    return {'hidden': hidden, **result}


def boundary_metrics(actual: dict, expected: dict) -> dict:
    """Apply the existing pointwise and nonzero relative-L2 gate to every named diagnostic field."""
    fields = {name: _metric(value, expected[name]) for name, value in actual.items()}
    return {'passed': all(value['passed'] for value in fields.values()), 'fields': fields}
