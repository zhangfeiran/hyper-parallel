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
"""Cold reconstruction of complete model fields for CP/TP numerical validation."""

from __future__ import annotations

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    model_reference_shapes,
)

_HEAD_AXES = {'q_b_proj.weight': 0, 'kv_b_proj.weight': 0, 'o_proj.weight': 1,
              'indexer.wq_b.weight': 0, 'indexer.weights_proj.weight': 0}


def _same(left, right):
    if left is None or right is None:
        return left is right
    return left.dtype == right.dtype and bool(torch.equal(left, right))


def _declared_shapes(heads: int, shapes: dict[str, list[int]] | None) -> dict[str, list[int]]:
    original = model_reference_shapes(heads)
    if shapes is None:
        return original
    if (not isinstance(shapes, dict) or set(shapes) != set(original)
            or any(not isinstance(shape, list)
                   or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
                   for shape in shapes.values())
            or shapes['loss'] != [] or len(shapes['output']) != 3 or shapes['output'][0] != 1
            or shapes['output'] != shapes['hidden']
            or any(shapes[name] != shapes['gradient/'+name[4:]] for name in shapes if name.startswith('sgd/'))):
        raise ValueError('declared global shapes require all model fields, matching parameter pairs and token geometry')
    return {name:list(shape) for name, shape in shapes.items()}


def assemble_global_model_fields(peers: list[dict], *, heads: int, cp_size: int, tp_size: int,
                                 declared_shapes: dict[str, list[int]] | None = None) -> dict:
    """Assemble all global fields after one CP gradient reduction without summing replicas.

    Every peer snapshot contains CPU tensors, CP/TP coordinates, global owner
    ids and all 25 model fields. Token ownership may be strided or empty. Head
    shards concatenate in TP order; already reduced gradients and updated
    parameters select one exactly verified CP replica. Replicated TP values
    must also agree exactly. This cold validation helper performs no collective
    and never chooses a replica by its agreement with the expected oracle.
    An explicit declared_shapes map extends mechanical reconstruction to a
    different model geometry; it does not alter a numerical acceptance policy.
    """
    if (isinstance(cp_size, bool) or isinstance(tp_size, bool)
            or not isinstance(cp_size, int) or not isinstance(tp_size, int)
            or cp_size not in (1, 2, 4, 8) or tp_size not in (1, 2, 4, 8)
            or cp_size*tp_size > 8 or len(peers) != cp_size*tp_size):
        raise ValueError('global model reconstruction requires a complete bounded CP/TP mesh')
    shapes = _declared_shapes(heads, declared_shapes)
    mesh = {}
    for peer in peers:
        coordinate = (peer['cp_rank'], peer['tp_rank'])
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
                or coordinate in mesh or coordinate[0] not in range(cp_size) or coordinate[1] not in range(tp_size)
                or set(peer['fields']) != set(shapes)):
            raise ValueError('global model reconstruction has duplicate coordinates or incomplete fields')
        ids = tuple(peer['owner_ids'])
        if len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise ValueError('global model reconstruction requires unique integer owner ids')
        for name, value in peer['fields'].items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or not value.is_floating_point():
                raise ValueError('global model reconstruction requires explicit floating CPU snapshots')
            shape = list(shapes[name])
            if name in ('output', 'hidden'):
                shape[1] = len(ids)
            elif name.startswith(('gradient/', 'sgd/')) and name.split('/', 1)[1] in _HEAD_AXES:
                shape[_HEAD_AXES[name.split('/', 1)[1]]] //= tp_size
            if list(value.shape) != shape:
                raise ValueError(f'global model reconstruction has an incompatible local shape: {name}')
        mesh[coordinate] = peer
    owned = [i for cp in range(cp_size) for i in mesh[cp, 0]['owner_ids']]
    if sorted(owned) != list(range(shapes['output'][1])):
        raise ValueError('global model reconstruction requires exactly one CP owner per global token')
    for cp in range(cp_size):
        for tp in range(1, tp_size):
            if mesh[cp, tp]['owner_ids'] != mesh[cp, 0]['owner_ids']:
                raise ValueError('global model reconstruction TP peers disagree on CP ownership')
    result = {}
    for name, shape in shapes.items():
        for coordinate, peer in mesh.items():
            cp, tp = coordinate
            reference = mesh[cp, 0] if name in ('output', 'hidden') else mesh[0, tp]
            if not _same(peer['fields'][name], reference['fields'][name]):
                raise ValueError(f'global model reconstruction has inconsistent replicas: {name}, {coordinate}')
        first = mesh[0, 0]['fields'][name]
        if name in ('output', 'hidden'):
            if any(peer['fields'][name] is None for peer in peers):
                raise ValueError(f'global model reconstruction requires every owner token field: {name}')
            result[name] = torch.empty(shape, dtype=first.dtype)
            for cp in range(cp_size):
                result[name][:, mesh[cp, 0]['owner_ids']] = mesh[cp, 0]['fields'][name]
        elif name.startswith(('gradient/', 'sgd/')) and name.split('/', 1)[1] in _HEAD_AXES:
            values = [mesh[0, tp]['fields'][name] for tp in range(tp_size)]
            if any(value is None for value in values) and not all(value is None for value in values):
                raise ValueError(f'global model reconstruction has inconsistent gradient availability: {name}')
            result[name] = (None if first is None else torch.cat(values, dim=_HEAD_AXES[name.split('/', 1)[1]]))
        else:
            if any(not _same(peer['fields'][name], first) for peer in peers):
                raise ValueError(f'global model reconstruction has inconsistent replicated TP fields: {name}')
            result[name] = first
    return result
