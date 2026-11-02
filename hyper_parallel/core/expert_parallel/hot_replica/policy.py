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
"""Private cost selection for default bounded replica adaptation."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state

from .cost import ExpertReplicaCostModel
from .critical_path import execution_fingerprint


_LOGGER = logging.getLogger(__name__)
_SHARED_SOC_CALIBRATIONS = {'Ascend910_9382': 'Ascend910_9392'}


def _cost_model_enabled() -> bool:
    """Read the diagnostic switch once per execution owner, before communication."""
    value = os.environ.get('HYPER_PARALLEL_REPLICA_COST_MODEL', '1')
    if value not in ('0', '1'):
        raise ValueError('HYPER_PARALLEL_REPLICA_COST_MODEL must be 0 or 1')
    return value == '1'


@lru_cache(maxsize=1)
def _calibrations() -> dict[tuple, ExpertReplicaCostModel]:
    """Own immutable package calibrations with exact hardware and workload selectors."""
    models = {}
    for path in sorted(Path(__file__).with_name('calibrations').glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        model = ExpertReplicaCostModel(**payload['model'])
        selector = payload['selector']
        key = (model.backend, model.hidden_size, model.intermediate_size, model.ep_size, model.transport,
               selector['num_experts'], selector['top_k'], selector['soc_name'])
        if model.critical_path is not None:
            key += (model.critical_path.execution_signature,)
        if key in models:
            raise ValueError(f'Duplicate replica calibration: {key}')
        models[key] = model
    return models


@lru_cache(maxsize=128)
def _calibration_available(backend: str, hidden: int, intermediate: int, ep_size: int, transport: str,
                           num_experts: int, top_k: int) -> bool:
    """Avoid hardware queries for executions with no packaged compute calibration."""
    prefix = backend, hidden, intermediate, ep_size, transport, num_experts, top_k
    return any(key[:7] == prefix for key in _calibrations())


@lru_cache(maxsize=128)
def _critical_path_available(backend: str, hidden: int, intermediate: int, ep_size: int, transport: str,
                             num_experts: int, top_k: int) -> bool:
    """Avoid payload hashing on executions with no critical-path profile."""
    prefix = backend, hidden, intermediate, ep_size, transport, num_experts, top_k
    return any(key[:7] == prefix and model.critical_path is not None
               for key, model in _calibrations().items())


@lru_cache(maxsize=128)
def _critical_path_identity(backend: str, hidden: int, intermediate: int, ep_size: int, transport: str,
                            num_experts: int, top_k: int) -> str:
    """Bind the existing group agreement to the same immutable cost profiles."""
    prefix = backend, hidden, intermediate, ep_size, transport, num_experts, top_k
    profiles = [(key, asdict(model)) for key, model in _calibrations().items() if key[:7] == prefix]
    return execution_fingerprint({'profiles':sorted(profiles, key=lambda item:repr(item[0]))})


@lru_cache(maxsize=128)
def _resolve_cost_model(backend: str, hidden: int, intermediate: int, ep_size: int, transport: str,
                        num_experts: int, top_k: int, soc_name: str,
                        checkpointed: bool, execution_signature: str | None = None) -> ExpertReplicaCostModel | None:
    """Match an existing calibration; unmeasured executions keep load-driven planning."""
    key = backend, hidden, intermediate, ep_size, transport, num_experts, top_k, soc_name
    model = _calibrations().get(key + (execution_signature,)) if execution_signature is not None else None
    model = model or _calibrations().get(key)
    table_soc = soc_name
    if model is None:
        table_soc = _SHARED_SOC_CALIBRATIONS.get(soc_name, soc_name)
        shared_key = key[:-1] + (table_soc,)
        model = (_calibrations().get(shared_key + (execution_signature,))
                 if execution_signature is not None else None)
        model = model or _calibrations().get(shared_key)
    if not dist.is_initialized() or dist.get_rank() == 0:
        _LOGGER.info('Replica cost calibration: status=%s backend=%s shape=%s/%s ep=%s experts=%s top_k=%s '
                     'actual_soc=%s table_soc=%s checkpointed=%s fanout=%s critical_path=%s signature=%s',
                     'matched' if model is not None else 'fallback', backend, hidden, intermediate, ep_size,
                     num_experts, top_k, soc_name, table_soc if model is not None else 'none', checkpointed,
                     bool(model is not None and model.weight_fanout_ms),
                     bool(model is not None and model.critical_path is not None), execution_signature)
    if model is None:
        return None
    return replace(model, forward_repeats=2 if checkpointed else 1,
                   backward_weight_ms=model.backward_weight_ms if checkpointed else model.weight_ms,
                   intra_node_backward_weight_ms=(model.intra_node_backward_weight_ms if checkpointed else
                                                 model.intra_node_weight_ms),
                   inter_node_backward_weight_ms=(model.inter_node_backward_weight_ms if checkpointed else
                                                 model.inter_node_weight_ms))


@lru_cache(maxsize=16)
def _soc_name(device: torch.device) -> str:
    """Query selected hardware once without probing a device from CPU-only setup."""
    return str(torch.npu.get_device_name(device)) if device.type == 'npu' else ''


def _runtime_cost_model(enabled: bool, backend: str, hidden: int, intermediate: int, ep_size: int,
                        transport: str, num_experts: int, top_k: int,
                        device: torch.device, *, execution_signature: str | None = None) -> ExpertReplicaCostModel | None:
    """Resolve invocation costs after device selection and checkpoint context binding."""
    if not enabled or not _calibration_available(backend, hidden, intermediate, ep_size, transport,
                                                num_experts, top_k):
        return None
    arguments = (backend, hidden, intermediate, ep_size, transport, num_experts, top_k,
                 _soc_name(device), get_recompute_state() is not None)
    return (_resolve_cost_model(*arguments) if execution_signature is None else
            _resolve_cost_model(*arguments, execution_signature))
