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
"""Matched all-off/all-on MegaMoe, MegaMhc and MegaGate benchmark support."""
# pylint: disable=forbidden-backend-import
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.mhc import PipelinedMhcModule
from hyper_parallel.core.activation_memory import is_recomputing
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import (
    _reference_shifted_boundary, _rms_norm_eps,
)
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class ReferenceMhc(PipelinedMhcModule):
    """Keep the legacy mHC operations within the same shifted boundary."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Reuse parameters and the source normalization epsilon."""
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        self.norm_eps = _rms_norm_eps(self.input_norm)

    def advance(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor, norm_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run post, coefficient prediction, pre mixing and RMSNorm without MegaMhc."""
        return _reference_shifted_boundary(
            self, previous_output, residual, previous_pre_mix, previous_post_mix,
            previous_residual_mix, norm_weight, self.norm_eps,
        )


def select_kernels(recipe: dict[str, Any], enabled: bool) -> None:
    """Select the matched mHC and Gate paths; the caller selects the MoE backend."""
    for entry in recipe['plan_overrides']:
        target = entry.get('replace_module', {}).get('_target_', '')
        if target.endswith('.DeepseekV41HyperMegaMhc') and not enabled:
            entry['replace_module'] = {
                '_target_': 'examples.training_demo.deepseek_v41.deepseek_v41_mega_ab.ReferenceMhc',
            }
    if not enabled:
        recipe['plan_overrides'] = [
            entry for entry in recipe['plan_overrides']
            if not entry.get('replace_module', {}).get('_target_', '').endswith('.DeepseekV41MegaGate')
        ]


def _tensors(value: Any) -> list[torch.Tensor]:
    """Find differentiable inputs without modifying or retaining their containers."""
    if isinstance(value, torch.Tensor):
        return [value] if value.requires_grad else []
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensors(item)]
    return []


class ModuleTiming:
    """Time matched module boundaries without tensor wrappers or graph retention.

    Current-stream intervals include visible waits. Backward ends when all used
    input gradients are ready and may include replay; phases are not additive.
    mHC includes post, next mappings, pre mixing and block-input RMSNorm in both
    cases. Gate includes projection and routing; MoE includes Gate and experts.
    """

    def __init__(self, model: nn.Module, event_factory: Callable | None = None) -> None:
        """Wrap existing calls and retain only removable gradient observers."""
        self.event_factory = event_factory or (lambda: torch.npu.Event(enable_timing=True))
        self.records: list[dict[str, Any]] = []
        self.grad_hooks: list[Any] = []
        self.originals: list[tuple[nn.Module, str, Callable]] = []
        self.groups: dict[str, list[str]] = {'moe': [], 'mhc': [], 'gate': []}
        for name, module in model.named_modules():
            if name.endswith('.mlp') and hasattr(module, 'gate'):
                self._install(module, 'forward', name, 'moe')
            elif name.endswith('.mlp.gate'):
                self._install(module, 'forward', name, 'gate')
            elif name.endswith(('.attn_hc', '.ffn_hc')):
                if not callable(getattr(module, 'advance', None)):
                    raise ValueError(f'Matched mHC timing requires advance(): {name}')
                self._install(module, 'advance', name, 'mhc')
        if any(not names for names in self.groups.values()):
            raise ValueError(f'Missing benchmark module group: {self.groups}')

    def _event(self) -> Any:
        event = self.event_factory()
        event.record()
        return event

    def _install(self, module: nn.Module, method: str, name: str, group: str) -> None:
        original = getattr(module, method)
        self.originals.append((module, method, original))
        self.groups[group].append(name)

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            """Observe the original call and its input-gradient boundary."""
            replay = is_recomputing()
            record = {'layer': name, 'group': group,
                      'phase': 'recompute_forward' if replay else 'forward',
                      'start': self._event(), 'end': None, 'completed': False}
            self.records.append(record)
            try:
                result = original(*args, **kwargs)
                record['completed'] = True
            finally:
                record['end'] = self._event()
            if not replay and torch.is_grad_enabled():
                self._observe_backward(record, _tensors((args, kwargs)), _tensors(result))
            return result

        setattr(module, method, measured)

    def _observe_backward(self, forward: dict[str, Any], inputs: list[torch.Tensor],
                          outputs: list[torch.Tensor]) -> None:
        if not inputs or not outputs:
            return
        record = {**forward, 'phase': 'backward_to_input_grad', 'start': None,
                  'end': None, 'completed': False}
        self.records.append(record)

        def output_ready(_gradients: Any) -> None:
            """Open the interval when the first used output gradient arrives."""
            record['start'] = self._event()

        def input_ready(_gradients: Any) -> None:
            """Close the interval after all used input gradients are available."""
            record['end'] = self._event()
            record['completed'] = True

        self.grad_hooks.append(torch.autograd.graph.register_multi_grad_hook(outputs, output_ready, mode='any'))
        self.grad_hooks.append(torch.autograd.graph.register_multi_grad_hook(inputs, input_ready, mode='all'))

    def snapshot(self) -> dict[str, Any]:
        """Resolve all groups after the benchmark synchronizes the device."""
        groups = {group: {name: {} for name in names} for group, names in self.groups.items()}
        incomplete = []
        for record in self.records:
            name, group, phase = record['layer'], record['group'], record['phase']
            if record['start'] is None or record['end'] is None:
                incomplete.append({'group': group, 'layer': name, 'phase': phase})
                continue
            entry = groups[group][name].setdefault(
                phase, {'current_stream_ms': 0.0, 'calls': 0, 'completed_calls': 0})
            entry['current_stream_ms'] += record['start'].elapsed_time(record['end'])
            entry['calls'] += 1
            entry['completed_calls'] += int(record['completed'])
        return {'method': 'current_stream_events', 'layers': groups['moe'],
                'module_groups': groups, 'incomplete_intervals': incomplete,
                'backward_includes_replay_if_nested': True, 'phases_additive': False}

    def clear(self) -> None:
        """Release observers and event references after the previous synchronized step."""
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks.clear()
        self.records.clear()

    def close(self) -> None:
        """Restore every original call and remove outstanding observers."""
        self.clear()
        for module, method, original in self.originals:
            setattr(module, method, original)
        self.originals.clear()
