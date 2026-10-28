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
"""Observe MoE current-stream spans without changing tensors or retaining graphs."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from hyper_parallel.core.activation_memory import is_recomputing


class MoeTiming:
    """Record module forward/replay and output-to-input-gradient intervals.

    Events stay on the current stream. They include waits visible to that stream,
    and do not claim to delimit work remaining on other communication streams or
    weight-gradient tails. Backward intervals can include checkpoint replay;
    phases must not be added to infer training-step wall time.
    """

    def __init__(self, model: torch.nn.Module, event_factory: Callable | None = None) -> None:
        """Attach tensor-preserving hooks to routed-plus-shared MoE modules."""
        self.event_factory = event_factory or (lambda: torch.npu.Event(enable_timing=True))
        self.records: list[dict[str, Any]] = []
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.hooks = []
        self.grad_hooks = []
        self.layers = [name for name, module in model.named_modules()
                       if name.endswith('.mlp') and hasattr(module, 'gate')]
        if not self.layers:
            raise ValueError('No MoE modules found for timing')
        for name, module in model.named_modules():
            if name in self.layers:
                self.hooks.append(module.register_forward_pre_hook(self._start(name)))
                self.hooks.append(module.register_forward_hook(self._finish(name), always_call=True))

    def _event(self) -> Any:
        event = self.event_factory()
        event.record()
        return event

    def _start(self, name: str) -> Callable:
        def before(_module: torch.nn.Module, inputs: tuple) -> None:
            """Open a forward interval and observe original input-gradient readiness."""
            replay = is_recomputing()
            record = {'layer': name, 'phase': 'recompute_forward' if replay else 'forward',
                      'start': self._event(), 'end': None, 'completed': False}
            self.records.append(record)
            self.pending.setdefault(name, []).append(record)
            if not replay and inputs[0].requires_grad:
                backward = {'layer': name, 'phase': 'backward_to_input_grad',
                            'start': None, 'end': None, 'completed': False}
                record['backward'] = backward
                self.records.append(backward)

                def input_ready(_gradient: torch.Tensor) -> None:
                    """Close the backward interval without changing its gradient."""
                    backward['end'] = self._event()
                    backward['completed'] = True

                self.grad_hooks.append(inputs[0].register_hook(input_ready))
        return before

    def _finish(self, name: str) -> Callable:
        def after(_module: torch.nn.Module, _inputs: tuple, output: Any) -> None:
            """Close forward timing and observe the original output gradient."""
            record = self.pending[name].pop()
            record['end'] = self._event()
            # Early-stop checkpoint replay may legitimately leave output absent.
            record['completed'] = output is not None
            backward = record.get('backward')
            if backward is not None and isinstance(output, torch.Tensor) and output.requires_grad:
                def output_ready(_gradient: torch.Tensor) -> None:
                    """Open the backward interval when the output gradient arrives."""
                    backward['start'] = self._event()

                self.grad_hooks.append(output.register_hook(output_ready))
        return after

    def snapshot(self) -> dict[str, Any]:
        """Resolve event pairs only after the caller synchronizes the device."""
        layers: dict[str, Any] = {name: {} for name in self.layers}
        incomplete = []
        for record in self.records:
            name, phase = record['layer'], record['phase']
            if record['start'] is None or record['end'] is None:
                incomplete.append({'layer': name, 'phase': phase})
                continue
            elapsed = record['start'].elapsed_time(record['end'])
            entry = layers[name].setdefault(phase, {'current_stream_ms': 0.0, 'calls': 0, 'completed_calls': 0})
            entry['current_stream_ms'] += elapsed
            entry['calls'] += 1
            entry['completed_calls'] += int(record['completed'])
        return {'method': 'current_stream_events', 'backward_includes_replay_if_nested': True,
                'layers': layers, 'incomplete_intervals': incomplete}

    def clear(self) -> None:
        """Release events and gradient hooks after the synchronized step."""
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks.clear()
        self.records.clear()
        self.pending.clear()

    def close(self) -> None:
        """Restore all module and gradient hooks."""
        self.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
