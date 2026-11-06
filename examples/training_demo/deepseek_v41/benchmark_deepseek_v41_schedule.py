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
"""Switch controlled logical routes between complete profiler-off Trainer steps."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from examples.training_demo.deepseek_v41.deepseek_v41_benchmark_control import record_failure
from examples.training_demo.deepseek_v41.deepseek_v41_route_schedule import RouteSchedule, load_schedule
from examples.training_demo.deepseek_v41.dirichlet_routes import make_route, route_metadata
from hyper_parallel.models.deepseek_v41.adapter.distributed import moe_engram_expert_parallel as adapter


class ScheduledRoutes:
    """Keep destinations stable across forward/replay and prepare routes only while settling."""

    def __init__(self, schedule: RouteSchedule, ep: int) -> None:
        """Bind real router and step functions before installing the experiment wrappers."""
        self.schedule, self.ep = schedule, ep
        self.original_router, self.original_step = adapter._router, benchmark._step
        self.layer_indices: dict[int, int] = {}
        self.cache: dict[tuple, tuple[Any, Any]] = {}
        self.metadata: dict[tuple[str, int], dict[int, Any]] = {}
        self.observed: dict[str, list[int]] = {window.name: [] for window in schedule.windows}
        self.current: Any = None
        self.current_index = -1
        self.next_step = 0

    def step(self, trainer: Any, iterator: Any, index: int, work: Path, *args: Any) -> dict[str, Any]:
        """Change profiles before the timing boundary, after the preceding optimizer completed."""
        if index != self.next_step:
            raise ValueError(f'Nonconsecutive route-window step: {index} != {self.next_step}')
        self.current = self.schedule.at(index)
        self.current_index = index
        if args:
            args[0].route_window = dict(routing_window=self.current.name, routing_profile=self.current.profile)
        if not self.layer_indices:
            for name, module in trainer.base.model.named_modules():
                match = re.search(r'(?:^|\.)layers\.(\d+)\.mlp$', name)
                if match:
                    self.layer_indices[id(module)] = int(match.group(1))
            if not self.layer_indices:
                raise RuntimeError('No decoder MoE layers found for controlled routing')
        result = self.original_step(trainer, iterator, index, work, *args)
        self.observed[self.current.name].append(index)
        self.next_step += 1
        return result

    def router(self, module: Any, states: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Retain real gate work/gradients and replace IDs with cached exact-count routes."""
        if self.current is None:
            raise RuntimeError('Router invoked before an optimizer-step route was selected')
        result = self.original_router(module, states, *args, **kwargs)
        ids, weights = result[:2]
        layer = self.layer_indices[id(module)]
        tokens = ids.numel() // ids.shape[-1]
        key = self.current.profile, self.current.seed, layer, tokens, str(ids.device), str(ids.dtype)
        if key not in self.cache:
            if self.current_index in self.current.measured_indices:
                raise RuntimeError('A measured step requires an already prepared route shape')
            alpha = None if self.current.profile == 'balanced' else float(self.current.profile)
            controlled, counts = make_route(tokens, module.gate.num_experts, ids.shape[-1],
                                             alpha, self.current.seed + layer * 1009)
            record = route_metadata(controlled, counts, self.ep)
            layers = self.metadata.setdefault((self.current.profile, self.current.seed), {})
            if layer in layers and layers[layer] != record:
                raise ValueError('Controlled full-model timing requires a constant token shape per layer')
            layers[layer] = record
            self.cache[key] = controlled.to(device=ids.device, dtype=ids.dtype), counts.to(ids.device)
        controlled, counts = self.cache[key]
        # Learned counts describe the original destinations and must never accompany replacement IDs.
        return ((controlled.view_as(ids), weights, counts) if len(result) == 3
                else (controlled.view_as(ids), weights))

    def summary(self) -> dict[str, Any]:
        """Describe applied window boundaries and immutable CPU route templates after training."""
        return dict(schedule=self.schedule.record(), schedule_sha256=self.schedule.sha256,
                    completed_steps=self.next_step, shared_source_profile=True, learned_weights_preserved=True,
                    synthetic_performance_only=True, plan_observation_hooks=False,
                    route_evidence='Exact CPU template uploaded by the installed router; no device ID readback',
                    windows=[dict(window=asdict(window), observed_steps=self.observed[window.name],
                                  layers=self.metadata.get((window.profile, window.seed), {}))
                             for window in self.schedule.windows])


def main() -> None:
    """Install the same schedule on Native/Push/Pull after CPU placement and before Trainer construction."""
    selected = load_schedule(Path(os.environ['DSV41_ROUTE_SCHEDULE']))
    if selected.sha256 != os.environ.get('DSV41_ROUTE_SCHEDULE_SHA256', selected.sha256):
        raise ValueError('Worker route schedule differs from the frozen launch')
    for flag, expected in (('--steps', selected.steps), ('--warmup-steps', selected.warmup)):
        if int(sys.argv[sys.argv.index(flag) + 1]) != expected:
            raise ValueError(f'{flag} differs from the frozen route schedule')
    if sys.argv[sys.argv.index('--modality') + 1] != 'text' or '--fix-router' in sys.argv:
        raise ValueError('Controlled route windows require text training without fix_router')
    output = Path(sys.argv[sys.argv.index('--output') + 1]) / f'rank{os.environ["RANK"]}'
    routes = ScheduledRoutes(selected, int(os.environ.get('EP_SIZE', '64')))
    original_step, original_router = benchmark._step, adapter._router
    benchmark._step, adapter._router = routes.step, routes.router
    try:
        benchmark.main()
    except BaseException as error:
        record_failure(output, error)
        raise
    finally:
        benchmark._step, adapter._router = original_step, original_router
        output.mkdir(parents=True, exist_ok=True)
        (output / 'routing-windows.json').write_text(json.dumps(routes.summary(), indent=2) + '\n')


if __name__ == '__main__':
    main()
