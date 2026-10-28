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
"""Collect bounded traces, optionally deferring all instrumentation until the tail."""

from __future__ import annotations

from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
import statistics
from typing import Any

from torch.profiler import record_function  # pylint: disable=forbidden-backend-import
import torch_npu

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from examples.training_demo.deepseek_v41.deepseek_v41_clean_abba import run_window
from hyper_parallel.core.activation_memory import is_recomputing
import hyper_parallel.core.multicore.modules.mega_moe.function as moe_function


ORIGINAL_RECIPE = benchmark._recipe
ORIGINAL_STEP = benchmark._step
SELECTED = int(os.environ.get("LOCAL_RANK", "0")) == 0
HOOKS = []
TAIL = None


class TailProfiler:
    """Leave clean steps untouched, then collect two Level0 steps on one rank per node."""

    def __init__(self, args: Any, work: Path) -> None:
        """Record the fixed diagnostic window before any device profiler is created."""
        self.window = run_window(os.environ.get('CLEAN_NUMA_TAIL') == '1')
        if args.steps != self.window.steps or args.warmup_steps != self.window.warmup:
            raise ValueError(f'Tail profiling requires {self.window.steps} total steps '
                             f'and {self.window.warmup} warmup steps')
        rank = int(os.environ.get('RANK', '0'))
        self.path = work / 'profiling-config.json'
        self.evidence = {
            'purpose': 'clean_prefix_and_tail_diagnosis', 'rank': rank, 'profiling_enabled': SELECTED,
            'profiler_level': 0, 'clean_steps_inclusive': [self.window.warmup + 1, self.window.clean_end],
            'profiler_warmup_step': self.window.profile_warmup,
            'profile_steps_inclusive': [self.window.profile_start, self.window.profile_end - 1],
            'executed_steps': args.steps, 'scheduler_horizon': args.steps, 'trace_exported': False,
            'trace_dir': str(Path(args.output).parents[1] / 'profiles' / Path(args.output).name / f'rank{rank}'),
        }
        self.profiler = None
        self.originals: list[tuple[Any, Any]] = []
        self.dry_run = getattr(args, 'dry_run', False)
        self.path.write_text(json.dumps(self.evidence, indent=2), encoding='utf-8')

    def start(self, trainer: Any) -> None:
        """Create the profiler and CPU ranges only after all accepted clean steps."""
        if not SELECTED:
            return
        install(trainer)
        for name, module in trainer.base.model.named_modules():
            if name.endswith(('.attn_hc', '.ffn_hc')):
                self.originals.append((module, annotate_advance(module, name)))
            elif name.endswith('.mlp.gate'):
                before, after = _module_hooks(name)
                HOOKS.append(module.register_forward_pre_hook(before))
                HOOKS.append(module.register_forward_hook(after, always_call=True))
        profiler = torch_npu.profiler
        handler = profiler.tensorboard_trace_handler(self.evidence['trace_dir'])

        def export(trace: Any) -> None:
            """Mark successful export only after the synchronous trace handler returns."""
            handler(trace)
            self.evidence['trace_exported'] = True
            self.path.write_text(json.dumps(self.evidence, indent=2), encoding='utf-8')

        self.profiler = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
            on_trace_ready=export, record_shapes=False, profile_memory=False, with_stack=False,
            with_modules=False,
            experimental_config=profiler._ExperimentalConfig(  # pylint: disable=protected-access
                profiler_level=profiler.ProfilerLevel.Level0, aic_metrics=profiler.AiCMetrics.AiCoreNone,
                data_simplification=False),
        )
        self.profiler.start()

    def close(self) -> None:
        """Flush the bounded trace and release all optional module hooks."""
        try:
            if self.profiler is not None:
                self.profiler.stop()
        finally:
            self.profiler = None
            for hook in HOOKS:
                hook.remove()
            HOOKS.clear()
            for module, original in self.originals:
                module.advance = original
            self.originals.clear()

    def finish(self) -> None:
        """Attach effective profile evidence after the production benchmark has returned."""
        self.close()
        if SELECTED and not self.evidence['trace_exported']:
            raise RuntimeError(f"Rank {self.evidence['rank']}: tail profiler did not export a trace")
        result_path = self.path.with_name('result.json')
        result = json.loads(result_path.read_text())
        result['profiling_enabled'] = SELECTED
        result['tail_profile'] = self.evidence
        measured = result['steps_detail'][self.window.warmup:self.window.clean_end]
        result['measurement']['purpose'] = 'raw_window_including_tail_profiler_not_for_speedup'
        result['clean_measurement'] = {
            'first_step': self.window.warmup + 1, 'last_step': self.window.clean_end, 'steps': len(measured),
            'mean_max_rank_seconds': statistics.mean(row['max_rank_seconds'] for row in measured),
            'median_max_rank_seconds': statistics.median(row['max_rank_seconds'] for row in measured),
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in measured),
            'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in measured),
        }
        self.path.write_text(json.dumps(self.evidence, indent=2), encoding='utf-8')
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')


def recipe(args: Any, work: Path, world: int) -> Any:
    """Enable the existing CPU/NPU profiler on one rank per node."""
    global TAIL  # pylint: disable=global-statement
    config = ORIGINAL_RECIPE(args, work, world)
    if os.environ.get('CLEAN_PROFILE_TAIL') == '1':
        config.profiling.enabled = False
        TAIL = TailProfiler(args, work)
        return config
    rank = int(os.environ.get("RANK", "0"))
    start = int(os.environ.get("PROFILE_START_STEP", "3"))
    end = int(os.environ.get("PROFILE_END_STEP", "5"))
    horizon = int(os.environ.get("SCHEDULE_TRAIN_ITERS", str(args.steps)))
    if not 1 <= start < end <= args.steps or horizon < args.steps:
        raise ValueError("Invalid profiling window or scheduler horizon")
    config.training.train_iters = horizon
    config.profiling.enabled = SELECTED
    config.profiling.rank = rank
    config.profiling.start_step = start
    config.profiling.end_step = end
    config.profiling.trace_dir = str(Path(args.output).parents[1] / "profiles" / Path(args.output).name / f"rank{rank}")
    config.profiling.record_shapes = False
    config.profiling.profile_memory = False
    config.profiling.with_stack = False
    config.profiling.with_modules = False
    evidence = {
        "purpose": "performance_diagnosis_not_speedup_acceptance",
        "rank": rank, "profiling_enabled": SELECTED,
        "profile_steps_inclusive": [start, end - 1],
        "executed_steps": args.steps, "scheduler_horizon": horizon,
        "trace_dir": config.profiling.trace_dir,
    }
    (work / "profiling-config.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return config


def annotate(module: Any, name: str, label: str) -> None:
    """Add a CPU range that preserves the wrapped function's semantics."""
    original = getattr(module, name)

    @wraps(original)
    def measured(*args: Any, **kwargs: Any) -> Any:
        """Preserve the observed function result within a named CPU range."""
        with record_function(label):
            return original(*args, **kwargs)

    setattr(module, name, measured)


def annotate_advance(module: Any, name: str) -> Any:
    """Observe the full shifted mHC boundary, which does not call module.forward on NPU."""
    original = module.advance

    @wraps(original)
    def measured(*args: Any, **kwargs: Any) -> Any:
        """Keep post, coefficient prediction, pre mixing and RMSNorm inside one CPU range."""
        phase = 'recompute' if is_recomputing() else 'forward'
        with record_function(f'dsv41.{name}.{phase}'):
            return original(*args, **kwargs)

    module.advance = measured
    return original


def _module_hooks(name: str) -> tuple[Any, Any]:
    pending: list[ExitStack] = []

    def before(_module: Any, _inputs: tuple) -> None:
        """Open a balanced CPU range for the current forward or replay."""
        phase = "recompute" if is_recomputing() else "forward"
        stack = ExitStack()
        stack.enter_context(record_function(f"dsv41.{name}.{phase}"))
        pending.append(stack)

    def after(_module: Any, _inputs: tuple, _output: Any) -> None:
        """Close the matching range, including checkpoint early-stop calls."""
        pending.pop().close()

    return before, after


def install(trainer: Any) -> None:
    """Name MegaMoe kernel launches and native/MegaMoe module boundaries."""
    annotate(moe_function, "_launch_forward_kernel", "dsv41.moe.launch_forward")
    annotate(moe_function, "_launch_backward_kernel", "dsv41.moe.launch_backward")
    for name, module in trainer.base.model.named_modules():
        if not name.endswith(".mlp") or not hasattr(module, "gate"):
            continue
        before, after = _module_hooks(name)
        HOOKS.append(module.register_forward_pre_hook(before))
        HOOKS.append(module.register_forward_hook(after, always_call=True))


def step(trainer: Any, iterator: Any, index: int, work: Path, *args: Any) -> dict[str, Any]:
    """Install ranges once, keeping tensors and production step ordering intact."""
    if TAIL is not None:
        number = index + 1
        if number == TAIL.window.profile_warmup:
            TAIL.start(trainer)
        if TAIL.profiler is None:
            return ORIGINAL_STEP(trainer, iterator, index, work, *args)
        with record_function('dsv41.train_step'):
            row = ORIGINAL_STEP(trainer, iterator, index, work, *args)
        TAIL.profiler.step()
        if number == TAIL.window.profile_end - 1:
            TAIL.close()
        return row
    if SELECTED and not HOOKS:
        install(trainer)
    with record_function("dsv41.train_step"):
        return ORIGINAL_STEP(trainer, iterator, index, work, *args)


def main() -> None:
    """Install model-local profiling hooks while preserving the normal trainer callbacks."""
    benchmark._recipe = recipe
    benchmark._step = step
    try:
        benchmark.main()
        if TAIL is not None and not TAIL.dry_run:
            TAIL.finish()
    finally:
        if TAIL is not None:
            TAIL.close()


if __name__ == "__main__":
    main()
