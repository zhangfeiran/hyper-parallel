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
"""Collect a bounded packed-push trace without changing the training horizon."""

from __future__ import annotations

from contextlib import ExitStack
from functools import wraps
import json
import os
from pathlib import Path
from typing import Any

from torch.profiler import record_function  # pylint: disable=forbidden-backend-import

from examples.training_demo.deepseek_v41 import benchmark_deepseek_v41_vlm as benchmark
from hyper_parallel.core.activation_memory import is_recomputing
import hyper_parallel.core.multicore.modules.mega_moe.function as moe_function


ORIGINAL_RECIPE = benchmark._recipe
ORIGINAL_STEP = benchmark._step
SELECTED = int(os.environ.get("LOCAL_RANK", "0")) == 0
HOOKS = []


def recipe(args: Any, work: Path, world: int) -> Any:
    """Enable the existing CPU/NPU profiler on one rank per node."""
    config = ORIGINAL_RECIPE(args, work, world)
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
    if SELECTED and not HOOKS:
        install(trainer)
    with record_function("dsv41.train_step"):
        return ORIGINAL_STEP(trainer, iterator, index, work, *args)


if __name__ == "__main__":
    benchmark._recipe = recipe
    benchmark._step = step
    benchmark.main()
