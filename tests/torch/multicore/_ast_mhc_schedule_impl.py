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
"""在独立进程中测量 MHC 完整前向、反向和 SGD step；原生任务保持不变。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu

import hyper_parallel
from hyper_parallel.core.multicore.frontend.examples.mhc_boundary import mhc_boundary
from hyper_parallel.core.multicore.runtime.mhc_spec import MhcSpec
from tests.torch.multicore._ast_mhc_impl import _inputs

_CASES = ((2593, 128, 32), (4097, 512, 64), (12000, 128, 96))


def _step(executable, values, optimizer):
    for value in values:
        value.grad = None
    outputs = executable(*values)
    loss = sum(value.float().square().mean() for value in outputs)
    loss.backward()
    optimizer.step()
    return loss


def _trajectory(plan):
    values = tuple(value.detach().npu().requires_grad_() for value in _inputs(
        plan.spec.token_count, plan.spec.hidden_size, 971))
    executable = plan.materialize()
    optimizer = torch.optim.SGD(values[5:], lr=0.0001)
    losses = []
    try:
        for _ in range(5):
            losses.append(float(_step(executable, values, optimizer).detach().cpu()))
        torch.npu.synchronize()
        weights = [value.detach().cpu().float().tolist() for value in values[5:]]
        return {"losses": losses, "parameters": weights}
    finally:
        executable.close()


def _case(tokens, hidden, tile, policy, iterations, warmup, root):
    cores = torch.npu.get_device_limit(0)["cube_core_num"]
    plan = mhc_boundary.plan(MhcSpec(tokens, hidden, cores, tile, tile), event_schedule=policy)
    label = f"t{tokens}-h{hidden}-tile{tile}"
    trajectory = _trajectory(plan)
    (root / f"{label}-trajectory.json").write_text(json.dumps(trajectory) + "\n")
    values = tuple(value.detach().npu().requires_grad_() for value in _inputs(tokens, hidden, 971))
    executable = plan.materialize()
    optimizer = torch.optim.SGD(values[5:], lr=0.0001)
    try:
        for _ in range(warmup):
            _step(executable, values, optimizer)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        baseline = torch.npu.memory_allocated()
        elapsed = []
        for _ in range(3):
            begin = time.perf_counter()
            for _ in range(iterations):
                _step(executable, values, optimizer)
            torch.npu.synchronize()
            elapsed.append((time.perf_counter() - begin) * 1000 / iterations)
        peak = torch.npu.max_memory_allocated()
        for value in values:
            value.grad = None
        outputs = executable(*values, profile=True)
        sum(value.float().square().mean() for value in outputs).backward()
        records = [record for profile in executable.take_profiles() for record in profile.records()]
        (root / f"{label}-cycles.json").write_text(json.dumps(records) + "\n")
        return {"shape": [tokens, hidden, tile], "policy": policy, "step_ms": elapsed,
                "iterations_per_block": iterations, "warmup": warmup, "allocated_before_bytes": baseline,
                "peak_allocated_bytes": peak, "incremental_peak_bytes": peak - baseline,
                "task_count": sum(len(queue) for queue in plan.backward.queues),
                "trajectory": f"{label}-trajectory.json", "profile": f"{label}-cycles.json"}
    finally:
        executable.close()


def run_benchmark(policy: str, result_dir: str, iterations: int = 20, warmup: int = 5) -> None:
    """测量独立训练轨迹、完整 step 和已分配 HBM，不混用 profiling 时间。

    Args:
        policy: compatibility、mhc_backward_overlap 或 mhc_backward_wavefront。
        result_dir: 当前独立进程的结果目录。
        iterations: 每个计时块的完整 step 数。
        warmup: 计时前的完整 step 数。
    """
    if iterations <= 0 or warmup < 0:
        raise ValueError("Benchmark iterations must be positive and warmup nonnegative")
    torch.npu.set_device(0)
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for tokens, hidden, tile in _CASES:
        record = _case(tokens, hidden, tile, policy, iterations, warmup, root)
        cases.append(record)
        print(json.dumps(record), flush=True)
    data = {"policy": policy, "checkout": hyper_parallel.__file__, "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "scope": "MHC_forward_backward_SGD", "cases": cases}
    (root / "benchmark.json").write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    """在已激活匹配原生产物的独立设备进程中执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("compatibility", "mhc_backward_overlap", "mhc_backward_wavefront",
                                             "mhc_backward_wavefront_fast"),
                        required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    options = parser.parse_args()
    run_benchmark(options.policy, options.result_dir, options.iterations, options.warmup)


if __name__ == "__main__":
    main()
