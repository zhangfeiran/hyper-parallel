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
"""不导入框架的 Source-NK 原始完整训练与新进程 ABBA 验收监督入口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _load(name, filename):
    specification = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# 按脚本路径复用既有归属、噪声和完整训练步分析，不导入 HyperParallel 包根。
BASE = _load("_source_acceptance_base", "acceptance.py")
CONTRACT = _load("_source_acceptance_contract", "source_contract.py")


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _numerical(record):
    # 原有八项阈值与完整 100 步检查继续适用，新增来源还必须证明实际参数组更新次数。
    if not BASE._numerical_passed(record):  # pylint: disable=protected-access
        return False
    for row in record["steps"]:
        for kind, check in row["checks"].items():
            expected_atol = 0.002 if kind in ("logits", "loss", "gradients", "parameters") else (
                0 if kind == "step" else 1e-8)
            if check.get("atol") != expected_atol or check.get("rtol") != (0 if kind == "step" else 0.02):
                return False
        if row["checks"]["step"].get("actual_update_count_passed") is not True:
            return False
    return True


def _resident(record, baseline):
    payloads = record.get("native_payloads") or {}
    if record["mode"] == "validate":
        payloads = payloads.get("candidate") or {}
    layers = record["workload"].get("config", {}).get("num_layers", 0)
    return (layers > 0 and set(payloads) == {f"layers.{index}.mlp" for index in range(layers)}
            and all(entries and all(CONTRACT.resident_source_entry(entry, baseline) for entry in entries)
                    for entries in payloads.values()))


def analyze_suite(suite: dict[str, object]) -> dict[str, object]:
    """保持两种原始数值契约的身份，并验收全部独立训练和完整优化器步。

    Args:
        suite: 包含既定原生对照、独立 100 步及至少三块 ABBA 的已完成任务日志。

    Returns:
        数值轨迹、设备执行与性能的分别判定；FSDP 及恢复仍由独立门禁证明。
    """
    expected_jobs = BASE.suite_jobs(suite["blocks"])
    if not suite["complete"] or len(suite["jobs"]) != len(expected_jobs):
        raise ValueError("Source-NK suite is incomplete")
    identities, jobs, processes, shared_identity = {}, [], set(), None
    for job, expected in zip(suite["jobs"], expected_jobs):
        if any(job[key] != value for key, value in expected.items()):
            raise ValueError("Source-NK suite order differs from the requested native/control experiment")
        baseline = job["baseline"]
        # 沿用同一来源、设备、数据、归属和时间窗口规则；每种数值契约内部身份必须完全相同。
        record, identity, process = BASE._checked_result(job, identities.get(baseline))  # pylint: disable=protected-access
        identities[baseline] = identity
        source = identity["identity"]
        if (source.get("numerical_contract") != baseline or source.get("source_weight_layout") != "nk"
                or source.get("deterministic_algorithms") is not True):
            raise ValueError("Source-NK record has a different numerical source or determinism contract")
        if job["mode"] == "measure":
            allocated, reserved = record.get("peak_allocated_bytes"), record.get("peak_reserved_bytes")
            if (type(allocated) not in (int,) or type(reserved) not in (int,)
                    or allocated <= 0 or reserved < allocated):
                raise ValueError("Source-NK performance requires actual peak allocated/reserved HBM evidence")
        common = {**identity, "identity": {key: value for key, value in source.items() if key != "numerical_contract"}}
        if shared_identity is not None and common != shared_identity:
            raise ValueError("Source-NK common/packed source, data, topology or optimizer identities differ")
        shared_identity = common
        if process in processes:
            raise ValueError("Source-NK suite requires a fresh process for every workload")
        processes.add(process)
        jobs.append((job, record))
    numerical = all(_numerical(record) for job, record in jobs if job["mode"] == "validate")
    resident = all(_resident(record, job["baseline"]) for job, record in jobs if job["backend"] == "mega_ffn")
    comparisons = {baseline: BASE.performance_verdict(
        BASE._performance_blocks(jobs, baseline, "control"),  # pylint: disable=protected-access
        BASE._performance_blocks(jobs, baseline, "comparison"))  # pylint: disable=protected-access
        for baseline in ("common", "packed")}
    performance = numerical and resident and all(value["performance_passed"] for value in comparisons.values())
    return {"identities": identities, "numerical_passed": numerical, "resident_source_present": resident,
            "formal_original_training_accepted": numerical and resident,
            "performance_accepted": performance, "comparisons": comparisons,
            "scope": "固定数据、随机初始化的原始 common/packed 完整 Qwen；FSDP、恢复和长期收敛另行验收"}


def _launch(args, index, job):
    directory = args.root / f"job-{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    output, ownership = directory / "result.json", directory / "ownership.json"
    benchmark = [sys.executable, str(Path(__file__).with_name("source_training.py")), "--mode", job["mode"],
                 "--baseline", job["baseline"], "--backend", job["backend"], "--output", str(output),
                 *args.benchmark_args]
    worker = [sys.executable, str(Path(__file__).with_name("acceptance.py")), "worker", "--device", str(args.device),
              "--output", str(ownership), "--", *benchmark]
    environment = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.device),
                       NPU_WAIT_VISIBLE_DEVICES=str(args.device),
                       NPU_WAIT_NUM_CARDS="1", NPU_WAIT_POLL_SECONDS="60", NPU_WAIT_IDLE_SAMPLES="1",
                       NPU_WAIT_FINAL_DELAY_SECONDS="20", NPU_WAIT_LOCK_WAIT_SECONDS="-1")
    with (directory / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(["bash", str(args.helper), *worker], env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    return {**job, "index": index, "result": str(output), "ownership": str(ownership),
            "log": str(directory / "run.log"), "exit_code": result.returncode}


def _run(args):
    if not args.helper.is_file():
        raise ValueError("Source-NK acceptance requires the existing readable idle helper")
    args.root = args.root.resolve()
    journal = args.root / "suite.json"
    if journal.exists():
        raise ValueError("Use a new Source-NK suite directory to preserve previous evidence")
    record = {"complete": False, "blocks": args.blocks, "jobs": []}
    _write(journal, record)
    for index, job in enumerate(BASE.suite_jobs(args.blocks)):
        record["jobs"].append(_launch(args, index, job))
        _write(journal, record)
        if record["jobs"][-1]["exit_code"]:
            raise RuntimeError(f"Source-NK suite stopped at job {index}; evidence retained")
    record["complete"] = True
    _write(journal, record)
    _write(args.root / "acceptance.json", analyze_suite(record))


def main() -> None:
    """通过既有空闲门禁启动新进程任务，或重新分析已保留的完整原始证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    run = modes.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--helper", type=Path, required=True)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--blocks", type=int, default=3)
    run.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    analyze = modes.add_parser("analyze")
    analyze.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        if args.device < 0 or args.blocks < 3:
            parser.error("device must be nonnegative and blocks must be at least three")
        args.benchmark_args = args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args
        if any(argument.split("=", 1)[0] in ("--mode", "--backend", "--baseline", "--output")
               for argument in args.benchmark_args):
            parser.error("Source-NK suite owns mode/backend/baseline/output")
        _run(args)
    else:
        suite = json.loads((args.root / "suite.json").read_text(encoding="utf-8"))
        _write(args.root / "acceptance.json", analyze_suite(suite))


if __name__ == "__main__":
    main()
