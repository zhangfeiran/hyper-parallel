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
"""Run customer LongCat precision diagnosis with explicitly reserved NPU devices."""

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from diagnose_reports import summarize_stage, write_summary


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
_WORKERS = Path("tests/torch/models/longcat_flash")
_ORACLE = "ca76a20089d03dc0d9f9f7e8852c93931c5396ef"


def parse_devices(value: str, stage: str) -> list[str]:
    """Validate explicit logical device IDs without selecting devices automatically.

    Args:
        value: Comma-separated logical device IDs already reserved by the caller.
        stage: Selected diagnostic stage.
    """
    parts = [part.strip() for part in value.split(",")]
    if any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("--devices requires comma-separated nonnegative logical NPU IDs")
    devices = [str(int(part)) for part in parts]
    count = 1 if stage == "ep1" else 4
    if len(devices) != count or len(set(devices)) != count:
        raise ValueError(f"--stage {stage} requires exactly {count} distinct reserved logical NPU IDs")
    return devices


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True,
                        help="Explicitly reserved logical NPU IDs; one for ep1, four otherwise")
    parser.add_argument("--output", type=Path, required=True, help="New output directory outside the repository")
    parser.add_argument("--cann-env", type=Path, required=True, help="Customer CANN set_env.sh")
    parser.add_argument("--native-env", type=Path,
                        default=REPO / "build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash")
    parser.add_argument("--stage", choices=("all", "ep1", "ep4"), default="all")
    parser.add_argument("--ep4-hf", choices=("on-failure", "always", "never"), default="on-failure")
    parser.add_argument("--build", action="store_true",
                        help="Build both supported targets and install this repo editable, without dependencies")
    args = parser.parse_args()
    try:
        args.devices = parse_devices(args.devices, args.stage)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.expanduser().resolve()
    if args.output.is_relative_to(REPO) or args.output.exists():
        parser.error("--output must be a new directory outside this repository")
    args.cann_env, args.native_env = args.cann_env.expanduser().resolve(), args.native_env.expanduser().resolve()
    return args


def _run(command: list[str], env: dict, log: Path) -> int:
    """Retain exact argv and stop only this child process group on interruption."""
    print(f"执行：{shlex.join(command)}\n日志：{log}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                              start_new_session=True) as process:
            try:
                return process.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise


def _source_environment(path: Path, env: dict, log: Path) -> dict:
    """Source a reviewed environment script using positional arguments, not interpolation."""
    if not path.is_file():
        raise FileNotFoundError(f"Environment script missing: {path}")
    script = 'set -e\nsource "$1" >>"$2" 2>&1\nenv -0'
    result = subprocess.run(["bash", "-c", script, "longcat-environment", str(path), str(log)],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        with log.open("ab") as stream:
            stream.write(result.stderr)
        raise RuntimeError(f"Environment activation failed: {path}; inspect {log.name}")
    return dict(os.fsdecode(item).split("=", 1) for item in result.stdout.split(b"\0") if b"=" in item)


def _prepare(args: argparse.Namespace, receipt: dict) -> dict:
    """Record source identity, optionally build, then verify the actual customer runtime."""
    receipt["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    receipt["source_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    oracle = subprocess.run(["git", "cat-file", "-e",
                             f"{_ORACLE}:hyper_parallel/models/longcat_flash/adapter/route_kernels.py"],
                            cwd=REPO, capture_output=True, check=False)
    if oracle.returncode:
        raise RuntimeError("Frozen reference is missing; fetch complete Git history before diagnosis")
    receipt["launcher_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in ROOT.glob("diagnose*.py")}
    receipt["worker_sha256"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (REPO / _WORKERS).glob("*.py") if path.name.startswith("_")}
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env = _source_environment(args.cann_env, env, args.output / "prepare.log")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if args.build:
        commands = [["bash", "build.sh", "--multicore", "on", "--soc-list",
                     "ascend910b,ascend910_93", "--strict", "on"],
                    [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(REPO)]]
        for command in commands:
            if _run(command, env, args.output / "build.log"):
                raise RuntimeError("Build/editable installation failed; inspect build.log")
    env = _source_environment(args.native_env, env, args.output / "prepare.log")
    env.update(HCCL_DETERMINISTIC="true", HCCL_HOST_SOCKET_PORT_RANGE="auto", OMP_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               ASCEND_RT_VISIBLE_DEVICES=",".join(args.devices))
    for command in ([sys.executable, str(ROOT / "checksums.py")],
                    [sys.executable, str(ROOT / "diagnose_probe.py"), str(REPO),
                     str(args.output / "environment.json"), str(len(args.devices))]):
        if _run(command, env, args.output / "preflight.log"):
            raise RuntimeError("Runtime preflight failed; inspect preflight.log and environment.json")
    return env


def _stage_command(stage: str, output: Path) -> list[str]:
    """Keep original widths, token count, TopK and per-rank expert cardinality."""
    semantic = stage == "ep1"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={1 if semantic else 4}", str(_WORKERS / (
                   "_diagnose_longcat_swiglu.py" if semantic else "_validate_longcat_route_migration.py"))]
    if not semantic:
        command.extend(["--check", "component"])
    return command + ["--backend", "push", "--ep", "1" if semantic else "4", "--tokens", "2048",
                      "--real", "8" if semantic else "32", "--zero", "4" if semantic else "16",
                      "--hidden", "6144", "--intermediate", "2048", "--top-k", "12",
                      "--routing-backend", "triton", "--swiglu-precision", "hf" if stage == "ep4-hf" else "fused",
                      "--initial-token-capacity", "24576", "--output", str(output / stage)]


def _execute(args: argparse.Namespace, receipt: dict, env: dict, stage: str) -> dict:
    """Run one fresh worker group, then require complete rank evidence."""
    current = env.copy()
    current["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(args.devices[:1] if stage == "ep1" else args.devices)
    command = _stage_command(stage, args.output)
    code = _run(command, current, args.output / f"{stage}.log")
    result = summarize_stage(args.output, stage, code)
    result["command"] = command
    receipt["stages"].append(result)
    write_summary(args.output, receipt)
    print(f"{stage}: {result['status']}", flush=True)
    return result


def main() -> int:
    """Execute requested diagnosis and leave an explicit final receipt on failures."""
    args = _arguments()
    args.output.mkdir(parents=True)
    receipt = {"stage_selection": args.stage, "devices": args.devices, "ep4_hf": args.ep4_hf,
               "stages": [], "error": None, "complete": False,
               "scope": "Synthetic expert components; customer project acceptance is separate"}
    interrupted = False
    try:
        env = _prepare(args, receipt)
        if args.stage in ("all", "ep1"):
            result = _execute(args, receipt, env, "ep1")
            if result["status"] == "ERROR":
                raise RuntimeError("EP1 did not finish with complete valid reports; EP4 was not started")
        if args.stage in ("all", "ep4"):
            result = _execute(args, receipt, env, "ep4-fused")
            if result["status"] != "ERROR" and (args.ep4_hf == "always" or
                                                (args.ep4_hf == "on-failure" and result["status"] == "NUMERICAL_FAIL")):
                _execute(args, receipt, env, "ep4-hf")
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    except KeyboardInterrupt:
        receipt["error"], interrupted = "Interrupted; this invocation's active worker group was stopped", True
        receipt["interrupted"] = True
    receipt["complete"] = True
    code = write_summary(args.output, receipt)
    print(f"诊断结束：{receipt['status']}\n报告：{args.output / 'summary.txt'}", flush=True)
    return 130 if interrupted else code


def _on_terminate(_signum: int, _frame: object) -> None:
    """Route termination through the same owned-child cleanup as Ctrl-C."""
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_terminate)
    sys.exit(main())
