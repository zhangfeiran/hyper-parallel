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
"""Bounded functional probe for multiple torchrun ranks sharing NPU devices.

This does not run a persistent MegaMoe kernel or produce performance numbers.
The parent stays framework-free, records worker progress, and owns the entire
launcher process group so a timeout also cleans up its workers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse physical device assignment and independent communication probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--backend", choices=["gloo", "hccl"], default="hccl")
    parser.add_argument(
        "--shmem-init", action="store_true", help="Also test SHMEM allocation/barrier; requires built adapter"
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("output/megamoe-rank-probe"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _process_start(pid: int) -> str | None:
    """Identify a Linux process without confusing a recycled PID for a worker."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rpartition(")")[2].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _cleanup_workers(directory: Path) -> None:
    """Stop recorded workers that survived torchrun's graceful termination.

    Torchrun can create separate worker process groups. A worker's recorded
    Linux start time guards PID reuse before sending it a final kill signal.
    """
    for path in directory.glob("rank*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        pid, start = record.get("pid"), record.get("process_start")
        if pid is not None and start is not None and _process_start(pid) == start:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _worker(args: argparse.Namespace) -> None:
    # Optional frameworks are loaded only inside an explicitly launched worker.
    torch = importlib.import_module("torch")
    distributed = importlib.import_module("torch.distributed")
    rank = int(os.environ["RANK"])
    device_index = int(os.environ["LOCAL_RANK"]) % len(args.devices)
    record = {
        "pid": os.getpid(),
        "process_start": _process_start(os.getpid()),
        "torch_version": torch.__version__,
        "rank": rank,
        "backend": args.backend,
        "logical_device": device_index,
        "physical_device": args.devices[device_index] if args.backend == "hccl" else None,
        "status": "starting",
        "completed_stages": [],
    }
    path = args.output / f"rank{rank}.json"
    _save(path, record)
    try:
        if args.backend == "hccl":
            importlib.import_module("torch_npu")
            torch.npu.set_device(device_index)
            device = f"npu:{device_index}"
        else:
            device = "cpu"
        distributed.init_process_group(args.backend, timeout=timedelta(seconds=max(10, args.timeout // 2)))
        record["completed_stages"].append("process_group_init")
        _save(path, record)
        tensor = torch.tensor([rank + 1.0], device=device)
        distributed.all_reduce(tensor)
        if tensor.item() != args.ranks * (args.ranks + 1) / 2:
            raise RuntimeError("all_reduce result does not match rank sum.")
        record["completed_stages"].append("all_reduce")
        _save(path, record)
        if args.ranks > 1:
            send = torch.tensor([rank + 1.0], device=device)
            receive = torch.empty_like(send)
            requests = distributed.batch_isend_irecv(
                [
                    distributed.P2POp(distributed.isend, send, (rank + 1) % args.ranks),
                    distributed.P2POp(distributed.irecv, receive, (rank - 1) % args.ranks),
                ]
            )
            for request in requests:
                request.wait()
            if receive.item() != (rank - 1) % args.ranks + 1:
                raise RuntimeError("P2P ring received data from the wrong rank.")
        record["completed_stages"].append("p2p_ring")
        _save(path, record)
        if args.shmem_init:
            module = importlib.import_module("hyper_parallel.platform.torch.symmetric_memory.lifecycle")
            owner = module.acquire_symmetric_memory()
            tensor = owner.empty((16,), torch.float32)
            tensor.fill_(rank)
            torch.npu.synchronize()
            owner.barrier()
            owner.free(tensor)
            owner.close()
            record["completed_stages"].append("shmem_allocate_barrier")
        distributed.destroy_process_group()
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(path, record)
        raise
    _save(path, record)


def launch(args: argparse.Namespace) -> dict[str, Any]:
    """Run a small probe with a deadline and report each completed stage."""
    if args.ranks < 1 or args.timeout < 10 or any(device < 0 for device in args.devices):
        raise ValueError("Ranks must be positive, timeout >= 10 seconds and device IDs nonnegative.")
    if len(set(args.devices)) != len(args.devices):
        raise ValueError("List each physical device once; ranks are mapped modulo this list.")
    if args.shmem_init and args.backend != "hccl":
        raise ValueError("--shmem-init requires --backend hccl.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob("rank*.json")):
        raise ValueError("Use a fresh output directory so stale rank results cannot be reported.")
    environment = dict(os.environ)
    environment.update(
        ASCEND_RT_VISIBLE_DEVICES=",".join(map(str, args.devices)),
        HYPER_PARALLEL_PLATFORM="torch",
        OMP_NUM_THREADS="1",
        HCCL_CONNECT_TIMEOUT=str(min(7200, max(120, args.timeout))),
        HCCL_EXEC_TIMEOUT=str(min(7200, max(120, args.timeout))),
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        shmem_port = reservation.getsockname()[1]
    environment.setdefault("SHMEM_IP_PORT", f"tcp://127.0.0.1:{shmem_port}")
    environment.setdefault("SYMMETRIC_MEMORY_HEAP_SIZE", str(64 * 1024 * 1024))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.ranks}",
        "-m",
        "examples.torch.multicore.probe_mega_moe_ranks",
        "--worker",
        "--backend",
        args.backend,
        "--ranks",
        str(args.ranks),
        "--devices",
        *map(str, args.devices),
        "--timeout",
        str(args.timeout),
        "--output",
        str(args.output),
    ]
    if args.shmem_init:
        command.append("--shmem-init")
    timed_out = False
    with (
        (args.output / "launcher.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as proc,
    ):
        try:
            returncode = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait(timeout=5)
            _cleanup_workers(args.output)
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.output.glob("rank*.json"))]
    passed = returncode == 0 and len(workers) == args.ranks and all(item["status"] == "passed" for item in workers)
    summary = {
        "evidence_kind": "functional_probe",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": passed,
        "timed_out": timed_out,
        "returncode": returncode,
        "workers": workers,
        "command": command,
        "shmem_endpoint": environment["SHMEM_IP_PORT"],
        "scope": "process group, scalar collective, P2P ring; optional SHMEM allocation/barrier; no MegaMoe kernel",
    }
    _save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        _worker(arguments)
    else:
        sys.exit(0 if launch(arguments)["passed"] else 1)
