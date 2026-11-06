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
"""Validate new routes against native GMM using a frozen historical route frontend."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch

from hyper_parallel.models.longcat_flash.adapter import fused_routing
from tests.torch.models.longcat_flash import _compare_longcat_flash, _validate_longcat_replicas


_DEFAULT_ORACLE = "ca76a20089d03dc0d9f9f7e8852c93931c5396ef"
_ROUTE_FILE = "hyper_parallel/models/longcat_flash/adapter/route_kernels.py"


@contextmanager
def _frozen_frontend(revision: str) -> Iterator[tuple[Any, dict[str, str]]]:
    """Load the historical source from this repository without changing the working tree.

    Args:
        revision: Historical Git commit containing the independent route frontend.
    """
    root = Path(__file__).resolve().parents[4]
    command = ["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"]
    resolved = subprocess.check_output(command).decode().strip()
    source = subprocess.check_output(["git", "-C", str(root), "show", f"{resolved}:{_ROUTE_FILE}"])
    digest = hashlib.sha256(source).hexdigest()
    if digest == hashlib.sha256((root / _ROUTE_FILE).read_bytes()).hexdigest():
        raise ValueError("The historical frontend must differ from the candidate frontend")
    name = f"_longcat_frozen_frontend_{digest}"
    with tempfile.TemporaryDirectory(prefix="longcat-route-oracle-") as directory:
        path = Path(directory) / "route_kernels.py"
        path.write_bytes(source)
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            raise RuntimeError("Cannot load the frozen route frontend")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        try:
            specification.loader.exec_module(module)
            yield module, {"revision": resolved, "sha256": digest, "applies_to": "native reference only"}
        finally:
            sys.modules.pop(name, None)


def _annotate_report(path: Path, oracle: dict[str, str]) -> None:
    """Retain oracle and actual binary provenance even when numerical checks fail.

    Args:
        path: This rank's completed or failed numerical report.
        oracle: Historical source revision and digest used by the native reference.
    """
    if not path.is_file():
        return
    root = Path(__file__).resolve().parents[4]
    loaded = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                     if "/hyper_parallel/core/multicore/" in line and line.split()[-1].startswith("/")})
    correct = bool(loaded) and all(Path(item).resolve().is_relative_to(root) for item in loaded)
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(reference_frontend=oracle,
                  candidate_frontend_sha256=hashlib.sha256((root / _ROUTE_FILE).read_bytes()).hexdigest(),
                  loaded_native_libraries=loaded, native_root_matches=correct)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not correct:
        raise RuntimeError(f"Foreign or missing native libraries: {loaded}")


def main() -> None:
    """Run component or replica precision validation with an independent route oracle."""
    root = Path(__file__).resolve().parents[4]
    if Path(fused_routing.__file__).resolve() != root / _ROUTE_FILE.replace("route_kernels.py", "fused_routing.py"):
        raise RuntimeError("Editable installation must point to this checkout")
    private = argparse.ArgumentParser(description=__doc__)
    private.add_argument("--oracle-revision", default=_DEFAULT_ORACLE)
    private.add_argument("--check", choices=("component", "replicas"), default="component")
    private.add_argument("--output", type=Path, required=True)
    private.add_argument("--routing-backend", choices=("triton",), default="triton")
    options, remaining = private.parse_known_args()
    worker = _compare_longcat_flash if options.check == "component" else _validate_longcat_replicas
    sys.argv = [sys.argv[0], *remaining, "--output", str(options.output), "--routing-backend", "triton"]
    report_path = options.output / f"rank{int(os.environ['LOCAL_RANK'])}.json"
    if report_path.exists():
        raise FileExistsError(f"Use a fresh output directory: {report_path}")
    original_step = worker._step  # pylint: disable=protected-access
    with _frozen_frontend(options.oracle_revision) as (reference, provenance):
        @wraps(original_step)
        def routed_step(experts: Any, *args: Any, **kwargs: Any) -> Any:
            """Use historical routing only for the independently executed native reference.

            Args:
                experts: Expert module identifying native versus candidate execution.
                *args: Original step operands.
                **kwargs: Original step options.
            """
            if experts.backend == "native":
                with (patch.object(fused_routing, "OrderedGather", reference.OrderedGather),
                      patch.object(fused_routing, "OrderedCombine", reference.OrderedCombine)):
                    return original_step(experts, *args, **kwargs)
            return original_step(experts, *args, **kwargs)

        try:
            with patch.object(worker, "_step", routed_step):
                worker.main()
        finally:
            _annotate_report(report_path, provenance)


if __name__ == "__main__":
    main()
