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
"""Two-process CPU worker; native resource release is represented by a barrier."""

import gc
import os
import signal
from datetime import timedelta
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch.distributed as dist

from hyper_parallel.core.multicore import lifecycle
from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.module import MulticoreModule


class _CPUResources:
    """Model the collective boundary without allocating any device buffers."""

    def __init__(self) -> None:
        """Initialize a ready fake allocation and its release counter."""
        self.ready = True
        self.close_count = 0
        self.runtime_release_count = 0

    @staticmethod
    def lifecycle_signature() -> tuple[str]:
        """Return identical resource metadata on both ranks."""
        return ("cpu_resource",)

    def can_close(self) -> bool:
        """Expose a controllable pending-work flag."""
        return self.ready

    def close(self) -> None:
        """Require both ranks to reach the same simulated symmetric free."""
        dist.barrier()
        self.close_count += 1

    def retain_runtime(self) -> Callable[[], None]:
        """Represent a session reference surviving the final workspace release."""
        return self._release_runtime

    def _release_runtime(self) -> None:
        """Model final runtime shutdown as another collective boundary."""
        dist.barrier()
        self.runtime_release_count += 1


class _CPUModule(MulticoreModule):
    """Use the real resource manager with CPU-only bindings."""

    def __init__(self) -> None:
        """Create a CPU-only owner with identical configuration on both ranks."""
        super().__init__(resource_specification="cpu", resource_compatibility_key="cpu", resource_scope_key="cpu")

    def _create_execution_resources(
        self, tensor: Any, *, shared: bool, active_specifications: tuple[Any, ...],
    ) -> _CPUResources:
        """Bind a fake native resource to the ordinary module ownership path."""
        return _CPUResources()


def test_lifecycle_coordination_worker() -> None:
    """Exercise asymmetric GC, pending work, and SIGTERM delivered to rank zero."""
    manager = module_api._MulticoreResourceManager()  # pylint: disable=protected-access
    closed_resources = []

    @lifecycle.managed_run
    def worker() -> None:
        """Run every control operation collectively before requesting shutdown."""
        dist.init_process_group("gloo", timeout=timedelta(seconds=15))
        rank = dist.get_rank()
        tensor = SimpleNamespace(device="cpu", dtype="float32")
        if rank == 0:
            # Construction-only modules must not change cross-rank allocation identity.
            unused = _CPUModule()
            del unused
        first = _CPUModule()
        first_resources = first._get_execution_resources(tensor)  # pylint: disable=protected-access
        if rank == 0:
            del first
            gc.collect()
        count = lifecycle.collect_resources()
        assert count == 0, f"Live peer must retain resource: expected=0, actual={count}"
        if rank == 1:
            del first
            gc.collect()
        count = lifecycle.collect_resources()
        assert count == 1, f"Orphan must be released: expected=1, actual={count}"
        second = _CPUModule()
        second_resources = second._get_execution_resources(tensor)  # pylint: disable=protected-access
        closed_resources.extend((first_resources, second_resources))
        second_resources.ready = rank != 0
        with pytest.raises(RuntimeError, match="pending backward"):
            lifecycle.shutdown()
        assert second_resources.close_count == 0, (
            f"Pending work must prevent free: expected=0, actual={second_resources.close_count}"
        )
        second_resources.ready = True
        if rank == 0:
            os.kill(os.getpid(), signal.SIGTERM)
        lifecycle.lifecycle_checkpoint()
        pytest.fail("An agreed termination request must leave the worker")

    with patch.object(module_api, "_RESOURCE_MANAGER", manager):
        with pytest.raises(SystemExit) as stopped:
            worker()
    assert stopped.value.code == 128 + signal.SIGTERM, (
        f"Cooperative exit code: expected={128 + signal.SIGTERM}, actual={stopped.value.code}"
    )
    assert not dist.is_initialized(), f"Process group must be destroyed: actual={dist.is_initialized()}"
    counts = [resource.close_count for resource in closed_resources]
    assert counts == [1, 1], f"Resources must close once: expected={[1, 1]}, actual={counts}"
    counts = [resource.runtime_release_count for resource in closed_resources]
    assert counts == [1, 0], f"Session reference must close once: expected={[1, 0]}, actual={counts}"
