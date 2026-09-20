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
"""CPU tests for collective lifecycle decisions and managed worker exits."""

import gc
import os
import signal
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore import lifecycle
from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.module import MulticoreModule

# These tests inspect ownership after failures and emulate remote metadata only.
# pylint: disable=protected-access


class TestMulticoreLifecycle(unittest.TestCase):
    """Check resource ownership independently of device allocation and collectives."""

    def setUp(self) -> None:
        """Install an isolated manager and a single-process communication boundary."""
        self.manager = module_api._MulticoreResourceManager()
        self.enterContext(patch.object(module_api, "_RESOURCE_MANAGER", self.manager))
        self.enterContext(patch.object(module_api.dist, "is_initialized", return_value=False))
        self.enterContext(patch.object(lifecycle, "_requested_signal", 0))
        self.enterContext(patch.object(lifecycle, "_agreed_signal", 0))
        self.tensor = SimpleNamespace(device="cpu", dtype="float32")

    def _bound_module(self) -> tuple[MulticoreModule, Mock]:
        """Bind one fake native owner through the real module path."""
        member = MulticoreModule(
            resource_specification="spec", resource_compatibility_key="key", resource_scope_key="scope",
        )
        resources = Mock()
        resources.lifecycle_signature.return_value = ("test", 128)
        resources.can_close.return_value = True
        with patch.object(member, "_create_execution_resources", return_value=resources):
            member._get_execution_resources(self.tensor)
        return member, resources

    def test_orphan_collection_preserves_live_groups(self) -> None:
        """Release discarded owners without closing an independently live module."""
        orphan, retired = self._bound_module()
        live, active = self._bound_module()
        del orphan
        gc.collect()
        self.assertEqual(lifecycle.collect_resources(), 1)
        retired.close.assert_called_once_with()
        active.close.assert_not_called()
        self.assertIs(live._get_execution_resources(self.tensor), active)
        self.assertEqual(self.manager.active_specifications("scope"), ("spec",))
        lifecycle.shutdown()
        active.close.assert_called_once_with()

    def test_orphan_with_pending_graph_waits_for_later_checkpoint(self) -> None:
        """Defer collection until the final autograd user is finished."""
        member, resources = self._bound_module()
        resources.can_close.return_value = False
        del member
        gc.collect()
        self.assertEqual(lifecycle.collect_resources(), 0)
        resources.close.assert_not_called()
        resources.can_close.return_value = True
        self.assertEqual(lifecycle.collect_resources(), 1)
        resources.close.assert_called_once_with()

    def test_repeated_model_disposal_returns_manager_to_baseline(self) -> None:
        """Repeated build and disposal must not accumulate orphan resource groups."""
        owners = []
        for _ in range(5):
            member, resources = self._bound_module()
            owners.append(resources)
            del member
            gc.collect()
            lifecycle.lifecycle_checkpoint()
            resources.close.assert_called_once_with()
            self.assertFalse(self.manager._groups)
            self.assertEqual(self.manager.active_specifications("scope"), ())
        owners[0].retain_runtime.assert_called_once_with()
        for owner in owners[1:]:
            owner.retain_runtime.assert_not_called()
        owners[0].retain_runtime.return_value.assert_not_called()
        lifecycle.shutdown()
        owners[0].retain_runtime.return_value.assert_called_once_with()
        self.assertIsNone(self.manager._runtime_release)

    def test_shutdown_invalidates_shared_live_members(self) -> None:
        """Close a shared group once and invalidate every surviving module handle."""
        members = [MulticoreModule(
            resource_specification="spec", resource_compatibility_key="key", resource_scope_key="scope",
        ) for _ in range(2)]
        MulticoreModule.share_execution_resources(members)
        resources = Mock()
        resources.lifecycle_signature.return_value = ("test", 128)
        resources.can_close.return_value = True
        with patch.object(members[1], "_create_execution_resources", return_value=resources):
            members[1]._get_execution_resources(self.tensor)
        lifecycle.shutdown()
        lifecycle.shutdown()
        resources.close.assert_called_once_with()
        for member in members:
            with self.assertRaisesRegex(RuntimeError, "closed multicore module"):
                member._get_execution_resources(self.tensor)
            member.close()
        self.assertFalse(self.manager._groups)

    def test_shutdown_failure_keeps_handles_and_process_group(self) -> None:
        """Do not destroy communication or lose handles when native cleanup fails."""
        member, resources = self._bound_module()
        resources.close.side_effect = RuntimeError("native busy")
        with patch.object(lifecycle.dist, "destroy_process_group") as destroy:
            with self.assertRaisesRegex(RuntimeError, "native busy"):
                lifecycle.shutdown(destroy_process_group=True)
            destroy.assert_not_called()
        self.assertIs(member._resource_group.resources, resources)
        with self.assertRaisesRegex(RuntimeError, "closing multicore module"):
            member._get_execution_resources(self.tensor)
        resources.close.side_effect = None
        lifecycle.shutdown()
        self.assertEqual(resources.close.call_count, 2)

    def test_shutdown_rejects_pending_backward_before_any_free(self) -> None:
        """Validate every resource before partially shutting down a model."""
        first, first_resources = self._bound_module()
        second, second_resources = self._bound_module()
        second_resources.can_close.return_value = False
        with self.assertRaisesRegex(RuntimeError, "pending backward"):
            lifecycle.shutdown()
        first_resources.close.assert_not_called()
        second_resources.close.assert_not_called()
        self.assertIs(first._resource_group.resources, first_resources)
        self.assertIs(second._resource_group.resources, second_resources)

    def test_different_rank_manifests_abort_before_free(self) -> None:
        """Refuse different allocation identities before entering native collectives."""
        member, resources = self._bound_module()

        def exchange(value: Any) -> list[Any]:
            """Simulate a peer with a different native allocation manifest."""
            operation, _, readiness, signum = value
            return [value, (operation, ((99, ("different",)),), readiness, signum)]

        with patch.object(self.manager, "_exchange", side_effect=exchange):
            with self.assertRaisesRegex(RuntimeError, "manifests differ"):
                lifecycle.shutdown()
        resources.close.assert_not_called()
        self.assertIs(member._resource_group.resources, resources)

    def test_remote_live_member_prevents_local_orphan_free(self) -> None:
        """Require all ranks to retire an allocation before orphan collection."""
        member, resources = self._bound_module()
        del member
        gc.collect()

        def exchange(value: Any) -> list[Any]:
            """Simulate a peer retaining its module membership."""
            operation, manifest, _, signum = value
            return [value, (operation, manifest, ((False, True),), signum)]

        with patch.object(self.manager, "_exchange", side_effect=exchange):
            self.assertEqual(lifecycle.collect_resources(), 0)
        resources.close.assert_not_called()

    def test_remote_close_failure_prevents_advancing_to_next_group(self) -> None:
        """Keep collective order when a peer reports a failed native close."""
        first, first_resources = self._bound_module()
        second, second_resources = self._bound_module()

        def exchange(value: Any) -> list[Any]:
            """Agree on readiness, then report a remote teardown failure."""
            return [value, "remote close failed"] if value is None else [value, value]

        with patch.object(self.manager, "_exchange", side_effect=exchange):
            with self.assertRaisesRegex(RuntimeError, "remote close failed"):
                lifecycle.shutdown()
        first_resources.close.assert_called_once_with()
        self.assertTrue(first._resource_group.closed)
        second_resources.close.assert_not_called()
        self.assertIs(second._resource_group.resources, second_resources)

    def test_remote_stop_request_reaches_every_checkpoint(self) -> None:
        """A rank with no local signal observes another rank's stop request."""
        def exchange(value: Any) -> list[Any]:
            """Report a termination signal from the remote rank."""
            return [value, (*value[:3], signal.SIGTERM)]

        with patch.object(self.manager, "_exchange", side_effect=exchange):
            with self.assertRaises(SystemExit) as raised:
                lifecycle.lifecycle_checkpoint()
        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)

    def test_late_signal_waits_for_next_collective_agreement(self) -> None:
        """Do not diverge when a signal arrives after this checkpoint's snapshot."""
        def exchange(value: Any) -> list[Any]:
            """Deliver a local signal after the collective snapshot was formed."""
            lifecycle._record_signal(signal.SIGINT, None)
            return [value, value]

        with patch.object(self.manager, "_exchange", side_effect=exchange):
            lifecycle.lifecycle_checkpoint()
        with self.assertRaises(SystemExit):
            lifecycle.lifecycle_checkpoint()

    def test_cleanup_precedes_process_group_destruction(self) -> None:
        """Keep communication available until all native owners have closed."""
        member, resources = self._bound_module()
        events = []
        resources.close.side_effect = lambda: events.append("close")
        with (
            patch.object(module_api.dist, "is_initialized", return_value=True),
            patch.object(self.manager, "_exchange", side_effect=lambda value: [value]),
            patch.object(lifecycle.dist, "destroy_process_group", side_effect=lambda: events.append("destroy")),
        ):
            lifecycle.shutdown(destroy_process_group=True)
        self.assertEqual(events, ["close", "destroy"])
        self.assertTrue(member._resource_group.closed)


class TestManagedRun(unittest.TestCase):
    """Check signal policy ownership without changing the test runner's handlers."""

    def setUp(self) -> None:
        """Isolate signal registration and global worker state."""
        self.enterContext(patch.object(lifecycle.signal, "getsignal", return_value=signal.SIG_DFL))
        self.signal_install = self.enterContext(patch.object(lifecycle.signal, "signal"))
        self.shutdown = self.enterContext(patch.object(lifecycle, "shutdown"))
        self.enterContext(patch.object(lifecycle, "_managed_run_active", False))
        self.enterContext(patch.object(lifecycle, "_requested_signal", 0))
        self.enterContext(patch.object(lifecycle, "_agreed_signal", 0))

    def test_normal_return_closes_and_restores_handlers(self) -> None:
        """Preserve the entry point result while automatically closing resources."""
        @lifecycle.managed_run
        def worker(value: int) -> int:
            """Return a result through the managed entry point."""
            return value + 1

        self.assertEqual(worker(4), 5)
        self.shutdown.assert_called_once_with(destroy_process_group=True)
        self.assertEqual(self.signal_install.call_count, 4)
        self.assertFalse(lifecycle._managed_run_active)
        self.assertEqual(worker.__name__, "worker")

    def test_unexpected_failure_does_not_start_cleanup_collectives(self) -> None:
        """Preserve worker failures for the external launcher to handle."""
        @lifecycle.managed_run
        def worker() -> None:
            """Raise an application failure outside a cooperative safe point."""
            raise RuntimeError("training failed")

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            worker()
        self.shutdown.assert_not_called()
        self.assertEqual(self.signal_install.call_count, 4)

    def test_custom_signal_handler_is_not_overwritten(self) -> None:
        """Leave a host framework's signal policy intact and request callback integration."""
        with patch.object(lifecycle.signal, "getsignal", return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, "custom signal handlers"):
                lifecycle.managed_run(lambda: None)()
        self.signal_install.assert_not_called()

    def test_signal_handler_only_records_request(self) -> None:
        """Keep collective and device cleanup outside asynchronous signal handling."""
        lifecycle._record_signal(signal.SIGTERM, None)
        self.assertEqual(lifecycle._requested_signal, signal.SIGTERM)
        self.shutdown.assert_not_called()


class TestManagedRunSignals(unittest.TestCase):
    """Deliver real signals only to subprocesses owned by these CPU tests."""

    def test_sigint_sigterm_and_sigkill(self) -> None:
        """Catch cooperative signals while confirming SIGKILL cannot run cleanup."""
        worker = textwrap.dedent("""
            import os
            import signal
            from unittest.mock import patch
            from hyper_parallel.core.multicore import lifecycle

            @lifecycle.managed_run
            def main():
                os.kill(os.getpid(), SIGNUM)
                print('checkpoint', flush=True)
                lifecycle.lifecycle_checkpoint()

            with patch.object(lifecycle, 'shutdown', side_effect=lambda **kw: print('closed', flush=True)):
                main()
        """)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signum=signum):
                result = subprocess.run(
                    [sys.executable, "-c", worker.replace("SIGNUM", str(int(signum)))],
                    capture_output=True, text=True, check=False, timeout=45, env=os.environ.copy(),
                )
                expected = -signal.SIGKILL if signum == signal.SIGKILL else 128 + signum
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertEqual("closed" in result.stdout, signum != signal.SIGKILL)
