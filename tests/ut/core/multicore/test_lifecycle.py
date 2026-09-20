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
"""CPU coverage for scoped and explicit collective resource cleanup."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore import lifecycle
from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.module import MulticoreModule

# Inspect private ownership to verify failure recovery without native allocations.
# pylint: disable=protected-access


class TestMulticoreLifecycle(unittest.TestCase):
    """Exercise real membership with mocked native resources and communication."""

    def setUp(self) -> None:
        """Isolate ownership and mock distributed initialization."""
        self.manager = module_api._MulticoreResourceManager()
        self.enterContext(patch.object(module_api, "_RESOURCE_MANAGER", self.manager))
        self.enterContext(patch.object(module_api.dist, "is_initialized", return_value=False))
        self.tensor = SimpleNamespace(device="cpu", dtype="float32")

    def _module(self) -> MulticoreModule:
        """Create an unbound module in the current scope."""
        return MulticoreModule(
            resource_specification="spec", resource_compatibility_key="key", resource_scope_key="scope",
        )

    def _bind(self, member: MulticoreModule) -> Mock:
        """Use ordinary lazy binding with a fake native resource."""
        resources = Mock()
        resources.lifecycle_signature.return_value = ("test", 128)
        resources.can_close.return_value = True
        with patch.object(member, "_create_execution_resources", return_value=resources):
            member._get_execution_resources(self.tensor)
        return resources

    def test_shutdown_closes_shared_and_orphan_groups_once(self) -> None:
        """Explicit cleanup needs neither live owners nor per-layer close calls."""
        members = [self._module(), self._module()]
        MulticoreModule.share_execution_resources(members)
        shared = self._bind(members[1])
        orphan = self._module()
        retired = self._bind(orphan)
        del orphan
        lifecycle.shutdown()
        lifecycle.shutdown()
        shared.close.assert_called_once_with()
        retired.close.assert_called_once_with()
        for member in members:
            with self.assertRaisesRegex(RuntimeError, "closed multicore module"):
                member._get_execution_resources(self.tensor)
            member.close()
        self.assertFalse(self.manager._groups)

    def test_failed_shutdown_can_retry_without_losing_handles(self) -> None:
        """Native failure preserves handles and disables further execution."""
        member = self._module()
        resources = self._bind(member)
        resources.close.side_effect = RuntimeError("native busy")
        with self.assertRaisesRegex(RuntimeError, "native busy"):
            lifecycle.shutdown()
        self.assertIs(member._resource_group.resources, resources)
        with self.assertRaisesRegex(RuntimeError, "closing multicore module"):
            member._get_execution_resources(self.tensor)
        resources.close.side_effect = None
        lifecycle.shutdown()
        self.assertEqual(resources.close.call_count, 2)

    def test_preflight_rejects_pending_work_and_remote_mismatch(self) -> None:
        """Reject unsafe cleanup before entering any native free."""
        member = self._module()
        resources = self._bind(member)
        resources.can_close.return_value = False
        with self.assertRaisesRegex(RuntimeError, "pending backward"):
            lifecycle.shutdown()
        resources.can_close.return_value = True
        with patch.object(self.manager, "_exchange", return_value=[(("different",), True)]):
            with self.assertRaisesRegex(RuntimeError, "manifests differ"):
                lifecycle.shutdown()
        resources.close.assert_not_called()

    def test_remote_close_failure_stops_before_next_group(self) -> None:
        """A returned remote error must prevent advancing native collective order."""
        members = [self._module(), self._module()]
        first, second = [self._bind(member) for member in members]
        with patch.object(
            self.manager, "_exchange",
            side_effect=lambda value: [value, "remote failed"] if value is None else [value, value],
        ):
            with self.assertRaisesRegex(RuntimeError, "remote failed"):
                lifecycle.shutdown()
        first.close.assert_called_once_with()
        second.close.assert_not_called()

    def test_nested_scopes_preserve_outer_resources(self) -> None:
        """Returning modules are closed, while outer owners survive inner scopes."""
        outside = self._module()
        outside_resources = self._bind(outside)

        @lifecycle.managed_run
        def inner() -> MulticoreModule:
            """Create one module whose ownership ends on return."""
            member = self._module()
            self._bind(member)
            return member

        @lifecycle.managed_run
        def outer() -> Mock:
            """Check nested cleanup without destroying communication."""
            member = self._module()
            resources = self._bind(member)
            returned = inner()
            self.assertTrue(returned._resource_group.closed)
            resources.close.assert_not_called()
            return resources

        with patch.object(module_api.dist, "destroy_process_group") as destroy:
            outer_resources = outer()
            destroy.assert_not_called()
        outer_resources.close.assert_called_once_with()
        outside_resources.close.assert_not_called()
        self.assertIsNone(self.manager.owner)
        self.assertEqual(outer.__name__, "outer")
        lifecycle.shutdown()

    def test_scope_exception_skips_collectives_and_preserves_ownership(self) -> None:
        """Do not start collective cleanup on arbitrary exceptions or interrupts."""
        for error in (RuntimeError("task failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                @lifecycle.managed_run
                def task() -> None:
                    """Leave an orphan behind when application execution fails."""
                    self._bind(self._module())
                    raise error

                with patch.object(self.manager, "_exchange") as exchange:
                    with self.assertRaises(type(error)):
                        task()
                    exchange.assert_not_called()
                self.assertIsNone(self.manager.owner)
                self.assertTrue(self.manager._groups)
                lifecycle.shutdown()

    def test_cross_scope_sharing_is_rejected(self) -> None:
        """An inner scope must not acquire ownership of an outer module."""
        outside = self._module()

        @lifecycle.managed_run
        def task() -> None:
            """Reject both possible orders of cross-scope sharing."""
            inside = self._module()
            for members in ([outside, inside], [inside, outside]):
                with self.assertRaisesRegex(ValueError, "across managed scopes"):
                    MulticoreModule.share_execution_resources(members)

        task()
        self.assertFalse(outside._resource_group.closed)
        lifecycle.shutdown()
