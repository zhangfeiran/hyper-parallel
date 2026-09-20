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
"""Tests for exception-safe ownership of multicore execution resources."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.module import MulticoreModule

# Ownership and finalizer invariants require inspecting the private manager state.
# pylint: disable=protected-access


class TestMulticoreModuleClose(unittest.TestCase):
    """Exercise public close calls with an isolated process resource manager."""

    def setUp(self) -> None:
        """Isolate process ownership without allocating accelerator resources."""
        self.manager = module_api._MulticoreResourceManager()
        manager_patch = patch.object(module_api, "_RESOURCE_MANAGER", self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.tensor = SimpleNamespace(device="cpu", dtype="float32")

    @staticmethod
    def _module() -> MulticoreModule:
        """Create an unbound member with a common resource specification."""
        return MulticoreModule(
            resource_specification="spec",
            resource_compatibility_key="compatible",
            resource_scope_key="scope",
        )

    def test_failed_last_close_preserves_ownership_and_can_retry(self) -> None:
        """Retry standalone and shared teardown after an exception or interrupt."""
        for member_count in (1, 2):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(member_count=member_count, error_type=error_type):
                    members = [self._module() for _ in range(member_count)]
                    MulticoreModule.share_execution_resources(members)
                    last = members[-1]
                    resources = Mock()
                    resources.close.side_effect = error_type("teardown interrupted")
                    with patch.object(last, "_create_execution_resources", return_value=resources):
                        last._get_execution_resources(self.tensor)
                    group = last._resource_group
                    binding = group.binding
                    for member in members[:-1]:
                        member.close()
                    resources.close.assert_not_called()

                    with self.assertRaisesRegex(error_type, "teardown interrupted"):
                        last.close()

                    self.assertFalse(last._resource_closed)
                    self.assertTrue(last._resource_finalizer.alive)
                    self.assertIs(self.manager._groups[group.identifier], group)
                    self.assertIs(group.resources, resources)
                    self.assertEqual(group.binding, binding)
                    self.assertEqual(group.members, {last._resource_member_token})
                    self.assertEqual(self.manager.active_specifications("scope"), ("spec",))
                    with self.assertRaisesRegex(RuntimeError, "closing multicore module"):
                        last._get_execution_resources(self.tensor)
                    with self.assertRaisesRegex(RuntimeError, "closing multicore modules"):
                        MulticoreModule.share_execution_resources([last])

                    resources.close.side_effect = None
                    last.close()
                    last.close()
                    self.assertEqual(resources.close.call_count, 2)
                    self.assertTrue(last._resource_closed)
                    self.assertFalse(last._resource_finalizer.alive)
                    self.assertIsNone(group.resources)
                    self.assertIsNone(group.binding)
                    self.assertFalse(group.members)
                    self.assertFalse(self.manager._groups)

    def test_gc_after_failed_close_preserves_native_handles(self) -> None:
        """Keep failed resources registered when their last module is discarded."""
        member = self._module()
        resources = Mock()
        resources.close.side_effect = RuntimeError("workspace busy")
        with patch.object(member, "_create_execution_resources", return_value=resources):
            member._get_execution_resources(self.tensor)
        group_id = member._resource_group.identifier
        with self.assertRaisesRegex(RuntimeError, "workspace busy"):
            member.close()

        resources.close.side_effect = None
        member_ref = weakref.ref(member)
        resource_ref = weakref.ref(resources)
        del member, resources
        gc.collect()

        self.assertIsNone(member_ref())
        group = self.manager._groups[group_id]
        self.assertFalse(group.members)
        self.assertIs(group.resources, resource_ref())
        group.resources.close.assert_called_once_with()
        self.assertEqual(self.manager.active_specifications("scope"), ("spec",))

    def test_unbound_close_is_idempotent(self) -> None:
        """Remove an unbound member without constructing native resources."""
        member = self._module()
        member.close()
        member.close()
        self.assertTrue(member._resource_closed)
        self.assertFalse(member._resource_finalizer.alive)
        self.assertFalse(self.manager._groups)
