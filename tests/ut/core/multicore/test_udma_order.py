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
"""Validate submission ordering for the process-wide UDMA QP bank."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hyper_parallel.core.multicore.modules.mega_moe.udma_order import SubmissionOrder, validate_transport
from hyper_parallel.core.multicore.modules.mega_moe import workspace as workspace_module


class TestSubmissionOrder(unittest.TestCase):
    """Exercise cross-workspace ordering without initializing an NPU."""

    def test_same_stream_keeps_its_natural_order(self) -> None:
        """Avoid a redundant event wait on the stream that recorded completion."""
        order = SubmissionOrder()
        stream = SimpleNamespace(npu_stream=12, wait_event=Mock())
        completion = object()

        order.claim(stream)
        order.release(stream, completion)
        order.claim(stream)

        stream.wait_event.assert_not_called()
        order.release(stream, object())

    def test_other_stream_waits_for_previous_workspace(self) -> None:
        """Keep the previous event alive and place the wait before the next submit."""
        order = SubmissionOrder()
        first = SimpleNamespace(npu_stream=12, wait_event=Mock())
        second = SimpleNamespace(npu_stream=19, wait_event=Mock())
        completion = object()

        order.claim(first)
        order.release(first, completion)
        order.claim(second)

        second.wait_event.assert_called_once_with(completion)
        order.release(second, object())

    def test_concurrent_host_submission_is_rejected(self) -> None:
        """Prevent two workspaces from concurrently writing the same QP bank."""
        order = SubmissionOrder()
        stream = SimpleNamespace(npu_stream=12, wait_event=Mock())
        order.claim(stream)

        with self.assertRaisesRegex(RuntimeError, 'concurrent workspace submissions'):
            order.claim(stream)

        order.release(stream, object())

    def test_release_requires_an_active_submission(self) -> None:
        """Reject an unmatched release instead of replacing the last completion."""
        order = SubmissionOrder()

        with self.assertRaisesRegex(RuntimeError, 'without an active submission'):
            order.release(SimpleNamespace(npu_stream=12), object())

    def test_default_mte_keeps_workspace_local_ordering(self) -> None:
        """Keep the legacy MTE runtime outside the process-wide QP lease."""
        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte"}):
            self.assertFalse(validate_transport())

        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma"}):
            self.assertTrue(validate_transport())

    def test_unknown_engine_is_rejected_before_workspace_creation(self) -> None:
        """Reject malformed engine names before any native task submission."""
        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "udma"}):
            with self.assertRaisesRegex(ValueError, "mte or mte_udma"):
                validate_transport()

    def test_workspace_uses_the_qp_lease_only_for_hybrid_transport(self) -> None:
        """Keep legacy MTE submissions unchanged while ordering shared hybrid QPs."""
        for engine in ('mte', 'mte_udma'):
            with self.subTest(engine=engine):
                stream = SimpleNamespace(npu_stream=12, wait_event=Mock())
                completion = Mock()
                framework = SimpleNamespace(npu=SimpleNamespace(current_stream=Mock(return_value=stream)))
                with (patch.dict(os.environ, {'HYPER_PARALLEL_SHMEM_DATA_ENGINE': engine}),
                      patch.object(workspace_module, 'torch', framework),
                      patch.object(workspace_module, 'SUBMISSION_ORDER') as order):
                    workspace = workspace_module.MegaMoeWorkspace(shared=False, completion_event=completion)
                    workspace.claim()
                    workspace.release()

                completion.record.assert_called_once_with(stream)
                if engine == 'mte_udma':
                    order.claim.assert_called_once_with(stream)
                    order.release.assert_called_once_with(stream, completion)
                else:
                    order.claim.assert_not_called()
                    order.release.assert_not_called()
