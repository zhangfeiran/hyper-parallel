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
"""CPU checks for rank-local verification failures and collective step agreement."""

import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.examples import mega_dsa_validation_control as control


class TestCollectiveValidation(unittest.TestCase):
    """A failing rank cannot leave peers to enter another forward or a different collective."""

    def setUp(self) -> None:
        """Mock the verification process group without initializing distributed hardware."""
        mocks = (patch.object(control.dist, "get_rank", return_value=0),
                 patch.object(control.dist, "get_world_size", return_value=2))
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)

    def test_success_preserves_evidence_after_peer_agreement(self):
        """Successful local evidence advances only when every peer reports success."""
        evidence = {"key_conversion_bits_exact": True}
        with patch.object(control.dist, "all_gather_object") as gather:
            actual = control.collective_validation_check(lambda: evidence)
        self.assertIs(actual, evidence)
        gather.assert_called_once()
        self.assertIsNone(gather.call_args.args[1])

    def test_local_failure_is_gathered_before_raising(self):
        """A local assertion becomes a common failure before any caller enters teardown."""
        def _check():
            raise AssertionError("conversion mismatch")
        def _gather(peers, failure, **_kwargs):
            peers[:] = [failure, None]
        with patch.object(control.dist, "all_gather_object", side_effect=_gather) as gather:
            with self.assertRaisesRegex(RuntimeError, "conversion mismatch"):
                control.collective_validation_check(_check)
        failure = gather.call_args.args[1]
        self.assertEqual(failure["rank"], 0)
        self.assertIn("AssertionError", failure["traceback"])

    def test_remote_failure_prevents_successful_rank_from_advancing(self):
        """A healthy rank receives the same failing verdict as its peer."""
        def _gather(peers, _failure, **_kwargs):
            peers[:] = [None, {"rank": 1, "error": "pointwise mismatch"}]
        with patch.object(control.dist, "all_gather_object", side_effect=_gather), \
                self.assertRaisesRegex(RuntimeError, "pointwise mismatch"):
            control.collective_validation_check(lambda: {"passed": True})
