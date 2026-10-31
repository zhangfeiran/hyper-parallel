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
"""Order independent workspaces sharing the process-wide UDMA QP bank."""
import os
import threading
from typing import Any


class SubmissionOrder:
    """Keep one host submitter and order device submissions across streams."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.stream = None
        self.completion = None

    def claim(self, stream: Any) -> None:
        """Order the next submission after the preceding workspace's stream."""
        with self.lock:
            if self.active:
                raise RuntimeError('UDMA QPs do not support concurrent workspace submissions')
            if self.completion is not None and self.stream != stream.npu_stream:
                stream.wait_event(self.completion)
            self.active = True

    def release(self, stream: Any, completion: Any) -> None:
        """Retain the completion event before admitting another workspace."""
        with self.lock:
            if not self.active:
                raise RuntimeError('UDMA QPs released without an active submission')
            self.stream = stream.npu_stream
            self.completion = completion
            self.active = False


def validate_transport() -> bool:
    """Reserve shared QP ordering only for the opt-in hybrid data engine."""
    engine = os.environ.get('HYPER_PARALLEL_SHMEM_DATA_ENGINE', 'mte')
    if engine == 'mte':
        return False
    if engine != 'mte_udma':
        raise ValueError('MegaMoE transport must use mte or mte_udma')
    return True


SUBMISSION_ORDER = SubmissionOrder()
