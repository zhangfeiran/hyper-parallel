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
"""Worker-only CP training composition with native LI and selected KL."""

import json
import os
import tempfile
import traceback
from pathlib import Path

from hyper_parallel.core.multicore.examples.mega_dsa_training_cp_validate import run_validation


def _validate(*, long_history: bool, kv_transfer: str = "full", zero_kl: bool = False) -> None:
    with tempfile.TemporaryDirectory(prefix="mega-dsa-training-") as directory:
        path = Path(os.environ.get("HP_DSA_TRAIN_CP_OUTPUT", directory))
        report = {"status": "running", "scope": "P4 native selection and selected KL training"}
        try:
            run_validation(report, path, long_history=long_history, kv_transfer=kv_transfer, zero_kl=zero_kl)
        except Exception as error:
            report.update(status="error", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank{os.environ.get('RANK', '0')}.json").write_text(json.dumps(report, indent=2) + "\n")


def test_mega_dsa_training_cp() -> None:
    """Verify native-selected LM/KL objectives and all seven owner-local derivatives."""
    _validate(long_history=False)


def test_mega_dsa_training_long_cp() -> None:
    """Verify truncated K=2048 training, packed boundaries and checkpoint owner return."""
    _validate(long_history=True)


def test_mega_dsa_selected_training_cp() -> None:
    """Verify actual device-generated main-KV requests/counts and all seven derivatives."""
    _validate(long_history=False, kv_transfer="selected")


def test_mega_dsa_selected_training_long_cp() -> None:
    """Verify request/count generation from genuine truncated native TopK."""
    _validate(long_history=True, kv_transfer="selected")


def test_mega_dsa_selected_zero_kl_cp() -> None:
    """Verify selected zero-KL phases, owner-local zeros and checkpoint/retained backward."""
    _validate(long_history=False, kv_transfer="selected", zero_kl=True)


def test_mega_dsa_selected_zero_kl_long_cp() -> None:
    """Verify genuine truncated selected zero-KL BF16/FP32 execution with empty owners."""
    _validate(long_history=True, kv_transfer="selected", zero_kl=True)
