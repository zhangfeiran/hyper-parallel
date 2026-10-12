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
"""Focused repeatability control for numerical differences in actual-model accumulation."""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

from hyper_parallel.core.multicore.examples import (
    mega_dsa_cp_tp_model_accumulation_validate as validation,
)


def main() -> None:
    """Repeat isolated native microbatches without changing precision or acceptance thresholds."""
    original_run = validation._run
    original_case = validation._case

    def _run(*args, **kwargs):
        if args[1] == 'cp2_tp2':
            return None
        kwargs['case_matrix'] = [('joint', 'reverse')]
        kwargs['case_runner'] = partial(original_case, repeat_isolated=True)
        return original_run(*args, **kwargs)

    with patch.object(validation, '_run', side_effect=_run):
        validation.main()


if __name__ == '__main__':
    main()
