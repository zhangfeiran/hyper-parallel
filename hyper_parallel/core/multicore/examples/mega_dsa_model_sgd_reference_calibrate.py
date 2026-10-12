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
"""Independent stock/FP32 model calibration with gradient-coupled BF16 SGD arithmetic bounds."""

from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples import (
    mega_dsa_model_reference_calibrate as validation,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_oracle_verify import (
    verify_model_oracle,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_precision import (
    coupled_model_reference_policy,
    fit_coupled_model_reference_profile,
)


def _measure_reference(*args, **kwargs):
    proofs = []

    def _verify(template, mode, hidden, embeddings, cotangent, meta, expected, policy):
        proof = verify_model_oracle(template, mode, hidden, embeddings, cotangent, meta,
                                   validation._complete_indices(meta), expected, policy)
        proofs.append(proof)
        return proof['passed']

    with patch.object(validation, '_oracle_check', side_effect=_verify):
        result = _ORIGINAL_MEASURE(*args, **kwargs, sgd_geometry=True)
    if len(proofs) != 1:
        raise RuntimeError('each reference sample requires one independent complete oracle proof')
    result['output']['oracle_proof'] = proofs[0]
    return result


_ORIGINAL_MEASURE = validation._measure


def main() -> None:
    """Reuse reference operators while collecting a separately declared disjoint policy."""
    torch.set_num_threads(coupled_model_reference_policy()['cpu_reference_threads'])
    with patch.object(validation, 'model_reference_policy', side_effect=coupled_model_reference_policy), \
            patch.object(validation, 'fit_model_reference_profile', side_effect=fit_coupled_model_reference_profile), \
            patch.object(validation, '_measure', side_effect=_measure_reference):
        validation.main()


if __name__ == '__main__':
    main()
