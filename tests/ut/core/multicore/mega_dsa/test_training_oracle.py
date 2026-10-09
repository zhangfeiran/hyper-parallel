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
"""Cross-check bounded training VJPs against expanded selected-KV FP32 autograd."""

import unittest

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_training_oracle import training_reference
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference, sparse_attention_reference,
)


class TestTrainingOracle(unittest.TestCase):
    """Keep the long-history checker independent of native arithmetic and sampling."""

    def test_dense_chunked_vjps_match_expanded_reference(self):
        """All seven gradients, objective isolation and global scales survive chunking."""
        generator = torch.Generator().manual_seed(20261009)
        shapes = ((7, 2, 4), (7, 4), (7, 2, 3), (7, 3), (7, 3, 5), (7, 5), (7, 3))
        inputs = tuple((torch.randn(shape, generator=generator) * .3).bfloat16() for shape in shapes)
        cotangent = torch.randn(shapes[0], generator=generator) * .1
        meta = DsaBatchMeta.packed((3, 4))
        indices = torch.tensor([[-1, 0, -1, 2, -1], [0, 1, -1, -1, -1], [2, 0, 1, -1, -1],
                                [0, 3, 6, -1, -1], [4, 3, 6, -1, -1], [-1, -1, -1, -1, -1],
                                [6, 4, -1, 3, -1]], dtype=torch.int32)
        for mode, coefficient, auxiliary, divisor in (
            ("joint", .3, 7, 1), ("joint", .3, 0, 4), ("lm_only", .3, 7, 1),
            ("kl_only", .3, 7, 4), ("coeff_zero", 0., 7, 4),
        ):
            for weight_dtype in (torch.bfloat16, torch.float32):
                values = (*inputs[:6], inputs[6].to(weight_dtype))
                full = tuple(value.float().clone().requires_grad_() for value in values)
                normalization = DsaLossNormalization(7, divisor)
                output, _ = sparse_attention_reference(*full[:4], indices, meta, attention_scale=.7)
                loss = selected_kl_reference(*full[4:], *full[:4], indices, meta, attention_scale=.7,
                                             normalization=normalization, loss_coeff=coefficient)
                objective = loss * auxiliary if mode == "kl_only" else (output * cotangent).sum()
                if mode not in ("kl_only", "lm_only"):
                    objective = objective + loss * auxiliary
                objective.backward()
                for chunk in (1, 3, 7):
                    with self.subTest(mode=mode, weight_dtype=weight_dtype, query_chunk=chunk):
                        actual = training_reference(values, cotangent, indices, normalization,
                                                    coefficient, mode, auxiliary, (3, 4),
                                                    attention_scale=.7, query_chunk=chunk)
                        torch.testing.assert_close(actual[0], output, rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(actual[1], loss, rtol=1e-5, atol=1e-7)
                        for gradient, leaf in zip(actual[2], full):
                            if leaf.grad is None:
                                self.assertIsNone(gradient)
                            else:
                                torch.testing.assert_close(gradient, leaf.grad, rtol=1e-5, atol=1e-7)

    def test_invalid_chunk_and_normalization_reject(self):
        """Reject partition parameters that would silently change the declared objective."""
        for chunk in (0, -1, True, 1.5):
            with self.subTest(query_chunk=chunk), self.assertRaisesRegex(ValueError, "positive integer"):
                training_reference((), torch.empty(0), torch.empty(0), DsaLossNormalization(7),
                                   .3, "joint", 7, (3, 4), attention_scale=.7, query_chunk=chunk)
        with self.assertRaisesRegex(ValueError, "complete packed query count"):
            training_reference((), torch.empty(0), torch.empty(0), DsaLossNormalization(8),
                               .3, "joint", 7, (3, 4), attention_scale=.7)
