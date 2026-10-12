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
"""CPU regression for formula verification independently from FP32 calibration tensors."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_oracle_verify import (
    verify_model_oracle,
)
from hyper_parallel.core.multicore.examples.mega_dsa_tp_model_common import (
    tp_model_fixture,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.model_sgd_precision import (
    coupled_model_reference_policy,
)


class TestModelOracleVerify(unittest.TestCase):
    """Keep exact FP32 replay and stricter FP64 formula equivalence as separate requirements."""

    def setUp(self) -> None:
        """Enforce the declared exact replay environment and restore the caller's CPU thread count."""
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    @staticmethod
    def _fixture(heads, seed, mode):
        policy = coupled_model_reference_policy()
        meta = DsaBatchMeta.packed(tuple(policy['lengths']))
        model = tp_model_fixture(heads=heads, seed=seed)
        source = copy.deepcopy(model).float()
        source.train(mode != 'eval')
        source.freeze_dsa = mode == 'freeze'
        source.dsa_loss_coeff = 0. if mode == 'zero' else .3
        hidden, embeddings = model_inputs(meta, dtype=torch.bfloat16, seed=seed+2)
        cotangent = torch.randn(hidden.shape, generator=torch.Generator().manual_seed(seed+3))*.01
        indices = torch.full((meta.global_valid_queries, 2048), -1, dtype=torch.int32)
        for token in range(meta.global_valid_queries):
            start = meta.global_cu_seqlens[meta.sequence_position(token)[0]]
            indices[token, :token-start+1] = torch.arange(start, token+1)
        inputs = hidden.float().requires_grad_()
        with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
                   side_effect=_rotary_mul_oracle):
            output, loss = unabsorbed_model_reference(
                source, inputs, tuple(x.float() for x in embeddings), meta, indices)
            objective = (output*cotangent).sum()
            if loss is not None:
                objective = objective+loss*policy['auxiliary']
            gradients = torch.autograd.grad(objective, (inputs, *source.parameters()), allow_unused=True)
        expected = {'output': output, 'hidden': gradients[0], 'loss': loss,
                    **{f'gradient/{name}': value for (name, _parameter), value in
                       zip(source.named_parameters(), gradients[1:])}}
        return (model, mode, hidden, embeddings, cotangent, meta, indices, expected, policy)

    def test_reassociation_fixtures_pass_fp64_identity_and_exact_fp32_replay(self):
        """Previously rejected FP32 matrix reassociations must satisfy the stricter FP64 formula proof."""
        for heads, seed in ((64, 20261215), (128, 20261201)):
            for mode in ('joint', 'freeze', 'zero', 'eval'):
                with self.subTest(heads=heads, mode=mode):
                    proof = verify_model_oracle(*self._fixture(heads, seed, mode))
                    self.assertTrue(proof['passed'], proof)
                    self.assertTrue(all(value['fp32_replay_exact'] for value in proof['fields'].values()))

    def test_changed_fp32_expected_tensor_is_rejected(self):
        """An otherwise valid formula proof cannot hide a substituted FP32 calibration target."""
        fixture = self._fixture(64, 20261215, 'eval')
        expected = fixture[-2]
        expected['hidden'] = expected['hidden'].clone()
        expected['hidden'][0, 0, 0] += .01
        proof = verify_model_oracle(*fixture)
        self.assertFalse(proof['passed'])
        self.assertTrue(proof['fields']['hidden']['formula_identity_passed'])
        self.assertFalse(proof['fields']['hidden']['fp32_replay_exact'])
