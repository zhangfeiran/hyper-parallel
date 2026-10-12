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
"""Selected unabsorbed model math against an independently absorbed ordinary autograd oracle."""

import copy
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_cp_tp_model_validate import (
    _reconstruct_selection,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
)
from hyper_parallel.core.multicore.examples.mega_dsa_selected_model_reference import (
    selected_unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)


class TestSelectedModelReference(unittest.TestCase):
    """Fixed non-dense histories retain full hidden and original Parameter derivatives."""

    def test_all_parameter_vjps_at_incomplete_selected_history(self):
        """Compare absorption identities without silently replacing the selection by dense history."""
        meta = DsaBatchMeta.packed((3,2))
        source = build_model_fixture(dtype=torch.float64,seed=20261265)
        absorbed = copy.deepcopy(source)
        unabsorbed = copy.deepcopy(source)
        hidden, embeddings = model_inputs(meta,dtype=torch.float64,seed=20261266)
        actual_hidden, expected_hidden = (hidden.clone().requires_grad_() for _ in range(2))
        indices = torch.tensor([[0,-1],[1,-1],[0,2],[3,-1],[3,4]],dtype=torch.int32)
        set_aux_loss_scale(torch.tensor(-1.75,dtype=torch.float64))
        with patch('hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul',
                   side_effect=_rotary_mul_oracle):
            q_resid,main,weights = absorbed._prepare_attention_states(actual_hidden,embeddings)
            q,c,qr,kr = main
            compressed,_stats = sparse_attention_reference(q.flatten(0,1),c.flatten(0,1)[:,0],
                qr.flatten(0,1),kr.flatten(0,1)[:,0],indices,meta,attention_scale=absorbed.scaling)
            values_up = weights[:,absorbed.qk_nope_head_dim:].transpose(1,2)
            restored = torch.einsum('thc,hcv->thv',compressed,values_up)
            actual = absorbed.o_proj(restored.reshape(1,5,-1))
            iq,ik,weight = absorbed._project_index_states(actual_hidden,q_resid,embeddings)
            actual_loss = selected_kl_reference(iq.flatten(0,1),ik.flatten(0,1)[:,0],weight.flatten(0,1),
                q.flatten(0,1),c.flatten(0,1)[:,0],qr.flatten(0,1),kr.flatten(0,1)[:,0],indices,meta,
                attention_scale=absorbed.scaling,normalization=DsaLossNormalization(5),loss_coeff=.3)
            actual_gradients = torch.autograd.grad(actual.square().sum()+actual_loss*-1.75,
                                                   (actual_hidden,*absorbed.parameters()),allow_unused=True)
            expected, loss = selected_unabsorbed_model_reference(unabsorbed,expected_hidden,embeddings,meta,indices)
            expected_gradients = torch.autograd.grad(expected.square().sum()+loss*-1.75,
                                                     (expected_hidden,*unabsorbed.parameters()),allow_unused=True)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)
        torch.testing.assert_close(actual_loss,loss,rtol=1e-9,atol=1e-11)
        self.assertEqual(len(actual_gradients),12)
        self.assertTrue(all(value is not None for value in actual_gradients))
        for value,target in zip(actual_gradients,expected_gradients):
            torch.testing.assert_close(value,target,rtol=1e-9,atol=1e-11)

    def test_global_selection_reconstruction_uses_every_owner_once(self):
        """Strided and empty ownership reconstruct real selected indices without head-shard summation."""
        meta = DsaBatchMeta.packed((3,2))
        selection = torch.full((5,2048),-1,dtype=torch.int32)
        selection[:,0] = torch.arange(5,dtype=torch.int32)
        for owners in (((0,2,4),(1,3)),(tuple(range(5)),())):
            with self.subTest(owners=owners):
                packets = [(ids,selection[list(ids)]) for ids in owners]
                actual = _reconstruct_selection(packets,meta)
                torch.testing.assert_close(actual,selection,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            _reconstruct_selection([((0,),selection[:1]),((0,),selection[:1])],meta)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            _reconstruct_selection([((0,),selection[:1])],meta)
