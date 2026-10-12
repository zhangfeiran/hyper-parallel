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
"""Standard physical parameter geometry and independently checked source YaRN frequencies."""

import math
import unittest

import numpy as np
import torch

from hyper_parallel.core.multicore.examples.mega_dsa_standard_model_common import (
    standard_model_fixture,
    standard_model_inputs,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


class TestStandardModelCommon(unittest.TestCase):
    """Meta tensors verify real standard parameter dimensions without allocating checkpoint-sized storage."""

    @staticmethod
    def _config():
        return {'hidden_size':7168, 'q_lora_rank':1536, 'kv_lora_rank':512, 'num_attention_heads':128,
                'qk_nope_head_dim':128, 'qk_rope_head_dim':64, 'v_head_dim':128,
                'index_n_heads':64, 'index_head_dim':128, 'index_topk':2048, 'rms_norm_eps':1e-6,
                'rope_theta':10000., 'max_position_embeddings':163840,
                'rope_scaling':{'type':'yarn', 'factor':40., 'mscale':1., 'mscale_all_dim':1.,
                                'original_max_position_embeddings':4096, 'beta_fast':32, 'beta_slow':1}}

    def test_standard_parameter_shapes_source_scale_and_rng_isolation(self):
        """The large O projection, Q latent and V expansion are all physical module parameters."""
        before = torch.get_rng_state().clone()
        model = standard_model_fixture(self._config(), device='meta')
        expected = {'linear_qkv.weight':(2112,7168), 'q_b_proj.weight':(24576,1536),
                    'kv_b_proj.weight':(32768,512), 'o_proj.weight':(7168,16384),
                    'q_a_layernorm.weight':(1536,), 'kv_a_layernorm.weight':(512,),
                    'indexer.wq_b.weight':(8192,1536), 'indexer.wk.weight':(128,7168),
                    'indexer.weights_proj.weight':(64,7168),
                    'indexer.k_norm.weight':(128,), 'indexer.k_norm.bias':(128,)}
        self.assertEqual({name:tuple(value.shape) for name,value in model.named_parameters()}, expected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(model.scaling, (1+.1*math.log(40))**2/math.sqrt(192))
        self.assertEqual(model.num_heads, 128)
        self.assertEqual(model.v_head_dim, 128)
        self.assertTrue(model.rotary_interleaved)
        self.assertFalse(model.index_rotary_interleaved)

    def test_packed_reset_yarn_band_and_long_position_inputs(self):
        """Independent NumPy correction bands verify packed reset and positions beyond the original context."""
        config = self._config()
        meta = DsaBatchMeta.packed((3, 4098))
        _hidden, (cos, sin) = standard_model_inputs(meta, config=config, dtype=torch.float32)
        inverse = 1/np.power(10000., np.arange(0,64,2,dtype=np.float64)/64)
        low = max(math.floor(32*math.log(4096/(32*2*math.pi))/math.log(10000)), 0)
        high = min(math.ceil(32*math.log(4096/(2*math.pi))/math.log(10000)), 63)
        ramp = np.clip((np.arange(32)-low)/(high-low),0,1)
        inverse = inverse/40*ramp+inverse*(1-ramp)
        for global_id, position in ((0,0),(3,0),(4,1),(4100,4097)):
            reference = position*inverse
            np.testing.assert_allclose(cos[0,global_id,:32].numpy(), np.cos(reference), rtol=0, atol=3e-4)
            np.testing.assert_allclose(sin[0,global_id,:32].numpy(), np.sin(reference), rtol=0, atol=3e-4)
        torch.testing.assert_close(cos[0,0],cos[0,3],rtol=0,atol=0)
        self.assertEqual(cos.shape, (1,4101,64))
        self.assertEqual(_hidden.shape, (1,4101,7168))
        for patch in ({'hidden_size':32}, {'rms_norm_eps':0}, {'attention_dropout':.1}, {'q_lora_rank':64}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                standard_model_fixture({**config, **patch}, device='meta')
        with self.assertRaises(ValueError):
            standard_model_inputs(DsaBatchMeta.packed((4,)), config={**config,'max_position_embeddings':3},
                                  dtype=torch.float32)
