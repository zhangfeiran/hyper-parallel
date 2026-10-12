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
"""Locked query workspace layout, scaling, dtype admission and storage lifetime."""

import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.query_workspace import (
    scaled_query_workspace_vjps,
)


class TestQueryWorkspace(unittest.TestCase):
    """Build an independent sentinel-coded ND producer plane rather than mocking the decoder."""

    @staticmethod
    def _workspace(rows, heads):
        # Mirror the locked producer's separate mm1/mm2, selected K/V, p/ds buffers.
        mm12 = heads*128*4*4
        selected_key = 128*576*4*2
        selected_value = 128*512*2*2
        predicted = min(rows,20)*(mm12*2+selected_key+selected_value)
        encoded = (torch.arange(rows*heads*576,dtype=torch.float32).reshape(rows,heads,576)%1009-504)/257
        storage = torch.full((predicted+encoded.numel()*4+512,),173,dtype=torch.uint8)
        storage[predicted:predicted+encoded.numel()*4].view(torch.float32).copy_(encoded.flatten())
        return storage, encoded

    def test_nd_head_rope_split_scale_and_scratch_ownership(self):
        """Head/row sentinels expose wrong offsets or strides on both sides of the 20-core cap."""
        for rows,heads in ((1,32),(3,64),(21,32)):
            with self.subTest(rows=rows,heads=heads):
                retained,encoded=self._workspace(rows,heads)
                scale=192**-.5
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    query,rope=scaled_query_workspace_vjps(retained,query_tokens=rows,key_tokens=31,
                                                          heads=heads,attention_scale=scale)
                torch.testing.assert_close(query,encoded[...,:512]*scale,rtol=0,atol=0)
                torch.testing.assert_close(rope,encoded[...,512:]*scale,rtol=0,atol=0)
                self.assertEqual(query.dtype,torch.float32)
                self.assertFalse(query.requires_grad)
                retained.zero_()
                torch.testing.assert_close(query,encoded[...,:512]*scale,rtol=0,atol=0)
                torch.testing.assert_close(rope,encoded[...,512:]*scale,rtol=0,atol=0)

    def test_empty_queries_never_decode_special_empty_layout(self):
        """An empty producer returns empty owned FP32 tensors without requiring positive-Q scratch."""
        query,rope=scaled_query_workspace_vjps(torch.empty(0,dtype=torch.uint8),query_tokens=0,key_tokens=13,
                                              heads=64,attention_scale=.1)
        self.assertEqual(query.shape,(0,64,512))
        self.assertEqual(rope.shape,(0,64,64))

    def test_invalid_geometry_alignment_scale_and_capacity(self):
        """Wrong producer declarations and incomplete/misaligned snapshots cannot yield gradients."""
        retained,_=self._workspace(1,32)
        args={'query_tokens':1,'key_tokens':13,'heads':32,'attention_scale':.1}
        for override in ({'query_tokens':-1},{'key_tokens':0},{'heads':16},{'attention_scale':True},
                         {'attention_scale':0.},{'attention_scale':float('nan')},{'attention_scale':float('inf')},
                         {'attention_scale':1e300},{'attention_scale':1e-300}):
            with self.subTest(override=override),self.assertRaises(ValueError):
                scaled_query_workspace_vjps(retained,**dict(args,**override))
        for value in (retained[:-1024],retained[1:],retained[::2],retained.float(),retained.view(1,-1)):
            with self.subTest(shape=value.shape,dtype=value.dtype),self.assertRaises(ValueError):
                scaled_query_workspace_vjps(value,**args)
