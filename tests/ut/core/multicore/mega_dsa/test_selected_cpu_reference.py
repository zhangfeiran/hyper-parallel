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
"""Bounded selected attention/KL against independent autograd and long-history selections."""

import unittest
from dataclasses import replace

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    selected_kl_reference,
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.selected_cpu_reference import (
    selected_attention_cpu,
    selected_kl_cpu,
)


def _leaves(shapes, seed=627):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape,generator=generator,dtype=torch.float64).requires_grad_() for shape in shapes)


class TestSelectedCpuReference(unittest.TestCase):
    """Require exact selection semantics, first-order derivatives and source storage lifetime."""

    def test_absorbed_forward_stats_all_gradients_and_alias_merge(self):
        """C as key/value matches ordinary autograd, with reordered storage, packed filtering and empty rows."""
        meta = replace(DsaBatchMeta.packed((3,2)),q_global_ids=(4,2,0),kv_global_ids=(3,1,4,0,2))
        main = _leaves(((3,2,3),(5,3),(3,2,2),(5,2)))
        indices = torch.tensor([[3,4,-1],[0,2,4],[-1,-1,-1]],dtype=torch.int32)
        query, compressed, qr, kr = main
        output,stats = selected_attention_cpu(query,compressed,compressed,qr,kr,indices,meta,attention_scale=.37)
        expected,expected_stats = sparse_attention_reference(*main,indices,meta,attention_scale=.37)
        torch.testing.assert_close(output,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(stats.maximum,expected_stats.maximum,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(stats.denominator,expected_stats.denominator,rtol=1e-12,atol=1e-12)
        upstream = torch.arange(output.numel(),dtype=output.dtype).reshape(output.shape)/17
        actual = torch.autograd.grad(output,main,upstream)
        reference = torch.autograd.grad(expected,main,upstream)
        for value,target in zip(actual,reference):
            torch.testing.assert_close(value,target,rtol=1e-12,atol=1e-12)
        self.assertFalse(stats.maximum.requires_grad)
        self.assertEqual(float(output[-1].abs().sum()),0.)

    def test_unabsorbed_per_head_key_value_against_explicit_head_loops(self):
        """Separate expanded K/V head dimensions retain all original projection edges."""
        meta = DsaBatchMeta.packed((3,))
        query,key,value,qr,kr = _leaves(((3,2,3),(3,2,3),(3,2,4),(3,2,2),(3,2)))
        indices = torch.tensor([[0,-1],[0,1],[0,2]],dtype=torch.int32)
        actual,_stats = selected_attention_cpu(query,key,value,qr,kr,indices,meta,attention_scale=.43)
        expected = torch.stack([torch.stack([
            ((query[row,head]@key[ids,head].T+qr[row,head]@kr[ids].T)*.43).softmax(-1)@value[ids,head]
            for head in range(2)]) for row in range(3)
            for ids in [[token for token in indices[row].tolist() if token>=0]]])
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        inputs = (query,key,value,qr,kr)
        gradients = torch.autograd.grad(actual.square().sum(),inputs)
        wanted = torch.autograd.grad(expected.square().sum(),inputs)
        for value,target in zip(gradients,wanted):
            torch.testing.assert_close(value,target,rtol=1e-12,atol=1e-12)

    def test_selected_kl_index_only_scale_and_empty_rows(self):
        """Signed weights, teacher isolation, arbitrary scalar upstream and CP normalization match the small oracle."""
        meta = DsaBatchMeta.packed((3,2))
        main = _leaves(((5,3,4),(5,4),(5,3,2),(5,2)))
        index = _leaves(((5,2,3),(5,3),(5,2)),seed=633)
        indices = torch.tensor([[0,-1,-1],[0,1,-1],[0,2,-1],[3,-1,-1],[-1,-1,-1]],dtype=torch.int32)
        normalization = DsaLossNormalization(5,2)
        actual = selected_kl_cpu(index,main,indices,meta,attention_scale=.37,
                                 normalization=normalization,loss_coeff=.23)
        expected = selected_kl_reference(*index,*main,indices,meta,attention_scale=.37,
                                         normalization=normalization,loss_coeff=.23)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        observed = torch.autograd.grad(actual*-1.75,(*index,*main),allow_unused=True)
        wanted = torch.autograd.grad(expected*-1.75,index)
        self.assertTrue(all(value is None for value in observed[3:]))
        for value,target in zip(observed[:3],wanted):
            torch.testing.assert_close(value,target,rtol=1e-12,atol=1e-12)

    def test_gradcheck_replay_and_source_versions(self):
        """First-order numerical derivatives agree under recompute; original tensor mutation remains an error."""
        meta = DsaBatchMeta.packed((3,))
        main = _leaves(((3,2,3),(3,3),(3,2,2),(3,2)))
        indices = torch.tensor([[0,-1],[0,1],[0,2]],dtype=torch.int32)

        def _attention(query,compressed,qr,kr):
            return selected_attention_cpu(query,compressed,compressed,qr,kr,indices,meta,attention_scale=.37)[0]

        self.assertTrue(torch.autograd.gradcheck(_attention,main,fast_mode=True))
        plain = _attention(*main)
        repeated = checkpoint(_attention,*main,use_reentrant=False)
        first = torch.autograd.grad(plain.sum(),main,retain_graph=True)
        second = torch.autograd.grad(plain.sum(),main)
        replay = torch.autograd.grad(repeated.sum(),main)
        for value,again,restored in zip(first,second,replay):
            torch.testing.assert_close(value,again,rtol=0,atol=0)
            torch.testing.assert_close(value,restored,rtol=0,atol=0)
        invalid = _attention(*main)
        with torch.no_grad():
            indices[0,0] = -1
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(invalid.sum(),main)

    def test_long_history_selection_and_saved_scratch_population(self):
        """A 4100-token history with K2048 is genuinely selected, including one late omitted key with a huge value."""
        meta = replace(DsaBatchMeta.packed((4100,)),q_global_ids=(4099,))
        query = torch.ones(1,2,3,dtype=torch.float64,requires_grad=True)
        compressed = torch.ones(4100,3,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():
            compressed[4098].fill_(10000.)
        qr = torch.zeros(1,2,2,dtype=torch.float64,requires_grad=True)
        kr = torch.zeros(4100,2,dtype=torch.float64,requires_grad=True)
        indices = torch.arange(2048,dtype=torch.int32).view(1,-1)
        saved_shapes = []

        def _pack(value):
            saved_shapes.append(tuple(value.shape))
            return value

        with torch.autograd.graph.saved_tensors_hooks(_pack,lambda value:value):
            output,_stats = selected_attention_cpu(query,compressed,compressed,qr,kr,indices,meta,attention_scale=.3)
        torch.testing.assert_close(output,torch.ones_like(output),rtol=1e-12,atol=1e-12)
        gradient = torch.autograd.grad(output.sum(),compressed)[0]
        self.assertEqual(float(gradient[2048:].abs().sum()),0.)
        self.assertEqual(len(saved_shapes),6)
        self.assertNotIn((1,2,2048),saved_shapes)
        with self.assertRaises(ValueError):
            selected_attention_cpu(query,compressed,compressed,qr,kr,indices.float(),meta,attention_scale=.3)
        duplicate = indices.clone()
        duplicate[0,1]=0
        with self.assertRaisesRegex(ValueError,'duplicate'):
            selected_attention_cpu(query,compressed,compressed,qr,kr,duplicate,meta,attention_scale=.3)


    def test_kl_gradcheck_checkpoint_and_detached_main_versions(self):
        """KL derivatives recompute faithfully and detect mutations of the detached teacher source."""
        meta = DsaBatchMeta.packed((3,))
        main = _leaves(((3,2,3),(3,3),(3,2,2),(3,2)),seed=672)
        index = _leaves(((3,2,3),(3,3),(3,2)),seed=678)
        indices = torch.tensor([[0,-1],[0,1],[0,2]],dtype=torch.int32)

        def _objective(query,key,weight):
            return selected_kl_cpu((query,key,weight),main,indices,meta,attention_scale=.37,
                                   normalization=DsaLossNormalization(3),loss_coeff=.3)

        self.assertTrue(torch.autograd.gradcheck(_objective,index,fast_mode=True))
        plain = _objective(*index)
        replay = checkpoint(_objective,*index,use_reentrant=False)
        first = torch.autograd.grad(plain*-1.75,index,retain_graph=True)
        second = torch.autograd.grad(plain*-1.75,index)
        restored = torch.autograd.grad(replay*-1.75,index)
        for value,again,recomputed in zip(first,second,restored):
            torch.testing.assert_close(value,again,rtol=0,atol=0)
            torch.testing.assert_close(value,recomputed,rtol=0,atol=0)
        invalid = _objective(*index)
        with torch.no_grad():
            main[1].add_(1)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(invalid,index)

    def test_fp32_precision_under_cpu_autocast(self):
        """Autocast cannot alter the independently declared FP32 oracle arithmetic."""
        meta = DsaBatchMeta.packed((3,))
        main = tuple(value.detach().float().requires_grad_() for value in
                     _leaves(((3,2,3),(3,3),(3,2,2),(3,2)),seed=693))
        indices = torch.tensor([[0,-1],[0,1],[0,2]],dtype=torch.int32)
        query,compressed,qr,kr = main
        baseline,_stats = selected_attention_cpu(query,compressed,compressed,qr,kr,indices,meta,attention_scale=.37)
        baseline_gradients = torch.autograd.grad(baseline.square().sum(),main)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual,_stats = selected_attention_cpu(query,compressed,compressed,qr,kr,indices,meta,attention_scale=.37)
            gradients = torch.autograd.grad(actual.square().sum(),main)
        torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
        for value,reference in zip(gradients,baseline_gradients):
            torch.testing.assert_close(value,reference,rtol=0,atol=0)
