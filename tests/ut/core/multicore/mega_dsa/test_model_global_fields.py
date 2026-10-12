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
"""Complete CP/TP numerical field reconstruction without duplicate gradient sums."""

import unittest

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.model_global_fields import (
    assemble_global_model_fields,
)
from hyper_parallel.core.multicore.modules.mega_dsa.model_reference_precision import (
    model_reference_shapes,
)


class TestModelGlobalFields(unittest.TestCase):
    """Prove strided/empty ownership and reject inconsistent replicas before normalized evaluation."""

    @staticmethod
    def _fixture(empty=False):
        axes = {'q_b_proj.weight': 0, 'kv_b_proj.weight': 0, 'o_proj.weight': 1,
                'indexer.wq_b.weight': 0, 'indexer.weights_proj.weight': 0}
        shapes = model_reference_shapes(64)
        expected = {name: torch.tensor(2., dtype=torch.bfloat16).expand(shape) for name, shape in shapes.items()}
        expected['loss'] = torch.tensor(.5)
        peers = []
        owners = [list(range(13)), []] if empty else [list(range(0, 13, 2)), list(range(1, 13, 2))]
        for cp in range(2):
            for tp in range(2):
                fields = {}
                for name, value in expected.items():
                    if name in ('output', 'hidden'):
                        value = value[:, owners[cp]]
                    elif name.startswith(('gradient/', 'sgd/')) and name.split('/', 1)[1] in axes:
                        value = value.chunk(2, dim=axes[name.split('/', 1)[1]])[tp]
                    fields[name] = value
                peers.append({'cp_rank': cp, 'tp_rank': tp, 'owner_ids': owners[cp], 'fields': fields})
        return peers, expected

    def test_complete_reconstruction_uses_each_head_and_owner_once(self):
        """CP-reduced gradients remain two rather than becoming four after reconstruction."""
        for empty in (False, True):
            with self.subTest(empty=empty):
                peers, expected = self._fixture(empty)
                actual = assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)
                self.assertEqual(set(actual), set(expected))
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_mismatched_replicas_missing_owners_and_local_fields_are_rejected(self):
        """Never choose a convenient replica or fill a missing global token with zeros."""
        peers, _ = self._fixture()
        peers[3]['fields']['gradient/linear_qkv.weight'] = peers[3]['fields']['gradient/linear_qkv.weight']+1
        with self.assertRaisesRegex(ValueError, 'inconsistent replicas'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)
        peers, _ = self._fixture()
        for peer in peers[2:]:
            peer['owner_ids'] = list(range(0, 13, 2))
            for name in ('output', 'hidden'):
                peer['fields'][name] = peers[0]['fields'][name]
        with self.assertRaisesRegex(ValueError, 'exactly one CP owner'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)
        peers, _ = self._fixture()
        peers[0]['fields']['gradient/o_proj.weight'] = torch.ones(32, 64)
        with self.assertRaisesRegex(ValueError, 'local shape'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)

    def test_all_shards_preserve_inactive_gradient_availability(self):
        """None cannot be replaced by a zero tensor or silently propagated over an active shard."""
        peers, _ = self._fixture()
        for peer in peers:
            peer['fields']['gradient/indexer.wq_b.weight'] = None
        actual = assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)
        self.assertIsNone(actual['gradient/indexer.wq_b.weight'])
        peers[1]['fields']['gradient/indexer.wq_b.weight'] = torch.zeros(4096, 64)
        peers[3]['fields']['gradient/indexer.wq_b.weight'] = torch.zeros(4096, 64)
        with self.assertRaisesRegex(ValueError, 'availability'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)

    def test_loss_replica_rounding_difference_remains_a_strict_error(self):
        """Global reconstruction rejects a two-ULP loss difference without averaging or tolerance."""
        peers, _ = self._fixture()
        for peer in peers:
            peer['fields']['loss'] = torch.tensor(
                0.003228132613003254 if peer['tp_rank'] == 0 else 0.0032281330786645412)
        with self.assertRaisesRegex(ValueError, 'inconsistent replicated TP fields: loss'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)


    def test_declared_geometry_reconstructs_each_token_without_changing_default_shapes(self):
        """An explicit source geometry admits a different hidden width but preserves exact replica checks."""
        peers, expected = self._fixture()
        shapes = model_reference_shapes(64)
        shapes['output'] = shapes['hidden'] = [1,13,7]
        token_values = torch.arange(91,dtype=torch.float32).reshape(1,13,7)
        for name in ('output','hidden'):
            expected[name] = token_values
            for peer in peers:
                peer['fields'][name] = token_values[:,peer['owner_ids']]
        actual = assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2, declared_shapes=shapes)
        torch.testing.assert_close(actual['output'], token_values, rtol=0, atol=0)
        self.assertEqual(model_reference_shapes(64)['output'], [1,13,32])
        with self.assertRaisesRegex(ValueError,'local shape'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2)
        malformed = dict(shapes)
        malformed['gradient/o_proj.weight'] = [1]
        with self.assertRaisesRegex(ValueError,'declared global shapes'):
            assemble_global_model_fields(peers, heads=64, cp_size=2, tp_size=2, declared_shapes=malformed)
