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
"""Real model projections with an explicit CPU oracle for the fused CP execution boundary."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
    unabsorbed_model_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.fused_cp import FusedDsaCpForwardProbe
from hyper_parallel.core.multicore.modules.mega_dsa.fused_model_boundary import FusedDsaCpAttention
from hyper_parallel.core.multicore.modules.mega_dsa.local_query import DsaLocalQueryLayout
from hyper_parallel.core.multicore.modules.mega_dsa.local_training import FusedDsaLocalCpTrainingProbe
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta, DsaLossNormalization
from hyper_parallel.core.multicore.modules.mega_dsa.mixed_tile import MixedSfaSchedule
from hyper_parallel.core.multicore.modules.mega_dsa.reference import (
    indexer_reference,
    selected_kl_reference,
    sparse_attention_reference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.workspace import (
    DsaWorkspaceInvocation,
    DsaWorkspaceSpec,
    MegaDsaWorkspace,
)
from tests.ut.core.multicore.shmem.test_consumer import SharedRootFixture


class _OracleTrainingProbe(FusedDsaLocalCpTrainingProbe):
    """Explicit CPU replacement of execution only; real projections remain under test."""

    def forward(self, invocation: DsaWorkspaceInvocation, main_states: tuple[torch.Tensor, ...],
                index_states: tuple[torch.Tensor, ...], *,
                query_layout: DsaLocalQueryLayout) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the independent CPU formulas with real projected inputs."""
        self.calls = getattr(self, "calls", 0)+1
        if not all(value.is_contiguous() for value in (*main_states, *index_states)):
            raise AssertionError("model boundary must normalize native state layouts")
        meta = invocation.batch_meta
        if not query_layout.global_ids:
            output = main_states[0]+sum(value.sum()*0 for value in main_states[1:])
            loss = sum(value.sum()*0 for value in index_states)
        else:
            indices = indexer_reference(*index_states, meta, sparse_count=min(2048, meta.global_valid_queries))
            output = sparse_attention_reference(*main_states, indices, meta,
                                                attention_scale=self.forward_backend.scale)[0]
            loss = (selected_kl_reference(*index_states, *main_states, indices, meta,
                                          attention_scale=self.forward_backend.scale,
                                          normalization=self.normalization, loss_coeff=self.loss_coeff)
                    if self.mode != "lm_only" else output.new_zeros(()))
            self.indices = indices
        self.last_loss = loss
        return output, loss


class TestFusedModelBoundary(SharedRootFixture):
    """Keep real HP parameters, autograd, absorption and restoration while mocking device execution."""

    def setUp(self) -> None:
        """Prepare actual model parameters and CPU-backed owner metadata."""
        super().setUp()
        self.source = build_model_fixture()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(712)
            self.source.indexer.wq_b = nn.Linear(64, 64*128, bias=False, dtype=torch.float64)
            self.source.indexer.weights_proj = nn.Linear(32, 64, bias=False, dtype=torch.float64)
        self.source.num_index_heads = 64
        self.source.config.index_n_heads = 64
        self.workspace = MegaDsaWorkspace(self.root, "model", DsaWorkspaceSpec(3, 3, 3, 1))
        self.consumers.append(self.workspace.consumer)
        self.addCleanup(self.workspace.close)
        self.workspace.bind()
        self.meta = replace(DsaBatchMeta.packed((1, 2)), heap_generation=self.root.generation)
        self.invocation = self.workspace.prepare(self.meta)
        self.backend = FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                             attention_scale=self.source.scaling, schedule=MixedSfaSchedule(7))
        self.queries = self.backend.prepare_local_queries()
        self.probe = self._probe()
        self.hidden, self.embeddings = model_inputs(self.meta, dtype=torch.float64)
        patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
              side_effect=_rotary_mul_oracle).start()
        patch("hyper_parallel.components.functional.aux_loss._AuxLossAutoScaler.main_loss_backward_scale",
              torch.tensor(1.0)).start()
        self.model = FusedDsaCpAttention(module=self.source, context={"cp": True})

    def _probe(self, mode="lm_plus_kl", coefficient=.3):
        return _OracleTrainingProbe(self.backend, normalization=DsaLossNormalization(3),
                                    loss_coeff=coefficient, mode=mode, fused_kl=True,
                                    sparse_main_return=True, sparse_index_return=True)

    def _forward(self, hidden=None, **kwargs):
        return self.model(self.hidden if hidden is None else hidden, position_embeddings=self.embeddings,
                          dsa_training=self.probe, dsa_invocation=self.invocation,
                          dsa_query_layout=self.queries, **kwargs)[0]

    def test_parameters_children_and_checkpoint_keys_are_preserved_by_identity(self):
        for name, value in self.source.named_parameters():
            self.assertIs(dict(self.model.named_parameters())[name], value)
        for name, value in self.source.named_children():
            self.assertIs(dict(self.model.named_children())[name], value)
        self.assertEqual(list(self.model.state_dict()), list(self.source.state_dict()))
        self.assertEqual(self.model.make_transforms(), [])
        self.assertIsNot(self.model._modules, self.source._modules)

    def test_actual_model_output_and_all_gradients_match_unabsorbed_fp64(self):
        reference = copy.deepcopy(self.source)
        hidden, expected_hidden = self.hidden.clone().requires_grad_(), self.hidden.clone().requires_grad_()
        set_aux_loss_scale(torch.tensor(-2.5))
        actual = self._forward(hidden)
        expected, loss = unabsorbed_model_reference(reference, expected_hidden, self.embeddings,
                                                    self.meta, self.probe.indices)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        left = torch.autograd.grad(actual.square().mean(), (hidden, *self.model.parameters()))
        right = torch.autograd.grad(expected.square().mean()+loss*-2.5,
                                    (expected_hidden, *reference.parameters()))
        for name, observed, target in zip(("hidden", *dict(self.model.named_parameters())), left, right):
            with self.subTest(parameter=name):
                torch.testing.assert_close(observed, target, rtol=1e-9, atol=1e-11)

    def test_freeze_eval_and_zero_coefficient_require_matching_prepared_probe(self):
        for mode in ("freeze", "eval", "zero"):
            with self.subTest(mode=mode):
                self.model.freeze_dsa = mode == "freeze"
                self.model.train(mode != "eval")
                self.model.dsa_loss_coeff = 0. if mode == "zero" else .3
                self.probe = self._probe("lm_only", self.model.dsa_loss_coeff)
                output = self._forward()
                gradients = torch.autograd.grad(output.square().mean(), tuple(self.model.parameters()),
                                                allow_unused=True)
                for name, gradient in zip(dict(self.model.named_parameters()), gradients):
                    if name.startswith("indexer."):
                        self.assertIsNone(gradient, name)
                    else:
                        self.assertIsNotNone(gradient, name)

    def test_missing_stale_or_incompatible_declarations_fail_before_projections(self):
        with patch.object(self.model.linear_qkv, "forward", side_effect=AssertionError("projection ran")):
            with self.assertRaisesRegex(TypeError, "dsa_training"):
                self.model(self.hidden)
            for kwargs, message in (({"actual_seq_len": (3,)}, "conflicts"),
                                    ({"actual_q_len": (1, 3)}, "aliases")):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                    self._forward(**kwargs)
            self.probe.loss_coeff = .5
            with self.assertRaisesRegex(ValueError, "matching"):
                self._forward()
            self.probe.loss_coeff = .3
            self.invocation = replace(self.invocation, batch_meta=replace(self.meta, heap_generation=99))
            with self.assertRaisesRegex(ValueError, "generation"):
                self._forward()
        with self.assertRaisesRegex(ValueError, "TP=1"):
            FusedDsaCpAttention(module=self.source, context={"tp": True})
        with self.assertRaisesRegex(ValueError, "Hi64"):
            FusedDsaCpAttention(module=build_model_fixture())

    def test_prepared_lengths_do_not_read_device_values(self):
        with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host read")):
            self.model._validate_call(self.hidden, self.queries.key_lengths,
                                      {"dsa_training": self.probe, "dsa_invocation": self.invocation,
                                       "dsa_query_layout": self.queries})

    def test_reordered_owner_rows_restore_caller_output_order(self):
        """Sorted packed query IDs must not replace the caller's storage ordering."""
        expected = self._forward()
        order = (2, 0, 1)
        self.meta = replace(self.meta, q_global_ids=order, kv_global_ids=order)
        self.invocation = self.workspace.prepare(self.meta)
        self.backend = FusedDsaCpForwardProbe(self.workspace, self.invocation, heads=32,
                                             attention_scale=self.model.scaling, schedule=MixedSfaSchedule(7))
        self.queries = self.backend.prepare_local_queries()
        self.probe = self._probe()
        self.hidden = self.hidden[:, list(order)]
        self.embeddings = tuple(value[:, list(order)] for value in self.embeddings)
        actual = self._forward()
        torch.testing.assert_close(actual, expected[:, list(order)], rtol=1e-10, atol=1e-12)

    @patch.object(DefaultDeviceType, "get_device_type", return_value="cpu")
    def test_checkpoint_preserves_real_model_input_and_parameter_gradients(self, _device):
        hidden = self.hidden.clone().requires_grad_()
        plain = torch.autograd.grad(self._forward(hidden).square().mean(), (hidden, *self.model.parameters()))
        hidden = self.hidden.clone().requires_grad_()
        output = checkpoint(self._forward, hidden, use_reentrant=False)
        replay = torch.autograd.grad(output.square().mean(), (hidden, *self.model.parameters()))
        for actual, expected in zip(replay, plain):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_empty_owner_still_calls_execution_and_returns_valid_local_shape(self):
        meta = replace(self.meta, q_global_ids=(), kv_global_ids=(), token_owners=(0, 0, 0),
                       cp_ranks=(0, 1), root_pes=(0, 1), cp_rank=1)
        backend = SimpleNamespace(workspace=self.workspace, heads=32, scale=self.model.scaling,
                                  layout=SimpleNamespace(batch_meta=meta, device=torch.device("cpu")))
        self.probe.forward_backend = backend
        self.invocation = replace(self.invocation, batch_meta=meta)
        self.queries = DsaLocalQueryLayout(meta, "cpu")
        hidden = self.hidden[:, :0].clone().requires_grad_()
        output = self._forward(hidden)
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(self.probe.calls, 1)
        output.sum().backward()
        self.assertEqual(hidden.grad.shape, hidden.shape)
