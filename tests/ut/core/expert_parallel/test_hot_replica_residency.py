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
"""Keep actual saved-forward reuse separate from shadow residency opportunities."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from tests.common.mark_utils import arg_mark
from tests.torch.expert_parallel import hot_replica_residency as residency


class TestHotReplicaResidency(unittest.TestCase):
    """Observe provider decisions without real devices or a distributed group."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_actual_reuse_and_fresh_copy_have_different_byte_accounting(self) -> None:
        """
        Feature: Actual provider reuse accounting
        Description: Compare a saved token hit with a refreshed backward even when shadow identities match.
        Expectation: Count actual copies and reused bytes separately from potential additional reuse.
        """
        weights = (torch.ones(1, 2, 4, dtype=torch.bfloat16), torch.ones(1, 4, 2, dtype=torch.bfloat16))
        transfer = SimpleNamespace(target_rank=0, target_slot=1, owner_rank=1, owner_slot=0, logical_expert=1)
        route = SimpleNamespace(rank=0, plan=SimpleNamespace(config=SimpleNamespace(home_experts=1),
                                                            transfers=(transfer,)))
        resources = SimpleNamespace(workspace=object(), heap_manager=SimpleNamespace(epoch=0))
        for reused in (False, True):
            with self.subTest(reused=reused):
                token = object()

                @contextmanager
                def _prefetch(*_args, _token=token, _reused=reused, **options):
                    generation = _token if not options.get("backward", False) or _reused else object()
                    yield SimpleNamespace(weight_generation=generation)

                with (patch.object(residency.function, "prefetch_weights", _prefetch),
                      patch.object(torch.npu, "Event", return_value=Mock(query=Mock(return_value=False))),
                      patch.object(torch.npu, "current_stream"),
                      residency.ResidencyAudit(resources) as audit):
                    with audit.invocation("layer-a", 0, "F"):
                        with residency.function.prefetch_weights(weights, route, overlap=True):
                            pass
                    with audit.invocation("layer-a", 0, "B"):
                        with residency.function.prefetch_weights(weights, route, overlap=True, backward=True,
                                                                 reuse_generation=token):
                            pass
                summary = residency._residency_summary(audit.records)
                self.assertEqual(summary["requested_weight_bytes"], 64)
                self.assertEqual(summary["potential_hit_bytes"], 32)
                self.assertEqual(summary["actual_weight_copy_bytes"], 32 if reused else 64)
                self.assertEqual(summary["actual_weight_reused_bytes"], 32 if reused else 0)
                self.assertEqual(summary["potential_additional_reuse_bytes"], 0 if reused else 32)
                self.assertEqual([row["actual_data_copy"] for row in audit.records],
                                 [True, True, not reused, not reused])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_checkpoint_scenario_records_original_recompute_and_backward(self) -> None:
        """
        Feature: Real checkpoint invocation sequence
        Description: Execute reentrant checkpoint and backward on a CPU differentiable module.
        Expectation: Label original and recomputed forwards independently and preserve the input gradient.
        """
        records = []

        @contextmanager
        def _invocation(layer, generation, name):
            records.append((layer, generation, name, torch.is_grad_enabled()))
            yield

        values = torch.tensor([[0.5, 1.0]], requires_grad=True)
        def _module(inputs, _ids, _scores):
            return inputs.sin()

        audit = SimpleNamespace(invocation=_invocation)
        # torch_npu changes checkpoint's default device even for CPU-only inputs.
        with patch.object(DefaultDeviceType, "get_device_type", return_value="cpu"):
            residency._audit_scenario("checkpoint_recompute", audit, (_module, _module), (), values,
                                      torch.zeros(1, 1, dtype=torch.int64), {},
                                      torch.ones(1, 1), torch.ones_like(values))
        self.assertEqual(records, [("layer-a", 0, "checkpoint_recompute-F", False),
                                   ("layer-a", 0, "checkpoint_recompute-B", True),
                                   ("layer-a", 0, "checkpoint_recompute-recompute-F", True)])
        torch.testing.assert_close(values.grad, values.detach().cos())
