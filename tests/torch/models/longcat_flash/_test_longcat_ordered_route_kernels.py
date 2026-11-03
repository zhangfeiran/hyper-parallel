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
"""Original-width numerical checks for the optional ordered route kernels."""

from typing import Any

import torch
import torch_npu  # pylint: disable=unused-import
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashTopkRouter

from hyper_parallel.core.multicore.modules.mega_moe.assignments import masked_expert_forward
from hyper_parallel.models.longcat_flash.adapter.experts import identity_contribution
from hyper_parallel.models.longcat_flash.adapter.routing import OrderedInputGather
from hyper_parallel.models.longcat_flash.adapter.route_kernels import OrderedCombine, OrderedGather


class _RouteOnlyExecutor:
    """Use a parameter-free stand-in to isolate route arithmetic from GEMM arithmetic."""

    num_experts = 512

    def __call__(self, hidden: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        """Return a deterministic stand-in with a nontrivial BF16 input derivative."""
        return hidden * 1.125


def test_longcat_ordered_route_kernels() -> None:
    """Compare every output, dX and dP element at the production router width."""
    torch.npu.set_device(0)
    torch.manual_seed(700)
    torch.npu.manual_seed(701)
    config = LongcatFlashConfig.from_dict({"num_layers": 1})
    router = LongcatFlashTopkRouter(config).to(device="npu", dtype=torch.bfloat16)
    data = torch.randn(2048, 6144, dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        probabilities, learned_ids = router(data)[-2:]
    for case in ("learned", "identity_only", "real_only", "large_real_only", "empty"):
        tokens = 2048 if case == "large_real_only" else 512
        ids = learned_ids[:tokens].clone()
        if case in ("identity_only", "real_only", "large_real_only"):
            ids = torch.arange(12, device="npu").expand(tokens, -1)
            ids = ids + (512 if case == "identity_only" else 0)
        source = data[:0] if case == "empty" else data[:tokens]
        ids = ids[:source.shape[0]]
        order = ids.float().argsort(-1)
        ids = ids.gather(1, order)
        weights = probabilities[:source.shape[0]].gather(1, order).detach().requires_grad_()
        other = weights.detach().clone().requires_grad_()
        reference = source.detach().clone().requires_grad_()
        candidate = source.detach().clone().requires_grad_()
        gather = OrderedInputGather(12)
        real = masked_expert_forward(_RouteOnlyExecutor(), reference, ids, weights, ids < 512,
                                     expert_weights=(), input_gather=gather)
        expected = identity_contribution(reference, ids, weights, 512, initial=real,
                                         inputs=gather.identity_inputs)
        token, slot = (ids < 512).nonzero(as_tuple=True)
        inverse = torch.full(ids.shape, -1, device="npu", dtype=torch.int32)
        inverse.view(-1).index_copy_(0, token * 12 + slot,
                                    torch.arange(token.numel(), device="npu", dtype=torch.int32))
        states, identity = OrderedGather.apply(candidate, token, inverse)
        actual = OrderedCombine.apply(identity, states * 1.125, other, inverse, token, slot)
        upstream = torch.randn(source.shape[0], 12288, device="npu", dtype=torch.bfloat16)[:, ::2]
        expected.backward(upstream)
        actual.backward(upstream)
        for observed, target in ((actual, expected), (candidate.grad, reference.grad), (other.grad, weights.grad)):
            torch.testing.assert_close(observed, target, rtol=0, atol=0, msg=case)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_longcat_ordered_route_kernels()
