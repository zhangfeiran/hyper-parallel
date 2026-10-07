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
"""CPU-only Gate Route example for semantic IR and reference validation."""

from __future__ import annotations

import torch

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml

TOKEN_EXPERT_SHAPE = ("T", "E")
EXPERT_SHAPE = ("E",)


@mc.program(target="ascend", schedule=mc.WorkerPipeline(axis=0, workers="aiv"))
def _route(
    logits: ml.Tensor[ml.fp32, TOKEN_EXPERT_SHAPE],
    bias: ml.Tensor[ml.fp32, EXPERT_SHAPE],
    k: ml.Constexpr[int],
    scale: ml.Constexpr[float],
):
    """Preserve the proposed Gate numerical order and selection-only bias."""
    scores = ml.sqrt(ml.softplus(logits))
    selection = ml.add(scores, ml.stop_gradient(bias))
    indices = ml.topk_indices(selection, k, axis=-1, sorted=False)
    selected = ml.gather(scores, indices, axis=-1)
    if k > 1:
        denominator = ml.add(ml.reduce_sum(selected, axis=-1, keepdim=True), 1.0e-20)
        selected = ml.divide(selected, denominator)
    weights = ml.multiply(selected, scale)
    return weights, ml.cast(indices, ml.int64)


def main() -> None:
    """Dump source-mapped semantic IR and execute CPU references for both branches."""
    logits = torch.tensor([[0.1, -0.7, 1.2, 0.6], [0.8, 0.3, -0.4, 1.7]], device="cpu")
    bias = torch.tensor([0.0, 1.0, -1.0, 0.2], device="cpu")
    for count in (1, 3):
        print(_route.explain(k=count, scale=2.5))
        weights, indices = _route.interpret(logits, bias, count, 2.5)
        print(f"k={count}, weights={weights.tolist()}, indices={indices.tolist()}")


if __name__ == "__main__":
    main()
