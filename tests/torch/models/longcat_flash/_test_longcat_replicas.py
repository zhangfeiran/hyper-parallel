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
"""Original-width persistent LongCat replica executors on four logical NPUs."""

from argparse import Namespace
from pathlib import Path

from tests.torch.models.longcat_flash._validate_longcat_replicas import \
    validate


def test_longcat_hot_replicas(tmp_path: Path) -> None:
    """Feature: Persistent EOI hot-expert replicas.

    Description: Keep eight real experts per rank through ten updates in both Push and Pull.
    Expectation: Every rank preserves strict native GMM parity and updated parameter hashes.

    Args:
        tmp_path: Pytest-owned external result directory for the worker.
    """
    for mode in ("push", "pull"):
        args = Namespace(backend="native", ep=4, tokens=2048, hidden=6144, intermediate=2048,
                         real=32, zero=16, top_k=12, swiglu_precision="fused", routing_backend="triton",
                         no_optimize_tasks=False, no_direct_weight_gradients=False,
                         initial_token_capacity=24576, output=tmp_path / mode)
        options = Namespace(dispatch_mode=mode, layout="EOI", budget=2, updates=10,
                            transport="shmem_signal_kernel_gradient_bands")
        validate(args, options)
