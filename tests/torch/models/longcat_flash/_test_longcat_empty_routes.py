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
"""Distributed empty-source and identity-only expert lifecycle contracts."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.dtensor import DTensor
from tests.torch.models.longcat_flash._test_longcat_flash import _build_case, _close_case
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime


def _check_case(experts: torch.nn.Module, reference: torch.nn.Module, rank: int,
                dtype: torch.dtype, identity_only: bool) -> dict:
    """Check fixed diagnostic routes without changing the learned-routing precision gate."""
    experts.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    device = experts.down_proj.device
    hidden = torch.randn(2, 128, dtype=dtype, device=device, requires_grad=True)
    source = hidden.detach().clone().requires_grad_()
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]], device=device, requires_grad=True)
    probabilities = weights.detach().clone().requires_grad_()
    routes = [[8, 9, 10], [9, 10, 11]] if identity_only or rank == 0 else [[0, 4, 8], [1, 5, 9]]
    ids = torch.tensor(routes, device=device)
    expected, actual = reference(source, ids, probabilities), experts(hidden, ids, weights)
    objective = torch.randn_like(actual)
    (expected.float() * objective.float()).sum().backward()
    (actual.float() * objective.float()).sum().backward()
    failures = []
    for label, value, target in (("output", actual, expected), ("input_gradient", hidden.grad, source.grad),
                                  ("probability_gradient", weights.grad, probabilities.grad)):
        try:
            torch.testing.assert_close(value, target, atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
                                       rtol=2e-2 if dtype == torch.bfloat16 else 2e-4)
        except AssertionError as error:
            failures.append(label + ": " + str(error))
    for name, parameter in experts.named_parameters():
        gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        if identity_only:
            if gradient is not None:
                failures.append(name + " must remain unused on a globally identity-only call")
        elif gradient is None or not bool(torch.isfinite(gradient).all()):
            failures.append(name + " must receive finite gradients while serving remote real assignments")
    return {"globally_identity_only": identity_only, "rank_has_no_real_source": identity_only or rank == 0,
            "max_output_error": float((actual.detach() - expected.detach()).abs().max()), "failures": failures}


def test_longcat_empty_routes() -> None:
    """Feature: Real-expert empty-source lifecycle.

    Description: Rank zero has no real source rows while both owners serve rank one.
        A second call selects only identities on every rank.
    Expectation: Every backward terminates and remote owner gradients are finite.
        Globally unused expert parameters have no gradient.
    """
    device_type = os.getenv("HP_LONGCAT_DEVICE", "npu")
    backend = os.getenv("HP_LONGCAT_BACKEND", "megamoe")
    mode = os.getenv("HP_LONGCAT_DISPATCH", "push")
    if device_type == "npu":
        import torch_npu  # pylint: disable=C0415,unused-import

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl" if device_type == "npu" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if device_type == "npu" else torch.device("cpu")
    if backend == "megamoe":
        start_shmem_lifetime()
    model, reference, _, dtype = _build_case(backend, device_type, mode, world, False, device)
    cases = [_check_case(model.model.layers[0].mlp.experts, reference.model.layers[0].mlp.experts,
                         rank, dtype, identity_only) for identity_only in (False, True)]
    output = Path(os.environ.get("HP_LONGCAT_REPORT_DIR", "/tmp/longcat-flash-reports"))
    output.mkdir(exist_ok=True, parents=True)
    report = {"rank": rank, "backend": backend, "dispatch": mode, "cases": cases,
              "scope": "fixed-route lifecycle diagnostic; separate from learned-routing precision"}
    (output / f"empty-{backend}-{mode}-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    _close_case(model, [])
    assert not any(case["failures"] for case in cases), report
