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
"""Original-width EP1 diagnosis of HF, native GMM and MegaMoE SwiGLU semantics."""

import argparse
import json
import os
import platform
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts

from hyper_parallel.core.multicore import shmem
from hyper_parallel.models.longcat_flash.adapter import fused_routing
from tests.torch.models.longcat_flash._benchmark_longcat_flash import _arguments, _build, _step
from tests.torch.models.longcat_flash._compare_longcat_flash import _values
from tests.torch.models.longcat_flash._precision import STRICT_GATE, compare_tensors, require_strict_pass
from tests.torch.models.longcat_flash._validate_longcat_route_migration import _DEFAULT_ORACLE, _frozen_frontend
from tests.torch.multicore._mega_moe_utils import environment_identity, start_shmem_lifetime


def _operands(module: torch.nn.Module, base: tuple) -> tuple:
    """Reuse fixed routes/upstream and clone leaves for one independent backward."""
    return (module, base[1].detach().clone().requires_grad_(), base[2],
            base[3].detach().clone().requires_grad_(), base[4])


def _snapshot(module: torch.nn.Module, base: tuple) -> dict:
    """Execute a module with the fixed operands and retain all results on CPU."""
    tensors = _operands(module, base)
    output = _step(*tensors, True)
    return {name: value.detach().cpu() for name, value in _values(tensors, output).items()}


def _record(records: list, name: str, expected: dict, actual: dict, primary: bool) -> None:
    """Separate same-semantics acceptance from literal-HF mismatch diagnostics."""
    comparison = {key: compare_tensors(value, expected[key]) for key, value in actual.items()}
    passed = all(value["strict_pass"] for value in comparison.values())
    records.append({"case": name, "primary": primary, "strict_pass": passed,
                    "all_ranks_strict_pass": passed, "comparison": comparison})
    print(json.dumps({"case": name, "primary": primary, "strict_pass": passed,
                      "relative_l2": {key: value["relative_l2"] for key, value in comparison.items()}}), flush=True)


def _identity_grid(native: tuple, candidate: tuple) -> tuple:
    """Isolate fused SwiGLU with exact identity projections at H6144/I2048."""
    with torch.no_grad():
        index = torch.arange(2048, device="npu")
        for tensors in (native, candidate):
            module = tensors[0]
            module.gate_up_proj.zero_()
            module.down_proj.zero_()
            for expert in range(8):
                module.gate_up_proj[expert, index, index] = 1
                module.gate_up_proj[expert, 2048 + index, 2048 + index] = 1
                module.down_proj[expert, index, index] = 1
    hidden = torch.zeros(257, 6144, device="npu", dtype=torch.bfloat16)
    values = torch.tensor([-80, -20, -10, -5, -2, -1, -0.1, 0, 0.1, 1, 2, 5, 10, 20, 80],
                          device="npu", dtype=torch.bfloat16)
    hidden[:, :2048] = values[torch.arange(2048, device="npu") % values.numel()]
    scales = torch.tensor([0.001, 1, 8, 64], device="npu", dtype=torch.bfloat16)
    hidden[:, 2048:4096] = scales[torch.arange(2048, device="npu") % scales.numel()]
    ids = torch.arange(12, device="npu").expand(257, -1)
    probabilities = torch.zeros(257, 12, device="npu", dtype=torch.float32)
    probabilities[:, 0] = 1
    upstream = torch.zeros_like(hidden)
    upstream[:, :2048] = 1
    return native[0], hidden, ids, probabilities, upstream


def _diagnose(args: argparse.Namespace, native: tuple, candidates: list, records: list) -> dict:
    """Triangulate literal HF and both activation modes using identical parameters."""
    config = LongcatFlashConfig.from_dict({"hidden_size": args.hidden, "expert_ffn_hidden_size": args.intermediate,
                                          "n_routed_experts": args.real, "zero_expert_num": args.zero,
                                          "moe_topk": args.top_k})
    with torch.device("meta"):
        hf = LongcatFlashExperts(config)
    hf.gate_up_proj, hf.down_proj = native[0].gate_up_proj, native[0].down_proj
    expected_hf = _snapshot(hf, native)
    _record(records, "HF_repeat", expected_hf, _snapshot(hf, native), True)
    with _frozen_frontend(_DEFAULT_ORACLE) as (old, provenance):
        for mode, candidate in zip(("hf", "fused"), candidates):
            native[0].swiglu_precision = mode
            with (patch.object(fused_routing, "OrderedGather", old.OrderedGather),
                  patch.object(fused_routing, "OrderedCombine", old.OrderedCombine)):
                expected = _snapshot(native[0], native)
            _record(records, f"native_GMM_{mode}_vs_original_HF", expected_hf, expected, False)
            _record(records, f"Mega_{mode}_vs_native_GMM_{mode}", expected, _snapshot(candidate[0], native), True)
            del expected
        del expected_hf
        base = _identity_grid(native, candidates[1])
        with (patch.object(fused_routing, "OrderedGather", old.OrderedGather),
              patch.object(fused_routing, "OrderedCombine", old.OrderedCombine)):
            expected = _snapshot(native[0], base)
        _record(records, "fixed_operand_fused_SwiGLU_grid", expected, _snapshot(candidates[1][0], base), True)
    return provenance


def main() -> None:
    """Write diagnostics even if a same-semantics primary comparison fails."""
    args = _arguments()
    if (args.ep, args.real, args.zero, args.hidden, args.intermediate, args.top_k) != (1, 8, 4, 6144, 2048, 12):
        raise ValueError("SwiGLU diagnosis requires EP1/E8/Z4/H6144/I2048/TopK12; only token count is configurable")
    if args.routing_backend != "triton":
        raise ValueError("SwiGLU diagnosis requires triton routing for the frozen independent native frontend")
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 1:
        raise ValueError("SwiGLU diagnosis must launch exactly one rank")
    report_path = args.output / "rank0.json"
    if report_path.exists():
        raise FileExistsError(f"Use a fresh output directory: {report_path}")
    start_shmem_lifetime()
    args.backend = "native"
    native, *candidates = [_build(args, rank) for _ in range(3)]
    for tensors, mode in zip(candidates, ("hf", "fused")):
        module = tensors[0]
        module.close_runtime()
        module.backend, module.swiglu_precision = "megamoe", mode
        module.configure(dist.group.WORLD, 1, args.top_k)
    records = []
    try:
        provenance = _diagnose(args, native, candidates, records)
    finally:
        for tensors in (native, *candidates):
            tensors[0].close_runtime()
    report = {"rank": 0, "shape": {key: getattr(args, key) for key in
              ("ep", "real", "zero", "hidden", "intermediate", "tokens", "top_k")},
              "records": records, "gate": STRICT_GATE, "reference_frontend": provenance,
              "host_arch": platform.machine(), "environment": environment_identity(),
              "shmem_after_close": shmem.debug_state(),
              "scope": "EP1 expert component; literal HF failures are diagnostics, not same-semantics acceptance"}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
    require_strict_pass([row for row in records if row["primary"]])


if __name__ == "__main__":
    main()
