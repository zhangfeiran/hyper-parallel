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
"""Distributed complete LongCat Flash parity through the production recipe and planner."""

import copy
from dataclasses import replace as dataclass_replace
import json
import os
from functools import partial
from typing import Any
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from hyper_parallel.models.longcat_flash.adapter.layout import ExpertOwnershipPacking
from hyper_parallel.models.longcat_flash.adapter.materialization import register_materialization
from hyper_parallel.models.longcat_flash.adapter.validation import cropped_config
from hyper_parallel.models.longcat_flash.modeling_longcat_flash import build_longcat_flash
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.parallelism import entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime


def _resolve_entries(backend: str, mode: str, device_type: str) -> list:
    """Resolve actual recipe actions with explicit test-only backend selection."""
    root = Path(__file__).resolve().parents[4]
    recipe = parse_training_args([str(root / "hyper_parallel/models/longcat_flash/recipes/train.yaml")])
    entries = list(recipe.plan_overrides)
    for index, entry in enumerate(entries):
        if entry.replace is not None and entry.replace.kind == "module":
            implementation = entry.replace.implementation
            if implementation.callable is LongcatFlashMegaMoeExperts:
                entries[index] = dataclass_replace(entry, replace=dataclass_replace(
                    entry.replace, implementation=implementation.replace(backend=backend, dispatch_mode=mode,
                                                        expert_weight_layout=os.getenv("HP_LONGCAT_LAYOUT", "EOI"))))
    if device_type == "cpu":
        entries = [entry for entry in entries if not (
            entry.replace is not None and entry.replace.kind == "module"
            and entry.replace.implementation.callable.__name__ == "RMSNorm")]
    return entries


def _build_case(backend: str, device_type: str, mode: str, world: int,
                full_depth: bool, device: torch.device) -> tuple:
    """Build the complete model through recipe replacements and actual sharding."""
    config = cropped_config(num_layers=28 if full_depth else 2)
    if not full_depth:
        config.update(hidden_size=128, ffn_hidden_size=256, q_lora_rank=64, kv_lora_rank=32,
                      qk_nope_head_dim=32, qk_rope_head_dim=32, v_head_dim=32, head_dim=32,
                      expert_ffn_hidden_size=128, n_routed_experts=8, zero_expert_num=4, moe_topk=3)
    torch.manual_seed(42)
    model = build_longcat_flash(config)
    reference = copy.deepcopy(model)
    entries = _resolve_entries(backend, mode, device_type)
    reference_mode = os.getenv("HP_LONGCAT_REFERENCE", "hf")
    if reference_mode in ("same_dense", "native"):
        reference_entries = _resolve_entries("native", mode, device_type) if reference_mode == "native" else entries
        dense_entries = [entry for entry in reference_entries if reference_mode == "native" or not (
            entry.replace is not None and entry.replace.kind == "module"
            and entry.replace.implementation.callable is LongcatFlashMegaMoeExperts)]
        dense_rules = entries_to_module_replacements(dense_entries)
        reference_context = {"ep_size": world, "ep": world > 1} if reference_mode == "native" else {}
        apply_module_replacements(reference, compile_module_replacements(reference, dense_rules),
                                  context=reference_context, weights_mapping=[])
        if reference_mode == "native":
            register_materialization(reference)
    replacements = entries_to_module_replacements(entries)
    conversions = []
    apply_module_replacements(model, compile_module_replacements(model, replacements),
                              context={"ep_size": world, "ep": world > 1}, weights_mapping=conversions)
    model._weight_conversions = conversions  # pylint: disable=protected-access
    register_materialization(model)
    dtype = torch.bfloat16 if device_type == "npu" else torch.float32
    model, reference = model.to(device=device, dtype=dtype), reference.to(device=device, dtype=dtype)
    mesh = MeshContext(dp_size=world, dp_shard_size=world, ep_size=world)
    mesh.build_meshs(device_type, world)
    overrides = entries_to_plan_overrides(entries, ep_size=world)
    plan = ShardingPlanner(plan_overrides=overrides).plan(model, mesh.device_mesh, ep_size=world)
    apply_sharding_plan(model, plan, mesh)
    if reference_mode == "native":
        reference_plan = ShardingPlanner(plan_overrides=overrides).plan(reference, mesh.device_mesh, ep_size=world)
        apply_sharding_plan(reference, reference_plan, mesh)
    return model, reference, config, dtype


def _capture_gradient(gradient: torch.Tensor, *, captures: dict, side: str, name: str) -> None:
    """Retain the first backward value without changing the diagnostic graph."""
    captures[side].setdefault(name, gradient.detach().float())


def _capture_inputs(module: Any, args: tuple, *, captures: dict, side: str, fqn: str) -> None:
    """Retain exact synthetic inputs for first-divergence diagnosis."""
    del module
    for index, argument in enumerate(args):
        name = fqn + f".input{index}"
        if name not in captures[side] and argument.requires_grad:
            argument.register_hook(partial(_capture_gradient, captures=captures, side=side, name=name + ".grad"))
        captures[side].setdefault(name, argument.detach().float())


def _capture_output(module: Any, args: tuple, output: Any, *,
                    captures: dict, side: str, fqn: str) -> None:
    """Retain outputs without changing the executed module's return value."""
    del module, args
    tensor = output[0] if isinstance(output, tuple) else output
    if fqn not in captures[side] and tensor.requires_grad:
        tensor.register_hook(partial(_capture_gradient, captures=captures, side=side, name=fqn + ".grad"))
    captures[side].setdefault(fqn, tensor.detach().float())


def _install_diagnostics(model: Any, reference: Any, enabled: bool) -> tuple:
    """Attach diagnostic hooks only when explicitly selected."""
    captures, handles = {"reference": {}, "candidate": {}}, []
    if enabled:
        suffixes = (".mlp", ".mlp.experts", ".mlp.router", ".self_attn.0", ".self_attn.1", "._executor")
        for side, network in (("reference", reference), ("candidate", model)):
            for fqn, module in network.named_modules():
                if not fqn.endswith(suffixes):
                    continue
                options = {"captures": captures, "side": side, "fqn": fqn}
                if fqn.endswith((".mlp.experts", "._executor")):
                    handles.append(module.register_forward_pre_hook(partial(_capture_inputs, **options)))
                handles.append(module.register_forward_hook(partial(_capture_output, **options)))
    return captures, handles


def _reference_gradient(name: str, parameters: dict) -> Any:
    """Map fused projections back to their independent HF parameter gradients."""
    if name in parameters:
        return parameters[name].grad
    if "linear_qkv" in name:
        names = (name.replace("linear_qkv", "q_a_proj"), name.replace("linear_qkv", "kv_a_proj_with_mqa"))
        return torch.cat(tuple(parameters[key].grad for key in names))
    if "linear_fc1" in name:
        names = (name.replace("linear_fc1", "gate_proj"), name.replace("linear_fc1", "up_proj"))
        return torch.cat(tuple(parameters[key].grad for key in names))
    return parameters[name.replace("linear_fc2", "down_proj")].grad


def _expert_reference_gradient(name: str, parameters: dict, gradient: Any) -> Any:
    """Agree on unused experts before reducing locally absent reference gradients."""
    parameter = parameters[name]
    present = torch.tensor(int(gradient is not None), dtype=torch.int64, device=parameter.device)
    dist.all_reduce(present)
    if not bool(present):
        return None
    gradient = torch.zeros_like(parameter) if gradient is None else gradient.clone()
    dist.all_reduce(gradient)
    return gradient


def _expected_gradient(name: str, parameters: dict, config: dict, world: int, rank: int) -> Any:
    """Resolve physical owner gradients or reduce and pack the unsharded HF oracle."""
    gradient = _reference_gradient(name, parameters)
    sharded_reference = ".experts." in name and os.getenv("HP_LONGCAT_REFERENCE", "hf") == "native"
    if ".experts." in name and not sharded_reference:
        gradient = _expert_reference_gradient(name, parameters, gradient)
    if isinstance(gradient, DTensor):
        gradient = gradient.to_local()
    if gradient is None:
        return None
    if ".experts." in name and not sharded_reference:
        if name.endswith("gate_up_proj"):
            gradient = ExpertOwnershipPacking(config["n_routed_experts"],
                                               config["zero_expert_num"], world).apply(gradient)
        gradient = gradient.chunk(world)[rank]
    return gradient


def _gradient_metrics(model: Any, reference: Any, config: dict, world: int,
                      rank: int, dtype: torch.dtype, rtol: float) -> tuple:
    """Collect all failures so every rank executes the same collective sequence."""
    observed, relative_errors, failures = {}, {}, {}
    parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        gradient = _expected_gradient(name, parameters, config, world, rank)
        if gradient is None:
            if parameter.grad is not None:
                failures[name] = "unexpected gradient for an unused parameter"
            continue
        actual = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        if actual is None:
            failures[name] = "missing gradient for a globally active parameter"
            continue
        atol = 2e-5 if dtype == torch.bfloat16 else 2e-6
        try:
            torch.testing.assert_close(actual, gradient, atol=atol, rtol=rtol)
        except AssertionError as error:
            failures[name] = str(error)
        delta = actual.detach().float() - gradient.detach().float()
        reference_norm = torch.linalg.vector_norm(gradient.float())  # pylint: disable=not-callable
        relative = float(torch.linalg.vector_norm(delta) / reference_norm.clamp_min(1e-12))  # pylint: disable=not-callable
        if relative > (0.05 if dtype == torch.bfloat16 else 2e-4):
            failures[name] = failures.get(name, "") + f" relative_L2={relative}"
        observed[name], relative_errors[name] = float(delta.abs().max()), relative
    return observed, relative_errors, failures


def _write_sample(captures: dict, reference: Any, output: Path, rank: int, mode: str, backend: str) -> None:
    """Preserve the first expert's inputs, outputs and original weights for replay."""
    first = "model.layers.0.mlp.experts"
    sample = {"reference": {name: value.cpu() for name, value in captures["reference"].items()
                             if name.startswith(first)},
              "candidate": {name: value.cpu() for name, value in captures["candidate"].items()
                             if name.startswith(first)},
              "weights": {name: value.detach().cpu() for name, value
                          in reference.get_submodule(first).state_dict().items()}}
    torch.save(sample, output / f"{backend}-{mode}-rank{rank}-first-expert.pt")


def _close_case(model: Any, handles: list) -> None:
    """Release every local runtime and converge all ranks before reporting failures."""
    for module in model.modules():
        if isinstance(module, LongcatFlashMegaMoeExperts):
            module.close_runtime()
    dist.barrier()
    dist.destroy_process_group()
    for handle in handles:
        handle.remove()


def test_longcat_flash_ep() -> None:
    """Feature: Complete LongCat Flash distributed expert execution.

    Description: Compare the final recipe/planner model against independently executed HF mathematics.
    Expectation: Outputs and every parameter gradient satisfy the declared precision limits on every rank.
    """
    backend = os.getenv("HP_LONGCAT_BACKEND", "megamoe")
    device_type = os.getenv("HP_LONGCAT_DEVICE", "npu")
    mode = os.getenv("HP_LONGCAT_DISPATCH", "push")
    full_depth = os.getenv("HP_LONGCAT_FULL_DEPTH", "0") == "1"
    if device_type == "npu":
        import torch_npu  # pylint: disable=C0415,unused-import

        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl" if device_type == "npu" else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(device_type, int(os.environ["LOCAL_RANK"])) if device_type == "npu" else torch.device("cpu")
    if backend == "megamoe":
        start_shmem_lifetime()
    model, reference, config, dtype = _build_case(backend, device_type, mode, world, full_depth, device)
    torch.manual_seed(100 + rank)
    tokens = torch.randint(0, config["vocab_size"], (1, 4), device=device)
    diagnostic = os.getenv("HP_LONGCAT_DIAGNOSTIC", "0") == "1"
    captures, handles = _install_diagnostics(model, reference, diagnostic)
    reference_output = reference(tokens, labels=tokens, use_cache=False)
    candidate_output = model(tokens, labels=tokens, use_cache=False)
    atol, rtol = (0.02, 0.05) if dtype == torch.bfloat16 else (2e-5, 2e-4)
    precision_failures = {}
    try:
        torch.testing.assert_close(candidate_output.logits, reference_output.logits, atol=atol, rtol=rtol)
    except AssertionError as error:
        precision_failures["logits"] = str(error)
    reference_output.loss.backward()
    candidate_output.loss.backward()
    observed, relative_errors, gradient_failures = _gradient_metrics(model, reference, config, world, rank, dtype, rtol)
    precision_failures.update(gradient_failures)
    if device_type == "npu":
        torch.npu.synchronize()
    report = {"rank": rank, "world_size": world, "backend": backend, "dispatch": mode,
              "reference": os.getenv("HP_LONGCAT_REFERENCE", "hf"), "depth": config["num_layers"],
              "real_experts": config["n_routed_experts"], "zero_experts": config["zero_expert_num"],
              "top_k": config["moe_topk"], "loss": float(candidate_output.loss.detach()),
              "reference_loss": float(reference_output.loss.detach()),
              "logit_max_abs": float((candidate_output.logits.detach().float()
                                      - reference_output.logits.detach().float()).abs().max()),
              "parameter_gradient_max_abs": observed, "parameter_gradient_relative_l2": relative_errors,
              "precision_failures": precision_failures,
              "intermediate_max_abs": {name: float((value - captures["reference"][name]).abs().max())
                                       for name, value in captures["candidate"].items()
                                       if name in captures["reference"]},
              "status": "FAIL" if precision_failures else "PASS"}
    output = Path(os.environ.get("HP_LONGCAT_REPORT_DIR", "/tmp/longcat-flash-reports"))
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{backend}-{mode}-rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    if diagnostic:
        _write_sample(captures, reference, output, rank, mode, backend)
    print(json.dumps(report), flush=True)
    _close_case(model, handles)
    if precision_failures:
        raise AssertionError(f"strict gradient precision failed for {len(precision_failures)} parameters")
