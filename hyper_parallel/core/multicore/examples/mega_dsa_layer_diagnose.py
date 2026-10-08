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
"""Explicit offline layer diagnosis; measurements do not change model acceptance."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

# These source helpers are reused to keep the diagnostic projection formulas
# identical to the model; there is no public projection/VJP capture API.
from hyper_parallel.components.modules.dsa_attention import (
    _restore_attention_projection,
)
from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.examples.mega_dsa_cann_validate import _metrics
from hyper_parallel.core.multicore.examples.mega_dsa_model_common import (
    _OracleDsaReference,
    _rotary_mul_oracle,
    build_model_fixture,
    model_inputs,
)
from hyper_parallel.core.multicore.examples.mega_dsa_model_validate import (
    _compare,
    _cpu_case,
    _device_case,
    _ObservedReference,
    _StockReference,
)
from hyper_parallel.core.multicore.examples.mega_dsa_restore_diagnose import (
    _restoration_diagnosis,
    _restoration_replay,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import CannDsaLayout
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import (
    DsaBatchMeta,
    DsaLossNormalization,
)

_MAIN = ("q_nope", "compressed_kv", "q_rope", "k_rope")
_INDEX = ("index_q", "index_k", "merge_weight")


def _project(model, hidden, embeddings):
    residual, main_states, weight = model._prepare_attention_states(hidden, embeddings)
    index = model._project_index_states(hidden, residual, embeddings)
    main_states = (main_states[0].flatten(0, 1), main_states[1].flatten(0, 1)[:, 0],
                   main_states[2].flatten(0, 1), main_states[3].flatten(0, 1)[:, 0])
    index = (index[0].flatten(0, 1), index[1].flatten(0, 1)[:, 0], index[2].flatten(0, 1))
    return main_states, index, weight


def _restore_states(model, sparse, hidden_shape, weight):
    restored = _restore_attention_projection(
        sparse.reshape(*hidden_shape[:2], model.num_heads, model.kv_lora_rank),
        weight[:, model.qk_nope_head_dim:].transpose(1, 2), num_heads=model.num_heads,
        batch_size=hidden_shape[0], seq_length=hidden_shape[1], kv_lora_rank=model.kv_lora_rank,
        value_head_dim=model.v_head_dim,
    )
    return restored, model.o_proj(restored)


def _snapshot(values):
    return tuple(value.detach().float().cpu() for value in values)


def _named_gradients(names, gradients):
    return {name: None if value is None else value.detach().float().cpu()
            for name, value in zip(names, gradients)}


def _projection_vjp(states, cotangents, hidden, model):
    names = ("hidden", *dict(model.named_parameters()))
    gradients = torch.autograd.grad(states, (hidden, *model.parameters()), grad_outputs=cotangents,
                                    retain_graph=True, allow_unused=True)
    return _named_gradients(names, gradients)


def _measure_projection(model, hidden, embeddings, backend, *, auxiliary_scale):
    """Capture actual model states, complete gradients and fixed-cotangent projection VJPs."""
    main_states, index, weight = _project(model, hidden, embeddings)
    selection = backend.indexer(*index)
    sparse, stats = backend.attention(*main_states, selection)
    loss = backend.kl_loss(*index, main_states, selection, stats,
                           normalization=DsaLossNormalization(backend.layout.batch_meta.global_valid_queries),
                           loss_coeff=model.dsa_loss_coeff)
    restored, output = _restore_states(model, sparse, hidden.shape, weight)
    objective = output.float().square().mean() * 13 + loss * auxiliary_scale
    targets = (*main_states, *index, sparse, restored, output, hidden, *model.parameters())
    gradients = torch.autograd.grad(objective, targets, retain_graph=True, allow_unused=True)
    boundary = gradients[:7]
    return {
        "states": _snapshot((*main_states, *index)), "cotangents": _snapshot(boundary),
        "sparse_cotangent": gradients[7].detach().float().cpu(),
        "restoration": {"sparse_input": sparse.detach().float().cpu(), "hidden_shape": list(hidden.shape),
                        "restored_input": restored.detach().float().cpu(),
                        "restored_cotangent": gradients[8].detach().float().cpu(),
                        "output_cotangent": gradients[9].detach().float().cpu()},
        "full": {"output": output.detach().float().cpu(), "kl_loss": loss.detach().float().cpu(),
                 **_named_gradients(("hidden", *dict(model.named_parameters())), gradients[10:])},
        "main_projection_vjp": _projection_vjp(main_states, boundary[:4], hidden, model),
        "index_projection_vjp": _projection_vjp(index, boundary[4:], hidden, model),
        "indices": selection.to_global_indices().detach().cpu(),
    }


def _operator_replay(values, cotangent, backend, indices, *, auxiliary_scale):
    """Use identical projected values, selection and output cotangent for every backend."""
    cpu = backend.layout.device.type == "cpu"
    dtype = torch.float32 if cpu else torch.bfloat16
    leaves = tuple(value.to(device=backend.layout.device, dtype=dtype).detach().requires_grad_() for value in values)
    main_states, index = leaves[:4], leaves[4:]
    selection = backend.prepare_selection(indices)
    output, stats = backend.attention(*main_states, selection)
    gradient = cotangent.to(device=backend.layout.device, dtype=dtype)
    main_grad = torch.autograd.grad(output, main_states, grad_outputs=gradient)
    loss = backend.kl_loss(*index, main_states, selection, stats,
                           normalization=DsaLossNormalization(backend.layout.batch_meta.global_valid_queries),
                           loss_coeff=0.3)
    index_grad = torch.autograd.grad(loss * auxiliary_scale, index)
    return {"output": output.detach().float().cpu(), "kl_loss": loss.detach().float().cpu(),
            **_named_gradients((*_MAIN, *_INDEX), (*main_grad, *index_grad))}


def _pair_metrics(actual, expected):
    availability = {name: (value is None) == (expected[name] is None) for name, value in actual.items()}
    measurements = {name: _metrics(value, expected[name]) for name, value in actual.items()
                    if value is not None and expected[name] is not None}
    return {"gradient_availability": availability, "measurements": measurements}


def _cpu_projection(template, hidden, embeddings, meta, indices, cotangents):
    model = copy.deepcopy(template).float()
    hidden = hidden.float().requires_grad_()
    embeddings = tuple(value.float() for value in embeddings)
    with patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
               side_effect=_rotary_mul_oracle):
        main_states, index, _ = _project(model, hidden, embeddings)
        main_vjp = _projection_vjp(main_states, cotangents[:4], hidden, model)
        index_vjp = _projection_vjp(index, cotangents[4:], hidden, model)
        backend = _OracleDsaReference(meta, attention_scale=model.scaling, indices=indices)
        full = _measure_projection(model, hidden, embeddings, backend, auxiliary_scale=7)
    return main_vjp, index_vjp, full


def _hidden_error_controls(template, hidden, embeddings, meta, native, cpu, operator_oracle):
    """Measure cumulative FP32 replays against the FP32 absorbed hidden gradient.

    These controls change several precision boundaries and are not additive
    error attribution or a production implementation. Index/KL is detached
    from the hidden path, so the controls only compare the main objective.
    """
    model = copy.deepcopy(template).float()
    hidden = hidden.float().requires_grad_()
    with patch("hyper_parallel.components.functional.rotary_embedding.torch_npu.npu_rotary_mul",
               side_effect=_rotary_mul_oracle):
        states, _, weight = _project(model, hidden, tuple(value.float() for value in embeddings))
    backend = _OracleDsaReference(meta, attention_scale=model.scaling, indices=native["indices"])
    restoration = _restoration_replay(model, native["restoration"], "combined")
    restored_vjp = _operator_replay(native["states"], restoration["input_cotangent"], backend,
                                    native["indices"], auxiliary_scale=7)
    sparse = operator_oracle["output"].detach().requires_grad_()
    _, output = _restore_states(model, sparse, hidden.shape, weight)
    downstream_cotangent = torch.autograd.grad(output.square().mean() * 13, sparse)[0]
    downstream = _operator_replay(native["states"], downstream_cotangent, backend,
                                 native["indices"], auxiliary_scale=7)
    controls = {
        "fp32_projection_jacobian_native_cotangents": native["cotangents"][:4],
        "fp32_projection_and_sparse_vjp": tuple(operator_oracle[name] for name in _MAIN),
        "fp32_projection_sparse_and_restoration_vjp": tuple(restored_vjp[name] for name in _MAIN),
        "fp32_downstream_on_bf16_projected_states": tuple(downstream[name] for name in _MAIN),
    }
    expected = cpu["full"]["hidden"]
    result = {"native_full_hidden": _metrics(native["full"]["hidden"], expected)}
    for name, cotangents in controls.items():
        gradient = torch.autograd.grad(states, hidden, grad_outputs=cotangents, retain_graph=True)[0]
        result[name] = _metrics(gradient, expected)
    return result


def _relu_crossings(native_states, fp32_states, meta):
    total = meta.global_valid_queries
    native = torch.einsum("thi,si->ths", native_states[4], native_states[5])
    fp32 = torch.einsum("thi,si->ths", fp32_states[4], fp32_states[5])
    tokens = torch.arange(total)
    starts = torch.tensor([meta.global_cu_seqlens[meta.sequence_position(q)[0]] for q in range(total)])
    legal = ((tokens[None] >= starts[:, None]) & (tokens[None] <= tokens[:, None]))[:, None]
    coordinates = (((native > 0) != (fp32 > 0)) & legal).nonzero().tolist()
    return [{"query": q, "head": h, "key": k, "bf16_projected_dot": float(native[q, h, k]),
             "fp32_projected_dot": float(fp32[q, h, k])} for q, h, k in coordinates]


def _save_snapshot(directory, seed, template, hidden, embeddings, native):
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{seed}.pt"
    torch.save({"seed": seed, "model_state_dict": template.state_dict(), "hidden": hidden,
                "position_embeddings": embeddings, "states": native["states"],
                "cotangents": native["cotangents"], "sparse_cotangent": native["sparse_cotangent"],
                "restoration": native["restoration"], "hidden_gradient": native["full"]["hidden"],
                "indices": native["indices"]}, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "explicit offline CPU capture; random model weights, inputs, projected states and cotangents"}


def _diagnose_seed(seed, device, meta, snapshot_dir):
    template = build_model_fixture(dtype=torch.bfloat16, seed=seed)
    hidden, embeddings = model_inputs(meta, dtype=torch.bfloat16, seed=seed + 1)
    source = copy.deepcopy(template).to(device)
    backend = _ObservedReference(CannDsaLayout(meta, device), attention_scale=source.scaling)
    native = _measure_projection(source, hidden.to(device).requires_grad_(),
                                 tuple(value.to(device) for value in embeddings), backend, auxiliary_scale=7)
    indices = native["indices"]
    full_boundary, _ = _device_case(template, hidden, embeddings, meta, "joint", 7)
    boundary_parity = _pair_metrics(native["full"], full_boundary)
    stock_backend = _StockReference(CannDsaLayout(meta, device), attention_scale=source.scaling, indices=indices)
    cpu_backend = _OracleDsaReference(meta, attention_scale=source.scaling, indices=indices)
    values, cotangent = native["states"], native["sparse_cotangent"]
    operator = _operator_replay(values, cotangent, backend, indices, auxiliary_scale=7)
    stock = _operator_replay(values, cotangent, stock_backend, indices, auxiliary_scale=7)
    oracle = _operator_replay(values, cotangent, cpu_backend, indices, auxiliary_scale=7)
    main_vjp, index_vjp, cpu = _cpu_projection(template, hidden, embeddings, meta, indices, native["cotangents"])
    unabsorbed = _cpu_case(template, hidden, embeddings, meta, indices, "joint", 7)
    state_metrics = {name: _metrics(actual, expected)
                     for name, actual, expected in zip((*_MAIN, *_INDEX), values, cpu["states"])}
    return {
        "snapshot": _save_snapshot(snapshot_dir, seed, template, hidden, embeddings, native),
        "manual_stage_vs_original_boundary": boundary_parity,
        "identical_state_operator_calibration": _compare(operator, stock, oracle),
        "enhance_vs_stock_identical_states": _pair_metrics(operator, stock),
        "projected_state_bf16_vs_fp32": state_metrics,
        "main_projection_fixed_cotangent_vjp": _pair_metrics(native["main_projection_vjp"], main_vjp),
        "index_projection_fixed_cotangent_vjp": _pair_metrics(native["index_projection_vjp"], index_vjp),
        "restoration_fixed_cotangent_vjp": _restoration_diagnosis(template, native, device, _pair_metrics),
        "hidden_precision_controls": _hidden_error_controls(template, hidden, embeddings, meta, native, cpu, oracle),
        "cpu_absorbed_vs_unabsorbed": _pair_metrics(cpu["full"], unabsorbed),
        "native_vs_fp32_absorbed": _pair_metrics(native["full"], cpu["full"]),
        "projection_relu_crossings": _relu_crossings(values, cpu["states"], meta),
    }


def run_diagnosis(report: dict, seeds: list[int], *, snapshot_dir: Path | None = None) -> None:
    """Execute one-card diagnosis without changing any frozen model calibration bounds."""
    # Optional extension is activated only for this explicitly selected NPU diagnostic.
    __import__("torch_npu")
    torch.npu.set_device(0)
    report["environment"] = probe_environment()
    report["device"] = torch.npu.get_device_name(0)
    meta = DsaBatchMeta.packed((3, 5))
    report["cases"] = {}
    for seed in seeds:
        report["stage"] = f"seed={seed}"
        report["cases"][str(seed)] = _diagnose_seed(seed, "npu:0", meta, snapshot_dir)
        torch.npu.synchronize()
        report["device_execution"] = True
        print(f"seed={seed}: measured", flush=True)
    report["stage"] = "complete"
    report["status"] = "diagnosis_measured"


def main() -> None:
    """Persist layer metrics and source identity; failed execution returns a nonzero exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261027, 20261028])
    parser.add_argument("--snapshot-dir", type=Path, help="optional offline CPU tensor captures")
    args = parser.parse_args()
    report = {"status": "running", "stage": "device_init", "device_execution": False,
              "model_acceptance_changed": False, "p0_complete": False, "performance_measurement": False,
              "seeds": args.seeds, "dtype": "bfloat16", "sequence_lengths": [3, 5], "grad_aux": 7,
              "scope": "random HP fixture; CP=TP=1; exact projected inputs and fixed cotangents; "
                       "diagnostic pointwise stock calibration does not replace frozen model acceptance",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "restoration_source_sha256": hashlib.sha256(
                  Path(__file__).with_name("mega_dsa_restore_diagnose.py").read_bytes()).hexdigest()}
    try:
        run_diagnosis(report, args.seeds, snapshot_dir=args.snapshot_dir)
    except Exception as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
