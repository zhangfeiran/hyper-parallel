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
"""Independent dense-Qwen trajectories and complete optimizer-step performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig, QwenDenseModel
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig
from hyper_parallel.core.optimizer import get_hyper_optimizer


@dataclass
class Workload:
    """One model with independently owned FP32 master weights and optimizer history."""

    model: QwenDenseModel
    optimizer: Float16OptimizerWithFloat16Params


def canonical_tensors(values: dict[str, torch.Tensor], backend: str) -> dict[str, torch.Tensor]:
    """Normalize physical FFN parameter layouts for numerical comparisons.

    Args:
        values: Per-parameter tensors, gradients, main weights or moment tensors.
        backend: common, packed or mega_ffn physical layout.
    """
    if backend not in ("common", "packed", "mega_ffn"):
        raise ValueError("Unknown dense comparison backend")
    result = {}
    for name, value in values.items():
        if ".mlp." not in name:
            result[name] = value
            continue
        prefix, suffix = name.split(".mlp.", 1)
        if backend == "common":
            if suffix == "gate_proj.weight":
                up = values[prefix + ".mlp.up_proj.weight"]
                if value.ndim == 0:
                    if not torch.equal(value, up):
                        raise ValueError("Packed gate/up optimizer step counters must agree")
                    result[prefix + ".mlp.gate_up"] = value
                else:
                    result[prefix + ".mlp.gate_up"] = torch.cat((value.t(), up.t()), dim=1)
            elif suffix == "down_proj.weight":
                result[prefix + ".mlp.down"] = value.t()
            elif suffix != "up_proj.weight":
                raise ValueError(f"Unexpected common FFN state: {name}")
        elif backend == "packed":
            target = {"linear_fc1.weight": "gate_up", "linear_fc2.weight": "down"}[suffix]
            result[prefix + ".mlp." + target] = value.t()
        else:
            result[name] = value
    return result


def compare_tensors(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                    atol: float, rtol: float = 0.02) -> dict[str, object]:
    """Compare every canonical tensor under fixed component/optimizer tolerances.

    Args:
        reference: Independent baseline state in canonical layouts.
        candidate: Candidate state in the same logical layouts.
        atol: Absolute threshold for the component being compared.
        rtol: Relative threshold, zero for exact initialization and step counters.
    """
    if set(reference) != set(candidate):
        raise ValueError("Dense comparison requires identical logical state keys")
    failures, maximum, nonfinite = [], 0.0, False
    for name, expected in reference.items():
        actual = candidate[name]
        if actual.shape != expected.shape:
            raise ValueError(f"Dense comparison shape mismatch: {name}")
        delta = (actual.float() - expected.float()).abs()
        finite = torch.isfinite(delta).all()
        nonfinite |= not bool(finite)
        if bool(finite):
            maximum = max(maximum, float(delta.max()) if delta.numel() else 0.0)
        valid = torch.isfinite(actual).all() & torch.isfinite(expected).all()
        valid &= (delta <= atol + rtol * expected.float().abs()).all()
        if not bool(valid):
            failures.append(name)
    return {"passed": not failures, "failed_keys": failures, "max_abs": None if nonfinite else maximum,
            "rtol": rtol, "atol": atol}


def _build(config, backend, seed, learning_rate, recompute=False, execution=None):
    torch.manual_seed(seed)
    model = QwenDenseModel(config, backend, recompute=recompute, execution=execution).to(device="npu:0",
                                                                                    dtype=torch.bfloat16)
    optimizer = get_hyper_optimizer(model=model, muon_params=[], adamw_params=[{"params": list(model.parameters())}],
                                    adamw_kwargs={"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8,
                                                  "weight_decay": 0.01})
    return Workload(model, Float16OptimizerWithFloat16Params(optimizer, model))


def _state(workload, kind):
    values = {}
    for name, parameter in workload.model.named_parameters():
        if kind == "parameters":
            value = parameter
        else:
            main_parameter = workload.optimizer.optimizer_param_by_model_param.get(parameter, parameter)
            if kind == "main":
                value = main_parameter
            else:
                states = [optimizer.state.get(main_parameter, {})
                          for optimizer in workload.optimizer.chained_optimizers]
                state = next((entry for entry in states if kind in entry), {})
                value = state[kind] if kind in state else (main_parameter.new_zeros(()) if kind == "step"
                                                          else torch.zeros_like(main_parameter))
        values[name] = value.detach()
    return canonical_tensors(values, workload.model.backend)


def _step(workload, tokens, capture):
    workload.optimizer.zero_grad(set_to_none=True)
    result = workload.model(tokens, tokens)
    result["loss"].backward()
    gradients = None
    if capture:
        values = {name: parameter.grad.detach().clone() for name, parameter in workload.model.named_parameters()}
        gradients = canonical_tensors(values, workload.model.backend)
    workload.optimizer.step()
    return result["loss"].detach(), result["logits"].detach(), gradients


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _identity():
    root = Path(__file__).resolve().parents[5]
    core = root / "hyper_parallel/core/multicore"
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(core.rglob("*.py"))}
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent"))
    version_file = cann / "opp/version.info"
    vendor_libraries = Path(torch_npu.__file__).resolve().parent / "lib"
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_hashes": sources, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "vendor_library_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(vendor_libraries.glob("*.so"))},
            "cann_root": str(cann), "cann_version": version_file.read_text(encoding="utf-8")
            if version_file.is_file() else None, "device": torch.npu.get_device_name(),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "torch_threads": torch.get_num_threads()}


def _payloads(workload):
    return {f"layers.{index}.mlp": layer.mlp.execution_manifest()
            for index, layer in enumerate(workload.model.layers) if hasattr(layer.mlp, "execution_manifest")}


def _scalar(value):
    result = float(value)
    return result if math.isfinite(result) else None


def _trajectory_checks(baseline, candidate, reference, actual):
    reference_loss, reference_logits, reference_gradients = reference
    loss, logits, gradients = actual
    checks = {"logits": compare_tensors({"logits": reference_logits}, {"logits": logits}, 0.002),
              "gradients": compare_tensors(reference_gradients, gradients, 0.002),
              "loss": compare_tensors({"loss": reference_loss}, {"loss": loss}, 0.002)}
    for kind in ("parameters", "main", "exp_avg", "exp_avg_sq", "step"):
        checks[kind] = compare_tensors(_state(baseline, kind), _state(candidate, kind),
                                       0.002 if kind == "parameters" else (0 if kind == "step" else 1e-8),
                                       rtol=0 if kind == "step" else 0.02)
    return {"baseline_loss": _scalar(reference_loss), "candidate_loss": _scalar(loss), "checks": checks,
            "passed": all(check["passed"] for check in checks.values())}


def _validate(args, config, tokens):
    execution = _execution(args)
    baseline = _build(config, args.baseline, args.seed, args.learning_rate, args.recompute, execution)
    candidate = _build(config, args.backend, args.seed, args.learning_rate, args.recompute, execution)
    initial = {kind: compare_tensors(_state(baseline, kind), _state(candidate, kind), 0.0, rtol=0.0)
               for kind in ("parameters", "main", "exp_avg", "exp_avg_sq", "step")}
    record = {"mode": "validate", "identity": _identity(), "baseline": args.baseline, "backend": args.backend,
              "workload": _workload_identity(args, config, tokens), "initial": initial,
              "expected_steps": args.steps, "complete": False,
              "numerical_passed": all(check["passed"] for check in initial.values()), "steps": []}
    _write(args.output, record)
    try:
        _validation_steps(args, baseline, candidate, tokens, record)
    finally:
        baseline.model.close()
        candidate.model.close()


def _validation_steps(args, baseline, candidate, tokens, record):
    if not record["numerical_passed"]:
        raise RuntimeError("Dense validation initial logical model/optimizer states must match exactly")
    for step in range(args.steps):
        reference = _step(baseline, tokens, True)
        actual = _step(candidate, tokens, True)
        row = {"step": step + 1, **_trajectory_checks(baseline, candidate, reference, actual)}
        record["steps"].append(row)
        record["numerical_passed"] &= row["passed"]
        record["native_payloads"] = {"baseline": _payloads(baseline), "candidate": _payloads(candidate)}
        _write(args.output, record)
        if not row["passed"]:
            raise RuntimeError(f"Dense independent trajectory failed at step {step + 1}; evidence retained")
    record["complete"] = True
    _write(args.output, record)


def _workload_identity(args, config, tokens):
    return {"config": asdict(config), "seed": args.seed, "learning_rate": args.learning_rate,
            "optimizer": "AdamW with FP32 master parameters", "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.01,
            "recompute": args.recompute,
            "dense_execution": {"backend": args.dense_backend, "soc": args.dense_soc,
                                "cann_root": str(args.cann_root) if args.cann_root else None},
            "validation_steps": args.steps, "warmup": args.warmup, "iterations": args.iterations,
            "tokens_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
            "tokens_shape": list(tokens.shape)}


def _measure(args, config, tokens):
    workload = _build(config, args.backend, args.seed, args.learning_rate, args.recompute, _execution(args))
    torch.npu.synchronize()
    start = time.perf_counter()
    _step(workload, tokens, False)
    torch.npu.synchronize()
    first_step_ms = 1000 * (time.perf_counter() - start)
    for _ in range(args.warmup):
        _step(workload, tokens, False)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    times = []
    measurement_started = time.time()
    for _ in range(args.iterations):
        start = time.perf_counter()
        _step(workload, tokens, False)
        torch.npu.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    measurement_finished = time.time()
    record = {"mode": "measure", "identity": _identity(), "backend": args.backend,
              "workload": _workload_identity(args, config, tokens), "native_payloads": _payloads(workload),
              "first_step_ms": first_step_ms, "warmup": args.warmup, "samples_ms": times,
              "measurement_window": [measurement_started, measurement_finished],
              "median_ms": statistics.median(times), "peak_allocated_bytes": torch.npu.max_memory_allocated(),
              "peak_reserved_bytes": torch.npu.max_memory_reserved(), "complete": True}
    _write(args.output, record)
    workload.model.close()


def _execution(args):
    return DenseExecutionConfig(backend=args.dense_backend, cann_root=args.cann_root, soc=args.dense_soc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse validated model and acceptance settings.

    Args:
        argv: CLI argument list, or None to read the process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "measure"), required=True)
    parser.add_argument("--backend", choices=("common", "packed", "mega_ffn"), default="mega_ffn")
    parser.add_argument("--baseline", choices=("common", "packed", "mega_ffn"), default="packed")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--key-value-heads", type=int, default=4)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dense-backend", choices=("host_stream", "resident_tiles"), default="resident_tiles")
    parser.add_argument("--dense-soc", choices=("Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4"))
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.iterations, args.batch_size) <= 0 or args.warmup < 0:
        parser.error("steps, iterations and batch-size must be positive; warmup must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seq_len < 2:
        parser.error("learning-rate must be finite and positive; seq-len must be at least two")
    return args


def main() -> None:
    """Run one device-gated validation or fresh-process performance workload."""
    args = parse_args()
    torch.npu.set_device(0)
    config = QwenDenseConfig(hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
                             max_seq_len=args.seq_len, vocab_size=args.vocab_size, num_layers=args.num_layers,
                             num_attention_heads=args.attention_heads, num_key_value_heads=args.key_value_heads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10000)
    tokens = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), generator=generator).to("npu:0")
    if args.mode == "validate":
        _validate(args, config, tokens)
    else:
        _measure(args, config, tokens)


if __name__ == "__main__":
    main()
