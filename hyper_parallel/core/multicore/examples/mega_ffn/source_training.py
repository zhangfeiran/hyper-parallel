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
"""原始 common/packed 完整模型的 Source-NK 独立训练与完整优化器步测量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.multicore.examples.mega_ffn import qwen_dense_benchmark as benchmark
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import QwenDenseConfig, QwenDenseModel
from hyper_parallel.core.multicore.examples.mega_ffn.source_contract import resident_source_entry
from hyper_parallel.core.multicore.examples.mega_ffn.source_layout import build_source_model, source_layout_tensors
from hyper_parallel.core.multicore.runtime.dense_execution import DenseExecutionConfig
from hyper_parallel.core.optimizer import get_hyper_optimizer


STATE_KINDS = ("parameters", "main", "exp_avg", "exp_avg_sq", "step")


@dataclass
class SourceTrainingWorkload:
    """独立持有完整模型、FP32 主参数、AdamW 历史及真实参数组计数。"""

    model: QwenDenseModel
    optimizer: Float16OptimizerWithFloat16Params

    def _owner(self, parameter):
        owners = [(leaf, group) for leaf in self.optimizer.chained_optimizers for group in leaf.param_groups
                  for item in group["params"] if item is parameter]
        if len(owners) != 1:
            raise ValueError("Every Source-NK main parameter must have exactly one AdamW group owner")
        return owners[0]

    def state(self, kind: str) -> dict[str, torch.Tensor]:
        """读取全部源布局状态；更新后缺失优化器矩时拒绝用零值替代。

        Args:
            kind: parameters、main、exp_avg、exp_avg_sq 或 step。

        Returns:
            原始三个 NK 投影及完整非 FFN 参数的状态视图。
        """
        if kind not in STATE_KINDS:
            raise ValueError("Unknown Source-NK optimizer state kind")
        values = {}
        for name, parameter in self.model.named_parameters():
            main_parameter = self.optimizer.optimizer_param_by_model_param.get(parameter, parameter)
            if kind in ("parameters", "main"):
                value = parameter if kind == "parameters" else main_parameter
            else:
                leaf, group = self._owner(main_parameter)
                counter = group.get("step", 0)
                if type(counter) not in (int,) or counter < 0:
                    raise ValueError("Source-NK AdamW requires an exact nonnegative parameter-group step")
                if kind == "step":
                    value = torch.tensor(counter, dtype=torch.int64)
                elif kind in leaf.state.get(main_parameter, {}):
                    value = leaf.state[main_parameter][kind]
                elif counter:
                    raise ValueError(f"Source-NK AdamW state is missing after an update: {name}: {kind}")
                else:
                    value = torch.zeros_like(main_parameter)
            values[name] = value.detach()
        return source_layout_tensors(self.model, values)

    def step(self, tokens: torch.Tensor, capture: bool) -> tuple:
        """执行前向、损失、完整反向与优化器更新，测量时不复制梯度。

        Args:
            tokens: 固定因果语言模型词元，同时作为标签。
            capture: 是否在更新前保留全部参数梯度以供数值比较。
        """
        self.optimizer.zero_grad(set_to_none=True)
        result = self.model(tokens, tokens)
        result["loss"].backward()
        gradients = None
        if capture:
            values = {}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    raise ValueError(f"Source-NK full training has a missing parameter gradient: {name}")
                values[name] = parameter.grad.detach().clone()
            gradients = source_layout_tensors(self.model, values)
        self.optimizer.step()
        return result["loss"].detach(), result["logits"].detach(), gradients

    def payloads(self) -> dict[str, object]:
        """保留每层每个形状的实际常驻前向与反向执行清单。"""
        return {f"layers.{index}.mlp": layer.mlp.execution_manifest()
                for index, layer in enumerate(self.model.layers) if hasattr(layer.mlp, "execution_manifest")}


def build_workload(config: QwenDenseConfig, baseline: str, candidate: bool, *, seed: int,
                   learning_rate: float, recompute: bool = False,
                   execution: DenseExecutionConfig | None = None) -> SourceTrainingWorkload:
    """只对齐初始化，分别创建原始模型或显式 Source-NK 候选及其优化器。

    Args:
        config: 完整模型结构，不删除注意力、嵌入、残差、归一化或语言模型头。
        baseline: 原始 common 或 packed 稠密数值契约。
        candidate: True 使用 Source-NK 替换；False 使用原始模型。
        seed: 两条独立轨迹共同的初始化种子。
        learning_rate: FP32 主参数 AdamW 学习率。
        recompute: 是否对 FFN 进行标准非重入重计算。
        execution: 候选执行配置及明确的缓存根目录。
    """
    if baseline not in ("common", "packed") or type(candidate) not in (bool,):
        raise ValueError("Source-NK training requires the original common or packed baseline")
    torch.manual_seed(seed)
    model = (build_source_model(config, baseline, recompute=recompute, execution=execution) if candidate else
             QwenDenseModel(config, baseline, recompute=recompute))
    model.to(device="npu:0", dtype=torch.bfloat16)
    inner = get_hyper_optimizer(model=model, muon_params=[], adamw_params=[{"params": list(model.parameters())}],
                                adamw_kwargs={"lr": learning_rate, "betas": (0.9, 0.999), "eps": 1e-8,
                                              "weight_decay": 0.01})
    return SourceTrainingWorkload(model, Float16OptimizerWithFloat16Params(inner, model))


def trajectory_checks(reference: SourceTrainingWorkload, actual: SourceTrainingWorkload,
                      reference_step: tuple, actual_step: tuple, expected_step: int) -> dict[str, object]:
    """按原有阈值核对八项独立状态，并拒绝两条轨迹共同停留在错误步数。

    Args:
        reference: 原始模型及其独立优化器。
        actual: 候选或原生对照及其独立优化器。
        reference_step: 原始损失、logits 和更新前梯度。
        actual_step: 候选损失、logits 和更新前梯度。
        expected_step: 当前应完成的实际优化器更新次数。
    """
    checks = {name: benchmark.compare_tensors({name: reference_step[index]}, {name: actual_step[index]}, 0.002)
              for index, name in enumerate(("loss", "logits"))}
    checks["gradients"] = benchmark.compare_tensors(reference_step[2], actual_step[2], 0.002)
    for kind in STATE_KINDS:
        left, right = reference.state(kind), actual.state(kind)
        checks[kind] = benchmark.compare_tensors(left, right, 0.002 if kind == "parameters" else
                                                 (0 if kind == "step" else 1e-8), rtol=0 if kind == "step" else 0.02)
        if kind == "step":
            expected = {name: torch.tensor(expected_step, dtype=torch.int64) for name in left}
            counters = [benchmark.compare_tensors(expected, values, 0, rtol=0) for values in (left, right)]
            checks[kind]["actual_update_count_passed"] = all(item["passed"] for item in counters)
            checks[kind]["passed"] &= checks[kind]["actual_update_count_passed"]
    return {"step": expected_step, "checks": checks,
            "passed": all(check["passed"] for check in checks.values())}


def resident_payloads(payloads: dict[str, object], baseline: str, layers: int) -> bool:
    """检查所有层的完整 Source-NK 设备执行，不用少数层代替完整模型。

    Args:
        payloads: JSON 往返后的逐层真实执行记录。
        baseline: common 四输入或 packed 三输入契约。
        layers: 完整模型配置中的层数。
    """
    return (layers > 0 and set(payloads) == {f"layers.{index}.mlp" for index in range(layers)}
            and all(entries and all(resident_source_entry(entry, baseline) for entry in entries)
                    for entries in payloads.values()))


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _identity(args, config, tokens):
    # 已有入口负责封存框架与供应商源码身份，此入口只补充明确的 Source-NK 数值契约。
    identity = benchmark._identity()  # pylint: disable=protected-access
    identity.update(source_training_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_weight_layout="nk", numerical_contract=args.baseline,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    workload = {"config": asdict(config), "seed": args.seed, "learning_rate": args.learning_rate,
                "optimizer": "AdamW with FP32 master parameters", "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.01, "recompute": args.recompute, "validation_steps": args.steps,
                "warmup": args.warmup, "iterations": args.iterations,
                "dense_execution": {"backend": args.dense_backend, "soc": args.dense_soc,
                                    "cann_root": str(args.cann_root) if args.cann_root else None},
                "tokens_sha256": hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                "tokens_shape": list(tokens.shape)}
    return {"identity": identity, "workload": workload}


def _build(args, config, candidate):
    execution = DenseExecutionConfig(backend=args.dense_backend, cann_root=args.cann_root,
                                     soc=args.dense_soc, cache_root=args.cache_root)
    return build_workload(config, args.baseline, candidate, seed=args.seed, learning_rate=args.learning_rate,
                          recompute=args.recompute, execution=execution)


def validate(args: argparse.Namespace, config: QwenDenseConfig, tokens: torch.Tensor) -> None:
    """保留初始状态与失败步骤，完成全部独立训练后才允许数值放行。

    Args:
        args: 验收 CLI 的配置与证据路径。
        config: 完整模型配置。
        tokens: 固定训练词元，不在后续步骤重新同步模型或优化器状态。
    """
    baseline, candidate = _build(args, config, False), None
    try:
        candidate = _build(args, config, args.backend == "mega_ffn")
        initial = {kind: benchmark.compare_tensors(baseline.state(kind), candidate.state(kind), 0, rtol=0)
                   for kind in STATE_KINDS}
        record = {"mode": "validate", "backend": args.backend, "baseline": args.baseline,
                  **_identity(args, config, tokens), "initial": initial, "steps": [], "expected_steps": args.steps,
                  "complete": False, "numerical_passed": all(check["passed"] for check in initial.values()),
                  "formal_original_training_accepted": False, "performance_accepted": False}
        _write(args.output, record)
        if not record["numerical_passed"]:
            raise RuntimeError("Source-NK initial model and optimizer states must match exactly")
        for index in range(args.steps):
            reference, actual = baseline.step(tokens, True), candidate.step(tokens, True)
            row = trajectory_checks(baseline, candidate, reference, actual, index + 1)
            record["steps"].append(row)
            record["numerical_passed"] &= row["passed"]
            record["native_payloads"] = {"baseline": baseline.payloads(), "candidate": candidate.payloads()}
            _write(args.output, record)
            if not row["passed"]:
                raise RuntimeError(f"Source-NK independent trajectory failed at step {index + 1}; evidence retained")
        record["complete"] = True
        _write(args.output, record)
    finally:
        baseline.model.close()
        if candidate is not None:
            candidate.model.close()


def measure(args: argparse.Namespace, config: QwenDenseConfig, tokens: torch.Tensor) -> None:
    """在全新进程中测量完整前向、损失、反向和 FP32 主参数 AdamW 更新。

    Args:
        args: 配置、预热次数、迭代次数及证据路径。
        config: 与独立轨迹验收一致的完整模型配置。
        tokens: 与全部 ABBA 进程一致的固定训练词元。
    """
    workload = _build(args, config, args.backend == "mega_ffn")
    try:
        identity = _identity(args, config, tokens)
        for _ in range(1 + args.warmup):
            workload.step(tokens, False)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        times = []
        started = time.time()
        for _ in range(args.iterations):
            start = time.perf_counter()
            workload.step(tokens, False)
            torch.npu.synchronize()
            times.append(1000 * (time.perf_counter() - start))
        finished = time.time()
        _write(args.output, {"mode": "measure", "backend": args.backend, "baseline": args.baseline, **identity,
                             "native_payloads": workload.payloads(), "measurement_window": [started, finished],
                             "samples_ms": times, "median_ms": statistics.median(times), "warmup": args.warmup,
                             "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                             "peak_reserved_bytes": torch.npu.max_memory_reserved(), "complete": True,
                             "formal_original_training_accepted": False, "performance_accepted": False})
    finally:
        workload.model.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """沿用完整模型配置，另行要求原始基线及显式 Source-NK 缓存根。

    Args:
        argv: 命令行参数；None 表示当前进程参数。
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-root", type=Path)
    cache, remaining = parser.parse_known_args(argv)
    args = benchmark.parse_args(remaining)
    args.cache_root = cache.cache_root
    if args.baseline not in ("common", "packed") or args.backend not in (args.baseline, "mega_ffn"):
        parser.error("Source-NK requires an original common/packed baseline and its matching native control")
    return args


def main() -> None:
    """从已有设备空闲启动器运行；显式确定性设置同时应用于原始模型与候选。"""
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(0)
    config = QwenDenseConfig(hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
                             max_seq_len=args.seq_len, vocab_size=args.vocab_size, num_layers=args.num_layers,
                             num_attention_heads=args.attention_heads, num_key_value_heads=args.key_value_heads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10000)
    tokens = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), generator=generator).to("npu:0")
    if args.mode == "validate":
        validate(args, config, tokens)
    else:
        measure(args, config, tokens)


if __name__ == "__main__":
    main()
