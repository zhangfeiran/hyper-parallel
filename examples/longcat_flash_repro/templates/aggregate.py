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
"""Aggregate the predeclared experiment without dropping samples or changing gates."""
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GIB = 1024**3
_REPORT_PARAGRAPH_1 = (
    "单层训练保留词表 131072、H6144、专家 I2048、dense12288、64 heads、Q1536/KV512、128/64 专家（每卡 8/4）"
    + "和 Top-12；逻辑层数裁为 1，并随 EP 缩减全局专家池。EP16、dense FSDP16、expert FSDP1，序列 2048、全局 batch1"
    + "6。每个正式进程在五次预热更新后测十次更新；三种实现各两个独立进程。"
)
_REPORT_PARAGRAPH_2 = (
    "参数、优化器、scheduler、RNG 初始哈希一致；每步输入、路由和 LR 一致。HF RMSNorm 与 FP32 objective 均有实际运行记录，"
    + "FP32 main 参数在所有 rank 终态有限。所有 rank 的退出、NPU PID 所有权、SHMEM 释放和原始样本均经检查。"
)
_REPORT_PARAGRAPH_3 = (
    "组件只预计算 learned Router 决策，计时包括专家路由/FFN/通信/combine；单层训练包含 Router、attention、dense M"
    + "LP、checkpoint recompute、梯度同步、优化器与其余训练步工作。组件收益不能代替训练步收益，也不能外推到全 28 层预训练吞吐。"
)
_REPORT_PARAGRAPH_4 = (
    "实现包含精确任务范围、只清零空专家梯度、单专家 GMM 寻址、可选 BF16 路由融合以及完整 Gate/Up 梯度直接写回。原始 HF 参数名、unused "
    + "Gate/Up 行和 checkpoint 布局均保留；仍存在真实专家 [R,H] scratch，不声称消除 sender 扩展 HBM。无权重 K=1 实验"
    + "因没有清晰诊断收益而在冻结前撤下。"
)
_REPORT_PARAGRAPH_5 = (
    "[结构化结果](results.json)，[冻结方案](optimization_plan.yaml)，[精度明细](precision-results.js"
    + "on)，[组件原始样本](component/results.json)，[训练原始样本](trainer/performance-results.json)，"
    + "[源码和制品哈希](frozen-provenance.json)，[包内参考结果](__PACKAGE__/reference/results.json)。"
)
_REPORT_PARAGRAPH_6 = (
    "Original-width expert optimization and one-logical-layer training comparison. Bo"
    + "th the original native routing and an identical Triton-routing native GMM contro"
    + "l are retained. No sample is removed and no threshold is changed after candidate"
    + " timing."
)
_REPORT_PARAGRAPH_7 = (
    "See the structured results for every p50/p90, raw rank-max sample, peak allocato"
    + "r metric and predeclared decision. Each component implementation has three fresh"
    + " processes; each training implementation has two fresh formal processes."
)
_REPORT_PARAGRAPH_8 = (
    "The numerical oracle is native NPU GMM with original Torch routing. Full compone"
    + "nt outputs and all gradients are exact. Two candidates each complete ten consecu"
    + "tive updates on all 16 ranks with exact updated model/main hashes, equal replay "
    + "inputs/routing/LR and finite FP32 main parameters. Literal HF F.linear failures "
    + "from the prior original-width investigation remain separate."
)
_REPORT_PARAGRAPH_9 = (
    "Scope is an EP4 original-width expert component plus EP16/FSDP16 one-logical-lay"
    + "er scratch training. All original widths, expert cardinalities, vocabulary, sequ"
    + "ence and batch are retained. This is not a full-depth pretrained training claim."
)
_REPORT_PARAGRAPH_10 = (
    "[Structured results](results.json), [Chinese detailed tables](summary.zh-CN.md),"
    + " [frozen plan](optimization_plan.yaml), [source/binary provenance](frozen-proven"
    + "ance.json)."
)
_REPORT_PARAGRAPH_11 = (
    "Constant-local-eight-expert EP4/E32/Z16 and EP16/E128/Z64 one-layer training; sc"
    + "ratch checkpoint, not full pretrained depth"
)


def read(path: Path) -> Any:
    """Load a structured experiment artifact.

    Args:
        path: Artifact path to read or write.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def process_median(rows: Any, key: str) -> float:
    """Combine independent process statistics without discarding samples.

    Args:
        rows: Raw aligned per-step records.
        key: Field to combine across independent processes.
    """
    return statistics.median((row[key] for row in rows))


def decision(reference: Any, candidate: Any, threshold: Any) -> dict[str, Any]:
    """Apply the predeclared speed, tail and allocator gates.

    Args:
        reference: Aggregated reference implementation statistics.
        candidate: Aggregated candidate implementation statistics.
        threshold: Predeclared numerical or performance gates.
    """
    speedup = reference["p50"] / candidate["p50"]
    tail_ratio = candidate["p90"] / reference["p90"]
    memory_ratio = candidate["peak_allocated_bytes"] / reference["peak_allocated_bytes"]
    passed = (
        speedup >= threshold["minimum_speedup"]
        and tail_ratio <= 1 + threshold["maximum_p90_regression"]
        and (memory_ratio <= 1.05)
    )
    return {
        "reference_over_candidate": speedup,
        "latency_reduction_percent": 100 * (1 - 1 / speedup),
        "throughput_gain_percent": 100 * (speedup - 1),
        "p90_ratio": tail_ratio,
        "allocated_ratio": memory_ratio,
        "threshold": threshold,
        "verdict": "PERFORMANCE_PASS" if passed else "PERFORMANCE_FAIL",
    }


def component_summaries(cases: list, phase: str, tokens: int) -> dict:
    """Collect all formal process quantiles for one component phase.

    Args:
        cases: All formal component process records.
        phase: Forward or forward-backward phase.
        tokens: Fixed token count per rank.
    """
    summaries = {}
    for implementation in ["native_torch", "native_triton_prefix", "push_direct_gradient", "pull_direct_gradient"]:
        rows = [row for row in cases if row["tokens"] == tokens and row["implementation"] == implementation]
        assert len(rows) == 3
        values = [row["phases"][phase] for row in rows]
        assert all((len(value["rank_max_ms"]) == 30 for value in values))
        summaries[implementation] = {
            "p50": process_median(values, "p50_ms"),
            "p90": process_median(values, "p90_ms"),
            "peak_allocated_bytes": max((v["peak_allocated_bytes"] for v in values)),
            "peak_reserved_bytes": max((v["peak_reserved_bytes"] for v in values)),
            "shmem_heap_bytes": max((v["shmem_heap_bytes"] for v in values)),
            "raw_rank_max_ms": [v["rank_max_ms"] for v in values],
            "cases": [row["case"] for row in rows],
        }
    return summaries


def aggregate_component() -> list[dict[str, Any]]:
    """Summarize all formal expert component processes."""
    data = read(ROOT / "component/results.json")
    assert read(ROOT / "component/status.json")["status"] == "COMPLETE"
    cases = [row for row in data["cases"] if row["phase"] == "formal"]
    output = []
    tokens = 2048
    for phase in ["forward", "forward_backward"]:
        summaries = component_summaries(cases, phase, tokens)
        for name, value in summaries.items():
            row = {"tokens": tokens, "phase": phase, "implementation": name, **value}
            row["versus_native_torch"] = decision(
                summaries["native_torch"], value, data["thresholds"][str(tokens)][phase]
            )
            row["versus_native_triton"] = decision(
                summaries["native_triton_prefix"], value, data["thresholds"][str(tokens)][phase]
            )
            if name.startswith("native") or name == "push_ordinary_gradient":
                row["verdict"] = "REFERENCE_ABLATION" if name == "push_ordinary_gradient" else "REFERENCE"
            else:
                row["verdict"] = row["versus_native_triton"]["verdict"]
            output.append(row)
    return output


def aggregate_trainer() -> list[dict[str, Any]]:
    """Summarize both formal processes for each Trainer implementation."""
    assert read(ROOT / "trainer/status.json")["status"] == "COMPLETE"
    data = read(ROOT / "trainer/performance-results.json")
    threshold = read(ROOT / "trainer/pilot-thresholds.json")
    summaries = {}
    for name in ["native", "fast-native", "megamoe"]:
        values = [row for row in data["cases"] if row["implementation"] == name]
        assert len(values) == 2 and all((len(value["rank_max_seconds"]) == 10 for value in values))
        summaries[name] = {
            "p50": process_median(values, "p50_seconds"),
            "p90": process_median(values, "p90_seconds"),
            "tokens_per_second": 32768 / process_median(values, "p50_seconds"),
            "peak_allocated_bytes": max((v["peak_allocated_bytes"] for v in values)),
            "peak_reserved_bytes": max((v["peak_reserved_bytes"] for v in values)),
            "raw_rank_max_seconds": [v["rank_max_seconds"] for v in values],
            "cases": [value["case"] for value in values],
        }
    output = []
    for name, value in summaries.items():
        row = {"implementation": name, **value}
        row["versus_native_torch"] = decision(summaries["native"], value, threshold)
        row["versus_native_triton"] = decision(summaries["fast-native"], value, threshold)
        row["verdict"] = row["versus_native_triton"]["verdict"] if name == "megamoe" else "REFERENCE"
        output.append(row)
    return output


def numerical() -> dict[str, Any]:
    """Verify aligned updates and complete model and main parameter hashes."""
    module = read(ROOT / "precision-results.json")
    assert module["status"] == "PASS"
    results = []
    reference = read(ROOT / "trainer/numerical-native/summary.json")
    for name in ["megamoe", "fast-native"]:
        comparison = read(ROOT / "trainer" / f"numerical-{name}-comparison.json")
        candidate = read(ROOT / "trainer" / f"numerical-{name}/summary.json")
        assert comparison["status"] == "PASS" and comparison["consecutive_steps_per_rank"] == 10
        assert len(comparison["rows"]) == 160 and comparison["state_identical"]
        pairs = 0
        for rank in range(16):
            for a, b in zip(reference["steps"][rank], candidate["steps"][rank]):
                assert a["updated_model_sha256"] == b["updated_model_sha256"]
                assert a["updated_main_sha256"] == b["updated_main_sha256"]
                assert a["objective_dtypes"] == b["objective_dtypes"] == ["torch.float32"]
                pairs += len(a["updated_model_sha256"]) + len(a["updated_main_sha256"])
        results.append(
            {
                "candidate": name,
                "status": "PASS",
                "rows": 160,
                "consecutive_updates_per_rank": 10,
                "maximum_loss_absolute_delta": max((row["loss_absolute_delta"] for row in comparison["rows"])),
                "maximum_preclip_norm_absolute_delta": max(
                    (row["preclip_norm_absolute_delta"] for row in comparison["rows"])
                ),
                "exact_updated_model_main_hash_pairs": pairs,
                "evidence": "trainer/numerical-" + f"{name}" + "-comparison.json",
            }
        )
    return {
        "status": "PRECISION_PASS",
        "module": module,
        "trainer": results,
        "literal_hf_linear": "Prior original-width strict FAIL retained; primary oracle is native GMM",
        "literal_hf_evidence": "Historical HF eager diagnostic is separate; see package reference/README.md",
    }


def verify_frozen_source() -> dict[str, Any]:
    """Require source and payload identities to match the frozen records."""
    frozen = read(ROOT / "frozen-provenance.json")
    spec = importlib.util.spec_from_file_location("bench_helpers", ROOT / "component/helpers.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    assert helpers.seal() == frozen["source_and_payload_sha256"]
    for folder in ["component", "trainer"]:
        assert read(ROOT / folder / "source-seal.json") == frozen["source_and_payload_sha256"]
    return frozen


def module_table_rows(module: list) -> list[str]:
    """Format ordered module table rows without changing statistics.

    Args:
        module: Aggregated process records."""
    lines = []
    for row in module:
        lines.append(
            "| "
            + f"{row['tokens']}"
            + " | "
            + f"{row['phase']}"
            + " | "
            + f"{row['implementation']}"
            + " | "
            + f"{row['p50']:.3f}"
            + " | "
            + f"{row['p90']:.3f}"
            + " | "
            + f"{row['versus_native_torch']['reference_over_candidate']:.4f}"
            + "x | "
            + f"{row['versus_native_triton']['reference_over_candidate']:.4f}"
            + "x | "
            + f"{row['verdict']}"
            + " |"
        )
    return lines


def trainer_table_rows(trainer: list) -> list[str]:
    """Format ordered trainer table rows without changing statistics.

    Args:
        trainer: Aggregated process records."""
    lines = []
    for row in trainer:
        lines.append(
            "| "
            + f"{row['implementation']}"
            + " | "
            + f"{row['p50']:.6f}"
            + " | "
            + f"{row['p90']:.6f}"
            + " | "
            + f"{row['tokens_per_second']:.1f}"
            + " | "
            + f"{row['versus_native_torch']['reference_over_candidate']:.4f}"
            + "x | "
            + f"{row['versus_native_triton']['reference_over_candidate']:.4f}"
            + "x | "
            + f"{row['peak_allocated_bytes'] / GIB:.3f}"
            + " / "
            + f"{row['peak_reserved_bytes'] / GIB:.3f}"
            + " | "
            + f"{row['verdict']}"
            + " |"
        )
    return lines


def write_chinese_report(frozen: dict, module: list, trainer: list, precision: dict, performance: str) -> None:
    """Write the detailed result while keeping verdict computation separate.

    Args:
        frozen: Captured model and runtime identities.
        module: Component process summaries.
        trainer: Trainer process summaries.
        precision: Aligned numerical acceptance records.
        performance: Overall performance verdict.
    """
    lines = [
        f"PRECISION_PASS / {performance}",
        "",
        "已完成每卡八个真实专家的 LongCat 裁剪验证。下表同时列出原始原生实现和使用相同路由融合和优化 prefix 梯度的强化原生对照；正式结果与此前诊断样本分"
        + "开。",
        "",
        "源码提交：`"
        + frozen["candidate_commit"]
        + (
            "`。全部比较采用同权重、同输入、同学习路由、同拓扑、同 fused SwiGLU 语义（native 调用 npu_swiglu，MegaMoE 嵌入 CANN"
            + " 实现）。"
        ),
        "",
        "| 每卡 tokens | 阶段 | 实现 | p50 ms | p90 ms | 对原生原路由加速比 | 对强化原生加速比 | 强化对照验收 |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(module_table_rows(module))
    lines += [
        "",
        "每个组件实现包含三个独立进程，每阶段每进程预热 10 次、保留 30 个所有 rank 最大耗时样本。p50/p90 均为进程分位数的中位数。",
        "",
        "| 单层训练实现 | p50 s | p90 s | 全局 tokens/s | 对原生原路由加速比 | 对强化原生加速比 | 进程 allocated/res"
        + "erved 峰值 GiB | 强化对照验收 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    lines.extend(trainer_table_rows(trainer))
    lines += [
        "",
        _REPORT_PARAGRAPH_1,
        "",
        "精度验收以原生 NPU GMM 为准，保留原始 Torch 路由作为独立数学对照。所有组件输出、dX、dP、两组 dW 全值比较最大绝对误差为 0；空 sour"
        + "ce、空 owner、全零专家、跨网格启动分别通过。",
        "",
    ]
    for row in precision["trainer"]:
        lines.append(
            f"{row['candidate']}"
            + "：连续 10 次更新 × 16 rank，loss 最大差 "
            + f"{row['maximum_loss_absolute_delta']:.9g}"
            + "，全局 preclip norm 最大差 "
            + f"{row['maximum_preclip_norm_absolute_delta']:.9g}"
            + "，完整模型/FP32 main 参数哈希精确相同 "
            + f"{row['exact_updated_model_main_hash_pairs']}"
            + " 对。"
        )
    lines += [
        "",
        _REPORT_PARAGRAPH_2,
        "",
        "本轮没有重新定义 HF per-expert F.linear 门限；此前原始宽度下它与原生 GMM 的 BF16 末位差异仍保留为严格失败明细。当前原生 GM"
        + "M 验收不等同于字面 HF F.linear 验收。",
        "",
        _REPORT_PARAGRAPH_3,
        "",
        _REPORT_PARAGRAPH_4,
        "",
        _REPORT_PARAGRAPH_5,
    ]
    (ROOT / "summary.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_english_report(performance: str) -> None:
    """Write the compact English result index.

    Args:
        performance: Overall performance verdict.
    """
    english = [
        f"PRECISION_PASS / {performance}",
        "",
        _REPORT_PARAGRAPH_6,
        "",
        _REPORT_PARAGRAPH_7,
        "",
        _REPORT_PARAGRAPH_8,
        "",
        _REPORT_PARAGRAPH_9,
        "",
        _REPORT_PARAGRAPH_10,
    ]
    (ROOT / "summary.md").write_text("\n".join(english) + "\n", encoding="utf-8")


def main() -> None:
    """Run this experiment stage and retain its acceptance records."""
    for name, value in read(ROOT / "aggregation-freeze.json")["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, name
    frozen = verify_frozen_source()
    module = aggregate_component()
    trainer = aggregate_trainer()
    precision = numerical()
    passed = all((row["verdict"] != "PERFORMANCE_FAIL" for row in module + trainer))
    performance = "PERFORMANCE_PASS" if passed else "PERFORMANCE_FAIL"
    results = {
        "source_commit": frozen["candidate_commit"],
        "source": frozen["candidate_source"],
        "precision": precision,
        "performance_verdict": performance,
        "component": module,
        "cropped_trainer": trainer,
        "aggregation": "Median process p50/p90; every raw rank-max sample retained; no post-result thres"
        + "hold changes",
        "scope": _REPORT_PARAGRAPH_11,
        "plan": "optimization_plan.yaml",
        "provenance": "frozen-provenance.json",
    }
    spec = importlib.util.spec_from_file_location("raw_audit", ROOT / "audit_raw.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    results["raw_audit"] = {"trainer": audit.audit_trainer(), "component": audit.audit_component(), "status": "PASS"}
    (ROOT / "raw-evidence-audit.json").write_text(json.dumps(results["raw_audit"], indent=2) + "\n", encoding="utf-8")
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_chinese_report(frozen, module, trainer, precision, performance)
    write_english_report(performance)
    print(precision["status"], performance)


if __name__ == "__main__":
    main()
