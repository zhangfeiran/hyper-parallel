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
"""Framework-free aggregation of complete LongCat diagnostic rank reports."""

import json
from pathlib import Path
from typing import Optional


SEMANTIC_CASES = {
    "HF_repeat": True,
    "native_GMM_hf_vs_original_HF": False,
    "Mega_hf_vs_native_GMM_hf": True,
    "native_GMM_fused_vs_original_HF": False,
    "Mega_fused_vs_native_GMM_fused": True,
    "fixed_operand_fused_SwiGLU_grid": True,
}
_TENSORS = {"output", "input_gradient", "probability_gradient", "gate_up_proj", "down_proj"}


def _rank_rows(report: dict, rank: int, stage: str) -> list[dict]:
    """Require every expected case, tensor, source-root receipt and closed runtime."""
    if report["rank"] != rank or report["shmem_after_close"]["reference_count"] != 0:
        raise ValueError("rank identity mismatch or SHMEM was not fully closed")
    semantic = stage == "ep1"
    if not semantic and report.get("native_root_matches") is not True:
        raise ValueError("native libraries were not verified against this checkout")
    expected = SEMANTIC_CASES if semantic else {"push": True, "pull": True}
    rows = report["records" if semantic else "results"]
    names = [row["case" if semantic else "mode"] for row in rows]
    if len(names) != len(expected) or set(names) != set(expected):
        raise ValueError(f"incomplete or duplicate cases: {names}")
    result = []
    for row, name in zip(rows, names):
        if semantic and row["primary"] is not expected[name]:
            raise ValueError(f"unexpected primary gate for {name}")
        if set(row["comparison"]) != _TENSORS:
            raise ValueError(f"incomplete tensor comparisons for {name}")
        if not isinstance(row["strict_pass"], bool) or not isinstance(row["all_ranks_strict_pass"], bool):
            raise ValueError(f"missing boolean strict verdict for {name}")
        if any(not isinstance(value["strict_pass"], bool) for value in row["comparison"].values()):
            raise ValueError(f"missing tensor strict verdict for {name}")
        passed = row["strict_pass"] and row["all_ranks_strict_pass"]
        passed = passed and all(value["strict_pass"] for value in row["comparison"].values())
        result.append({"rank": rank, "case": name, "primary": expected[name], "strict_pass": passed,
                       "comparison": row["comparison"]})
    return result


def summarize_stage(output: Path, stage: str, exit_code: int) -> dict:
    """Fail closed on absent reports, worker errors or any primary numerical failure.

    Args:
        output: New run output directory containing stage subdirectories.
        stage: ep1, ep4-fused or ep4-hf.
        exit_code: Actual worker launcher exit code.
    """
    result = {"stage": stage, "exit_code": exit_code, "status": "ERROR", "errors": [], "rows": []}
    expected_ranks = 1 if stage == "ep1" else 4
    for rank in range(expected_ranks):
        path = output / stage / f"rank{rank}.json"
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            result["rows"].extend(_rank_rows(report, rank, stage))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append(f"{path.name}: {error}")
    if result["errors"]:
        return result
    failed = any(row["primary"] and not row["strict_pass"] for row in result["rows"])
    if failed:
        result["status"] = "NUMERICAL_FAIL"
    elif exit_code:
        result["errors"].append("worker exited nonzero despite passing primary numerical reports")
    else:
        result["status"] = "PASS"
    return result


def overall_status(stages: list[dict], error: Optional[str]) -> tuple[str, int]:
    """Preserve the original fused failure even when an HF follow-up passes.

    Args:
        stages: All executed stage results.
        error: Optional preparation or execution error.
    """
    if error or not stages or any(row["status"] == "ERROR" for row in stages):
        return "ERROR", 2
    if any(row["status"] == "NUMERICAL_FAIL" for row in stages):
        return "NUMERICAL_FAIL", 1
    return "PASS", 0


def _diagnosis(stages: list[dict]) -> list[str]:
    """Explain directions to investigate without declaring a customer root cause."""
    notes = []
    semantic = {row["case"]: row for stage in stages if stage["stage"] == "ep1" for row in stage["rows"]}
    fused = semantic.get("Mega_fused_vs_native_GMM_fused", {})
    hf = semantic.get("native_GMM_fused_vs_original_HF", {})
    if fused.get("strict_pass") and hf and not hf["strict_pass"]:
        notes.append("同语义融合路径通过，但原生融合与原始 HF 不同；请核对项目参考是否仍分开执行 SiLU 和乘法。")
    if fused and not fused["strict_pass"]:
        notes.append("单卡同语义融合比较失败；优先查看固定操作数实验及输出/dX/专家梯度的首批失败位置。")
    ep1_pass = any(stage["stage"] == "ep1" and stage["status"] == "PASS" for stage in stages)
    if ep1_pass and any(stage["stage"].startswith("ep4") and stage["status"] == "NUMERICAL_FAIL" for stage in stages):
        notes.append("单卡通过、四卡失败；优先核对 EP 路由、专家归属、分组形状和通信/同步。")
    if stages and all(stage["status"] == "PASS" for stage in stages):
        notes.append("所选独立组件用例通过；客户原始失败批次、checkpoint 和完整训练仍需在项目中单独对齐。")
    if any(stage["status"] == "ERROR" for stage in stages):
        notes.append("存在运行错误或不完整报告；请先检查对应阶段日志，不能按数值通过处理。")
    return notes


def write_summary(output: Path, receipt: dict) -> int:
    """Write machine-readable results and a Chinese customer-facing receipt.

    Args:
        output: Run artifact directory.
        receipt: Stage records, provenance, controls and optional error.
    """
    status, code = overall_status(receipt["stages"], receipt.get("error"))
    if receipt.get("complete", True) is False and not receipt.get("error"):
        status, code = "RUNNING", 2
    if receipt.get("interrupted"):
        status, code = "INTERRUPTED", 130
    receipt.update(status=status, exit_code=code, diagnosis=_diagnosis(receipt["stages"]))
    (output / "summary.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"LongCat 诊断结果：{status}，退出码：{code}",
             "范围：固定种子的原宽度专家组件；不等同于客户原始失败用例或完整模型验收。"]
    if receipt.get("error"):
        lines.append(f"准备/执行错误：{receipt['error']}")
    for stage in receipt["stages"]:
        lines.append(f"\n[{stage['stage']}] {stage['status']}，worker 退出码={stage['exit_code']}")
        lines.extend(stage["errors"])
        for row in stage["rows"]:
            label = "主验收" if row["primary"] else "HF参考差异诊断"
            lines.append(f"rank={row['rank']} {row['case']} [{label}] strict_pass={row['strict_pass']}")
            for name, value in row["comparison"].items():
                relative = value.get("relative_l2")
                percent = "未知" if relative is None else f"{100 * relative:.6f}%"
                lines.append(f"  {name}: relative_l2={percent}, max_abs={value.get('max_abs')}, "
                             f"strict_bad={value.get('strict_bad')}, strict_pass={value['strict_pass']}")
                if value.get("first_strict_failures"):
                    lines.append(f"    首批失败：{value['first_strict_failures']}")
    lines.extend(["", *receipt["diagnosis"], "", "回传：summary.*、environment.json、各阶段 .log 与全部 rank*.json。"])
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code
