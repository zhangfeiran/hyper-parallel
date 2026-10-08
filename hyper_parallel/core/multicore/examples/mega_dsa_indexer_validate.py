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
"""Long-history CP=1 indexer selection gate; separate from model gradient acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from dataclasses import replace
from pathlib import Path

import torch

from hyper_parallel.core.multicore.examples.mega_dsa_backend_probe import (
    probe_environment,
)
from hyper_parallel.core.multicore.modules.mega_dsa.cann_reference import (
    CannDsaLayout,
    CannDsaReference,
)
from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta
from hyper_parallel.core.multicore.modules.mega_dsa.reference import indexer_reference

_K = 2048
_CASES = ("increasing", "signed_decreasing", "concentrated", "all_tied", "random_signed")


def _fixture(case, meta):
    total = meta.global_valid_queries
    query = torch.zeros(total, 8, 128)
    key = torch.zeros(total, 128)
    weights = torch.zeros(total, 8)
    positions = torch.tensor([meta.sequence_position(token)[1] for token in range(total)])
    if case == "random_signed":
        generator = torch.Generator().manual_seed(20261008)
        return tuple((torch.randn(shape, generator=generator) * 0.1).bfloat16()
                     for shape in ((total, 8, 128), (total, 128), (total, 8)))
    if case == "concentrated":
        query[:, 0, 0] = 1
        key[:, 0] = torch.where(positions < _K, 4, 1)
        weights[:, 0] = 1
    elif case in ("increasing", "signed_decreasing"):
        # Two exactly representable BF16 digits give unique FP32 dot products
        # beyond K; a single BF16 coordinate would introduce accidental ties.
        key[:, 0], key[:, 1] = positions // 128, positions % 128
        query[:, 0, 0], query[:, 0, 1] = 128, 1
        weights[:, 0] = 1
        if case == "signed_decreasing":
            query[:, 1, 0], query[:, 1, 1] = 256, 2
            weights[:, 1] = -1
    elif case != "all_tied":
        raise ValueError("unknown indexer validation fixture")
    return query.bfloat16(), key.bfloat16(), weights.bfloat16()


def _sample_queries(meta):
    queries = []
    for start, end in zip(meta.global_cu_seqlens, meta.global_cu_seqlens[1:]):
        positions = {0, 1, 31, 127, 1023, _K - 2, _K - 1, _K, _K + 63, end - start - 1}
        queries.extend(start + position for position in sorted(positions) if position < end - start)
    return tuple(queries)


def _raw_contract(indices, meta):
    """Check unfiltered native storage, including every unsampled query row."""
    if indices.device.type != "cpu" or indices.dtype != torch.int32 or indices.shape != (
            meta.global_valid_queries, 1, _K):
        raise ValueError("raw native snapshot must be CPU int32 [T,1,2048]")
    local = indices[:, 0]
    positions = torch.tensor([meta.sequence_position(token)[1] for token in range(meta.global_valid_queries)])
    present = local >= 0
    sorted_ids = local.sort(dim=-1).values
    duplicate = (sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)
    checks = {
        "legal_or_padding": ((local == -1) | (present & (local <= positions[:, None]))).all(dim=1),
        "native_cardinality": present.sum(dim=1) == (positions + 1).clamp_max(_K),
        "unique": ~duplicate.any(dim=1),
        "trailing_padding": ~(present[:, 1:] & ~present[:, :-1]).any(dim=1),
    }
    failed = ~torch.stack(tuple(checks.values())).all(dim=0)
    return {"passed": not bool(failed.any()), "queries_checked": meta.global_valid_queries,
            "checks": {name: bool(value.all()) for name, value in checks.items()},
            "failed_query_ids": failed.nonzero().flatten().tolist()}


def _global_samples(raw, meta, queries):
    rows = raw[torch.tensor(queries), 0]
    starts = torch.tensor([meta.global_cu_seqlens[meta.sequence_position(q)[0]] for q in queries])
    return torch.where(rows >= 0, rows + starts[:, None], rows).to(torch.int32)


def _score_certificate(scores, selected, meta, *, sparse_count):
    """Certify Top-K membership with exact cutoff ties, without assuming tie ordering."""
    rows = {}
    for row, query in enumerate(meta.q_global_ids):
        sequence, position = meta.sequence_position(query)
        start = meta.global_cu_seqlens[sequence]
        legal = scores[row, start:query + 1]
        count = min(sparse_count, position + 1)
        ranked = legal.sort(descending=True).values
        cutoff = ranked[count - 1]
        ids = selected[row][selected[row] != -1]
        valid = bool(((ids >= start) & (ids <= query)).all())
        unique = ids.unique().numel() == ids.numel()
        eligible = valid and unique and ids.numel() == count
        strict_ids = (legal > cutoff).nonzero().flatten() + start
        missing = sorted(set(strict_ids.tolist()) - set(ids.tolist()))
        wrong = [] if not eligible else ids[scores[row, ids.long()] < cutoff].tolist()
        rows[str(query)] = {
            "passed": eligible and not missing and not wrong and bool(legal.isfinite().all()),
            "legal_candidates": position + 1, "selected_count": ids.numel(), "cutoff": float(cutoff),
            "strictly_better_count": strict_ids.numel(), "cutoff_tie_count": int((legal == cutoff).sum()),
            "cutoff_gap": None if count == legal.numel() else float(cutoff - ranked[count]),
            "missing_strict_ids": missing, "below_cutoff_ids": wrong,
            "hypothetical_prefix_partition_counts": [int((ids < start + sparse_count).sum()),
                                                       int((ids >= start + sparse_count).sum())],
        }
    return {"passed": all(row["passed"] for row in rows.values()), "queries": rows}


def _oracle_samples(values, meta, queries):
    sample_meta = replace(meta, q_global_ids=queries)
    query, key, weight = values
    query, weight = query[list(queries)].float(), weight[list(queries)].float()
    scores = (torch.einsum("thd,sd->ths", query, key.float()).relu() * weight[..., None]).sum(dim=1)
    expected = indexer_reference(query, key.float(), weight, sample_meta, sparse_count=_K)
    return sample_meta, scores, expected


def _set_agreement(first, second):
    equal = (first.sort(dim=-1).values == second.sort(dim=-1).values).all(dim=-1)
    return {"all_equal": bool(equal.all()), "mismatch_row_ids": (~equal).nonzero().flatten().tolist()}


def _sha256_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _analytic_expected(case, meta):
    """Construct all-row exact winners for the known integer-score fixtures."""
    if case not in ("increasing", "signed_decreasing", "concentrated"):
        raise ValueError("analytic winners require an ordered or concentrated fixture")
    positions = torch.tensor([meta.sequence_position(q)[1] for q in range(meta.global_valid_queries)])[:, None]
    slots = torch.arange(_K)[None]
    begin = (positions + 1 - _K).clamp_min(0) if case == "increasing" else torch.zeros_like(positions)
    return torch.where(slots < (positions + 1).clamp_max(_K), slots + begin, -1).to(torch.int32)[:, None]


def _capture_case(case, meta, backend):
    values = _fixture(case, meta)
    device_values = tuple(value.to(backend.layout.device) for value in values)
    selection = backend.indexer(*device_values)
    # The public export filters illegal IDs. This diagnostic intentionally
    # inspects raw owned storage before conversion so violations cannot disappear.
    candidate = backend._native_indices(selection).detach().cpu()
    query, key, weights = device_values
    stock, _ = torch.ops.npu.npu_lightning_indexer(
        query, key[:, None].contiguous(), weights,
        actual_seq_lengths_query=backend.layout.length_tensor, actual_seq_lengths_key=backend.layout.length_tensor,
        layout_query="TND", layout_key="TND", sparse_count=_K, sparse_mode=3, return_value=False,
    )
    stock = stock.cpu()
    repeated = backend._native_indices(backend.indexer(*device_values)).detach().cpu()
    return values, candidate, stock, repeated


def _zero_tie_proof(values, query, first, second, first_certificate, second_certificate):
    """Certify structural ReLU-zero ties with a conservative FP32 dot error bound."""
    changed = sorted(set(first.tolist()) ^ set(second.tolist()))
    eligible = (first_certificate["passed"] and second_certificate["passed"]
                and first_certificate["cutoff"] == second_certificate["cutoff"] == 0 and bool(changed))
    if not eligible:
        return {"certified": False, "changed_ids": changed, "reason": "not a valid exact zero-cutoff tie"}
    query_states, keys, _ = values
    products = query_states[query].double()[:, None] * keys[changed].double()[None]
    dots = products.sum(dim=-1)
    unit_roundoff = 2**-24
    terms = query_states.shape[-1]
    gamma = terms * unit_roundoff / (1 - terms * unit_roundoff)
    bound = gamma * products.abs().sum(dim=-1)
    margin = -(dots + bound)
    return {"certified": bool((margin > 0).all()), "changed_ids": changed,
            "maximum_fp64_head_dot": float(dots.max()), "maximum_fp32_dot_error_bound": float(bound.max()),
            "minimum_negative_margin": float(margin.min()),
            "scope": "every exchanged key has inactive ReLU in every head even under the FP32 accumulation bound"}


def _mismatch_ties(agreement, first, second, scores, sample_meta, values):
    evidence = {}
    rows = {query: row for row, query in enumerate(sample_meta.q_global_ids)}
    for query in agreement["mismatch_row_ids"]:
        if query not in rows:
            evidence[str(query)] = {"certified": False, "reason": "outside bounded diagnostic query sample"}
            continue
        row = rows[query]
        meta = replace(sample_meta, q_global_ids=(query,))
        certificate_a = _score_certificate(scores[row:row + 1], first[row:row + 1], meta, sparse_count=_K)
        certificate_b = _score_certificate(scores[row:row + 1], second[row:row + 1], meta, sparse_count=_K)
        evidence[str(query)] = _zero_tie_proof(values, query, first[row], second[row],
                                              certificate_a["queries"][str(query)],
                                              certificate_b["queries"][str(query)])
    return {"certified": all(item["certified"] for item in evidence.values()), "queries": evidence}


def _case_passed(case, contracts, ties, certificate, stock_certificate, analytic):
    raw_pass = all(check["passed"] for check in contracts.values())
    tie_pass = case == "all_tied" or all(proof["certified"] for proof in ties.values())
    score_pass = case == "random_signed" or (certificate["passed"] and stock_certificate["passed"])
    analytic_pass = all(check["all_equal"] for check in analytic.values())
    return all((raw_pass, tie_pass, score_pass, analytic_pass))


def _check_case(case, values, candidate, stock, repeated, meta):
    agreement = _set_agreement(candidate[:, 0], stock[:, 0])
    repeat_agreement = _set_agreement(candidate[:, 0], repeated[:, 0])
    unexpected = sorted(set(agreement["mismatch_row_ids"] + repeat_agreement["mismatch_row_ids"]))
    extra = unexpected[:32] if case != "all_tied" else []
    queries = tuple(sorted(set(_sample_queries(meta)) | set(extra)))
    sample_meta, scores, expected = _oracle_samples(values, meta, queries)
    candidate_samples = _global_samples(candidate, meta, queries)
    stock_samples = _global_samples(stock, meta, queries)
    certificate = _score_certificate(scores, candidate_samples, sample_meta, sparse_count=_K)
    stock_certificate = _score_certificate(scores, stock_samples, sample_meta, sparse_count=_K)
    repeated_samples = _global_samples(repeated, meta, queries)
    ties = {} if case == "all_tied" else {
        "stock": _mismatch_ties(agreement, candidate_samples, stock_samples, scores, sample_meta, values),
        "repeat": _mismatch_ties(repeat_agreement, candidate_samples, repeated_samples, scores, sample_meta, values),
    }
    contracts = {name: _raw_contract(raw, meta) for name, raw in
                 (("enhance", candidate), ("stock", stock), ("enhance_repeat", repeated))}
    analytic = {}
    if case in ("increasing", "signed_decreasing", "concentrated"):
        exact = _analytic_expected(case, meta)
        analytic = {name: _set_agreement(raw[:, 0], exact[:, 0]) for name, raw in
                    (("enhance", candidate), ("stock", stock), ("enhance_repeat", repeated))}
    accepted = _case_passed(case, contracts, ties, certificate, stock_certificate, analytic)
    strict = all((all(check["passed"] for check in contracts.values()),
                  agreement["all_equal"], repeat_agreement["all_equal"]))
    report = {
        "accepted": accepted, "raw_contracts": contracts, "sample_query_ids": list(queries),
        "strict_set_agreement_passed": strict, "structural_zero_tie_evidence": ties,
        "analytic_all_row_membership": analytic,
        "enhance_vs_stock_all_rows": agreement,
        "enhance_repeat_set_agreement": repeat_agreement,
        "enhance_repeat_order_equal": bool(torch.equal(candidate, repeated)),
        "enhance_fp32_score_certificate": certificate, "stock_fp32_score_certificate": stock_certificate,
        "fp32_certificate_required": case != "random_signed",
        "enhance_vs_stable_global_id_oracle": _set_agreement(candidate_samples, expected),
        "raw_native_sha256": {name: _sha256_tensor(raw) for name, raw in
                              (("enhance", candidate), ("stock", stock), ("enhance_repeat", repeated))},
    }
    return report, candidate_samples, stock_samples


def run_indexer_validation(report: dict, lengths: tuple[int, ...], snapshot_dir: Path | None) -> None:
    """Run full native queries and bounded CPU oracle samples over complete key storage."""
    if not lengths or any(length <= _K for length in lengths):
        raise ValueError("each sequence must exceed K=2048 for this long-context gate")
    # Native registration/device activation belongs only to this explicit validator.
    __import__("torch_npu")
    torch.npu.set_device(0)
    report["environment"] = probe_environment()
    report["device"] = torch.npu.get_device_name(0)
    meta = DsaBatchMeta.packed(lengths)
    backend = CannDsaReference(CannDsaLayout(meta, "npu:0"), attention_scale=192**-0.5)
    report["cases"] = {}
    for case in _CASES:
        report["stage"] = case
        values, candidate, stock, repeated = _capture_case(case, meta, backend)
        result, candidate_samples, stock_samples = _check_case(case, values, candidate, stock, repeated, meta)
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = snapshot_dir / f"{case}.pt"
            torch.save({"case": case, "sequence_lengths": lengths, "inputs": values,
                        "sample_query_ids": result["sample_query_ids"], "enhance_samples": candidate_samples,
                        "stock_samples": stock_samples}, path)
            result["snapshot"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "scope": "CPU inputs and sampled global IDs; "
                                           "full native indices are hashed separately"}
        report["cases"][case] = result
        report["device_execution"] = True
        print(f"{case}: accepted={result['accepted']}", flush=True)
    report["selection_gate_passed"] = all(case["accepted"] for case in report["cases"].values())
    report["strict_set_gate_passed"] = all(case["strict_set_agreement_passed"] for case in report["cases"].values())
    report["status"] = "failed_selection"
    if report["selection_gate_passed"]:
        report["status"] = "selection_verified" if report["strict_set_gate_passed"] else "selection_verified_with_ties"
    report["stage"] = "complete"


def main() -> None:
    """Record exact environment, sampled certificates and failures; exit nonzero on a failed gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[2176, 2240])
    args = parser.parse_args()
    report = {"status": "running", "stage": "metadata", "device_execution": False,
              "sequence_lengths": args.sequence_lengths, "sparse_count": _K, "cp_size": 1, "tp_size": 1,
              "tie_policy": "any legal exact-cutoff tied subset; every strictly better candidate is mandatory; "
                            "stable global-ID oracle match is reported separately",
              "random_policy": "strict set agreement reported separately; a mismatch is admitted only with "
                               "valid exact-cutoff membership and a structural ReLU-zero proof for every exchanged ID; "
                               "at most 32 additional mismatch queries are certified; other differences fail",
              "partition_scope": "hypothetical per-sequence prefix split at K, not an executed CP mesh",
              "oracle_scope": "FP32 full-key candidates for sampled queries; "
                              "raw native contracts/stock sets for all rows",
              "p0_complete": False, "model_acceptance_changed": False, "performance_measurement": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        run_indexer_validation(report, tuple(args.sequence_lengths), args.snapshot_dir)
        if not report["selection_gate_passed"]:
            raise RuntimeError("long-context indexer selection gate failed")
    except Exception as error:
        if report["status"] == "running":
            report["status"] = "error"
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
