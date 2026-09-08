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
"""Offline profiler import, calibration and held-out error accounting."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any

from .cost import Hardware, gemm, gemm_curve_key
from .profile_import import cube_services, import_profiles
from .workload import Shape

CYCLE_SOURCE = (
    "https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0282.html"
)


def summarize_tasks(profile: dict, *, cycles_per_us: float = 50.0) -> dict[str, Any]:
    """Separate same-core pre-compute waiting, task execution and publication.

    GetSystemCycle is a 50 MHz system counter on the documented A2/A3 targets.
    Never substitute the Cube frequency. Compute spans include task-internal
    protocol waits; sampled sums are neither complete totals nor wall latency.
    """
    if not math.isfinite(cycles_per_us) or cycles_per_us <= 0:
        raise ValueError("cycles_per_us must be finite and positive")
    if profile.get("schema_version") != 1 or profile.get("unit") != "system_cycle":
        raise ValueError("unsupported task profile schema or unit")
    phases: dict[str, list[dict[str, float]]] = defaultdict(list)
    for worker in profile["workers"]:
        for record in worker["records"]:
            start, ready, done, end = (
                record[key]
                for key in (
                    "start_cycle",
                    "ready_cycle",
                    "compute_end_cycle",
                    "end_cycle",
                )
            )
            if not start <= ready <= done <= end:
                raise ValueError("nonmonotonic local task timestamps")
            key = f"{worker['kind']}/type{record['task_type']}/tiling{record['tiling_position']}/wave{record['wave']}"
            phases[key].append(
                {
                    "pre_compute_us": (ready - start) / cycles_per_us,
                    "compute_us": (done - ready) / cycles_per_us,
                    "publication_us": (end - done) / cycles_per_us,
                }
            )
    return {
        "counter_source": CYCLE_SOURCE,
        "cycles_per_us": cycles_per_us,
        "accounting": "sampled task spans; pre_compute includes decode; compute may contain internal waits",
        "phases": {
            key: {
                "samples": len(rows),
                **{
                    name: {
                        "median": statistics.median(row[name] for row in rows),
                        "max": max(row[name] for row in rows),
                    }
                    for name in rows[0]
                },
            }
            for key, rows in phases.items()
        },
    }


def union_us(intervals: list[tuple[float, float]]) -> float:
    """Measure interval union rather than adding overlapping profiler scopes."""
    end = -math.inf
    duration = 0.0
    for start, finish in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(finish) or finish < start:
            raise ValueError("invalid trace interval")
        duration += max(0.0, finish - max(start, end))
        end = max(end, finish)
    return duration


def summarize_trace(trace: dict) -> dict[str, Any]:
    """Group Chrome complete events by pid/tid and name; scopes stay inclusive."""
    by_name: dict[str, list] = defaultdict(list)
    streams: dict[str, list] = defaultdict(list)
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X" or "dur" not in event:
            continue
        start, duration = float(event["ts"]), float(event["dur"])
        interval = (start, start + duration)
        key = f"{event.get('pid')}/{event.get('tid')}"
        streams[key].append(interval)
        by_name[f"{key}/{event.get('name', '')}"].append(interval)
    return {
        "unit": "us",
        "stream_busy_union_us": {key: union_us(value) for key, value in streams.items()},
        "inclusive_scopes": {key: {"calls": len(value), "union_us": union_us(value)} for key, value in by_name.items()},
        "accounting": "CPU and device streams remain separate; nested CPU scopes are not additive",
    }


def calibrate(samples: list[dict], hardware: Hardware, *, identity: dict) -> Hardware:
    """Fit positive GEMM service multipliers on explicitly marked training samples.

    Each sample represents the slowest participating core for one expert GMM,
    not the sum across cores and not its dependency wait. Holdout cases are
    rejected here so callers cannot silently train on evaluation measurements.
    """
    ratios: dict[str, list[float]] = defaultdict(list)
    domains: dict[str, list[int]] = defaultdict(list)
    curve_buckets: dict[str, dict[int, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if sample.get("split") != "train":
            raise ValueError("calibration accepts train samples only; keep holdout measurements separate")
        measured = float(sample["compute_us"])
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError("compute_us must be finite and positive")
        shape = Shape(**sample["shape"])
        rows = int(sample["expert_rows"])
        if rows <= 0:
            continue
        phase = sample["phase"]
        predicted = gemm(
            phase,
            [rows],
            shape,
            replace(hardware, phase_scales={}, gemm_row_scales={}),
            int(sample.get("gradient_bytes", 4)),
        ).service_us
        ratios[phase].append(measured / predicted)
        domains[phase].append(rows)
        key = gemm_curve_key(phase, shape, int(sample.get("gradient_bytes", 4)))
        curve_buckets[key][(rows - 1).bit_length()].append((rows, measured / predicted))
    if not ratios or not identity:
        raise ValueError("calibration requires nonempty service samples and source/device identity")
    return replace(
        hardware,
        phase_scales={phase: statistics.median(values) for phase, values in ratios.items()},
        gemm_row_scales={
            key: [
                (statistics.median(row[0] for row in values), statistics.median(row[1] for row in values))
                for _, values in sorted(buckets.items())
            ]
            for key, buckets in curve_buckets.items()
        },
        provenance={
            "status": "calibrated task service; end-to-end still requires holdout validation",
            "identity": identity,
            "shapes": list({tuple(sorted(sample["shape"].items())): sample["shape"] for sample in samples}.values()),
            "ranks": sorted({sample["ranks"] for sample in samples if "ranks" in sample}),
            "domain": {
                phase: {
                    "rows_min": min(rows),
                    "rows_max": max(rows),
                    "samples": len(rows),
                }
                for phase, rows in domains.items()
            },
            "row_curves": "power-of-two row buckets; median ratios; log-row interpolation; endpoints clamp",
            "curve_domain": {
                key: {
                    "rows_min": min(row[0] for values in buckets.values() for row in values),
                    "rows_max": max(row[0] for values in buckets.values() for row in values),
                    "samples": sum(len(values) for values in buckets.values()),
                }
                for key, buckets in curve_buckets.items()
            },
        },
    )


def calibrate_descriptor_service(documents: list[dict], hardware: Hardware, *, train_pairs: set[str]) -> Hardware:
    """Fit decode service using all-task counters, independently of sampled spans.

    Args:
        documents: Raw per-rank results from ``profile_mega_moe_costs``.
        hardware: Existing hardware rates; GEMM and transfer fits are preserved.
        train_pairs: Explicit route alpha/seed identifiers allowed into this fit.

    Returns:
        Hardware with measured active-vector-worker count and per-task decode
        service. Dependency wait and holdout routes never enter the fit.
    """
    observations, identities, worker_counts = [], set(), set()
    for document in documents:
        identity = document["identity"]
        mode = identity.get("runtime_environment", {}).get("HP_MEGA_MOE_COMPACT_RUNTIME")
        sources = identity.get("sources_and_payload", {})
        reader_hash = sources.get("hyper_parallel/core/multicore/ops/runtime/runtime_config.hpp")
        worker_hash = sources.get("hyper_parallel/core/multicore/ops/runtime/worker_kernel.h")
        native = {name: value for name, value in sources.items() if name.endswith((".so", ".o"))}
        if mode not in {"0", "1"} or not reader_hash or not worker_hash or not native:
            raise ValueError("descriptor calibration requires explicit runtime format and native source identity")
        native_hash = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
        identities.add((mode, reader_hash, worker_hash, native_hash))
        for result in document["results"]:
            if result["pair"] not in train_pairs:
                continue
            for direction, profile in result["task_profiles"].items():
                cycles = result["cycle_profiles"][direction]
                if profile["unit"] != "system_cycle" or cycles["unit"] != "system_cycle":
                    raise ValueError("descriptor calibration requires system-cycle counters")
                vectors = [worker for worker in profile["workers"] if worker["kind"] == "vector"]
                per_core = cycles["vector"]["per_core"]
                if not vectors or len(per_core) != len(vectors):
                    raise ValueError("descriptor calibration requires complete active vector counters")
                seen = sum(worker["seen"] for worker in vectors)
                decode_cycles = sum(row[cycles["categories"].index("descriptor_decode")] for row in per_core)
                if seen <= 0 or decode_cycles <= 0:
                    raise ValueError("descriptor calibration requires positive task counts and decode duration")
                worker_counts.add(len(vectors))
                observations.append(decode_cycles / (50.0 * seen))
    if not observations or len(identities) != 1 or len(worker_counts) != 1:
        raise ValueError("descriptor calibration requires training samples from one runtime format and worker topology")
    mode, reader_hash, worker_hash, native_hash = identities.pop()
    workers = worker_counts.pop()
    return replace(
        hardware, descriptor_us=statistics.median(observations), vector_workers=workers,
        provenance=hardware.provenance | {"descriptor": {
            "format": "compact" if mode == "1" else "dense",
            "reader_sha256": reader_hash, "worker_sha256": worker_hash,
            "native_payload_sha256": native_hash,
            "vector_workers": workers, "train_pairs": sorted(train_pairs),
            "observations": len(observations), "min_us": min(observations), "max_us": max(observations),
            "accounting": "median rank/direction all-task decode averages; excludes dependency waits; not wall time",
        }},
    )


def evaluate(predictions: list[float], measurements: list[float]) -> dict[str, Any]:
    """Report signed, median and worst relative errors on held-out cases."""
    if not predictions or len(predictions) != len(measurements):
        raise ValueError("nonempty prediction and measurement vectors must have equal length")
    if any(not math.isfinite(value) or value <= 0 for value in predictions + measurements):
        raise ValueError("latency samples must be finite and positive")
    errors = [(predicted - measured) / measured for predicted, measured in zip(predictions, measurements)]
    return {
        "cases": len(errors),
        "signed_relative_errors": errors,
        "median_absolute_relative_error": statistics.median(abs(error) for error in errors),
        "max_absolute_relative_error": max(abs(error) for error in errors),
    }


def calibrate_transfer_windows(samples: list[dict], hardware: Hardware, *, identity: dict) -> Hardware:
    """Fit a nonnegative per-window startup from isolated gradient reductions.

    ``base_us`` must include the selected network and owner-add service with
    zero window startup. ``measured_us`` excludes kernel arrival skew. This
    residual is an effective protocol/launch term for the recorded domain,
    not a claim that all HCCL latency is independent of peer count and bytes.
    """
    residuals = []
    for sample in samples:
        if sample.get("split") != "train":
            raise ValueError("transfer calibration accepts train samples only")
        measured, base, windows = (float(sample[key]) for key in ("measured_us", "base_us", "windows"))
        if any(not math.isfinite(value) or value < 0 for value in (measured, base, windows)) or windows < 1:
            raise ValueError("transfer costs must be finite and nonnegative, with at least one window")
        residuals.append(max(0.0, measured - base) / windows)
    if not residuals or not identity:
        raise ValueError("transfer calibration requires samples and their source/device identity")
    return replace(
        hardware,
        replica_gradient_window_us=statistics.median(residuals),
        provenance={
            **hardware.provenance,
            "replica_windows": {
                "identity": identity,
                "samples": len(samples),
                "scope": "isolated gradient reduction; effective per-window startup; no multi-node validation",
            },
        },
    )


def task_service_samples(documents: list[dict], train_pairs: set[str]) -> list[dict]:
    """Convert complete Cube samples to per-expert service observations.

    These observations use the maximum compute span among a GEMM's workers;
    no dependency wait enters the fit. All Cube queues must have stride one.
    Vector samples remain diagnostic because communication contains protocol waits.
    """
    if any(document.get("profile_schema_version") == 2 for document in documents):
        import_profiles(documents, expected_identity=documents[0]["identity"], read_traces=False)
    samples = []
    for document in documents:
        for result in document["results"]:
            source_shape = result["shape"]
            shape = {
                "experts": source_shape["num_experts"],
                "tokens_per_rank": source_shape["local_num_tokens"],
                "top_k": source_shape["top_k"],
                "hidden": source_shape["hidden_size"],
                "intermediate": source_shape["intermediate_size"],
            }
            # MegaMoeExperts selects its home-only runtime whenever it fits C,
            # even when the module has spare slots configured. That runtime
            # writes BF16 dW; using the configured B would mislabel calibration.
            gradient_bytes = 4 if document["arguments"]["slots"] and result["route"]["baseline_wave_count"] > 1 else 2
            for direction, profile in result["task_profiles"].items():
                context = result.get("task_contexts", {}).get(direction, {})
                cores = context.get("num_cube_cores", sum(worker["kind"] == "cube" for worker in profile["workers"]))
                for service in cube_services(profile, expected_cores=cores):
                    samples.append({
                        **service, "shape": shape, "gradient_bytes": gradient_bytes,
                        "pair": result["pair"], "rank": document["rank"], "ranks": source_shape["ep_size"],
                        "direction": direction, "context": context or None,
                        "split": "train" if result["pair"] in train_pairs else "holdout",
                    })
    return samples


def hardware_dict(hardware: Hardware) -> dict[str, Any]:
    """Serialize a fitted model without device tensors or framework objects."""
    return asdict(hardware)
