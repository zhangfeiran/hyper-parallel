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
"""Offline selected-key locality measurements from explicit CPU trace snapshots."""

from collections import Counter
from itertools import pairwise

import torch

from hyper_parallel.core.multicore.modules.mega_dsa.metadata import DsaBatchMeta


def _legal_rows(indices: torch.Tensor, meta: DsaBatchMeta) -> tuple[list[list[int]], dict]:
    rows = []
    counts = Counter()
    for query, raw in zip(meta.q_global_ids, indices.tolist()):
        sequence, _ = meta.sequence_position(query)
        start = meta.global_cu_seqlens[sequence]
        chosen = []
        seen = set()
        for token in raw:
            if token == -1:
                counts["padding_slots"] += 1
                continue
            if not 0 <= token < meta.global_valid_queries:
                raise ValueError("trace contains an unknown global token ID")
            if token in seen:
                raise ValueError("trace selections must be unique within each query")
            seen.add(token)
            if not start <= token <= query:
                counts["masked_slots"] += 1
                continue
            chosen.append(token)
        counts["empty_queries"] += not chosen
        rows.append(chosen)
    return rows, {name: counts[name] for name in ("padding_slots", "masked_slots", "empty_queries")}


def _owner_storage_runs(tokens: set[int], meta: DsaBatchMeta) -> list[int]:
    addresses = sorted((meta.token_owners[token], meta.token_local_offsets[token]) for token in tokens)
    if not addresses:
        return []
    runs = []
    length = 1
    for previous, current in pairwise(addresses):
        if current[0] == previous[0] and current[1] == previous[1] + 1:
            length += 1
        else:
            runs.append(length)
            length = 1
    return runs + [length]


def _tile_profile(rows: list[list[int]], meta: DsaBatchMeta, tile_size: int, token_bytes: int) -> dict:
    tiles = []
    previous = set()
    all_tokens = set()
    for begin in range(0, len(rows), tile_size):
        selections = [token for row in rows[begin:begin + tile_size] for token in row]
        distinct = set(selections)
        owner_references = Counter(meta.token_owners[token] for token in selections)
        owner_distinct = Counter(meta.token_owners[token] for token in distinct)
        reused = len(distinct & previous)
        remote = sum(count for owner, count in owner_distinct.items() if owner != meta.cp_rank)
        tiles.append({
            "query_storage_begin": begin,
            "query_count": len(rows[begin:begin + tile_size]),
            "selected_references": len(selections),
            "distinct_selected_keys": len(distinct),
            "owner_reference_histogram": [owner_references[owner] for owner in range(len(meta.cp_ranks))],
            "owner_distinct_histogram": [owner_distinct[owner] for owner in range(len(meta.cp_ranks))],
            "owner_storage_run_lengths": _owner_storage_runs(distinct, meta),
            "previous_tile_reused_keys": reused,
            "previous_tile_reuse_fraction": reused / len(distinct) if distinct else 0.0,
            "previously_seen_keys": len(distinct & all_tokens),
            "remote_distinct_keys": remote,
            "remote_fraction": remote / len(distinct) if distinct else 0.0,
            "deduplicated_payload_bytes": len(distinct) * token_bytes,
            "remote_payload_bytes": remote * token_bytes,
        })
        previous = distinct
        all_tokens.update(distinct)
    references = sum(tile["selected_references"] for tile in tiles)
    reads = sum(tile["distinct_selected_keys"] for tile in tiles)
    return {
        "tile_size": tile_size,
        "selected_references": references,
        "tile_distinct_reads": reads,
        "invocation_distinct_keys": len(all_tokens),
        "intra_tile_repeated_reads": references - reads,
        "inter_tile_repeated_reads": reads - len(all_tokens),
        "deduplicated_payload_bytes": reads * token_bytes,
        "tiles": tiles,
    }


def profile_index_trace(
    indices: torch.Tensor, meta: DsaBatchMeta, *, kv_bytes_per_token: int,
    tile_sizes: tuple[int, ...] = (16, 32, 64, 128),
) -> dict:
    """Measure selected-key unions, ownership and storage locality offline.

    Args:
        indices: Explicit CPU int32 snapshot in global packed [Tq,K] layout.
        meta: Invocation metadata, including ordered owner/storage addresses.
        kv_bytes_per_token: Declared bytes for the KV payload under study.
        tile_sizes: Positive query tile sizes, applied in local Q storage order.

    Returns:
        JSON-compatible trace measurements. Byte counts assume one fetch per
        distinct key per tile; repeated keys describe reuse opportunities, not
        measured cache hits, actual transport requests or allocation peaks.

    Raises:
        ValueError: Invalid snapshot, token IDs, duplicates or tile sizes.

    Note:
        This function never copies device tensors to the host. Capture must be
        requested explicitly outside the training hot path. Missing owner-local
        KV is allowed: this analyzes final global selections, not local buffers.
    """
    if indices.device.type != "cpu" or indices.dtype != torch.int32 or indices.ndim != 2:
        raise ValueError("trace requires an explicit CPU int32 [Tq,K] snapshot")
    if indices.shape[0] != len(meta.q_global_ids) or indices.shape[1] <= 0:
        raise ValueError("trace shape must match local queries and have positive K")
    if not isinstance(kv_bytes_per_token, int) or isinstance(kv_bytes_per_token, bool) or kv_bytes_per_token <= 0:
        raise ValueError("kv_bytes_per_token must be a positive integer")
    if (not isinstance(tile_sizes, tuple) or not tile_sizes
            or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in tile_sizes)
            or len(set(tile_sizes)) != len(tile_sizes)):
        raise ValueError("tile_sizes must contain unique positive integers")
    rows, counts = _legal_rows(indices, meta)
    return {
        "cache_identity": list(meta.cache_identity),
        "index_namespace": meta.index_namespace,
        "cp_ranks": list(meta.cp_ranks),
        "root_pes": list(meta.root_pes),
        "cp_rank": meta.cp_rank,
        "query_count": len(rows),
        "kv_bytes_per_token": kv_bytes_per_token,
        **counts,
        "profiles": [_tile_profile(rows, meta, size, kv_bytes_per_token) for size in tile_sizes],
    }
