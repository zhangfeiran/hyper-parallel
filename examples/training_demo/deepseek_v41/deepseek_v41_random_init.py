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
"""Generate topology-independent canonical random shards without weight files."""
# pylint: disable=forbidden-backend-import

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
from itertools import islice
import math
import os
from typing import Any

import numpy as np
import torch

from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.utils.random_init import canonical_normal
from hyper_parallel.models.external_state import get_model_external_state


_CHUNK_ELEMENTS = 1 << 20
_INITIALIZATION_WORKERS = 2


def _linear_shard_base(shape: tuple[int, ...], area: tuple[tuple[int, int], ...]) -> int | None:
    """Return the global start only when all local elements are contiguous in row-major order."""
    start, stride, full_suffix = 0, 1, True
    for extent, (begin, end) in reversed(tuple(zip(shape, area, strict=True))):
        if end - begin > 1 and not full_suffix:
            return None
        start += begin * stride
        full_suffix = full_suffix and begin == 0 and end == extent
        stride *= extent
    return start


def canonical_indices(start: int, count: int, shape: tuple[int, ...],
                      area: tuple[tuple[int, int], ...], transpose: bool) -> np.ndarray:
    """Translate local flat coordinates into native-layout global flat indices."""
    if not transpose:
        base = _linear_shard_base(shape, area)
        if base is not None:
            return np.arange(base + start, base + start + count, dtype=np.uint64)
    elif len(shape) >= 2:
        base = _linear_shard_base(shape[:-2], area[:-2])
        if base is not None:
            rows, columns = shape[-2:]
            (first_row, last_row), (first_column, last_column) = area[-2:]
            offsets = np.arange(start, start + count, dtype=np.uint64)
            row_offsets, column = np.divmod(offsets, last_column - first_column)
            prefix, row = np.divmod(row_offsets, last_row - first_row)
            return ((prefix + base) * columns + column + first_column) * rows + row + first_row
    local_shape = tuple(end - begin for begin, end in area)
    offsets = np.arange(start, start + count, dtype=np.int64)
    coordinates = []
    for size, (begin, _) in reversed(list(zip(local_shape, area))):
        coordinates.append(offsets % size + begin)
        offsets //= size
    coordinates.reverse()
    canonical_shape = list(shape)
    if transpose:
        canonical_shape[-2:] = canonical_shape[-2:][::-1]
        coordinates[-2:] = coordinates[-2:][::-1]
    indices = np.zeros(count, dtype=np.uint64)
    for size, coordinate in zip(canonical_shape, coordinates):
        indices = indices * np.uint64(size) + coordinate.astype(np.uint64)
    return indices


def _generate_chunk(start: int, *, numel: int, shape: tuple[int, ...],
                    area: tuple[tuple[int, int], ...], transpose: bool,
                    name: str, seed: int, std: float, constant: float | None,
                    padding_id: int | None) -> np.ndarray:
    """Generate one CPU chunk without touching model or device state."""
    size = min(_CHUNK_ELEMENTS, numel - start)
    if constant is not None and padding_id is None:
        return np.full(size, constant, dtype=np.float32)
    indices = canonical_indices(start, size, shape, area, transpose)
    values = (canonical_normal(indices, name, seed, std) if constant is None
              else np.full(size, constant, dtype=np.float32))
    if padding_id is not None:
        values[indices // shape[-1] == padding_id] = 0
    return values


def _initialization_chunks(generate: Callable[[int], np.ndarray], numel: int, *,
                           pool: ThreadPoolExecutor | None) -> Iterator[tuple[int, np.ndarray]]:
    """Generate bounded chunks ahead of the ordered copies on the caller thread."""
    offsets = range(0, numel, _CHUNK_ELEMENTS)
    if pool is None or len(offsets) <= 1:
        for start in offsets:
            yield start, generate(start)
        return
    remaining = iter(offsets)
    pending = deque((start, pool.submit(generate, start)) for start in islice(remaining, _INITIALIZATION_WORKERS))
    while pending:
        start, future = pending.popleft()
        values = future.result()
        if (next_start := next(remaining, None)) is not None:
            pending.append((next_start, pool.submit(generate, next_start)))
        yield start, values


def initialize_random_shards(trainer: Any, *, seed: int, backend: str, std: float = 0.02) -> dict[str, Any]:
    """Fill local shards using the native parameter layout and reload optimizer masters.

    Random matrices follow N(0, std); norms/Engram gates/scales keep unit values,
    and biases, mHC bases and attention sinks keep zeros. Persistent model buffers
    retain the builder's deterministic values. Only bounded CPU chunks are used.
    """
    external = get_model_external_state(trainer.base.model)
    host_tables = {id(table.weight): table for table in external.tables.values()} if external else {}
    count = 0
    signature = np.uint64(0)
    cpu_affinity = getattr(os, "sched_getaffinity", lambda _: {0})(0)
    pool_context = (ThreadPoolExecutor(max_workers=_INITIALIZATION_WORKERS)
                    if _INITIALIZATION_WORKERS > 1 and len(cpu_affinity) >= _INITIALIZATION_WORKERS
                    else nullcontext())
    with torch.no_grad(), pool_context as pool:
        for name, parameter in trainer.base.model.named_parameters():
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            if not local.is_contiguous():
                raise ValueError(f"Random initialization requires contiguous storage: {name}")
            shape = tuple(parameter.shape)
            layout = getattr(parameter, "layout", None)
            if id(parameter) in host_tables:
                table = host_tables[id(parameter)]
                if table.global_row_start is None or table.rows_per_owner is None:
                    raise RuntimeError(f"Host table is not bound before random initialization: {name}")
                shape = (table.physical_rows, table.width)
                area = ((table.global_row_start, table.global_row_start + table.rows_per_owner), (0, table.width))
            elif hasattr(layout, "rank_list"):
                area = infer_slice_area_by_layout(layout, layout.rank_list.index(layout.mesh.rank), shape)
            else:
                area = tuple((0, size) for size in shape)
            if math.prod(end - begin for begin, end in area) != local.numel():
                raise ValueError(f"Unsupported nonrectangular parameter shard: {name}")
            leaf = name.rsplit(".", 1)[-1]
            constant = None
            if leaf in {"bias", "bias_vl", "base", "hc_base", "sinks", "position_bias"}:
                constant = 0.0
            elif leaf in {"scale", "hc_scale", "q_weight", "k_weight"} or (leaf == "weight" and len(shape) == 1):
                constant = 1.0
            flat = local.view(-1)
            transpose = backend == "megamoe" and ".mlp.experts." in name
            # The token embedding padding row stays zero, as in the HF initializer.
            padding_id = trainer.base.model.config.pad_token_id if name == "model.embed_tokens.weight" else None
            generate = partial(_generate_chunk, numel=local.numel(), shape=shape, area=area, transpose=transpose,
                               name=name, seed=seed, std=std, constant=constant, padding_id=padding_id)
            for start, values in _initialization_chunks(generate, local.numel(),
                                                        pool=pool if constant is None else None):
                flat[start:start + values.size].copy_(torch.from_numpy(values))
                # An order-independent checksum permits comparison after expert transposition.
                with np.errstate(over="ignore"):
                    signature += np.sum(values.view(np.uint32), dtype=np.uint64)
            count += local.numel()
    optimizers = trainer.base.optimizer
    for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
        optimizer.reload_model_params()
    return {"algorithm": "global_index_splitmix64_box_muller_v1", "seed": seed,
            "std": std, "local_parameter_elements": count, "fp32_bits_sum_mod_2_64": str(int(signature)),
            "weight_files": False}
