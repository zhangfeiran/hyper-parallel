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

import hashlib
import math
from typing import Any

import numpy as np
import torch

from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout


_CHUNK_ELEMENTS = 1 << 20


def canonical_normal(indices: np.ndarray, name: str, seed: int, std: float) -> np.ndarray:
    """Map canonical global element indices to reproducible normal FP32 values."""
    key = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
    with np.errstate(over="ignore"):
        bits = indices.astype(np.uint64) + np.uint64(key)
        bits = (bits ^ (bits >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        bits = (bits ^ (bits >> 27)) * np.uint64(0x94D049BB133111EB)
        bits ^= bits >> 31
    first = ((bits >> 32).astype(np.float64) + 0.5) / 2**32
    second = ((bits & np.uint64(0xFFFFFFFF)).astype(np.float64) + 0.5) / 2**32
    return (np.sqrt(-2 * np.log(first)) * np.cos(2 * np.pi * second) * std).astype(np.float32)


def canonical_indices(start: int, count: int, shape: tuple[int, ...],
                      area: tuple[tuple[int, int], ...], transpose: bool) -> np.ndarray:
    """Translate local flat coordinates into native-layout global flat indices."""
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


def initialize_random_shards(trainer: Any, *, seed: int, backend: str, std: float = 0.02) -> dict[str, Any]:
    """Fill local shards using the native parameter layout and reload optimizer masters.

    Random matrices follow N(0, std); norms/Engram gates/scales keep unit values,
    and biases, mHC bases and attention sinks keep zeros. Persistent model buffers
    retain the builder's deterministic values. Only bounded CPU chunks are used.
    """
    count = 0
    signature = np.uint64(0)
    with torch.no_grad():
        for name, parameter in trainer.base.model.named_parameters():
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            if not local.is_contiguous():
                raise ValueError(f"Random initialization requires contiguous storage: {name}")
            shape = tuple(parameter.shape)
            layout = getattr(parameter, "layout", None)
            if hasattr(layout, "rank_list"):
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
            for start in range(0, local.numel(), _CHUNK_ELEMENTS):
                size = min(_CHUNK_ELEMENTS, local.numel() - start)
                indices = canonical_indices(start, size, shape, area, transpose)
                values = (canonical_normal(indices, name, seed, std) if constant is None
                          else np.full(size, constant, dtype=np.float32))
                # The token embedding padding row stays zero, as in the HF initializer.
                if name == "model.embed_tokens.weight":
                    values[indices // shape[-1] == trainer.base.model.config.pad_token_id] = 0
                flat[start:start + size].copy_(torch.from_numpy(values))
                # An order-independent checksum permits comparison after expert transposition.
                with np.errstate(over="ignore"):
                    signature += np.sum(values.view(np.uint32).astype(np.uint64), dtype=np.uint64)
            count += local.numel()
    optimizers = trainer.base.optimizer
    for optimizer in optimizers if isinstance(optimizers, list) else [optimizers]:
        optimizer.reload_model_params()
    return {"algorithm": "global_index_splitmix64_box_muller_v1", "seed": seed,
            "std": std, "local_parameter_elements": count, "fp32_bits_sum_mod_2_64": str(int(signature)),
            "weight_files": False}
