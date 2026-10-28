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
"""Generate deterministic normal values from canonical global element indices."""

import hashlib

import numpy as np


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
