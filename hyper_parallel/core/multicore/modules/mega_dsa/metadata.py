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
"""Host-side packed-token identity, CP ownership, and loss normalization."""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _integer_tuple(name: str, values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple) or any(not _is_integer(value) for value in values):
        raise ValueError(f"{name} must be a tuple of integers")


@dataclass(frozen=True)
class DsaBatchMeta:
    """Immutable logical addresses for packed TND inputs.

    IDs use the ``global_packed`` namespace: zero-based offsets in the global
    packed batch. Sequence and causal position follow ``global_cu_seqlens``
    (including its initial zero), independently of Q/K storage order. All local
    rows are valid; internal tile padding is not part of these ID lists.

    ``token_owners`` uses indices into ordered ``cp_ranks``, not WORLD ranks.
    ``token_local_offsets`` addresses each owner's original local storage.
    K rows may be an owner-local shard or a replicated/reordered buffer; missing
    selected K must be fetched before evaluating the attention oracle.
    ``root_pes`` explicitly maps each CP member into a future SHMEM root.
    Host metadata is constructed outside the training hot path.
    """

    global_cu_seqlens: tuple[int, ...]
    q_global_ids: tuple[int, ...]
    kv_global_ids: tuple[int, ...]
    token_owners: tuple[int, ...]
    token_local_offsets: tuple[int, ...]
    cp_ranks: tuple[int, ...] = (0,)
    root_pes: tuple[int, ...] = (0,)
    cp_rank: int = 0
    layout_id: str = "packed"
    layer: int = 0
    microbatch: int = 0
    invocation: int = 0
    heap_generation: int = 0
    index_namespace: str = "global_packed"

    def __post_init__(self) -> None:
        for name in ("global_cu_seqlens", "q_global_ids", "kv_global_ids", "token_owners",
                     "token_local_offsets", "cp_ranks", "root_pes"):
            _integer_tuple(name, getattr(self, name))
        lengths = self.global_cu_seqlens
        if len(lengths) < 2 or lengths[0] != 0 or any(a >= b for a, b in pairwise(lengths)):
            raise ValueError("global_cu_seqlens must start at zero and describe nonempty sequences")
        if self.index_namespace != "global_packed":
            raise ValueError("only the global_packed index namespace is supported")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id must be a nonempty string")
        self._validate_topology()
        for name in ("layer", "microbatch", "invocation", "heap_generation"):
            value = getattr(self, name)
            if not _is_integer(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("q_global_ids", "kv_global_ids"):
            ids = getattr(self, name)
            if len(set(ids)) != len(ids) or any(token < 0 or token >= lengths[-1] for token in ids):
                raise ValueError(f"{name} must contain unique valid global token IDs")
        if any(self.token_owners[token] != self.cp_rank for token in self.q_global_ids):
            raise ValueError("local queries must belong to cp_rank")

    def _validate_topology(self) -> None:
        for name in ("cp_ranks", "root_pes"):
            members = getattr(self, name)
            if not members or len(set(members)) != len(members) or min(members) < 0:
                raise ValueError(f"{name} must contain unique nonnegative members")
        if len(self.root_pes) != len(self.cp_ranks):
            raise ValueError("root_pes must map every ordered CP member")
        if not _is_integer(self.cp_rank) or not 0 <= self.cp_rank < len(self.cp_ranks):
            raise ValueError("cp_rank must be an index into cp_ranks")
        total = self.global_cu_seqlens[-1]
        if len(self.token_owners) != total or len(self.token_local_offsets) != total:
            raise ValueError("owner and offset maps must cover every global token")
        if any(not 0 <= owner < len(self.cp_ranks) for owner in self.token_owners):
            raise ValueError("token owner is outside the ordered CP group")
        addresses = tuple(zip(self.token_owners, self.token_local_offsets))
        if any(offset < 0 for offset in self.token_local_offsets) or len(set(addresses)) != total:
            raise ValueError("owner-local addresses must be nonnegative and unique")
        for owner in range(len(self.cp_ranks)):
            offsets = sorted(offset for rank, offset in addresses if rank == owner)
            if offsets != list(range(len(offsets))):
                raise ValueError("each owner's local offsets must form a contiguous storage range")

    @classmethod
    def packed(cls, sequence_lengths: tuple[int, ...]) -> "DsaBatchMeta":
        """Construct a complete single-rank packed batch without alignment limits."""
        _integer_tuple("sequence_lengths", sequence_lengths)
        if not sequence_lengths or min(sequence_lengths) <= 0:
            raise ValueError("sequence_lengths must contain positive lengths")
        cumulative = [0]
        for length in sequence_lengths:
            cumulative.append(cumulative[-1] + length)
        ids = tuple(range(cumulative[-1]))
        return cls(tuple(cumulative), ids, ids, (0,) * len(ids), ids)

    @property
    def global_valid_queries(self) -> int:
        """Return the global query count for training over the complete packed batch."""
        return self.global_cu_seqlens[-1]

    @property
    def cache_identity(self) -> tuple:
        """Return the layer/invocation/layout/heap identity for activation reuse."""
        return (self.layer, self.microbatch, self.invocation, self.layout_id, self.heap_generation)

    def sequence_position(self, global_id: int) -> tuple[int, int]:
        """Resolve a logical token ID into its sequence and causal position."""
        if not _is_integer(global_id) or not 0 <= global_id < self.global_valid_queries:
            raise ValueError("global_id is outside the packed batch")
        sequence = bisect_right(self.global_cu_seqlens, global_id) - 1
        return sequence, global_id - self.global_cu_seqlens[sequence]


@dataclass(frozen=True)
class DsaLossNormalization:
    """Scale a local KL sum for an explicitly declared downstream reducer.

    ``reducer_divisor`` is the divisor applied when summing these CP
    contributions: one for a sum and CP size for a CP average. It must not
    include unrelated DP averages. The training adapter is responsible for
    declaring the actual reducer and applying the auxiliary upstream scale.
    """

    global_valid_queries: int
    reducer_divisor: int = 1

    def __post_init__(self) -> None:
        for name in ("global_valid_queries", "reducer_divisor"):
            value = getattr(self, name)
            if not _is_integer(value) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def local_sum_scale(self) -> float:
        """Return the factor mapping a local sum to the global mean gradient."""
        return self.reducer_divisor / self.global_valid_queries
