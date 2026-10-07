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
"""Device-independent shape, tile and hardware contract for the fixed MHC family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MhcSpec:
    """Static native topology and tiles; runtime tensor pointers never enter the contract."""

    token_count: int
    hidden_size: int
    num_cube_cores: int = 24
    token_tile: int = 32
    grad_token_tile: int = 32
    need_backward: bool = True

    def __post_init__(self):
        for name in ("token_count", "hidden_size", "num_cube_cores", "token_tile", "grad_token_tile"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 < value < 1 << 32:
                raise ValueError(f"MHC {name} must be a positive uint32 integer")
        if self.num_cube_cores > 24 or self.hidden_size % 128:
            raise ValueError("Native MHC requires <=24 Cube cores and hidden size divisible by 128")
        if not isinstance(self.need_backward, bool):
            raise TypeError("MHC need_backward must be a boolean")
        if self.need_backward and (self.token_count < 2 * self.num_cube_cores or self.hidden_size > 5760):
            raise ValueError("Native MHC backward requires T >= AIV count and H <= 5760")

    @property
    def num_vector_cores(self) -> int:
        """Return the original mixed-kernel 1:2 core ratio."""
        return 2 * self.num_cube_cores
