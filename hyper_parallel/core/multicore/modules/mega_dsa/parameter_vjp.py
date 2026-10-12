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
"""Explicit pre-leaf parameter VJP capture for one caller-owned reduction cycle."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ParameterVjpCapture:
    """Collect full-precision intermediate weight VJPs before their original leaf cast.

    Original Parameters, leaf gradients and checkpoint keys remain intact.
    The caller brackets each backward or accumulation cycle, consumes this
    capture exactly once, and reduces captured tensors before one leaf-dtype
    cast. Intermediate hooks perform no collective and never mutate incoming
    gradients. Concurrent backward cycles are unsupported.
    """

    def __init__(self, parameters: Mapping[str, nn.Parameter]) -> None:
        """Bind named plain floating Parameters after physical projection sharding."""
        if not parameters or any(not isinstance(name, str) or not name for name in parameters):
            raise ValueError('parameter VJP capture requires declared nonempty names')
        if any(type(value) is not nn.Parameter or value.layout != torch.strided  # pylint: disable=unidiomatic-typecheck
               or value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
               for value in parameters.values()):
            raise TypeError('parameter VJP capture requires plain dense floating Parameters')
        if len({id(value) for value in parameters.values()}) != len(parameters):
            raise ValueError('parameter VJP capture rejects tied declared parameters')
        self.parameters = dict(parameters)
        self._gradients = {}
        self._state = 'idle'

    def view(self, name: str, *, dtype: torch.dtype) -> torch.Tensor:
        """Create a differentiable intermediate view with a pre-leaf VJP hook.

        Views may be prepared by several forwards before backward, or during
        checkpoint recomputation. A float/double identity view keeps hooks off
        the Parameter itself. The original leaf's version is checked in the hook.
        """
        parameter = self.parameters[name]
        expected = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        if dtype != expected:
            raise ValueError('parameter VJP capture requires the declared full-precision compute dtype')
        value = parameter.to(dtype=dtype).view_as(parameter)
        if value.requires_grad:
            version = parameter._version

            def _capture(gradient):
                if self._state != 'collecting':
                    raise RuntimeError('parameter VJP capture requires an active backward cycle')
                if parameter._version != version:
                    raise RuntimeError('parameter changed after its captured projection forward')
                detached = gradient.detach()
                if name in self._gradients:
                    self._gradients[name].add_(detached)
                else:
                    self._gradients[name] = detached.clone()
                return gradient

            value.register_hook(_capture)
        return value

    def begin_backward(self) -> None:
        """Start a fresh cycle; previous data must first be consumed or explicitly discarded."""
        if self._state != 'idle':
            raise RuntimeError('parameter VJP capture has an unfinished or unconsumed backward cycle')
        self._gradients = {}
        self._state = 'collecting'

    def finish_backward(self) -> None:
        """Seal capture after the caller's complete backward or accumulation sequence."""
        if self._state != 'collecting':
            raise RuntimeError('parameter VJP capture has no active backward cycle')
        self._state = 'ready'

    def take(self) -> dict[str, torch.Tensor | None]:
        """Transfer owned full-precision gradients once, including None for unused parameters."""
        if self._state != 'ready':
            raise RuntimeError('parameter VJP capture is not ready for reduction')
        result = {name: self._gradients.get(name) for name in self.parameters}
        self._gradients = {}
        self._state = 'idle'
        return result

    def discard(self) -> None:
        """Release a failed or intentionally ignored cycle without retaining stale gradients."""
        self._gradients = {}
        self._state = 'idle'
