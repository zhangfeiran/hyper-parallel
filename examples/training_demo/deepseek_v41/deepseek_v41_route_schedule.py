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
"""Declare reproducible route windows without importing the training framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RouteWindow:
    """One zero-based contiguous window, with settling excluded from measurement."""

    name: str
    profile: str
    seed: int
    start: int
    settle: int
    measured: int

    @property
    def end(self) -> int:
        """Return the exclusive optimizer-step boundary."""
        return self.start + self.settle + self.measured

    @property
    def measured_indices(self) -> range:
        """Return indices into result.steps_detail, excluding transition steps."""
        return range(self.start + self.settle, self.end)


@dataclass(frozen=True)
class RouteSchedule:
    """Keep one immutable route order shared by every backend and rank."""

    windows: tuple[RouteWindow, ...]

    def __post_init__(self) -> None:
        end, names = 0, set()
        if not self.windows:
            raise ValueError('A schedule must contain route windows')
        for window in self.windows:
            if (any(type(value) is not int for value in
                    (window.seed, window.start, window.settle, window.measured))
                    or not isinstance(window.name, str) or not isinstance(window.profile, str)):
                raise ValueError('Route windows require integer boundaries and string identities')
            if (window.start != end or window.name in names or not window.name
                    or min(window.seed, window.settle, window.measured) < 0 or window.end <= window.start):
                raise ValueError(f'Invalid contiguous route window: {window}')
            if window.profile != 'balanced' and (not math.isfinite(float(window.profile))
                                                or float(window.profile) < .001):
                raise ValueError(f'Invalid routing profile: {window.profile}')
            end = window.end
            names.add(window.name)
        if self.windows[0].measured or self.windows[0].settle < 1:
            raise ValueError('A schedule must start with a nonempty untimed primer')
        if any(not window.measured for window in self.windows[1:]):
            raise ValueError('Every window after the primer must contain measured steps')

    @property
    def steps(self) -> int:
        """Return the total optimizer steps in this initialized model process."""
        return self.windows[-1].end

    @property
    def warmup(self) -> int:
        """Return the benchmark-wide primer; each later window has its own settling."""
        return self.windows[0].end

    def at(self, index: int) -> RouteWindow:
        """Select the route before executing forward and its checkpoint replay."""
        for window in self.windows:
            if window.start <= index < window.end:
                return window
        raise ValueError(f'Step outside route schedule: {index}')

    def record(self) -> dict[str, Any]:
        """Serialize the exact schedule used for route and timing audits."""
        return {'version': 1, 'windows': [asdict(window) for window in self.windows]}

    @property
    def sha256(self) -> str:
        """Hash canonical schedule content, independent of its bundle path."""
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()


def load_schedule(path: Path) -> RouteSchedule:
    """Reject unsupported wire versions and validate all optimizer-step boundaries."""
    record = json.loads(path.read_text())
    if record['version'] != 1:
        raise ValueError('Unsupported route schedule version')
    return RouteSchedule(tuple(RouteWindow(**window) for window in record['windows']))


def default_schedule() -> RouteSchedule:
    """Prime high-water buffers once and measure three loads in the same model."""
    windows = [RouteWindow('primer', '0.1', 42, 0, 5, 0)]
    for name, profile in (('balanced', 'balanced'), ('medium', '1'), ('hot', '0.1')):
        windows.append(RouteWindow(name, profile, 42, windows[-1].end, 2, 8))
    return RouteSchedule(tuple(windows))


def dense_schedule() -> RouteSchedule:
    """Measure near-balanced profiles before growing resources for later hotspots."""
    profiles = ('balanced', '10000', '1000', '100', '50', '30', '20', '15', '12', '10',
                '8', '6', '5', '4', '3', '2.5', '2', '1.5', '1', '0.5', '0.1')
    windows = [RouteWindow('primer', 'balanced', 42, 0, 5, 0)]
    for profile in profiles:
        name = 'balanced' if profile == 'balanced' else 'alpha_' + profile.replace('.', 'p')
        windows.append(RouteWindow(name, profile, 42, windows[-1].end, 2, 7))
    return RouteSchedule(tuple(windows))
