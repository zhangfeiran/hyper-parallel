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
"""Correlation identities must survive nesting without retaining device state."""

import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

from examples.torch.multicore.mega_moe_profile_context import (
    ProfileContext,
    parse_scope_name,
    scope_name,
)


class TestProfileContext(unittest.TestCase):
    """Exercise context lifetime and stable layer/resource identities."""

    def test_nested_context_restores_parent_after_exception(self) -> None:
        """A failed batch must not leak its identity into the next layer."""
        names = []

        @contextmanager
        def record(name: str) -> Iterator[None]:
            """Retain scalar scope names without a device profiler."""
            names.append(name)
            yield

        context = ProfileContext(rank=2, world_size=4, capture_id="run/trace", record_function=record)
        with context.span("call", layer=3, resource_group=1, direction="backward", call_id=0):
            with (
                self.assertRaisesRegex(RuntimeError, "injected"),
                context.span("batch", batch_index=7, placement_generation=9),
            ):
                self.assertEqual(context.metadata()["layer"], 3)
                self.assertEqual(context.metadata()["batch_index"], 7)
                raise RuntimeError("injected")
            self.assertNotIn("batch_index", context.metadata())
        self.assertNotIn("layer", context.metadata())
        decoded = [parse_scope_name(name) for name in names]
        self.assertEqual(decoded[1]["placement_generation"], 9)
        self.assertEqual(decoded[1]["capture_id"], "run/trace")

    def test_layer_registry_uses_group_membership_not_process_addresses(self) -> None:
        """Layer numbering is stable across ranks with different Python object ids."""
        for rank in (0, 1):
            first, second = object(), object()
            layers = [SimpleNamespace(_resource_group=group) for group in (first, second, first)]
            context = ProfileContext(rank=rank, world_size=2, capture_id="run/trace")
            context.bind_layers(layers)
            self.assertEqual(context.layer_registry, [
                {"layer": 0, "resource_group": 0},
                {"layer": 1, "resource_group": 1},
                {"layer": 2, "resource_group": 0},
            ])

    def test_scope_codec_rejects_invalid_identity(self) -> None:
        """Unknown ordinary annotations remain separate; malformed identities fail."""
        self.assertIsNone(parse_scope_name("megamoe::function._launch_backward"))
        valid = {"capture_id": "run/trace", "rank": 0, "world_size": 2, "step": 0}
        self.assertEqual(parse_scope_name(scope_name("call", valid)), {"role": "call", **valid})
        for change in ({"rank": True}, {"rank": 2}, {"step": -1}, {"capture_id": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scope_name("call", valid | change)
