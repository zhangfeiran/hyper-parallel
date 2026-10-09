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
"""Actual dense worker queue heads, ownership and descriptor admission on CPU."""

from __future__ import annotations

import struct
import unittest
from collections import Counter
from dataclasses import replace

import torch
from torch.nn import functional

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.core.multicore.backends.dense_tile import DenseSdkTiler, HEADER, OPERATION, bind_dense_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles, simulate_dense_tiles
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from tests.common.mark_utils import arg_mark


def _plan(rows, hidden=8, intermediate=6, **policy):
    spec = mc.DenseSpec({"T": rows, "H": hidden, "PackedI": 2 * intermediate, "I": intermediate})
    return compile_dense_tiles(dense_ffn.plan(spec, intermediate_size=intermediate), DenseTilePolicy(**policy))


def _execute_queues(plan, inputs):
    values = {value.id: tensor for value, tensor in zip(plan.dense.ir.inputs, inputs)}
    for buffer in plan.dense.buffers:
        values[buffer.value_id] = torch.empty(dict(plan.dense.value_types)[buffer.value_id].shape,
                                              dtype=torch.bfloat16)
    for task in simulate_dense_tiles(plan):
        operation = plan.dense.ir.operations[task.operation]
        arguments = dict(operation.arguments)
        first, end = task.row_begin, task.row_end
        if operation.logical_name == "dense.matmul":
            right = values[arguments["right"].id]
            right = right.t() if arguments["transpose_right"] else right
            result = values[arguments["left"].id][first:end] @ right
        else:
            gate, up = values[arguments["packed"].id][first:end].float().chunk(2, dim=-1)
            result = (functional.silu(gate) * up).to(torch.bfloat16)
        values[operation.outputs[0].id][first:end] = result
    return tuple(values[value.id] for value in plan.dense.ir.outputs)


class TestDenseTile(unittest.TestCase):
    """Exercise scheduling rather than treating source emission as device proof."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_real_worker_queues_terminate_and_cover_every_row(self):
        """Feature: Resident dense forward queue expansion.
        Description: Run real queue heads at multiple prefetch/core/tail sizes.
        Expectation: Every primitive owns each row exactly once and each vector join has two signals.
        """
        for rows in (0, 1, 7, 64, 65, 129, 1025):
            for cores in (1, 3, 20):
                for prefetch in (1, 2, 4):
                    with self.subTest(rows=rows, cores=cores, prefetch=prefetch):
                        plan = _plan(rows, cube_workers=cores, prefetch_tiles=prefetch)
                        completed = simulate_dense_tiles(plan)
                        self.assertEqual(len(completed), len(plan.tasks))
                        for operation in range(3):
                            owners = Counter(row for task in completed if task.operation == operation
                                             for row in range(task.row_begin, task.row_end))
                            self.assertEqual(owners, Counter(range(rows)))
                        signals = Counter(task.trigger for task in completed)
                        for event, count in signals.items():
                            self.assertEqual(count, 2 if event % 3 == 1 else 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_empty_vector_lane_participates_in_join(self):
        """Feature: Partial tile producer counts.
        Description: Split one token between two paired AIV lanes.
        Expectation: The empty lane signals completion and down projection waits for both.
        """
        plan = _plan(1, cube_workers=1)
        vector = [task for task in plan.tasks if task.worker_kind == "vector"]
        self.assertEqual([(task.row_begin, task.row_end) for task in vector], [(0, 1), (1, 1)])
        self.assertEqual(plan.tasks[-1].dependencies, ((1, 2),))
        self.assertEqual(len(simulate_dense_tiles(plan)), 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_queue_prefetch_and_deadlock_detection(self):
        """Feature: Worker queue ordering.
        Description: Inspect producer prefetch and create an actual cyclic blocked head.
        Expectation: Gate/up tiles precede down tiles and a cyclic queue is rejected.
        """
        plan = _plan(129, cube_workers=1, prefetch_tiles=2)
        cube = plan.queues()[("cube", 0)]
        self.assertEqual([(task.operation, task.tile) for task in cube[:4]], [(0, 0), (0, 1), (2, 0), (2, 1)])
        broken = replace(plan, tasks=(replace(plan.tasks[0], dependencies=((1, 2),)), *plan.tasks[1:]))
        with self.assertRaisesRegex(ValueError, "deadlock"):
            simulate_dense_tiles(broken)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_semantic_cpu_replay_of_queues_matches_whole_ffn(self):
        """Feature: Tiled operation ownership.
        Description: Replay queue slices on CPU for zero rows, tails and nonaligned channels.
        Expectation: Logical outputs match independently executed whole dense primitives.
        """
        torch.manual_seed(11)
        for rows in (0, 1, 7, 129):
            with self.subTest(rows=rows):
                plan = _plan(rows, hidden=80, intermediate=131, cube_workers=3)
                inputs = tuple(torch.randn(shape, dtype=torch.bfloat16) * 0.05
                               for shape in ((rows, 80), (80, 262), (131, 80)))
                actual, = _execute_queues(plan, inputs)
                expected = plan.dense.materialize("cpu")(*inputs)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_generic_fanout_and_linear_transpose_need_no_ffn_matcher(self):
        """Feature: Generic resident SSA admission.
        Description: Tile two independent matrix outputs including a transposed Linear weight.
        Expectation: Both outputs preserve rows and execute through independent queues.
        """
        program = mc.from_source(
            "def fork(x, a, b):\n    first = ml.matmul(x, a)\n    second = ml.matmul(x, b, transpose_right=True)\n"
            "    return first, second\n", signature={"x": ml.Tensor[ml.bf16, (65, 4)],
                                                     "a": ml.Tensor[ml.bf16, (4, 2)],
                                                     "b": ml.Tensor[ml.bf16, (5, 4)]},
            symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        plan = compile_dense_tiles(program.plan(mc.DenseSpec({})), DenseTilePolicy(cube_workers=1))
        inputs = tuple(torch.randn(shape, dtype=torch.bfloat16) for shape in ((65, 4), (4, 2), (5, 4)))
        actual = _execute_queues(plan, inputs)
        for result, expected in zip(actual, (inputs[0] @ inputs[1], inputs[0] @ inputs[2].t())):
            torch.testing.assert_close(result, expected)
        self.assertTrue(bind_dense_tiles(plan).matmul_shapes[1][-1])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_descriptor_has_exact_golden_abi_and_no_addresses(self):
        """Feature: Dense device task ABI.
        Description: Serialize a nonaligned FFN to the native 32-byte header and 36-byte operations.
        Expectation: Compact SSA slots, primitive widths and two-vector joins match golden bytes.
        """
        binding = bind_dense_tiles(_plan(129, hidden=80, intermediate=131, cube_workers=2))
        golden = struct.pack("<8I", 0x444E5331, 1, 129, 64, 2, 2, 3, 6)
        golden += struct.pack("<7IiI", 0, 0, 1, 3, 262, 80, 0, -1, 1)
        golden += struct.pack("<7IiI", 1, 3, 0, 4, 131, 262, 0, 0, 1)
        golden += struct.pack("<7IiI", 0, 4, 2, 5, 80, 131, 0, 1, 2)
        self.assertEqual((HEADER.size, OPERATION.size), (32, 36))
        self.assertEqual(binding.config, golden)
        self.assertEqual(binding.value_ids, tuple(range(6)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_admission_rejects_reductions_dtype_and_event_overflow(self):
        """Feature: Resident forward capability boundary.
        Description: Request token contraction, FP32 storage and overflowing native event offsets.
        Expectation: Unsupported schedules fail before tile expansion or a native call.
        """
        for source, signature in (
                ("def reduction(x, y):\n    return ml.matmul(x, y, transpose_left=True)\n",
                 {"x": ml.Tensor[ml.bf16, (4, 3)], "y": ml.Tensor[ml.bf16, (4, 2)]}),
                ("def fp32(x, y):\n    return ml.matmul(x, y)\n",
                 {"x": ml.Tensor[ml.fp32, (4, 3)], "y": ml.Tensor[ml.fp32, (3, 2)]})):
            program = mc.from_source(source, signature=signature, symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
            with self.subTest(source=source), self.assertRaises(ValueError):
                compile_dense_tiles(program.plan(mc.DenseSpec({})))
        with self.assertRaisesRegex(ValueError, "uint32"):
            _plan(2**31 - 1, rows_per_tile=1)
        with self.assertRaisesRegex(ValueError, "nonzero channel"):
            _plan(1, hidden=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sdk_bank_uses_reported_size_and_hardware_limits(self):
        """Feature: SDK tiling boundary.
        Description: Supply a host tiler double with explicit core count and raw struct size.
        Expectation: Empty tokens make no tiling call and worker oversubscription is rejected.
        """
        tiler = DenseSdkTiler.__new__(DenseSdkTiler)
        tiler.soc, tiler.cube_workers, tiler.tiling_bytes = "test-target", 2, 200
        calls = []

        def _generate(_soc, rows, tile, columns, contracted, transpose, block, capacity):
            calls.append((rows, tile, columns, contracted, transpose, capacity))
            block.raw = struct.pack("<5I", rows, tile, columns, contracted, transpose) + bytes(capacity - 20)
            return 0

        tiler.generate = _generate
        plan = _plan(129, cube_workers=2)
        bank = tiler.bank(plan, bind_dense_tiles(plan))
        self.assertEqual(len(bank), 600)
        self.assertEqual(bank[200:400], bytes(200))
        self.assertEqual(calls, [(129, 64, 12, 8, 0, 200), (129, 64, 8, 6, 0, 200)])
        empty = _plan(0, cube_workers=2)
        self.assertEqual(tiler.bank(empty, bind_dense_tiles(empty)), bytes(600))
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(ValueError, "hardware target"):
            oversubscribed = _plan(1, cube_workers=3)
            tiler.bank(oversubscribed, bind_dense_tiles(oversubscribed))
