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
"""显式 Linear 的来源、共享设备队列、独立梯度及检查点契约。"""

from __future__ import annotations

import struct
import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional

import hyper_parallel.core.multicore.frontend as mc
import hyper_parallel.core.multicore.language as ml
from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.core.multicore.backends.dense_backward import DenseBackwardSdkTiler, bind_dense_backward_tiles
from hyper_parallel.core.multicore.backends.dense_codegen import generate_dense_cpp
from hyper_parallel.core.multicore.backends.dense_linear import DenseLinearSdkTiler, bind_dense_linear_tiles
from hyper_parallel.core.multicore.backends.dense_reference import ACCESS, REFERENCE_FIELDS, reference_access
from hyper_parallel.core.multicore.backends.dense_tile import bind_dense_tiles
from hyper_parallel.core.multicore.compiler.dense_backward import compile_dense_backward
from hyper_parallel.core.multicore.compiler.dense_backward_tiles import compile_dense_backward_tiles
from hyper_parallel.core.multicore.compiler.dense_tile import DenseTilePolicy, compile_dense_tiles
from hyper_parallel.core.multicore.frontend.examples.dense_ffn import dense_ffn
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.multicore.modules.mega_ffn.module import MegaFFN
from tests.common.mark_utils import arg_mark
from tests.ut.core.multicore.backends.test_dense_backward_binding import _decode_queues


def _plan(rows):
    return linear_ffn.plan(mc.DenseSpec({"T": rows, "H": 17, "I": 19, "PackedI": 38}), intermediate_size=19)


def _source(values):
    x, gate_up, down = values
    gate, up = functional.linear(x, gate_up).float().chunk(2, dim=-1)  # pylint: disable=not-callable
    activation = (functional.silu(gate) * up).to(x.dtype)
    return functional.linear(activation, down)  # pylint: disable=not-callable


def _tiler(kind, queries):
    tiler = kind.__new__(kind)
    tiler.sdk = SimpleNamespace(cube_workers=20, soc="Ascend910B3", tiling_bytes=200, workspace_bytes=4096)
    tiler.generate = lambda *_args: 0
    tiler.reference_record, tiler.reference_shapes = None, ()

    def _query(shapes):
        queries.append(shapes)
        results = []
        for shape in shapes:
            fields = {name: 0 for name in REFERENCE_FIELDS}
            fields.update(m_ori=shape[0], n_ori=shape[1], k_ori=shape[2])
            raw = struct.pack("<38i", *(fields[name] for name in REFERENCE_FIELDS)).hex()
            results.append({"shape": shape, "native_family": "MatMulV3", "v3_with_split_k_false_true": (1, 1),
                            "run_info": {"tiling_key": 0, "tiling_data": raw}})
        return {"results": results}

    tiler.reference = SimpleNamespace(query=_query)
    return tiler


class TestDenseLinear(unittest.TestCase):
    """独立源算子、原始权重布局及解码后的真实工作队列必须一致。"""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_full_l1_reference_requires_complete_k_and_partition_proof(self):
        """Feature: 单侧完整 L1 K 模板。
        Description: 使用真实 SDK 满载查询，分别破坏 K 覆盖、因子、分区及数据类型。
        Expectation: 仅显式原始 Linear 可使用已证明的顺序规则，破坏任一条件均拒绝。
        """
        path = Path(__file__).parent / "fixtures/dense_reference_linear_full_l1_910b3.json"
        result = json.loads(path.read_text())["query"]["results"][0]
        shape = tuple(result["shape"])
        self.assertEqual(reference_access(result, shape, 20, allow_left_transpose=True,
                                           allow_sequential_reference=True),
                         (bytes(ACCESS.size), "sdk_full_l1_k_sequential"))
        self.assertEqual(reference_access(result, shape, 20, allow_left_transpose=True)[1],
                         "unsupported_reference_template")
        for name, value in (("kbl1_16", 63), ("kal1_16", 0), ("kbl1_factor", 2), ("kl1_times", 7),
                            ("kal0_factor", 1), ("kbl0_factor", 15), ("k_dim", 2), ("m_dim", 0),
                            ("n_dim", 1), ("datatype_bf16", 0), ("hf32_flag", 1)):
            with self.subTest(field=name):
                corrupt = dict(result, run_info=dict(result["run_info"]))
                fields = dict(zip(REFERENCE_FIELDS, struct.unpack("<38i", bytes.fromhex(
                    result["run_info"]["tiling_data"]))))
                fields[name] = value
                corrupt["run_info"]["tiling_data"] = struct.pack(
                    "<38i", *(fields[key] for key in REFERENCE_FIELDS)).hex()
                reason = reference_access(corrupt, shape, 20, allow_left_transpose=True,
                                          allow_sequential_reference=True)[1]
                self.assertEqual(reason, "unsupported_reference_template")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sdk_weight_reference_maps_mn_access_without_changing_physical_tn(self):
        """Feature: 原始权重梯度的转置坐标。
        Description: 使用真实 SDK 数值查询，独立核对物理 TN 分块参数及映射后的 bank。
        Expectation: 借用权重不复制，M/N 跨度和核分区转置，完整 K 块及禁用标志保持原样。
        """
        program = mc.from_source("def project(x, weight):\n    return ml.linear(x, weight)\n",
                                 signature={"x": ml.Tensor[ml.bf16, (1024, 1024)],
                                            "weight": ml.Tensor[ml.bf16, (4096, 1024)]},
                                 symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        forward = program.plan(mc.DenseSpec({}))
        backward = compile_dense_backward_tiles(compile_dense_backward(forward), DenseTilePolicy(cube_workers=20),
                                                 input_mask=(False, True))
        binding = bind_dense_backward_tiles(backward)
        path = Path(__file__).parent / "fixtures/dense_reference_linear_weight_coordinates_910b3.json"
        record = json.loads(path.read_text())["query"]
        calls, queries = [], []
        tiler = _tiler(DenseBackwardSdkTiler, queries)
        tiler.reference = SimpleNamespace(query=lambda shapes: queries.append(shapes) or record)
        tiler.generate = lambda *args: calls.append(args[1:8]) or 0
        bank = tiler.bank(backward, binding)
        self.assertEqual(binding.matmul_shapes, ((4096, 1024, 1024, True, False),))
        self.assertEqual(queries, [((1024, 4096, 1024, True, False),)])
        self.assertEqual(calls[0][0], 4096)
        self.assertEqual(calls[0][2], 1024)
        self.assertEqual(calls[0][4:], (1024, 1, 0))
        self.assertEqual(ACCESS.unpack_from(bank, tiler.sdk.tiling_bytes), (896, 256, 5, 4, 128, 0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_single_k_block_reference_requires_actual_complete_l0_geometry(self):
        """Feature: 单 K 块顺序模板。
        Description: 使用实际 SDK 查询快照，再破坏 L0 的完整 K 覆盖。
        Expectation: 只有显式准入的单块参考保留顺序；多个 K 块仍须原有访问证明。
        """
        path = Path(__file__).parent / "fixtures/dense_reference_linear_single_k_910b3.json"
        record = json.loads(path.read_text())
        for result in record["results"]:
            with self.subTest(shape=result["shape"]):
                shape = tuple(result["shape"])
                self.assertEqual(reference_access(result, shape, 20)[1], "unsupported_reference_template")
                flags = {"allow_left_transpose": True, "allow_right_transpose": True,
                         "allow_sequential_reference": True}
                self.assertEqual(reference_access(result, shape, 20, **flags),
                                 (bytes(ACCESS.size), "sdk_single_k_block_sequential"))
                fields = list(struct.unpack("<38i", bytes.fromhex(result["run_info"]["tiling_data"])))
                fields[REFERENCE_FIELDS.index("k_l0")] -= 1
                corrupt = {**result, "run_info": {**result["run_info"],
                           "tiling_data": struct.pack("<38i", *fields).hex()}}
                self.assertEqual(reference_access(corrupt, shape, 20, **flags)[1], "unsupported_reference_template")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_single_partition_reference_preserves_non_aligned_multi_l0_k(self):
        """Feature: 非对齐 K 的单分区顺序模板。
        Description: 使用实际 TN 查询，分别破坏核分区、完整 L1、L0 因子与 split-K 条件。
        Expectation: 原始单分区可准入，其他配置仍须独立访问顺序证明。
        """
        path = Path(__file__).parent / "fixtures/dense_reference_linear_single_partition_910b3.json"
        result, = json.loads(path.read_text())["results"]
        shape = tuple(result["shape"])
        flags = {"allow_left_transpose": True, "allow_sequential_reference": True}
        self.assertEqual(reference_access(result, shape, 20)[1], "unsupported_reference_template")
        self.assertEqual(reference_access(result, shape, 20, **flags),
                         (bytes(ACCESS.size), "sdk_single_partition_sequential"))
        raw = struct.unpack("<38i", bytes.fromhex(result["run_info"]["tiling_data"]))
        for name, value in (("m_dim", 2), ("n_dim", 2), ("k_dim", 2), ("kal1_16", 6),
                            ("kbl1_16", 6), ("k_l0", 0), ("kal0_factor", 2), ("kbl0_factor", 2)):
            with self.subTest(field=name):
                fields = list(raw)
                fields[REFERENCE_FIELDS.index(name)] = value
                corrupt = {**result, "run_info": {**result["run_info"],
                           "tiling_data": struct.pack("<38i", *fields).hex()}}
                self.assertEqual(reference_access(corrupt, shape, 20, **flags)[1], "unsupported_reference_template")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_original_linear_outputs_and_gradients_keep_nk_parameters(self):
        """Feature: 原始 Linear 布局。
        Description: 对照独立 F.linear 图，覆盖空输入、尾部及较大行数。
        Expectation: 输出和全部梯度逐值一致，输入地址与版本不变。
        """
        torch.manual_seed(83)
        for rows in (0, 1, 7, 129):
            with self.subTest(rows=rows):
                module = MegaFFN(17, 19, program=linear_ffn)
                x = torch.randn(rows, 17, dtype=torch.bfloat16, requires_grad=True)
                values = (x, module.gate_up, module.down)
                copies = tuple(value.detach().clone().requires_grad_() for value in values)
                identities = tuple((value.data_ptr(), value._version) for value in values)
                actual, expected = module(x), _source(copies)
                cotangent = torch.randn_like(actual)
                actual_gradients = torch.autograd.grad(actual, values, cotangent)
                expected_gradients = torch.autograd.grad(expected, copies, cotangent)
                self.assertEqual(module.weight_layout, "nk")
                self.assertEqual((tuple(module.gate_up.shape), tuple(module.down.shape)), ((38, 17), (17, 19)))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                self.assertEqual(tuple((value.data_ptr(), value._version) for value in values), identities)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_linear_backward_keeps_physical_nk_and_original_transposed_numerical_reference(self):
        """Feature: Linear 配套数值来源。
        Description: 从语义图生成真实输入梯度和源布局权重梯度。
        Expectation: dX 使用 NN，dW 的物理 TN 直接写 NK，参考 X.T @ dY 并映射转置坐标。
        """
        forward = _plan(129)
        backward = compile_dense_backward(forward)
        matrices = [node for node in backward.nodes if node.matrix is not None]
        self.assertEqual([node.kind for node in matrices],
                         ["matmul_left_grad", "linear_weight_grad", "matmul_left_grad", "linear_weight_grad"])
        self.assertEqual([node.matrix for node in matrices], [(129, 19, 17, False, False), (17, 19, 129, True, False),
                                                            (129, 17, 38, False, False), (38, 17, 129, True, False)])
        self.assertEqual([node.numeric_reference.matrix for node in matrices],
                         [(129, 19, 17, False, False), (19, 17, 129, True, False),
                          (129, 17, 38, False, False), (17, 38, 129, True, False)])
        self.assertEqual([node.numeric_reference.coordinates_transposed for node in matrices],
                         [False, True, False, True])
        self.assertTrue(all(not node.transpose_output for node in matrices))
        self.assertEqual([node.numeric_reference.source for node in matrices],
                         ["dense.linear.input_grad", "dense.linear.weight_grad"] * 2)
        source = generate_dense_cpp(forward, "hp_dense_linear_test")
        self.assertIn("at::linear(left, weight)", source)
        self.assertIn("(dense_matmul((v4).t(), gradients[5])).t()", source)
        self.assertIn("(dense_matmul((v0).t(), gradients[3])).t()", source)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_shared_forward_abi_matches_independent_native_queue_decoder(self):
        """Feature: 共享矩阵 ABI 的真实前向队列。
        Description: 独立解码设备循环，比较窗口、列分区、空通道与所有依赖。
        Expectation: 原始前向计划只改变显式事件编号，全部片段和队头顺序一致。
        """
        for rows in (0, 1, 65, 129):
            for workers in (1, 3, 20):
                for columns in (1, 4):
                    with self.subTest(rows=rows, workers=workers, columns=columns):
                        plan = compile_dense_tiles(_plan(rows), DenseTilePolicy(cube_workers=workers,
                                                                              column_workers=columns))
                        binding = bind_dense_linear_tiles(plan)
                        mapping = binding.event_map
                        expected = {worker: [(task.operation, task.tile, task.row_begin, task.row_end,
                                              task.column_begin, task.column_end, task.column_tile,
                                              tuple(sorted((mapping[event], count)
                                                           for event, count in task.dependencies)),
                                              mapping[task.trigger]) for task in tasks]
                                    for worker, tasks in plan.queues().items()}
                        self.assertEqual(_decode_queues(binding), expected)
                        binding.admit_native()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_sdk_queries_pair_forward_nt_with_original_backward_geometry(self):
        """Feature: 配套原始参考查询。
        Description: 记录前向和反向查询，复用 bank 并导出真实数值来源。
        Expectation: 查询保留两侧转置和全 K，重复构建不重复查询，损坏绑定被拒绝。
        """
        forward = compile_dense_tiles(_plan(129), DenseTilePolicy(cube_workers=3))
        binding, queries = bind_dense_linear_tiles(forward), []
        tiler = _tiler(DenseLinearSdkTiler, queries)
        bank = tiler.bank(forward, binding)
        self.assertEqual(bank, tiler.bank(forward, binding))
        self.assertEqual(queries, [((129, 38, 17, False, True), (129, 17, 19, False, True))])
        manifest = tiler.export_manifest(binding, bank)
        self.assertEqual(manifest["direction"], "forward")
        self.assertEqual(manifest["event_layout"], "operation_major")
        self.assertEqual(manifest["matrix_references"][0]["source"], "dense.linear")
        backward = compile_dense_backward_tiles(compile_dense_backward(forward.dense), forward.policy)
        backward_binding = bind_dense_backward_tiles(backward)
        reverse = _tiler(DenseBackwardSdkTiler, queries)
        reverse.bank(backward, backward_binding)
        self.assertEqual(queries[-1], tuple(value.matrix for value in backward_binding.matrix_references
                                            if value is not None))
        with self.assertRaisesRegex(ValueError, "canonical"):
            tiler.bank(forward, replace(binding, event_map=()))
        missing = _tiler(DenseLinearSdkTiler, [])
        missing.reference = None
        with self.assertRaisesRegex(ValueError, "not admitted.*no_reference_query"):
            missing.bank(forward, binding)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_checkpoint_restores_original_separate_nk_projections(self):
        """Feature: 原始 Linear 检查点布局。
        Description: 通过正式反向变换入口恢复三个源投影，down 仅需改名。
        Expectation: 原始 NK 权重逐值恢复，不依赖空操作转换器或转置复制。
        """
        source = nn.Module()
        source.gate_proj = nn.Linear(17, 19, bias=False, dtype=torch.bfloat16)
        source.up_proj = nn.Linear(17, 19, bias=False, dtype=torch.bfloat16)
        source.down_proj = nn.Linear(19, 17, bias=False, dtype=torch.bfloat16)
        adapter = MegaFFNAdapter(module=source, context={"mega_ffn_program": linear_ffn})
        adapter._weight_conversions = adapter.make_transforms()
        recovered = revert_weight_conversion(adapter, adapter.state_dict())
        self.assertEqual(set(recovered), set(source.state_dict()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(recovered[name], value, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_physical_kn_linear_and_corrupt_forward_plan_cannot_enter_resident(self):
        """Feature: 原始借用权重准入。
        Description: 尝试使用 KN Linear 权重，伪造事件数或以旧 Matmul 代替 Linear。
        Expectation: 设备路径在分配前拒绝，不通过热路径权重复制改变源布局。
        """
        program = mc.from_source("def project(x, weight):\n    return ml.linear(x, weight, transpose_right=False)\n",
                                 signature={"x": ml.Tensor[ml.bf16, (7, 17)],
                                            "weight": ml.Tensor[ml.bf16, (17, 19)]},
                                 symbols={"ml": ml}, schedule=mc.TaskDAG("dense_v1"))
        with self.assertRaisesRegex(ValueError, "original contiguous NK"):
            compile_dense_tiles(program.plan(mc.DenseSpec({})))
        plan = compile_dense_tiles(_plan(7))
        with self.assertRaisesRegex(ValueError, "shared matrix"):
            bind_dense_tiles(plan)
        with self.assertRaisesRegex(ValueError, "canonical"):
            bind_dense_linear_tiles(replace(plan, event_count=plan.event_count + 1))
        legacy = dense_ffn.plan(mc.DenseSpec({"T": 7, "H": 17, "I": 19, "PackedI": 38}), intermediate_size=19)
        with self.assertRaisesRegex(ValueError, "explicit Linear"):
            bind_dense_linear_tiles(compile_dense_tiles(legacy))
