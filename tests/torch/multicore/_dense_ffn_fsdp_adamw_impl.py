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
"""双卡 FSDP 下完整设备反向、FP32 主参数 AdamW 及磁盘检查点恢复验证。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.components.modules import SwiGLUMLP
from hyper_parallel.components.optim import Float16OptimizerWithFloat16Params
from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy, wait_grad_handle
from hyper_parallel.core.multicore.examples.mega_ffn.acceptance import _resident_backward_entry
from hyper_parallel.core.multicore.examples.mega_ffn.qwen_dense_model import DenseMLP, QwenDenseConfig
from hyper_parallel.core.multicore.examples.mega_ffn.source_contract import resident_source_entry
from hyper_parallel.core.multicore.examples.mega_ffn.source_layout import source_layout_tensors
from hyper_parallel.core.multicore.frontend.examples.common_ffn import common_ffn
from hyper_parallel.core.multicore.frontend.examples.linear_ffn import linear_ffn
from hyper_parallel.core.multicore.modules.mega_ffn.adapter import MegaFFNAdapter
from hyper_parallel.core.optimizer import get_hyper_optimizer
from tests.torch.multicore._dense_ffn_fsdp_impl import _SourceFFN, _execution_config


def _global_cpu(value: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        value = value.full_tensor()
    elif value.ndim and isinstance(parameter, DTensor):
        value = DTensor.from_local(value, parameter.device_mesh, parameter.placements,
                                   shape=tuple(parameter.shape)).full_tensor()
    return value.detach().cpu().clone()


def _optimizer(model: nn.Module) -> Float16OptimizerWithFloat16Params:
    inner = get_hyper_optimizer(model=model, muon_params=[], adamw_params=[{"params": list(model.parameters())}],
                                adamw_kwargs={"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8,
                                              "weight_decay": 0.01})
    return Float16OptimizerWithFloat16Params(inner, model)


def _state(model, optimizer, kind, *, layout="kn"):
    values = {}
    names = {"linear_fc1.weight": "gate_up", "linear_fc2.weight": "down"}
    for name, parameter in model.named_parameters():
        main = optimizer.optimizer_param_by_model_param[parameter]
        if main.dtype != torch.float32:
            raise AssertionError("FSDP 优化器必须持有 FP32 主参数")
        if kind in ("parameters", "gradients"):
            value = parameter if kind == "parameters" else parameter.grad
        elif kind == "main":
            value = main
        else:
            leaf = optimizer.chained_optimizers[0]
            if kind == "step":
                group = next(group for group in leaf.param_groups if any(item is main for item in group["params"]))
                value = torch.tensor(group.get("step", 0))
            else:
                value = leaf.state.get(main, {}).get(kind)
                if value is None:
                    value = torch.zeros_like(main)
        if value is None:
            raise AssertionError(f"FSDP 梯度缺失：{name}")
        value = _global_cpu(value, parameter)
        values[name] = value
    if layout == "source_nk":
        return source_layout_tensors(model, values)
    if layout != "kn":
        raise ValueError("FSDP 状态比较必须显式选择 kn 或 source_nk 布局")
    values = {names.get(name, name): value.t() if name in names and value.ndim else value
              for name, value in values.items()}
    return values


def _updated_state(model, optimizer, *, layout="kn"):
    return {kind: _state(model, optimizer, kind, layout=layout)
            for kind in ("parameters", "main", "exp_avg", "exp_avg_sq", "step")}


def _compare(actual, expected, *, exact=False):
    if actual.keys() != expected.keys():
        raise AssertionError("训练状态比较必须覆盖相同的逻辑字段")
    checks = {}
    for kind, values in actual.items():
        if values.keys() != expected[kind].keys():
            raise AssertionError(f"训练状态键不一致：{kind}")
        atol = 0 if exact or kind == "step" else (0.002 if kind in ("output", "dX", "gradients", "parameters")
                                                 else 1e-8)
        rtol = 0 if exact or kind == "step" else 0.02
        maximum = 0.0
        for name, value in values.items():
            reference = expected[kind][name]
            torch.testing.assert_close(value, reference, rtol=rtol, atol=atol, msg=f"{kind}: {name}")
            delta = (value.float() - reference.float()).abs()
            maximum = max(maximum, float(delta.max()) if delta.numel() else 0)
        checks[kind] = {"passed": True, "max_abs": maximum, "atol": atol, "rtol": rtol}
    return checks


def _assert_copyback(model, optimizer):
    for parameter in model.parameters():
        main = optimizer.optimizer_param_by_model_param[parameter]
        torch.testing.assert_close(parameter.to_local(), main.to_local().to(torch.bfloat16), rtol=0, atol=0)


def _advance(models, optimizers, value, cotangent, recompute, *, layout="kn"):
    records = []
    for model, optimizer in zip(models, optimizers):
        optimizer.zero_grad(set_to_none=True)
        tokens = value.detach().clone().requires_grad_()
        output = checkpoint(model, tokens, use_reentrant=False) if recompute else model(tokens)
        output.backward(cotangent)
        wait_grad_handle()
        record = {"output": {"value": output.detach().cpu()}, "dX": {"value": tokens.grad.detach().cpu()},
                  "gradients": _state(model, optimizer, "gradients", layout=layout)}
        with SkipDTensorDispatch():
            optimizer.step()
        _assert_copyback(model, optimizer)
        record.update(_updated_state(model, optimizer, layout=layout))
        records.append(record)
    return records


def _checkpoint_cpu(value):
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        return local.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _checkpoint_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_checkpoint_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_checkpoint_cpu(item) for item in value)
    return deepcopy(value)


def _restore(source, mesh, candidate, optimizer, directory, rank, *, layout="kn", program=None):
    path = directory / f"fsdp-adamw-checkpoint-rank-{rank}.pt"
    saved = {"model": {name: _global_cpu(parameter, parameter) for name, parameter in candidate.named_parameters()},
             "optimizer": _checkpoint_cpu(optimizer.state_dict())}
    torch.save(saved, path)
    restored = MegaFFNAdapter(module=source, context={"mega_ffn_execution": _execution_config(),
                                                    "mega_ffn_program": program}).to(
        torch.device("npu", rank))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  output_dtype=torch.bfloat16, cast_forward_inputs=False)
    fully_shard(restored, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    restored_optimizer = _optimizer(restored)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(loaded["model"])
    with SkipDTensorDispatch():
        restored_optimizer.load_state_dict(loaded["optimizer"])
    checks = _compare(_updated_state(restored, restored_optimizer, layout=layout),
                      _updated_state(candidate, optimizer, layout=layout), exact=True)
    _assert_copyback(restored, restored_optimizer)
    return restored, restored_optimizer, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "layout": "full_model_and_rank_local_optimizer", "checks": checks}


def _native(model, *, source_baseline=None):
    # 验收分析器读取磁盘 JSON；梯度掩码遵循相同的列表契约。
    entries = json.loads(json.dumps(model.execution_manifest()))
    if not entries:
        raise AssertionError("FSDP 必须保留真实的原生执行记录")
    for entry in entries:
        complete = (resident_source_entry(entry, source_baseline) if source_baseline is not None else
                    entry["native"].get("execution_mode") == "resident_dense_tile_candidate"
                    and _resident_backward_entry(entry))
        if not complete:
            raise AssertionError("FSDP 必须执行 resident 前向和全部声明输入的完整设备反向，禁止主机补算")
    return entries


def _fsdp_adamw(source_baseline=None):
    programs = {None: None, "common": common_ffn, "packed": linear_ffn}
    if source_baseline not in programs:
        raise ValueError("FSDP Source-NK 验收必须选择原始 common 或 packed 数值契约")
    program = programs[source_baseline]
    layout = "kn" if source_baseline is None else "source_nk"
    rank, world_size = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError("Dense FSDP AdamW 验收要求恰好两个 worker rank")
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group("hccl")
    candidate, restored = None, None
    try:
        mesh = init_device_mesh("npu", (world_size,), mesh_dim_names=("dp",))
        torch.manual_seed(17)
        source = (DenseMLP(QwenDenseConfig(hidden_size=64, intermediate_size=128))
                  if source_baseline == "common" else _SourceFFN()).to(dtype=torch.bfloat16)
        candidate = MegaFFNAdapter(module=source, context={"mega_ffn_execution": _execution_config(),
                                                         "mega_ffn_program": program}).to(device)
        baseline = (deepcopy(source) if source_baseline == "common" else SwiGLUMLP(module=deepcopy(source))).to(device)
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      output_dtype=torch.bfloat16, cast_forward_inputs=False)
        models = [candidate, baseline]
        for model in models:
            fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
        optimizers = [_optimizer(model) for model in models]
        initial = _compare(_updated_state(candidate, optimizers[0], layout=layout),
                           _updated_state(baseline, optimizers[1], layout=layout), exact=True)
        rows = []
        checkpoint_record = None
        with TemporaryDirectory(prefix="hp-ffn-fsdp-adamw-") as temporary:
            directory = Path(os.environ.get("HP_FFN_EVIDENCE_DIR", temporary))
            if source_baseline is not None:
                directory = directory / source_baseline
            directory.mkdir(parents=True, exist_ok=True)
            progress_path = directory / f"fsdp-adamw-progress-rank-{rank}.json"
            for step, tokens in enumerate((7, 129, 7, 1, 129, 7, 1, 7)):
                generator = torch.Generator(device="cpu").manual_seed(1000 + rank * 100 + step)
                value = torch.randn(tokens, 64, generator=generator, dtype=torch.bfloat16).to(device)
                cotangent = torch.randn(tokens, 64, generator=generator, dtype=torch.bfloat16).to(device)
                records = _advance(models, optimizers, value, cotangent, step in (2, 5), layout=layout)
                checks = _compare(records[0], records[1])
                for record in records:
                    if any(int(count) != step + 1 for count in record["step"].values()):
                        raise AssertionError("AdamW 参数组计数必须与实际更新次数一致")
                row = {"step": step + 1, "tokens": tokens, "recompute": step in (2, 5), "checks": checks}
                if restored is not None:
                    row["resumed_checks"] = _compare(records[2], records[0], exact=True)
                rows.append(row)
                progress_path.write_text(json.dumps({"rank": rank, "initial": initial, "steps": rows},
                                                     indent=2) + "\n", encoding="utf-8")
                if step == 3:
                    restored, restored_optimizer, checkpoint_record = _restore(
                        source, mesh, candidate, optimizers[0], directory, rank, layout=layout, program=program)
                    models.append(restored)
                    optimizers.append(restored_optimizer)
            if checkpoint_record is None or restored is None:
                raise AssertionError("训练验收必须完成磁盘恢复及后续独立更新")
            torch.npu.synchronize()
            record = {"rank": rank, "world_size": world_size, "torch": torch.__version__,
                      "torch_npu": torch_npu.__version__, "scope": "BF16 FSDP FP32-main AdamW component lifecycle",
                      "source_baseline": source_baseline, "parameter_layout": layout,
                      "model": {"hidden_size": 64, "intermediate_size": 128, "seed": 17},
                      "optimizer": {"family": "HyperParallel AdamW", "fp32_main_params": True,
                                    "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
                      "initial": initial, "steps": rows, "checkpoint": checkpoint_record,
                      "native_payloads": _native(candidate, source_baseline=source_baseline),
                      "resumed_native_payloads": _native(restored, source_baseline=source_baseline),
                      "formal_original_training_accepted": False, "performance_accepted": False}
            (directory / f"fsdp-adamw-rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n",
                                                                 encoding="utf-8")
    finally:
        if restored is not None:
            restored.close()
        if candidate is not None:
            candidate.close()
        dist.destroy_process_group()


def test_dense_ffn_fully_shard_adamw() -> None:
    """比较原有 KN 模块的独立 AdamW 轨迹、双卡磁盘恢复及后续更新。"""
    _fsdp_adamw()


def test_dense_source_packed_fully_shard_adamw() -> None:
    """比较原始 packed 与 NK Linear 的独立 AdamW 轨迹、双卡磁盘恢复及后续更新。"""
    _fsdp_adamw("packed")


def test_dense_source_common_fully_shard_adamw() -> None:
    """比较原始 common 与三个 NK Linear 的独立 AdamW 轨迹、双卡磁盘恢复及后续更新。"""
    _fsdp_adamw("common")
