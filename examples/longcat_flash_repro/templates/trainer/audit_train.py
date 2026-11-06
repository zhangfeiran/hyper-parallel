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
# Standalone accelerator workers use the native Torch APIs required by the project.
# pylint: disable=forbidden-backend-import
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from functools import partial, wraps
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashRMSNorm

import hyper_parallel
from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.multicore.modules.mega_moe.module import _MegaMoeExecutionResources
from hyper_parallel.models.longcat_flash.adapter.experts import LongcatFlashMegaMoeExperts
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer

ROOT = Path(os.environ["HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["RANK"])


def save(name: str, value: Any) -> None:
    """Write a rank report outside the measurement interval.

    Args:
        name: Artifact or case label.
        value: Complete nested value to inspect.
    """
    (ROOT / f"{name}-rank{rank}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def hash_value(value: Any, digest: Any) -> None:
    """Add the complete value and tensor metadata to a bounded hash stream.

    Args:
        value: Complete nested value to inspect.
        digest: Streaming hash accumulator.
    """
    if isinstance(value, torch.Tensor):
        local = value.to_local() if hasattr(value, "to_local") else value
        data = local.detach().reshape(-1)
        digest.update(str((tuple(local.shape), local.dtype)).encode())
        for offset in range(0, data.numel(), 16 * 1024 * 1024):
            digest.update(
                data[offset : offset + 16 * 1024 * 1024].cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
    elif dataclasses.is_dataclass(value) and (not isinstance(value, type)):
        digest.update(type(value).__qualname__.encode())
        for field in dataclasses.fields(value):
            if field.init:
                digest.update(field.name.encode())
                hash_value(getattr(value, field.name), digest)
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            hash_value(value[key], digest)
    elif isinstance(value, (list, tuple)):
        digest.update(str(type(value)).encode())
        for item in value:
            hash_value(item, digest)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, sort_keys=True).encode())
    else:
        raise TypeError(f"Unsupported state identity value: {type(value).__name__}")


def sha(value: Any) -> str:
    """Hash a complete nested model or training state.

    Args:
        value: Complete nested value to inspect.
    """
    digest = hashlib.sha256()
    hash_value(value, digest)
    return digest.hexdigest()


config = parse_training_args()
trainer = TextTrainer(config)
base = trainer.base
model = base.model
architecture_keys = [
    "vocab_size",
    "hidden_size",
    "ffn_hidden_size",
    "expert_ffn_hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "head_dim",
    "max_position_embeddings",
    "moe_topk",
    "n_routed_experts",
    "zero_expert_num",
    "routed_scaling_factor",
    "rms_norm_eps",
]
reference_config = LongcatFlashConfig.from_dict(
    {"num_layers": 1, "use_cache": False, "n_routed_experts": 128, "zero_expert_num": 64}
)
architecture = {module_name: getattr(model.config, module_name) for module_name in architecture_keys}
if model.config.num_layers != 1 or architecture != {
    module_name: getattr(reference_config, module_name) for module_name in architecture_keys
}:
    raise RuntimeError("original widths and EP16 local ownership of eight real/four identity experts required")
if Path(hyper_parallel.__file__).resolve().parents[1] != Path.cwd():
    raise RuntimeError("wrong editable checkout")
norm_classes = {
    module_name: type(expert_module).__name__
    for module_name, expert_module in model.named_modules()
    if module_name == "model.norm" or ".input_layernorm." in module_name or ".post_attention_layernorm." in module_name
}
if len(norm_classes) != 5:
    raise RuntimeError("missing one-layer HF normalization coverage")
for module_name, expert_module in model.named_modules():
    if module_name in norm_classes and (not isinstance(expert_module, LongcatFlashRMSNorm)):
        raise RuntimeError("unexpected non-HF normalization: " + module_name)
counts = {}
routing = []
loss_dtypes = []
original_fb = base.forward_backward_step


@wraps(original_fb)
def forward_backward(*args: Any, **kwargs: Any) -> Any:
    """Retain the actual objective dtype from the Trainer."""
    loss, metrics = original_fb(*args, **kwargs)
    loss_dtypes.append(str(loss.dtype))
    return (loss, metrics)


base.forward_backward_step = forward_backward


def capture_routing(_module: Any, _args: tuple, output: Any, *, name: str) -> None:
    """Record learned routing for aligned input identity checks.

    Args:
        _module: Observed layer supplied by the hook interface.
        _args: Original hook arguments.
        output: Worker output or fresh report directory.
        name: Artifact or case label.
    """
    routing.append((name, output[-2].detach(), output[-1].detach()))


def count_call(_module: Any, _args: tuple, *, name: str) -> None:
    """Count executed layer entries including checkpoint recomputation.

    Args:
        _module: Observed layer supplied by the hook interface.
        _args: Original hook arguments.
        name: Artifact or case label.
    """
    counts[name] = counts.get(name, 0) + 1


handles = [
    expert_module.register_forward_pre_hook(partial(count_call, name=module_name))
    for module_name, expert_module in model.named_modules()
    if module_name.startswith("model.layers.") and module_name.count(".") == 2
]
handles.extend(
    (
        expert_module.register_forward_hook(partial(capture_routing, name=module_name))
        for module_name, expert_module in model.named_modules()
        if module_name.endswith(".mlp.router")
    )
)
world = dist.get_world_size()
if world != 16:
    raise RuntimeError("EP16 trainer required")
for expert_module in model.modules():
    if isinstance(expert_module, LongcatFlashMegaMoeExperts):
        if (
            expert_module.ep_size != 16
            or expert_module.num_experts // 16 != 8
            or expert_module.zero_expert_num // 16 != 4
        ):
            raise RuntimeError("wrong local expert counts")
original_begin = trainer.on_train_begin


def begin() -> None:
    """Record the complete restored state before aligned updates."""
    original_begin()
    parameters = {name: sha(param) for name, param in model.named_parameters()}
    optimizers = base.optimizer if isinstance(base.optimizer, list) else [base.optimizer]
    schedulers = base.lr_scheduler if isinstance(base.lr_scheduler, list) else [base.lr_scheduler]
    optimizer_states = [opt.state_dict() for opt in optimizers]
    save(
        "initial",
        {
            "norm_classes": norm_classes,
            "global_step": base.state.global_step,
            "model_architecture": architecture,
            "parameters": parameters,
            "optimizer_sha256": sha(optimizer_states),
            "scheduler_sha256": sha([s.state_dict() for s in schedulers]),
            "python_rng_sha256": sha(random.getstate()),
            "cpu_rng_sha256": sha(torch.get_rng_state()),
            "npu_rng_sha256": sha(torch.npu.get_rng_state()),
            "optimizer_classes": [type(optimizer).__name__ for optimizer in optimizers],
        },
    )


trainer.on_train_begin = begin
batches = []
original_batch = base.get_batch


@wraps(original_batch)
def get_batch(iterator: Any) -> Any:
    """Record the actual replay batch before forwarding it.

    Args:
        iterator: Original Trainer batch iterator.
    """
    value = original_batch(iterator)
    batches.append(value)
    return value


base.get_batch = get_batch
original_prepare = base.prepare_optimizer_step
main_gradient_metadata = []


def prepare() -> Any:
    """Record main gradient dtypes before the optimizer step."""
    result = original_prepare()
    if os.environ.get("HP_LONGCAT_EXACT_STATE_PROBES") == "1":
        metadata = {
            name: (
                str(getattr(param, "main_grad", None).dtype) if getattr(param, "main_grad", None) is not None else None
            )
            for name, param in model.named_parameters()
        }
        main_gradient_metadata.append({"step": base.state.global_step + 1, "dtypes": metadata})
    return result


base.prepare_optimizer_step = prepare
original_step = trainer.train_step


def train_step(iterator: Any) -> Any:
    """Measure the original Trainer step and retain untimed precision probes.

    Args:
        iterator: Original Trainer batch iterator.
    """
    batches.clear()
    routing.clear()
    loss_dtypes.clear()
    lrs = [float(group["lr"]) for group in base.optimizer.param_groups]
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    metrics = original_step(iterator)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    if not all((math.isfinite(v) for v in metrics.values())):
        raise RuntimeError("nonfinite trainer metrics")
    maximum = torch.tensor([elapsed], device="npu", dtype=torch.float32)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    step = base.state.global_step
    if not loss_dtypes or set(loss_dtypes) != {"torch.float32"}:
        raise RuntimeError("objective must remain FP32")
    row = {
        "objective_dtypes": list(loss_dtypes),
        "step": step,
        "local_seconds": elapsed,
        "max_rank_seconds": maximum.item(),
        "input_sha256": sha(batches),
        "routing_sha256": sha(routing),
        "lr": lrs,
        "phase": "measured" if step >= 7 else "warmup",
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        **metrics,
    }
    heap_epochs = {}
    for name, module in model.named_modules():
        if isinstance(module, LongcatFlashMegaMoeExperts) and module.backend == "megamoe":
            resources = module._executor._resource_group.resources
            if resources is not None:
                heap_epochs[name] = resources.heap_manager.epoch
    row["heap_epochs"] = heap_epochs
    if os.environ.get("HP_LONGCAT_EXACT_STATE_PROBES") == "1":
        row["updated_model_sha256"] = {name: sha(param) for name, param in model.named_parameters()}
        row["updated_main_sha256"] = {name: sha(param.main_param) for name, param in model.named_parameters()}
        row["main_gradient_metadata"] = main_gradient_metadata[-1]
    with (ROOT / f"steps-rank{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    batches.clear()
    routing.clear()
    return metrics


heap_state = {}
original_close = _MegaMoeExecutionResources.close


def capture_close(resources: Any) -> None:
    """Record persistent capacity before automatic resource cleanup.

    Args:
        resources: Persistent resources captured before collective cleanup.
    """
    heap_state["experts"] = {
        "epoch": resources.heap_manager.epoch,
        "growth_records": list(resources.heap_manager.growth_records),
        "source_capacity": resources._token_capacity,
        "initial_source_capacity": resources.spec.local_num_tokens,
        "initial_receive_capacity": resources.spec.receive_capacity,
        "forward_hf_bf16": resources.last_execution_plan.forward_graph.get_op("swiglu").fill_config.hf_bf16,
        "backward_hf_bf16": resources.last_execution_plan.backward_graph.get_op("swiglu_grad").fill_config.hf_bf16,
    }
    return original_close(resources)


_MegaMoeExecutionResources.close = capture_close
trainer.train_step = train_step
trainer.train()
final_parameters = {}
for module_name, param in model.named_parameters():
    main = getattr(param, "main_param", None)
    local_main = main.to_local() if hasattr(main, "to_local") else main
    final_parameters[module_name] = {
        "main_dtype": str(main.dtype) if main is not None else None,
        "main_finite": bool(torch.isfinite(local_main).all()) if local_main is not None else False,
        "logical_shape": list(main.shape) if main is not None else None,
        "local_shape": list(local_main.shape) if local_main is not None else None,
        "placements": list(map(str, main.placements)) if hasattr(main, "placements") else None,
    }
for expert_module in model.modules():
    if isinstance(expert_module, LongcatFlashMegaMoeExperts):
        expert_module.close_runtime()
torch.npu.synchronize()
report = {
    "rank": rank,
    "world": world,
    "pid": os.getpid(),
    "source": hyper_parallel.__file__,
    "global_step": base.state.global_step,
    "layer_forward_entry_counts": counts,
    "gradient_checkpointing_layers": [
        module_name
        for module_name, expert_module in model.named_modules()
        if module_name in counts and getattr(expert_module, "gradient_checkpointing", False)
    ],
    "parameters": final_parameters,
    "all_main_fp32_finite": all(
        (v["main_dtype"] == "torch.float32" and v["main_finite"] for v in final_parameters.values())
    ),
    "shmem_after_close": shmem.debug_state(),
    "native_library_maps": [
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if os.environ["HP_REPRO_NATIVE_ROOT"] in line
    ],
    "HCCL_DETERMINISTIC": os.environ.get("HCCL_DETERMINISTIC"),
    "heap_state": heap_state,
    "expert_backends": {
        module_name: {
            "backend": expert_module.backend,
            "swiglu_precision": expert_module.swiglu_precision,
            "routing_backend": expert_module.routing_backend,
            "optimize_tasks": expert_module.optimize_tasks,
            "direct_weight_gradients": expert_module.direct_weight_gradients,
        }
        for module_name, expert_module in model.named_modules()
        if isinstance(expert_module, LongcatFlashMegaMoeExperts)
    },
}
save("terminal", report)
for handle in handles:
    handle.remove()
assert report["all_main_fp32_finite"], report
