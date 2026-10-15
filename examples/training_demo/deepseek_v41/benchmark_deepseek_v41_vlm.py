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
"""Compare native/GMM, MegaMoE push and pull on the production VLM training loop."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist
import yaml

from examples.training_demo.deepseek_v41.deepseek_v41_random_init import initialize_random_shards
from examples.training_demo.deepseek_v41.deepseek_v41_moe_timing import MoeTiming
from hyper_parallel.core.multicore import _loader, shmem
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.models.deepseek_v41.adapter.policies import activation_checkpointing


def _recipe(args: argparse.Namespace, work: Path, world: int) -> Any:
    modality = getattr(args, "modality", "vlm")
    filename = "train_deepseek_v41_vlm_online.yaml" if modality == "vlm" else "train_deepseek_v41_online.yaml"
    recipe = yaml.safe_load(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
    recipe["model"].update(config_path=args.model_dir, engram_assets_path=args.engram_assets,
                          num_routed_experts=args.experts, root_owned_vision=modality == "vlm")
    recipe["model_init_dtype"] = "float32"
    packing_mode = getattr(args, "packing_mode", "packed")
    if modality == "vlm":
        recipe["dataloader"].update(token_budget=1 if packing_mode == "dynamic" else 4096,
                                    min_buffered_samples=1 if packing_mode == "dynamic" else 16)
    recipe["training"].update(train_iters=args.steps, global_batch_size=world,
                               empty_cache_steps=args.empty_cache_steps, seed=args.random_init_seed)
    recipe["activation_checkpoint"]["mode"] = args.activation_checkpoint
    if args.activation_checkpoint == "full":
        recipe["activation_checkpoint"]["selection"] = {
            "source": "model_adapter_safe_regions", "layer_count": 0,
        }
    ep_size = world if args.ep_size is None else args.ep_size
    recipe["accelerator"]["ep_size"] = ep_size
    edp_shard_size = args.edp_shard_size or world // ep_size
    recipe["fsdp_config"].update(dp_shard_size=world, edp_shard_size=edp_shard_size)
    recipe["dataset"]["data_path"] = args.data
    if modality == "vlm":
        recipe["dataset"]["model_assets"]["config_path"] = args.model_dir
    else:
        recipe["dataset"]["model_assets"]["tokenizer"]["pretrained_model_name_or_path"] = args.model_dir
    if args.validation_crop:
        recipe["model"].update(
            text_parameter_divisor=1, vision_parameter_divisor=1,
            num_hidden_layers=args.num_hidden_layers or 4,
            vision_num_hidden_layers=(args.vision_num_hidden_layers or 1) if modality == "vlm" else None,
            exercise_post_training_indexer=False,
        )
    if args.full_model:
        source = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
        text, vision = source["text_config"], source["vision_config"]
        assets = json.loads(Path(args.engram_assets).read_text(encoding="utf-8"))
        expected = {
            "num_hidden_layers": text["num_hidden_layers"],
            "bucket_base": text["engram_vocab_size"],
            "layer_ids": text["engram_layer_ids"],
            "num_embeddings": text["engram_num_embeddings"],
            "compressed_vocab_size": text["engram_compressed_vocab_size"],
        }
        for key, value in expected.items():
            if assets.get(key) != value:
                raise ValueError(f"Full-model Engram mismatch for {key}: {assets.get(key)} != {value}")
        if args.experts != text["n_routed_experts"]:
            raise ValueError("Full-model expert count must match the released config")
        depth = args.num_hidden_layers or text["num_hidden_layers"]
        vision_depth = (args.vision_num_hidden_layers or vision["num_hidden_layers"]) if modality == "vlm" else None
        if not 4 <= depth <= text["num_hidden_layers"]:
            raise ValueError("Language depth must be between 4 and the released depth")
        if vision_depth is not None and not 1 <= vision_depth <= vision["num_hidden_layers"]:
            raise ValueError("Vision depth must be between 1 and the released depth")
        retained = [index for index, layer in enumerate(assets["layer_ids"]) if layer < depth]
        for key in ("layer_ids", "primes", "num_embeddings", "padded_num_embeddings", "multipliers"):
            assets[key] = [assets[key][index] for index in retained]
        assets["num_hidden_layers"] = text["num_hidden_layers"]
        # Pad only the physical storage; hash addresses keep the released sizes.
        assets["table_pad_multiple"] = ep_size
        assets["padded_num_embeddings"] = [
            (size + ep_size - 1) // ep_size * ep_size for size in assets["num_embeddings"]
        ]
        asset_path = work / "engram.json"
        asset_path.write_text(json.dumps(assets), encoding="utf-8")
        recipe["model"].update(
            text_parameter_divisor=1, vision_parameter_divisor=1,
            num_hidden_layers=depth,
            vision_num_hidden_layers=vision_depth,
            engram_assets_path=str(asset_path), exercise_post_training_indexer=False,
        )
        for override in recipe["plan_overrides"]:
            if isinstance(override["match"], str):
                override["match"] = override["match"].replace("model.vision.blocks.0.", "model.vision.blocks.*.")
    if args.activation_checkpoint == "full":
        depth = recipe["model"].get("num_hidden_layers")
        if depth is None:
            source = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
            depth = source["text_config"]["num_hidden_layers"]
        recipe["activation_checkpoint"]["selection"]["layer_count"] = depth
    recipe["model"]["engram_storage_backend"] = args.engram_storage_backend
    if args.engram_storage_backend == "host":
        adamw = recipe["optimizer"]["adamw_config"]
        recipe["model"]["engram_sparse_adam_config"] = {
            "lr": adamw["adamw_lr"], "betas": adamw["adamw_betas"], "eps": adamw["adamw_eps"],
        }
        for override in recipe["plan_overrides"]:
            if override.get("module_type", "").endswith(".DeepseekV41Engram"):
                override["replace_module"]["_target_"] = (
                    "hyper_parallel.models.deepseek_v41.adapter.engram.host_replacement.HostEngramModule"
                )
    compute = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel."
    moe_override = next(entry for entry in recipe["plan_overrides"]
                        if entry.get("local_compute_fn", {}).get("_target_", "").endswith("deepseek_v41_ep_compute_fn"))
    if args.backend == "megamoe":
        replacement = {
            "_target_": "hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts",
            "local_num_tokens": 4096, "dispatch_mode": args.dispatch_mode,
        }
        if args.dispatch_mode == "push" and args.capacity_factor is not None:
            replacement["initial_capacity_factor"] = args.capacity_factor
        recipe["plan_overrides"].insert(0, {
            "match": "model.layers.*.mlp.experts",
            "module_type": "transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts",
            "replace_module": replacement,
        })
        moe_override.pop("when", None)
        moe_override["local_compute_fn"] = {"_target_": compute + "deepseek_v41_megamoe_compute_fn"}
    recipe["checkpoint"].update(save_ckpt=False, restore_from=None, checkpoint_dir=str(work / "unused-checkpoint"))
    path = work / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return parse_training_args([str(path)])


class _RouteDiagnostics:
    """Record actual source token rows without retaining activations."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Attach observations to each routed-plus-shared MoE boundary."""
        self.token_rows: dict[str, int] = {}
        self.hooks = [module.register_forward_pre_hook(self._record_rows(name))
                      for name, module in model.named_modules()
                      if name.endswith(".mlp") and hasattr(module, "gate")]

    def _record_rows(self, name: str) -> Any:
        def hook(_module: torch.nn.Module, inputs: tuple) -> None:
            """Record rows without changing the source tensor."""
            self.token_rows[name] = inputs[0].numel() // inputs[0].shape[-1]
        return hook

    def clear(self) -> None:
        """Clear observations before the next training step."""
        self.token_rows.clear()

    def close(self) -> None:
        """Remove all observations after training."""
        for hook in self.hooks:
            hook.remove()


def _step(trainer: Any, iterator: Any, step: int, work: Path,
          diagnostics: _RouteDiagnostics, warmup_steps: int, moe_timing: MoeTiming | None = None) -> dict[str, Any]:
    dist.barrier()
    torch.npu.synchronize()
    diagnostics.clear()
    if moe_timing is not None:
        moe_timing.clear()
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    metrics = trainer.train_step(iterator)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc).isoformat()
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite metrics at step {step}: {metrics}")
    times = torch.tensor([elapsed], device="npu", dtype=torch.float32)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    row = {"step": step + 1, "local_seconds": elapsed, "max_rank_seconds": times.item(),
           "allocated_bytes": torch.npu.memory_allocated(), "reserved_bytes": torch.npu.memory_reserved(),
           "phase": "warmup" if step < warmup_steps else "measured",
           "started_at": started_at, "finished_at": finished_at,
           "moe_input_rows": dict(diagnostics.token_rows),
           "replica_routes": {},
           "moe_timing": moe_timing.snapshot() if moe_timing is not None else None,
           "peak_allocated_bytes": torch.npu.max_memory_allocated(),
           "peak_reserved_bytes": torch.npu.max_memory_reserved(),
           "trainer_env": dict(trainer.base.step_env_metrics),
           "trainer_metrics": dict(trainer.base.step_train_metrics), **metrics}
    with (work / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    return row


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engram-assets", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--backend", choices=("owner_ep", "megamoe"), required=True)
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--modality", choices=("text", "vlm"), default="vlm")
    parser.add_argument("--empty-cache-steps", type=int, default=1)
    parser.add_argument("--attention-recompute-tier", choices=("none", "reuse", "reuse_reindex", "all"),
                        default="all")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--full-model", action="store_true", help="Use released dimensions, with optional depth crops")
    scale.add_argument("--validation-crop", action="store_true",
                       help="Use full widths and 4+1 layers with supplied validation Engram assets and expert count")
    parser.add_argument("--num-hidden-layers", type=int,
                        help="Language depth override for full-model or validation-crop")
    parser.add_argument("--vision-num-hidden-layers", type=int,
                        help="Vision depth override for full-model or validation-crop")
    parser.add_argument("--edp-shard-size", type=int, help="Expert FSDP shard degree; remaining EDP ranks replicate")
    parser.add_argument("--ep-size", type=int, help="EP degree; dense FSDP spans the full world")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=0, help="Exclude initial steps from measured statistics")
    parser.add_argument("--packing-mode", choices=("dynamic", "packed"), default="packed",
                        help="Dynamic emits one complete source sample; packed selects up to 4096 tokens")
    parser.add_argument("--moe-timing", action="store_true", help="Record MoE current-stream event spans")
    parser.add_argument("--activation-checkpoint", choices=("off", "full"), default="off")
    parser.add_argument("--engram-storage-backend", choices=("device", "host"), default="device")
    parser.add_argument("--random-init-seed", type=int, default=42,
                        help="Generate identical canonical shards in memory")
    parser.add_argument("--native-lib", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.empty_cache_steps < 0:
        parser.error("empty-cache-steps must be nonnegative")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("warmup-steps must be nonnegative and smaller than steps")
    world = int(os.environ.get("WORLD_SIZE", "16"))
    ep_size = world if args.ep_size is None else args.ep_size
    if world < 1 or ep_size < 2 or world % ep_size or args.steps < 1 or args.experts < 6 or args.experts % ep_size:
        parser.error("requires positive steps, EP >= 2 dividing WORLD_SIZE, and experts >= 6 divisible by EP")
    for depth in (args.num_hidden_layers, args.vision_num_hidden_layers):
        if depth is not None and (not (args.full_model or args.validation_crop) or depth <= 0):
            parser.error("depth overrides require --full-model or --validation-crop and positive values")
    if args.edp_shard_size is not None:
        if args.edp_shard_size <= 0 or (world // ep_size) % args.edp_shard_size:
            parser.error("edp-shard-size must be positive and divide WORLD_SIZE/EP")
    if args.dispatch_mode == "pull" and args.capacity_factor is not None:
        parser.error("pull comparison must not impose a push receive-capacity bound")
    return args


def main() -> None:
    """Run fixed-horizon training, preserving epoch order and both timing scopes."""
    args = _arguments()
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "16"))
    work = Path(args.output) / f"rank{rank}"
    work.mkdir(parents=True, exist_ok=True)
    config = _recipe(args, work, world)
    if args.dry_run:
        return
    if args.native_lib:
        _loader._component_root = args.native_lib.resolve
    activation_checkpointing.ATTENTION_RECOMPUTE_TIER = args.attention_recompute_tier
    trainer = (VLMTrainer if args.modality == "vlm" else TextTrainer)(config)
    experts = [module for module in trainer.base.model.modules() if isinstance(module, DeepseekV41MegaMoeExperts)]
    initialization = initialize_random_shards(trainer, seed=args.random_init_seed, backend=args.backend)
    (work / "initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    dist.barrier()
    trainer.on_train_begin()
    torch.npu.reset_peak_memory_stats()
    diagnostics = _RouteDiagnostics(trainer.base.model)
    moe_timing = MoeTiming(trainer.base.model) if args.moe_timing else None
    rows = []
    try:
        for epoch in range(trainer.base.train_epochs):
            loader = trainer.base.train_dataloader
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            trainer.on_epoch_begin()
            iterator = iter(loader)
            count = min(trainer.base.train_steps, args.steps - len(rows))
            for _ in range(count):
                if len(rows) == args.warmup_steps:
                    torch.npu.reset_peak_memory_stats()
                rows.append(_step(trainer, iterator, len(rows), work, diagnostics, args.warmup_steps, moe_timing))
            trainer.on_epoch_end()
            trainer.base.state.epoch = epoch + 1
            if len(rows) == args.steps:
                break
    finally:
        diagnostics.close()
        if moe_timing is not None:
            moe_timing.close()
    if len(rows) != args.steps:
        raise RuntimeError(f"Incomplete training: {len(rows)} of {args.steps} steps")
    measured = rows[args.warmup_steps:]
    result = {"vision_enabled": args.modality == "vlm", **vars(args), "native_lib": str(args.native_lib),
              "rank": rank, "world_size": world,
              "steps_detail": rows, "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
              "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows), "finite": True,
              "measurement": {"first_step": args.warmup_steps + 1, "steps": len(measured),
                              "mean_max_rank_seconds": statistics.mean(row["max_rank_seconds"] for row in measured),
                              "median_max_rank_seconds": statistics.median(row["max_rank_seconds"] for row in measured),
                              "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                              "peak_reserved_bytes": torch.npu.max_memory_reserved()},
              "weights_initialization": "canonical_random_not_pretrained", "workspace_shared": bool(experts),
              "shmem_state": shmem.debug_state() if experts else None,
              "owner_use_grouped_gemm": args.backend == "owner_ep"}
    trainer.on_train_end()
    (work / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
