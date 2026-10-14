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

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import yaml

from examples.training_demo.deepseek_v41.deepseek_v41_random_init import initialize_random_shards
from hyper_parallel.core.multicore import _loader
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import DeepseekV41MegaMoeExperts
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


def _recipe(args: argparse.Namespace, work: Path, world: int) -> Any:
    recipe = yaml.safe_load(Path(__file__).with_name("train_deepseek_v41_vlm_online.yaml").read_text())
    recipe["model"].update(config_path=args.model_dir, engram_assets_path=args.engram_assets,
                          num_routed_experts=args.experts)
    recipe["model_init_dtype"] = "float32"
    recipe["training"].update(train_iters=args.steps, global_batch_size=world)
    recipe["activation_checkpoint"]["mode"] = args.activation_checkpoint
    ep_size = world if args.ep_size is None else args.ep_size
    recipe["accelerator"]["ep_size"] = ep_size
    edp_shard_size = args.edp_shard_size or world // ep_size
    recipe["fsdp_config"].update(dp_shard_size=world, edp_shard_size=edp_shard_size)
    recipe["dataset"]["data_path"] = args.data
    recipe["dataset"]["model_assets"]["config_path"] = args.model_dir
    if args.full_model:
        source = json.loads((Path(args.model_dir) / "config.json").read_text())
        text, vision = source["text_config"], source["vision_config"]
        assets = json.loads(Path(args.engram_assets).read_text())
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
        vision_depth = args.vision_num_hidden_layers or vision["num_hidden_layers"]
        if not 4 <= depth <= text["num_hidden_layers"]:
            raise ValueError("Language depth must be between 4 and the released depth")
        if not 1 <= vision_depth <= vision["num_hidden_layers"]:
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
    replica_options = {"replica_slots_per_rank": args.replica_slots_per_rank,
                       "replica_minimum_load_reduction": 0.0}
    compute = "hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel."
    moe_override = next(entry for entry in recipe["plan_overrides"]
                        if entry.get("local_compute_fn", {}).get("_target_", "").endswith("deepseek_v41_ep_compute_fn"))
    if args.backend == "megamoe":
        replacement = {
            "_target_": "hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe.DeepseekV41MegaMoeExperts",
            "local_num_tokens": 4096, "dispatch_mode": args.dispatch_mode, **replica_options,
        }
        if args.dispatch_mode == "push" and args.capacity_factor is not None:
            replacement["initial_capacity_factor"] = args.capacity_factor
        recipe["plan_overrides"].insert(0, {
            "match": "*.mlp.experts",
            "module_type": "transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts",
            "replace_module": replacement,
        })
        moe_override.pop("when", None)
        moe_override["local_compute_fn"] = {"_target_": compute + "deepseek_v41_megamoe_compute_fn"}
    else:
        moe_override["local_compute_fn"].update(replica_options)
    recipe["checkpoint"].update(save_ckpt=False, restore_from=None, checkpoint_dir=str(work / "unused-checkpoint"))
    path = work / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return parse_training_args([str(path)])


def _step(trainer, iterator, step, work):
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    metrics = trainer.train_step(iterator)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite metrics at step {step}: {metrics}")
    times = torch.tensor([elapsed], device="npu", dtype=torch.float32)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    row = {"step": step + 1, "local_seconds": elapsed, "max_rank_seconds": times.item(),
           "allocated_bytes": torch.npu.memory_allocated(), "reserved_bytes": torch.npu.memory_reserved(),
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
    parser.add_argument("--replica-slots-per-rank", type=int, default=6)
    parser.add_argument("--full-model", action="store_true", help="Use released dimensions, with optional depth crops")
    parser.add_argument("--num-hidden-layers", type=int, help="Language depth override; requires --full-model")
    parser.add_argument("--vision-num-hidden-layers", type=int, help="Vision depth override; requires --full-model")
    parser.add_argument("--edp-shard-size", type=int, help="Expert FSDP shard degree; remaining EDP ranks replicate")
    parser.add_argument("--ep-size", type=int, help="EP degree; dense FSDP spans the full world")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--activation-checkpoint", choices=("off", "full"), default="off")
    parser.add_argument("--random-init-seed", type=int, default=42, help="Generate identical canonical shards in memory")
    parser.add_argument("--native-lib", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.replica_slots_per_rank < 0:
        parser.error("replica-slots-per-rank must be nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "16"))
    ep_size = world if args.ep_size is None else args.ep_size
    if world < 1 or ep_size < 2 or world % ep_size or args.steps < 1 or args.experts < 6 or args.experts % ep_size:
        parser.error("requires positive steps, EP >= 2 dividing WORLD_SIZE, and experts >= 6 divisible by EP")
    for depth in (args.num_hidden_layers, args.vision_num_hidden_layers):
        if depth is not None and (not args.full_model or depth <= 0):
            parser.error("depth overrides require --full-model and positive values")
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
        _loader._component_root = lambda: args.native_lib.resolve()
    trainer = VLMTrainer(config)
    experts = [module for module in trainer.base.model.modules() if isinstance(module, DeepseekV41MegaMoeExperts)]
    initialization = initialize_random_shards(trainer, seed=args.random_init_seed, backend=args.backend)
    (work / "initialization.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")
    dist.barrier()
    trainer.on_train_begin()
    torch.npu.reset_peak_memory_stats()
    rows = []
    for epoch in range(trainer.base.train_epochs):
        loader = trainer.base.train_dataloader
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        trainer.on_epoch_begin()
        iterator = iter(loader)
        count = min(trainer.base.train_steps, args.steps - len(rows))
        for _ in range(count):
            rows.append(_step(trainer, iterator, len(rows), work))
        trainer.on_epoch_end()
        trainer.base.state.epoch = epoch + 1
        if len(rows) == args.steps:
            break
    if len(rows) != args.steps:
        raise RuntimeError(f"Incomplete training: {len(rows)} of {args.steps} steps")
    result = {**vars(args), "native_lib": str(args.native_lib), "rank": rank, "world_size": world,
              "steps_detail": rows, "peak_allocated_bytes": torch.npu.max_memory_allocated(),
              "peak_reserved_bytes": torch.npu.max_memory_reserved(), "finite": True,
              "weights_initialization": "canonical_random_not_pretrained", "workspace_shared": bool(experts),
              "owner_use_grouped_gemm": args.backend == "owner_ep"}
    trainer.on_train_end()
    (work / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
