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

"""Backend-free model configuration and CLI for the Qwen MoE optimizer benchmark."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Literal

RoutingMode = Literal["balanced", "topk", "dirichlet"]
CompactLayoutStrategy = Literal[
    "lazy_on_first_replica",
    "eager_on_first_forward",
]


@dataclass(frozen=True)
class QwenMoeConfig:
    """Fixed-shape synthetic Qwen model configuration.

    The example intentionally models one TP1/EP8 optimizer-step workload. Its
    deterministic balanced default gives every destination the same load;
    optional Dirichlet routing reproduces load-skew and multi-wave behavior.
    """

    vocab_size: int = 32000
    hidden_size: int = 5120
    num_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_seq_len: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    num_experts: int = 48
    top_k: int = 8
    intermediate_size: int = 1792
    shared_expert_intermediate_size: int = 512
    routed_scaling_factor: float = 1.0
    routing: RoutingMode = "balanced"
    router_load_alpha: float = 1.0
    router_load_seed: int = 42
    local_num_tokens: int = 4096
    expert_capacity_factor: float | None = None
    replica_slots_per_rank: int = 0
    compact_physical_weights: bool = False
    shared_physical_weights: bool = False
    compact_layout_strategy: CompactLayoutStrategy = "eager_on_first_forward"
    ep_size: int = 8

    def __post_init__(self) -> None:
        """Validate the fixed optimizer-step topology."""
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "intermediate_size": self.intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "local_num_tokens": self.local_num_tokens,
            "ep_size": self.ep_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({self.ep_size})."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})."
            )
        if (
            not isinstance(self.replica_slots_per_rank, int)
            or isinstance(self.replica_slots_per_rank, bool)
            or self.replica_slots_per_rank < 0
        ):
            raise ValueError(
                "replica_slots_per_rank must be a nonnegative integer, "
                f"got {self.replica_slots_per_rank!r}."
            )
        if not isinstance(self.compact_physical_weights, bool):
            raise ValueError(
                "compact_physical_weights must be a bool, got "
                f"{self.compact_physical_weights!r}."
            )
        if not isinstance(self.shared_physical_weights, bool):
            raise ValueError("shared_physical_weights must be a bool.")
        if self.shared_physical_weights and (self.compact_physical_weights or not self.replica_slots_per_rank):
            raise ValueError("shared_physical_weights requires replica slots and excludes compact_physical_weights.")
        if self.compact_layout_strategy not in (
            "lazy_on_first_replica",
            "eager_on_first_forward",
        ):
            raise ValueError(
                "compact_layout_strategy must be 'lazy_on_first_replica' or "
                f"'eager_on_first_forward', got {self.compact_layout_strategy!r}."
            )
        if self.local_num_tokens % 128:
            raise ValueError(
                f"local_num_tokens ({self.local_num_tokens}) must be divisible by 128."
            )
        if self.max_seq_len > self.local_num_tokens:
            raise ValueError(
                f"max_seq_len ({self.max_seq_len}) cannot exceed local_num_tokens "
                f"({self.local_num_tokens}) in the fixed batch-one workload."
            )
        self._validate_routing()

    def _validate_routing(self) -> None:
        """Validate learned and synthetic router parameters."""
        if self.routing not in ("balanced", "topk", "dirichlet"):
            raise ValueError(f"unsupported routing mode: {self.routing!r}.")
        if self.routing == "balanced" and self.top_k != self.ep_size:
            raise ValueError(
                "balanced routing requires top_k to equal ep_size so every token "
                f"selects one expert per destination, got top_k={self.top_k} and "
                f"ep_size={self.ep_size}."
            )
        if not math.isfinite(self.router_load_alpha) or self.router_load_alpha < 0.01:
            raise ValueError(
                "router_load_alpha must be finite and at least 0.01, got "
                f"{self.router_load_alpha}."
            )
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError(
                "routed_scaling_factor must be finite and positive, got "
                f"{self.routed_scaling_factor}."
            )


def _capacity_factor(value: str) -> float | None:
    """Parse ``none`` or a finite factor of at least one."""
    if value.lower() == "none":
        return None
    try:
        factor = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be 'none' or a number, got {value!r}."
        ) from error
    if not math.isfinite(factor) or factor < 1.0:
        raise argparse.ArgumentTypeError(
            f"capacity factor must be finite and at least 1.0, got {value!r}."
        )
    return factor


def parse_args(
    argv: list[str] | None = None, *, parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse the minimal optimizer benchmark interface."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Run an eight-NPU Qwen MoE optimizer-step benchmark.")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--expert-capacity-factor", type=_capacity_factor, default=None)
    parser.add_argument("--routing", choices=("balanced", "topk", "dirichlet"), default="balanced")
    parser.add_argument("--router-load-alpha", type=float, default=1.0)
    parser.add_argument("--router-load-seed", type=int, default=42)
    parser.add_argument("--replica-slots-per-rank", type=int, default=0)
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--compact-physical-weights", action="store_true")
    storage.add_argument("--shared-physical-weights", action="store_true")
    defaults = QwenMoeConfig()
    for name in ("hidden_size", "intermediate_size", "num_layers", "vocab_size"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--sequence-length", type=int, default=defaults.local_num_tokens)
    parser.add_argument(
        "--compact-layout-strategy",
        choices=("lazy_on_first_replica", "eager_on_first_forward"),
        default="eager_on_first_forward",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("both", "common", "mega_moe"), default="both")
    snapshots = parser.add_mutually_exclusive_group()
    snapshots.add_argument("--snapshot-dir", help="write CPU accuracy references for a single backend")
    snapshots.add_argument("--reference-dir", help="compare a single backend against completed CPU references")
    parser.add_argument(
        "--output",
        default="output/qwen_moe_benchmark.json",
        help="rank-zero JSON result path",
    )
    args = parser.parse_args(argv)
    has_snapshots = bool(args.snapshot_dir or args.reference_dir)
    if (args.backend != "both") != has_snapshots:
        raise ValueError("single-backend runs require --snapshot-dir or --reference-dir; paired runs exclude both.")
    if args.warmup_steps < 1:
        raise ValueError(f"warmup_steps must be positive, got {args.warmup_steps}.")
    if args.measured_steps <= 0:
        raise ValueError(f"measured_steps must be positive, got {args.measured_steps}.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError(
            f"learning_rate must be finite and positive, got {args.learning_rate}."
        )
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError(
            f"weight_decay must be finite and nonnegative, got {args.weight_decay}."
        )
    make_config(args)
    return args


def make_config(args: argparse.Namespace) -> QwenMoeConfig:
    """Build and validate the complete batch-one workload before initializing EP."""
    return replace(
        QwenMoeConfig(),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.sequence_length,
        local_num_tokens=args.sequence_length,
        replica_slots_per_rank=args.replica_slots_per_rank,
        compact_physical_weights=args.compact_physical_weights,
        shared_physical_weights=args.shared_physical_weights,
        compact_layout_strategy=args.compact_layout_strategy,
        expert_capacity_factor=args.expert_capacity_factor,
        routing=args.routing,
        router_load_alpha=args.router_load_alpha,
        router_load_seed=args.router_load_seed,
    )


def optimizer_contract(args: argparse.Namespace) -> dict:
    """Describe the fixed numerical optimizer contract shared by workers and parents."""
    return {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999], "eps": 1e-8, "model_dtype": "bfloat16", "main_dtype": "float32"}
