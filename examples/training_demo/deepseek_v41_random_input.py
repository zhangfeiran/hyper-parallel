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
"""Deterministic random-content VLM inputs with unchanged image geometry and supervision mask."""

import hashlib
import json
from typing import Optional

import torch

from hyper_parallel.models.deepseek_v41.adapter.vlm_data import DeepseekV41VLMTransform, TEXT


def randomize_sample(sample: dict, vocabulary: torch.Tensor, seed: int) -> dict:
    """Replace text and pixels using a private CPU generator, preserving structural fields.

    Args:
        sample: A transformed image-text sample with fixed sequence and image geometry.
        vocabulary: Valid non-special token IDs.
        seed: Stable per-record random seed.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = dict(sample)
    text = sample["token_types"].eq(TEXT)
    ids = sample["input_ids"].clone()
    ids[text] = vocabulary[torch.randint(vocabulary.numel(), (int(text.sum()),), generator=generator)]
    result["input_ids"] = ids
    labels = sample["labels"].clone()
    supervised = labels.ne(-100)
    labels[supervised] = ids[supervised]
    result["labels"] = labels
    result["attention_mask"] = torch.ones_like(sample["attention_mask"])
    pixels = sample["pixel_values"]
    result["pixel_values"] = (torch.rand(pixels.shape, generator=generator) * 2 - 1).to(pixels.dtype)
    return result


class RandomContentVLMTransform(DeepseekV41VLMTransform):
    """Keep the vision path and loss mask but eliminate repeated padding content."""

    def __init__(self, config_path: str, *, max_seq_len: int = 4096,
                 max_image_tokens: Optional[int] = None, random_input_seed: int = 20260921,
                 processor: object = None) -> None:
        """Resolve valid token IDs without initializing any model weights."""
        del processor
        super().__init__(config_path, max_seq_len=max_seq_len, max_image_tokens=max_image_tokens)
        config = json.loads((self.config_path / "config.json").read_text())
        size = min(self.tokenizer.vocab_size, config["text_config"]["vocab_size"])
        excluded = set(self.tokenizer.all_special_ids)
        excluded.update((self.pad_token_id, self.image_token_id))
        self.vocabulary = torch.tensor([index for index in range(size) if index not in excluded])
        self.random_input_seed = random_input_seed

    def __call__(self, record: dict) -> dict:
        """Use record identity so data-loader worker scheduling cannot change inputs."""
        identity = str(record["id"]).encode("utf-8")
        digest = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        seed = (digest + self.random_input_seed) % (2**63 - 1)
        return randomize_sample(super().__call__(record), self.vocabulary, seed)
