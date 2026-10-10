/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/library.h>
#include <array>
#include <tuple>
#include "csrc/cached_op_api.h"

namespace {
using Result = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>;
// Pinned arch22: 20 * (2*512*512*4 + 8*2*2*2048*4 + 8*2*16*8).
constexpr int64_t kRetainedBytes = 47226880;

void check_states(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, bool local = false) {
  TORCH_CHECK(query.dim() == 3 && (local || query.size(0) > 0) && query.size(1) == 64 && query.size(2) == 128 &&
                query.scalar_type() == at::kBFloat16,
              "mixed LI query must be BF16 [T,64,128]");
  const auto tokens = query.size(0);
  TORCH_CHECK(key.dim() == 3 && key.size(0) > 0 && key.size(1) == 1 && key.size(2) == 128 &&
                (local ? tokens <= key.size(0) : tokens == key.size(0)) && key.scalar_type() == at::kBFloat16,
              "mixed LI key must be BF16 [T,1,128]");
  TORCH_CHECK(weights.sizes() == at::IntArrayRef({tokens, 64}) &&
                (weights.scalar_type() == at::kBFloat16 || weights.scalar_type() == at::kFloat),
              "mixed LI weights must be BF16 or FP32 [T,64]");
}

void check_trace(const at::Tensor &config, const at::Tensor &trace, int64_t phase) {
  TORCH_CHECK(config.sizes() == at::IntArrayRef({4}) && config.scalar_type() == at::kLong &&
                ((phase == 2 && trace.sizes() == at::IntArrayRef({2, 20, 64})) ||
                 (phase != 2 && trace.sizes() == at::IntArrayRef({20, 64}))) &&
                trace.scalar_type() == at::kLong,
              "mixed LI expects int64 config[4] and trace[20,64], or fused trace[2,20,64]");
}

void check_runtime(const at::Tensor &query, const at::Tensor &actualQuery, const at::Tensor &actualKv,
                   const at::Tensor &config, const at::Tensor &trace, const at::Tensor &retained,
                   const at::Tensor &indices, const at::Tensor &values, int64_t phase, bool local = false) {
  const auto tokens = query.size(0);
  TORCH_CHECK(actualQuery.dim() == 1 && actualQuery.numel() > 0 && actualKv.sizes() == actualQuery.sizes() &&
                actualQuery.scalar_type() == at::kInt && actualKv.scalar_type() == at::kInt,
              "mixed LI cumulative lengths must be nonempty int32 vectors of equal size");
  check_trace(config, trace, phase);
  TORCH_CHECK(retained.dim() == 1 && retained.numel() >= (local && tokens == 0 ? 0 : kRetainedBytes) &&
                retained.scalar_type() == at::kByte,
              "mixed LI retained uint8 workspace requires at least 47226880 bytes");
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({tokens, 1, 2048}) && indices.scalar_type() == at::kInt &&
                values.sizes() == indices.sizes() && values.scalar_type() == at::kBFloat16,
              "mixed LI outputs must be int32 indices and BF16 values [T,1,2048]");
}

Result dsa_mixed_indexer_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                             const at::Tensor &actualQuery, const at::Tensor &actualKv, const at::Tensor &config,
                             at::Tensor &trace, at::Tensor &retained, int64_t mergePhase, at::Tensor &indices,
                             at::Tensor &values) {
  const std::array<const at::Tensor *, 10> tensors = {&query,  &key,   &weights,  &actualQuery, &actualKv,
                                                      &config, &trace, &retained, &indices,     &values};
  for (const auto *tensor : tensors) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                "mixed LI requires contiguous tensors on one NPU");
    TORCH_CHECK(!tensor->requires_grad(), "mixed LI probe has no backward; detach inputs explicitly");
  }
  check_states(query, key, weights);
  check_runtime(query, actualQuery, actualKv, config, trace, retained, indices, values, mergePhase);
  TORCH_CHECK(mergePhase >= 0 && mergePhase <= 2, "mixed LI phase must be main=0, merge=1 or fused=2");
  for (const auto *output : {&trace, &retained, &indices, &values}) {
    for (const auto *tensor : tensors) {
      TORCH_CHECK(output == tensor || !output->is_alias_of(*tensor),
                  "mixed LI mutable buffers must own independent storage");
    }
  }
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaMixedIndexer",
                                                          "aclnnHyperDsaMixedIndexerGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, weights, actualQuery, actualKv, config, trace, retained,
                                               mergePhase, indices, values);
  return Result(indices, values, trace, retained);
}
Result dsa_local_indexer_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                             const at::Tensor &actualQuery, const at::Tensor &actualKv, const at::Tensor &config,
                             at::Tensor &trace, at::Tensor &retained, const at::Tensor &queryPositions,
                             at::Tensor &indices, at::Tensor &values) {
  const std::array<const at::Tensor *, 11> tensors{&query, &key,      &weights,        &actualQuery, &actualKv, &config,
                                                   &trace, &retained, &queryPositions, &indices,     &values};
  for (const auto *tensor : tensors) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "local LI requires detached contiguous tensors on one NPU");
  }
  TORCH_CHECK(queryPositions.sizes() == at::IntArrayRef({query.size(0)}) && queryPositions.scalar_type() == at::kLong,
              "local LI positions must be int64 [local Q]");
  check_states(query, key, weights, true);
  check_runtime(query, actualQuery, actualKv, config, trace, retained, indices, values, 2, true);
  for (const auto *output : {&trace, &retained, &indices, &values}) {
    for (const auto *tensor : tensors) {
      TORCH_CHECK(output == tensor || !output->is_alias_of(*tensor),
                  "local LI mutable buffers must own independent storage");
    }
  }
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaLocalIndexer",
                                                          "aclnnHyperDsaLocalIndexerGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, weights, actualQuery, actualKv, config, trace, retained,
                                               queryPositions, indices, values);
  return Result(indices, values, trace, retained);
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_mixed_indexer_out", &dsa_mixed_indexer_npu);
  m.impl("dsa_local_indexer_out", &dsa_local_indexer_npu);
}
