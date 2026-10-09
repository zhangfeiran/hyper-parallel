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
#include <ATen/Context.h>
#include <array>
#include <cmath>
#include <tuple>
#include "csrc/cached_op_api.h"

namespace {
using Result = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>;

Result dsa_mixed_kl_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &indexQuery,
                        const at::Tensor &indexKey, const at::Tensor &weight, const at::Tensor &indices,
                        const at::Tensor &maximum, const at::Tensor &sum, const at::Tensor &queryRope,
                        const at::Tensor &keyRope, at::IntArrayRef actualQuery, at::IntArrayRef actualKey,
                        const at::Tensor &config, at::Tensor &trace, at::Tensor &retained, double scale, int64_t phase,
                        at::Tensor &gradQuery, at::Tensor &gradKey, at::Tensor &gradWeight, at::Tensor &loss) {
  const std::array<const at::Tensor *, 17> tensors = {
    &query,   &key,    &indexQuery, &indexKey, &weight,    &indices, &maximum,    &sum, &queryRope,
    &keyRope, &config, &trace,      &retained, &gradQuery, &gradKey, &gradWeight, &loss};
  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "mixed KL requires an NPU");
  for (const auto *tensor : tensors) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "mixed KL requires detached contiguous tensors on one NPU");
  }
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(), "mixed KL supports non-deterministic math only");
  TORCH_CHECK(std::isfinite(scale) && scale > 0 && phase >= 0 && phase <= 2, "mixed KL invalid scale/phase");
  TORCH_CHECK(
    query.dim() == 3 && query.size(0) > 0 && query.size(2) == 512 && (query.size(1) == 32 || query.size(1) == 64),
    "mixed KL query must be [T,32|64,512]");
  const auto tokens = query.size(0), heads = query.size(1);
  TORCH_CHECK(key.sizes() == at::IntArrayRef({tokens, 1, 512}) &&
                queryRope.sizes() == at::IntArrayRef({tokens, heads, 64}) &&
                keyRope.sizes() == at::IntArrayRef({tokens, 1, 64}),
              "mixed KL main shapes are incompatible");
  TORCH_CHECK(indexQuery.sizes() == at::IntArrayRef({tokens, 64, 128}) &&
                indexKey.sizes() == at::IntArrayRef({tokens, 1, 128}) &&
                weight.sizes() == at::IntArrayRef({tokens, 64}),
              "mixed KL index shapes are incompatible");
  for (const auto *tensor : {&query, &key, &queryRope, &keyRope, &indexQuery, &indexKey}) {
    TORCH_CHECK(tensor->scalar_type() == at::kBFloat16, "mixed KL states must be BF16");
  }
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kFloat,
              "mixed KL merge weights must be BF16 or FP32");
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({tokens, 1, 2048}) && indices.scalar_type() == at::kInt,
              "mixed KL indices must be int32 [T,1,2048]");
  TORCH_CHECK(!actualQuery.empty() && actualQuery == actualKey && actualQuery.back() == tokens,
              "mixed KL cumulative lengths must describe matching complete query/key sequences");
  int64_t previous = 0;
  for (const auto length : actualQuery) {
    TORCH_CHECK(length > previous, "mixed KL cumulative lengths must be strictly increasing");
    previous = length;
  }
  TORCH_CHECK(maximum.sizes() == at::IntArrayRef({1, tokens, heads}) && sum.sizes() == maximum.sizes() &&
                maximum.scalar_type() == at::kFloat && sum.scalar_type() == at::kFloat,
              "mixed KL statistics must be FP32 [1,T,H]");
  TORCH_CHECK(config.sizes() == at::IntArrayRef({4}) && config.scalar_type() == at::kLong &&
                trace.sizes() == at::IntArrayRef({20, 64}) && trace.scalar_type() == at::kLong && retained.dim() == 1 &&
                retained.numel() > 0 && retained.scalar_type() == at::kByte,
              "mixed KL runtime buffers have invalid shapes/dtypes");
  TORCH_CHECK(gradQuery.sizes() == indexQuery.sizes() && gradKey.sizes() == indexKey.sizes() &&
                gradWeight.sizes() == weight.sizes() && gradQuery.scalar_type() == at::kBFloat16 &&
                gradKey.scalar_type() == at::kBFloat16 && gradWeight.scalar_type() == weight.scalar_type() &&
                loss.sizes() == at::IntArrayRef({1}) && loss.scalar_type() == at::kFloat,
              "mixed KL output shapes/dtypes are incompatible");
  for (const auto *output : {&trace, &retained, &gradQuery, &gradKey, &gradWeight, &loss}) {
    for (const auto *tensor : tensors) {
      TORCH_CHECK(output == tensor || !output->is_alias_of(*tensor), "mixed KL mutable buffers must not alias");
    }
  }
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaMixedKl",
                                                          "aclnnHyperDsaMixedKlGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, indexQuery, indexKey, weight, indices, maximum, sum,
                                               queryRope, keyRope, actualQuery, actualKey, config, trace, retained,
                                               scale, phase, gradQuery, gradKey, gradWeight, loss);
  return Result(gradQuery, gradKey, gradWeight, loss, trace, retained);
}
}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) { m.impl("dsa_mixed_kl_out", &dsa_mixed_kl_npu); }
