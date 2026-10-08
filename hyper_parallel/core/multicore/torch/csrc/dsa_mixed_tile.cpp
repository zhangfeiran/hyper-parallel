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
#include <tuple>
#include <array>
#include <cmath>
#include "csrc/cached_op_api.h"

namespace {
using Result = std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>;
void check_states(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& queryRope, const at::Tensor& keyRope, const at::Tensor& out) {
  TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == 512 &&
              (query.size(1) == 32 || query.size(1) == 64), "DSA mixed query must be [T,32|64,512]");
  const auto tokens = query.size(0);
  const auto heads = query.size(1);
  TORCH_CHECK(key.sizes() == at::IntArrayRef({tokens, 1, 512}) && value.sizes() == key.sizes(),
              "DSA mixed key/value must be [T,1,512]");
  TORCH_CHECK(queryRope.sizes() == at::IntArrayRef({tokens, heads, 64}) &&
              keyRope.sizes() == at::IntArrayRef({tokens, 1, 64}), "DSA mixed rope shapes are incompatible");
  for (const auto* tensor : {&query, &key, &value, &queryRope, &keyRope, &out}) {
    TORCH_CHECK(tensor->scalar_type() == at::kBFloat16, "DSA mixed states must be BF16");
  }
}

void check_runtime(const at::Tensor& query, const at::Tensor& indices,
                   const at::Tensor& actualQuery, const at::Tensor& actualKv,
                   const at::Tensor& config, const at::Tensor& trace) {
  const auto tokens = query.size(0);
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({tokens, 1, 2048}) && indices.scalar_type() == at::kInt,
              "DSA mixed indices must be int32 [T,1,2048]");
  TORCH_CHECK(actualQuery.dim() == 1 && actualQuery.numel() > 0 && actualKv.sizes() == actualQuery.sizes() &&
              actualQuery.scalar_type() == at::kInt && actualKv.scalar_type() == at::kInt,
              "DSA mixed sequence lengths must be int32 vectors of equal size");
  TORCH_CHECK(config.sizes() == at::IntArrayRef({4}) && config.scalar_type() == at::kLong &&
              trace.sizes() == at::IntArrayRef({20, 64}) && trace.scalar_type() == at::kLong,
              "DSA mixed runtime ABI expects int64 config[4] and trace[20,64]");
}

void check_outputs(const at::Tensor& query, const at::Tensor& out,
                   const at::Tensor& maximum, const at::Tensor& sum) {
  const auto tokens = query.size(0);
  const auto heads = query.size(1);
  TORCH_CHECK(out.sizes() == query.sizes() && maximum.sizes() == at::IntArrayRef({1, tokens, heads}) &&
              sum.sizes() == maximum.sizes() && maximum.scalar_type() == at::kFloat &&
              sum.scalar_type() == at::kFloat, "DSA mixed output/statistic shapes are incompatible");
}

void check_inputs(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& indices, const at::Tensor& actualQuery, const at::Tensor& actualKv,
                  const at::Tensor& queryRope, const at::Tensor& keyRope, const at::Tensor& config,
                  const at::Tensor& trace, const at::Tensor& out, const at::Tensor& maximum,
                  const at::Tensor& sum) {
  const std::array<const at::Tensor*, 13> tensors = {&query, &key, &value, &indices, &actualQuery,
      &actualKv, &queryRope, &keyRope, &config, &trace, &out, &maximum, &sum};
  for (const auto* tensor : tensors) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                "DSA mixed tile requires contiguous tensors on one NPU");
    TORCH_CHECK(!tensor->requires_grad(), "DSA mixed tile probe has no backward; detach inputs explicitly");
  }
  check_states(query, key, value, queryRope, keyRope, out);
  check_runtime(query, indices, actualQuery, actualKv, config, trace);
  check_outputs(query, out, maximum, sum);
  for (const auto* output : {&out, &maximum, &sum, &trace}) {
    for (const auto* tensor : tensors) {
      TORCH_CHECK(output == tensor || !output->is_alias_of(*tensor),
                  "DSA mixed mutable buffers must own independent storage");
    }
  }
}

Result dsa_mixed_tile_npu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& indices, const at::Tensor& actualQuery, const at::Tensor& actualKv,
    const at::Tensor& queryRope, const at::Tensor& keyRope, const at::Tensor& config,
    at::Tensor& trace, double scale, at::Tensor& out, at::Tensor& maximum, at::Tensor& sum) {
  check_inputs(query, key, value, indices, actualQuery, actualKv, queryRope, keyRope, config,
               trace, out, maximum, sum);
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "DSA mixed attention scale must be finite and positive");
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnHyperDsaMixedTile", "aclnnHyperDsaMixedTileGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, value, indices, actualQuery, actualKv,
      queryRope, keyRope, config, trace, scale, out, maximum, sum);
  return Result(out, maximum, sum, trace);
}
}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_mixed_tile_out", &dsa_mixed_tile_npu);
}
