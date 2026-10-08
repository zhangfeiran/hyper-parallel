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
#include <cmath>
#include <tuple>
#include "csrc/cached_op_api.h"

namespace {
using Tensors = std::array<const at::Tensor *, 15>;
using Result = std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>;

void CheckStorage(const Tensors &tensors) {
  for (const auto *tensor : tensors) {
    TORCH_CHECK(tensor->device() == tensors[0]->device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "CP attention raw buffers must be detached contiguous tensors on the same NPU");
  }
  for (size_t mutableIndex : {1U, 3U, 7U, 8U, 11U, 12U, 13U, 14U}) {
    for (size_t index = 0; index < tensors.size(); ++index) {
      TORCH_CHECK(index == mutableIndex || !tensors[mutableIndex]->is_alias_of(*tensors[index]),
                  "CP attention mutable buffers must own independent storage");
    }
  }
}
void CheckMath(const Tensors &tensors) {
  const auto &query = *tensors[0];
  TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && (query.size(1) == 32 || query.size(1) == 64) &&
              query.size(2) == 512, "CP attention query must be [T,32|64,512]");
  const auto tokens = query.size(0);
  const auto heads = query.size(1);
  TORCH_CHECK(tensors[1]->sizes() == at::IntArrayRef({tokens, 1, 512}) &&
              tensors[2]->sizes() == at::IntArrayRef({tokens, heads, 64}) &&
              tensors[3]->sizes() == at::IntArrayRef({tokens, 1, 64}), "CP attention main shapes are incompatible");
  for (size_t index : {0U, 1U, 2U, 3U, 12U}) {
    TORCH_CHECK(tensors[index]->scalar_type() == at::kBFloat16, "CP attention main dtype must be BF16");
  }
  TORCH_CHECK(tensors[4]->sizes() == at::IntArrayRef({tokens, 1, 2048}) &&
              tensors[4]->scalar_type() == at::kInt, "CP attention indices must be int32 [T,1,2048]");
  TORCH_CHECK(tensors[5]->dim() == 1 && tensors[5]->numel() > 0 && tensors[5]->scalar_type() == at::kInt,
              "CP attention lengths must be nonempty int32");
  TORCH_CHECK(tensors[12]->sizes() == query.sizes() &&
              tensors[13]->sizes() == at::IntArrayRef({1, tokens, heads}) &&
              tensors[14]->sizes() == tensors[13]->sizes() && tensors[13]->scalar_type() == at::kFloat &&
              tensors[14]->scalar_type() == at::kFloat, "CP attention output shapes or dtypes are incompatible");
}
void CheckRuntime(const Tensors &tensors) {
  TORCH_CHECK(tensors[6]->sizes() == at::IntArrayRef({4}) && tensors[6]->scalar_type() == at::kLong &&
              tensors[7]->sizes() == at::IntArrayRef({20, 64}) && tensors[7]->scalar_type() == at::kLong,
              "CP attention runtime expects config[4] and trace[20,64] int64");
  TORCH_CHECK(tensors[8]->dim() == 1 && tensors[8]->scalar_type() == at::kByte &&
              tensors[9]->sizes() == at::IntArrayRef({18}) && tensors[9]->scalar_type() == at::kLong &&
              tensors[10]->dim() == 2 && tensors[10]->size(0) > 0 && tensors[10]->size(1) == 4 &&
              tensors[10]->scalar_type() == at::kLong && tensors[11]->sizes() == at::IntArrayRef({32}) &&
              tensors[11]->scalar_type() == at::kLong, "CP attention transport ABI mismatch");
}
Result CpAttentionNpu(
    const at::Tensor &query, at::Tensor &compressed, const at::Tensor &queryRope, at::Tensor &keyRope,
    const at::Tensor &indices, const at::Tensor &lengths, const at::Tensor &config, at::Tensor &trace,
    at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests, at::Tensor &transportTrace,
    double scale, at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum) {
  const Tensors tensors{&query, &compressed, &queryRope, &keyRope, &indices, &lengths, &config, &trace,
                        &arena, &metadata, &requests, &transportTrace, &attention, &maximum, &sum};
  CheckStorage(tensors);
  CheckMath(tensors);
  CheckRuntime(tensors);
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "CP attention scale must be finite and positive");
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnHyperDsaCpAttention", "aclnnHyperDsaCpAttentionGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, compressed, queryRope, keyRope, indices,
      lengths, config, trace, arena, metadata, requests, transportTrace, scale, attention, maximum, sum);
  return Result(attention, maximum, sum, trace, transportTrace);
}
}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_cp_attention_out", &CpAttentionNpu);
}
