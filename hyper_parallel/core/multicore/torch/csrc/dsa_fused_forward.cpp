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
using Tensors = std::array<const at::Tensor *, 16>;
using TensorRef = at::Tensor &;
using Result = std::tuple<TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef>;

void check_storage(const Tensors &tensors) {
  for (const auto *tensor : tensors) {
    TORCH_CHECK(tensor->device() == tensors[0]->device() && tensor->is_contiguous(),
                "fused DSA requires contiguous tensors on one NPU");
    TORCH_CHECK(!tensor->requires_grad(), "fused DSA forward probe requires explicitly detached inputs");
  }
  for (size_t mutableIndex = 9; mutableIndex < tensors.size(); ++mutableIndex) {
    for (size_t index = 0; index < tensors.size(); ++index) {
      TORCH_CHECK(index == mutableIndex || !tensors[mutableIndex]->is_alias_of(*tensors[index]),
                  "fused DSA mutable buffers must own independent storage");
    }
  }
}

void check_states(const Tensors &tensors) {
  const auto &query = *tensors[2];
  const auto tokens = query.size(0);
  TORCH_CHECK(query.dim() == 3 && tokens > 0 && (query.size(1) == 32 || query.size(1) == 64) && query.size(2) == 512,
              "fused DSA query must be [T,32|64,512]");
  TORCH_CHECK(tensors[0]->sizes() == at::IntArrayRef({tokens, 64, 128}) &&
              tensors[1]->sizes() == at::IntArrayRef({tokens, 1, 128}), "fused DSA LI shapes are incompatible");
  TORCH_CHECK(tensors[3]->sizes() == at::IntArrayRef({tokens, 1, 512}) &&
              tensors[4]->sizes() == at::IntArrayRef({tokens, query.size(1), 64}) &&
              tensors[5]->sizes() == at::IntArrayRef({tokens, 1, 64}), "fused DSA KV/RoPE shapes are incompatible");
  for (size_t index = 0; index < 6; ++index) {
    TORCH_CHECK(tensors[index]->scalar_type() == at::kBFloat16, "fused DSA states must be BF16");
  }
  TORCH_CHECK(tensors[6]->sizes() == at::IntArrayRef({tokens, 64}) &&
              (tensors[6]->scalar_type() == at::kBFloat16 || tensors[6]->scalar_type() == at::kFloat),
              "fused DSA LI weights must be BF16 or FP32 [T,64]");
}

void check_runtime(const Tensors &tensors) {
  TORCH_CHECK(tensors[7]->dim() == 1 && tensors[7]->numel() > 0 && tensors[7]->scalar_type() == at::kInt,
              "fused DSA packed cumulative lengths must be a nonempty int32 vector");
  TORCH_CHECK(tensors[8]->sizes() == at::IntArrayRef({4}) && tensors[8]->scalar_type() == at::kLong &&
              tensors[9]->sizes() == at::IntArrayRef({3, 20, 64}) && tensors[9]->scalar_type() == at::kLong,
              "fused DSA runtime ABI expects int64 config[4] and trace[3,20,64]");
  TORCH_CHECK(tensors[10]->dim() == 1 && tensors[10]->numel() >= 47226880 &&
              tensors[10]->scalar_type() == at::kByte, "fused DSA retained scratch requires 47226880 uint8 bytes");
}

void check_outputs(const Tensors &tensors) {
  const auto &query = *tensors[2];
  const auto tokens = query.size(0);
  TORCH_CHECK(tensors[11]->sizes() == at::IntArrayRef({tokens, 1, 2048}) &&
              tensors[11]->scalar_type() == at::kInt && tensors[12]->sizes() == tensors[11]->sizes() &&
              tensors[12]->scalar_type() == at::kBFloat16, "fused DSA LI output shapes or dtypes are incompatible");
  TORCH_CHECK(tensors[13]->sizes() == query.sizes() && tensors[13]->scalar_type() == at::kBFloat16 &&
              tensors[14]->sizes() == at::IntArrayRef({1, tokens, query.size(1)}) &&
              tensors[15]->sizes() == tensors[14]->sizes() && tensors[14]->scalar_type() == at::kFloat &&
              tensors[15]->scalar_type() == at::kFloat, "fused DSA SFA output shapes or dtypes are incompatible");
}

Result fused_forward_npu(
    const at::Tensor &indexQuery, const at::Tensor &indexKey, const at::Tensor &query,
    const at::Tensor &compressed, const at::Tensor &queryRope, const at::Tensor &keyRope,
    const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
    at::Tensor &trace, at::Tensor &retained, double scale,
    at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum) {
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed, &queryRope, &keyRope, &weights,
                        &lengths, &config, &trace, &retained, &indices, &values, &attention, &maximum, &sum};
  check_storage(tensors);
  check_states(tensors);
  check_runtime(tensors);
  check_outputs(tensors);
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "fused DSA scale must be finite and positive");
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnHyperDsaFusedForward", "aclnnHyperDsaFusedForwardGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, indexQuery, indexKey, query, compressed, queryRope, keyRope,
      weights, lengths, config, trace, retained, scale, indices, values, attention, maximum, sum);
  return Result(indices, values, attention, maximum, sum, trace, retained);
}
}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_fused_forward_out", &fused_forward_npu);
}
