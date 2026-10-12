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
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/library.h>
#include <array>
#include <tuple>
#include "csrc/cached_op_api.h"
namespace {
using Result = std::tuple<at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &>;
Result fp32_index_kl_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weight,
                         const at::Tensor &teacher, const at::Tensor &globalLogits, const at::Tensor &indices,
                         int64_t phase, at::Tensor &localLogits, at::Tensor &gradQuery, at::Tensor &gradKey,
                         at::Tensor &gradWeight, at::Tensor &loss, at::Tensor &trace) {
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(), "FP32 index KL key atomics are nondeterministic");
  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1 && query.dim() == 3 && query.size(0) > 0 &&
                query.size(1) > 0 && query.size(1) <= 128 && query.size(2) == 128 &&
                key.dim() == 2 && key.size(0) > 0 && key.size(0) < (1LL << 31) && key.size(1) == 128 &&
                phase >= 0 && phase <= 1, "FP32 index KL requires NPU Q[T,H<=128,128], K[S,128] and phase 0/1");
  const auto tokens = query.size(0), heads = query.size(1);
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({tokens, heads}) &&
                teacher.sizes() == at::IntArrayRef({tokens, 2048}) && globalLogits.sizes() == teacher.sizes() &&
                indices.sizes() == teacher.sizes() && localLogits.sizes() == teacher.sizes() &&
                gradQuery.sizes() == query.sizes() && gradKey.sizes() == key.sizes() &&
                gradWeight.sizes() == weight.sizes() && loss.sizes() == at::IntArrayRef({tokens}) &&
                trace.sizes() == at::IntArrayRef({40, 4}), "FP32 index KL buffers have incompatible geometry");
  const std::array<const at::Tensor *, 12> tensors{&query, &key, &weight, &teacher, &globalLogits, &indices,
                                                 &localLogits, &gradQuery, &gradKey, &gradWeight, &loss, &trace};
  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto *tensor = tensors[index];
    const auto dtype = index == 5 ? at::kInt : index == 11 ? at::kLong : at::kFloat;
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad() &&
                  tensor->scalar_type() == dtype, "FP32 index KL requires detached contiguous FP32 buffers");
    for (size_t other = 0; other < index; ++other) {
      TORCH_CHECK(!tensor->is_alias_of(*tensors[other]), "FP32 index KL buffers must not alias");
    }
  }
  if (phase == 1) {
    gradKey.zero_();
  }
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaFp32IndexKl",
                                                          "aclnnHyperDsaFp32IndexKlGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, weight, teacher, globalLogits, indices, phase,
                                              localLogits, gradQuery, gradKey, gradWeight, loss, trace);
  return Result(localLogits, gradQuery, gradKey, gradWeight, loss, trace);
}
}  // namespace
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_fp32_index_kl_out", &fp32_index_kl_npu);
}
