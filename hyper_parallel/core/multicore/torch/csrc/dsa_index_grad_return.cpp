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
#include <array>
#include <ATen/ATen.h>
#include <torch/library.h>
#include "csrc/cached_op_api.h"

namespace {
std::tuple<at::Tensor &, at::Tensor &> index_grad_return_npu(const at::Tensor &gradient, at::Tensor &arena,
                                                             const at::Tensor &metadata, const at::Tensor &requests,
                                                             at::Tensor &trace, at::Tensor &owner) {
  TORCH_CHECK(gradient.device().type() == c10::DeviceType::PrivateUse1 && gradient.scalar_type() == at::kFloat &&
                gradient.dim() == 2 && gradient.size(0) > 0 && gradient.size(1) == 128,
              "index-K return requires an NPU FP32 [global keys,128] partial");
  const std::array<const at::Tensor *, 6> tensors{&gradient, &arena, &metadata, &requests, &trace, &owner};
  for (size_t index = 0; index < tensors.size(); ++index) {
    TORCH_CHECK(tensors[index]->device() == gradient.device() && tensors[index]->is_contiguous() &&
                  !tensors[index]->requires_grad(),
                "index-K return buffers must be detached contiguous tensors on one NPU");
    for (size_t other = 0; other < index; ++other) {
      TORCH_CHECK(!tensors[index]->is_alias_of(*tensors[other]), "index-K return buffers must not alias");
    }
  }
  TORCH_CHECK(arena.scalar_type() == at::kByte && arena.dim() == 1 && arena.numel() > 0,
              "index-K return requires a bound symmetric uint8 arena");
  TORCH_CHECK(metadata.scalar_type() == at::kLong && metadata.sizes() == at::IntArrayRef({18}) &&
                requests.scalar_type() == at::kLong && requests.dim() == 2 && requests.size(0) > 0 &&
                requests.size(1) == 4 && trace.scalar_type() == at::kLong && trace.sizes() == at::IntArrayRef({32}),
              "index-K return requires int64 metadata[18], requests[runs,4], trace[32]");
  TORCH_CHECK(owner.scalar_type() == at::kFloat && owner.dim() == 2 && owner.size(0) > 0 && owner.size(1) == 128,
              "index-K return requires an FP32 [owner capacity,128] output");
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaIndexGradReturn",
                                                          "aclnnHyperDsaIndexGradReturnGetWorkspaceSize");
  int64_t capacity = owner.size(0);
  hyper_parallel::multicore::execute_cached_op(api, gradient, arena, metadata, requests, trace, capacity, owner);
  return {owner, trace};
}
}  // namespace
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) { m.impl("dsa_sparse_index_grad_out", &index_grad_return_npu); }
