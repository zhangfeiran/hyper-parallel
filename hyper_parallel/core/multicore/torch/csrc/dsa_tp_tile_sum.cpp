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
at::Tensor &tp_tile_sum_npu(const at::Tensor &partial, at::Tensor &arena, const at::Tensor &metadata,
                            at::Tensor &summed) {
  TORCH_CHECK(partial.device().type() == c10::DeviceType::PrivateUse1 && partial.scalar_type() == at::kFloat &&
                partial.dim() == 3 && partial.size(0) > 0 && partial.size(0) < (1LL << 32) && partial.size(1) > 0 &&
                partial.size(1) <= 40 && partial.size(2) > 0 && partial.size(2) <= 2048 && partial.size(2) % 8 == 0,
              "TP tile SUM requires NPU FP32 [rounds,slots<=40,width<=2048] with width a multiple of eight");
  const std::array<const at::Tensor *, 4> tensors{&partial, &arena, &metadata, &summed};
  for (size_t index = 0; index < tensors.size(); ++index) {
    TORCH_CHECK(tensors[index]->device() == partial.device() && tensors[index]->is_contiguous() &&
                  !tensors[index]->requires_grad(),
                "TP tile SUM buffers must be detached contiguous tensors on one NPU");
    for (size_t other = 0; other < index; ++other) {
      TORCH_CHECK(!tensors[index]->is_alias_of(*tensors[other]), "TP tile SUM buffers must not alias");
    }
  }
  TORCH_CHECK(arena.scalar_type() == at::kByte && arena.dim() == 1 && arena.numel() > 0 &&
                metadata.scalar_type() == at::kLong && metadata.sizes() == at::IntArrayRef({24}) &&
                summed.scalar_type() == at::kFloat && summed.sizes() == partial.sizes(),
              "invalid TP tile SUM arena/metadata/output");
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaTpTileSum",
                                                          "aclnnHyperDsaTpTileSumGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, partial, arena, metadata, summed);
  return summed;
}
}  // namespace
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) { m.impl("dsa_tp_tile_sum_out", &tp_tile_sum_npu); }
