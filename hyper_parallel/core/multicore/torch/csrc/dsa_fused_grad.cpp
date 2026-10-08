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
#include "csrc/dsa_grad_checks.h"

namespace {
using Ref = at::Tensor &;
using Result = std::tuple<Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref>;
using Extras = std::array<const at::Tensor *, 6>;

void check_extra_shapes(const at::Tensor &query, const at::Tensor &trace, const Extras &extra) {
  TORCH_CHECK(trace.sizes() == at::IntArrayRef({3, 20, 64}) && trace.scalar_type() == at::kLong,
              "fused SFA grad expects int64 phase trace[3,20,64]");
  TORCH_CHECK(extra[0]->dim() == 1 && extra[0]->scalar_type() == at::kByte &&
              extra[1]->sizes() == at::IntArrayRef({18}) && extra[1]->scalar_type() == at::kLong &&
              extra[2]->dim() == 2 && extra[2]->size(0) > 0 && extra[2]->size(1) == 4 &&
              extra[2]->scalar_type() == at::kLong && extra[3]->sizes() == at::IntArrayRef({32}) &&
              extra[3]->scalar_type() == at::kLong, "fused SFA grad transport buffer ABI mismatch");
}

void check_grad_buffers(const at::Tensor &query, const Extras &extra) {
  TORCH_CHECK(extra[4]->dim() == 2 && extra[4]->size(0) > 0 && extra[4]->size(1) == 576 &&
              extra[4]->scalar_type() == at::kFloat &&
              extra[5]->sizes() == at::IntArrayRef({query.size(0), 1088}) && extra[5]->scalar_type() == at::kFloat,
              "fused SFA grad requires FP32 owner[capacity,576] and partials[T,1088]");
}

void check_extra_storage(const hyper_parallel::multicore::DsaGradInputs &original, const Extras &extras) {
  for (const auto *tensor : extras) {
    TORCH_CHECK(tensor->device() == original[0]->device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "fused SFA grad transport requires detached contiguous buffers on the prepared NPU");
    for (const auto *input : original) {
      TORCH_CHECK(!tensor->is_alias_of(*input), "fused SFA grad transport must own independent storage");
    }
  }
  for (size_t index : {0U, 3U, 4U, 5U}) {
    for (size_t other = 0; other < extras.size(); ++other) {
      TORCH_CHECK(index == other || !extras[index]->is_alias_of(*extras[other]),
                  "fused SFA grad mutable transport buffers must own independent storage");
    }
  }
}

Result fused_cp_grad_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &indices, const at::Tensor &gradOut, const at::Tensor &out,
    const at::Tensor &maximum, const at::Tensor &sum, const at::Tensor &actualQuery, const at::Tensor &actualKv,
    const at::Tensor &queryRope, const at::Tensor &keyRope, const at::Tensor &config,
    at::Tensor &trace, at::Tensor &retained, double scale,
    at::Tensor &gradQuery, at::Tensor &gradKey, at::Tensor &gradValue,
    at::Tensor &gradQueryRope, at::Tensor &gradKeyRope,
    at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests,
    at::Tensor &transportTrace, at::Tensor &ownerGradient, at::Tensor &partials) {
  const Extras extras{&arena, &metadata, &requests, &transportTrace, &ownerGradient, &partials};
  check_extra_shapes(query, trace, extras);
  check_grad_buffers(query, extras);
  const auto firstPhase = trace.select(0, 0);
  const hyper_parallel::multicore::DsaGradInputs tensors{
      &query, &key, &value, &indices, &gradOut, &out, &maximum, &sum, &actualQuery, &actualKv,
      &queryRope, &keyRope, &config, &firstPhase, &retained, &gradQuery, &gradKey, &gradValue,
      &gradQueryRope, &gradKeyRope};
  hyper_parallel::multicore::check_dsa_grad_inputs(tensors, scale);
  check_extra_storage(tensors, extras);
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnHyperDsaFusedGrad", "aclnnHyperDsaFusedGradGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, value, indices, gradOut, out, maximum, sum,
      actualQuery, actualKv, queryRope, keyRope, config, trace, retained, scale,
      gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope,
      arena, metadata, requests, transportTrace, ownerGradient, partials);
  return Result(gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope, trace, retained,
                ownerGradient, partials, transportTrace);
}
}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_fused_cp_grad_out", &fused_cp_grad_npu);
}
