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
#include "csrc/dsa_grad_checks.h"

namespace {
using Result = std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>;
using Tensors = std::array<const at::Tensor*, 20>;

void check_states(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& queryRope, const at::Tensor& keyRope,
                  const at::Tensor& out, const at::Tensor& gradOut) {
  TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == 512 &&
              (query.size(1) == 32 || query.size(1) == 64), "mixed SFA grad query must be [T,32|64,512]");
  const auto tokens = query.size(0);
  const auto heads = query.size(1);
  TORCH_CHECK(key.sizes() == at::IntArrayRef({tokens, 1, 512}) && value.sizes() == key.sizes(),
              "mixed SFA grad key/value must be [T,1,512]");
  TORCH_CHECK(queryRope.sizes() == at::IntArrayRef({tokens, heads, 64}) &&
              keyRope.sizes() == at::IntArrayRef({tokens, 1, 64}), "mixed SFA grad RoPE shapes are incompatible");
  TORCH_CHECK(out.sizes() == query.sizes() && gradOut.sizes() == out.sizes(),
              "mixed SFA grad attention output and gradient must match query shape");
  for (const auto* tensor : {&query, &key, &value, &queryRope, &keyRope, &out, &gradOut}) {
    TORCH_CHECK(tensor->scalar_type() == at::kBFloat16, "mixed SFA grad states must be BF16");
  }
}

void check_runtime(const at::Tensor& query, const at::Tensor& indices,
                   const at::Tensor& actualQuery, const at::Tensor& actualKv,
                   const at::Tensor& maximum, const at::Tensor& sum,
                   const at::Tensor& config, const at::Tensor& trace, const at::Tensor& retained) {
  const auto tokens = query.size(0);
  const auto heads = query.size(1);
  TORCH_CHECK(indices.sizes() == at::IntArrayRef({tokens, 1, 2048}) && indices.scalar_type() == at::kInt,
              "mixed SFA grad indices must be int32 [T,1,2048]");
  TORCH_CHECK(actualQuery.dim() == 1 && actualQuery.numel() > 0 && actualKv.sizes() == actualQuery.sizes() &&
              actualQuery.scalar_type() == at::kInt && actualKv.scalar_type() == at::kInt,
              "mixed SFA grad cumulative lengths must be nonempty equal-size int32 vectors");
  TORCH_CHECK(maximum.sizes() == at::IntArrayRef({1, tokens, heads}) && sum.sizes() == maximum.sizes() &&
              maximum.scalar_type() == at::kFloat && sum.scalar_type() == at::kFloat,
              "mixed SFA grad statistics must be FP32 [1,T,H]");
  TORCH_CHECK(config.sizes() == at::IntArrayRef({4}) && config.scalar_type() == at::kLong &&
              trace.sizes() == at::IntArrayRef({20, 64}) && trace.scalar_type() == at::kLong,
              "mixed SFA grad expects int64 config[4] and trace[20,64]");
  TORCH_CHECK(retained.dim() == 1 && retained.numel() > 0 && retained.scalar_type() == at::kByte,
              "mixed SFA grad retained workspace must be a nonempty uint8 vector");
}

void check_gradients(const at::Tensor& query, const at::Tensor& key, const at::Tensor& queryRope,
                     const at::Tensor& keyRope, const std::array<const at::Tensor*, 5>& gradients) {
  const std::array<const at::Tensor*, 5> states = {&query, &key, &key, &queryRope, &keyRope};
  for (size_t index = 0; index < gradients.size(); ++index) {
    TORCH_CHECK(gradients[index]->sizes() == states[index]->sizes() &&
                gradients[index]->scalar_type() == at::kBFloat16, "mixed SFA output gradient shape/dtype mismatch");
  }
}

Result dsa_mixed_grad_npu(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& indices, const at::Tensor& gradOut, const at::Tensor& out,
    const at::Tensor& maximum, const at::Tensor& sum, const at::Tensor& actualQuery, const at::Tensor& actualKv,
    const at::Tensor& queryRope, const at::Tensor& keyRope, const at::Tensor& config,
    at::Tensor& trace, at::Tensor& retained, double scale, int64_t phase,
    at::Tensor& gradQuery, at::Tensor& gradKey, at::Tensor& gradValue,
    at::Tensor& gradQueryRope, at::Tensor& gradKeyRope) {
  const Tensors tensors = {&query, &key, &value, &indices, &gradOut, &out, &maximum, &sum,
      &actualQuery, &actualKv, &queryRope, &keyRope, &config, &trace, &retained,
      &gradQuery, &gradKey, &gradValue, &gradQueryRope, &gradKeyRope};
  hyper_parallel::multicore::check_dsa_grad_inputs(tensors, scale);
  TORCH_CHECK(phase >= 0 && phase <= 2, "mixed SFA grad invalid phase");
  static const hyper_parallel::multicore::CachedOpApi api(
      "aclnnHyperDsaMixedGrad", "aclnnHyperDsaMixedGradGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, query, key, value, indices, gradOut, out, maximum, sum,
      actualQuery, actualKv, queryRope, keyRope, config, trace, retained, scale, phase,
      gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope);
  return Result(gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope, trace, retained);
}
}  // namespace

namespace hyper_parallel::multicore {
void check_dsa_grad_inputs(const DsaGradInputs &tensors, double scale) {
  const auto &query = *tensors[0], &key = *tensors[1], &value = *tensors[2];
  const auto &indices = *tensors[3], &gradOut = *tensors[4], &out = *tensors[5];
  const auto &maximum = *tensors[6], &sum = *tensors[7], &actualQuery = *tensors[8], &actualKv = *tensors[9];
  const auto &queryRope = *tensors[10], &keyRope = *tensors[11], &config = *tensors[12];
  const auto &trace = *tensors[13], &retained = *tensors[14];
  const auto &gradQuery = *tensors[15], &gradKey = *tensors[16], &gradValue = *tensors[17];
  const auto &gradQueryRope = *tensors[18], &gradKeyRope = *tensors[19];
  for (const auto* tensor : tensors) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                "mixed SFA grad requires contiguous tensors on one NPU");
    TORCH_CHECK(!tensor->requires_grad(), "mixed SFA grad native inputs must be detached");
  }
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(), "mixed SFA grad does not implement deterministic mode");
  check_states(query, key, value, queryRope, keyRope, out, gradOut);
  check_runtime(query, indices, actualQuery, actualKv, maximum, sum, config, trace, retained);
  check_gradients(query, key, queryRope, keyRope, {&gradQuery, &gradKey, &gradValue, &gradQueryRope, &gradKeyRope});
  for (const auto* output : {&trace, &retained, &gradQuery, &gradKey, &gradValue, &gradQueryRope, &gradKeyRope}) {
    for (const auto* tensor : tensors) {
      TORCH_CHECK(output == tensor || !output->is_alias_of(*tensor), "mixed SFA grad mutable buffers must not alias");
    }
  }
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "mixed SFA grad invalid scale");
}
}  // namespace hyper_parallel::multicore

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_mixed_grad_out", &dsa_mixed_grad_npu);
}
