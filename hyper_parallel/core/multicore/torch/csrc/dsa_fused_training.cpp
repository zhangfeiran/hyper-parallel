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
#include <vector>
#include "csrc/cached_op_api.h"

namespace {
using Tensors = std::array<const at::Tensor *, 21>;
using TensorRef = at::Tensor &;
using Result = std::tuple<TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef,
                          TensorRef, TensorRef, TensorRef, TensorRef>;

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

void check_states(const Tensors &tensors, bool local = false) {
  const auto &query = *tensors[2];
  const auto tokens = query.size(0);
  TORCH_CHECK(
    query.dim() == 3 && (local || tokens > 0) && (query.size(1) == 32 || query.size(1) == 64) && query.size(2) == 512,
    "fused DSA query must be [T,32|64,512]");
  TORCH_CHECK(tensors[0]->sizes() == at::IntArrayRef({tokens, 64, 128}) && tensors[1]->dim() == 3 &&
                tensors[1]->size(0) > 0 && (local ? tokens <= tensors[1]->size(0) : tokens == tensors[1]->size(0)) &&
                tensors[1]->sizes() == at::IntArrayRef({tensors[1]->size(0), 1, 128}),
              "fused DSA LI shapes are incompatible");
  TORCH_CHECK(tensors[3]->sizes() == at::IntArrayRef({tensors[1]->size(0), 1, 512}) &&
                tensors[4]->sizes() == at::IntArrayRef({tokens, query.size(1), 64}) &&
                tensors[5]->sizes() == at::IntArrayRef({tensors[1]->size(0), 1, 64}),
              "fused DSA KV/RoPE shapes are incompatible");
  for (size_t index = 0; index < 6; ++index) {
    TORCH_CHECK(tensors[index]->scalar_type() == at::kBFloat16, "fused DSA states must be BF16");
  }
  TORCH_CHECK(tensors[6]->sizes() == at::IntArrayRef({tokens, 64}) &&
                (tensors[6]->scalar_type() == at::kBFloat16 || tensors[6]->scalar_type() == at::kFloat),
              "fused DSA LI weights must be BF16 or FP32 [T,64]");
}

void check_runtime(const Tensors &tensors, int64_t phases, bool local) {
  TORCH_CHECK(tensors[7]->dim() == 1 && tensors[7]->numel() > 0 && tensors[7]->scalar_type() == at::kInt,
              "fused DSA packed cumulative lengths must be a nonempty int32 vector");
  TORCH_CHECK(tensors[8]->sizes() == at::IntArrayRef({4}) && tensors[8]->scalar_type() == at::kLong &&
                tensors[9]->sizes() == at::IntArrayRef({phases, 20, 64}) && tensors[9]->scalar_type() == at::kLong,
              "fused DSA runtime ABI expects int64 config[4] and the declared phase trace");
  TORCH_CHECK(tensors[10]->dim() == 1 && tensors[10]->numel() >= (local && tensors[2]->size(0) == 0 ? 0 : 47226880) &&
                tensors[10]->scalar_type() == at::kByte,
              "fused DSA retained scratch requires 47226880 uint8 bytes");
}

void check_kl_outputs(const Tensors &tensors, bool local = false) {
  TORCH_CHECK(tensors[11]->dim() == 1 && (tensors[11]->numel() > 0 || (local && tensors[2]->size(0) == 0)) &&
                tensors[11]->scalar_type() == at::kByte,
              "fused KL retained scratch must be a nonempty uint8 buffer");
  TORCH_CHECK(tensors[17]->sizes() == tensors[0]->sizes() && tensors[18]->sizes() == tensors[1]->sizes() &&
                tensors[19]->sizes() == tensors[6]->sizes() && tensors[17]->scalar_type() == at::kBFloat16 &&
                tensors[18]->scalar_type() == at::kBFloat16 &&
                tensors[19]->scalar_type() == tensors[6]->scalar_type() &&
                tensors[20]->sizes() == at::IntArrayRef({1}) && tensors[20]->scalar_type() == at::kFloat,
              "fused KL gradient/loss shapes or dtypes are incompatible");
}

void check_lengths(at::IntArrayRef lengths, const at::Tensor &query, const at::Tensor &deviceLengths) {
  TORCH_CHECK(!lengths.empty() && lengths.size() == deviceLengths.numel() && lengths.back() == query.size(0),
              "fused KL host cumulative lengths must describe the complete packed query/key count");
  int64_t previous = 0;
  for (const auto length : lengths) {
    TORCH_CHECK(length > previous, "fused KL host cumulative lengths must be strictly increasing");
    previous = length;
  }
}

void check_outputs(const Tensors &tensors) {
  const auto &query = *tensors[2];
  const auto tokens = query.size(0);
  TORCH_CHECK(tensors[12]->sizes() == at::IntArrayRef({tokens, 1, 2048}) && tensors[12]->scalar_type() == at::kInt &&
                tensors[13]->sizes() == tensors[12]->sizes() && tensors[13]->scalar_type() == at::kBFloat16,
              "fused DSA LI output shapes or dtypes are incompatible");
  TORCH_CHECK(tensors[14]->sizes() == query.sizes() && tensors[14]->scalar_type() == at::kBFloat16 &&
                tensors[15]->sizes() == at::IntArrayRef({1, tokens, query.size(1)}) &&
                tensors[16]->sizes() == tensors[15]->sizes() && tensors[15]->scalar_type() == at::kFloat &&
                tensors[16]->scalar_type() == at::kFloat,
              "fused DSA SFA output shapes or dtypes are incompatible");
}

void check_fused_inputs(const Tensors &tensors, double scale, int64_t phases = 6, bool withKl = true,
                        bool local = false) {
  TORCH_CHECK(tensors[0]->device().type() == c10::DeviceType::PrivateUse1, "fused training requires an NPU");
  TORCH_CHECK(!withKl || !at::globalContext().deterministicAlgorithms(),
              "fused training supports non-deterministic KL only");
  check_storage(tensors);
  check_states(tensors, local);
  check_runtime(tensors, phases, local);
  check_outputs(tensors);
  if (withKl) {
    check_kl_outputs(tensors, local);
  } else {
    for (size_t index : {11U, 17U, 18U, 19U, 20U}) {
      TORCH_CHECK(tensors[index]->sizes() == at::IntArrayRef({0}), "selected forward requires empty KL descriptors");
    }
  }
  TORCH_CHECK(std::isfinite(scale) && scale > 0, "fused DSA scale must be finite and positive");
}

Result fused_training_npu(const at::Tensor &indexQuery, const at::Tensor &indexKey, const at::Tensor &query,
                          const at::Tensor &compressed, const at::Tensor &queryRope, const at::Tensor &keyRope,
                          const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
                          at::Tensor &trace, at::Tensor &retained, at::Tensor &klRetained, at::IntArrayRef klLengths,
                          double scale, at::Tensor &indices, at::Tensor &values, at::Tensor &attention,
                          at::Tensor &maximum, at::Tensor &sum, at::Tensor &gradIndexQuery, at::Tensor &gradIndexKey,
                          at::Tensor &gradWeight, at::Tensor &loss) {
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale);
  check_lengths(klLengths, query, lengths);
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaFusedTraining",
                                                          "aclnnHyperDsaFusedTrainingGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, indexQuery, indexKey, query, compressed, queryRope, keyRope,
                                               weights, lengths, config, trace, retained, klRetained, klLengths, scale,
                                               indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey,
                                               gradWeight, loss);
  return Result(indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight, loss, trace,
                retained, klRetained);
}
void check_transport_shapes(const at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests,
                            const at::Tensor &transportTrace) {
  TORCH_CHECK(arena.dim() == 1 && arena.scalar_type() == at::kByte &&
                (metadata.dim() == 1 && (metadata.numel() == 18 || metadata.numel() == 27)) &&
                metadata.scalar_type() == at::kLong && requests.dim() == 2 && requests.size(0) > 0 &&
                requests.size(1) == 4 && requests.scalar_type() == at::kLong &&
                transportTrace.sizes() == at::IntArrayRef({32}) && transportTrace.scalar_type() == at::kLong,
              "fused CP transport buffer ABI mismatch");
}

void check_transport(const Tensors &tensors, const at::Tensor &arena, const at::Tensor &metadata,
                     const at::Tensor &requests, const at::Tensor &transportTrace) {
  check_transport_shapes(arena, metadata, requests, transportTrace);
  for (const auto *tensor : {&arena, &metadata, &requests, &transportTrace}) {
    TORCH_CHECK(tensor->device() == tensors[0]->device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "fused CP transport requires detached contiguous buffers on the prepared NPU");
    for (const auto *original : tensors) {
      TORCH_CHECK(!tensor->is_alias_of(*original), "fused CP transport must own independent storage");
    }
  }
  const std::array<const at::Tensor *, 4> extras{&arena, &metadata, &requests, &transportTrace};
  for (size_t index : {0U, 3U}) {
    for (size_t other = 0; other < extras.size(); ++other) {
      TORCH_CHECK(index == other || !extras[index]->is_alias_of(*extras[other]),
                  "fused CP mutable transport buffers must own independent storage");
    }
  }
  for (size_t index : {1U, 3U, 5U}) {
    for (size_t other = 0; other < tensors.size(); ++other) {
      TORCH_CHECK(index == other || !tensors[index]->is_alias_of(*tensors[other]),
                  "fused CP key destinations must own independent storage");
    }
  }
}

void check_selected(const Tensors &tensors, const std::array<const at::Tensor *, 8> &extras) {
  const auto tokens = tensors[1]->size(0);
  TORCH_CHECK(tokens <= INT32_MAX / 2048, "selected membership occurrence counts exceed int32 capacity");
  TORCH_CHECK(extras[4]->sizes() == at::IntArrayRef({tokens, 3}) && extras[4]->scalar_type() == at::kLong &&
                extras[5]->sizes() == at::IntArrayRef({(tokens + 7) / 8 * 8}) && extras[5]->scalar_type() == at::kInt &&
                extras[6]->sizes() == at::IntArrayRef({tokens, 4}) && extras[6]->scalar_type() == at::kLong &&
                extras[7]->sizes() == at::IntArrayRef({16}) && extras[7]->scalar_type() == at::kLong,
              "selected request buffers must match prepared packed tokens and bounded capacity");
  for (size_t index = 4; index < extras.size(); ++index) {
    const auto &tensor = *extras[index];
    TORCH_CHECK(tensor.device() == tensors[0]->device() && tensor.is_contiguous() && !tensor.requires_grad(),
                "selected request buffers must be detached contiguous tensors on the prepared NPU");
    for (const auto *original : tensors) {
      TORCH_CHECK(!tensor.is_alias_of(*original), "selected request buffers must own independent storage");
    }
    for (size_t other = 0; other < extras.size(); ++other) {
      TORCH_CHECK(index == other || !tensor.is_alias_of(*extras[other]), "selected request buffers must not alias");
    }
  }
}

Result fused_cp_training_npu(const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query,
                             at::Tensor &compressed, const at::Tensor &queryRope, at::Tensor &keyRope,
                             const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
                             at::Tensor &trace, at::Tensor &retained, at::Tensor &klRetained, at::IntArrayRef klLengths,
                             double scale, at::Tensor &indices, at::Tensor &values, at::Tensor &attention,
                             at::Tensor &maximum, at::Tensor &sum, at::Tensor &gradIndexQuery, at::Tensor &gradIndexKey,
                             at::Tensor &gradWeight, at::Tensor &loss, at::Tensor &arena, const at::Tensor &metadata,
                             const at::Tensor &requests, at::Tensor &transportTrace) {
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale);
  check_lengths(klLengths, query, lengths);
  check_transport(tensors, arena, metadata, requests, transportTrace);
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaFusedCpTraining",
                                                          "aclnnHyperDsaFusedCpTrainingGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(api, indexQuery, indexKey, query, compressed, queryRope, keyRope,
                                               weights, lengths, config, trace, retained, klRetained, klLengths, scale,
                                               indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey,
                                               gradWeight, loss, arena, metadata, requests, transportTrace);
  return Result(indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight, loss, trace,
                retained, klRetained);
}
Result selected_cp_training_npu(const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query,
                                at::Tensor &compressed, const at::Tensor &queryRope, at::Tensor &keyRope,
                                const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
                                at::Tensor &trace, at::Tensor &retained, at::Tensor &klRetained,
                                at::IntArrayRef klLengths, double scale, at::Tensor &indices, at::Tensor &values,
                                at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum, at::Tensor &gradIndexQuery,
                                at::Tensor &gradIndexKey, at::Tensor &gradWeight, at::Tensor &loss, at::Tensor &arena,
                                const at::Tensor &metadata, const at::Tensor &requests, at::Tensor &transportTrace,
                                const at::Tensor &selectedRows, at::Tensor &membership, at::Tensor &selectedRequests,
                                at::Tensor &selectedCounts) {
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale, 9);
  check_selected(tensors, {&arena, &metadata, &requests, &transportTrace, &selectedRows, &membership, &selectedRequests,
                           &selectedCounts});
  check_lengths(klLengths, query, lengths);
  check_transport(tensors, arena, metadata, requests, transportTrace);
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaSelectedCpTraining",
                                                          "aclnnHyperDsaSelectedCpTrainingGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(
    api, indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained,
    klRetained, klLengths, scale, indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight,
    loss, arena, metadata, requests, transportTrace, selectedRows, membership, selectedRequests, selectedCounts);
  return Result(indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight, loss, trace,
                retained, klRetained);
}
std::tuple<TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef> selected_cp_forward_npu(
  const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query, at::Tensor &compressed,
  const at::Tensor &queryRope, at::Tensor &keyRope, const at::Tensor &weights, const at::Tensor &lengths,
  const at::Tensor &config, at::Tensor &trace, at::Tensor &retained, at::IntArrayRef klLengths, double scale,
  at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum,
  at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests, at::Tensor &transportTrace,
  const at::Tensor &selectedRows, at::Tensor &membership, at::Tensor &selectedRequests, at::Tensor &selectedCounts) {
  auto klRetained = at::empty({0}, retained.options());
  auto gradIndexQuery = at::empty({0}, indexQuery.options());
  auto gradIndexKey = at::empty({0}, indexKey.options());
  auto gradWeight = at::empty({0}, weights.options());
  auto loss = at::empty({0}, maximum.options());
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale, 6, false);
  check_selected(tensors, {&arena, &metadata, &requests, &transportTrace, &selectedRows, &membership, &selectedRequests,
                           &selectedCounts});
  check_lengths(klLengths, query, lengths);
  check_transport(tensors, arena, metadata, requests, transportTrace);
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaSelectedCpForward",
                                                          "aclnnHyperDsaSelectedCpForwardGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(
    api, indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained,
    klRetained, klLengths, scale, indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight,
    loss, arena, metadata, requests, transportTrace, selectedRows, membership, selectedRequests, selectedCounts);
  return {indices, values, attention, maximum, sum, trace, retained};
}
std::tuple<TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef, TensorRef> local_cp_forward_npu(
  const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query, at::Tensor &compressed,
  const at::Tensor &queryRope, at::Tensor &keyRope, const at::Tensor &weights, const at::Tensor &lengths,
  const at::Tensor &config, at::Tensor &trace, at::Tensor &retained, at::IntArrayRef klLengths, double scale,
  at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum,
  at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests, at::Tensor &transportTrace,
  const at::Tensor &selectedRows, at::Tensor &membership, at::Tensor &selectedRequests, at::Tensor &selectedCounts,
  const at::Tensor &queryLengths, const at::Tensor &queryPositions, const at::Tensor &queryIds) {
  auto klRetained = at::empty({0}, retained.options());
  auto gradIndexQuery = at::empty({0}, indexQuery.options());
  auto gradIndexKey = at::empty({0}, indexKey.options());
  auto gradWeight = at::empty({0}, weights.options());
  auto loss = at::empty({0}, maximum.options());
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale, 6, false, true);
  check_selected(tensors, {&arena, &metadata, &requests, &transportTrace, &selectedRows, &membership, &selectedRequests,
                           &selectedCounts});
  check_lengths(klLengths, indexKey, lengths);
  check_transport(tensors, arena, metadata, requests, transportTrace);
  TORCH_CHECK(queryLengths.sizes() == lengths.sizes() && queryLengths.scalar_type() == at::kInt &&
                queryPositions.sizes() == at::IntArrayRef({query.size(0)}) &&
                queryPositions.scalar_type() == at::kLong && queryIds.sizes() == queryPositions.sizes() &&
                queryIds.scalar_type() == at::kLong,
              "local CP requires genuine int32 query lengths and int64 positions/global IDs");
  for (const auto *tensor : {&queryLengths, &queryPositions, &queryIds}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "local CP metadata must be detached contiguous tensors on the prepared NPU");
    for (const auto *original : tensors) {
      TORCH_CHECK(!tensor->is_alias_of(*original), "local metadata must not alias arithmetic buffers");
    }
    const std::array<const at::Tensor *, 8> extras{&arena,        &metadata,   &requests,         &transportTrace,
                                                   &selectedRows, &membership, &selectedRequests, &selectedCounts};
    for (const auto *extra : extras) {
      TORCH_CHECK(!tensor->is_alias_of(*extra), "local metadata must not alias transport/request buffers");
    }
  }
  static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaLocalCpForward",
                                                          "aclnnHyperDsaLocalCpForwardGetWorkspaceSize");
  hyper_parallel::multicore::execute_cached_op(
    api, indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained,
    klRetained, klLengths, scale, indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight,
    loss, arena, metadata, requests, transportTrace, selectedRows, membership, selectedRequests, selectedCounts,
    queryLengths, queryPositions, queryIds);
  return {indices, values, attention, maximum, sum, trace, retained};
}
Result RunLocalCpTraining(const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query,
                          at::Tensor &compressed, const at::Tensor &queryRope, at::Tensor &keyRope,
                          const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
                          at::Tensor &trace, at::Tensor &retained, at::IntArrayRef klLengths, double scale,
                          at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum,
                          at::Tensor &sum, at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests,
                          at::Tensor &transportTrace, const at::Tensor &selectedRows, at::Tensor &membership,
                          at::Tensor &selectedRequests, at::Tensor &selectedCounts, const at::Tensor &queryLengths,
                          const at::Tensor &queryPositions, const at::Tensor &queryIds, at::IntArrayRef klQueryLengths,
                          at::Tensor &klRetained, at::Tensor &gradIndexQuery, at::Tensor &gradIndexKey,
                          at::Tensor &gradWeight, at::Tensor &loss, bool withKl, at::Tensor *tpLiArena,
                          const at::Tensor *tpLiMetadata, const c10::optional<at::Tensor> &tpKlArena,
                          const c10::optional<at::Tensor> &tpKlMetadata) {
  const Tensors tensors{&indexQuery, &indexKey, &query, &compressed,     &queryRope,    &keyRope,    &weights,
                        &lengths,    &config,   &trace, &retained,       &klRetained,   &indices,    &values,
                        &attention,  &maximum,  &sum,   &gradIndexQuery, &gradIndexKey, &gradWeight, &loss};
  check_fused_inputs(tensors, scale, withKl ? 9 : 6, withKl, true);
  check_selected(tensors, {&arena, &metadata, &requests, &transportTrace, &selectedRows, &membership, &selectedRequests,
                           &selectedCounts});
  check_lengths(klLengths, indexKey, lengths);
  TORCH_CHECK(
    klQueryLengths.size() == klLengths.size() && !klQueryLengths.empty() && klQueryLengths.back() == query.size(0),
    "local KL cumulative queries must describe actual rows");
  int64_t previousQuery = 0, previousKey = 0;
  for (size_t index = 0; index < klQueryLengths.size(); ++index) {
    TORCH_CHECK(
      klQueryLengths[index] >= previousQuery && klQueryLengths[index] - previousQuery <= klLengths[index] - previousKey,
      "local KL query counts must be nondecreasing within key sequence sizes");
    previousQuery = klQueryLengths[index];
    previousKey = klLengths[index];
  }
  check_transport(tensors, arena, metadata, requests, transportTrace);
  TORCH_CHECK(queryLengths.sizes() == lengths.sizes() && queryLengths.scalar_type() == at::kInt &&
                queryPositions.sizes() == at::IntArrayRef({query.size(0)}) &&
                queryPositions.scalar_type() == at::kLong && queryIds.sizes() == queryPositions.sizes() &&
                queryIds.scalar_type() == at::kLong,
              "local CP requires genuine int32 query lengths and int64 positions/global IDs");
  for (const auto *tensor : {&queryLengths, &queryPositions, &queryIds}) {
    TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad(),
                "local CP metadata must be detached contiguous tensors on the prepared NPU");
    for (const auto *original : tensors) {
      TORCH_CHECK(!tensor->is_alias_of(*original), "local metadata must not alias arithmetic buffers");
    }
    const std::array<const at::Tensor *, 8> extras{&arena,        &metadata,   &requests,         &transportTrace,
                                                   &selectedRows, &membership, &selectedRequests, &selectedCounts};
    for (const auto *extra : extras) {
      TORCH_CHECK(!tensor->is_alias_of(*extra), "local metadata must not alias transport/request buffers");
    }
  }
  if (tpLiArena != nullptr) {
    TORCH_CHECK(tpLiMetadata != nullptr && tpKlArena.has_value() == withKl && tpKlMetadata.has_value() == withKl,
                "fused TP requires LI transport and a KL transport pair exactly when KL is enabled");
    std::vector<const at::Tensor *> transport{tpLiArena, tpLiMetadata};
    if (withKl) {
      transport.push_back(&tpKlArena.value());
      transport.push_back(&tpKlMetadata.value());
    }
    const std::array<const at::Tensor *, 11> extra{&arena,        &metadata,       &requests,         &transportTrace,
                                                   &selectedRows, &membership,     &selectedRequests, &selectedCounts,
                                                   &queryLengths, &queryPositions, &queryIds};
    for (size_t index = 0; index < transport.size(); ++index) {
      const auto *tensor = transport[index];
      TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous() && !tensor->requires_grad() &&
                    (index % 2 == 0 ? (tensor->dim() == 1 && tensor->numel() > 0 && tensor->scalar_type() == at::kByte)
                                    : (tensor->sizes() == at::IntArrayRef({24}) && tensor->scalar_type() == at::kLong)),
                  "fused TP requires detached uint8 arenas and int64[24] metadata on the prepared NPU");
      for (const auto *original : tensors) {
        TORCH_CHECK(!tensor->is_alias_of(*original), "TP transport must not alias arithmetic buffers");
      }
      for (const auto *original : extra) {
        TORCH_CHECK(!tensor->is_alias_of(*original), "TP transport must not alias CP transport or local metadata");
      }
      for (size_t previous = 0; previous < index; ++previous) {
        TORCH_CHECK(!tensor->is_alias_of(*transport[previous]), "TP transport arenas/metadata must not alias");
      }
    }
    static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaTpLocalCpTraining",
                                                            "aclnnHyperDsaTpLocalCpTrainingGetWorkspaceSize");
    hyper_parallel::multicore::execute_cached_op(
      api, indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained,
      klRetained, klLengths, scale, indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight,
      loss, arena, metadata, requests, transportTrace, selectedRows, membership, selectedRequests, selectedCounts,
      queryLengths, queryPositions, queryIds, klQueryLengths, *tpLiArena, *tpLiMetadata, tpKlArena, tpKlMetadata,
      withKl);
  } else {
    static const hyper_parallel::multicore::CachedOpApi api("aclnnHyperDsaLocalCpTraining",
                                                            "aclnnHyperDsaLocalCpTrainingGetWorkspaceSize");
    hyper_parallel::multicore::execute_cached_op(
      api, indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained,
      klRetained, klLengths, scale, indices, values, attention, maximum, sum, gradIndexQuery, gradIndexKey, gradWeight,
      loss, arena, metadata, requests, transportTrace, selectedRows, membership, selectedRequests, selectedCounts,
      queryLengths, queryPositions, queryIds, klQueryLengths);
  }
  return {indices,      values,     attention, maximum, sum,      gradIndexQuery,
          gradIndexKey, gradWeight, loss,      trace,   retained, klRetained};
}
Result local_cp_training_npu(const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query,
                             at::Tensor &compressed, const at::Tensor &queryRope, at::Tensor &keyRope,
                             const at::Tensor &weights, const at::Tensor &lengths, const at::Tensor &config,
                             at::Tensor &trace, at::Tensor &retained, at::IntArrayRef klLengths, double scale,
                             at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum,
                             at::Tensor &sum, at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests,
                             at::Tensor &transportTrace, const at::Tensor &selectedRows, at::Tensor &membership,
                             at::Tensor &selectedRequests, at::Tensor &selectedCounts, const at::Tensor &queryLengths,
                             const at::Tensor &queryPositions, const at::Tensor &queryIds,
                             at::IntArrayRef klQueryLengths, at::Tensor &klRetained, at::Tensor &gradIndexQuery,
                             at::Tensor &gradIndexKey, at::Tensor &gradWeight, at::Tensor &loss) {
  return RunLocalCpTraining(
    indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained, klLengths,
    scale, indices, values, attention, maximum, sum, arena, metadata, requests, transportTrace, selectedRows,
    membership, selectedRequests, selectedCounts, queryLengths, queryPositions, queryIds, klQueryLengths, klRetained,
    gradIndexQuery, gradIndexKey, gradWeight, loss, true, nullptr, nullptr, c10::nullopt, c10::nullopt);
}

Result tp_local_cp_training_npu(
  const at::Tensor &indexQuery, at::Tensor &indexKey, const at::Tensor &query, at::Tensor &compressed,
  const at::Tensor &queryRope, at::Tensor &keyRope, const at::Tensor &weights, const at::Tensor &lengths,
  const at::Tensor &config, at::Tensor &trace, at::Tensor &retained, at::IntArrayRef klLengths, double scale,
  at::Tensor &indices, at::Tensor &values, at::Tensor &attention, at::Tensor &maximum, at::Tensor &sum,
  at::Tensor &arena, const at::Tensor &metadata, const at::Tensor &requests, at::Tensor &transportTrace,
  const at::Tensor &selectedRows, at::Tensor &membership, at::Tensor &selectedRequests, at::Tensor &selectedCounts,
  const at::Tensor &queryLengths, const at::Tensor &queryPositions, const at::Tensor &queryIds,
  at::IntArrayRef klQueryLengths, at::Tensor &klRetained, at::Tensor &gradIndexQuery, at::Tensor &gradIndexKey,
  at::Tensor &gradWeight, at::Tensor &loss, at::Tensor &tpLiArena, const at::Tensor &tpLiMetadata,
  const c10::optional<at::Tensor> &tpKlArena, const c10::optional<at::Tensor> &tpKlMetadata, bool withKl) {
  return RunLocalCpTraining(
    indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, config, trace, retained, klLengths,
    scale, indices, values, attention, maximum, sum, arena, metadata, requests, transportTrace, selectedRows,
    membership, selectedRequests, selectedCounts, queryLengths, queryPositions, queryIds, klQueryLengths, klRetained,
    gradIndexQuery, gradIndexKey, gradWeight, loss, withKl, &tpLiArena, &tpLiMetadata, tpKlArena, tpKlMetadata);
}

}  // namespace

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("dsa_fused_training_out", &fused_training_npu);
  m.impl("dsa_fused_cp_training_out", &fused_cp_training_npu);
  m.impl("dsa_selected_cp_training_out", &selected_cp_training_npu);
  m.impl("dsa_selected_cp_forward_out", &selected_cp_forward_npu);
  m.impl("dsa_local_cp_forward_out", &local_cp_forward_npu);
  m.impl("dsa_local_cp_training_out", &local_cp_training_npu);
  m.impl("dsa_tp_local_cp_training_out", &tp_local_cp_training_npu);
}
