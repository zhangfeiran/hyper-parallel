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
// SDK umbrella and generated tiling headers belong to this assembled include closure.
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "arch22/sparse_flash_attention_grad_bs1_basic.h"
#include "runtime/dsa_mixed_group.h"

using namespace AscendC;  // NOLINT(build/namespaces)
using namespace DsaMixed;  // NOLINT(build/namespaces)

namespace {
__aicore__ inline bool ValidConfig(GlobalTensor<int64_t> &config, uint32_t groups, uint32_t count) {
  return config.GetValue(0) == kMixedMagic && config.GetValue(1) == 1 &&
         groups > 0 && groups < count && config.GetValue(3) == 1;
}
}  // namespace

extern "C" __global__ __aicore__ void hyper_dsa_mixed_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *indices,
    __gm__ uint8_t *gradOut, __gm__ uint8_t *out, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
    __gm__ uint8_t *value, __gm__ uint8_t *actualQuery, __gm__ uint8_t *actualKv,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *runtimeConfig,
    __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained,
    __gm__ uint8_t *gradQuery, __gm__ uint8_t *gradKey, __gm__ uint8_t *gradValue,
    __gm__ uint8_t *gradQueryRope, __gm__ uint8_t *gradKeyRope,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GlobalTensor<int64_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (!ValidConfig(config, groups, physicalCount)) {
    return;
  }
  uint32_t group = GetBlockIdx();
  if ASCEND_IS_AIV {
    group /= 2;
  }
  if (group >= groups) {
    return;
  }
  GlobalTensor<int64_t> trace;
  trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupTrace), physicalCount * kGroupWords);
  GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionGradBasicTilingData, tilingData, tiling);
  uint32_t count = 0;
  uint32_t checksum = 0;
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    using TileType = SFAG_BASIC::SFAG_TYPE<SparseFlashAttentionGradBasicTilingData,
        bfloat16_t, 3, true, true, false, false, false>;
    {
      SFAG_BASIC::SelectedAttentionGradBasic<TileType> tile;
      tile.Process(query, key, value, out, gradOut, maximum, sum, indices, actualQuery, actualKv,
                   queryRope, keyRope, gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope,
                   retained, &tilingData, ticket, group, tilingData.mixedPhase);
    }
    ++count;
    checksum += ticket + 1;
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
}
