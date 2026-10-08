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
// CANN compiles its umbrella and template headers from the SDK/assembled include roots.
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "runtime/dsa_mixed_group.h"
#include "lightning_indexer_template_tiling_key.h"  // NOLINT(build/include_subdir)
#include "arch22/lightning_indexer_kernel.h"

using AscendC::CacheLine;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DcciDst;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetSubBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::PipeBarrier;
using AscendC::TPipe;

using namespace DsaMixed;  // NOLINT(build/namespaces)

namespace {
__aicore__ inline bool ValidConfig(GlobalTensor<int64_t> &config, uint32_t groups, uint32_t physicalCount) {
  return config.GetValue(0) == kMixedMagic && config.GetValue(1) == 1 &&
         groups > 0 && groups < physicalCount && config.GetValue(3) == 1;
}
}  // namespace


template<int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void hyper_dsa_mixed_indexer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *actualQuery, __gm__ uint8_t *actualKv, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained,
    __gm__ uint8_t *indices, __gm__ uint8_t *values, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if constexpr (DT_Q != LI_TPL_BF16 || DT_K != LI_TPL_BF16 || PAGE_ATTENTION ||
                LAYOUT_T != LI_LAYOUT_TND || K_LAYOUT_T != LI_LAYOUT_TND) {
    return;
  }
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
  GET_TILING_DATA_WITH_STRUCT(LITilingData, tilingData, tiling);
  const bool mergePhase = tilingData.mergePhase != 0;
  uint32_t count = 0;
  uint32_t checksum = 0;
  uint32_t ldCount = 0;
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    {
      TPipe pipe;
      using TileType = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, false,
          LICommon::LI_LAYOUT::TND, LICommon::LI_LAYOUT::TND, DT_W_FLAG>;
      LIKernel::LightningIndexerKernel<TileType> tile;
      tile.Init(query, key, weights, actualQuery, actualKv, blockTable, indices, values,
                retained, &tilingData, &pipe, ticket, physicalCount, group, physicalCount, mergePhase);
      ldCount += tile.IsLdPartition() ? 1 : 0;
      tile.ProcessPhase(mergePhase);
    }
    ++count;
    checksum += ticket + 1;
    uint32_t member = 0;
    if ASCEND_IS_AIV {
      member = 1 + GetSubBlockIdx();
    }
    trace.SetValue(group * kGroupWords + member * kMemberWords + 4, ldCount);
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
}
