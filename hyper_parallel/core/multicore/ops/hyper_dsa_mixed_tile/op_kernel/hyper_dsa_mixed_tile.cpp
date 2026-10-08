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
#include "sparse_flash_attention_template_tiling_key.h"  // NOLINT(build/include_subdir)
#include "arch22/sparse_flash_attention_kernel_mla.h"

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

namespace {
constexpr int64_t kMixedMagic = 0x48504453414D4958;
constexpr uint32_t kDispatchFlag = 13;
constexpr uint32_t kCompleteFlag = 14;
constexpr uint32_t kGroupWords = 64;
constexpr uint32_t kMemberWords = 16;

__aicore__ inline void FlushLine(GlobalTensor<int64_t> &tensor, uint32_t offset) {
  DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tensor[offset]);
}

// Each role owns a separate cache line; a cube ticket is visible only after dispatch.
__aicore__ inline uint32_t Dispatch(GlobalTensor<int64_t> &trace, uint32_t row, uint32_t logical) {
  if ASCEND_IS_AIC {
    trace.SetValue(row, logical);
    FlushLine(trace, row);
    // The ticket is a scalar GM store; FIX notification alone does not order its publication.
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<2, PIPE_FIX>(kDispatchFlag);
    return logical;
  } else {
    CrossCoreWaitFlag(kDispatchFlag);
    FlushLine(trace, row);
    PipeBarrier<PIPE_ALL>();
    volatile __gm__ int64_t *ticket = trace.GetPhyAddr(row);
    return static_cast<uint32_t>(*ticket);
  }
}

__aicore__ inline void Complete(GlobalTensor<int64_t> &trace, uint32_t row,
                               uint32_t count, uint32_t checksum, uint32_t logical) {
  uint32_t member = 0;
  if ASCEND_IS_AIV {
    member = 1 + GetSubBlockIdx();
  }
  const uint32_t offset = row + member * kMemberWords;
  trace.SetValue(offset + 1, count);
  trace.SetValue(offset + 2, checksum);
  trace.SetValue(offset + 3, logical);
  FlushLine(trace, offset);
  PipeBarrier<PIPE_ALL>();
  if ASCEND_IS_AIV {
    CrossCoreSetFlag<2, PIPE_MTE3>(kCompleteFlag);
  } else {
    CrossCoreWaitFlag(kCompleteFlag);
  }
}
}  // namespace

template<int FLASH_DECODE, int PAGE_ATTENTION, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
__global__ __aicore__ void hyper_dsa_mixed_tile(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualQuery, __gm__ uint8_t *actualKv,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if constexpr (FLASH_DECODE || PAGE_ATTENTION || LAYOUT_T != 1 || KV_LAYOUT_T != 1) {
    return;
  }
  GlobalTensor<int64_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t rounds = static_cast<uint32_t>(config.GetValue(3));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 ||
      groups == 0 || groups >= physicalCount || rounds == 0 || rounds > 64) {
    return;
  }
  uint32_t group = GetBlockIdx();
  if ASCEND_IS_AIV {
    group /= 2;
  }
  // At least one complete physical group remains available to a future progress worker.
  if (group >= groups) {
    return;
  }
  GlobalTensor<int64_t> trace;
  trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupTrace), physicalCount * kGroupWords);
  GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingDataMla, tilingData, tiling);
  auto *user = GetUserWorkspace(workspace);
  uint32_t count = 0;
  uint32_t checksum = 0;
  for (uint32_t round = 0; round < rounds; ++round) {
    for (uint32_t logical = group; logical < physicalCount; logical += groups) {
      const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
      {
        TPipe pipe;
        using TileType = SFAType<bfloat16_t, bfloat16_t, bfloat16_t, 0,
            SFA_LAYOUT::TND, SFA_LAYOUT::TND, TEMPLATE_MODE>;
        SparseFlashAttentionMla<TileType> tile;
        tile.Init(query, key, value, sparseIndices, actualQuery, actualKv, blockTable,
                  queryRope, keyRope, attentionOut, softmaxMax, softmaxSum,
                  user, &tilingData, tiling, &pipe, ticket, physicalCount, group, physicalCount);
        tile.Process();
      }
      ++count;
      checksum += ticket + 1;
      Complete(trace, group * kGroupWords, count, checksum, ticket);
    }
  }
}
