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
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "arch22/sparse_flash_attention_grad_bs1_basic.h"
#include "runtime/dsa_mixed_group.h"
#include "runtime/dsa_cp_grad_transport.h"

using namespace AscendC;  // NOLINT(build/namespaces)
using namespace DsaMixed;  // NOLINT(build/namespaces)

namespace {
struct Buffers {
  __gm__ uint8_t *query, *key, *value, *indices, *gradOut, *out, *maximum, *sum;
  __gm__ uint8_t *actualQuery, *actualKv, *queryRope, *keyRope, *retained;
  __gm__ uint8_t *gradQuery, *gradKey, *gradValue, *gradQueryRope, *gradKeyRope;
};

__aicore__ inline void ClosePhase(GlobalTensor<int64_t> &trace, uint32_t groups, int64_t epoch) {
  TPipe pipe;
  uint32_t member = 0;
  if ASCEND_IS_AIV {
    member = 1 + GetSubBlockIdx();
    if (GetSubBlockIdx() == 0) {
      CoordinatePhase(trace, groups, epoch);
    } else {
      WaitPhase(trace, groups, epoch);
    }
  } else {
    WaitPhase(trace, groups, epoch);
  }
  PublishControl(trace, groups * kGroupWords + member * kMemberWords + kArrivalWord, epoch);
}

__aicore__ inline void ComputePhase(const Buffers &buffers, const SparseFlashAttentionGradBasicTilingData *data,
                                   GlobalTensor<int64_t> &trace, uint32_t phase,
                                   uint32_t group, uint32_t groups, uint32_t physicalCount) {
  uint32_t count = 0, checksum = 0;
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    using TileType = SFAG_BASIC::SFAG_TYPE<SparseFlashAttentionGradBasicTilingData,
        bfloat16_t, 3, true, true, false, false, false>;
    {
      SFAG_BASIC::SelectedAttentionGradBasic<TileType> tile;
      tile.Process(buffers.query, buffers.key, buffers.value, buffers.out, buffers.gradOut,
                   buffers.maximum, buffers.sum, buffers.indices, buffers.actualQuery, buffers.actualKv,
                   buffers.queryRope, buffers.keyRope, buffers.gradQuery, buffers.gradKey, buffers.gradValue,
                   buffers.gradQueryRope, buffers.gradKeyRope, buffers.retained, data, ticket, group, phase);
    }
    ++count;
    checksum += ticket + 1;
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
  ArriveAndWait(trace, group, groups, phase + 1);
}

__aicore__ inline void Run(const Buffers &buffers, CpGradTransport &transport,
                           __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace, __gm__ uint8_t *tiling) {
  GlobalTensor<int64_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 || groups == 0 ||
      groups >= physicalCount || config.GetValue(3) != 1) { return; }
  uint32_t group = GetBlockIdx();
  if ASCEND_IS_AIV {
    group /= 2;
  }
  if (group > groups) {
    return;
  }
  GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionGradBasicTilingData, data, tiling);
  for (uint32_t phase = 0; phase < 3; ++phase) {
    GlobalTensor<int64_t> trace;
    trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupTrace) + phase * physicalCount * kGroupWords,
                          physicalCount * kGroupWords);
    if (group == groups) {
      if ASCEND_IS_AIV {
        if (GetSubBlockIdx() == 0 && phase == 0) {
          transport.Prepare();
        }
        if (GetSubBlockIdx() == 0 && phase == 2) {
          transport.Progress(buffers.retained, data.postTilingData.dkWorkSpaceOffset,
                             data.postTilingData.dvWorkSpaceOffset);
        }
      }
      ClosePhase(trace, groups, phase + 1);
    } else {
      ComputePhase(buffers, &data, trace, phase, group, groups, physicalCount);
    }
  }
}
}  // namespace

extern "C" __global__ __aicore__ void hyper_dsa_fused_grad(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *indices,
    __gm__ uint8_t *gradOut, __gm__ uint8_t *out, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
    __gm__ uint8_t *value, __gm__ uint8_t *actualQuery, __gm__ uint8_t *actualKv,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *runtimeConfig,
    __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained, __gm__ uint8_t *arena,
    __gm__ uint8_t *metadata, __gm__ uint8_t *requests, __gm__ uint8_t *transportTrace,
    __gm__ uint8_t *ownerGradient, __gm__ uint8_t *partials,
    __gm__ uint8_t *gradQuery, __gm__ uint8_t *gradKey, __gm__ uint8_t *gradValue,
    __gm__ uint8_t *gradQueryRope, __gm__ uint8_t *gradKeyRope,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Buffers buffers{query, key, value, indices, gradOut, out, maximum, sum, actualQuery, actualKv,
                  queryRope, keyRope, retained, gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope};
  CpGradTransport transport;
  transport.Init(arena, metadata, requests, transportTrace, partials, ownerGradient);
  Run(buffers, transport, runtimeConfig, groupTrace, tiling);
}
