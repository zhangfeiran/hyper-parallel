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
// Both vendor tile closures compile against the generated composite tiling declarations.
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "runtime/dsa_mixed_group.h"
#include "runtime/dsa_cp_transport.h"
#include "li/arch22/lightning_indexer_kernel.h"
#include "sfa_template_modes.h"  // NOLINT(build/include_subdir)
#include "hyper_dsa_fused_forward_tiling_key.h"  // NOLINT(build/include_subdir)
#include "sfa/arch22/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;  // NOLINT(build/namespaces)
using namespace DsaMixed;  // NOLINT(build/namespaces)

namespace {
struct Buffers {
  __gm__ uint8_t *indexQuery;
  __gm__ uint8_t *indexKey;
  __gm__ uint8_t *query;
  __gm__ uint8_t *compressed;
  __gm__ uint8_t *queryRope;
  __gm__ uint8_t *keyRope;
  __gm__ uint8_t *weights;
  __gm__ uint8_t *lengths;
  __gm__ uint8_t *retained;
  __gm__ uint8_t *indices;
  __gm__ uint8_t *values;
  __gm__ uint8_t *attention;
  __gm__ uint8_t *maximum;
  __gm__ uint8_t *sum;
  __gm__ uint8_t *workspace;
  __gm__ uint8_t *tiling;
};

template<bool FloatWeights>
__aicore__ inline uint32_t RunIndexer(const Buffers &buffers, const LITilingData *data,
                                    uint32_t ticket, uint32_t group, uint32_t count, bool merge) {
  TPipe pipe;
  using TileType = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, false,
      LICommon::LI_LAYOUT::TND, LICommon::LI_LAYOUT::TND, FloatWeights>;
  LIKernel::LightningIndexerKernel<TileType> tile;
  tile.Init(buffers.indexQuery, buffers.indexKey, buffers.weights, buffers.lengths, buffers.lengths, nullptr,
            buffers.indices, buffers.values, buffers.retained, data, &pipe, ticket, count, group, count, merge);
  const uint32_t ld = tile.IsLdPartition() ? 1 : 0;
  tile.ProcessPhase(merge);
  return ld;
}

__aicore__ inline void RunAttention(const Buffers &buffers, const SparseFlashAttentionTilingDataMla *data,
                                  uint32_t ticket, uint32_t group, uint32_t count) {
  TPipe pipe;
  using TileType = SFAType<bfloat16_t, bfloat16_t, bfloat16_t, 0, SFA_LAYOUT::TND, SFA_LAYOUT::TND, 1>;
  SparseFlashAttentionMla<TileType> tile;
  tile.Init(buffers.query, buffers.compressed, buffers.compressed, buffers.indices,
            buffers.lengths, buffers.lengths, nullptr, buffers.queryRope, buffers.keyRope,
            buffers.attention, buffers.maximum, buffers.sum, GetUserWorkspace(buffers.workspace), data,
            buffers.tiling + __builtin_offsetof(DsaFusedForwardTilingData, sfa), &pipe, ticket, count, group, count);
  tile.Process();
}

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

template<bool FloatWeights, bool Transport>
__aicore__ inline void RunComputePhase(const Buffers &buffers, const LITilingData *li,
                                      const SparseFlashAttentionTilingDataMla *sfa,
                                      GlobalTensor<int64_t> &trace, uint32_t phase,
                                      uint32_t group, uint32_t groups, uint32_t physicalCount) {
  uint32_t count = 0;
  uint32_t checksum = 0;
  uint32_t ldCount = 0;
  uint32_t member = 0;
  if ASCEND_IS_AIV {
    member = 1 + GetSubBlockIdx();
  }
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    if constexpr (Transport) {
      if (phase == 0 && count == 0) {
        const uint32_t offset = group * kGroupWords + member * kMemberWords;
        PublishControl(trace, offset + 8, GetSystemCycle());
        PublishControl(trace, offset + 7, 1);
      }
    }
    if (phase < 2) {
      ldCount += RunIndexer<FloatWeights>(buffers, li, ticket, group, physicalCount, phase == 1);
    } else {
      RunAttention(buffers, sfa, ticket, group, physicalCount);
    }
    ++count;
    checksum += ticket + 1;
    trace.SetValue(group * kGroupWords + member * kMemberWords + 4, ldCount);
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
  if constexpr (Transport) {
    if (phase == 0) {
      PublishControl(trace, group * kGroupWords + member * kMemberWords + 9, GetSystemCycle());
    }
  }
  ArriveAndWait(trace, group, groups, phase + 1);
}

template<bool Transport>
__aicore__ inline void PrepareTransport(CpTransport &transport, const Buffers &buffers,
                                      uint32_t group, uint32_t groups) {
  if constexpr (Transport) {
    if (group == groups) {
      if ASCEND_IS_AIV {
        if (GetSubBlockIdx() == 0) {
          transport.PublishReady();
          transport.PullIndex(buffers.indexKey);
        }
      }
    } else {
      transport.WaitIndex();
    }
  }
}

template<bool Transport>
__aicore__ inline void ProgressPhase(CpTransport &transport, const Buffers &buffers,
                                    GlobalTensor<int64_t> &trace, uint32_t groups, uint32_t phase) {
      if constexpr (Transport) {
        if ASCEND_IS_AIV {
          if (GetSubBlockIdx() == 0 && phase == 0) {
            transport.PullMain(buffers.compressed, buffers.keyRope, trace, groups);
          }
          if (GetSubBlockIdx() == 0 && phase == 2) {
            transport.WaitAcknowledged();
          }
        }
      }
}

template<bool FloatWeights, bool Transport>
__aicore__ inline void Run(const Buffers &buffers, __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace,
                           __gm__ uint8_t *arena, __gm__ uint8_t *metadata,
                           __gm__ uint8_t *requests, __gm__ uint8_t *transportTrace) {
  GlobalTensor<int64_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 ||
      groups == 0 || groups >= physicalCount || config.GetValue(3) != 1) {
    return;
  }
  uint32_t group = GetBlockIdx();
  if ASCEND_IS_AIV {
    group /= 2;
  }
  if (group > groups) {
    return;
  }
  CpTransport transport;
  if constexpr (Transport) {
    transport.Init(arena, metadata, requests, transportTrace);
  }
  PrepareTransport<Transport>(transport, buffers, group, groups);
  GET_TILING_DATA_WITH_STRUCT(DsaFusedForwardTilingData, data, buffers.tiling);
  for (uint32_t phase = 0; phase < 3; ++phase) {
    GlobalTensor<int64_t> trace;
    auto *address = reinterpret_cast<__gm__ int64_t *>(groupTrace) + phase * physicalCount * kGroupWords;
    trace.SetGlobalBuffer(address, physicalCount * kGroupWords);
    if (group == groups) {
      ProgressPhase<Transport>(transport, buffers, trace, groups, phase);
      ClosePhase(trace, groups, phase + 1);
    } else {
      RunComputePhase<FloatWeights, Transport>(
          buffers, &data.li, &data.sfa, trace, phase, group, groups, physicalCount);
    }
  }
}
}  // namespace

template<bool FLOAT_WEIGHTS, bool TRANSPORT>
__global__ __aicore__ void hyper_dsa_fused_forward(
    __gm__ uint8_t *indexQuery, __gm__ uint8_t *indexKey, __gm__ uint8_t *query,
    __gm__ uint8_t *compressed, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *weights, __gm__ uint8_t *lengths, __gm__ uint8_t *runtimeConfig,
    __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained,
    __gm__ uint8_t *arena, __gm__ uint8_t *metadata, __gm__ uint8_t *requests, __gm__ uint8_t *transportTrace,
    __gm__ uint8_t *indices, __gm__ uint8_t *values,
    __gm__ uint8_t *attention, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Buffers buffers{indexQuery, indexKey, query, compressed, queryRope, keyRope, weights, lengths, retained,
                  indices, values, attention, maximum, sum, workspace, tiling};
  Run<FLOAT_WEIGHTS, TRANSPORT>(buffers, runtimeConfig, groupTrace, arena, metadata, requests, transportTrace);
}
