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
#include "runtime/dsa_selected_requests.h"
#include "runtime/dsa_tp_tile_sum.h"
#define HYPER_DSA_TP_LI
#define HYPER_DSA_TP_KL
#include "li/arch22/lightning_indexer_kernel.h"
#include "kl/arch22/sparse_lightning_indexer_grad_kl_loss_base.h"
#include "sfa_template_modes.h"                   // NOLINT(build/include_subdir)
#include "hyper_dsa_fused_training_tiling_key.h"  // NOLINT(build/include_subdir)
#include "sfa/arch22/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;   // NOLINT(build/namespaces)
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
  __gm__ uint8_t *klRetained;
  __gm__ uint8_t *klLengths;
  __gm__ uint8_t *klQueryLengths;
  __gm__ uint8_t *gradIndexQuery;
  __gm__ uint8_t *gradIndexKey;
  __gm__ uint8_t *gradWeight;
  __gm__ uint8_t *loss;
  __gm__ uint8_t *indices;
  __gm__ uint8_t *values;
  __gm__ uint8_t *attention;
  __gm__ uint8_t *maximum;
  __gm__ uint8_t *sum;
  __gm__ uint8_t *workspace;
  __gm__ uint8_t *tiling;
  __gm__ uint8_t *selectedRows;
  __gm__ uint8_t *membership;
  __gm__ uint8_t *selectedRequests;
  __gm__ uint8_t *selectedCounts;
  __gm__ uint8_t *queryLengths;
  __gm__ uint8_t *queryPositions;
  __gm__ uint8_t *queryIds;
  __gm__ uint8_t *tpLiArena;
  __gm__ uint8_t *tpLiMetadata;
  __gm__ uint8_t *tpKlArena;
  __gm__ uint8_t *tpKlMetadata;
};

template <bool FloatWeights>
__aicore__ inline uint32_t RunIndexer(const Buffers &buffers, const LITilingData *data, uint32_t ticket, uint32_t group,
                                      uint32_t count, bool merge, TpTileSum *tpTransport) {
  TPipe pipe;
  using TileType = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, false, LICommon::LI_LAYOUT::TND,
                                    LICommon::LI_LAYOUT::TND, FloatWeights>;
  LIKernel::LightningIndexerKernel<TileType> tile;
  tile.Init(buffers.indexQuery, buffers.indexKey, buffers.weights,
            buffers.queryLengths == nullptr ? buffers.lengths : buffers.queryLengths, buffers.lengths, nullptr,
            buffers.indices, buffers.values, buffers.retained, data, &pipe, ticket, count, group, count, merge,
            buffers.queryPositions);
  if ASCEND_IS_AIV {
    if (!merge && tpTransport != nullptr) {
      tile.SetTpTransport(tpTransport);
    }
  }
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
            buffers.queryLengths == nullptr ? buffers.lengths : buffers.queryLengths, buffers.lengths, nullptr,
            buffers.queryRope, buffers.keyRope, buffers.attention, buffers.maximum, buffers.sum,
            GetUserWorkspace(buffers.workspace), data,
            buffers.tiling + __builtin_offsetof(DsaFusedTrainingTilingData, sfa), &pipe, ticket, count, group, count,
            buffers.queryPositions);
  tile.Process();
}

template <bool FloatWeights>
__aicore__ inline void RunKl(const Buffers &buffers, const optiling::SparseLightningIndexerGradKLLossTilingData *data,
                             uint32_t ticket, uint32_t group, uint32_t count, uint32_t phase, TpTileSum *tpTransport,
                             uint32_t tpSize) {
  TPipe pipe;
  using WeightType = typename std::conditional<FloatWeights, float, bfloat16_t>::type;
  using TileType = SLIType<bfloat16_t, bfloat16_t, WeightType, bfloat16_t, SLITopKRange::TOPK_2k, SLILayout::TND,
                           SLILayout::TND, SLISparseMode::RightDown, true, false>;
  SparseLightningIndexerGradKLLossBase<TileType> tile;
  tile.Init(buffers.query, buffers.compressed, buffers.indexQuery, buffers.indexKey, buffers.weights, buffers.indices,
            buffers.maximum, buffers.sum, buffers.queryRope, buffers.keyRope,
            buffers.klQueryLengths == nullptr ? buffers.klLengths : buffers.klQueryLengths, buffers.klLengths,
            buffers.gradIndexQuery, buffers.gradIndexKey, buffers.gradWeight, buffers.loss, buffers.klRetained, data,
            &pipe, ticket, group, count, phase);
  tile.SetQueryPositions(buffers.queryPositions);
  if ASCEND_IS_AIV {
    if (phase == 1 && tpTransport != nullptr) {
      tile.SetTpTransport(tpTransport, tpSize);
    }
  }
  tile.ProcessPhase();
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

template <bool FloatWeights, bool Transport, bool Selected>
__aicore__ inline void RunComputePhase(const Buffers &buffers, const LITilingData *li,
                                       const SparseFlashAttentionTilingDataMla *sfa,
                                       const optiling::SparseLightningIndexerGradKLLossTilingData *kl,
                                       GlobalTensor<int64_t> &trace, uint32_t phase, uint32_t group, uint32_t groups,
                                       uint32_t physicalCount, SelectedRequests &requests, TpTileSum *tpLi,
                                       TpTileSum *tpKl, uint32_t tpSize) {
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
    const uint32_t mathPhase = Selected && phase >= 5 ? phase - 3 : phase;
    if constexpr (Selected) {
      if (phase == 2) {
        requests.Clear(ticket, physicalCount);
      }
      if (phase == 3) {
        requests.Mark(buffers.indices, ticket, physicalCount);
      }
    }
    if (Selected && phase >= 2 && phase <= 4) {
      // Membership and packing have their own releases before any SFA/KL reader.
    } else if (mathPhase < 2) {
      if (li->queryTokens != 0) {
        ldCount += RunIndexer<FloatWeights>(buffers, li, ticket, group, physicalCount, phase == 1, tpLi);
      }
    } else if (mathPhase == 2) {
      if (li->queryTokens != 0) {
        RunAttention(buffers, sfa, ticket, group, physicalCount);
      }
    } else if (li->queryTokens != 0) {
      RunKl<FloatWeights>(buffers, kl, ticket, group, physicalCount, mathPhase - 3, tpKl, tpSize);
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
  if ASCEND_IS_AIV {
    if (phase == 0 && tpLi != nullptr) {
      tpLi->Finish();
    }
    if (phase == 7 && tpKl != nullptr) {
      tpKl->Finish();
    }
  }
  ArriveAndWait(trace, group, groups, phase + 1);
}

template <bool Transport>
__aicore__ inline void PrepareTransport(CpTransport &transport, const Buffers &buffers, uint32_t group,
                                        uint32_t groups) {
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

template <bool Transport, bool Selected>
__aicore__ inline void ProgressPhase(CpTransport &transport, const Buffers &buffers, GlobalTensor<int64_t> &trace,
                                     uint32_t groups, uint32_t phase, SelectedRequests &requests,
                                     __gm__ uint8_t *metadata, uint32_t phaseCount) {
  if constexpr (Transport) {
    if ASCEND_IS_AIV {
      if (GetSubBlockIdx() == 0 && phase == 0 && !Selected) {
        transport.PullMain(buffers.compressed, buffers.keyRope, trace, groups);
      }
      if constexpr (Selected) {
        if (GetSubBlockIdx() == 0 && phase == 2) {
          requests.ResetCounts();
        }
        if (GetSubBlockIdx() == 0 && phase == 4) {
          GlobalTensor<int64_t> meta;
          meta.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 18);
          requests.Pack(meta.GetValue(3), meta.GetValue(4), meta.GetValue(17), meta.GetValue(2));
          transport.PullSelectedMain(buffers.compressed, buffers.keyRope, buffers.selectedRequests,
                                     buffers.selectedCounts);
        }
      }
      if (GetSubBlockIdx() == 0 && phase + 1 == phaseCount) {
        transport.WaitAcknowledged();
      }
    }
  }
}

template <bool FloatWeights, bool Transport, bool Selected>
__aicore__ inline void Run(const Buffers &buffers, __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace,
                           __gm__ uint8_t *arena, __gm__ uint8_t *metadata, __gm__ uint8_t *requests,
                           __gm__ uint8_t *transportTrace) {
  GlobalTensor<int64_t> config;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 || groups == 0 || groups >= physicalCount ||
      config.GetValue(3) != 1) {
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
  GET_TILING_DATA_WITH_STRUCT(DsaFusedTrainingTilingData, data, buffers.tiling);
  TpTileSum tpLi, tpKl;
  TpTileSum *liTransport = nullptr, *klTransport = nullptr;
  uint32_t tpSize = 1;
  if ASCEND_IS_AIV {
    if (group < groups && data.tpLiArenaBytes != 0) {
      GlobalTensor<int64_t> meta;
      meta.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buffers.tpLiMetadata), 24);
      tpSize = static_cast<uint32_t>(meta.GetValue(4));
      tpLi.Init(buffers.tpLiArena, buffers.tpLiMetadata, data.tpLiArenaBytes, group * 2 + GetSubBlockIdx(),
                meta.GetValue(10), groups * 2, 512);
      liTransport = &tpLi;
      if (data.tpKlArenaBytes != 0) {
        GlobalTensor<int64_t> klMeta;
        klMeta.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(buffers.tpKlMetadata), 24);
        if (klMeta.GetValue(4) != tpSize || klMeta.GetValue(3) != meta.GetValue(3) ||
            klMeta.GetValue(2) != meta.GetValue(2)) {
          AscendC::Trap();
        }
        for (uint32_t word = 11; word < 24; ++word) {
          if (klMeta.GetValue(word) != meta.GetValue(word)) {
            AscendC::Trap();
          }
        }
        tpKl.Init(buffers.tpKlArena, buffers.tpKlMetadata, data.tpKlArenaBytes, group * 2 + GetSubBlockIdx(),
                  klMeta.GetValue(10), groups * 2, 2048);
        klTransport = &tpKl;
      }
    }
  }
  SelectedRequests requestsBuilder;
  if constexpr (Selected) {
    GlobalTensor<int64_t> meta;
    meta.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 18);
    requestsBuilder.Init(buffers.selectedRows, buffers.membership, buffers.selectedRequests, buffers.selectedCounts,
                         meta.GetValue(15), data.li.queryTokens, buffers.queryIds);
  }
  const uint32_t phaseCount = Selected ? (data.withKl ? 9 : 6) : 6;
  for (uint32_t phase = 0; phase < phaseCount; ++phase) {
    GlobalTensor<int64_t> trace;
    auto *address = reinterpret_cast<__gm__ int64_t *>(groupTrace) + phase * physicalCount * kGroupWords;
    trace.SetGlobalBuffer(address, physicalCount * kGroupWords);
    if (group == groups) {
      ProgressPhase<Transport, Selected>(transport, buffers, trace, groups, phase, requestsBuilder, metadata,
                                         phaseCount);
      ClosePhase(trace, groups, phase + 1);
    } else {
      RunComputePhase<FloatWeights, Transport, Selected>(
        buffers, &data.li, &data.sfa,
        reinterpret_cast<const optiling::SparseLightningIndexerGradKLLossTilingData *>(data.kl), trace, phase, group,
        groups, physicalCount, requestsBuilder, liTransport, klTransport, tpSize);
    }
  }
}
}  // namespace

template <bool FLOAT_WEIGHTS, bool TRANSPORT, bool SELECTED>
__global__ __aicore__ void hyper_dsa_fused_training(
  __gm__ uint8_t *indexQuery, __gm__ uint8_t *indexKey, __gm__ uint8_t *query, __gm__ uint8_t *compressed,
  __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *weights, __gm__ uint8_t *lengths,
  __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained, __gm__ uint8_t *klRetained,
  __gm__ uint8_t *klLengths, __gm__ uint8_t *arena, __gm__ uint8_t *metadata, __gm__ uint8_t *requests,
  __gm__ uint8_t *transportTrace, __gm__ uint8_t *selectedRows, __gm__ uint8_t *membership,
  __gm__ uint8_t *selectedRequests, __gm__ uint8_t *selectedCounts, __gm__ uint8_t *queryLengths,
  __gm__ uint8_t *queryPositions, __gm__ uint8_t *queryIds, __gm__ uint8_t *klQueryLengths, __gm__ uint8_t *tpLiArena,
  __gm__ uint8_t *tpLiMetadata, __gm__ uint8_t *tpKlArena, __gm__ uint8_t *tpKlMetadata, __gm__ uint8_t *indices,
  __gm__ uint8_t *values, __gm__ uint8_t *attention, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
  __gm__ uint8_t *gradIndexQuery, __gm__ uint8_t *gradIndexKey, __gm__ uint8_t *gradWeight, __gm__ uint8_t *loss,
  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  Buffers buffers{indexQuery,     indexKey, query,        compressed,   queryRope,        keyRope,        weights,
                  lengths,        retained, klRetained,   klLengths,    klQueryLengths,   gradIndexQuery, gradIndexKey,
                  gradWeight,     loss,     indices,      values,       attention,        maximum,        sum,
                  workspace,      tiling,   selectedRows, membership,   selectedRequests, selectedCounts, queryLengths,
                  queryPositions, queryIds, tpLiArena,    tpLiMetadata, tpKlArena,        tpKlMetadata};
  Run<FLOAT_WEIGHTS, TRANSPORT, SELECTED>(buffers, runtimeConfig, groupTrace, arena, metadata, requests,
                                          transportTrace);
}
