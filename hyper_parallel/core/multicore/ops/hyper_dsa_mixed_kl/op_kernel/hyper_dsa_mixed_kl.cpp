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
#include "kernel_operator.h"
#include "arch22/sparse_lightning_indexer_grad_kl_loss_base.h"
#include "runtime/dsa_mixed_group.h"

using namespace AscendC;
using namespace DsaMixed;

extern "C" __global__ __aicore__ void hyper_dsa_mixed_kl(
  __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *indexQuery, __gm__ uint8_t *indexKey,
  __gm__ uint8_t *weight, __gm__ uint8_t *indices, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
  __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *actualQuery, __gm__ uint8_t *actualKey,
  __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace, __gm__ uint8_t *retained, __gm__ uint8_t *gradQuery,
  __gm__ uint8_t *gradKey, __gm__ uint8_t *gradWeight, __gm__ uint8_t *loss, __gm__ uint8_t *workspace,
  __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  REGISTER_TILING_DEFAULT(optiling::SparseLightningIndexerGradKLLossTilingData);
  GlobalTensor<int64_t> config, trace;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupTrace), 20 * kGroupWords);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 || groups == 0 || groups >= physicalCount ||
      physicalCount != 20 || config.GetValue(3) != 1) {
    return;
  }
  uint32_t group = GetBlockIdx();
  if ASCEND_IS_AIV {
    group /= 2;
  }
  if (group >= groups) {
    return;
  }
  GET_TILING_DATA_WITH_STRUCT(optiling::SparseLightningIndexerGradKLLossTilingData, data, tiling);
  uint32_t count = 0, checksum = 0;
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    {
      TPipe pipe;
      using TileType = SLIType<bfloat16_t, bfloat16_t, DTYPE_WEIGHT, bfloat16_t, SLITopKRange::TOPK_2k, SLILayout::TND,
                               SLILayout::TND, SLISparseMode::RightDown, true, false>;
      SparseLightningIndexerGradKLLossBase<TileType> tile;
      tile.Init(query, key, indexQuery, indexKey, weight, indices, maximum, sum, queryRope, keyRope, actualQuery,
                actualKey, gradQuery, gradKey, gradWeight, loss, retained, &data, &pipe, ticket, group, physicalCount,
                data.mixedPhase);
      tile.ProcessPhase();
    }
    ++count;
    checksum += ticket + 1;
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
}
