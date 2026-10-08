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
#include "runtime/dsa_mixed_group.h"
#include "runtime/dsa_cp_transport.h"
#include "sfa_template_modes.h"  // NOLINT(build/include_subdir)
#include "sfa/arch22/sparse_flash_attention_kernel_mla.h"

using namespace AscendC;  // NOLINT(build/namespaces)
using namespace DsaMixed;  // NOLINT(build/namespaces)

extern "C" __global__ __aicore__ void hyper_dsa_cp_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *compressed, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *indices, __gm__ uint8_t *lengths, __gm__ uint8_t *runtimeConfig, __gm__ uint8_t *groupTrace,
    __gm__ uint8_t *arena, __gm__ uint8_t *metadata, __gm__ uint8_t *requests, __gm__ uint8_t *transportTrace,
    __gm__ uint8_t *attention, __gm__ uint8_t *maximum, __gm__ uint8_t *sum,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GlobalTensor<int64_t> config, trace, transportEvidence;
  config.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(runtimeConfig), 4);
  trace.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupTrace), 20 * kGroupWords);
  transportEvidence.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(transportTrace), 32);
  const uint32_t groups = static_cast<uint32_t>(config.GetValue(2));
  const uint32_t physicalCount = GetBlockNum();
  if (config.GetValue(0) != kMixedMagic || config.GetValue(1) != 1 || groups == 0 ||
      groups >= physicalCount || physicalCount != 20 || config.GetValue(3) != 1) {
    return;
  }
  uint32_t group = GetBlockIdx();
  uint32_t member = 0;
  if ASCEND_IS_AIV {
    group /= 2;
    member = 1 + GetSubBlockIdx();
  }
  if (group > groups) {
    return;
  }
  CpTransport transport;
  transport.Init(arena, metadata, requests, transportTrace);
  if (group == groups) {
    TPipe pipe;
    if ASCEND_IS_AIV {
      if (GetSubBlockIdx() == 0) {
        transport.PublishReady();
        transport.PullMain<false>(compressed, keyRope, trace, groups);
        transport.WaitAcknowledged();
        CoordinatePhase(trace, groups, 1);
      } else {
        WaitPhase(trace, groups, 1);
      }
    } else {
      WaitPhase(trace, groups, 1);
    }
    PublishControl(trace, groups * kGroupWords + member * kMemberWords + kArrivalWord, 1);
    return;
  }
  while (ReadVisible(transportEvidence, 1) == 0) {}
  GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingDataMla, data, tiling);
  uint32_t count = 0;
  uint32_t checksum = 0;
  for (uint32_t logical = group; logical < physicalCount; logical += groups) {
    const uint32_t ticket = Dispatch(trace, group * kGroupWords, logical);
    {
      TPipe pipe;
      using TileType = SFAType<bfloat16_t, bfloat16_t, bfloat16_t, 0, SFA_LAYOUT::TND, SFA_LAYOUT::TND, 1>;
      SparseFlashAttentionMla<TileType> tile;
      tile.Init(query, compressed, compressed, indices, lengths, lengths, nullptr,
                queryRope, keyRope, attention, maximum, sum, GetUserWorkspace(workspace),
                &data, tiling, &pipe, ticket, physicalCount, group, physicalCount);
      tile.Process();
    }
    ++count;
    checksum += ticket + 1;
    Complete(trace, group * kGroupWords, count, checksum, ticket);
  }
  ArriveAndWait(trace, group, groups, 1);
}
