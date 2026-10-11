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
#include "runtime/dsa_tp_tile_sum.h"
extern "C" __global__ __aicore__ void hyper_dsa_tp_tile_sum(__gm__ uint8_t *partial, __gm__ uint8_t *arena,
                                                            __gm__ uint8_t *metadata, __gm__ uint8_t *summed,
                                                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<int64_t> data;
  data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiling), 4);
  const int64_t rounds = data.GetValue(0), slots = data.GetValue(1), width = data.GetValue(2);
  const uint32_t slot = AscendC::GetBlockIdx();
  if (slot >= slots) {
    return;
  }
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
  pipe.InitBuffer(storage, width * 8);
  auto tile = storage.Get<float>();
  auto scratch = tile[width];
  AscendC::GlobalTensor<float> input, output;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(partial));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(summed));
  DsaMixed::TpTileSum transport;
  transport.Init(arena, metadata, data.GetValue(3), slot, rounds, slots, width);
  for (int64_t round = 0; round < rounds; ++round) {
    const int64_t offset = (round * slots + slot) * width;
    AscendC::DataCopyPad(tile, input[offset], AscendC::DataCopyExtParams{1, static_cast<uint32_t>(width * 4), 0, 0, 0},
                         AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
    transport.Sum(tile, scratch, round);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::DataCopyPad(output[offset], tile,
                         AscendC::DataCopyExtParams{1, static_cast<uint32_t>(width * 4), 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
  }
  transport.Finish();
}
