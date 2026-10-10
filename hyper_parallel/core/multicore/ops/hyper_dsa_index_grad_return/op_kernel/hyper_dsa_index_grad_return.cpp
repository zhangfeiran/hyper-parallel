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
#include "runtime/dsa_index_grad_transport.h"
extern "C" __global__ __aicore__ void hyper_dsa_index_grad_return(__gm__ uint8_t *gradient, __gm__ uint8_t *arena,
                                                                  __gm__ uint8_t *metadata, __gm__ uint8_t *requests,
                                                                  __gm__ uint8_t *trace, __gm__ uint8_t *owner,
                                                                  __gm__ uint8_t *workspace, __gm__ uint8_t *tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (AscendC::GetBlockIdx() != 0) {
    return;
  }
  AscendC::GlobalTensor<int64_t> data;
  data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiling), 4);
  DsaMixed::IndexGradTransport transport;
  transport.Init(gradient, arena, metadata, requests, trace, owner);
  transport.Process(data.GetValue(0), data.GetValue(1), data.GetValue(2), data.GetValue(3));
}
