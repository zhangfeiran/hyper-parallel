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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_GROUP_PARTIALS_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_GROUP_PARTIALS_H_
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
using namespace AscendC;  // NOLINT(build/namespaces)

namespace DsaMixed {
class GroupPartials {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *scratch, uint64_t keys, uint32_t groups) {
    scratch_ = scratch;
    keys_ = keys;
    groups_ = groups;
  }

  __aicore__ inline __gm__ uint8_t *Group(uint32_t group) const {
    // Slot zero remains the canonical interleaved dK/dV transport output.
    return scratch_ + (group + 1) * keys_ * 1088 * sizeof(float);
  }

  __aicore__ inline void Clear(uint32_t group, uint32_t member) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, kChunk * sizeof(float));
    auto zero = storage.Get<float>();
    Duplicate(zero, 0.0f, kChunk);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> target;
    target.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Group(group)));
    const uint64_t words = keys_ * 1088;
    for (uint64_t offset = member * kChunk; offset < words; offset += 2 * kChunk) {
      const uint32_t count = words - offset > kChunk ? kChunk : static_cast<uint32_t>(words - offset);
      const uint32_t bytes = count * static_cast<uint32_t>(sizeof(float));
      DataCopyPad(target[offset], zero, DataCopyExtParams{1, bytes, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
    }
  }

  __aicore__ inline void Reduce(__gm__ uint8_t *retained, uint64_t keyOffset, uint64_t valueOffset) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 2 * kChunk * sizeof(float));
    auto sum = storage.Get<float>();
    auto next = sum[kChunk];
    for (uint32_t kind = 0; kind < 2; ++kind) {
      const uint64_t words = keys_ * (kind == 0 ? 576 : 512);
      const uint64_t sourceOffset = kind == 0 ? 0 : keys_ * 576;
      GlobalTensor<float> output;
      output.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(retained + (kind == 0 ? keyOffset : valueOffset)));
      for (uint64_t offset = 0; offset < words; offset += kChunk) {
        const uint32_t count = words - offset > kChunk ? kChunk : static_cast<uint32_t>(words - offset);
        const uint32_t bytes = count * static_cast<uint32_t>(sizeof(float));
        const DataCopyExtParams copy{1, bytes, 0, 0, 0};
        for (uint32_t group = 0; group < groups_; ++group) {
          GlobalTensor<float> input;
          input.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Group(group)));
          DataCopyPad(group == 0 ? sum : next, input[sourceOffset + offset], copy,
                      DataCopyPadExtParams<float>{false, 0, 0, 0});
          PipeBarrier<PIPE_ALL>();
          if (group != 0) {
            Add(sum, sum, next, count);
            PipeBarrier<PIPE_ALL>();
          }
        }
        DataCopyPad(output[offset], sum, copy);
        PipeBarrier<PIPE_ALL>();
      }
    }
  }

 private:
  static constexpr uint32_t kChunk = 4096;
  __gm__ uint8_t *scratch_ = nullptr;
  uint64_t keys_ = 0;
  uint32_t groups_ = 0;
};
}  // namespace DsaMixed
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_GROUP_PARTIALS_H_
