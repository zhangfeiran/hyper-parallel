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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_MIXED_GROUP_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_MIXED_GROUP_H_
// This header is compiled inside each assembled kernel's SDK include closure.
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
using namespace AscendC;  // NOLINT(build/namespaces)

namespace DsaMixed {
constexpr int64_t kMixedMagic = 0x48504453414D4958;
constexpr uint32_t kDispatchFlag = 13;
constexpr uint32_t kCompleteFlag = 14;
constexpr uint32_t kGroupWords = 64;
constexpr uint32_t kMemberWords = 16;
constexpr uint32_t kArrivalWord = 6;
constexpr uint32_t kReleaseWord = kMemberWords + 8;

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

__aicore__ inline int64_t ReadVisible(GlobalTensor<int64_t> &trace, uint32_t offset) {
  FlushLine(trace, offset);
  PipeBarrier<PIPE_ALL>();
  volatile __gm__ int64_t *address = trace.GetPhyAddr(offset);
  return *address;
}

__aicore__ inline void PublishControl(GlobalTensor<int64_t> &trace, uint32_t offset, int64_t value) {
  // Other physical groups observe this store inside the same kernel, before kernel completion.
  volatile __gm__ int64_t *address = trace.GetPhyAddr(offset);
  *address = value;
  FlushLine(trace, offset);
  PipeBarrier<PIPE_ALL>();
}

// A reserved vector remains runnable while every compute member waits for phase closure.
__aicore__ inline void CoordinatePhase(GlobalTensor<int64_t> &trace, uint32_t groups, int64_t epoch) {
  const uint32_t row = groups * kGroupWords + kMemberWords;
  PublishControl(trace, row + 5, epoch);
  for (uint32_t group = 0; group < groups; ++group) {
    for (uint32_t member = 0; member < 3; ++member) {
      const uint32_t offset = group * kGroupWords + member * kMemberWords + kArrivalWord;
      while (ReadVisible(trace, offset) != epoch) {}
    }
  }
  PublishControl(trace, row + 9, groups * 3);
  PublishControl(trace, row + 8, epoch);
}

__aicore__ inline void WaitPhase(GlobalTensor<int64_t> &trace, uint32_t groups, int64_t epoch) {
  const uint32_t release = groups * kGroupWords + kReleaseWord;
  while (ReadVisible(trace, release) != epoch) {}
  PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ArriveAndWait(GlobalTensor<int64_t> &trace, uint32_t group,
                                    uint32_t groups, int64_t epoch) {
  uint32_t member = 0;
  if ASCEND_IS_AIV {
    member = 1 + GetSubBlockIdx();
  }
  // Complete every tile's DMA before announcing that its retained partials can be consumed.
  PipeBarrier<PIPE_ALL>();
  const uint32_t offset = group * kGroupWords + member * kMemberWords + kArrivalWord;
  PublishControl(trace, offset, epoch);
  WaitPhase(trace, groups, epoch);
}
}  // namespace DsaMixed
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_MIXED_GROUP_H_
