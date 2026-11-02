/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char *;
using event_t = int;
constexpr int PIPE_ALL = 0;
uint64_t clock_cycle = 0, fence_cycle = 0;
namespace AscendC {
enum class HardEvent { FIX_MTE2 };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
inline void Trap() { throw std::runtime_error("invalid producer state"); }
template <int> void PipeBarrier() { ++clock_cycle; }
template <HardEvent> void SetFlag(event_t) { ++clock_cycle; }
template <HardEvent> void WaitFlag(event_t) { fence_cycle = ++clock_cycle; }
template <class T> struct GlobalTensor {
  T *data;
  void SetGlobalBuffer(T *value, uint32_t) { data = value; }
  void SetValue(uint32_t index, T value) { data[index] = value; ++clock_cycle; }
};
template <class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) { ++clock_cycle; }
}  // namespace AscendC
struct Pipe { int FetchEventID(AscendC::HardEvent) { return 1; } } pipe_state;
Pipe *GetTPipePtr() { return &pipe_state; }
struct TraceRecord { uint32_t desc; uint64_t start, end; };
struct CycleTraceRecorder {
  std::vector<TraceRecord> records;
  uint64_t Now() { return ++clock_cycle; }
  void Record(uint32_t desc, uint32_t, uint32_t, uint32_t, uint64_t start, uint64_t end) {
    records.push_back({desc, start, end});
  }
};
#include "native_producer.h"
int main() {
  for (bool profile : {false, true}) {
    std::array<int32_t, 32> flags = {};
    GradientBandProducer producer;
    CycleTraceRecorder trace;
    producer.Set(reinterpret_cast<GM_ADDR>(flags.data()), 7, 4);
    producer.SetProfile(profile ? &trace : nullptr, 1, 2, 6);
    producer.Begin(8, 0, 1);
    producer.Expect(0, 4);
    producer.PublishEmpty();
    if (flags[0] != 0 || flags[16] != 7) {
      throw std::runtime_error("empty contribution changed unfinished data readiness");
    }
    producer.Retire(0, 4);
    if (flags[0] != 7 || !fence_cycle) {
      throw std::runtime_error("nonempty output was not fenced before publication");
    }
    if (profile) {
      if (trace.records.size() != 2 || trace.records[0].desc != 0x11109 ||
          trace.records[1].desc != 0x11101 || trace.records[1].start <= fence_cycle ||
          trace.records[1].end <= trace.records[1].start) {
        throw std::runtime_error("readable/publication observations violate the fence or event contract");
      }
    } else if (!trace.records.empty()) {
      throw std::runtime_error("disabled publisher emitted observations");
    }
  }
  std::cout << "Verified producer fence, empty contribution, and disabled observations\n";
}
