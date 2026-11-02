/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
 *
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
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = unsigned char *;
constexpr int32_t DATA_CACHE_LINE_SIZE = 64;
constexpr int32_t VECTOR_WORKER_STRIDE = 2;
namespace AscendC {
inline uint64_t GetSystemCycle() { static uint64_t cycles = 0; return ++cycles; }
inline void Trap() { throw std::runtime_error("invalid native schedule"); }
}  // namespace AscendC
namespace hyper_parallel::multicore::shmem::data_plane {
inline void *remote_ptr(void *address, int32_t) { return address; }
}  // namespace hyper_parallel::multicore::shmem::data_plane
struct GradientBandProducer { static constexpr int32_t kMaxBands = 64; };
struct CycleTraceRecorder {
  uint64_t Now() { return AscendC::GetSystemCycle(); }
  std::vector<uint32_t> descriptors;
  void Record(uint32_t desc, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t) {
    descriptors.push_back(desc);
  }
};
GM_ADDR blocked_flag = nullptr;
bool released = true;
bool blocked_seen = false;
uint32_t failed_reads = 0;
int32_t blocked_band = -1;
int64_t matrix_elements = 0;
int64_t band_elements = 0;
inline bool IsBandProducerReady(GM_ADDR address, int32_t) {
  if (address == blocked_flag && !released) {
    blocked_seen = true;
    if (++failed_reads < 20) {
      return false;
    }
    released = true;
  }
  return true;
}
inline void WaitBandProducer(GM_ADDR address, int32_t epoch) {
  while (!IsBandProducerReady(address, epoch)) {}
}

std::vector<float> output, guest_template;
std::array<std::vector<float>, 4> guests;
std::vector<std::vector<int32_t>> visits;
std::vector<std::tuple<int32_t, int64_t>> calls;

// Only device reads/writes are replaced. The production traversal is compiled below.
void AddReplicaGradient(GM_ADDR target, GM_ADDR source, int64_t elements,
                        int32_t peer, int32_t worker, int32_t workers) {
  auto destination = reinterpret_cast<float *>(target) - output.data();
  auto origin = reinterpret_cast<float *>(source) - guest_template.data();
  if (!released) {
    throw std::runtime_error("static traversal consumed a producer before publication");
  }
  calls.emplace_back(peer, origin);
  for (int64_t index = worker; index < elements; index += workers) {
    output[destination + index] += guests[peer][origin + index];
    visits[destination + index].push_back(peer);
  }
}

#include "native_schedule.h"

void Check(int32_t cores, int32_t bands, bool delay = false, bool profile = false) {
  constexpr int32_t home = 3;
  constexpr int32_t slots = 3;
  constexpr int32_t band_rows = 5;
  constexpr int32_t columns = 17;
  const int32_t rows = bands * band_rows - 1;
  const int64_t elements = rows * columns;
  matrix_elements = elements;
  band_elements = band_rows * columns;
  std::vector<uint64_t> config(18 + 9 * 4, 0);
  auto owned = config.data() + 18;
  const std::array<std::array<int32_t, 3>, 3> orders = {{{2, 0, 1}, {0, 2, 1}, {1, 0, 2}}};
  for (int32_t peer = 1; peer <= 3; ++peer) {
    for (int32_t slot = 0; slot < slots; ++slot) {
      const int32_t index = (peer - 1) * slots + slot;
      owned[4 * index] = peer;
      owned[4 * index + 1] = slot;
      owned[4 * index + 2] = orders[peer - 1][slot];
    }
    guests[peer].resize(slots * elements);
    for (int64_t index = 0; index < slots * elements; ++index) {
      guests[peer][index] = peer == 1 ? 1e20f : peer == 2 ? -1e20f : static_cast<float>(index % 19 + 1);
    }
  }
  output.assign(home * elements, 0.25f);
  guest_template.resize(slots * elements);
  visits.assign(home * elements, {});
  std::vector<float> expected = output;
  std::vector<std::vector<int32_t>> expected_visits(home * elements);
  for (int32_t index = 0; index < 9; ++index) {
    for (int64_t item = 0; item < elements; ++item) {
      const int64_t target = owned[4 * index + 2] * elements + item;
      expected[target] += guests[owned[4 * index]][owned[4 * index + 1] * elements + item];
      expected_visits[target].push_back(owned[4 * index]);
    }
  }
  std::vector<unsigned char> flags(home * bands * cores * DATA_CACHE_LINE_SIZE);
  config[0] = 7;
  config[2] = slots;
  config[3] = 1;
  config[4] = elements;
  config[5] = reinterpret_cast<uint64_t>(output.data());
  config[6] = reinterpret_cast<uint64_t>(guest_template.data());
  config[11] = 9;
  config[12] = config[13] = reinterpret_cast<uint64_t>(flags.data());
  config[14] = band_rows;
  config[15] = rows;
  config[16] = columns;
  config[17] = bands;
  for (int32_t worker = 0; worker < cores; ++worker) {
    calls.clear();
    const int32_t groups = cores < 4 ? cores : 4;
    blocked_band = worker % groups;
    blocked_flag = delay ? flags.data() + (blocked_band * cores + cores - 1) * DATA_CACHE_LINE_SIZE : nullptr;
    released = !delay;
    blocked_seen = false;
    failed_reads = 0;
    NativeSchedule kernel{worker * VECTOR_WORKER_STRIDE, cores, home};
    CycleTraceRecorder trace;
    kernel.ProcessGradientBands(config.data(), owned, 0x10008, trace, profile);
    if (!profile && !trace.descriptors.empty()) {
      throw std::runtime_error("disabled profiling recorded events");
    }
    const int32_t max_events = 5 * 9 * ((bands + groups - 1) / groups);
    if (trace.descriptors.size() > static_cast<size_t>(max_events)) {
      throw std::runtime_error("state transitions exceeded the per-head bound");
    }
    if (delay && profile && worker < groups &&
        std::find(trace.descriptors.begin(), trace.descriptors.end(), 0x11005) == trace.descriptors.end()) {
      throw std::runtime_error("independent ready owner was not observed while the head remained blocked");
    }
    int64_t previous_band = -1;
    int64_t previous_edge = -1;
    for (const auto &[peer, origin] : calls) {
      const int64_t band = (origin % elements) / band_elements;
      const int64_t edge = (peer - 1) * slots + origin / elements;
      if (band < previous_band || (band == previous_band && edge < previous_edge)) {
        throw std::runtime_error("static traversal changed band or owned-edge order");
      }
      previous_edge = edge;
      previous_band = band;
    }
    if (delay && !blocked_seen) {
      throw std::runtime_error("delayed producer did not block the static head");
    }
  }
  if (visits != expected_visits || std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) {
    throw std::runtime_error("native traversal changed coverage or FP32 peer order");
  }
}

int main() {
  for (bool profile : {false, true}) {
  for (int32_t cores : {1, 2, 3, 20, 24, 32}) {
      for (int32_t bands : {1, 2, 3, 5, 23, 64}) {
        Check(cores, bands, false, profile);
      }
    }
    Check(24, 23, true, profile);
  }
  std::cout << "Verified 74 static native schedules with exact FP32 peer order and delayed producer observations.\n";
}
