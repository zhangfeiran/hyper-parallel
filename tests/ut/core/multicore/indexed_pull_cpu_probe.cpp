// Copyright 2026 Huawei Technologies Co., Ltd.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0.
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
constexpr int EVENT_ID0 = 0;
constexpr int PIPE_ALL = 0;
struct TaskDesc {
  uint32_t trigger_event;
};

namespace Probe {
struct Copy {
  void *destination;
  const void *source;
  uint64_t bytes;
};

struct Machine {
  uintptr_t ub_begin = 0;
  uint64_t ub_bytes = 16384;
  const uint16_t *source = nullptr;
  const int32_t *metadata = nullptr;
  uint16_t *destination = nullptr;
  int32_t *events = nullptr;
  int32_t *tiles = nullptr;
  int64_t hidden = 0;
  int64_t tile_stride = 0;
  const std::vector<int64_t> *groups = nullptr;
  std::vector<int64_t> copied;
  std::vector<Copy> reads;
  std::vector<Copy> stores;
  uint64_t payload_calls = 0;
  uint64_t metadata_calls = 0;

  static bool Overlap(uintptr_t left, uint64_t left_bytes, uintptr_t right, uint64_t right_bytes) {
    return left < right + right_bytes && right < left + left_bytes;
  }

  void Read(void *target, const void *input, uint64_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(target);
    assert(address >= ub_begin && address + bytes <= ub_begin + ub_bytes);
    for (const auto &pending : stores) {
      assert(!Overlap(address, bytes, reinterpret_cast<uintptr_t>(pending.source), pending.bytes));
    }
    const auto input_address = reinterpret_cast<uintptr_t>(input);
    const auto source_address = reinterpret_cast<uintptr_t>(source);
    if (input_address >= source_address && input_address < source_address + hidden * source_tokens * 2) {
      assert(input_address + bytes <= source_address + hidden * source_tokens * 2);
      ++payload_calls;
    } else {
      const auto raw = reinterpret_cast<uintptr_t>(input);
      const auto start = reinterpret_cast<uintptr_t>(metadata);
      assert(raw >= start && raw + bytes <= start + metadata_bytes);
      ++metadata_calls;
    }
    reads.push_back({target, input, bytes});
  }

  void Store(void *target, const void *input, uint64_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(input);
    assert(address >= ub_begin && address + bytes <= ub_begin + ub_bytes);
    assert(reads.empty());
    stores.push_back({target, input, bytes});
  }

  void DrainReads() {
    for (const auto &pending : reads) {
      std::memcpy(pending.destination, pending.source, pending.bytes);
    }
    reads.clear();
  }

  void DrainStores() {
    for (const auto &pending : stores) {
      auto offset = static_cast<uint16_t *>(pending.destination) - destination;
      assert(offset >= 0 &&
             offset + static_cast<int64_t>(pending.bytes / 2) <= static_cast<int64_t>(copied.size()) * hidden);
      const int64_t row = offset / hidden;
      assert(offset % hidden + static_cast<int64_t>(pending.bytes / 2) <= hidden);
      copied[row] += pending.bytes / 2;
      assert(copied[row] <= hidden);
      std::memcpy(pending.destination, pending.source, pending.bytes);
    }
    stores.clear();
  }

  int32_t Credit(int32_t *target, int32_t delta) {
    assert(reads.empty() && stores.empty());
    const bool is_tile =
      reinterpret_cast<uintptr_t>(target) >= reinterpret_cast<uintptr_t>(tiles) &&
      reinterpret_cast<uintptr_t>(target) < reinterpret_cast<uintptr_t>(tiles + groups->size() * tile_stride);
    const int64_t index = is_tile ? target - tiles : target - events;
    const int64_t expert = is_tile ? index / tile_stride : index;
    assert(expert >= 0 && expert < static_cast<int64_t>(groups->size()));
    const int64_t begin = expert == 0 ? 0 : groups->at(expert - 1);
    const int64_t first = is_tile ? begin + (index % tile_stride) * 128 : begin;
    const int64_t end = is_tile ? std::min(first + 128, groups->at(expert)) : groups->at(expert);
    int32_t complete = 0;
    for (int64_t row = first; row < end; ++row) {
      complete += copied[row] == hidden;
    }
    assert(delta > 0 && *target + delta <= complete);
    const int32_t previous = *target;
    *target += delta;
    return previous;
  }

  int64_t source_tokens = 0;
  uint64_t metadata_bytes = 0;
};

Machine *active = nullptr;
}  // namespace Probe

namespace AscendC {
enum class HardEvent { MTE2_S, MTE2_MTE3, MTE3_MTE2 };
template <HardEvent Event>
void SetFlag(int) {}
template <HardEvent Event>
void WaitFlag(int) {
  if constexpr (Event == HardEvent::MTE3_MTE2) {
    Probe::active->DrainStores();
  } else {
    Probe::active->DrainReads();
  }
}
template <typename T>
T AtomicAdd(T *target, T value) {
  return Probe::active->Credit(target, value);
}
inline void Trap() { throw std::runtime_error("Device contract trap"); }
}  // namespace AscendC

template <int Pipe>
void PipeBarrier() {
  Probe::active->DrainReads();
  Probe::active->DrainStores();
}

inline void aclshmemi_copy_gm2ub(void *target, const void *source, uint64_t bytes) {
  Probe::active->Read(target, source, bytes);
}
inline void aclshmemi_copy_ub2gm(void *target, const void *source, uint64_t bytes) {
  Probe::active->Store(target, source, bytes);
}

#include "production_token_pull.h"

constexpr int64_t kPeers = 4;
constexpr int64_t kExperts = 32;
constexpr int64_t kLocalExperts = kExperts / kPeers;

struct Fixture {
  int64_t hidden;
  int64_t tokens;
  uint32_t topk;
  int rank;
  std::array<std::vector<int32_t>, kPeers> metadata;
  std::array<std::vector<uint16_t>, kPeers> inputs;
  std::array<std::array<int64_t, kExperts>, kPeers> counts{};
  std::vector<int64_t> groups = std::vector<int64_t>(kLocalExperts);
  std::vector<int64_t> offsets = std::vector<int64_t>(kPeers * kLocalExperts);
  std::vector<uint16_t> destination;
  std::vector<uint16_t> expected;
  std::vector<int32_t> events = std::vector<int32_t>(kLocalExperts);
  int64_t stride;
  std::vector<int32_t> tiles;
  std::array<uint32_t, (16384 + 32) / 4> ub;
  Probe::Machine machine;

  Fixture(int64_t width, int64_t rows, uint32_t choices, int pattern, int owner, bool reverse_ordinals)
      : hidden(width),
        tokens(rows),
        topk(choices),
        rank(owner),
        stride((kPeers * rows + 127) / 128 + 1),
        tiles(kLocalExperts * stride) {
    BuildRoutes(pattern, reverse_ordinals);
    BuildDestination();
    ConfigureMachine();
  }

  void BuildRoutes(int pattern, bool reverse_ordinals);
  void BuildDestination();
  void ConfigureMachine();
  std::pair<uint64_t, uint64_t> ExpectedPeer(int64_t peer);
  template <bool InputTiles>
  void Verify();
};

void Fixture::BuildRoutes(int pattern, bool reverse_ordinals) {
  std::mt19937 random(211 + static_cast<uint32_t>(tokens));
  for (int64_t peer = 0; peer < kPeers; ++peer) {
    inputs[peer].resize(tokens * hidden);
    metadata[peer].resize(tokens * topk * 2);
    for (size_t value = 0; value < inputs[peer].size(); ++value) {
      inputs[peer][value] = static_cast<uint16_t>((value * 71 + peer * 199) % 65521);
    }
    for (int64_t token = 0; token < tokens; ++token) {
      std::array<int, kExperts> order{};
      std::iota(order.begin(), order.end(), 0);
      if (pattern == 1) {
        std::shuffle(order.begin(), order.end(), random);
      }
      for (uint32_t selected = 0; selected < topk; ++selected) {
        int expert = pattern == 1 ? order[selected] : (token * topk + selected) % kExperts;
        if (pattern == 2) {
          expert = selected;
        }
        const int64_t index = (token * topk + selected) * 2;
        metadata[peer][index] = expert;
        metadata[peer][index + 1] = counts[peer][expert]++;
      }
    }
    if (reverse_ordinals) {
      for (int64_t row = 0; row < tokens * topk; ++row) {
        auto *record = metadata[peer].data() + row * 2;
        record[1] = counts[peer][record[0]] - 1 - record[1];
      }
    }
  }
}

void Fixture::BuildDestination() {
  int64_t total = 0;
  for (int64_t local = 0; local < kLocalExperts; ++local) {
    for (int64_t peer = 0; peer < kPeers; ++peer) {
      offsets[peer * kLocalExperts + local] = total * hidden;
      total += counts[peer][rank * kLocalExperts + local];
    }
    groups[local] = total;
  }
  destination.resize(std::max(int64_t{1}, total * hidden), 0xFA17);
  expected = destination;
  machine.copied.resize(total);
}

void Fixture::ConfigureMachine() {
  ub.fill(0xABABABAB);
  machine.ub_begin = reinterpret_cast<uintptr_t>(ub.data());
  machine.destination = destination.data();
  machine.events = events.data();
  machine.tiles = tiles.data();
  machine.hidden = hidden;
  machine.tile_stride = stride;
  machine.groups = &groups;
  machine.source_tokens = tokens;
  machine.metadata_bytes = tokens * topk * 2 * sizeof(int32_t);
}

std::pair<uint64_t, uint64_t> Fixture::ExpectedPeer(int64_t peer) {
  uint64_t wanted = 0;
  uint64_t routes = 0;
  for (int64_t token = 0; token < tokens; ++token) {
    bool selected_peer = false;
    for (uint32_t selected = 0; selected < topk; ++selected) {
      const auto *record = metadata[peer].data() + (token * topk + selected) * 2;
      if (record[0] / kLocalExperts != rank) {
        continue;
      }
      selected_peer = true;
      ++routes;
      const int64_t output = offsets[peer * kLocalExperts + record[0] % kLocalExperts] + record[1] * hidden;
      std::copy_n(inputs[peer].data() + token * hidden, hidden, expected.data() + output);
    }
    wanted += selected_peer;
  }
  return {wanted, routes};
}

template <bool InputTiles>
void Fixture::Verify() {
  assert(destination == expected);
  for (int64_t local = 0; local < kLocalExperts; ++local) {
    const int64_t begin = local == 0 ? 0 : groups[local - 1];
    const int64_t rows = groups[local] - begin;
    if constexpr (InputTiles) {
      for (int64_t tile = 0; tile < stride; ++tile) {
        const int64_t expected_rows = std::max(int64_t{0}, std::min(int64_t{128}, rows - tile * 128));
        assert(tiles[local * stride + tile] == expected_rows);
      }
    } else {
      assert(events[local] == rows);
    }
  }
  for (const auto copied : machine.copied) {
    assert(copied == hidden);
  }
  for (size_t index = 16384 / 4; index < ub.size(); ++index) {
    assert(ub[index] == 0xABABABAB);
  }
}

template <bool InputTiles, bool Count = true>
uint64_t Run(int64_t hidden, int64_t tokens, uint32_t topk, int pattern, int rank, bool reverse_ordinals) {
  Fixture fixture(hidden, tokens, topk, pattern, rank, reverse_ordinals);
  auto &machine = fixture.machine;
  Probe::active = &machine;
  uint64_t total_wanted = 0;
  for (int64_t peer = 0; peer < kPeers; ++peer) {
    const auto [wanted, routes] = fixture.ExpectedPeer(peer);
    total_wanted += wanted;
    machine.source = fixture.inputs[peer].data();
    machine.metadata = fixture.metadata[peer].data();
    std::array<uint64_t, 6> statistics{};
    MulticoreRuntime::PullTokenBlocks<uint16_t, Count, InputTiles>(
      TaskDesc{0}, fixture.inputs[peer].data(), fixture.destination.data(), fixture.metadata[peer].data(),
      fixture.offsets.data(), fixture.events.data(), machine.ub_begin, 0, tokens, hidden, topk, rank, peer,
      kLocalExperts, tokens, statistics.data(), fixture.groups.data(), fixture.tiles.data(), fixture.stride);
    assert(machine.reads.empty() && machine.stores.empty());
    if constexpr (Count) {
      assert(statistics[0] == wanted * hidden * 2);
      assert(statistics[1] == routes * hidden * 2);
      assert(statistics[2] == machine.metadata_bytes);
      assert(statistics[3] == wanted);
    } else {
      assert(std::all_of(statistics.begin(), statistics.end(), [](uint64_t value) { return value == 0; }));
    }
  }
  fixture.Verify<InputTiles>();
  if (total_wanted == 0) {
    assert(machine.payload_calls == 0);
  }
  return machine.payload_calls / std::max(uint64_t{1}, total_wanted);
}

template <bool Count>
uint64_t RunRoutingSuite() {
  uint64_t cases = 0;
  for (int pattern = 0; pattern < 3; ++pattern) {
    for (int rank = 0; rank < 4; ++rank) {
      Run<false, Count>(5120, 129, 8, pattern, rank, true);
      Run<true, Count>(5120, 129, 8, pattern, rank, true);
      cases += 2;
    }
  }
  for (auto topk : {1U, 8U, 12U, 16U}) {
    for (auto tokens : {1, 31, 32, 33, 41, 42, 43, 63, 64, 65, 127, 128, 129, 257}) {
      Run<false, Count>(5120, tokens, topk, 1, 1, true);
      Run<true, Count>(5120, tokens, topk, 1, 1, true);
      cases += 2;
    }
  }
  return cases;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }
  const std::string suite = argv[1];
  uint64_t cases = 0;
  if (suite == "geometry") {
    const auto observed = Run<false>(5120, 129, 8, 0, 0, false);
    constexpr uint64_t expected_calls = __NPU_ARCH__ == 3510 ? 1 : 2;
    if (observed != expected_calls) {
      std::fprintf(stderr, "expected_payload_calls_per_wanted_row=%llu actual=%llu\n",
                   static_cast<unsigned long long>(expected_calls), static_cast<unsigned long long>(observed));
      return 1;
    }
    ++cases;
  } else if (suite == "fallback") {
    for (auto hidden : {256, 3072, 4096, 6144}) {
      const auto observed = Run<false>(hidden, 129, 8, 1, 2, true);
      assert(observed == static_cast<uint64_t>((hidden + 3071) / 3072));
      ++cases;
    }
    const auto wide_type = MulticoreRuntime::GetTokenPullGeometry<float>(5120);
    assert(wide_type.metadata_bytes == 8192 && wide_type.chunk_elements == 3072);
  } else if (suite == "routing") {
    cases = RunRoutingSuite<true>();
  } else if (suite == "normal") {
    cases = RunRoutingSuite<false>();
  } else {
    return 2;
  }
  std::printf("{\"suite\":\"%s\",\"cases\":%llu}\n", suite.c_str(), static_cast<unsigned long long>(cases));
}
