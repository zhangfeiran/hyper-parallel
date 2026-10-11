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
#ifndef HYPER_PARALLEL_DSA_TP_TILE_SUM_H_
#define HYPER_PARALLEL_DSA_TP_TILE_SUM_H_
#include "kernel_operator.h"
#include "data_plane/rma.h"
#include "dsa_mixed_group.h"

namespace DsaMixed {
constexpr int64_t kTpTileSumMagic = 0x4850445341545031LL;
constexpr uint32_t kTpSlotWords = 80;

// Every runnable vector slot owns one bounded source tile. Readers ACK before
// reuse, so an arbitrarily long candidate scan does not grow symmetric storage.
class TpTileSum {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *arena, __gm__ uint8_t *metadata, int64_t arenaBytes, uint32_t slot,
                              int64_t rounds, int64_t slots, int64_t width) {
    arena_ = arena;
    meta_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 24);
    rank_ = static_cast<int32_t>(meta_.GetValue(3));
    size_ = static_cast<int32_t>(meta_.GetValue(4));
    slot_ = slot;
    width_ = static_cast<uint32_t>(width);
    rounds_ = rounds;
    epoch_ = meta_.GetValue(2);
#ifndef __DAV_C220_CUBE__
    auto *state = aclshmemi_get_state();
    if (meta_.GetValue(0) != kTpTileSumMagic || meta_.GetValue(1) != 1 || epoch_ <= 0 || epoch_ >= (1LL << 31) ||
        size_ <= 0 || size_ > 8 || rank_ < 0 || rank_ >= size_ || meta_.GetValue(5) != state->npes || slots <= 0 ||
        slots > 40 || slot >= slots || width <= 0 || width > 2048 || width % 8 != 0 || rounds <= 0 ||
        rounds >= (1LL << 32) || meta_.GetValue(6) != slots || meta_.GetValue(7) != width ||
        meta_.GetValue(10) != rounds || meta_.GetValue(8) < 0 || meta_.GetValue(9) < 0 || meta_.GetValue(8) % 64 != 0 ||
        meta_.GetValue(9) % 64 != 0 || meta_.GetValue(8) > meta_.GetValue(9) || meta_.GetValue(9) > arenaBytes ||
        slots * width * 4 > meta_.GetValue(9) - meta_.GetValue(8) ||
        slots * kTpSlotWords * 8 > arenaBytes - meta_.GetValue(9)) {
      AscendC::Trap();
    }
    for (int32_t peer = 0; peer < 8; ++peer) {
      const int64_t pe = meta_.GetValue(16 + peer);
      if (peer >= size_) {
        if (pe != -1) {
          AscendC::Trap();
        }
        continue;
      }
      if (pe < 0 || pe >= state->npes || (peer != rank_ && !(state->topo_list[pe] & ACLSHMEM_TRANSPORT_MTE))) {
        AscendC::Trap();
      }
      for (int32_t previous = 0; previous < peer; ++previous) {
        if (meta_.GetValue(16 + previous) == pe) {
          AscendC::Trap();
        }
      }
    }
    if (state->mype != meta_.GetValue(16 + rank_)) {
      AscendC::Trap();
    }
    auto local = Control(rank_);
    for (uint32_t word = 0; word < 8; ++word) {
      // Geometry, invocation, generation and caller's ordered-row signature.
      const uint32_t index = word < 3 ? (word == 0 ? 6 : word == 1 ? 7 : 10) : 8 + word;
      PublishControl(local, 8 + word, meta_.GetValue(index));
    }
#endif
  }

  __aicore__ inline void Sum(AscendC::LocalTensor<float> tile, AscendC::LocalTensor<float> scratch, int64_t round) {
#ifndef __DAV_C220_CUBE__
    if (round != completedRounds_ || round < 0 || round >= rounds_) {
      AscendC::Trap();
    }
    const int64_t sequence = epoch_ * (1LL << 32) + round + 1;
    if (round > 0) {
      WaitReaders(sequence - 1);
    }
    auto source = Source(rank_);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
    AscendC::DataCopyPad(source, tile, AscendC::DataCopyExtParams{1, width_ * 4, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID7);
    auto local = Control(rank_);
    PublishControl(local, 0, sequence);
    AscendC::Duplicate(tile, 0.0f, width_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      auto remote = Control(peer);
      while (ReadVisible(remote, 0) < sequence) {
      }
      if (ReadVisible(remote, 0) != sequence) {
        AscendC::Trap();
      }
      if (round == 0) {
        for (uint32_t word = 0; word < 8; ++word) {
          if (ReadVisible(remote, 8 + word) != ReadVisible(local, 8 + word)) {
            AscendC::Trap();
          }
        }
      }
      auto input = Source(peer);
      AscendC::DataCopyPad(scratch, input, AscendC::DataCopyExtParams{1, width_ * 4, 0, 0, 0},
                           AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
      AscendC::Add(tile, tile, scratch, width_);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
      PublishControl(remote, (2 + rank_) * 8, sequence);
    }
    ++completedRounds_;
#endif
  }

  __aicore__ inline void SumNext(AscendC::LocalTensor<float> tile, AscendC::LocalTensor<float> scratch) {
    Sum(tile, scratch, completedRounds_);
  }
  __aicore__ inline void Finish() {
    // LI partitions can consume fewer than the declared upper bound, including zero.
    if (completedRounds_ > 0) {
      WaitReaders(epoch_ * (1LL << 32) + completedRounds_);
    }
  }

 private:
  __aicore__ inline __gm__ uint8_t *Address(int32_t peer, int64_t offset) {
    auto *address = arena_ + offset;
    return peer == rank_ ? address
                         : reinterpret_cast<__gm__ uint8_t *>(hyper_parallel::multicore::shmem::data_plane::remote_ptr(
                             address, meta_.GetValue(16 + peer)));
  }
  __aicore__ inline GlobalTensor<int64_t> Control(int32_t peer) {
    GlobalTensor<int64_t> result;
    result.SetGlobalBuffer(
      reinterpret_cast<__gm__ int64_t *>(Address(peer, meta_.GetValue(9) + slot_ * kTpSlotWords * 8)), kTpSlotWords);
    return result;
  }
  __aicore__ inline GlobalTensor<float> Source(int32_t peer) {
    GlobalTensor<float> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Address(peer, meta_.GetValue(8) + slot_ * width_ * 4)),
                           width_);
    return result;
  }
  __aicore__ inline void WaitReaders(int64_t sequence) {
#ifndef __DAV_C220_CUBE__
    auto local = Control(rank_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      while (ReadVisible(local, (2 + peer) * 8) < sequence) {
      }
    }
#endif
  }
  __gm__ uint8_t *arena_ = nullptr;
  GlobalTensor<int64_t> meta_;
  int32_t rank_ = 0, size_ = 0;
  uint32_t slot_ = 0, width_ = 0;
  int64_t epoch_ = 0, rounds_ = 0, completedRounds_ = 0;
};
}  // namespace DsaMixed
#endif
