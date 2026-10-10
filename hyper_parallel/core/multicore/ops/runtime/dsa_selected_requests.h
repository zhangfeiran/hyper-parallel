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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_SELECTED_REQUESTS_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_SELECTED_REQUESTS_H_
#include "dsa_mixed_group.h"

namespace DsaMixed {
constexpr int64_t kSelectedRequestsMagic = 0x4850445341525131LL;

class SelectedRequests {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *rows, __gm__ uint8_t *membership, __gm__ uint8_t *requests,
                              __gm__ uint8_t *counts, uint32_t tokens) {
    rows_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(rows), tokens * 3);
    membership_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(membership), (tokens + 7) / 8 * 8);
    requests_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(requests), tokens * 4);
    counts_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(counts), 16);
    tokens_ = tokens;
  }

  __aicore__ inline void Clear(uint32_t logical, uint32_t partitions) {
#ifndef __DAV_C220_CUBE__
    TPipe pipe;
    TBuf<TPosition::VECOUT> storage;
    pipe.InitBuffer(storage, 32);
    auto zero = storage.Get<int32_t>();
    for (uint32_t word = 0; word < 8; ++word) {
      zero.SetValue(word, 0);
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    for (uint32_t block = logical * 2 + GetSubBlockIdx(); block < (tokens_ + 7) / 8; block += partitions * 2) {
      DataCopy(membership_[block * 8], zero, 8);
    }
    PipeBarrier<PIPE_ALL>();
#endif
  }

  __aicore__ inline void ResetCounts() {
#ifndef __DAV_C220_CUBE__
    for (uint32_t word = 0; word < 16; ++word) {
      PublishControl(counts_, word, 0);
    }
#endif
  }

  __aicore__ inline void Mark(__gm__ uint8_t *indices, uint32_t logical, uint32_t partitions) {
#ifndef __DAV_C220_CUBE__
    TPipe pipe;
    TBuf<TPosition::VECIN> indexStorage;
    TBuf<TPosition::VECOUT> deltaStorage;
    pipe.InitBuffer(indexStorage, 2048 * sizeof(int32_t));
    pipe.InitBuffer(deltaStorage, 32);
    auto selected = indexStorage.Get<int32_t>();
    auto delta = deltaStorage.Get<int32_t>();
    for (uint32_t word = 0; word < 8; ++word) {
      delta.SetValue(word, 0);
    }
    GlobalTensor<int32_t> input;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices), tokens_ * 2048);
    uint32_t previous = 0;
    for (uint32_t query = logical * 2 + GetSubBlockIdx(); query < tokens_; query += partitions * 2) {
      DataCopy(selected, input[query * 2048], 2048);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
      const int64_t start = rows_.GetValue(query * 3);
      if (start < 0 || start > query) {
        Trap();
      }
      for (uint32_t slot = 0; slot < 2048; ++slot) {
        const int32_t index = selected.GetValue(slot);
        if (index < 0 || index > static_cast<int64_t>(query) - start) {
          continue;
        }
        const uint32_t key = static_cast<uint32_t>(start + index);
        delta.SetValue(previous, 0);
        delta.SetValue(key % 8, 1);
        previous = key % 8;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        // Atomic aligned blocks prevent cache-line races between neighboring keys.
        SetAtomicAdd<int32_t>();
        DataCopy(membership_[key / 8 * 8], delta, 8);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        SetAtomicNone();
      }
    }
    PipeBarrier<PIPE_ALL>();
#endif
  }

  __aicore__ inline void Pack(int32_t rank, int32_t peers, int64_t sourceCapacity, int64_t epoch) {
#ifndef __DAV_C220_CUBE__
    TPipe pipe;
    PublishControl(counts_, 8, GetSystemCycle());
    int64_t runs = 0, unique = 0, occurrences = 0, remoteKeys = 0, remoteRuns = 0;
    int64_t previousOwner = -1, previousSource = -1, previousDestination = -1, runLength = 0;
    for (uint32_t key = 0; key < tokens_; ++key) {
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(membership_[key]);
      PipeBarrier<PIPE_ALL>();
      const int32_t count = membership_.GetValue(key);
      if (count <= 0) {
        continue;
      }
      ++unique;
      occurrences += count;
      const int64_t owner = rows_.GetValue(key * 3 + 1);
      const int64_t source = rows_.GetValue(key * 3 + 2);
      if (owner < 0 || owner >= peers || source < 0 || source >= sourceCapacity) {
        Trap();
      }
      if (owner != rank) {
        ++remoteKeys;
      }
      if (owner == previousOwner && source == previousSource + 1 && key == previousDestination + 1) {
        PublishControl(requests_, (runs - 1) * 4 + 3, ++runLength);
      } else {
        PublishControl(requests_, runs * 4, owner);
        PublishControl(requests_, runs * 4 + 1, source);
        PublishControl(requests_, runs * 4 + 2, key);
        PublishControl(requests_, runs * 4 + 3, 1);
        ++runs;
        runLength = 1;
        if (owner != rank) {
          ++remoteRuns;
        }
      }
      previousOwner = owner;
      previousSource = source;
      previousDestination = key;
    }
    PublishControl(counts_, 3, unique);
    PublishControl(counts_, 4, occurrences);
    PublishControl(counts_, 5, remoteKeys);
    PublishControl(counts_, 6, remoteRuns);
    PublishControl(counts_, 9, GetSystemCycle());
    PublishControl(counts_, 0, kSelectedRequestsMagic);
    PublishControl(counts_, 1, epoch);
    PublishControl(counts_, 2, runs);
#endif
  }

 private:
  GlobalTensor<int64_t> rows_, requests_, counts_;
  GlobalTensor<int32_t> membership_;
  uint32_t tokens_ = 0;
};
}  // namespace DsaMixed
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_SELECTED_REQUESTS_H_
