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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_GRAD_TRANSPORT_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_GRAD_TRANSPORT_H_
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "data_plane/rma.h"
#include "dsa_mixed_group.h"  // NOLINT(build/include_subdir)

namespace DsaMixed {
class CpGradTransport {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *arena, __gm__ uint8_t *metadata, __gm__ uint8_t *requests,
                              __gm__ uint8_t *trace, __gm__ uint8_t *partials, __gm__ uint8_t *ownerGradient) {
    arena_ = arena;
    meta_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 18);
    requests_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(requests));
    trace_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace), 32);
    partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(partials));
    owner_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ownerGradient));
    epoch_ = meta_.GetValue(2);
    rank_ = static_cast<int32_t>(meta_.GetValue(3));
    size_ = static_cast<int32_t>(meta_.GetValue(4));
    sparse_ = meta_.GetValue(1) == 2;
  }

  __aicore__ inline void Prepare() {
#ifndef __DAV_C220_CUBE__
    auto *state = aclshmemi_get_state();
    if (meta_.GetValue(0) != 0x4850445341434731LL || (!sparse_ && meta_.GetValue(1) != 1) || state->mype != rank_ ||
        state->npes != size_ || epoch_ <= 0 ||
        (sparse_ && (size_ <= 0 || size_ > 8 || meta_.GetValue(11) <= 0 || meta_.GetValue(12) < 0 ||
                     meta_.GetValue(12) > meta_.GetValue(11)))) {
      AscendC::Trap();
    }
    if (sparse_) {
      ClearSparseInbox();
    }
    auto local = Signal(rank_);
    for (uint32_t word = 0; word < 5; ++word) {
      PublishControl(local, word, meta_.GetValue(5 + word));
    }
    PublishControl(local, 5, meta_.GetValue(1));
    PublishControl(local, 24, epoch_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      if (peer != rank_ && !(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) {
        AscendC::Trap();
      }
      auto remote = Signal(peer);
      while (ReadVisible(remote, 24) < epoch_) {
      }
      for (uint32_t word = 0; word < 5; ++word) {
        if (ReadVisible(remote, word) != meta_.GetValue(5 + word)) {
          AscendC::Trap();
        }
      }
      if (ReadVisible(remote, 5) != meta_.GetValue(1)) {
        AscendC::Trap();
      }
    }
    PublishControl(trace_, 0, epoch_);
#endif
  }

  __aicore__ inline void Progress(__gm__ uint8_t *retained, int64_t keyOffset, int64_t valueOffset) {
#ifndef __DAV_C220_CUBE__
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 12288);
    auto buffer = storage.Get<float>();
    GlobalTensor<float> key, value;
    key.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(retained + keyOffset));
    value.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(retained + valueOffset));
    PublishControl(trace_, 20, keyOffset);
    PublishControl(trace_, 21, valueOffset);
    PublishControl(trace_, 8, AscendC::GetSystemCycle());
    CopyPartials(key, value, buffer);
    Push(buffer);
    PublishControl(trace_, 9, AscendC::GetSystemCycle());
    auto local = Signal(rank_);
    PublishControl(local, 8, epoch_);
    PublishControl(trace_, 1, epoch_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      auto remote = Signal(peer);
      while (ReadVisible(remote, 8) < epoch_) {
      }
    }
    PublishControl(trace_, 2, size_);
    PublishControl(trace_, 10, AscendC::GetSystemCycle());
    Reduce(buffer);
    PublishControl(trace_, 11, AscendC::GetSystemCycle());
    PublishControl(local, 16, epoch_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      auto remote = Signal(peer);
      while (ReadVisible(remote, 16) < epoch_) {
      }
    }
    PublishControl(trace_, 3, size_);
#endif
  }

 private:
  __aicore__ inline GlobalTensor<int64_t> Counter(int32_t peer, int32_t sender) {
    auto *address = arena_ + meta_.GetValue(10) + sender * meta_.GetValue(11) * 704 * 4 + 576 * 4;
    if (peer != rank_) {
      address =
        reinterpret_cast<__gm__ uint8_t *>(hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peer));
    }
    GlobalTensor<int64_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address), 1);
    return counter;
  }

  __aicore__ inline void ClearSparseInbox() {
#ifndef __DAV_C220_CUBE__
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 576 * sizeof(float));
    auto zero = storage.Get<float>();
    AscendC::Duplicate(zero, 0.0f, 576);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (int32_t sender = 0; sender < size_; ++sender) {
      auto inbox = Inbox(rank_, sender);
      for (int64_t row = 0; row < meta_.GetValue(12); ++row) {
        Store(inbox, row * 704, zero, 576);
      }
      auto counter = Counter(rank_, sender);
      PublishControl(counter, 0, 0);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
  }

  __aicore__ inline GlobalTensor<int64_t> Signal(int32_t peer) {
    auto *address = arena_ + meta_.GetValue(13);
    if (peer != rank_) {
      address =
        reinterpret_cast<__gm__ uint8_t *>(hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peer));
    }
    GlobalTensor<int64_t> signal;
    signal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address), 32);
    return signal;
  }

  __aicore__ inline void Load(AscendC::LocalTensor<float> destination, GlobalTensor<float> &source, int64_t offset,
                              uint32_t count) {
#ifndef __DAV_C220_CUBE__
    AscendC::DataCopyExtParams params{1, count * 4, 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(destination, source[offset], params, padding);
#endif
  }

  __aicore__ inline void Store(GlobalTensor<float> &destination, int64_t offset, AscendC::LocalTensor<float> source,
                               uint32_t count) {
#ifndef __DAV_C220_CUBE__
    AscendC::DataCopyExtParams params{1, count * 4, 0, 0, 0};
    AscendC::DataCopyPad(destination[offset], source, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
#endif
  }

  __aicore__ inline void CopyPartials(GlobalTensor<float> &key, GlobalTensor<float> &value,
                                      AscendC::LocalTensor<float> &buffer) {
#ifndef __DAV_C220_CUBE__
    for (int64_t token = 0; token < meta_.GetValue(15); ++token) {
      Load(buffer, key, token * 576, 576);
      Load(buffer[576], value, token * 512, 512);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
      Store(partials_, token * 1088, buffer, 1088);
    }
#endif
  }

  __aicore__ inline GlobalTensor<float> Inbox(int32_t peer, int32_t sender) {
    auto *address = arena_ + meta_.GetValue(10) + sender * meta_.GetValue(11) * 704 * 4;
    if (peer != rank_) {
      address =
        reinterpret_cast<__gm__ uint8_t *>(hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peer));
    }
    GlobalTensor<float> inbox;
    inbox.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
    return inbox;
  }

  __aicore__ inline void Push(AscendC::LocalTensor<float> &buffer) {
#ifndef __DAV_C220_CUBE__
    auto merged = buffer[1088];
    int64_t remote = 0, sentRows = 0;
    int64_t counts[8] = {};
    for (int64_t run = 0; run < meta_.GetValue(14); ++run) {
      const int32_t peer = static_cast<int32_t>(requests_.GetValue(run * 4));
      auto inbox = Inbox(peer, rank_);
      for (int64_t row = 0; row < requests_.GetValue(run * 4 + 3); ++row) {
        const int64_t token = requests_.GetValue(run * 4 + 2) + row;
        Load(buffer, partials_, token * 1088, 1088);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Add(merged, buffer, buffer[576], 512);
        AscendC::DataCopy(merged[512], buffer[512], 64);
        AscendC::Duplicate(merged[576], 0.0f, 128);
        if (sparse_) {
          AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
          bool nonzero = false;
          for (uint32_t column = 0; column < 576; ++column) {
            if (merged.GetValue(column) != 0.0f) {
              nonzero = true;
              break;
            }
          }
          if (!nonzero) {
            continue;
          }
          ++counts[peer];
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        Store(inbox, (requests_.GetValue(run * 4 + 1) + row) * 704, merged, 704);
        ++sentRows;
        if (peer != rank_) {
          remote += 704 * 4;
        }
      }
    }
    if (sparse_) {
      for (int32_t peer = 0; peer < size_; ++peer) {
        auto counter = Counter(peer, rank_);
        PublishControl(counter, 0, counts[peer]);
      }
      PublishControl(trace_, 12, sentRows);
      PublishControl(trace_, 13, meta_.GetValue(15) - sentRows);
      PublishControl(trace_, 16, (size_ - 1) * sizeof(int64_t));
    }
    PublishControl(trace_, 4, sentRows * 704 * 4);
    PublishControl(trace_, 5, remote);
#endif
  }

  __aicore__ inline void Reduce(AscendC::LocalTensor<float> &buffer) {
#ifndef __DAV_C220_CUBE__
    auto sum = buffer[704];
    for (int64_t row = 0; row < meta_.GetValue(12); ++row) {
      for (int32_t sender = 0; sender < size_; ++sender) {
        auto inbox = Inbox(rank_, sender);
        Load(buffer, inbox, row * 704, 576);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (sender == 0) {
          AscendC::DataCopy(sum, buffer, 576);
        } else {
          AscendC::Add(sum, sum, buffer, 576);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      Store(owner_, row * 576, sum, 576);
    }
    int64_t received = size_ * meta_.GetValue(12), remote = (size_ - 1) * meta_.GetValue(12);
    if (sparse_) {
      received = 0;
      remote = 0;
      for (int32_t sender = 0; sender < size_; ++sender) {
        auto counter = Counter(rank_, sender);
        const auto rows = ReadVisible(counter, 0);
        if (rows < 0 || rows > meta_.GetValue(12)) {
          AscendC::Trap();
        }
        received += rows;
        if (sender != rank_) {
          remote += rows;
        }
      }
      PublishControl(trace_, 14, received);
      PublishControl(trace_, 15, remote);
      PublishControl(trace_, 17, (size_ - 1) * sizeof(int64_t));
      PublishControl(trace_, 18, size_ * meta_.GetValue(12) * 576 * 4);
    }
    PublishControl(trace_, 6, received * 704 * 4);
    PublishControl(trace_, 7, remote * 704 * 4);
#endif
  }
  __gm__ uint8_t *arena_ = nullptr;
  GlobalTensor<int64_t> meta_, requests_, trace_;
  GlobalTensor<float> partials_, owner_;
  int64_t epoch_ = 0;
  int32_t rank_ = 0, size_ = 0;
  bool sparse_ = false;
};
}  // namespace DsaMixed
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_GRAD_TRANSPORT_H_
