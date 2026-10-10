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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_INDEX_GRAD_TRANSPORT_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_INDEX_GRAD_TRANSPORT_H_
#include "kernel_operator.h"
#include "data_plane/rma.h"
#include "dsa_cp_peer_mapping.h"
#include "dsa_mixed_group.h"

namespace DsaMixed {
class IndexGradTransport {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *gradient, __gm__ uint8_t *arena, __gm__ uint8_t *metadata,
                              __gm__ uint8_t *requests, __gm__ uint8_t *trace, __gm__ uint8_t *owner) {
    arena_ = arena;
    meta_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 18);
    requests_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(requests));
    trace_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace), 32);
    gradient_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradient));
    owner_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(owner));
    epoch_ = meta_.GetValue(2);
    rank_ = static_cast<int32_t>(meta_.GetValue(3));
    size_ = static_cast<int32_t>(meta_.GetValue(4));
    peers_.Init(metadata, meta_.GetValue(1) == 2, rank_, size_);
  }

  __aicore__ inline void Process(int64_t tokens, int64_t capacity, int64_t arenaBytes, int64_t runs) {
#ifndef __DAV_C220_CUBE__
    if (meta_.GetValue(0) != 0x4850445341494731LL || (meta_.GetValue(1) != 1 && meta_.GetValue(1) != 2) || size_ <= 0 ||
        size_ > 8 || epoch_ <= 0 || meta_.GetValue(11) != capacity || capacity <= 0 || meta_.GetValue(12) < 0 ||
        meta_.GetValue(12) > capacity || meta_.GetValue(15) != tokens || meta_.GetValue(14) != runs ||
        meta_.GetValue(10) < 0 || meta_.GetValue(13) < 0 ||
        meta_.GetValue(10) + size_ * capacity * 704 * 4 > arenaBytes || meta_.GetValue(13) + 32 * 8 > arenaBytes) {
      AscendC::Trap();
    }
    peers_.Validate();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 1024);
    auto buffer = storage.Get<float>();
    Prepare(buffer);
    PublishControl(trace_, 8, AscendC::GetSystemCycle());
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
  __aicore__ inline __gm__ uint8_t *Address(int32_t peer, int32_t sender) {
    auto *address = arena_ + meta_.GetValue(10) + sender * meta_.GetValue(11) * 704 * 4;
    return peer == rank_ ? address
                         : reinterpret_cast<__gm__ uint8_t *>(
                             hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peers_.RootPe(peer)));
  }
  __aicore__ inline GlobalTensor<float> Inbox(int32_t peer, int32_t sender) {
    GlobalTensor<float> inbox;
    inbox.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(Address(peer, sender) + 576 * 4));
    return inbox;
  }
  __aicore__ inline GlobalTensor<int64_t> Counter(int32_t peer, int32_t sender) {
    GlobalTensor<int64_t> counter;
    counter.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(Address(peer, sender)), 1);
    return counter;
  }
  __aicore__ inline GlobalTensor<int64_t> Signal(int32_t peer) {
    auto *address = arena_ + meta_.GetValue(13);
    if (peer != rank_) {
      address = reinterpret_cast<__gm__ uint8_t *>(
        hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peers_.RootPe(peer)));
    }
    GlobalTensor<int64_t> signal;
    signal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address), 32);
    return signal;
  }
  __aicore__ inline void Load(AscendC::LocalTensor<float> destination, GlobalTensor<float> &source, int64_t offset) {
#ifndef __DAV_C220_CUBE__
    AscendC::DataCopyPad(destination, source[offset], AscendC::DataCopyExtParams{1, 512, 0, 0, 0},
                         AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
#endif
  }
  __aicore__ inline void Store(GlobalTensor<float> &destination, int64_t offset, AscendC::LocalTensor<float> source) {
#ifndef __DAV_C220_CUBE__
    AscendC::DataCopyPad(destination[offset], source, AscendC::DataCopyExtParams{1, 512, 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
#endif
  }
  __aicore__ inline void Prepare(AscendC::LocalTensor<float> buffer) {
#ifndef __DAV_C220_CUBE__
    AscendC::Duplicate(buffer, 0.0f, 128);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    for (int32_t sender = 0; sender < size_; ++sender) {
      auto inbox = Inbox(rank_, sender);
      for (int64_t row = 0; row < meta_.GetValue(12); ++row) {
        Store(inbox, row * 704, buffer);
      }
      auto counter = Counter(rank_, sender);
      PublishControl(counter, 0, 0);
    }
    auto local = Signal(rank_);
    for (uint32_t word = 0; word < 5; ++word) {
      PublishControl(local, word, meta_.GetValue(5 + word));
    }
    PublishControl(local, 5, meta_.GetValue(0));
    PublishControl(local, 24, epoch_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      if (peer != rank_ && !(aclshmemi_get_state()->topo_list[peers_.RootPe(peer)] & ACLSHMEM_TRANSPORT_MTE)) {
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
      if (ReadVisible(remote, 5) != meta_.GetValue(0)) {
        AscendC::Trap();
      }
    }
    PublishControl(trace_, 0, epoch_);
#endif
  }
  __aicore__ inline void Push(AscendC::LocalTensor<float> buffer) {
#ifndef __DAV_C220_CUBE__
    int64_t counts[8] = {}, sent = 0, remoteRows = 0;
    for (int64_t run = 0; run < meta_.GetValue(14); ++run) {
      const auto peer = static_cast<int32_t>(requests_.GetValue(run * 4));
      if (peer < 0 || peer >= size_) {
        AscendC::Trap();
      }
      auto inbox = Inbox(peer, rank_);
      for (int64_t row = 0; row < requests_.GetValue(run * 4 + 3); ++row) {
        const int64_t token = requests_.GetValue(run * 4 + 2) + row;
        const int64_t offset = requests_.GetValue(run * 4 + 1) + row;
        if (token < 0 || token >= meta_.GetValue(15) || offset < 0 || offset >= meta_.GetValue(11)) {
          AscendC::Trap();
        }
        Load(buffer, gradient_, token * 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        auto bits = buffer.ReinterpretCast<uint32_t>();
        bool nonzero = false;
        for (uint32_t column = 0; column < 128; ++column) {
          if ((bits.GetValue(column) & 0x7fffffffU) != 0) {
            nonzero = true;
            break;
          }
        }
        if (!nonzero) {
          continue;
        }
        Store(inbox, offset * 704, buffer);
        ++counts[peer];
        ++sent;
        if (peer != rank_) {
          ++remoteRows;
        }
      }
    }
    for (int32_t peer = 0; peer < size_; ++peer) {
      auto counter = Counter(peer, rank_);
      PublishControl(counter, 0, counts[peer]);
    }
    PublishControl(trace_, 4, sent * 512);
    PublishControl(trace_, 5, remoteRows * 512);
    PublishControl(trace_, 12, sent);
    PublishControl(trace_, 13, meta_.GetValue(15) - sent);
    PublishControl(trace_, 16, (size_ - 1) * 8);
#endif
  }
  __aicore__ inline void Reduce(AscendC::LocalTensor<float> buffer) {
#ifndef __DAV_C220_CUBE__
    auto sum = buffer[128];
    for (int64_t row = 0; row < meta_.GetValue(12); ++row) {
      for (int32_t sender = 0; sender < size_; ++sender) {
        auto inbox = Inbox(rank_, sender);
        Load(buffer, inbox, row * 704);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (sender == 0) {
          AscendC::DataCopy(sum, buffer, 128);
        } else {
          AscendC::Add(sum, sum, buffer, 128);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      Store(owner_, row * 128, sum);
    }
    int64_t received = 0, remoteRows = 0;
    for (int32_t sender = 0; sender < size_; ++sender) {
      auto counter = Counter(rank_, sender);
      const auto rows = ReadVisible(counter, 0);
      if (rows < 0 || rows > meta_.GetValue(12)) {
        AscendC::Trap();
      }
      received += rows;
      if (sender != rank_) {
        remoteRows += rows;
      }
    }
    PublishControl(trace_, 6, received * 512);
    PublishControl(trace_, 7, remoteRows * 512);
    PublishControl(trace_, 14, received);
    PublishControl(trace_, 15, remoteRows);
    PublishControl(trace_, 17, (size_ - 1) * 8);
    PublishControl(trace_, 18, size_ * meta_.GetValue(12) * 512);
#endif
  }
  CpPeerMapping peers_;
  __gm__ uint8_t *arena_ = nullptr;
  GlobalTensor<int64_t> meta_, requests_, trace_;
  GlobalTensor<float> gradient_, owner_;
  int64_t epoch_ = 0;
  int32_t rank_ = 0, size_ = 0;
};
}  // namespace DsaMixed
#endif
