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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_TRANSPORT_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_TRANSPORT_H_
#include "kernel_operator.h"  // NOLINT(build/include_subdir)
#include "data_plane/rma.h"
#include "dsa_mixed_group.h"  // NOLINT(build/include_subdir)

namespace DsaMixed {
class CpTransport {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *arena, __gm__ uint8_t *metadata,
                             __gm__ uint8_t *requests, __gm__ uint8_t *trace) {
    arena_ = arena;
    meta_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata), 18);
    requests_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(requests));
    trace_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace), 32);
    epoch_ = meta_.GetValue(2);
    rank_ = static_cast<int32_t>(meta_.GetValue(3));
    size_ = static_cast<int32_t>(meta_.GetValue(4));
  }

  __aicore__ inline void PublishReady() {
#ifndef __DAV_C220_CUBE__
    auto *state = aclshmemi_get_state();
    if (meta_.GetValue(0) != 0x4850445341435031LL || meta_.GetValue(1) != 1 ||
        state->mype != rank_ || state->npes != size_ || epoch_ <= 0) {
      AscendC::Trap();
    }
    auto signal = Signal(rank_);
    for (uint32_t index = 0; index < 5; ++index) {
      PublishControl(signal, index, meta_.GetValue(5 + index));
    }
    PublishControl(signal, 8, epoch_);
    for (int32_t peer = 0; peer < size_; ++peer) {
      if (peer != rank_ && !(state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE)) {
        AscendC::Trap();
      }
      auto remote = Signal(peer);
      while (ReadVisible(remote, 8) < epoch_) {}
      for (uint32_t index = 0; index < 5; ++index) {
        if (ReadVisible(remote, index) != meta_.GetValue(5 + index)) {
          AscendC::Trap();
        }
      }
    }
#endif
  }

  __aicore__ inline void PullIndex(__gm__ uint8_t *indexKey) {
#ifndef __DAV_C220_CUBE__
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 8192);
    auto buffer = storage.Get<uint8_t>();
    PublishControl(trace_, 8, AscendC::GetSystemCycle());
    PullField(indexKey, meta_.GetValue(10), 128, buffer, 4, 6);
    PublishControl(trace_, 9, AscendC::GetSystemCycle());
    PublishControl(trace_, 0, epoch_);
#endif
  }

  __aicore__ inline void PullMain(__gm__ uint8_t *compressed, __gm__ uint8_t *rope,
                                 GlobalTensor<int64_t> &compute, uint32_t groups) {
#ifndef __DAV_C220_CUBE__
    for (uint32_t group = 0; group < groups; ++group) {
      for (uint32_t member = 0; member < 3; ++member) {
        while (ReadVisible(compute, group * kGroupWords + member * kMemberWords + 7) != 1) {}
      }
    }
    PublishControl(trace_, 13, 3 * groups);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    pipe.InitBuffer(storage, 8192);
    auto buffer = storage.Get<uint8_t>();
    PublishControl(trace_, 10, AscendC::GetSystemCycle());
    PullField(compressed, meta_.GetValue(11), 512, buffer, 5, 7);
    PullField(rope, meta_.GetValue(12), 64, buffer, 5, 7);
    PublishControl(trace_, 11, AscendC::GetSystemCycle());
    auto local = Signal(rank_);
    PublishControl(local, 16, epoch_);
    PublishControl(trace_, 2, size_);
    PublishControl(trace_, 1, epoch_);
#endif
  }

  __aicore__ inline void WaitIndex() {
    while (ReadVisible(trace_, 0) != epoch_) {}
  }

  __aicore__ inline void WaitAcknowledged() {
#ifndef __DAV_C220_CUBE__
    for (int32_t peer = 0; peer < size_; ++peer) {
      auto remote = Signal(peer);
      while (ReadVisible(remote, 16) < epoch_) {}
    }
    PublishControl(trace_, 3, size_);
#endif
  }

 private:
  __aicore__ inline GlobalTensor<int64_t> Signal(int32_t peer) {
    auto *address = arena_ + meta_.GetValue(13);
    if (peer != rank_) {
      address = reinterpret_cast<__gm__ uint8_t *>(
          hyper_parallel::multicore::shmem::data_plane::remote_ptr(address, peer));
    }
    GlobalTensor<int64_t> signal;
    signal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(address), 32);
    return signal;
  }

  __aicore__ inline void PullField(__gm__ uint8_t *destination, int64_t sourceOffset, uint32_t width,
                                  AscendC::LocalTensor<uint8_t> &buffer, uint32_t totalWord, uint32_t remoteWord) {
#ifndef __DAV_C220_CUBE__
    int64_t total = ReadVisible(trace_, totalWord);
    int64_t remote = ReadVisible(trace_, remoteWord);
    const uint32_t rowBytes = width * 2;
    for (int64_t run = 0; run < meta_.GetValue(14); ++run) {
      const int32_t peer = static_cast<int32_t>(requests_.GetValue(run * 4));
      auto *source = arena_ + sourceOffset + requests_.GetValue(run * 4 + 1) * rowBytes;
      auto *target = destination + requests_.GetValue(run * 4 + 2) * rowBytes;
      const int64_t bytes = requests_.GetValue(run * 4 + 3) * rowBytes;
      if (peer != rank_) {
        source = reinterpret_cast<__gm__ uint8_t *>(
            hyper_parallel::multicore::shmem::data_plane::remote_ptr(source, peer));
        remote += bytes;
      }
      Copy(target, source, bytes, buffer);
      total += bytes;
    }
    PublishControl(trace_, totalWord, total);
    PublishControl(trace_, remoteWord, remote);
#endif
  }

  __aicore__ inline void Copy(__gm__ uint8_t *destination, __gm__ uint8_t *source,
                             int64_t bytes, AscendC::LocalTensor<uint8_t> &buffer) {
#ifndef __DAV_C220_CUBE__
    GlobalTensor<uint8_t> input, output;
    input.SetGlobalBuffer(source, bytes);
    output.SetGlobalBuffer(destination, bytes);
    for (int64_t offset = 0; offset < bytes; offset += 8192) {
      const uint32_t count = static_cast<uint32_t>(bytes - offset < 8192 ? bytes - offset : 8192);
      AscendC::DataCopyExtParams params{1, count, 0, 0, 0};
      AscendC::DataCopyPadExtParams<uint8_t> padding{false, 0, 0, 0};
      AscendC::DataCopyPad(buffer, input[offset], params, padding);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
      AscendC::DataCopyPad(output[offset], buffer, params);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
#endif
  }
  __gm__ uint8_t *arena_ = nullptr;
  GlobalTensor<int64_t> meta_, requests_, trace_;
  int64_t epoch_ = 0;
  int32_t rank_ = 0;
  int32_t size_ = 0;
};
}  // namespace DsaMixed
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_RUNTIME_DSA_CP_TRANSPORT_H_
