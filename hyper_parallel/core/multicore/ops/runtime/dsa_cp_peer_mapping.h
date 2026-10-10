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
#ifndef HYPER_PARALLEL_DSA_CP_PEER_MAPPING_H_
#define HYPER_PARALLEL_DSA_CP_PEER_MAPPING_H_
#include "kernel_operator.h"
#include "data_plane/rma.h"
#include "dsa_mixed_group.h"
namespace DsaMixed {
class CpPeerMapping {
 public:
  __aicore__ inline void Init(__gm__ uint8_t *metadata, bool mapped, int32_t rank, int32_t size) {
    mapped_ = mapped;
    rank_ = rank;
    size_ = size;
    if (mapped_) {
      extension_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(metadata) + 18, 9);
    }
  }
  __aicore__ inline int32_t RootPe(int32_t peer) {
    return mapped_ ? static_cast<int32_t>(extension_.GetValue(1 + peer)) : peer;
  }
  __aicore__ inline void Validate() {
#ifndef __DAV_C220_CUBE__
    auto *state = aclshmemi_get_state();
    const auto rootSize = mapped_ ? extension_.GetValue(0) : size_;
    if (size_ <= 0 || rank_ < 0 || rank_ >= size_ || rootSize != state->npes || (mapped_ && size_ > 8)) {
      AscendC::Trap();
    }
    for (int32_t peer = 0; peer < size_; ++peer) {
      const int32_t pe = RootPe(peer);
      if (pe < 0 || pe >= state->npes) {
        AscendC::Trap();
      }
      if (mapped_) {
        for (int32_t previous = 0; previous < peer; ++previous) {
          if (RootPe(previous) == pe) {
            AscendC::Trap();
          }
        }
      }
    }
    if (mapped_) {
      for (int32_t peer = size_; peer < 8; ++peer) {
        if (extension_.GetValue(1 + peer) != -1) {
          AscendC::Trap();
        }
      }
    }
    if (state->mype != RootPe(rank_)) {
      AscendC::Trap();
    }
#endif
  }

 private:
  GlobalTensor<int64_t> extension_;
  bool mapped_ = false;
  int32_t rank_ = 0, size_ = 0;
};
}  // namespace DsaMixed
#endif
