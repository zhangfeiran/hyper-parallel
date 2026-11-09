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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __ubuf__
using GM_ADDR = unsigned char *;
constexpr size_t DATA_CACHE_LINE_SIZE = 128;
constexpr size_t ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE = 1024;
constexpr int ACLSHMEM_TRANSPORT_UDMA = 1;
inline int aclshmem_n_pes() { return 4; }
inline int aclshmem_my_pe() { return 3; }
struct DeviceState { int topo_list[4] = {1, 1, 1, 1}; };
inline DeviceState *aclshmemi_get_state() { static DeviceState state; return &state; }
struct QPInfo { int qp_num = 8; };
inline QPInfo *aclshmemi_udma_qp_info_fetch() { static QPInfo info; return &info; }
struct PendingGet { unsigned char *target = nullptr, *source = nullptr; size_t bytes = 0; };
static PendingGet pending[4][8];
template <class T> void aclshmemx_udma_qp_get_nbi(T *target, T *source, unsigned char *,
                                               size_t bytes, int peer, int qp, int) {
  auto &get = pending[peer][qp];
  assert(get.target == nullptr);
  get = {reinterpret_cast<unsigned char *>(target), reinterpret_cast<unsigned char *>(source), bytes};
}
inline void aclshmemx_udma_qp_quiet(int peer, int qp) {
  auto &get = pending[peer][qp];
  assert(get.target != nullptr);
  std::memcpy(get.target, get.source, get.bytes);
  get = {};
}

static size_t max_copy_bytes = 0;
static size_t max_buffer_bytes = 0;
static float *output_base = nullptr;
static std::vector<int> writes;

namespace hyper_parallel::multicore::shmem::data_plane {
inline void *remote_ptr(void *address, int peer) {
  assert(peer >= 0 && peer < 3);
  return address;
}
}

namespace AscendC {
enum class TPosition { VECCALC, VECOUT };
enum class HardEvent { MTE2_V, V_MTE3, MTE3_MTE2 };
constexpr int EVENT_ID0 = 0;
inline void Trap() { throw std::runtime_error("Native guard trap"); }
template <HardEvent event> void SetFlag(int) {}
template <HardEvent event> void WaitFlag(int) {}
inline void SetAtomicNone() {}

template <class T> struct LocalTensor {
  T *data;
  size_t count;
  void *GetPhyAddr() const { return data; }
  LocalTensor operator[](size_t offset) const {
    assert(offset <= count);
    return {data + offset, count - offset};
  }
};
template <class T> struct GlobalTensor {
  T *data = nullptr;
  size_t count = 0;
  void SetGlobalBuffer(T *address, int64_t elements) {
    assert(elements >= 0);
    data = address;
    count = static_cast<size_t>(elements);
  }
  GlobalTensor operator[](size_t offset) const {
    assert(offset <= count);
    return {data + offset, count - offset};
  }
};
template <TPosition position> struct TBuf {
  std::vector<float> storage;
  template <class T> LocalTensor<T> Get() {
    return {reinterpret_cast<T *>(storage.data()), storage.size() * sizeof(float) / sizeof(T)};
  }
};
struct TPipe {
  template <class T> void InitBuffer(T &buffer, size_t bytes) {
    assert(bytes % sizeof(float) == 0);
    max_buffer_bytes = std::max(max_buffer_bytes, bytes);
    buffer.storage.resize(bytes / sizeof(float));
  }
  void Destroy() {}
};
struct DataCopyExtParams { uint32_t blocks, bytes, source_stride, target_stride, reserved; };
template <class T> struct DataCopyPadExtParams { bool enabled; int left, right; T value; };
inline void DataCopyPad(LocalTensor<float> target, GlobalTensor<float> source,
                        DataCopyExtParams params, DataCopyPadExtParams<float>) {
  size_t count = params.bytes / sizeof(float);
  assert(params.blocks == 1 && count <= target.count && count <= source.count);
  max_copy_bytes = std::max(max_copy_bytes, static_cast<size_t>(params.bytes));
  std::memcpy(target.data, source.data, params.bytes);
}
inline void DataCopyPad(GlobalTensor<float> target, LocalTensor<float> source, DataCopyExtParams params) {
  size_t count = params.bytes / sizeof(float);
  assert(params.blocks == 1 && count <= target.count && count <= source.count);
  for (size_t i = 0; i < count; ++i) ++writes.at(target.data - output_base + i);
  std::memcpy(target.data, source.data, params.bytes);
}
inline void Add(LocalTensor<float> target, LocalTensor<float> left, LocalTensor<float> right, uint32_t count) {
  assert(count <= target.count && count <= left.count && count <= right.count);
  for (uint32_t i = 0; i < count; ++i) target.data[i] = left.data[i] + right.data[i];
}
}
using AscendC::EVENT_ID0;
#include "native_gradient.h"

int RunReturn(GM_ADDR source, size_t elements, int peer, GM_ADDR scratch, size_t chunk) {
      for (int worker = 0; worker < 4; ++worker) {
#if HP_UDMA
        if (elements == 0) {
          bool trapped = false;
          try {
            AddReplicaGradientUdma(reinterpret_cast<GM_ADDR>(output_base),
                source, elements, peer, worker, 4,
                scratch, chunk, worker + 1);
          } catch (const std::runtime_error &) { trapped = true; }
          if (!trapped) return 7;
        } else {
          AddReplicaGradientUdma(reinterpret_cast<GM_ADDR>(output_base),
              source, elements, peer, worker, 4,
              scratch, chunk, worker + 1);
        }
#else
        AddReplicaGradient(reinterpret_cast<GM_ADDR>(output_base),
                           source, elements, peer, worker, 4);
#endif
      }
  return 0;
}

int CheckShape(size_t elements, GM_ADDR scratch, size_t chunk) {
  constexpr size_t guard = 8;
    std::vector<float> owner(elements + 2 * guard, -123.5f), reference(elements, 1.25f);
    std::vector<float> guests[3];
    std::fill(owner.begin() + guard, owner.end() - guard, 1.25f);
    for (int peer = 0; peer < 3; ++peer) {
      guests[peer].resize(elements);
      for (size_t i = 0; i < elements; ++i) {
        guests[peer][i] = peer == 0 ? 1.0e20f : peer == 1 ? -1.0e20f : float((i % 17) + 1) / 8.0f;
        reference[i] += guests[peer][i];
      }
      writes.assign(elements, 0);
      output_base = owner.data() + guard;
      const int code = RunReturn(reinterpret_cast<GM_ADDR>(guests[peer].data()), elements, peer, scratch, chunk);
      if (code) return code;
      if (std::any_of(writes.begin(), writes.end(), [](int value) { return value != 1; })) return 2;
    }
    if (elements && std::memcmp(output_base, reference.data(), elements * sizeof(float))) return 3;
    for (size_t i = 0; i < guard; ++i) {
      if (owner[i] != -123.5f || owner[guard + elements + i] != -123.5f) return 4;
    }
  return 0;
}

int main() {
  const size_t shapes[] = {0, 1, 7, 4095, 4096, 4097, 8191, 8192, 8193, 32771, 131089};
  constexpr size_t chunk = 8192;
  std::vector<unsigned char> scratch_storage(8 * chunk * sizeof(float) + DATA_CACHE_LINE_SIZE);
  auto scratch_bits = (reinterpret_cast<uintptr_t>(scratch_storage.data()) + DATA_CACHE_LINE_SIZE - 1) &
                      ~(DATA_CACHE_LINE_SIZE - 1);
  auto scratch = reinterpret_cast<GM_ADDR>(scratch_bits);
  for (size_t elements : shapes) {
    const int code = CheckShape(elements, scratch, chunk);
    if (code) return code;
  }
  for (const auto &peer : pending) for (const auto &get : peer) if (get.target != nullptr) return 8;
  if (max_copy_bytes != HP_EXPECTED_TILE * sizeof(float)) return 5;
  if (max_buffer_bytes != 2 * HP_EXPECTED_TILE * sizeof(float)) return 6;
  std::cout << "Verified 11 guarded ordered FP32 gradient shapes" << std::endl;
  return 0;
}
