/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#ifndef HYPER_PARALLEL_INDEXED_PULL_H
#define HYPER_PARALLEL_INDEXED_PULL_H

// Forward-only scalar route publication at port 16. Blaze F2 leaves this
// legacy workspace argument unused; all backward transport remains unchanged.
namespace MulticoreRuntime {
  template <typename T>
  __aicore__ inline void PullOriginalRows(__gm__ T *source, __gm__ T *destination, __ubuf__ int32_t *metadata,
                                         uint64_t scratch, int64_t rows, int64_t hidden, uint32_t top_k,
                                         uint32_t source_tokens) {
    constexpr int64_t CHUNK = 8192;
    uint64_t value_bytes = CHUNK * sizeof(T);
    if (scratch + 2 * value_bytes > 192 * 1024) {
      AscendC::Trap();
    }
    __ubuf__ T *buffers[2] = {reinterpret_cast<__ubuf__ T *>(scratch),
                            reinterpret_cast<__ubuf__ T *>(scratch + value_bytes)};
    for (int64_t col = 0; col < hidden; col += CHUNK) {
      int64_t elements = hidden - col < CHUNK ? hidden - col : CHUNK;
      uint64_t bytes = elements * sizeof(T);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      int64_t first = metadata[0] / top_k;
      if (first < 0 || first >= source_tokens) { AscendC::Trap(); }
      aclshmemi_copy_gm2ub(buffers[0], source + first * hidden + col, bytes);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      for (int64_t row = 0; row < rows; ++row) {
        uint32_t parity = row & 1;
        AscendC::TEventID event = parity ? EVENT_ID1 : EVENT_ID0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        if (row + 1 < rows) {
          AscendC::TEventID next = parity ? EVENT_ID0 : EVENT_ID1;
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(next);
          int64_t token = metadata[(row + 1) * 4] / top_k;
          if (token < 0 || token >= source_tokens) { AscendC::Trap(); }
          aclshmemi_copy_gm2ub(buffers[1 - parity], source + token * hidden + col, bytes);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        aclshmemi_copy_ub2gm(destination + row * hidden + col, buffers[parity], bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
      }
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
  }

  template <typename T>
  __aicore__ inline void ExecuteIndexedPull(const TaskDesc &task, GM_ADDR *input_list, uint32_t tiling_index) {
#ifdef __DAV_CUBE__
    AscendC::Trap();
#else
    int64_t ep = getExtraValueFromTiling(input_list[tiling_index], 1);
    int64_t experts = getExtraValueFromTiling(input_list[tiling_index], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[tiling_index], 3);
    int64_t tokens = getExtraValueFromTiling(input_list[tiling_index], 4);
    int64_t tiles = task.task_split_num / experts;
    int source_pe = task.task_index / ((experts / ep) * tiles);
    int64_t size = reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[
      task.inputs[3].base_ptr_offset];
    int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    if (start >= size) { return; }
    int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
    int64_t elements = size - start < tile_elements ? size - start : tile_elements;
    int64_t destination_offset = reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position])[
      task.inputs[0].base_ptr_offset] + start;
    int64_t source_offset = reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position])[
      task.inputs[2].base_ptr_offset] + start;
    namespace plane = hyper_parallel::multicore::shmem::data_plane;
    constexpr int64_t META_ROWS = 128;
    // Each task drains all transfers before another task reuses this worker's UB.
    uint64_t scratch_ub;
    uint64_t transport_bytes;
    if (!hyper_parallel::multicore::shmem::cann::device::query_mte_resources(
          source_pe, scratch_ub, transport_bytes) || transport_bytes == 0) {
      AscendC::Trap();
    }
    auto *remote_data = reinterpret_cast<__gm__ T *>(plane::remote_ptr(
      input_list[task.inputs[1].input_position], source_pe));
    auto *remote_metadata = reinterpret_cast<__gm__ int32_t *>(plane::remote_ptr(
      input_list[task.extra_value_0], source_pe));
    if (remote_data == nullptr || remote_metadata == nullptr) {
      AscendC::Trap();
    }
    auto *destination = reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]) + destination_offset;
    int64_t source_row = source_offset / hidden;
    int64_t total_rows = elements / hidden;
    for (int64_t begin = 0; begin < total_rows; begin += META_ROWS) {
      int64_t rows = total_rows - begin < META_ROWS ? total_rows - begin : META_ROWS;
      auto *metadata = reinterpret_cast<__ubuf__ int32_t *>(scratch_ub);
      aclshmemi_copy_gm2ub(metadata, remote_metadata + (source_row + begin) * 4,
                           rows * 4 * sizeof(int32_t));
      // No TPipe is alive in the transport task. Own a fixed event rather
      // than querying FetchEventID through the previous task's pipe pointer.
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      PullOriginalRows<T>(remote_data, destination + begin * hidden, metadata,
                         scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden,
                         task.inputs[1].dim[3], tokens);
    }
#endif
  }

}  // namespace MulticoreRuntime
#endif
