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
  template <typename T, bool Count = false>
  __aicore__ inline void PullOriginalRows(__gm__ T *source, __gm__ T *destination, __ubuf__ int32_t *metadata,
                                         uint64_t scratch, int64_t rows, int64_t hidden, uint32_t top_k,
                                         uint32_t source_tokens, uint64_t *statistics) {
    constexpr int64_t CHUNK = 3072;
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
      if constexpr (Count) { statistics[0] += bytes; }
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
          if constexpr (Count) { statistics[0] += bytes; }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        aclshmemi_copy_ub2gm(destination + row * hidden + col, buffers[parity], bytes);
        if constexpr (Count) { statistics[1] += bytes; }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
      }
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
  }

  template <typename T, bool Count = false>
  __aicore__ inline void PullPeerRows(
      __gm__ T *source, __gm__ T *destination, __gm__ int32_t *remote_metadata,
      __ubuf__ int32_t *sorted_metadata, uint64_t scratch, int64_t rows, int64_t hidden,
      uint32_t top_k, uint32_t source_tokens, int64_t rank, int64_t source_pe, int64_t local_experts,
      __gm__ int64_t *target_offsets, uint64_t *statistics) {
    constexpr int64_t CHUNK = 3072;
    // Match the indexed bridge chunk while honoring the SDK 16 KiB UB lease.
    constexpr uint64_t TOKEN_METADATA_BYTES = 16 * 4 * sizeof(int32_t);
    auto *token_metadata = reinterpret_cast<__ubuf__ int32_t *>(scratch);
    auto *values = reinterpret_cast<__ubuf__ T *>(scratch + TOKEN_METADATA_BYTES);
    if (top_k == 0 || top_k > 16 || scratch + TOKEN_METADATA_BYTES + CHUNK * sizeof(T) > 192 * 1024) {
      AscendC::Trap();
    }
    for (int64_t row = 0; row < rows; ++row) {
      int64_t slot = sorted_metadata[row * 4];
      int64_t token = slot / top_k;
      if (slot < 0 || token >= source_tokens) { AscendC::Trap(); }
      aclshmemi_copy_gm2ub(token_metadata, remote_metadata + token * top_k * 4,
                         top_k * 4 * sizeof(int32_t));
      if constexpr (Count) { statistics[2] += top_k * 4 * sizeof(int32_t); }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      int64_t owner = -1;
      for (uint32_t selected = 0; selected < top_k; ++selected) {
        int64_t expert = token_metadata[selected * 4 + 2];
        if (expert / local_experts == rank) { owner = selected; break; }
      }
      if (owner < 0) { AscendC::Trap(); }
      // Exactly the first selected expert on this destination pulls the token.
      // The prepass kernel boundary makes every fanout row ready before GMM.
      if (slot % top_k != owner) { continue; }
      if constexpr (Count) { ++statistics[3]; }
      for (int64_t col = 0; col < hidden; col += CHUNK) {
        int64_t elements = hidden - col < CHUNK ? hidden - col : CHUNK;
        uint64_t bytes = elements * sizeof(T);
        aclshmemi_copy_gm2ub(values, source + token * hidden + col, bytes);
        if constexpr (Count) { statistics[0] += bytes; }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          int64_t expert = token_metadata[selected * 4 + 2];
          if (expert / local_experts != rank) { continue; }
          int64_t ordinal = token_metadata[selected * 4 + 3];
          if (ordinal < 0 || ordinal >= source_tokens) { AscendC::Trap(); }
          int64_t offset = target_offsets[source_pe * local_experts + expert % local_experts];
          aclshmemi_copy_ub2gm(destination + offset + ordinal * hidden + col, values, bytes);
          if constexpr (Count) { statistics[1] += bytes; }
        }
        // Every local destination has consumed this UB row before its reuse.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
  }

  __aicore__ inline void StoreIngressStatistics(const TaskDesc &task, GM_ADDR metadata,
                                                int64_t tokens, uint64_t *statistics) {
    if (task.inputs[1].dim[2] != 1) { return; }
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(
      metadata + tokens * task.inputs[1].dim[3] * 4 * sizeof(int32_t)) + task.task_index * 16, 16);
    for (uint32_t column = 0; column < 4; ++column) { output.SetValue(column, statistics[column]); }
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(output);
    PipeBarrier<PIPE_ALL>();
  }

  template <typename T>
  __aicore__ inline void ExecuteIndexedPull(const TaskDesc &task, GM_ADDR *input_list, uint32_t tiling_index) {
#ifdef __DAV_CUBE__
    AscendC::Trap();
#else
    if (task.extra_value_1 == 3) { return; }
    int64_t ep = getExtraValueFromTiling(input_list[tiling_index], 1);
    int64_t experts = getExtraValueFromTiling(input_list[tiling_index], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[tiling_index], 3);
    int64_t tokens = getExtraValueFromTiling(input_list[tiling_index], 4);
    int64_t tiles = task.task_split_num / experts;
    int source_pe = task.task_index / ((experts / ep) * tiles);
    int64_t size = reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[
      task.inputs[3].base_ptr_offset];
    int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    uint64_t statistics[4] = {0, 0, 0, 0};
    if (start >= size) {
      StoreIngressStatistics(task, input_list[task.extra_value_0], tokens, statistics);
      return;
    }
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
          source_pe, scratch_ub, transport_bytes) || transport_bytes < 128 * 4 * sizeof(int32_t) + 2 * 3072 * sizeof(T)) {
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
      if (task.inputs[1].dim[2] == 1) { statistics[2] += rows * 4 * sizeof(int32_t); }
      // No TPipe is alive in the transport task. Own a fixed event rather
      // than querying FetchEventID through the previous task's pipe pointer.
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      if (task.extra_value_1 == 2) {
        int64_t rank = getExtraValueFromTiling(input_list[tiling_index], 0) % ep;
        if (task.inputs[1].dim[2] == 1) {
          PullPeerRows<T, true>(remote_data,
          reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
          scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
          rank, source_pe, experts / ep,
          reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics);
        } else {
          PullPeerRows<T, false>(remote_data,
          reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
          scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
          rank, source_pe, experts / ep,
          reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics);
        }
      } else {
        if (task.inputs[1].dim[2] == 1) {
          PullOriginalRows<T, true>(remote_data, destination + begin * hidden, metadata,
                           scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden,
                           task.inputs[1].dim[3], tokens, statistics);
          statistics[3] += rows;
        } else {
          PullOriginalRows<T, false>(remote_data, destination + begin * hidden, metadata,
                           scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden,
                           task.inputs[1].dim[3], tokens, statistics);
        }
      }
    }
    StoreIngressStatistics(task, input_list[task.extra_value_0], tokens, statistics);
#endif
  }

}  // namespace MulticoreRuntime
#endif
