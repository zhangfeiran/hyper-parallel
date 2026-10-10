// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_ACCESS_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_ACCESS_H_

#include "dense_tile_abi.h"

namespace HyperParallelDense {
__aicore__ inline uint32_t MatrixSegmentEnd(uint32_t first, uint32_t span, uint32_t end) {
  const uint32_t boundary = (first / span + 1) * span;
  return boundary < end ? boundary : end;
}

__aicore__ inline uint64_t MatrixKState(const DenseMatrixAccess &access, uint32_t rows, uint32_t columns,
                                       uint32_t contracted, uint32_t first_row, uint32_t first_column) {
  if (access.k_chunk == 0 || access.close_shift != 0) {
    return 0;
  }
  const bool row_axis = access.m_extent >= access.n_extent;
  const uint32_t span = row_axis ? access.m_span : access.n_span;
  const uint32_t extent = row_axis ? access.m_extent : access.n_extent;
  const uint32_t index = (row_axis ? first_row : first_column) / span;
  const uint32_t size = row_axis ? rows : columns;
  const uint32_t padded = (size + 15) / 16 * 16;
  const uint32_t chunks = contracted / access.k_chunk;
  // 与 SDK 的 ShiftBlockAccess 一致：尾部重叠区域必须保留原始收缩顺序。
  if (extent < 4 || extent / 2 > chunks ||
      (padded % span != 0 && static_cast<uint64_t>(index + 2) * span > padded)) {
    return 0;
  }
  const uint32_t rotation = (chunks / (extent / 2)) * ((index / 2) % (extent / 2));
  const uint32_t order = rotation | (index % 2 ? 0x80000000u : 0);
  return order == 0 ? 0 : (static_cast<uint64_t>(order) << 32) | access.k_chunk;
}

__aicore__ inline uint32_t MatrixKOffset(uint32_t offset, uint32_t contracted, uint64_t state) {
  if (state == 0) {
    return offset;
  }
  const uint32_t chunk = static_cast<uint32_t>(state);
  const uint32_t order = static_cast<uint32_t>(state >> 32);
  const uint32_t chunks = contracted / chunk;
  uint32_t index = offset / chunk;
  index = (order & 0x80000000u) ? chunks - index - 1 : index;
  index = (index + (order & 0x7fffffffu)) % chunks;
  return index * chunk + offset % chunk;
}
}  // namespace HyperParallelDense

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_ACCESS_H_
