// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_TILE_ABI_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_TILE_ABI_H_

#include <cstdint>

namespace HyperParallelDense {
constexpr uint32_t DENSE_TILE_MAGIC = 0x444E5331;
constexpr uint32_t DENSE_TILE_VERSION = 1;
constexpr uint32_t DENSE_MAX_OPERATIONS = 16;
constexpr uint32_t DENSE_MAX_VALUES = 32;
constexpr uint32_t DENSE_EVENT_STRIDE = 32;
constexpr uint32_t DENSE_MATMUL = 0;
constexpr uint32_t DENSE_SWIGLU = 1;

struct DenseTileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t rows;
  uint32_t rows_per_tile;
  uint32_t cube_workers;
  uint32_t prefetch_tiles;
  uint32_t operation_count;
  uint32_t value_count;
};

struct DenseTileOperation {
  uint32_t kind;
  uint32_t left;
  uint32_t right;
  uint32_t output;
  uint32_t columns;
  uint32_t contracted;
  uint32_t transpose_right;
  int32_t dependency;
  uint32_t dependency_count;
};

static_assert(sizeof(DenseTileHeader) == 32, "Dense tile header ABI drift");
static_assert(sizeof(DenseTileOperation) == 36, "Dense tile operation ABI drift");
}  // namespace HyperParallelDense

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_TILE_ABI_H_
