// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_ABI_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_ABI_H_

#include <cstdint>

namespace HyperParallelDense {
constexpr uint32_t DENSE_BACKWARD_MAGIC = 0x444E4231;
constexpr uint32_t DENSE_BACKWARD_VERSION = 1;
constexpr uint32_t DENSE_BACKWARD_MAX_OPERATIONS = 64;
constexpr uint32_t DENSE_BACKWARD_MAX_VALUES = 128;
constexpr uint32_t DENSE_SWIGLU_GRAD = 2;
constexpr uint32_t DENSE_GRADIENT_ADD = 3;
constexpr uint32_t DENSE_ZERO = 4;
constexpr uint32_t DENSE_SILU = 5;
constexpr uint32_t DENSE_MULTIPLY = 6;
constexpr uint32_t DENSE_SILU_GRAD = 7;

struct DenseBackwardHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t rows_per_tile;
  uint32_t cube_workers;
  uint32_t prefetch_tiles;
  uint32_t operation_count;
  uint32_t value_count;
  uint32_t event_count;
  uint32_t dependency_count;
  uint32_t phase_tiles[2];
};

struct DenseBackwardOperation {
  uint32_t kind;
  uint32_t left;
  uint32_t right;
  uint32_t output;
  uint32_t rows;
  uint32_t columns;
  uint32_t contracted;
  uint32_t transpose_left;
  uint32_t transpose_right;
  uint32_t transpose_output;
  uint32_t phase;
  uint32_t columns_per_tile;
  uint32_t column_tiles;
  uint32_t event_begin;
  uint32_t tile_count;
  uint32_t signal_count;
  uint32_t dependency_begin;
  uint32_t dependency_count;
};

struct DenseBackwardDependency {
  uint32_t event_begin;
  uint32_t tile_count;
  uint32_t signal_count;
  uint32_t row_local;
};

static_assert(sizeof(DenseBackwardHeader) == 44, "Dense backward header ABI drift");
static_assert(sizeof(DenseBackwardOperation) == 72, "Dense backward operation ABI drift");
static_assert(sizeof(DenseBackwardDependency) == 16, "Dense backward dependency ABI drift");
}  // namespace HyperParallelDense

#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_ABI_H_
