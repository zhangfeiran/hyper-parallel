// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "matmul/matmul_tiling.h"
#include "matmul/bmm_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
platform_ascendc::PlatformAscendC *Platform(const char *soc) {
  return platform_ascendc::PlatformAscendCManager::GetInstance(soc);
}

bool Configure(matmul_tiling::MatmulApiTilingBase &builder, uint32_t rows, uint32_t tile_rows, uint32_t columns,
               uint32_t tile_columns, uint32_t contracted, bool transpose_right) {
  using matmul_tiling::CubeFormat;
  using matmul_tiling::DataType;
  using matmul_tiling::TPosition;
  return builder.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, false) == 0 &&
         builder.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16, transpose_right) == 0 &&
         builder.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_BF16) == 0 &&
         builder.SetShape(tile_rows, tile_columns, contracted) == 0 &&
         builder.SetOrgShape(rows, columns, contracted) == 0 &&
         builder.SetBias(false) == 0 && builder.SetBufferSpace(-1, -1, -1) == 0;
}
}  // namespace

extern "C" int hyper_parallel_dense_platform(const char *soc, uint32_t *cube_workers, uint64_t *workspace_bytes,
                                             uint32_t *tiling_bytes) {
  auto *platform = Platform(soc);
  if (platform == nullptr || cube_workers == nullptr || workspace_bytes == nullptr || tiling_bytes == nullptr) {
    return -1;
  }
  *cube_workers = platform->GetCoreNumAic();
  *workspace_bytes = platform->GetLibApiWorkSpaceSize();
  *tiling_bytes = sizeof(AscendC::tiling::TCubeTiling);
  return 0;
}

extern "C" int hyper_parallel_dense_partition(const char *soc, uint32_t rows, uint32_t columns,
                                               uint32_t contracted, uint32_t transpose_right,
                                               uint32_t *tile_rows, uint32_t *tile_columns) {
  auto *platform = Platform(soc);
  if (platform == nullptr || rows == 0 || columns == 0 || contracted == 0 || tile_rows == nullptr ||
      tile_columns == nullptr) {
    return -1;
  }
  matmul_tiling::MultiCoreMatmulTiling builder(*platform);
  if (builder.SetDim(platform->GetCoreNumAic()) != 0 ||
      !Configure(builder, rows, rows, columns, columns, contracted, transpose_right != 0)) {
    return -2;
  }
  AscendC::tiling::TCubeTiling tiling{};
  if (builder.GetTiling(tiling) != 0 || tiling.singleCoreM <= 0 || tiling.singleCoreN <= 0) {
    return -3;
  }
  *tile_rows = tiling.singleCoreM;
  *tile_columns = tiling.singleCoreN;
  return 0;
}

extern "C" int hyper_parallel_dense_matmul_tiling(const char *soc, uint32_t rows, uint32_t tile_rows, uint32_t columns,
                                                  uint32_t tile_columns, uint32_t contracted,
                                                  uint32_t transpose_right, void *output,
                                                  uint32_t capacity) {
  auto *platform = Platform(soc);
  if (platform == nullptr || output == nullptr || capacity != sizeof(AscendC::tiling::TCubeTiling) || rows == 0 ||
      tile_rows == 0 || columns == 0 || tile_columns == 0 || tile_columns > columns || contracted == 0) {
    return -1;
  }
  matmul_tiling::MatmulApiTiling builder(*platform);
  if (!Configure(builder, rows, tile_rows, columns, tile_columns, contracted, transpose_right != 0)) {
    return -2;
  }
  AscendC::tiling::TCubeTiling tiling{};
  if (builder.GetTiling(tiling) != 0) {
    return -3;
  }
  std::memcpy(output, &tiling, sizeof(tiling));
  return 0;
}
