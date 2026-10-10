// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "exe_graph/runtime/tensor.h"
#include "platform/platform_info.h"

// 选定 SDK 的 legacy 包间接口；原型与桥接源码、库哈希一起封存。
extern "C" bool LegacyMmCheckHitV3Shape(const gert::Tensor *, const gert::Tensor *, const gert::Tensor *,
                                        bool, bool, ge::Format, bool, uint32_t, const std::string &);

extern "C" int hyper_parallel_dense_compile_platform(const char *soc, uint32_t cube_workers) {
  if (soc == nullptr || cube_workers == 0) {
    return -1;
  }
  for (auto *manager : {&fe::PlatformInfoManager::Instance(), &fe::PlatformInfoManager::GeInstance()}) {
    if (manager->InitializePlatformInfo() != 0) {
      return -2;
    }
    fe::OptionalInfos optional;
    if (!optional.Init()) {
      return -3;
    }
    optional.SetSocVersion(soc);
    optional.SetCoreType("AiCore");
    optional.SetAICoreNum(cube_workers);
    optional.SetL1FusionFlag("");
    manager->SetOptionalCompilationInfo(optional);
    fe::PlatFormInfos platform;
    if (manager->GetPlatformInfoWithOutSocVersion(platform, optional) != 0) {
      return -4;
    }
  }
  return 0;
}

extern "C" int hyper_parallel_dense_native_family(const char *soc, uint32_t cube_workers, uint32_t rows,
                                                   uint32_t columns, uint32_t contracted, uint32_t transpose,
                                                   uint32_t split_k) {
  if (soc == nullptr || cube_workers == 0 || rows == 0 || columns == 0 || contracted == 0) {
    return -1;
  }
  gert::Tensor left, right;
  left.MutableOriginShape() = gert::Shape({rows, contracted});
  left.MutableStorageShape() = left.GetOriginShape();
  right.MutableOriginShape() = transpose ? gert::Shape({columns, contracted}) : gert::Shape({contracted, columns});
  right.MutableStorageShape() = right.GetOriginShape();
  for (auto *tensor : {&left, &right}) {
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    tensor->SetDataType(ge::DT_BF16);
  }
  return LegacyMmCheckHitV3Shape(&left, &right, nullptr, false, transpose != 0, ge::FORMAT_ND, split_k != 0,
                                 cube_workers, std::string(soc));
}
