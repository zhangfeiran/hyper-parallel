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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_HYPER_DSA_FUSED_GRAD_TILING_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_HYPER_DSA_FUSED_GRAD_TILING_H_
#include "tiling/tiling_api.h"
#include "grad_tiling_data.h"  // NOLINT(build/include_subdir)
namespace optiling {
REGISTER_TILING_DATA_CLASS(HyperDsaFusedGrad, SparseFlashAttentionGradBasicTilingData)
}
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_HYPER_DSA_FUSED_GRAD_TILING_H_
