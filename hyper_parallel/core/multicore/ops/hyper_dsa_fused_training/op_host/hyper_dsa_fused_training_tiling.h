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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_HOST_HYPER_DSA_FUSED_TRAINING_TILING_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_HOST_HYPER_DSA_FUSED_TRAINING_TILING_H_
#include "register/tilingdata_base.h"
#include "../op_kernel/kl/arch22/sparse_lightning_indexer_grad_kl_loss_tiling.h"
#include "li_tiling_data.h"   // NOLINT(build/include_subdir)
#include "sfa_tiling_data.h"  // NOLINT(build/include_subdir)

namespace optiling {
REGISTER_TILING_DATA_CLASS(LITilingDataOp, LITilingData)
REGISTER_TILING_DATA_CLASS(SparseFlashAttentionTilingDataMlaOp, SparseFlashAttentionTilingDataMla)
BEGIN_TILING_DATA_DEF(DsaFusedTrainingTilingData)
TILING_DATA_FIELD_DEF_STRUCT(LITilingData, li);
TILING_DATA_FIELD_DEF_STRUCT(SparseFlashAttentionTilingDataMla, sfa);
TILING_DATA_FIELD_DEF(uint64_t, withKl);
TILING_DATA_FIELD_DEF_ARR(uint64_t, (sizeof(SparseLightningIndexerGradKLLossTilingData) + 7) / 8, kl);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(HyperDsaFusedTraining, DsaFusedTrainingTilingData)
}  // namespace optiling
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_HOST_HYPER_DSA_FUSED_TRAINING_TILING_H_
