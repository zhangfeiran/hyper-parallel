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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FP32_INDEX_KL_OP_HOST_OP_API_ACLNN_HYPER_DSA_FP32_INDEX_KL_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FP32_INDEX_KL_OP_HOST_OP_API_ACLNN_HYPER_DSA_FP32_INDEX_KL_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnHyperDsaFp32IndexKlGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weight, const aclTensor *teacher,
    const aclTensor *globalLogits, const aclTensor *indices, int64_t phase, const aclTensor *localLogits,
    const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradWeight, const aclTensor *loss,
    const aclTensor *trace, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnHyperDsaFp32IndexKl(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FP32_INDEX_KL_OP_HOST_OP_API_ACLNN_HYPER_DSA_FP32_INDEX_KL_H_
