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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_OP_API_ACLNN_HYPER_DSA_FUSED_GRAD_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_OP_API_ACLNN_HYPER_DSA_FUSED_GRAD_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"  // NOLINT(build/include_subdir)
extern "C" {
ACLNN_API aclnnStatus aclnnHyperDsaFusedGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *indices, const aclTensor *gradOut, const aclTensor *out,
    const aclTensor *maximum, const aclTensor *sum, const aclTensor *actualQuery,
    const aclTensor *actualKv, const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *config, const aclTensor *trace, const aclTensor *retained, double scale,
    const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue,
    const aclTensor *gradQueryRope, const aclTensor *gradKeyRope,
    const aclTensor *arena, const aclTensor *metadata, const aclTensor *requests,
    const aclTensor *transportTrace, const aclTensor *ownerGradient, const aclTensor *partials,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnHyperDsaFusedGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
}
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_GRAD_OP_HOST_OP_API_ACLNN_HYPER_DSA_FUSED_GRAD_H_
