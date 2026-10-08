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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_MIXED_INDEXER_OP_HOST_OP_API_ACLNN_HYPER_DSA_MIXED_INDEXER_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_MIXED_INDEXER_OP_HOST_OP_API_ACLNN_HYPER_DSA_MIXED_INDEXER_H_
#include "aclnn/aclnn_base.h"
// CANN exposes this public utility header at the SDK include root.
#include "aclnn_util.h"  // NOLINT(build/include_subdir)

#ifdef __cplusplus
extern "C" {
#endif
ACLNN_API aclnnStatus aclnnHyperDsaMixedIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualQuery, const aclTensor *actualKv, const aclTensor *config,
    const aclTensor *trace, const aclTensor *retained, int64_t mergePhase,
    const aclTensor *indices, const aclTensor *values, uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnHyperDsaMixedIndexer(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_MIXED_INDEXER_OP_HOST_OP_API_ACLNN_HYPER_DSA_MIXED_INDEXER_H_
