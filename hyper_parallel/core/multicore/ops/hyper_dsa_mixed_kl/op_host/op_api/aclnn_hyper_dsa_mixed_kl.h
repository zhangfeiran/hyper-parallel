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
#ifndef HYPER_PARALLEL_MULTICORE_ACLNN_HYPER_DSA_MIXED_KL_H_
#define HYPER_PARALLEL_MULTICORE_ACLNN_HYPER_DSA_MIXED_KL_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
extern "C" {
ACLNN_API aclnnStatus aclnnHyperDsaMixedKlGetWorkspaceSize(
  const aclTensor *query, const aclTensor *key, const aclTensor *indexQuery, const aclTensor *indexKey,
  const aclTensor *weight, const aclTensor *indices, const aclTensor *maximum, const aclTensor *sum,
  const aclTensor *queryRope, const aclTensor *keyRope, const aclIntArray *actualQuery, const aclIntArray *actualKey,
  const aclTensor *config, const aclTensor *trace, const aclTensor *retained, double scale, int64_t phase,
  const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradWeight, const aclTensor *loss,
  uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnHyperDsaMixedKl(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream);
}
#endif  // HYPER_PARALLEL_MULTICORE_ACLNN_HYPER_DSA_MIXED_KL_H_
