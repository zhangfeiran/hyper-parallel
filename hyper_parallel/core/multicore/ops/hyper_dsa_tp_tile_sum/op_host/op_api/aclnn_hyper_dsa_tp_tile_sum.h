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
#ifndef HYPER_PARALLEL_ACLNN_DSA_TP_TILE_SUM_H_
#define HYPER_PARALLEL_ACLNN_DSA_TP_TILE_SUM_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
extern "C" {
ACLNN_API aclnnStatus aclnnHyperDsaTpTileSumGetWorkspaceSize(const aclTensor *partial, const aclTensor *arena,
                                                             const aclTensor *metadata, const aclTensor *summed,
                                                             uint64_t *workspaceSize, aclOpExecutor **executor);
ACLNN_API aclnnStatus aclnnHyperDsaTpTileSum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);
}
#endif
