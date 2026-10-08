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
// The same adjacent API header is installed at the vendor include root.
#include "aclnn_hyper_dsa_mixed_indexer.h"  // NOLINT(build/include_subdir)
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(HyperDsaMixedIndexer);
}
using l0op::HyperDsaMixedIndexerOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaMixedIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualQuery, const aclTensor *actualKv, const aclTensor *config,
    const aclTensor *trace, const aclTensor *retained, int64_t mergePhase,
    const aclTensor *indices, const aclTensor *values, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperDsaMixedIndexer, DFX_IN(query, key, weights, config), DFX_OUT(indices, values));
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  auto *queryCont = l0op::Contiguous(query, owner.get());
  CHECK_RET(queryCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *keyCont = l0op::Contiguous(key, owner.get());
  CHECK_RET(keyCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *weightsCont = l0op::Contiguous(weights, owner.get());
  CHECK_RET(weightsCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *actualQueryCont = l0op::Contiguous(actualQuery, owner.get());
  CHECK_RET(actualQueryCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *actualKvCont = l0op::Contiguous(actualKv, owner.get());
  CHECK_RET(actualKvCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *indicesCont = l0op::Contiguous(indices, owner.get());
  CHECK_RET(indicesCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *valuesCont = l0op::Contiguous(values, owner.get());
  CHECK_RET(valuesCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor *blockTable = nullptr;
  auto *indicesMutable = const_cast<aclTensor *>(indicesCont);
  auto *valuesMutable = const_cast<aclTensor *>(valuesCont);
  const char *layout = "TND";
  const int64_t sparseCount = 2048;
  const int64_t mode = 3;
  const int64_t preTokens = INT64_MAX;
  const int64_t nextTokens = INT64_MAX;
  const bool returnValues = true;
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
      HyperDsaMixedIndexer,
      OP_INPUT(queryCont, keyCont, weightsCont, actualQueryCont, actualKvCont, blockTable, config, trace, retained),
      OP_OUTPUT(indicesMutable, valuesMutable),
      OP_ATTR(layout, layout, sparseCount, mode, preTokens, nextTokens, returnValues, mergePhase));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  CHECK_RET(l0op::ViewCopy(indicesCont, indices, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(l0op::ViewCopy(valuesCont, values, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnHyperDsaMixedIndexer(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaMixedIndexer);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
