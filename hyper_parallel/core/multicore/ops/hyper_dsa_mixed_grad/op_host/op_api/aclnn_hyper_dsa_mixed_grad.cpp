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
#include "aclnn_hyper_dsa_mixed_grad.h"  // NOLINT(build/include_subdir)
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(HyperDsaMixedGrad);
}
using l0op::HyperDsaMixedGradOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaMixedGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *indices, const aclTensor *gradOut, const aclTensor *out,
    const aclTensor *maximum, const aclTensor *sum, const aclTensor *actualQuery,
    const aclTensor *actualKv, const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *config, const aclTensor *trace, const aclTensor *retained,
    double scale, int64_t phase,
    const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue,
    const aclTensor *gradQueryRope, const aclTensor *gradKeyRope,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperDsaMixedGrad, DFX_IN(query, key, value, indices), DFX_OUT(gradQuery, gradKey, gradValue));
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  auto *queryCont = l0op::Contiguous(query, owner.get());
  CHECK_RET(queryCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *keyCont = l0op::Contiguous(key, owner.get());
  CHECK_RET(keyCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *valueCont = l0op::Contiguous(value, owner.get());
  CHECK_RET(valueCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *indicesCont = l0op::Contiguous(indices, owner.get());
  CHECK_RET(indicesCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradOutCont = l0op::Contiguous(gradOut, owner.get());
  CHECK_RET(gradOutCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *outCont = l0op::Contiguous(out, owner.get());
  CHECK_RET(outCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *maximumCont = l0op::Contiguous(maximum, owner.get());
  CHECK_RET(maximumCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *sumCont = l0op::Contiguous(sum, owner.get());
  CHECK_RET(sumCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *actualQueryCont = l0op::Contiguous(actualQuery, owner.get());
  CHECK_RET(actualQueryCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *actualKvCont = l0op::Contiguous(actualKv, owner.get());
  CHECK_RET(actualKvCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *queryRopeCont = l0op::Contiguous(queryRope, owner.get());
  CHECK_RET(queryRopeCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *keyRopeCont = l0op::Contiguous(keyRope, owner.get());
  CHECK_RET(keyRopeCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradQueryCont = l0op::Contiguous(gradQuery, owner.get());
  CHECK_RET(gradQueryCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradKeyCont = l0op::Contiguous(gradKey, owner.get());
  CHECK_RET(gradKeyCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradValueCont = l0op::Contiguous(gradValue, owner.get());
  CHECK_RET(gradValueCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradQueryRopeCont = l0op::Contiguous(gradQueryRope, owner.get());
  CHECK_RET(gradQueryRopeCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradKeyRopeCont = l0op::Contiguous(gradKeyRope, owner.get());
  CHECK_RET(gradKeyRopeCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto *gradQueryMutable = const_cast<aclTensor *>(gradQueryCont);
  auto *gradKeyMutable = const_cast<aclTensor *>(gradKeyCont);
  auto *gradValueMutable = const_cast<aclTensor *>(gradValueCont);
  auto *gradQueryRopeMutable = const_cast<aclTensor *>(gradQueryRopeCont);
  auto *gradKeyRopeMutable = const_cast<aclTensor *>(gradKeyRopeCont);
  const float scaleValue = static_cast<float>(scale);
  const int64_t blockSize = 1;
  const char *layout = "TND";
  const int64_t mode = 3;
  const int64_t preTokens = INT64_MAX;
  const int64_t nextTokens = INT64_MAX;
  const bool deterministic = false;
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
      HyperDsaMixedGrad,
      OP_INPUT(queryCont, keyCont, indicesCont, gradOutCont, outCont, maximumCont, sumCont, valueCont,
               actualQueryCont, actualKvCont, queryRopeCont, keyRopeCont, config, trace, retained),
      OP_OUTPUT(gradQueryMutable, gradKeyMutable, gradValueMutable, gradQueryRopeMutable, gradKeyRopeMutable),
      OP_ATTR(scaleValue, blockSize, layout, mode, preTokens, nextTokens, deterministic, phase));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  CHECK_RET(l0op::ViewCopy(gradQueryCont, gradQuery, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(l0op::ViewCopy(gradKeyCont, gradKey, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(l0op::ViewCopy(gradValueCont, gradValue, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(l0op::ViewCopy(gradQueryRopeCont, gradQueryRope, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(l0op::ViewCopy(gradKeyRopeCont, gradKeyRope, owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnHyperDsaMixedGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaMixedGrad);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
