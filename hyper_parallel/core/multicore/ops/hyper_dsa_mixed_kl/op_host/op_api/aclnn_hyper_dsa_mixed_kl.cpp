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
#include "aclnn_hyper_dsa_mixed_kl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

namespace l0op {
OP_TYPE_REGISTER(HyperDsaMixedKl);
}
using namespace op;
using l0op::HyperDsaMixedKlOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaMixedKlGetWorkspaceSize(
  const aclTensor *query, const aclTensor *key, const aclTensor *indexQuery, const aclTensor *indexKey,
  const aclTensor *weight, const aclTensor *indices, const aclTensor *maximum, const aclTensor *sum,
  const aclTensor *queryRope, const aclTensor *keyRope, const aclIntArray *actualQuery, const aclIntArray *actualKey,
  const aclTensor *config, const aclTensor *trace, const aclTensor *retained, double scale, int64_t phase,
  const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradWeight, const aclTensor *loss,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperDsaMixedKl, DFX_IN(query, key, indexQuery, indexKey),
                 DFX_OUT(gradQuery, gradKey, gradWeight, loss));
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const aclTensor *inputs[] = {query, key, indexQuery, indexKey, weight, indices, maximum, sum, queryRope, keyRope};
  const aclTensor *contiguousInputs[10];
  for (size_t index = 0; index < 10; ++index) {
    contiguousInputs[index] = l0op::Contiguous(inputs[index], owner.get());
    CHECK_RET(contiguousInputs[index] != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto *actualQueryConst = owner->ConvertToTensor(actualQuery, op::DataType::DT_INT64);
  auto *actualKeyConst = owner->ConvertToTensor(actualKey, op::DataType::DT_INT64);
  CHECK_RET(actualQueryConst != nullptr && actualKeyConst != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor *outputs[] = {gradQuery, gradKey, gradWeight, loss};
  const aclTensor *contiguousOutputs[4];
  for (size_t index = 0; index < 4; ++index) {
    contiguousOutputs[index] = l0op::Contiguous(outputs[index], owner.get());
    CHECK_RET(contiguousOutputs[index] != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  const float scaleValue = static_cast<float>(scale);
  const char *layout = "TND";
  const int64_t sparseMode = 3, preTokens = INT64_MAX, nextTokens = INT64_MAX;
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
      HyperDsaMixedKl,
      OP_INPUT(contiguousInputs[0], contiguousInputs[1], contiguousInputs[2], contiguousInputs[3], contiguousInputs[4],
               contiguousInputs[5], contiguousInputs[6], contiguousInputs[7], contiguousInputs[8], contiguousInputs[9],
               actualQueryConst, actualKeyConst, config, trace, retained),
      OP_OUTPUT(const_cast<aclTensor *>(contiguousOutputs[0]), const_cast<aclTensor *>(contiguousOutputs[1]),
                const_cast<aclTensor *>(contiguousOutputs[2]), const_cast<aclTensor *>(contiguousOutputs[3])),
      OP_ATTR(scaleValue, layout, sparseMode, preTokens, nextTokens, phase));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  for (size_t index = 0; index < 4; ++index) {
    CHECK_RET(l0op::ViewCopy(contiguousOutputs[index], outputs[index], owner.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
  }
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnHyperDsaMixedKl(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaMixedKl);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
