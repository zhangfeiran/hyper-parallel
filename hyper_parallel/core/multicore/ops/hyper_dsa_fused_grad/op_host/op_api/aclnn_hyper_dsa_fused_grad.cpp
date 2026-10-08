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
#include "aclnn_hyper_dsa_fused_grad.h"  // NOLINT(build/include_subdir)
#include <array>
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(HyperDsaFusedGrad);
}
using l0op::HyperDsaFusedGradOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaFusedGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *indices, const aclTensor *gradOut, const aclTensor *out,
    const aclTensor *maximum, const aclTensor *sum, const aclTensor *actualQuery,
    const aclTensor *actualKv, const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *config, const aclTensor *trace, const aclTensor *retained, double scale,
    const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradValue,
    const aclTensor *gradQueryRope, const aclTensor *gradKeyRope,
    const aclTensor *arena, const aclTensor *metadata, const aclTensor *requests,
    const aclTensor *transportTrace, const aclTensor *ownerGradient, const aclTensor *partials,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperDsaFusedGrad, DFX_IN(query, key, value, indices), DFX_OUT(gradQuery, gradKey));
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const std::array<const aclTensor *, 21> original{
      query, key, indices, gradOut, out, maximum, sum, value, actualQuery, actualKv, queryRope, keyRope,
      config, trace, retained, arena, metadata, requests, transportTrace, ownerGradient, partials};
  auto inputs = original;
  const std::array<const aclTensor *, 5> destinations{gradQuery, gradKey, gradValue, gradQueryRope, gradKeyRope};
  auto outputs = destinations;
  for (size_t index = 0; index < 12; ++index) {
    inputs[index] = l0op::Contiguous(inputs[index], owner.get());
    CHECK_RET(inputs[index] != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (size_t index : {19U, 20U}) {
    inputs[index] = l0op::Contiguous(inputs[index], owner.get());
    CHECK_RET(inputs[index] != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (auto &output : outputs) {
    output = l0op::Contiguous(output, owner.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  const float scaleValue = static_cast<float>(scale);
  const int64_t blockSize = 1, mode = 3, preTokens = INT64_MAX, nextTokens = INT64_MAX, phase = 1;
  const char *layout = "TND";
  const bool deterministic = false;
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(HyperDsaFusedGrad,
        OP_INPUT(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7],
                 inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[13], inputs[14], inputs[15],
                 inputs[16], inputs[17], inputs[18], inputs[19], inputs[20]),
        OP_OUTPUT(const_cast<aclTensor *>(outputs[0]), const_cast<aclTensor *>(outputs[1]),
                  const_cast<aclTensor *>(outputs[2]), const_cast<aclTensor *>(outputs[3]),
                  const_cast<aclTensor *>(outputs[4])),
        OP_ATTR(scaleValue, blockSize, layout, mode, preTokens, nextTokens, deterministic, phase));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  for (size_t index = 0; index < outputs.size(); ++index) {
    CHECK_RET(l0op::ViewCopy(outputs[index], destinations[index], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (size_t index : {19U, 20U}) {
    CHECK_RET(l0op::ViewCopy(inputs[index], original[index], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnHyperDsaFusedGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaFusedGrad);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
