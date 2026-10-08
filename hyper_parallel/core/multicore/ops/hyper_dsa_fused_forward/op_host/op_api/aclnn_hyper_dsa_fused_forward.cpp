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
#include "aclnn_hyper_dsa_fused_forward.h"  // NOLINT(build/include_subdir)
#include <array>
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(HyperDsaFusedForward);
}
using l0op::HyperDsaFusedForwardOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaFusedForwardGetWorkspaceSize(
    const aclTensor *indexQuery, const aclTensor *indexKey, const aclTensor *query,
    const aclTensor *compressed, const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *weights, const aclTensor *lengths, const aclTensor *config,
    const aclTensor *trace, const aclTensor *retained, double scale,
    const aclTensor *indices, const aclTensor *values, const aclTensor *attention,
    const aclTensor *maximum, const aclTensor *sum,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnHyperDsaFusedForward, DFX_IN(indexQuery, query, config), DFX_OUT(indices, attention));
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  std::array<const aclTensor *, 8> inputs{indexQuery, indexKey, query, compressed,
                                         queryRope, keyRope, weights, lengths};
  std::array<const aclTensor *, 5> outputs{indices, values, attention, maximum, sum};
  for (auto &input : inputs) {
    input = l0op::Contiguous(input, owner.get());
    CHECK_RET(input != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (auto &output : outputs) {
    output = l0op::Contiguous(output, owner.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  const float scaleValue = static_cast<float>(scale);
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
        HyperDsaFusedForward,
        OP_INPUT(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                 inputs[6], inputs[7], config, trace, retained),
        OP_OUTPUT(const_cast<aclTensor *>(outputs[0]), const_cast<aclTensor *>(outputs[1]),
                  const_cast<aclTensor *>(outputs[2]), const_cast<aclTensor *>(outputs[3]),
                  const_cast<aclTensor *>(outputs[4])),
        OP_ATTR(scaleValue));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  const std::array<const aclTensor *, 5> destinations{indices, values, attention, maximum, sum};
  for (size_t index = 0; index < outputs.size(); ++index) {
    CHECK_RET(l0op::ViewCopy(outputs[index], destinations[index], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnHyperDsaFusedForward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaFusedForward);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
