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
#include "aclnn_hyper_dsa_cp_attention.h"  // NOLINT(build/include_subdir)
#include <array>
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
namespace l0op {
OP_TYPE_REGISTER(HyperDsaCpAttention);
}
using l0op::HyperDsaCpAttentionOpTypeId;

extern "C" aclnnStatus aclnnHyperDsaCpAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *compressed, const aclTensor *queryRope, const aclTensor *keyRope,
    const aclTensor *indices, const aclTensor *lengths, const aclTensor *config, const aclTensor *trace,
    const aclTensor *arena, const aclTensor *metadata, const aclTensor *requests, const aclTensor *transportTrace,
    double scale, const aclTensor *attention, const aclTensor *maximum, const aclTensor *sum,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnHyperDsaCpAttention, DFX_IN(query, indices, config), DFX_OUT(attention));
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  auto owner = CREATE_EXECUTOR();
  CHECK_RET(owner.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const std::array<const aclTensor *, 6> originals{query, compressed, queryRope, keyRope, indices, lengths};
  const std::array<const aclTensor *, 3> destinations{attention, maximum, sum};
  auto inputs = originals;
  auto outputs = destinations;
  for (auto &tensor : inputs) {
    tensor = l0op::Contiguous(tensor, owner.get());
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (auto &tensor : outputs) {
    tensor = l0op::Contiguous(tensor, owner.get());
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  const float scaleValue = static_cast<float>(scale);
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(HyperDsaCpAttention,
      OP_INPUT(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
               config, trace, arena, metadata, requests, transportTrace),
      OP_OUTPUT(const_cast<aclTensor *>(outputs[0]), const_cast<aclTensor *>(outputs[1]),
                const_cast<aclTensor *>(outputs[2])), OP_ATTR(scaleValue));
  };
  const auto status = launch(owner.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  for (size_t index = 0; index < outputs.size(); ++index) {
    CHECK_RET(l0op::ViewCopy(outputs[index], destinations[index], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  for (size_t index : {1U, 3U}) {
    CHECK_RET(l0op::ViewCopy(inputs[index], originals[index], owner.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  *workspaceSize = owner->GetWorkspaceSize();
  owner.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnHyperDsaCpAttention(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperDsaCpAttention);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
