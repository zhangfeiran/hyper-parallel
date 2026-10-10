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
#include "aclnn_hyper_dsa_index_grad_return.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
namespace l0op {
OP_TYPE_REGISTER(HyperDsaIndexGradReturn);
}
using l0op::HyperDsaIndexGradReturnOpTypeId;
extern "C" aclnnStatus aclnnHyperDsaIndexGradReturnGetWorkspaceSize(const aclTensor *gradient, const aclTensor *arena,
                                                                    const aclTensor *metadata,
                                                                    const aclTensor *requests, const aclTensor *trace,
                                                                    int64_t capacity, const aclTensor *owner,
                                                                    uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  auto holder = CREATE_EXECUTOR();
  CHECK_RET(holder.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  const auto *denseGradient = l0op::Contiguous(gradient, holder.get());
  const auto *denseRequests = l0op::Contiguous(requests, holder.get());
  const auto *denseOwner = l0op::Contiguous(owner, holder.get());
  CHECK_RET(denseGradient != nullptr && denseRequests != nullptr && denseOwner != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(HyperDsaIndexGradReturn,
                                       OP_INPUT(denseGradient, arena, metadata, denseRequests, trace),
                                       OP_OUTPUT(const_cast<aclTensor *>(denseOwner)), OP_ATTR(capacity));
  };
  const auto status = launch(holder.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  CHECK_RET(l0op::ViewCopy(denseOwner, owner, holder.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  *workspaceSize = holder->GetWorkspaceSize();
  holder.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnHyperDsaIndexGradReturn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                    aclrtStream stream) {
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
