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
#include "aclnn_hyper_dsa_fp32_index_kl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
namespace l0op {
OP_TYPE_REGISTER(HyperDsaFp32IndexKl);
}
using l0op::HyperDsaFp32IndexKlOpTypeId;
extern "C" aclnnStatus aclnnHyperDsaFp32IndexKlGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weight, const aclTensor *teacher,
    const aclTensor *globalLogits, const aclTensor *indices, int64_t phase, const aclTensor *localLogits,
    const aclTensor *gradQuery, const aclTensor *gradKey, const aclTensor *gradWeight, const aclTensor *loss,
    const aclTensor *trace, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  auto holder = CREATE_EXECUTOR();
  CHECK_RET(holder.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // Torch passes view geometry separately from flat storage, even for contiguous tensors.
  const aclTensor *original[] = {query, key, weight, teacher, globalLogits, indices,
                                 localLogits, gradQuery, gradKey, gradWeight, loss, trace};
  const aclTensor *dense[12];
  for (size_t index = 0; index < 12; ++index) {
    dense[index] = l0op::Contiguous(original[index], holder.get());
    CHECK_RET(dense[index] != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto launch = [&](aclOpExecutor *executor) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
      HyperDsaFp32IndexKl, OP_INPUT(dense[0], dense[1], dense[2], dense[3], dense[4], dense[5]),
      OP_OUTPUT(const_cast<aclTensor *>(dense[6]), const_cast<aclTensor *>(dense[7]),
                const_cast<aclTensor *>(dense[8]), const_cast<aclTensor *>(dense[9]),
                const_cast<aclTensor *>(dense[10]), const_cast<aclTensor *>(dense[11])), OP_ATTR(phase));
  };
  const auto status = launch(holder.get());
  CHECK_RET(status == ACL_SUCCESS, status);
  for (size_t index = 6; index < 12; ++index) {
    CHECK_RET(l0op::ViewCopy(dense[index], original[index], holder.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  *workspaceSize = holder->GetWorkspaceSize();
  holder.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnHyperDsaFp32IndexKl(void *workspace, uint64_t workspaceSize,
                                               aclOpExecutor *executor, aclrtStream stream) {
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
