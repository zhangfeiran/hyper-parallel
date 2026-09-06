/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_hyper_mega_moe_grad.h"

#include "aclnn_kernels/reshape.h"
#include "hyper_mega_moe_grad.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"

#define RT_MEMORY_POLICY_HUGE_PAGE_FIRST_P2P (0x2000u)
#define RT_MEMORY_POLICY_HUGE_PAGE_ONLY_P2P (0x4000u)
#define RT_MEMORY_POLICY_DEFAULT_PAGE_ONLY_P2P (0x8000u)
#define RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY (0x10000u)
#define RT_MEMORY_POLICY_HUGE1G_PAGE_ONLY_P2P (0x20000u)

#ifdef __cplusplus
extern "C" {
#endif

aclTensor *CreateContiguousTensorList(const aclTensor *tensorList, aclOpExecutor *executor) {
  op::Shape shape;
  const aclTensor *inputTensor = tensorList;
  op::Shape viewShape = inputTensor->GetViewShape();
  uint32_t viewShapeDimsNum = viewShape.GetDimNum();
  shape.SetScalar();
  // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
  for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
    shape.AppendDim(viewShape.GetDim(i));
  }
  // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last
  // dim.
  shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
  shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2));  // 2:the second last dim.
  aclTensor *tensor =
    executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());  // use executor to create tensor
  tensor->SetStorageFormat(inputTensor->GetStorageFormat());
  return tensor;
}

static void SetTransposedTensorListContiguous(
    HyperMegaMoeGradParams &params, aclOpExecutor *executorPtr,
    bool transposePermuteOut) {
  aclTensor *hidden = CreateContiguousTensorList(params.hidden, executorPtr);
  params.hidden = hidden;

  aclTensor *w1 = CreateContiguousTensorList(params.w1, executorPtr);
  params.w1 = w1;

  if (transposePermuteOut) {
    aclTensor *permute_out = CreateContiguousTensorList(params.permute_out, executorPtr);
    params.permute_out = permute_out;
  }

  aclTensor *weight = CreateContiguousTensorList(params.weight, executorPtr);
  params.weight = weight;
}

bool IsTransposeLastTwoDims(const aclTensor *tensor) {
  auto shape = tensor->GetViewShape();
  int64_t dim1 = shape.GetDimNum() - 1;
  int64_t dim2 = shape.GetDimNum() - 2;
  auto strides = tensor->GetViewStrides();
  if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
    int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
    for (int64_t batchDim = shape.GetDimNum() - 3; batchDim >= 0; batchDim--) {
      if (strides[batchDim] != tmpNxD) {
        return false;
      }
      tmpNxD *= shape.GetDim(batchDim);
    }
    return true;
  }
  return false;
}

ACLNN_API aclnnStatus aclnnHyperMegaMoeGradGetWorkspaceSize(
  const aclTensor *dispatch_target, const aclTensor *dispatch_target_off, const aclTensor *dy,
  const aclTensor *dispatch_src_off, const aclTensor *dispatch_size, const aclTensor *hidden,
  const aclTensor *hidden_dw,
  const aclTensor *weight,  // w2
  const aclTensor *y,       // act_grad_y
  const aclTensor *gate, const aclTensor *grad_gate, const aclTensor *w1, const aclTensor *gate_dx,
  const aclTensor *grad_x, const aclTensor *combine_target_off, const aclTensor *combine_src_off,
  const aclTensor *combine_size, const aclTensor *permute_out, const aclTensor *gate_dw, const aclTensor *group_list,
  const aclTensor *act_grad_tiling, const aclTensor *gate_grad_tiling, const aclTensor *w1_grad_tiling,
  const aclTensor *w2_grad_tiling, const aclTensor *swiglu_grad_tiling, const aclTensor *gmm_workspace,
  const aclTensor *swiglu_grad_workspace, const aclTensor *runtime_config, const aclTensor *all_event_counters,
  const aclTensor *permute_src_index, const aclTensor *route_weight, const aclTensor *reduce_route_index,
  int64_t rankId, int64_t ep, int64_t expert_num, int64_t hidden_size, int64_t seq_size,
  int64_t receive_capacity, int64_t wave_count, int64_t cycle_profile, int64_t home_experts,
  uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(
    aclnnHyperMegaMoeGrad,
    DFX_IN(dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size, hidden, hidden_dw, weight, y,
           gate, grad_gate, w1, gate_dx, grad_x, combine_target_off, combine_src_off, combine_size, permute_out,
           gate_dw, group_list, act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling, swiglu_grad_tiling,
           gmm_workspace, swiglu_grad_workspace, runtime_config, all_event_counters, permute_src_index, route_weight,
           reduce_route_index, rankId, ep, expert_num, hidden_size, seq_size, receive_capacity,
           wave_count, cycle_profile, home_experts),
    DFX_OUT(dispatch_target, hidden_dw, y, grad_gate, gate_dx, grad_x, permute_out, gate_dw));

  auto uniqueExecutor = CREATE_EXECUTOR();
  aclOpExecutor *executorPtr = uniqueExecutor.get();
  CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  HyperMegaMoeGradParams params{hidden, w1, permute_out, weight};
  // Multi-wave dW1 reads its wave-local hidden copy from position 12, while
  // position 17 remains the symmetric Combine staging buffer. Only the
  // ordinary graph needs the historical transposed dW1 input view.
  SetTransposedTensorListContiguous(params, executorPtr, wave_count <= 1);

  // Invoke the underlying operator.
  auto bwd_output = l0op::HyperMegaMoeGrad(
    dispatch_target, dispatch_target_off, dy, dispatch_src_off, dispatch_size, params.hidden, hidden_dw, params.weight,
    y, gate, grad_gate, params.w1, gate_dx, grad_x, combine_target_off, combine_src_off, combine_size,
    params.permute_out, gate_dw, group_list, act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
    swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace, runtime_config, all_event_counters, permute_src_index,
    route_weight, reduce_route_index, rankId, ep, expert_num, hidden_size, seq_size,
    receive_capacity, wave_count, cycle_profile, home_experts, executorPtr);
  bool bwd_output_success =
    std::all_of(bwd_output.begin(), bwd_output.end(), [](const aclTensor *ptr) { return ptr != nullptr; });
  CHECK_RET(bwd_output_success, ACLNN_ERR_INNER_NULLPTR);

  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnHyperMegaMoeGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnHyperMegaMoeGrad);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
