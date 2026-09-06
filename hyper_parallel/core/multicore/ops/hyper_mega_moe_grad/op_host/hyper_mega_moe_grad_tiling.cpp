/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file hyper_mega_moe_grad_tiling.cpp
 */
#include "hyper_mega_moe_grad_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include "log/log.h"
#include "../op_kernel/hyper_mega_moe_grad_tiling_key.h"

namespace optiling {
const uint64_t BLOCK_SIZE = 32;
const uint64_t BUFFER_NUM = 2;

static ge::graphStatus GetIntermediateSize(gert::TilingContext *context, int64_t hiddenSize, int64_t expertNum,
                                           int64_t ep, int64_t homeExperts, int64_t *intermediateSize) {
  constexpr size_t kWeight2Index = 7;
  constexpr size_t kWeight1Index = 11;
  auto weight2Shape = context->GetInputShape(kWeight2Index);
  auto weight1Shape = context->GetInputShape(kWeight1Index);
  OP_CHECK_NULL_WITH_CONTEXT(context, weight2Shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, weight1Shape);
  const auto &weight2 = weight2Shape->GetOriginShape();
  const auto &weight1 = weight1Shape->GetOriginShape();
  if (ep <= 0 || expertNum <= 0 || expertNum % ep != 0 || homeExperts < 0 || homeExperts >= expertNum / ep) {
    OP_LOGE(context->GetNodeName(), "home_experts must be zero or a proper physical-expert prefix");
    return ge::GRAPH_FAILED;
  }
  const int64_t weightExperts = homeExperts > 0 ? homeExperts : expertNum / ep;
  if (weight2.GetDimNum() != 3 || weight1.GetDimNum() != 3 ||
      weight2.GetDim(0) != weightExperts || weight1.GetDim(0) != weightExperts || weight2.GetDim(1) != hiddenSize ||
      weight1.GetDim(2) != hiddenSize || weight1.GetDim(1) != 2 * weight2.GetDim(2) || weight2.GetDim(2) <= 0) {
    OP_LOGE(context->GetNodeName(), "HyperMegaMoeGrad weights must have shapes [E,H,I] and [E,2I,H]");
    return ge::GRAPH_FAILED;
  }
  *intermediateSize = weight2.GetDim(2);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto coreNum = ascendcPlatform.GetCoreNumAic();
  context->SetBlockDim(coreNum);
  // OP_LOGE(context->GetNodeName(), "coreNum:%d", coreNum);

  auto input0Desc = context->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
  auto attr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attr);
  auto cycleProfileAttr = attr->GetAttrPointer<int64_t>(7);
  OP_CHECK_NULL_WITH_CONTEXT(context, cycleProfileAttr);
  uint32_t cycleProfile = *cycleProfileAttr == 0 ? 0U : 1U;
  ge::DataType input0DType = input0Desc->GetDataType();
  auto weightGrad1Desc = context->GetInputDesc(18);
  auto weightGrad2Desc = context->GetInputDesc(6);
  OP_CHECK_NULL_WITH_CONTEXT(context, weightGrad1Desc);
  OP_CHECK_NULL_WITH_CONTEXT(context, weightGrad2Desc);
  ge::DataType weightGradDType = weightGrad1Desc->GetDataType();
  if (weightGrad2Desc->GetDataType() != weightGradDType ||
      (weightGradDType != input0DType && weightGradDType != ge::DT_FLOAT)) {
    OP_LOGE(context->GetNodeName(), "Weight gradients must share the activation dtype or both use float32");
    return ge::GRAPH_FAILED;
  }
  if (input0DType == ge::DT_FLOAT16) {
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(
      static_cast<uint32_t>(1), static_cast<uint32_t>(1), static_cast<uint32_t>(weightGradDType),
      0UL, 0UL, static_cast<uint32_t>(0),
      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0),
      cycleProfile);
    context->SetTilingKey(tilingKey_);
  } else if (input0DType == ge::DT_BF16) {
    uint64_t tilingKey_ =
      GET_TPL_TILING_KEY(static_cast<uint32_t>(27), static_cast<uint32_t>(27),
                         static_cast<uint32_t>(weightGradDType), 0UL, 0UL,
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), cycleProfile);
    context->SetTilingKey(tilingKey_);
  } else {
    OP_LOGE(context->GetNodeName(), "Input0[dispatch_target]:%s is only support float16, bfloat16",
            Ops::Base::ToString(static_cast<ge::DataType>(input0DType)).c_str());
    return ge::GRAPH_FAILED;
  }

  HyperMegaMoeGradTilingData tiling;
  if (attr != nullptr) {
    tiling.set_rankId((*(attr->GetAttrPointer<int64_t>(0))));
    tiling.set_ep((*(attr->GetAttrPointer<int64_t>(1))));
    tiling.set_expertNum((*(attr->GetAttrPointer<int64_t>(2))));
    tiling.set_hiddenSize((*(attr->GetAttrPointer<int64_t>(3))));
    tiling.set_seqSize((*(attr->GetAttrPointer<int64_t>(4))));
    tiling.set_coreNum(static_cast<int64_t>(coreNum));
    auto homeExperts = attr->GetAttrPointer<int64_t>(8);
    OP_CHECK_NULL_WITH_CONTEXT(context, homeExperts);
    tiling.set_homeExperts(*homeExperts);
    int64_t intermediateSize = 0;
    if (GetIntermediateSize(context, tiling.get_hiddenSize(), tiling.get_expertNum(), tiling.get_ep(),
                            *homeExperts, &intermediateSize) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    tiling.set_intermediateSize(intermediateSize);
    tiling.set_receiveCapacity((*(attr->GetAttrPointer<int64_t>(5))));
    tiling.set_waveCount((*(attr->GetAttrPointer<int64_t>(6))));
  }
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 99615232;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareTilingFunc(gert::TilingParseContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

struct HyperMegaMoeGradCompileInfo {};

IMPL_OP_OPTILING(HyperMegaMoeGrad)
  .Tiling(TilingFunc)
  .TilingParse<HyperMegaMoeGradCompileInfo>(TilingPrepareTilingFunc);

}  // namespace optiling
