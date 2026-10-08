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
// Adjacent upstream tiling declarations retain the pinned implementation.
#include "arch22/sparse_flash_attention_grad_tiling_bs1_basic.h"

namespace optiling {
namespace sfag {
namespace {
bool CheckRuntime(gert::TilingContext *context) {
  const auto *config = context->GetRequiredInputShape(12);
  const auto *trace = context->GetRequiredInputShape(13);
  const auto *phase = context->GetAttrs()->GetInt(7);
  return config != nullptr && config->GetStorageShape().GetShapeSize() == 4 &&
         trace != nullptr && trace->GetStorageShape().GetShapeSize() == 1280 &&
         phase != nullptr && *phase >= 0 && *phase <= 2 && context->GetDeterministic() != 1;
}

bool CheckSupport(gert::TilingContext *context) {
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const auto *query = context->GetRequiredInputShape(0);
  const auto *desc = context->GetRequiredInputDesc(0);
  return platform.GetCoreNumAic() == 20 && platform.GetCoreNumAiv() == 40 &&
         query != nullptr && query->GetStorageShape().GetDimNum() == 3 &&
         desc != nullptr && desc->GetDataType() == ge::DT_BF16;
}

ge::graphStatus MixedGradTiling(gert::TilingContext *context) {
  if (context == nullptr || context->GetAttrs() == nullptr || !CheckSupport(context) || !CheckRuntime(context)) {
    return ge::GRAPH_FAILED;
  }
  SparseFlashAttentionGradBasicTiling tile(context);
  const auto status = tile.DoTiling();
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }
  // Only complete causal TND with separate K/V and RoPE is callable in this adapter.
  if (context->GetTilingKey() != 111000) {
    return ge::GRAPH_FAILED;
  }
  // The fixed callable wrapper has one compiled entry; its mathematical variant is checked above.
  context->SetTilingKey(0);
  const auto *retained = context->GetRequiredInputShape(14);
  const uint64_t systemReserved = 32 * 1024 * 1024;
  auto *sizes = context->GetWorkspaceSizes(1);
  const uint64_t required = sizes[0] - systemReserved;
  if (retained == nullptr || retained->GetStorageShape().GetShapeSize() < static_cast<int64_t>(required)) {
    OP_LOGE(context->GetNodeName(), "mixed SFA gradient retained workspace is too small");
    return ge::GRAPH_FAILED;
  }
  sizes[0] = systemReserved;
  tile.tilingData.set_mixedPhase(*context->GetAttrs()->GetInt(7));
  tile.tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tile.tilingData.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(HyperDsaMixedGrad).Tiling(MixedGradTiling);
}  // namespace sfag
}  // namespace optiling
