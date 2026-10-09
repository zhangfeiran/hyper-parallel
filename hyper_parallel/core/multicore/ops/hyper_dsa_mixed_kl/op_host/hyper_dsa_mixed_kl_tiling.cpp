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
#include "register/op_impl_registry.h"
#include "arch22/sparse_lightning_indexer_grad_kl_loss_tiling_general.h"

namespace optiling {
ge::graphStatus TilingHyperDsaMixedKl(gert::TilingContext *context) {
  if (context == nullptr || context->GetAttrs() == nullptr || context->GetDeterministic() == 1) {
    return ge::GRAPH_FAILED;
  }
  const auto *phase = context->GetAttrs()->GetInt(5);
  const auto *config = context->GetRequiredInputShape(12);
  const auto *trace = context->GetRequiredInputShape(13);
  const auto *retained = context->GetRequiredInputShape(14);
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  if (phase == nullptr || *phase < 0 || *phase > 2 || platform.GetCoreNumAic() != 20 ||
      platform.GetCoreNumAiv() != 40 || config == nullptr || config->GetStorageShape().GetShapeSize() != 4 ||
      trace == nullptr || trace->GetStorageShape().GetShapeSize() != 1280 || retained == nullptr) {
    OP_LOGE(context->GetNodeName(), "mixed KL runtime/platform invalid: phase=%ld AIC=%u AIV=%u config=%ld trace=%ld",
            phase == nullptr ? -1 : *phase, platform.GetCoreNumAic(), platform.GetCoreNumAiv(),
            config == nullptr ? -1 : config->GetStorageShape().GetShapeSize(),
            trace == nullptr ? -1 : trace->GetStorageShape().GetShapeSize());
    return ge::GRAPH_FAILED;
  }
  SparseLightningIndexerGradKLLossTilingBase tile(context);
  if (tile.DoTiling() != ge::GRAPH_SUCCESS) {
    OP_LOGE(context->GetNodeName(), "locked KL child tiling failed");
    return ge::GRAPH_FAILED;
  }
  auto *data = context->GetTilingData<SparseLightningIndexerGradKLLossTilingData>();
  auto &base = data->baseParams;
  if (base.dSizeQuery != 512 || base.dSizeQueryIndex != 128 || base.kSize != 2048 || base.gSizeQueryIndex != 64 ||
      (base.gSizeQuery != 32 && base.gSizeQuery != 64)) {
    OP_LOGE(context->GetNodeName(), "mixed KL unsupported geometry: C=%u Di=%u K=%u Hi=%u H=%u", base.dSizeQuery,
            base.dSizeQueryIndex, base.kSize, base.gSizeQueryIndex, base.gSizeQuery);
    return ge::GRAPH_FAILED;
  }
  const uint64_t coreBytes = 2 * (2048 * 576 * 2 + 2048 * 128 * 2 + base.gSizeQuery * 2048 * 4 + 64 * 2048 * 4 * 2 +
                                  2048 * 2 * 4 + 2048 * 128 * 4);
  const uint64_t required = coreBytes * 20 + 512 + base.t2Size * 128 * 4;
  if (retained->GetStorageShape().GetShapeSize() < static_cast<int64_t>(required)) {
    OP_LOGE(context->GetNodeName(), "mixed KL retained workspace capacity %ld is below required %lu",
            retained->GetStorageShape().GetShapeSize(), required);
    return ge::GRAPH_FAILED;
  }
  data->mixedPhase = *phase;
  context->SetBlockDim(20);
  context->SetTilingKey(0);
  context->GetRawTilingData()->SetDataSize(sizeof(*data));
  context->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024;
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaMixedKl).Tiling(TilingHyperDsaMixedKl);
}  // namespace optiling
