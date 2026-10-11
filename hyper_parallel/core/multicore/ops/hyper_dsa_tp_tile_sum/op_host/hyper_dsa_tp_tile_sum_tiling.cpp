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
#include "hyper_dsa_tp_tile_sum_tiling.h"
namespace optiling {
ge::graphStatus TilingTpTileSum(gert::TilingContext *context) {
  const auto *partial = context->GetRequiredInputShape(0);
  const auto *arena = context->GetRequiredInputShape(1);
  const auto *metadata = context->GetRequiredInputShape(2);
  const auto *output = context->GetOutputShape(0);
  if (partial == nullptr || arena == nullptr || metadata == nullptr || output == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &p = partial->GetOriginShape();
  const auto &a = arena->GetOriginShape();
  const auto &m = metadata->GetOriginShape();
  if (p.GetDimNum() != 3 || p.GetDim(0) <= 0 || p.GetDim(0) >= (1LL << 32) || p.GetDim(1) <= 0 || p.GetDim(1) > 40 ||
      p.GetDim(2) <= 0 || p.GetDim(2) > 2048 || p.GetDim(2) % 8 != 0 || a.GetDimNum() != 1 || a.GetDim(0) <= 0 ||
      m.GetDimNum() != 1 || m.GetDim(0) != 24 || output->GetOriginShape() != p) {
    return ge::GRAPH_FAILED;
  }
  TpTileSumTilingData data;
  data.set_rounds(p.GetDim(0));
  data.set_slots(p.GetDim(1));
  data.set_width(p.GetDim(2));
  data.set_arenaBytes(a.GetDim(0));
  auto *raw = context->GetRawTilingData();
  if (raw == nullptr || raw->GetCapacity() < data.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = 0;
  context->SetBlockDim(p.GetDim(1));
  context->SetScheduleMode(1);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaTpTileSum).Tiling(TilingTpTileSum);
}  // namespace optiling
