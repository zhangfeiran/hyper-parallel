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
#include "log/log.h"
#include "hyper_dsa_index_grad_return_tiling.h"
namespace optiling {
ge::graphStatus TilingIndexGradReturn(gert::TilingContext *context) {
  const auto *gradient = context->GetRequiredInputShape(0);
  const auto *arena = context->GetRequiredInputShape(1);
  const auto *metadata = context->GetRequiredInputShape(2);
  const auto *requests = context->GetRequiredInputShape(3);
  const auto *trace = context->GetRequiredInputShape(4);
  const auto *owner = context->GetOutputShape(0);
  const auto *capacity = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (gradient == nullptr || arena == nullptr || metadata == nullptr || requests == nullptr || trace == nullptr ||
      owner == nullptr || capacity == nullptr || *capacity <= 0) {
    OP_LOGE(context->GetNodeName(), "index-K return requires all shape descriptors and positive capacity");
    return ge::GRAPH_FAILED;
  }
  // The ACLNN adapter normalizes flattened ND storage descriptors before applying the logical shape contract.
  const auto &g = gradient->GetOriginShape();
  const auto &a = arena->GetOriginShape();
  const auto &m = metadata->GetOriginShape();
  const auto &r = requests->GetOriginShape();
  const auto &t = trace->GetOriginShape();
  const auto &o = owner->GetOriginShape();
  if (g.GetDimNum() != 2 || g.GetDim(0) <= 0 || g.GetDim(1) != 128 || a.GetDimNum() != 1 || a.GetDim(0) <= 0 ||
      m.GetDimNum() != 1 || (m.GetDim(0) != 18 && m.GetDim(0) != 27) || r.GetDimNum() != 2 || r.GetDim(1) != 4 ||
      r.GetDim(0) <= 0 || t.GetDimNum() != 1 || t.GetDim(0) != 32 || o.GetDimNum() != 2 || o.GetDim(0) != *capacity ||
      o.GetDim(1) != 128) {
    const auto dim = [](const auto &shape, size_t axis) {
      return shape.GetDimNum() > axis ? shape.GetDim(axis) : int64_t{-1};
    };
    OP_LOGE(context->GetNodeName(),
            "index-K shapes: gradient=[%ld,%ld], arena=%ld, metadata=%ld, requests=[%ld,%ld], trace=%ld, "
            "owner=[%ld,%ld], capacity=%ld",
            dim(g, 0), dim(g, 1), dim(a, 0), dim(m, 0), dim(r, 0), dim(r, 1), dim(t, 0), dim(o, 0), dim(o, 1),
            *capacity);
    return ge::GRAPH_FAILED;
  }
  IndexGradReturnTilingData data;
  data.set_tokens(g.GetDim(0));
  data.set_capacity(*capacity);
  data.set_arenaBytes(a.GetDim(0));
  data.set_runs(r.GetDim(0));
  auto *raw = context->GetRawTilingData();
  if (raw == nullptr || raw->GetCapacity() < data.GetDataSize()) {
    OP_LOGE(context->GetNodeName(), "index-K raw tiling buffer has insufficient capacity for %zu bytes",
            static_cast<size_t>(data.GetDataSize()));
    return ge::GRAPH_FAILED;
  }
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = 0;
  context->SetBlockDim(1);
  context->SetScheduleMode(1);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaIndexGradReturn).Tiling(TilingIndexGradReturn);
}  // namespace optiling
