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
#include "hyper_dsa_fp32_index_kl_tiling.h"
namespace optiling {
ge::graphStatus TilingFp32IndexKl(gert::TilingContext *context) {
  if (context == nullptr || context->GetAttrs() == nullptr || context->GetDeterministic() == 1) {
    return ge::GRAPH_FAILED;
  }
  const auto *query = context->GetRequiredInputShape(0);
  const auto *key = context->GetRequiredInputShape(1);
  const auto *phase = context->GetAttrs()->GetInt(0);
  if (query == nullptr || key == nullptr || phase == nullptr || *phase < 0 || *phase > 1) {
    return ge::GRAPH_FAILED;
  }
  const auto &q = query->GetOriginShape();
  const auto &k = key->GetOriginShape();
  if (q.GetDimNum() != 3 || q.GetDim(0) <= 0 || q.GetDim(1) <= 0 || q.GetDim(1) > 128 || q.GetDim(2) != 128 ||
      k.GetDimNum() != 2 || k.GetDim(0) <= 0 || k.GetDim(0) >= (1LL << 31) || k.GetDim(1) != 128) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape expected[] = {gert::Shape({q.GetDim(0), q.GetDim(1)}), gert::Shape({q.GetDim(0), 2048}),
                                 gert::Shape({q.GetDim(0), 2048}), gert::Shape({q.GetDim(0), 2048})};
  for (size_t input = 2; input < 6; ++input) {
    const auto *shape = context->GetRequiredInputShape(input);
    if (shape == nullptr || shape->GetOriginShape() != expected[input - 2]) {
      return ge::GRAPH_FAILED;
    }
  }
  const gert::Shape outputs[] = {expected[3], q, k, expected[0], gert::Shape({q.GetDim(0)}), gert::Shape({40, 4})};
  for (size_t output = 0; output < 6; ++output) {
    const auto *shape = context->GetOutputShape(output);
    if (shape == nullptr || shape->GetOriginShape() != outputs[output]) {
      return ge::GRAPH_FAILED;
    }
  }
  Fp32IndexKlTilingData data;
  data.set_queries(q.GetDim(0));
  data.set_keys(k.GetDim(0));
  data.set_heads(q.GetDim(1));
  data.set_phase(*phase);
  auto *raw = context->GetRawTilingData();
  if (raw == nullptr || raw->GetCapacity() < data.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = 0;
  context->SetBlockDim(40);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaFp32IndexKl).Tiling(TilingFp32IndexKl);
}  // namespace optiling
