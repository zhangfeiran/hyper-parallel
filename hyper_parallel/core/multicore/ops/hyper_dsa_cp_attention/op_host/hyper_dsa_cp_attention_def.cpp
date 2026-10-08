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
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace ops {
class HyperDsaCpAttention : public OpDef {
 public:
  explicit HyperDsaCpAttention(const char *name) : OpDef(name) {
    for (const char *input : {"query", "compressed", "query_rope", "key_rope"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
          .Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"indices", "lengths"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32})
          .Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"runtime_config", "group_trace"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Input("arena").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    for (const char *input : {"transport_meta", "requests", "transport_trace"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Output("attention").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    for (const char *output : {"maximum", "sum"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    }
    this->Attr("scale").AttrType(REQUIRED).Float();
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaCpAttention);
ge::graphStatus InferCpAttentionShape(gert::InferShapeContext *context) {
  const auto *query = context->GetInputShape(0);
  if (query == nullptr || query->GetDimNum() != 3) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (context->GetOutputShape(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
  }
  *context->GetOutputShape(0) = *query;
  *context->GetOutputShape(1) = gert::Shape({1, query->GetDim(0), query->GetDim(1)});
  *context->GetOutputShape(2) = *context->GetOutputShape(1);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferCpAttentionType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_BF16);
  context->SetOutputDataType(1, ge::DT_FLOAT);
  context->SetOutputDataType(2, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaCpAttention).InferShape(InferCpAttentionShape).InferDataType(InferCpAttentionType);
}  // namespace ops
