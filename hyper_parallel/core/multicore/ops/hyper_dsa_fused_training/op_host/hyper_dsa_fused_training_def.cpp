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
class HyperDsaFusedTraining : public OpDef {
 public:
  explicit HyperDsaFusedTraining(const char *name) : OpDef(name) {
    for (const char *input : {"index_query", "index_key", "query", "compressed", "query_rope", "key_rope"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    this->Input("weights")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BF16, ge::DT_FLOAT})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("lengths")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    for (const char *input : {"runtime_config", "group_trace"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    this->Input("retained")
      .ParamType(REQUIRED)
      .DataType({ge::DT_UINT8, ge::DT_UINT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("kl_retained")
      .ParamType(REQUIRED)
      .DataType({ge::DT_UINT8, ge::DT_UINT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("kl_lengths")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .ValueDepend(REQUIRED)
      .AutoContiguous();
    this->Input("arena")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_UINT8, ge::DT_UINT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    for (const char *input : {"transport_meta", "requests", "transport_trace"}) {
      this->Input(input)
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    this->Input("selected_rows")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_INT64, ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("selected_membership")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    for (const char *input : {"selected_requests", "selected_counts"}) {
      this->Input(input)
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    this->Output("indices")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    for (const char *output : {"values", "attention"}) {
      this->Output(output)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    for (const char *output : {"maximum", "sum"}) {
      this->Output(output)
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    for (const char *output : {"d_index_query", "d_index_key"}) {
      this->Output(output)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }
    this->Output("d_weight")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BF16, ge::DT_FLOAT})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("loss")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("scale").AttrType(REQUIRED).Float();
    this->Attr("with_kl").AttrType(OPTIONAL).Bool(true);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaFusedTraining);

ge::graphStatus InferTrainingShape(gert::InferShapeContext *context) {
  const auto *query = context->GetInputShape(2);
  if (query == nullptr || query->GetDimNum() != 3) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < 9; ++index) {
    if (context->GetOutputShape(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
  }
  const auto tokens = query->GetDim(0);
  const auto heads = query->GetDim(1);
  *context->GetOutputShape(0) = gert::Shape({tokens, 1, 2048});
  *context->GetOutputShape(1) = gert::Shape({tokens, 1, 2048});
  *context->GetOutputShape(2) = *query;
  *context->GetOutputShape(3) = gert::Shape({1, tokens, heads});
  *context->GetOutputShape(4) = gert::Shape({1, tokens, heads});
  const auto *withKl = context->GetAttrs()->GetAttrPointer<bool>(1);
  if (withKl == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (!*withKl) {
    for (size_t index = 5; index < 9; ++index) {
      *context->GetOutputShape(index) = gert::Shape({0});
    }
    return ge::GRAPH_SUCCESS;
  }
  *context->GetOutputShape(5) = *context->GetInputShape(0);
  *context->GetOutputShape(6) = *context->GetInputShape(1);
  *context->GetOutputShape(7) = *context->GetInputShape(6);
  *context->GetOutputShape(8) = gert::Shape({1});
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferTrainingType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  context->SetOutputDataType(1, ge::DT_BF16);
  context->SetOutputDataType(2, ge::DT_BF16);
  context->SetOutputDataType(3, ge::DT_FLOAT);
  context->SetOutputDataType(4, ge::DT_FLOAT);
  context->SetOutputDataType(5, ge::DT_BF16);
  context->SetOutputDataType(6, ge::DT_BF16);
  context->SetOutputDataType(7, context->GetInputDataType(6));
  context->SetOutputDataType(8, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaFusedTraining).InferShape(InferTrainingShape).InferDataType(InferTrainingType);
}  // namespace ops
