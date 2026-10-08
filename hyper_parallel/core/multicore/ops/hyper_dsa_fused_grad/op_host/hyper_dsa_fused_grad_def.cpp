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
class HyperDsaFusedGrad : public OpDef {
 public:
  explicit HyperDsaFusedGrad(const char *name) : OpDef(name) {
    for (const char *input : {"query", "key"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Input("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND}).AutoContiguous();
    for (const char *input : {"d_out", "out"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"softmax_max", "softmax_sum"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Input("value").ParamType(OPTIONAL).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
    for (const char *input : {"actual_seq_lengths_query", "actual_seq_lengths_kv"}) {
      this->Input(input).ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"query_rope", "key_rope"}) {
      this->Input(input).ParamType(OPTIONAL).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"runtime_config", "group_trace"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"retained", "arena"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"transport_meta", "requests", "transport_trace"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *input : {"owner_gradient", "partials"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    for (const char *output : {"d_query", "d_key", "d_value", "d_query_rope", "d_key_rope"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
    }
    this->Attr("scale_value").AttrType(REQUIRED).Float();
    this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
    this->Attr("layout").AttrType(OPTIONAL).String("TND");
    this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
    this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    this->Attr("deterministic").AttrType(OPTIONAL).Bool(false);
    this->Attr("phase").AttrType(REQUIRED).Int();
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaFusedGrad);

ge::graphStatus InferFusedGradShape(gert::InferShapeContext *context) {
  const size_t sources[] = {0, 1, 7, 10, 11};
  for (size_t index = 0; index < 5; ++index) {
    if (context->GetInputShape(sources[index]) == nullptr || context->GetOutputShape(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(index) = *context->GetInputShape(sources[index]);
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferFusedGradType(gert::InferDataTypeContext *context) {
  for (size_t index = 0; index < 5; ++index) {
    context->SetOutputDataType(index, ge::DT_BF16);
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaFusedGrad).InferShape(InferFusedGradShape).InferDataType(InferFusedGradType);
}  // namespace ops
