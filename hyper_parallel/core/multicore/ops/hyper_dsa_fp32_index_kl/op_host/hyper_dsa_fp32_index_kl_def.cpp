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
class HyperDsaFp32IndexKl : public OpDef {
 public:
  explicit HyperDsaFp32IndexKl(const char *name) : OpDef(name) {
    for (const char *input : {"query", "key", "weight", "teacher", "global_logits"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    for (const char *output : {"local_logits", "grad_query", "grad_key", "grad_weight", "loss"}) {
      this->Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    }
    this->Output("trace").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
    this->Attr("phase").AttrType(REQUIRED).Int();
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaFp32IndexKl);
ge::graphStatus InferFp32IndexKlShape(gert::InferShapeContext *context) {
  for (size_t output = 0; output < 4; ++output) {
    const size_t input = output == 0 ? 5 : output - 1;
    if (context->GetInputShape(input) == nullptr || context->GetOutputShape(output) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(output) = *context->GetInputShape(input);
  }
  if (context->GetInputShape(0) == nullptr || context->GetOutputShape(4) == nullptr ||
      context->GetOutputShape(5) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *context->GetOutputShape(4) = gert::Shape({context->GetInputShape(0)->GetDim(0)});
  *context->GetOutputShape(5) = gert::Shape({40, 4});
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferFp32IndexKlType(gert::InferDataTypeContext *context) {
  for (size_t output = 0; output < 5; ++output) {
    context->SetOutputDataType(output, ge::DT_FLOAT);
  }
  context->SetOutputDataType(5, ge::DT_INT64);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaFp32IndexKl).InferShape(InferFp32IndexKlShape).InferDataType(InferFp32IndexKlType);
}  // namespace ops
