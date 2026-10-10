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
class HyperDsaIndexGradReturn : public OpDef {
 public:
  explicit HyperDsaIndexGradReturn(const char *name) : OpDef(name) {
    this->Input("gradient").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("arena").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    for (const char *input : {"metadata", "requests", "trace"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    }
    this->Output("owner").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Attr("capacity").AttrType(REQUIRED).Int();
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaIndexGradReturn);
ge::graphStatus InferIndexGradShape(gert::InferShapeContext *context) {
  const auto *capacity = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  if (capacity == nullptr || *capacity <= 0 || context->GetOutputShape(0) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = gert::Shape({*capacity, 128});
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferIndexGradType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaIndexGradReturn).InferShape(InferIndexGradShape).InferDataType(InferIndexGradType);
}  // namespace ops
