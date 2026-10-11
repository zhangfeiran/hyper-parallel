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
class HyperDsaTpTileSum : public OpDef {
 public:
  explicit HyperDsaTpTileSum(const char *name) : OpDef(name) {
    this->Input("partial").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("arena").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("metadata").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Output("summed").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaTpTileSum);
ge::graphStatus InferTpTileShape(gert::InferShapeContext *context) {
  if (context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferTpTileType(gert::InferDataTypeContext *context) {
  context->SetOutputDataType(0, ge::DT_FLOAT);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(HyperDsaTpTileSum).InferShape(InferTpTileShape).InferDataType(InferTpTileType);
}  // namespace ops
