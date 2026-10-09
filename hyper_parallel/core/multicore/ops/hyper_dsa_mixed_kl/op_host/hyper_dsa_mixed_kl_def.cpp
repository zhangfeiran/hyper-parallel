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
class HyperDsaMixedKl : public OpDef {
 public:
  explicit HyperDsaMixedKl(const char *name) : OpDef(name) {
    for (const char *input : {"query", "key", "query_index", "key_index"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    this->Input("weight")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BF16, ge::DT_FLOAT})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Input("sparse_indices")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    for (const char *input : {"softmax_max", "softmax_sum"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    for (const char *input : {"query_rope", "key_rope"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    }
    for (const char *input : {"actual_seq_lengths_query", "actual_seq_lengths_key"}) {
      this->Input(input)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .ValueDepend(REQUIRED)
        .AutoContiguous();
    }
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
    for (const char *output : {"d_query_index", "d_key_index"}) {
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
    this->Attr("scale_value").AttrType(REQUIRED).Float();
    this->Attr("input_layout").AttrType(OPTIONAL).String("TND");
    this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
    this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
    this->Attr("phase").AttrType(REQUIRED).Int();
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(HyperDsaMixedKl);
}  // namespace ops
