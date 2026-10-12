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
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#include "base/context_builder/op_tiling_context_builder.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "hyper_dsa_fused_grad_tiling.h"  // NOLINT(build/include_subdir)

namespace optiling {
namespace sfag {
ge::graphStatus TilingHyperDsaMixedGrad(gert::TilingContext *context);
}
namespace {
gert::Tensor InputMetadata(gert::TilingContext *context, size_t index) {
  const auto *desc = context->GetInputDesc(index);
  return gert::Tensor(*context->GetRequiredInputShape(index),
                      gert::StorageFormat(desc->GetOriginFormat(), desc->GetStorageFormat(), desc->GetExpandDimsType()),
                      desc->GetDataType());
}
gert::Tensor OutputMetadata(gert::TilingContext *context, size_t index) {
  const auto *desc = context->GetOutputDesc(index);
  return gert::Tensor(*context->GetOutputShape(index),
                      gert::StorageFormat(desc->GetOriginFormat(), desc->GetStorageFormat(), desc->GetExpandDimsType()),
                      desc->GetDataType());
}
ge::graphStatus Metadata(gert::TilingContext *context, std::array<gert::Tensor, 15> &inputs,
                         std::array<gert::Tensor, 5> &outputs) {
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (context->GetInputDesc(index) == nullptr || context->GetRequiredInputShape(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    inputs[index] = InputMetadata(context, index);
  }
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (context->GetOutputDesc(index) == nullptr || context->GetOutputShape(index) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    outputs[index] = OutputMetadata(context, index);
  }
  return ge::GRAPH_SUCCESS;
}
void Configure(gert::TilingContext *context, gert::OpTilingContextBuilder &builder,
               std::array<gert::Tensor, 15> &inputs, std::array<gert::Tensor, 5> &outputs, gert::Tensor &trace) {
  std::vector<gert::Tensor *> in;
  std::vector<gert::Tensor *> out;
  for (size_t index = 0; index < inputs.size(); ++index) {
    in.push_back(index == 13 ? &trace : &inputs[index]);
  }
  for (auto &tensor : outputs) {
    out.push_back(&tensor);
  }
  builder.OpName(ge::AscendString(context->GetNodeName()))
    .OpType(ge::AscendString("HyperDsaMixedGrad"))
    .IOInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1})
    .InputTensors(in)
    .OutputTensors(out)
    .PlatformInfo(context->GetPlatformInfo())
    .Deterministic(0)
    .AppendAttr(*context->GetAttrs()->GetAttrPointer<float>(0))
    .AppendAttr(int64_t{1})
    .AppendAttr(ge::AscendString("TND"))
    .AppendAttr(int64_t{3})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(false)
    .AppendAttr(int64_t{1});
}
ge::graphStatus PrivateGroups(gert::TilingContext *context, uint64_t &groups) {
  const auto *shape = context->GetRequiredInputShape(20);
  const auto *keys = context->GetRequiredInputShape(1);
  if (shape == nullptr || keys == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto &partials = shape->GetStorageShape();
  const int64_t tokens = keys->GetStorageShape().GetDim(0);
  groups = 0;
  if (partials.GetDimNum() == 2 && partials.GetDim(0) == tokens && partials.GetDim(1) == 1088) {
    return ge::GRAPH_SUCCESS;
  }
  if (partials.GetDimNum() != 3 || partials.GetDim(0) < 2 || partials.GetDim(0) > 20 ||
      partials.GetDim(1) != tokens || partials.GetDim(2) != 1088) {
    return ge::GRAPH_FAILED;
  }
  groups = partials.GetDim(0) - 1;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus EmptyQueryTiling(gert::TilingContext *context, const std::array<gert::Tensor, 15> &inputs,
                                uint64_t privateGroups) {
  const auto &query = inputs[0].GetStorageShape();
  const auto &key = inputs[1].GetStorageShape();
  if (query.GetDimNum() != 3 || query.GetDim(0) != 0 || query.GetDim(2) != 512 ||
      (query.GetDim(1) != 32 && query.GetDim(1) != 64) || key.GetDimNum() != 3 || key.GetDim(0) <= 0 ||
      key.GetDim(1) != 1 || key.GetDim(2) != 512) {
    return ge::GRAPH_FAILED;
  }
  const uint64_t keys = key.GetDim(0);
  const uint64_t keyBytes = (keys * 576 * sizeof(float) + 511) / 512 * 512;
  const uint64_t valueBytes = (keys * 512 * sizeof(float) + 511) / 512 * 512;
  if (inputs[14].GetStorageShape().GetShapeSize() < static_cast<int64_t>(keyBytes + valueBytes)) {
    return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
  if (platform.GetCoreNumAic() != 20 || platform.GetCoreNumAiv() != 40) {
    return ge::GRAPH_FAILED;
  }
  DsaFusedGradTilingData data;
  data.grad.opInfo.set_S1(0);
  data.grad.opInfo.set_S2(keys);
  data.grad.postTilingData.set_dkWorkSpaceOffset(0);
  data.grad.postTilingData.set_dvWorkSpaceOffset(keyBytes);
  data.set_privateGroups(privateGroups);
  auto *raw = context->GetRawTilingData();
  auto *sizes = context->GetWorkspaceSizes(1);
  if (raw == nullptr || raw->GetCapacity() < data.GetDataSize() || sizes == nullptr) {
    return ge::GRAPH_FAILED;
  }
  data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
  raw->SetDataSize(data.GetDataSize());
  // The generated MIX entry owns a system region even when no query math is issued.
  sizes[0] = std::max<uint64_t>(32 * 1024 * 1024, platform.GetLibApiWorkSpaceSize());
  context->SetBlockDim(20);
  context->SetScheduleMode(1);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus TilingDsaFusedGrad(gert::TilingContext *context) {
  if (context == nullptr || context->GetAttrs() == nullptr || context->GetDeterministic() == 1) {
    return ge::GRAPH_FAILED;
  }
  std::array<gert::Tensor, 15> inputs;
  std::array<gert::Tensor, 5> outputs;
  if (Metadata(context, inputs, outputs) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (inputs[13].GetStorageShape().GetShapeSize() != 3 * 20 * 64) {
    return ge::GRAPH_FAILED;
  }
  uint64_t privateGroups = 0;
  if (PrivateGroups(context, privateGroups) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const bool local = context->GetOptionalInputDesc(21) != nullptr;
  if (local) {
    const auto *positions = context->GetOptionalInputShape(21);
    if (positions == nullptr || positions->GetStorageShape().GetDimNum() != 1 ||
        context->GetOptionalInputDesc(21)->GetDataType() != ge::DT_INT64 ||
        positions->GetStorageShape().GetShapeSize() != inputs[0].GetStorageShape().GetDim(0)) {
      return ge::GRAPH_FAILED;
    }
    if (inputs[0].GetStorageShape().GetDim(0) == 0) {
      return EmptyQueryTiling(context, inputs, privateGroups);
    }
  }
  gert::Tensor trace(gert::StorageShape({20, 64}, {20, 64}), gert::StorageFormat(), ge::DT_INT64);
  gert::OpTilingContextBuilder builder;
  Configure(context, builder, inputs, outputs, trace);
  auto workspaceOwner = gert::ContinuousVector::Create<size_t>(1);
  if (workspaceOwner == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceOwner.get());
  const int64_t compileCoreCount = 20;
  builder.CompileInfo(&compileCoreCount).Workspace(workspace).TilingDataSize(4096);
  auto holder = builder.Build();
  auto *child = holder.GetContext();
  if (child == nullptr || sfag::TilingHyperDsaMixedGrad(child) != ge::GRAPH_SUCCESS || child->GetBlockDim() != 20) {
    return ge::GRAPH_FAILED;
  }
  const auto *raw = child->GetRawTilingData();
  auto *target = context->GetRawTilingData();
  DsaFusedGradTilingData data;
  data.set_privateGroups(privateGroups);
  if (raw == nullptr || target == nullptr || target->GetCapacity() < data.GetDataSize() ||
      raw->GetDataSize() != data.grad.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  data.SaveToBuffer(target->GetData(), target->GetCapacity());
  std::memcpy(target->GetData(), raw->GetData(), raw->GetDataSize());
  target->SetDataSize(data.GetDataSize());
  context->GetWorkspaceSizes(1)[0] = child->GetWorkspaceSizes(1)[0];
  context->SetBlockDim(20);
  context->SetScheduleMode(1);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaFusedGrad).Tiling(TilingDsaFusedGrad);
}  // namespace optiling
