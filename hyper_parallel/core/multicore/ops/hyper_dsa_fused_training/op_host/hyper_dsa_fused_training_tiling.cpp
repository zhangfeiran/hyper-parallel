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
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>
#include "base/context_builder/op_tiling_context_builder.h"
#include "register/op_impl_registry.h"
#include "err/ops_err.h"
#include "hyper_dsa_fused_training_tiling.h"  // NOLINT(build/include_subdir)

namespace optiling {
ge::graphStatus TilingForLightningIndexer(gert::TilingContext *context);
ge::graphStatus TilingHyperDsaMixedTile(gert::TilingContext *context);
ge::graphStatus TilingHyperDsaMixedKl(gert::TilingContext *context);

namespace {
gert::Tensor InputMetadata(gert::TilingContext *context, size_t input) {
  const auto *desc = context->GetInputDesc(input);
  const gert::StorageFormat format(desc->GetOriginFormat(), desc->GetStorageFormat(), desc->GetExpandDimsType());
  return gert::Tensor(*context->GetRequiredInputShape(input), format, desc->GetDataType());
}

gert::Tensor OutputMetadata(gert::TilingContext *context, size_t output) {
  const auto *desc = context->GetOutputDesc(output);
  const gert::StorageFormat format(desc->GetOriginFormat(), desc->GetStorageFormat(), desc->GetExpandDimsType());
  return gert::Tensor(*context->GetOutputShape(output), format, desc->GetDataType());
}

void IndexerAttrs(gert::OpTilingContextBuilder &builder) {
  builder.AppendAttr(ge::AscendString("TND"))
    .AppendAttr(ge::AscendString("TND"))
    .AppendAttr(int64_t{2048})
    .AppendAttr(int64_t{3})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(true)
    .AppendAttr(int64_t{0});
}

void AttentionAttrs(gert::OpTilingContextBuilder &builder, float scale) {
  builder.AppendAttr(scale)
    .AppendAttr(int64_t{1})
    .AppendAttr(ge::AscendString("TND"))
    .AppendAttr(ge::AscendString("TND"))
    .AppendAttr(int64_t{3})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{2})
    .AppendAttr(true);
}

ge::graphStatus SaveNested(gert::TilingContext *context, DsaFusedTrainingTilingData &data, bool indexer,
                           void *destination) {
  const auto *raw = context->GetRawTilingData();
  const size_t capacity = indexer ? data.li.GetDataSize() : data.sfa.GetDataSize();
  if (raw == nullptr || raw->GetDataSize() != capacity) {
    return ge::GRAPH_FAILED;
  }
  std::memcpy(destination, raw->GetData(), raw->GetDataSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Metadata(gert::TilingContext *context, std::array<gert::Tensor, 11> &inputs,
                         std::array<gert::Tensor, 5> &outputs) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (context->GetInputDesc(i) == nullptr || context->GetRequiredInputShape(i) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    inputs[i] = InputMetadata(context, i);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (context->GetOutputDesc(i) == nullptr || context->GetOutputShape(i) == nullptr) {
      return ge::GRAPH_FAILED;
    }
    outputs[i] = OutputMetadata(context, i);
  }
  return ge::GRAPH_SUCCESS;
}

void ConfigureIo(gert::OpTilingContextBuilder &builder, std::array<gert::Tensor, 11> &inputs,
                 std::array<gert::Tensor, 5> &outputs, gert::Tensor &trace, bool indexer) {
  const std::vector<gert::Tensor *> liInputs = {&inputs[0], &inputs[1], &inputs[6], &inputs[7],
                                                &inputs[7], &inputs[8], &trace,     &inputs[10]};
  const std::vector<gert::Tensor *> sfaInputs = {&inputs[2], &inputs[3], &inputs[3], &outputs[0], &inputs[7],
                                                 &inputs[7], &inputs[4], &inputs[5], &inputs[8],  &trace};
  const std::vector<gert::Tensor *> liOutputs = {&outputs[0], &outputs[1]};
  const std::vector<gert::Tensor *> sfaOutputs = {&outputs[2], &outputs[3], &outputs[4]};
  if (indexer) {
    builder.OpType(ge::AscendString("HyperDsaMixedIndexer"))
      .IOInstanceNum({1, 1, 1, 1, 1, 0, 1, 1, 1}, {1, 1})
      .InputTensors(liInputs)
      .OutputTensors(liOutputs);
    IndexerAttrs(builder);
  } else {
    builder.OpType(ge::AscendString("HyperDsaMixedTile"))
      .IOInstanceNum({1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
      .InputTensors(sfaInputs)
      .OutputTensors(sfaOutputs);
  }
}

ge::graphStatus Compose(gert::TilingContext *context, DsaFusedTrainingTilingData &data, bool indexer,
                        uint64_t &workspaceBytes) {
  std::array<gert::Tensor, 11> inputs;
  std::array<gert::Tensor, 5> outputs;
  if (Metadata(context, inputs, outputs) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  // The child tilers use a phase trace; the parent owns all three records.
  gert::Tensor trace(gert::StorageShape({20, 64}, {20, 64}), gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                     ge::DT_INT64);
  gert::OpTilingContextBuilder builder;
  ConfigureIo(builder, inputs, outputs, trace, indexer);
  auto workspaceOwner = gert::ContinuousVector::Create<size_t>(1);
  if (workspaceOwner == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceOwner.get());
  // The builder requires compile storage; both locked tilers derive hardware exclusively from PlatformInfo.
  const int64_t compileCoreCount = 20;
  builder.OpName(ge::AscendString(context->GetNodeName()))
    .PlatformInfo(context->GetPlatformInfo())
    .CompileInfo(&compileCoreCount)
    .Deterministic(0)
    .Workspace(workspace)
    .TilingDataSize(4096);
  if (!indexer) {
    AttentionAttrs(builder, *context->GetAttrs()->GetAttrPointer<float>(0));
  }
  auto holder = builder.Build();
  auto *child = holder.GetContext();
  if (child == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const auto status = indexer ? TilingForLightningIndexer(child) : TilingHyperDsaMixedTile(child);
  if (status != ge::GRAPH_SUCCESS || child->GetBlockDim() != 20 ||
      SaveNested(child, data, indexer,
                 static_cast<uint8_t *>(context->GetRawTilingData()->GetData()) +
                   (indexer ? 0 : data.li.GetDataSize())) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  const auto *sizes = child->GetWorkspaceSizes(1);
  if (sizes == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspaceBytes = sizes[0];
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus ComposeKl(gert::TilingContext *context, DsaFusedTrainingTilingData &data) {
  std::array<gert::Tensor, 13> inputs;
  std::array<gert::Tensor, 9> outputs;
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
  const auto *hostLengths = context->GetRequiredInputTensor(12);
  if (hostLengths == nullptr || hostLengths->GetData<int64_t>() == nullptr) {
    OP_LOGE(context->GetNodeName(), "fused KL requires host constant cumulative lengths");
    return ge::GRAPH_FAILED;
  }
  // The parent executor owns the host constant until the synchronous child tiling completes.
  inputs[12].SetData(gert::TensorData(const_cast<int64_t *>(hostLengths->GetData<int64_t>()), nullptr,
                                      inputs[12].GetShapeSize() * sizeof(int64_t), gert::kOnHost));
  gert::Tensor trace(gert::StorageShape({20, 64}, {20, 64}), gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                     ge::DT_INT64);
  gert::OpTilingContextBuilder builder;
  builder.OpType(ge::AscendString("HyperDsaMixedKl"))
    .IOInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1})
    .InputTensors({&inputs[2], &inputs[3], &inputs[0], &inputs[1], &inputs[6], &outputs[0], &outputs[3], &outputs[4],
                   &inputs[4], &inputs[5], &inputs[12], &inputs[12], &inputs[8], &trace, &inputs[11]})
    .OutputTensors({&outputs[5], &outputs[6], &outputs[7], &outputs[8]})
    .AppendAttr(*context->GetAttrs()->GetAttrPointer<float>(0))
    .AppendAttr(ge::AscendString("TND"))
    .AppendAttr(int64_t{3})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{INT64_MAX})
    .AppendAttr(int64_t{0});
  auto workspaceOwner = gert::ContinuousVector::Create<size_t>(1);
  if (workspaceOwner == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceOwner.get());
  const int64_t compileCoreCount = 20;
  builder.OpName(ge::AscendString(context->GetNodeName()))
    .PlatformInfo(context->GetPlatformInfo())
    .CompileInfo(&compileCoreCount)
    .Deterministic(0)
    .Workspace(workspace)
    .TilingDataSize(sizeof(SparseLightningIndexerGradKLLossTilingData));
  auto holder = builder.Build();
  auto *child = holder.GetContext();
  if (child == nullptr || TilingHyperDsaMixedKl(child) != ge::GRAPH_SUCCESS || child->GetBlockDim() != 20 ||
      child->GetRawTilingData()->GetDataSize() != sizeof(SparseLightningIndexerGradKLLossTilingData)) {
    OP_LOGE(context->GetNodeName(), "fused KL child tiling failed");
    return ge::GRAPH_FAILED;
  }
  const size_t capacity = (sizeof(SparseLightningIndexerGradKLLossTilingData) + 7) / 8 * 8;
  auto *destination = static_cast<uint8_t *>(context->GetRawTilingData()->GetData()) + data.GetDataSize() - capacity;
  std::memcpy(destination, child->GetRawTilingData()->GetData(), sizeof(SparseLightningIndexerGradKLLossTilingData));
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus TilingDsaFusedTraining(gert::TilingContext *context) {
  if (context->GetAttrs() == nullptr || context->GetAttrs()->GetAttrPointer<bool>(1) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const bool withKl = *context->GetAttrs()->GetAttrPointer<bool>(1);
  const auto *trace = context->GetRequiredInputShape(9);
  const bool selected = context->GetOptionalInputDesc(17) != nullptr;
  if (!withKl && !selected) {
    return ge::GRAPH_FAILED;
  }
  if (selected && (context->GetOptionalInputDesc(13) == nullptr || context->GetOptionalInputDesc(18) == nullptr ||
                   context->GetOptionalInputDesc(19) == nullptr || context->GetOptionalInputDesc(20) == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  if (trace == nullptr || trace->GetStorageShape().GetShapeSize() != (selected ? (withKl ? 9 : 6) : 6) * 20 * 64 ||
      context->GetAttrs() == nullptr || context->GetAttrs()->GetAttrPointer<float>(0) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  DsaFusedTrainingTilingData data;
  auto *raw = context->GetRawTilingData();
  if (raw == nullptr || raw->GetCapacity() < data.GetDataSize()) {
    return ge::GRAPH_FAILED;
  }
  std::memset(raw->GetData(), 0, data.GetDataSize());
  uint64_t liBytes = 0;
  uint64_t sfaBytes = 0;
  if (Compose(context, data, true, liBytes) != ge::GRAPH_SUCCESS ||
      Compose(context, data, false, sfaBytes) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (withKl && ComposeKl(context, data) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  auto *workspace = context->GetWorkspaceSizes(1);
  if (workspace == nullptr || context->GetInputDesc(6) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspace[0] = sfaBytes;
  context->SetBlockDim(20);
  context->SetScheduleMode(1);
  const uint64_t transport = context->GetOptionalInputDesc(13) == nullptr ? 0 : 2;
  context->SetTilingKey((selected ? 4 : 0) + transport +
                        (context->GetInputDesc(6)->GetDataType() == ge::DT_FLOAT ? 1 : 0));
  const uint64_t klEnabled = withKl ? 1 : 0;
  const size_t klCapacity = (sizeof(SparseLightningIndexerGradKLLossTilingData) + 7) / 8 * 8;
  std::memcpy(static_cast<uint8_t *>(raw->GetData()) + data.GetDataSize() - klCapacity - sizeof(uint64_t), &klEnabled,
              sizeof(klEnabled));
  raw->SetDataSize(data.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HyperDsaFusedTraining).Tiling(TilingDsaFusedTraining);
}  // namespace optiling
