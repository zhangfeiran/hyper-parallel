// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <torch/library.h>

#include <mutex>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "dense_tile_binary.inc"

extern "C" {
uint32_t RegisterAscendBinary(const char *buffer, size_t size, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, uint64_t key, uint32_t blocks, void **arguments, uint32_t size,
                            const void *stream);
uint32_t GetAscendCoreSyncAddr(void **address);
bool AscendCheckSoCVersion(const char *soc, char *error);
}

namespace {
struct KernelState {
  std::mutex mutex;
  void *handle = nullptr;
};

void *RegisteredKernel() {
  // The process owns its registration. Module close cannot unload a kernel still used by a saved graph.
  static KernelState state;
  std::lock_guard<std::mutex> guard(state.mutex);
  if (state.handle == nullptr) {
    char error[1024] = {};
    TORCH_CHECK(AscendCheckSoCVersion(HP_DENSE_SOC, error), "Dense compiled/device SoC mismatch: ", error);
    const auto status =
      RegisterAscendBinary(reinterpret_cast<const char *>(HP_DENSE_BINARY), sizeof(HP_DENSE_BINARY), 0, &state.handle);
    TORCH_CHECK(status == 0 && state.handle != nullptr, "Dense binary registration failed: ", status);
  }
  return state.handle;
}

struct KernelArguments {
  void *ffts;
  void *values;
  void *config;
  void *tilings;
  void *events;
  void *ones;
  void *workspace;
  void *overflow;
};
static_assert(sizeof(KernelArguments) == 8 * sizeof(void *), "Dense standalone launch argument ABI drift");

void CheckMetadata(const at::Tensor &config, const at::Tensor &tilings, const at::Tensor &events,
                   const at::Tensor &ones, const at::Tensor &workspace, const at::Tensor &overflow) {
  TORCH_CHECK(config.scalar_type() == at::kByte && config.numel() >= 68 && (config.numel() - 32) % 36 == 0 &&
                tilings.scalar_type() == at::kByte &&
                tilings.numel() == (config.numel() - 32) / 36 * HP_DENSE_BANK_STRIDE,
              "Invalid dense task/tiling storage");
  TORCH_CHECK(
    events.scalar_type() == at::kInt && events.numel() % 32 == 0 && ones.scalar_type() == at::kInt && ones.numel() == 8,
    "Invalid dense event storage");
  TORCH_CHECK(workspace.scalar_type() == at::kByte && overflow.scalar_type() == at::kByte && overflow.numel() == 8,
              "Invalid dense workspace/overflow storage");
}

void Launch(const std::vector<at::Tensor> &inputs, const std::vector<at::Tensor> &outputs, const at::Tensor &pointers,
            const at::Tensor &config, const at::Tensor &tilings, const at::Tensor &events, const at::Tensor &ones,
            const at::Tensor &workspace, const at::Tensor &overflow, int64_t cube_workers) {
  TORCH_CHECK(cube_workers > 0 && cube_workers <= 24, "Dense launch requires 1..24 resident cube workers");
  TORCH_CHECK(!inputs.empty() && !outputs.empty() && pointers.scalar_type() == at::kLong &&
                pointers.numel() == static_cast<int64_t>(inputs.size() + outputs.size()),
              "Invalid dense pointer table");
  CheckMetadata(config, tilings, events, ones, workspace, overflow);
  std::vector<at::Tensor> retained(inputs);
  retained.insert(retained.end(), outputs.begin(), outputs.end());
  for (const auto &value : retained) {
    TORCH_CHECK(value.scalar_type() == at::kBFloat16, "Dense resident values require BF16 storage");
  }
  auto stream = c10_npu::getCurrentNPUStream(pointers.device().index());
  retained.insert(retained.end(), {pointers, config, tilings, events, ones, workspace, overflow});
  for (const auto &tensor : retained) {
    TORCH_CHECK(tensor.device() == pointers.device() && tensor.is_contiguous(),
                "Dense launch requires contiguous tensors on one NPU device");
    if (tensor.numel()) {
      c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
  }
  auto handler = [retained, pointers, config, tilings, events, ones, workspace, overflow, cube_workers, stream]() {
    void *ffts = nullptr;
    const auto status = GetAscendCoreSyncAddr(&ffts);
    TORCH_CHECK(status == 0 && ffts != nullptr, "Dense core synchronization address failed: ", status);
    KernelArguments arguments{ffts,
                              pointers.data_ptr(),
                              config.data_ptr(),
                              tilings.data_ptr(),
                              events.data_ptr(),
                              ones.data_ptr(),
                              workspace.data_ptr(),
                              overflow.data_ptr()};
    return static_cast<int>(LaunchAscendKernel(RegisteredKernel(), 0, static_cast<uint32_t>(cube_workers),
                                               reinterpret_cast<void **>(&arguments), sizeof(arguments),
                                               stream.stream(false)));
  };
  // TorchNPU queues preceding Tensor copies and this launch in order; the handler retains their storage.
  at_npu::native::OpCommand::RunOpApiV3("HyperParallelDenseTile", handler, false, &stream);
}
}  // namespace

#define HP_DENSE_LIBRARY(ns, module) TORCH_LIBRARY(ns, module)
#define HP_DENSE_LIBRARY_IMPL(ns, key, module) TORCH_LIBRARY_IMPL(ns, key, module)

HP_DENSE_LIBRARY(HP_DENSE_NAMESPACE, module) {
  module.def(
    "launch(Tensor[] inputs, Tensor(a!)[] outputs, Tensor pointers, Tensor config, Tensor tilings, Tensor(b!) events, "
    "Tensor ones, Tensor(c!) workspace, Tensor(d!) overflow, int cube_workers) -> ()");
}
HP_DENSE_LIBRARY_IMPL(HP_DENSE_NAMESPACE, PrivateUse1, module) { module.impl("launch", Launch); }
