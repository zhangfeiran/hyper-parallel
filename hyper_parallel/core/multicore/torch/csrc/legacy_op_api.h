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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_LEGACY_OP_API_H_
#define HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_LEGACY_OP_API_H_

#include <ATen/ATen.h>
#include <acl/acl_base.h>
#include <dlfcn.h>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

struct aclTensor;
struct aclIntArray;
struct aclOpExecutor;

namespace hyper_parallel::multicore::legacy {

/** Keep ACLNN symbols local to the selected SDK and custom vendor. */
inline void *Resolve(const char *name) {
  static const auto libraries = [] {
    std::vector<void *> handles;
    const auto add = [&](const std::string &path) {
      if (auto *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)) handles.push_back(handle);
    };
    if (const auto *vendors = std::getenv("ASCEND_CUSTOM_OPP_PATH")) {
      std::string paths(vendors);
      size_t start = 0;
      do {
        const auto end = paths.find(':', start);
        const auto path = paths.substr(start, end == std::string::npos ? end : end - start);
        if (!path.empty()) add(path + "/op_api/lib/libcust_opapi.so");
        if (end == std::string::npos) break;
        start = end + 1;
      } while (start < paths.size());
    }
    const auto *sdk = std::getenv("ASCEND_HOME_PATH");
    TORCH_CHECK(sdk != nullptr, "Source the selected CANN SDK before loading multicore");
    for (const auto *library : {"libopapi.so", "libnnopbase.so"}) add(std::string(sdk) + "/lib64/" + library);
    return handles;
  }();
  for (auto *library : libraries) {
    if (auto *symbol = dlsym(library, name)) return symbol;
  }
  TORCH_CHECK(false, "Selected CANN/vendor does not export ", name);
}

inline aclDataType DataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat: return ACL_FLOAT;
    case at::kHalf: return ACL_FLOAT16;
    case at::kBFloat16: return ACL_BF16;
    case at::kDouble: return ACL_DOUBLE;
    case at::kLong: return ACL_INT64;
    case at::kInt: return ACL_INT32;
    case at::kShort: return ACL_INT16;
    case at::kChar: return ACL_INT8;
    case at::kByte: return ACL_UINT8;
    case at::kBool: return ACL_BOOL;
    default: TORCH_CHECK(false, "Unsupported multicore ACLNN tensor dtype: ", type);
  }
}

/** Own descriptors through workspace query and asynchronous launch submission. */
template <class Handle>
class Descriptor {
 public:
  using Destroy = int (*)(const Handle *);
  Descriptor(Handle *value, Destroy destroy) : value_(value), destroy_(destroy) {}
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  Descriptor(Descriptor &&other) noexcept : value_(std::exchange(other.value_, nullptr)), destroy_(other.destroy_) {}
  ~Descriptor() { if (value_ != nullptr) destroy_(value_); }
  const Handle *Get() const { return value_; }

 private:
  Handle *value_;
  Destroy destroy_;
};

inline Descriptor<aclTensor> Convert(const at::Tensor &value) {
  using Create = aclTensor *(*)(const int64_t *, uint64_t, aclDataType, const int64_t *, int64_t, aclFormat,
                                const int64_t *, uint64_t, void *);
  static const auto create = reinterpret_cast<Create>(Resolve("aclCreateTensor"));
  static const auto destroy = reinterpret_cast<Descriptor<aclTensor>::Destroy>(Resolve("aclDestroyTensor"));
  if (!value.defined()) return {nullptr, destroy};
  TORCH_CHECK(value.device().type() == c10::DeviceType::PrivateUse1, "Multicore ACLNN requires NPU tensors");
  const auto *storage = static_cast<const torch_npu::NPUStorageImpl *>(value.storage().unsafeGetStorageImpl());
  const auto format = storage->npu_desc_.npu_format_;
  auto dimensions = storage->npu_desc_.storage_sizes_;
  if (format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW || format == ACL_FORMAT_NHWC ||
      format == ACL_FORMAT_NCL || format == ACL_FORMAT_NCDHW) {
    dimensions = {static_cast<int64_t>(value.storage().nbytes() / value.element_size())};
  }
  auto *tensor = create(value.sizes().data(), value.dim(), DataType(value.scalar_type()), value.strides().data(),
                        value.storage_offset(), format, dimensions.data(), dimensions.size(),
                        const_cast<void *>(value.storage().data()));
  TORCH_CHECK(tensor != nullptr, "aclCreateTensor failed");
  return {tensor, destroy};
}

using OwnedShape = std::optional<std::vector<int64_t>>;
inline Descriptor<aclIntArray> Convert(const OwnedShape &shape) {
  using Create = aclIntArray *(*)(const int64_t *, uint64_t);
  static const auto create = reinterpret_cast<Create>(Resolve("aclCreateIntArray"));
  static const auto destroy = reinterpret_cast<Descriptor<aclIntArray>::Destroy>(Resolve("aclDestroyIntArray"));
  if (!shape) return {nullptr, destroy};
  auto *array = create(shape->data(), shape->size());
  TORCH_CHECK(array != nullptr, "aclCreateIntArray failed");
  return {array, destroy};
}

template <class Value>
struct Scalar {
  Value value;
  Value Get() const { return value; }
};
inline Scalar<int64_t> Convert(int64_t value) { return {value}; }
inline Scalar<double> Convert(double value) { return {value}; }
inline Scalar<bool> Convert(bool value) { return {value}; }

template <class Value>
Value Own(const Value &value) { return value; }
inline OwnedShape Own(const c10::optional<at::IntArrayRef> &shape) {
  return shape ? OwnedShape(std::vector<int64_t>(shape->begin(), shape->end())) : std::nullopt;
}

inline void Record(const at::Tensor &value, c10_npu::NPUStream stream) {
  if (value.defined()) c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
}
template <class Value>
void Record(const Value &, c10_npu::NPUStream) {}

/** Use the wheel's public task queue, retaining tensors and optional array storage. */
template <class... Args>
void Submit(const char *name, void *query, void *execute, const Args &...args) {
  TORCH_CHECK(!at::globalContext().deterministicAlgorithms(),
              "The legacy multicore ACLNN bridge does not support deterministic-algorithm mode");
  auto stream = c10_npu::getCurrentNPUStream();
  auto acl_stream = stream.stream(false);
  auto owned = std::make_tuple(Own(args)...);
  std::apply([&](const auto &...value) { (Record(value, stream), ...); }, owned);
  at_npu::native::OpCommand::RunOpApiV2(name, [owned, query, execute, acl_stream]() -> int {
    auto converted = std::apply([](const auto &...value) { return std::make_tuple(Convert(value)...); }, owned);
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    const auto status = std::apply([&](auto &...value) {
      using Query = int (*)(decltype(value.Get())..., uint64_t *, aclOpExecutor **);
      return reinterpret_cast<Query>(query)(value.Get()..., &bytes, &executor);
    }, converted);
    TORCH_CHECK(status == 0, "Multicore ACLNN workspace query failed: ", status);
    std::unique_ptr<void, decltype(&c10_npu::NPUCachingAllocator::raw_delete)> workspace(
        bytes == 0 ? nullptr : c10_npu::NPUCachingAllocator::raw_alloc_with_stream(bytes, acl_stream),
        &c10_npu::NPUCachingAllocator::raw_delete);
    using Execute = int (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    const auto result = reinterpret_cast<Execute>(execute)(workspace.get(), bytes, executor, acl_stream);
    TORCH_CHECK(result == 0, "Multicore ACLNN execution failed: ", result);
    return result;
  });
}

}  // namespace hyper_parallel::multicore::legacy

#define HYPER_EXEC_NPU_CMD_EXT(aclnn_api, ...)                                                         \
  do {                                                                                                \
    static auto *hyper_query = ::hyper_parallel::multicore::legacy::Resolve(#aclnn_api "GetWorkspaceSize"); \
    static auto *hyper_execute = ::hyper_parallel::multicore::legacy::Resolve(#aclnn_api);                 \
    ::hyper_parallel::multicore::legacy::Submit(#aclnn_api, hyper_query, hyper_execute, __VA_ARGS__);       \
  } while (false)

#endif  // HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_LEGACY_OP_API_H_
