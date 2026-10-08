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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_DSA_GRAD_CHECKS_H_
#define HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_DSA_GRAD_CHECKS_H_
#include <ATen/ATen.h>
#include <array>
namespace hyper_parallel::multicore {
using DsaGradInputs = std::array<const at::Tensor *, 20>;
void check_dsa_grad_inputs(const DsaGradInputs &tensors, double scale);
}
#endif  // HYPER_PARALLEL_CORE_MULTICORE_TORCH_CSRC_DSA_GRAD_CHECKS_H_
