# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(hyper_parallel_multicore_vendor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(HP_GATE_COMPONENT ${HP_GATE_PINNED_ROOT}/hyper_parallel/core/multicore)
include(${HP_GATE_COMPONENT}/cmake/hardening.cmake)

set(HP_MULTICORE_SOURCE_ROOT "" CACHE PATH "Assembled Gate source root")
set(HP_VENDOR_STAGE_ROOT "${CMAKE_BINARY_DIR}/stage"
    CACHE PATH "CANN vendor stage root")

set(GATE_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_gate/op_host)
set(GATE_GRAD_HOST ${HP_MULTICORE_SOURCE_ROOT}/hyper_mega_gate_grad/op_host)

if(NOT HP_MULTICORE_SOURCE_ROOT
        OR NOT EXISTS "${GATE_HOST}/hyper_mega_gate_route_def.cpp"
        OR NOT EXISTS "${GATE_GRAD_HOST}/hyper_mega_gate_route_grad_def.cpp")
    message(FATAL_ERROR
        "HP_MULTICORE_SOURCE_ROOT must identify an assembled Gate source tree")
endif()
find_package(ASC REQUIRED)

set(package_name hyper_parallel_multicore_gate_v1)
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${HP_VENDOR_STAGE_ROOT}
        VENDOR_NAME hyper_parallel_multicore_gate_v1
        ENABLE_CPACK False
        ENABLE_SOURCE_PACKAGE False
        ENABLE_BINARY_PACKAGE True
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
