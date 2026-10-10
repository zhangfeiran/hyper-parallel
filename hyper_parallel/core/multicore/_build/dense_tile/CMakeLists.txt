# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(hyper_parallel_dense_tile LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "Selected CANN SDK root")
set(SOC_VERSION Ascend910B3 CACHE STRING "Dense candidate target")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(HP_ASCENDC_CMAKE_ROOT
    "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
set(HP_ASCENDC_CMAKE "${HP_ASCENDC_CMAKE_ROOT}/ascendc.cmake")
if(NOT EXISTS "${HP_ASCENDC_CMAKE}")
    message(FATAL_ERROR
        "The selected CANN SDK lacks AscendC standalone build support")
endif()
include(${HP_ASCENDC_CMAKE})

set(HP_DENSE_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../ops/dense")
set(HP_ASCEND_ARCH_ROOT
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux")
ascendc_library(hyper_parallel_dense_tile STATIC
    ${HP_DENSE_SOURCE_ROOT}/dense_tile_kernel.cpp)
ascendc_include_directories(hyper_parallel_dense_tile PRIVATE
    ${HP_DENSE_SOURCE_ROOT}/.. ${HP_ASCEND_ARCH_ROOT})

add_library(hyper_parallel_dense_tiling SHARED
    ${HP_DENSE_SOURCE_ROOT}/dense_tile_tiling.cpp)
target_include_directories(hyper_parallel_dense_tiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${HP_ASCEND_ARCH_ROOT}/asc/include/adv_api
    ${HP_ASCEND_ARCH_ROOT}/asc/include/utils)
target_link_directories(hyper_parallel_dense_tiling PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(hyper_parallel_dense_tiling PRIVATE
    tiling_api platform register unified_dlog c_sec)

# SDK 的编译平台接口使用旧版 libstdc++ ABI，单独隔离于 Torch 启动器。
add_library(hyper_parallel_dense_reference SHARED
    ${HP_DENSE_SOURCE_ROOT}/dense_reference_tiling.cpp)
target_compile_definitions(hyper_parallel_dense_reference PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(hyper_parallel_dense_reference PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(hyper_parallel_dense_reference PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(hyper_parallel_dense_reference PRIVATE platform
    "${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${CMAKE_SYSTEM_PROCESSOR}/libophost_comm_legacy.so")
target_link_options(hyper_parallel_dense_reference PRIVATE "-Wl,-z,defs")
