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

cmake_minimum_required(VERSION 3.18)
project(hyper_parallel_multicore_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(HP_MHC_COMPONENT ${HP_MHC_PINNED_ROOT}/hyper_parallel/core/multicore)
include(${HP_MHC_COMPONENT}/cmake/hardening.cmake)
if(NOT HP_MHC_TORCH_SOURCE_ROOT)
    set(HP_MHC_TORCH_SOURCE_ROOT ${HP_MHC_COMPONENT}/torch)
endif()

if(NOT HP_MULTICORE_VENDOR_ROOT OR
    NOT EXISTS "${HP_MULTICORE_VENDOR_ROOT}/op_api/include")
    message(FATAL_ERROR
        "HP_MULTICORE_VENDOR_ROOT must identify a complete "
        "MHC vendor")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(find_python_package_root package_name output_variable)
    string(CONCAT package_lookup_code
        "import importlib.util, sys; "
        "spec = importlib.util.find_spec(sys.argv[1]); "
        "locations = getattr(spec, 'submodule_search_locations', None) or []; "
        "print(next(iter(locations), ''))")
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
                -c "${package_lookup_code}" "${package_name}"
        RESULT_VARIABLE package_lookup_status
        OUTPUT_VARIABLE package_root
        ERROR_VARIABLE package_lookup_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT package_lookup_status EQUAL 0 OR
      NOT IS_DIRECTORY "${package_root}")
        message(FATAL_ERROR
            "Cannot locate ${package_name} for ${Python3_EXECUTABLE}: "
            "${package_lookup_error}")
    endif()
    set(${output_variable} "${package_root}" PARENT_SCOPE)
endfunction()

find_python_package_root("torch" TORCH_PACKAGE_ROOT)
find_python_package_root("torch_npu" TORCH_NPU_PACKAGE_ROOT)
set(torch_abi_code
    "import torch; print(int(torch.compiled_with_cxx11_abi()))")
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
            ${Python3_EXECUTABLE} -c "${torch_abi_code}"
    RESULT_VARIABLE torch_abi_status
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    ERROR_VARIABLE torch_abi_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT torch_abi_status EQUAL 0 OR
  NOT TORCH_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR
        "Cannot determine the active Torch C++ ABI: ${torch_abi_error}")
endif()

set(TORCH_INCLUDE_ROOT "${TORCH_PACKAGE_ROOT}/include")
set(TORCH_NPU_INCLUDE_ROOT "${TORCH_NPU_PACKAGE_ROOT}/include")
set(OP_PLUGIN_ROOT "${TORCH_NPU_INCLUDE_ROOT}/third_party/op-plugin")
set(TORCH_LINK_LIBRARIES
    "${TORCH_PACKAGE_ROOT}/lib/libc10.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch.so"
    "${TORCH_PACKAGE_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_NPU_PACKAGE_ROOT}/lib/libtorch_npu.so")

foreach(required_path IN ITEMS
    ${Python3_INCLUDE_DIRS}
    "${TORCH_INCLUDE_ROOT}"
    "${TORCH_INCLUDE_ROOT}/torch/csrc/api/include"
    "${TORCH_NPU_INCLUDE_ROOT}"
    "${OP_PLUGIN_ROOT}/op_plugin/include/npu_cpp_extension.h"
    ${TORCH_LINK_LIBRARIES})
    if(NOT EXISTS "${required_path}")
        message(FATAL_ERROR
            "The active Torch environment is missing a build input: "
            "${required_path}")
    endif()
endforeach()

add_library(hyper_parallel_mega_mhc_torch SHARED
    ${HP_MHC_TORCH_SOURCE_ROOT}/csrc/registration_mhc.cpp
    ${HP_MHC_TORCH_SOURCE_ROOT}/csrc/mega_mhc.cpp)
target_compile_definitions(hyper_parallel_mega_mhc_torch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_include_directories(hyper_parallel_mega_mhc_torch PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_ROOT}
    ${TORCH_INCLUDE_ROOT}/torch/csrc/api/include
    ${TORCH_NPU_INCLUDE_ROOT}
    ${TORCH_NPU_INCLUDE_ROOT}/third_party/acl/inc
    ${OP_PLUGIN_ROOT}
    ${OP_PLUGIN_ROOT}/op_plugin/include
    ${HP_MULTICORE_VENDOR_ROOT}/op_api/include)
target_link_libraries(hyper_parallel_mega_mhc_torch PRIVATE
    ${TORCH_LINK_LIBRARIES})
hp_enable_elf_hardening(hyper_parallel_mega_mhc_torch)

install(TARGETS hyper_parallel_mega_mhc_torch DESTINATION .)
