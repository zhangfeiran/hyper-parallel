# Copyright 2026 Huawei Technologies Co., Ltd
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
set(CMAKE_CXX_COMPILER bisheng)
project(hyper_parallel_replica_planner LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CANN $ENV{ASCEND_HOME_PATH})
if(NOT EXISTS "${CANN}/include/acl/acl.h")
  message(FATAL_ERROR "Source the CANN set_env.sh before building the replica planner")
endif()
add_library(kernel OBJECT kernel.cpp)
target_include_directories(kernel PRIVATE ${CANN}/compiler/tikcpp ${CANN}/compiler/tikcpp/tikcfw
  ${CANN}/compiler/tikcpp/tikcfw/impl ${CANN}/compiler/tikcpp/tikcfw/interface
  ${CANN}/include ${CANN}/include/experiment/runtime)
target_compile_options(kernel PRIVATE -O2 -xcce --cce-aicore-arch=dav-c220-vec
  -Wno-macro-redefined -Wno-ignored-attributes -DL2_CACHE_HINT -DTILING_KEY_VAR
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000" "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-addr-transform" "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")
add_library(hyper_parallel_replica_planner SHARED $<TARGET_OBJECTS:kernel>)
target_link_options(hyper_parallel_replica_planner PRIVATE --cce-fatobj-link)
target_link_directories(hyper_parallel_replica_planner PRIVATE ${CANN}/lib64)
target_link_libraries(hyper_parallel_replica_planner PRIVATE runtime ascendcl stdc++ m dl)

install(TARGETS hyper_parallel_replica_planner LIBRARY DESTINATION .)
