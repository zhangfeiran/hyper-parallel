/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <iostream>

#include "runtime/config.h"

int main() {
  const auto config = hyper_parallel::multicore::shmem::runtime::LoadConfigFromEnvironment();
  if (!config.ok()) {
    std::cerr << config.error().message << std::endl;
    return 2;
  }
  std::cout << config.value().udma_qp_num << std::endl;
  return 0;
}
