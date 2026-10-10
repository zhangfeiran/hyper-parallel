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
#ifndef HYPER_PARALLEL_DSA_INDEX_GRAD_RETURN_TILING_H_
#define HYPER_PARALLEL_DSA_INDEX_GRAD_RETURN_TILING_H_
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(IndexGradReturnTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens);
TILING_DATA_FIELD_DEF(int64_t, capacity);
TILING_DATA_FIELD_DEF(int64_t, arenaBytes);
TILING_DATA_FIELD_DEF(int64_t, runs);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(HyperDsaIndexGradReturn, IndexGradReturnTilingData)
}  // namespace optiling
#endif
