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
#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_KERNEL_HYPER_DSA_FUSED_TRAINING_TILING_KEY_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_KERNEL_HYPER_DSA_FUSED_TRAINING_TILING_KEY_H_
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(HyperDsaFusedTraining, ASCENDC_TPL_BOOL_DECL(FLOAT_WEIGHTS, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TRANSPORT, 0, 1), ASCENDC_TPL_BOOL_DECL(SELECTED, 0, 1));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLOAT_WEIGHTS, 0, 1), ASCENDC_TPL_BOOL_SEL(TRANSPORT, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(SELECTED, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(FLOAT_WEIGHTS, 0, 1), ASCENDC_TPL_BOOL_SEL(TRANSPORT, 1),
                                     ASCENDC_TPL_BOOL_SEL(SELECTED, 1)));
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_HYPER_DSA_FUSED_TRAINING_OP_KERNEL_HYPER_DSA_FUSED_TRAINING_TILING_KEY_H_
