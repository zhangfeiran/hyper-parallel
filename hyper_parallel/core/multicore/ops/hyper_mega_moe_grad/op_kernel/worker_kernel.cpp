/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.cpp
 * @brief MoE-FFN backward — op-specific compute kernels + KernelWorker specialization.
 */

#include "kernel_operator.h"
#include "swi_glu/swi_glu.cpp"
#include "swi_glu_grad/swi_glu_grad.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/worker_kernel.h"

using namespace AscendC;

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
  // input_list layout for hyper_mega_moe_grad (backward):
  //   [33] = tiling params  [34] = all_event_counters
  //   [25] = gmm_workspace  [26] = swi_glu_grad_workspace
  static constexpr uint32_t TILING_IDX = 33;
  static constexpr uint32_t EVENT_IDX = 34;
  static constexpr uint32_t WORKSPACE_IDX = 25;
  static constexpr uint32_t SWIGLU_GRAD_WORKSPACE_IDX = 26;

#ifndef __DAV_C220_CUBE__
  // Weight-gradient GMMs fully overwrite nonempty experts but skip zero-row
  // experts. Clear only those skipped blocks before the scheduler starts.
  // worker_id_ spans both AIV subblocks, so each lane owns disjoint rows.
  template <typename T>
  __aicore__ inline void ZeroEmptyExpertMatrix(GM_ADDR output_addr, GM_ADDR group_list_addr, int64_t expert,
                                               int64_t rows, int64_t cols, int64_t local_experts) {
    if (output_addr == nullptr || group_list_addr == nullptr || rows <= 0 || cols <= 0 || local_experts <= 0 ||
        expert < 0 || expert >= local_experts) {
      return;
    }
    GlobalTensor<int64_t> group_list;
    group_list.SetGlobalBuffer((__gm__ int64_t *)group_list_addr);
    const int64_t end = group_list.GetValue(static_cast<uint32_t>(expert));
    const int64_t begin = expert == 0 ? 0 : group_list.GetValue(static_cast<uint32_t>(expert - 1));
    if (end != begin) {
      return;
    }

    const int64_t lane = static_cast<int64_t>(this->worker_id_);
    const int64_t lanes = this->vector_num;
    if (lanes <= 0) {
      return;
    }
    // Keep row partitions aligned to one 32-byte bf16/fp16 block.
    const int64_t rows_per_lane = ((rows + lanes - 1) / lanes + 15) / 16 * 16;
    const int64_t row_begin = lane * rows_per_lane;
    if (row_begin >= rows) {
      return;
    }
    const int64_t row_count = (rows - row_begin < rows_per_lane) ? rows - row_begin : rows_per_lane;
    const uint64_t expert_offset =
      static_cast<uint64_t>(expert) * static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    const uint64_t row_offset = static_cast<uint64_t>(row_begin) * static_cast<uint64_t>(cols);
    GlobalTensor<T> output;
    output.SetGlobalBuffer((__gm__ T *)output_addr);
    InitOutput<T>(output[expert_offset + row_offset], static_cast<uint32_t>(row_count * cols), 0);
  }

  __aicore__ inline void ZeroEmptyWeightGradBlocks() {
    const int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    const int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    const int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    const int64_t intermediate_size = getExtraValueFromTiling(input_list[TILING_IDX], 6);
    if (ep <= 0 || expert_num <= 0 || hidden_size <= 0 || intermediate_size <= 0 || expert_num % ep != 0) {
      return;
    }
    const int64_t local_experts = expert_num / ep;
    GM_ADDR group_list = input_list[19];
    for (int64_t expert = 0; expert < local_experts; ++expert) {
      ZeroEmptyExpertMatrix<DTYPE_DISPATCH_TARGET>(input_list[6], group_list, expert, intermediate_size, hidden_size,
                                                   local_experts);
      ZeroEmptyExpertMatrix<DTYPE_DISPATCH_TARGET>(input_list[18], group_list, expert, hidden_size,
                                                   intermediate_size * 2, local_experts);
    }
  }
#endif

  __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
    switch (task_desc.task_type) {
      case TASK_BEGIN_TASK_GRAPH:
        break;
      case TASK_MATMUL:
        ExecuteMatmul(task_desc);
        break;
      case TASK_GROUPED_MATMUL:
        ExecuteGroupedMatmul(task_desc);
        break;
      case TASK_SHMEM_PUT_MEM_SIGNAL:
        ExecuteShmemPutMem(task_desc);
        break;
      case TASK_SWI_GLU_GRAD:
        ExecuteSwiGluGrad(task_desc);
        break;
      case TASK_UNPERMUTE_REDUCE_GRAD:
        ExecuteUnpermuteReduceGrad(task_desc);
        break;
#ifndef __DAV_C220_CUBE__
      case TASK_WEIGHTED_UNPERMUTE:
        if (task_desc.extra_value_1 != 0) {
          AcquireRemoteWrites();
        }
        break;
      case TASK_TOPK_WEIGHT_GRAD:
        ExecuteTopKWeightGrad(task_desc);
        break;
#endif
      default:
        break;
    }
  }

 private:
  __aicore__ inline bool getTransposeData(uint32_t data) { return data == 1; }

  __aicore__ inline void ExecuteMatmul(const TaskDesc &task_desc) {}

#ifndef __DAV_C220_CUBE__
  __aicore__ inline void AcquireRemoteWrites() {
    AscendC::GlobalTensor<uint8_t> cache;
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
      cache);
    __asm__ __volatile__("");
  }
#endif

  template <typename T>
  __aicore__ inline LocalTensor<T> MakeLocalTensorFromUb(uint64_t address, uint32_t elements) {
    LocalTensor<T> local;
    TBuffAddr buffer_address;
    buffer_address.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
    buffer_address.bufferAddr = address;
#if ASCENDC_CPU_DEBUG
    buffer_address.dataLen = elements * sizeof(T);
    buffer_address.absAddr = reinterpret_cast<uint8_t *>(address);
#endif
    local.SetAddr(buffer_address);
    local.SetSize(elements);
    return local;
  }

  __aicore__ inline void ExecuteUnpermuteReduceGrad(const TaskDesc &task_desc) {
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t top_k = static_cast<int64_t>(task_desc.extra_value_0);
    int64_t row_begin = static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
    int64_t row_end = row_begin + task_desc.task_split_value;
    int64_t total_rows = task_desc.outputs[0].dim[0];
    if (row_end > total_rows) {
      row_end = total_rows;
    }
    if (row_begin >= row_end) {
      return;
    }

    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *src =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[0].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *dst =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.outputs[0].input_position]);
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t reduce_ub = device_state->mte_config.aclshmem_ub + device_state->mte_config.ub_size;
    constexpr int64_t chunk_elems = 4096;
    uint64_t value_bytes = chunk_elems * sizeof(DTYPE_DISPATCH_TARGET);
    __ubuf__ DTYPE_DISPATCH_TARGET *value_ping_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub);
    __ubuf__ DTYPE_DISPATCH_TARGET *value_pong_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub + value_bytes);
    __ubuf__ DTYPE_DISPATCH_TARGET *output_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(reduce_ub + value_bytes * 2);
    LocalTensor<DTYPE_DISPATCH_TARGET> value_ping =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> value_pong =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub + value_bytes, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> output =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(reduce_ub + value_bytes * 2, chunk_elems);
    LocalTensor<float> accumulator = MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3, chunk_elems);
    LocalTensor<float> temporary_ping =
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + chunk_elems * sizeof(float), chunk_elems);
    LocalTensor<float> temporary_pong =
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + chunk_elems * sizeof(float) * 2, chunk_elems);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    for (int64_t row = row_begin; row < row_end; ++row) {
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        int64_t first_row = route_index[row * top_k];
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(value_ping_ub, src + first_row * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (top_k > 1) {
          int64_t second_row = route_index[row * top_k + 1];
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
          aclshmemi_copy_gm2ub(value_pong_ub, src + second_row * hidden_size + col, bytes);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        }
        Cast(accumulator, value_ping, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int64_t k = 1; k < top_k; ++k) {
          bool pong = (k & 1) != 0;
          AscendC::TEventID event_id = pong ? EVENT_ID1 : EVENT_ID0;
          LocalTensor<DTYPE_DISPATCH_TARGET> value = pong ? value_pong : value_ping;
          LocalTensor<float> temporary = pong ? temporary_pong : temporary_ping;
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
          if (k + 1 < top_k) {
            bool next_pong = ((k + 1) & 1) != 0;
            AscendC::TEventID next_event_id = next_pong ? EVENT_ID1 : EVENT_ID0;
            __ubuf__ DTYPE_DISPATCH_TARGET *next_value_ub = next_pong ? value_pong_ub : value_ping_ub;
            int64_t next_row = route_index[row * top_k + k + 1];
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next_event_id);
            aclshmemi_copy_gm2ub(next_value_ub, src + next_row * hidden_size + col, bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next_event_id);
          }
          Cast(temporary, value, RoundMode::CAST_NONE, elements);
          PipeBarrier<PIPE_V>();
          Add(accumulator, accumulator, temporary, elements);
          PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        Cast(output, accumulator, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        aclshmemi_copy_ub2gm(dst + row * hidden_size + col, output_ub, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
  }

#ifndef __DAV_C220_CUBE__
  __aicore__ inline void RunTopKWeightGrad(__gm__ DTYPE_DISPATCH_TARGET *expert_output,
                                           __gm__ DTYPE_DISPATCH_TARGET *grad_output, __gm__ int32_t *route_index,
                                           __gm__ float *grad_topk_weight, int64_t hidden_size, int64_t top_k,
                                           int64_t token_begin, int64_t token_end) {
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t dot_ub = device_state->mte_config.aclshmem_ub + device_state->mte_config.ub_size;
    constexpr int64_t chunk_elems = 2048;
    uint64_t value_bytes = chunk_elems * sizeof(DTYPE_DISPATCH_TARGET);
    uint64_t float_bytes = chunk_elems * sizeof(float);
    __ubuf__ DTYPE_DISPATCH_TARGET *expert_ping_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub);
    __ubuf__ DTYPE_DISPATCH_TARGET *expert_pong_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub + value_bytes);
    __ubuf__ DTYPE_DISPATCH_TARGET *grad_ub =
      reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(dot_ub + value_bytes * 2);
    LocalTensor<DTYPE_DISPATCH_TARGET> expert_ping = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> expert_pong =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub + value_bytes, chunk_elems);
    LocalTensor<DTYPE_DISPATCH_TARGET> grad_local =
      MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(dot_ub + value_bytes * 2, chunk_elems);
    LocalTensor<float> expert_float = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3, chunk_elems);
    LocalTensor<float> grad_float = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes, chunk_elems);
    LocalTensor<float> reduce_work =
      MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes * 2, chunk_elems);
    LocalTensor<float> reduce_out = MakeLocalTensorFromUb<float>(dot_ub + value_bytes * 3 + float_bytes * 3, 8);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    for (int64_t token = token_begin; token < token_end; ++token) {
      int32_t routed_rows[16];
      float dots[16];
      for (int64_t k = 0; k < top_k; ++k) {
        routed_rows[k] = route_index[token * top_k + k];
        dots[k] = 0.0f;
      }
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        aclshmemi_copy_gm2ub(grad_ub, grad_output + token * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        Cast(grad_float, grad_local, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(expert_ping_ub, expert_output + static_cast<int64_t>(routed_rows[0]) * hidden_size + col,
                             bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t k = 0; k < top_k; ++k) {
          bool pong = (k & 1) != 0;
          AscendC::TEventID event_id = pong ? EVENT_ID1 : EVENT_ID0;
          LocalTensor<DTYPE_DISPATCH_TARGET> expert = pong ? expert_pong : expert_ping;
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
          if (k + 1 < top_k) {
            bool next_pong = ((k + 1) & 1) != 0;
            AscendC::TEventID next_event_id = next_pong ? EVENT_ID1 : EVENT_ID0;
            __ubuf__ DTYPE_DISPATCH_TARGET *next_expert_ub = next_pong ? expert_pong_ub : expert_ping_ub;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next_event_id);
            aclshmemi_copy_gm2ub(next_expert_ub,
                                 expert_output + static_cast<int64_t>(routed_rows[k + 1]) * hidden_size + col, bytes);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next_event_id);
          }
          Cast(expert_float, expert, RoundMode::CAST_NONE, elements);
          PipeBarrier<PIPE_V>();
          Mul(expert_float, expert_float, grad_float, elements);
          PipeBarrier<PIPE_V>();
          ReduceSum(reduce_out, expert_float, reduce_work, elements);
          SyncFunc<AscendC::HardEvent::V_S>();
          dots[k] += reduce_out.GetValue(0);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        }
      }
      for (int64_t k = 0; k < top_k; ++k) {
        grad_topk_weight[token * top_k + k] = dots[k];
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
  }

  __aicore__ inline void ExecuteTopKWeightGrad(const TaskDesc &task_desc) {
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t top_k = static_cast<int64_t>(task_desc.extra_value_0);
    int64_t token_begin = static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
    int64_t token_end = token_begin + task_desc.task_split_value;
    int64_t token_count = task_desc.outputs[0].dim[0];
    if (token_end > token_count) {
      token_end = token_count;
    }
    if (token_begin >= token_end) {
      return;
    }
    __gm__ DTYPE_DISPATCH_TARGET *expert_output =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[0].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *grad_output =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.inputs[2].input_position]);
    __gm__ float *grad_topk_weight = (__gm__ float *)(input_list[task_desc.outputs[0].input_position]);
    RunTopKWeightGrad(expert_output, grad_output, route_index, grad_topk_weight, hidden_size, top_k, token_begin,
                      token_end);
  }
#endif

  __aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, int64_t length) {
    __gm__ uint8_t *start =
      (__gm__ uint8_t *)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
      (__gm__ uint8_t *)(((int64_t)sourceAddr + length) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
      __asm__ __volatile__("");
      AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(global[i]);
      __asm__ __volatile__("");
    }
  }

  __aicore__ inline void ExecuteSwiGluGrad(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.outputs[0].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[dynamic_data.dynamic_input_position];
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        uint32_t grouped_list_shape = dynamic_data.dynamic_group_size;
        uint32_t per_task_num = task_desc.task_split_num / grouped_list_shape;
        uint32_t data_index = task_desc.task_index / per_task_num;
        uint32_t task_index = task_desc.task_index % per_task_num;
        int64_t start = 0;
        int64_t end = grouped_list_tensor.GetValue(data_index);
        if (data_index != 0) {
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        int64_t current_seq_start = task_index * task_desc.task_split_value;
        int64_t current_seq_end = (task_index + 1) * task_desc.task_split_value;
        int64_t base_ptr_offset = start + current_seq_start;

        if (base_ptr_offset >= end) {
          return;
        }
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
        if (start + current_seq_end <= end) {
          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                       input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset);
          return;
        } else {
          GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 80 * (AscendC::GetBlockIdx() + 1);
          __gm__ SwiGluTilingData *tilingdata_data = reinterpret_cast<__gm__ SwiGluTilingData *>(tiling_data_addr);
          tilingdata_data->rowLen = end - (start + current_seq_start);
          if (tilingdata_data->rowLen == 0) {
            return;
          }
          if (end - (start + current_seq_start) < 19) {
            tilingdata_data->baseRowLen = end - (start + current_seq_start);
          }
          cacheWriteThrough(tiling_data_addr, 10);
          PipeBarrier<PIPE_ALL>();

          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX], tiling_data_addr);
        }
      }
      return;
    }
  }

  __aicore__ inline void ExecuteGroupedMatmul(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.inputs[1].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[task_desc.inputs[2].input_position] +
                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        GM_ADDR grouped_list_real = this->runtimeConfigPtr + getGroupedMatmulGroupListOffsetById(
                                                               this->runtimeConfigPtr, AscendC::GetBlockIdx() * 2);
        GlobalTensor<int64_t> grouped_list_tensor_real;
        grouped_list_tensor_real.SetGlobalBuffer((__gm__ int64_t *)(grouped_list_real), expert_num_single_rank);
        uint32_t data_index = task_desc.task_index / this->core_num;
        int64_t value = grouped_list_tensor.GetValue(data_index);
        int64_t start = 0;
        if (data_index != 0) {
          value = value - grouped_list_tensor.GetValue(data_index - 1);
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        if (value == 0) {
          return;
        }
        for (uint32_t i = 0; i < expert_num_single_rank; i++) {
          if (i > data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else if (i == data_index) {
            grouped_list_tensor_real.SetValue(i, value);
          } else {
            grouped_list_tensor_real.SetValue(i, 0);
          }
        }
        cacheWriteThrough(grouped_list_real, expert_num_single_rank);

        GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 2016 * AscendC::GetBlockIdx();
        __gm__ GMMTilingData *tilingdata_data = reinterpret_cast<__gm__ GMMTilingData *>(tiling_data_addr);
        if (getTransposeData(task_desc.inputs[0].transpose_flag)) {
          tilingdata_data->mmTilingData.Ka = value;
          tilingdata_data->mmTilingData.Kb = value;
          tilingdata_data->mmTilingData.singleCoreK = value;
        } else {
          tilingdata_data->gmmBaseParams.m = value;
          tilingdata_data->mmTilingData.M = value;
          tilingdata_data->mmTilingData.singleCoreM = value;
        }
        cacheWriteThrough(tiling_data_addr, 220);
        PipeBarrier<PIPE_ALL>();

        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? start * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? start * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? start * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;

        grouped_matmul(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset, nullptr, nullptr, nullptr,
                       nullptr, nullptr, grouped_list_real, nullptr,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset, input_list[WORKSPACE_IDX],
                       tiling_data_addr, getTransposeData(task_desc.inputs[0].transpose_flag),
                       getTransposeData(task_desc.inputs[1].transpose_flag));
      }
      return;
    }
  }

  __aicore__ inline void ExecuteShmemPutMem(const TaskDesc &task_desc) {
    if (task_desc.extra_value_0 != 0xFFFFFFFFU) {
      ExecuteRoutedShmemPutMem(task_desc);
      return;
    }
    GM_ADDR signal = input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      send_data_size_ = 0;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    put_mem_signal_kernel(input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_, send_data_size_, signal, static_cast<int64_t>(task_desc.trigger_event), 1,
                          nullptr, 1, target_pe, false);
  }

  __aicore__ inline void ExecuteRoutedShmemPutMem(const TaskDesc &task_desc) {
    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    int64_t tasks_per_expert = task_desc.task_split_num / expert_num;
    int64_t target_pe = task_desc.task_index / (local_experts * tasks_per_expert);

    int32_t size = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                        task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int64_t tile = task_desc.task_index % tasks_per_expert;
    int64_t start = tile * task_desc.task_split_value * hidden_size;
    if (start >= size) {
      this->PublishRouteDispatchArrival(task_desc.task_index, task_desc.task_split_num, target_pe);
#ifndef __DAV_C220_CUBE__
      if (task_desc.extra_value_2 != EVENT_INVALID_ID) {
        this->TriggerEvent(task_desc.extra_value_2);
      }
#endif
      return;
    }

    int64_t target_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                 task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                              task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int64_t send_elems = task_desc.task_split_value * hidden_size;
    if (start + send_elems > size) {
      send_elems = size - start;
    }
    int64_t rows = send_elems / hidden_size;
    int64_t logical_src_row = (src_offset + start) / hidden_size;
    int64_t target_base = target_offset + start;

    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.extra_value_0]);
    __gm__ float *route_weight = (__gm__ float *)(input_list[task_desc.extra_value_1]);
    __gm__ DTYPE_DISPATCH_TARGET *src =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *target =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.outputs[0].input_position] +
                                       task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type);
    __gm__ DTYPE_DISPATCH_TARGET *dst = (__gm__ DTYPE_DISPATCH_TARGET *)(aclshmem_ptr(target + target_base, target_pe));

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    constexpr int64_t chunk_elems = 8192;
    __ubuf__ DTYPE_DISPATCH_TARGET *value_ub = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(copy_ub);
    LocalTensor<DTYPE_DISPATCH_TARGET> value = MakeLocalTensorFromUb<DTYPE_DISPATCH_TARGET>(copy_ub, chunk_elems);
    LocalTensor<float> temporary =
      MakeLocalTensorFromUb<float>(copy_ub + chunk_elems * sizeof(DTYPE_DISPATCH_TARGET), chunk_elems);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t row = 0; row < rows; ++row) {
      int64_t route_row = logical_src_row + row;
      int32_t token_row = route_index[route_row];
      float weight = route_weight[route_row];
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        aclshmemi_copy_gm2ub(value_ub, src + static_cast<int64_t>(token_row) * hidden_size + col, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Cast(temporary, value, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(temporary, temporary, weight, elements);
        PipeBarrier<PIPE_V>();
        Cast(value, temporary, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        aclshmemi_copy_ub2gm(dst + row * hidden_size + col, value_ub, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    aclshmem_fence();
    this->PublishRouteDispatchArrival(task_desc.task_index, task_desc.task_split_num, target_pe);
#ifndef __DAV_C220_CUBE__
    if (task_desc.extra_value_2 != EVENT_INVALID_ID) {
      this->TriggerEvent(task_desc.extra_value_2);
    }
#endif
  }
};

extern "C" inline __aicore__ void worker_kernel(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr,
                                                GM_ADDR *input_list) {
  KernelWorker worker;
  worker.Init(worker_id, runtimeConfigPtr, input_list);
#ifndef __DAV_C220_CUBE__
  worker.ZeroEmptyWeightGradBlocks();
#endif
  worker.Process();
}
