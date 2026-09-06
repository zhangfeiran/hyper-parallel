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
 * @brief MoE-FFN forward — op-specific compute kernels + KernelWorker specialization.
 */

#include "kernel_operator.h"
#include "swi_glu/swi_glu.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/worker_kernel.h"

using namespace AscendC;

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
  // input_list layout for hyper_mega_moe (forward):
  //   [28] = tiling params  [29] = all_event_counters  [11] = gmm_workspace
  static constexpr uint32_t TILING_IDX = 28;
  static constexpr uint32_t EVENT_IDX = 29;
  static constexpr uint32_t WORKSPACE_IDX = 11;
  static constexpr uint32_t RECEIVE_CAPACITY_TILING_INDEX = 6;
  static constexpr uint32_t WAVE_COUNT_TILING_INDEX = 7;
  static constexpr uint32_t HOME_EXPERTS_TILING_INDEX = 8;
  static constexpr bool HAS_MID_WAVE_ROUTE_BARRIER = false;
  static constexpr bool CYCLE_PROFILE_ENABLED = false;

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
      case TASK_SWI_GLU:
        ExecuteSwiglu(task_desc);
        break;
      case TASK_SHMEM_PUT_MEM_SIGNAL:
        ExecuteShmemPutMem(task_desc);
        break;
#ifndef __DAV_C220_CUBE__
      case TASK_WEIGHTED_UNPERMUTE:
        ExecuteWeightedUnpermute(task_desc);
        break;
#endif
      default:
        break;
    }
  }

 private:
  __aicore__ inline void ExecuteMatmul(const TaskDesc &task_desc) {}

  __aicore__ inline uint32_t DecodeRouteToken(uint32_t route_slot, uint32_t top_k) {
    switch (top_k) {
      case 1U:
        return route_slot;
      case 2U:
        return route_slot >> 1;
      case 3U:
        return route_slot / 3U;
      case 4U:
        return route_slot >> 2;
      case 5U:
        return route_slot / 5U;
      case 6U:
        return route_slot / 6U;
      case 7U:
        return route_slot / 7U;
      case 8U:
        return route_slot >> 3;
      case 9U:
        return route_slot / 9U;
      case 10U:
        return route_slot / 10U;
      case 11U:
        return route_slot / 11U;
      case 12U:
        return route_slot / 12U;
      case 13U:
        return route_slot / 13U;
      case 14U:
        return route_slot / 14U;
      case 15U:
        return route_slot / 15U;
      case 16U:
        return route_slot >> 4;
      default:
        return 0U;
    }
  }

#ifndef __DAV_C220_CUBE__
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

  __aicore__ inline void AcquireRemoteWrites() {
    AscendC::GlobalTensor<uint8_t> cache;
    __asm__ __volatile__("");
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
      cache);
    __asm__ __volatile__("");
  }

  __aicore__ inline void LoadRouteChunk(__gm__ DTYPE_DISPATCH_TARGET *source, __ubuf__ DTYPE_DISPATCH_TARGET *buffer,
                                        uint64_t bytes, AscendC::TEventID value_event, AscendC::TEventID save_event) {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(value_event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(save_event);
    aclshmemi_copy_gm2ub(buffer, source, bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(value_event);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(save_event);
  }

  __aicore__ inline void SaveRouteChunk(__gm__ DTYPE_DISPATCH_TARGET *target, __ubuf__ DTYPE_DISPATCH_TARGET *buffer,
                                        uint64_t bytes, AscendC::TEventID save_event) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(save_event);
    aclshmemi_copy_ub2gm(target, buffer, bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(save_event);
  }

  __aicore__ inline void RunWeightedUnpermute(__gm__ DTYPE_DISPATCH_TARGET *expert_output, __gm__ int32_t *route_index,
                                              __gm__ float *topk_weights, __gm__ DTYPE_DISPATCH_TARGET *final_out,
                                              __gm__ DTYPE_DISPATCH_TARGET *saved_expert_output, int64_t hidden_size,
                                              int64_t top_k, int64_t token_begin, int64_t token_end) {
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t reduce_ub = device_state->mte_config.aclshmem_ub + device_state->mte_config.ub_size;
    constexpr int64_t chunk_elems = 4096;
    uint64_t value_bytes = chunk_elems * sizeof(DTYPE_DISPATCH_TARGET);
    uint64_t float_bytes = chunk_elems * sizeof(float);
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
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + float_bytes, chunk_elems);
    LocalTensor<float> temporary_pong =
      MakeLocalTensorFromUb<float>(reduce_ub + value_bytes * 3 + float_bytes * 2, chunk_elems);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
    for (int64_t token = token_begin; token < token_end; ++token) {
      int64_t route_base = token * top_k;
      for (int64_t col = 0; col < hidden_size; col += chunk_elems) {
        int64_t elements = hidden_size - col;
        if (elements > chunk_elems) {
          elements = chunk_elems;
        }
        uint64_t bytes = elements * sizeof(DTYPE_DISPATCH_TARGET);
        int32_t first_row = route_index[route_base];
        LoadRouteChunk(expert_output + static_cast<int64_t>(first_row) * hidden_size + col, value_ping_ub, bytes,
                       EVENT_ID0, EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (top_k > 1) {
          int32_t second_row = route_index[route_base + 1];
          LoadRouteChunk(expert_output + static_cast<int64_t>(second_row) * hidden_size + col, value_pong_ub, bytes,
                         EVENT_ID1, EVENT_ID4);
        }
        SaveRouteChunk(saved_expert_output + static_cast<int64_t>(first_row) * hidden_size + col, value_ping_ub, bytes,
                       EVENT_ID3);
        Cast(temporary_ping, value_ping, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        float first_weight = topk_weights[route_base];
        Muls(accumulator, temporary_ping, first_weight, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        for (int64_t k = 1; k < top_k; ++k) {
          int64_t route_offset = route_base + k;
          float weight = topk_weights[route_offset];
          bool pong = (k & 1) != 0;
          AscendC::TEventID event_id = pong ? EVENT_ID1 : EVENT_ID0;
          AscendC::TEventID save_event_id = pong ? EVENT_ID4 : EVENT_ID3;
          LocalTensor<DTYPE_DISPATCH_TARGET> value = pong ? value_pong : value_ping;
          __ubuf__ DTYPE_DISPATCH_TARGET *value_ub = pong ? value_pong_ub : value_ping_ub;
          LocalTensor<float> temporary = pong ? temporary_pong : temporary_ping;
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
          SaveRouteChunk(saved_expert_output + static_cast<int64_t>(route_index[route_offset]) * hidden_size + col,
                         value_ub, bytes, save_event_id);
          if (k + 1 < top_k) {
            bool next_pong = ((k + 1) & 1) != 0;
            AscendC::TEventID next_event_id = next_pong ? EVENT_ID1 : EVENT_ID0;
            AscendC::TEventID next_save_event_id = next_pong ? EVENT_ID4 : EVENT_ID3;
            __ubuf__ DTYPE_DISPATCH_TARGET *next_value_ub = next_pong ? value_pong_ub : value_ping_ub;
            int32_t next_row = route_index[route_offset + 1];
            LoadRouteChunk(expert_output + static_cast<int64_t>(next_row) * hidden_size + col, next_value_ub, bytes,
                           next_event_id, next_save_event_id);
          }
          Cast(temporary, value, RoundMode::CAST_NONE, elements);
          PipeBarrier<PIPE_V>();
          Axpy(accumulator, temporary, weight, elements);
          PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        Cast(output, accumulator, RoundMode::CAST_RINT, elements);
        PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        aclshmemi_copy_ub2gm(final_out + token * hidden_size + col, output_ub, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
  }

  __aicore__ inline void ExecuteWeightedUnpermute(const TaskDesc &task_desc) {
    if (task_desc.extra_value_1 != 0) {
      AcquireRemoteWrites();
      return;
    }
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
    __gm__ int32_t *route_index = (__gm__ int32_t *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ float *topk_weights = (__gm__ float *)(input_list[task_desc.inputs[2].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *final_out =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.outputs[0].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *saved_expert_output =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[3].input_position]);
    RunWeightedUnpermute(expert_output, route_index, topk_weights, final_out, saved_expert_output, hidden_size, top_k,
                         token_begin, token_end);
  }
#endif

  __aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, int64_t length) {
    if (length <= 0) {
      return;
    }
    __gm__ uint8_t *start =
      (__gm__ uint8_t *)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
      (__gm__ uint8_t *)(((int64_t)sourceAddr + length - 1) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
      AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(global[i]);
    }
  }

  __aicore__ inline void ExecuteSwiglu(const TaskDesc &task_desc) {
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
        int64_t end = 0;
        this->GetWaveExpertRange(grouped_list_tensor, data_index, &start, &end);
        int64_t current_seq_start = task_index * task_desc.task_split_value;
        int64_t current_seq_end = (task_index + 1) * task_desc.task_split_value;
        int64_t base_ptr_offset = start + current_seq_start;
        int64_t retained_row_offset =
          this->GetWaveGlobalRowBase() + base_ptr_offset;
        int64_t static_row_offset =
          static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
        if (base_ptr_offset >= end) {
          return;
        }
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? retained_row_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : static_row_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : static_row_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type;
        if (start + current_seq_end <= end) {
          swi_glu(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                  input_list[task_desc.outputs[0].input_position] + output_0_offset, nullptr,
                  input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset);
        } else {
          uint32_t tail_rows = end - (start + current_seq_start);
          if (tail_rows == 0) {
            return;
          }
          uint32_t tail_tiling_index = 2 * this->core_num + tail_rows;
          GM_ADDR tiling_data_addr =
            input_list[task_desc.tiling_data_position] + 80 * tail_tiling_index;
          swi_glu(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                  input_list[task_desc.outputs[0].input_position] + output_0_offset, nullptr, tiling_data_addr);
        }
      }
      return;
    }
    int64_t row_offset =
      static_cast<int64_t>(task_desc.task_index) * task_desc.task_split_value;
    swi_glu(input_list[task_desc.inputs[0].input_position] +
              row_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type,
            input_list[task_desc.outputs[0].input_position] +
              row_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type,
            nullptr, input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset);
  }

  __aicore__ inline void ExecuteSingleGroupMatmul(
      const TaskDesc &task_desc,
      GlobalTensor<int64_t> &grouped_list_tensor) {
    GM_ADDR grouped_list_real =
      this->runtimeConfigPtr + getGroupedMatmulGroupListOffsetById(
        this->runtimeConfigPtr, AscendC::GetBlockIdx() * 2);
    GlobalTensor<int64_t> grouped_list_tensor_real;
    grouped_list_tensor_real.SetGlobalBuffer(
      (__gm__ int64_t *)(grouped_list_real), 1);
    const uint32_t data_index = task_desc.task_index / this->core_num;
    int64_t start = 0;
    int64_t expert_end = 0;
    this->GetWaveExpertRange(
      grouped_list_tensor, data_index, &start, &expert_end);
    const int64_t value = expert_end - start;
    if (value == 0) {
      return;
    }
    grouped_list_tensor_real.SetValue(0, value);
    cacheWriteThrough(grouped_list_real, INT64_T_SIZE);

    GM_ADDR tiling_data_addr =
      input_list[task_desc.tiling_data_position] +
      task_desc.tiling_data_offset + 2016 * AscendC::GetBlockIdx();
    __gm__ GMMTilingData *tilingdata_data =
      reinterpret_cast<__gm__ GMMTilingData *>(tiling_data_addr);
    tilingdata_data->gmmBaseParams.groupNum = 1;
    tilingdata_data->gmmBaseParams.m = value;
    tilingdata_data->mmTilingData.M = value;
    tilingdata_data->mmTilingData.singleCoreM = value;
    tilingdata_data->gmmBaseParams.activeType = 0;
    cacheWriteThrough(tiling_data_addr, 250);
    PipeBarrier<PIPE_ALL>();

    const int64_t global_start = this->GetWaveGlobalRowBase() + start;
    const int64_t input_row = start;
    const int64_t output_row =
      this->IsMultiWave() && task_desc.outputs[0].input_position == 7
        ? global_start
        : start;
    const int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
      ? input_row * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
      : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
    const int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
      ? output_row * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
      : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
    const int64_t matrix_bytes =
      static_cast<int64_t>(task_desc.inputs[1].dim[1]) *
      task_desc.inputs[1].dim[2] * task_desc.inputs[1].data_type;
    const uint32_t weight_position = task_desc.inputs[1].input_position;
    grouped_matmul(
      input_list[task_desc.inputs[0].input_position] + input_0_offset,
      this->GetExpertMatrix(weight_position, data_index, matrix_bytes, weight_position == 5 ? 0 : 1),
      nullptr, nullptr, nullptr, nullptr, nullptr, grouped_list_real, nullptr,
      input_list[task_desc.outputs[0].input_position] + output_0_offset,
      input_list[WORKSPACE_IDX], tiling_data_addr, false, false);
  }

  __aicore__ inline void ExecuteGroupedMatmul(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape != 1 &&
        task_desc.inputs[1].dynamic_shape != 1) {
      return;
    }
    DynamicData dynamic_data;
    getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
    if (dynamic_data.dynamic_type != DynamicType::DYNAMIC_DSV3_MOE) {
      return;
    }
    GM_ADDR grouped_list =
      input_list[task_desc.inputs[2].input_position] +
      task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
    GlobalTensor<int64_t> grouped_list_tensor;
    const int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    const int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    const int64_t local_experts = expert_num / ep;
    grouped_list_tensor.SetGlobalBuffer(
      (__gm__ int64_t *)(grouped_list), local_experts);
    ExecuteSingleGroupMatmul(task_desc, grouped_list_tensor);
  }

  __aicore__ inline void ExecuteShmemPutMem(TaskDesc task_desc) {
    if (task_desc.extra_value_0 != 0xFFFFFFFFU) {
      ExecuteRoutedShmemPutMem(task_desc);
      return;
    }
    GM_ADDR signal = input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);
    bool offsets_in_tokens = (task_desc.extra_value_4 & 0x1U) != 0;

    int64_t rank_id = (*(__gm__ int64_t *)(input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    if (this->IsMultiWave()) {
      if (!offsets_in_tokens) {
        target_offset_ /= hidden_size;
        src_offset_ /= hidden_size;
        size_ /= hidden_size;
        offsets_in_tokens = true;
      }
      if ((task_desc.extra_value_3 & 0xFFU) == 1U) {
        this->SliceWaveSegment(&target_offset_, &src_offset_, &size_);
      } else {
        this->SliceWaveSegment(&src_offset_, &target_offset_, &size_);
      }
    }
    if (offsets_in_tokens) {
      target_offset_ *= hidden_size;
      src_offset_ *= hidden_size;
      size_ *= hidden_size;
    }
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
    uint32_t top_k = task_desc.extra_value_4 >> 8;
    int64_t tasks_per_expert = task_desc.task_split_num / expert_num;
    int64_t target_pe = task_desc.task_index / (local_experts * tasks_per_expert);

    int64_t target_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                 task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                              task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                        task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    bool offsets_in_tokens = (task_desc.extra_value_4 & 0x1U) != 0;
    if (this->IsMultiWave()) {
      if (!offsets_in_tokens) {
        target_offset /= hidden_size;
        src_offset /= hidden_size;
        size /= hidden_size;
        offsets_in_tokens = true;
      }
      if ((task_desc.extra_value_3 & 0xFFU) == 1U) {
        this->SliceWaveSegment(&target_offset, &src_offset, &size);
      } else {
        this->SliceWaveSegment(&src_offset, &target_offset, &size);
      }
    }
    if (offsets_in_tokens) {
      target_offset *= hidden_size;
      src_offset *= hidden_size;
      size *= hidden_size;
    }
    int64_t tile = task_desc.task_index % tasks_per_expert;
    int64_t start = tile * task_desc.task_split_value * hidden_size;
    int64_t send_elems = task_desc.task_split_value * hidden_size;
    if (start >= size) {
      send_elems = 0;
    } else if (start + send_elems > size) {
      send_elems = size - start;
    }
    int64_t rows = send_elems / hidden_size;
    int64_t logical_src_row = (src_offset + start) / hidden_size;
    int64_t target_base = target_offset + start;
    if (rows == 0) {
      this->PublishRouteDispatchArrival(
        task_desc.task_index, task_desc.task_split_num, target_pe);
      return;
    }

    __gm__ int32_t *index_ptr = (__gm__ int32_t *)(input_list[task_desc.extra_value_0]);
    __gm__ DTYPE_DISPATCH_TARGET *src_base =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.inputs[1].input_position]);
    __gm__ DTYPE_DISPATCH_TARGET *target_ptr =
      (__gm__ DTYPE_DISPATCH_TARGET *)(input_list[task_desc.outputs[0].input_position] +
                                       task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type);
    __gm__ DTYPE_DISPATCH_TARGET *dst_base =
      (__gm__ DTYPE_DISPATCH_TARGET *)(aclshmem_ptr(target_ptr + target_base, target_pe));

    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint64_t half_ub = device_state->mte_config.ub_size / 2;
    __ubuf__ DTYPE_DISPATCH_TARGET *ping = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(copy_ub);
    __ubuf__ DTYPE_DISPATCH_TARGET *pong = reinterpret_cast<__ubuf__ DTYPE_DISPATCH_TARGET *>(copy_ub + half_ub);
    uint64_t row_bytes = hidden_size * sizeof(DTYPE_DISPATCH_TARGET);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    uint64_t chunk = 0;
    for (int64_t row = 0; row < rows; ++row) {
      uint32_t route_slot = static_cast<uint32_t>(index_ptr[logical_src_row + row]);
      uint32_t src_row = DecodeRouteToken(route_slot, top_k);
      __gm__ DTYPE_DISPATCH_TARGET *src = src_base + static_cast<int64_t>(src_row) * hidden_size;
      __gm__ DTYPE_DISPATCH_TARGET *dst = dst_base + row * hidden_size;
      for (uint64_t copied = 0; copied < row_bytes; ++chunk) {
        uint64_t bytes = row_bytes - copied;
        if (bytes > half_ub) {
          bytes = half_ub;
        }
        AscendC::TEventID event_id = (chunk & 1) ? EVENT_ID1 : EVENT_ID0;
        __ubuf__ DTYPE_DISPATCH_TARGET *buffer = (chunk & 1) ? pong : ping;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        aclshmemi_copy_gm2ub(buffer, src + copied / sizeof(DTYPE_DISPATCH_TARGET), bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        aclshmemi_copy_ub2gm(dst + copied / sizeof(DTYPE_DISPATCH_TARGET), buffer, bytes);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        copied += bytes;
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    aclshmem_fence();
    this->PublishRouteDispatchArrival(task_desc.task_index, task_desc.task_split_num, target_pe);
  }
};

extern "C" inline __aicore__ void worker_kernel(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr,
                                                GM_ADDR *input_list) {
  KernelWorker worker;
  worker.Init(worker_id, runtimeConfigPtr, input_list);
  worker.Process();
}
