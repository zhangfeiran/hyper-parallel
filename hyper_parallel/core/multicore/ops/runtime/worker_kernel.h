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
 * @file worker_kernel.h
 * @brief Shared CRTP base class for AIC/AIV scheduling loop and event synchronization.
 *
 * Each op derives KernelWorker : KernelWorkerBase<KernelWorker> and supplies:
 *   - static constexpr uint32_t TILING_IDX  — index into input_list for tiling params
 *   - static constexpr uint32_t EVENT_IDX   — index into input_list for all_event_counters
 *   - void ExecuteComputeKernel(TaskDesc)    — op-specific task dispatch switch
 *
 * Compute kernels (ExecuteMatmul, ExecuteShmemPutMem, etc.) are NOT part of this class;
 * they belong in each op's worker_kernel.cpp.
 */

#ifndef MULTICORE_SCHEDULER_WORKER_KERNEL_H
#define MULTICORE_SCHEDULER_WORKER_KERNEL_H

#include "kernel_operator.h"
#include "runtime_config.hpp"

using namespace AscendC;  // NOLINT(build/namespaces)

template <typename Derived>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    this->tpipe_.InitBuffer(tBuf, DISPATCH_TOKEN_UB_SIZE);
    tpipe_.Destroy();

    all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[Derived::EVENT_IDX]), MAX_EVENT_NUM);

    all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(this->runtimeConfigPtr + getAllEventNumTriggersOffset()),
                                           MAX_EVENT_NUM);

    uint32_t runtime_task_capacity = getRuntimeTaskCapacity(this->runtimeConfigPtr);
    vector_task_indexs.SetGlobalBuffer(
      (__gm__ int32_t *)(this->runtimeConfigPtr + getVectorTaskIndexsOffset(this->runtimeConfigPtr)),
      runtime_task_capacity);

    cube_task_indexs.SetGlobalBuffer(
      (__gm__ int32_t *)(this->runtimeConfigPtr + getCubeTaskIndexsOffset(this->runtimeConfigPtr)),
      runtime_task_capacity);

    atomic_add_values.SetGlobalBuffer(
      (__gm__ int32_t *)(this->runtimeConfigPtr + getAtomicAddValuesOffset(this->runtimeConfigPtr)), 8);

    this->input_list = input_list;
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      for (uint32_t category = 0; category < CYCLE_PROFILE_CATEGORY_COUNT; ++category) {
        this->cycle_profile_[category] = 0;
      }
    }
    this->task_num = getTaskNum(this->runtimeConfigPtr);
    this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
    this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);
    this->core_num = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 5);
    this->vector_num = this->core_num * 2;
    this->receive_capacity_ = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], Derived::RECEIVE_CAPACITY_TILING_INDEX);
    int64_t wave_count = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], Derived::WAVE_COUNT_TILING_INDEX);
    this->wave_count_ = wave_count > 0 ? static_cast<uint32_t>(wave_count) : 1;
    this->home_experts_ = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], Derived::HOME_EXPERTS_TILING_INDEX);
    if (this->home_experts_ > 0) {
      getReplicaMatrixBases(this->runtimeConfigPtr, this->replica_matrix_bases_);
    }
    InitTaskProfile();
  }

  __aicore__ inline void Process() {
    ReadyHandshakeMeta meta;
    bool has_ready_handshake = LoadReadyHandshakeMeta(&meta);
    this->has_ready_handshake_ = has_ready_handshake;
    if (has_ready_handshake) {
      this->ready_handshake_meta_ = meta;
    }
    if (has_ready_handshake) {
      int64_t profile_start = ProfileStart();
      WaitForDeviceReady(meta);
      ProfileRecord(CYCLE_PROFILE_SYNCHRONIZATION, profile_start);
    }
    if (!IsMultiWave()) {
      ProcessStatic();
      FlushCycleProfile();
      return;
    }
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ % 2 == 0) {
      return;
    }
#endif
    for (this->wave_index_ = 0; this->wave_index_ < this->wave_count_;
         ++this->wave_index_) {
      this->mid_wave_route_synchronized_ = false;
      ProcessStatic();
      if (this->wave_index_ + 1 < this->wave_count_) {
        int64_t profile_start = ProfileStart();
        SynchronizeWave(meta, has_ready_handshake);
        ProfileRecord(CYCLE_PROFILE_SYNCHRONIZATION, profile_start);
      }
    }
    FlushCycleProfile();
  }

 protected:
  __aicore__ inline GM_ADDR GetExpertMatrix(
      uint32_t input_position, int64_t expert, int64_t matrix_bytes, uint32_t replica_index) const {
    if (this->home_experts_ > 0 && expert >= this->home_experts_) {
      return this->replica_matrix_bases_[replica_index] +
        (expert - this->home_experts_) * matrix_bytes;
    }
    return this->input_list[input_position] + expert * matrix_bytes;
  }

  __aicore__ inline bool IsMultiWave() const {
    return this->wave_count_ > 1 && this->receive_capacity_ > 0;
  }

  __aicore__ inline bool IsLastWave() const {
    return this->wave_index_ + 1 == this->wave_count_;
  }

  __aicore__ inline int64_t ProfileStart() const {
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      return AscendC::GetSystemCycle();
    }
    return 0;
  }

  __aicore__ inline void ProfileRecord(
      CycleProfileCategory category, int64_t start) {
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      this->cycle_profile_[static_cast<uint32_t>(category)] +=
        AscendC::GetSystemCycle() - start;
    }
  }

  static constexpr uint32_t TASK_PROFILE_MARKER = 0x4D545001;
  static constexpr uint32_t TASK_PROFILE_CAPACITY = 256;
  static constexpr uint32_t TASK_PROFILE_VALUES = 16;

  __aicore__ inline void InitTaskProfile() {
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      GM_ADDR base = input_list[Derived::CYCLE_PROFILE_WORKSPACE_IDX] + Derived::CYCLE_PROFILE_OFFSET_BYTES;
      GlobalTensor<int64_t> marker;
      marker.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(base + 3968), 1);
      this->task_profile_enabled_ = marker.GetValue(0) == TASK_PROFILE_MARKER;
      if (!this->task_profile_enabled_) {
        return;
      }
      uint32_t slot = this->worker_id_;
      uint32_t tasks = this->cube_task_num;
#ifndef __DAV_C220_CUBE__
      slot = NUM_WORKERS_CUBE + this->worker_id_ / 2;
      tasks = this->vector_task_num;
#endif
      uint32_t per_core = (tasks + this->core_num - 1) / this->core_num;
      uint32_t visits = per_core * this->wave_count_;
      this->task_profile_stride_ = (visits + TASK_PROFILE_CAPACITY - 1) / TASK_PROFILE_CAPACITY;
      if (this->task_profile_stride_ == 0) {
        this->task_profile_stride_ = 1;
      }
      this->task_profile_output_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t *>(base + 4096) +
          slot * (TASK_PROFILE_CAPACITY + 1) * TASK_PROFILE_VALUES,
        (TASK_PROFILE_CAPACITY + 1) * TASK_PROFILE_VALUES);
      this->task_profile_start_ = ProfileStart();
    }
  }

  __aicore__ inline void RecordTaskProfile(
      TaskId task_id, const TaskDesc &desc, int64_t start, int64_t ready, int64_t compute_end, int64_t end) {
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      if (!this->task_profile_enabled_ || this->task_profile_recorded_ >= TASK_PROFILE_CAPACITY ||
          (this->task_profile_seen_ - 1) % this->task_profile_stride_ != 0) {
        return;
      }
      int64_t rows = -1;
      if (desc.task_type == TASK_GROUPED_MATMUL) {
        GlobalTensor<int64_t> groups;
        groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(
          input_list[desc.inputs[2].input_position] + desc.inputs[2].base_ptr_offset * desc.inputs[2].data_type));
        int64_t begin = 0;
        int64_t finish = 0;
        GetWaveExpertRange(groups, desc.task_index / this->core_num, &begin, &finish);
        rows = finish - begin;
      }
      const int64_t values[TASK_PROFILE_VALUES] = {
        task_id, desc.task_type, desc.task_index, this->wave_index_, desc.tiling_data_position,
        desc.dependent_event, desc.trigger_event, start, ready, compute_end, end, rows,
        desc.outputs[0].input_position, desc.extra_value_3, this->task_profile_seen_ - 1, 0};
      const uint32_t offset = (++this->task_profile_recorded_) * TASK_PROFILE_VALUES;
      for (uint32_t i = 0; i < TASK_PROFILE_VALUES; ++i) {
        this->task_profile_output_.SetValue(offset + i, values[i]);
      }
      for (uint32_t i = 0; i < TASK_PROFILE_VALUES; i += DATA_CACHE_LINE_SIZE / sizeof(int64_t)) {
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          this->task_profile_output_[offset + i]);
      }
    }
  }

  __aicore__ inline void FlushTaskProfile() {
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      if (this->task_profile_enabled_) {
        this->task_profile_output_.SetValue(0, this->task_profile_seen_);
        this->task_profile_output_.SetValue(1, this->task_profile_recorded_);
        this->task_profile_output_.SetValue(2, this->task_profile_stride_);
        this->task_profile_output_.SetValue(3, this->task_profile_start_);
        this->task_profile_output_.SetValue(4, ProfileStart());
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          this->task_profile_output_[0]);
      }
    }
  }

  __aicore__ inline int64_t GetWaveGlobalRowBase() const {
    return IsMultiWave()
      ? static_cast<int64_t>(this->wave_index_) * this->receive_capacity_
      : 0;
  }

  __aicore__ inline void SliceWaveSegment(
      int64_t *bounded_offset, int64_t *peer_offset, int32_t *size) const {
    if (!IsMultiWave() || *size <= 0) {
      return;
    }
    const int64_t wave_begin =
      static_cast<int64_t>(this->wave_index_) * this->receive_capacity_;
    const int64_t wave_end = wave_begin + this->receive_capacity_;
    const int64_t segment_begin = *bounded_offset;
    const int64_t segment_end = segment_begin + *size;
    const int64_t intersection_begin =
      segment_begin > wave_begin ? segment_begin : wave_begin;
    const int64_t intersection_end =
      segment_end < wave_end ? segment_end : wave_end;
    if (intersection_begin >= intersection_end) {
      *bounded_offset = 0;
      *peer_offset = 0;
      *size = 0;
      return;
    }
    *peer_offset += intersection_begin - segment_begin;
    *bounded_offset = intersection_begin - wave_begin;
    *size = static_cast<int32_t>(intersection_end - intersection_begin);
  }

  __aicore__ inline void GetWaveExpertRange(
      const GlobalTensor<int64_t> &group_list, uint32_t expert,
      int64_t *begin, int64_t *end) const {
    const int64_t full_begin =
      expert == 0 ? 0 : group_list.GetValue(expert - 1);
    const int64_t full_end = group_list.GetValue(expert);
    if (!IsMultiWave()) {
      *begin = full_begin;
      *end = full_end;
      return;
    }
    const int64_t wave_begin =
      static_cast<int64_t>(this->wave_index_) * this->receive_capacity_;
    const int64_t wave_end = wave_begin + this->receive_capacity_;
    const int64_t intersection_begin =
      full_begin > wave_begin ? full_begin : wave_begin;
    const int64_t intersection_end = full_end < wave_end ? full_end : wave_end;
    if (intersection_begin >= intersection_end) {
      *begin = 0;
      *end = 0;
      return;
    }
    *begin = intersection_begin - wave_begin;
    *end = intersection_end - wave_begin;
  }

  __aicore__ inline void ProcessStatic() {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTask(task_index);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    if (this->worker_id_ % 2 == 0) {
      return;
    }
    uint32_t block_idx = this->worker_id_ / 2;
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTask(task_index);
      uint32_t half_num = this->vector_num / 2;
      block_idx = block_idx + half_num;
    } while (1);
#endif
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    if (getConfigExtraValue(this->runtimeConfigPtr) != READY_HANDSHAKE_MAGIC) {
      return false;
    }
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->magic == READY_HANDSHAKE_MAGIC &&
      (meta->version == READY_HANDSHAKE_VERSION ||
       meta->version == READY_HANDSHAKE_COPY_ACK_VERSION ||
       meta->version == READY_HANDSHAKE_DESTINATION_FENCE_VERSION);
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + MAX_EVENT_NUM;

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t generation = ready_state.GetValue(generation_index) + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= 2, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    TriggerEvent(meta.ready_event);
#endif
  }

  __aicore__ inline int32_t LoadReadyGeneration() {
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready =
      (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + MAX_EVENT_NUM;
    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(
      ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    const uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<
      int32_t, CacheLine::SINGLE_CACHE_LINE,
      DcciDst::CACHELINE_OUT>(ready_state[generation_index]);
    const int32_t generation = ready_state.GetValue(generation_index);
    PipeBarrier<PIPE_ALL>();
    return generation;
  }

  __aicore__ inline void PublishDestinationReady(
      const ReadyHandshakeMeta &meta) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank =
      getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready =
      (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + MAX_EVENT_NUM;
    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(
      ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    const uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<
      int32_t, CacheLine::SINGLE_CACHE_LINE,
      DcciDst::CACHELINE_OUT>(ready_state[generation_index]);
    const int32_t generation = ready_state.GetValue(generation_index) + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<
      int32_t, CacheLine::SINGLE_CACHE_LINE,
      DcciDst::CACHELINE_OUT>(ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *destination_ready =
      ready + static_cast<uint32_t>(rank) * ready_stride;
    for (int64_t target = 0; target < ep; ++target) {
      if (target == rank) {
        continue;
      }
      aclshmemx_signal_op(
        destination_ready, generation, ACLSHMEM_SIGNAL_SET,
        static_cast<int>(target));
    }
    TriggerEvent(meta.ready_event);
#endif
  }

  __aicore__ inline void WaitForDestinationReady(
      const TaskDesc &task_desc) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank =
      getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    int64_t expert_num =
      getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    int64_t tasks_per_expert = task_desc.task_split_num / expert_num;
    int64_t target = task_desc.task_index /
      (local_experts * tasks_per_expert);
    if (target == rank) {
      return;
    }
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready =
      (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + MAX_EVENT_NUM;
    __gm__ int32_t *destination_ready =
      ready + static_cast<uint32_t>(target) * ready_stride;
    aclshmem_signal_wait_until(
      destination_ready, ACLSHMEM_CMP_GE,
      this->destination_ready_generation_);
#endif
  }

  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef __DAV_C220_CUBE__
    WaitForDependency(meta.ready_event);
#else
    if (this->worker_id_ % 2 == 0) {
      return;
    }
    if (this->worker_id_ == 1) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }

  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef __DAV_C220_CUBE__
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
#ifdef __DAV_C220_CUBE__
    this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicNone();
    tBuf.FreeTensor(this->localSet);
#else
    this->localSet = tBuf.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
    this->localSet.SetValue(0, 1);
    for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
      this->localSet.SetValue(i, 0);
    }
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(this->all_event_counters[event_index], this->localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
    tBuf.FreeTensor(this->localSet);
#endif
  }

  __aicore__ inline int64_t GetRouteArrivalOffset() const {
    int64_t ep = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 1);
    return MAX_EVENT_NUM * INT32_T_SIZE +
      (ep + 1) * DATA_CACHE_LINE_SIZE;
  }

  __aicore__ inline int64_t GetRouteCopyAckOffset(
      uint32_t tasks_per_expert) const {
    int64_t expert_num = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 2);
    return GetRouteArrivalOffset() +
      expert_num * tasks_per_expert * ROUTE_ARRIVAL_STRIDE_BYTES;
  }

  __aicore__ inline void WaitForLocalRouteGeneration(
      int64_t arena_offset, uint32_t slot, int32_t generation) {
    __gm__ int32_t *address = reinterpret_cast<__gm__ int32_t *>(
      input_list[Derived::EVENT_IDX] + arena_offset +
      static_cast<int64_t>(slot) * ROUTE_ARRIVAL_STRIDE_BYTES);
    GlobalTensor<int32_t> signal;
    signal.SetGlobalBuffer(address, 1);
    DataCacheCleanAndInvalid<
      int32_t, CacheLine::SINGLE_CACHE_LINE,
      DcciDst::CACHELINE_OUT>(signal[0]);
    int32_t current = signal.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    int64_t refresh_cycle = AscendC::GetSystemCycle();
    while (current < generation) {
      int64_t now = AscendC::GetSystemCycle();
      if ((now - refresh_cycle) / 50 > 150) {
        DataCacheCleanAndInvalid<
          int32_t, CacheLine::SINGLE_CACHE_LINE,
          DcciDst::CACHELINE_OUT>(signal[0]);
        current = signal.GetValue(0);
        refresh_cycle = AscendC::GetSystemCycle();
      }
    }
  }

  __aicore__ inline void WaitForRouteDispatchArrivalTile(
      const TaskDesc &task_desc) {
    int64_t ep = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    uint32_t tasks_per_expert = task_desc.task_split_num / expert_num;
    uint32_t tasks_per_source =
      static_cast<uint32_t>(local_experts) * tasks_per_expert;
    uint32_t source = task_desc.task_index / tasks_per_source;
    uint32_t within_source = task_desc.task_index % tasks_per_source;
    uint32_t local_expert = within_source / tasks_per_expert;
    uint32_t tile = within_source % tasks_per_expert;
    uint32_t slot = (
      (local_expert * static_cast<uint32_t>(ep) + source) *
      tasks_per_expert + tile);
    const int32_t generation = static_cast<int32_t>(
      this->wave_index_ * task_desc.extra_value_2 +
      task_desc.extra_value_1 + 1);
    WaitForLocalRouteGeneration(
      GetRouteArrivalOffset(), slot, generation);
  }

  __aicore__ inline void PublishRouteCopyAck(
      const TaskDesc &task_desc) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    int64_t rank = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 0) % ep;
    uint32_t tasks_per_expert = task_desc.task_split_num / expert_num;
    uint32_t tasks_per_source =
      static_cast<uint32_t>(local_experts) * tasks_per_expert;
    uint32_t source = task_desc.task_index / tasks_per_source;
    uint32_t within_source = task_desc.task_index % tasks_per_source;
    uint32_t local_expert = within_source / tasks_per_expert;
    uint32_t tile = within_source % tasks_per_expert;
    uint32_t global_expert =
      static_cast<uint32_t>(rank * local_experts) + local_expert;
    uint32_t slot = global_expert * tasks_per_expert + tile;
    int64_t ack_offset = GetRouteCopyAckOffset(tasks_per_expert);
    __gm__ int32_t *ack = reinterpret_cast<__gm__ int32_t *>(
      input_list[Derived::EVENT_IDX] + ack_offset +
      static_cast<int64_t>(slot) * ROUTE_ARRIVAL_STRIDE_BYTES);
    const int32_t generation = static_cast<int32_t>(this->wave_index_ + 1);
    PipeBarrier<PIPE_ALL>();
    if (source == static_cast<uint32_t>(rank)) {
      GlobalTensor<int32_t> local_ack;
      local_ack.SetGlobalBuffer(ack, 1);
      local_ack.SetValue(0, generation);
      DataCacheCleanAndInvalid<
        int32_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(local_ack[0]);
      PipeBarrier<PIPE_ALL>();
    } else {
      aclshmemx_signal_op(
        ack, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(source));
    }
#endif
  }

  __aicore__ inline void PublishRouteComputeAck(
      const TaskDesc &task_desc) {
#ifndef __DAV_C220_CUBE__
    const int64_t ep = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 1);
    const int64_t expert_num = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 2);
    const int64_t local_experts = expert_num / ep;
    const int64_t rank = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 0) % ep;
    const uint32_t local_expert =
      task_desc.task_index / static_cast<uint32_t>(ep);
    const uint32_t source =
      task_desc.task_index % static_cast<uint32_t>(ep);
    const uint32_t route_tiles_per_expert = task_desc.extra_value_2;
    const uint32_t global_expert =
      static_cast<uint32_t>(rank * local_experts) + local_expert;
    const int64_t ack_offset = GetRouteCopyAckOffset(
      route_tiles_per_expert);
    const int32_t generation = static_cast<int32_t>(this->wave_index_ + 1);

    const uint32_t slot = global_expert * route_tiles_per_expert;
    __gm__ int32_t *ack = reinterpret_cast<__gm__ int32_t *>(
      input_list[Derived::EVENT_IDX] + ack_offset +
      static_cast<int64_t>(slot) * ROUTE_ARRIVAL_STRIDE_BYTES);
    if (source == static_cast<uint32_t>(rank)) {
      GlobalTensor<int32_t> local_ack;
      local_ack.SetGlobalBuffer(ack, 1);
      local_ack.SetValue(0, generation);
      DataCacheCleanAndInvalid<
        int32_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(local_ack[0]);
      PipeBarrier<PIPE_ALL>();
    } else {
      aclshmemx_signal_op(
        ack, generation, ACLSHMEM_SIGNAL_SET,
        static_cast<int>(source));
    }
#endif
  }

  __aicore__ inline void WaitForRouteCopyAck(
      const TaskDesc &task_desc) {
    int64_t expert_num = getExtraValueFromTiling(
      input_list[Derived::TILING_IDX], 2);
    uint32_t tasks_per_expert = task_desc.task_split_num / expert_num;
    const uint32_t slot =
      (task_desc.extra_value_3 & ROUTE_EXPERT_ACK_FLAG) != 0
        ? (task_desc.task_index / tasks_per_expert) * tasks_per_expert
        : task_desc.task_index;
    WaitForLocalRouteGeneration(
      GetRouteCopyAckOffset(tasks_per_expert), slot,
      static_cast<int32_t>(this->wave_index_ + 1));
  }

  __aicore__ inline void PublishRouteDispatchArrival(
      uint32_t task_index, uint32_t task_split_num, int64_t target_pe,
      uint32_t route_generation_phase = 0,
      uint32_t route_generation_count = 1) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    int64_t tasks_per_expert = task_split_num / expert_num;
    int64_t source_rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0);
    int64_t source_pe = source_rank % ep;
    int64_t expert_index = task_index / tasks_per_expert;
    int64_t local_expert = expert_index % local_experts;
    int64_t tile = task_index % tasks_per_expert;
    int64_t slot = (local_expert * ep + source_pe) * tasks_per_expert + tile;
    int64_t route_arrival_offset = MAX_EVENT_NUM * INT32_T_SIZE + (ep + 1) * DATA_CACHE_LINE_SIZE;
    __gm__ int32_t *arrival = reinterpret_cast<__gm__ int32_t *>(input_list[Derived::EVENT_IDX] + route_arrival_offset +
                                                                 slot * ROUTE_ARRIVAL_STRIDE_BYTES);
    const int32_t generation = static_cast<int32_t>(
      this->wave_index_ * route_generation_count + route_generation_phase + 1);
    if (target_pe == source_pe) {
      GlobalTensor<int32_t> local_arrival;
      local_arrival.SetGlobalBuffer(arrival, 1);
      local_arrival.SetValue(0, generation);
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(local_arrival[0]);
      PipeBarrier<PIPE_ALL>();
    } else {
      aclshmemx_signal_op(arrival, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target_pe));
    }
#endif
  }

  __aicore__ inline void WaitForRouteDispatchArrivals(uint32_t event_index, int32_t packed_needed) {
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    int64_t local_experts = expert_num / ep;
    uint32_t needed = static_cast<uint32_t>(packed_needed);
    uint32_t route_generation_phase = 0;
    uint32_t route_generation_count = 1;
    if (needed > 0xFFFFU) {
      route_generation_phase = (needed >> 16) & 0xFFU;
      route_generation_count = (needed >> 24) & 0xFFU;
      needed &= 0xFFFFU;
    }
    int64_t tasks_per_source = static_cast<int64_t>(needed) / ep;
    int64_t local_expert = (static_cast<int64_t>(event_index) - 1) % local_experts;
    int64_t route_arrival_offset = MAX_EVENT_NUM * INT32_T_SIZE + (ep + 1) * DATA_CACHE_LINE_SIZE;
    constexpr uint32_t stride = ROUTE_ARRIVAL_STRIDE_BYTES / INT32_T_SIZE;
    GlobalTensor<int32_t> arrivals;
    arrivals.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(input_list[Derived::EVENT_IDX] + route_arrival_offset),
                             static_cast<uint32_t>(expert_num * tasks_per_source * stride));

    for (int64_t source = 0; source < ep; ++source) {
      for (int64_t tile = 0; tile < tasks_per_source; ++tile) {
        uint32_t index = static_cast<uint32_t>(((local_expert * ep + source) * tasks_per_source + tile) * stride);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(arrivals[index]);
        int32_t current = arrivals.GetValue(index);
        PipeBarrier<PIPE_ALL>();
        int64_t system_cycle_before = AscendC::GetSystemCycle();
        const int32_t generation = static_cast<int32_t>(
          this->wave_index_ * route_generation_count +
          route_generation_phase + 1);
        while (current < generation) {
          int64_t system_cycle_after = AscendC::GetSystemCycle();
          if ((system_cycle_after - system_cycle_before) / 50 > 150) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(arrivals[index]);
            current = arrivals.GetValue(index);
            system_cycle_before = AscendC::GetSystemCycle();
          }
        }
      }
    }
  }

  __aicore__ inline void ExecuteTask(TaskId task_id) {
    const int64_t task_start = ProfileStart();
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      ++this->task_profile_seen_;
    }
    TaskDesc task_desc;
    int64_t profile_start = ProfileStart();
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    ProfileRecord(CYCLE_PROFILE_DESCRIPTOR_DECODE, profile_start);
    const bool first_wave_topk =
      task_desc.task_type == TASK_TOPK_WEIGHT_GRAD &&
      task_desc.extra_value_1 == 1;
    if (IsMultiWave()) {
      if (task_desc.task_type == TASK_TERMINATE) {
        return;
      }
      const bool terminal_task =
        (task_desc.task_type == TASK_WEIGHTED_UNPERMUTE &&
         task_desc.extra_value_1 == 1) ||
        task_desc.task_type == TASK_UNPERMUTE_REDUCE_GRAD ||
        (task_desc.task_type == TASK_TOPK_WEIGHT_GRAD &&
         !first_wave_topk);
      if (terminal_task && !IsLastWave()) {
        return;
      }
      if (first_wave_topk && this->wave_index_ != 0) {
        return;
      }
    }
    if (task_desc.task_type == TASK_PHASE_BARRIER) {
      profile_start = ProfileStart();
      SynchronizePhase();
      ProfileRecord(CYCLE_PROFILE_SYNCHRONIZATION, profile_start);
      const int64_t end = ProfileStart();
      RecordTaskProfile(task_id, task_desc, task_start, end, end, end);
      return;
    }
    const bool route_tile_copy =
      IsMultiWave() && task_desc.task_type == TASK_LOCAL_COPY &&
      task_desc.extra_value_0 == ROUTE_COPY_ACK_FLAG;
    const bool route_compute_ack =
      IsMultiWave() && task_desc.task_type == TASK_LOCAL_COPY &&
      task_desc.extra_value_0 == ROUTE_COMPUTE_ACK_FLAG;
    const bool route_tile_wait = route_tile_copy ||
      (IsMultiWave() && task_desc.task_type == TASK_LOCAL_COPY &&
       task_desc.extra_value_0 == ROUTE_ARRIVAL_WAIT_FLAG);
    if (route_tile_wait) {
      profile_start = ProfileStart();
      WaitForRouteDispatchArrivalTile(task_desc);
      ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
    }
    if (route_compute_ack) {
      profile_start = ProfileStart();
      WaitForDependency(task_desc.extra_value_1);
      ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
    }
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      uint32_t one_time_predecessors = 0;
      if (IsMultiWave() &&
          ((task_desc.task_type == TASK_WEIGHTED_UNPERMUTE &&
            task_desc.extra_value_1 == 1) ||
           task_desc.task_type == TASK_UNPERMUTE_REDUCE_GRAD ||
           task_desc.task_type == TASK_TOPK_WEIGHT_GRAD)) {
        one_time_predecessors = task_desc.extra_value_3;
      }
      profile_start = ProfileStart();
      WaitForDependency(task_desc.dependent_event, one_time_predecessors);
      ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
    }
    if (IsMultiWave() && this->wave_index_ > 0 &&
        this->has_ready_handshake_ &&
        this->ready_handshake_meta_.version ==
          READY_HANDSHAKE_DESTINATION_FENCE_VERSION &&
        task_desc.task_type == TASK_SHMEM_PUT_MEM_SIGNAL &&
        (task_desc.extra_value_3 & 0xFFU) == 1U) {
      profile_start = ProfileStart();
      WaitForDestinationReady(task_desc);
      ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
    }
    // Multi-wave dTopK reuses the route-index slot and must finish before the
    // first Combine overwrites the retained expert output. Combine keeps its
    // ordinary per-expert gate dependency and waits on this second, one-time
    // barrier only in wave zero.
    if ((!IsMultiWave() || this->wave_index_ == 0) &&
        task_desc.task_type == TASK_SHMEM_PUT_MEM_SIGNAL &&
        (task_desc.extra_value_3 & 0xFFU) == 2U &&
        task_desc.extra_value_2 != EVENT_INVALID_ID) {
      profile_start = ProfileStart();
      WaitForDependency(task_desc.extra_value_2);
      ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
    }
    if constexpr (Derived::HAS_MID_WAVE_ROUTE_BARRIER) {
      if (task_desc.task_type == TASK_SHMEM_PUT_MEM_SIGNAL &&
          (task_desc.extra_value_3 & 0xFFU) == 2U) {
        profile_start = ProfileStart();
        SynchronizeCombineReuse(task_desc);
        ProfileRecord(CYCLE_PROFILE_SYNCHRONIZATION, profile_start);
      }
    }
    const uint32_t route_generation_phase =
      (task_desc.extra_value_3 >> 8) & 0xFFU;
    const uint32_t route_generation_count =
      (task_desc.extra_value_3 >> 16) & 0xFFU;
    const bool wait_for_copy_ack =
      (task_desc.extra_value_3 & ROUTE_COPY_ACK_FLAG) != 0;
    if (IsMultiWave() && Derived::HAS_MID_WAVE_ROUTE_BARRIER &&
        task_desc.task_type == TASK_SHMEM_PUT_MEM_SIGNAL &&
        route_generation_count > 1 && route_generation_phase == 1) {
      profile_start = ProfileStart();
      if (wait_for_copy_ack) {
        WaitForRouteCopyAck(task_desc);
        ProfileRecord(CYCLE_PROFILE_DEPENDENCY_WAIT, profile_start);
      } else {
        SynchronizeMidWaveRoute(task_desc.task_index == 0);
        ProfileRecord(CYCLE_PROFILE_SYNCHRONIZATION, profile_start);
      }
    }
    const int64_t task_ready = ProfileStart();
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    const int64_t task_compute_end = ProfileStart();
    if (route_tile_copy) {
      profile_start = ProfileStart();
      PublishRouteCopyAck(task_desc);
      ProfileRecord(CYCLE_PROFILE_LOCAL_COMPUTE, profile_start);
    }
    if (route_compute_ack) {
      profile_start = ProfileStart();
      PublishRouteComputeAck(task_desc);
      ProfileRecord(CYCLE_PROFILE_LOCAL_COMPUTE, profile_start);
    }
    if (task_desc.task_type != TASK_SHMEM_PUT_MEM_SIGNAL && task_desc.trigger_event != EVENT_INVALID_ID) {
      TriggerEvent(task_desc.trigger_event);
    }
    RecordTaskProfile(task_id, task_desc, task_start, task_ready, task_compute_end, ProfileStart());
  }

  __aicore__ inline void WaitForDependency(
      uint32_t event_index, uint32_t one_time_predecessors = 0) {
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    if (needed < 0) {
      WaitForRouteDispatchArrivals(event_index, -needed);
      return;
    }
    if (IsMultiWave()) {
      needed *= static_cast<int32_t>(this->wave_index_ + 1);
      needed -= static_cast<int32_t>(
        one_time_predecessors * this->wave_index_);
    }
    WaitForEventCount(event_index, needed);
  }

  __aicore__ inline void WaitForEventCount(uint32_t event_index, int32_t needed) {
#ifdef __DAV_C220_CUBE__
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleBefore = AscendC::GetSystemCycle();
    do {
      if (current >= needed) {
        break;
      }
      int64_t systemCycleAfter = AscendC::GetSystemCycle();
      int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
      int64_t CycleToTimeBase = 50;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > 50) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        systemCycleBefore = AscendC::GetSystemCycle();
      }
    } while (1);
#else
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t systemCycleBefore = AscendC::GetSystemCycle();
    do {
      if (current >= needed) {
        break;
      }
      int64_t systemCycleAfter = AscendC::GetSystemCycle();
      int64_t GetBlockNumCycle = systemCycleAfter - systemCycleBefore;
      int64_t CycleToTimeBase = 50;
      int64_t GetBlockNumTime = GetBlockNumCycle / CycleToTimeBase;
      if (GetBlockNumTime > 150) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        systemCycleBefore = AscendC::GetSystemCycle();
      }
    } while (1);
#endif
  }

  __aicore__ inline void SynchronizeWave(
      const ReadyHandshakeMeta &meta, bool has_ready_handshake) {
    TriggerEvent(WAVE_BARRIER_EVENT);
    const int32_t active_workers = static_cast<int32_t>(this->core_num * 2);
    const int32_t local_generation =
      active_workers * static_cast<int32_t>(this->wave_index_ + 1);
    WaitForEventCount(WAVE_BARRIER_EVENT, local_generation);
    if (!has_ready_handshake) {
      return;
    }
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ == 1) {
      if (meta.version == READY_HANDSHAKE_DESTINATION_FENCE_VERSION) {
        PublishDestinationReady(meta);
      } else {
        PublishDeviceReady(meta);
      }
    }
#endif
    const int32_t ready_events_per_wave =
      meta.version == READY_HANDSHAKE_VERSION
        ? (Derived::HAS_MID_WAVE_ROUTE_BARRIER ? 2 : 1)
        : 1;
    WaitForEventCount(
      meta.ready_event,
      1 + ready_events_per_wave * static_cast<int32_t>(this->wave_index_ + 1));
    if (meta.version == READY_HANDSHAKE_DESTINATION_FENCE_VERSION) {
      this->destination_ready_generation_ = LoadReadyGeneration();
    }
  }

  __aicore__ inline void SynchronizePhase() {
    PipeBarrier<PIPE_ALL>();
    TriggerEvent(PHASE_BARRIER_EVENT);
    const int32_t active_workers = static_cast<int32_t>(this->core_num * 2);
    const int32_t generation =
      active_workers * static_cast<int32_t>(this->wave_index_ + 1);
    WaitForEventCount(PHASE_BARRIER_EVENT, generation);
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void SynchronizeMidWaveRoute(bool publish) {
    if (this->mid_wave_route_synchronized_) {
      return;
    }
    if (this->has_ready_handshake_) {
#ifndef __DAV_C220_CUBE__
      if (publish) {
        PublishDeviceReady(this->ready_handshake_meta_);
      }
#endif
      WaitForEventCount(
        this->ready_handshake_meta_.ready_event,
        2 + 2 * static_cast<int32_t>(this->wave_index_));
    }
    this->mid_wave_route_synchronized_ = true;
  }

  __aicore__ inline void SynchronizeCombineReuse(const TaskDesc &task_desc) {
#ifndef __DAV_C220_CUBE__
    if (this->combine_reuse_synchronized_ || !this->has_ready_handshake_ ||
        (IsMultiWave() && this->wave_index_ != 0)) {
      return;
    }
    if (task_desc.outputs[0].input_position != 0 &&
        task_desc.extra_value_2 == EVENT_INVALID_ID) {
      return;
    }
    // A local gate dependency cannot release a peer's dY buffer. In the
    // one-wave Cube queue, every act-grad/dW2 precedes every gate-grad, so
    // reaching Combine proves local dY consumption is complete. Early dTopK
    // has its own local dependency above and also needs peer completion
    // before Combine can overwrite the retained forward expert output.
    // Rendezvous after those local readers, before any remote overwrite.
    ReadyHandshakeMeta reuse_meta = this->ready_handshake_meta_;
    // Keep the release counter separate from wave/hidden-dispatch readiness;
    // the persistent dissemination generation still advances on every rank.
    reuse_meta.ready_event += 1;
    // The first Combine tile always exists, while a fixed worker may receive
    // no Combine tile for small schedules.
    if (task_desc.task_index == 0) {
      PublishDeviceReady(reuse_meta);
    }
    WaitForEventCount(reuse_meta.ready_event, 1);
    this->combine_reuse_synchronized_ = true;
#endif
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  __aicore__ inline void FlushCycleProfile() {
    FlushTaskProfile();
    if constexpr (Derived::CYCLE_PROFILE_ENABLED) {
      uint32_t worker_slot = this->worker_id_;
#ifndef __DAV_C220_CUBE__
      worker_slot = NUM_WORKERS_CUBE + this->worker_id_ / 2;
#endif
      if (worker_slot >= CYCLE_PROFILE_WORKER_SLOTS) {
        return;
      }
      GlobalTensor<int64_t> output;
      output.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t *>(
          input_list[Derived::CYCLE_PROFILE_WORKSPACE_IDX] +
          Derived::CYCLE_PROFILE_OFFSET_BYTES),
        CYCLE_PROFILE_WORKER_SLOTS * CYCLE_PROFILE_CATEGORY_COUNT);
      const uint32_t offset = worker_slot * CYCLE_PROFILE_CATEGORY_COUNT;
      for (uint32_t category = 0; category < CYCLE_PROFILE_CATEGORY_COUNT; ++category) {
        output.SetValue(offset + category, this->cycle_profile_[category]);
      }
      DataCacheCleanAndInvalid<
        int64_t, CacheLine::SINGLE_CACHE_LINE,
        DcciDst::CACHELINE_OUT>(output[offset]);
      PipeBarrier<PIPE_ALL>();
    }
  }

  bool task_profile_enabled_ = false;
  uint32_t task_profile_seen_ = 0;
  uint32_t task_profile_recorded_ = 0;
  uint32_t task_profile_stride_ = 1;
  int64_t task_profile_start_ = 0;
  GlobalTensor<int64_t> task_profile_output_;
  uint32_t worker_id_ = 0;
  uint32_t task_num = 0;
  int32_t vector_task_num = 0;
  int32_t cube_task_num = 0;

  TPipe tpipe_;
#ifdef __DAV_C220_CUBE__
  TBuf<AscendC::TPosition::A1> tBuf;
#else
  TBuf<AscendC::TPosition::VECOUT> tBuf;
#endif
  LocalTensor<int32_t> localSet;

  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> all_event_num_triggers;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> vector_task_indexs;
  GlobalTensor<int32_t> cube_task_indexs;

  __gm__ uint8_t *runtimeConfigPtr;
  GM_ADDR *input_list = nullptr;
  int64_t core_num = 0;
  int64_t vector_num = 0;
  int64_t receive_capacity_ = 0;
  int64_t home_experts_ = 0;
  GM_ADDR replica_matrix_bases_[4] = {nullptr, nullptr, nullptr, nullptr};
  uint32_t wave_count_ = 1;
  uint32_t wave_index_ = 0;
  bool has_ready_handshake_ = false;
  bool mid_wave_route_synchronized_ = false;
  bool combine_reuse_synchronized_ = false;
  int32_t destination_ready_generation_ = 0;
  ReadyHandshakeMeta ready_handshake_meta_;
  int64_t cycle_profile_[CYCLE_PROFILE_CATEGORY_COUNT];
};

#endif  // MULTICORE_SCHEDULER_WORKER_KERNEL_H
