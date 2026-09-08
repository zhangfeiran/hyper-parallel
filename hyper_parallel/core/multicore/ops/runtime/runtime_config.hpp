/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP
#define MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP

constexpr uint32_t MAX_TENSOR_DIMS = 4;
constexpr uint32_t MAX_INPUTS_PER_TASK = 4;
constexpr uint32_t MAX_OUTPUTS_PER_TASK = 4;

constexpr uint32_t MAX_TASK_NUM = 256 * 100;
constexpr uint32_t MAX_EVENT_NUM = 1024;
constexpr uint32_t PHASE_BARRIER_EVENT = MAX_EVENT_NUM - 2;
constexpr uint32_t WAVE_BARRIER_EVENT = MAX_EVENT_NUM - 1;
constexpr uint32_t NUM_WORKERS_VECTOR = 48;
constexpr uint32_t NUM_WORKERS_CUBE = 24;
constexpr uint32_t CYCLE_PROFILE_CATEGORY_COUNT = 8;
constexpr uint32_t CYCLE_PROFILE_WORKER_SLOTS = 2 * NUM_WORKERS_CUBE;
constexpr uint32_t QUEUE_CAPACITY = 100;
constexpr uint32_t TASK_TYPE_INDEX_NUM = 256 * 100;
constexpr uint32_t MAX_GROUP_LIST = 512;
constexpr uint32_t ATOMIC_ADD_VALUE_LEN = 8;
constexpr uint32_t READY_HANDSHAKE_MAGIC = 0x4D4D5231;
constexpr uint32_t READY_HANDSHAKE_VERSION = 1;
constexpr uint32_t READY_HANDSHAKE_COPY_ACK_VERSION = 2;
constexpr uint32_t READY_HANDSHAKE_DESTINATION_FENCE_VERSION = 3;
constexpr uint32_t RUNTIME_TASK_CAPACITY_MAGIC = 0x80000000;
constexpr uint32_t RUNTIME_TASK_CAPACITY_MASK = RUNTIME_TASK_CAPACITY_MAGIC - 1;
constexpr uint32_t RUNTIME_EXTENDED_MAGIC = 0x40000000;
constexpr uint32_t RUNTIME_EXTENDED_CAPACITY_MASK = RUNTIME_EXTENDED_MAGIC - 1;
constexpr uint32_t RUNTIME_FORMAT_VERSION = 2;
constexpr uint32_t COMPACT_TASK_DESC_SIZE = 128;

constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t DATA_CACHE_LINE_SIZE = 64;
constexpr uint32_t ROUTE_ARRIVAL_STRIDE_BYTES = DATA_CACHE_LINE_SIZE;
constexpr uint32_t ROUTE_COPY_ACK_FLAG = 1U << 24;
constexpr uint32_t ROUTE_ARRIVAL_WAIT_FLAG = 1U << 25;
constexpr uint32_t ROUTE_COMPUTE_ACK_FLAG = 1U << 26;
constexpr uint32_t ROUTE_EXPERT_ACK_FLAG = 1U << 27;
constexpr uint32_t RETAINED_DW_SCRATCH_FLAG = 1U << 28;
constexpr uint32_t ATOMIC_DW_ACCUMULATION_FLAG = 1U << 29;
constexpr uint32_t EXP_TOKEN_COUNT_FLAG_CNT = UB_32B_ALIGN / sizeof(int32_t);  // 8
constexpr uint32_t DISPATCH_TOKEN_UB_SIZE = 3 * 32;

constexpr uint32_t UINT32_T_SIZE = sizeof(uint32_t);
constexpr uint32_t INT32_T_SIZE = sizeof(int32_t);
constexpr uint32_t INT64_T_SIZE = sizeof(int64_t);

constexpr uint32_t EVENT_INVALID_ID = 0xFFFFFFFF;

typedef uint32_t TaskId;

enum TaskAiCoreType : uint32_t {
  TASK_AICORE_INVALID = 0,
  TASK_AICORE_CUBE = 1,
  TASK_AICORE_VECTOR = 2,
  TASK_AICORE_MIX = 3,
};

enum TaskType : uint32_t {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_ADD_CUSTOM = 101,
  TASK_SWI_GLU = 102,
  TASK_MATMUL = 103,
  TASK_GROUPED_MATMUL = 104,
  TASK_SHMEM_PUT_MEM_SIGNAL = 105,
  TASK_SWI_GLU_GRAD = 106,
  TASK_UNPERMUTE_REDUCE_GRAD = 110,
  TASK_TOPK_WEIGHT_GRAD = 111,
  TASK_WEIGHTED_UNPERMUTE = 112,
  TASK_LOCAL_COPY = 113,
  TASK_LOCAL_ACCUMULATE = 114,
  TASK_PHASE_BARRIER = 115,
};

enum EventType : uint32_t {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,  // TASK_TERMINATE
  EVENT_INVALID = 999,
};

enum CycleProfileCategory : uint32_t {
  CYCLE_PROFILE_DESCRIPTOR_DECODE = 0,
  CYCLE_PROFILE_DEPENDENCY_WAIT = 1,
  CYCLE_PROFILE_GMM = 2,
  CYCLE_PROFILE_DISPATCH_DY = 3,
  CYCLE_PROFILE_DISPATCH_HIDDEN = 4,
  CYCLE_PROFILE_COMBINE = 5,
  CYCLE_PROFILE_LOCAL_COMPUTE = 6,
  CYCLE_PROFILE_SYNCHRONIZATION = 7,
};

struct TensorDesc {
  uint32_t tensor_type;
  uint32_t num_dims;
  uint32_t dim[MAX_TENSOR_DIMS];
  uint32_t stride[MAX_TENSOR_DIMS];
  uint32_t data_type;
  uint32_t input_position;
  uint32_t base_ptr_offset;
  uint32_t transpose_flag;
  uint32_t dynamic_shape;
  uint32_t dynamic_dim;
};

struct EventDesc {
  EventType event_type;
  uint32_t num_triggers;
  uint32_t first_task_id, last_task_id;
};

enum struct DynamicType : uint32_t {
  DYNAMIC_EMPTY = 0,
  DYNAMIC_DSV3_MOE = 101,
};

struct DynamicData {
  DynamicType dynamic_type;
  uint32_t dynamic_input_position;
  uint32_t dynamic_group_size;
  uint32_t dynamic_max_seq_len;
};

struct TaskDesc {
  TaskType task_type;
  TaskAiCoreType task_aicore_type;
  uint32_t num_inputs, num_outputs;
  uint32_t trigger_event;
  uint32_t dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  uint32_t tiling_data_position;
  uint32_t tiling_data_offset;
  uint32_t task_index;
  uint32_t task_split_num;
  uint32_t task_split_value;
  uint32_t extra_value_0;
  uint32_t extra_value_1;
  uint32_t extra_value_2;
  uint32_t extra_value_3;
  uint32_t extra_value_4;
};

struct RuntimeConfig {
  uint32_t task_num;
  uint32_t num_workers;
  uint32_t queue_capacity;
  uint32_t config_extra_value;

  int32_t *all_event_num_triggers;

  TaskDesc *all_tasks;
  EventDesc *all_events;

  int32_t *task_index_num;      // 3
  int32_t *cube_task_indexs;    // TASK_TYPE_INDEX_NUM
  int32_t *vector_task_indexs;  // TASK_TYPE_INDEX_NUM
  int32_t *mix_task_indexs;     // TASK_TYPE_INDEX_NUM

  DynamicData dynamic_data;
  int64_t *grouped_matmul_group_list;
  int32_t *atomic_add_values;
};

struct ReadyHandshakeMeta {
  uint32_t magic;
  uint32_t version;
  uint32_t ready_event;
};

__aicore__ inline uint32_t getTaskNum(__gm__ uint8_t *tiling) { return (*(__gm__ uint32_t *)(tiling)); }

__aicore__ inline uint32_t getConfigExtraValue(__gm__ uint8_t *tiling) {
  return (*(__gm__ uint32_t *)(tiling + UINT32_T_SIZE * 3));
}

__aicore__ inline bool hasExtendedRuntime(__gm__ uint8_t *tiling) {
  uint32_t encoded = (*(__gm__ uint32_t *)(tiling + UINT32_T_SIZE * 2));
  return (encoded & (RUNTIME_TASK_CAPACITY_MAGIC | RUNTIME_EXTENDED_MAGIC)) ==
    (RUNTIME_TASK_CAPACITY_MAGIC | RUNTIME_EXTENDED_MAGIC);
}

__aicore__ inline uint32_t getRuntimeTaskCapacity(__gm__ uint8_t *tiling) {
  uint32_t encoded = (*(__gm__ uint32_t *)(tiling + UINT32_T_SIZE * 2));
  if (hasExtendedRuntime(tiling)) {
    return encoded & RUNTIME_EXTENDED_CAPACITY_MASK;
  }
  return (encoded & RUNTIME_TASK_CAPACITY_MAGIC) != 0 ? encoded & RUNTIME_TASK_CAPACITY_MASK : MAX_TASK_NUM;
}

__aicore__ inline uint32_t getRuntimeEventCapacity(__gm__ uint8_t *tiling) {
  return hasExtendedRuntime(tiling) ? (*(__gm__ uint32_t *)(tiling + 16)) : MAX_EVENT_NUM;
}

__aicore__ inline uint32_t getRuntimeTensorCount(__gm__ uint8_t *tiling) {
  return hasExtendedRuntime(tiling) ? (*(__gm__ uint32_t *)(tiling + 20)) : 0;
}

__aicore__ inline uint32_t getRuntimeTaskStride(__gm__ uint8_t *tiling) {
  return hasExtendedRuntime(tiling) ? (*(__gm__ uint32_t *)(tiling + 24)) : sizeof(TaskDesc);
}

__aicore__ inline void validateRuntimeFormat(__gm__ uint8_t *tiling) {
  if (!hasExtendedRuntime(tiling)) {
    return;
  }
  uint32_t stride = getRuntimeTaskStride(tiling);
  if ((*(__gm__ uint32_t *)(tiling + 28)) != RUNTIME_FORMAT_VERSION ||
      (stride != sizeof(TaskDesc) && stride != COMPACT_TASK_DESC_SIZE) ||
      getRuntimeEventCapacity(tiling) < MAX_EVENT_NUM || getRuntimeEventCapacity(tiling) % 16 != 0) {
    AscendC::Trap();
  }
}

__aicore__ inline uint32_t getAllEventNumTriggersOffset(__gm__ uint8_t *tiling) {
  return hasExtendedRuntime(tiling) ? DATA_CACHE_LINE_SIZE : UINT32_T_SIZE * 4;
}

__aicore__ inline uint32_t getAllTasksOffset(__gm__ uint8_t *tiling) {
  return getAllEventNumTriggersOffset(tiling) + INT32_T_SIZE * getRuntimeEventCapacity(tiling);
}

__aicore__ inline uint32_t getAllEventsOffset(__gm__ uint8_t *tiling) {
  return getAllTasksOffset(tiling) + getRuntimeTaskStride(tiling) * getRuntimeTaskCapacity(tiling);
}

__aicore__ inline uint32_t getRuntimeTensorTableOffset(__gm__ uint8_t *tiling);

__aicore__ inline void getTensorTemplate(__gm__ uint32_t *source, TensorDesc *tensor) {
  tensor->tensor_type = source[0];
  tensor->num_dims = source[1];
  for (uint32_t dimension = 0; dimension < MAX_TENSOR_DIMS; ++dimension) {
    tensor->dim[dimension] = source[2 + dimension];
    tensor->stride[dimension] = source[6 + dimension];
  }
  tensor->data_type = source[10];
  tensor->input_position = source[11];
  tensor->base_ptr_offset = source[12];
  tensor->transpose_flag = source[13];
  tensor->dynamic_shape = source[14];
  tensor->dynamic_dim = source[15];
}

__aicore__ inline void getCompactTaskDesc(__gm__ uint8_t *tiling, TaskDesc *task, uint32_t index) {
  __gm__ uint32_t *record = reinterpret_cast<__gm__ uint32_t *>(
    tiling + getAllTasksOffset(tiling) + index * COMPACT_TASK_DESC_SIZE);
  task->task_type = static_cast<TaskType>(record[0]);
  task->task_aicore_type = static_cast<TaskAiCoreType>(record[1]);
  task->num_inputs = record[2];
  task->num_outputs = record[3];
  task->trigger_event = record[4];
  task->dependent_event = record[5];
  __gm__ uint8_t *table = tiling + getRuntimeTensorTableOffset(tiling);
  for (uint32_t tensor_index = 0; tensor_index < MAX_INPUTS_PER_TASK + MAX_OUTPUTS_PER_TASK; ++tensor_index) {
    TensorDesc *tensor = tensor_index < MAX_INPUTS_PER_TASK ?
      &task->inputs[tensor_index] : &task->outputs[tensor_index - MAX_INPUTS_PER_TASK];
    const uint32_t template_id = record[6 + tensor_index * 2];
    __gm__ uint32_t *source = reinterpret_cast<__gm__ uint32_t *>(table + template_id * sizeof(TensorDesc));
    getTensorTemplate(source, tensor);
    tensor->base_ptr_offset = record[7 + tensor_index * 2];
  }
  task->tiling_data_position = record[22];
  task->tiling_data_offset = record[23];
  task->task_index = record[24];
  task->task_split_num = record[25];
  task->task_split_value = record[26];
  task->extra_value_0 = record[27];
  task->extra_value_1 = record[28];
  task->extra_value_2 = record[29];
  task->extra_value_3 = record[30];
  task->extra_value_4 = record[31];
}

__aicore__ inline void getTaskDesc(__gm__ uint8_t *tiling, TaskDesc *tilingData, uint32_t index_size) {
  uint32_t tensor_desc_size = UINT32_T_SIZE * 8 + UINT32_T_SIZE * MAX_TENSOR_DIMS * 2;
  uint32_t task_desc_size = UINT32_T_SIZE * 6 + MAX_INPUTS_PER_TASK * tensor_desc_size +
                            MAX_OUTPUTS_PER_TASK * tensor_desc_size + UINT32_T_SIZE * 10;
  if (getRuntimeTaskStride(tiling) == COMPACT_TASK_DESC_SIZE) {
    getCompactTaskDesc(tiling, tilingData, index_size);
    return;
  }
  uint32_t size = getAllTasksOffset(tiling) + index_size * task_desc_size;

  tilingData->task_type = (*(__gm__ TaskType *)(tiling + size));
  tilingData->task_aicore_type = (*(__gm__ TaskAiCoreType *)(tiling + size + UINT32_T_SIZE));
  tilingData->num_inputs = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->num_outputs = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
  tilingData->trigger_event = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 4));
  tilingData->dependent_event = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 5));
  uint32_t start_size = size + UINT32_T_SIZE * 6;
  for (uint32_t i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    (tilingData->inputs)[i].tensor_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].num_dims = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->inputs)[i].dim[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->inputs)[i].stride[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    (tilingData->inputs)[i].data_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].input_position = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].base_ptr_offset = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].transpose_flag = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].dynamic_shape = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->inputs)[i].dynamic_dim = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
  }
  for (uint32_t i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    (tilingData->outputs)[i].tensor_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].num_dims = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->outputs)[i].dim[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    for (uint32_t j = 0; j < MAX_TENSOR_DIMS; j++) {
      (tilingData->outputs)[i].stride[j] = (*(__gm__ uint32_t *)(tiling + start_size));
      start_size = start_size + UINT32_T_SIZE;
    }
    (tilingData->outputs)[i].data_type = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].input_position = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].base_ptr_offset = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].transpose_flag = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].dynamic_shape = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
    (tilingData->outputs)[i].dynamic_dim = (*(__gm__ uint32_t *)(tiling + start_size));
    start_size = start_size + UINT32_T_SIZE;
  }
  tilingData->tiling_data_position = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->tiling_data_offset = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_index = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_split_num = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->task_split_value = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->extra_value_0 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->extra_value_1 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->extra_value_2 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->extra_value_3 = (*(__gm__ uint32_t *)(tiling + start_size));
  start_size = start_size + UINT32_T_SIZE;
  tilingData->extra_value_4 = (*(__gm__ uint32_t *)(tiling + start_size));
}

__aicore__ inline void getEventDesc(__gm__ uint8_t *tiling, EventDesc *tilingData, uint32_t index_size) {
  uint32_t size = getAllEventsOffset(tiling) + index_size * 4 * UINT32_T_SIZE;

  tilingData->event_type = (*(__gm__ EventType *)(tiling + size));
  tilingData->num_triggers = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->first_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->last_task_id = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getTaskIndexNumOffset(__gm__ uint8_t *tiling) {
  return getAllEventsOffset(tiling) + getRuntimeEventCapacity(tiling) * 4 * UINT32_T_SIZE;
}

__aicore__ inline int32_t getTaskIndexNumByTaskType(__gm__ uint8_t *tiling, TaskAiCoreType task_aicore_type) {
  uint32_t size = getTaskIndexNumOffset(tiling);
  if (task_aicore_type == TaskAiCoreType::TASK_AICORE_VECTOR) {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE));
  } else if (task_aicore_type == TaskAiCoreType::TASK_AICORE_CUBE) {
    return (*(__gm__ int32_t *)(tiling + size));
  } else {
    return (*(__gm__ int32_t *)(tiling + size + INT32_T_SIZE * 2));
  }
}

__aicore__ inline uint32_t getCubeTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getTaskIndexNumOffset(tiling) + 4 * INT32_T_SIZE;
}

__aicore__ inline uint32_t getVectorTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getCubeTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline uint32_t getMixTaskIndexsOffset(__gm__ uint8_t *tiling) {
  return getVectorTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline uint32_t getDynamicDataOffset(__gm__ uint8_t *tiling) {
  return getMixTaskIndexsOffset(tiling) + getRuntimeTaskCapacity(tiling) * INT32_T_SIZE;
}

__aicore__ inline void getDynamicData(__gm__ uint8_t *tiling, DynamicData *tilingData) {
  uint32_t size = getDynamicDataOffset(tiling);
  tilingData->dynamic_type = (*(__gm__ DynamicType *)(tiling + size));
  tilingData->dynamic_input_position = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE));
  tilingData->dynamic_group_size = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 2));
  tilingData->dynamic_max_seq_len = (*(__gm__ uint32_t *)(tiling + size + UINT32_T_SIZE * 3));
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffset(__gm__ uint8_t *tiling) {
  return getDynamicDataOffset(tiling) + 4 * UINT32_T_SIZE;
}

__aicore__ inline uint32_t getGroupedMatmulGroupListOffsetById(__gm__ uint8_t *tiling, uint32_t block_idx) {
  return getGroupedMatmulGroupListOffset(tiling) + 8 * INT64_T_SIZE * block_idx;
}

__aicore__ inline uint32_t getAtomicAddValuesOffset(__gm__ uint8_t *tiling) {
  return getGroupedMatmulGroupListOffset(tiling) + MAX_GROUP_LIST * INT64_T_SIZE;
}

__aicore__ inline uint32_t getReadyHandshakeMetaOffset(__gm__ uint8_t *tiling) {
  return getAtomicAddValuesOffset(tiling) + ATOMIC_ADD_VALUE_LEN * INT32_T_SIZE;
}

__aicore__ inline void getReadyHandshakeMeta(__gm__ uint8_t *tiling, ReadyHandshakeMeta *meta) {
  __gm__ uint8_t *base = tiling + getReadyHandshakeMetaOffset(tiling);
  meta->magic = (*(__gm__ uint32_t *)(base));
  meta->version = (*(__gm__ uint32_t *)(base + UINT32_T_SIZE));
  meta->ready_event = (*(__gm__ uint32_t *)(base + UINT32_T_SIZE * 2));
}

// Tensor templates start on a cache line. Legacy descriptors retain their
// original 8-byte-aligned tail so persisted offline blobs remain readable.
__aicore__ inline uint32_t getRuntimeTensorTableOffset(__gm__ uint8_t *tiling) {
  uint32_t alignment = hasExtendedRuntime(tiling) ? DATA_CACHE_LINE_SIZE : INT64_T_SIZE;
  return (getReadyHandshakeMetaOffset(tiling) + 3 * UINT32_T_SIZE + alignment - 1) & ~(alignment - 1);
}

// Optional split-weight ABI v1, after the compact RuntimeConfig's 8-byte
// aligned end. Home tensors remain operator inputs; only group-owned B-slot
// weights and gradients are addressed here. Never read this for dense weights.
__aicore__ inline void getReplicaMatrixBases(__gm__ uint8_t *tiling, GM_ADDR *bases) {
  const uint32_t offset = getRuntimeTensorTableOffset(tiling) + getRuntimeTensorCount(tiling) * sizeof(TensorDesc);
  __gm__ uint8_t *base = tiling + offset;
  const uint32_t magic = *reinterpret_cast<__gm__ uint32_t *>(base);
  const uint32_t version = *reinterpret_cast<__gm__ uint32_t *>(base + UINT32_T_SIZE);
  if (magic != 0x53505754U || version != 1U) {
    AscendC::Trap();
  }
  for (uint32_t index = 0; index < 4; ++index) {
    bases[index] = reinterpret_cast<GM_ADDR>(
      *reinterpret_cast<__gm__ uint64_t *>(base + 2 * UINT32_T_SIZE + index * INT64_T_SIZE));
  }
}

__aicore__ inline int64_t getExtraValueFromTiling(__gm__ uint8_t *tiling, uint32_t index) {
  return (*(__gm__ int64_t *)(tiling + index * INT64_T_SIZE));
}

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc() {
  uint32_t eventID = static_cast<uint32_t>(GetTPipePtr()->FetchEventID(event));
  AscendC::SetFlag<event>(eventID);
  AscendC::WaitFlag<event>(eventID);
}

#endif  // MULTICORE_SCHEDULER_RUNTIME_CONFIG_HPP
