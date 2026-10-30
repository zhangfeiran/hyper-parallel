/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef HYPER_PARALLEL_NATIVE_ORDER_HPP_
#define HYPER_PARALLEL_NATIVE_ORDER_HPP_

namespace MulticoreRuntime {

__aicore__ inline int64_t NativeOrderMapOffset(int64_t rows, int64_t intermediate, int64_t hidden, bool include_input) {
  int64_t offset = 0;
  const int64_t widths[] = {intermediate * 2, intermediate, hidden};
  for (uint32_t operand = 0; operand < 3; ++operand) {
    offset = (offset + rows * widths[operand] * sizeof(DTYPE_DISPATCH_TARGET) + 127) / 128 * 128;
  }
  if (include_input) {
    offset = (offset + rows * hidden * sizeof(DTYPE_DISPATCH_TARGET) + 127) / 128 * 128;
  }
  return offset;
}

__aicore__ inline void InitNativeOrderInputs(GM_ADDR *inputs, GM_ADDR config) {
  const auto *header = reinterpret_cast<__gm__ RuntimeHeader *>(config);
  const int64_t intermediate = header->native_order_intermediate;
  if (intermediate == 0) {
    return;
  }
  const int64_t experts = getExtraValueFromTiling(inputs[30], 2) / getExtraValueFromTiling(inputs[30], 1);
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(inputs[19]), experts);
  const int64_t rows = groups.GetValue(experts - 1);
  const int64_t activation_offset = (rows * intermediate * 4 + 127) / 128 * 128;
  const int64_t dy_offset = (activation_offset + rows * intermediate * 2 + 127) / 128 * 128;
  inputs[33] = inputs[26] + activation_offset;
  inputs[34] = inputs[26] + dy_offset;
  inputs[35] = inputs[26] + (dy_offset + rows * getExtraValueFromTiling(inputs[30], 3) * 2 + 127) / 128 * 128;
}

template <AscendC::HardEvent Type>
__aicore__ inline void NativeOrderSetFlag(uint32_t slot) {
  if (slot == 0) {
    AscendC::SetFlag<Type>(EVENT_ID0);
  } else {
    AscendC::SetFlag<Type>(EVENT_ID1);
  }
}

template <AscendC::HardEvent Type>
__aicore__ inline void NativeOrderWaitFlag(uint32_t slot) {
  if (slot == 0) {
    AscendC::WaitFlag<Type>(EVENT_ID0);
  } else {
    AscendC::WaitFlag<Type>(EVENT_ID1);
  }
}

__aicore__ inline void NativeBitReorder(GM_ADDR *inputs, GM_ADDR tiling, const TaskDesc &task) {
#ifndef __DAV_C220_CUBE__
  using Element = DTYPE_DISPATCH_TARGET;
  const int64_t experts = getExtraValueFromTiling(tiling, 2) / getExtraValueFromTiling(tiling, 1);
  AscendC::GlobalTensor<int64_t> groups;
  groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(inputs[19]), experts);
  const int64_t tasks_per_expert = task.task_split_num / experts;
  const int64_t expert = task.task_index / tasks_per_expert;
  const int64_t start = expert == 0 ? 0 : groups.GetValue(expert - 1);
  const int64_t end = groups.GetValue(expert);
  const int64_t first = start + (task.task_index % tasks_per_expert) * task.task_split_value;
  if (first >= end) {
    return;
  }
  const int64_t limit = first + task.task_split_value < end ? first + task.task_split_value : end;
  const int64_t rows = groups.GetValue(experts - 1);
  // The byte workspace starts with BF16 dU; the fresh INT64 row map follows it.
  const int64_t hidden = getExtraValueFromTiling(tiling, 3);
  const int64_t map_offset = NativeOrderMapOffset(rows, task.extra_value_0 / 2, hidden, task.extra_value_2 != 0);
  AscendC::GlobalTensor<int64_t> order;
  order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(inputs[26] + map_offset), rows);
  uint32_t width = 0;
  for (uint32_t operand = 0; operand < task.num_inputs; ++operand) {
    width = width > task.inputs[operand].dim[1] ? width : task.inputs[operand].dim[1];
  }
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> staging;
  pipe.InitBuffer(staging, width * sizeof(Element) * 2);
  auto buffer = staging.Get<Element>();
  uint32_t copied = 0;
  GM_ADDR pending_target = nullptr;
  uint32_t pending_columns = 0;
  for (int64_t row = first; row < limit; ++row) {
    const int64_t mapped_row = order.GetValue(row);
    const int64_t source_row = task.extra_value_1 == 0 ? mapped_row : row;
    const int64_t target_row = task.extra_value_1 == 0 ? row : mapped_row;
    if (mapped_row < start || mapped_row >= end) {
      AscendC::Trap();
    }
    for (uint32_t operand = 0; operand < task.num_inputs; ++operand) {
      const uint32_t columns = task.inputs[operand].dim[1];
      AscendC::GlobalTensor<Element> source;
      source.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(inputs[task.inputs[operand].input_position]));
      const uint32_t slot = copied % 2;
      if (copied >= 2) {
        NativeOrderWaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
      }
      AscendC::DataCopy(buffer[slot * width], source[source_row * columns], columns);
      NativeOrderSetFlag<AscendC::HardEvent::MTE2_MTE3>(slot);
      if (copied > 0) {
        const uint32_t previous = 1 - slot;
        NativeOrderWaitFlag<AscendC::HardEvent::MTE2_MTE3>(previous);
        AscendC::GlobalTensor<Element> target;
        target.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(pending_target));
        AscendC::DataCopy(target, buffer[previous * width], pending_columns);
        NativeOrderSetFlag<AscendC::HardEvent::MTE3_MTE2>(previous);
      }
      pending_target = inputs[task.outputs[operand].input_position] + target_row * columns * sizeof(Element);
      pending_columns = columns;
      ++copied;
    }
  }
  const uint32_t last = (copied - 1) % 2;
  NativeOrderWaitFlag<AscendC::HardEvent::MTE2_MTE3>(last);
  AscendC::GlobalTensor<Element> target;
  target.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(pending_target));
  AscendC::DataCopy(target, buffer[last * width], pending_columns);
  NativeOrderSetFlag<AscendC::HardEvent::MTE3_MTE2>(last);
  if (copied > 1) {
    NativeOrderWaitFlag<AscendC::HardEvent::MTE3_MTE2>(1 - last);
  }
  NativeOrderWaitFlag<AscendC::HardEvent::MTE3_MTE2>(last);
  AscendC::PipeBarrier<PIPE_ALL>();
  pipe.Destroy();
#endif
}

}  // namespace MulticoreRuntime
#endif
