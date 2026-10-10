// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_VECTOR_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_VECTOR_H_

#ifndef __DAV_C220_CUBE__
#include "swi_glu_grad/swi_glu_grad_bf16.hpp"
#include "dense/dense_backward_abi.h"

namespace HyperParallelDense {
constexpr uint32_t BACKWARD_VECTOR_COLUMNS = 256;
constexpr uint32_t BACKWARD_VECTOR_ROWS = 16;
constexpr uint32_t BACKWARD_VECTOR_ELEMENTS = BACKWARD_VECTOR_COLUMNS * BACKWARD_VECTOR_ROWS;

__aicore__ inline void CopyBackwardInput(AscendC::LocalTensor<bfloat16_t> local,
                                        AscendC::GlobalTensor<bfloat16_t> source, uint64_t offset,
                                        uint32_t rows, uint32_t columns, uint32_t stride, uint32_t padded) {
  AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), columns * 2, (stride - columns) * 2, 0, 0};
  AscendC::DataCopyPadExtParams<bfloat16_t> pad{true, 0, static_cast<uint8_t>(padded - columns), 0};
  AscendC::DataCopyPad(local, source[offset], copy, pad);
}

__aicore__ inline void CopyBackwardOutput(AscendC::GlobalTensor<bfloat16_t> target, uint64_t offset,
                                         AscendC::LocalTensor<bfloat16_t> local, uint32_t rows,
                                         uint32_t columns, uint32_t stride) {
  AscendC::DataCopyExtParams copy{static_cast<uint16_t>(rows), columns * 2, 0, (stride - columns) * 2, 0};
  AscendC::DataCopyPad(target[offset], local, copy);
}

class DenseSwiGluGrad : public SwiGluGradBF16<bfloat16_t, float, bfloat16_t, 1> {
 public:
  __aicore__ inline void Process(GM_ADDR packed, GM_ADDR gradient, GM_ADDR output, uint32_t width,
                                 uint32_t first, uint32_t end, uint32_t column_begin, uint32_t column_end) {
    this->beta = -1.0f;
    this->InitUbBuffer(BACKWARD_VECTOR_ELEMENTS);
    this->aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(packed));
    this->lGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gradient));
    this->mGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
    for (uint32_t row = first; row < end; row += BACKWARD_VECTOR_ROWS) {
      const uint32_t rows = end - row < BACKWARD_VECTOR_ROWS ? end - row : BACKWARD_VECTOR_ROWS;
      for (uint32_t column = column_begin; column < column_end; column += BACKWARD_VECTOR_COLUMNS) {
        const uint32_t columns = column_end - column < BACKWARD_VECTOR_COLUMNS ? column_end - column
                                                                             : BACKWARD_VECTOR_COLUMNS;
        const uint32_t padded = (columns + 15) / 16 * 16;
        auto gate = this->inQueueA.AllocTensor<bfloat16_t>();
        auto up = this->inQueueB.AllocTensor<bfloat16_t>();
        auto grad = this->inQueueL.AllocTensor<bfloat16_t>();
        CopyBackwardInput(gate, this->aGm, static_cast<uint64_t>(row) * width * 2 + column,
                           rows, columns, width * 2, padded);
        CopyBackwardInput(up, this->aGm, static_cast<uint64_t>(row) * width * 2 + width + column,
                           rows, columns, width * 2, padded);
        CopyBackwardInput(grad, this->lGm, static_cast<uint64_t>(row) * width + column,
                           rows, columns, width, padded);
        this->inQueueA.EnQue(gate);
        this->inQueueB.EnQue(up);
        this->inQueueL.EnQue(grad);
        // 直接调用选定 SDK 的 BF16 recipe，保留它的 FP32 操作顺序及两次输出舍入。
        this->Compute(rows * padded);
        auto gate_grad = this->outQueueM.DeQue<bfloat16_t>();
        auto up_grad = this->outQueueN.DeQue<bfloat16_t>();
        CopyBackwardOutput(this->mGm, static_cast<uint64_t>(row) * width * 2 + column,
                            gate_grad, rows, columns, width * 2);
        CopyBackwardOutput(this->mGm, static_cast<uint64_t>(row) * width * 2 + width + column,
                            up_grad, rows, columns, width * 2);
        this->outQueueM.FreeTensor(gate_grad);
        this->outQueueN.FreeTensor(up_grad);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
};

class DenseGradientAdd {
 public:
  __aicore__ inline void Process(GM_ADDR left_address, GM_ADDR right_address, GM_ADDR output_address,
                                 uint32_t width, uint32_t first, uint32_t end,
                                 uint32_t column_begin, uint32_t column_end, bool zero) {
    using AscendC::TPosition;
    AscendC::TPipe pipe;
    AscendC::TQue<TPosition::VECIN, 1> left_queue, right_queue;
    AscendC::TQue<TPosition::VECOUT, 1> output_queue;
    AscendC::TBuf<TPosition::VECCALC> left_float, right_float;
    pipe.InitBuffer(left_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(right_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(output_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(left_float, BACKWARD_VECTOR_ELEMENTS * 4);
    pipe.InitBuffer(right_float, BACKWARD_VECTOR_ELEMENTS * 4);
    AscendC::GlobalTensor<bfloat16_t> left, right, output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(left_address));
    right.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(right_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output_address));
    for (uint32_t row = first; row < end; row += BACKWARD_VECTOR_ROWS) {
      const uint32_t rows = end - row < BACKWARD_VECTOR_ROWS ? end - row : BACKWARD_VECTOR_ROWS;
      for (uint32_t column = column_begin; column < column_end; column += BACKWARD_VECTOR_COLUMNS) {
        const uint32_t columns = column_end - column < BACKWARD_VECTOR_COLUMNS ? column_end - column
                                                                             : BACKWARD_VECTOR_COLUMNS;
        const uint32_t padded = (columns + 15) / 16 * 16;
        const uint32_t count = rows * padded;
        const uint64_t offset = static_cast<uint64_t>(row) * width + column;
        auto result = output_queue.AllocTensor<bfloat16_t>();
        if (zero) {
          AscendC::Duplicate(result, static_cast<bfloat16_t>(0.0f), count);
        } else {
          auto first_value = left_queue.AllocTensor<bfloat16_t>();
          auto second_value = right_queue.AllocTensor<bfloat16_t>();
          CopyBackwardInput(first_value, left, offset, rows, columns, width, padded);
          CopyBackwardInput(second_value, right, offset, rows, columns, width, padded);
          left_queue.EnQue(first_value);
          right_queue.EnQue(second_value);
          first_value = left_queue.DeQue<bfloat16_t>();
          second_value = right_queue.DeQue<bfloat16_t>();
          auto first_float = left_float.Get<float>();
          auto second_float = right_float.Get<float>();
          AscendC::Cast(first_float, first_value, AscendC::RoundMode::CAST_NONE, count);
          AscendC::Cast(second_float, second_value, AscendC::RoundMode::CAST_NONE, count);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(first_float, first_float, second_float, count);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(result, first_float, AscendC::RoundMode::CAST_RINT, count);
          left_queue.FreeTensor(first_value);
          right_queue.FreeTensor(second_value);
        }
        output_queue.EnQue(result);
        result = output_queue.DeQue<bfloat16_t>();
        CopyBackwardOutput(output, offset, result, rows, columns, width);
        output_queue.FreeTensor(result);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
};
}  // namespace HyperParallelDense
#endif
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_BACKWARD_VECTOR_H_
