// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_COMMON_VECTOR_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_COMMON_VECTOR_H_

#include "dense/dense_backward_abi.h"

#ifndef __DAV_C220_CUBE__
#include "dense/dense_backward_vector.h"

namespace HyperParallelDense {
class DenseCommonVector {
 public:
  __aicore__ inline void Process(GM_ADDR left_address, GM_ADDR right_address, GM_ADDR output_address, uint32_t width,
                                 uint32_t first, uint32_t end, uint32_t column_begin, uint32_t column_end,
                                 uint32_t kind) {
    if (kind < DENSE_SILU || kind > DENSE_SILU_GRAD) {
      AscendC::Trap();
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> left_queue, right_queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> left_float, right_float, result_float, temporary;
    pipe.InitBuffer(left_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(right_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(output_queue, 1, BACKWARD_VECTOR_ELEMENTS * 2);
    pipe.InitBuffer(left_float, BACKWARD_VECTOR_ELEMENTS * 4);
    pipe.InitBuffer(right_float, BACKWARD_VECTOR_ELEMENTS * 4);
    pipe.InitBuffer(result_float, BACKWARD_VECTOR_ELEMENTS * 4);
    pipe.InitBuffer(temporary, BACKWARD_VECTOR_ELEMENTS * 4);
    AscendC::GlobalTensor<bfloat16_t> left, right, output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(left_address));
    right.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(right_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output_address));
    for (uint32_t row = first; row < end; row += BACKWARD_VECTOR_ROWS) {
      const uint32_t rows = end - row < BACKWARD_VECTOR_ROWS ? end - row : BACKWARD_VECTOR_ROWS;
      for (uint32_t column = column_begin; column < column_end; column += BACKWARD_VECTOR_COLUMNS) {
        const uint32_t columns =
          column_end - column < BACKWARD_VECTOR_COLUMNS ? column_end - column : BACKWARD_VECTOR_COLUMNS;
        const uint32_t padded = (columns + 15) / 16 * 16;
        const uint32_t elements = rows * padded;
        const uint64_t offset = static_cast<uint64_t>(row) * width + column;
        auto left_value = left_queue.AllocTensor<bfloat16_t>();
        CopyBackwardInput(left_value, left, offset, rows, columns, width, padded);
        left_queue.EnQue(left_value);
        left_value = left_queue.DeQue<bfloat16_t>();
        auto x = left_float.Get<float>();
        auto y = right_float.Get<float>();
        auto result = result_float.Get<float>();
        auto temp = temporary.Get<float>();
        AscendC::Cast(x, left_value, AscendC::RoundMode::CAST_NONE, elements);
        if (kind != DENSE_SILU) {
          auto right_value = right_queue.AllocTensor<bfloat16_t>();
          CopyBackwardInput(right_value, right, offset, rows, columns, width, padded);
          right_queue.EnQue(right_value);
          right_value = right_queue.DeQue<bfloat16_t>();
          AscendC::Cast(y, right_value, AscendC::RoundMode::CAST_NONE, elements);
          right_queue.FreeTensor(right_value);
        }
        AscendC::PipeBarrier<PIPE_V>();
        Compute(x, y, result, temp, elements, kind);
        auto value = output_queue.AllocTensor<bfloat16_t>();
        AscendC::Cast(value, result, AscendC::RoundMode::CAST_RINT, elements);
        output_queue.EnQue(value);
        value = output_queue.DeQue<bfloat16_t>();
        CopyBackwardOutput(output, offset, value, rows, columns, width);
        output_queue.FreeTensor(value);
        left_queue.FreeTensor(left_value);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 private:
  __aicore__ inline void Compute(AscendC::LocalTensor<float> x, AscendC::LocalTensor<float> y,
                                 AscendC::LocalTensor<float> result, AscendC::LocalTensor<float> temp,
                                 uint32_t elements, uint32_t kind) {
    if (kind == DENSE_MULTIPLY) {
      AscendC::Mul(result, x, y, elements);
      AscendC::PipeBarrier<PIPE_V>();
      return;
    }
    AscendC::Muls(temp, x, -1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(temp, temp, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(temp, temp, 1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    if (kind == DENSE_SILU) {
      AscendC::Div(result, x, temp, elements);
      AscendC::PipeBarrier<PIPE_V>();
      return;
    }
    // 对应选定 SDK 的 legacy SiluGrad BF16 规则，逐条保留 FP32 乘法次序。
    AscendC::Duplicate(result, 1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(temp, result, temp, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(result, result, temp, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(result, result, x, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(result, result, 1.0f, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(result, temp, result, elements);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(result, y, result, elements);
    AscendC::PipeBarrier<PIPE_V>();
  }
};
}  // namespace HyperParallelDense
#endif
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_COMMON_VECTOR_H_
