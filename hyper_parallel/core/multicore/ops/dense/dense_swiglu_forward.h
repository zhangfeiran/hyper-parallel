// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_SWIGLU_FORWARD_H_
#define HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_SWIGLU_FORWARD_H_

#ifndef __DAV_C220_CUBE__
namespace HyperParallelDense {
constexpr uint32_t VECTOR_CHUNK = 512;
constexpr uint32_t VECTOR_ROWS = 16;
constexpr uint32_t VECTOR_ELEMENTS = VECTOR_CHUNK * VECTOR_ROWS;

// 与已验证前向保持相同的 FP32 运算和 BF16 舍入；每次调用独立持有 AscendC::TPipe。
class DenseSwiGluForward {
 public:
  __aicore__ inline void Process(GM_ADDR packed_address, GM_ADDR output_address, uint32_t width, uint32_t first,
                                 uint32_t end, uint32_t column_begin, uint32_t column_end) {
    if (first == end) {
      return;
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> gate_queue, up_queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> output_queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gate_float, up_float, temporary;
    pipe.InitBuffer(gate_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(up_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(output_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(gate_float, VECTOR_ELEMENTS * sizeof(float));
    pipe.InitBuffer(up_float, VECTOR_ELEMENTS * sizeof(float));
    pipe.InitBuffer(temporary, VECTOR_ELEMENTS * sizeof(float));
    AscendC::GlobalTensor<bfloat16_t> packed, output;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(packed_address));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output_address));
    for (uint32_t row = first; row < end; row += VECTOR_ROWS) {
      const uint32_t remaining_rows = end - row;
      const uint32_t rows = remaining_rows < VECTOR_ROWS ? remaining_rows : VECTOR_ROWS;
      for (uint32_t column = column_begin; column < column_end; column += VECTOR_CHUNK) {
        const uint32_t remaining = column_end - column;
        const uint32_t size = remaining < VECTOR_CHUNK ? remaining : VECTOR_CHUNK;
        auto gate = gate_queue.AllocTensor<bfloat16_t>();
        auto up = up_queue.AllocTensor<bfloat16_t>();
        const uint32_t block_elements = 32 / sizeof(bfloat16_t);
        const uint32_t padded_size = (size + block_elements - 1) / block_elements * block_elements;
        const uint32_t elements = rows * padded_size;
        // 行间空隙由 DMA 步长处理；UB 尾部补零，保持每行独立的打包布局。
        AscendC::DataCopyExtParams copy_in{static_cast<uint16_t>(rows),
                                           static_cast<uint32_t>(size * sizeof(bfloat16_t)),
                                           static_cast<uint32_t>(((width * 2) - size) * sizeof(bfloat16_t)), 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> pad{true, 0, static_cast<uint8_t>(padded_size - size), 0};
        AscendC::DataCopyPad(gate, packed[static_cast<uint64_t>(row) * (width * 2) + column], copy_in, pad);
        AscendC::DataCopyPad(up, packed[static_cast<uint64_t>(row) * (width * 2) + width + column], copy_in, pad);
        gate_queue.EnQue(gate);
        up_queue.EnQue(up);
        gate = gate_queue.DeQue<bfloat16_t>();
        up = up_queue.DeQue<bfloat16_t>();
        auto gate32 = gate_float.Get<float>();
        auto up32 = up_float.Get<float>();
        auto temp = temporary.Get<float>();
        AscendC::Cast(gate32, gate, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::Cast(up32, up, AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(temp, gate32, -1.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(temp, temp, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(temp, temp, 1.0f, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(temp, gate32, temp, elements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(temp, temp, up32, elements);
        AscendC::PipeBarrier<PIPE_V>();
        auto result = output_queue.AllocTensor<bfloat16_t>();
        AscendC::Cast(result, temp, AscendC::RoundMode::CAST_RINT, elements);
        output_queue.EnQue(result);
        result = output_queue.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams copy_out{static_cast<uint16_t>(rows),
                                            static_cast<uint32_t>(size * sizeof(bfloat16_t)), 0,
                                            static_cast<uint32_t>((width - size) * sizeof(bfloat16_t)), 0};
        AscendC::DataCopyPad(output[static_cast<uint64_t>(row) * width + column], result, copy_out);
        output_queue.FreeTensor(result);
        gate_queue.FreeTensor(gate);
        up_queue.FreeTensor(up);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
};
}  // namespace HyperParallelDense
#endif
#endif  // HYPER_PARALLEL_CORE_MULTICORE_OPS_DENSE_DENSE_SWIGLU_FORWARD_H_
