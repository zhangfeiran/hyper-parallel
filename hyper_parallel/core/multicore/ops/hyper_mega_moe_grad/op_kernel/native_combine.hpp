/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYPER_PARALLEL_NATIVE_COMBINE_HPP_
#define HYPER_PARALLEL_NATIVE_COMBINE_HPP_

namespace MulticoreRuntime {

__aicore__ inline void NativeBitCombine(
  GM_ADDR expert_output, GM_ADDR scores, GM_ADDR inverse, GM_ADDR output,
  uint32_t tokens, uint32_t hidden, uint32_t top_k, uint32_t worker, uint32_t workers) {
#ifndef __DAV_C220_CUBE__
  using Element = DTYPE_DISPATCH_TARGET;
  AscendC::GlobalTensor<Element> source;
  AscendC::GlobalTensor<Element> target;
  AscendC::GlobalTensor<uint16_t> score_bits;
  AscendC::GlobalTensor<int64_t> inverse_rows;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(expert_output));
  target.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(output));
  score_bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(scores));
  inverse_rows.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(inverse));
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> input_queue;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> output_queue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> accumulator_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> rounded_buffer;
  pipe.InitBuffer(input_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(output_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(accumulator_buffer, hidden * sizeof(float));
  pipe.InitBuffer(product_buffer, hidden * sizeof(float));
  pipe.InitBuffer(rounded_buffer, hidden * sizeof(Element));
  auto accumulator = accumulator_buffer.Get<float>();
  auto product = product_buffer.Get<float>();
  auto rounded = rounded_buffer.Get<Element>();
  for (uint32_t token = worker; token < tokens; token += workers) {
    int64_t ordered_rows[8];
    for (uint32_t slot = 0; slot < top_k; ++slot) {
      const int64_t value = inverse_rows.GetValue(static_cast<int64_t>(token) * top_k + slot);
      uint32_t position = slot;
      while (position > 0 && ordered_rows[position - 1] > value) {
        ordered_rows[position] = ordered_rows[position - 1];
        --position;
      }
      ordered_rows[position] = value;
    }
    // IndexAdd starts from positive zero and visits the original source rows in order.
    AscendC::Duplicate(accumulator, 0.0f, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t slot = 0; slot < top_k; ++slot) {
      const int64_t row = ordered_rows[slot];
      union {
        uint32_t bits;
        float value;
      } score;
      score.bits = static_cast<uint32_t>(score_bits.GetValue(row)) << 16;
      auto input = input_queue.AllocTensor<Element>();
      AscendC::DataCopy(input, source[row * hidden], hidden);
      input_queue.EnQue(input);
      input = input_queue.DeQue<Element>();
      AscendC::Cast(product, input, AscendC::RoundMode::CAST_NONE, hidden);
      input_queue.FreeTensor(input);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(product, product, score.value, hidden);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(rounded, product, AscendC::RoundMode::CAST_RINT, hidden);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(product, rounded, AscendC::RoundMode::CAST_NONE, hidden);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(accumulator, accumulator, product, hidden);
      AscendC::PipeBarrier<PIPE_V>();
    }
    auto result = output_queue.AllocTensor<Element>();
    AscendC::Cast(result, accumulator, AscendC::RoundMode::CAST_RINT, hidden);
    output_queue.EnQue(result);
    result = output_queue.DeQue<Element>();
    AscendC::DataCopy(target[static_cast<int64_t>(token) * hidden], result, hidden);
    output_queue.FreeTensor(result);
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  pipe.Destroy();
#endif
}

}  // namespace MulticoreRuntime
#endif
