/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYPER_PARALLEL_NATIVE_COMBINE_GRAD_HPP_
#define HYPER_PARALLEL_NATIVE_COMBINE_GRAD_HPP_

namespace MulticoreRuntime {

__aicore__ inline void NativeBitCombineGrad(
  GM_ADDR expert_output, GM_ADDR grad_output, GM_ADDR scores, GM_ADDR source_order,
  GM_ADDR grad_expert, GM_ADDR product_output, uint32_t rows, uint32_t hidden, uint32_t top_k,
  uint32_t need_weight_grad, uint32_t worker, uint32_t workers, bool reduce_scores) {
#ifndef __DAV_C220_CUBE__
  using Element = DTYPE_DISPATCH_TARGET;
  AscendC::GlobalTensor<Element> expert;
  AscendC::GlobalTensor<Element> dy;
  AscendC::GlobalTensor<Element> dx;
  AscendC::GlobalTensor<Element> product;
  AscendC::GlobalTensor<uint16_t> score_bits;
  AscendC::GlobalTensor<int64_t> order;
  expert.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(expert_output));
  dy.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(grad_output));
  dx.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(grad_expert));
  product.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(product_output));
  score_bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(scores));
  order.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(source_order));
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> dy_queue;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> expert_queue;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> dx_queue;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> product_queue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> dy_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> product_buffer;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buffer;
  pipe.InitBuffer(dy_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(expert_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(dx_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(product_queue, 1, hidden * sizeof(Element));
  pipe.InitBuffer(dy_buffer, hidden * sizeof(float));
  pipe.InitBuffer(product_buffer, hidden * sizeof(float));
  pipe.InitBuffer(sum_buffer, 32);
  auto float_dy = dy_buffer.Get<float>();
  auto float_product = product_buffer.Get<float>();
  for (uint32_t row = worker; row < rows; row += workers) {
    const int64_t token = order.GetValue(row) / top_k;
    union {
      uint32_t bits;
      float value;
    } score;
    score.bits = static_cast<uint32_t>(score_bits.GetValue(row)) << 16;
    auto input_dy = dy_queue.AllocTensor<Element>();
    AscendC::DataCopy(input_dy, dy[token * hidden], hidden);
    dy_queue.EnQue(input_dy);
    input_dy = dy_queue.DeQue<Element>();
    AscendC::Cast(float_dy, input_dy, AscendC::RoundMode::CAST_NONE, hidden);
    dy_queue.FreeTensor(input_dy);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(float_product, float_dy, score.value, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    auto result_dx = dx_queue.AllocTensor<Element>();
    AscendC::Cast(result_dx, float_product, AscendC::RoundMode::CAST_RINT, hidden);
    dx_queue.EnQue(result_dx);
    result_dx = dx_queue.DeQue<Element>();
    AscendC::DataCopy(dx[static_cast<int64_t>(row) * hidden], result_dx, hidden);
    dx_queue.FreeTensor(result_dx);
    if (need_weight_grad) {
      auto input_expert = expert_queue.AllocTensor<Element>();
      AscendC::DataCopy(input_expert, expert[static_cast<int64_t>(row) * hidden], hidden);
      expert_queue.EnQue(input_expert);
      input_expert = expert_queue.DeQue<Element>();
      AscendC::Cast(float_product, input_expert, AscendC::RoundMode::CAST_NONE, hidden);
      expert_queue.FreeTensor(input_expert);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(float_product, float_dy, float_product, hidden);
      AscendC::PipeBarrier<PIPE_V>();
      auto result_product = product_queue.AllocTensor<Element>();
      AscendC::Cast(result_product, float_product, AscendC::RoundMode::CAST_RINT, hidden);
      if (reduce_scores) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(float_product, result_product, AscendC::RoundMode::CAST_NONE, hidden);
        AscendC::PipeBarrier<PIPE_V>();
        // Preserve the deployed BF16 ReduceSum's FP32 addition tree, including the tail order.
        AscendC::Add(float_product, float_product, float_product[2048], 2048);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(float_product, float_product, float_product[4096], 1024);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t width = 1024; width >= 64; width /= 2) {
          AscendC::Add(float_product, float_product, float_product[width], width);
          AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum(sum_buffer.Get<float>(), float_product, 64, 1, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(result_product, sum_buffer.Get<float>(), AscendC::RoundMode::CAST_RINT, 1);
        AscendC::PipeBarrier<PIPE_V>();
      }
      product_queue.EnQue(result_product);
      result_product = product_queue.DeQue<Element>();
      AscendC::DataCopy(product[static_cast<int64_t>(row) * (reduce_scores ? 16 : hidden)],
                        result_product, reduce_scores ? 16 : hidden);
      product_queue.FreeTensor(result_product);
    }
  }
  AscendC::PipeBarrier<PIPE_ALL>();
  pipe.Destroy();
#endif
}

}  // namespace MulticoreRuntime
#endif
