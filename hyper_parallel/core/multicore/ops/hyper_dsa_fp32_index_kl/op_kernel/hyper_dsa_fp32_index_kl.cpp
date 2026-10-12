/*
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"
using namespace AscendC;
namespace {
constexpr uint32_t kWidth = 128;
constexpr uint32_t kSelected = 2048;
class Fp32IndexKl {
 public:
  __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR teacher, GM_ADDR logits,
                             GM_ADDR indices, GM_ADDR local, GM_ADDR dq, GM_ADDR dk, GM_ADDR dw, GM_ADDR loss,
                             GM_ADDR trace, GM_ADDR tiling) {
    GlobalTensor<int64_t> data;
    data.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(tiling), 4);
    queries_ = data.GetValue(0);
    keys_ = data.GetValue(1);
    heads_ = data.GetValue(2);
    phase_ = data.GetValue(3);
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(query));
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(key));
    w_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
    teacher_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(teacher));
    logits_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(logits));
    ids_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indices));
    local_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(local));
    dq_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dq));
    dk_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dk));
    dw_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dw));
    loss_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(loss));
    trace_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace));
    pipe_.InitBuffer(storage_, 64 * 1024);
    pipe_.InitBuffer(idsStorage_, kSelected * sizeof(int32_t));
    pipe_.InitBuffer(traceStorage_, 32);
    traceLocal_ = traceStorage_.Get<int64_t>();
    pipe_.InitBuffer(maskStorage_, kSelected / 8);
    auto all = storage_.Get<float>();
    scores_ = all;
    teacherLocal_ = all[kSelected];
    factor_ = all[kSelected * 2];
    temp_ = all[kSelected * 3];
    scratch_ = all[kSelected * 4];
    scalars_ = all[kSelected * 5];
    queryLocal_ = scalars_[128];
    keyLocal_ = queryLocal_[128];
    product_ = keyLocal_[128];
    gradient_ = product_[128];
    weights_ = gradient_[128];
    weightGradient_ = weights_[128];
    idsLocal_ = idsStorage_.Get<int32_t>();
    mask_ = maskStorage_.Get<uint8_t>();
  }
  __aicore__ inline void Process() {
    int64_t count = 0, checksum = 0, last = -1;
    for (int64_t row = GetBlockIdx(); row < queries_; row += GetBlockNum()) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
      DataCopy(idsLocal_, ids_[row * kSelected], kSelected);
      SetFlag<HardEvent::MTE2_S>(EVENT_ID7);
      WaitFlag<HardEvent::MTE2_S>(EVENT_ID7);
      uint32_t valid = 0;
      for (uint32_t selected = 0; selected < kSelected; ++selected) {
        const int32_t key = idsLocal_.GetValue(selected);
        if (key < -1 || key >= keys_) {
          WriteTrace(-1, 0, 0, -1);
          return;
        }
        valid += key >= 0;
      }
      if (valid == 0) {
        WriteTrace(-1, 0, 0, -1);
        return;
      }
      Load(weights_, w_, row * heads_, heads_);
      if (phase_ == 0) {
        Scores(row);
      } else {
        Distribution(row);
        Gradients(row);
      }
      ++count;
      checksum += row + 1;
      last = row;
    }
    WriteTrace(phase_ + 1, count, checksum, last);
  }

 private:
  __aicore__ inline void WriteTrace(int64_t phase, int64_t count, int64_t checksum, int64_t last) {
    // Disjoint 32-byte MTE3 writes avoid scalar GM cache-line sharing across AIVs.
    traceLocal_.SetValue(0, phase);
    traceLocal_.SetValue(1, count);
    traceLocal_.SetValue(2, checksum);
    traceLocal_.SetValue(3, last);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID7);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID7);
    DataCopy(trace_[GetBlockIdx() * 4], traceLocal_, 4);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
  }
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<float> src, int64_t offset, uint32_t elements) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    DataCopyPad(dst, src[offset], DataCopyExtParams{1, elements * 4, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID7);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID7);
  }
  __aicore__ inline void Store(GlobalTensor<float> dst, int64_t offset, LocalTensor<float> src,
                              uint32_t elements, bool atomic = false) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
    if (atomic) {
      SetAtomicAdd<float>();
    }
    DataCopyPad(dst[offset], src, DataCopyExtParams{1, elements * 4, 0, 0, 0});
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
  }
  __aicore__ inline float Scalar(LocalTensor<float> value) {
    SetFlag<HardEvent::V_S>(EVENT_ID7);
    WaitFlag<HardEvent::V_S>(EVENT_ID7);
    return value.GetValue(0);
  }
  __aicore__ inline void ScalarWrites() {
    SetFlag<HardEvent::S_V>(EVENT_ID7);
    WaitFlag<HardEvent::S_V>(EVENT_ID7);
  }
  __aicore__ inline float Dot(int32_t key) {
    Load(keyLocal_, k_, static_cast<int64_t>(key) * kWidth, kWidth);
    Mul(product_, queryLocal_, keyLocal_, kWidth);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalars_, product_, scratch_, kWidth);
    return Scalar(scalars_);
  }
  __aicore__ inline void Scores(int64_t row) {
    Duplicate(scores_, 0.0f, kSelected);
    PipeBarrier<PIPE_V>();
    for (int64_t head = 0; head < heads_; ++head) {
      Load(queryLocal_, q_, (row * heads_ + head) * kWidth, kWidth);
      const float weight = Scalar(weights_[head]);
      for (uint32_t selected = 0; selected < kSelected; ++selected) {
        const int32_t key = idsLocal_.GetValue(selected);
        if (key >= 0) {
          const float dot = Dot(key);
          const float relu = dot > 0.0f ? dot : 0.0f;
          scores_.SetValue(selected, scores_.GetValue(selected) + weight * relu);
        }
      }
    }
    ScalarWrites();
    Store(local_, row * kSelected, scores_, kSelected);
  }
  __aicore__ inline void Distribution(int64_t row) {
    Load(scores_, logits_, row * kSelected, kSelected);
    Load(teacherLocal_, teacher_, row * kSelected, kSelected);
    for (uint32_t selected = 0; selected < kSelected; ++selected) {
      if (idsLocal_.GetValue(selected) < 0) {
        scores_.SetValue(selected, -3.402823466e+38f);
      }
    }
    ScalarWrites();
    ReduceMax(scalars_, scores_, scratch_, kSelected);
    const float maximum = Scalar(scalars_);
    Adds(scores_, scores_, -maximum, kSelected);
    PipeBarrier<PIPE_V>();
    Exp(factor_, scores_, kSelected);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalars_, factor_, scratch_, kSelected);
    const float denominator = Scalar(scalars_);
    Duplicate(scalars_, denominator, 8);
    PipeBarrier<PIPE_V>();
    Ln(scalars_, scalars_, 8);
    const float logDenominator = Scalar(scalars_);
    Muls(factor_, factor_, 1.0f / denominator, kSelected);
    PipeBarrier<PIPE_V>();
    Adds(temp_, scores_, -logDenominator, kSelected);
    PipeBarrier<PIPE_V>();
    Ln(scratch_, teacherLocal_, kSelected);
    PipeBarrier<PIPE_V>();
    Sub(scratch_, scratch_, temp_, kSelected);
    PipeBarrier<PIPE_V>();
    Mul(scratch_, scratch_, teacherLocal_, kSelected);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask_, teacherLocal_, 0.0f, CMPMODE::GT, kSelected);
    PipeBarrier<PIPE_V>();
    Select(scratch_, mask_, scratch_, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, kSelected);
    PipeBarrier<PIPE_V>();
    ReduceSum(scalars_, scratch_, temp_, kSelected);
    const float loss = Scalar(scalars_);
    scalars_.SetValue(0, loss);
    ScalarWrites();
    Store(loss_, row, scalars_, 1);
    ReduceSum(scalars_, teacherLocal_, scratch_, kSelected);
    const float mass = Scalar(scalars_);
    Muls(factor_, factor_, mass, kSelected);
    PipeBarrier<PIPE_V>();
    Sub(factor_, factor_, teacherLocal_, kSelected);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Gradients(int64_t row) {
    Duplicate(weightGradient_, 0.0f, 128);
    for (int64_t head = 0; head < heads_; ++head) {
      Load(queryLocal_, q_, (row * heads_ + head) * kWidth, kWidth);
      Duplicate(gradient_, 0.0f, kWidth);
      PipeBarrier<PIPE_V>();
      const float weight = Scalar(weights_[head]);
      for (uint32_t selected = 0; selected < kSelected; ++selected) {
        const int32_t key = idsLocal_.GetValue(selected);
        scores_.SetValue(selected, 0.0f);
        if (key < 0) {
          continue;
        }
        const float dot = Dot(key);
        if (dot <= 0.0f) {
          continue;
        }
        scores_.SetValue(selected, dot);
        const float coefficient = factor_.GetValue(selected) * weight;
        Muls(product_, keyLocal_, coefficient, kWidth);
        PipeBarrier<PIPE_V>();
        Add(gradient_, gradient_, product_, kWidth);
        PipeBarrier<PIPE_V>();
        Muls(product_, queryLocal_, coefficient, kWidth);
        Store(dk_, static_cast<int64_t>(key) * kWidth, product_, kWidth, true);
      }
      ScalarWrites();
      Mul(scores_, scores_, factor_, kSelected);
      PipeBarrier<PIPE_V>();
      ReduceSum(scalars_, scores_, scratch_, kSelected);
      weightGradient_.SetValue(head, Scalar(scalars_));
      Store(dq_, (row * heads_ + head) * kWidth, gradient_, kWidth);
    }
    ScalarWrites();
    Store(dw_, row * heads_, weightGradient_, heads_);
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> storage_, idsStorage_, maskStorage_, traceStorage_;
  GlobalTensor<float> q_, k_, w_, teacher_, logits_, local_, dq_, dk_, dw_, loss_;
  GlobalTensor<int32_t> ids_;
  GlobalTensor<int64_t> trace_;
  LocalTensor<float> scores_, teacherLocal_, factor_, temp_, scratch_, scalars_, queryLocal_, keyLocal_, product_;
  LocalTensor<float> gradient_, weights_, weightGradient_;
  LocalTensor<int64_t> traceLocal_;
  LocalTensor<int32_t> idsLocal_;
  LocalTensor<uint8_t> mask_;
  int64_t queries_, keys_, heads_, phase_;
};
}  // namespace
extern "C" __global__ __aicore__ void hyper_dsa_fp32_index_kl(
  GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR teacher, GM_ADDR globalLogits, GM_ADDR indices,
  GM_ADDR localLogits, GM_ADDR gradQuery, GM_ADDR gradKey, GM_ADDR gradWeight, GM_ADDR loss, GM_ADDR trace,
  GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  Fp32IndexKl kernel;
  kernel.Init(query, key, weight, teacher, globalLogits, indices, localLogits, gradQuery, gradKey, gradWeight, loss,
              trace, tiling);
  kernel.Process();
}
