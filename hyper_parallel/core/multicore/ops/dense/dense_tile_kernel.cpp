// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#define ASCENDC_CUBE_ONLY
#include "asc/include/kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dense/dense_tile_abi.h"

using AscendC::Adds;
using AscendC::CacheLine;
using AscendC::Cast;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DcciDst;
using AscendC::Div;
using AscendC::Exp;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::RoundMode;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::Trap;
using HyperParallelDense::DENSE_EVENT_STRIDE;
using HyperParallelDense::DENSE_MATMUL;
using HyperParallelDense::DENSE_MAX_OPERATIONS;
using HyperParallelDense::DENSE_MAX_VALUES;
using HyperParallelDense::DENSE_SWIGLU;
using HyperParallelDense::DENSE_TILE_MAGIC;
using HyperParallelDense::DENSE_TILE_VERSION;
using HyperParallelDense::DenseTileHeader;
using HyperParallelDense::DenseTileOperation;
using matmul::Matmul;
using matmul::MatmulType;

namespace {
using Matrix = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using DenseMatmul = Matmul<Matrix, Matrix, Matrix>;
constexpr uint32_t VECTOR_CHUNK = 512;

template <typename Data>
__aicore__ inline Data LoadDescriptor(const __gm__ Data *source) {
  Data result;
  auto *target = reinterpret_cast<uint32_t *>(&result);
  const auto *words = reinterpret_cast<const __gm__ uint32_t *>(source);
  for (uint32_t index = 0; index < sizeof(Data) / sizeof(uint32_t); ++index) {
    target[index] = words[index];
  }
  return result;
}

class DenseTileWorker {
 public:
  __aicore__ inline void Init(GM_ADDR values, GM_ADDR config, GM_ADDR tilings, GM_ADDR events, GM_ADDR ones,
                              GM_ADDR workspace) {
    const auto *header = reinterpret_cast<const __gm__ DenseTileHeader *>(config);
    header_ = LoadDescriptor(header);
    if (header_.magic != DENSE_TILE_MAGIC || header_.version != DENSE_TILE_VERSION || header_.rows_per_tile == 0 ||
        header_.cube_workers == 0 || header_.cube_workers > 24 || header_.operation_count > DENSE_MAX_OPERATIONS ||
        header_.value_count > DENSE_MAX_VALUES || header_.prefetch_tiles == 0) {
      Trap();
    }
    operations_ = reinterpret_cast<const __gm__ DenseTileOperation *>(config + sizeof(DenseTileHeader));
    values_ = reinterpret_cast<const __gm__ uint64_t *>(values);
    tilings_ = tilings;
    counters_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(events));
    ones_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ones));
    workspace_ = workspace;
  }

  __aicore__ inline void Process() {
#ifdef __DAV_C220_CUBE__
    const uint32_t cube = GetBlockIdx();
#else
    const uint32_t cube = GetBlockIdx() / 2;
#endif
    if (cube >= header_.cube_workers) {
      return;
    }
    const uint32_t tiles = (header_.rows + header_.rows_per_tile - 1) / header_.rows_per_tile;
    const uint32_t window = header_.cube_workers * header_.prefetch_tiles;
    for (uint32_t base = 0; base < tiles; base += window) {
      for (uint32_t index = 0; index < header_.operation_count; ++index) {
        const DenseTileOperation operation = LoadDescriptor(&operations_[index]);
#ifdef __DAV_C220_CUBE__
        if (operation.kind != DENSE_MATMUL) {
          continue;
        }
#else
        if (operation.kind != DENSE_SWIGLU) {
          continue;
        }
#endif
        for (uint32_t tile = base + cube; tile < tiles && tile < base + window; tile += header_.cube_workers) {
          Execute(operation, index, tile);
        }
      }
    }
  }

 private:
  __aicore__ inline void Wait(uint32_t event, uint32_t count) {
    const uint32_t position = event * DENSE_EVENT_STRIDE;
    const __gm__ volatile int32_t *ready = counters_.GetPhyAddr(position);
    // Other cores publish asynchronously; each poll must reload the GM value.
    while (true) {
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counters_[position]);
      PipeBarrier<PIPE_ALL>();
      if (*ready >= count) {
        return;
      }
    }
  }

  template <AscendC::HardEvent event>
  __aicore__ inline void SyncEvent(TPipe &pipe) {
    const auto id = static_cast<event_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
  }

  __aicore__ inline void Signal(uint32_t event) {
    TPipe pipe;
#ifdef __DAV_C220_CUBE__
    TBuf<TPosition::A1> buffer;
#else
    TBuf<TPosition::VECOUT> buffer;
#endif
    pipe.InitBuffer(buffer, 32);
    auto value = buffer.Get<int32_t>();
#ifdef __DAV_C220_CUBE__
    SyncEvent<AscendC::HardEvent::S_MTE2>(pipe);
    DataCopy(value, ones_, 8);
    SyncEvent<AscendC::HardEvent::MTE2_S>(pipe);
    SetAtomicAdd<int32_t>();
    SyncEvent<AscendC::HardEvent::S_MTE2>(pipe);
    DataCopy(counters_[event * DENSE_EVENT_STRIDE], value, 8);
    SyncEvent<AscendC::HardEvent::MTE2_S>(pipe);
#else
    value.SetValue(0, 1);
    for (uint32_t index = 1; index < 8; ++index) {
      value.SetValue(index, 0);
    }
    SetAtomicAdd<int32_t>();
    SyncEvent<AscendC::HardEvent::S_MTE3>(pipe);
    DataCopy(counters_[event * DENSE_EVENT_STRIDE], value, 8);
    SyncEvent<AscendC::HardEvent::MTE3_S>(pipe);
#endif
    SetAtomicNone();
    buffer.FreeTensor(value);
    pipe.Destroy();
  }

  __aicore__ inline void Execute(const DenseTileOperation &operation, uint32_t index, uint32_t tile) {
    if (operation.dependency >= 0) {
      Wait(tile * header_.operation_count + operation.dependency, operation.dependency_count);
    }
    uint32_t first = tile * header_.rows_per_tile;
    uint32_t end = first + header_.rows_per_tile;
    end = end > header_.rows ? header_.rows : end;
#ifdef __DAV_C220_CUBE__
    Matmul(operation, index, first, end - first);
#else
    const uint32_t middle = first + (end - first + 1) / 2;
    if (GetBlockIdx() % 2 == 0) {
      end = middle;
    } else {
      first = middle;
    }
    Swiglu(operation, first, end);
#endif
    Signal(tile * header_.operation_count + index);
  }

#ifdef __DAV_C220_CUBE__
  __aicore__ inline void Matmul(const DenseTileOperation &operation, uint32_t index, uint32_t first, uint32_t rows) {
    TPipe pipe;
    TCubeTiling tiling;
    const __gm__ int32_t *source = reinterpret_cast<const __gm__ int32_t *>(tilings_ + index * sizeof(TCubeTiling));
    int32_t *target = reinterpret_cast<int32_t *>(&tiling);
    for (uint32_t word = 0; word < sizeof(TCubeTiling) / sizeof(int32_t); ++word) {
      target[word] = source[word];
    }
    DenseMatmul mm;
    REGIST_MATMUL_OBJ(&pipe, workspace_, mm, &tiling);
    GlobalTensor<bfloat16_t> left, right, output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.left]));
    right.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.right]));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.output]));
    mm.SetOrgShape(header_.rows, operation.columns, operation.contracted);
    mm.SetTail(rows, operation.columns, operation.contracted);
    mm.SetTensorA(left[static_cast<uint64_t>(first) * operation.contracted]);
    mm.SetTensorB(right, operation.transpose_right != 0);
    mm.IterateAll(output[static_cast<uint64_t>(first) * operation.columns]);
    mm.End();
  }
#else
  __aicore__ inline void Swiglu(const DenseTileOperation &operation, uint32_t first, uint32_t end) {
    TPipe pipe;
    TQue<TPosition::VECIN, 1> gate_queue, up_queue;
    TQue<TPosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> gate_float, up_float, temporary;
    pipe.InitBuffer(gate_queue, 1, VECTOR_CHUNK * sizeof(bfloat16_t));
    pipe.InitBuffer(up_queue, 1, VECTOR_CHUNK * sizeof(bfloat16_t));
    pipe.InitBuffer(output_queue, 1, VECTOR_CHUNK * sizeof(bfloat16_t));
    pipe.InitBuffer(gate_float, VECTOR_CHUNK * sizeof(float));
    pipe.InitBuffer(up_float, VECTOR_CHUNK * sizeof(float));
    pipe.InitBuffer(temporary, VECTOR_CHUNK * sizeof(float));
    GlobalTensor<bfloat16_t> packed, output;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.left]));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.output]));
    for (uint32_t row = first; row < end; ++row) {
      for (uint32_t column = 0; column < operation.columns; column += VECTOR_CHUNK) {
        const uint32_t remaining = operation.columns - column;
        const uint32_t size = remaining < VECTOR_CHUNK ? remaining : VECTOR_CHUNK;
        auto gate = gate_queue.AllocTensor<bfloat16_t>();
        auto up = up_queue.AllocTensor<bfloat16_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(size * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyPad(gate, packed[static_cast<uint64_t>(row) * operation.contracted + column], copy, pad);
        DataCopyPad(up, packed[static_cast<uint64_t>(row) * operation.contracted + operation.columns + column], copy,
                    pad);
        gate_queue.EnQue(gate);
        up_queue.EnQue(up);
        gate = gate_queue.DeQue<bfloat16_t>();
        up = up_queue.DeQue<bfloat16_t>();
        auto gate32 = gate_float.Get<float>();
        auto up32 = up_float.Get<float>();
        auto temp = temporary.Get<float>();
        Cast(gate32, gate, RoundMode::CAST_NONE, size);
        Cast(up32, up, RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Muls(temp, gate32, -1.0f, size);
        PipeBarrier<PIPE_V>();
        Exp(temp, temp, size);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, 1.0f, size);
        PipeBarrier<PIPE_V>();
        Div(temp, gate32, temp, size);
        PipeBarrier<PIPE_V>();
        Mul(temp, temp, up32, size);
        PipeBarrier<PIPE_V>();
        auto result = output_queue.AllocTensor<bfloat16_t>();
        Cast(result, temp, RoundMode::CAST_RINT, size);
        output_queue.EnQue(result);
        result = output_queue.DeQue<bfloat16_t>();
        DataCopyPad(output[static_cast<uint64_t>(row) * operation.columns + column], result, copy);
        output_queue.FreeTensor(result);
        gate_queue.FreeTensor(gate);
        up_queue.FreeTensor(up);
      }
    }
    PipeBarrier<PIPE_ALL>();
  }
#endif

  DenseTileHeader header_{};
  const __gm__ DenseTileOperation *operations_ = nullptr;
  const __gm__ uint64_t *values_ = nullptr;
  GlobalTensor<int32_t> counters_, ones_;
  GM_ADDR tilings_ = nullptr;
  GM_ADDR workspace_ = nullptr;
};
}  // namespace

extern "C" __global__ __aicore__ void hyper_parallel_dense_tile(GM_ADDR values, GM_ADDR config, GM_ADDR tilings,
                                                                GM_ADDR events, GM_ADDR ones, GM_ADDR workspace) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  DenseTileWorker worker;
  worker.Init(values, config, tilings, events, ones, workspace);
  worker.Process();
}
