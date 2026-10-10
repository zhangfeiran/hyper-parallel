// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#define ASCENDC_CUBE_ONLY
#include "asc/include/kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dense/dense_tile_abi.h"
#include "dense/dense_access.h"

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
using HyperParallelDense::DenseMatrixAccess;
using HyperParallelDense::MatrixKOffset;
using HyperParallelDense::MatrixKState;
using HyperParallelDense::MatrixSegmentEnd;
using matmul::Matmul;
using matmul::MatmulType;

namespace {
using Matrix = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using DenseMatmul = Matmul<Matrix, Matrix, Matrix>;
constexpr uint32_t VECTOR_CHUNK = 512;
constexpr uint32_t VECTOR_ROWS = 16;
constexpr uint32_t VECTOR_ELEMENTS = VECTOR_CHUNK * VECTOR_ROWS;

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

#ifdef __DAV_C220_CUBE__
__aicore__ inline void OrderedCopyA(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm,
                                   int row, int column, int use_m, int use_k, uint64_t tiling_ptr,
                                   uint64_t state) {
  const auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(tiling_ptr));
  GlobalTensor<bfloat16_t> source;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(const_cast<__gm__ void *>(gm)));
  auto target = local.ReinterpretCast<bfloat16_t>();
  const uint32_t chunk = static_cast<uint32_t>(state);
  const uint32_t stride = (use_m + 15) / 16 * 16;
  for (uint32_t done = 0; done < use_k;) {
    const uint32_t offset = column * tiling.baseK + done;
    const uint32_t remaining = chunk - offset % chunk;
    const uint32_t count = use_k - done < remaining ? use_k - done : remaining;
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = use_m;
    copy.dValue = count;
    copy.srcDValue = tiling.Ka;
    copy.dstNzC0Stride = stride;
    copy.dstNzNStride = 1;
    const uint64_t position = static_cast<uint64_t>(row * tiling.baseM) * tiling.Ka +
                              MatrixKOffset(offset, tiling.Ka, state);
    DataCopy(target[done * stride], source[position], copy);
    done += count;
  }
}

__aicore__ inline void OrderedCopyB(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm,
                                   int row, int column, int use_k, int use_n, uint64_t tiling_ptr,
                                   uint64_t state) {
  const auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(tiling_ptr));
  GlobalTensor<bfloat16_t> source;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(const_cast<__gm__ void *>(gm)));
  auto target = local.ReinterpretCast<bfloat16_t>();
  const uint32_t chunk = static_cast<uint32_t>(state);
  for (uint32_t done = 0; done < use_k;) {
    const uint32_t offset = row * tiling.baseK + done;
    const uint32_t remaining = chunk - offset % chunk;
    const uint32_t count = use_k - done < remaining ? use_k - done : remaining;
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = count;
    copy.dValue = use_n;
    copy.srcDValue = tiling.N;
    copy.dstNzC0Stride = (use_k + 15) / 16 * 16;
    copy.dstNzNStride = 1;
    const uint64_t position = static_cast<uint64_t>(MatrixKOffset(offset, tiling.Ka, state)) * tiling.N +
                              column * tiling.baseN;
    DataCopy(target[done * 16], source[position], copy);
    done += count;
  }
}

using OrderedMatmul = Matmul<Matrix, Matrix, Matrix, Matrix, CFG_NORM,
                             AscendC::MatmulCallBackFunc<nullptr, OrderedCopyA, OrderedCopyB>>;
#endif

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
        if (operation.columns_per_tile == 0 || operation.column_tiles == 0 ||
            operation.column_tiles !=
              (operation.columns + operation.columns_per_tile - 1) / operation.columns_per_tile) {
          Trap();
        }
        const uint32_t stop = tiles < base + window ? tiles : base + window;
        for (uint32_t slice = base * operation.column_tiles + cube; slice < stop * operation.column_tiles;
             slice += header_.cube_workers) {
          Execute(operation, index, slice / operation.column_tiles, slice % operation.column_tiles);
        }
      }
    }
  }

 private:
  __aicore__ inline void Wait(uint32_t event, uint32_t count) {
    const uint32_t position = event * DENSE_EVENT_STRIDE;
    const __gm__ volatile int32_t *ready = counters_.GetPhyAddr(position);
    // 其他核异步发布事件，每次轮询必须重新读取 GM。
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

  __aicore__ inline void Execute(const DenseTileOperation &operation, uint32_t index, uint32_t tile,
                                 uint32_t column_tile) {
    if (operation.dependency >= 0) {
      Wait(tile * header_.operation_count + operation.dependency, operation.dependency_count);
    }
    uint32_t first = tile * header_.rows_per_tile;
    uint32_t end = first + header_.rows_per_tile;
    end = end > header_.rows ? header_.rows : end;
    const uint32_t column = column_tile * operation.columns_per_tile;
    const uint32_t column_stop = column + operation.columns_per_tile;
    const uint32_t column_end = column_stop < operation.columns ? column_stop : operation.columns;
#ifdef __DAV_C220_CUBE__
    Matmul(operation, index, first, end - first, column, column_end);
#else
    const uint32_t middle = first + (end - first + 1) / 2;
    if (GetBlockIdx() % 2 == 0) {
      end = middle;
    } else {
      first = middle;
    }
    Swiglu(operation, first, end, column, column_end);
#endif
    Signal(tile * header_.operation_count + index);
  }

#ifdef __DAV_C220_CUBE__
  __aicore__ inline void Matmul(const DenseTileOperation &operation, uint32_t index, uint32_t first, uint32_t rows,
                                uint32_t first_column, uint32_t end_column) {
    GM_ADDR source = tilings_ + index * (sizeof(TCubeTiling) + sizeof(DenseMatrixAccess));
    auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(source));
    const auto access =
      LoadDescriptor(reinterpret_cast<const __gm__ DenseMatrixAccess *>(source + sizeof(TCubeTiling)));
    const uint32_t extent = access.m_extent >= access.n_extent ? access.m_extent : access.n_extent;
    if (access.k_chunk == 0 || access.close_shift != 0 || operation.transpose_right != 0 || extent < 4 ||
        extent / 2 > operation.contracted / access.k_chunk) {
      MatmulSegment<DenseMatmul>(operation, source, tiling, first, rows, first_column,
                                 end_column - first_column, 0);
      return;
    }
    // 分段继承原来的行任务及事件归属，仅在 SDK 访问顺序改变的位置切开矩阵。
    const bool row_axis = access.m_extent >= access.n_extent;
    for (uint32_t row = first; row < first + rows;) {
      const uint32_t row_end = row_axis ? MatrixSegmentEnd(row, access.m_span, first + rows) : first + rows;
      for (uint32_t column = first_column; column < end_column;) {
        const uint32_t column_end =
          row_axis ? end_column : MatrixSegmentEnd(column, access.n_span, end_column);
        const uint64_t state = MatrixKState(access, header_.rows, operation.columns, operation.contracted, row, column);
        if (state == 0) {
          MatmulSegment<DenseMatmul>(operation, source, tiling, row, row_end - row, column, column_end - column, 0);
        } else {
          MatmulSegment<OrderedMatmul>(operation, source, tiling, row, row_end - row, column,
                                       column_end - column, state);
        }
        column = column_end;
      }
      row = row_end;
    }
  }

  template <typename MatrixKernel>
  __aicore__ inline void MatmulSegment(const DenseTileOperation &operation, GM_ADDR source, TCubeTiling &tiling,
                                      uint32_t first, uint32_t rows, uint32_t column, uint32_t columns,
                                      uint64_t state) {
    // SDK 只允许一个活动 TPipe；Matmul 在退出作用域时先于所属 pipe 析构。
    TPipe pipe;
    MatrixKernel mm;
    REGIST_MATMUL_OBJ(&pipe, workspace_, mm, &tiling);
    GlobalTensor<bfloat16_t> left, right, output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.left]));
    right.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.right]));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.output]));
    mm.SetOrgShape(header_.rows, operation.columns, operation.contracted);
    mm.SetTail(rows, columns, operation.contracted);
    mm.SetUserDefInfo(reinterpret_cast<uint64_t>(source));
    mm.SetSelfDefineData(state);
    mm.SetTensorA(left[static_cast<uint64_t>(first) * operation.contracted]);
    const uint64_t right_offset = operation.transpose_right ? static_cast<uint64_t>(column) * operation.contracted
                                                            : column;
    mm.SetTensorB(right[right_offset], operation.transpose_right != 0);
    mm.IterateAll(output[static_cast<uint64_t>(first) * operation.columns + column]);
    mm.End();
  }
#else
  __aicore__ inline void Swiglu(const DenseTileOperation &operation, uint32_t first, uint32_t end,
                                uint32_t column_begin, uint32_t column_end) {
    if (first == end) {
      return;
    }
    TPipe pipe;
    TQue<TPosition::VECIN, 1> gate_queue, up_queue;
    TQue<TPosition::VECOUT, 1> output_queue;
    TBuf<TPosition::VECCALC> gate_float, up_float, temporary;
    pipe.InitBuffer(gate_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(up_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(output_queue, 1, VECTOR_ELEMENTS * sizeof(bfloat16_t));
    pipe.InitBuffer(gate_float, VECTOR_ELEMENTS * sizeof(float));
    pipe.InitBuffer(up_float, VECTOR_ELEMENTS * sizeof(float));
    pipe.InitBuffer(temporary, VECTOR_ELEMENTS * sizeof(float));
    GlobalTensor<bfloat16_t> packed, output;
    packed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.left]));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.output]));
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
        DataCopyExtParams copy_in{static_cast<uint16_t>(rows), static_cast<uint32_t>(size * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>((operation.contracted - size) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{true, 0, static_cast<uint8_t>(padded_size - size), 0};
        DataCopyPad(gate, packed[static_cast<uint64_t>(row) * operation.contracted + column], copy_in, pad);
        DataCopyPad(up, packed[static_cast<uint64_t>(row) * operation.contracted + operation.columns + column],
                    copy_in, pad);
        gate_queue.EnQue(gate);
        up_queue.EnQue(up);
        gate = gate_queue.DeQue<bfloat16_t>();
        up = up_queue.DeQue<bfloat16_t>();
        auto gate32 = gate_float.Get<float>();
        auto up32 = up_float.Get<float>();
        auto temp = temporary.Get<float>();
        Cast(gate32, gate, RoundMode::CAST_NONE, elements);
        Cast(up32, up, RoundMode::CAST_NONE, elements);
        PipeBarrier<PIPE_V>();
        Muls(temp, gate32, -1.0f, elements);
        PipeBarrier<PIPE_V>();
        Exp(temp, temp, elements);
        PipeBarrier<PIPE_V>();
        Adds(temp, temp, 1.0f, elements);
        PipeBarrier<PIPE_V>();
        Div(temp, gate32, temp, elements);
        PipeBarrier<PIPE_V>();
        Mul(temp, temp, up32, elements);
        PipeBarrier<PIPE_V>();
        auto result = output_queue.AllocTensor<bfloat16_t>();
        Cast(result, temp, RoundMode::CAST_RINT, elements);
        output_queue.EnQue(result);
        result = output_queue.DeQue<bfloat16_t>();
        DataCopyExtParams copy_out{static_cast<uint16_t>(rows), static_cast<uint32_t>(size * sizeof(bfloat16_t)),
                                  0, static_cast<uint32_t>((operation.columns - size) * sizeof(bfloat16_t)), 0};
        DataCopyPad(output[static_cast<uint64_t>(row) * operation.columns + column], result, copy_out);
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
