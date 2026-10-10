// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#define ASCENDC_CUBE_ONLY
#include "asc/include/kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dense/dense_tile_abi.h"
#include "dense/dense_backward_abi.h"
#include "dense/dense_backward_vector.h"
#include "dense/dense_swiglu_forward.h"
#include "dense/dense_common_vector.h"
#include "dense/dense_access.h"

namespace {
using AscendC::CacheLine;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DataCopy;
using AscendC::DcciDst;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::PipeBarrier;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::Trap;
using AscendC::WaitFlag;
using HyperParallelDense::DENSE_BACKWARD_MAGIC;
using HyperParallelDense::DENSE_BACKWARD_MAX_OPERATIONS;
using HyperParallelDense::DENSE_BACKWARD_MAX_VALUES;
using HyperParallelDense::DENSE_BACKWARD_VERSION;
using HyperParallelDense::DENSE_EVENT_STRIDE;
using HyperParallelDense::DENSE_MATMUL;
using HyperParallelDense::DENSE_SILU;
using HyperParallelDense::DENSE_SILU_GRAD;
using HyperParallelDense::DENSE_SWIGLU;
using HyperParallelDense::DENSE_SWIGLU_GRAD;
using HyperParallelDense::DENSE_ZERO;
using HyperParallelDense::DenseBackwardDependency;
using HyperParallelDense::DenseBackwardHeader;
using HyperParallelDense::DenseBackwardOperation;
using HyperParallelDense::DenseMatrixAccess;
using HyperParallelDense::MatrixKOffset;
using HyperParallelDense::MatrixKState;
using HyperParallelDense::MatrixSegmentEnd;
#ifndef __DAV_C220_CUBE__
using HyperParallelDense::DenseCommonVector;
using HyperParallelDense::DenseGradientAdd;
using HyperParallelDense::DenseSwiGluForward;
using HyperParallelDense::DenseSwiGluGrad;
#endif
using Matrix = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using LeftTransposeMatrix = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using RightTransposeMatrix = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using DenseMatmul = matmul::Matmul<Matrix, Matrix, Matrix>;
using DenseLeftTransposeMatmul = matmul::Matmul<LeftTransposeMatrix, Matrix, Matrix>;
using DenseRightTransposeMatmul = matmul::Matmul<Matrix, RightTransposeMatrix, Matrix>;

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
__aicore__ inline void BackwardOrderedCopyA(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm, int row,
                                            int column, int use_m, int use_k, uint64_t tiling_ptr, uint64_t state) {
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
    const uint64_t position =
      static_cast<uint64_t>(row * tiling.baseM) * tiling.Ka + MatrixKOffset(offset, tiling.Ka, state);
    DataCopy(target[done * stride], source[position], copy);
    done += count;
  }
}

__aicore__ inline void BackwardOrderedCopyB(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm, int row,
                                            int column, int use_k, int use_n, uint64_t tiling_ptr, uint64_t state) {
  const auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(tiling_ptr));
  GlobalTensor<bfloat16_t> source;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(const_cast<__gm__ void *>(gm)));
  auto target = local.ReinterpretCast<bfloat16_t>();
  const uint32_t chunk = static_cast<uint32_t>(state);
  const uint32_t stride = (use_n + 15) / 16 * 16;
  // SDK 的回调保留逻辑 K/N 参数；转置 B 的 GM 与 L1 按物理 N/K 布局搬运。
  for (uint32_t done = 0; done < use_k;) {
    const uint32_t offset = row * tiling.baseK + done;
    const uint32_t remaining = chunk - offset % chunk;
    const uint32_t count = use_k - done < remaining ? use_k - done : remaining;
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = use_n;
    copy.dValue = count;
    copy.srcDValue = tiling.Kb;
    copy.dstNzC0Stride = stride;
    copy.dstNzNStride = 1;
    const uint64_t position =
      static_cast<uint64_t>(column * tiling.baseN) * tiling.Kb + MatrixKOffset(offset, tiling.Kb, state);
    DataCopy(target[done * stride], source[position], copy);
    done += count;
  }
}

using OrderedRightTransposeMatmul =
  matmul::Matmul<Matrix, RightTransposeMatrix, Matrix, Matrix, CFG_NORM,
                 AscendC::MatmulCallBackFunc<nullptr, BackwardOrderedCopyA, BackwardOrderedCopyB>>;

__aicore__ inline void OrderedCopyNormalB(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm, int row,
                                          int column, int use_k, int use_n, uint64_t tiling_ptr, uint64_t state) {
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
    const uint64_t position =
      static_cast<uint64_t>(MatrixKOffset(offset, tiling.Kb, state)) * tiling.N + column * tiling.baseN;
    DataCopy(target[done * 16], source[position], copy);
    done += count;
  }
}

__aicore__ inline void OrderedCopyTransposeA(const AscendC::LocalTensor<int8_t> &local, const __gm__ void *gm, int row,
                                             int column, int use_m, int use_k, uint64_t tiling_ptr, uint64_t state) {
  const auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(tiling_ptr));
  GlobalTensor<bfloat16_t> source;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(const_cast<__gm__ void *>(gm)));
  auto target = local.ReinterpretCast<bfloat16_t>();
  const uint32_t chunk = static_cast<uint32_t>(state);
  for (uint32_t done = 0; done < use_k;) {
    const uint32_t offset = column * tiling.baseK + done;
    const uint32_t remaining = chunk - offset % chunk;
    const uint32_t count = use_k - done < remaining ? use_k - done : remaining;
    AscendC::Nd2NzParams copy{};
    copy.ndNum = 1;
    copy.nValue = count;
    copy.dValue = use_m;
    copy.srcDValue = tiling.M;
    copy.dstNzC0Stride = (use_k + 15) / 16 * 16;
    copy.dstNzNStride = 1;
    const uint64_t position =
      static_cast<uint64_t>(MatrixKOffset(offset, tiling.Ka, state)) * tiling.M + row * tiling.baseM;
    DataCopy(target[done * 16], source[position], copy);
    done += count;
  }
}

using OrderedNormalMatmul =
  matmul::Matmul<Matrix, Matrix, Matrix, Matrix, CFG_NORM,
                 AscendC::MatmulCallBackFunc<nullptr, BackwardOrderedCopyA, OrderedCopyNormalB>>;
using OrderedLeftTransposeMatmul =
  matmul::Matmul<LeftTransposeMatrix, Matrix, Matrix, Matrix, CFG_NORM,
                 AscendC::MatmulCallBackFunc<nullptr, OrderedCopyTransposeA, OrderedCopyNormalB>>;
#endif

class DenseBackwardWorker {
 public:
  __aicore__ inline void Init(GM_ADDR values, GM_ADDR config, GM_ADDR tilings, GM_ADDR events, GM_ADDR ones,
                              GM_ADDR workspace) {
    header_ = LoadDescriptor(reinterpret_cast<const __gm__ DenseBackwardHeader *>(config));
    if (header_.magic != DENSE_BACKWARD_MAGIC || header_.version != DENSE_BACKWARD_VERSION ||
        header_.rows_per_tile == 0 || header_.cube_workers == 0 || header_.cube_workers > 24 ||
        header_.prefetch_tiles == 0 || header_.prefetch_tiles > 8 ||
        header_.operation_count > DENSE_BACKWARD_MAX_OPERATIONS || header_.value_count > DENSE_BACKWARD_MAX_VALUES) {
      Trap();
    }
    operations_ = reinterpret_cast<const __gm__ DenseBackwardOperation *>(config + sizeof(DenseBackwardHeader));
    dependencies_ = reinterpret_cast<const __gm__ DenseBackwardDependency *>(
      config + sizeof(DenseBackwardHeader) + header_.operation_count * sizeof(DenseBackwardOperation));
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
    const uint32_t window = header_.cube_workers * header_.prefetch_tiles;
    // 所有行窗口必须先于完整收缩阶段；否则 dW 等待未来的行生产者会形成死锁。
    for (uint32_t phase = 0; phase < 2; ++phase) {
      for (uint32_t base = 0; base < header_.phase_tiles[phase]; base += window) {
        for (uint32_t index = 0; index < header_.operation_count; ++index) {
          const auto operation = LoadDescriptor(&operations_[index]);
          if (operation.phase != phase || operation.column_tiles == 0) {
            continue;
          }
#ifdef __DAV_C220_CUBE__
          if (operation.kind != DENSE_MATMUL) {
            continue;
          }
#else
          if (operation.kind == DENSE_MATMUL) {
            continue;
          }
#endif
          Validate(operation);
          const uint32_t stop = operation.tile_count < base + window ? operation.tile_count : base + window;
          for (uint32_t slice = base * operation.column_tiles + cube; slice < stop * operation.column_tiles;
               slice += header_.cube_workers) {
            Execute(operation, index, slice / operation.column_tiles, slice % operation.column_tiles);
          }
        }
      }
    }
  }

 private:
  __aicore__ inline void Validate(const DenseBackwardOperation &operation) {
    if (operation.left >= header_.value_count || operation.right >= header_.value_count ||
        operation.output >= header_.value_count || operation.transpose_right > 1 ||
        (operation.kind == DENSE_MATMUL && operation.transpose_left && operation.transpose_right) ||
        (operation.transpose_output != 0 && operation.kind != DENSE_ZERO) || operation.transpose_left > 1 ||
        operation.kind > DENSE_SILU_GRAD ||
        (operation.kind == DENSE_SWIGLU && operation.contracted != operation.columns * 2) ||
        operation.columns_per_tile == 0 ||
        operation.column_tiles != (operation.columns + operation.columns_per_tile - 1) / operation.columns_per_tile ||
        operation.tile_count != (operation.rows + header_.rows_per_tile - 1) / header_.rows_per_tile ||
        operation.event_begin + operation.tile_count > header_.event_count ||
        operation.dependency_begin + operation.dependency_count > header_.dependency_count) {
      Trap();
    }
  }

  __aicore__ inline void Wait(uint32_t event, uint32_t count) {
    if (event >= header_.event_count || count == 0) {
      Trap();
    }
    const uint32_t position = event * DENSE_EVENT_STRIDE;
    const __gm__ volatile int32_t *ready = counters_.GetPhyAddr(position);
    while (true) {
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counters_[position]);
      PipeBarrier<PIPE_ALL>();
      if (*ready >= count) {
        return;
      }
    }
  }

  __aicore__ inline void WaitDependencies(const DenseBackwardOperation &operation, uint32_t tile) {
    for (uint32_t index = 0; index < operation.dependency_count; ++index) {
      const auto dependency = LoadDescriptor(&dependencies_[operation.dependency_begin + index]);
      if (dependency.row_local) {
        if (tile >= dependency.tile_count) {
          Trap();
        }
        Wait(dependency.event_begin + tile, dependency.signal_count);
      } else {
        for (uint32_t parent = 0; parent < dependency.tile_count; ++parent) {
          Wait(dependency.event_begin + parent, dependency.signal_count);
        }
      }
    }
  }

  template <HardEvent event>
  __aicore__ inline void SyncEvent(TPipe &pipe) {
    const auto id = static_cast<event_t>(pipe.FetchEventID(event));
    SetFlag<event>(id);
    WaitFlag<event>(id);
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
    SyncEvent<HardEvent::S_MTE2>(pipe);
    DataCopy(value, ones_, 8);
    SyncEvent<HardEvent::MTE2_S>(pipe);
    SetAtomicAdd<int32_t>();
    SyncEvent<HardEvent::S_MTE2>(pipe);
    DataCopy(counters_[event * DENSE_EVENT_STRIDE], value, 8);
    SyncEvent<HardEvent::MTE2_S>(pipe);
#else
    value.SetValue(0, 1);
    for (uint32_t index = 1; index < 8; ++index) {
      value.SetValue(index, 0);
    }
    SetAtomicAdd<int32_t>();
    SyncEvent<HardEvent::S_MTE3>(pipe);
    DataCopy(counters_[event * DENSE_EVENT_STRIDE], value, 8);
    SyncEvent<HardEvent::MTE3_S>(pipe);
#endif
    SetAtomicNone();
    buffer.FreeTensor(value);
    pipe.Destroy();
  }

  __aicore__ inline void Execute(const DenseBackwardOperation &operation, uint32_t index, uint32_t tile,
                                 uint32_t column_tile) {
    WaitDependencies(operation, tile);
    uint32_t first = tile * header_.rows_per_tile;
    uint32_t end = first + header_.rows_per_tile;
    end = end < operation.rows ? end : operation.rows;
    const uint32_t column = column_tile * operation.columns_per_tile;
    uint32_t column_end = column + operation.columns_per_tile;
    column_end = column_end < operation.columns ? column_end : operation.columns;
#ifdef __DAV_C220_CUBE__
    Matrix(operation, index, first, end - first, column, column_end);
#else
    const uint32_t middle = first + (end - first + 1) / 2;
    if (GetBlockIdx() % 2 == 0) {
      end = middle;
    } else {
      first = middle;
    }
    if (first < end) {
      if (operation.kind == DENSE_SWIGLU) {
        DenseSwiGluForward worker;
        worker.Process(reinterpret_cast<GM_ADDR>(values_[operation.left]),
                       reinterpret_cast<GM_ADDR>(values_[operation.output]), operation.columns, first, end, column,
                       column_end);
      } else if (operation.kind == DENSE_SWIGLU_GRAD) {
        DenseSwiGluGrad worker;
        worker.Process(
          reinterpret_cast<GM_ADDR>(values_[operation.left]), reinterpret_cast<GM_ADDR>(values_[operation.right]),
          reinterpret_cast<GM_ADDR>(values_[operation.output]), operation.columns, first, end, column, column_end);
      } else if (operation.kind >= DENSE_SILU) {
        DenseCommonVector worker;
        worker.Process(reinterpret_cast<GM_ADDR>(values_[operation.left]),
                       reinterpret_cast<GM_ADDR>(values_[operation.right]),
                       reinterpret_cast<GM_ADDR>(values_[operation.output]), operation.columns, first, end, column,
                       column_end, operation.kind);
      } else {
        DenseGradientAdd worker;
        worker.Process(reinterpret_cast<GM_ADDR>(values_[operation.left]),
                       reinterpret_cast<GM_ADDR>(values_[operation.right]),
                       reinterpret_cast<GM_ADDR>(values_[operation.output]), operation.columns, first, end, column,
                       column_end, operation.kind == DENSE_ZERO);
      }
    }
#endif
    Signal(operation.event_begin + tile);
  }

#ifdef __DAV_C220_CUBE__
  __aicore__ inline void Matrix(const DenseBackwardOperation &operation, uint32_t index, uint32_t first, uint32_t rows,
                                uint32_t first_column, uint32_t end_column) {
    GM_ADDR source = tilings_ + index * (sizeof(TCubeTiling) + sizeof(DenseMatrixAccess));
    auto tiling = LoadDescriptor(reinterpret_cast<const __gm__ TCubeTiling *>(source));
    const auto access =
      LoadDescriptor(reinterpret_cast<const __gm__ DenseMatrixAccess *>(source + sizeof(TCubeTiling)));
    const uint32_t extent = access.m_extent >= access.n_extent ? access.m_extent : access.n_extent;
    if (access.k_chunk == 0 || access.close_shift != 0 || extent < 4 ||
        extent / 2 > operation.contracted / access.k_chunk) {
      MatrixPart(operation, source, tiling, first, rows, first_column, end_column - first_column, 0);
      return;
    }
    const bool row_axis = access.m_extent >= access.n_extent;
    for (uint32_t row = first; row < first + rows;) {
      const uint32_t row_end = row_axis ? MatrixSegmentEnd(row, access.m_span, first + rows) : first + rows;
      for (uint32_t column = first_column; column < end_column;) {
        const uint32_t column_end = row_axis ? end_column : MatrixSegmentEnd(column, access.n_span, end_column);
        const uint64_t state =
          MatrixKState(access, operation.rows, operation.columns, operation.contracted, row, column);
        MatrixPart(operation, source, tiling, row, row_end - row, column, column_end - column, state);
        column = column_end;
      }
      row = row_end;
    }
  }

  __aicore__ inline void MatrixPart(const DenseBackwardOperation &operation, GM_ADDR source, TCubeTiling &tiling,
                                    uint32_t first, uint32_t rows, uint32_t column, uint32_t columns, uint64_t state) {
    if (operation.transpose_left) {
      if (state == 0) {
        MatrixSegment<DenseLeftTransposeMatmul>(operation, source, tiling, first, rows, column, columns, state);
      } else {
        MatrixSegment<OrderedLeftTransposeMatmul>(operation, source, tiling, first, rows, column, columns, state);
      }
    } else if (operation.transpose_right) {
      if (state == 0) {
        MatrixSegment<DenseRightTransposeMatmul>(operation, source, tiling, first, rows, column, columns, state);
      } else {
        MatrixSegment<OrderedRightTransposeMatmul>(operation, source, tiling, first, rows, column, columns, state);
      }
    } else if (state == 0) {
      MatrixSegment<DenseMatmul>(operation, source, tiling, first, rows, column, columns, state);
    } else {
      MatrixSegment<OrderedNormalMatmul>(operation, source, tiling, first, rows, column, columns, state);
    }
  }

  template <typename MatrixKernel>
  __aicore__ inline void MatrixSegment(const DenseBackwardOperation &operation, GM_ADDR source, TCubeTiling &tiling,
                                       uint32_t first, uint32_t rows, uint32_t column, uint32_t columns,
                                       uint64_t state) {
    TPipe pipe;
    MatrixKernel mm;
    REGIST_MATMUL_OBJ(&pipe, workspace_, mm, &tiling);
    GlobalTensor<bfloat16_t> left, right, output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.left]));
    right.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.right]));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(values_[operation.output]));
    mm.SetOrgShape(operation.rows, operation.columns, operation.contracted);
    mm.SetTail(rows, columns, operation.contracted);
    mm.SetUserDefInfo(reinterpret_cast<uint64_t>(source));
    mm.SetSelfDefineData(state);
    const uint64_t offset = operation.transpose_left ? first : static_cast<uint64_t>(first) * operation.contracted;
    mm.SetTensorA(left[offset], operation.transpose_left != 0);
    const uint64_t right_offset =
      operation.transpose_right ? static_cast<uint64_t>(column) * operation.contracted : column;
    mm.SetTensorB(right[right_offset], operation.transpose_right != 0);
    mm.IterateAll(output[static_cast<uint64_t>(first) * operation.columns + column]);
    mm.End();
  }
#endif

  DenseBackwardHeader header_{};
  const __gm__ DenseBackwardOperation *operations_ = nullptr;
  const __gm__ DenseBackwardDependency *dependencies_ = nullptr;
  const __gm__ uint64_t *values_ = nullptr;
  GlobalTensor<int32_t> counters_, ones_;
  GM_ADDR tilings_ = nullptr;
  GM_ADDR workspace_ = nullptr;
};
}  // namespace

extern "C" __global__ __aicore__ void hyper_parallel_dense_backward(GM_ADDR values, GM_ADDR config, GM_ADDR tilings,
                                                                    GM_ADDR events, GM_ADDR ones, GM_ADDR workspace) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  DenseBackwardWorker worker;
  worker.Init(values, config, tilings, events, ones, workspace);
  worker.Process();
}
