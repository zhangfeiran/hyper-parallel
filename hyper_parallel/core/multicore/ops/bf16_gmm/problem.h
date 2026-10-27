// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_GMM_PROBLEM_H_
#define HYPER_MULTICORE_BF16_GMM_PROBLEM_H_
#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define MULTICORE_BF16_INLINE __aicore__ inline
#else
#define MULTICORE_BF16_INLINE inline
#endif

namespace MulticoreBf16 {
struct Problem {
  uint64_t m, n, k;
  uint64_t a, b, c;
};

MULTICORE_BF16_INLINE Problem MakeProblem(uint64_t m, uint64_t n, uint64_t k, uint64_t expert,
                                        uint64_t start, uint64_t rows, bool trans_a) {
  if (trans_a) return {m, n, rows, start * m, start * n, expert * m * n};
  return {rows, n, k, start * k, expert * k * n, start * n};
}
}  // namespace MulticoreBf16
#undef MULTICORE_BF16_INLINE
#endif
