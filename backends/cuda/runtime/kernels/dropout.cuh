#pragma once

#include <stdint.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

template <int EPT, int N_ROUNDS, bool FULL_TILE>
__device__ __forceinline__ void dropout_f32(
    const float* __restrict__ X,
    float p,
    uint32_t seed_u32,
    float* __restrict__ Y,
    int64_t n_elements) {
  static_assert(EPT > 0 && EPT <= 8, "dropout_f32 supports EPT in [1,8]");
  static_assert(N_ROUNDS > 0 && N_ROUNDS <= 10, "dropout_f32 supports N_ROUNDS in [1,10]");
  const int tid = (int)threadIdx.x;
  const int64_t stride = (int64_t)blockDim.x;
  const int64_t base = (int64_t)blockIdx.x * stride * (int64_t)EPT + (int64_t)tid;
  const uint64_t seed = (uint64_t)seed_u32;

  if (p <= 0.0f) {
    int64_t i = base;
    #pragma unroll
    for (int e = 0; e < EPT; ++e, i += stride) {
      if constexpr (!FULL_TILE) {
        if (i >= n_elements) break;
      }
      Y[i] = intentir_ldg_f32(X + i);
    }
    return;
  }
  if (p >= 1.0f) {
    int64_t i = base;
    #pragma unroll
    for (int e = 0; e < EPT; ++e, i += stride) {
      if constexpr (!FULL_TILE) {
        if (i >= n_elements) break;
      }
      Y[i] = 0.0f;
    }
    return;
  }

  const float inv_keep = __fdividef(1.0f, (1.0f - p));
  // Compare in the integer domain to avoid per-element int->float conversion.
  // This matches the interpreter mapping:
  //   xi = (int32_t)u; xi ^= (xi >> 31); r = (float)xi * 2^-31; keep = r > p
  // Rearranged:
  //   keep iff xi > p * 2^31
  const uint32_t keep_thresh = (uint32_t)(p * 2147483648.0f);  // 2^31
  int64_t i = base;
  #pragma unroll
  for (int e = 0; e < EPT; ++e, i += stride) {
    if constexpr (!FULL_TILE) {
      if (i >= n_elements) break;
    }
    const float x = intentir_ldg_f32(X + i);
    const uint32_t ctr = (uint32_t)i;
    const uint32_t rnd_u32 = intentir_philox_randint_u32_rounds<N_ROUNDS>(seed, ctr);
    int32_t xi = (int32_t)rnd_u32;
    xi ^= (xi >> 31);
    const bool keep = ((uint32_t)xi > keep_thresh);
    Y[i] = keep ? (x * inv_keep) : 0.0f;
  }
}

template <int EPT, int N_ROUNDS, bool FULL_TILE>
__device__ __forceinline__ void dropout_f32(
    const float* __restrict__ X,
    const float* p_ptr,
    const int* seed_ptr,
    float* __restrict__ Y,
    int64_t n_elements) {
  const float p = p_ptr ? p_ptr[0] : 0.0f;
  const uint32_t seed_u32 = (uint32_t)(seed_ptr ? seed_ptr[0] : 0);
  dropout_f32<EPT, N_ROUNDS, FULL_TILE>(X, p, seed_u32, Y, n_elements);
}

}  // namespace intentir_cuda
