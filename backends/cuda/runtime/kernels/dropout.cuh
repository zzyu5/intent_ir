#pragma once

#include <stdint.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

template <int EPT, int N_ROUNDS>
__device__ __forceinline__ void dropout_f32(
    const float* __restrict__ X,
    const float* p_ptr,
    const int* seed_ptr,
    float* __restrict__ Y,
    int64_t n_elements) {
  static_assert(EPT > 0 && EPT <= 8, "dropout_f32 supports EPT in [1,8]");
  static_assert(N_ROUNDS > 0 && N_ROUNDS <= 10, "dropout_f32 supports N_ROUNDS in [1,10]");
  const int tid = (int)threadIdx.x;
  const int64_t base = (int64_t)blockIdx.x * (int64_t)blockDim.x * (int64_t)EPT + (int64_t)tid;
  // Loading these scalars per-thread avoids an unconditional __syncthreads().
  const float p = p_ptr ? p_ptr[0] : 0.0f;
  const uint64_t seed = (uint64_t)(seed_ptr ? (uint32_t)seed_ptr[0] : 0u);

  if (p <= 0.0f) {
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
      const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
      if (i >= n_elements) break;
      Y[i] = intentir_ldg_f32(X + i);
    }
    return;
  }
  if (p >= 1.0f) {
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
      const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
      if (i >= n_elements) break;
      Y[i] = 0.0f;
    }
    return;
  }

  const float inv_keep = __fdividef(1.0f, (1.0f - p));
  #pragma unroll
  for (int e = 0; e < EPT; ++e) {
    const int64_t i = base + (int64_t)e * (int64_t)blockDim.x;
    if (i >= n_elements) break;
    const float x = intentir_ldg_f32(X + i);
    const uint32_t ctr = (uint32_t)i;
    const float r = intentir_uint_to_uniform_float_u32(intentir_philox_randint_u32_rounds<N_ROUNDS>(seed, ctr));
    Y[i] = (r > p) ? (x * inv_keep) : 0.0f;
  }
}

}  // namespace intentir_cuda

