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
__device__ __forceinline__ void dropout_f32_vec4(
    const float* __restrict__ X,
    float p,
    uint32_t seed_u32,
    float* __restrict__ Y,
    int64_t n_elements) {
  static_assert(EPT > 0 && EPT <= 8, "dropout_f32_vec4 supports EPT in [1,8]");
  static_assert((EPT % 4) == 0, "dropout_f32_vec4 requires EPT a multiple of 4");
  static_assert(N_ROUNDS > 0 && N_ROUNDS <= 10, "dropout_f32_vec4 supports N_ROUNDS in [1,10]");
  const int tid = (int)threadIdx.x;
  const int64_t threads = (int64_t)blockDim.x;
  const int64_t tile = threads * (int64_t)EPT;
  const int64_t base = (int64_t)blockIdx.x * tile + (int64_t)tid * (int64_t)EPT;
  const uint64_t seed = (uint64_t)seed_u32;
  const bool aligned = (((uintptr_t)(X + base) & 15u) == 0u) && (((uintptr_t)(Y + base) & 15u) == 0u);

  if (p <= 0.0f) {
    #pragma unroll
    for (int e = 0; e < EPT; e += 4) {
      const int64_t i = base + (int64_t)e;
      if constexpr (!FULL_TILE) {
        if (i >= n_elements) break;
      }
      if (aligned && (FULL_TILE || (i + 3) < n_elements)) {
        const float4 x4 = *reinterpret_cast<const float4*>(X + i);
        *reinterpret_cast<float4*>(Y + i) = x4;
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int64_t idx = i + (int64_t)j;
          if constexpr (!FULL_TILE) {
            if (idx >= n_elements) break;
          }
          Y[idx] = intentir_ldg_f32(X + idx);
        }
      }
    }
    return;
  }
  if (p >= 1.0f) {
    const float4 z4 = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int e = 0; e < EPT; e += 4) {
      const int64_t i = base + (int64_t)e;
      if constexpr (!FULL_TILE) {
        if (i >= n_elements) break;
      }
      if (aligned && (FULL_TILE || (i + 3) < n_elements)) {
        *reinterpret_cast<float4*>(Y + i) = z4;
      } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int64_t idx = i + (int64_t)j;
          if constexpr (!FULL_TILE) {
            if (idx >= n_elements) break;
          }
          Y[idx] = 0.0f;
        }
      }
    }
    return;
  }

  const float inv_keep = __fdividef(1.0f, (1.0f - p));
  const uint32_t keep_thresh = (uint32_t)(p * 2147483648.0f);  // 2^31

  #pragma unroll
  for (int e = 0; e < EPT; e += 4) {
    const int64_t i = base + (int64_t)e;
    if constexpr (!FULL_TILE) {
      if (i >= n_elements) break;
    }
    const uint32_t ctr = (uint32_t)((uint64_t)i >> 2);
    const intentir_uint4 rnd = intentir_philox_rand4_u32_rounds<N_ROUNDS>(seed, ctr);

    if (aligned && (FULL_TILE || (i + 3) < n_elements)) {
      const float4 x4 = *reinterpret_cast<const float4*>(X + i);
      float4 y4;
      int32_t xi0 = (int32_t)rnd.x;
      int32_t xi1 = (int32_t)rnd.y;
      int32_t xi2 = (int32_t)rnd.z;
      int32_t xi3 = (int32_t)rnd.w;
      xi0 ^= (xi0 >> 31);
      xi1 ^= (xi1 >> 31);
      xi2 ^= (xi2 >> 31);
      xi3 ^= (xi3 >> 31);
      y4.x = ((uint32_t)xi0 > keep_thresh) ? (x4.x * inv_keep) : 0.0f;
      y4.y = ((uint32_t)xi1 > keep_thresh) ? (x4.y * inv_keep) : 0.0f;
      y4.z = ((uint32_t)xi2 > keep_thresh) ? (x4.z * inv_keep) : 0.0f;
      y4.w = ((uint32_t)xi3 > keep_thresh) ? (x4.w * inv_keep) : 0.0f;
      *reinterpret_cast<float4*>(Y + i) = y4;
    } else {
      const uint32_t rv[4] = {rnd.x, rnd.y, rnd.z, rnd.w};
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int64_t idx = i + (int64_t)j;
        if constexpr (!FULL_TILE) {
          if (idx >= n_elements) break;
        }
        const float x = intentir_ldg_f32(X + idx);
        int32_t xi = (int32_t)rv[j];
        xi ^= (xi >> 31);
        const bool keep = ((uint32_t)xi > keep_thresh);
        Y[idx] = keep ? (x * inv_keep) : 0.0f;
      }
    }
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

template <int EPT, int N_ROUNDS, bool FULL_TILE>
__device__ __forceinline__ void dropout_f32_vec4(
    const float* __restrict__ X,
    const float* p_ptr,
    const int* seed_ptr,
    float* __restrict__ Y,
    int64_t n_elements) {
  const float p = p_ptr ? p_ptr[0] : 0.0f;
  const uint32_t seed_u32 = (uint32_t)(seed_ptr ? seed_ptr[0] : 0);
  dropout_f32_vec4<EPT, N_ROUNDS, FULL_TILE>(X, p, seed_u32, Y, n_elements);
}

}  // namespace intentir_cuda
