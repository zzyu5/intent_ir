#pragma once

#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "kernels/reduce.cuh"

namespace intentir_cuda {

template <int BLOCK_THREADS>
__device__ __forceinline__ void layernorm_2d_f32(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const float* __restrict__ W,
    const float* __restrict__ B,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    int M,
    int N,
    float eps) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "layernorm block size must be in (0,1024]");

  const int m = (int)blockIdx.x;
  if (m >= M) return;

  __shared__ BlockAllreduceF32<BLOCK_THREADS> red;
  const float* xrow = X + (size_t)m * (size_t)N;
  float* yrow = Y + (size_t)m * (size_t)N;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const bool aligned = ((((uintptr_t)xrow) & 15u) == 0u) && ((((uintptr_t)yrow) & 15u) == 0u) && ((((uintptr_t)W) & 15u) == 0u) &&
                       ((((uintptr_t)B) & 15u) == 0u) && ((N & 3) == 0);
  if (aligned) {
    const int n4 = N >> 2;
    const float4* __restrict__ x4 = (const float4* __restrict__)xrow;
    float tsum4 = 0.0f;
    for (int i = (int)threadIdx.x; i < n4; i += BLOCK_THREADS) {
      const float4 v = x4[i];
      tsum4 += ((v.x + v.y) + (v.z + v.w));
    }
    const float mean = block_allreduce_sum<BLOCK_THREADS>(tsum4, &red) / (float)N;

    float tsq4 = 0.0f;
    for (int i = (int)threadIdx.x; i < n4; i += BLOCK_THREADS) {
      const float4 v = x4[i];
      const float c0 = v.x - mean;
      const float c1 = v.y - mean;
      const float c2 = v.z - mean;
      const float c3 = v.w - mean;
      tsq4 += ((c0 * c0 + c1 * c1) + (c2 * c2 + c3 * c3));
    }
    const float var = block_allreduce_sum<BLOCK_THREADS>(tsq4, &red) / (float)N;
    const float rstd = rsqrtf(var + eps);
    if ((int)threadIdx.x == 0) {
      Mean[m] = mean;
      Rstd[m] = rstd;
    }

    float4* __restrict__ y4 = (float4* __restrict__)yrow;
    const float4* __restrict__ w4 = (const float4* __restrict__)W;
    const float4* __restrict__ b4 = (const float4* __restrict__)B;
    for (int i = (int)threadIdx.x; i < n4; i += BLOCK_THREADS) {
      const float4 x = x4[i];
      const float4 w = w4[i];
      const float4 b = b4[i];
      float4 y;
      y.x = ((x.x - mean) * rstd) * w.x + b.x;
      y.y = ((x.y - mean) * rstd) * w.y + b.y;
      y.z = ((x.z - mean) * rstd) * w.z + b.z;
      y.w = ((x.w - mean) * rstd) * w.w + b.w;
      y4[i] = y;
    }
    return;
  }
#endif

  float tsum = 0.0f;
  for (int n = (int)threadIdx.x; n < N; n += BLOCK_THREADS) {
    tsum += xrow[n];
  }
  const float mean = block_allreduce_sum<BLOCK_THREADS>(tsum, &red) / (float)N;

  float tsq = 0.0f;
  for (int n = (int)threadIdx.x; n < N; n += BLOCK_THREADS) {
    float c = xrow[n] - mean;
    tsq += c * c;
  }
  const float var = block_allreduce_sum<BLOCK_THREADS>(tsq, &red) / (float)N;
  const float rstd = rsqrtf(var + eps);
  if ((int)threadIdx.x == 0) {
    Mean[m] = mean;
    Rstd[m] = rstd;
  }
  for (int n = (int)threadIdx.x; n < N; n += BLOCK_THREADS) {
    float c = xrow[n] - mean;
    yrow[n] = (c * rstd) * W[n] + B[n];
  }
}

}  // namespace intentir_cuda
