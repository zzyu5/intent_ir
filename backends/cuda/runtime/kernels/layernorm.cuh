#pragma once

#include <stddef.h>
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
  float tsum = 0.0f;
  const float* xrow = X + (size_t)m * (size_t)N;
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
  float* yrow = Y + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += BLOCK_THREADS) {
    float c = xrow[n] - mean;
    yrow[n] = (c * rstd) * W[n] + B[n];
  }
}

}  // namespace intentir_cuda
