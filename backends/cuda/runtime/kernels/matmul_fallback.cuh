#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace intentir_cuda {

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int ROWS_PER_THREAD>
__device__ __forceinline__ void matmul_f32_fallback(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K,
    float* __restrict__ As,
    float* __restrict__ Bs) {
  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int col = (int)(blockIdx.x * BLOCK_N + tx);
  const int block_row = (int)(blockIdx.y * BLOCK_M);
  const int row0 = block_row + ty;

  float acc[ROWS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ROWS_PER_THREAD; ++i) acc[i] = 0.0f;

  for (int kt = 0; kt < K; kt += BLOCK_K) {
    // Cooperative load (guarded).
    if (tx < BLOCK_K) {
      #pragma unroll
      for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        const int r = ty + i * THREAD_M;
        if (r < BLOCK_M) {
          const int row = block_row + r;
          if (row < M && (kt + tx) < K)
            As[r * BLOCK_K + tx] = A[(size_t)row * (size_t)K + (size_t)(kt + tx)];
          else
            As[r * BLOCK_K + tx] = 0.0f;
        }
      }
    }
    if (ty < BLOCK_K) {
      if (col < N && (kt + ty) < K)
        Bs[ty * BLOCK_N + tx] = B[(size_t)(kt + ty) * (size_t)N + (size_t)col];
      else
        Bs[ty * BLOCK_N + tx] = 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int k0 = 0; k0 < BLOCK_K; ++k0) {
      const float b0 = Bs[k0 * BLOCK_N + tx];
      #pragma unroll
      for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        const int r = ty + i * THREAD_M;
        if (r < BLOCK_M) acc[i] = fmaf(As[r * BLOCK_K + k0], b0, acc[i]);
      }
    }
    __syncthreads();
  }

  if (col < N) {
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
      const int row = row0 + i * THREAD_M;
      if (row < M) C[(size_t)row * (size_t)N + (size_t)col] = acc[i];
    }
  }
}

}  // namespace intentir_cuda

