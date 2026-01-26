#pragma once

#include <math.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  }
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, off);
  }
  return v;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_max(float v) {
  __shared__ float shared[32];
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  constexpr int NUM_WARPS = (BLOCK_THREADS + 31) >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (warp == 0) ? ((lane < NUM_WARPS) ? shared[lane] : -INFINITY) : -INFINITY;
  if (warp == 0) v = warp_reduce_max(v);
  if ((int)threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_sum(float v) {
  __shared__ float shared[32];
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  constexpr int NUM_WARPS = (BLOCK_THREADS + 31) >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = (warp == 0) ? ((lane < NUM_WARPS) ? shared[lane] : 0.0f) : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  if ((int)threadIdx.x == 0) shared[0] = v;
  __syncthreads();
  return shared[0];
}

template <int BLOCK_THREADS, int EPT>
__device__ __forceinline__ void softmax_2d_last_f32(const float* __restrict__ inp, float* __restrict__ out, int R, int C) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "softmax block size must be in (0,1024]");
  static_assert((BLOCK_THREADS % 32) == 0, "softmax block must be a multiple of 32 threads");
  static_assert(EPT > 0 && EPT <= 32, "softmax EPT must be in (0,32]");
  const int r = (int)blockIdx.x;
  if (r >= R) return;
  const float* __restrict__ in_row = inp + (size_t)r * (size_t)C;
  float* __restrict__ out_row = out + (size_t)r * (size_t)C;
  const int tid = (int)threadIdx.x;

  float tmax = -INFINITY;
  float expv[EPT];
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    float v = -INFINITY;
    if (c < C) v = intentir_ldg_f32(in_row + (size_t)c);
    expv[i] = v;
    tmax = fmaxf(tmax, v);
  }
  const float mx = block_allreduce_max<BLOCK_THREADS>(tmax);

  float tsum = 0.0f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    float e = 0.0f;
    if (c < C) {
      e = __expf(expv[i] - mx);
      tsum += e;
    }
    expv[i] = e;
  }
  const float sum = block_allreduce_sum<BLOCK_THREADS>(tsum);
  const float inv = __fdividef(1.0f, sum);
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    if (c < C) {
      out_row[(size_t)c] = expv[i] * inv;
    }
  }
}

}  // namespace intentir_cuda

