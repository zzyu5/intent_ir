#pragma once

#include <math.h>

#include "intentir_cuda_ops.cuh"
#include "kernels/reduce.cuh"

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

__device__ __forceinline__ float warp_allreduce_max(float v) {
  v = warp_reduce_max(v);
  return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float warp_allreduce_sum(float v) {
  v = warp_reduce_sum(v);
  return __shfl_sync(0xffffffff, v, 0);
}

template <int BLOCK_THREADS>
struct SoftmaxBlockReduceF32 {
  static_assert(BLOCK_THREADS > 0, "SoftmaxBlockReduceF32: BLOCK_THREADS must be > 0");
  static_assert((BLOCK_THREADS % 32) == 0, "SoftmaxBlockReduceF32: BLOCK_THREADS must be a multiple of 32");
  static constexpr int WARPS = BLOCK_THREADS / 32;
  float warp_out[WARPS];
  float out;
};

template <int BLOCK_THREADS>
__device__ __forceinline__ float softmax_block_allreduce_max(float v, SoftmaxBlockReduceF32<BLOCK_THREADS>* st) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = warp_reduce_max(v);
  if (lane == 0) st->warp_out[warp] = v;
  __syncthreads();

  float out = -INFINITY;
  if (warp == 0) {
    out = (lane < SoftmaxBlockReduceF32<BLOCK_THREADS>::WARPS) ? st->warp_out[lane] : -INFINITY;
    out = warp_reduce_max(out);
    if (lane == 0) st->out = out;
  }
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float softmax_block_allreduce_sum(float v, SoftmaxBlockReduceF32<BLOCK_THREADS>* st) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) st->warp_out[warp] = v;
  __syncthreads();

  float out = 0.0f;
  if (warp == 0) {
    out = (lane < SoftmaxBlockReduceF32<BLOCK_THREADS>::WARPS) ? st->warp_out[lane] : 0.0f;
    out = warp_reduce_sum(out);
    if (lane == 0) st->out = out;
  }
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS, int EPT, bool USE_WARP_REDUCE = false>
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
  if constexpr (USE_WARP_REDUCE) {
    __shared__ SoftmaxBlockReduceF32<BLOCK_THREADS> red;
    const float mx = softmax_block_allreduce_max<BLOCK_THREADS>(tmax, &red);

    float tsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int c = tid + i * BLOCK_THREADS;
      (void)c;
      const float e = __expf(expv[i] - mx);
      expv[i] = e;
      tsum += e;
    }
    const float sum = softmax_block_allreduce_sum<BLOCK_THREADS>(tsum, &red);
    const float inv = __fdividef(1.0f, sum);
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int c = tid + i * BLOCK_THREADS;
      if (c < C) out_row[(size_t)c] = expv[i] * inv;
    }
  } else {
    __shared__ BlockAllreduceF32<BLOCK_THREADS> red;
    const float mx = block_allreduce_max<BLOCK_THREADS>(tmax, &red);

    float tsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int c = tid + i * BLOCK_THREADS;
      (void)c;
      const float e = __expf(expv[i] - mx);
      expv[i] = e;
      tsum += e;
    }
    const float sum = block_allreduce_sum<BLOCK_THREADS>(tsum, &red);
    const float inv = __fdividef(1.0f, sum);
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int c = tid + i * BLOCK_THREADS;
      if (c < C) out_row[(size_t)c] = expv[i] * inv;
    }
  }
}

template <int BLOCK_THREADS, int TILES, bool USE_WARP_REDUCE = false>
__device__ __forceinline__ void softmax_2d_last_f32_vec4(const float* __restrict__ inp, float* __restrict__ out, int R, int C) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "softmax vec4 block size must be in (0,1024]");
  static_assert((BLOCK_THREADS % 32) == 0, "softmax vec4 block must be a multiple of 32 threads");
  static_assert(TILES > 0 && TILES <= 8, "softmax vec4 supports 1..8 tiles");
  constexpr int VEC = 4;
  const int r = (int)blockIdx.x;
  if (r >= R) return;
  if (C > 1024) return;
  const float* __restrict__ in_row = inp + (size_t)r * (size_t)C;
  float* __restrict__ out_row = out + (size_t)r * (size_t)C;
  const bool aligned = (((uintptr_t)in_row & 15u) == 0u) && (((uintptr_t)out_row & 15u) == 0u);
  const int tid = (int)threadIdx.x;

  float4 vals[TILES];
  float tmax = -INFINITY;
  #pragma unroll
  for (int t = 0; t < TILES; ++t) {
    const int base = t * (BLOCK_THREADS * VEC) + tid * VEC;
    float4 v;
    v.x = -INFINITY;
    v.y = -INFINITY;
    v.z = -INFINITY;
    v.w = -INFINITY;
    if (base < C) {
      if (aligned && (base + (VEC - 1) < C)) {
        v = *reinterpret_cast<const float4*>(in_row + (size_t)base);
      } else {
        if (base + 0 < C) v.x = intentir_ldg_f32(in_row + (size_t)(base + 0));
        if (base + 1 < C) v.y = intentir_ldg_f32(in_row + (size_t)(base + 1));
        if (base + 2 < C) v.z = intentir_ldg_f32(in_row + (size_t)(base + 2));
        if (base + 3 < C) v.w = intentir_ldg_f32(in_row + (size_t)(base + 3));
      }
    }
    vals[t] = v;
    tmax = fmaxf(tmax, v.x);
    tmax = fmaxf(tmax, v.y);
    tmax = fmaxf(tmax, v.z);
    tmax = fmaxf(tmax, v.w);
  }
  if constexpr (USE_WARP_REDUCE) {
    __shared__ SoftmaxBlockReduceF32<BLOCK_THREADS> red;
    const float mx = softmax_block_allreduce_max<BLOCK_THREADS>(tmax, &red);

    float tsum = 0.0f;
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
      float4 v = vals[t];
      v.x = __expf(v.x - mx);
      v.y = __expf(v.y - mx);
      v.z = __expf(v.z - mx);
      v.w = __expf(v.w - mx);
      vals[t] = v;
      tsum += (v.x + v.y + v.z + v.w);
    }
    const float sum = softmax_block_allreduce_sum<BLOCK_THREADS>(tsum, &red);
    const float inv = __fdividef(1.0f, sum);

    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
      const int base = t * (BLOCK_THREADS * VEC) + tid * VEC;
      if (base >= C) continue;
      float4 v = vals[t];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      if (aligned && (base + (VEC - 1) < C)) {
        *reinterpret_cast<float4*>(out_row + (size_t)base) = v;
      } else {
        if (base + 0 < C) out_row[(size_t)(base + 0)] = v.x;
        if (base + 1 < C) out_row[(size_t)(base + 1)] = v.y;
        if (base + 2 < C) out_row[(size_t)(base + 2)] = v.z;
        if (base + 3 < C) out_row[(size_t)(base + 3)] = v.w;
      }
    }
    return;
  } else {
    __shared__ BlockAllreduceF32<BLOCK_THREADS> red;
    const float mx = block_allreduce_max<BLOCK_THREADS>(tmax, &red);

    float tsum = 0.0f;
    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
      float4 v = vals[t];
      v.x = __expf(v.x - mx);
      v.y = __expf(v.y - mx);
      v.z = __expf(v.z - mx);
      v.w = __expf(v.w - mx);
      vals[t] = v;
      tsum += (v.x + v.y + v.z + v.w);
    }
    const float sum = block_allreduce_sum<BLOCK_THREADS>(tsum, &red);
    const float inv = __fdividef(1.0f, sum);

    #pragma unroll
    for (int t = 0; t < TILES; ++t) {
      const int base = t * (BLOCK_THREADS * VEC) + tid * VEC;
      if (base >= C) continue;
      float4 v = vals[t];
      v.x *= inv;
      v.y *= inv;
      v.z *= inv;
      v.w *= inv;
      if (aligned && (base + (VEC - 1) < C)) {
        *reinterpret_cast<float4*>(out_row + (size_t)base) = v;
      } else {
        if (base + 0 < C) out_row[(size_t)(base + 0)] = v.x;
        if (base + 1 < C) out_row[(size_t)(base + 1)] = v.y;
        if (base + 2 < C) out_row[(size_t)(base + 2)] = v.z;
        if (base + 3 < C) out_row[(size_t)(base + 3)] = v.w;
      }
    }
    return;
  }

  // Unreachable.
}

template <int WARPS_PER_BLOCK>
__device__ __forceinline__ void softmax_2d_last_f32_warp4(const float* __restrict__ inp, float* __restrict__ out, int R, int C) {
  static_assert(WARPS_PER_BLOCK > 0 && WARPS_PER_BLOCK <= 8, "softmax warp4 supports 1..8 warps per block");
  constexpr int VEC = 4;
  constexpr int MAX_C = 1024;
  __shared__ __align__(16) float smem[WARPS_PER_BLOCK * MAX_C];

  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (warp >= WARPS_PER_BLOCK) return;

  const int r = (int)blockIdx.x * WARPS_PER_BLOCK + warp;
  if (r >= R) return;
  if (C > MAX_C) return;

  float* __restrict__ buf = smem + (int)warp * MAX_C;
  const float* __restrict__ in_row = inp + (size_t)r * (size_t)C;
  float* __restrict__ out_row = out + (size_t)r * (size_t)C;

  const int shift_in = (int)(((16u - (unsigned)((uintptr_t)in_row & 15u)) & 15u) >> 2);   // 0..3 floats
  const int shift_out = (int)(((16u - (unsigned)((uintptr_t)out_row & 15u)) & 15u) >> 2);  // 0..3 floats
  const int shift = (shift_in < C) ? shift_in : C;
  const int len0 = C - shift;  // elements in the aligned suffix [shift, C)

  float tmax = -INFINITY;
  // Rotate the row by `shift` (0..3) so the bulk begins at a 16B boundary:
  //   - buf[0..len0)      <- in_row[shift..C)
  //   - buf[len0..C)      <- in_row[0..shift)
  // This keeps buf float4-aligned while allowing aligned float4 loads for the bulk.
  #pragma unroll 1
  for (int base = lane * VEC; base < len0; base += 32 * VEC) {
    const int in_base = shift + base;
    if (base + (VEC - 1) < len0) {
      const float4 x = *reinterpret_cast<const float4*>(in_row + (size_t)in_base);
      *reinterpret_cast<float4*>(buf + (size_t)base) = x;
      tmax = fmaxf(tmax, x.x);
      tmax = fmaxf(tmax, x.y);
      tmax = fmaxf(tmax, x.z);
      tmax = fmaxf(tmax, x.w);
    } else {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const int c = base + j;
        if (c < len0) {
          const float v = intentir_ldg_f32(in_row + (size_t)(shift + c));
          buf[(size_t)c] = v;
          tmax = fmaxf(tmax, v);
        }
      }
    }
  }
  if (lane < shift) {
    const int c = lane;
    const int buf_idx = len0 + c;
    if (buf_idx < C) {
      const float v = intentir_ldg_f32(in_row + (size_t)c);
      buf[(size_t)buf_idx] = v;
      tmax = fmaxf(tmax, v);
    }
  }
  const float mx = warp_allreduce_max(tmax);

  float tsum = 0.0f;
  // The shared buffer is always 16B-aligned; we can safely vectorize across it.
  #pragma unroll 1
  for (int base = lane * VEC; base < C; base += 32 * VEC) {
    if (base + (VEC - 1) < C) {
      const float4 v = *reinterpret_cast<const float4*>(buf + (size_t)base);
      float4 e;
      e.x = __expf(v.x - mx);
      e.y = __expf(v.y - mx);
      e.z = __expf(v.z - mx);
      e.w = __expf(v.w - mx);
      *reinterpret_cast<float4*>(buf + (size_t)base) = e;
      tsum += (e.x + e.y + e.z + e.w);
    } else {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const int c = base + j;
        if (c < C) {
          const float v = buf[(size_t)c];
          const float e = __expf(v - mx);
          buf[(size_t)c] = e;
          tsum += e;
        }
      }
    }
  }
  const float sum = warp_allreduce_sum(tsum);
  const float inv = __fdividef(1.0f, sum);

  // Store: inverse-rotate back to the original order.
  if (shift_out == shift) {
    #pragma unroll 1
    for (int base = lane * VEC; base < len0; base += 32 * VEC) {
      const int out_base = shift + base;
      if (base + (VEC - 1) < len0) {
        float4 e = *reinterpret_cast<const float4*>(buf + (size_t)base);
        e.x *= inv;
        e.y *= inv;
        e.z *= inv;
        e.w *= inv;
        *reinterpret_cast<float4*>(out_row + (size_t)out_base) = e;
      } else {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
          const int c = base + j;
          if (c < len0) out_row[(size_t)(shift + c)] = buf[(size_t)c] * inv;
        }
      }
    }
    if (lane < shift) {
      const int c = lane;
      const int buf_idx = len0 + c;
      if (buf_idx < C) out_row[(size_t)c] = buf[(size_t)buf_idx] * inv;
    }
  } else {
    #pragma unroll 1
    for (int base = lane * VEC; base < len0; base += 32 * VEC) {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const int c = base + j;
        if (c < len0) out_row[(size_t)(shift + c)] = buf[(size_t)c] * inv;
      }
    }
    if (lane < shift) {
      const int c = lane;
      const int buf_idx = len0 + c;
      if (buf_idx < C) out_row[(size_t)c] = buf[(size_t)buf_idx] * inv;
    }
  }
}

}  // namespace intentir_cuda
