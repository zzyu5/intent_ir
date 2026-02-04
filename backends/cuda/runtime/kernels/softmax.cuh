#pragma once

#include <math.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

// Softmax uses exp heavily; allow switching between exp and exp2. We keep this
// local to softmax to avoid changing semantics elsewhere.
template <bool USE_EXP2>
__device__ __forceinline__ float softmax_fast_exp(float x) {
  if constexpr (USE_EXP2) {
    // exp(x) = exp2(x * log2(e))
    float y;
    const float a = x * 1.4426950408889634f;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(a));
    return y;
  } else {
    return __expf(x);
  }
}

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
__device__ __forceinline__ float block_allreduce_max_f32(float v) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "block_allreduce_max_f32 supports BLOCK_THREADS in (0,1024]");
  static_assert((BLOCK_THREADS % 32) == 0, "block_allreduce_max_f32 requires BLOCK_THREADS a multiple of 32");
  constexpr int WARPS = BLOCK_THREADS / 32;
  __shared__ float warp_out[WARPS];
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float w = warp_reduce_max(v);
  if (lane == 0) warp_out[warp] = w;
  __syncthreads();
  float out = -INFINITY;
  if (warp == 0) {
    out = (lane < WARPS) ? warp_out[lane] : -INFINITY;
    out = warp_reduce_max(out);
    if (lane == 0) warp_out[0] = out;
  }
  __syncthreads();
  return warp_out[0];
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_sum_f32(float v) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "block_allreduce_sum_f32 supports BLOCK_THREADS in (0,1024]");
  static_assert((BLOCK_THREADS % 32) == 0, "block_allreduce_sum_f32 requires BLOCK_THREADS a multiple of 32");
  constexpr int WARPS = BLOCK_THREADS / 32;
  __shared__ float warp_out[WARPS];
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float w = warp_reduce_sum(v);
  if (lane == 0) warp_out[warp] = w;
  __syncthreads();
  float out = 0.0f;
  if (warp == 0) {
    out = (lane < WARPS) ? warp_out[lane] : 0.0f;
    out = warp_reduce_sum(out);
    if (lane == 0) warp_out[0] = out;
  }
  __syncthreads();
  return warp_out[0];
}

template <int BLOCK_THREADS, int EPT, bool USE_EXP2, bool FULL_TILE = false>
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
    if constexpr (FULL_TILE) {
      v = intentir_ldg_f32(in_row + (size_t)c);
    } else {
      if (c < C) v = intentir_ldg_f32(in_row + (size_t)c);
    }
    expv[i] = v;
    tmax = fmaxf(tmax, v);
  }
  const float mx = block_allreduce_max_f32<BLOCK_THREADS>(tmax);

  float tsum = 0.0f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const float e = softmax_fast_exp<USE_EXP2>(expv[i] - mx);
    expv[i] = e;
    tsum += e;
  }
  const float sum = block_allreduce_sum_f32<BLOCK_THREADS>(tsum);
  const float inv = __fdividef(1.0f, sum);
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    if constexpr (FULL_TILE) {
      out_row[(size_t)c] = expv[i] * inv;
    } else {
      if (c < C) {
        out_row[(size_t)c] = expv[i] * inv;
      }
    }
  }
}

// Contract-gated fast path: when C is a compile-time constant but not a multiple of
// BLOCK_THREADS, we can still eliminate per-element bounds checks by distributing the
// remainder across the first (C % BLOCK_THREADS) lanes. This keeps the inner loop
// branch-free and makes the "contract enables aggressive specialization" story
// measurable (contract_off disables this variant).
template <int BLOCK_THREADS, int C, bool USE_EXP2>
__device__ __forceinline__ void softmax_2d_last_f32_ragged(const float* __restrict__ inp, float* __restrict__ out, int R) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "softmax ragged block size must be in (0,1024]");
  static_assert((BLOCK_THREADS % 32) == 0, "softmax ragged block must be a multiple of 32 threads");
  static_assert(C > 0 && C <= 1024, "softmax ragged supports C in (0,1024]");
  constexpr int Q = C / BLOCK_THREADS;
  constexpr int REM = C - Q * BLOCK_THREADS;
  static_assert(Q >= 0, "softmax ragged requires non-negative Q");
  static_assert(Q <= 32, "softmax ragged requires Q<=32");

  const int r = (int)blockIdx.x;
  if (r >= R) return;
  const float* __restrict__ in_row = inp + (size_t)r * (size_t)C;
  float* __restrict__ out_row = out + (size_t)r * (size_t)C;
  const int tid = (int)threadIdx.x;

  float tmax = -INFINITY;
  float expv[Q + ((REM > 0) ? 1 : 0)];

  #pragma unroll
  for (int i = 0; i < Q; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    const float v = intentir_ldg_f32(in_row + (size_t)c);
    expv[i] = v;
    tmax = fmaxf(tmax, v);
  }
  if constexpr (REM > 0) {
    float v = -INFINITY;
    if (tid < REM) {
      const int c = tid + Q * BLOCK_THREADS;
      v = intentir_ldg_f32(in_row + (size_t)c);
    }
    expv[Q] = v;
    tmax = fmaxf(tmax, v);
  }

  const float mx = block_allreduce_max_f32<BLOCK_THREADS>(tmax);

  float tsum = 0.0f;
  #pragma unroll
  for (int i = 0; i < Q; ++i) {
    const float e = softmax_fast_exp<USE_EXP2>(expv[i] - mx);
    expv[i] = e;
    tsum += e;
  }
  if constexpr (REM > 0) {
    float e = 0.0f;
    if (tid < REM) e = softmax_fast_exp<USE_EXP2>(expv[Q] - mx);
    expv[Q] = e;
    tsum += e;
  }

  const float sum = block_allreduce_sum_f32<BLOCK_THREADS>(tsum);
  const float inv = __fdividef(1.0f, sum);

  #pragma unroll
  for (int i = 0; i < Q; ++i) {
    const int c = tid + i * BLOCK_THREADS;
    out_row[(size_t)c] = expv[i] * inv;
  }
  if constexpr (REM > 0) {
    if (tid < REM) {
      const int c = tid + Q * BLOCK_THREADS;
      out_row[(size_t)c] = expv[Q] * inv;
    }
  }
}

template <int BLOCK_THREADS, int TILES, bool USE_EXP2, bool FULL_TILE = false>
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
    if constexpr (FULL_TILE) {
      if (aligned) {
        v = *reinterpret_cast<const float4*>(in_row + (size_t)base);
      } else {
        v.x = intentir_ldg_f32(in_row + (size_t)(base + 0));
        v.y = intentir_ldg_f32(in_row + (size_t)(base + 1));
        v.z = intentir_ldg_f32(in_row + (size_t)(base + 2));
        v.w = intentir_ldg_f32(in_row + (size_t)(base + 3));
      }
    } else {
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
    }
    vals[t] = v;
    tmax = fmaxf(tmax, v.x);
    tmax = fmaxf(tmax, v.y);
    tmax = fmaxf(tmax, v.z);
    tmax = fmaxf(tmax, v.w);
  }
  const float mx = block_allreduce_max_f32<BLOCK_THREADS>(tmax);

  float tsum = 0.0f;
  #pragma unroll
  for (int t = 0; t < TILES; ++t) {
    float4 v = vals[t];
    const int base = t * (BLOCK_THREADS * VEC) + tid * VEC;
    if constexpr (FULL_TILE) {
      v.x = softmax_fast_exp<USE_EXP2>(v.x - mx);
      v.y = softmax_fast_exp<USE_EXP2>(v.y - mx);
      v.z = softmax_fast_exp<USE_EXP2>(v.z - mx);
      v.w = softmax_fast_exp<USE_EXP2>(v.w - mx);
    } else {
      if (base + 0 < C) v.x = softmax_fast_exp<USE_EXP2>(v.x - mx); else v.x = 0.0f;
      if (base + 1 < C) v.y = softmax_fast_exp<USE_EXP2>(v.y - mx); else v.y = 0.0f;
      if (base + 2 < C) v.z = softmax_fast_exp<USE_EXP2>(v.z - mx); else v.z = 0.0f;
      if (base + 3 < C) v.w = softmax_fast_exp<USE_EXP2>(v.w - mx); else v.w = 0.0f;
    }
    vals[t] = v;
    tsum += (v.x + v.y + v.z + v.w);
  }
  const float sum = block_allreduce_sum_f32<BLOCK_THREADS>(tsum);
  const float inv = __fdividef(1.0f, sum);

  #pragma unroll
  for (int t = 0; t < TILES; ++t) {
    const int base = t * (BLOCK_THREADS * VEC) + tid * VEC;
    float4 v = vals[t];
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    v.w *= inv;
    if constexpr (FULL_TILE) {
      if (aligned) {
        *reinterpret_cast<float4*>(out_row + (size_t)base) = v;
      } else {
        out_row[(size_t)(base + 0)] = v.x;
        out_row[(size_t)(base + 1)] = v.y;
        out_row[(size_t)(base + 2)] = v.z;
        out_row[(size_t)(base + 3)] = v.w;
      }
    } else {
      if (base >= C) continue;
      if (aligned && (base + (VEC - 1) < C)) {
        *reinterpret_cast<float4*>(out_row + (size_t)base) = v;
      } else {
        if (base + 0 < C) out_row[(size_t)(base + 0)] = v.x;
        if (base + 1 < C) out_row[(size_t)(base + 1)] = v.y;
        if (base + 2 < C) out_row[(size_t)(base + 2)] = v.z;
        if (base + 3 < C) out_row[(size_t)(base + 3)] = v.w;
      }
    }
  }
}

template <int WARPS_PER_BLOCK, bool USE_EXP2>
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
      e.x = softmax_fast_exp<USE_EXP2>(v.x - mx);
      e.y = softmax_fast_exp<USE_EXP2>(v.y - mx);
      e.z = softmax_fast_exp<USE_EXP2>(v.z - mx);
      e.w = softmax_fast_exp<USE_EXP2>(v.w - mx);
      *reinterpret_cast<float4*>(buf + (size_t)base) = e;
      tsum += (e.x + e.y + e.z + e.w);
    } else {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const int c = base + j;
        if (c < C) {
          const float v = buf[(size_t)c];
          const float e = softmax_fast_exp<USE_EXP2>(v - mx);
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

template <int WARPS_PER_BLOCK, bool USE_EXP2>
__device__ __forceinline__ void softmax_2d_last_f32_warp_expbuf(const float* __restrict__ inp, float* __restrict__ out, int R, int C) {
  static_assert(WARPS_PER_BLOCK > 0 && WARPS_PER_BLOCK <= 8, "softmax warp expbuf supports 1..8 warps per block");
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
  #pragma unroll 1
  for (int base = lane * VEC; base < len0; base += 32 * VEC) {
    const int in_base = shift + base;
    if (base + (VEC - 1) < len0) {
      const float4 x = *reinterpret_cast<const float4*>(in_row + (size_t)in_base);
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
          tmax = fmaxf(tmax, v);
        }
      }
    }
  }
  if (lane < shift) {
    const int c = lane;
    const float v = intentir_ldg_f32(in_row + (size_t)c);
    tmax = fmaxf(tmax, v);
  }
  const float mx = warp_allreduce_max(tmax);

  float tsum = 0.0f;
  #pragma unroll 1
  for (int base = lane * VEC; base < len0; base += 32 * VEC) {
    const int in_base = shift + base;
    if (base + (VEC - 1) < len0) {
      const float4 x = *reinterpret_cast<const float4*>(in_row + (size_t)in_base);
      float4 e;
      e.x = softmax_fast_exp<USE_EXP2>(x.x - mx);
      e.y = softmax_fast_exp<USE_EXP2>(x.y - mx);
      e.z = softmax_fast_exp<USE_EXP2>(x.z - mx);
      e.w = softmax_fast_exp<USE_EXP2>(x.w - mx);
      *reinterpret_cast<float4*>(buf + (size_t)base) = e;
      tsum += (e.x + e.y + e.z + e.w);
    } else {
      #pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const int c = base + j;
        if (c < len0) {
          const float v = intentir_ldg_f32(in_row + (size_t)(shift + c));
          const float e = softmax_fast_exp<USE_EXP2>(v - mx);
          buf[(size_t)c] = e;
          tsum += e;
        }
      }
    }
  }
  if (lane < shift) {
    const int c = lane;
    const int buf_idx = len0 + c;
    if (buf_idx < C) {
      const float v = intentir_ldg_f32(in_row + (size_t)c);
      const float e = softmax_fast_exp<USE_EXP2>(v - mx);
      buf[(size_t)buf_idx] = e;
      tsum += e;
    }
  }
  const float sum = warp_allreduce_sum(tsum);
  const float inv = __fdividef(1.0f, sum);

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
