#pragma once

#include <stddef.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

template <int HEADS_PER_BLOCK, int ROPE_VEC, int BLOCK_X, int ITERS, bool FULL_HEADS, bool FULL_TILE, typename idx_t>
__device__ __forceinline__ void rope_f32(
    const float* __restrict__ inp,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ out,
    int SEQ_LEN,
    int BATCH_NUM,
    int HEAD_NUM,
    int HEAD_DIM) {
  static_assert(HEADS_PER_BLOCK > 0 && HEADS_PER_BLOCK <= 16, "rope HEADS_PER_BLOCK must be in (0,16]");
  static_assert(ROPE_VEC == 1 || ROPE_VEC == 2 || ROPE_VEC == 4, "rope ROPE_VEC must be 1,2,4");
  static_assert(BLOCK_X > 0 && BLOCK_X <= 1024, "rope BLOCK_X must be in (0,1024]");
  static_assert(ITERS > 0 && ITERS <= 1024, "rope ITERS must be in (0,1024]");

  const int pid_head_group = (int)blockIdx.x;
  const int pid_batch = (int)blockIdx.y;
  const int pid_seq = (int)blockIdx.z;
  // Grid is chosen from bindings; for the canonical mapping (HEADS_PER_BLOCK=1),
  // (pid_seq, pid_batch, pid_head) are always in range, so these checks are redundant.
  // Keep them only for non-canonical head grouping.
  if constexpr (HEADS_PER_BLOCK != 1) {
    if (pid_batch >= BATCH_NUM || pid_seq >= SEQ_LEN) return;
  }
  const int half = (int)(HEAD_DIM >> 1);
  const int head0 = pid_head_group * HEADS_PER_BLOCK;
  if constexpr (HEADS_PER_BLOCK != 1 && !FULL_HEADS) {
    if (head0 >= HEAD_NUM) return;
  }

  const idx_t base0 = (idx_t)(((idx_t)pid_seq * (idx_t)BATCH_NUM + (idx_t)pid_batch) * (idx_t)HEAD_NUM) * (idx_t)HEAD_DIM;
  const idx_t cb0 = (idx_t)pid_seq * (idx_t)half;
  const int tid = (int)threadIdx.x;

  if constexpr (ROPE_VEC == 4) {
    const int half4 = (int)(half >> 2);  // # of float4 packs
    const float4* __restrict__ cos4 = (const float4* __restrict__)(cos + cb0);
    const float4* __restrict__ sin4 = (const float4* __restrict__)(sin + cb0);
    if constexpr (HEADS_PER_BLOCK == 1) {
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      const float4* __restrict__ x14 = (const float4* __restrict__)(inp + base);
      const float4* __restrict__ x24 = (const float4* __restrict__)(inp + base + (idx_t)half);
      float4* __restrict__ y14 = (float4* __restrict__)(out + base);
      float4* __restrict__ y24 = (float4* __restrict__)(out + base + (idx_t)half);
      #pragma unroll
      for (int k = 0; k < ITERS; ++k) {
        const int j4 = tid + k * BLOCK_X;
        if constexpr (!FULL_TILE) {
          if (j4 >= half4) break;
        }
        const float4 c4 = cos4[j4];
        const float4 s4 = sin4[j4];
        const float4 a = x14[j4];
        const float4 b = x24[j4];
        float4 y1;
        float4 y2;
        y1.x = __fmaf_rn(-b.x, s4.x, a.x * c4.x);
        y1.y = __fmaf_rn(-b.y, s4.y, a.y * c4.y);
        y1.z = __fmaf_rn(-b.z, s4.z, a.z * c4.z);
        y1.w = __fmaf_rn(-b.w, s4.w, a.w * c4.w);
        y2.x = __fmaf_rn(a.x, s4.x, b.x * c4.x);
        y2.y = __fmaf_rn(a.y, s4.y, b.y * c4.y);
        y2.z = __fmaf_rn(a.z, s4.z, b.z * c4.z);
        y2.w = __fmaf_rn(a.w, s4.w, b.w * c4.w);
        y14[j4] = y1;
        y24[j4] = y2;
      }
    } else {
      for (int j4 = tid; j4 < half4; j4 += BLOCK_X) {
        const float4 c4 = cos4[j4];
        const float4 s4 = sin4[j4];
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {
          const int head = head0 + gh;
          if constexpr (!FULL_HEADS) {
            if (head >= HEAD_NUM) break;
          }
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float4* __restrict__ x14 = (const float4* __restrict__)(inp + base);
          const float4* __restrict__ x24 = (const float4* __restrict__)(inp + base + (idx_t)half);
          float4* __restrict__ y14 = (float4* __restrict__)(out + base);
          float4* __restrict__ y24 = (float4* __restrict__)(out + base + (idx_t)half);
          const float4 a = x14[j4];
          const float4 b = x24[j4];
          float4 y1;
          float4 y2;
          y1.x = __fmaf_rn(-b.x, s4.x, a.x * c4.x);
          y1.y = __fmaf_rn(-b.y, s4.y, a.y * c4.y);
          y1.z = __fmaf_rn(-b.z, s4.z, a.z * c4.z);
          y1.w = __fmaf_rn(-b.w, s4.w, a.w * c4.w);
          y2.x = __fmaf_rn(a.x, s4.x, b.x * c4.x);
          y2.y = __fmaf_rn(a.y, s4.y, b.y * c4.y);
          y2.z = __fmaf_rn(a.z, s4.z, b.z * c4.z);
          y2.w = __fmaf_rn(a.w, s4.w, b.w * c4.w);
          y14[j4] = y1;
          y24[j4] = y2;
        }
      }
    }
  } else if constexpr (ROPE_VEC == 2) {
    const int half2 = (int)(half >> 1);  // # of float2 packs
    const float2* __restrict__ cos2 = (const float2* __restrict__)(cos + cb0);
    const float2* __restrict__ sin2 = (const float2* __restrict__)(sin + cb0);
    if constexpr (HEADS_PER_BLOCK == 1) {
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      const float2* __restrict__ x12 = (const float2* __restrict__)(inp + base);
      const float2* __restrict__ x22 = (const float2* __restrict__)(inp + base + (idx_t)half);
      float2* __restrict__ y12 = (float2* __restrict__)(out + base);
      float2* __restrict__ y22 = (float2* __restrict__)(out + base + (idx_t)half);
      #pragma unroll
      for (int k = 0; k < ITERS; ++k) {
        const int j2 = tid + k * BLOCK_X;
        if constexpr (!FULL_TILE) {
          if (j2 >= half2) break;
        }
        const float2 c2 = cos2[j2];
        const float2 s2 = sin2[j2];
        const float2 a = x12[j2];
        const float2 b = x22[j2];
        float2 y1;
        float2 y2;
        y1.x = __fmaf_rn(-b.x, s2.x, a.x * c2.x);
        y1.y = __fmaf_rn(-b.y, s2.y, a.y * c2.y);
        y2.x = __fmaf_rn(a.x, s2.x, b.x * c2.x);
        y2.y = __fmaf_rn(a.y, s2.y, b.y * c2.y);
        y12[j2] = y1;
        y22[j2] = y2;
      }
    } else {
      for (int j2 = tid; j2 < half2; j2 += BLOCK_X) {
        const float2 c2 = cos2[j2];
        const float2 s2 = sin2[j2];
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {
          const int head = head0 + gh;
          if constexpr (!FULL_HEADS) {
            if (head >= HEAD_NUM) break;
          }
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float2* __restrict__ x12 = (const float2* __restrict__)(inp + base);
          const float2* __restrict__ x22 = (const float2* __restrict__)(inp + base + (idx_t)half);
          float2* __restrict__ y12 = (float2* __restrict__)(out + base);
          float2* __restrict__ y22 = (float2* __restrict__)(out + base + (idx_t)half);
          const float2 a = x12[j2];
          const float2 b = x22[j2];
          float2 y1;
          float2 y2;
          y1.x = __fmaf_rn(-b.x, s2.x, a.x * c2.x);
          y1.y = __fmaf_rn(-b.y, s2.y, a.y * c2.y);
          y2.x = __fmaf_rn(a.x, s2.x, b.x * c2.x);
          y2.y = __fmaf_rn(a.y, s2.y, b.y * c2.y);
          y12[j2] = y1;
          y22[j2] = y2;
        }
      }
    }
  } else {
    if constexpr (HEADS_PER_BLOCK == 1) {
      const int head = head0;
      const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
      #pragma unroll
      for (int k = 0; k < ITERS; ++k) {
        const int j = tid + k * BLOCK_X;
        if constexpr (!FULL_TILE) {
          if (j >= half) break;
        }
        const idx_t cb = cb0 + (idx_t)j;
        const float c = intentir_ldg_f32(&cos[cb]);
        const float s0 = intentir_ldg_f32(&sin[cb]);
        const float x1 = intentir_ldg_f32(&inp[base + (idx_t)j]);
        const float x2 = intentir_ldg_f32(&inp[base + (idx_t)half + (idx_t)j]);
        out[base + (idx_t)j] = __fmaf_rn(-x2, s0, x1 * c);
        out[base + (idx_t)half + (idx_t)j] = __fmaf_rn(x1, s0, x2 * c);
      }
    } else {
      for (int j = tid; j < half; j += BLOCK_X) {
        const idx_t cb = cb0 + (idx_t)j;
        const float c = intentir_ldg_f32(&cos[cb]);
        const float s0 = intentir_ldg_f32(&sin[cb]);
        #pragma unroll
        for (int gh = 0; gh < HEADS_PER_BLOCK; ++gh) {
          const int head = head0 + gh;
          if constexpr (!FULL_HEADS) {
            if (head >= HEAD_NUM) break;
          }
          const idx_t base = base0 + (idx_t)head * (idx_t)HEAD_DIM;
          const float x1 = intentir_ldg_f32(&inp[base + (idx_t)j]);
          const float x2 = intentir_ldg_f32(&inp[base + (idx_t)half + (idx_t)j]);
          out[base + (idx_t)j] = __fmaf_rn(-x2, s0, x1 * c);
          out[base + (idx_t)half + (idx_t)j] = __fmaf_rn(x1, s0, x2 * c);
        }
      }
    }
  }
}

}  // namespace intentir_cuda
