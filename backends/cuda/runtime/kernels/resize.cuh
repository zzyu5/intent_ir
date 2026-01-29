#pragma once

#include <stddef.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

template <int BLOCK_W>
__device__ __forceinline__ void resize_bilinear2x_i8(
    const int8_t* __restrict__ src,
    int8_t* __restrict__ out,
    int C,
    int H,
    int W) {
  static_assert(BLOCK_W > 0 && BLOCK_W <= 1024, "resize block size must be in (0,1024]");

  const int OH = H * 2;
  const int OW = W * 2;

  const int h_idx = (int)blockIdx.y;
  const int c = (int)blockIdx.z;
  const int x0 = (int)blockIdx.x * BLOCK_W + (int)threadIdx.x;
  if (c >= C || h_idx >= OH || x0 >= W) return;

  const int y0 = h_idx >> 1;
  const int x1 = (x0 + 1 < W) ? (x0 + 1) : (W - 1);
  const int y1 = (y0 + 1 < H) ? (y0 + 1) : (H - 1);

  const int64_t src_hw = (int64_t)H * (int64_t)W;
  const int64_t dst_hw = (int64_t)OH * (int64_t)OW;
  const int64_t src_base = (int64_t)c * src_hw;
  const int64_t dst_base = (int64_t)c * dst_hw + (int64_t)h_idx * (int64_t)OW;

  const int64_t row0 = src_base + (int64_t)y0 * (int64_t)W;
  const int64_t row1 = src_base + (int64_t)y1 * (int64_t)W;

  const int16_t a = (int16_t)intentir_ldg<int8_t>(src + row0 + x0);
  const int16_t b = (int16_t)intentir_ldg<int8_t>(src + row0 + x1);
  const int16_t c0 = (int16_t)intentir_ldg<int8_t>(src + row1 + x0);
  const int16_t d = (int16_t)intentir_ldg<int8_t>(src + row1 + x1);

  // Compute 2 output pixels: w=2*x0 (even) and w=2*x0+1 (odd).
  const int y_odd = (h_idx & 1);
  const int32_t sum1_even = (int32_t)a;
  const int32_t sum2_even = (int32_t)c0;
  const int32_t sum1_odd = (((int32_t)a + (int32_t)b) >> 1);
  const int32_t sum2_odd = (((int32_t)c0 + (int32_t)d) >> 1);
  const int32_t out_even = y_odd ? ((sum1_even + sum2_even) >> 1) : sum1_even;
  const int32_t out_odd = y_odd ? ((sum1_odd + sum2_odd) >> 1) : sum1_odd;
  // Store two adjacent int8 outputs at once. The destination row base is
  // 2-byte aligned because OW = 2*W is always even; 2*x0 is also even.
  const uint16_t packed =
      (uint16_t)((uint8_t)out_even) | (uint16_t)((uint8_t)out_odd) << 8;
  reinterpret_cast<uint16_t*>(out + dst_base)[x0] = packed;
}

}  // namespace intentir_cuda
