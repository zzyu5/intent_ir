#pragma once

#include <stddef.h>
#include <stdint.h>

namespace intentir_cuda {

template <int BLOCK_W, bool FULL_W>
__device__ __forceinline__ void warp_q8_8_i8_i16(
    const int8_t* __restrict__ src,
    const int16_t* __restrict__ offset,
    int8_t* __restrict__ out,
    int C,
    int H,
    int W) {
  static_assert(BLOCK_W > 0 && BLOCK_W <= 1024, "warp block size must be in (0,1024]");

  const int h = (int)blockIdx.y;
  const int c = (int)blockIdx.z;
  const int w = (int)blockIdx.x * BLOCK_W + (int)threadIdx.x;
  // `h` and `c` are in-bounds by construction (grid.y = H, grid.z = C).
  (void)H;
  (void)C;
  if constexpr (!FULL_W) {
    if (w >= W) return;
  }
  const int64_t hw = (int64_t)H * (int64_t)W;
  const int64_t row_base = (int64_t)c * hw + (int64_t)h * (int64_t)W;
  const int64_t off_base = (int64_t)h * (int64_t)W;
  const int16_t ov = offset[off_base + w];
  const int8_t offset_int = (int8_t)(ov >> 8);
  const int8_t offset_frac = (int8_t)(((int16_t)(ov << 8)) >> 8);
  const int8_t indvar = (int8_t)w;
  const int8_t right_i8 = (int8_t)(indvar - offset_int);
  const int8_t left_i8 = (int8_t)(right_i8 - 1);
  const int right = (int)right_i8;
  const int left = (int)left_i8;
  int8_t right_val = 0;
  int8_t left_val = 0;
  if (right >= 0 && right < W) right_val = src[row_base + (int64_t)right];
  if (left >= 0 && left < W) left_val = src[row_base + (int64_t)left];
  int16_t outv = (int16_t)((int16_t)right_val << 8);
  outv = (int16_t)(outv + (int16_t)((int16_t)(left_val - right_val) * (int16_t)offset_frac));
  outv = (int16_t)(outv >> 8);
  out[row_base + w] = (int8_t)outv;
}

}  // namespace intentir_cuda
