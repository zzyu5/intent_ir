#pragma once

#include <stddef.h>
#include <stdint.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

template <int BLOCK_THREADS>
__device__ __forceinline__ void correlation_i8(
    const int8_t* __restrict__ src0,
    const int8_t* __restrict__ src1,
    int8_t* __restrict__ out,
    int out_channel,
    int in_channel,
    int height,
    int width,
    int out_shift) {
  static_assert(BLOCK_THREADS > 0 && BLOCK_THREADS <= 1024, "correlation block size must be in (0,1024]");

  const int64_t hw = (int64_t)height * (int64_t)width;
  const int64_t total = (int64_t)out_channel * hw;
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)BLOCK_THREADS + (int64_t)threadIdx.x;
  if (tid >= total) return;

  const int oc = (int)(tid / hw);
  const int64_t rem = tid - (int64_t)oc * hw;
  const int h = (int)(rem / (int64_t)width);
  const int w = (int)(rem - (int64_t)h * (int64_t)width);

  int sh = out_shift;
  if (sh < 0) sh = 0;
  if (sh > 30) sh = 30;

  if (oc >= width || w < oc) {
    out[(size_t)tid] = 0;
    return;
  }

  int32_t acc = 0;
  const int64_t off0 = (int64_t)h * (int64_t)width + (int64_t)w;
  const int64_t off1 = (int64_t)h * (int64_t)width + (int64_t)(w - oc);
#pragma unroll 4
  for (int k = 0; k < in_channel; ++k) {
    const int64_t base = (int64_t)k * hw;
    const int8_t a = intentir_ldg<int8_t>(src0 + base + off0);
    const int8_t b = intentir_ldg<int8_t>(src1 + base + off1);
    acc += (int32_t)a * (int32_t)b;
  }
  out[(size_t)tid] = (int8_t)(acc >> sh);
}

}  // namespace intentir_cuda
