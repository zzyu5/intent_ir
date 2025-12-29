// TileLang CUDA shim: define missing fp8 type used by TileLang templates.
//
// Some TileLang releases reference `__nv_fp8_e8m0` from their templates, but
// CUDA headers only define `__nv_fp8_e4m3` / `__nv_fp8_e5m2`. For our pipeline
// we don't need fp8_e8m0 functionality; we only need the type to exist so
// kernels can compile on mainstream CUDA toolchains.
#pragma once

#include <cuda_fp8.h>

#ifndef __nv_fp8_e8m0
struct __align__(1) __nv_fp8_e8m0 {
  __nv_fp8_storage_t __x;
};
#endif

