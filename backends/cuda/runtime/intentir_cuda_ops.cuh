#pragma once

#include <stdint.h>

// -----------------------------------------------------------------------------
// Common CUDA device helpers for IntentIR-generated kernels.
//
// The goal is to keep codegen in Python, but move reusable low-level pieces
// (loads, RNG, reductions) into a stable runtime header, similar in spirit to
// backends/spmd_rvv/runtime/*.
// -----------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T intentir_ldg(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  return __ldg(p);
#else
  return *p;
#endif
}

__device__ __forceinline__ float intentir_ldg_f32(const float* p) { return intentir_ldg<float>(p); }

// -----------------------------------------------------------------------------
// Async copy helpers (Ampere+).
//
// We use cp.async to overlap global->shared loads with tensor core compute in
// generated matmul kernels. For older GPUs, we fall back to a normal load/store.
// -----------------------------------------------------------------------------

__device__ __forceinline__ void intentir_cp_async_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  // Cache global (cg): bypass L1 and cache in L2.
  // For GEMM-style streaming loads this often reduces L1 thrash and can improve
  // overlap with tensor-core compute.
  const unsigned int smem = __cvta_generic_to_shared(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem), "l"(gmem_src) : "memory");
#else
  // Fallback: synchronous copy (16 bytes).
  *reinterpret_cast<float4*>(smem_dst) = *reinterpret_cast<const float4*>(gmem_src);
#endif
}

__device__ __forceinline__ void intentir_cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

template <int N>
__device__ __forceinline__ void intentir_cp_async_wait_group();

template <>
__device__ __forceinline__ void intentir_cp_async_wait_group<0>() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

template <>
__device__ __forceinline__ void intentir_cp_async_wait_group<1>() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 1;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void intentir_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  intentir_cp_async_wait_group<0>();
#endif
}

// -----------------------------------------------------------------------------
// Philox RNG (matches semantics used by the RVV backend runtime).
// -----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t intentir_philox_randint_u32(uint64_t seed, uint32_t c0, int n_rounds) {
  uint32_t c1 = 0u, c2 = 0u, c3 = 0u;
  uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFu);
  uint32_t k1 = (uint32_t)((seed >> 32) & 0xFFFFFFFFu);
  const uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  const uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  const uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  const uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;
  if (n_rounds <= 0) n_rounds = 10;
  #pragma unroll
  for (int r = 0; r < 10; ++r) {
    if (r >= n_rounds) break;
    const uint32_t _c0 = c0;
    const uint32_t _c2 = c2;
    const uint32_t hi0 = __umulhi(PHILOX_ROUND_A, _c0);
    const uint32_t hi1 = __umulhi(PHILOX_ROUND_B, _c2);
    const uint32_t lo0 = (uint32_t)(PHILOX_ROUND_A * _c0);
    const uint32_t lo1 = (uint32_t)(PHILOX_ROUND_B * _c2);
    c0 = hi1 ^ c1 ^ k0;
    c2 = hi0 ^ c3 ^ k1;
    c1 = lo1;
    c3 = lo0;
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }
  return c0;
}

__device__ __forceinline__ float intentir_uint_to_uniform_float_u32(uint32_t x) {
  // Uniform float in [0, 1). Mirrors the "signed int then abs-ish" mapping used by RVV runtime.
  int32_t xi = (int32_t)x;
  if (xi < 0) xi = ~xi;  // -x-1
  return (float)xi * 4.6566127342e-10f;
}
