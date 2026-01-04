#ifndef INTENTIR_RUNTIME_H
#define INTENTIR_RUNTIME_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int intentir_read_bytes(const char* path, void* dst, size_t bytes);

int intentir_compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol);
int intentir_compare_u8(const char* name, const uint8_t* got, const uint8_t* ref, size_t n);

uint64_t intentir_now_ns(void);

#ifdef __cplusplus
}  // extern "C"
#endif

// Schedule hint: cap vector length when RVV is available.
#ifndef INTENTIR_VEC_WIDTH
#define INTENTIR_VEC_WIDTH 0
#endif

#if defined(__riscv_vector) || defined(__riscv_v)
#include <riscv_vector.h>

static inline size_t intentir_vsetvl_e32m1(size_t rem) {
  if (INTENTIR_VEC_WIDTH > 0 && rem > (size_t)INTENTIR_VEC_WIDTH) rem = (size_t)INTENTIR_VEC_WIDTH;
  return __riscv_vsetvl_e32m1(rem);
}

static inline size_t intentir_vsetvl_e8m1(size_t rem) {
  if (INTENTIR_VEC_WIDTH > 0) {
    // INTENTIR_VEC_WIDTH is expressed in f32 lanes; e8 has 4x lanes.
    size_t cap = (size_t)INTENTIR_VEC_WIDTH * 4;
    if (rem > cap) rem = cap;
  }
  return __riscv_vsetvl_e8m1(rem);
}
#endif

static inline size_t idx2(int i, int j, int D1) { return (size_t)i * (size_t)D1 + (size_t)j; }
static inline size_t idx3(int i, int j, int k, int D1, int D2) { return ((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k; }
static inline size_t idx4(int i, int j, int k, int l, int D1, int D2, int D3) {
  return (((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k) * (size_t)D3 + (size_t)l;
}

#endif  // INTENTIR_RUNTIME_H
