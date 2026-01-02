#include "intentir_ops.h"

void intentir_reduce_sum_2d_axis1_f32(const float* a, float* out, int64_t M, int64_t K, float scale, int has_scale) {
  if (!a || !out || M <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (int64_t m = 0; m < M; ++m) {
    size_t vlmax = intentir_vsetvl_e32m1((size_t)K);
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[idx2((int)m, (int)k, (int)K)], vl);
      vsum = __riscv_vfadd_vv_f32m1(vsum, vx, vl);
      k += (int64_t)vl;
    }
    vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vres = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0, vlmax);
    float s = __riscv_vfmv_f_s_f32m1_f32(vres);
    if (has_scale) s *= scale;
    out[(size_t)m] = s;
  }
#else
  for (int64_t m = 0; m < M; ++m) {
    double acc = 0.0;
    for (int64_t k = 0; k < K; ++k) acc += (double)a[idx2((int)m, (int)k, (int)K)];
    if (has_scale) acc *= (double)scale;
    out[(size_t)m] = (float)acc;
  }
#endif
}

void intentir_reduce_sum_4d_axis23_f32(
    const float* a, float* out, int64_t N, int64_t G, int64_t GS, int64_t HW, float scale, int has_scale) {
  if (!a || !out || N <= 0 || G <= 0 || GS <= 0 || HW <= 0) return;
  const int64_t len = GS * HW;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (int64_t n0 = 0; n0 < N; ++n0) {
    for (int64_t g0 = 0; g0 < G; ++g0) {
      const size_t base = ((size_t)n0 * (size_t)G + (size_t)g0) * (size_t)len;
      size_t vlmax = intentir_vsetvl_e32m1((size_t)len);
      vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      for (size_t i = 0; i < (size_t)len;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)len - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + i], vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, vx, vl);
        i += vl;
      }
      vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t vres = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0, vlmax);
      float s = __riscv_vfmv_f_s_f32m1_f32(vres);
      if (has_scale) s *= scale;
      out[(size_t)n0 * (size_t)G + (size_t)g0] = s;
    }
  }
#else
  for (int64_t n0 = 0; n0 < N; ++n0) {
    for (int64_t g0 = 0; g0 < G; ++g0) {
      double acc = 0.0;
      for (int64_t gs = 0; gs < GS; ++gs) {
        for (int64_t hw = 0; hw < HW; ++hw) {
          acc += (double)a[idx4((int)n0, (int)g0, (int)gs, (int)hw, (int)G, (int)GS, (int)HW)];
        }
      }
      if (has_scale) acc *= (double)scale;
      out[(size_t)n0 * (size_t)G + (size_t)g0] = (float)acc;
    }
  }
#endif
}

void intentir_reduce_max_2d_axis1_f32(const float* a, float* out, int64_t M, int64_t K) {
  if (!a || !out || M <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (int64_t m = 0; m < M; ++m) {
    size_t vlmax = intentir_vsetvl_e32m1((size_t)K);
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[idx2((int)m, (int)k, (int)K)], vl);
      vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
      k += (int64_t)vl;
    }
    vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
    vfloat32m1_t vres = __riscv_vfredmax_vs_f32m1_f32m1(vmax, v0, vlmax);
    out[(size_t)m] = __riscv_vfmv_f_s_f32m1_f32(vres);
  }
#else
  for (int64_t m = 0; m < M; ++m) {
    float mx = -INFINITY;
    for (int64_t k = 0; k < K; ++k) mx = fmaxf(mx, a[idx2((int)m, (int)k, (int)K)]);
    out[(size_t)m] = mx;
  }
#endif
}

