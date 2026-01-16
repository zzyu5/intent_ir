#include "intentir_ops.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__riscv_vector) || defined(__riscv_v)
static inline vfloat32m1_t intentir_vexp_approx_f32m1(vfloat32m1_t x, size_t vl) {
  const float LOG2E = 1.4426950408889634f;
  const float LN2 = 0.6931471805599453f;

  vfloat32m1_t y = __riscv_vfmul_vf_f32m1(x, LOG2E, vl);

  // n = floor(y) using rtz trunc + correction for negative fractional part.
  vint32m1_t n = __riscv_vfcvt_x_f_v_i32m1(y, vl);
  vfloat32m1_t nt = __riscv_vfcvt_f_x_v_f32m1(n, vl);
  vbool32_t m = __riscv_vmflt_vv_f32m1_b32(y, nt, vl);
  vint32m1_t n1 = __riscv_vsub_vx_i32m1(n, 1, vl);
  n = __riscv_vmerge_vvm_i32m1(n, n1, m, vl);
  nt = __riscv_vfcvt_f_x_v_f32m1(n, vl);

  vfloat32m1_t f = __riscv_vfsub_vv_f32m1(y, nt, vl);
  vfloat32m1_t z = __riscv_vfmul_vf_f32m1(f, LN2, vl);

  // p = exp(z) on z in [0, ln(2)) using 5th-order Taylor.
  vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  p = __riscv_vfadd_vv_f32m1(p, z, vl);
  vfloat32m1_t z2 = __riscv_vfmul_vv_f32m1(z, z, vl);
  p = __riscv_vfadd_vv_f32m1(p, __riscv_vfmul_vf_f32m1(z2, 0.5f, vl), vl);
  vfloat32m1_t z3 = __riscv_vfmul_vv_f32m1(z2, z, vl);
  p = __riscv_vfadd_vv_f32m1(p, __riscv_vfmul_vf_f32m1(z3, 0.1666666716f, vl), vl);
  vfloat32m1_t z4 = __riscv_vfmul_vv_f32m1(z2, z2, vl);
  p = __riscv_vfadd_vv_f32m1(p, __riscv_vfmul_vf_f32m1(z4, 0.0416666679f, vl), vl);
  vfloat32m1_t z5 = __riscv_vfmul_vv_f32m1(z4, z, vl);
  p = __riscv_vfadd_vv_f32m1(p, __riscv_vfmul_vf_f32m1(z5, 0.0083333338f, vl), vl);

  // scale = 2^n via exponent bits.
  vint32m1_t ncl = __riscv_vmax_vx_i32m1(n, -126, vl);
  ncl = __riscv_vmin_vx_i32m1(ncl, 127, vl);
  vint32m1_t e = __riscv_vadd_vx_i32m1(ncl, 127, vl);  // [1,254]
  vuint32m1_t eb = __riscv_vreinterpret_v_i32m1_u32m1(e);
  eb = __riscv_vsll_vx_u32m1(eb, 23, vl);
  vfloat32m1_t scale = __riscv_vreinterpret_v_u32m1_f32m1(eb);
  return __riscv_vfmul_vv_f32m1(p, scale, vl);
}
#endif

void intentir_reduce_sum_2d_axis1_f32(const float* a, float* out, int64_t M, int64_t K, float scale, int has_scale) {
  if (!a || !out || M <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((N * G) >= 4)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((N * G) >= 4)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    float mx = -INFINITY;
    for (int64_t k = 0; k < K; ++k) mx = fmaxf(mx, a[idx2((int)m, (int)k, (int)K)]);
    out[(size_t)m] = mx;
  }
#endif
}

void intentir_layernorm_2d_f32(
    const float* X, float* Y, const float* W, const float* B, float* Mean, float* Rstd, int64_t M, int64_t N, float eps) {
  if (!X || !Y || !W || !B || !Mean || !Rstd) return;
  if (M <= 0 || N <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const float* xrow = &X[(size_t)m * (size_t)N];
    float* yrow = &Y[(size_t)m * (size_t)N];

    // Pass 1: mean
    size_t vlmax = intentir_vsetvl_e32m1((size_t)N);
    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int64_t n = 0; n < N;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(N - n));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&xrow[(size_t)n], vl);
      vsum = __riscv_vfadd_vv_f32m1(vsum, vx, vl);
      n += (int64_t)vl;
    }
    vfloat32m1_t v0s = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vress = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0s, vlmax);
    float sum = __riscv_vfmv_f_s_f32m1_f32(vress);
    float mean = sum / (float)N;

    // Pass 2: variance of centered values.
    vfloat32m1_t vsumsq = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int64_t n = 0; n < N;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(N - n));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&xrow[(size_t)n], vl);
      vfloat32m1_t vc = __riscv_vfsub_vf_f32m1(vx, mean, vl);
      vfloat32m1_t vq = __riscv_vfmul_vv_f32m1(vc, vc, vl);
      vsumsq = __riscv_vfadd_vv_f32m1(vsumsq, vq, vl);
      n += (int64_t)vl;
    }
    vfloat32m1_t v0q = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vresq = __riscv_vfredusum_vs_f32m1_f32m1(vsumsq, v0q, vlmax);
    float sumsq = __riscv_vfmv_f_s_f32m1_f32(vresq);
    float var = sumsq / (float)N;
    float rstd = 1.0f / sqrtf(var + eps);

    Mean[(size_t)m] = mean;
    Rstd[(size_t)m] = rstd;

    // Pass 3: normalize + affine
    for (int64_t n = 0; n < N;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(N - n));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&xrow[(size_t)n], vl);
      vfloat32m1_t vw = __riscv_vle32_v_f32m1(&W[(size_t)n], vl);
      vfloat32m1_t vb = __riscv_vle32_v_f32m1(&B[(size_t)n], vl);
      vfloat32m1_t vc = __riscv_vfsub_vf_f32m1(vx, mean, vl);
      vfloat32m1_t vn = __riscv_vfmul_vf_f32m1(vc, rstd, vl);
      vfloat32m1_t vs = __riscv_vfmul_vv_f32m1(vn, vw, vl);
      vfloat32m1_t vy = __riscv_vfadd_vv_f32m1(vs, vb, vl);
      __riscv_vse32_v_f32m1(&yrow[(size_t)n], vy, vl);
      n += (int64_t)vl;
    }
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    const float* xrow = &X[(size_t)m * (size_t)N];
    float* yrow = &Y[(size_t)m * (size_t)N];
    double sum = 0.0;
    for (int64_t n = 0; n < N; ++n) sum += (double)xrow[(size_t)n];
    double mean = sum / (double)N;
    double sumsq = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      double c = (double)xrow[(size_t)n] - mean;
      sumsq += c * c;
    }
    double var = sumsq / (double)N;
    float rstd = 1.0f / sqrtf((float)var + eps);
    Mean[(size_t)m] = (float)mean;
    Rstd[(size_t)m] = rstd;
    for (int64_t n = 0; n < N; ++n) {
      float c = xrow[(size_t)n] - (float)mean;
      yrow[(size_t)n] = (c * rstd) * W[(size_t)n] + B[(size_t)n];
    }
  }
#endif
}

static inline uint32_t intentir_umulhi_u32(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32); }

static inline uint32_t intentir_philox_randint_u32(uint64_t seed, uint32_t offset, int n_rounds) {
  uint32_t c0 = offset, c1 = 0u, c2 = 0u, c3 = 0u;
  uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFu);
  uint32_t k1 = (uint32_t)((seed >> 32) & 0xFFFFFFFFu);
  const uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  const uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  const uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  const uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;
  if (n_rounds <= 0) n_rounds = 10;
  for (int r = 0; r < n_rounds; ++r) {
    uint32_t _c0 = c0, _c2 = c2;
    c0 = intentir_umulhi_u32(PHILOX_ROUND_B, _c2) ^ c1 ^ k0;
    c2 = intentir_umulhi_u32(PHILOX_ROUND_A, _c0) ^ c3 ^ k1;
    c1 = (uint32_t)((uint64_t)PHILOX_ROUND_B * (uint64_t)_c2);
    c3 = (uint32_t)((uint64_t)PHILOX_ROUND_A * (uint64_t)_c0);
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }
  return c0;
}

static inline float intentir_uint_to_uniform_float_u32(uint32_t x) {
  // Matches triton.language.random.uint_to_uniform_float for uint32/int32:
  // x = bitcast to int32; x = where(x < 0, -x-1, x); return x * 4.6566127342e-10
  int32_t xi = (int32_t)x;  // bitcast
  if (xi < 0) xi = ~xi;     // -x-1 in two's complement
  return (float)xi * 4.6566127342e-10f;
}

#if defined(__riscv_vector) || defined(__riscv_v)
static inline vuint32m1_t intentir_philox_randint_u32m1(vuint32m1_t c0, uint64_t seed, int n_rounds, size_t vl) {
  vuint32m1_t c1 = __riscv_vmv_v_x_u32m1(0u, vl);
  vuint32m1_t c2 = __riscv_vmv_v_x_u32m1(0u, vl);
  vuint32m1_t c3 = __riscv_vmv_v_x_u32m1(0u, vl);
  vuint32m1_t k0 = __riscv_vmv_v_x_u32m1((uint32_t)(seed & 0xFFFFFFFFu), vl);
  vuint32m1_t k1 = __riscv_vmv_v_x_u32m1((uint32_t)((seed >> 32) & 0xFFFFFFFFu), vl);
  const uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  const uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  const uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  const uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;
  if (n_rounds <= 0) n_rounds = 10;
  vuint32m1_t vA = __riscv_vmv_v_x_u32m1(PHILOX_ROUND_A, vl);
  vuint32m1_t vB = __riscv_vmv_v_x_u32m1(PHILOX_ROUND_B, vl);
  for (int r = 0; r < n_rounds; ++r) {
    vuint32m1_t _c0 = c0, _c2 = c2;
    vuint32m1_t hi_B_c2 = __riscv_vmulhu_vv_u32m1(vB, _c2, vl);
    vuint32m1_t hi_A_c0 = __riscv_vmulhu_vv_u32m1(vA, _c0, vl);
    c0 = __riscv_vxor_vv_u32m1(hi_B_c2, c1, vl);
    c0 = __riscv_vxor_vv_u32m1(c0, k0, vl);
    c2 = __riscv_vxor_vv_u32m1(hi_A_c0, c3, vl);
    c2 = __riscv_vxor_vv_u32m1(c2, k1, vl);
    c1 = __riscv_vmul_vv_u32m1(vB, _c2, vl);
    c3 = __riscv_vmul_vv_u32m1(vA, _c0, vl);
    k0 = __riscv_vadd_vx_u32m1(k0, PHILOX_KEY_A, vl);
    k1 = __riscv_vadd_vx_u32m1(k1, PHILOX_KEY_B, vl);
  }
  return c0;
}
#endif

void intentir_dropout_f32(const float* X, float* Y, size_t n, float p, uint64_t seed, int n_rounds) {
  if (!X || !Y) return;
  if (n == 0) return;
  if (p <= 0.0f) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 4096)
#endif
    for (size_t i = 0; i < n; ++i) Y[i] = X[i];
    return;
  }
  if (p >= 1.0f) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 4096)
#endif
    for (size_t i = 0; i < n; ++i) Y[i] = 0.0f;
    return;
  }
  const float inv_keep = 1.0f / (1.0f - p);
#if defined(__riscv_vector) || defined(__riscv_v)
  const int64_t CHUNK = 16384;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((int64_t)n >= (CHUNK * 2))
#endif
  for (int64_t base = 0; base < (int64_t)n; base += CHUNK) {
    size_t i = (size_t)base;
    size_t end = i + (size_t)CHUNK;
    if (end > n) end = n;
    while (i < end) {
      size_t vl = intentir_vsetvl_e32m1(end - i);
      vuint32m1_t vid = __riscv_vid_v_u32m1(vl);
      vuint32m1_t offs = __riscv_vadd_vx_u32m1(vid, (uint32_t)i, vl);
      vuint32m1_t rnd_u = intentir_philox_randint_u32m1(offs, seed, n_rounds, vl);
      vint32m1_t xi = __riscv_vreinterpret_v_u32m1_i32m1(rnd_u);
      vbool32_t neg = __riscv_vmslt_vx_i32m1_b32(xi, 0, vl);
      vint32m1_t xnot = __riscv_vnot_v_i32m1(xi, vl);
      vint32m1_t xabs = __riscv_vmerge_vvm_i32m1(xi, xnot, neg, vl);
      vfloat32m1_t r = __riscv_vfcvt_f_x_v_f32m1(xabs, vl);
      r = __riscv_vfmul_vf_f32m1(r, 4.6566127342e-10f, vl);
      vbool32_t keep = __riscv_vmfgt_vf_f32m1_b32(r, p, vl);
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&X[i], vl);
      vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vx, inv_keep, vl);
      vfloat32m1_t zeros = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      vy = __riscv_vmerge_vvm_f32m1(zeros, vy, keep, vl);
      __riscv_vse32_v_f32m1(&Y[i], vy, vl);
      i += vl;
    }
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 4096)
#endif
  for (size_t i = 0; i < n; ++i) {
    float r = intentir_uint_to_uniform_float_u32(intentir_philox_randint_u32(seed, (uint32_t)i, n_rounds));
    Y[i] = (r > p) ? (X[i] * inv_keep) : 0.0f;
  }
#endif
}

void intentir_softmax_1d_last_f32(const float* a, float* out, int64_t K) {
  if (!a || !out || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  size_t vlmax = intentir_vsetvl_e32m1((size_t)K);

  vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
  for (int64_t k = 0; k < K;) {
    size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[(size_t)k], vl);
    vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
    k += (int64_t)vl;
  }
  vfloat32m1_t v0m = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
  vfloat32m1_t vresm = __riscv_vfredmax_vs_f32m1_f32m1(vmax, v0m, vlmax);
  float mx = __riscv_vfmv_f_s_f32m1_f32(vresm);

  vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  for (int64_t k = 0; k < K;) {
    size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[(size_t)k], vl);
    vx = __riscv_vfsub_vf_f32m1(vx, mx, vl);
    vfloat32m1_t ve = intentir_vexp_approx_f32m1(vx, vl);
    __riscv_vse32_v_f32m1(&out[(size_t)k], ve, vl);
    vsum = __riscv_vfadd_vv_f32m1(vsum, ve, vl);
    k += (int64_t)vl;
  }

  vfloat32m1_t v0s = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
  vfloat32m1_t vress = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0s, vlmax);
  float sum = __riscv_vfmv_f_s_f32m1_f32(vress);
  float inv = 1.0f / sum;

  for (int64_t k = 0; k < K;) {
    size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&out[(size_t)k], vl);
    vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vx, inv, vl);
    __riscv_vse32_v_f32m1(&out[(size_t)k], vy, vl);
    k += (int64_t)vl;
  }
  return;
#else
  double mx = -1e30;
  for (int64_t k = 0; k < K; ++k) {
    double v = (double)a[(size_t)k];
    if (v > mx) mx = v;
  }
  double sum = 0.0;
  for (int64_t k = 0; k < K; ++k) {
    double e = exp((double)a[(size_t)k] - mx);
    out[(size_t)k] = (float)e;
    sum += e;
  }
  double inv = 1.0 / sum;
  for (int64_t k = 0; k < K; ++k) out[(size_t)k] = (float)((double)out[(size_t)k] * inv);
#endif
}

void intentir_softmax_2d_last_f32(const float* a, float* out, int64_t M, int64_t K) {
  if (!a || !out || M <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    size_t vlmax = intentir_vsetvl_e32m1((size_t)K);
    size_t base = idx2((int)m, 0, (int)K);

    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
      vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
      k += (int64_t)vl;
    }
    vfloat32m1_t v0m = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
    vfloat32m1_t vresm = __riscv_vfredmax_vs_f32m1_f32m1(vmax, v0m, vlmax);
    float mx = __riscv_vfmv_f_s_f32m1_f32(vresm);

    vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
      vx = __riscv_vfsub_vf_f32m1(vx, mx, vl);
      vfloat32m1_t ve = intentir_vexp_approx_f32m1(vx, vl);
      __riscv_vse32_v_f32m1(&out[base + (size_t)k], ve, vl);
      vsum = __riscv_vfadd_vv_f32m1(vsum, ve, vl);
      k += (int64_t)vl;
    }

    vfloat32m1_t v0s = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vress = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0s, vlmax);
    float sum = __riscv_vfmv_f_s_f32m1_f32(vress);
    float inv = 1.0f / sum;

    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&out[base + (size_t)k], vl);
      vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vx, inv, vl);
      __riscv_vse32_v_f32m1(&out[base + (size_t)k], vy, vl);
      k += (int64_t)vl;
    }
  }
  return;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (M >= 4)
#endif
  for (int64_t m = 0; m < M; ++m) {
    double mx = -1e30;
    for (int64_t k = 0; k < K; ++k) {
      double v = (double)a[idx2((int)m, (int)k, (int)K)];
      if (v > mx) mx = v;
    }
    double sum = 0.0;
    for (int64_t k = 0; k < K; ++k) {
      double e = exp((double)a[idx2((int)m, (int)k, (int)K)] - mx);
      out[idx2((int)m, (int)k, (int)K)] = (float)e;
      sum += e;
    }
    double inv = 1.0 / sum;
    for (int64_t k = 0; k < K; ++k) {
      size_t i = idx2((int)m, (int)k, (int)K);
      out[i] = (float)((double)out[i] * inv);
    }
  }
}

void intentir_softmax_3d_last_f32(const float* a, float* out, int64_t A0, int64_t A1, int64_t K) {
  if (!a || !out || A0 <= 0 || A1 <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((A0 * A1) >= 4)
#endif
  for (int64_t i0 = 0; i0 < A0; ++i0) {
    for (int64_t i1 = 0; i1 < A1; ++i1) {
      size_t vlmax = intentir_vsetvl_e32m1((size_t)K);
      size_t base = idx3((int)i0, (int)i1, 0, (int)A1, (int)K);

      vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
      for (int64_t k = 0; k < K;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
        vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
        k += (int64_t)vl;
      }
      vfloat32m1_t v0m = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
      vfloat32m1_t vresm = __riscv_vfredmax_vs_f32m1_f32m1(vmax, v0m, vlmax);
      float mx = __riscv_vfmv_f_s_f32m1_f32(vresm);

      vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      for (int64_t k = 0; k < K;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
        vx = __riscv_vfsub_vf_f32m1(vx, mx, vl);
        vfloat32m1_t ve = intentir_vexp_approx_f32m1(vx, vl);
        __riscv_vse32_v_f32m1(&out[base + (size_t)k], ve, vl);
        vsum = __riscv_vfadd_vv_f32m1(vsum, ve, vl);
        k += (int64_t)vl;
      }

      vfloat32m1_t v0s = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t vress = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0s, vlmax);
      float sum = __riscv_vfmv_f_s_f32m1_f32(vress);
      float inv = 1.0f / sum;

      for (int64_t k = 0; k < K;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&out[base + (size_t)k], vl);
        vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vx, inv, vl);
        __riscv_vse32_v_f32m1(&out[base + (size_t)k], vy, vl);
        k += (int64_t)vl;
      }
    }
  }
  return;
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((A0 * A1) >= 4)
#endif
  for (int64_t i0 = 0; i0 < A0; ++i0) {
    for (int64_t i1 = 0; i1 < A1; ++i1) {
      double mx = -1e30;
      for (int64_t k = 0; k < K; ++k) {
        double v = (double)a[idx3((int)i0, (int)i1, (int)k, (int)A1, (int)K)];
        if (v > mx) mx = v;
      }
      double sum = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        size_t i = idx3((int)i0, (int)i1, (int)k, (int)A1, (int)K);
        double e = exp((double)a[i] - mx);
        out[i] = (float)e;
        sum += e;
      }
      double inv = 1.0 / sum;
      for (int64_t k = 0; k < K; ++k) {
        size_t i = idx3((int)i0, (int)i1, (int)k, (int)A1, (int)K);
        out[i] = (float)((double)out[i] * inv);
      }
    }
  }
}

void intentir_softmax_4d_last_f32(const float* a, float* out, int64_t B, int64_t H, int64_t Q, int64_t K) {
  if (!a || !out || B <= 0 || H <= 0 || Q <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) if (((B * H) * Q) >= 4)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t q = 0; q < Q; ++q) {
        size_t vlmax = intentir_vsetvl_e32m1((size_t)K);
        size_t base = idx4((int)b, (int)h, (int)q, 0, (int)H, (int)Q, (int)K);

        vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
        for (int64_t k = 0; k < K;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
          vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
          vmax = __riscv_vfmax_vv_f32m1(vmax, vx, vl);
          k += (int64_t)vl;
        }
        vfloat32m1_t v0m = __riscv_vfmv_v_f_f32m1(-INFINITY, vlmax);
        vfloat32m1_t vresm = __riscv_vfredmax_vs_f32m1_f32m1(vmax, v0m, vlmax);
        float mx = __riscv_vfmv_f_s_f32m1_f32(vresm);

        vfloat32m1_t vsum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        for (int64_t k = 0; k < K;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
          vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[base + (size_t)k], vl);
          vx = __riscv_vfsub_vf_f32m1(vx, mx, vl);
          vfloat32m1_t ve = intentir_vexp_approx_f32m1(vx, vl);
          __riscv_vse32_v_f32m1(&out[base + (size_t)k], ve, vl);
          vsum = __riscv_vfadd_vv_f32m1(vsum, ve, vl);
          k += (int64_t)vl;
        }

        vfloat32m1_t v0s = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        vfloat32m1_t vress = __riscv_vfredusum_vs_f32m1_f32m1(vsum, v0s, vlmax);
        float sum = __riscv_vfmv_f_s_f32m1_f32(vress);
        float inv = 1.0f / sum;

        for (int64_t k = 0; k < K;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
          vfloat32m1_t vx = __riscv_vle32_v_f32m1(&out[base + (size_t)k], vl);
          vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vx, inv, vl);
          __riscv_vse32_v_f32m1(&out[base + (size_t)k], vy, vl);
          k += (int64_t)vl;
        }
      }
    }
  }
  return;
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) if (((B * H) * Q) >= 4)
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t q = 0; q < Q; ++q) {
        double mx = -1e30;
        for (int64_t k = 0; k < K; ++k) {
          double v = (double)a[idx4((int)b, (int)h, (int)q, (int)k, (int)H, (int)Q, (int)K)];
          if (v > mx) mx = v;
        }
        double sum = 0.0;
        for (int64_t k = 0; k < K; ++k) {
          size_t i = idx4((int)b, (int)h, (int)q, (int)k, (int)H, (int)Q, (int)K);
          double e = exp((double)a[i] - mx);
          out[i] = (float)e;
          sum += e;
        }
        double inv = 1.0 / sum;
        for (int64_t k = 0; k < K; ++k) {
          size_t i = idx4((int)b, (int)h, (int)q, (int)k, (int)H, (int)Q, (int)K);
          out[i] = (float)((double)out[i] * inv);
        }
      }
    }
  }
}

static inline int64_t intentir_min_i64(int64_t a, int64_t b) { return (a < b) ? a : b; }

static inline size_t intentir_numel_rank(const int64_t* shape, int rank) {
  size_t n = 1;
  for (int i = 0; i < rank; ++i) n *= (size_t)shape[i];
  return n;
}

static inline int intentir_shapes_equal(const int64_t* a, const int64_t* b, int rank) {
  for (int i = 0; i < rank; ++i) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

static inline void intentir_unravel_index(size_t linear, const int64_t* shape, int rank, int64_t* coords) {
  for (int d = rank - 1; d >= 0; --d) {
    size_t dim = (size_t)shape[d];
    coords[d] = (int64_t)(linear % dim);
    linear /= dim;
  }
}

static inline size_t intentir_ravel_broadcast(const int64_t* coords, const int64_t* shape, int rank) {
  size_t idx = 0;
  for (int d = 0; d < rank; ++d) {
    size_t dim = (size_t)shape[d];
    size_t c = (dim == 1) ? 0 : (size_t)coords[d];
    idx = idx * dim + c;
  }
  return idx;
}

static inline float intentir_apply_f32_bin(float x, float y, int op) {
  switch (op) {
    case INTENTIR_F32_BIN_ADD:
      return x + y;
    case INTENTIR_F32_BIN_SUB:
      return x - y;
    case INTENTIR_F32_BIN_MUL:
      return x * y;
    case INTENTIR_F32_BIN_DIV:
      return x / y;
    case INTENTIR_F32_BIN_MAX:
      return fmaxf(x, y);
    case INTENTIR_F32_BIN_MIN:
      return fminf(x, y);
    default:
      return x;
  }
}

static void intentir_f32_bin_contig(const float* a, const float* b, float* out, size_t n, int op) {
  if (!a || !b || !out || n == 0) return;
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
#if defined(__riscv_vector) || defined(__riscv_v)
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
        vfloat32m1_t vc;
        if (op == INTENTIR_F32_BIN_ADD) vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
        else if (op == INTENTIR_F32_BIN_SUB) vc = __riscv_vfsub_vv_f32m1(va, vb, vl);
        else if (op == INTENTIR_F32_BIN_MUL) vc = __riscv_vfmul_vv_f32m1(va, vb, vl);
        else if (op == INTENTIR_F32_BIN_DIV) vc = __riscv_vfdiv_vv_f32m1(va, vb, vl);
        else if (op == INTENTIR_F32_BIN_MAX) vc = __riscv_vfmax_vv_f32m1(va, vb, vl);
        else if (op == INTENTIR_F32_BIN_MIN) vc = __riscv_vfmin_vv_f32m1(va, vb, vl);
        else vc = va;
        __riscv_vse32_v_f32m1(&out[i], vc, vl);
        i += vl;
      }
#else
      for (size_t i = i0; i < i1; ++i) out[i] = intentir_apply_f32_bin(a[i], b[i], op);
#endif
    }
    return;
  }
#endif
#if defined(__riscv_vector) || defined(__riscv_v)
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
    vfloat32m1_t vc;
    if (op == INTENTIR_F32_BIN_ADD) vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
    else if (op == INTENTIR_F32_BIN_SUB) vc = __riscv_vfsub_vv_f32m1(va, vb, vl);
    else if (op == INTENTIR_F32_BIN_MUL) vc = __riscv_vfmul_vv_f32m1(va, vb, vl);
    else if (op == INTENTIR_F32_BIN_DIV) vc = __riscv_vfdiv_vv_f32m1(va, vb, vl);
    else if (op == INTENTIR_F32_BIN_MAX) vc = __riscv_vfmax_vv_f32m1(va, vb, vl);
    else if (op == INTENTIR_F32_BIN_MIN) vc = __riscv_vfmin_vv_f32m1(va, vb, vl);
    else vc = va;
    __riscv_vse32_v_f32m1(&out[i], vc, vl);
    i += vl;
  }
#else
  for (size_t i = 0; i < n; ++i) out[i] = intentir_apply_f32_bin(a[i], b[i], op);
#endif
}

#if defined(__riscv_vector) || defined(__riscv_v)
static inline void intentir_make_bcast_strides(const int64_t* shape, int rank, int64_t* strides) {
  int64_t s = 1;
  for (int i = rank - 1; i >= 0; --i) {
    int64_t d = shape[i];
    if (d <= 1) strides[i] = 0;
    else strides[i] = s;
    if (d > 0) s *= d;
  }
}

static inline vfloat32m1_t intentir_apply_f32_bin_vec(vfloat32m1_t a, vfloat32m1_t b, int op, size_t vl) {
  if (op == INTENTIR_F32_BIN_ADD) return __riscv_vfadd_vv_f32m1(a, b, vl);
  if (op == INTENTIR_F32_BIN_SUB) return __riscv_vfsub_vv_f32m1(a, b, vl);
  if (op == INTENTIR_F32_BIN_MUL) return __riscv_vfmul_vv_f32m1(a, b, vl);
  if (op == INTENTIR_F32_BIN_DIV) return __riscv_vfdiv_vv_f32m1(a, b, vl);
  if (op == INTENTIR_F32_BIN_MAX) return __riscv_vfmax_vv_f32m1(a, b, vl);
  if (op == INTENTIR_F32_BIN_MIN) return __riscv_vfmin_vv_f32m1(a, b, vl);
  return a;
}

static inline vbool32_t intentir_apply_cmp_f32_vec(vfloat32m1_t a, vfloat32m1_t b, int op, size_t vl) {
  if (op == INTENTIR_CMP_LT) return __riscv_vmflt_vv_f32m1_b32(a, b, vl);
  if (op == INTENTIR_CMP_LE) return __riscv_vmfle_vv_f32m1_b32(a, b, vl);
  if (op == INTENTIR_CMP_GT) return __riscv_vmfgt_vv_f32m1_b32(a, b, vl);
  if (op == INTENTIR_CMP_GE) return __riscv_vmfge_vv_f32m1_b32(a, b, vl);
  if (op == INTENTIR_CMP_NE) return __riscv_vmfne_vv_f32m1_b32(a, b, vl);
  return __riscv_vmfne_vv_f32m1_b32(a, b, vl);
}

static inline vbool32_t intentir_apply_cmp_i32_vec(vint32m1_t a, vint32m1_t b, int op, size_t vl) {
  if (op == INTENTIR_CMP_LT) return __riscv_vmslt_vv_i32m1_b32(a, b, vl);
  if (op == INTENTIR_CMP_LE) {
    vbool32_t lt = __riscv_vmslt_vv_i32m1_b32(a, b, vl);
    vbool32_t eq = __riscv_vmseq_vv_i32m1_b32(a, b, vl);
    return __riscv_vmor_mm_b32(lt, eq, vl);
  }
  if (op == INTENTIR_CMP_GT) return __riscv_vmslt_vv_i32m1_b32(b, a, vl);
  if (op == INTENTIR_CMP_GE) {
    vbool32_t gt = __riscv_vmslt_vv_i32m1_b32(b, a, vl);
    vbool32_t eq = __riscv_vmseq_vv_i32m1_b32(a, b, vl);
    return __riscv_vmor_mm_b32(gt, eq, vl);
  }
  if (op == INTENTIR_CMP_NE) return __riscv_vmsne_vv_i32m1_b32(a, b, vl);
  return __riscv_vmsne_vv_i32m1_b32(a, b, vl);
}

static void intentir_cmp_f32_broadcast_vec(
    const float* a, const float* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  int64_t so[4] = {0, 0, 0, 0};
  int64_t sa[4] = {0, 0, 0, 0};
  int64_t sb[4] = {0, 0, 0, 0};
  intentir_make_bcast_strides(out_shape, rank, so);
  intentir_make_bcast_strides(a_shape, rank, sa);
  intentir_make_bcast_strides(b_shape, rank, sb);
  const int64_t last = out_shape[rank - 1];
  if (last <= 0) return;

  if (rank == 1) {
    for (size_t j = 0; j < (size_t)last;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
      vfloat32m1_t va = (a_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(a[0], vl) : __riscv_vle32_v_f32m1(&a[j], vl);
      vfloat32m1_t vb = (b_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(b[0], vl) : __riscv_vle32_v_f32m1(&b[j], vl);
      vbool32_t m = intentir_apply_cmp_f32_vec(va, vb, op, vl);
      vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
      vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
      vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
      __riscv_vse8_v_u8mf4(&out[j], vo, vl);
      j += vl;
    }
    return;
  }

  if (rank == 2) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      const size_t ob = (size_t)(i0 * so[0]);
      const size_t ab = (size_t)(i0 * sa[0]);
      const size_t bb = (size_t)(i0 * sb[0]);
      for (size_t j = 0; j < (size_t)last;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
        vfloat32m1_t va = (a_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
        vfloat32m1_t vb = (b_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
        vbool32_t m = intentir_apply_cmp_f32_vec(va, vb, op, vl);
        vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
        vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
        vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
        __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
        j += vl;
      }
    }
    return;
  }

  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        const size_t ob = (size_t)(i0 * so[0] + i1 * so[1]);
        const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1]);
        const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1]);
        for (size_t j = 0; j < (size_t)last;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
          vfloat32m1_t va = (a_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
          vfloat32m1_t vb = (b_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
          vbool32_t m = intentir_apply_cmp_f32_vec(va, vb, op, vl);
          vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
          vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
          vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
          __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
          j += vl;
        }
      }
    }
    return;
  }

  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          const size_t ob = (size_t)(i0 * so[0] + i1 * so[1] + i2 * so[2]);
          const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1] + i2 * sa[2]);
          const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1] + i2 * sb[2]);
          for (size_t j = 0; j < (size_t)last;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
            vfloat32m1_t va = (a_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
            vfloat32m1_t vb = (b_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
            vbool32_t m = intentir_apply_cmp_f32_vec(va, vb, op, vl);
            vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
            vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
            vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
            __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
            j += vl;
          }
        }
      }
    }
    return;
  }
}

static void intentir_cmp_i32_broadcast_vec(
    const int32_t* a, const int32_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  int64_t so[4] = {0, 0, 0, 0};
  int64_t sa[4] = {0, 0, 0, 0};
  int64_t sb[4] = {0, 0, 0, 0};
  intentir_make_bcast_strides(out_shape, rank, so);
  intentir_make_bcast_strides(a_shape, rank, sa);
  intentir_make_bcast_strides(b_shape, rank, sb);
  const int64_t last = out_shape[rank - 1];
  if (last <= 0) return;

  if (rank == 1) {
    for (size_t j = 0; j < (size_t)last;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
      vint32m1_t va = (a_shape[0] == 1) ? __riscv_vmv_v_x_i32m1(a[0], vl) : __riscv_vle32_v_i32m1(&a[j], vl);
      vint32m1_t vb = (b_shape[0] == 1) ? __riscv_vmv_v_x_i32m1(b[0], vl) : __riscv_vle32_v_i32m1(&b[j], vl);
      vbool32_t m = intentir_apply_cmp_i32_vec(va, vb, op, vl);
      vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
      vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
      vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
      __riscv_vse8_v_u8mf4(&out[j], vo, vl);
      j += vl;
    }
    return;
  }

  if (rank == 2) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      const size_t ob = (size_t)(i0 * so[0]);
      const size_t ab = (size_t)(i0 * sa[0]);
      const size_t bb = (size_t)(i0 * sb[0]);
      for (size_t j = 0; j < (size_t)last;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
        vint32m1_t va = (a_shape[1] == 1) ? __riscv_vmv_v_x_i32m1(a[ab], vl) : __riscv_vle32_v_i32m1(&a[ab + j], vl);
        vint32m1_t vb = (b_shape[1] == 1) ? __riscv_vmv_v_x_i32m1(b[bb], vl) : __riscv_vle32_v_i32m1(&b[bb + j], vl);
        vbool32_t m = intentir_apply_cmp_i32_vec(va, vb, op, vl);
        vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
        vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
        vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
        __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
        j += vl;
      }
    }
    return;
  }

  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        const size_t ob = (size_t)(i0 * so[0] + i1 * so[1]);
        const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1]);
        const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1]);
        for (size_t j = 0; j < (size_t)last;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
          vint32m1_t va = (a_shape[2] == 1) ? __riscv_vmv_v_x_i32m1(a[ab], vl) : __riscv_vle32_v_i32m1(&a[ab + j], vl);
          vint32m1_t vb = (b_shape[2] == 1) ? __riscv_vmv_v_x_i32m1(b[bb], vl) : __riscv_vle32_v_i32m1(&b[bb + j], vl);
          vbool32_t m = intentir_apply_cmp_i32_vec(va, vb, op, vl);
          vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
          vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
          vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
          __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
          j += vl;
        }
      }
    }
    return;
  }

  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          const size_t ob = (size_t)(i0 * so[0] + i1 * so[1] + i2 * so[2]);
          const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1] + i2 * sa[2]);
          const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1] + i2 * sb[2]);
          for (size_t j = 0; j < (size_t)last;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
            vint32m1_t va = (a_shape[3] == 1) ? __riscv_vmv_v_x_i32m1(a[ab], vl) : __riscv_vle32_v_i32m1(&a[ab + j], vl);
            vint32m1_t vb = (b_shape[3] == 1) ? __riscv_vmv_v_x_i32m1(b[bb], vl) : __riscv_vle32_v_i32m1(&b[bb + j], vl);
            vbool32_t m = intentir_apply_cmp_i32_vec(va, vb, op, vl);
            vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
            vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
            vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
            __riscv_vse8_v_u8mf4(&out[ob + j], vo, vl);
            j += vl;
          }
        }
      }
    }
    return;
  }
}

static void intentir_bool_bin_broadcast_vec(
    const uint8_t* a, const uint8_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  int64_t so[4] = {0, 0, 0, 0};
  int64_t sa[4] = {0, 0, 0, 0};
  int64_t sb[4] = {0, 0, 0, 0};
  intentir_make_bcast_strides(out_shape, rank, so);
  intentir_make_bcast_strides(a_shape, rank, sa);
  intentir_make_bcast_strides(b_shape, rank, sb);
  const int64_t last = out_shape[rank - 1];
  if (last <= 0) return;

  if (rank == 1) {
    for (size_t j = 0; j < (size_t)last;) {
      size_t vl = intentir_vsetvl_e8m1((size_t)last - j);
      vuint8m1_t va = (a_shape[0] == 1) ? __riscv_vmv_v_x_u8m1(a[0], vl) : __riscv_vle8_v_u8m1(&a[j], vl);
      vuint8m1_t vb = (b_shape[0] == 1) ? __riscv_vmv_v_x_u8m1(b[0], vl) : __riscv_vle8_v_u8m1(&b[j], vl);
      vbool8_t ma = __riscv_vmsne_vx_u8m1_b8(va, 0, vl);
      vbool8_t mb = __riscv_vmsne_vx_u8m1_b8(vb, 0, vl);
      vbool8_t m = (op == INTENTIR_BOOL_BIN_OR) ? __riscv_vmor_mm_b8(ma, mb, vl) : __riscv_vmand_mm_b8(ma, mb, vl);
      vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);
      vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
      vuint8m1_t vo = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);
      __riscv_vse8_v_u8m1(&out[j], vo, vl);
      j += vl;
    }
    return;
  }

  if (rank == 2) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      const size_t ob = (size_t)(i0 * so[0]);
      const size_t ab = (size_t)(i0 * sa[0]);
      const size_t bb = (size_t)(i0 * sb[0]);
      for (size_t j = 0; j < (size_t)last;) {
        size_t vl = intentir_vsetvl_e8m1((size_t)last - j);
        vuint8m1_t va = (a_shape[1] == 1) ? __riscv_vmv_v_x_u8m1(a[ab], vl) : __riscv_vle8_v_u8m1(&a[ab + j], vl);
        vuint8m1_t vb = (b_shape[1] == 1) ? __riscv_vmv_v_x_u8m1(b[bb], vl) : __riscv_vle8_v_u8m1(&b[bb + j], vl);
        vbool8_t ma = __riscv_vmsne_vx_u8m1_b8(va, 0, vl);
        vbool8_t mb = __riscv_vmsne_vx_u8m1_b8(vb, 0, vl);
        vbool8_t m = (op == INTENTIR_BOOL_BIN_OR) ? __riscv_vmor_mm_b8(ma, mb, vl) : __riscv_vmand_mm_b8(ma, mb, vl);
        vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);
        vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
        vuint8m1_t vo = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);
        __riscv_vse8_v_u8m1(&out[ob + j], vo, vl);
        j += vl;
      }
    }
    return;
  }

  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        const size_t ob = (size_t)(i0 * so[0] + i1 * so[1]);
        const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1]);
        const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1]);
        for (size_t j = 0; j < (size_t)last;) {
          size_t vl = intentir_vsetvl_e8m1((size_t)last - j);
          vuint8m1_t va = (a_shape[2] == 1) ? __riscv_vmv_v_x_u8m1(a[ab], vl) : __riscv_vle8_v_u8m1(&a[ab + j], vl);
          vuint8m1_t vb = (b_shape[2] == 1) ? __riscv_vmv_v_x_u8m1(b[bb], vl) : __riscv_vle8_v_u8m1(&b[bb + j], vl);
          vbool8_t ma = __riscv_vmsne_vx_u8m1_b8(va, 0, vl);
          vbool8_t mb = __riscv_vmsne_vx_u8m1_b8(vb, 0, vl);
          vbool8_t m = (op == INTENTIR_BOOL_BIN_OR) ? __riscv_vmor_mm_b8(ma, mb, vl) : __riscv_vmand_mm_b8(ma, mb, vl);
          vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);
          vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
          vuint8m1_t vo = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);
          __riscv_vse8_v_u8m1(&out[ob + j], vo, vl);
          j += vl;
        }
      }
    }
    return;
  }

  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          const size_t ob = (size_t)(i0 * so[0] + i1 * so[1] + i2 * so[2]);
          const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1] + i2 * sa[2]);
          const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1] + i2 * sb[2]);
          for (size_t j = 0; j < (size_t)last;) {
            size_t vl = intentir_vsetvl_e8m1((size_t)last - j);
            vuint8m1_t va = (a_shape[3] == 1) ? __riscv_vmv_v_x_u8m1(a[ab], vl) : __riscv_vle8_v_u8m1(&a[ab + j], vl);
            vuint8m1_t vb = (b_shape[3] == 1) ? __riscv_vmv_v_x_u8m1(b[bb], vl) : __riscv_vle8_v_u8m1(&b[bb + j], vl);
            vbool8_t ma = __riscv_vmsne_vx_u8m1_b8(va, 0, vl);
            vbool8_t mb = __riscv_vmsne_vx_u8m1_b8(vb, 0, vl);
            vbool8_t m = (op == INTENTIR_BOOL_BIN_OR) ? __riscv_vmor_mm_b8(ma, mb, vl) : __riscv_vmand_mm_b8(ma, mb, vl);
            vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);
            vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
            vuint8m1_t vo = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);
            __riscv_vse8_v_u8m1(&out[ob + j], vo, vl);
            j += vl;
          }
        }
      }
    }
    return;
  }
}

static void intentir_where_broadcast_vec(
    const uint8_t* cond, const float* x, const float* y, float* out, const int64_t* out_shape, const int64_t* cond_shape, const int64_t* x_shape,
    const int64_t* y_shape, int rank) {
  int64_t so[4] = {0, 0, 0, 0};
  int64_t sc[4] = {0, 0, 0, 0};
  int64_t sx[4] = {0, 0, 0, 0};
  int64_t sy[4] = {0, 0, 0, 0};
  intentir_make_bcast_strides(out_shape, rank, so);
  intentir_make_bcast_strides(cond_shape, rank, sc);
  intentir_make_bcast_strides(x_shape, rank, sx);
  intentir_make_bcast_strides(y_shape, rank, sy);
  const int64_t last = out_shape[rank - 1];
  if (last <= 0) return;

  if (rank == 1) {
    for (size_t j = 0; j < (size_t)last;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
      vuint8mf4_t vc = (cond_shape[0] == 1) ? __riscv_vmv_v_x_u8mf4(cond[0], vl) : __riscv_vle8_v_u8mf4(&cond[j], vl);
      vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
      vfloat32m1_t vx = (x_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(x[0], vl) : __riscv_vle32_v_f32m1(&x[j], vl);
      vfloat32m1_t vy = (y_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(y[0], vl) : __riscv_vle32_v_f32m1(&y[j], vl);
      vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(vy, vx, m, vl);
      __riscv_vse32_v_f32m1(&out[j], vo, vl);
      j += vl;
    }
    return;
  }

  if (rank == 2) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      const size_t ob = (size_t)(i0 * so[0]);
      const size_t cb = (size_t)(i0 * sc[0]);
      const size_t xb = (size_t)(i0 * sx[0]);
      const size_t yb = (size_t)(i0 * sy[0]);
      for (size_t j = 0; j < (size_t)last;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
        vuint8mf4_t vc =
            (cond_shape[1] == 1) ? __riscv_vmv_v_x_u8mf4(cond[cb], vl) : __riscv_vle8_v_u8mf4(&cond[cb + j], vl);
        vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
        vfloat32m1_t vx = (x_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(x[xb], vl) : __riscv_vle32_v_f32m1(&x[xb + j], vl);
        vfloat32m1_t vy = (y_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(y[yb], vl) : __riscv_vle32_v_f32m1(&y[yb + j], vl);
        vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(vy, vx, m, vl);
        __riscv_vse32_v_f32m1(&out[ob + j], vo, vl);
        j += vl;
      }
    }
    return;
  }

  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        const size_t ob = (size_t)(i0 * so[0] + i1 * so[1]);
        const size_t cb = (size_t)(i0 * sc[0] + i1 * sc[1]);
        const size_t xb = (size_t)(i0 * sx[0] + i1 * sx[1]);
        const size_t yb = (size_t)(i0 * sy[0] + i1 * sy[1]);
        for (size_t j = 0; j < (size_t)last;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
          vuint8mf4_t vc =
              (cond_shape[2] == 1) ? __riscv_vmv_v_x_u8mf4(cond[cb], vl) : __riscv_vle8_v_u8mf4(&cond[cb + j], vl);
          vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
          vfloat32m1_t vx = (x_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(x[xb], vl) : __riscv_vle32_v_f32m1(&x[xb + j], vl);
          vfloat32m1_t vy = (y_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(y[yb], vl) : __riscv_vle32_v_f32m1(&y[yb + j], vl);
          vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(vy, vx, m, vl);
          __riscv_vse32_v_f32m1(&out[ob + j], vo, vl);
          j += vl;
        }
      }
    }
    return;
  }

  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          const size_t ob = (size_t)(i0 * so[0] + i1 * so[1] + i2 * so[2]);
          const size_t cb = (size_t)(i0 * sc[0] + i1 * sc[1] + i2 * sc[2]);
          const size_t xb = (size_t)(i0 * sx[0] + i1 * sx[1] + i2 * sx[2]);
          const size_t yb = (size_t)(i0 * sy[0] + i1 * sy[1] + i2 * sy[2]);
          for (size_t j = 0; j < (size_t)last;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
            vuint8mf4_t vc =
                (cond_shape[3] == 1) ? __riscv_vmv_v_x_u8mf4(cond[cb], vl) : __riscv_vle8_v_u8mf4(&cond[cb + j], vl);
            vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
            vfloat32m1_t vx =
                (x_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(x[xb], vl) : __riscv_vle32_v_f32m1(&x[xb + j], vl);
            vfloat32m1_t vy =
                (y_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(y[yb], vl) : __riscv_vle32_v_f32m1(&y[yb + j], vl);
            vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(vy, vx, m, vl);
            __riscv_vse32_v_f32m1(&out[ob + j], vo, vl);
            j += vl;
          }
        }
      }
    }
    return;
  }
}

static void intentir_f32_bin_broadcast_vec(
    const float* a, const float* b, float* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  int64_t so[4] = {0, 0, 0, 0};
  int64_t sa[4] = {0, 0, 0, 0};
  int64_t sb[4] = {0, 0, 0, 0};
  intentir_make_bcast_strides(out_shape, rank, so);
  intentir_make_bcast_strides(a_shape, rank, sa);
  intentir_make_bcast_strides(b_shape, rank, sb);

  const int64_t last = out_shape[rank - 1];
  if (last <= 0) return;

  if (rank == 1) {
    for (size_t j = 0; j < (size_t)last;) {
      size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
      vfloat32m1_t va = (a_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(a[0], vl) : __riscv_vle32_v_f32m1(&a[j], vl);
      vfloat32m1_t vb = (b_shape[0] == 1) ? __riscv_vfmv_v_f_f32m1(b[0], vl) : __riscv_vle32_v_f32m1(&b[j], vl);
      vfloat32m1_t vc = intentir_apply_f32_bin_vec(va, vb, op, vl);
      __riscv_vse32_v_f32m1(&out[j], vc, vl);
      j += vl;
    }
    return;
  }

  if (rank == 2) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      const size_t ob = (size_t)(i0 * so[0]);
      const size_t ab = (size_t)(i0 * sa[0]);
      const size_t bb = (size_t)(i0 * sb[0]);
      for (size_t j = 0; j < (size_t)last;) {
        size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
        vfloat32m1_t va = (a_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
        vfloat32m1_t vb = (b_shape[1] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
        vfloat32m1_t vc = intentir_apply_f32_bin_vec(va, vb, op, vl);
        __riscv_vse32_v_f32m1(&out[ob + j], vc, vl);
        j += vl;
      }
    }
    return;
  }

  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        const size_t ob = (size_t)(i0 * so[0] + i1 * so[1]);
        const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1]);
        const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1]);
        for (size_t j = 0; j < (size_t)last;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
          vfloat32m1_t va = (a_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
          vfloat32m1_t vb = (b_shape[2] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
          vfloat32m1_t vc = intentir_apply_f32_bin_vec(va, vb, op, vl);
          __riscv_vse32_v_f32m1(&out[ob + j], vc, vl);
          j += vl;
        }
      }
    }
    return;
  }

  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          const size_t ob = (size_t)(i0 * so[0] + i1 * so[1] + i2 * so[2]);
          const size_t ab = (size_t)(i0 * sa[0] + i1 * sa[1] + i2 * sa[2]);
          const size_t bb = (size_t)(i0 * sb[0] + i1 * sb[1] + i2 * sb[2]);
          for (size_t j = 0; j < (size_t)last;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)last - j);
            vfloat32m1_t va = (a_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(a[ab], vl) : __riscv_vle32_v_f32m1(&a[ab + j], vl);
            vfloat32m1_t vb = (b_shape[3] == 1) ? __riscv_vfmv_v_f_f32m1(b[bb], vl) : __riscv_vle32_v_f32m1(&b[bb + j], vl);
            vfloat32m1_t vc = intentir_apply_f32_bin_vec(va, vb, op, vl);
            __riscv_vse32_v_f32m1(&out[ob + j], vc, vl);
            j += vl;
          }
        }
      }
    }
    return;
  }
}
#endif

void intentir_f32_bin_broadcast(
    const float* a, const float* b, float* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  if (!a || !b || !out || !out_shape || !a_shape || !b_shape) return;
  if (rank < 1 || rank > 4) return;

  const size_t n = intentir_numel_rank(out_shape, rank);
  if (intentir_shapes_equal(a_shape, out_shape, rank) && intentir_shapes_equal(b_shape, out_shape, rank)) {
    intentir_f32_bin_contig(a, b, out, n, op);
    return;
  }

#if defined(__riscv_vector) || defined(__riscv_v)
  // General broadcast: vectorize over the innermost dimension.
  intentir_f32_bin_broadcast_vec(a, b, out, out_shape, a_shape, b_shape, rank, op);
  return;
#endif

  if (rank == 1) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      int64_t a0 = (a_shape[0] == 1) ? 0 : i0;
      int64_t b0 = (b_shape[0] == 1) ? 0 : i0;
      out[(size_t)i0] = intentir_apply_f32_bin(a[(size_t)a0], b[(size_t)b0], op);
    }
    return;
  }
  if (rank == 2) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        int64_t a0 = (a_shape[0] == 1) ? 0 : i0;
        int64_t a1 = (a_shape[1] == 1) ? 0 : i1;
        int64_t b0 = (b_shape[0] == 1) ? 0 : i0;
        int64_t b1 = (b_shape[1] == 1) ? 0 : i1;
        size_t oi = (size_t)i0 * (size_t)D1 + (size_t)i1;
        size_t ai = (size_t)a0 * (size_t)a_shape[1] + (size_t)a1;
        size_t bi = (size_t)b0 * (size_t)b_shape[1] + (size_t)b1;
        out[oi] = intentir_apply_f32_bin(a[ai], b[bi], op);
      }
    }
    return;
  }
  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          int64_t a0 = (a_shape[0] == 1) ? 0 : i0;
          int64_t a1 = (a_shape[1] == 1) ? 0 : i1;
          int64_t a2 = (a_shape[2] == 1) ? 0 : i2;
          int64_t b0 = (b_shape[0] == 1) ? 0 : i0;
          int64_t b1 = (b_shape[1] == 1) ? 0 : i1;
          int64_t b2 = (b_shape[2] == 1) ? 0 : i2;
          size_t oi = ((size_t)i0 * (size_t)D1 + (size_t)i1) * (size_t)D2 + (size_t)i2;
          size_t ai = ((size_t)a0 * (size_t)a_shape[1] + (size_t)a1) * (size_t)a_shape[2] + (size_t)a2;
          size_t bi = ((size_t)b0 * (size_t)b_shape[1] + (size_t)b1) * (size_t)b_shape[2] + (size_t)b2;
          out[oi] = intentir_apply_f32_bin(a[ai], b[bi], op);
        }
      }
    }
    return;
  }
  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2], D3 = out_shape[3];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          for (int64_t i3 = 0; i3 < D3; ++i3) {
            int64_t a0 = (a_shape[0] == 1) ? 0 : i0;
            int64_t a1 = (a_shape[1] == 1) ? 0 : i1;
            int64_t a2 = (a_shape[2] == 1) ? 0 : i2;
            int64_t a3 = (a_shape[3] == 1) ? 0 : i3;
            int64_t b0 = (b_shape[0] == 1) ? 0 : i0;
            int64_t b1 = (b_shape[1] == 1) ? 0 : i1;
            int64_t b2 = (b_shape[2] == 1) ? 0 : i2;
            int64_t b3 = (b_shape[3] == 1) ? 0 : i3;
            size_t oi = (((size_t)i0 * (size_t)D1 + (size_t)i1) * (size_t)D2 + (size_t)i2) * (size_t)D3 + (size_t)i3;
            size_t ai = (((size_t)a0 * (size_t)a_shape[1] + (size_t)a1) * (size_t)a_shape[2] + (size_t)a2) * (size_t)a_shape[3] +
                         (size_t)a3;
            size_t bi = (((size_t)b0 * (size_t)b_shape[1] + (size_t)b1) * (size_t)b_shape[2] + (size_t)b2) * (size_t)b_shape[3] +
                         (size_t)b3;
            out[oi] = intentir_apply_f32_bin(a[ai], b[bi], op);
          }
        }
      }
    }
    return;
  }
}

static inline int intentir_apply_cmp_f32(float x, float y, int op) {
  switch (op) {
    case INTENTIR_CMP_LT:
      return x < y;
    case INTENTIR_CMP_LE:
      return x <= y;
    case INTENTIR_CMP_GT:
      return x > y;
    case INTENTIR_CMP_GE:
      return x >= y;
    case INTENTIR_CMP_NE:
      return x != y;
    default:
      return 0;
  }
}

static inline int intentir_apply_cmp_i32(int32_t x, int32_t y, int op) {
  switch (op) {
    case INTENTIR_CMP_LT:
      return x < y;
    case INTENTIR_CMP_LE:
      return x <= y;
    case INTENTIR_CMP_GT:
      return x > y;
    case INTENTIR_CMP_GE:
      return x >= y;
    case INTENTIR_CMP_NE:
      return x != y;
    default:
      return 0;
  }
}

void intentir_cmp_f32_broadcast_u8(
    const float* a, const float* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  if (!a || !b || !out || !out_shape || !a_shape || !b_shape) return;
  if (rank < 1 || rank > 4) return;
  const size_t n = intentir_numel_rank(out_shape, rank);
#if defined(__riscv_vector) || defined(__riscv_v)
  if (!intentir_shapes_equal(a_shape, out_shape, rank) || !intentir_shapes_equal(b_shape, out_shape, rank)) {
    intentir_cmp_f32_broadcast_vec(a, b, out, out_shape, a_shape, b_shape, rank, op);
    return;
  }
  if (intentir_shapes_equal(a_shape, out_shape, rank) && intentir_shapes_equal(b_shape, out_shape, rank)) {
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e32m1(n - i);
      vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[i], vl);
      vfloat32m1_t vb = __riscv_vle32_v_f32m1(&b[i], vl);
      vbool32_t m;
      if (op == INTENTIR_CMP_LT) m = __riscv_vmflt_vv_f32m1_b32(va, vb, vl);
      else if (op == INTENTIR_CMP_LE) m = __riscv_vmfle_vv_f32m1_b32(va, vb, vl);
      else if (op == INTENTIR_CMP_GT) m = __riscv_vmfgt_vv_f32m1_b32(va, vb, vl);
      else if (op == INTENTIR_CMP_GE) m = __riscv_vmfge_vv_f32m1_b32(va, vb, vl);
      else if (op == INTENTIR_CMP_NE) m = __riscv_vmfne_vv_f32m1_b32(va, vb, vl);
      else m = __riscv_vmfne_vv_f32m1_b32(va, vb, vl);
      vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
      vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
      vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
      __riscv_vse8_v_u8mf4(&out[i], vo, vl);
      i += vl;
    }
    return;
  }
#endif
  int64_t coords[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    intentir_unravel_index(i, out_shape, rank, coords);
    size_t ai = intentir_ravel_broadcast(coords, a_shape, rank);
    size_t bi = intentir_ravel_broadcast(coords, b_shape, rank);
    out[i] = intentir_apply_cmp_f32(a[ai], b[bi], op) ? 1 : 0;
  }
}

void intentir_cmp_i32_broadcast_u8(
    const int32_t* a, const int32_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  if (!a || !b || !out || !out_shape || !a_shape || !b_shape) return;
  if (rank < 1 || rank > 4) return;
  const size_t n = intentir_numel_rank(out_shape, rank);
#if defined(__riscv_vector) || defined(__riscv_v)
  if (!intentir_shapes_equal(a_shape, out_shape, rank) || !intentir_shapes_equal(b_shape, out_shape, rank)) {
    intentir_cmp_i32_broadcast_vec(a, b, out, out_shape, a_shape, b_shape, rank, op);
    return;
  }
  if (intentir_shapes_equal(a_shape, out_shape, rank) && intentir_shapes_equal(b_shape, out_shape, rank)) {
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e32m1(n - i);
      vint32m1_t va = __riscv_vle32_v_i32m1(&a[i], vl);
      vint32m1_t vb = __riscv_vle32_v_i32m1(&b[i], vl);
      vbool32_t m;
      if (op == INTENTIR_CMP_LT) {
        m = __riscv_vmslt_vv_i32m1_b32(va, vb, vl);
      } else if (op == INTENTIR_CMP_LE) {
        vbool32_t lt = __riscv_vmslt_vv_i32m1_b32(va, vb, vl);
        vbool32_t eq = __riscv_vmseq_vv_i32m1_b32(va, vb, vl);
        m = __riscv_vmor_mm_b32(lt, eq, vl);
      } else if (op == INTENTIR_CMP_GT) {
        m = __riscv_vmslt_vv_i32m1_b32(vb, va, vl);
      } else if (op == INTENTIR_CMP_GE) {
        vbool32_t gt = __riscv_vmslt_vv_i32m1_b32(vb, va, vl);
        vbool32_t eq = __riscv_vmseq_vv_i32m1_b32(va, vb, vl);
        m = __riscv_vmor_mm_b32(gt, eq, vl);
      } else if (op == INTENTIR_CMP_NE) {
        m = __riscv_vmsne_vv_i32m1_b32(va, vb, vl);
      } else {
        m = __riscv_vmsne_vv_i32m1_b32(va, vb, vl);
      }
      vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
      vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
      vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
      __riscv_vse8_v_u8mf4(&out[i], vo, vl);
      i += vl;
    }
    return;
  }
#endif
  int64_t coords[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    intentir_unravel_index(i, out_shape, rank, coords);
    size_t ai = intentir_ravel_broadcast(coords, a_shape, rank);
    size_t bi = intentir_ravel_broadcast(coords, b_shape, rank);
    out[i] = intentir_apply_cmp_i32(a[ai], b[bi], op) ? 1 : 0;
  }
}

void intentir_bool_bin_broadcast_u8(
    const uint8_t* a, const uint8_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  if (!a || !b || !out || !out_shape || !a_shape || !b_shape) return;
  if (rank < 1 || rank > 4) return;
  const size_t n = intentir_numel_rank(out_shape, rank);
#if defined(__riscv_vector) || defined(__riscv_v)
  if (!intentir_shapes_equal(a_shape, out_shape, rank) || !intentir_shapes_equal(b_shape, out_shape, rank)) {
    intentir_bool_bin_broadcast_vec(a, b, out, out_shape, a_shape, b_shape, rank, op);
    return;
  }
  if (intentir_shapes_equal(a_shape, out_shape, rank) && intentir_shapes_equal(b_shape, out_shape, rank)) {
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e8m1(n - i);
      vuint8m1_t va = __riscv_vle8_v_u8m1(&a[i], vl);
      vuint8m1_t vb = __riscv_vle8_v_u8m1(&b[i], vl);
      vbool8_t ma = __riscv_vmsne_vx_u8m1_b8(va, 0, vl);
      vbool8_t mb = __riscv_vmsne_vx_u8m1_b8(vb, 0, vl);
      vbool8_t m = (op == INTENTIR_BOOL_BIN_OR) ? __riscv_vmor_mm_b8(ma, mb, vl) : __riscv_vmand_mm_b8(ma, mb, vl);
      vuint8m1_t ones = __riscv_vmv_v_x_u8m1(1, vl);
      vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
      vuint8m1_t vo = __riscv_vmerge_vvm_u8m1(zeros, ones, m, vl);
      __riscv_vse8_v_u8m1(&out[i], vo, vl);
      i += vl;
    }
    return;
  }
#endif
  int64_t coords[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    intentir_unravel_index(i, out_shape, rank, coords);
    size_t ai = intentir_ravel_broadcast(coords, a_shape, rank);
    size_t bi = intentir_ravel_broadcast(coords, b_shape, rank);
    int av = (a[ai] != 0);
    int bv = (b[bi] != 0);
    int r = 0;
    if (op == INTENTIR_BOOL_BIN_AND) r = av && bv;
    else if (op == INTENTIR_BOOL_BIN_OR) r = av || bv;
    out[i] = r ? 1 : 0;
  }
}

static inline double intentir_read_as_f64(const void* p, size_t i, int from_type) {
  switch (from_type) {
    case INTENTIR_TYPE_U8:
      return (double)((const uint8_t*)p)[i];
    case INTENTIR_TYPE_I8:
      return (double)((const int8_t*)p)[i];
    case INTENTIR_TYPE_I32:
      return (double)((const int32_t*)p)[i];
    case INTENTIR_TYPE_I64:
      return (double)((const int64_t*)p)[i];
    case INTENTIR_TYPE_F32:
      return (double)((const float*)p)[i];
    case INTENTIR_TYPE_F64:
      return ((const double*)p)[i];
    default:
      return 0.0;
  }
}

void intentir_cast_1d(const void* inp, void* out, size_t n, int from_type, int to_type) {
  if (!inp || !out || n == 0) return;

  if (to_type == INTENTIR_TYPE_U8) {
    uint8_t* o = (uint8_t*)out;
#if defined(__riscv_vector) || defined(__riscv_v)
    if (from_type == INTENTIR_TYPE_F32) {
      const float* a = (const float*)inp;
      for (size_t i = 0; i < n;) {
        size_t vl = intentir_vsetvl_e32m1(n - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vbool32_t m = __riscv_vmfne_vv_f32m1_b32(vx, v0, vl);
        vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, vl);
        vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
        vuint8mf4_t vo = __riscv_vmerge_vvm_u8mf4(zeros, ones, m, vl);
        __riscv_vse8_v_u8mf4(&o[i], vo, vl);
        i += vl;
      }
      return;
    }
#endif
    for (size_t i = 0; i < n; ++i) {
      double v = intentir_read_as_f64(inp, i, from_type);
      o[i] = (v != 0.0) ? 1 : 0;
    }
    return;
  }

  if (from_type == INTENTIR_TYPE_U8 && (to_type == INTENTIR_TYPE_F32 || to_type == INTENTIR_TYPE_F64)) {
    const uint8_t* a = (const uint8_t*)inp;
    if (to_type == INTENTIR_TYPE_F32) {
      float* o = (float*)out;
#if defined(__riscv_vector) || defined(__riscv_v)
      for (size_t i = 0; i < n;) {
        size_t vl = intentir_vsetvl_e32m1(n - i);
        vuint8mf4_t vc = __riscv_vle8_v_u8mf4(&a[i], vl);
        vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
        vfloat32m1_t zeros = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t ones = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(zeros, ones, m, vl);
        __riscv_vse32_v_f32m1(&o[i], vo, vl);
        i += vl;
      }
      return;
#endif
      for (size_t i = 0; i < n; ++i) o[i] = (a[i] != 0) ? 1.0f : 0.0f;
    } else {
      double* o = (double*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (a[i] != 0) ? 1.0 : 0.0;
    }
    return;
  }

  switch (to_type) {
    case INTENTIR_TYPE_I8: {
      int8_t* o = (int8_t*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (int8_t)intentir_read_as_f64(inp, i, from_type);
      return;
    }
    case INTENTIR_TYPE_I32: {
      int32_t* o = (int32_t*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (int32_t)intentir_read_as_f64(inp, i, from_type);
      return;
    }
    case INTENTIR_TYPE_I64: {
      int64_t* o = (int64_t*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (int64_t)intentir_read_as_f64(inp, i, from_type);
      return;
    }
    case INTENTIR_TYPE_F32: {
      float* o = (float*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (float)intentir_read_as_f64(inp, i, from_type);
      return;
    }
    case INTENTIR_TYPE_F64: {
      double* o = (double*)out;
      for (size_t i = 0; i < n; ++i) o[i] = (double)intentir_read_as_f64(inp, i, from_type);
      return;
    }
    default:
      return;
  }
}

void intentir_iota_i32(int32_t* out, const int64_t* out_shape, int rank, int axis) {
  if (!out || !out_shape) return;
  if (rank < 1 || rank > 4) return;
  if (axis < 0 || axis >= rank) return;
  const size_t n = intentir_numel_rank(out_shape, rank);
  int64_t coords[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    intentir_unravel_index(i, out_shape, rank, coords);
    out[i] = (int32_t)coords[axis];
  }
}

void intentir_gather_f32_i32(
    const float* data, float* out, const int32_t* const* idxs, int data_rank, const int64_t* data_shape, int data_shape_rank,
    const int64_t* out_shape, int out_rank, const int64_t* idx_shapes_flat) {
  if (!data || !out || !idxs || !data_shape || !out_shape || !idx_shapes_flat) return;
  if (data_rank < 1 || data_rank > 4) return;
  if (data_shape_rank != data_rank) return;
  if (out_rank < 1 || out_rank > 4) return;

#if defined(__riscv_vector) || defined(__riscv_v)
  // Fast path: vectorize along the innermost output dimension using indexed loads.
  // This assumes indices are in-bounds (frontends clamp/mask where needed).
  {
    int64_t data_numel = 1;
    for (int ax = 0; ax < data_rank; ++ax) {
      if (data_shape[ax] <= 0) return;
      if (data_numel > (INT64_MAX / data_shape[ax])) {
        data_numel = 0;
        break;
      }
      data_numel *= data_shape[ax];
    }
    if (data_numel > 0 && (uint64_t)data_numel * (uint64_t)sizeof(float) < (uint64_t)UINT32_MAX) {
      const int last_od = out_rank - 1;
      const int64_t last_dim = out_shape[last_od];
      if (last_dim > 0) {
        int64_t idx_strides[4][4] = {{0}};
        int idx_last_var[4] = {0, 0, 0, 0};
        for (int ax = 0; ax < data_rank; ++ax) {
          const int64_t* ish = &idx_shapes_flat[ax * out_rank];
          int64_t s = 1;
          for (int k = out_rank - 1; k >= 0; --k) {
            idx_strides[ax][k] = s;
            s *= ish[k];
          }
          idx_last_var[ax] = (ish[last_od] != 1);
        }

        const uint32_t dim1 = (data_rank >= 2) ? (uint32_t)data_shape[1] : 1u;
        const uint32_t dim2 = (data_rank >= 3) ? (uint32_t)data_shape[2] : 1u;
        const uint32_t dim3 = (data_rank >= 4) ? (uint32_t)data_shape[3] : 1u;

        if (out_rank == 1) {
          size_t base[4] = {0, 0, 0, 0};
          for (int ax = 0; ax < data_rank; ++ax) base[ax] = 0;
          for (int64_t o = 0; o < last_dim;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o));
            vuint32m1_t i0 = __riscv_vmv_v_x_u32m1(0, vl);
            vuint32m1_t i1 = __riscv_vmv_v_x_u32m1(0, vl);
            vuint32m1_t i2 = __riscv_vmv_v_x_u32m1(0, vl);
            vuint32m1_t i3 = __riscv_vmv_v_x_u32m1(0, vl);
            if (data_rank >= 1) {
              i0 = idx_last_var[0] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[0][base[0] + (size_t)o], vl)
                                   : __riscv_vmv_v_x_u32m1((uint32_t)idxs[0][base[0]], vl);
            }
            if (data_rank >= 2) {
              i1 = idx_last_var[1] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[1][base[1] + (size_t)o], vl)
                                   : __riscv_vmv_v_x_u32m1((uint32_t)idxs[1][base[1]], vl);
            }
            if (data_rank >= 3) {
              i2 = idx_last_var[2] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[2][base[2] + (size_t)o], vl)
                                   : __riscv_vmv_v_x_u32m1((uint32_t)idxs[2][base[2]], vl);
            }
            if (data_rank >= 4) {
              i3 = idx_last_var[3] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[3][base[3] + (size_t)o], vl)
                                   : __riscv_vmv_v_x_u32m1((uint32_t)idxs[3][base[3]], vl);
            }
            vuint32m1_t di = i0;
            if (data_rank >= 2) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim1, vl), i1, vl);
            if (data_rank >= 3) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim2, vl), i2, vl);
            if (data_rank >= 4) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim3, vl), i3, vl);
            vuint32m1_t off = __riscv_vsll_vx_u32m1(di, 2, vl);
            vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(data, off, vl);
            __riscv_vse32_v_f32m1(&out[(size_t)o], vx, vl);
            o += (int64_t)vl;
          }
          return;
        }

        if (out_rank == 2) {
          const int64_t D0 = out_shape[0];
          for (int64_t o0 = 0; o0 < D0; ++o0) {
            const size_t ob = (size_t)o0 * (size_t)last_dim;
            size_t base[4] = {0, 0, 0, 0};
            for (int ax = 0; ax < data_rank; ++ax) {
              const int64_t* ish = &idx_shapes_flat[ax * out_rank];
              size_t b = 0;
              if (ish[0] != 1) b += (size_t)o0 * (size_t)idx_strides[ax][0];
              base[ax] = b;
            }
            for (int64_t o1 = 0; o1 < last_dim;) {
              size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o1));
              vuint32m1_t i0 = __riscv_vmv_v_x_u32m1(0, vl);
              vuint32m1_t i1 = __riscv_vmv_v_x_u32m1(0, vl);
              vuint32m1_t i2 = __riscv_vmv_v_x_u32m1(0, vl);
              vuint32m1_t i3 = __riscv_vmv_v_x_u32m1(0, vl);
              if (data_rank >= 1) {
                i0 = idx_last_var[0] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[0][base[0] + (size_t)o1], vl)
                                     : __riscv_vmv_v_x_u32m1((uint32_t)idxs[0][base[0]], vl);
              }
              if (data_rank >= 2) {
                i1 = idx_last_var[1] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[1][base[1] + (size_t)o1], vl)
                                     : __riscv_vmv_v_x_u32m1((uint32_t)idxs[1][base[1]], vl);
              }
              if (data_rank >= 3) {
                i2 = idx_last_var[2] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[2][base[2] + (size_t)o1], vl)
                                     : __riscv_vmv_v_x_u32m1((uint32_t)idxs[2][base[2]], vl);
              }
              if (data_rank >= 4) {
                i3 = idx_last_var[3] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[3][base[3] + (size_t)o1], vl)
                                     : __riscv_vmv_v_x_u32m1((uint32_t)idxs[3][base[3]], vl);
              }

              vuint32m1_t di = i0;
              if (data_rank >= 2) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim1, vl), i1, vl);
              if (data_rank >= 3) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim2, vl), i2, vl);
              if (data_rank >= 4) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim3, vl), i3, vl);

              vuint32m1_t off = __riscv_vsll_vx_u32m1(di, 2, vl);
              vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(data, off, vl);
              __riscv_vse32_v_f32m1(&out[ob + (size_t)o1], vx, vl);
              o1 += (int64_t)vl;
            }
          }
          return;
        }

        if (out_rank == 3) {
          const int64_t D0 = out_shape[0], D1 = out_shape[1];
          for (int64_t o0 = 0; o0 < D0; ++o0) {
            for (int64_t o1 = 0; o1 < D1; ++o1) {
              const size_t ob = ((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)last_dim;
              size_t base[4] = {0, 0, 0, 0};
              for (int ax = 0; ax < data_rank; ++ax) {
                const int64_t* ish = &idx_shapes_flat[ax * out_rank];
                size_t b = 0;
                if (ish[0] != 1) b += (size_t)o0 * (size_t)idx_strides[ax][0];
                if (ish[1] != 1) b += (size_t)o1 * (size_t)idx_strides[ax][1];
                base[ax] = b;
              }
              for (int64_t o2 = 0; o2 < last_dim;) {
                size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o2));
                vuint32m1_t i0 = __riscv_vmv_v_x_u32m1(0, vl);
                vuint32m1_t i1 = __riscv_vmv_v_x_u32m1(0, vl);
                vuint32m1_t i2 = __riscv_vmv_v_x_u32m1(0, vl);
                vuint32m1_t i3 = __riscv_vmv_v_x_u32m1(0, vl);
                if (data_rank >= 1) {
                  i0 = idx_last_var[0] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[0][base[0] + (size_t)o2], vl)
                                       : __riscv_vmv_v_x_u32m1((uint32_t)idxs[0][base[0]], vl);
                }
                if (data_rank >= 2) {
                  i1 = idx_last_var[1] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[1][base[1] + (size_t)o2], vl)
                                       : __riscv_vmv_v_x_u32m1((uint32_t)idxs[1][base[1]], vl);
                }
                if (data_rank >= 3) {
                  i2 = idx_last_var[2] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[2][base[2] + (size_t)o2], vl)
                                       : __riscv_vmv_v_x_u32m1((uint32_t)idxs[2][base[2]], vl);
                }
                if (data_rank >= 4) {
                  i3 = idx_last_var[3] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[3][base[3] + (size_t)o2], vl)
                                       : __riscv_vmv_v_x_u32m1((uint32_t)idxs[3][base[3]], vl);
                }

                vuint32m1_t di = i0;
                if (data_rank >= 2) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim1, vl), i1, vl);
                if (data_rank >= 3) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim2, vl), i2, vl);
                if (data_rank >= 4) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim3, vl), i3, vl);

                vuint32m1_t off = __riscv_vsll_vx_u32m1(di, 2, vl);
                vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(data, off, vl);
                __riscv_vse32_v_f32m1(&out[ob + (size_t)o2], vx, vl);
                o2 += (int64_t)vl;
              }
            }
          }
          return;
        }

        if (out_rank == 4) {
          const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
          for (int64_t o0 = 0; o0 < D0; ++o0) {
            for (int64_t o1 = 0; o1 < D1; ++o1) {
              for (int64_t o2 = 0; o2 < D2; ++o2) {
                const size_t ob = (((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2) * (size_t)last_dim;
                size_t base[4] = {0, 0, 0, 0};
                for (int ax = 0; ax < data_rank; ++ax) {
                  const int64_t* ish = &idx_shapes_flat[ax * out_rank];
                  size_t b = 0;
                  if (ish[0] != 1) b += (size_t)o0 * (size_t)idx_strides[ax][0];
                  if (ish[1] != 1) b += (size_t)o1 * (size_t)idx_strides[ax][1];
                  if (ish[2] != 1) b += (size_t)o2 * (size_t)idx_strides[ax][2];
                  base[ax] = b;
                }
                for (int64_t o3 = 0; o3 < last_dim;) {
                  size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o3));
                  vuint32m1_t i0 = __riscv_vmv_v_x_u32m1(0, vl);
                  vuint32m1_t i1 = __riscv_vmv_v_x_u32m1(0, vl);
                  vuint32m1_t i2 = __riscv_vmv_v_x_u32m1(0, vl);
                  vuint32m1_t i3 = __riscv_vmv_v_x_u32m1(0, vl);
                  if (data_rank >= 1) {
                    i0 = idx_last_var[0] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[0][base[0] + (size_t)o3], vl)
                                         : __riscv_vmv_v_x_u32m1((uint32_t)idxs[0][base[0]], vl);
                  }
                  if (data_rank >= 2) {
                    i1 = idx_last_var[1] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[1][base[1] + (size_t)o3], vl)
                                         : __riscv_vmv_v_x_u32m1((uint32_t)idxs[1][base[1]], vl);
                  }
                  if (data_rank >= 3) {
                    i2 = idx_last_var[2] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[2][base[2] + (size_t)o3], vl)
                                         : __riscv_vmv_v_x_u32m1((uint32_t)idxs[2][base[2]], vl);
                  }
                  if (data_rank >= 4) {
                    i3 = idx_last_var[3] ? __riscv_vle32_v_u32m1((const uint32_t*)&idxs[3][base[3] + (size_t)o3], vl)
                                         : __riscv_vmv_v_x_u32m1((uint32_t)idxs[3][base[3]], vl);
                  }

                  vuint32m1_t di = i0;
                  if (data_rank >= 2) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim1, vl), i1, vl);
                  if (data_rank >= 3) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim2, vl), i2, vl);
                  if (data_rank >= 4) di = __riscv_vadd_vv_u32m1(__riscv_vmul_vx_u32m1(di, dim3, vl), i3, vl);

                  vuint32m1_t off = __riscv_vsll_vx_u32m1(di, 2, vl);
                  vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(data, off, vl);
                  __riscv_vse32_v_f32m1(&out[ob + (size_t)o3], vx, vl);
                  o3 += (int64_t)vl;
                }
              }
            }
          }
          return;
        }
      }
    }
  }
#endif

  const size_t n = intentir_numel_rank(out_shape, out_rank);
  int64_t coords[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    intentir_unravel_index(i, out_shape, out_rank, coords);

    int32_t di32[4] = {0, 0, 0, 0};
    for (int ax = 0; ax < data_rank; ++ax) {
      const int64_t* ish = &idx_shapes_flat[ax * out_rank];
      size_t ii = intentir_ravel_broadcast(coords, ish, out_rank);
      di32[ax] = idxs[ax][ii];
    }

    size_t di = 0;
    for (int ax = 0; ax < data_rank; ++ax) {
      size_t dim = (size_t)data_shape[ax];
      size_t c = (size_t)di32[ax];
      di = di * dim + c;
    }
    out[i] = data[di];
  }
}

void intentir_reduce_any_2d_axis1_u8(const uint8_t* a, uint8_t* out, int64_t M, int64_t K) {
  if (!a || !out || M <= 0 || K <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (int64_t m = 0; m < M; ++m) {
    const uint8_t* row = &a[(size_t)m * (size_t)K];
    uint8_t acc = 0;
    for (int64_t k = 0; k < K;) {
      size_t vl = intentir_vsetvl_e8m1((size_t)(K - k));
      vuint8m1_t vx = __riscv_vle8_v_u8m1(&row[(size_t)k], vl);
      vbool8_t msk = __riscv_vmsne_vx_u8m1_b8(vx, 0, vl);
      size_t cnt = __riscv_vcpop_m_b8(msk, vl);
      if (cnt != 0) {
        acc = 1;
        break;
      }
      k += (int64_t)vl;
    }
    out[(size_t)m] = acc;
  }
  return;
#endif
  for (int64_t m = 0; m < M; ++m) {
    uint8_t acc = 0;
    for (int64_t k = 0; k < K; ++k) {
      if (a[idx2((int)m, (int)k, (int)K)] != 0) {
        acc = 1;
        break;
      }
    }
    out[(size_t)m] = acc;
  }
}

void intentir_abs_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vy = __riscv_vfabs_v_f32m1(vx, vl);
        __riscv_vse32_v_f32m1(&out[i], vy, vl);
        i += vl;
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vy = __riscv_vfabs_v_f32m1(vx, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
  for (size_t i = 0; i < n; ++i) out[i] = fabsf(a[i]);
#endif
}

void intentir_floor_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        // rtz trunc -> floor correction for negative, non-integer values.
        vint32m1_t vi = __riscv_vfcvt_x_f_v_i32m1(vx, vl);
        vfloat32m1_t vt = __riscv_vfcvt_f_x_v_f32m1(vi, vl);
        vbool32_t m = __riscv_vmflt_vv_f32m1_b32(vx, vt, vl);  // true when vx is negative with fractional part
        vfloat32m1_t vt1 = __riscv_vfadd_vf_f32m1(vt, -1.0f, vl);
        vfloat32m1_t vy = __riscv_vmerge_vvm_f32m1(vt, vt1, m, vl);
        __riscv_vse32_v_f32m1(&out[i], vy, vl);
        i += vl;
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    // rtz trunc -> floor correction for negative, non-integer values.
    vint32m1_t vi = __riscv_vfcvt_x_f_v_i32m1(vx, vl);
    vfloat32m1_t vt = __riscv_vfcvt_f_x_v_f32m1(vi, vl);
    vbool32_t m = __riscv_vmflt_vv_f32m1_b32(vx, vt, vl);  // true when vx is negative with fractional part
    vfloat32m1_t vt1 = __riscv_vfadd_vf_f32m1(vt, -1.0f, vl);
    vfloat32m1_t vy = __riscv_vmerge_vvm_f32m1(vt, vt1, m, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
  for (size_t i = 0; i < n; ++i) out[i] = floorf(a[i]);
#endif
}

void intentir_rsqrt_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vs = __riscv_vfsqrt_v_f32m1(vx, vl);
        vfloat32m1_t vy = __riscv_vfrdiv_vf_f32m1(vs, 1.0f, vl);
        __riscv_vse32_v_f32m1(&out[i], vy, vl);
        i += vl;
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vs = __riscv_vfsqrt_v_f32m1(vx, vl);
    vfloat32m1_t vy = __riscv_vfrdiv_vf_f32m1(vs, 1.0f, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / sqrtf(a[i]);
#endif
}

void intentir_exp_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t vy = intentir_vexp_approx_f32m1(vx, vl);
        __riscv_vse32_v_f32m1(&out[i], vy, vl);
        i += vl;
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vy = intentir_vexp_approx_f32m1(vx, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
  return;
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
  for (size_t i = 0; i < n; ++i) out[i] = expf(a[i]);
}

void intentir_relu_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
  if (n >= 16384) {
#pragma omp parallel
    {
      size_t tid = (size_t)omp_get_thread_num();
      size_t nt = (size_t)omp_get_num_threads();
      size_t chunk = (n + nt - 1) / nt;
      size_t i0 = tid * chunk;
      size_t i1 = i0 + chunk;
      if (i1 > n) i1 = n;
      for (size_t i = i0; i < i1;) {
        size_t vl = intentir_vsetvl_e32m1(i1 - i);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
        vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vy = __riscv_vfmax_vv_f32m1(vx, v0, vl);
        __riscv_vse32_v_f32m1(&out[i], vy, vl);
        i += vl;
      }
    }
    return;
  }
#endif
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vy = __riscv_vfmax_vv_f32m1(vx, v0, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
  for (size_t i = 0; i < n; ++i) {
    float v = a[i];
    out[i] = v > 0.0f ? v : 0.0f;
  }
#endif
}

void intentir_transpose_4d_0132_f32(const float* inp, float* out, int64_t B, int64_t H, int64_t K, int64_t D) {
  if (!inp || !out || B <= 0 || H <= 0 || K <= 0 || D <= 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  // [B,H,K,D] -> [B,H,D,K] (perm 0,1,3,2)
  // Vectorize over K: strided load from inp (stride=D) and contiguous store to out.
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t d = 0; d < D; ++d) {
        const size_t in_base = (((size_t)b * (size_t)H + (size_t)h) * (size_t)K) * (size_t)D + (size_t)d;
        const size_t out_base = (((size_t)b * (size_t)H + (size_t)h) * (size_t)D + (size_t)d) * (size_t)K;
        for (int64_t k = 0; k < K;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)(K - k));
          const float* p = &inp[in_base + (size_t)k * (size_t)D];
          vfloat32m1_t vx = __riscv_vlse32_v_f32m1(p, (ptrdiff_t)((size_t)D * sizeof(float)), vl);
          __riscv_vse32_v_f32m1(&out[out_base + (size_t)k], vx, vl);
          k += (int64_t)vl;
        }
      }
    }
  }
  return;
#endif
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t k = 0; k < K; ++k) {
        for (int64_t d = 0; d < D; ++d) {
          size_t oi = (((size_t)b * (size_t)H + (size_t)h) * (size_t)D + (size_t)d) * (size_t)K + (size_t)k;
          size_t ii = (((size_t)b * (size_t)H + (size_t)h) * (size_t)K + (size_t)k) * (size_t)D + (size_t)d;
          out[oi] = inp[ii];
        }
      }
    }
  }
}

void intentir_transpose_f32(const float* inp, float* out, const int64_t* in_shape, const int64_t* out_shape, const int* perm, int rank) {
  if (!inp || !out || !in_shape || !out_shape || !perm) return;
  if (rank < 1 || rank > 4) return;

  if (rank == 1) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) out[(size_t)i0] = inp[(size_t)i0];
    return;
  }
  if (rank == 2) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        int64_t o[2] = {o0, o1};
        int64_t in[2] = {0, 0};
        for (int od = 0; od < 2; ++od) in[perm[od]] = o[od];
        size_t oi = (size_t)o0 * (size_t)D1 + (size_t)o1;
        size_t ii = (size_t)in[0] * (size_t)in_shape[1] + (size_t)in[1];
        out[oi] = inp[ii];
      }
    }
    return;
  }
  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        for (int64_t o2 = 0; o2 < D2; ++o2) {
          int64_t o[3] = {o0, o1, o2};
          int64_t in[3] = {0, 0, 0};
          for (int od = 0; od < 3; ++od) in[perm[od]] = o[od];
          size_t oi = ((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2;
          size_t ii = ((size_t)in[0] * (size_t)in_shape[1] + (size_t)in[1]) * (size_t)in_shape[2] + (size_t)in[2];
          out[oi] = inp[ii];
        }
      }
    }
    return;
  }
  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2], D3 = out_shape[3];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        for (int64_t o2 = 0; o2 < D2; ++o2) {
          for (int64_t o3 = 0; o3 < D3; ++o3) {
            int64_t o[4] = {o0, o1, o2, o3};
            int64_t in[4] = {0, 0, 0, 0};
            for (int od = 0; od < 4; ++od) in[perm[od]] = o[od];
            size_t oi = (((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2) * (size_t)D3 + (size_t)o3;
            size_t ii = (((size_t)in[0] * (size_t)in_shape[1] + (size_t)in[1]) * (size_t)in_shape[2] + (size_t)in[2]) *
                             (size_t)in_shape[3] +
                         (size_t)in[3];
            out[oi] = inp[ii];
          }
        }
      }
    }
    return;
  }
}

void intentir_where_broadcast_f32(
    const uint8_t* cond, const float* x, const float* y, float* out, const int64_t* out_shape, const int64_t* cond_shape,
    const int64_t* x_shape, const int64_t* y_shape, int rank) {
  if (!cond || !x || !y || !out || !out_shape || !cond_shape || !x_shape || !y_shape) return;
  if (rank < 1 || rank > 4) return;

  const size_t n = intentir_numel_rank(out_shape, rank);
  if (intentir_shapes_equal(cond_shape, out_shape, rank) && intentir_shapes_equal(x_shape, out_shape, rank) &&
      intentir_shapes_equal(y_shape, out_shape, rank)) {
#if defined(__riscv_vector) || defined(__riscv_v)
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e32m1(n - i);
      // Use u8mf4 so its mask type is vbool32_t (matches f32m1).
      vuint8mf4_t vc = __riscv_vle8_v_u8mf4(&cond[i], vl);
      vbool32_t m = __riscv_vmsne_vx_u8mf4_b32(vc, 0, vl);
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(&x[i], vl);
      vfloat32m1_t vy = __riscv_vle32_v_f32m1(&y[i], vl);
      vfloat32m1_t vo = __riscv_vmerge_vvm_f32m1(vy, vx, m, vl);
      __riscv_vse32_v_f32m1(&out[i], vo, vl);
      i += vl;
    }
    return;
#else
    for (size_t i = 0; i < n; ++i) out[i] = (cond[i] != 0) ? x[i] : y[i];
    return;
#endif
  }

#if defined(__riscv_vector) || defined(__riscv_v)
  intentir_where_broadcast_vec(cond, x, y, out, out_shape, cond_shape, x_shape, y_shape, rank);
  return;
#endif

  if (rank == 1) {
    const int64_t D0 = out_shape[0];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      int64_t c0 = (cond_shape[0] == 1) ? 0 : i0;
      int64_t x0 = (x_shape[0] == 1) ? 0 : i0;
      int64_t y0 = (y_shape[0] == 1) ? 0 : i0;
      out[(size_t)i0] = (cond[(size_t)c0] != 0) ? x[(size_t)x0] : y[(size_t)y0];
    }
    return;
  }
  if (rank == 2) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        int64_t c0 = (cond_shape[0] == 1) ? 0 : i0;
        int64_t c1 = (cond_shape[1] == 1) ? 0 : i1;
        int64_t x0 = (x_shape[0] == 1) ? 0 : i0;
        int64_t x1 = (x_shape[1] == 1) ? 0 : i1;
        int64_t y0 = (y_shape[0] == 1) ? 0 : i0;
        int64_t y1 = (y_shape[1] == 1) ? 0 : i1;
        size_t oi = (size_t)i0 * (size_t)D1 + (size_t)i1;
        size_t ci = (size_t)c0 * (size_t)cond_shape[1] + (size_t)c1;
        size_t xi = (size_t)x0 * (size_t)x_shape[1] + (size_t)x1;
        size_t yi = (size_t)y0 * (size_t)y_shape[1] + (size_t)y1;
        out[oi] = (cond[ci] != 0) ? x[xi] : y[yi];
      }
    }
    return;
  }
  if (rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          int64_t c0 = (cond_shape[0] == 1) ? 0 : i0;
          int64_t c1 = (cond_shape[1] == 1) ? 0 : i1;
          int64_t c2 = (cond_shape[2] == 1) ? 0 : i2;
          int64_t x0 = (x_shape[0] == 1) ? 0 : i0;
          int64_t x1 = (x_shape[1] == 1) ? 0 : i1;
          int64_t x2 = (x_shape[2] == 1) ? 0 : i2;
          int64_t y0 = (y_shape[0] == 1) ? 0 : i0;
          int64_t y1 = (y_shape[1] == 1) ? 0 : i1;
          int64_t y2 = (y_shape[2] == 1) ? 0 : i2;
          size_t oi = ((size_t)i0 * (size_t)D1 + (size_t)i1) * (size_t)D2 + (size_t)i2;
          size_t ci = ((size_t)c0 * (size_t)cond_shape[1] + (size_t)c1) * (size_t)cond_shape[2] + (size_t)c2;
          size_t xi = ((size_t)x0 * (size_t)x_shape[1] + (size_t)x1) * (size_t)x_shape[2] + (size_t)x2;
          size_t yi = ((size_t)y0 * (size_t)y_shape[1] + (size_t)y1) * (size_t)y_shape[2] + (size_t)y2;
          out[oi] = (cond[ci] != 0) ? x[xi] : y[yi];
        }
      }
    }
    return;
  }
  if (rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2], D3 = out_shape[3];
    for (int64_t i0 = 0; i0 < D0; ++i0) {
      for (int64_t i1 = 0; i1 < D1; ++i1) {
        for (int64_t i2 = 0; i2 < D2; ++i2) {
          for (int64_t i3 = 0; i3 < D3; ++i3) {
            int64_t c0 = (cond_shape[0] == 1) ? 0 : i0;
            int64_t c1 = (cond_shape[1] == 1) ? 0 : i1;
            int64_t c2 = (cond_shape[2] == 1) ? 0 : i2;
            int64_t c3 = (cond_shape[3] == 1) ? 0 : i3;
            int64_t x0 = (x_shape[0] == 1) ? 0 : i0;
            int64_t x1 = (x_shape[1] == 1) ? 0 : i1;
            int64_t x2 = (x_shape[2] == 1) ? 0 : i2;
            int64_t x3 = (x_shape[3] == 1) ? 0 : i3;
            int64_t y0 = (y_shape[0] == 1) ? 0 : i0;
            int64_t y1 = (y_shape[1] == 1) ? 0 : i1;
            int64_t y2 = (y_shape[2] == 1) ? 0 : i2;
            int64_t y3 = (y_shape[3] == 1) ? 0 : i3;
            size_t oi = (((size_t)i0 * (size_t)D1 + (size_t)i1) * (size_t)D2 + (size_t)i2) * (size_t)D3 + (size_t)i3;
            size_t ci =
                (((size_t)c0 * (size_t)cond_shape[1] + (size_t)c1) * (size_t)cond_shape[2] + (size_t)c2) * (size_t)cond_shape[3] +
                (size_t)c3;
            size_t xi = (((size_t)x0 * (size_t)x_shape[1] + (size_t)x1) * (size_t)x_shape[2] + (size_t)x2) * (size_t)x_shape[3] +
                        (size_t)x3;
            size_t yi = (((size_t)y0 * (size_t)y_shape[1] + (size_t)y1) * (size_t)y_shape[2] + (size_t)y2) * (size_t)y_shape[3] +
                        (size_t)y3;
            out[oi] = (cond[ci] != 0) ? x[xi] : y[yi];
          }
        }
      }
    }
    return;
  }
}

void intentir_broadcast_in_dim_f32(
    const float* inp, float* out, const int64_t* in_shape, int in_rank, const int64_t* out_shape, int out_rank, const int* bcast_dims) {
  if (!inp || !out || !in_shape || !out_shape || !bcast_dims) return;
  if (in_rank < 0 || out_rank < 0) return;
  if (in_rank > 4 || out_rank > 4) return;
  if (out_rank < in_rank) return;

  // Scalar broadcast (common for consts/eps): fill output with inp[0].
  int is_scalar = (in_rank == 0);
  for (int i = 0; i < in_rank; ++i) {
    if (in_shape[i] != 1) {
      is_scalar = 0;
      break;
    }
  }
  if (is_scalar) {
    const float v = inp[0];
    const size_t n = intentir_numel_rank(out_shape, out_rank);
#if defined(__riscv_vector) || defined(__riscv_v)
#ifdef _OPENMP
    if (n >= 16384) {
#pragma omp parallel
      {
        size_t tid = (size_t)omp_get_thread_num();
        size_t nt = (size_t)omp_get_num_threads();
        size_t chunk = (n + nt - 1) / nt;
        size_t i0 = tid * chunk;
        size_t i1 = i0 + chunk;
        if (i1 > n) i1 = n;
        for (size_t i = i0; i < i1;) {
          size_t vl = intentir_vsetvl_e32m1(i1 - i);
          vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(v, vl);
          __riscv_vse32_v_f32m1(&out[i], vv, vl);
          i += vl;
        }
      }
      return;
    }
#endif
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e32m1(n - i);
      vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(v, vl);
      __riscv_vse32_v_f32m1(&out[i], vv, vl);
      i += vl;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n >= 16384)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = v;
#endif
    return;
  }

  int64_t strides[4] = {1, 1, 1, 1};
  int64_t s = 1;
  for (int i = in_rank - 1; i >= 0; --i) {
    strides[i] = s;
    s *= in_shape[i];
  }

#if defined(__riscv_vector) || defined(__riscv_v)
  // Vectorized broadcast over the innermost output dimension.
  if (out_rank >= 1) {
    const int last_od = out_rank - 1;
    const int64_t last_dim = out_shape[last_od];
    if (last_dim > 0) {
      int in_last = -1;
      for (int d = 0; d < in_rank; ++d) {
        if (bcast_dims[d] == last_od) {
          in_last = d;
          break;
        }
      }
      const int64_t stride_last = (in_last >= 0 && in_shape[in_last] != 1) ? strides[in_last] : 0;

      if (out_rank == 1) {
        const size_t ob = 0;
        const size_t base0 = 0;
        for (int64_t o = 0; o < last_dim;) {
          size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o));
          if (stride_last == 0) {
            vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(inp[base0], vl);
            __riscv_vse32_v_f32m1(&out[ob + (size_t)o], vv, vl);
          } else if (stride_last == 1) {
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(&inp[base0 + (size_t)o], vl);
            __riscv_vse32_v_f32m1(&out[ob + (size_t)o], vx, vl);
          } else {
            const size_t ib = base0 + (size_t)o * (size_t)stride_last;
            vfloat32m1_t vx = __riscv_vlse32_v_f32m1(&inp[ib], (ptrdiff_t)((size_t)stride_last * sizeof(float)), vl);
            __riscv_vse32_v_f32m1(&out[ob + (size_t)o], vx, vl);
          }
          o += (int64_t)vl;
        }
        return;
      }

      if (out_rank == 2) {
        const int64_t D0 = out_shape[0];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (D0 >= 4)
#endif
        for (int64_t o0 = 0; o0 < D0; ++o0) {
          const size_t ob = (size_t)o0 * (size_t)last_dim;
          size_t base0 = 0;
          for (int d = 0; d < in_rank; ++d) {
            if (in_shape[d] == 1) continue;
            int od = bcast_dims[d];
            if (od == last_od) continue;
            base0 += (size_t)o0 * (size_t)strides[d];
          }
          for (int64_t o1 = 0; o1 < last_dim;) {
            size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o1));
            if (stride_last == 0) {
              vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(inp[base0], vl);
              __riscv_vse32_v_f32m1(&out[ob + (size_t)o1], vv, vl);
            } else if (stride_last == 1) {
              vfloat32m1_t vx = __riscv_vle32_v_f32m1(&inp[base0 + (size_t)o1], vl);
              __riscv_vse32_v_f32m1(&out[ob + (size_t)o1], vx, vl);
            } else {
              const size_t ib = base0 + (size_t)o1 * (size_t)stride_last;
              vfloat32m1_t vx = __riscv_vlse32_v_f32m1(&inp[ib], (ptrdiff_t)((size_t)stride_last * sizeof(float)), vl);
              __riscv_vse32_v_f32m1(&out[ob + (size_t)o1], vx, vl);
            }
            o1 += (int64_t)vl;
          }
        }
        return;
      }

      if (out_rank == 3) {
        const int64_t D0 = out_shape[0], D1 = out_shape[1];
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((D0 * D1) >= 4)
#endif
        for (int64_t o0 = 0; o0 < D0; ++o0) {
          for (int64_t o1 = 0; o1 < D1; ++o1) {
            const size_t ob = ((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)last_dim;
            size_t base0 = 0;
            for (int d = 0; d < in_rank; ++d) {
              if (in_shape[d] == 1) continue;
              int od = bcast_dims[d];
              if (od == last_od) continue;
              if (od == 0) base0 += (size_t)o0 * (size_t)strides[d];
              else if (od == 1) base0 += (size_t)o1 * (size_t)strides[d];
            }
            for (int64_t o2 = 0; o2 < last_dim;) {
              size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o2));
              if (stride_last == 0) {
                vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(inp[base0], vl);
                __riscv_vse32_v_f32m1(&out[ob + (size_t)o2], vv, vl);
              } else if (stride_last == 1) {
                vfloat32m1_t vx = __riscv_vle32_v_f32m1(&inp[base0 + (size_t)o2], vl);
                __riscv_vse32_v_f32m1(&out[ob + (size_t)o2], vx, vl);
              } else {
                const size_t ib = base0 + (size_t)o2 * (size_t)stride_last;
                vfloat32m1_t vx = __riscv_vlse32_v_f32m1(&inp[ib], (ptrdiff_t)((size_t)stride_last * sizeof(float)), vl);
                __riscv_vse32_v_f32m1(&out[ob + (size_t)o2], vx, vl);
              }
              o2 += (int64_t)vl;
            }
          }
        }
        return;
      }

      if (out_rank == 4) {
        const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) if (((D0 * D1) * D2) >= 4)
#endif
        for (int64_t o0 = 0; o0 < D0; ++o0) {
          for (int64_t o1 = 0; o1 < D1; ++o1) {
            for (int64_t o2 = 0; o2 < D2; ++o2) {
              const size_t ob = (((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2) * (size_t)last_dim;
              size_t base0 = 0;
              for (int d = 0; d < in_rank; ++d) {
                if (in_shape[d] == 1) continue;
                int od = bcast_dims[d];
                if (od == last_od) continue;
                if (od == 0) base0 += (size_t)o0 * (size_t)strides[d];
                else if (od == 1) base0 += (size_t)o1 * (size_t)strides[d];
                else if (od == 2) base0 += (size_t)o2 * (size_t)strides[d];
              }
              for (int64_t o3 = 0; o3 < last_dim;) {
                size_t vl = intentir_vsetvl_e32m1((size_t)(last_dim - o3));
                if (stride_last == 0) {
                  vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(inp[base0], vl);
                  __riscv_vse32_v_f32m1(&out[ob + (size_t)o3], vv, vl);
                } else if (stride_last == 1) {
                  vfloat32m1_t vx = __riscv_vle32_v_f32m1(&inp[base0 + (size_t)o3], vl);
                  __riscv_vse32_v_f32m1(&out[ob + (size_t)o3], vx, vl);
                } else {
                  const size_t ib = base0 + (size_t)o3 * (size_t)stride_last;
                  vfloat32m1_t vx = __riscv_vlse32_v_f32m1(&inp[ib], (ptrdiff_t)((size_t)stride_last * sizeof(float)), vl);
                  __riscv_vse32_v_f32m1(&out[ob + (size_t)o3], vx, vl);
                }
                o3 += (int64_t)vl;
              }
            }
          }
        }
        return;
      }
    }
  }
#endif

  if (out_rank == 0) {
    out[0] = inp[0];
    return;
  }
  if (out_rank == 1) {
    const int64_t D0 = out_shape[0];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      size_t in_i = 0;
      for (int in_d = 0; in_d < in_rank; ++in_d) {
        int od = bcast_dims[in_d];
        int64_t v = (in_shape[in_d] == 1) ? 0 : (od == 0 ? o0 : 0);
        in_i += (size_t)v * (size_t)strides[in_d];
      }
      out[(size_t)o0] = inp[in_i];
    }
    return;
  }
  if (out_rank == 2) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        size_t in_i = 0;
        for (int in_d = 0; in_d < in_rank; ++in_d) {
          int od = bcast_dims[in_d];
          int64_t v = 0;
          if (in_shape[in_d] != 1) v = (od == 0) ? o0 : (od == 1) ? o1 : 0;
          in_i += (size_t)v * (size_t)strides[in_d];
        }
        out[(size_t)o0 * (size_t)D1 + (size_t)o1] = inp[in_i];
      }
    }
    return;
  }
  if (out_rank == 3) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        for (int64_t o2 = 0; o2 < D2; ++o2) {
          size_t in_i = 0;
          for (int in_d = 0; in_d < in_rank; ++in_d) {
            int od = bcast_dims[in_d];
            int64_t v = 0;
            if (in_shape[in_d] != 1) v = (od == 0) ? o0 : (od == 1) ? o1 : (od == 2) ? o2 : 0;
            in_i += (size_t)v * (size_t)strides[in_d];
          }
          out[((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2] = inp[in_i];
        }
      }
    }
    return;
  }
  if (out_rank == 4) {
    const int64_t D0 = out_shape[0], D1 = out_shape[1], D2 = out_shape[2], D3 = out_shape[3];
    for (int64_t o0 = 0; o0 < D0; ++o0) {
      for (int64_t o1 = 0; o1 < D1; ++o1) {
        for (int64_t o2 = 0; o2 < D2; ++o2) {
          for (int64_t o3 = 0; o3 < D3; ++o3) {
            size_t in_i = 0;
            for (int in_d = 0; in_d < in_rank; ++in_d) {
              int od = bcast_dims[in_d];
              int64_t v = 0;
              if (in_shape[in_d] != 1) v = (od == 0) ? o0 : (od == 1) ? o1 : (od == 2) ? o2 : (od == 3) ? o3 : 0;
              in_i += (size_t)v * (size_t)strides[in_d];
            }
            size_t oi =
                (((size_t)o0 * (size_t)D1 + (size_t)o1) * (size_t)D2 + (size_t)o2) * (size_t)D3 + (size_t)o3;
            out[oi] = inp[in_i];
          }
        }
      }
    }
    return;
  }
}

void intentir_matmul_2d_f32(
    const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K, int transpose_a, int transpose_b,
    int64_t tile_m, int64_t tile_n, int64_t tile_k) {
  if (!a || !b || !out || M <= 0 || N <= 0 || K <= 0) return;
  const int64_t tm = (tile_m > 0) ? intentir_min_i64(tile_m, M) : M;
  const int64_t tn = (tile_n > 0) ? intentir_min_i64(tile_n, N) : N;
  const int64_t tk = (tile_k > 0) ? intentir_min_i64(tile_k, K) : K;

#if defined(__riscv_vector) || defined(__riscv_v)
  if (!transpose_a) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((M * N) >= 4096)
#endif
    for (int64_t m_base = 0; m_base < M; m_base += tm) {
      int64_t m_end = m_base + tm;
      if (m_end > M) m_end = M;
      for (int64_t n_base = 0; n_base < N; n_base += tn) {
        int64_t n_end = n_base + tn;
        if (n_end > N) n_end = N;
        for (int64_t m = m_base; m < m_end; ++m) {
          for (int64_t n0 = n_base; n0 < n_end;) {
            size_t rem = (size_t)(n_end - n0);
            size_t vl = intentir_vsetvl_e32m1(rem);
            vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
            for (int64_t k_base = 0; k_base < K; k_base += tk) {
              int64_t k_end = k_base + tk;
              if (k_end > K) k_end = K;
              for (int64_t k0 = k_base; k0 < k_end; ++k0) {
                float av = a[idx2((int)m, (int)k0, (int)K)];
                vfloat32m1_t vb;
                if (!transpose_b) {
                  vb = __riscv_vle32_v_f32m1(&b[idx2((int)k0, (int)n0, (int)N)], vl);
                } else {
                  vb = __riscv_vlse32_v_f32m1(
                      &b[idx2((int)n0, (int)k0, (int)K)], (ptrdiff_t)((size_t)K * sizeof(float)), vl);
                }
                vacc = __riscv_vfmacc_vf_f32m1(vacc, av, vb, vl);
              }
            }
            __riscv_vse32_v_f32m1(&out[idx2((int)m, (int)n0, (int)N)], vacc, vl);
            n0 += (int64_t)vl;
          }
        }
      }
    }
    return;
  }
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((M * N) >= 4096)
#endif
  for (int64_t m_base = 0; m_base < M; m_base += tm) {
    int64_t m_end = m_base + tm;
    if (m_end > M) m_end = M;
    for (int64_t n_base = 0; n_base < N; n_base += tn) {
      int64_t n_end = n_base + tn;
      if (n_end > N) n_end = N;
      for (int64_t m = m_base; m < m_end; ++m) {
        for (int64_t n0 = n_base; n0 < n_end; ++n0) {
          double acc = 0.0;
          for (int64_t k_base = 0; k_base < K; k_base += tk) {
            int64_t k_end = k_base + tk;
            if (k_end > K) k_end = K;
            for (int64_t k0 = k_base; k0 < k_end; ++k0) {
              size_t ai = (size_t)(transpose_a ? idx2((int)k0, (int)m, (int)M) : idx2((int)m, (int)k0, (int)K));
              size_t bi = (size_t)(transpose_b ? idx2((int)n0, (int)k0, (int)K) : idx2((int)k0, (int)n0, (int)N));
              acc += (double)a[ai] * (double)b[bi];
            }
          }
          out[idx2((int)m, (int)n0, (int)N)] = (float)acc;
        }
      }
    }
  }
}

void intentir_matmul_4d_f32(
    const float* a, const float* b, float* out, int64_t B, int64_t H, int64_t M, int64_t N, int64_t K, int transpose_a,
    int transpose_b, int64_t tile_m, int64_t tile_n, int64_t tile_k) {
  if (!a || !b || !out || B <= 0 || H <= 0 || M <= 0 || N <= 0 || K <= 0) return;
  const int64_t tm = (tile_m > 0) ? intentir_min_i64(tile_m, M) : M;
  const int64_t tn = (tile_n > 0) ? intentir_min_i64(tile_n, N) : N;
  const int64_t tk = (tile_k > 0) ? intentir_min_i64(tile_k, K) : K;

#if defined(__riscv_vector) || defined(__riscv_v)
  if (!transpose_a) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((B * H) >= 2)
#endif
    for (int64_t b0 = 0; b0 < B; ++b0) {
      for (int64_t h0 = 0; h0 < H; ++h0) {
        for (int64_t m_base = 0; m_base < M; m_base += tm) {
          int64_t m_end = m_base + tm;
          if (m_end > M) m_end = M;
          for (int64_t n_base = 0; n_base < N; n_base += tn) {
            int64_t n_end = n_base + tn;
            if (n_end > N) n_end = N;
            for (int64_t m0 = m_base; m0 < m_end; ++m0) {
              for (int64_t n0 = n_base; n0 < n_end;) {
                size_t rem = (size_t)(n_end - n0);
                size_t vl = intentir_vsetvl_e32m1(rem);
                vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
                for (int64_t k_base = 0; k_base < K; k_base += tk) {
                  int64_t k_end = k_base + tk;
                  if (k_end > K) k_end = K;
                  for (int64_t k0 = k_base; k0 < k_end; ++k0) {
                    float av = a[idx4((int)b0, (int)h0, (int)m0, (int)k0, (int)H, (int)M, (int)K)];
                    vfloat32m1_t vb;
                    if (!transpose_b) {
                      vb = __riscv_vle32_v_f32m1(
                          &b[idx4((int)b0, (int)h0, (int)k0, (int)n0, (int)H, (int)K, (int)N)], vl);
                    } else {
                      vb = __riscv_vlse32_v_f32m1(
                          &b[idx4((int)b0, (int)h0, (int)n0, (int)k0, (int)H, (int)N, (int)K)],
                          (ptrdiff_t)((size_t)K * sizeof(float)), vl);
                    }
                    vacc = __riscv_vfmacc_vf_f32m1(vacc, av, vb, vl);
                  }
                }
                __riscv_vse32_v_f32m1(&out[idx4((int)b0, (int)h0, (int)m0, (int)n0, (int)H, (int)M, (int)N)], vacc, vl);
                n0 += (int64_t)vl;
              }
            }
          }
        }
      }
    }
    return;
  }
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if ((B * H) >= 2)
#endif
  for (int64_t b0 = 0; b0 < B; ++b0) {
    for (int64_t h0 = 0; h0 < H; ++h0) {
      for (int64_t m_base = 0; m_base < M; m_base += tm) {
        int64_t m_end = m_base + tm;
        if (m_end > M) m_end = M;
        for (int64_t n_base = 0; n_base < N; n_base += tn) {
          int64_t n_end = n_base + tn;
          if (n_end > N) n_end = N;
          for (int64_t m0 = m_base; m0 < m_end; ++m0) {
            for (int64_t n0 = n_base; n0 < n_end; ++n0) {
              double acc = 0.0;
              for (int64_t k_base = 0; k_base < K; k_base += tk) {
                int64_t k_end = k_base + tk;
                if (k_end > K) k_end = K;
                for (int64_t k0 = k_base; k0 < k_end; ++k0) {
                  size_t ai =
                      (size_t)(transpose_a ? idx4((int)b0, (int)h0, (int)k0, (int)m0, (int)H, (int)K, (int)M)
                                           : idx4((int)b0, (int)h0, (int)m0, (int)k0, (int)H, (int)M, (int)K));
                  size_t bi =
                      (size_t)(transpose_b ? idx4((int)b0, (int)h0, (int)n0, (int)k0, (int)H, (int)N, (int)K)
                                           : idx4((int)b0, (int)h0, (int)k0, (int)n0, (int)H, (int)K, (int)N));
                  acc += (double)a[ai] * (double)b[bi];
                }
              }
              out[idx4((int)b0, (int)h0, (int)m0, (int)n0, (int)H, (int)M, (int)N)] = (float)acc;
            }
          }
        }
      }
    }
  }
}
