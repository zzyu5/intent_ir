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

void intentir_softmax_1d_last_f32(const float* a, float* out, int64_t K) {
  if (!a || !out || K <= 0) return;
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
}

void intentir_softmax_2d_last_f32(const float* a, float* out, int64_t M, int64_t K) {
  if (!a || !out || M <= 0 || K <= 0) return;
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

void intentir_f32_bin_broadcast(
    const float* a, const float* b, float* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op) {
  if (!a || !b || !out || !out_shape || !a_shape || !b_shape) return;
  if (rank < 1 || rank > 4) return;

  const size_t n = intentir_numel_rank(out_shape, rank);
  if (intentir_shapes_equal(a_shape, out_shape, rank) && intentir_shapes_equal(b_shape, out_shape, rank)) {
    intentir_f32_bin_contig(a, b, out, n, op);
    return;
  }

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

void intentir_abs_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vy = __riscv_vfabs_v_f32m1(vx, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
  for (size_t i = 0; i < n; ++i) out[i] = fabsf(a[i]);
#endif
}

void intentir_floor_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) out[i] = floorf(a[i]);
}

void intentir_rsqrt_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t vs = __riscv_vfsqrt_v_f32m1(vx, vl);
    vfloat32m1_t vy = __riscv_vfrdiv_vf_f32m1(vs, 1.0f, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / sqrtf(a[i]);
#endif
}

void intentir_exp_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) out[i] = expf(a[i]);
}

void intentir_relu_f32(const float* a, float* out, size_t n) {
  if (!a || !out || n == 0) return;
#if defined(__riscv_vector) || defined(__riscv_v)
  for (size_t i = 0; i < n;) {
    size_t vl = intentir_vsetvl_e32m1(n - i);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(&a[i], vl);
    vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vy = __riscv_vfmax_vv_f32m1(vx, v0, vl);
    __riscv_vse32_v_f32m1(&out[i], vy, vl);
    i += vl;
  }
#else
  for (size_t i = 0; i < n; ++i) {
    float v = a[i];
    out[i] = v > 0.0f ? v : 0.0f;
  }
#endif
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
