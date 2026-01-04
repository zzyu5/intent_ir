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

void intentir_transpose_4d_0132_f32(const float* inp, float* out, int64_t B, int64_t H, int64_t K, int64_t D) {
  if (!inp || !out || B <= 0 || H <= 0 || K <= 0 || D <= 0) return;
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
    for (size_t i = 0; i < n;) {
      size_t vl = intentir_vsetvl_e32m1(n - i);
      vfloat32m1_t vv = __riscv_vfmv_v_f_f32m1(v, vl);
      __riscv_vse32_v_f32m1(&out[i], vv, vl);
      i += vl;
    }
#else
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
