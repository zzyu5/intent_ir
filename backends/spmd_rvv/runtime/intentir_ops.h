#ifndef INTENTIR_OPS_H
#define INTENTIR_OPS_H

#include "intentir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reduce-sum over the last axis of a 2D f32 tensor [M,K]. Output is length M.
void intentir_reduce_sum_2d_axis1_f32(const float* a, float* out, int64_t M, int64_t K, float scale, int has_scale);

// Reduce-sum over the last 2 axes of a 4D f32 tensor [N,G,GS,HW]. Output is length N*G.
void intentir_reduce_sum_4d_axis23_f32(
    const float* a, float* out, int64_t N, int64_t G, int64_t GS, int64_t HW, float scale, int has_scale);

// Reduce-max over the last axis of a 2D f32 tensor [M,K]. Output is length M.
void intentir_reduce_max_2d_axis1_f32(const float* a, float* out, int64_t M, int64_t K);

// LayerNorm forward (f32), fused: Y = (X-mean)*rstd*W + B, also writes Mean/Rstd per row.
// Shapes: X,Y are [M,N], W,B are [N], Mean/Rstd are [M].
void intentir_layernorm_2d_f32(
    const float* X, float* Y, const float* W, const float* B, float* Mean, float* Rstd, int64_t M, int64_t N, float eps);

// Dropout (f32): Y[i] = (rand(seed, i) > p) ? X[i] / (1-p) : 0
// Uses Triton-compatible Philox (n_rounds default is 10).
void intentir_dropout_f32(const float* X, float* Y, size_t n, float p, uint64_t seed, int n_rounds);

// Correlation (int8): out[oc,h,w] = (sum_k src0[k,h,w] * src1[k,h,w-oc] >> out_shift).to(int8)
void intentir_correlation_i8(
    const int8_t* src0, const int8_t* src1, int8_t* out, int64_t out_channel, int64_t in_channel, int64_t height, int64_t width, int32_t out_shift);

// Resize (int8): bilinear 2x upsample (fixed-point hw_fl=7 in the baseline kernel).
void intentir_resize_bilinear2x_i8(const int8_t* src, int8_t* out, int64_t channel, int64_t height, int64_t width, int hw_fl);

// Warp (int8,int16): per-pixel horizontal warp using Q8.8 packed offsets in int16.
void intentir_warp_q8_8_i8_i16(const int8_t* src, const int16_t* offset, int8_t* out, int64_t channel, int64_t height, int64_t width);

// Softmax over the last axis for rank-1..4 f32 tensors.
void intentir_softmax_1d_last_f32(const float* a, float* out, int64_t K);
void intentir_softmax_2d_last_f32(const float* a, float* out, int64_t M, int64_t K);
void intentir_softmax_3d_last_f32(const float* a, float* out, int64_t A0, int64_t A1, int64_t K);
void intentir_softmax_4d_last_f32(const float* a, float* out, int64_t B, int64_t H, int64_t Q, int64_t K);

// Elementwise (f32) with numpy-style broadcasting (rank<=4).
// Shapes are length `rank` and follow the codegen's padded broadcasting convention.
#define INTENTIR_F32_BIN_ADD 0
#define INTENTIR_F32_BIN_SUB 1
#define INTENTIR_F32_BIN_MUL 2
#define INTENTIR_F32_BIN_DIV 3
#define INTENTIR_F32_BIN_MAX 4
#define INTENTIR_F32_BIN_MIN 5
void intentir_f32_bin_broadcast(
    const float* a, const float* b, float* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op);

// Comparisons producing bool (u8), with numpy-style broadcasting (rank<=4).
// Shapes are length `rank` and follow the codegen's padded broadcasting convention.
#define INTENTIR_CMP_LT 0
#define INTENTIR_CMP_LE 1
#define INTENTIR_CMP_GT 2
#define INTENTIR_CMP_GE 3
#define INTENTIR_CMP_NE 4
void intentir_cmp_f32_broadcast_u8(
    const float* a, const float* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op);
void intentir_cmp_i32_broadcast_u8(
    const int32_t* a, const int32_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op);

// Boolean ops (u8) with numpy-style broadcasting (rank<=4).
#define INTENTIR_BOOL_BIN_AND 0
#define INTENTIR_BOOL_BIN_OR 1
void intentir_bool_bin_broadcast_u8(
    const uint8_t* a, const uint8_t* b, uint8_t* out, const int64_t* out_shape, const int64_t* a_shape, const int64_t* b_shape, int rank, int op);

// Unary elementwise (f32).
void intentir_abs_f32(const float* a, float* out, size_t n);
void intentir_floor_f32(const float* a, float* out, size_t n);
void intentir_rsqrt_f32(const float* a, float* out, size_t n);
void intentir_exp_f32(const float* a, float* out, size_t n);
void intentir_relu_f32(const float* a, float* out, size_t n);

// Casts between simple scalar types over 1D buffers.
#define INTENTIR_TYPE_U8 0
#define INTENTIR_TYPE_I8 1
#define INTENTIR_TYPE_I32 2
#define INTENTIR_TYPE_I64 3
#define INTENTIR_TYPE_F32 4
#define INTENTIR_TYPE_F64 5
#define INTENTIR_TYPE_I16 6
void intentir_cast_1d(const void* inp, void* out, size_t n, int from_type, int to_type);

// Transpose (f32), rank<=4.
void intentir_transpose_4d_0132_f32(const float* inp, float* out, int64_t B, int64_t H, int64_t K, int64_t D);
void intentir_transpose_f32(const float* inp, float* out, const int64_t* in_shape, const int64_t* out_shape, const int* perm, int rank);

// Where (f32), with numpy-style broadcasting (rank<=4).
// Shapes are length `rank` and follow the codegen's padded broadcasting convention.
void intentir_where_broadcast_f32(
    const uint8_t* cond, const float* x, const float* y, float* out, const int64_t* out_shape, const int64_t* cond_shape,
    const int64_t* x_shape, const int64_t* y_shape, int rank);

// broadcast_in_dim (f32), rank<=4.
// bcast_dims maps each input dim -> output dim (length in_rank).
void intentir_broadcast_in_dim_f32(
    const float* inp, float* out, const int64_t* in_shape, int in_rank, const int64_t* out_shape, int out_rank, const int* bcast_dims);

// iota (i32): out[...] = index_along_axis.
void intentir_iota_i32(int32_t* out, const int64_t* out_shape, int rank, int axis);

// Gather (f32) with per-axis indices (i32), with numpy-style broadcasting of each index tensor to `out_shape`.
// idx_shapes_flat is a row-major matrix of shape [data_rank, out_rank].
void intentir_gather_f32_i32(
    const float* data, float* out, const int32_t* const* idxs, int data_rank, const int64_t* data_shape, int data_shape_rank,
    const int64_t* out_shape, int out_rank, const int64_t* idx_shapes_flat);

// Reduce-any over the last axis of a 2D u8 tensor [M,K]. Output is length M (u8 {0,1}).
void intentir_reduce_any_2d_axis1_u8(const uint8_t* a, uint8_t* out, int64_t M, int64_t K);

// Matmul (GEMM) in row-major layouts, with optional transpose flags:
// - 2D: [M,K] x [K,N] -> [M,N]
// - 4D: [B,H,M,K] x [B,H,K,N] -> [B,H,M,N]
// Transpose flags follow the IntentIR `matmul` op: transpose_a / transpose_b.
void intentir_matmul_2d_f32(
    const float* a, const float* b, float* out, int64_t M, int64_t N, int64_t K, int transpose_a, int transpose_b,
    int64_t tile_m, int64_t tile_n, int64_t tile_k);
void intentir_matmul_4d_f32(
    const float* a, const float* b, float* out, int64_t B, int64_t H, int64_t M, int64_t N, int64_t K, int transpose_a,
    int transpose_b, int64_t tile_m, int64_t tile_n, int64_t tile_k);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // INTENTIR_OPS_H
