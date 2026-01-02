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

// Softmax over the last axis for rank-1..4 f32 tensors.
void intentir_softmax_1d_last_f32(const float* a, float* out, int64_t K);
void intentir_softmax_2d_last_f32(const float* a, float* out, int64_t M, int64_t K);
void intentir_softmax_3d_last_f32(const float* a, float* out, int64_t A0, int64_t A1, int64_t K);
void intentir_softmax_4d_last_f32(const float* a, float* out, int64_t B, int64_t H, int64_t Q, int64_t K);

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
