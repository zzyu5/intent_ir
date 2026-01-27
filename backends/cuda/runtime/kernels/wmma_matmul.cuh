#pragma once

#include <mma.h>

#include "intentir_cuda_ops.cuh"

namespace intentir_cuda {

enum class CpAsyncPolicy : int { CA = 0, CG = 1 };

template <CpAsyncPolicy POLICY>
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
  if constexpr (POLICY == CpAsyncPolicy::CA) {
    intentir_cp_async_ca_16(smem_dst, gmem_src);
  } else {
    intentir_cp_async_cg_16(smem_dst, gmem_src);
  }
}

template <int TILE_M, int TILE_N, int STAGE_K, int AS_PAD, int BS_PAD, CpAsyncPolicy CP_A_POLICY, CpAsyncPolicy CP_B_POLICY>
__device__ __forceinline__ void cp_async_tile_f32(
    int buf,
    int k_base,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ As,
    float* __restrict__ Bs,
    int row0,
    int col0,
    int K,
    int N) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int AS_LD = STAGE_K + AS_PAD;
  constexpr int BS_LD = TILE_N + BS_PAD;
  constexpr int VEC = 4;
  constexpr int A_EL = TILE_M * STAGE_K;
  constexpr int B_EL = STAGE_K * TILE_N;
  constexpr int A_V = A_EL / VEC;
  constexpr int B_V = B_EL / VEC;
  const int tid = (int)threadIdx.x;
  #pragma unroll
  for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {
    const int off = idx * VEC;
    const int r = off / STAGE_K;
    const int kk = off - r * STAGE_K;
    const int gr = row0 + r;
    const int gk = k_base + kk;
    cp_async_16<CP_A_POLICY>(
        As + (size_t)buf * (size_t)(TILE_M * AS_LD) + (size_t)r * (size_t)AS_LD + (size_t)kk,
        A + (size_t)gr * (size_t)K + (size_t)gk);
  }
  #pragma unroll
  for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {
    const int off = bidx * VEC;
    const int kk = off / TILE_N;
    const int n = off - kk * TILE_N;
    const int gn = col0 + n;
    const int gk = k_base + kk;
    cp_async_16<CP_B_POLICY>(
        Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk * (size_t)BS_LD + (size_t)n,
        B + (size_t)gk * (size_t)N + (size_t)gn);
  }
#else
  (void)buf;
  (void)k_base;
  (void)A;
  (void)B;
  (void)As;
  (void)Bs;
  (void)row0;
  (void)col0;
  (void)K;
  (void)N;
#endif
}

__device__ __forceinline__ void mma_tf32_m16n16k8_rr(
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float>& acc,
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& a_frag,
    const nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>& b_frag) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const int a0 = __float_as_int(a_frag.x[0]);
  const int a1 = __float_as_int(a_frag.x[1]);
  const int a2 = __float_as_int(a_frag.x[2]);
  const int a3 = __float_as_int(a_frag.x[3]);
  const int b0 = __float_as_int(b_frag.x[0]);
  const int b1 = __float_as_int(b_frag.x[1]);
  const int b2 = __float_as_int(b_frag.x[2]);
  const int b3 = __float_as_int(b_frag.x[3]);

  float d0 = acc.x[0];
  float d1 = acc.x[1];
  float d2 = acc.x[2];
  float d3 = acc.x[3];
  float d4 = acc.x[4];
  float d5 = acc.x[5];
  float d6 = acc.x[6];
  float d7 = acc.x[7];

  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3));

  acc.x[0] = d0;
  acc.x[1] = d1;
  acc.x[2] = d2;
  acc.x[3] = d3;
  acc.x[4] = d4;
  acc.x[5] = d5;
  acc.x[6] = d6;
  acc.x[7] = d7;
#else
  nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
#endif
}

template <
    int WARPS_M,
    int WARPS_N,
    int FRAG_N,
    int STAGE_K,
    int AS_PAD,
    int BS_PAD,
    int PIPE_STAGES,
    bool USE_CP_ASYNC,
    CpAsyncPolicy CP_A_POLICY,
    CpAsyncPolicy CP_B_POLICY,
    bool ENABLE_FASTPATH,
    bool SPECIALIZE_FULL_TILE>
__device__ __forceinline__ void wmma_matmul_f32_tf32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
  using namespace nvcuda::wmma;
  constexpr int TILE_M = 16 * WARPS_M;
  constexpr int TILE_N = 16 * WARPS_N * FRAG_N;
  constexpr int AS_LD = STAGE_K + AS_PAD;
  constexpr int BS_LD = TILE_N + BS_PAD;

  extern __shared__ __align__(16) float intentir_smem[];
  float* __restrict__ As = intentir_smem;
  float* __restrict__ Bs = intentir_smem + (size_t)PIPE_STAGES * (size_t)TILE_M * (size_t)AS_LD;

  const int warp = (int)(threadIdx.x >> 5);  // 0..(WARPS_M*WARPS_N-1)
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp - warp_m * WARPS_N;

  const int row0 = (int)blockIdx.y * TILE_M;
  const int col0 = (int)blockIdx.x * TILE_N;
  if constexpr (!SPECIALIZE_FULL_TILE) {
    if (row0 >= M || col0 >= N) return;
  }

  const bool full_tile = SPECIALIZE_FULL_TILE
                             ? true
                             : (ENABLE_FASTPATH && (row0 + TILE_M <= M) && (col0 + TILE_N <= N) && ((K % STAGE_K) == 0) &&
                                ((K & 3) == 0) && ((N & 3) == 0));

  fragment<accumulator, 16, 16, 8, float> acc0;
  fill_fragment(acc0, 0.0f);
  fragment<accumulator, 16, 16, 8, float> acc1;
  if constexpr (FRAG_N > 1) {
    fill_fragment(acc1, 0.0f);
  }

  if (full_tile) {
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;

    if constexpr (USE_CP_ASYNC) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      if constexpr (PIPE_STAGES == 2) {
        cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD, CP_A_POLICY, CP_B_POLICY>(0, 0, A, B, As, Bs, row0, col0, K, N);
        intentir_cp_async_commit();
        intentir_cp_async_wait_all();
        __syncthreads();

        int buf = 0;
        for (int k0 = 0; k0 < K; k0 += STAGE_K) {
          const int next_k0 = k0 + STAGE_K;
          const int next_buf = buf ^ 1;
          if (next_k0 < K) {
            cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD, CP_A_POLICY, CP_B_POLICY>(
                next_buf, next_k0, A, B, As, Bs, row0, col0, K, N);
            intentir_cp_async_commit();
          }

          #pragma unroll
          for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {
            load_matrix_sync(
                a_frag,
                As + (size_t)buf * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                AS_LD);
            load_matrix_sync(
                b_frag,
                Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
                BS_LD);
            mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
            if constexpr (FRAG_N > 1) {
              load_matrix_sync(
                  b_frag,
                  Bs + (size_t)buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
                  BS_LD);
              mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
            }
          }

          if (next_k0 < K) {
            intentir_cp_async_wait_all();
            __syncthreads();
          }
          buf = next_buf;
        }
      } else {
        static_assert(PIPE_STAGES == 3, "PIPE_STAGES must be 3 for this pipeline path");
        const int num_tiles = K / STAGE_K;

        // Prime the pipeline with the first (PIPE_STAGES - 1) tiles. Tile `t` is stored in
        // buffer `(t % PIPE_STAGES)`.
        cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD, CP_A_POLICY, CP_B_POLICY>(0, 0, A, B, As, Bs, row0, col0, K, N);
        intentir_cp_async_commit();
        if (num_tiles > 1) {
          cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD, CP_A_POLICY, CP_B_POLICY>(1, STAGE_K, A, B, As, Bs, row0, col0, K, N);
          intentir_cp_async_commit();
        }

        // Ensure tile0 is ready before the first load_matrix_sync. If we prefetched tile1,
        // keep it in-flight (wait_group<1>); otherwise wait for all.
        if (num_tiles > 1) {
          intentir_cp_async_wait_group<1>();
        } else {
          intentir_cp_async_wait_group<0>();
        }
        __syncthreads();

        for (int tile = 0; tile < num_tiles; ++tile) {
          const int read_buf = tile % PIPE_STAGES;
          const int pf_tile = tile + (PIPE_STAGES - 1);  // overlap: prefetch two tiles ahead
          const int pf_buf = pf_tile % PIPE_STAGES;
          if (pf_tile < num_tiles) {
            cp_async_tile_f32<TILE_M, TILE_N, STAGE_K, AS_PAD, BS_PAD, CP_A_POLICY, CP_B_POLICY>(
                pf_buf, pf_tile * STAGE_K, A, B, As, Bs, row0, col0, K, N);
            intentir_cp_async_commit();
          }

          #pragma unroll
          for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {
            load_matrix_sync(
                a_frag,
                As + (size_t)read_buf * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0,
                AS_LD);
            load_matrix_sync(
                b_frag,
                Bs + (size_t)read_buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
                BS_LD);
            mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
            if constexpr (FRAG_N > 1) {
              load_matrix_sync(
                  b_frag,
                  Bs + (size_t)read_buf * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
                  BS_LD);
              mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
            }
          }

          if (tile + 1 < num_tiles) {
            // Next tile is `tile+1`. If there is also a tile `tile+2` in flight, keep one group
            // outstanding (wait_group<1>) to overlap; otherwise drain fully so the next buffer is ready.
            if (tile + 2 < num_tiles) {
              intentir_cp_async_wait_group<1>();
            } else {
              intentir_cp_async_wait_group<0>();
            }
            __syncthreads();
          }
        }
      }
#else
      constexpr int VEC = 4;
      constexpr int A_EL = TILE_M * STAGE_K;
      constexpr int B_EL = STAGE_K * TILE_N;
      constexpr int A_V = A_EL / VEC;
      constexpr int B_V = B_EL / VEC;
      const int tid = (int)threadIdx.x;
      for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {
          const int off = idx * VEC;
          const int r = off / STAGE_K;
          const int kk = off - r * STAGE_K;
          const int gr = row0 + r;
          const int gk = k0 + kk;
          *reinterpret_cast<float4*>(As + (size_t)r * (size_t)AS_LD + (size_t)kk) =
              *reinterpret_cast<const float4*>(A + (size_t)gr * (size_t)K + (size_t)gk);
        }
        for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {
          const int off = bidx * VEC;
          const int kk = off / TILE_N;
          const int n = off - kk * TILE_N;
          const int gn = col0 + n;
          const int gk = k0 + kk;
          *reinterpret_cast<float4*>(Bs + (size_t)kk * (size_t)BS_LD + (size_t)n) =
              *reinterpret_cast<const float4*>(B + (size_t)gk * (size_t)N + (size_t)gn);
        }
        __syncthreads();
        #pragma unroll
        for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {
          load_matrix_sync(a_frag, As + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0, AS_LD);
          load_matrix_sync(b_frag, Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16), BS_LD);
          mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
          if constexpr (FRAG_N > 1) {
            load_matrix_sync(b_frag, Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16), BS_LD);
            mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
          }
        }
        __syncthreads();
      }
#endif
    } else {
      constexpr int VEC = 4;
      constexpr int A_EL = TILE_M * STAGE_K;
      constexpr int B_EL = STAGE_K * TILE_N;
      constexpr int A_V = A_EL / VEC;
      constexpr int B_V = B_EL / VEC;
      const int tid = (int)threadIdx.x;
      for (int k0 = 0; k0 < K; k0 += STAGE_K) {
        for (int idx = tid; idx < A_V; idx += (int)blockDim.x) {
          const int off = idx * VEC;
          const int r = off / STAGE_K;
          const int kk = off - r * STAGE_K;
          const int gr = row0 + r;
          const int gk = k0 + kk;
          *reinterpret_cast<float4*>(As + (size_t)r * (size_t)AS_LD + (size_t)kk) =
              *reinterpret_cast<const float4*>(A + (size_t)gr * (size_t)K + (size_t)gk);
        }
        for (int bidx = tid; bidx < B_V; bidx += (int)blockDim.x) {
          const int off = bidx * VEC;
          const int kk = off / TILE_N;
          const int n = off - kk * TILE_N;
          const int gn = col0 + n;
          const int gk = k0 + kk;
          *reinterpret_cast<float4*>(Bs + (size_t)kk * (size_t)BS_LD + (size_t)n) =
              *reinterpret_cast<const float4*>(B + (size_t)gk * (size_t)N + (size_t)gn);
        }
        __syncthreads();
        #pragma unroll
        for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {
          load_matrix_sync(a_frag, As + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0, AS_LD);
          load_matrix_sync(b_frag, Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16), BS_LD);
          mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
          if constexpr (FRAG_N > 1) {
            load_matrix_sync(b_frag, Bs + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16), BS_LD);
            mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
          }
        }
        __syncthreads();
      }
    }

    const int out_r = row0 + warp_m * 16;
    const int out_c0 = col0 + (warp_n * FRAG_N + 0) * 16;
    store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c0, acc0, (unsigned)N, mem_row_major);
    if constexpr (FRAG_N > 1) {
      const int out_c1 = col0 + (warp_n * FRAG_N + 1) * 16;
      store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c1, acc1, (unsigned)N, mem_row_major);
    }
    return;
  }

  if constexpr (!SPECIALIZE_FULL_TILE) {
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
    for (int k0 = 0; k0 < K; k0 += STAGE_K) {
      const int tid = (int)threadIdx.x;
      const int total = TILE_M * STAGE_K + STAGE_K * TILE_N;
      for (int idx = tid; idx < total; idx += (int)blockDim.x) {
        if (idx < TILE_M * STAGE_K) {
          const int r = idx / STAGE_K;
          const int kk = idx - r * STAGE_K;
          const int gr = row0 + r;
          const int gk = k0 + kk;
          As[(size_t)0 * (size_t)(TILE_M * AS_LD) + (size_t)r * (size_t)AS_LD + (size_t)kk] =
              (gr < M && gk < K) ? intentir_ldg_f32(A + (size_t)gr * (size_t)K + (size_t)gk) : 0.0f;
        } else {
          const int bidx = idx - TILE_M * STAGE_K;
          const int kk = bidx / TILE_N;
          const int n = bidx - kk * TILE_N;
          const int gn = col0 + n;
          const int gk = k0 + kk;
          Bs[(size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk * (size_t)BS_LD + (size_t)n] =
              (gn < N && gk < K) ? intentir_ldg_f32(B + (size_t)gk * (size_t)N + (size_t)gn) : 0.0f;
        }
      }
      __syncthreads();
      #pragma unroll
      for (int kk0 = 0; kk0 < STAGE_K; kk0 += 8) {
        load_matrix_sync(a_frag, As + (size_t)0 * (size_t)(TILE_M * AS_LD) + (size_t)(warp_m * 16) * (size_t)AS_LD + (size_t)kk0, AS_LD);
        load_matrix_sync(
            b_frag,
            Bs + (size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 0) * 16),
            BS_LD);
        mma_tf32_m16n16k8_rr(acc0, a_frag, b_frag);
        if constexpr (FRAG_N > 1) {
          load_matrix_sync(
              b_frag,
              Bs + (size_t)0 * (size_t)(STAGE_K * BS_LD) + (size_t)kk0 * (size_t)BS_LD + (size_t)((warp_n * FRAG_N + 1) * 16),
              BS_LD);
          mma_tf32_m16n16k8_rr(acc1, a_frag, b_frag);
        }
      }
      __syncthreads();
    }

    const int out_r = row0 + warp_m * 16;
    const int out_c0 = col0 + (warp_n * FRAG_N + 0) * 16;
    if (out_r < M && out_c0 < N) {
      store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c0, acc0, (unsigned)N, mem_row_major);
    }
    if constexpr (FRAG_N > 1) {
      const int out_c1 = col0 + (warp_n * FRAG_N + 1) * 16;
      if (out_r < M && out_c1 < N) {
        store_matrix_sync(C + (size_t)out_r * (size_t)N + (size_t)out_c1, acc1, (unsigned)N, mem_row_major);
      }
    }
  }
}

}  // namespace intentir_cuda
