#pragma once

#if !defined(__CUDACC_RTC__)
#if defined(__has_include)
#if __has_include(<cub/block/block_reduce.cuh>)
#define INTENTIR_CUDA_HAS_CUB 1
#else
#define INTENTIR_CUDA_HAS_CUB 0
#endif
#else
#define INTENTIR_CUDA_HAS_CUB 1
#endif
#else
#define INTENTIR_CUDA_HAS_CUB 0
#endif

// NVCC + CUB support for very new SM versions (e.g., Blackwell sm_120) can be
// fragile across environments. Prefer our shuffle-based fallback on SM >= 1000
// to keep the runtime headers portable across clusters.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#undef INTENTIR_CUDA_HAS_CUB
#define INTENTIR_CUDA_HAS_CUB 0
#endif

#if INTENTIR_CUDA_HAS_CUB
#include <cub/block/block_reduce.cuh>
#endif

namespace intentir_cuda {

#if INTENTIR_CUDA_HAS_CUB
template <typename T, int BLOCK_THREADS>
struct BlockAllreduce {
  using BlockReduce = cub::BlockReduce<T, BLOCK_THREADS>;
  typename BlockReduce::TempStorage temp;
  T out;
};

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_sum(T v, BlockAllreduce<T, BLOCK_THREADS>* st) {
  using BlockReduce = cub::BlockReduce<T, BLOCK_THREADS>;
  const T sum = BlockReduce(st->temp).Reduce(v, cub::Sum());
  if ((int)threadIdx.x == 0) st->out = sum;
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_max(T v, BlockAllreduce<T, BLOCK_THREADS>* st) {
  using BlockReduce = cub::BlockReduce<T, BLOCK_THREADS>;
  const T mx = BlockReduce(st->temp).Reduce(v, cub::Max());
  if ((int)threadIdx.x == 0) st->out = mx;
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_sum(T v) {
  __shared__ BlockAllreduce<T, BLOCK_THREADS> st;
  return block_allreduce_sum<BLOCK_THREADS>(v, &st);
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_max(T v) {
  __shared__ BlockAllreduce<T, BLOCK_THREADS> st;
  return block_allreduce_max<BLOCK_THREADS>(v, &st);
}

template <int BLOCK_THREADS>
using BlockAllreduceF32 = BlockAllreduce<float, BLOCK_THREADS>;

template <int BLOCK_THREADS>
using BlockAllreduceI32 = BlockAllreduce<int, BLOCK_THREADS>;

}  // namespace intentir_cuda

#else  // INTENTIR_CUDA_HAS_CUB

template <typename T, int BLOCK_THREADS>
struct BlockAllreduce {
  static constexpr int WARP_THREADS = 32;
  static constexpr int WARPS = (BLOCK_THREADS + (WARP_THREADS - 1)) / WARP_THREADS;
  T warp_out[WARPS];
  T out;
};

template <typename T>
__device__ __forceinline__ T _intentir_warp_reduce_sum(T v, unsigned mask) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(mask, v, off);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T _intentir_warp_reduce_max(T v, unsigned mask) {
  for (int off = 16; off > 0; off >>= 1) {
    const T other = __shfl_down_sync(mask, v, off);
    v = (other > v) ? other : v;
  }
  return v;
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_sum(T v, BlockAllreduce<T, BLOCK_THREADS>* st) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned mask = __activemask();
  v = _intentir_warp_reduce_sum(v, mask);
  if (lane == 0) st->warp_out[warp] = v;
  __syncthreads();

  constexpr int WARPS = BlockAllreduce<T, BLOCK_THREADS>::WARPS;
  constexpr unsigned FULL_MASK = 0xffffffffu;
  constexpr unsigned warp_mask = (WARPS >= 32) ? FULL_MASK : ((1u << WARPS) - 1u);
  if (warp == 0) {
    T w = (lane < WARPS) ? st->warp_out[lane] : (T)0;
    w = _intentir_warp_reduce_sum(w, warp_mask);
    if (lane == 0) st->out = w;
  }
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_max(T v, BlockAllreduce<T, BLOCK_THREADS>* st) {
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned mask = __activemask();
  v = _intentir_warp_reduce_max(v, mask);
  if (lane == 0) st->warp_out[warp] = v;
  __syncthreads();

  constexpr int WARPS = BlockAllreduce<T, BLOCK_THREADS>::WARPS;
  constexpr unsigned FULL_MASK = 0xffffffffu;
  constexpr unsigned warp_mask = (WARPS >= 32) ? FULL_MASK : ((1u << WARPS) - 1u);
  if (warp == 0) {
    T w = (lane < WARPS) ? st->warp_out[lane] : st->warp_out[0];
    w = _intentir_warp_reduce_max(w, warp_mask);
    if (lane == 0) st->out = w;
  }
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_sum(T v) {
  __shared__ BlockAllreduce<T, BLOCK_THREADS> st;
  return block_allreduce_sum<BLOCK_THREADS>(v, &st);
}

template <int BLOCK_THREADS, typename T>
__device__ __forceinline__ T block_allreduce_max(T v) {
  __shared__ BlockAllreduce<T, BLOCK_THREADS> st;
  return block_allreduce_max<BLOCK_THREADS>(v, &st);
}

template <int BLOCK_THREADS>
using BlockAllreduceF32 = BlockAllreduce<float, BLOCK_THREADS>;

template <int BLOCK_THREADS>
using BlockAllreduceI32 = BlockAllreduce<int, BLOCK_THREADS>;

}  // namespace intentir_cuda

#endif  // INTENTIR_CUDA_HAS_CUB
