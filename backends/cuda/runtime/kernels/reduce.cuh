#pragma once

#include <cub/block/block_reduce.cuh>

namespace intentir_cuda {

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
