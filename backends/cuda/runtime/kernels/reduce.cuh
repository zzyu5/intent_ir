#pragma once

#include <math.h>

#include <cub/block/block_reduce.cuh>

namespace intentir_cuda {

template <int BLOCK_THREADS>
struct BlockAllreduceF32 {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  typename BlockReduce::TempStorage temp;
  float out;
};

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_sum(float v, BlockAllreduceF32<BLOCK_THREADS>* st) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  const float sum = BlockReduce(st->temp).Reduce(v, cub::Sum());
  if ((int)threadIdx.x == 0) st->out = sum;
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_max(float v, BlockAllreduceF32<BLOCK_THREADS>* st) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  const float mx = BlockReduce(st->temp).Reduce(v, cub::Max());
  if ((int)threadIdx.x == 0) st->out = mx;
  __syncthreads();
  return st->out;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_sum(float v) {
  __shared__ BlockAllreduceF32<BLOCK_THREADS> st;
  return block_allreduce_sum<BLOCK_THREADS>(v, &st);
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_allreduce_max(float v) {
  __shared__ BlockAllreduceF32<BLOCK_THREADS> st;
  return block_allreduce_max<BLOCK_THREADS>(v, &st);
}

}  // namespace intentir_cuda

