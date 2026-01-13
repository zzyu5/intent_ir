// 2D row reduction: out[m] = sum_n inp[m, n]
// inp: [M, N] row-major, out: [M]

extern "C" __global__ void row_sum(
    const float* inp,
    float* out,
    int M,
    int N
) {
  int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (m < M) {
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
      acc += inp[m * N + n];
    }
    out[m] = acc;
  }
}

