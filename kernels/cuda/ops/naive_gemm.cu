// Naive GEMM (row-major):
//   C[m, n] = sum_k A[m, k] * B[k, n]
// A: [M, K], B: [K, N], C: [M, N]
//
// This kernel is intentionally simple (Tier-A anchor) to produce stable PTX
// evidence for the CUDA frontend (reduce + dot-like pattern).

extern "C" __global__ void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
  int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int m = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (m < M && n < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[m * K + k] * B[k * N + n];
    }
    C[m * N + n] = acc;
  }
}

