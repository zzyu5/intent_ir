// Minimal CUDA kernel for IntentIR CUDA frontend MVP.
// Semantics: C[i] = A[i] + B[i]

extern "C" __global__ void vec_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

