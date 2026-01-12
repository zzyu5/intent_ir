// 2D transpose: out[x, y] = inp[y, x]
// inp: [M, N], out: [N, M] (row-major)

extern "C" __global__ void transpose2d(
    const float* inp,
    float* out,
    int M,
    int N
) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (x < N && y < M) {
    out[x * M + y] = inp[y * N + x];
  }
}

