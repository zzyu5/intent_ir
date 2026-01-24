"""
IntentIR -> CUDA kernel codegen (MVP).

Scope (initial):
- AI-Bench8 kernels used in paper experiments:
  - matmul, dropout, softmax, layernorm, correlation, resize, rope, warp

This codegen intentionally focuses on producing *runnable* CUDA for the paper.
It does not try to cover the full IntentIR op-set yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from intent_ir.ir import IntentFunction, ScheduleSketch

from backends.cuda.runtime import CudaLaunch


class CudaLoweringError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaLoweredKernel:
    kernel_name: str
    cuda_src: str
    io_spec: Dict[str, Any]
    launch: CudaLaunch
    output_names: list[str]
    # Bindings needed by the runtime to materialize shapes/scalars.
    bindings: Dict[str, Any]


def _as_int(v: Any, *, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise CudaLoweringError(f"expected int for {name}, got {v!r}") from e


def _resolve_schedule_int(v: str | int | None, bindings: Mapping[str, Any], *, default: int) -> int:
    if v is None:
        return int(default)
    if isinstance(v, int):
        return int(v)
    key = str(v)
    if key in bindings:
        return _as_int(bindings[key], name=f"schedule.{key}")
    # Accept "BLOCK_M" style names without explicit bindings (fallback to default).
    return int(default)


def _kernel_matmul_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "matmul":
        raise CudaLoweringError("matmul lowering expects a single matmul op")
    op = intent.ops[0]
    if len(op.inputs) != 2:
        raise CudaLoweringError("matmul expects 2 inputs")
    a, b = op.inputs
    c = op.output
    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    K = _as_int(bindings.get("K"), name="K")

    sched = intent.schedule or ScheduleSketch()
    block_y = _resolve_schedule_int(sched.tile_m, bindings, default=16)  # rows
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=16)  # cols
    # Clamp to CUDA limits (1024 threads per block).
    if block_x <= 0:
        block_x = 16
    if block_y <= 0:
        block_y = 16
    if block_x * block_y > 1024:
        # Prefer keeping X dimension; shrink Y.
        block_y = max(1, 1024 // max(1, block_x))

    grid_x = (N + block_x - 1) // block_x
    grid_y = (M + block_y - 1) // block_y

    cuda_src = f"""
extern "C" __global__ void {intent.name}(
    const float* A, const float* B, float* C,
    int M, int N, int K) {{
  int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  const int a_base = row * K;
  for (int k = 0; k < K; ++k) {{
    acc += A[a_base + k] * B[k * N + col];
  }}
  C[row * N + col] = acc;
}}
""".lstrip()

    io_spec = {
        "arg_names": [a, b, c, "M", "N", "K"],
        "tensors": {
            a: {"dtype": "f32", "shape": ["M", "K"]},
            b: {"dtype": "f32", "shape": ["K", "N"]},
            c: {"dtype": "f32", "shape": ["M", "N"]},
        },
        "scalars": {"M": "i32", "N": "i32", "K": "i32"},
    }
    launch = CudaLaunch(grid=(grid_x, grid_y, 1), block=(block_x, block_y, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[c], bindings=dict(bindings))


def _kernel_dropout_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "dropout":
        raise CudaLoweringError("dropout lowering expects a single dropout op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("dropout expects 3 inputs (X,p,seed)")
    X, p_name, seed_name = op.inputs
    Y = op.output
    n = _as_int(bindings.get("n_elements"), name="n_elements")

    # Default to 10 Philox rounds (matches Triton reference).
    rounds = int(op.attrs.get("n_rounds") or 10)

    # Use descriptor-like block size if present.
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024

    grid_x = (n + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>
#include <math.h>

// Philox (matches backends/spmd_rvv/runtime/intentir_ops.c semantics)
__device__ __forceinline__ uint32_t intentir_philox_randint_u32(uint64_t seed, uint32_t c0, int n_rounds) {{
  uint32_t c1 = 0u, c2 = 0u, c3 = 0u;
  uint32_t k0 = (uint32_t)(seed & 0xFFFFFFFFu);
  uint32_t k1 = (uint32_t)((seed >> 32) & 0xFFFFFFFFu);
  const uint32_t PHILOX_KEY_A = 0x9E3779B9u;
  const uint32_t PHILOX_KEY_B = 0xBB67AE85u;
  const uint32_t PHILOX_ROUND_A = 0xD2511F53u;
  const uint32_t PHILOX_ROUND_B = 0xCD9E8D57u;
  if (n_rounds <= 0) n_rounds = 10;
  #pragma unroll
  for (int r = 0; r < 10; ++r) {{
    if (r >= n_rounds) break;
    const uint32_t _c0 = c0;
    const uint32_t _c2 = c2;
    const uint64_t prod0 = (uint64_t)PHILOX_ROUND_A * (uint64_t)_c0;
    const uint64_t prod1 = (uint64_t)PHILOX_ROUND_B * (uint64_t)_c2;
    c0 = (uint32_t)(prod1 >> 32) ^ c1 ^ k0;
    c2 = (uint32_t)(prod0 >> 32) ^ c3 ^ k1;
    c1 = (uint32_t)prod1;
    c3 = (uint32_t)prod0;
    k0 += PHILOX_KEY_A;
    k1 += PHILOX_KEY_B;
  }}
  return c0;
}}

__device__ __forceinline__ float intentir_uint_to_uniform_float_u32(uint32_t x) {{
  int32_t xi = (int32_t)x;
  if (xi < 0) xi = ~xi; // -x-1
  return (float)xi * 4.6566127342e-10f;
}}

extern "C" __global__ void {intent.name}(const float* X, const float* p_ptr, const int* seed_ptr, float* Y, int64_t n_elements) {{
  const int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= n_elements) return;
  const float p = p_ptr ? p_ptr[0] : 0.0f;
  const uint64_t seed = seed_ptr ? (uint64_t)(uint32_t)seed_ptr[0] : 0ull;
  if (p <= 0.0f) {{ Y[i] = X[i]; return; }}
  if (p >= 1.0f) {{ Y[i] = 0.0f; return; }}
  const float inv_keep = 1.0f / (1.0f - p);
  float r = intentir_uint_to_uniform_float_u32(intentir_philox_randint_u32(seed, (uint32_t)i, {rounds}));
  Y[i] = (r > p) ? (X[i] * inv_keep) : 0.0f;
}}
""".lstrip()

    io_spec = {
        "arg_names": [X, p_name, seed_name, Y, "n_elements"],
        "tensors": {
            X: {"dtype": "f32", "shape": ["n_elements"]},
            p_name: {"dtype": "f32", "shape": []},
            seed_name: {"dtype": "i32", "shape": []},
            Y: {"dtype": "f32", "shape": ["n_elements"]},
        },
        "scalars": {"n_elements": "i64"},
    }
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, int] = dict(bindings)
    out_bindings.setdefault("n_elements", n)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[Y], bindings=out_bindings)


def _kernel_softmax_2d_last_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Softmax kernel: one block per row, reduce over last dimension.
    R = _as_int(bindings.get("R"), name="R")
    C = _as_int(bindings.get("C"), name="C")

    # Default block size: next pow2(C) capped at 1024 (match Triton ref).
    def _next_pow2(x: int) -> int:
        if x <= 1:
            return 1
        return 1 << (int(x - 1).bit_length())

    default_block = min(1024, _next_pow2(C))
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=default_block)
    if block_x <= 0:
        block_x = default_block
    if block_x > 1024:
        block_x = 1024

    # Require power-of-two for the simple shared-memory reduction.
    if (block_x & (block_x - 1)) != 0:
        # Round down to nearest power-of-two.
        b = 1 << (int(block_x).bit_length() - 1)
        block_x = max(1, min(1024, b))

    # Prefer names from the intent (AI-Bench uses in_ptr/out_ptr).
    out_name = str(intent.outputs[0]) if intent.outputs else "out"
    # Find a distinct input tensor with the same [R,C] shape.
    in_name = None
    try:
        out_shape = intent.tensors[out_name].shape
        for tn, ts in intent.tensors.items():
            if tn == out_name:
                continue
            if ts.dtype == "f32" and ts.shape == out_shape:
                in_name = str(tn)
                break
    except Exception:
        in_name = None
    if not in_name:
        # Fallback: common AI-Bench naming.
        in_name = "in_ptr" if "in_ptr" in intent.tensors else "inp"

    cuda_src = f"""
#include <math.h>

extern "C" __global__ void {intent.name}(const float* {in_name}, float* {out_name}, int R, int C) {{
  const int r = (int)blockIdx.x;
  if (r >= R) return;
  // Shared reductions (blockDim.x must equal the compiled BLOCK_SIZE).
  __shared__ float smem[1024];
  float tmax = -INFINITY;
  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {{
    float v = {in_name}[r * C + c];
    tmax = fmaxf(tmax, v);
  }}
  smem[threadIdx.x] = tmax;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + off]);
    __syncthreads();
  }}
  const float mx = smem[0];

  float tsum = 0.0f;
  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {{
    float e = expf({in_name}[r * C + c] - mx);
    {out_name}[r * C + c] = e;
    tsum += e;
  }}
  smem[threadIdx.x] = tsum;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }}
  const float inv = 1.0f / smem[0];
  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {{
    {out_name}[r * C + c] *= inv;
  }}
}}
""".lstrip()

    # NOTE: We allocate a fixed 1024 shared array; block_x must be <=1024 (enforced).
    io_spec = {
        "arg_names": [in_name, out_name, "R", "C"],
        "tensors": {
            in_name: {"dtype": "f32", "shape": ["R", "C"]},
            out_name: {"dtype": "f32", "shape": ["R", "C"]},
        },
        "scalars": {"R": "i32", "C": "i32"},
    }
    launch = CudaLaunch(grid=(R, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_layernorm_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    M = _as_int(bindings.get("M"), name="M")
    N = _as_int(bindings.get("N"), name="N")
    eps = float(bindings.get("eps", 1e-5))

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    if (block_x & (block_x - 1)) != 0:
        b = 1 << (int(block_x).bit_length() - 1)
        block_x = max(1, min(1024, b))

    cuda_src = f"""
#include <math.h>

extern "C" __global__ void {intent.name}(
    const float* X, float* Y, const float* W, const float* B, float* Mean, float* Rstd,
    int M, int N, float eps) {{
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ float smem[1024];
  float tsum = 0.0f;
  const float* xrow = X + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    tsum += xrow[n];
  }}
  smem[threadIdx.x] = tsum;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }}
  const float mean = smem[0] / (float)N;

  float tsq = 0.0f;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    float c = xrow[n] - mean;
    tsq += c * c;
  }}
  smem[threadIdx.x] = tsq;
  __syncthreads();
  for (int off = ((int)blockDim.x >> 1); off > 0; off >>= 1) {{
    if ((int)threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
    __syncthreads();
  }}
  const float var = smem[0] / (float)N;
  const float rstd = rsqrtf(var + eps);
  if ((int)threadIdx.x == 0) {{
    Mean[m] = mean;
    Rstd[m] = rstd;
  }}
  float* yrow = Y + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    float c = xrow[n] - mean;
    yrow[n] = (c * rstd) * W[n] + B[n];
  }}
}}
""".lstrip()

    io_spec = {
        "arg_names": ["X", "Y", "W", "B", "Mean", "Rstd", "M", "N", "eps"],
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"]},
            "Y": {"dtype": "f32", "shape": ["M", "N"]},
            "W": {"dtype": "f32", "shape": ["N"]},
            "B": {"dtype": "f32", "shape": ["N"]},
            "Mean": {"dtype": "f32", "shape": ["M"]},
            "Rstd": {"dtype": "f32", "shape": ["M"]},
        },
        "scalars": {"M": "i32", "N": "i32", "eps": "f32"},
    }
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    # Use canonical output names when available.
    outs = list(intent.outputs) if intent.outputs else ["Y"]
    out_bindings: Dict[str, Any] = dict(bindings)
    out_bindings.setdefault("eps", float(eps))
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=outs, bindings=out_bindings)


def _kernel_rope_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    SEQ = _as_int(bindings.get("SEQ_LEN"), name="SEQ_LEN")
    B = _as_int(bindings.get("BATCH_NUM"), name="BATCH_NUM")
    H = _as_int(bindings.get("HEAD_NUM"), name="HEAD_NUM")
    D = _as_int(bindings.get("HEAD_DIM"), name="HEAD_DIM")
    if (D & 1) != 0:
        raise CudaLoweringError("rope expects even HEAD_DIM")
    half = D // 2
    total_pairs = SEQ * B * H * half

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total_pairs + block_x - 1) // block_x

    out_name = str(intent.outputs[0]) if intent.outputs else "output"
    cuda_src = f"""
extern "C" __global__ void {intent.name}(
    const float* input, const float* cos, const float* sin, float* {out_name},
    int SEQ_LEN, int BATCH_NUM, int HEAD_NUM, int HEAD_DIM) {{
  const int half = (int)(HEAD_DIM >> 1);
  const int64_t total = (int64_t)SEQ_LEN * (int64_t)BATCH_NUM * (int64_t)HEAD_NUM * (int64_t)half;
  const int64_t t = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (t >= total) return;
  int j = (int)(t % (int64_t)half);
  int64_t tmp = t / (int64_t)half;
  int h = (int)(tmp % (int64_t)HEAD_NUM);
  tmp /= (int64_t)HEAD_NUM;
  int b = (int)(tmp % (int64_t)BATCH_NUM);
  int s = (int)(tmp / (int64_t)BATCH_NUM);
  const size_t base = (((size_t)s * (size_t)BATCH_NUM + (size_t)b) * (size_t)HEAD_NUM + (size_t)h) * (size_t)HEAD_DIM;
  const size_t cb = (size_t)s * (size_t)half + (size_t)j;
  float c = cos[cb];
  float s0 = sin[cb];
  float x1 = input[base + (size_t)j];
  float x2 = input[base + (size_t)half + (size_t)j];
  {out_name}[base + (size_t)j] = x1 * c - x2 * s0;
  {out_name}[base + (size_t)half + (size_t)j] = x1 * s0 + x2 * c;
}}
""".lstrip()

    io_spec = {
        "arg_names": ["input", "cos", "sin", out_name, "SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"],
        "tensors": {
            "input": {"dtype": "f32", "shape": ["SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"]},
            "cos": {"dtype": "f32", "shape": ["SEQ_LEN", "HEAD_DIM_DIV2"]},
            "sin": {"dtype": "f32", "shape": ["SEQ_LEN", "HEAD_DIM_DIV2"]},
            out_name: {"dtype": "f32", "shape": ["SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"]},
        },
        "scalars": {"SEQ_LEN": "i32", "BATCH_NUM": "i32", "HEAD_NUM": "i32", "HEAD_DIM": "i32"},
    }
    # Provide derived symbol for cos/sin shapes.
    bindings = dict(bindings)
    bindings.setdefault("HEAD_DIM_DIV2", half)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=bindings)


def _kernel_resize_bilinear2x_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    C = _as_int(bindings.get("C"), name="C")
    H = _as_int(bindings.get("H"), name="H")
    W = _as_int(bindings.get("W"), name="W")
    OH = _as_int(bindings.get("OH", 2 * H), name="OH")
    OW = _as_int(bindings.get("OW", 2 * W), name="OW")
    if OH != 2 * H or OW != 2 * W:
        raise CudaLoweringError("resize MVP supports only 2x upsample")
    hw_fl = 7
    if intent.ops and intent.ops[0].op == "resize":
        try:
            hw_fl = int(intent.ops[0].attrs.get("hw_fl", 7))
        except Exception:
            hw_fl = 7
    if hw_fl <= 0:
        hw_fl = 7
    total = C * OH * OW

    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(const int8_t* src, int8_t* out, int C, int H, int W, int hw_fl) {{
  const int OH = H * 2;
  const int OW = W * 2;
  const int64_t total = (int64_t)C * (int64_t)OH * (int64_t)OW;
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (tid >= total) return;
  const int w_idx = (int)(tid % (int64_t)OW);
  int64_t tmp = tid / (int64_t)OW;
  const int h_idx = (int)(tmp % (int64_t)OH);
  const int c = (int)(tmp / (int64_t)OH);

  const int factor = 1 << hw_fl;
  const int input_y = h_idx << (hw_fl - 1);
  const int y0 = input_y >> hw_fl;
  const int h1 = input_y - (y0 << hw_fl);
  const int h0 = factor - h1;
  int y1 = y0 + 1;
  if (y1 >= H) y1 = H - 1;

  const int input_x = w_idx << (hw_fl - 1);
  const int x0 = input_x >> hw_fl;
  const int w1 = input_x - (x0 << hw_fl);
  const int w0 = factor - w1;
  int x1 = x0 + 1;
  if (x1 >= W) x1 = W - 1;

  const int64_t src_hw = (int64_t)H * (int64_t)W;
  const int64_t dst_hw = (int64_t)OH * (int64_t)OW;
  const int64_t src_base = (int64_t)c * src_hw;
  const int64_t src_row0 = src_base + (int64_t)y0 * (int64_t)W;
  const int64_t src_row1 = src_base + (int64_t)y1 * (int64_t)W;

  const int16_t y0x0 = (int16_t)src[src_row0 + x0];
  const int16_t y0x1 = (int16_t)src[src_row0 + x1];
  const int16_t y1x0 = (int16_t)src[src_row1 + x0];
  const int16_t y1x1 = (int16_t)src[src_row1 + x1];
  const int32_t sum1 = (((int32_t)y0x0 * (int32_t)w0) + ((int32_t)y0x1 * (int32_t)w1)) >> hw_fl;
  const int32_t sum2 = (((int32_t)y1x0 * (int32_t)w0) + ((int32_t)y1x1 * (int32_t)w1)) >> hw_fl;
  const int32_t sum = (((sum1 * (int32_t)h0) + (sum2 * (int32_t)h1)) >> hw_fl);

  out[(size_t)tid] = (int8_t)sum;
}}
""".lstrip()

    io_spec = {
        "arg_names": ["src", "out", "C", "H", "W", "hw_fl"],
        "tensors": {
            "src": {"dtype": "i8", "shape": ["C", "H", "W"]},
            "out": {"dtype": "i8", "shape": ["C", "OH", "OW"]},
        },
        "scalars": {"C": "i32", "H": "i32", "W": "i32", "hw_fl": "i32"},
    }
    bindings = dict(bindings)
    bindings.setdefault("OH", OH)
    bindings.setdefault("OW", OW)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    bindings.setdefault("hw_fl", hw_fl)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[intent.outputs[0]], bindings=bindings)


def _kernel_warp_q8_8_i8_i16(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    C = _as_int(bindings.get("C"), name="C")
    H = _as_int(bindings.get("H"), name="H")
    W = _as_int(bindings.get("W"), name="W")
    total = C * H * W
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=256)
    if block_x <= 0:
        block_x = 256
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(const int8_t* src, const int16_t* offset, int8_t* out, int C, int H, int W) {{
  const int64_t total = (int64_t)C * (int64_t)H * (int64_t)W;
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (tid >= total) return;
  const int w = (int)(tid % (int64_t)W);
  int64_t tmp = tid / (int64_t)W;
  const int h = (int)(tmp % (int64_t)H);
  const int c = (int)(tmp / (int64_t)H);
  const int64_t hw = (int64_t)H * (int64_t)W;
  const int64_t row_base = (int64_t)c * hw + (int64_t)h * (int64_t)W;
  const int64_t off_base = (int64_t)h * (int64_t)W;
  const int16_t ov = offset[off_base + w];
  const int8_t offset_int = (int8_t)(ov >> 8);
  const int8_t offset_frac = (int8_t)(((int16_t)(ov << 8)) >> 8);
  const int8_t indvar = (int8_t)w;
  const int8_t right_i8 = (int8_t)(indvar - offset_int);
  const int8_t left_i8 = (int8_t)(right_i8 - 1);
  const int right = (int)right_i8;
  const int left = (int)left_i8;
  int8_t right_val = 0;
  int8_t left_val = 0;
  if (right >= 0 && right < W) right_val = src[row_base + (int64_t)right];
  if (left >= 0 && left < W) left_val = src[row_base + (int64_t)left];
  int16_t outv = (int16_t)((int16_t)right_val << 8);
  outv = (int16_t)(outv + (int16_t)((int16_t)(left_val - right_val) * (int16_t)offset_frac));
  outv = (int16_t)(outv >> 8);
  out[row_base + w] = (int8_t)outv;
}}
""".lstrip()

    io_spec = {
        "arg_names": ["src", "offset", "out", "C", "H", "W"],
        "tensors": {
            "src": {"dtype": "i8", "shape": ["C", "H", "W"]},
            "offset": {"dtype": "i16", "shape": ["H", "W"]},
            "out": {"dtype": "i8", "shape": ["C", "H", "W"]},
        },
        "scalars": {"C": "i32", "H": "i32", "W": "i32"},
    }
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[intent.outputs[0]], bindings=dict(bindings))


def _kernel_correlation_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    OC = _as_int(bindings.get("out_channel"), name="out_channel")
    IC = _as_int(bindings.get("in_channel"), name="in_channel")
    H = _as_int(bindings.get("height"), name="height")
    W = _as_int(bindings.get("width"), name="width")
    total = OC * H * W
    sched = intent.schedule or ScheduleSketch()
    block_x = _resolve_schedule_int(sched.tile_n, bindings, default=128)
    if block_x <= 0:
        block_x = 128
    if block_x > 1024:
        block_x = 1024
    grid_x = (total + block_x - 1) // block_x

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(
    const int8_t* src0, const int8_t* src1, int8_t* out,
    int out_channel, int in_channel, int height, int width, int out_shift) {{
  const int64_t hw = (int64_t)height * (int64_t)width;
  const int64_t total = (int64_t)out_channel * hw;
  const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (tid >= total) return;
  const int oc = (int)(tid / hw);
  const int64_t rem = tid - (int64_t)oc * hw;
  const int h = (int)(rem / (int64_t)width);
  const int w = (int)(rem - (int64_t)h * (int64_t)width);

  int sh = out_shift;
  if (sh < 0) sh = 0;
  if (sh > 30) sh = 30;

  if (oc >= width || w < oc) {{
    out[(size_t)tid] = 0;
    return;
  }}

  int32_t acc = 0;
  const int64_t off0 = (int64_t)h * (int64_t)width + (int64_t)w;
  const int64_t off1 = (int64_t)h * (int64_t)width + (int64_t)(w - oc);
  for (int k = 0; k < in_channel; ++k) {{
    const int64_t base = (int64_t)k * hw;
    acc += (int32_t)src0[base + off0] * (int32_t)src1[base + off1];
  }}
  out[(size_t)tid] = (int8_t)(acc >> sh);
}}
""".lstrip()

    io_spec = {
        "arg_names": ["src0", "src1", "out", "out_channel", "in_channel", "height", "width", "out_shift"],
        "tensors": {
            "src0": {"dtype": "i8", "shape": ["in_channel", "height", "width"]},
            "src1": {"dtype": "i8", "shape": ["in_channel", "height", "width"]},
            "out": {"dtype": "i8", "shape": ["out_channel", "height", "width"]},
        },
        "scalars": {"out_channel": "i32", "in_channel": "i32", "height": "i32", "width": "i32", "out_shift": "i32"},
    }
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[intent.outputs[0]], bindings=dict(bindings))


def lower_intent_to_cuda_kernel(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    schedule_override: ScheduleSketch | Dict[str, Any] | None = None,
) -> CudaLoweredKernel:
    """
    Lower an IntentFunction into a single CUDA kernel (MVP).

    Note: The lowering currently targets the AI-Bench8 kernels and a small set
    of common patterns (softmax2d-last, layernorm2d).
    """
    bindings: Dict[str, Any] = {}
    for k, v in dict(shape_bindings).items():
        key = str(k)
        if isinstance(v, bool):
            bindings[key] = int(v)
            continue
        if isinstance(v, int):
            bindings[key] = int(v)
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        bindings[key] = int(fv) if float(fv).is_integer() else float(fv)

    # Allow caller overrides for schedule knobs (used by tuning / freeze-vs-retune).
    if schedule_override is not None:
        if isinstance(schedule_override, ScheduleSketch):
            intent.schedule = schedule_override
        elif isinstance(schedule_override, dict):
            intent.schedule = ScheduleSketch(
                tile_m=schedule_override.get("tile_m"),
                tile_n=schedule_override.get("tile_n"),
                tile_k=schedule_override.get("tile_k"),
                vec_width=schedule_override.get("vec_width"),
                pipeline_depth=schedule_override.get("pipeline_depth"),
                axis_bindings=dict(schedule_override.get("axis_bindings") or {}),
                vec_axis=(schedule_override.get("vec_axis") if isinstance(schedule_override.get("vec_axis"), str) else None),
                parallel_axes=[str(x) for x in (schedule_override.get("parallel_axes") or [])],
                memory_hint=dict(schedule_override.get("memory_hint") or {}),
            )

    # 1) Direct single-op kernels.
    if intent.ops and len(intent.ops) == 1:
        op0 = intent.ops[0].op
        if op0 == "matmul":
            return _kernel_matmul_f32(intent, bindings)
        if op0 == "dropout":
            return _kernel_dropout_f32(intent, bindings)
        if op0 == "correlation":
            return _kernel_correlation_i8(intent, bindings)
        if op0 == "resize":
            return _kernel_resize_bilinear2x_i8(intent, bindings)
        if op0 == "warp":
            return _kernel_warp_q8_8_i8_i16(intent, bindings)
        if op0 == "rope":
            return _kernel_rope_f32(intent, bindings)

    # 2) Pattern-based kernels (fused).
    outs = set(intent.outputs or [])
    if {"Y", "Mean", "Rstd"}.issubset(outs):
        return _kernel_layernorm_2d_f32(intent, bindings)
    # Softmax: recognize by name or by presence of reduce_max + exp + reduce_sum.
    op_names = {o.op for o in (intent.ops or [])}
    if ("softmax" in str(intent.name).lower()) or ({"reduce_max", "reduce_sum", "exp", "div"}.issubset(op_names)):
        return _kernel_softmax_2d_last_f32(intent, bindings)

    raise CudaLoweringError(f"CUDA lowering unsupported for intent: name={intent.name} ops={sorted(op_names)}")


__all__ = ["CudaLoweringError", "CudaLoweredKernel", "lower_intent_to_cuda_kernel"]
