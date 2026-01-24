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


def _dim_value(d: Any) -> str | int:
    # `Dim` is defined in intent_ir.ir_types, but we keep this helper generic.
    try:
        # Dim(kind=..., value=...)
        v = d.value  # type: ignore[attr-defined]
    except Exception:
        v = d
    if isinstance(v, (int, str)):
        return v
    return str(v)


def _shape_values(intent: IntentFunction, name: str) -> list[str | int]:
    if name not in intent.tensors:
        raise CudaLoweringError(f"unknown tensor in intent.tensors: {name}")
    t = intent.tensors[name]
    return [_dim_value(d) for d in (t.shape or [])]


def _tensor_io_spec(intent: IntentFunction, name: str) -> Dict[str, Any]:
    t = intent.tensors[name]
    return {"dtype": str(t.dtype), "shape": _shape_values(intent, name)}


def _resolve_dim_int(dim: str | int, bindings: Mapping[str, Any], *, name: str) -> int:
    if isinstance(dim, int):
        return int(dim)
    key = str(dim)
    if key not in bindings:
        raise CudaLoweringError(f"missing binding for dim {name} ({key})")
    return _as_int(bindings[key], name=name)


def _is_scalar_tensor(intent: IntentFunction, name: str, *, dtype: str | None = None) -> bool:
    t = intent.tensors.get(name)
    if t is None:
        return False
    if t.shape:
        return False
    if dtype is None:
        return True
    return str(t.dtype) == str(dtype)


def _io_spec_from_args(
    intent: IntentFunction,
    *,
    tensor_args: Sequence[str],
    scalar_args: Mapping[str, str],
    arg_names: Sequence[str],
) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for n in tensor_args:
        if n not in intent.tensors:
            raise CudaLoweringError(f"io_spec tensor arg missing from intent.tensors: {n}")
        tensors[n] = _tensor_io_spec(intent, n)
    return {"arg_names": list(arg_names), "tensors": tensors, "scalars": dict(scalar_args)}


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
    # Derive dims from tensor shapes to support both:
    # - scalar-tensor dims (AI-Bench)
    # - pure symbolic dims (unit tests)
    a_shape = _shape_values(intent, a)
    b_shape = _shape_values(intent, b)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise CudaLoweringError("matmul expects rank-2 inputs")
    M_dim, K_dim = a_shape
    K2_dim, N_dim = b_shape
    if str(K_dim) != str(K2_dim):
        raise CudaLoweringError(f"matmul K mismatch: A is {K_dim} but B is {K2_dim}")
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")
    K = _resolve_dim_int(K_dim, bindings, name="K")

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

    # Respect scalar-tensor dims if present in the IntentIR signature.
    m_is_tensor = _is_scalar_tensor(intent, str(M_dim), dtype="i32")
    n_is_tensor = _is_scalar_tensor(intent, str(N_dim), dtype="i32")
    k_is_tensor = _is_scalar_tensor(intent, str(K_dim), dtype="i32")

    m_param = f"const int* {str(M_dim)}_ptr" if m_is_tensor else "int M"
    n_param = f"const int* {str(N_dim)}_ptr" if n_is_tensor else "int N"
    k_param = f"const int* {str(K_dim)}_ptr" if k_is_tensor else "int K"

    m_load = f"const int M = {str(M_dim)}_ptr ? {str(M_dim)}_ptr[0] : 0;" if m_is_tensor else ""
    n_load = f"const int N = {str(N_dim)}_ptr ? {str(N_dim)}_ptr[0] : 0;" if n_is_tensor else ""
    k_load = f"const int K = {str(K_dim)}_ptr ? {str(K_dim)}_ptr[0] : 0;" if k_is_tensor else ""

    cuda_src = f"""
extern "C" __global__ void {intent.name}(
    const float* A, const float* B, float* C,
    {m_param}, {n_param}, {k_param}) {{
  {m_load}
  {n_load}
  {k_load}
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

    tensor_args = [a, b, c]
    scalar_args: Dict[str, str] = {}
    arg_names = [a, b, c]
    if m_is_tensor:
        tensor_args.append(str(M_dim))
        arg_names.append(str(M_dim))
    else:
        scalar_args[str(M_dim)] = "i32"
        arg_names.append(str(M_dim))
    if n_is_tensor:
        tensor_args.append(str(N_dim))
        arg_names.append(str(N_dim))
    else:
        scalar_args[str(N_dim)] = "i32"
        arg_names.append(str(N_dim))
    if k_is_tensor:
        tensor_args.append(str(K_dim))
        arg_names.append(str(K_dim))
    else:
        scalar_args[str(K_dim)] = "i32"
        arg_names.append(str(K_dim))

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
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

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[X, p_name, seed_name, Y],
        scalar_args={"n_elements": "i64"},
        arg_names=[X, p_name, seed_name, Y, "n_elements"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, int] = dict(bindings)
    out_bindings.setdefault("n_elements", n)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[Y], bindings=out_bindings)


def _kernel_softmax_2d_last_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Softmax kernel: one block per row, reduce over last dimension.
    # Identify the input matrix by tracing the reduce_max input (robust against extra tensors).
    out_name = str(intent.outputs[0]) if intent.outputs else "out"
    in_name = None
    for op in intent.ops or []:
        if op.op == "reduce_max" and op.inputs:
            in_name = str(op.inputs[0])
            break
    if not in_name:
        # Fallback: any rank-2 f32 tensor that is not produced by ops and not the output.
        produced = {o.output for o in (intent.ops or [])}
        for tn, tt in intent.tensors.items():
            if tn == out_name or tn in produced:
                continue
            if str(tt.dtype) == "f32" and len(tt.shape or []) == 2:
                in_name = str(tn)
                break
    if not in_name:
        raise CudaLoweringError("softmax lowering failed to identify input tensor")

    in_shape = _shape_values(intent, in_name)
    if len(in_shape) != 2:
        raise CudaLoweringError("softmax expects rank-2 input")
    R_dim, C_dim = in_shape
    R = _resolve_dim_int(R_dim, bindings, name="R")
    C = _resolve_dim_int(C_dim, bindings, name="C")

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

    r_is_tensor = _is_scalar_tensor(intent, str(R_dim), dtype="i32")
    c_is_tensor = _is_scalar_tensor(intent, str(C_dim), dtype="i32")
    r_param = f"const int* {str(R_dim)}_ptr" if r_is_tensor else "int R"
    c_param = f"const int* {str(C_dim)}_ptr" if c_is_tensor else "int C"
    r_load = f"const int R = {str(R_dim)}_ptr ? {str(R_dim)}_ptr[0] : 0;" if r_is_tensor else ""
    c_load = f"const int C = {str(C_dim)}_ptr ? {str(C_dim)}_ptr[0] : 0;" if c_is_tensor else ""

    cuda_src = f"""
#include <math.h>

extern "C" __global__ void {intent.name}(const float* {in_name}, float* {out_name}, {r_param}, {c_param}) {{
  {r_load}
  {c_load}
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
    tensor_args = [in_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [in_name, out_name]
    if r_is_tensor:
        tensor_args.append(str(R_dim))
        arg_names.append(str(R_dim))
    else:
        scalar_args[str(R_dim)] = "i32"
        arg_names.append(str(R_dim))
    if c_is_tensor:
        tensor_args.append(str(C_dim))
        arg_names.append(str(C_dim))
    else:
        scalar_args[str(C_dim)] = "i32"
        arg_names.append(str(C_dim))
    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(R, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_layernorm_2d_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Identify inputs from the true input set (ignore intermediates present in `intent.tensors`).
    produced = {o.output for o in (intent.ops or [])}
    outs = set(intent.outputs or [])
    inputs = [n for n in intent.tensors.keys() if n not in produced and n not in outs]

    X_name = "X" if "X" in inputs else None
    if not X_name:
        for n in inputs:
            if str(intent.tensors[n].dtype) == "f32" and len(intent.tensors[n].shape or []) == 2:
                X_name = n
                break
    if not X_name:
        raise CudaLoweringError("layernorm lowering cannot find input X")

    # Prefer canonical names for gamma/beta if present.
    W_name = "W" if "W" in inputs else None
    B_name = "B" if "B" in inputs else None
    if not W_name or not B_name:
        rank1 = [n for n in inputs if str(intent.tensors[n].dtype) == "f32" and len(intent.tensors[n].shape or []) == 1]
        if not W_name and rank1:
            W_name = rank1[0]
        if not B_name and len(rank1) > 1:
            B_name = rank1[1]
    if not W_name or not B_name:
        raise CudaLoweringError("layernorm lowering cannot find W/B inputs")

    out_names = list(intent.outputs) if intent.outputs else ["Y", "Mean", "Rstd"]
    if len(out_names) != 3:
        raise CudaLoweringError("layernorm lowering expects 3 outputs (Y, Mean, Rstd)")
    Y_name, Mean_name, Rstd_name = (str(x) for x in out_names)

    x_shape = _shape_values(intent, X_name)
    if len(x_shape) != 2:
        raise CudaLoweringError("layernorm expects rank-2 X")
    M_dim, N_dim = x_shape
    M = _resolve_dim_int(M_dim, bindings, name="M")
    N = _resolve_dim_int(N_dim, bindings, name="N")

    # eps from const() if present, else fallback.
    eps = None
    for op in intent.ops or []:
        if op.op == "const" and op.output == "eps":
            try:
                eps = float(op.attrs.get("value"))
            except Exception:
                eps = None
            break
    if eps is None:
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
    const float* {X_name}, float* {Y_name}, const float* {W_name}, const float* {B_name}, float* {Mean_name}, float* {Rstd_name},
    int M, int N, float eps) {{
  const int m = (int)blockIdx.x;
  if (m >= M) return;
  __shared__ float smem[1024];
  float tsum = 0.0f;
  const float* xrow = {X_name} + (size_t)m * (size_t)N;
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
    {Mean_name}[m] = mean;
    {Rstd_name}[m] = rstd;
  }}
  float* yrow = {Y_name} + (size_t)m * (size_t)N;
  for (int n = (int)threadIdx.x; n < N; n += (int)blockDim.x) {{
    float c = xrow[n] - mean;
    yrow[n] = (c * rstd) * {W_name}[n] + {B_name}[n];
  }}
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[X_name, Y_name, W_name, B_name, Mean_name, Rstd_name],
        scalar_args={"M": "i32", "N": "i32", "eps": "f32"},
        arg_names=[X_name, Y_name, W_name, B_name, Mean_name, Rstd_name, "M", "N", "eps"],
    )
    launch = CudaLaunch(grid=(M, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    out_bindings: Dict[str, Any] = dict(bindings)
    out_bindings.setdefault("eps", float(eps))
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[Y_name, Mean_name, Rstd_name], bindings=out_bindings)


def _kernel_rope_f32(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    # Respect scalar-tensor dims when present; fall back to scalar args otherwise.
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

    if not intent.ops or intent.ops[0].op != "rope":
        raise CudaLoweringError("rope lowering expects a single rope op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("rope expects 3 inputs (input, cos, sin)")
    in_name, cos_name, sin_name = op.inputs
    out_name = op.output

    # Scalar-tensor dims (AI-Bench) if present.
    seq_is_tensor = _is_scalar_tensor(intent, "SEQ_LEN", dtype="i32")
    b_is_tensor = _is_scalar_tensor(intent, "BATCH_NUM", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "HEAD_NUM", dtype="i32")
    d_is_tensor = _is_scalar_tensor(intent, "HEAD_DIM", dtype="i32")

    def _dim_param(name: str) -> str:
        return f"const int* {name}_ptr" if _is_scalar_tensor(intent, name, dtype="i32") else f"int {name}"

    def _dim_load(name: str) -> str:
        if _is_scalar_tensor(intent, name, dtype="i32"):
            return f"const int {name} = {name}_ptr ? {name}_ptr[0] : 0;"
        return ""

    cuda_src = f"""
extern "C" __global__ void {intent.name}(
    const float* {in_name}, const float* {cos_name}, const float* {sin_name}, float* {out_name},
    {_dim_param("SEQ_LEN")}, {_dim_param("BATCH_NUM")}, {_dim_param("HEAD_NUM")}, {_dim_param("HEAD_DIM")}) {{
  {_dim_load("SEQ_LEN")}
  {_dim_load("BATCH_NUM")}
  {_dim_load("HEAD_NUM")}
  {_dim_load("HEAD_DIM")}
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
  float c = {cos_name}[cb];
  float s0 = {sin_name}[cb];
  float x1 = {in_name}[base + (size_t)j];
  float x2 = {in_name}[base + (size_t)half + (size_t)j];
  {out_name}[base + (size_t)j] = x1 * c - x2 * s0;
  {out_name}[base + (size_t)half + (size_t)j] = x1 * s0 + x2 * c;
}}
""".lstrip()

    tensor_args = [in_name, cos_name, sin_name, out_name]
    scalar_args: Dict[str, str] = {}
    arg_names = [in_name, cos_name, sin_name, out_name]
    for dim_name in ["SEQ_LEN", "BATCH_NUM", "HEAD_NUM", "HEAD_DIM"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    # Provide derived symbol for cos/sin shapes (e.g., HEAD_DIM_DIV_2).
    bindings = dict(bindings)
    try:
        cos_shape = _shape_values(intent, cos_name)
        if len(cos_shape) >= 2 and isinstance(cos_shape[1], str):
            bindings.setdefault(str(cos_shape[1]), half)
    except Exception:
        bindings.setdefault("HEAD_DIM_DIV_2", half)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=bindings)


def _kernel_resize_bilinear2x_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "resize":
        raise CudaLoweringError("resize lowering expects a single resize op")
    op = intent.ops[0]
    if len(op.inputs) != 1:
        raise CudaLoweringError("resize expects 1 input")
    src_name = op.inputs[0]
    out_name = op.output

    C = _as_int(bindings.get("C"), name="C")
    H = _as_int(bindings.get("H"), name="H")
    W = _as_int(bindings.get("W"), name="W")
    OH = _as_int(bindings.get("OH", 2 * H), name="OH")
    OW = _as_int(bindings.get("OW", 2 * W), name="OW")
    if OH != 2 * H or OW != 2 * W:
        raise CudaLoweringError("resize MVP supports only 2x upsample")
    hw_fl = 7
    try:
        hw_fl = int(op.attrs.get("hw_fl", 7))
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

    c_is_tensor = _is_scalar_tensor(intent, "C", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "H", dtype="i32")
    w_is_tensor = _is_scalar_tensor(intent, "W", dtype="i32")

    c_param = "const int* C_ptr" if c_is_tensor else "int C"
    h_param = "const int* H_ptr" if h_is_tensor else "int H"
    w_param = "const int* W_ptr" if w_is_tensor else "int W"
    c_load = "const int C = C_ptr ? C_ptr[0] : 0;" if c_is_tensor else ""
    h_load = "const int H = H_ptr ? H_ptr[0] : 0;" if h_is_tensor else ""
    w_load = "const int W = W_ptr ? W_ptr[0] : 0;" if w_is_tensor else ""

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(const int8_t* {src_name}, int8_t* {out_name}, {c_param}, {h_param}, {w_param}, int hw_fl) {{
  {c_load}
  {h_load}
  {w_load}
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

  const int16_t y0x0 = (int16_t){src_name}[src_row0 + x0];
  const int16_t y0x1 = (int16_t){src_name}[src_row0 + x1];
  const int16_t y1x0 = (int16_t){src_name}[src_row1 + x0];
  const int16_t y1x1 = (int16_t){src_name}[src_row1 + x1];
  const int32_t sum1 = (((int32_t)y0x0 * (int32_t)w0) + ((int32_t)y0x1 * (int32_t)w1)) >> hw_fl;
  const int32_t sum2 = (((int32_t)y1x0 * (int32_t)w0) + ((int32_t)y1x1 * (int32_t)w1)) >> hw_fl;
  const int32_t sum = (((sum1 * (int32_t)h0) + (sum2 * (int32_t)h1)) >> hw_fl);

  {out_name}[(size_t)tid] = (int8_t)sum;
}}
""".lstrip()

    # Allow scalar-tensor dims if present (AI-Bench uses scalar tensors for C/H/W).
    tensor_args = [src_name, out_name]
    scalar_args: Dict[str, str] = {"hw_fl": "i32"}
    arg_names = [src_name, out_name]
    for dim_name in ["C", "H", "W"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)
    arg_names.append("hw_fl")

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    bindings = dict(bindings)
    bindings.setdefault("OH", OH)
    bindings.setdefault("OW", OW)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    bindings.setdefault("hw_fl", hw_fl)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=bindings)


def _kernel_warp_q8_8_i8_i16(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "warp":
        raise CudaLoweringError("warp lowering expects a single warp op")
    op = intent.ops[0]
    if len(op.inputs) != 2:
        raise CudaLoweringError("warp expects 2 inputs (src, offset)")
    src_name, offset_name = op.inputs
    out_name = op.output

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
extern "C" __global__ void {intent.name}(const int8_t* {src_name}, const int16_t* {offset_name}, int8_t* {out_name}, int C, int H, int W) {{
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
  const int16_t ov = {offset_name}[off_base + w];
  const int8_t offset_int = (int8_t)(ov >> 8);
  const int8_t offset_frac = (int8_t)(((int16_t)(ov << 8)) >> 8);
  const int8_t indvar = (int8_t)w;
  const int8_t right_i8 = (int8_t)(indvar - offset_int);
  const int8_t left_i8 = (int8_t)(right_i8 - 1);
  const int right = (int)right_i8;
  const int left = (int)left_i8;
  int8_t right_val = 0;
  int8_t left_val = 0;
  if (right >= 0 && right < W) right_val = {src_name}[row_base + (int64_t)right];
  if (left >= 0 && left < W) left_val = {src_name}[row_base + (int64_t)left];
  int16_t outv = (int16_t)((int16_t)right_val << 8);
  outv = (int16_t)(outv + (int16_t)((int16_t)(left_val - right_val) * (int16_t)offset_frac));
  outv = (int16_t)(outv >> 8);
  {out_name}[row_base + w] = (int8_t)outv;
}}
""".lstrip()

    io_spec = _io_spec_from_args(
        intent,
        tensor_args=[src_name, offset_name, out_name],
        scalar_args={"C": "i32", "H": "i32", "W": "i32"},
        arg_names=[src_name, offset_name, out_name, "C", "H", "W"],
    )
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


def _kernel_correlation_i8(intent: IntentFunction, bindings: Dict[str, int]) -> CudaLoweredKernel:
    if not intent.ops or intent.ops[0].op != "correlation":
        raise CudaLoweringError("correlation lowering expects a single correlation op")
    op = intent.ops[0]
    if len(op.inputs) != 3:
        raise CudaLoweringError("correlation expects 3 inputs (src0,src1,out_shift)")
    src0_name, src1_name, out_shift_name = op.inputs
    out_name = op.output

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

    oc_is_tensor = _is_scalar_tensor(intent, "out_channel", dtype="i32")
    ic_is_tensor = _is_scalar_tensor(intent, "in_channel", dtype="i32")
    h_is_tensor = _is_scalar_tensor(intent, "height", dtype="i32")
    w_is_tensor = _is_scalar_tensor(intent, "width", dtype="i32")
    sh_is_tensor = _is_scalar_tensor(intent, "out_shift", dtype="i32")

    oc_param = "const int* out_channel_ptr" if oc_is_tensor else "int out_channel"
    ic_param = "const int* in_channel_ptr" if ic_is_tensor else "int in_channel"
    h_param = "const int* height_ptr" if h_is_tensor else "int height"
    w_param = "const int* width_ptr" if w_is_tensor else "int width"
    sh_param = "const int* out_shift_ptr" if sh_is_tensor else "int out_shift"

    oc_load = "const int out_channel = out_channel_ptr ? out_channel_ptr[0] : 0;" if oc_is_tensor else ""
    ic_load = "const int in_channel = in_channel_ptr ? in_channel_ptr[0] : 0;" if ic_is_tensor else ""
    h_load = "const int height = height_ptr ? height_ptr[0] : 0;" if h_is_tensor else ""
    w_load = "const int width = width_ptr ? width_ptr[0] : 0;" if w_is_tensor else ""
    sh_load = "const int out_shift = out_shift_ptr ? out_shift_ptr[0] : 0;" if sh_is_tensor else ""

    cuda_src = f"""
#include <stdint.h>
extern "C" __global__ void {intent.name}(
    const int8_t* {src0_name}, const int8_t* {src1_name}, int8_t* {out_name},
    {oc_param}, {ic_param}, {h_param}, {w_param}, {sh_param}) {{
  {oc_load}
  {ic_load}
  {h_load}
  {w_load}
  {sh_load}
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
    acc += (int32_t){src0_name}[base + off0] * (int32_t){src1_name}[base + off1];
  }}
  {out_name}[(size_t)tid] = (int8_t)(acc >> sh);
}}
""".lstrip()

    # Prefer scalar tensors if present in the IntentIR signature (AI-Bench).
    tensor_args = [src0_name, src1_name, out_name]
    arg_names = [src0_name, src1_name, out_name]
    scalar_args: Dict[str, str] = {}
    for dim_name in ["out_channel", "in_channel", "height", "width", "out_shift"]:
        if _is_scalar_tensor(intent, dim_name, dtype="i32"):
            tensor_args.append(dim_name)
            arg_names.append(dim_name)
        else:
            scalar_args[dim_name] = "i32"
            arg_names.append(dim_name)

    io_spec = _io_spec_from_args(intent, tensor_args=tensor_args, scalar_args=scalar_args, arg_names=arg_names)
    launch = CudaLaunch(grid=(grid_x, 1, 1), block=(block_x, 1, 1), shared_mem=0)
    return CudaLoweredKernel(kernel_name=intent.name, cuda_src=cuda_src, io_spec=io_spec, launch=launch, output_names=[out_name], bindings=dict(bindings))


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
