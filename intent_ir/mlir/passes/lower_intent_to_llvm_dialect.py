from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backends.cuda.codegen.cpp_driver import lower_intent_to_cuda_kernel_cpp
from backends.spmd_rvv.codegen.cpp_driver import lower_intent_to_c_with_files_cpp
from intent_ir.ir import IntentFunction

from ..convert_to_intent import to_intent
from ..module import IntentMLIRModule
from ..toolchain import detect_mlir_toolchain


_MISSING_BINDINGS_PATTERNS = [
    re.compile(r"missing(?:/invalid)?\s+bindings?:\s*([A-Za-z0-9_/\s,+-]+)", re.IGNORECASE),
    re.compile(r"missing\s+bindings?\s+for\s+([A-Za-z0-9_/\s,+-]+)", re.IGNORECASE),
]
_RVV_TARGET_TRIPLE = "riscv64-unknown-linux-gnu"
_RVV_TARGET_DATALAYOUT = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"


def _missing_binding_names_from_error(msg: str) -> list[str]:
    text = str(msg or "")
    out: list[str] = []
    for pat in _MISSING_BINDINGS_PATTERNS:
        m = pat.search(text)
        if m is None:
            continue
        raw = str(m.group(1) or "")
        for tok in re.split(r"[/,\s]+", raw):
            name = str(tok or "").strip()
            if not name:
                continue
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                continue
            if name not in out:
                out.append(name)
    return out


def _ensure_symbolic_default_bindings(
    *,
    bindings: dict[str, int],
    intent: IntentFunction,
    error: Exception,
) -> dict[str, int]:
    out = dict(bindings or {})
    requested = _missing_binding_names_from_error(str(error))
    if not requested:
        # Some backends only report a generic "missing dim binding". Fall back to
        # all symbolic dims present in the recovered intent payload.
        requested = sorted(_collect_symbols(intent.to_json_dict()))
    changed = False
    for sym in requested:
        key = str(sym).strip()
        if not key:
            continue
        if key in out:
            continue
        out[key] = 1
        changed = True
    return out if changed else dict(bindings or {})


def lower_intent_to_llvm_dialect(module: IntentMLIRModule, *, backend: str | None = None) -> IntentMLIRModule:
    """
    Lower Intent-dialect payload to textual LLVM IR.

    The downstream llvm_* pipelines consume textual LLVM IR, so this pass emits
    non-stub `.ll` by lowering to C first and then compiling to LLVM IR via clang.
    """
    text = str(module.module_text or "")
    if _looks_like_llvm_ir(text):
        out = _clone(module, module_text=text)
        out.meta["llvm_dialect_origin"] = "already_llvm_ir"
        if backend:
            out.meta["llvm_dialect_backend"] = str(backend)
        return out

    intent = _recover_intent(module)
    shape_bindings = _resolve_shape_bindings(module=module, intent=intent)
    selected_backend = str(backend or "").strip().lower()

    if selected_backend == "cuda":
        try:
            try:
                lowered = lower_intent_to_cuda_kernel_cpp(intent, bindings=shape_bindings)
            except Exception as bind_err:
                # Keep CUDA lowering on the CUDA path when the only blocker is missing
                # symbolic bindings (for example output-only symbols like `U`).
                retry_bindings = _ensure_symbolic_default_bindings(
                    bindings=shape_bindings,
                    intent=intent,
                    error=bind_err,
                )
                if retry_bindings != shape_bindings:
                    shape_bindings = dict(retry_bindings)
                    lowered = lower_intent_to_cuda_kernel_cpp(intent, bindings=shape_bindings)
                else:
                    raise
            kernel_name = str(lowered.get("kernel_name") or intent.name or "intent")
            cuda_src = str(lowered.get("cuda_src") or "")
            if not cuda_src.strip():
                raise RuntimeError("empty cuda_src from cuda codegen")
            llvm_ir_text, cc_path = _compile_cuda_src_to_device_llvm_ir(cuda_src, kernel_name=kernel_name)
            out = _clone(module, module_text=llvm_ir_text)
            out.meta["llvm_dialect_origin"] = "lowered_from_intent_cuda_codegen"
            out.meta["llvm_shape_bindings"] = dict(shape_bindings)
            out.meta["llvm_cuda_compiler"] = str(cc_path)
            out.meta["llvm_cuda_kernel_name"] = str(kernel_name)
            out.meta["llvm_target_triple"] = str(_llvm_target_triple(llvm_ir_text))
            out.meta["llvm_dialect_backend"] = "cuda"
            return out
        except Exception as cuda_err:
            # Compatibility fallback: keep previous RVV-style C->LLVM path alive.
            c_src = lower_intent_to_c_with_files_cpp(
                intent,
                shape_bindings=shape_bindings,
                atol=1e-3,
                rtol=1e-3,
                mode="verify",
            )
            llvm_ir_text, cc_path = _compile_c_to_llvm_ir(c_src)
            out = _clone(module, module_text=llvm_ir_text)
            out.meta["llvm_dialect_origin"] = "lowered_from_intent_c_codegen_fallback_for_cuda"
            out.meta["llvm_shape_bindings"] = dict(shape_bindings)
            out.meta["llvm_c_compiler"] = str(cc_path)
            out.meta["llvm_cuda_backend_error"] = f"{type(cuda_err).__name__}: {cuda_err}"
            out.meta["llvm_dialect_backend"] = "cuda"
            out.meta["llvm_target_triple"] = str(_llvm_target_triple(llvm_ir_text))
            return out

    c_src = lower_intent_to_c_with_files_cpp(
        intent,
        shape_bindings=shape_bindings,
        atol=1e-3,
        rtol=1e-3,
        mode="verify",
    )
    llvm_ir_text, cc_path = _compile_c_to_llvm_ir(c_src)
    rvv_retargeted = False
    if selected_backend == "rvv":
        llvm_ir_text, rvv_retargeted = _retarget_host_llvm_ir_to_rvv(llvm_ir_text)

    out = _clone(module, module_text=llvm_ir_text)
    out.meta["llvm_dialect_origin"] = (
        "lowered_from_intent_c_codegen_rvv"
        if selected_backend == "rvv"
        else "lowered_from_intent_c_codegen"
    )
    out.meta["llvm_shape_bindings"] = dict(shape_bindings)
    out.meta["llvm_c_compiler"] = str(cc_path)
    out.meta["llvm_target_triple"] = str(_llvm_target_triple(llvm_ir_text))
    if selected_backend == "rvv":
        out.meta["llvm_rvv_retargeted_from_host"] = bool(rvv_retargeted)
        out.meta["llvm_rvv_target_triple"] = str(_rvv_target_triple())
    if backend:
        out.meta["llvm_dialect_backend"] = str(backend)
    return out


def _recover_intent(module: IntentMLIRModule) -> IntentFunction:
    if isinstance(module.intent_json, dict):
        try:
            return IntentFunction.from_json_dict(dict(module.intent_json))
        except Exception:
            pass
    return to_intent(module)


def _resolve_shape_bindings(*, module: IntentMLIRModule, intent: IntentFunction) -> dict[str, int]:
    out: dict[str, int] = {}

    meta_bindings = (module.meta or {}).get("shape_bindings")
    if isinstance(meta_bindings, dict):
        for k, v in meta_bindings.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                out[key] = max(1, int(v))
            except Exception:
                continue

    intent_json = intent.to_json_dict()
    symbols = _collect_symbols(intent_json)
    for sym in symbols:
        out.setdefault(sym, 1)
    return out


def _collect_symbols(intent_json: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    tensors = intent_json.get("tensors")
    if not isinstance(tensors, dict):
        return out
    for spec in tensors.values():
        if not isinstance(spec, dict):
            continue
        for d in list(spec.get("shape") or []):
            if isinstance(d, dict):
                if str(d.get("kind") or "").strip().lower() == "sym":
                    key = str(d.get("value") or "").strip()
                    if key:
                        out.add(key)
                continue
            if isinstance(d, str):
                key = d.strip()
                if key and (not key.lstrip("-").isdigit()):
                    out.add(key)
    return out


def _compile_c_to_llvm_ir(c_src: str) -> tuple[str, str]:
    toolchain = detect_mlir_toolchain()
    tools = dict(toolchain.get("tools") or {})
    clang_path = str(((tools.get("clang") or {}).get("path") or "")).strip()
    if not clang_path:
        clang_path = str(((tools.get("rvv_cc") or {}).get("path") or "")).strip()
    if not clang_path:
        raise RuntimeError("lower_intent_to_llvm_dialect: clang unavailable for C->LLVM lowering")

    repo_root = Path(__file__).resolve().parents[3]
    runtime_dir = repo_root / "backends" / "spmd_rvv" / "runtime"
    if not runtime_dir.is_dir():
        raise RuntimeError(f"lower_intent_to_llvm_dialect: runtime include dir missing: {runtime_dir}")

    with tempfile.TemporaryDirectory(prefix="intentir_lower_to_llvm_") as td:
        td_path = Path(td)
        c_path = td_path / "kernel.c"
        ll_path = td_path / "kernel.ll"
        c_path.write_text(str(c_src or ""), encoding="utf-8")
        cmd = [
            clang_path,
            "-S",
            "-emit-llvm",
            "-O2",
            "-std=c11",
            "-Wno-unknown-pragmas",
            "-I",
            str(runtime_dir),
            str(c_path),
            "-o",
            str(ll_path),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                "lower_intent_to_llvm_dialect: clang C->LLVM failed: "
                + (str(p.stderr or p.stdout).strip() or f"rc={p.returncode}")
            )
        if not ll_path.is_file():
            raise RuntimeError("lower_intent_to_llvm_dialect: clang did not produce LLVM IR file")
        out = str(ll_path.read_text(encoding="utf-8") or "")
        if not _looks_like_llvm_ir(out):
            raise RuntimeError("lower_intent_to_llvm_dialect: produced LLVM IR is empty/invalid")
        return out, clang_path


def _rvv_target_triple() -> str:
    raw = str(os.getenv("INTENTIR_RVV_LLVM_TARGET_TRIPLE", "")).strip().lower()
    return str(raw or _RVV_TARGET_TRIPLE)


def _retarget_host_llvm_ir_to_rvv(llvm_ir_text: str) -> tuple[str, bool]:
    text = str(llvm_ir_text or "")
    if not text.strip():
        return text, False
    changed = False
    triple = _rvv_target_triple()

    triple_pat = re.compile(r'(target\s+triple\s*=\s*")([^"]*)(")')
    if triple_pat.search(text):
        text2 = triple_pat.sub(rf'\g<1>{triple}\3', text, count=1)
        changed = bool(changed or text2 != text)
        text = text2
    else:
        text = f'target triple = "{triple}"\n{text}'
        changed = True

    datalayout_pat = re.compile(r'(target\s+datalayout\s*=\s*")([^"]*)(")')
    if datalayout_pat.search(text):
        text2 = datalayout_pat.sub(rf'\g<1>{_RVV_TARGET_DATALAYOUT}\3', text, count=1)
        changed = bool(changed or text2 != text)
        text = text2
    else:
        text = f'target datalayout = "{_RVV_TARGET_DATALAYOUT}"\n{text}'
        changed = True

    for key in ("target-cpu", "target-features", "tune-cpu", "target-abi"):
        pat = re.compile(rf'\s*"{re.escape(str(key))}"="[^"]*"')
        text2 = pat.sub("", text)
        changed = bool(changed or text2 != text)
        text = text2
    return text, changed


def _cuda_sm_arch() -> str:
    raw = str(os.getenv("INTENTIR_CUDA_SM", "")).strip().lower()
    if raw.startswith("sm_"):
        return str(raw)
    if raw.isdigit():
        return f"sm_{raw}"
    return "sm_80"


def _resolve_cuda_clang_compiler() -> str:
    toolchain = detect_mlir_toolchain()
    tools = dict(toolchain.get("tools") or {})
    clang_path = str(((tools.get("clang") or {}).get("path") or "")).strip()
    candidates: list[str] = []
    if clang_path:
        cp = Path(clang_path)
        sibling = cp.with_name("clang++")
        if sibling.is_file():
            candidates.append(str(sibling))
        candidates.append(str(cp))
    which_cpp = shutil.which("clang++")
    if which_cpp:
        candidates.append(str(which_cpp))
    which_clang = shutil.which("clang")
    if which_clang:
        candidates.append(str(which_clang))
    seen: set[str] = set()
    dedup: list[str] = []
    for cand in candidates:
        c = str(cand).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        dedup.append(c)
    if not dedup:
        raise RuntimeError("lower_intent_to_llvm_dialect: clang/clang++ unavailable for CUDA C->LLVM lowering")
    return dedup[0]


def _cuda_device_llvm_preamble(*, include_half_bfloat_stubs: bool = True) -> str:
    preamble = (
        "#define __global__ __attribute__((global))\n"
        "#define __host__ __attribute__((host))\n"
        "#define __device__ __attribute__((device))\n"
        "#define __shared__ __attribute__((shared))\n"
        "#define __align__(n) __attribute__((aligned(n)))\n"
        "#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))\n"
        "#define __forceinline__ __inline__ __attribute__((always_inline))\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#include \"__clang_cuda_builtin_vars.h\"\n"
        "#ifndef INFINITY\n"
        "#define INFINITY (__builtin_inff())\n"
        "#endif\n"
        "#ifndef NAN\n"
        "#define NAN (__builtin_nanf(\"\"))\n"
        "#endif\n"
        "__device__ __forceinline__ unsigned int __cvta_generic_to_shared(const void* p) {\n"
        "  return (unsigned int)((unsigned long long)p);\n"
        "}\n"
        "__device__ __forceinline__ unsigned int __umulhi(unsigned int a, unsigned int b) {\n"
        "  return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);\n"
        "}\n"
        "template <typename T>\n"
        "__device__ __forceinline__ T __ldg(const T* p) { return *p; }\n"
        "__device__ __forceinline__ unsigned int __activemask() {\n"
        "  unsigned int m;\n"
        "  asm volatile(\"activemask.b32 %0;\" : \"=r\"(m));\n"
        "  return m;\n"
        "}\n"
        "__device__ __forceinline__ int __shfl_down_sync(unsigned int mask, int var, unsigned int delta, int width = 32) {\n"
        "  int ret;\n"
        "  asm volatile(\"shfl.sync.down.b32 %0, %1, %2, %3, %4;\"\n"
        "               : \"=r\"(ret)\n"
        "               : \"r\"(var), \"r\"(delta), \"r\"(width - 1), \"r\"(mask));\n"
        "  return ret;\n"
        "}\n"
        "__device__ __forceinline__ unsigned int __shfl_down_sync(\n"
        "    unsigned int mask, unsigned int var, unsigned int delta, int width = 32) {\n"
        "  unsigned int ret;\n"
        "  asm volatile(\"shfl.sync.down.b32 %0, %1, %2, %3, %4;\"\n"
        "               : \"=r\"(ret)\n"
        "               : \"r\"(var), \"r\"(delta), \"r\"(width - 1), \"r\"(mask));\n"
        "  return ret;\n"
        "}\n"
        "__device__ __forceinline__ float __shfl_down_sync(unsigned int mask, float var, unsigned int delta, int width = 32) {\n"
        "  union {\n"
        "    float f;\n"
        "    unsigned int i;\n"
        "  } in_u, out_u;\n"
        "  in_u.f = var;\n"
        "  out_u.i = __shfl_down_sync(mask, in_u.i, delta, width);\n"
        "  return out_u.f;\n"
        "}\n"
        "__device__ __forceinline__ long long __shfl_down_sync(unsigned int mask, long long var, unsigned int delta, int width = 32) {\n"
        "  unsigned long long bits = (unsigned long long)var;\n"
        "  unsigned int lo = (unsigned int)(bits & 0xffffffffull);\n"
        "  unsigned int hi = (unsigned int)(bits >> 32);\n"
        "  lo = __shfl_down_sync(mask, lo, delta, width);\n"
        "  hi = __shfl_down_sync(mask, hi, delta, width);\n"
        "  return (long long)((((unsigned long long)hi) << 32) | (unsigned long long)lo);\n"
        "}\n"
        "__device__ __forceinline__ unsigned long long __shfl_down_sync(\n"
        "    unsigned int mask, unsigned long long var, unsigned int delta, int width = 32) {\n"
        "  unsigned int lo = (unsigned int)(var & 0xffffffffull);\n"
        "  unsigned int hi = (unsigned int)(var >> 32);\n"
        "  lo = __shfl_down_sync(mask, lo, delta, width);\n"
        "  hi = __shfl_down_sync(mask, hi, delta, width);\n"
        "  return (((unsigned long long)hi) << 32) | (unsigned long long)lo;\n"
        "}\n"
        "__device__ __forceinline__ long __shfl_down_sync(unsigned int mask, long var, unsigned int delta, int width = 32) {\n"
        "  return (long)__shfl_down_sync(mask, (long long)var, delta, width);\n"
        "}\n"
        "__device__ __forceinline__ unsigned long __shfl_down_sync(\n"
        "    unsigned int mask, unsigned long var, unsigned int delta, int width = 32) {\n"
        "  return (unsigned long)__shfl_down_sync(mask, (unsigned long long)var, delta, width);\n"
        "}\n"
        "__device__ __forceinline__ int __shfl_sync(unsigned int mask, int var, int lane, int width = 32) {\n"
        "  int ret;\n"
        "  asm volatile(\"shfl.sync.idx.b32 %0, %1, %2, %3, %4;\"\n"
        "               : \"=r\"(ret)\n"
        "               : \"r\"(var), \"r\"(lane), \"r\"(width - 1), \"r\"(mask));\n"
        "  return ret;\n"
        "}\n"
        "__device__ __forceinline__ unsigned int __shfl_sync(\n"
        "    unsigned int mask, unsigned int var, int lane, int width = 32) {\n"
        "  unsigned int ret;\n"
        "  asm volatile(\"shfl.sync.idx.b32 %0, %1, %2, %3, %4;\"\n"
        "               : \"=r\"(ret)\n"
        "               : \"r\"(var), \"r\"(lane), \"r\"(width - 1), \"r\"(mask));\n"
        "  return ret;\n"
        "}\n"
        "__device__ __forceinline__ float __shfl_sync(unsigned int mask, float var, int lane, int width = 32) {\n"
        "  union {\n"
        "    float f;\n"
        "    unsigned int i;\n"
        "  } in_u, out_u;\n"
        "  in_u.f = var;\n"
        "  out_u.i = __shfl_sync(mask, in_u.i, lane, width);\n"
        "  return out_u.f;\n"
        "}\n"
        "__device__ __forceinline__ long long __shfl_sync(unsigned int mask, long long var, int lane, int width = 32) {\n"
        "  unsigned long long bits = (unsigned long long)var;\n"
        "  unsigned int lo = (unsigned int)(bits & 0xffffffffull);\n"
        "  unsigned int hi = (unsigned int)(bits >> 32);\n"
        "  lo = __shfl_sync(mask, lo, lane, width);\n"
        "  hi = __shfl_sync(mask, hi, lane, width);\n"
        "  return (long long)((((unsigned long long)hi) << 32) | (unsigned long long)lo);\n"
        "}\n"
        "__device__ __forceinline__ unsigned long long __shfl_sync(\n"
        "    unsigned int mask, unsigned long long var, int lane, int width = 32) {\n"
        "  unsigned int lo = (unsigned int)(var & 0xffffffffull);\n"
        "  unsigned int hi = (unsigned int)(var >> 32);\n"
        "  lo = __shfl_sync(mask, lo, lane, width);\n"
        "  hi = __shfl_sync(mask, hi, lane, width);\n"
        "  return (((unsigned long long)hi) << 32) | (unsigned long long)lo;\n"
        "}\n"
        "__device__ __forceinline__ long __shfl_sync(unsigned int mask, long var, int lane, int width = 32) {\n"
        "  return (long)__shfl_sync(mask, (long long)var, lane, width);\n"
        "}\n"
        "__device__ __forceinline__ unsigned long __shfl_sync(\n"
        "    unsigned int mask, unsigned long var, int lane, int width = 32) {\n"
        "  return (unsigned long)__shfl_sync(mask, (unsigned long long)var, lane, width);\n"
        "}\n"
        "__device__ __forceinline__ int atomicExch(int* addr, int val) {\n"
        "  int old;\n"
        "  asm volatile(\"atom.exch.b32 %0, [%1], %2;\" : \"=r\"(old) : \"l\"(addr), \"r\"(val));\n"
        "  return old;\n"
        "}\n"
        "__device__ __forceinline__ unsigned int atomicExch(unsigned int* addr, unsigned int val) {\n"
        "  unsigned int old;\n"
        "  asm volatile(\"atom.exch.b32 %0, [%1], %2;\" : \"=r\"(old) : \"l\"(addr), \"r\"(val));\n"
        "  return old;\n"
        "}\n"
        "namespace intentir_cuda {\n"
        "template <int BLOCK_THREADS>\n"
        "struct BlockAllreduceF32 { float shared[BLOCK_THREADS]; };\n"
        "template <int BLOCK_THREADS>\n"
        "__device__ __forceinline__ float block_allreduce_sum(float v, BlockAllreduceF32<BLOCK_THREADS>* red) {\n"
        "  red->shared[(int)threadIdx.x] = v;\n"
        "  __syncthreads();\n"
        "  for (int stride = (BLOCK_THREADS >> 1); stride > 0; stride >>= 1) {\n"
        "    if ((int)threadIdx.x < stride) red->shared[(int)threadIdx.x] += red->shared[(int)threadIdx.x + stride];\n"
        "    __syncthreads();\n"
        "  }\n"
        "  return red->shared[0];\n"
        "}\n"
        "template <int BLOCK_THREADS>\n"
        "__device__ __forceinline__ float block_allreduce_max(float v, BlockAllreduceF32<BLOCK_THREADS>* red) {\n"
        "  red->shared[(int)threadIdx.x] = v;\n"
        "  __syncthreads();\n"
        "  for (int stride = (BLOCK_THREADS >> 1); stride > 0; stride >>= 1) {\n"
        "    if ((int)threadIdx.x < stride) {\n"
        "      const float a = red->shared[(int)threadIdx.x];\n"
        "      const float b = red->shared[(int)threadIdx.x + stride];\n"
        "      red->shared[(int)threadIdx.x] = (a > b) ? a : b;\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  return red->shared[0];\n"
        "}\n"
        "template <int BLOCK_THREADS>\n"
        "__device__ __forceinline__ float block_allreduce_min(float v, BlockAllreduceF32<BLOCK_THREADS>* red) {\n"
        "  red->shared[(int)threadIdx.x] = v;\n"
        "  __syncthreads();\n"
        "  for (int stride = (BLOCK_THREADS >> 1); stride > 0; stride >>= 1) {\n"
        "    if ((int)threadIdx.x < stride) {\n"
        "      const float a = red->shared[(int)threadIdx.x];\n"
        "      const float b = red->shared[(int)threadIdx.x + stride];\n"
        "      red->shared[(int)threadIdx.x] = (a < b) ? a : b;\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  return red->shared[0];\n"
        "}\n"
        "template <int BLOCK_THREADS>\n"
        "struct BlockAllreduceI32 { int shared[BLOCK_THREADS]; };\n"
        "template <int BLOCK_THREADS>\n"
        "__device__ __forceinline__ int block_allreduce_max(int v, BlockAllreduceI32<BLOCK_THREADS>* red) {\n"
        "  red->shared[(int)threadIdx.x] = v;\n"
        "  __syncthreads();\n"
        "  for (int stride = (BLOCK_THREADS >> 1); stride > 0; stride >>= 1) {\n"
        "    if ((int)threadIdx.x < stride) {\n"
        "      const int a = red->shared[(int)threadIdx.x];\n"
        "      const int b = red->shared[(int)threadIdx.x + stride];\n"
        "      red->shared[(int)threadIdx.x] = (a > b) ? a : b;\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  return red->shared[0];\n"
        "}\n"
        "template <typename T>\n"
        "__device__ __forceinline__ float to_f32(T v) {\n"
        "  return static_cast<float>(v);\n"
        "}\n"
        "template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int ROWS_PER_THREAD, typename TA, typename TB>\n"
        "__device__ __forceinline__ void matmul_f32_accum_fallback(\n"
        "    const TA* __restrict__ A,\n"
        "    const TB* __restrict__ B,\n"
        "    float* __restrict__ C,\n"
        "    int M,\n"
        "    int N,\n"
        "    int K,\n"
        "    float* __restrict__ As,\n"
        "    float* __restrict__ Bs) {\n"
        "  const int tx = (int)threadIdx.x;\n"
        "  const int ty = (int)threadIdx.y;\n"
        "  const int col = (int)(blockIdx.x * BLOCK_N + tx);\n"
        "  const int block_row = (int)(blockIdx.y * BLOCK_M);\n"
        "  const int row0 = block_row + ty;\n"
        "  float acc[ROWS_PER_THREAD];\n"
        "  #pragma unroll\n"
        "  for (int i = 0; i < ROWS_PER_THREAD; ++i) acc[i] = 0.0f;\n"
        "  for (int kt = 0; kt < K; kt += BLOCK_K) {\n"
        "    if (tx < BLOCK_K) {\n"
        "      #pragma unroll\n"
        "      for (int i = 0; i < ROWS_PER_THREAD; ++i) {\n"
        "        const int r = ty + i * THREAD_M;\n"
        "        if (r < BLOCK_M) {\n"
        "          const int row = block_row + r;\n"
        "          if (row < M && (kt + tx) < K)\n"
        "            As[r * BLOCK_K + tx] = to_f32(A[(size_t)row * (size_t)K + (size_t)(kt + tx)]);\n"
        "          else\n"
        "            As[r * BLOCK_K + tx] = 0.0f;\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "    if (ty < BLOCK_K) {\n"
        "      if (col < N && (kt + ty) < K)\n"
        "        Bs[ty * BLOCK_N + tx] = to_f32(B[(size_t)(kt + ty) * (size_t)N + (size_t)col]);\n"
        "      else\n"
        "        Bs[ty * BLOCK_N + tx] = 0.0f;\n"
        "    }\n"
        "    __syncthreads();\n"
        "    #pragma unroll\n"
        "    for (int k0 = 0; k0 < BLOCK_K; ++k0) {\n"
        "      const float b0 = Bs[k0 * BLOCK_N + tx];\n"
        "      #pragma unroll\n"
        "      for (int i = 0; i < ROWS_PER_THREAD; ++i) {\n"
        "        const int r = ty + i * THREAD_M;\n"
        "        if (r < BLOCK_M) acc[i] = __builtin_fmaf(As[r * BLOCK_K + k0], b0, acc[i]);\n"
        "      }\n"
        "    }\n"
        "    __syncthreads();\n"
        "  }\n"
        "  if (col < N) {\n"
        "    #pragma unroll\n"
        "    for (int i = 0; i < ROWS_PER_THREAD; ++i) {\n"
        "      const int row = row0 + i * THREAD_M;\n"
        "      if (row < M) C[(size_t)row * (size_t)N + (size_t)col] = acc[i];\n"
        "    }\n"
        "  }\n"
        "}\n"
        "} // namespace intentir_cuda\n"
        "__device__ __forceinline__ float acosf(float x) { return __builtin_acosf(x); }\n"
        "__device__ __forceinline__ float acoshf(float x) { return __builtin_acoshf(x); }\n"
        "__device__ __forceinline__ float asinf(float x) { return __builtin_asinf(x); }\n"
        "__device__ __forceinline__ float asinhf(float x) { return __builtin_asinhf(x); }\n"
        "__device__ __forceinline__ float atanf(float x) { return __builtin_atanf(x); }\n"
        "__device__ __forceinline__ float atan2f(float y, float x) { return __builtin_atan2f(y, x); }\n"
        "__device__ __forceinline__ float atanhf(float x) { return __builtin_atanhf(x); }\n"
        "__device__ __forceinline__ float ceilf(float x) { return __builtin_ceilf(x); }\n"
        "__device__ __forceinline__ float copysignf(float x, float y) { return __builtin_copysignf(x, y); }\n"
        "__device__ __forceinline__ float cosf(float x) { return __builtin_cosf(x); }\n"
        "__device__ __forceinline__ float coshf(float x) { return __builtin_coshf(x); }\n"
        "__device__ __forceinline__ float erff(float x) { return __builtin_erff(x); }\n"
        "__device__ __forceinline__ float erfcf(float x) { return __builtin_erfcf(x); }\n"
        "__device__ __forceinline__ float exp2f(float x) { return __builtin_exp2f(x); }\n"
        "__device__ __forceinline__ float __expf(float x) { return __builtin_expf(x); }\n"
        "__device__ __forceinline__ float expf(float x) { return __builtin_expf(x); }\n"
        "__device__ __forceinline__ float expm1f(float x) { return __builtin_expm1f(x); }\n"
        "__device__ __forceinline__ float floorf(float x) { return __builtin_floorf(x); }\n"
        "__device__ __forceinline__ float fmaf(float x, float y, float z) { return __builtin_fmaf(x, y, z); }\n"
        "__device__ __forceinline__ float fmaxf(float a, float b) { return (a > b) ? a : b; }\n"
        "__device__ __forceinline__ float fminf(float a, float b) { return (a < b) ? a : b; }\n"
        "__device__ __forceinline__ float fabsf(float a) { return (a < 0.0f) ? -a : a; }\n"
        "__device__ __forceinline__ int isfinite(float x) { return (x == x) && (x != INFINITY) && (x != -INFINITY); }\n"
        "__device__ __forceinline__ int isnan(float x) { return !(x == x); }\n"
        "__device__ __forceinline__ int isinf(float x) { return (x == INFINITY) || (x == -INFINITY); }\n"
        "__device__ __forceinline__ int isfinite(double x) { return (x == x) && (x != (double)INFINITY) && (x != -(double)INFINITY); }\n"
        "__device__ __forceinline__ int isnan(double x) { return !(x == x); }\n"
        "__device__ __forceinline__ int isinf(double x) { return (x == (double)INFINITY) || (x == -(double)INFINITY); }\n"
        "__device__ __forceinline__ float log10f(float x) { return __builtin_log10f(x); }\n"
        "__device__ __forceinline__ float log1pf(float x) { return __builtin_log1pf(x); }\n"
        "__device__ __forceinline__ float log2f(float x) { return __builtin_log2f(x); }\n"
        "__device__ __forceinline__ float logf(float a) { return __builtin_logf(a); }\n"
        "__device__ __forceinline__ float powf(float x, float y) { return __builtin_powf(x, y); }\n"
        "__device__ __forceinline__ float rintf(float x) { return __builtin_rintf(x); }\n"
        "__device__ __forceinline__ float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }\n"
        "__device__ __forceinline__ float sinf(float x) { return __builtin_sinf(x); }\n"
        "__device__ __forceinline__ float sinhf(float x) { return __builtin_sinhf(x); }\n"
        "__device__ __forceinline__ float sqrtf(float x) { return __builtin_sqrtf(x); }\n"
        "__device__ __forceinline__ float tanf(float x) { return __builtin_tanf(x); }\n"
        "__device__ __forceinline__ float tanhf(float x) { return __builtin_tanhf(x); }\n"
        "__device__ __forceinline__ float truncf(float x) { return __builtin_truncf(x); }\n"
        "__device__ __forceinline__ float __fdividef(float a, float b) { return a / b; }\n"
        "typedef struct __align__(16) { float x, y, z, w; } float4;\n"
    )
    if not include_half_bfloat_stubs:
        return preamble
    return preamble + (
        "typedef struct __align__(2) { unsigned short x; } __half;\n"
        "__device__ __forceinline__ __half __float2half(float v) {\n"
        "  __half out;\n"
        "  asm(\"cvt.rn.f16.f32 %0, %1;\" : \"=h\"(out.x) : \"f\"(v));\n"
        "  return out;\n"
        "}\n"
        "__device__ __forceinline__ float __half2float(__half v) {\n"
        "  float out;\n"
        "  asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(out) : \"h\"(v.x));\n"
        "  return out;\n"
        "}\n"
        "typedef struct __align__(2) { unsigned short x; } __nv_bfloat16;\n"
        "__device__ __forceinline__ __nv_bfloat16 __float2bfloat16(float v) {\n"
        "  union { float f; unsigned int u; } in_u;\n"
        "  in_u.f = v;\n"
        "  const unsigned int lsb = (in_u.u >> 16) & 1u;\n"
        "  const unsigned int rounded = in_u.u + 0x7FFFu + lsb;\n"
        "  __nv_bfloat16 out;\n"
        "  out.x = (unsigned short)(rounded >> 16);\n"
        "  return out;\n"
        "}\n"
        "__device__ __forceinline__ float __bfloat162float(__nv_bfloat16 v) {\n"
        "  union { float f; unsigned int u; } out_u;\n"
        "  out_u.u = ((unsigned int)v.x) << 16;\n"
        "  return out_u.f;\n"
        "}\n"
    )


def _sanitize_cuda_codegen_src(cuda_src: str) -> str:
    src = str(cuda_src or "")
    if not src.strip():
        return ""
    out_lines: list[str] = []
    for raw in src.splitlines():
        line = str(raw or "")
        stripped = line.strip().replace(" ", "")
        if stripped in {
            "#include<math.h>",
            "#include<cmath>",
            "#include<cuda_fp16.h>",
            "#include<cuda_bf16.h>",
            "#include\"kernels/reduce.cuh\"",
            "#include<kernels/reduce.cuh>",
            "#include\"kernels/matmul_fallback.cuh\"",
            "#include<kernels/matmul_fallback.cuh>",
        }:
            continue
        out_lines.append(line)
    out = "\n".join(out_lines)
    if src.endswith("\n"):
        out += "\n"
    return out


def _cuda_codegen_requires_half_bfloat_stubs(cuda_src: str) -> bool:
    s = str(cuda_src or "").lower()
    # Runtime kernels like reduce.cuh include CUDA half/bfloat headers themselves.
    if "reduce.cuh" in s:
        return False
    # matmul_fallback.cuh pulls cuda_fp16.h transitively.
    if "matmul_fallback.cuh" in s:
        return False
    if "cuda_fp16.h" in s or "cuda_bf16.h" in s:
        return False
    return True


def _resolve_cuda_include_dirs() -> list[str]:
    candidates: list[Path] = []

    def _add_if_dir(p: Path) -> None:
        if p.is_dir():
            candidates.append(p)

    for key in ("INTENTIR_CUDA_HOME", "CUDA_HOME", "CUDA_PATH"):
        raw = str(os.getenv(key, "")).strip()
        if raw:
            _add_if_dir(Path(raw) / "include")

    ptxas_path = shutil.which("ptxas")
    if ptxas_path:
        p = Path(ptxas_path).resolve()
        _add_if_dir(p.parent.parent / "include")

    for raw in ("/usr/local/cuda/include", "/opt/cuda/include"):
        _add_if_dir(Path(raw))

    out: list[str] = []
    seen: set[str] = set()
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _compile_cuda_src_to_device_llvm_ir(cuda_src: str, *, kernel_name: str) -> tuple[str, str]:
    clang_path = _resolve_cuda_clang_compiler()
    sm_arch = _cuda_sm_arch()
    runtime_dir = Path(__file__).resolve().parents[3] / "backends" / "cuda" / "runtime"
    if not runtime_dir.is_dir():
        raise RuntimeError(
            "lower_intent_to_llvm_dialect: CUDA runtime include dir missing: "
            f"{runtime_dir}"
        )
    with tempfile.TemporaryDirectory(prefix=f"intentir_cuda_to_llvm_{kernel_name}_") as td:
        td_path = Path(td)
        cu_path = td_path / f"{kernel_name}.cu"
        ll_path = td_path / f"{kernel_name}.ll"
        sanitized = _sanitize_cuda_codegen_src(cuda_src)
        wrapped = _cuda_device_llvm_preamble(
            include_half_bfloat_stubs=_cuda_codegen_requires_half_bfloat_stubs(sanitized)
        ) + "\n" + sanitized
        cu_path.write_text(str(wrapped), encoding="utf-8")
        cmd = [
            clang_path,
            "--cuda-device-only",
            f"--cuda-gpu-arch={sm_arch}",
            "-nocudainc",
            "-nocudalib",
            "-I",
            str(runtime_dir),
            "-O3",
            "-S",
            "-emit-llvm",
            str(cu_path),
            "-o",
            str(ll_path),
        ]
        for include_dir in _resolve_cuda_include_dirs():
            cmd.extend(["-isystem", str(include_dir)])
        # Some environments only expose `clang` and require `-x cuda`.
        if Path(clang_path).name == "clang":
            cmd.insert(1, "-x")
            cmd.insert(2, "cuda")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and ll_path.is_file():
            out = str(ll_path.read_text(encoding="utf-8") or "")
            if _looks_like_llvm_ir(out):
                triple = _llvm_target_triple(out)
                if "nvptx" in triple:
                    return out, clang_path
            first_error = "lower_intent_to_llvm_dialect: produced CUDA LLVM IR is empty/invalid"
        else:
            first_error = (
                "lower_intent_to_llvm_dialect: clang CUDA->LLVM failed: "
                + (str(p.stderr or p.stdout).strip() or f"rc={p.returncode}")
            )

        # Fallback: compile raw CUDA source with system CUDA includes for
        # kernels that rely on extra CUDA headers/intrinsics.
        raw_cu_path = td_path / f"{kernel_name}.raw.cu"
        raw_ll_path = td_path / f"{kernel_name}.raw.ll"
        raw_cu_path.write_text(str(cuda_src or ""), encoding="utf-8")
        fallback_cmd = [
            clang_path,
            "--cuda-device-only",
            f"--cuda-gpu-arch={sm_arch}",
            "-std=c++17",
            "-O3",
            "-S",
            "-emit-llvm",
            "-I",
            str(runtime_dir),
            str(raw_cu_path),
            "-o",
            str(raw_ll_path),
        ]
        for include_dir in _resolve_cuda_include_dirs():
            fallback_cmd.extend(["-isystem", str(include_dir)])
        if Path(clang_path).name == "clang":
            fallback_cmd.insert(1, "-x")
            fallback_cmd.insert(2, "cuda")
        p2 = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if p2.returncode != 0:
            second_error = str(p2.stderr or p2.stdout).strip() or f"rc={p2.returncode}"
            raise RuntimeError(f"{first_error}\n-- fallback compile --\n{second_error}")
        if not raw_ll_path.is_file():
            raise RuntimeError(f"{first_error}\n-- fallback compile --\nclang did not produce CUDA LLVM IR file")
        out2 = str(raw_ll_path.read_text(encoding="utf-8") or "")
        if not _looks_like_llvm_ir(out2):
            raise RuntimeError(f"{first_error}\n-- fallback compile --\nproduced CUDA LLVM IR is empty/invalid")
        triple2 = _llvm_target_triple(out2)
        if "nvptx" not in triple2:
            raise RuntimeError(
                f"{first_error}\n-- fallback compile --\nCUDA LLVM target triple is not nvptx: {triple2!r}"
            )
        return out2, clang_path


def _looks_like_llvm_ir(text: str) -> bool:
    s = str(text or "")
    return ("define " in s and "{" in s and "}" in s) or ("; ModuleID" in s)


def _llvm_target_triple(llvm_ir_text: str) -> str:
    m = re.search(r'target\s+triple\s*=\s*"([^"]+)"', str(llvm_ir_text or ""))
    return str(m.group(1) if m else "").strip().lower()


def _clone(module: IntentMLIRModule, *, module_text: str) -> IntentMLIRModule:
    return IntentMLIRModule(
        module_text=str(module_text or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
