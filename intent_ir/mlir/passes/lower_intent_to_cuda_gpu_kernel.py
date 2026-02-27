from __future__ import annotations

import base64
import json
import math
import os
import re
from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(str(name), "")
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if not s:
        return bool(default)
    return s in {"1", "true", "yes", "y", "on"}


def _resolve_dim_int(dim: Any, bindings: Mapping[str, Any]) -> int | None:
    if dim is None:
        return None
    kind = getattr(dim, "kind", None)
    raw = getattr(dim, "value", dim) if kind in {"sym", "const"} else dim
    if isinstance(raw, int):
        return int(raw)
    key = str(raw).strip()
    if not key:
        return None
    if key in bindings:
        try:
            return int(bindings[key])
        except Exception:
            return None
    # Minimal support for legacy dims like "M + 1".
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(\d+)$", key)
    if m:
        base = str(m.group(1))
        delta = int(m.group(2))
        if base in bindings:
            try:
                return int(bindings[base]) + int(delta)
            except Exception:
                return None
    try:
        return int(key)
    except Exception:
        return None


def _mlir_ident(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return "v"
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    if out and out[0].isdigit():
        out = f"v_{out}"
    return out or "v"


def _b64_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _symbols_from_intent(intent: IntentFunction) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in dict(intent.tensors or {}).values():
        for d in list(getattr(t, "shape", []) or []):
            if getattr(d, "kind", None) != "sym":
                continue
            name = str(getattr(d, "value", "")).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def _dtype_to_mlir(dtype: str) -> str:
    dt = str(dtype or "").strip().lower()
    if dt == "f32":
        return "f32"
    if dt == "f16":
        return "f16"
    if dt == "bf16":
        return "bf16"
    if dt in {"i1", "bool"}:
        return "i1"
    if dt == "i8":
        return "i8"
    if dt == "i32":
        return "i32"
    if dt == "i64":
        return "i64"
    raise RuntimeError(f"unsupported dtype for cuda real-mlir wave: {dtype}")


def _dtype_to_mlir_memref_elem(dtype: str) -> str:
    """
    Memory element type for tensors passed across the ABI boundary.

    Note: torch bool tensors are byte-addressed; keep memrefs as i8 and convert
    to i1 for control-flow and comparisons.
    """
    dt = str(dtype or "").strip().lower()
    if dt in {"i1", "bool"}:
        return "i8"
    return _dtype_to_mlir(dt)


def _collect_io_arg_order(intent: IntentFunction) -> tuple[list[str], list[str]]:
    tensors = dict(intent.tensors or {})
    ops = list(intent.ops or [])
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    produced = {str(getattr(op, "output", "")).strip() for op in ops if str(getattr(op, "output", "")).strip()}
    used: set[str] = set()
    for op in ops:
        for inp in list(getattr(op, "inputs", []) or []):
            name = str(inp).strip()
            if name:
                used.add(name)
    external_inputs = sorted([n for n in used if n in tensors and n not in produced])
    out_names = [n for n in outputs if n in tensors and n not in set(external_inputs)]
    return external_inputs, out_names


def lower_intent_to_cuda_gpu_kernel(
    module: IntentMLIRModule,
    *,
    backend: str | None = None,
    **_: object,
) -> IntentMLIRModule:
    """
    Phase3B (CUDA wave): emit a minimal GPU-dialect kernel for simple elementwise intent graphs.

    The output is a *parseable* MLIR module (gpu+scf+arith+math), specialized to the
    concrete `shape_bindings` provided in `module.meta`. Downstream pipeline lowers it
    to NVVM/LLVM and then `mlir-translate` emits textual LLVM IR.

    This pass is gated by `INTENTIR_REAL_MLIR=1` to avoid perturbing legacy runs.
    """
    b = str(backend or "").strip().lower()
    if b and not b.startswith("cuda"):
        return module
    if not _env_flag("INTENTIR_REAL_MLIR", default=False):
        return module

    intent = to_intent(module)
    if not isinstance(intent, IntentFunction):
        raise RuntimeError("to_intent did not return IntentFunction")

    bindings_raw = (module.meta or {}).get("shape_bindings") if isinstance(module.meta, dict) else None
    if not isinstance(bindings_raw, Mapping) or not dict(bindings_raw):
        raise RuntimeError("cuda real-mlir lowering requires module.meta.shape_bindings")
    bindings: dict[str, Any] = {str(k): v for k, v in dict(bindings_raw).items() if str(k).strip()}

    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if len(outputs) != 1:
        raise RuntimeError(f"cuda real-mlir wave supports single-output intents only; outputs={outputs}")
    out_name = outputs[0]
    out_tt = (intent.tensors or {}).get(out_name)
    if out_tt is None:
        raise RuntimeError(f"missing output tensor spec: {out_name}")

    out_shape = list(getattr(out_tt, "shape", []) or [])
    out_dims: list[int] = []
    for d in out_shape:
        iv = _resolve_dim_int(d, bindings)
        if iv is None or iv <= 0:
            raise RuntimeError(f"unbound output dim: tensor={out_name} dim={d}")
        out_dims.append(int(iv))
    out_total = int(math.prod(out_dims)) if out_dims else 1
    if out_total <= 0:
        raise RuntimeError("invalid output numel for cuda real-mlir wave")

    ext_inputs, out_names = _collect_io_arg_order(intent)
    if out_name not in out_names:
        raise RuntimeError(f"output tensor not in io arg order: {out_name}")

    # Build flattened, static-shape memref args so `convert-gpu-to-nvvm` can use
    # bare-ptr calling convention.
    #
    # This pass is intentionally small-scope. We support:
    # - elementwise tensors matching the output numel
    # - scalar tensors (shape=[])
    # - simple broadcast (rank-2 output, rank-1 input) for bias-style patterns
    #   (shape=[N] or shape=[M]).
    arg_specs: dict[str, dict[str, Any]] = {}
    out_rank = int(len(out_dims))
    out_m = int(out_dims[0]) if out_rank == 2 else None
    out_n = int(out_dims[1]) if out_rank == 2 else None
    for name in [*ext_inputs, *out_names]:
        tt = (intent.tensors or {}).get(name)
        if tt is None:
            raise RuntimeError(f"missing tensor spec: {name}")
        dtype = str(getattr(tt, "dtype", "f32"))
        scalar_ty = _dtype_to_mlir(dtype)
        memref_elem_ty = _dtype_to_mlir_memref_elem(dtype)
        shape = list(getattr(tt, "shape", []) or [])
        dims: list[int] = []
        for d in shape:
            iv = _resolve_dim_int(d, bindings)
            if iv is None or iv <= 0:
                raise RuntimeError(f"unbound tensor dim: tensor={name} dim={d}")
            dims.append(int(iv))

        is_scalar = len(dims) == 0
        numel = 1 if is_scalar else int(math.prod(dims))
        if numel <= 0:
            raise RuntimeError(f"invalid tensor numel: tensor={name} numel={numel}")

        broadcast = "scalar" if is_scalar else ""
        if not is_scalar:
            if int(numel) == int(out_total):
                broadcast = "elementwise"
            elif out_rank == 2 and len(dims) == 1 and out_m is not None and out_n is not None:
                if int(numel) == int(out_n):
                    broadcast = "broadcast_n"
                elif int(numel) == int(out_m):
                    broadcast = "broadcast_m"
                else:
                    raise RuntimeError(
                        "cuda real-mlir wave supports only elementwise tensors + 1D bias broadcast; "
                        f"tensor={name} shape={dims} numel={numel} expected_out=[{out_m},{out_n}]"
                    )
            else:
                raise RuntimeError(
                    "cuda real-mlir wave supports only elementwise tensors + 1D bias broadcast; "
                    f"tensor={name} shape={dims} numel={numel} expected_out_rank={out_rank} out_total={out_total}"
                )

        memref_n = 1 if is_scalar else int(numel)
        arg_specs[str(name)] = {
            "dtype": str(dtype),
            "scalar_ty": str(scalar_ty),
            "memref_elem_ty": str(memref_elem_ty),
            "memref": f"memref<{memref_n}x{memref_elem_ty}, 1>",
            "scalar": bool(is_scalar),
            "broadcast": str(broadcast),
            "dims": list(dims),
        }

    kernel_name = _mlir_ident(str(intent.name or out_name or "kernel"))
    intent_json = intent.to_json_dict()
    b64 = _b64_json(intent_json)
    symbols_attr = "[" + ", ".join([f"\"{s}\"" for s in _symbols_from_intent(intent)]) + "]"

    # Build kernel args in runtime_io_spec order.
    arg_list = [f"%{_mlir_ident(n)}: {arg_specs[n]['memref']}" for n in [*ext_inputs, *out_names]]
    arg_sig = ", ".join(arg_list)

    need_row = any(str(spec.get("broadcast") or "") == "broadcast_m" for spec in arg_specs.values())
    need_col = any(str(spec.get("broadcast") or "") == "broadcast_n" for spec in arg_specs.values())
    need_rowcol = bool(need_row or need_col)
    if need_rowcol and out_rank != 2:
        raise RuntimeError("internal error: broadcast patterns require rank-2 output")

    tmp_idx = 0

    def _fresh(prefix: str) -> str:
        nonlocal tmp_idx
        tmp_idx += 1
        return f"%{_mlir_ident(prefix)}_{tmp_idx}"

    elems_per_thread = 1
    raw_elems = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_ELEMS_PER_THREAD", "") or "").strip()
    if raw_elems:
        try:
            elems_per_thread = int(raw_elems)
        except Exception:
            raise RuntimeError(
                "invalid INTENTIR_CUDA_REAL_MLIR_ELEMS_PER_THREAD; expected int, got "
                f"{raw_elems!r}"
            )
    if elems_per_thread not in {1, 2, 4}:
        raise RuntimeError(
            "unsupported cuda real-mlir elems_per_thread; expected one of {1,2,4}, got "
            f"{elems_per_thread}"
        )

    def _eligible_for_vectorization() -> bool:
        if elems_per_thread <= 1:
            return False
        if int(out_total) <= 0:
            return False
        if int(out_total) % int(elems_per_thread) != 0:
            return False
        # First cut: only contiguous elementwise + scalar broadcasts, no row/col addressing.
        if need_rowcol:
            return False
        for spec in arg_specs.values():
            if str(spec.get("broadcast") or "") not in {"scalar", "elementwise"}:
                return False
            if str(spec.get("scalar_ty") or "") != "f32":
                return False
            if str(spec.get("memref_elem_ty") or "") != "f32":
                return False
        # Require an f32-only dataflow (no comparisons, no bool, no casts).
        for op in list(intent.ops or []):
            op_name = str(getattr(op, "op", "")).strip()
            if op_name in {"eq", "ne", "lt", "le", "gt", "ge", "where", "cast"}:
                return False
        return True

    vectorize = _eligible_for_vectorization()

    # SSA names for memref arguments. In vectorize mode we may introduce
    # alignment-assumed aliases to unlock aligned vector memory ops downstream.
    arg_ssa: dict[str, str] = {str(name): f"%{_mlir_ident(name)}" for name in arg_specs}

    raw_fast_math = os.getenv("INTENTIR_CUDA_USE_FAST_MATH", "0").strip().lower()
    use_fast_math = raw_fast_math in {"1", "true", "yes", "y", "on"}
    fm = " fastmath<fast>" if use_fast_math else ""

    def _as_f32_const(v: Any) -> str:
        try:
            return repr(float(v))
        except Exception:
            return repr(float(str(v)))

    out_spec = arg_specs[out_name]
    out_memref = str(out_spec["memref"])
    out_memref_elem_ty = str(out_spec.get("memref_elem_ty") or "")

    vec_ty = f"vector<{int(elems_per_thread)}xf32>"

    def _emit_elementwise_vector_for_base(base_idx_ssa: str) -> list[str]:
        if not vectorize:
            raise RuntimeError("internal error: vector emitter called when vectorize=false")

        loaded: dict[str, str] = {}
        loaded_ty: dict[str, str] = {}
        computed: dict[str, str] = {}
        computed_ty: dict[str, str] = {}
        out_lines: list[str] = []

        def _infer_value_ty(name: str) -> str:
            if name in computed_ty:
                return str(computed_ty[name])
            if name in loaded_ty:
                return str(loaded_ty[name])
            # Vector mode is currently f32-only.
            return str(vec_ty)

        def _load_tensor(name: str) -> list[str]:
            if name in computed:
                return []
            if name in loaded:
                return []
            spec = arg_specs.get(name)
            if spec is None:
                raise RuntimeError(f"unknown tensor referenced: {name}")
            memref = str(spec["memref"])
            bcast = str(spec.get("broadcast") or "")
            if bcast == "scalar":
                ssa_scalar = _fresh(f"{name}_scalar")
                ssa_vec = _fresh(f"{name}_splat")
                lines = [
                    f"        {ssa_scalar} = memref.load {arg_ssa[str(name)]}[%c0] : {memref}",
                    f"        {ssa_vec} = vector.splat {ssa_scalar} : {vec_ty}",
                ]
                loaded[name] = ssa_vec
                loaded_ty[name] = str(vec_ty)
                return lines
            if bcast == "elementwise":
                ssa_vec = _fresh(f"{name}_vec")
                lines = [
                    f"        {ssa_vec} = vector.load {arg_ssa[str(name)]}[{base_idx_ssa}] : {memref}, {vec_ty}",
                ]
                loaded[name] = ssa_vec
                loaded_ty[name] = str(vec_ty)
                return lines
            raise RuntimeError(f"unsupported broadcast kind in cuda real-mlir vectorize mode: {bcast}")

        # Emit ops (vector elementwise evaluation).
        for op in list(intent.ops or []):
            op_name = str(getattr(op, "op", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            outv = str(getattr(op, "output", "")).strip()
            attrs = dict(getattr(op, "attrs", {}) or {})
            if not op_name or not outv:
                raise RuntimeError("invalid op in intent")

            # Materialize inputs.
            in_ssa: list[str] = []
            in_ty: list[str] = []
            for inp in inputs:
                if inp in computed:
                    in_ssa.append(computed[inp])
                else:
                    out_lines.extend(_load_tensor(inp))
                    in_ssa.append(loaded[inp])
                in_ty.append(_infer_value_ty(inp))

            out_ty = _infer_value_ty(outv)

            if op_name == "const":
                if inputs:
                    raise RuntimeError("const expects 0 inputs")
                if out_ty != vec_ty:
                    raise RuntimeError(f"const vectorize mismatch (out_ty={out_ty})")
                ssa_scalar = _fresh("const_scalar")
                dst = _fresh("const")
                out_lines.append(f"        {ssa_scalar} = arith.constant {_as_f32_const(attrs.get('value'))} : f32")
                out_lines.append(f"        {dst} = vector.splat {ssa_scalar} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "identity":
                if len(in_ssa) != 1:
                    raise RuntimeError("identity expects 1 input")
                computed[outv] = in_ssa[0]
                computed_ty[outv] = in_ty[0]
                continue

            if op_name == "abs":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("abs vectorize expects vector<f32>")
                dst = _fresh("abs")
                out_lines.append(f"        {dst} = math.absf {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "floor":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("floor vectorize expects vector<f32>")
                dst = _fresh("floor")
                out_lines.append(f"        {dst} = math.floor {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "ceil":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("ceil vectorize expects vector<f32>")
                dst = _fresh("ceil")
                out_lines.append(f"        {dst} = math.ceil {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "neg":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("neg vectorize expects vector<f32>")
                dst = _fresh("neg")
                out_lines.append(f"        {dst} = arith.negf {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name in {"add", "sub", "mul", "div"}:
                if len(in_ssa) != 2 or in_ty[0] != vec_ty or in_ty[1] != vec_ty:
                    raise RuntimeError(f"{op_name} vectorize expects vector<f32> binary")
                dst = _fresh(op_name)
                arith = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf"}[op_name]
                out_lines.append(f"        {dst} = {arith} {in_ssa[0]}, {in_ssa[1]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name in {"max", "min"}:
                if len(in_ssa) != 2 or in_ty[0] != vec_ty or in_ty[1] != vec_ty:
                    raise RuntimeError(f"{op_name} vectorize expects vector<f32> binary")
                dst = _fresh(op_name)
                arith = {"max": "arith.maximumf", "min": "arith.minimumf"}[op_name]
                out_lines.append(f"        {dst} = {arith} {in_ssa[0]}, {in_ssa[1]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "relu":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("relu vectorize expects vector<f32>")
                c0s = _fresh("c0f")
                c0v = _fresh("c0v")
                dst = _fresh("relu")
                out_lines.append(f"        {c0s} = arith.constant 0.0 : f32")
                out_lines.append(f"        {c0v} = vector.splat {c0s} : {vec_ty}")
                out_lines.append(f"        {dst} = arith.maximumf {in_ssa[0]}, {c0v}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "exp":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("exp vectorize expects vector<f32>")
                dst = _fresh("exp")
                out_lines.append(f"        {dst} = math.exp {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "exp2":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("exp2 vectorize expects vector<f32>")
                dst = _fresh("exp2")
                out_lines.append(f"        {dst} = math.exp2 {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "log":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("log vectorize expects vector<f32>")
                dst = _fresh("log")
                out_lines.append(f"        {dst} = math.log {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "sqrt":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("sqrt vectorize expects vector<f32>")
                dst = _fresh("sqrt")
                out_lines.append(f"        {dst} = math.sqrt {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "rsqrt":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("rsqrt vectorize expects vector<f32>")
                dst = _fresh("rsqrt")
                out_lines.append(f"        {dst} = math.rsqrt {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name == "erf":
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError("erf vectorize expects vector<f32>")
                dst = _fresh("erf")
                out_lines.append(f"        {dst} = math.erf {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            if op_name in {"sin", "cos", "tan"}:
                if len(in_ssa) != 1 or in_ty[0] != vec_ty:
                    raise RuntimeError(f"{op_name} vectorize expects vector<f32>")
                dst = _fresh(op_name)
                mop = {"sin": "math.sin", "cos": "math.cos", "tan": "math.tan"}[op_name]
                out_lines.append(f"        {dst} = {mop} {in_ssa[0]}{fm} : {vec_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(vec_ty)
                continue

            raise RuntimeError(f"unsupported op in cuda real-mlir vectorize mode: {op_name}")

        final_ssa = computed.get(out_name)
        if final_ssa is None:
            last_out = str(getattr(list(intent.ops or [])[-1], "output", "")).strip() if intent.ops else ""
            final_ssa = computed.get(last_out)
        if not final_ssa:
            raise RuntimeError(f"no computed value for output tensor: {out_name}")
        out_val_ty = _infer_value_ty(out_name)
        if out_memref_elem_ty != "f32" or out_val_ty != vec_ty:
            raise RuntimeError("vectorize mode supports only f32 outputs")
        out_lines.append(
            f"        vector.store {final_ssa}, {arg_ssa[str(out_name)]}[{base_idx_ssa}] : {out_memref}, {vec_ty}"
        )
        return out_lines

    def _emit_elementwise_for_index(idx_ssa: str) -> list[str]:
        loaded: dict[str, str] = {}
        loaded_ty: dict[str, str] = {}
        computed: dict[str, str] = {}
        computed_ty: dict[str, str] = {}

        row_ssa = ""
        col_ssa = ""
        out_lines: list[str] = []
        if need_rowcol:
            assert out_n is not None
            if need_row:
                row_ssa = _fresh("row")
                out_lines.append(f"        {row_ssa} = arith.divui {idx_ssa}, %cN : index")
            if need_col:
                col_ssa = _fresh("col")
                out_lines.append(f"        {col_ssa} = arith.remui {idx_ssa}, %cN : index")

        def _infer_value_ty(name: str) -> str:
            if name in computed_ty:
                return str(computed_ty[name])
            if name in loaded_ty:
                return str(loaded_ty[name])
            tt = (intent.tensors or {}).get(name)
            if tt is None:
                return "f32"
            return _dtype_to_mlir(str(getattr(tt, "dtype", "f32")))

        def _load_tensor(name: str) -> list[str]:
            if name in computed:
                return []
            if name in loaded:
                return []
            spec = arg_specs.get(name)
            if spec is None:
                raise RuntimeError(f"unknown tensor referenced: {name}")
            memref = str(spec["memref"])
            memref_elem_ty = str(spec.get("memref_elem_ty") or "")
            scalar_ty = str(spec.get("scalar_ty") or "")
            ssa_loaded = _fresh(f"{name}_loaded")
            ssa_value = _fresh(f"{name}_v")
            bcast = str(spec.get("broadcast") or "")
            idx = str(idx_ssa)
            if bcast == "scalar":
                idx = "%c0"
            elif bcast == "elementwise":
                idx = str(idx_ssa)
            elif bcast == "broadcast_n":
                if not col_ssa:
                    raise RuntimeError("internal error: broadcast_n requires col SSA")
                idx = str(col_ssa)
            elif bcast == "broadcast_m":
                if not row_ssa:
                    raise RuntimeError("internal error: broadcast_m requires row SSA")
                idx = str(row_ssa)
            else:
                raise RuntimeError(f"unsupported broadcast kind in cuda real-mlir wave: {bcast}")

            lines = [f"        {ssa_loaded} = memref.load {arg_ssa[str(name)]}[{idx}] : {memref}"]
            if scalar_ty and memref_elem_ty and scalar_ty != memref_elem_ty:
                # ABI-facing bool tensors use i8 elements; convert to i1 in registers.
                if scalar_ty == "i1" and memref_elem_ty == "i8":
                    c0i8 = _fresh("c0i8")
                    lines.append(f"        {c0i8} = arith.constant 0 : i8")
                    lines.append(f"        {ssa_value} = arith.cmpi ne, {ssa_loaded}, {c0i8} : i8")
                else:
                    raise RuntimeError(
                        "unsupported memref->scalar conversion in cuda real-mlir wave: "
                        f"{memref_elem_ty} -> {scalar_ty} for tensor={name}"
                    )
            else:
                ssa_value = ssa_loaded

            loaded[name] = ssa_value
            loaded_ty[name] = scalar_ty or memref_elem_ty or "f32"
            return lines

        # Emit ops (single-thread elementwise evaluation).
        for op in list(intent.ops or []):
            op_name = str(getattr(op, "op", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            outv = str(getattr(op, "output", "")).strip()
            attrs = dict(getattr(op, "attrs", {}) or {})
            if not op_name or not outv:
                raise RuntimeError("invalid op in intent")

            # Materialize inputs.
            in_ssa: list[str] = []
            in_ty: list[str] = []
            for inp in inputs:
                if inp in computed:
                    in_ssa.append(computed[inp])
                else:
                    out_lines.extend(_load_tensor(inp))
                    in_ssa.append(loaded[inp])
                in_ty.append(_infer_value_ty(inp))

            out_ty = _infer_value_ty(outv)

            if op_name == "const":
                if inputs:
                    raise RuntimeError("const expects 0 inputs")
                if out_ty != "f32":
                    raise RuntimeError(f"const currently supports f32 only (out_ty={out_ty})")
                dst = _fresh("const")
                out_lines.append(f"        {dst} = arith.constant {_as_f32_const(attrs.get('value'))} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "cast":
                if len(in_ssa) != 1:
                    raise RuntimeError("cast expects 1 input")
                to_ty = _dtype_to_mlir(str(attrs.get("to") or out_ty or "f32"))
                src_ty = in_ty[0]
                if src_ty == to_ty:
                    computed[outv] = in_ssa[0]
                    computed_ty[outv] = str(to_ty)
                    continue
                dst = _fresh("cast")
                if src_ty in {"f16", "bf16"} and to_ty == "f32":
                    out_lines.append(f"        {dst} = arith.extf {in_ssa[0]} : {src_ty} to {to_ty}")
                elif src_ty == "f32" and to_ty in {"f16", "bf16"}:
                    out_lines.append(f"        {dst} = arith.truncf {in_ssa[0]} : {src_ty} to {to_ty}")
                elif src_ty == "i1" and to_ty == "i32":
                    # FlagGems comparison ops often cast bool -> i32 for output.
                    out_lines.append(f"        {dst} = arith.extui {in_ssa[0]} : i1 to i32")
                else:
                    raise RuntimeError(f"unsupported cast: {src_ty} -> {to_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(to_ty)
                continue

            if op_name == "identity":
                if len(in_ssa) != 1:
                    raise RuntimeError("identity expects 1 input")
                computed[outv] = in_ssa[0]
                computed_ty[outv] = in_ty[0]
                continue

            if op_name == "abs":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("abs currently supports f32 only")
                dst = _fresh("abs")
                out_lines.append(f"        {dst} = math.absf {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "floor":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("floor currently supports f32 only")
                dst = _fresh("floor")
                out_lines.append(f"        {dst} = math.floor {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "ceil":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("ceil currently supports f32 only")
                dst = _fresh("ceil")
                out_lines.append(f"        {dst} = math.ceil {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "neg":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("neg currently supports f32 only")
                dst = _fresh("neg")
                out_lines.append(f"        {dst} = arith.negf {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name in {"add", "sub", "mul", "div"}:
                if len(in_ssa) != 2 or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError(f"{op_name} currently supports f32 binary only")
                dst = _fresh(op_name)
                arith = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf"}[op_name]
                out_lines.append(f"        {dst} = {arith} {in_ssa[0]}, {in_ssa[1]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name in {"max", "min"}:
                if len(in_ssa) != 2 or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError(f"{op_name} currently supports f32 binary only")
                dst = _fresh(op_name)
                arith = {"max": "arith.maximumf", "min": "arith.minimumf"}[op_name]
                out_lines.append(f"        {dst} = {arith} {in_ssa[0]}, {in_ssa[1]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
                if len(in_ssa) != 2 or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError(f"{op_name} currently supports f32 binary only")
                dst = _fresh(op_name)
                pred = {
                    "eq": "oeq",
                    "ne": "one",
                    "lt": "olt",
                    "le": "ole",
                    "gt": "ogt",
                    "ge": "oge",
                }[op_name]
                out_lines.append(f"        {dst} = arith.cmpf {pred}, {in_ssa[0]}, {in_ssa[1]} : f32")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue

            if op_name == "relu":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("relu currently supports f32 only")
                c0f = _fresh("c0f")
                dst = _fresh("relu")
                out_lines.append(f"        {c0f} = arith.constant 0.0 : f32")
                out_lines.append(f"        {dst} = arith.maximumf {in_ssa[0]}, {c0f}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "exp":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("exp currently supports f32 only")
                dst = _fresh("exp")
                out_lines.append(f"        {dst} = math.exp {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "exp2":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("exp2 currently supports f32 only")
                dst = _fresh("exp2")
                out_lines.append(f"        {dst} = math.exp2 {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "log":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("log currently supports f32 only")
                dst = _fresh("log")
                out_lines.append(f"        {dst} = math.log {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "sqrt":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("sqrt currently supports f32 only")
                dst = _fresh("sqrt")
                out_lines.append(f"        {dst} = math.sqrt {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "rsqrt":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("rsqrt currently supports f32 only")
                dst = _fresh("rsqrt")
                out_lines.append(f"        {dst} = math.rsqrt {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "erf":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("erf currently supports f32 only")
                dst = _fresh("erf")
                out_lines.append(f"        {dst} = math.erf {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name in {"sin", "cos", "tan"}:
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError(f"{op_name} currently supports f32 only")
                dst = _fresh(op_name)
                mop = {"sin": "math.sin", "cos": "math.cos", "tan": "math.tan"}[op_name]
                out_lines.append(f"        {dst} = {mop} {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "where":
                if len(in_ssa) != 3:
                    raise RuntimeError("where expects 3 inputs")
                if in_ty[0] != "i1":
                    raise RuntimeError(f"where condition must be bool/i1, got {in_ty[0]}")
                if in_ty[1] != "f32" or in_ty[2] != "f32":
                    raise RuntimeError("where currently supports f32 branches only")
                dst = _fresh("where")
                out_lines.append(f"        {dst} = arith.select {in_ssa[0]}, {in_ssa[1]}, {in_ssa[2]} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            raise RuntimeError(f"unsupported op in cuda real-mlir wave: {op_name}")

        # Store final output element.
        final_ssa = computed.get(out_name)
        if final_ssa is None:
            last_out = str(getattr(list(intent.ops or [])[-1], "output", "")).strip() if intent.ops else ""
            final_ssa = computed.get(last_out)
        if not final_ssa:
            raise RuntimeError(f"no computed value for output tensor: {out_name}")

        out_val_ty = _infer_value_ty(out_name)
        store_ssa = str(final_ssa)
        if out_memref_elem_ty and out_val_ty and out_memref_elem_ty != out_val_ty:
            if out_memref_elem_ty == "i8" and out_val_ty == "i1":
                tmp = _fresh("out_i8")
                out_lines.append(f"        {tmp} = arith.extui {store_ssa} : i1 to i8")
                store_ssa = tmp
            else:
                raise RuntimeError(
                    "unsupported scalar->memref conversion in cuda real-mlir wave: "
                    f"{out_val_ty} -> {out_memref_elem_ty} for output={out_name}"
                )
        out_lines.append(f"        memref.store {store_ssa}, {arg_ssa[str(out_name)]}[{idx_ssa}] : {out_memref}")
        return out_lines

    # Assemble module text.
    lines: list[str] = []
    lines.append("module attributes {")
    lines.append("  gpu.container_module,")
    lines.append('  intentir.format = "std_mlir_v1",')
    lines.append(f'  intentir.intent_name = "{kernel_name}",')
    lines.append(f'  intentir.intent_json_b64 = "{b64}",')
    lines.append(f"  intentir.symbols = {symbols_attr},")
    lines.append('  llvm.target_triple = "nvptx64-nvidia-cuda"')
    lines.append("} {")
    lines.append("  gpu.module @kernels {")
    lines.append(f"    gpu.func @{kernel_name}({arg_sig}) kernel {{")
    lines.append("      %tid = gpu.thread_id x")
    lines.append("      %bid = gpu.block_id x")
    lines.append("      %bdim = gpu.block_dim x")
    lines.append("      %tmp = arith.muli %bid, %bdim : index")
    lines.append("      %lin = arith.addi %tmp, %tid : index")
    lines.append("      %c0 = arith.constant 0 : index")
    if need_rowcol:
        assert out_n is not None
        lines.append(f"      %cN = arith.constant {int(out_n)} : index")
    lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
    if elems_per_thread == 1:
        idx_ssa = "%lin"
        lines.append(f"      %pred = arith.cmpi ult, {idx_ssa}, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.extend(_emit_elementwise_for_index(idx_ssa))
        lines.append("      }")
    else:
        lines.append(f"      %c_elems = arith.constant {int(elems_per_thread)} : index")
        lines.append("      %base = arith.muli %lin, %c_elems : index")
        if vectorize:
            # Assume base pointer alignment to unlock aligned vector memory ops
            # downstream (load/store <N x f32>). Since %base is a multiple of
            # elems_per_thread, this is safe for contiguous tensors.
            align_bytes = int(elems_per_thread) * 4
            for name, spec in arg_specs.items():
                aligned = f"%{_mlir_ident(name)}_aligned"
                original = f"%{_mlir_ident(name)}"
                memref_ty = str(spec["memref"])
                lines.append(f"      {aligned} = memref.assume_alignment {original}, {align_bytes} : {memref_ty}")
                arg_ssa[str(name)] = str(aligned)
            lines.append("      %pred_vec = arith.cmpi ult, %base, %c_total : index")
            lines.append("      scf.if %pred_vec {")
            lines.extend(_emit_elementwise_vector_for_base("%base"))
            lines.append("      }")
        else:
            for lane in range(int(elems_per_thread)):
                lane_c = f"%c_lane_{lane}"
                lane_i = f"%i_{lane}"
                lane_p = f"%pred_{lane}"
                lines.append(f"      {lane_c} = arith.constant {int(lane)} : index")
                lines.append(f"      {lane_i} = arith.addi %base, {lane_c} : index")
                lines.append(f"      {lane_p} = arith.cmpi ult, {lane_i}, %c_total : index")
                lines.append(f"      scf.if {lane_p} {{")
                lines.extend(_emit_elementwise_for_index(lane_i))
                lines.append("      }")
    lines.append("      gpu.return")
    lines.append("    }")
    lines.append("  }")
    lines.append("  // intentir_json_begin")
    lines.append(f"  // {b64}")
    lines.append("  // intentir_json_end")
    lines.append("}")

    out = IntentMLIRModule(
        module_text="\n".join(lines) + "\n",
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=dict(intent_json),
    )
    out.meta["cuda_real_mlir_kernel_emitted"] = True
    out.meta["kernel_name"] = str(kernel_name)
    out.meta["cuda_real_mlir_output_total"] = int(out_total)
    out.meta["cuda_real_mlir_elems_per_thread"] = int(elems_per_thread)
    return out


__all__ = ["lower_intent_to_cuda_gpu_kernel"]
