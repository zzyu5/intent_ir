from __future__ import annotations

import base64
import json
import math
import os
import re
from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from intent_ir.ir.repair import materialize_missing_op_output_tensors
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
    repair_actions = materialize_missing_op_output_tensors(intent)

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

        # Broadcast classification is relative to the *final* output shape. For
        # non-elementwise ops (pad/concat/gather) an input tensor may have a
        # different numel; keep it as "none" and let op-specific lowering compute
        # explicit indices.
        broadcast = "scalar" if is_scalar else "none"
        if not is_scalar:
            if int(numel) == int(out_total):
                broadcast = "elementwise"
            elif out_rank == 2 and len(dims) == 1 and out_m is not None and out_n is not None:
                if int(numel) == int(out_n):
                    broadcast = "broadcast_n"
                elif int(numel) == int(out_m):
                    broadcast = "broadcast_m"
                else:
                    broadcast = "none"
            else:
                broadcast = "none"

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

    # Some ops need 2D row/col addressing on the *output* index.
    need_out_rowcol = False
    if out_rank == 2:
        need_out_rowcol = any(
            str(spec.get("broadcast") or "") in {"broadcast_m", "broadcast_n"} for spec in arg_specs.values()
        )
        if not need_out_rowcol:
            need_out_rowcol = any(
                str(getattr(op, "op", "")).strip() in {"concat", "pad"} for op in list(intent.ops or [])
            )
    if need_out_rowcol and out_rank != 2:
        raise RuntimeError("internal error: broadcast patterns require rank-2 output")

    tmp_idx = 0

    def _fresh(prefix: str) -> str:
        nonlocal tmp_idx
        tmp_idx += 1
        return f"%{_mlir_ident(prefix)}_{tmp_idx}"

    def _eligible_for_vectorization_with(elems: int) -> bool:
        if elems <= 1:
            return False
        if int(out_total) <= 0:
            return False
        if int(out_total) % int(elems) != 0:
            return False
        # First cut: only contiguous elementwise + scalar broadcasts, no row/col addressing.
        if need_out_rowcol:
            return False
        for spec in arg_specs.values():
            if str(spec.get("broadcast") or "") not in {"scalar", "elementwise"}:
                return False
            if str(spec.get("scalar_ty") or "") != "f32":
                return False
            if str(spec.get("memref_elem_ty") or "") != "f32":
                return False
        allowed_vec_ops = {
            "abs",
            "add",
            "ceil",
            "const",
            "cos",
            "div",
            "erf",
            "exp",
            "exp2",
            "floor",
            "identity",
            "log",
            "max",
            "min",
            "mul",
            "neg",
            "relu",
            "rsqrt",
            "sin",
            "sqrt",
            "sub",
            "tan",
        }
        # Vector path is intentionally narrow; bail out on anything we don't explicitly lower.
        for op in list(intent.ops or []):
            op_name = str(getattr(op, "op", "")).strip()
            if op_name not in allowed_vec_ops:
                return False
        return True

    elems_per_thread = 1
    elems_per_thread_source = "default"
    raw_elems = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_ELEMS_PER_THREAD", "") or "").strip()
    if raw_elems:
        try:
            elems_per_thread = int(raw_elems)
        except Exception:
            raise RuntimeError(
                "invalid INTENTIR_CUDA_REAL_MLIR_ELEMS_PER_THREAD; expected int, got "
                f"{raw_elems!r}"
            )
        elems_per_thread_source = "env"
    else:
        # IntentIR evidence-driven default: prefer a wider vector width when the intent
        # reports a contiguous dominant range. This keeps tuning knobs visible in
        # the artifacts while allowing deterministic defaults.
        auto = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_ELEMS_PER_THREAD_AUTO", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if auto:
            meta = intent_json.get("meta") if isinstance(intent_json, dict) else None
            access = meta.get("access_witness") if isinstance(meta, dict) else None
            has_contig = bool(access.get("has_contiguous_range")) if isinstance(access, dict) else False
            try:
                dom_len = int(access.get("dominant_range_len") or 0) if isinstance(access, dict) else 0
            except Exception:
                dom_len = 0
            pref = 1
            if has_contig and dom_len >= 256:
                pref = 4
            elif has_contig and dom_len >= 128:
                pref = 2
            if pref in {2, 4} and _eligible_for_vectorization_with(pref):
                elems_per_thread = int(pref)
                elems_per_thread_source = f"access_witness:{dom_len}"

    if elems_per_thread not in {1, 2, 4}:
        raise RuntimeError(
            "unsupported cuda real-mlir elems_per_thread; expected one of {1,2,4}, got "
            f"{elems_per_thread}"
        )

    def _eligible_for_vectorization() -> bool:
        return _eligible_for_vectorization_with(int(elems_per_thread))

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

    def _resolve_symbolic_int(v: Any) -> int:
        iv = _resolve_dim_int(v, bindings)
        if iv is None:
            raise RuntimeError(f"unable to resolve int const: {v!r}")
        return int(iv)

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

    def _emit_elementwise_for_index(
        idx_ssa: str,
        *,
        precomputed: dict[str, tuple[str, str]] | None = None,
        skip_rank2_outputs: bool = False,
    ) -> list[str]:
        loaded: dict[str, str] = {}
        loaded_ty: dict[str, str] = {}
        computed: dict[str, str] = {}
        computed_ty: dict[str, str] = {}

        if precomputed:
            for name, pair in dict(precomputed).items():
                if not (isinstance(pair, tuple) and len(pair) == 2):
                    continue
                ssa, ty = str(pair[0]), str(pair[1])
                if ssa and ty:
                    computed[str(name)] = ssa
                    computed_ty[str(name)] = ty

        row_ssa = ""
        col_ssa = ""
        out_lines: list[str] = []
        if need_out_rowcol:
            assert out_n is not None
            row_ssa = _fresh("row")
            col_ssa = _fresh("col")
            out_lines.append(f"        {row_ssa} = arith.divui {idx_ssa}, %cN : index")
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
            elif bcast == "none":
                raise RuntimeError(
                    "cuda real-mlir lowering: tensor cannot be indexed by output element; "
                    f"tensor={name} shape={spec.get('dims')} out_dims={out_dims} op_requires_custom_indexing"
                )
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

        def _load_at(name: str, idx: str) -> tuple[str, str]:
            spec = arg_specs.get(name)
            if spec is None:
                raise RuntimeError(f"unknown tensor referenced: {name}")
            memref = str(spec["memref"])
            memref_elem_ty = str(spec.get("memref_elem_ty") or "")
            scalar_ty = str(spec.get("scalar_ty") or "")
            ssa_loaded = _fresh(f"{name}_loaded")
            ssa_value = _fresh(f"{name}_v")
            out_lines.append(f"        {ssa_loaded} = memref.load {arg_ssa[str(name)]}[{idx}] : {memref}")
            if scalar_ty and memref_elem_ty and scalar_ty != memref_elem_ty:
                # ABI-facing bool tensors use i8 elements; convert to i1 in registers.
                if scalar_ty == "i1" and memref_elem_ty == "i8":
                    c0i8 = _fresh("c0i8")
                    out_lines.append(f"        {c0i8} = arith.constant 0 : i8")
                    out_lines.append(f"        {ssa_value} = arith.cmpi ne, {ssa_loaded}, {c0i8} : i8")
                    return ssa_value, "i1"
                raise RuntimeError(
                    "unsupported memref->scalar conversion in cuda real-mlir wave: "
                    f"{memref_elem_ty} -> {scalar_ty} for tensor={name}"
                )
            return ssa_loaded, (scalar_ty or memref_elem_ty or "f32")

        def _as_index(ssa: str, ty: str) -> str:
            if ty == "index":
                return ssa
            if ty not in {"i32", "i64"}:
                raise RuntimeError(f"expected integer index tensor, got {ty}")
            out = _fresh("idx")
            out_lines.append(f"        {out} = arith.index_cast {ssa} : {ty} to index")
            return out

        def _coerce_scalar(ssa: str, src_ty: str, dst_ty: str) -> str:
            if src_ty == dst_ty:
                return ssa
            out = _fresh("cast")
            # float widen/narrow
            if src_ty in {"f16", "bf16"} and dst_ty == "f32":
                out_lines.append(f"        {out} = arith.extf {ssa} : {src_ty} to f32")
                return out
            if src_ty == "f32" and dst_ty in {"f16", "bf16"}:
                out_lines.append(f"        {out} = arith.truncf {ssa} : f32 to {dst_ty}")
                return out
            # bool <-> int
            if src_ty == "i1" and dst_ty in {"i32", "i64"}:
                out_lines.append(f"        {out} = arith.extui {ssa} : i1 to {dst_ty}")
                return out
            if src_ty in {"i32", "i64"} and dst_ty == "i1":
                c0 = _fresh("c0")
                out_lines.append(f"        {c0} = arith.constant 0 : {src_ty}")
                out_lines.append(f"        {out} = arith.cmpi ne, {ssa}, {c0} : {src_ty}")
                return out
            # int widen/narrow
            if src_ty == "i32" and dst_ty == "i64":
                out_lines.append(f"        {out} = arith.extsi {ssa} : i32 to i64")
                return out
            if src_ty == "i64" and dst_ty == "i32":
                out_lines.append(f"        {out} = arith.trunci {ssa} : i64 to i32")
                return out
            # int <-> float (signed)
            if src_ty in {"i32", "i64"} and dst_ty == "f32":
                out_lines.append(f"        {out} = arith.sitofp {ssa} : {src_ty} to f32")
                return out
            if src_ty == "f32" and dst_ty in {"i32", "i64"}:
                out_lines.append(f"        {out} = arith.fptosi {ssa} : f32 to {dst_ty}")
                return out
            if src_ty == "i1" and dst_ty == "f32":
                tmp_i32 = _fresh("b_i32")
                out_lines.append(f"        {tmp_i32} = arith.extui {ssa} : i1 to i32")
                out_lines.append(f"        {out} = arith.uitofp {tmp_i32} : i32 to f32")
                return out
            if src_ty == "f32" and dst_ty == "i1":
                c0 = _fresh("c0f")
                out_lines.append(f"        {c0} = arith.constant 0.0 : f32")
                out_lines.append(f"        {out} = arith.cmpf one, {ssa}, {c0} : f32")
                return out
            raise RuntimeError(f"unsupported cast: {src_ty} -> {dst_ty}")

        def _value(name: str) -> tuple[str, str]:
            if name in computed:
                return computed[name], str(computed_ty[name])
            out_lines.extend(_load_tensor(name))
            return loaded[name], str(loaded_ty[name])

        # Emit ops (single-thread elementwise evaluation).
        for op in list(intent.ops or []):
            op_name = str(getattr(op, "op", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            outv = str(getattr(op, "output", "")).strip()
            attrs = dict(getattr(op, "attrs", {}) or {})
            if not op_name or not outv:
                raise RuntimeError("invalid op in intent")

            if skip_rank2_outputs:
                tt_out = (intent.tensors or {}).get(outv)
                out_shape_raw = list(getattr(tt_out, "shape", []) or []) if tt_out is not None else []
                if len(out_shape_raw) > 1:
                    continue

            if op_name in {"reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_any", "reduce_all"}:
                if outv in computed:
                    continue
                raise RuntimeError(f"unsupported reduction op in cuda real-mlir wave: {op_name}")

            # Non-elementwise ops: handle explicit indexing first so we don't try to
            # load broadcast="none" tensors with elementwise rules.
            if op_name == "iota":
                if inputs:
                    raise RuntimeError("iota expects 0 inputs")
                tt = (intent.tensors or {}).get(outv)
                if tt is None:
                    raise RuntimeError(f"missing iota output tensor spec: {outv}")
                dims_raw = list(getattr(tt, "shape", []) or [])
                dims: list[int] = []
                for d in dims_raw:
                    iv = _resolve_dim_int(d, bindings)
                    if iv is None or iv <= 0:
                        raise RuntimeError(f"unbound iota dim: tensor={outv} dim={d}")
                    dims.append(int(iv))
                rank = int(len(dims))
                axis_raw = attrs.get("axis", 0)
                try:
                    axis = int(axis_raw)
                except Exception:
                    axis = 0

                idx_axis = str(idx_ssa)
                if rank == 0:
                    idx_axis = "%c0"
                elif rank == 1:
                    if axis != 0:
                        raise RuntimeError(f"iota rank-1 expects axis=0, got {axis}")
                    idx_axis = str(idx_ssa)
                elif rank == 2:
                    if axis not in {0, 1}:
                        raise RuntimeError(f"iota rank-2 expects axis in {{0,1}}, got {axis}")
                    cN = _fresh("cN_iota")
                    out_lines.append(f"        {cN} = arith.constant {int(dims[1])} : index")
                    if axis == 0:
                        row_i = _fresh("row_iota")
                        out_lines.append(f"        {row_i} = arith.divui {idx_ssa}, {cN} : index")
                        idx_axis = row_i
                    else:
                        col_i = _fresh("col_iota")
                        out_lines.append(f"        {col_i} = arith.remui {idx_ssa}, {cN} : index")
                        idx_axis = col_i
                else:
                    raise RuntimeError(f"iota unsupported rank: {rank}")

                out_ty = _infer_value_ty(outv)
                if out_ty not in {"i32", "i64"}:
                    raise RuntimeError(f"iota unsupported dtype: {out_ty}")
                dst = _fresh("iota")
                out_lines.append(f"        {dst} = arith.index_cast {idx_axis} : index to {out_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(out_ty)
                continue

            if op_name == "concat":
                if len(inputs) != 2:
                    raise RuntimeError("concat currently supports exactly 2 inputs")
                if out_rank != 2 or not need_out_rowcol:
                    raise RuntimeError("concat currently supports rank-2 outputs only")
                axis = int(attrs.get("axis") or 0)
                if axis not in {0, 1}:
                    raise RuntimeError(f"concat unsupported axis: {axis}")
                a, b = str(inputs[0]), str(inputs[1])
                a_dims = list(arg_specs.get(a, {}).get("dims") or [])
                b_dims = list(arg_specs.get(b, {}).get("dims") or [])
                if len(a_dims) != 2 or len(b_dims) != 2:
                    raise RuntimeError("concat expects 2D inputs")
                a_m, a_n = int(a_dims[0]), int(a_dims[1])
                b_m, b_n = int(b_dims[0]), int(b_dims[1])
                if axis == 1 and a_m != b_m:
                    raise RuntimeError("concat axis=1 expects matching M")
                if axis == 0 and a_n != b_n:
                    raise RuntimeError("concat axis=0 expects matching N")
                out_ty = _infer_value_ty(outv)

                if axis == 1:
                    cA = _fresh("cA")
                    pred = _fresh("pred_concat")
                    dst = _fresh("concat")
                    out_lines.append(f"        {cA} = arith.constant {int(a_n)} : index")
                    out_lines.append(f"        {pred} = arith.cmpi ult, {col_ssa}, {cA} : index")
                    out_lines.append(f"        {dst} = scf.if {pred} -> ({out_ty}) {{")
                    mul_a = _fresh("mul_a")
                    idx_a = _fresh("idx_a")
                    out_lines.append(f"          {mul_a} = arith.muli {row_ssa}, {cA} : index")
                    out_lines.append(f"          {idx_a} = arith.addi {mul_a}, {col_ssa} : index")
                    val_a, ty_a = _load_at(a, idx_a)
                    if ty_a != out_ty:
                        raise RuntimeError("concat dtype mismatch on A")
                    out_lines.append(f"          scf.yield {val_a} : {out_ty}")
                    out_lines.append("        } else {")
                    cB = _fresh("cB")
                    col_b = _fresh("col_b")
                    mul_b = _fresh("mul_b")
                    idx_b = _fresh("idx_b")
                    out_lines.append(f"          {cB} = arith.constant {int(b_n)} : index")
                    out_lines.append(f"          {col_b} = arith.subi {col_ssa}, {cA} : index")
                    out_lines.append(f"          {mul_b} = arith.muli {row_ssa}, {cB} : index")
                    out_lines.append(f"          {idx_b} = arith.addi {mul_b}, {col_b} : index")
                    val_b, ty_b = _load_at(b, idx_b)
                    if ty_b != out_ty:
                        raise RuntimeError("concat dtype mismatch on B")
                    out_lines.append(f"          scf.yield {val_b} : {out_ty}")
                    out_lines.append("        }")
                    computed[outv] = dst
                    computed_ty[outv] = out_ty
                    continue

                # axis == 0
                cA = _fresh("cA")
                pred = _fresh("pred_concat")
                dst = _fresh("concat")
                out_lines.append(f"        {cA} = arith.constant {int(a_m)} : index")
                out_lines.append(f"        {pred} = arith.cmpi ult, {row_ssa}, {cA} : index")
                out_lines.append(f"        {dst} = scf.if {pred} -> ({out_ty}) {{")
                cN = _fresh("cN_in")
                mul_a = _fresh("mul_a")
                idx_a = _fresh("idx_a")
                out_lines.append(f"          {cN} = arith.constant {int(a_n)} : index")
                out_lines.append(f"          {mul_a} = arith.muli {row_ssa}, {cN} : index")
                out_lines.append(f"          {idx_a} = arith.addi {mul_a}, {col_ssa} : index")
                val_a, ty_a = _load_at(a, idx_a)
                if ty_a != out_ty:
                    raise RuntimeError("concat dtype mismatch on A")
                out_lines.append(f"          scf.yield {val_a} : {out_ty}")
                out_lines.append("        } else {")
                row_b = _fresh("row_b")
                cN = _fresh("cN_in")
                mul_b = _fresh("mul_b")
                idx_b = _fresh("idx_b")
                out_lines.append(f"          {row_b} = arith.subi {row_ssa}, {cA} : index")
                out_lines.append(f"          {cN} = arith.constant {int(b_n)} : index")
                out_lines.append(f"          {mul_b} = arith.muli {row_b}, {cN} : index")
                out_lines.append(f"          {idx_b} = arith.addi {mul_b}, {col_ssa} : index")
                val_b, ty_b = _load_at(b, idx_b)
                if ty_b != out_ty:
                    raise RuntimeError("concat dtype mismatch on B")
                out_lines.append(f"          scf.yield {val_b} : {out_ty}")
                out_lines.append("        }")
                computed[outv] = dst
                computed_ty[outv] = out_ty
                continue

            if op_name == "pad":
                if len(inputs) != 1:
                    raise RuntimeError("pad expects 1 input")
                if out_rank != 2 or not need_out_rowcol:
                    raise RuntimeError("pad currently supports rank-2 outputs only")
                mode = str(attrs.get("mode") or "").strip().lower()
                if mode and mode != "constant":
                    raise RuntimeError(f"pad unsupported mode: {mode}")
                pad_width = dict(attrs.get("pad_width") or {})
                pairs = list(pad_width.get("pairs") or [])
                if len(pairs) != 2:
                    raise RuntimeError("pad expects 2D pad_width pairs")
                try:
                    pad0_before, _pad0_after = int(pairs[0][0]), int(pairs[0][1])
                    pad1_before, _pad1_after = int(pairs[1][0]), int(pairs[1][1])
                except Exception as e:
                    raise RuntimeError(f"pad invalid pad_width pairs: {pairs}") from e
                inp = str(inputs[0])
                inp_dims = list(arg_specs.get(inp, {}).get("dims") or [])
                if len(inp_dims) != 2:
                    raise RuntimeError("pad expects 2D input")
                in_m, in_n = int(inp_dims[0]), int(inp_dims[1])
                out_ty = _infer_value_ty(outv)
                if out_ty != "f32":
                    raise RuntimeError("pad currently supports f32 only")
                value = _as_f32_const(attrs.get("value", 0.0))

                c_row_lo = _fresh("c_row_lo")
                c_row_hi = _fresh("c_row_hi")
                c_col_lo = _fresh("c_col_lo")
                c_col_hi = _fresh("c_col_hi")
                out_lines.append(f"        {c_row_lo} = arith.constant {int(pad0_before)} : index")
                out_lines.append(f"        {c_row_hi} = arith.constant {int(pad0_before + in_m)} : index")
                out_lines.append(f"        {c_col_lo} = arith.constant {int(pad1_before)} : index")
                out_lines.append(f"        {c_col_hi} = arith.constant {int(pad1_before + in_n)} : index")

                row_ge = _fresh("row_ge")
                row_lt = _fresh("row_lt")
                col_ge = _fresh("col_ge")
                col_lt = _fresh("col_lt")
                out_lines.append(f"        {row_ge} = arith.cmpi uge, {row_ssa}, {c_row_lo} : index")
                out_lines.append(f"        {row_lt} = arith.cmpi ult, {row_ssa}, {c_row_hi} : index")
                out_lines.append(f"        {col_ge} = arith.cmpi uge, {col_ssa}, {c_col_lo} : index")
                out_lines.append(f"        {col_lt} = arith.cmpi ult, {col_ssa}, {c_col_hi} : index")
                row_ok = _fresh("row_ok")
                col_ok = _fresh("col_ok")
                pred = _fresh("pred_pad")
                out_lines.append(f"        {row_ok} = arith.andi {row_ge}, {row_lt} : i1")
                out_lines.append(f"        {col_ok} = arith.andi {col_ge}, {col_lt} : i1")
                out_lines.append(f"        {pred} = arith.andi {row_ok}, {col_ok} : i1")

                dst = _fresh("pad")
                out_lines.append(f"        {dst} = scf.if {pred} -> (f32) {{")
                in_row = _fresh("in_row")
                in_col = _fresh("in_col")
                out_lines.append(f"          {in_row} = arith.subi {row_ssa}, {c_row_lo} : index")
                out_lines.append(f"          {in_col} = arith.subi {col_ssa}, {c_col_lo} : index")
                cN = _fresh("cN_in")
                mul_in = _fresh("mul_in")
                idx_in = _fresh("idx_in")
                out_lines.append(f"          {cN} = arith.constant {int(in_n)} : index")
                out_lines.append(f"          {mul_in} = arith.muli {in_row}, {cN} : index")
                out_lines.append(f"          {idx_in} = arith.addi {mul_in}, {in_col} : index")
                val_in, ty_in = _load_at(inp, idx_in)
                if ty_in != "f32":
                    raise RuntimeError("pad input must be f32")
                out_lines.append(f"          scf.yield {val_in} : f32")
                out_lines.append("        } else {")
                cval = _fresh("cval")
                out_lines.append(f"          {cval} = arith.constant {value} : f32")
                out_lines.append(f"          scf.yield {cval} : f32")
                out_lines.append("        }")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "gather":
                if len(inputs) != 3:
                    raise RuntimeError("gather expects 3 inputs (inp,row_idx,col_idx)")
                inp, row_idx, col_idx = [str(x) for x in inputs]
                inp_dims = list(arg_specs.get(inp, {}).get("dims") or [])
                if len(inp_dims) != 2:
                    raise RuntimeError("gather currently supports 2D input tensors only")
                inp_n = int(inp_dims[1])
                row_v, row_ty = _value(row_idx)
                col_v, col_ty = _value(col_idx)
                row_i = _as_index(row_v, row_ty)
                col_i = _as_index(col_v, col_ty)
                cN = _fresh("cN_in")
                mul_in = _fresh("mul_in")
                idx_in = _fresh("idx_in")
                out_lines.append(f"        {cN} = arith.constant {int(inp_n)} : index")
                out_lines.append(f"        {mul_in} = arith.muli {row_i}, {cN} : index")
                out_lines.append(f"        {idx_in} = arith.addi {mul_in}, {col_i} : index")
                val, ty = _load_at(inp, idx_in)
                out_ty = _infer_value_ty(outv)
                if ty != out_ty:
                    raise RuntimeError(f"gather dtype mismatch: {ty} vs {out_ty}")
                computed[outv] = val
                computed_ty[outv] = out_ty
                continue

            # Elementwise path: materialize inputs using broadcast rules.
            in_ssa: list[str] = []
            in_ty: list[str] = []
            for inp in inputs:
                v, ty = _value(inp)
                in_ssa.append(v)
                in_ty.append(ty)

            out_ty = _infer_value_ty(outv)

            if op_name == "const":
                if inputs:
                    raise RuntimeError("const expects 0 inputs")
                dst = _fresh("const")
                if out_ty in {"i32", "i64"}:
                    iv = _resolve_symbolic_int(attrs.get("value"))
                    out_lines.append(f"        {dst} = arith.constant {int(iv)} : {out_ty}")
                    computed[outv] = dst
                    computed_ty[outv] = str(out_ty)
                    continue
                if out_ty == "f32":
                    raw = attrs.get("value")
                    # Some seeds encode symbol refs as strings (e.g. "N").
                    sym_iv = _resolve_dim_int(raw, bindings)
                    if sym_iv is not None:
                        raw = float(sym_iv)
                    out_lines.append(f"        {dst} = arith.constant {_as_f32_const(raw)} : f32")
                    computed[outv] = dst
                    computed_ty[outv] = "f32"
                    continue
                raise RuntimeError(f"const unsupported output dtype: {out_ty}")

            if op_name == "cast":
                if len(in_ssa) != 1:
                    raise RuntimeError("cast expects 1 input")
                to_ty = _dtype_to_mlir(str(attrs.get("to") or out_ty or "f32"))
                src_ty = in_ty[0]
                if src_ty == to_ty:
                    computed[outv] = in_ssa[0]
                    computed_ty[outv] = str(to_ty)
                    continue
                dst = _coerce_scalar(in_ssa[0], src_ty, str(to_ty))
                computed[outv] = dst
                computed_ty[outv] = str(to_ty)
                continue

            if op_name == "identity":
                if len(in_ssa) != 1:
                    raise RuntimeError("identity expects 1 input")
                computed[outv] = in_ssa[0]
                computed_ty[outv] = in_ty[0]
                continue

            if op_name in {"broadcast_in_dim", "reshape"}:
                if len(in_ssa) != 1:
                    raise RuntimeError(f"{op_name} expects 1 input")
                # Shape-only ops in the current CUDA real-MLIR wave. Require elementwise
                # indexing and forward the scalar value for this output element.
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
                if len(in_ssa) != 2:
                    raise RuntimeError(f"{op_name} expects 2 inputs")
                if out_ty == "f32":
                    lhs = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                    rhs = _coerce_scalar(in_ssa[1], in_ty[1], "f32")
                    dst = _fresh(op_name)
                    arith = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf"}[
                        op_name
                    ]
                    out_lines.append(f"        {dst} = {arith} {lhs}, {rhs}{fm} : f32")
                    computed[outv] = dst
                    computed_ty[outv] = "f32"
                    continue
                if out_ty in {"i32", "i64"}:
                    lhs = _coerce_scalar(in_ssa[0], in_ty[0], str(out_ty))
                    rhs = _coerce_scalar(in_ssa[1], in_ty[1], str(out_ty))
                    dst = _fresh(op_name)
                    arith = {"add": "arith.addi", "sub": "arith.subi", "mul": "arith.muli", "div": "arith.divsi"}[
                        op_name
                    ]
                    out_lines.append(f"        {dst} = {arith} {lhs}, {rhs} : {out_ty}")
                    computed[outv] = dst
                    computed_ty[outv] = str(out_ty)
                    continue
                raise RuntimeError(f"{op_name} unsupported output dtype: {out_ty}")

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
                if len(in_ssa) != 2:
                    raise RuntimeError(f"{op_name} expects 2 inputs")
                dst = _fresh(op_name)
                float_tys = {"f16", "bf16", "f32"}
                int_tys = {"i1", "i32", "i64"}
                if in_ty[0] in float_tys or in_ty[1] in float_tys:
                    lhs = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                    rhs = _coerce_scalar(in_ssa[1], in_ty[1], "f32")
                    pred = {
                        "eq": "oeq",
                        "ne": "one",
                        "lt": "olt",
                        "le": "ole",
                        "gt": "ogt",
                        "ge": "oge",
                    }[op_name]
                    out_lines.append(f"        {dst} = arith.cmpf {pred}, {lhs}, {rhs} : f32")
                elif in_ty[0] in int_tys and in_ty[1] in int_tys:
                    cmp_ty = "i64" if ("i64" in {in_ty[0], in_ty[1]}) else ("i32" if ("i32" in {in_ty[0], in_ty[1]}) else "i1")
                    lhs = _coerce_scalar(in_ssa[0], in_ty[0], cmp_ty)
                    rhs = _coerce_scalar(in_ssa[1], in_ty[1], cmp_ty)
                    pred = {
                        "eq": "eq",
                        "ne": "ne",
                        "lt": "slt",
                        "le": "sle",
                        "gt": "sgt",
                        "ge": "sge",
                    }[op_name]
                    out_lines.append(f"        {dst} = arith.cmpi {pred}, {lhs}, {rhs} : {cmp_ty}")
                else:
                    raise RuntimeError(f"{op_name} unsupported input dtypes: {in_ty[0]} vs {in_ty[1]}")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue

            if op_name == "and":
                if len(in_ssa) != 2 or in_ty[0] != "i1" or in_ty[1] != "i1":
                    raise RuntimeError("and currently supports i1/i1 only")
                dst = _fresh("and")
                out_lines.append(f"        {dst} = arith.andi {in_ssa[0]}, {in_ssa[1]} : i1")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue

            if op_name == "not":
                if len(in_ssa) != 1 or in_ty[0] != "i1":
                    raise RuntimeError("not currently supports i1 only")
                c1 = _fresh("c1")
                dst = _fresh("not")
                out_lines.append(f"        {c1} = arith.constant 1 : i1")
                out_lines.append(f"        {dst} = arith.xori {in_ssa[0]}, {c1} : i1")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue

            if op_name == "bitwise_and":
                if len(in_ssa) != 2 or in_ty[0] != "i32" or in_ty[1] != "i32":
                    raise RuntimeError("bitwise_and currently supports i32/i32 only")
                dst = _fresh("band")
                out_lines.append(f"        {dst} = arith.andi {in_ssa[0]}, {in_ssa[1]} : i32")
                computed[outv] = dst
                computed_ty[outv] = "i32"
                continue

            if op_name == "bitwise_or":
                if len(in_ssa) != 2 or in_ty[0] != "i32" or in_ty[1] != "i32":
                    raise RuntimeError("bitwise_or currently supports i32/i32 only")
                dst = _fresh("bor")
                out_lines.append(f"        {dst} = arith.ori {in_ssa[0]}, {in_ssa[1]} : i32")
                computed[outv] = dst
                computed_ty[outv] = "i32"
                continue

            if op_name == "bitwise_left_shift":
                if len(in_ssa) != 2 or in_ty[0] != "i32" or in_ty[1] != "i32":
                    raise RuntimeError("bitwise_left_shift currently supports i32/i32 only")
                dst = _fresh("bshl")
                out_lines.append(f"        {dst} = arith.shli {in_ssa[0]}, {in_ssa[1]} : i32")
                computed[outv] = dst
                computed_ty[outv] = "i32"
                continue

            if op_name == "bitwise_right_shift":
                if len(in_ssa) != 2 or in_ty[0] != "i32" or in_ty[1] != "i32":
                    raise RuntimeError("bitwise_right_shift currently supports i32/i32 only")
                dst = _fresh("bshr")
                # Torch bitwise_right_shift follows arithmetic right shift for signed ints.
                out_lines.append(f"        {dst} = arith.shrsi {in_ssa[0]}, {in_ssa[1]} : i32")
                computed[outv] = dst
                computed_ty[outv] = "i32"
                continue

            if op_name == "bitwise_not":
                if len(in_ssa) != 1 or in_ty[0] != "i32":
                    raise RuntimeError("bitwise_not currently supports i32 only")
                c_all = _fresh("call")
                dst = _fresh("bnot")
                out_lines.append(f"        {c_all} = arith.constant -1 : i32")
                out_lines.append(f"        {dst} = arith.xori {in_ssa[0]}, {c_all} : i32")
                computed[outv] = dst
                computed_ty[outv] = "i32"
                continue

            if op_name == "remainder":
                if len(in_ssa) != 2 or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError("remainder currently supports f32/f32 only")
                dst = _fresh("rem")
                out_lines.append(f"        {dst} = arith.remf {in_ssa[0]}, {in_ssa[1]} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "pow":
                if len(in_ssa) != 2 or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError("pow currently supports f32/f32 only")
                dst = _fresh("pow")
                out_lines.append(f"        {dst} = math.powf {in_ssa[0]}, {in_ssa[1]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "acos":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("acos currently supports f32 only")
                dst = _fresh("acos")
                out_lines.append(f"        {dst} = math.acos {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "atan":
                if len(in_ssa) != 1 or in_ty[0] != "f32":
                    raise RuntimeError("atan currently supports f32 only")
                dst = _fresh("atan")
                out_lines.append(f"        {dst} = math.atan {in_ssa[0]}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
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

    row_reduce_sum_axis1: dict[str, Any] | None = None
    if out_rank == 1:
        red_ops: list[tuple[int, Any]] = []
        for i, op in enumerate(list(intent.ops or [])):
            if str(getattr(op, "op", "")).strip() == "reduce_sum":
                red_ops.append((int(i), op))
        if len(red_ops) == 1:
            red_op_idx, red_op = red_ops[0]
            red_out = str(getattr(red_op, "output", "")).strip()
            red_inputs = [str(x) for x in list(getattr(red_op, "inputs", []) or []) if str(x).strip()]
            red_attrs = dict(getattr(red_op, "attrs", {}) or {})
            dims = red_attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list == [1] and red_out and len(red_inputs) == 1:
                red_in = str(red_inputs[0])
                red_in_tt = (intent.tensors or {}).get(red_in)
                red_out_tt = (intent.tensors or {}).get(red_out)
                if red_in_tt is not None and red_out_tt is not None:
                    red_in_shape = list(getattr(red_in_tt, "shape", []) or [])
                    red_out_shape = list(getattr(red_out_tt, "shape", []) or [])
                    if len(red_in_shape) == 2 and len(red_out_shape) == 1:
                        m0 = _resolve_dim_int(red_in_shape[0], bindings)
                        n0 = _resolve_dim_int(red_in_shape[1], bindings)
                        if m0 == int(out_total) and n0 is not None and int(n0) > 0:
                            red_out_ty = _dtype_to_mlir(str(getattr(red_out_tt, "dtype", "f32")))
                            if red_out_ty == "f32":
                                row_reduce_sum_axis1 = {
                                    "op_index": int(red_op_idx),
                                    "red_in": str(red_in),
                                    "red_out": str(red_out),
                                    "reduce_n": int(n0),
                                }

    # Assemble module text.
    launch_override: dict[str, Any] | None = None
    kernel_kind = "elementwise_v1"
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
    if row_reduce_sum_axis1 is not None:
        kernel_kind = "row_reduce_sum_axis1_v1"
        # One block per output row; let each block reduce across the N dimension.
        # This keeps global loads coalesced (threads iterate columns).
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}

        red_in = str(row_reduce_sum_axis1["red_in"])
        red_out = str(row_reduce_sum_axis1["red_out"])
        red_op_index = int(row_reduce_sum_axis1["op_index"])
        red_n = int(row_reduce_sum_axis1["reduce_n"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        # Per-thread partial sum.
        lines.append("        %partial = scf.for %j = %tid to %cN_red step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")

        # Pre-reduction elementwise evaluation up to the reduce_sum op (limited subset).
        elem_lines: list[str] = []
        computed: dict[str, str] = {}
        computed_ty: dict[str, str] = {}
        loaded: dict[str, str] = {}
        loaded_ty: dict[str, str] = {}

        def _infer_elem_ty(name: str) -> str:
            if name in computed_ty:
                return str(computed_ty[name])
            if name in loaded_ty:
                return str(loaded_ty[name])
            tt = (intent.tensors or {}).get(name)
            if tt is None:
                return "f32"
            return _dtype_to_mlir(str(getattr(tt, "dtype", "f32")))

        def _load_elem(name: str) -> str:
            if name in computed:
                return computed[name]
            if name in loaded:
                return loaded[name]
            spec = arg_specs.get(name)
            if spec is None:
                raise RuntimeError(f"unknown tensor referenced in row-reduce: {name}")
            dims = list(spec.get("dims") or [])
            idx = "%idx"
            if bool(spec.get("scalar")) or len(dims) == 0:
                idx = "%c0"
            elif len(dims) == 1 and int(dims[0]) == int(out_total):
                idx = "%bid"
            elif len(dims) == 1 and int(dims[0]) == int(red_n):
                idx = "%j"
            elif len(dims) == 2 and int(dims[0]) == int(out_total) and int(dims[1]) == int(red_n):
                idx = "%idx"
            else:
                raise RuntimeError(
                    "row-reduce supports only scalar/[M]/[N]/[M,N] external tensors; "
                    f"tensor={name} dims={dims} M={int(out_total)} N={int(red_n)}"
                )
            memref = str(spec["memref"])
            memref_elem_ty = str(spec.get("memref_elem_ty") or "")
            scalar_ty = str(spec.get("scalar_ty") or "")
            ssa_loaded = _fresh(f"{name}_loaded")
            ssa_value = _fresh(f"{name}_v")
            elem_lines.append(f"          {ssa_loaded} = memref.load {arg_ssa[str(name)]}[{idx}] : {memref}")
            if scalar_ty and memref_elem_ty and scalar_ty != memref_elem_ty:
                if scalar_ty == "i1" and memref_elem_ty == "i8":
                    c0i8 = _fresh("c0i8")
                    elem_lines.append(f"          {c0i8} = arith.constant 0 : i8")
                    elem_lines.append(f"          {ssa_value} = arith.cmpi ne, {ssa_loaded}, {c0i8} : i8")
                    loaded[name] = ssa_value
                    loaded_ty[name] = "i1"
                    return ssa_value
                raise RuntimeError(
                    "unsupported memref->scalar conversion in row-reduce: "
                    f"{memref_elem_ty} -> {scalar_ty} for tensor={name}"
                )
            loaded[name] = ssa_loaded
            loaded_ty[name] = scalar_ty or memref_elem_ty or "f32"
            return ssa_loaded

        def _coerce_elem(ssa: str, src_ty: str, dst_ty: str) -> str:
            if src_ty == dst_ty:
                return ssa
            out = _fresh("cast")
            if src_ty in {"f16", "bf16"} and dst_ty == "f32":
                elem_lines.append(f"          {out} = arith.extf {ssa} : {src_ty} to f32")
                return out
            if src_ty == "f32" and dst_ty in {"f16", "bf16"}:
                elem_lines.append(f"          {out} = arith.truncf {ssa} : f32 to {dst_ty}")
                return out
            if src_ty == "i1" and dst_ty in {"i32", "i64"}:
                elem_lines.append(f"          {out} = arith.extui {ssa} : i1 to {dst_ty}")
                return out
            if src_ty in {"i32", "i64"} and dst_ty == "f32":
                elem_lines.append(f"          {out} = arith.sitofp {ssa} : {src_ty} to f32")
                return out
            if src_ty == "f32" and dst_ty in {"i32", "i64"}:
                elem_lines.append(f"          {out} = arith.fptosi {ssa} : f32 to {dst_ty}")
                return out
            if src_ty == "i32" and dst_ty == "i64":
                elem_lines.append(f"          {out} = arith.extsi {ssa} : i32 to i64")
                return out
            if src_ty == "i64" and dst_ty == "i32":
                elem_lines.append(f"          {out} = arith.trunci {ssa} : i64 to i32")
                return out
            raise RuntimeError(f"unsupported cast in row-reduce: {src_ty} -> {dst_ty}")

        def _value_elem(name: str) -> tuple[str, str]:
            if name in computed:
                return computed[name], str(computed_ty[name])
            v = _load_elem(name)
            return v, _infer_elem_ty(name)

        # Evaluate ops up to reduce_sum to materialize `red_in` at this element.
        for op in list(intent.ops or [])[:red_op_index]:
            op_name = str(getattr(op, "op", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            outv = str(getattr(op, "output", "")).strip()
            attrs = dict(getattr(op, "attrs", {}) or {})
            if not op_name or not outv:
                raise RuntimeError("invalid op in intent (row-reduce pre-eval)")
            if op_name == "const":
                if inputs:
                    raise RuntimeError("const expects 0 inputs")
                out_ty = _infer_elem_ty(outv)
                dst = _fresh("const")
                if out_ty in {"i32", "i64"}:
                    iv = _resolve_symbolic_int(attrs.get("value"))
                    elem_lines.append(f"          {dst} = arith.constant {int(iv)} : {out_ty}")
                elif out_ty == "f32":
                    raw = attrs.get("value")
                    sym_iv = _resolve_dim_int(raw, bindings)
                    if sym_iv is not None:
                        raw = float(sym_iv)
                    elem_lines.append(f"          {dst} = arith.constant {_as_f32_const(raw)} : f32")
                else:
                    raise RuntimeError(f"const unsupported dtype in row-reduce: {out_ty}")
                computed[outv] = dst
                computed_ty[outv] = out_ty
                continue
            if op_name == "cast":
                if len(inputs) != 1:
                    raise RuntimeError("cast expects 1 input")
                src_v, src_ty = _value_elem(inputs[0])
                to_ty = _dtype_to_mlir(str(attrs.get("to") or _infer_elem_ty(outv) or "f32"))
                dst = _coerce_elem(src_v, src_ty, str(to_ty))
                computed[outv] = dst
                computed_ty[outv] = str(to_ty)
                continue
            if op_name in {"identity", "reshape", "broadcast_in_dim"}:
                if len(inputs) != 1:
                    raise RuntimeError(f"{op_name} expects 1 input")
                src_v, src_ty = _value_elem(inputs[0])
                computed[outv] = src_v
                computed_ty[outv] = src_ty
                continue
            if op_name in {"add", "sub", "mul", "div"}:
                if len(inputs) != 2:
                    raise RuntimeError(f"{op_name} expects 2 inputs")
                a_v, a_ty = _value_elem(inputs[0])
                b_v, b_ty = _value_elem(inputs[1])
                out_ty = _infer_elem_ty(outv)
                if out_ty != "f32":
                    raise RuntimeError(f"{op_name} row-reduce pre-eval supports f32 outputs only (out_ty={out_ty})")
                a = _coerce_elem(a_v, a_ty, "f32")
                b = _coerce_elem(b_v, b_ty, "f32")
                dst = _fresh(op_name)
                arith = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf"}[op_name]
                elem_lines.append(f"          {dst} = {arith} {a}, {b}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue
            raise RuntimeError(f"unsupported op in row-reduce pre-eval: {op_name}")

        val_ssa, val_ty = _value_elem(red_in)
        if val_ty != "f32":
            raise RuntimeError(f"row-reduce expects f32 reduction input, got {val_ty} for {red_in}")
        elem_lines.append(f"          %acc_next = arith.addf %acc, {val_ssa}{fm} : f32")
        elem_lines.append("          scf.yield %acc_next : f32")

        lines.extend(elem_lines)
        lines.append("        }")

        # Shared-memory reduce across threads in the block (assume block.x==256).
        lines.append("        %sh = memref.alloca() : memref<256xf32, 3>")
        lines.append("        memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_{stride}"
            pS = f"%pS_{stride}"
            tid2 = f"%tid_{stride}"
            a = f"%a_{stride}"
            b = f"%b_{stride}"
            s = f"%s_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"          {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"          {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %sum0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        body_lines = _emit_elementwise_for_index(
            "%bid",
            precomputed={red_out: ("%sum0", "f32")},
            skip_rank2_outputs=True,
        )
        for l in body_lines:
            lines.append("  " + l)
        lines.append("        }")
        lines.append("      }")
    else:
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        if need_out_rowcol:
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
    out.meta["cuda_real_mlir_elems_per_thread_source"] = str(elems_per_thread_source)
    out.meta["cuda_real_mlir_kernel_kind"] = str(kernel_kind)
    if launch_override:
        out.meta["cuda_real_mlir_launch_override"] = dict(launch_override)
    if repair_actions:
        out.meta["cuda_real_mlir_intent_repair_actions"] = list(repair_actions)
    return out


__all__ = ["lower_intent_to_cuda_gpu_kernel"]
