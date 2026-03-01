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
    if dt == "i16":
        return "i16"
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
    # Macro ops may reference scalar ABI inputs implicitly (not present in op.inputs).
    # Keep this narrow: only enable for known macro ops so non-macro graphs remain stable.
    has_macro = any(str(getattr(op, "op", "")).strip() == "upsample_bicubic2d_aa" for op in ops)
    if has_macro:
        extra_scalars: list[str] = []
        for name, tt in tensors.items():
            nm = str(name).strip()
            if not nm or nm in produced or nm in outputs or nm in external_inputs:
                continue
            shape = list(getattr(tt, "shape", []) or [])
            if len(shape) == 0:
                extra_scalars.append(nm)
        external_inputs.extend(sorted(extra_scalars))
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

    intent_name = str(intent.name or "").strip()
    outputs = [str(x) for x in list(intent.outputs or []) if str(x).strip()]
    if not outputs:
        raise RuntimeError("cuda real-mlir lowering requires at least one output tensor")
    multi_output_ok = intent_name in {
        "group_norm_kernel",
        "layer_norm_persistent",
        "layer_norm_residual2d",
        "ai_bench_layernorm",
        "rms_norm2d",
        "rms_norm_residual2d",
        "min_dim2d",
        "max_pool2d_with_indices_nchw",
        "per_token_group_quant_fp8_2d",
        "batch_norm2d",
    }
    if len(outputs) != 1 and not multi_output_ok:
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
                str(getattr(op, "op", "")).strip() in {"concat", "pad", "transpose"} for op in list(intent.ops or [])
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
        # MLIR float literals must be parseable by the assembly parser. Python's
        # `repr(1e-5)` -> "1e-05" is *not* accepted (it can be mis-tokenized).
        try:
            x = float(v)
        except Exception:
            x = float(str(v))
        if math.isfinite(x):
            if float(x).is_integer() and abs(x) < 1e12:
                return f"{int(x)}.0"
            return format(x, ".9e")
        # Keep inf/nan spellings stable for the parser.
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return "nan"

    def _extract_f32_const(output_name: str) -> float | None:
        """
        Extract a scalar f32 constant from the intent graph.

        Many Triton-native kernels materialize constants (e.g. `eps`) as `const`
        ops rather than ABI inputs; those values are not present in `arg_specs`.
        """
        target = str(output_name or "").strip()
        if not target:
            return None
        for op in list(intent.ops or []):
            if str(getattr(op, "op", "")).strip() != "const":
                continue
            if str(getattr(op, "output", "")).strip() != target:
                continue
            attrs = dict(getattr(op, "attrs", {}) or {})
            dt = str(attrs.get("dtype") or "f32").strip().lower()
            if dt and dt not in {"f32", "float32"}:
                continue
            v = attrs.get("value")
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v))
                except Exception:
                    return None
        return None

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
                if intent_name == "logspace1d":
                    cLOG2E = _fresh("cLOG2E")
                    cLOG2E_v = _fresh("cLOG2E_v")
                    x2 = _fresh("x2")
                    out_lines.append(f"        {cLOG2E} = arith.constant 1.44269504 : f32")
                    out_lines.append(f"        {cLOG2E_v} = vector.splat {cLOG2E} : {vec_ty}")
                    out_lines.append(f"        {x2} = arith.mulf {in_ssa[0]}, {cLOG2E_v}{fm} : {vec_ty}")
                    out_lines.append(f"        {dst} = math.exp2 {x2}{fm} : {vec_ty}")
                else:
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
                # In row-reduce kernels we re-use this scalar emitter with a
                # 1D index (%bid). Skip true rank>1 tensors (e.g. [M,N]) that
                # cannot be indexed by a single row index; allow keepdims-style
                # row outputs like [M,1] where flat index == row.
                if len(out_shape_raw) > 1:
                    if len(out_shape_raw) == 2:
                        d1 = _resolve_dim_int(out_shape_raw[1], bindings)
                        if d1 == 1:
                            pass
                        else:
                            continue
                    else:
                        continue

            if op_name in {"reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_any", "reduce_all"}:
                if outv in computed:
                    continue
                raise RuntimeError(f"unsupported reduction op in cuda real-mlir wave: {op_name}")

            if op_name == "matmul":
                if outv in computed:
                    continue
                raise RuntimeError("unsupported matmul op in cuda real-mlir wave")

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
                else:
                    if axis < 0 or axis >= rank:
                        raise RuntimeError(f"iota expects axis in [0,{rank - 1}], got {axis}")
                    # Flattened row-major index -> axis coordinate:
                    #   coord = (idx / prod(dims[axis+1:])) % dims[axis]
                    stride = 1
                    for d in dims[axis + 1 :]:
                        stride *= int(d)
                    q = str(idx_ssa)
                    if int(stride) != 1:
                        c_stride = _fresh("cStride_iota")
                        q_i = _fresh("q_iota")
                        out_lines.append(f"        {c_stride} = arith.constant {int(stride)} : index")
                        out_lines.append(f"        {q_i} = arith.divui {idx_ssa}, {c_stride} : index")
                        q = q_i
                    c_dim = _fresh("cDim_iota")
                    coord = _fresh("coord_iota")
                    out_lines.append(f"        {c_dim} = arith.constant {int(dims[axis])} : index")
                    out_lines.append(f"        {coord} = arith.remui {q}, {c_dim} : index")
                    idx_axis = coord

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
                if len(inputs) < 3:
                    raise RuntimeError("gather expects at least 3 inputs (inp + indices)")
                inp = str(inputs[0])
                idx_names = [str(x) for x in list(inputs[1:]) if str(x).strip()]
                inp_dims = list(arg_specs.get(inp, {}).get("dims") or [])
                if not inp_dims:
                    raise RuntimeError("gather missing input tensor dims")
                rank = int(len(inp_dims))
                if len(idx_names) != rank:
                    raise RuntimeError(f"gather expects {1 + rank} inputs (inp + {rank} indices); got {len(inputs)}")
                idx_ssas: list[str] = []
                for nm in idx_names:
                    v, ty0 = _value(nm)
                    idx_ssas.append(_as_index(v, ty0))

                # Flatten row-major: (((i0*d1 + i1)*d2 + i2)*... + i{r-1})
                acc = str(idx_ssas[0])
                for k in range(1, rank):
                    cD = _fresh(f"cD{k}_in")
                    mul_in = _fresh("mul_in")
                    add_in = _fresh("idx_in")
                    out_lines.append(f"        {cD} = arith.constant {int(inp_dims[k])} : index")
                    out_lines.append(f"        {mul_in} = arith.muli {acc}, {cD} : index")
                    out_lines.append(f"        {add_in} = arith.addi {mul_in}, {idx_ssas[k]} : index")
                    acc = add_in

                val, ty = _load_at(inp, acc)
                out_ty = _infer_value_ty(outv)
                if ty != out_ty:
                    raise RuntimeError(f"gather dtype mismatch: {ty} vs {out_ty}")
                computed[outv] = val
                computed_ty[outv] = out_ty
                continue

            if op_name == "transpose":
                if len(inputs) != 1:
                    raise RuntimeError("transpose expects 1 input")
                perm = attrs.get("perm")
                perm_list = list(perm) if isinstance(perm, list) else []
                if perm_list != [1, 0]:
                    raise RuntimeError(f"transpose currently supports perm=[1,0] only (got {perm_list})")
                if out_rank != 2 or not need_out_rowcol:
                    raise RuntimeError("transpose currently supports rank-2 outputs only")
                if not row_ssa or not col_ssa:
                    raise RuntimeError("internal error: transpose requires output row/col SSA")

                inp = str(inputs[0])
                inp_dims = list(arg_specs.get(inp, {}).get("dims") or [])
                if len(inp_dims) != 2:
                    raise RuntimeError("transpose currently supports 2D input tensors only")
                inp_n = int(inp_dims[1])

                # Output index is [row, col] in the transposed shape [N, M].
                # Input index should be [col, row] in the original [M, N].
                cN = _fresh("cN_in")
                mul_in = _fresh("mul_in")
                idx_in = _fresh("idx_in")
                out_lines.append(f"        {cN} = arith.constant {int(inp_n)} : index")
                out_lines.append(f"        {mul_in} = arith.muli {col_ssa}, {cN} : index")
                out_lines.append(f"        {idx_in} = arith.addi {mul_in}, {row_ssa} : index")
                val, ty = _load_at(inp, idx_in)
                out_ty = _infer_value_ty(outv)
                if ty != out_ty:
                    raise RuntimeError(f"transpose dtype mismatch: {ty} vs {out_ty}")
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
                if len(in_ssa) != 1:
                    raise RuntimeError("abs expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("abs")
                out_lines.append(f"        {dst} = math.absf {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "floor":
                if len(in_ssa) != 1:
                    raise RuntimeError("floor expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("floor")
                out_lines.append(f"        {dst} = math.floor {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "ceil":
                if len(in_ssa) != 1:
                    raise RuntimeError("ceil expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("ceil")
                out_lines.append(f"        {dst} = math.ceil {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "neg":
                if len(in_ssa) != 1:
                    raise RuntimeError("neg expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("neg")
                out_lines.append(f"        {dst} = arith.negf {x}{fm} : f32")
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
                if len(in_ssa) != 2:
                    raise RuntimeError(f"{op_name} expects 2 inputs")
                if out_ty == "i1" and in_ty[0] == "i1" and in_ty[1] == "i1":
                    dst = _fresh(op_name)
                    arith = {"max": "arith.ori", "min": "arith.andi"}[op_name]
                    out_lines.append(f"        {dst} = {arith} {in_ssa[0]}, {in_ssa[1]} : i1")
                    computed[outv] = dst
                    computed_ty[outv] = "i1"
                    continue
                if out_ty != "f32" or in_ty[0] != "f32" or in_ty[1] != "f32":
                    raise RuntimeError(f"{op_name} currently supports f32 or i1 binary only")
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
                if len(in_ssa) != 1:
                    raise RuntimeError("acos expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("acos")
                out_lines.append(f"        {dst} = math.acos {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "atan":
                if len(in_ssa) != 1:
                    raise RuntimeError("atan expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("atan")
                out_lines.append(f"        {dst} = math.atan {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "relu":
                if len(in_ssa) != 1:
                    raise RuntimeError("relu expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                c0f = _fresh("c0f")
                dst = _fresh("relu")
                out_lines.append(f"        {c0f} = arith.constant 0.0 : f32")
                out_lines.append(f"        {dst} = arith.maximumf {x}, {c0f}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "exp":
                if len(in_ssa) != 1:
                    raise RuntimeError("exp expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("exp")
                if intent_name == "logspace1d":
                    cLOG2E = _fresh("cLOG2E")
                    x2 = _fresh("x2")
                    out_lines.append(f"        {cLOG2E} = arith.constant 1.44269504 : f32")
                    out_lines.append(f"        {x2} = arith.mulf {x}, {cLOG2E}{fm} : f32")
                    out_lines.append(f"        {dst} = math.exp2 {x2}{fm} : f32")
                else:
                    out_lines.append(f"        {dst} = math.exp {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "exp2":
                if len(in_ssa) != 1:
                    raise RuntimeError("exp2 expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("exp2")
                out_lines.append(f"        {dst} = math.exp2 {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "log":
                if len(in_ssa) != 1:
                    raise RuntimeError("log expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("log")
                out_lines.append(f"        {dst} = math.log {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "sqrt":
                if len(in_ssa) != 1:
                    raise RuntimeError("sqrt expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("sqrt")
                out_lines.append(f"        {dst} = math.sqrt {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "rsqrt":
                if len(in_ssa) != 1:
                    raise RuntimeError("rsqrt expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("rsqrt")
                out_lines.append(f"        {dst} = math.rsqrt {x}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue

            if op_name == "erf":
                if len(in_ssa) != 1:
                    raise RuntimeError("erf expects 1 input")
                x = _coerce_scalar(in_ssa[0], in_ty[0], "f32")
                dst = _fresh("erf")
                out_lines.append(f"        {dst} = math.erf {x}{fm} : f32")
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
                cond_ssa = str(in_ssa[0])
                cond_ty = str(in_ty[0])
                if cond_ty != "i1":
                    # Some providers materialize boolean masks as i32 (0/1). Treat any
                    # non-zero value as true.
                    if cond_ty in {"i8", "i32", "i64"}:
                        c0 = _fresh("c0")
                        out_lines.append(f"        {c0} = arith.constant 0 : {cond_ty}")
                        cond_i1 = _fresh("cond")
                        out_lines.append(f"        {cond_i1} = arith.cmpi ne, {cond_ssa}, {c0} : {cond_ty}")
                        cond_ssa = cond_i1
                        cond_ty = "i1"
                    else:
                        raise RuntimeError(f"where condition must be bool/i1, got {cond_ty}")
                out_ty = _infer_value_ty(outv)
                if not out_ty:
                    raise RuntimeError("where missing output dtype")
                v_t = _coerce_scalar(str(in_ssa[1]), str(in_ty[1]), str(out_ty))
                v_f = _coerce_scalar(str(in_ssa[2]), str(in_ty[2]), str(out_ty))
                dst = _fresh("where")
                out_lines.append(f"        {dst} = arith.select {cond_ssa}, {v_t}, {v_f} : {out_ty}")
                computed[outv] = dst
                computed_ty[outv] = str(out_ty)
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
    row_reduce_max_axis1: dict[str, Any] | None = None
    row_reduce_any_axis1: dict[str, Any] | None = None
    row_argmax_axis1: dict[str, Any] | None = None
    row_argmin_axis1: dict[str, Any] | None = None
    row_reduce_prod_axis1: dict[str, Any] | None = None
    row_reduce_min_argmin_axis1: dict[str, Any] | None = None
    reduce_min_all_v1: dict[str, Any] | None = None
    reduce_prod_all_v1: dict[str, Any] | None = None
    trace2d_v1: dict[str, Any] | None = None
    count_nonzero2d_v1: dict[str, Any] | None = None
    allclose2d_v1: dict[str, Any] | None = None
    mse_loss2d_v1: dict[str, Any] | None = None
    nll_loss_forward_v1: dict[str, Any] | None = None
    nll_loss2d_forward_v1: dict[str, Any] | None = None
    per_token_group_quant_fp8_2d_v1: dict[str, Any] | None = None
    batch_norm2d_v1: dict[str, Any] | None = None
    row_std_axis1_v1: dict[str, Any] | None = None
    row_var_axis1_v1: dict[str, Any] | None = None
    weight_norm2d_v1: dict[str, Any] | None = None
    stack2d_v1: dict[str, Any] | None = None
    polar2d_v1: dict[str, Any] | None = None
    diag_embed2d_v1: dict[str, Any] | None = None
    upsample_nearest1d_ncl_v1: dict[str, Any] | None = None
    upsample_nearest2d_nchw_v1: dict[str, Any] | None = None
    glu2d_v1: dict[str, Any] | None = None
    row_log_softmax_axis1_v1: dict[str, Any] | None = None
    row_softmax_axis1: dict[str, Any] | None = None
    row_masked_softmax_axis1: dict[str, Any] | None = None
    row_grouped_row_sum2d: dict[str, Any] | None = None
    group_norm_kernel_v1: dict[str, Any] | None = None
    row_layer_norm_persistent: dict[str, Any] | None = None
    row_layer_norm_residual2d: dict[str, Any] | None = None
    row_rms_norm2d: dict[str, Any] | None = None
    row_rms_norm_residual2d: dict[str, Any] | None = None
    dropout_v1: dict[str, Any] | None = None
    correlation_v1: dict[str, Any] | None = None
    resize_v1: dict[str, Any] | None = None
    rope_v1: dict[str, Any] | None = None
    warp_v1: dict[str, Any] | None = None
    matmul_v1: dict[str, Any] | None = None
    matvec_v1: dict[str, Any] | None = None
    mlp2d_v1: dict[str, Any] | None = None
    attn2d_v1: dict[str, Any] | None = None
    attn_fwd_v1: dict[str, Any] | None = None
    sdpa_bhsd_v1: dict[str, Any] | None = None
    conv1d_ncl_v1: dict[str, Any] | None = None
    conv2d_nchw_v1: dict[str, Any] | None = None
    conv3d_ncdhw_v1: dict[str, Any] | None = None
    conv_depthwise2d_nchw_v1: dict[str, Any] | None = None
    unique2d_v1: dict[str, Any] | None = None
    nonzero2d_v1: dict[str, Any] | None = None
    upsample_bicubic2d_aa_v1: dict[str, Any] | None = None
    kron2d_v1: dict[str, Any] | None = None
    cumsum2d_v1: dict[str, Any] | None = None
    normed_cumsum2d_v1: dict[str, Any] | None = None
    cummax1d_v1: dict[str, Any] | None = None
    cummin1d_v1: dict[str, Any] | None = None
    row_sort_axis1_bitonic_v1: dict[str, Any] | None = None
    row_topk_axis1_bitonic_v1: dict[str, Any] | None = None
    row_quantile_axis1_sort_v1: dict[str, Any] | None = None
    index_add2d_axis0_v1: dict[str, Any] | None = None
    index_put2d_v1: dict[str, Any] | None = None
    scatter2d_dim1_v1: dict[str, Any] | None = None
    select_scatter2d_dim1_v1: dict[str, Any] | None = None
    slice_scatter2d_dim1_v1: dict[str, Any] | None = None
    masked_select2d_v1: dict[str, Any] | None = None
    masked_scatter2d_v1: dict[str, Any] | None = None
    avg_pool2d_nchw_v1: dict[str, Any] | None = None
    max_pool2d_with_indices_nchw_v1: dict[str, Any] | None = None

    ops_list = [op for op in list(intent.ops or []) if op is not None]
    if ops_list:
        # Kernel-identity patterns (Triton-native coverage kernels).
        if attn2d_v1 is None and intent_name in {"masked_attention2d", "flash_attention2d"}:
            required = {"Q", "K", "V", "sm_scale", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                q_dims = list(arg_specs["Q"].get("dims") or [])
                k_dims = list(arg_specs["K"].get("dims") or [])
                v_dims = list(arg_specs["V"].get("dims") or [])
                o_dims = list(arg_specs[str(out_name)].get("dims") or [])
                if len(q_dims) == 2 and len(k_dims) == 2 and len(v_dims) == 2 and len(o_dims) == 2:
                    q_ctx, hd = int(q_dims[0]), int(q_dims[1])
                    kv_ctx, hd_k = int(k_dims[0]), int(k_dims[1])
                    kv_ctx_v, hd_v = int(v_dims[0]), int(v_dims[1])
                    if (
                        q_ctx > 0
                        and kv_ctx > 0
                        and int(o_dims[0]) == q_ctx
                        and int(o_dims[1]) == int(hd)
                        and hd_k == int(hd)
                        and kv_ctx_v == kv_ctx
                        and hd_v == int(hd)
                        and str(arg_specs["Q"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["K"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["V"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["sm_scale"].get("memref_elem_ty") or "") == "f32"
                    ):
                        attn2d_v1 = {
                            "Q": "Q",
                            "K": "K",
                            "V": "V",
                            "out": str(out_name),
                            "sm_scale": "sm_scale",
                            "Q_CTX": int(q_ctx),
                            "KV_CTX": int(kv_ctx),
                            "HEAD_DIM": int(hd),
                            "causal": True,
                        }

        if attn_fwd_v1 is None and intent_name == "_attn_fwd":
            required = {"Q", "K", "V", "sm_scale", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                q_dims = list(arg_specs["Q"].get("dims") or [])
                k_dims = list(arg_specs["K"].get("dims") or [])
                v_dims = list(arg_specs["V"].get("dims") or [])
                o_dims = list(arg_specs[str(out_name)].get("dims") or [])
                if len(q_dims) == 4 and len(k_dims) == 4 and len(v_dims) == 4 and len(o_dims) == 4:
                    z, qh, q_ctx, hd = (int(q_dims[0]), int(q_dims[1]), int(q_dims[2]), int(q_dims[3]))
                    z2, kh, kv_ctx, hd_k = (int(k_dims[0]), int(k_dims[1]), int(k_dims[2]), int(k_dims[3]))
                    z3, vh, kv_ctx_v, hd_v = (int(v_dims[0]), int(v_dims[1]), int(v_dims[2]), int(v_dims[3]))
                    if (
                        z > 0
                        and qh > 0
                        and q_ctx > 0
                        and hd > 0
                        and z2 == z
                        and z3 == z
                        and kh == qh
                        and vh == qh
                        and kv_ctx > 0
                        and kv_ctx_v == kv_ctx
                        and hd_k == hd
                        and hd_v == hd
                        and [int(o_dims[0]), int(o_dims[1]), int(o_dims[2]), int(o_dims[3])] == [z, qh, q_ctx, hd]
                        and str(arg_specs["Q"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["K"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["V"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["sm_scale"].get("memref_elem_ty") or "") == "f32"
                    ):
                        # `attn_mask` is present in the intent seed but is a no-op (where(x,a,a)).
                        # Keep it as an ABI arg (so runtime bindings remain stable) but do not
                        # apply it in real-MLIR lowering.
                        attn_fwd_v1 = {
                            "Q": "Q",
                            "K": "K",
                            "V": "V",
                            "out": str(out_name),
                            "sm_scale": "sm_scale",
                            "attn_mask": ("attn_mask" if "attn_mask" in arg_specs else None),
                            "Z": int(z),
                            "num_head": int(qh),
                            "Q_CTX": int(q_ctx),
                            "KV_CTX": int(kv_ctx),
                            "HEAD_DIM": int(hd),
                            "causal": False,
                        }

        if sdpa_bhsd_v1 is None and intent_name in {"scaled_dot_product_attention_bhsd", "flash_attn_varlen_func_bhsd"}:
            if len(ops_list) != 1:
                raise RuntimeError(f"{intent_name} expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "scaled_dot_product_attention" or len(op0_inputs) != 3 or op0_out != str(out_name):
                raise RuntimeError(f"{intent_name} expects scaled_dot_product_attention(query,key,value)->{out_name}")
            q_name, k_name, v_name = (str(op0_inputs[0]), str(op0_inputs[1]), str(op0_inputs[2]))
            required = {q_name, k_name, v_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"{intent_name} missing ABI tensors: {missing}")

            q_dims = list(arg_specs[q_name].get("dims") or [])
            k_dims = list(arg_specs[k_name].get("dims") or [])
            v_dims = list(arg_specs[v_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(q_dims) != 4 or len(k_dims) != 4 or len(v_dims) != 4 or len(o_dims) != 4:
                raise RuntimeError(f"{intent_name} expects rank-4 tensors (B,H,S,D)")
            b_dim, h_dim, q_len, d_dim = (int(q_dims[0]), int(q_dims[1]), int(q_dims[2]), int(q_dims[3]))
            b2, h2, k_len, d2 = (int(k_dims[0]), int(k_dims[1]), int(k_dims[2]), int(k_dims[3]))
            b3, h3, k_len3, d3 = (int(v_dims[0]), int(v_dims[1]), int(v_dims[2]), int(v_dims[3]))
            if (
                b_dim <= 0
                or h_dim <= 0
                or q_len <= 0
                or k_len <= 0
                or d_dim <= 0
                or b2 != b_dim
                or b3 != b_dim
                or h2 != h_dim
                or h3 != h_dim
                or k_len3 != k_len
                or d2 != d_dim
                or d3 != d_dim
                or [int(o_dims[0]), int(o_dims[1]), int(o_dims[2]), int(o_dims[3])] != [b_dim, h_dim, q_len, d_dim]
            ):
                raise RuntimeError(
                    f"{intent_name} shape mismatch: "
                    f"query={q_dims} key={k_dims} value={v_dims} out={o_dims}"
                )
            if int(d_dim) > 256:
                raise RuntimeError(f"{intent_name} currently requires D<=256, got D={d_dim}")

            ok = (
                str(arg_specs[q_name].get("memref_elem_ty")) == "f32"
                and str(arg_specs[k_name].get("memref_elem_ty")) == "f32"
                and str(arg_specs[v_name].get("memref_elem_ty")) == "f32"
                and str(arg_specs[str(out_name)].get("memref_elem_ty")) == "f32"
            )
            if not ok:
                raise RuntimeError(f"{intent_name} expects f32 query/key/value/out")

            attrs = dict(getattr(op0, "attrs", {}) or {})
            is_causal = bool(attrs.get("is_causal", False))
            if "IS_CAUSAL" in bindings:
                try:
                    is_causal = bool(int(bindings["IS_CAUSAL"]))
                except Exception:
                    is_causal = bool(is_causal)
            sdpa_bhsd_v1 = {
                "query": str(q_name),
                "key": str(k_name),
                "value": str(v_name),
                "out": str(out_name),
                "B": int(b_dim),
                "H": int(h_dim),
                "Q": int(q_len),
                "K": int(k_len),
                "D": int(d_dim),
                "is_causal": bool(is_causal),
            }

        if conv1d_ncl_v1 is None and intent_name == "conv1d_ncl":
            if len(ops_list) != 1:
                raise RuntimeError(f"conv1d_ncl expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "conv1d" or len(op0_inputs) != 3 or op0_out != str(out_name):
                raise RuntimeError("conv1d_ncl expects conv1d(input, weight, bias) -> out")
            inp_name, w_name, b_name = (str(op0_inputs[0]), str(op0_inputs[1]), str(op0_inputs[2]))
            required = {inp_name, w_name, b_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"conv1d_ncl missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            w_dims = list(arg_specs[w_name].get("dims") or [])
            b_dims = list(arg_specs[b_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 3 or len(w_dims) != 3 or len(b_dims) != 1 or len(o_dims) != 3:
                raise RuntimeError(f"conv1d_ncl expects input[N,C_IN,L], weight[C_OUT,C_PER_G,K], bias[C_OUT], out[N,C_OUT,OL]")
            n_dim, c_in, l_dim = (int(in_dims[0]), int(in_dims[1]), int(in_dims[2]))
            c_out, c_per_g, k_dim = (int(w_dims[0]), int(w_dims[1]), int(w_dims[2]))
            n_out, c_out2, ol_dim = (int(o_dims[0]), int(o_dims[1]), int(o_dims[2]))
            if n_out != n_dim or c_out2 != c_out or int(b_dims[0]) != c_out:
                raise RuntimeError(f"conv1d_ncl shape mismatch: input={in_dims} weight={w_dims} bias={b_dims} out={o_dims}")
            groups = int(bindings.get("GROUPS") or 1)
            stride = int(bindings.get("STRIDE") or 1)
            padding = int(bindings.get("PADDING") or 0)
            dilation = int(bindings.get("DILATION") or 1)
            if groups <= 0 or stride <= 0 or dilation <= 0 or padding < 0:
                raise RuntimeError("conv1d_ncl invalid stride/padding/dilation/groups")
            if c_in % groups != 0 or c_out % groups != 0:
                raise RuntimeError(f"conv1d_ncl expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
            if c_per_g != (c_in // groups):
                raise RuntimeError(f"conv1d_ncl expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv1d_ncl expects f32 input tensor")
            if str(arg_specs[w_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv1d_ncl expects f32 weight tensor")
            if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv1d_ncl expects f32 bias tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv1d_ncl expects f32 output tensor")
            conv1d_ncl_v1 = {
                "input": str(inp_name),
                "weight": str(w_name),
                "bias": str(b_name),
                "out": str(out_name),
                "N": int(n_dim),
                "C_IN": int(c_in),
                "C_OUT": int(c_out),
                "L": int(l_dim),
                "K": int(k_dim),
                "STRIDE": int(stride),
                "PADDING": int(padding),
                "DILATION": int(dilation),
                "GROUPS": int(groups),
                "C_PER_G": int(c_per_g),
                "OL": int(ol_dim),
            }

        if conv2d_nchw_v1 is None and intent_name == "conv2d_nchw":
            if len(ops_list) != 1:
                raise RuntimeError(f"conv2d_nchw expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "conv2d" or len(op0_inputs) != 3 or op0_out != str(out_name):
                raise RuntimeError("conv2d_nchw expects conv2d(input, weight, bias) -> out")
            inp_name, w_name, b_name = (str(op0_inputs[0]), str(op0_inputs[1]), str(op0_inputs[2]))
            required = {inp_name, w_name, b_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"conv2d_nchw missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            w_dims = list(arg_specs[w_name].get("dims") or [])
            b_dims = list(arg_specs[b_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 4 or len(w_dims) != 4 or len(b_dims) != 1 or len(o_dims) != 4:
                raise RuntimeError("conv2d_nchw expects input[N,C_IN,H,W], weight[C_OUT,C_PER_G,KH,KW], bias[C_OUT], out[N,C_OUT,OH,OW]")
            n_dim, c_in, h_dim, w_dim = (int(in_dims[0]), int(in_dims[1]), int(in_dims[2]), int(in_dims[3]))
            c_out, c_per_g, kh, kw = (int(w_dims[0]), int(w_dims[1]), int(w_dims[2]), int(w_dims[3]))
            n_out, c_out2, oh, ow = (int(o_dims[0]), int(o_dims[1]), int(o_dims[2]), int(o_dims[3]))
            if n_out != n_dim or c_out2 != c_out or int(b_dims[0]) != c_out:
                raise RuntimeError(f"conv2d_nchw shape mismatch: input={in_dims} weight={w_dims} bias={b_dims} out={o_dims}")
            groups = int(bindings.get("GROUPS") or 1)
            sh = int(bindings.get("SH") or 1)
            sw = int(bindings.get("SW") or 1)
            ph = int(bindings.get("PH") or 0)
            pw = int(bindings.get("PW") or 0)
            dh = int(bindings.get("DH") or 1)
            dw = int(bindings.get("DW") or 1)
            if groups <= 0 or sh <= 0 or sw <= 0 or dh <= 0 or dw <= 0 or ph < 0 or pw < 0:
                raise RuntimeError("conv2d_nchw invalid stride/padding/dilation/groups")
            if c_in % groups != 0 or c_out % groups != 0:
                raise RuntimeError(f"conv2d_nchw expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
            if c_per_g != (c_in // groups):
                raise RuntimeError(f"conv2d_nchw expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv2d_nchw expects f32 input tensor")
            if str(arg_specs[w_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv2d_nchw expects f32 weight tensor")
            if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv2d_nchw expects f32 bias tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv2d_nchw expects f32 output tensor")
            conv2d_nchw_v1 = {
                "input": str(inp_name),
                "weight": str(w_name),
                "bias": str(b_name),
                "out": str(out_name),
                "N": int(n_dim),
                "C_IN": int(c_in),
                "C_OUT": int(c_out),
                "H": int(h_dim),
                "W": int(w_dim),
                "KH": int(kh),
                "KW": int(kw),
                "SH": int(sh),
                "SW": int(sw),
                "PH": int(ph),
                "PW": int(pw),
                "DH": int(dh),
                "DW": int(dw),
                "GROUPS": int(groups),
                "C_PER_G": int(c_per_g),
                "OH": int(oh),
                "OW": int(ow),
            }

        if conv3d_ncdhw_v1 is None and intent_name == "conv3d_ncdhw":
            if len(ops_list) != 1:
                raise RuntimeError(f"conv3d_ncdhw expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "conv3d" or len(op0_inputs) != 3 or op0_out != str(out_name):
                raise RuntimeError("conv3d_ncdhw expects conv3d(input, weight, bias) -> out")
            inp_name, w_name, b_name = (str(op0_inputs[0]), str(op0_inputs[1]), str(op0_inputs[2]))
            required = {inp_name, w_name, b_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"conv3d_ncdhw missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            w_dims = list(arg_specs[w_name].get("dims") or [])
            b_dims = list(arg_specs[b_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 5 or len(w_dims) != 5 or len(b_dims) != 1 or len(o_dims) != 5:
                raise RuntimeError("conv3d_ncdhw expects input[N,C_IN,D,H,W], weight[C_OUT,C_PER_G,KD,KH,KW], bias[C_OUT], out[N,C_OUT,OD,OH,OW]")
            n_dim, c_in, d_dim, h_dim, w_dim = (int(in_dims[0]), int(in_dims[1]), int(in_dims[2]), int(in_dims[3]), int(in_dims[4]))
            c_out, c_per_g, kd, kh, kw = (int(w_dims[0]), int(w_dims[1]), int(w_dims[2]), int(w_dims[3]), int(w_dims[4]))
            n_out, c_out2, od, oh, ow = (int(o_dims[0]), int(o_dims[1]), int(o_dims[2]), int(o_dims[3]), int(o_dims[4]))
            if n_out != n_dim or c_out2 != c_out or int(b_dims[0]) != c_out:
                raise RuntimeError(f"conv3d_ncdhw shape mismatch: input={in_dims} weight={w_dims} bias={b_dims} out={o_dims}")
            groups = int(bindings.get("GROUPS") or 1)
            sd = int(bindings.get("SD") or 1)
            sh = int(bindings.get("SH") or 1)
            sw = int(bindings.get("SW") or 1)
            pd = int(bindings.get("PD") or 0)
            ph = int(bindings.get("PH") or 0)
            pw = int(bindings.get("PW") or 0)
            dd = int(bindings.get("DD") or 1)
            dh = int(bindings.get("DH") or 1)
            dw = int(bindings.get("DW") or 1)
            if groups <= 0 or sd <= 0 or sh <= 0 or sw <= 0 or dd <= 0 or dh <= 0 or dw <= 0 or pd < 0 or ph < 0 or pw < 0:
                raise RuntimeError("conv3d_ncdhw invalid stride/padding/dilation/groups")
            if c_in % groups != 0 or c_out % groups != 0:
                raise RuntimeError(f"conv3d_ncdhw expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
            if c_per_g != (c_in // groups):
                raise RuntimeError(f"conv3d_ncdhw expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv3d_ncdhw expects f32 input tensor")
            if str(arg_specs[w_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv3d_ncdhw expects f32 weight tensor")
            if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv3d_ncdhw expects f32 bias tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv3d_ncdhw expects f32 output tensor")
            conv3d_ncdhw_v1 = {
                "input": str(inp_name),
                "weight": str(w_name),
                "bias": str(b_name),
                "out": str(out_name),
                "N": int(n_dim),
                "C_IN": int(c_in),
                "C_OUT": int(c_out),
                "D": int(d_dim),
                "H": int(h_dim),
                "W": int(w_dim),
                "KD": int(kd),
                "KH": int(kh),
                "KW": int(kw),
                "SD": int(sd),
                "SH": int(sh),
                "SW": int(sw),
                "PD": int(pd),
                "PH": int(ph),
                "PW": int(pw),
                "DD": int(dd),
                "DH": int(dh),
                "DW": int(dw),
                "GROUPS": int(groups),
                "C_PER_G": int(c_per_g),
                "OD": int(od),
                "OH": int(oh),
                "OW": int(ow),
            }

        if conv_depthwise2d_nchw_v1 is None and intent_name == "conv_depthwise2d_nchw":
            if len(ops_list) != 1:
                raise RuntimeError(f"conv_depthwise2d_nchw expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "conv_depthwise2d" or len(op0_inputs) != 3 or op0_out != str(out_name):
                raise RuntimeError("conv_depthwise2d_nchw expects conv_depthwise2d(input, weight, bias) -> out")
            inp_name, w_name, b_name = (str(op0_inputs[0]), str(op0_inputs[1]), str(op0_inputs[2]))
            required = {inp_name, w_name, b_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"conv_depthwise2d_nchw missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            w_dims = list(arg_specs[w_name].get("dims") or [])
            b_dims = list(arg_specs[b_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 4 or len(w_dims) != 4 or len(b_dims) != 1 or len(o_dims) != 4:
                raise RuntimeError("conv_depthwise2d_nchw expects input[N,C_IN,H,W], weight[C_OUT,1,KH,KW], bias[C_OUT], out[N,C_OUT,OH,OW]")
            n_dim, c_in, h_dim, w_dim = (int(in_dims[0]), int(in_dims[1]), int(in_dims[2]), int(in_dims[3]))
            c_out, one_dim, kh, kw = (int(w_dims[0]), int(w_dims[1]), int(w_dims[2]), int(w_dims[3]))
            n_out, c_out2, oh, ow = (int(o_dims[0]), int(o_dims[1]), int(o_dims[2]), int(o_dims[3]))
            if n_out != n_dim or c_out2 != c_out or int(b_dims[0]) != c_out:
                raise RuntimeError(
                    f"conv_depthwise2d_nchw shape mismatch: input={in_dims} weight={w_dims} bias={b_dims} out={o_dims}"
                )
            if one_dim != 1:
                raise RuntimeError(f"conv_depthwise2d_nchw expects weight dim1==1, got {one_dim}")
            sh = int(bindings.get("SH") or 1)
            sw = int(bindings.get("SW") or 1)
            ph = int(bindings.get("PH") or 0)
            pw = int(bindings.get("PW") or 0)
            dh = int(bindings.get("DH") or 1)
            dw = int(bindings.get("DW") or 1)
            mult = int(bindings.get("MULT") or 1)
            if sh <= 0 or sw <= 0 or dh <= 0 or dw <= 0 or ph < 0 or pw < 0 or mult <= 0:
                raise RuntimeError("conv_depthwise2d_nchw invalid stride/padding/dilation/mult")
            if c_out != (c_in * mult):
                raise RuntimeError(f"conv_depthwise2d_nchw expects C_OUT=C_IN*MULT, got C_OUT={c_out} C_IN={c_in} MULT={mult}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv_depthwise2d_nchw expects f32 input tensor")
            if str(arg_specs[w_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv_depthwise2d_nchw expects f32 weight tensor")
            if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv_depthwise2d_nchw expects f32 bias tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "f32":
                raise RuntimeError("conv_depthwise2d_nchw expects f32 output tensor")
            conv_depthwise2d_nchw_v1 = {
                "input": str(inp_name),
                "weight": str(w_name),
                "bias": str(b_name),
                "out": str(out_name),
                "N": int(n_dim),
                "C_IN": int(c_in),
                "C_OUT": int(c_out),
                "H": int(h_dim),
                "W": int(w_dim),
                "KH": int(kh),
                "KW": int(kw),
                "SH": int(sh),
                "SW": int(sw),
                "PH": int(ph),
                "PW": int(pw),
                "DH": int(dh),
                "DW": int(dw),
                "MULT": int(mult),
                "OH": int(oh),
                "OW": int(ow),
            }

        if unique2d_v1 is None and intent_name == "unique2d":
            if len(ops_list) != 1:
                raise RuntimeError(f"unique2d expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "unique" or len(op0_inputs) != 1 or op0_out != str(out_name):
                raise RuntimeError("unique2d expects unique(inp) -> out")
            inp_name = str(op0_inputs[0])
            required = {inp_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"unique2d missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 1 or len(o_dims) != 1:
                raise RuntimeError("unique2d expects input[N], out[U]")
            n_dim = int(in_dims[0])
            u_dim = int(o_dims[0])
            if n_dim <= 0 or u_dim <= 0:
                raise RuntimeError(f"unique2d expects positive dims, got N={n_dim} U={u_dim}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "i32":
                raise RuntimeError("unique2d expects i32 input tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "i32":
                raise RuntimeError("unique2d expects i32 output tensor")
            unique2d_v1 = {"inp": inp_name, "out": str(out_name), "N": int(n_dim), "U": int(u_dim)}

        if nonzero2d_v1 is None and intent_name == "nonzero2d":
            if len(ops_list) != 1:
                raise RuntimeError(f"nonzero2d expects 1 op, got {len(ops_list)}")
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            if op0_name != "nonzero" or len(op0_inputs) != 1 or op0_out != str(out_name):
                raise RuntimeError("nonzero2d expects nonzero(inp) -> out")
            inp_name = str(op0_inputs[0])
            required = {inp_name, str(out_name)}
            if not required.issubset(set(arg_specs.keys())):
                missing = sorted([x for x in required if x not in arg_specs])
                raise RuntimeError(f"nonzero2d missing ABI tensors: {missing}")
            in_dims = list(arg_specs[inp_name].get("dims") or [])
            o_dims = list(arg_specs[str(out_name)].get("dims") or [])
            if len(in_dims) != 2 or len(o_dims) != 2:
                raise RuntimeError("nonzero2d expects input[M,N], out[num_nonzeros,2]")
            m_dim, n_dim = (int(in_dims[0]), int(in_dims[1]))
            nnz_dim, two_dim = (int(o_dims[0]), int(o_dims[1]))
            if m_dim <= 0 or n_dim <= 0 or nnz_dim <= 0 or two_dim != 2:
                raise RuntimeError(f"nonzero2d invalid dims: input={in_dims} out={o_dims}")
            if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
                raise RuntimeError("nonzero2d expects f32 input tensor")
            if str(arg_specs[str(out_name)].get("memref_elem_ty")) != "i64":
                raise RuntimeError("nonzero2d expects i64 output tensor")
            nonzero2d_v1 = {"inp": inp_name, "out": str(out_name), "M": int(m_dim), "N": int(n_dim), "NNZ": int(nnz_dim)}

        if upsample_bicubic2d_aa_v1 is None and intent_name == "upsample_bicubic2d_aa":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                if op0_name == "upsample_bicubic2d_aa" and len(op0_inputs) == 1 and op0_out == str(out_name):
                    inp_name = str(op0_inputs[0])
                    out_name2 = str(op0_out)
                    required = {inp_name, out_name2, "reciprocal_scale_h", "reciprocal_scale_w"}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        out_dims2 = list(arg_specs[out_name2].get("dims") or [])
                        if len(in_dims) == 4 and len(out_dims2) == 4:
                            n_dim, c_dim, ih_dim, iw_dim = map(int, in_dims)
                            n2, c2, oh_dim, ow_dim = map(int, out_dims2)
                            ok = (
                                n_dim > 0
                                and c_dim > 0
                                and ih_dim > 0
                                and iw_dim > 0
                                and oh_dim > 0
                                and ow_dim > 0
                                and n2 == n_dim
                                and c2 == c_dim
                                and out_rank == 4
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[out_name2].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs["reciprocal_scale_h"].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs["reciprocal_scale_w"].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                try:
                                    a = float(attrs.get("a", -0.5))
                                except Exception:
                                    a = -0.5
                                try:
                                    support = float(attrs.get("support", 2.0))
                                except Exception:
                                    support = 2.0
                                try:
                                    invscale = float(attrs.get("invscale", 1.0))
                                except Exception:
                                    invscale = 1.0
                                taps = 5
                                try:
                                    impl = attrs.get("impl") if isinstance(attrs.get("impl"), dict) else {}
                                    index_plan = (
                                        impl.get("index_plan")
                                        if isinstance(impl, dict) and isinstance(impl.get("index_plan"), dict)
                                        else {}
                                    )
                                    taps_raw = index_plan.get("taps")
                                    if isinstance(taps_raw, int):
                                        taps = int(taps_raw)
                                except Exception:
                                    taps = 5
                                if taps != 5:
                                    raise RuntimeError(f"upsample_bicubic2d_aa real-mlir currently requires taps=5, got {taps}")
                                upsample_bicubic2d_aa_v1 = {
                                    "inp": inp_name,
                                    "out": out_name2,
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "IH": int(ih_dim),
                                    "IW": int(iw_dim),
                                    "OH": int(oh_dim),
                                    "OW": int(ow_dim),
                                    "reciprocal_scale_h": "reciprocal_scale_h",
                                    "reciprocal_scale_w": "reciprocal_scale_w",
                                    "a": float(a),
                                    "support": float(support),
                                    "invscale": float(invscale),
                                }

        # Wave13 (coverage_batches backfill): stack/polar/diag_embed/upsample_nearest/glu/log_softmax.
        if stack2d_v1 is None and intent_name == "stack2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs.get("axis")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                if op0_name == "stack" and len(op0_inputs) == 2 and op0_out == str(out_name) and axis_i == 0:
                    inp0, inp1 = op0_inputs
                    required = {inp0, inp1, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in0_dims = list(arg_specs[inp0].get("dims") or [])
                        in1_dims = list(arg_specs[inp1].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in0_dims) == 2 and in1_dims == in0_dims and len(out_dims2) == 3:
                            m_dim, n_dim = int(in0_dims[0]), int(in0_dims[1])
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and out_dims2 == [2, m_dim, n_dim]
                                and out_rank == 3
                                and str(arg_specs[inp0].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[inp1].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                stack2d_v1 = {"inp0": inp0, "inp1": inp1, "out": op0_out, "M": int(m_dim), "N": int(n_dim)}

        if polar2d_v1 is None and intent_name == "polar2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "polar" and len(op0_inputs) == 2 and op0_out == str(out_name):
                    abs_name, ang_name = op0_inputs
                    required = {abs_name, ang_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        abs_dims = list(arg_specs[abs_name].get("dims") or [])
                        ang_dims = list(arg_specs[ang_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(abs_dims) == 2 and ang_dims == abs_dims and len(out_dims2) == 3:
                            m_dim, n_dim = int(abs_dims[0]), int(abs_dims[1])
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and out_dims2 == [m_dim, n_dim, 2]
                                and out_rank == 3
                                and str(arg_specs[abs_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[ang_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                polar2d_v1 = {"abs": abs_name, "angle": ang_name, "out": op0_out, "M": int(m_dim), "N": int(n_dim)}

        if upsample_nearest1d_ncl_v1 is None and intent_name == "upsample_nearest1d_ncl":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "upsample_nearest1d" and len(op0_inputs) == 1 and op0_out == str(out_name):
                    inp_name = str(op0_inputs[0])
                    required = {inp_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 3 and len(out_dims2) == 3:
                            n_dim, c_dim, il_dim = map(int, in_dims)
                            n2, c2, ol_dim = map(int, out_dims2)
                            ok = (
                                n_dim > 0
                                and c_dim > 0
                                and il_dim > 0
                                and ol_dim > 0
                                and n2 == n_dim
                                and c2 == c_dim
                                and out_rank == 3
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                upsample_nearest1d_ncl_v1 = {
                                    "inp": inp_name,
                                    "out": op0_out,
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "IL": int(il_dim),
                                    "OL": int(ol_dim),
                                }

        if upsample_nearest2d_nchw_v1 is None and intent_name == "upsample_nearest2d_nchw":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "upsample_nearest2d" and len(op0_inputs) == 1 and op0_out == str(out_name):
                    inp_name = str(op0_inputs[0])
                    required = {inp_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 4 and len(out_dims2) == 4:
                            n_dim, c_dim, ih_dim, iw_dim = map(int, in_dims)
                            n2, c2, oh_dim, ow_dim = map(int, out_dims2)
                            ok = (
                                n_dim > 0
                                and c_dim > 0
                                and ih_dim > 0
                                and iw_dim > 0
                                and oh_dim > 0
                                and ow_dim > 0
                                and n2 == n_dim
                                and c2 == c_dim
                                and out_rank == 4
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                upsample_nearest2d_nchw_v1 = {
                                    "inp": inp_name,
                                    "out": op0_out,
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "IH": int(ih_dim),
                                    "IW": int(iw_dim),
                                    "OH": int(oh_dim),
                                    "OW": int(ow_dim),
                                }

        if glu2d_v1 is None and intent_name == "glu2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs.get("axis")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                if op0_name == "glu" and len(op0_inputs) == 1 and op0_out == str(out_name) and axis_i == 1:
                    x_name = str(op0_inputs[0])
                    required = {x_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        x_dims = list(arg_specs[x_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(x_dims) == 2 and len(out_dims2) == 2:
                            m_dim, n_dim = int(x_dims[0]), int(x_dims[1])
                            m2, nh_dim = int(out_dims2[0]), int(out_dims2[1])
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and nh_dim > 0
                                and m2 == m_dim
                                and n_dim == 2 * nh_dim
                                and out_rank == 2
                                and str(arg_specs[x_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                glu2d_v1 = {
                                    "x": x_name,
                                    "out": op0_out,
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "N_HALF": int(nh_dim),
                                }

        if diag_embed2d_v1 is None and intent_name == "diag_embed2d":
            required = {"x", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                x_dims = list(arg_specs["x"].get("dims") or [])
                out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                if len(x_dims) == 2 and len(out_dims2) == 3:
                    b_dim, n_dim = int(x_dims[0]), int(x_dims[1])
                    ok = (
                        b_dim > 0
                        and n_dim > 0
                        and out_dims2 == [b_dim, n_dim, n_dim]
                        and out_rank == 3
                        and str(arg_specs["x"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                    )
                    if ok:
                        diag_embed2d_v1 = {"x": "x", "out": str(out_name), "B": int(b_dim), "N": int(n_dim)}

        if row_log_softmax_axis1_v1 is None and intent_name == "log_softmax2d":
            required = {"inp", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                in_dims = list(arg_specs["inp"].get("dims") or [])
                out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                if len(in_dims) == 2 and out_dims2 == in_dims and out_rank == 2:
                    m_dim, n_dim = int(in_dims[0]), int(in_dims[1])
                    ok = (
                        m_dim > 0
                        and n_dim > 0
                        and str(arg_specs["inp"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                    )
                    if ok:
                        row_log_softmax_axis1_v1 = {"inp": "inp", "out": str(out_name), "reduce_n": int(n_dim)}

        if mse_loss2d_v1 is None and intent_name == "mse_loss2d":
            if len(ops_list) == 1 and out_rank == 0:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                reduction = attrs.get("reduction")
                try:
                    reduction_i = int(reduction) if reduction is not None else None
                except Exception:
                    reduction_i = None
                if (
                    op0_name == "mse_loss"
                    and len(op0_inputs) == 2
                    and op0_out == str(out_name)
                    and reduction_i == 1  # mean
                ):
                    inp_name, tgt_name = str(op0_inputs[0]), str(op0_inputs[1])
                    required = {inp_name, tgt_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        inp_dims = list(arg_specs[inp_name].get("dims") or [])
                        tgt_dims = list(arg_specs[tgt_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(inp_dims) == 2 and tgt_dims == inp_dims and len(out_dims2) == 0:
                            m_dim, n_dim = map(int, inp_dims)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[tgt_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                mse_loss2d_v1 = {
                                    "inp": inp_name,
                                    "target": tgt_name,
                                    "out": op0_out,
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "reduction": int(reduction_i),
                                }

        if nll_loss_forward_v1 is None and intent_name == "nll_loss_forward":
            if len(ops_list) == 1 and out_rank == 0:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                reduction = attrs.get("reduction")
                ignore_index = attrs.get("ignore_index", -100)
                try:
                    reduction_i = int(reduction) if reduction is not None else None
                except Exception:
                    reduction_i = None
                try:
                    ignore_i = int(ignore_index)
                except Exception:
                    ignore_i = -100
                if op0_name == "nll_loss_forward" and len(op0_inputs) == 3 and op0_out == str(out_name) and reduction_i == 1:
                    logits_name, tgt_name, w_name = map(str, op0_inputs)
                    required = {logits_name, tgt_name, w_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        logits_dims = list(arg_specs[logits_name].get("dims") or [])
                        tgt_dims = list(arg_specs[tgt_name].get("dims") or [])
                        w_dims = list(arg_specs[w_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(logits_dims) == 2 and len(tgt_dims) == 1 and len(w_dims) == 1 and len(out_dims2) == 0:
                            n_dim, c_dim = map(int, logits_dims)
                            ok = (
                                n_dim > 0
                                and c_dim > 1
                                and tgt_dims == [n_dim]
                                and w_dims == [c_dim]
                                and str(arg_specs[logits_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[tgt_name].get("memref_elem_ty") or "") == "i64"
                                and str(arg_specs[w_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                nll_loss_forward_v1 = {
                                    "self": logits_name,
                                    "target": tgt_name,
                                    "weight": w_name,
                                    "out": op0_out,
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "reduction": int(reduction_i),
                                    "ignore_index": int(ignore_i),
                                }

        if nll_loss2d_forward_v1 is None and intent_name == "nll_loss2d_forward":
            if len(ops_list) == 1 and out_rank == 0:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                reduction = attrs.get("reduction")
                ignore_index = attrs.get("ignore_index", -100)
                try:
                    reduction_i = int(reduction) if reduction is not None else None
                except Exception:
                    reduction_i = None
                try:
                    ignore_i = int(ignore_index)
                except Exception:
                    ignore_i = -100
                if op0_name == "nll_loss2d_forward" and len(op0_inputs) == 3 and op0_out == str(out_name) and reduction_i == 1:
                    logits_name, tgt_name, w_name = map(str, op0_inputs)
                    required = {logits_name, tgt_name, w_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        logits_dims = list(arg_specs[logits_name].get("dims") or [])
                        tgt_dims = list(arg_specs[tgt_name].get("dims") or [])
                        w_dims = list(arg_specs[w_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(logits_dims) == 4 and len(tgt_dims) == 3 and len(w_dims) == 1 and len(out_dims2) == 0:
                            n_dim, c_dim, h_dim, w_dim = map(int, logits_dims)
                            ok = (
                                n_dim > 0
                                and c_dim > 1
                                and h_dim > 0
                                and w_dim > 0
                                and tgt_dims == [n_dim, h_dim, w_dim]
                                and w_dims == [c_dim]
                                and str(arg_specs[logits_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[tgt_name].get("memref_elem_ty") or "") == "i64"
                                and str(arg_specs[w_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                nll_loss2d_forward_v1 = {
                                    "self": logits_name,
                                    "target": tgt_name,
                                    "weight": w_name,
                                    "out": op0_out,
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "H": int(h_dim),
                                    "W": int(w_dim),
                                    "reduction": int(reduction_i),
                                    "ignore_index": int(ignore_i),
                                }

        if row_std_axis1_v1 is None and intent_name == "std2d":
            if len(ops_list) == 1 and out_rank == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                dims = attrs.get("dims", attrs.get("axis"))
                dims_list = list(dims) if isinstance(dims, list) else []
                keepdims = bool(attrs.get("keepdims", False))
                correction = attrs.get("correction", attrs.get("ddof", 0))
                try:
                    ddof = int(correction)
                except Exception:
                    ddof = 0
                if op0_name == "std" and len(op0_inputs) == 1 and op0_out == str(out_name):
                    if dims_list == [1] and not keepdims and ddof == 1:
                        inp_name = str(op0_inputs[0])
                        required = {inp_name, op0_out}
                        if required.issubset(set(arg_specs.keys())):
                            inp_dims = list(arg_specs[inp_name].get("dims") or [])
                            out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                            if len(inp_dims) == 2 and len(out_dims2) == 1:
                                m_dim, n_dim = map(int, inp_dims)
                                ok = (
                                    m_dim > 0
                                    and n_dim > 1
                                    and out_dims2 == [int(m_dim)]
                                    and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                )
                                if ok:
                                    row_std_axis1_v1 = {
                                        "inp": inp_name,
                                        "out": op0_out,
                                        "M": int(m_dim),
                                        "N": int(n_dim),
                                        "ddof": int(ddof),
                                    }

        if row_var_axis1_v1 is None and intent_name == "var_mean2d":
            if len(ops_list) == 2 and out_rank == 1:
                op0, op1 = ops_list
                op0_name = str(getattr(op0, "op", "")).strip()
                op1_name = str(getattr(op1, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op1_inputs = [str(x) for x in list(getattr(op1, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                op1_out = str(getattr(op1, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                dims = attrs.get("dims", attrs.get("axis"))
                dims_list = list(dims) if isinstance(dims, list) else []
                keepdims = bool(attrs.get("keepdims", False))
                correction = attrs.get("correction", attrs.get("ddof", 0))
                try:
                    ddof = int(correction)
                except Exception:
                    ddof = 0
                if (
                    op0_name == "std"
                    and op1_name == "mul"
                    and len(op0_inputs) == 1
                    and len(op1_inputs) == 2
                    and op1_out == str(out_name)
                    and op1_inputs[0] == op0_out
                    and op1_inputs[1] == op0_out
                ):
                    if dims_list == [1] and not keepdims and ddof == 1:
                        inp_name = str(op0_inputs[0])
                        required = {inp_name, op1_out}
                        if required.issubset(set(arg_specs.keys())):
                            inp_dims = list(arg_specs[inp_name].get("dims") or [])
                            out_dims2 = list(arg_specs[op1_out].get("dims") or [])
                            if len(inp_dims) == 2 and len(out_dims2) == 1:
                                m_dim, n_dim = map(int, inp_dims)
                                ok = (
                                    m_dim > 0
                                    and n_dim > 1
                                    and out_dims2 == [int(m_dim)]
                                    and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[op1_out].get("memref_elem_ty") or "") == "f32"
                                )
                                if ok:
                                    row_var_axis1_v1 = {
                                        "inp": inp_name,
                                        "out": op1_out,
                                        "M": int(m_dim),
                                        "N": int(n_dim),
                                        "ddof": int(ddof),
                                    }

        if weight_norm2d_v1 is None and intent_name == "weight_norm2d":
            if len(ops_list) == 6 and out_rank == 2:
                op0, op1, op2, op3, op4, op5 = ops_list
                n0 = str(getattr(op0, "op", "")).strip()
                n1 = str(getattr(op1, "op", "")).strip()
                n2 = str(getattr(op2, "op", "")).strip()
                n3 = str(getattr(op3, "op", "")).strip()
                n4 = str(getattr(op4, "op", "")).strip()
                n5 = str(getattr(op5, "op", "")).strip()
                if (n0, n1, n2, n3, n4, n5) == ("mul", "reduce_sum", "sqrt", "div", "broadcast_in_dim", "mul"):
                    mul_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                    mul_out = str(getattr(op0, "output", "")).strip()
                    red_inputs = [str(x) for x in list(getattr(op1, "inputs", []) or []) if str(x).strip()]
                    red_out = str(getattr(op1, "output", "")).strip()
                    red_attrs = dict(getattr(op1, "attrs", {}) or {})
                    dims = red_attrs.get("dims", red_attrs.get("axis"))
                    dims_list = list(dims) if isinstance(dims, list) else []
                    keepdims = bool(red_attrs.get("keepdims", False))
                    sqrt_inputs = [str(x) for x in list(getattr(op2, "inputs", []) or []) if str(x).strip()]
                    sqrt_out = str(getattr(op2, "output", "")).strip()
                    div_inputs = [str(x) for x in list(getattr(op3, "inputs", []) or []) if str(x).strip()]
                    div_out = str(getattr(op3, "output", "")).strip()
                    bc_inputs = [str(x) for x in list(getattr(op4, "inputs", []) or []) if str(x).strip()]
                    bc_out = str(getattr(op4, "output", "")).strip()
                    bc_attrs = dict(getattr(op4, "attrs", {}) or {})
                    bc_dims = bc_attrs.get("broadcast_dims")
                    bc_dims_list = list(bc_dims) if isinstance(bc_dims, list) else []
                    out_shape = bc_attrs.get("out_shape")
                    out_shape_list = list(out_shape) if isinstance(out_shape, list) else []
                    mul2_inputs = [str(x) for x in list(getattr(op5, "inputs", []) or []) if str(x).strip()]
                    mul2_out = str(getattr(op5, "output", "")).strip()
                    v_name = str(mul_inputs[0]) if len(mul_inputs) == 2 and mul_inputs[0] == mul_inputs[1] else ""
                    if (
                        v_name
                        and mul_out
                        and red_out
                        and sqrt_out
                        and div_out
                        and bc_out
                        and mul2_out == str(out_name)
                        and len(red_inputs) == 1
                        and red_inputs[0] == mul_out
                        and len(sqrt_inputs) == 1
                        and sqrt_inputs[0] == red_out
                        and len(div_inputs) == 2
                        and div_inputs[1] == sqrt_out
                        and len(bc_inputs) == 1
                        and bc_inputs[0] == div_out
                        and len(mul2_inputs) == 2
                        and mul2_inputs[0] == v_name
                        and mul2_inputs[1] == bc_out
                        and dims_list == [1]
                        and not keepdims
                        and bc_dims_list == [0]
                        and len(out_shape_list) == 2
                    ):
                        g_name = str(div_inputs[0])
                        required = {v_name, g_name, str(out_name)}
                        if required.issubset(set(arg_specs.keys())):
                            v_dims = list(arg_specs[v_name].get("dims") or [])
                            g_dims = list(arg_specs[g_name].get("dims") or [])
                            out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                            if len(v_dims) == 2 and len(g_dims) == 1 and out_dims2 == v_dims and out_rank == 2:
                                m_dim, n_dim = map(int, v_dims)
                                ok = (
                                    m_dim > 0
                                    and n_dim > 0
                                    and g_dims == [int(m_dim)]
                                    and str(arg_specs[v_name].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[g_name].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                                )
                                if ok:
                                    weight_norm2d_v1 = {
                                        "v": v_name,
                                        "g": g_name,
                                        "out": str(out_name),
                                        "M": int(m_dim),
                                        "N": int(n_dim),
                                    }

        # Pattern: ai_bench_resize (bilinear 2x, fixed-point hw_fl=7).
        if len(ops_list) == 1:
            op0 = ops_list[0]
            op0_name = str(getattr(op0, "op", "")).strip()
            op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
            op0_out = str(getattr(op0, "output", "")).strip()
            attrs = dict(getattr(op0, "attrs", {}) or {})
            if op0_name == "resize" and len(op0_inputs) == 1 and op0_out:
                scale = attrs.get("scale_factor")
                mode = str(attrs.get("mode") or "").strip()
                hw_fl = attrs.get("hw_fl")
                if int(scale or 0) == 2 and mode == "bilinear" and int(hw_fl or 0) == 7:
                    src_name = str(op0_inputs[0])
                    out_name2 = str(op0_out)
                    src_tt = (intent.tensors or {}).get(src_name)
                    out_tt2 = (intent.tensors or {}).get(out_name2)
                    if src_tt is not None and out_tt2 is not None:
                        src_shape = list(getattr(src_tt, "shape", []) or [])
                        out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                        if len(src_shape) == 3 and len(out_shape2) == 3:
                            c_dim = _resolve_dim_int(src_shape[0], bindings)
                            h_dim = _resolve_dim_int(src_shape[1], bindings)
                            w_dim = _resolve_dim_int(src_shape[2], bindings)
                            oh_dim = _resolve_dim_int(out_shape2[1], bindings)
                            ow_dim = _resolve_dim_int(out_shape2[2], bindings)
                            if (
                                c_dim is not None
                                and h_dim is not None
                                and w_dim is not None
                                and oh_dim is not None
                                and ow_dim is not None
                                and int(c_dim) > 0
                                and int(h_dim) > 0
                                and int(w_dim) > 0
                                and int(oh_dim) == 2 * int(h_dim)
                                and int(ow_dim) == 2 * int(w_dim)
                            ):
                                resize_v1 = {
                                    "src": str(src_name),
                                    "out": str(out_name2),
                                    "C": int(c_dim),
                                    "H": int(h_dim),
                                    "W": int(w_dim),
                                    "OH": int(oh_dim),
                                    "OW": int(ow_dim),
                                    "hw_fl": 7,
                                }
            if op0_name == "correlation" and len(op0_inputs) == 3 and op0_out:
                src0_name = str(op0_inputs[0])
                src1_name = str(op0_inputs[1])
                shift_name = str(op0_inputs[2])
                out_name2 = str(op0_out)
                src0_tt = (intent.tensors or {}).get(src0_name)
                src1_tt = (intent.tensors or {}).get(src1_name)
                shift_tt = (intent.tensors or {}).get(shift_name)
                out_tt2 = (intent.tensors or {}).get(out_name2)
                if src0_tt is not None and src1_tt is not None and shift_tt is not None and out_tt2 is not None:
                    src0_shape = list(getattr(src0_tt, "shape", []) or [])
                    src1_shape = list(getattr(src1_tt, "shape", []) or [])
                    out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                    if len(src0_shape) == 3 and len(src1_shape) == 3 and len(out_shape2) == 3:
                        ic_dim = _resolve_dim_int(src0_shape[0], bindings)
                        h_dim = _resolve_dim_int(src0_shape[1], bindings)
                        w_dim = _resolve_dim_int(src0_shape[2], bindings)
                        oc_dim = _resolve_dim_int(out_shape2[0], bindings)
                        oh_dim = _resolve_dim_int(out_shape2[1], bindings)
                        ow_dim = _resolve_dim_int(out_shape2[2], bindings)
                        if (
                            ic_dim is not None
                            and h_dim is not None
                            and w_dim is not None
                            and oc_dim is not None
                            and oh_dim is not None
                            and ow_dim is not None
                            and int(ic_dim) > 0
                            and int(h_dim) > 0
                            and int(w_dim) > 0
                            and int(oc_dim) > 0
                            and int(oh_dim) == int(h_dim)
                            and int(ow_dim) == int(w_dim)
                            and list(src1_shape) == list(src0_shape)
                        ):
                            correlation_v1 = {
                                "src0": str(src0_name),
                                "src1": str(src1_name),
                                "out": str(out_name2),
                                "out_shift": str(shift_name),
                                "IC": int(ic_dim),
                                "OC": int(oc_dim),
                                "H": int(h_dim),
                                "W": int(w_dim),
                            }
            if op0_name == "dropout" and len(op0_inputs) == 3 and op0_out:
                x_name = str(op0_inputs[0])
                p_name = str(op0_inputs[1])
                seed_name = str(op0_inputs[2])
                out_name2 = str(op0_out)
                x_tt = (intent.tensors or {}).get(x_name)
                p_tt = (intent.tensors or {}).get(p_name)
                seed_tt = (intent.tensors or {}).get(seed_name)
                out_tt2 = (intent.tensors or {}).get(out_name2)
                if x_tt is not None and p_tt is not None and seed_tt is not None and out_tt2 is not None:
                    x_shape = list(getattr(x_tt, "shape", []) or [])
                    out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                    if len(x_shape) == 1 and len(out_shape2) == 1:
                        n_dim = _resolve_dim_int(x_shape[0], bindings)
                        n2_dim = _resolve_dim_int(out_shape2[0], bindings)
                        if n_dim is not None and n2_dim is not None and int(n_dim) > 0 and int(n2_dim) == int(n_dim):
                            dropout_v1 = {
                                "x": str(x_name),
                                "p": str(p_name),
                                "seed": str(seed_name),
                                "out": str(out_name2),
                                "N": int(n_dim),
                                "n_rounds": 10,
                            }

        # Wave16 (coverage_batches backfill): scan + kron (static outputs).
        if kron2d_v1 is None and intent_name == "kron2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "kron" and len(op0_inputs) == 2 and op0_out == str(out_name):
                    a_name, b_name = str(op0_inputs[0]), str(op0_inputs[1])
                    required = {a_name, b_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        a_dims = list(arg_specs[a_name].get("dims") or [])
                        b_dims = list(arg_specs[b_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(a_dims) == 2 and len(b_dims) == 2 and len(out_dims2) == 2 and out_rank == 2:
                            m_dim, n_dim = map(int, a_dims)
                            p_dim, q_dim = map(int, b_dims)
                            mp_dim, nq_dim = map(int, out_dims2)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and p_dim > 0
                                and q_dim > 0
                                and mp_dim == m_dim * p_dim
                                and nq_dim == n_dim * q_dim
                                and str(arg_specs[a_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[b_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                kron2d_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": op0_out,
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "P": int(p_dim),
                                    "Q": int(q_dim),
                                    "MP": int(mp_dim),
                                    "NQ": int(nq_dim),
                                }

        if cumsum2d_v1 is None and intent_name == "cumsum2d":
            required = {"inp", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                in_dims = list(arg_specs["inp"].get("dims") or [])
                out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                if len(in_dims) == 2 and out_dims2 == in_dims and out_rank == 2:
                    m_dim, n_dim = map(int, in_dims)
                    if (
                        m_dim > 0
                        and n_dim > 0
                        and str(arg_specs["inp"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                    ):
                        axis_ok = False
                        for op in ops_list:
                            if str(getattr(op, "op", "")).strip() != "cumsum":
                                continue
                            attrs = dict(getattr(op, "attrs", {}) or {})
                            axis = attrs.get("axis")
                            try:
                                axis_i = int(axis) if axis is not None else None
                            except Exception:
                                axis_i = None
                            axis_ok = axis_i == 1
                        if axis_ok:
                            cumsum2d_v1 = {"inp": "inp", "out": str(out_name), "M": int(m_dim), "N": int(n_dim)}

        if normed_cumsum2d_v1 is None and intent_name == "normed_cumsum2d":
            required = {"inp", "EPS", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                in_dims = list(arg_specs["inp"].get("dims") or [])
                out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                if len(in_dims) == 2 and out_dims2 == in_dims and out_rank == 2:
                    m_dim, n_dim = map(int, in_dims)
                    if (
                        m_dim > 0
                        and n_dim > 0
                        and str(arg_specs["inp"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["EPS"].get("memref_elem_ty") or "") == "f32"
                        and bool(arg_specs["EPS"].get("scalar"))
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "f32"
                    ):
                        cumsum_out = None
                        denom_out = None
                        add_out = None
                        bc_out = None
                        for op in ops_list:
                            name = str(getattr(op, "op", "")).strip()
                            out2 = str(getattr(op, "output", "")).strip()
                            ins = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
                            attrs = dict(getattr(op, "attrs", {}) or {})
                            if name == "cumsum" and len(ins) == 1 and ins[0] == "inp":
                                axis = attrs.get("axis")
                                try:
                                    axis_i = int(axis) if axis is not None else None
                                except Exception:
                                    axis_i = None
                                if axis_i == 1:
                                    cumsum_out = str(out2)
                            elif name == "reduce_sum" and len(ins) == 1 and ins[0] == "inp":
                                dims = attrs.get("dims")
                                dims_list = list(dims) if isinstance(dims, list) else []
                                keepdims = bool(attrs.get("keepdims", False))
                                if dims_list == [1] and keepdims:
                                    denom_out = str(out2)
                            elif name == "add" and len(ins) == 2 and "EPS" in set(ins):
                                other = [x for x in ins if x != "EPS"]
                                if len(other) == 1 and denom_out and other[0] == denom_out:
                                    add_out = str(out2)
                            elif name == "broadcast_in_dim" and len(ins) == 1:
                                bcast_dims = attrs.get("broadcast_dims")
                                out_shape = attrs.get("out_shape")
                                bcast_dims_list = list(bcast_dims) if isinstance(bcast_dims, list) else []
                                out_shape_list = list(out_shape) if isinstance(out_shape, list) else []
                                if add_out and ins[0] == add_out and bcast_dims_list == [0, 1] and out_shape_list == ["M", "N"]:
                                    bc_out = str(out2)
                        # backend_legalize may remove explicit broadcast_in_dim and rely on implicit
                        # broadcasting for div([M,N],[M,1]). Accept both forms.
                        if cumsum_out and (bc_out or add_out):
                            for op in ops_list:
                                name = str(getattr(op, "op", "")).strip()
                                out2 = str(getattr(op, "output", "")).strip()
                                ins = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
                                if name == "div" and len(ins) == 2 and out2 == str(out_name):
                                    in_set = set(map(str, ins))
                                    ok_bc = bool(bc_out) and in_set == {str(cumsum_out), str(bc_out)}
                                    ok_implicit = bool(add_out) and in_set == {str(cumsum_out), str(add_out)}
                                    if ok_bc or ok_implicit:
                                        normed_cumsum2d_v1 = {
                                            "inp": "inp",
                                            "eps": "EPS",
                                            "out": str(out_name),
                                            "M": int(m_dim),
                                            "N": int(n_dim),
                                        }

        if cummax1d_v1 is None and intent_name == "cummax1d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs.get("axis")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                if op0_name == "cummax" and len(op0_inputs) == 1 and op0_out == str(out_name) and axis_i == 0:
                    x_name = str(op0_inputs[0])
                    required = {x_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        x_dims = list(arg_specs[x_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(x_dims) == 1 and out_dims2 == x_dims and out_rank == 1:
                            n_dim = int(x_dims[0])
                            ok = (
                                n_dim > 0
                                and str(arg_specs[x_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                cummax1d_v1 = {"x": x_name, "out": op0_out, "N": int(n_dim)}

        if cummin1d_v1 is None and intent_name == "cummin1d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs.get("axis")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                if op0_name == "cummin" and len(op0_inputs) == 1 and op0_out == str(out_name) and axis_i == 0:
                    x_name = str(op0_inputs[0])
                    required = {x_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        x_dims = list(arg_specs[x_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(x_dims) == 1 and out_dims2 == x_dims and out_rank == 1:
                            n_dim = int(x_dims[0])
                            ok = (
                                n_dim > 0
                                and str(arg_specs[x_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                cummin1d_v1 = {"x": x_name, "out": op0_out, "N": int(n_dim)}

        # Wave17 (coverage_batches backfill): sort/topk/quantile (static rank-2 outputs).
        if intent_name in {"sort2d", "sort_stable2d"} and row_sort_axis1_bitonic_v1 is None:
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs.get("axis")
                descending = attrs.get("descending")
                stable = attrs.get("stable")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                desc_b = bool(descending) if descending is not None else False
                stable_b = bool(stable) if stable is not None else False
                if intent_name == "sort2d" and stable_b:
                    stable_b = False
                if intent_name == "sort_stable2d" and not stable_b:
                    stable_b = True
                # Stable+descending is not supported in v1 (would require stable reverse).
                if stable_b and desc_b:
                    pass
                elif op0_name == "sort" and axis_i == 1 and len(op0_inputs) == 1 and op0_out == str(out_name):
                    in_name = str(op0_inputs[0])
                    required = {in_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[in_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and out_dims2 == in_dims and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and n_dim <= 256
                                and str(arg_specs[in_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                row_sort_axis1_bitonic_v1 = {
                                    "inp": str(in_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "descending": bool(desc_b),
                                    "stable": bool(stable_b),
                                }

        if intent_name == "topk2d" and row_topk_axis1_bitonic_v1 is None:
            # Canonical graph: sort(axis=1, descending=true) -> iota -> iota -> gather.
            if len(ops_list) == 4:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs0.get("axis")
                descending = attrs0.get("descending")
                stable = attrs0.get("stable")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                desc_b = bool(descending) if descending is not None else False
                stable_b = bool(stable) if stable is not None else False
                if op0_name == "sort" and axis_i == 1 and desc_b and not stable_b and len(op0_inputs) == 1:
                    in_name = str(op0_inputs[0])
                    out_name2 = str(out_name)
                    required = {in_name, out_name2}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[in_name].get("dims") or [])
                        out_dims2 = list(arg_specs[out_name2].get("dims") or [])
                        if len(in_dims) == 2 and len(out_dims2) == 2 and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            m2_dim, k_dim = map(int, out_dims2)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and m2_dim == m_dim
                                and k_dim > 0
                                and k_dim <= n_dim
                                and n_dim <= 256
                                and str(arg_specs[in_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[out_name2].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                row_topk_axis1_bitonic_v1 = {
                                    "inp": str(in_name),
                                    "out": str(out_name2),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "K": int(k_dim),
                                }

        if intent_name == "quantile2d" and row_quantile_axis1_sort_v1 is None:
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                dim = attrs0.get("dim")
                keepdim = attrs0.get("keepdim")
                interpolation = str(attrs0.get("interpolation") or "").strip().lower()
                try:
                    dim_i = int(dim) if dim is not None else None
                except Exception:
                    dim_i = None
                keepdim_b = bool(keepdim) if keepdim is not None else False
                if (
                    op0_name == "quantile"
                    and len(op0_inputs) == 2
                    and op0_out == str(out_name)
                    and dim_i == 1
                    and not keepdim_b
                    and interpolation in {"", "linear"}
                ):
                    in_name = str(op0_inputs[0])
                    q_name = str(op0_inputs[1])
                    required = {in_name, q_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[in_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and len(out_dims2) == 1 and out_rank == 1:
                            m_dim, n_dim = map(int, in_dims)
                            m2_dim = int(out_dims2[0])
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and m2_dim == m_dim
                                and n_dim <= 256
                                and str(arg_specs[in_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[q_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                row_quantile_axis1_sort_v1 = {
                                    "inp": str(in_name),
                                    "q": str(q_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                }

        # Wave18 (coverage_batches backfill): NCHW pooling (static output shapes).
        if avg_pool2d_nchw_v1 is None and intent_name == "avg_pool2d_nchw":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs = dict(getattr(op0, "attrs", {}) or {})
                kernel_size = attrs.get("kernel_size")
                stride = attrs.get("stride")
                padding = attrs.get("padding")
                ceil_mode = bool(attrs.get("ceil_mode")) if attrs.get("ceil_mode") is not None else False
                kernel_size_list = list(kernel_size) if isinstance(kernel_size, list) else []
                stride_list = list(stride) if isinstance(stride, list) else []
                padding_list = list(padding) if isinstance(padding, list) else []
                if (
                    op0_name == "avg_pool2d"
                    and len(op0_inputs) == 1
                    and op0_out == str(out_name)
                    and kernel_size_list == [2, 2]
                    and stride_list == [2, 2]
                    and padding_list == [0, 0]
                    and not ceil_mode
                ):
                    in_name = str(op0_inputs[0])
                    required = {in_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[in_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 4 and len(out_dims2) == 4 and out_rank == 4:
                            n_dim, c_dim, h_dim, w_dim = map(int, in_dims)
                            n2_dim, c2_dim, oh_dim, ow_dim = map(int, out_dims2)
                            ok = (
                                n_dim > 0
                                and c_dim > 0
                                and h_dim > 0
                                and w_dim > 0
                                and oh_dim > 0
                                and ow_dim > 0
                                and n2_dim == n_dim
                                and c2_dim == c_dim
                                and str(arg_specs[in_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                avg_pool2d_nchw_v1 = {
                                    "inp": str(in_name),
                                    "out": str(op0_out),
                                    "N": int(n_dim),
                                    "C": int(c_dim),
                                    "H": int(h_dim),
                                    "W": int(w_dim),
                                    "OH": int(oh_dim),
                                    "OW": int(ow_dim),
                                    "KH": 2,
                                    "KW": 2,
                                    "SH": 2,
                                    "SW": 2,
                                    "PH": 0,
                                    "PW": 0,
                                }

        if max_pool2d_with_indices_nchw_v1 is None and intent_name == "max_pool2d_with_indices_nchw":
            # Canonical graph: two ops with select=values/indices producing two outputs.
            if len(ops_list) == 2:
                in_name = ""
                out_values = ""
                out_indices = ""
                attrs0 = None
                ok_graph = True
                for op in ops_list:
                    name = str(getattr(op, "op", "")).strip()
                    ins = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
                    out2 = str(getattr(op, "output", "")).strip()
                    attrs = dict(getattr(op, "attrs", {}) or {})
                    if name != "max_pool2d_with_indices" or len(ins) != 1 or not out2:
                        ok_graph = False
                        break
                    if not in_name:
                        in_name = str(ins[0])
                    if str(ins[0]) != str(in_name):
                        ok_graph = False
                        break
                    select = str(attrs.get("select") or "").strip().lower()
                    if select == "values":
                        out_values = str(out2)
                    elif select == "indices":
                        out_indices = str(out2)
                    else:
                        ok_graph = False
                        break
                    if attrs0 is None:
                        attrs0 = dict(attrs)
                if ok_graph and in_name and out_values and out_indices and isinstance(attrs0, dict):
                    kernel_size = attrs0.get("kernel_size")
                    stride = attrs0.get("stride")
                    padding = attrs0.get("padding")
                    dilation = attrs0.get("dilation")
                    ceil_mode = bool(attrs0.get("ceil_mode")) if attrs0.get("ceil_mode") is not None else False
                    kernel_size_list = list(kernel_size) if isinstance(kernel_size, list) else []
                    stride_list = list(stride) if isinstance(stride, list) else []
                    padding_list = list(padding) if isinstance(padding, list) else []
                    dilation_list = list(dilation) if isinstance(dilation, list) else []
                    if (
                        kernel_size_list == [2, 2]
                        and stride_list == [2, 2]
                        and padding_list == [0, 0]
                        and dilation_list == [1, 1]
                        and not ceil_mode
                    ):
                        required = {in_name, out_values, out_indices}
                        if required.issubset(set(arg_specs.keys())):
                            in_dims = list(arg_specs[in_name].get("dims") or [])
                            out_dims2 = list(arg_specs[out_values].get("dims") or [])
                            idx_dims = list(arg_specs[out_indices].get("dims") or [])
                            if len(in_dims) == 4 and len(out_dims2) == 4 and out_dims2 == idx_dims and out_rank == 4:
                                n_dim, c_dim, h_dim, w_dim = map(int, in_dims)
                                n2_dim, c2_dim, oh_dim, ow_dim = map(int, out_dims2)
                                ok = (
                                    n_dim > 0
                                    and c_dim > 0
                                    and h_dim > 0
                                    and w_dim > 0
                                    and oh_dim > 0
                                    and ow_dim > 0
                                    and n2_dim == n_dim
                                    and c2_dim == c_dim
                                    and str(arg_specs[in_name].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[out_values].get("memref_elem_ty") or "") == "f32"
                                    and str(arg_specs[out_indices].get("memref_elem_ty") or "") == "i64"
                                )
                                if ok:
                                    max_pool2d_with_indices_nchw_v1 = {
                                        "inp": str(in_name),
                                        "out": str(out_values),
                                        "indices": str(out_indices),
                                        "N": int(n_dim),
                                        "C": int(c_dim),
                                        "H": int(h_dim),
                                        "W": int(w_dim),
                                        "OH": int(oh_dim),
                                        "OW": int(ow_dim),
                                        "KH": 2,
                                        "KW": 2,
                                        "SH": 2,
                                        "SW": 2,
                                        "PH": 0,
                                        "PW": 0,
                                        "DH": 1,
                                        "DW": 1,
                                    }

        # Wave19 (coverage_batches backfill): index/scatter family (row-parallel).
        if index_add2d_axis0_v1 is None and intent_name == "index_add2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                axis = attrs0.get("axis")
                alpha = attrs0.get("alpha")
                try:
                    axis_i = int(axis) if axis is not None else None
                except Exception:
                    axis_i = None
                try:
                    alpha_f = float(alpha) if alpha is not None else 1.0
                except Exception:
                    alpha_f = 1.0
                if op0_name == "index_add" and len(op0_inputs) == 3 and op0_out == str(out_name) and axis_i == 0:
                    base_name, index_name, src_name = map(str, op0_inputs)
                    required = {base_name, index_name, src_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        base_dims = list(arg_specs[base_name].get("dims") or [])
                        idx_dims = list(arg_specs[index_name].get("dims") or [])
                        src_dims = list(arg_specs[src_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(base_dims) == 2 and len(out_dims2) == 2 and len(idx_dims) == 1 and len(src_dims) == 2 and out_rank == 2:
                            m_dim, n_dim = map(int, base_dims)
                            l_dim = int(idx_dims[0])
                            l2_dim, n2_dim = map(int, src_dims)
                            m_out, n_out = map(int, out_dims2)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and l_dim > 0
                                and l2_dim == l_dim
                                and n2_dim == n_dim
                                and m_out == m_dim
                                and n_out == n_dim
                                and str(arg_specs[base_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[src_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[index_name].get("memref_elem_ty") or "") == "i32"
                            )
                            if ok:
                                index_add2d_axis0_v1 = {
                                    "base": str(base_name),
                                    "index": str(index_name),
                                    "src": str(src_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "L": int(l_dim),
                                    "alpha": float(alpha_f),
                                }

        if index_put2d_v1 is None and intent_name == "index_put2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                accumulate = bool(attrs0.get("accumulate")) if attrs0.get("accumulate") is not None else False
                if op0_name == "index_put" and len(op0_inputs) == 4 and op0_out == str(out_name) and not accumulate:
                    base_name, row_name, col_name, val_name = map(str, op0_inputs)
                    required = {base_name, row_name, col_name, val_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        base_dims = list(arg_specs[base_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        row_dims = list(arg_specs[row_name].get("dims") or [])
                        col_dims = list(arg_specs[col_name].get("dims") or [])
                        val_dims = list(arg_specs[val_name].get("dims") or [])
                        if (
                            len(base_dims) == 2
                            and len(out_dims2) == 2
                            and len(row_dims) == 1
                            and row_dims == col_dims
                            and row_dims == val_dims
                            and out_rank == 2
                        ):
                            m_dim, n_dim = map(int, base_dims)
                            m_out, n_out = map(int, out_dims2)
                            l_dim = int(row_dims[0])
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and l_dim > 0
                                and m_out == m_dim
                                and n_out == n_dim
                                and str(arg_specs[base_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[val_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[row_name].get("memref_elem_ty") or "") == "i32"
                                and str(arg_specs[col_name].get("memref_elem_ty") or "") == "i32"
                            )
                            if ok:
                                index_put2d_v1 = {
                                    "base": str(base_name),
                                    "row_idx": str(row_name),
                                    "col_idx": str(col_name),
                                    "values": str(val_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "L": int(l_dim),
                                }

        if scatter2d_dim1_v1 is None and intent_name == "scatter2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                dim = attrs0.get("dim")
                try:
                    dim_i = int(dim) if dim is not None else None
                except Exception:
                    dim_i = None
                if op0_name == "scatter" and len(op0_inputs) == 3 and op0_out == str(out_name) and dim_i == 1:
                    inp_name, idx_name, src_name = map(str, op0_inputs)
                    required = {inp_name, idx_name, src_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        idx_dims = list(arg_specs[idx_name].get("dims") or [])
                        src_dims = list(arg_specs[src_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and in_dims == idx_dims and in_dims == src_dims and in_dims == out_dims2 and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[src_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[idx_name].get("memref_elem_ty") or "") == "i32"
                            )
                            if ok:
                                scatter2d_dim1_v1 = {
                                    "inp": str(inp_name),
                                    "index": str(idx_name),
                                    "src": str(src_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                }

        if select_scatter2d_dim1_v1 is None and intent_name == "select_scatter2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                dim = attrs0.get("dim")
                index = attrs0.get("index")
                try:
                    dim_i = int(dim) if dim is not None else None
                except Exception:
                    dim_i = None
                try:
                    idx_i = int(index) if index is not None else None
                except Exception:
                    idx_i = None
                if op0_name == "select_scatter" and len(op0_inputs) == 2 and op0_out == str(out_name) and dim_i == 1 and idx_i is not None:
                    inp_name, src_name = map(str, op0_inputs)
                    required = {inp_name, src_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        src_dims = list(arg_specs[src_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and len(out_dims2) == 2 and len(src_dims) == 1 and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            m2_dim, n2_dim = map(int, out_dims2)
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and m2_dim == m_dim
                                and n2_dim == n_dim
                                and int(src_dims[0]) == m_dim
                                and 0 <= int(idx_i) < n_dim
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[src_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                select_scatter2d_dim1_v1 = {
                                    "inp": str(inp_name),
                                    "src": str(src_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "index": int(idx_i),
                                }

        if slice_scatter2d_dim1_v1 is None and intent_name == "slice_scatter2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                attrs0 = dict(getattr(op0, "attrs", {}) or {})
                dim = attrs0.get("dim")
                start = attrs0.get("start")
                end = attrs0.get("end")
                step = attrs0.get("step")
                try:
                    dim_i = int(dim) if dim is not None else None
                except Exception:
                    dim_i = None
                try:
                    start_i = int(start) if start is not None else None
                except Exception:
                    start_i = None
                try:
                    end_i = int(end) if end is not None else None
                except Exception:
                    end_i = None
                try:
                    step_i = int(step) if step is not None else 1
                except Exception:
                    step_i = 1
                if (
                    op0_name == "slice_scatter"
                    and len(op0_inputs) == 2
                    and op0_out == str(out_name)
                    and dim_i == 1
                    and start_i is not None
                    and end_i is not None
                    and step_i > 0
                ):
                    inp_name, src_name = map(str, op0_inputs)
                    required = {inp_name, src_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        src_dims = list(arg_specs[src_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and len(out_dims2) == 2 and len(src_dims) == 2 and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            m2_dim, n2_dim = map(int, out_dims2)
                            m_src, l_src = map(int, src_dims)
                            slice_len = len(list(range(int(start_i), int(end_i), int(step_i))))
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and m2_dim == m_dim
                                and n2_dim == n_dim
                                and m_src == m_dim
                                and slice_len == l_src
                                and 0 <= int(start_i) <= int(end_i) <= n_dim
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[src_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                            )
                            if ok:
                                slice_scatter2d_dim1_v1 = {
                                    "inp": str(inp_name),
                                    "src": str(src_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "start": int(start_i),
                                    "end": int(end_i),
                                    "step": int(step_i),
                                    "L": int(l_src),
                                }

        # Wave20 (coverage_batches backfill): masked_select/masked_scatter (single-CTA prefix-sum).
        if masked_select2d_v1 is None and intent_name == "masked_select2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "masked_select" and len(op0_inputs) == 2 and op0_out == str(out_name):
                    inp_name, mask_name = map(str, op0_inputs)
                    required = {inp_name, mask_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        mask_dims = list(arg_specs[mask_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and in_dims == mask_dims and len(out_dims2) == 1 and out_rank == 1:
                            m_dim, n_dim = map(int, in_dims)
                            l_dim = int(out_dims2[0])
                            t_dim = int(m_dim * n_dim)
                            block_threads = 1 << (int(t_dim - 1).bit_length()) if t_dim > 0 else 1
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and l_dim > 0
                                and t_dim > 0
                                and l_dim <= t_dim
                                and t_dim <= 1024
                                and block_threads <= 1024
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[mask_name].get("memref_elem_ty") or "") in {"i1", "i8", "i32"}
                            )
                            if ok:
                                masked_select2d_v1 = {
                                    "inp": str(inp_name),
                                    "mask": str(mask_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "L": int(l_dim),
                                    "T": int(t_dim),
                                    "block_threads": int(block_threads),
                                }

        if masked_scatter2d_v1 is None and intent_name == "masked_scatter2d":
            if len(ops_list) == 1:
                op0 = ops_list[0]
                op0_name = str(getattr(op0, "op", "")).strip()
                op0_inputs = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                op0_out = str(getattr(op0, "output", "")).strip()
                if op0_name == "masked_scatter" and len(op0_inputs) == 3 and op0_out == str(out_name):
                    inp_name, mask_name, src_name = map(str, op0_inputs)
                    required = {inp_name, mask_name, src_name, op0_out}
                    if required.issubset(set(arg_specs.keys())):
                        in_dims = list(arg_specs[inp_name].get("dims") or [])
                        mask_dims = list(arg_specs[mask_name].get("dims") or [])
                        src_dims = list(arg_specs[src_name].get("dims") or [])
                        out_dims2 = list(arg_specs[op0_out].get("dims") or [])
                        if len(in_dims) == 2 and in_dims == mask_dims and len(src_dims) == 1 and len(out_dims2) == 2 and out_rank == 2:
                            m_dim, n_dim = map(int, in_dims)
                            l_dim = int(src_dims[0])
                            m_out, n_out = map(int, out_dims2)
                            t_dim = int(m_dim * n_dim)
                            block_threads = 1 << (int(t_dim - 1).bit_length()) if t_dim > 0 else 1
                            ok = (
                                m_dim > 0
                                and n_dim > 0
                                and l_dim > 0
                                and t_dim > 0
                                and l_dim <= t_dim
                                and m_out == m_dim
                                and n_out == n_dim
                                and t_dim <= 1024
                                and block_threads <= 1024
                                and str(arg_specs[inp_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[src_name].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[op0_out].get("memref_elem_ty") or "") == "f32"
                                and str(arg_specs[mask_name].get("memref_elem_ty") or "") in {"i1", "i8", "i32"}
                            )
                            if ok:
                                masked_scatter2d_v1 = {
                                    "inp": str(inp_name),
                                    "mask": str(mask_name),
                                    "src": str(src_name),
                                    "out": str(op0_out),
                                    "M": int(m_dim),
                                    "N": int(n_dim),
                                    "L": int(l_dim),
                                    "T": int(t_dim),
                                    "block_threads": int(block_threads),
                                }

        # Pattern: matmul family (Triton-native perf/coverage kernels).
        #
        # These are currently lowered as a single CTA (256 threads) per output tile,
        # with cooperative shared-memory loads and a fused epilogue (bias/relu/masks).
        if intent_name in {
            "ai_bench_matmul",
            "mm2d",
            "addmm2d",
            "bmm3d",
            "baddbmm3d",
            "mv2d",
            "matmul_relu2d",
            "matmul_bias_relu2d",
            "matmul_fused_epilogue2d",
        }:
            op_names = [str(getattr(op, "op", "")).strip() for op in ops_list]

            def _op_inputs(op: Any) -> list[str]:
                return [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]

            def _op_out(op: Any) -> str:
                return str(getattr(op, "output", "") or "").strip()

            def _dims(name: str) -> list[int] | None:
                spec = arg_specs.get(str(name))
                if not isinstance(spec, dict):
                    return None
                dims = list(spec.get("dims") or [])
                return [int(x) for x in dims] if dims else []

            def _elem(name: str) -> str:
                return str((arg_specs.get(str(name)) or {}).get("memref_elem_ty") or "").strip()

            if matmul_v1 is None and intent_name == "ai_bench_matmul" and op_names == ["matmul"]:
                op0 = ops_list[0]
                ins = _op_inputs(op0)
                out0 = _op_out(op0)
                if len(ins) == 2 and out0 == str(out_name):
                    a_name, b_name = str(ins[0]), str(ins[1])
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    if a_dims and b_dims and c_dims and len(a_dims) == 2 and len(b_dims) == 2 and len(c_dims) == 2:
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if k2 == k and int(c_dims[0]) == m and int(c_dims[1]) == n:
                            if _elem(a_name) == "f32" and _elem(b_name) == "f32" and _elem(str(out_name)) == "f32":
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 64,
                                    "BN": 16,
                                    "BK": 16,
                                    "relu": False,
                                    "bias": None,
                                    "row_mask": None,
                                    "col_mask": None,
                                }

            if matmul_v1 is None and intent_name == "mm2d" and op_names == ["matmul"]:
                op0 = ops_list[0]
                ins = _op_inputs(op0)
                out0 = _op_out(op0)
                if len(ins) == 2 and out0 == str(out_name):
                    a_name, b_name = str(ins[0]), str(ins[1])
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    if a_dims and b_dims and c_dims and len(a_dims) == 2 and len(b_dims) == 2 and len(c_dims) == 2:
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if k2 == k and int(c_dims[0]) == m and int(c_dims[1]) == n:
                            if _elem(a_name) == "f32" and _elem(b_name) == "f32" and _elem(str(out_name)) == "f32":
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 64,
                                    "BN": 16,
                                    "BK": 16,
                                    "relu": False,
                                    "bias": None,
                                    "row_mask": None,
                                    "col_mask": None,
                                }

            if matmul_v1 is None and intent_name == "bmm3d" and op_names == ["matmul"]:
                op0 = ops_list[0]
                ins = _op_inputs(op0)
                out0 = _op_out(op0)
                if len(ins) == 2 and out0 == str(out_name):
                    a_name, b_name = str(ins[0]), str(ins[1])
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    if (
                        a_dims
                        and b_dims
                        and c_dims
                        and len(a_dims) == 3
                        and len(b_dims) == 3
                        and len(c_dims) == 3
                        and _elem(a_name) == "f32"
                        and _elem(b_name) == "f32"
                        and _elem(str(out_name)) == "f32"
                    ):
                        batch, m, k = (int(a_dims[0]), int(a_dims[1]), int(a_dims[2]))
                        batch2, k2, n = (int(b_dims[0]), int(b_dims[1]), int(b_dims[2]))
                        if (
                            batch > 0
                            and batch2 == batch
                            and m > 0
                            and n > 0
                            and k > 0
                            and k2 == k
                            and int(c_dims[0]) == int(batch)
                            and int(c_dims[1]) == int(m)
                            and int(c_dims[2]) == int(n)
                        ):
                            matmul_v1 = {
                                "A": a_name,
                                "B": b_name,
                                "out": str(out_name),
                                "BATCH": int(batch),
                                "M": int(m),
                                "N": int(n),
                                "K": int(k),
                                "BM": 64,
                                "BN": 16,
                                "BK": 16,
                                "relu": False,
                                "bias": None,
                                "row_mask": None,
                                "col_mask": None,
                            }

            if matmul_v1 is None and intent_name == "addmm2d" and tuple(op_names) in {
                ("matmul", "mul", "mul", "add"),
                ("matmul", "mul", "mul", "add", "cast"),
            }:
                cast_op = ops_list[-1] if op_names[-1] == "cast" else None
                if cast_op is not None:
                    matmul_op, mul_a, mul_b, add_op, cast_op = ops_list
                else:
                    matmul_op, mul_a, mul_b, add_op = ops_list

                ins0 = _op_inputs(matmul_op)
                out0 = _op_out(matmul_op)
                ins1 = _op_inputs(mul_a)
                out1 = _op_out(mul_a)
                ins2 = _op_inputs(mul_b)
                out2 = _op_out(mul_b)
                ins3 = _op_inputs(add_op)
                out3 = _op_out(add_op)

                if cast_op is not None:
                    ins4 = _op_inputs(cast_op)
                    out4 = _op_out(cast_op)
                    attrs4 = dict(getattr(cast_op, "attrs", {}) or {})
                    to4 = str(attrs4.get("to") or "f32").strip().lower()
                    out_ok = (
                        len(ins4) == 1
                        and out4 == str(out_name)
                        and str(ins4[0]) == str(out3)
                        and to4 in {"f32", "float32"}
                    )
                else:
                    out_ok = (out3 == str(out_name))

                if (
                    len(ins0) == 2
                    and len(ins1) == 2
                    and len(ins2) == 2
                    and len(ins3) == 2
                    and out_ok
                ):
                    # scaled_mm = mul(mm_out, alpha) (alpha is scalar)
                    # Identify alpha/beta via scalar-ness in arg_specs.
                    alpha_name = None
                    mm_out_name = str(out0)
                    for cand in ins1:
                        spec = arg_specs.get(str(cand))
                        if isinstance(spec, dict) and bool(spec.get("scalar")) and str(spec.get("memref_elem_ty") or "") == "f32":
                            alpha_name = str(cand)
                    if alpha_name and set(map(str, ins1)) == {mm_out_name, str(alpha_name)} and out1:
                        # scaled_bias = mul(input, beta) (beta scalar)
                        beta_name = None
                        input_name = None
                        for cand in ins2:
                            spec = arg_specs.get(str(cand))
                            if isinstance(spec, dict) and bool(spec.get("scalar")) and str(spec.get("memref_elem_ty") or "") == "f32":
                                beta_name = str(cand)
                            elif isinstance(spec, dict) and not bool(spec.get("scalar")) and str(spec.get("memref_elem_ty") or "") == "f32":
                                input_name = str(cand)
                        if beta_name and input_name and set(map(str, ins2)) == {str(input_name), str(beta_name)} and out2:
                            # add_out = add(scaled_mm, scaled_bias)
                            if set(map(str, ins3)) == {str(out1), str(out2)}:
                                mat1_name, mat2_name = str(ins0[0]), str(ins0[1])
                                a_dims = _dims(mat1_name)
                                b_dims = _dims(mat2_name)
                                c_dims = _dims(str(out_name))
                                input_dims = _dims(input_name)
                                if (
                                    a_dims
                                    and b_dims
                                    and c_dims
                                    and input_dims
                                    and len(a_dims) == 2
                                    and len(b_dims) == 2
                                    and len(c_dims) == 2
                                    and len(input_dims) == 2
                                ):
                                    m, k = int(a_dims[0]), int(a_dims[1])
                                    k2, n = int(b_dims[0]), int(b_dims[1])
                                    if (
                                        k2 == k
                                        and int(c_dims[0]) == m
                                        and int(c_dims[1]) == n
                                        and int(input_dims[0]) == m
                                        and int(input_dims[1]) == n
                                    ):
                                        if (
                                            _elem(mat1_name) == "f32"
                                            and _elem(mat2_name) == "f32"
                                            and _elem(input_name) == "f32"
                                            and _elem(str(out_name)) == "f32"
                                        ):
                                            matmul_v1 = {
                                                "A": mat1_name,
                                                "B": mat2_name,
                                                "out": str(out_name),
                                                "M": int(m),
                                                "N": int(n),
                                                "K": int(k),
                                                "BM": 64,
                                                "BN": 16,
                                                "BK": 16,
                                                "relu": False,
                                                "bias": None,
                                                "row_mask": None,
                                                "col_mask": None,
                                                "add_inp": str(input_name),
                                            "alpha": str(alpha_name),
                                            "beta": str(beta_name),
                                        }

            if matmul_v1 is None and intent_name == "baddbmm3d" and tuple(op_names) in {("matmul", "mul", "mul", "add")}:
                matmul_op, mul_a, mul_b, add_op = ops_list
                ins0 = _op_inputs(matmul_op)
                out0 = _op_out(matmul_op)
                ins1 = _op_inputs(mul_a)
                out1 = _op_out(mul_a)
                ins2 = _op_inputs(mul_b)
                out2 = _op_out(mul_b)
                ins3 = _op_inputs(add_op)
                out3 = _op_out(add_op)
                if len(ins0) == 2 and len(ins1) == 2 and len(ins2) == 2 and len(ins3) == 2 and out3 == str(out_name):
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    matmul_out = str(out0)

                    alpha_name = None
                    beta_name = None
                    bias_name = None
                    matmul_scaled = None
                    bias_scaled = None

                    # Identify alpha scaling of matmul output.
                    for ins, outv in ((ins1, out1), (ins2, out2)):
                        if matmul_out in ins:
                            other = [str(x) for x in ins if str(x) != matmul_out]
                            if len(other) != 1:
                                continue
                            cand_alpha = str(other[0])
                            spec = arg_specs.get(str(cand_alpha))
                            if (
                                isinstance(spec, dict)
                                and bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                alpha_name = cand_alpha
                                matmul_scaled = str(outv)

                    # Identify beta scaling of bias tensor.
                    for ins, outv in ((ins1, out1), (ins2, out2)):
                        if matmul_out in ins:
                            continue
                        scalar = None
                        tensor = None
                        for cand in ins:
                            spec = arg_specs.get(str(cand))
                            if (
                                isinstance(spec, dict)
                                and bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                scalar = str(cand)
                            elif (
                                isinstance(spec, dict)
                                and not bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                tensor = str(cand)
                        if scalar is None or tensor is None:
                            continue
                        beta_name = scalar
                        bias_name = tensor
                        bias_scaled = str(outv)

                    if (
                        alpha_name
                        and beta_name
                        and bias_name
                        and matmul_scaled
                        and bias_scaled
                        and set(map(str, ins3)) == {str(matmul_scaled), str(bias_scaled)}
                    ):
                        a_dims = _dims(a_name)
                        b_dims = _dims(b_name)
                        out_dims2 = _dims(str(out_name))
                        bias_dims = _dims(str(bias_name))
                        if (
                            a_dims
                            and b_dims
                            and out_dims2
                            and bias_dims
                            and len(a_dims) == 3
                            and len(b_dims) == 3
                            and len(out_dims2) == 3
                            and len(bias_dims) == 3
                        ):
                            batch, m, k = (int(a_dims[0]), int(a_dims[1]), int(a_dims[2]))
                            batch2, k2, n = (int(b_dims[0]), int(b_dims[1]), int(b_dims[2]))
                            if (
                                batch > 0
                                and batch2 == batch
                                and m > 0
                                and n > 0
                                and k > 0
                                and k2 == k
                                and list(out_dims2) == [int(batch), int(m), int(n)]
                                and list(bias_dims) == [int(batch), int(m), int(n)]
                                and _elem(a_name) == "f32"
                                and _elem(b_name) == "f32"
                                and _elem(str(out_name)) == "f32"
                            ):
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "BATCH": int(batch),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 64,
                                    "BN": 16,
                                    "BK": 16,
                                    "relu": False,
                                    "bias": None,
                                    "row_mask": None,
                                    "col_mask": None,
                                    "add_inp": str(bias_name),
                                    "alpha": str(alpha_name),
                                    "beta": str(beta_name),
                                }

            if matvec_v1 is None and intent_name == "mv2d" and tuple(op_names) in {
                ("matmul", "mul", "mul", "add"),
                ("matmul", "mul", "mul", "add", "cast"),
            }:
                cast_op = ops_list[-1] if op_names[-1] == "cast" else None
                if cast_op is not None:
                    matmul_op, mul_a, mul_b, add_op, cast_op = ops_list
                else:
                    matmul_op, mul_a, mul_b, add_op = ops_list

                ins0 = _op_inputs(matmul_op)
                out0 = _op_out(matmul_op)
                ins1 = _op_inputs(mul_a)
                out1 = _op_out(mul_a)
                ins2 = _op_inputs(mul_b)
                out2 = _op_out(mul_b)
                ins3 = _op_inputs(add_op)
                out3 = _op_out(add_op)

                if cast_op is not None:
                    ins4 = _op_inputs(cast_op)
                    out4 = _op_out(cast_op)
                    attrs4 = dict(getattr(cast_op, "attrs", {}) or {})
                    to4 = str(attrs4.get("to") or "f32").strip().lower()
                    out_ok = (
                        len(ins4) == 1
                        and out4 == str(out_name)
                        and str(ins4[0]) == str(out3)
                        and to4 in {"f32", "float32"}
                    )
                else:
                    out_ok = (out3 == str(out_name))

                if len(ins0) == 2 and len(ins1) == 2 and len(ins2) == 2 and len(ins3) == 2 and out_ok:
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    mv_out_name = str(out0)

                    alpha_name = None
                    beta_name = None
                    inp_name = None
                    mv_scaled_out = None
                    inp_scaled_out = None

                    # Identify scaling: alpha multiplies the matmul output (mv_out_name);
                    # beta multiplies the input vector.
                    for ins, outv in ((ins1, out1), (ins2, out2)):
                        if mv_out_name in ins:
                            other = [str(x) for x in ins if str(x) != mv_out_name]
                            if len(other) != 1:
                                continue
                            cand_alpha = str(other[0])
                            spec = arg_specs.get(str(cand_alpha))
                            if (
                                isinstance(spec, dict)
                                and bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                alpha_name = cand_alpha
                                mv_scaled_out = str(outv)
                            continue

                        scalar = None
                        tensor = None
                        for cand in ins:
                            spec = arg_specs.get(str(cand))
                            if (
                                isinstance(spec, dict)
                                and bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                scalar = str(cand)
                            elif (
                                isinstance(spec, dict)
                                and not bool(spec.get("scalar"))
                                and str(spec.get("memref_elem_ty") or "") == "f32"
                            ):
                                tensor = str(cand)
                        if scalar is None or tensor is None:
                            continue
                        beta_name = scalar
                        inp_name = tensor
                        inp_scaled_out = str(outv)

                    if (
                        alpha_name
                        and beta_name
                        and inp_name
                        and mv_scaled_out
                        and inp_scaled_out
                        and set(map(str, ins3)) == {str(mv_scaled_out), str(inp_scaled_out)}
                    ):
                        a_dims = _dims(a_name)
                        b_dims = _dims(b_name)
                        out_dims2 = _dims(str(out_name))
                        inp_dims = _dims(str(inp_name))
                        if (
                            a_dims
                            and b_dims
                            and out_dims2
                            and inp_dims
                            and len(a_dims) == 2
                            and len(b_dims) == 1
                            and len(out_dims2) == 1
                            and len(inp_dims) == 1
                        ):
                            n_dim = int(out_dims2[0])
                            m_dim = int(b_dims[0])
                            if (
                                n_dim > 0
                                and m_dim > 0
                                and int(a_dims[0]) == int(n_dim)
                                and int(a_dims[1]) == int(m_dim)
                                and int(inp_dims[0]) == int(n_dim)
                            ):
                                if (
                                    _elem(a_name) == "f32"
                                    and _elem(b_name) == "f32"
                                    and _elem(inp_name) == "f32"
                                    and _elem(str(out_name)) == "f32"
                                    and bool((arg_specs.get(str(alpha_name)) or {}).get("scalar"))
                                    and bool((arg_specs.get(str(beta_name)) or {}).get("scalar"))
                                ):
                                    matvec_v1 = {
                                        "A": str(a_name),
                                        "B": str(b_name),
                                        "mv_out": str(mv_out_name),
                                        "out": str(out_name),
                                        "Inp": str(inp_name),
                                        "alpha": str(alpha_name),
                                        "beta": str(beta_name),
                                        "N": int(n_dim),
                                        "M": int(m_dim),
                                    }

            if matmul_v1 is None and intent_name == "matmul_relu2d" and op_names == ["matmul", "relu"]:
                op0, op1 = ops_list
                ins0 = _op_inputs(op0)
                out0 = _op_out(op0)
                ins1 = _op_inputs(op1)
                out1 = _op_out(op1)
                if len(ins0) == 2 and len(ins1) == 1 and out1 == str(out_name) and str(ins1[0]) == str(out0):
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    if a_dims and b_dims and c_dims and len(a_dims) == 2 and len(b_dims) == 2 and len(c_dims) == 2:
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if k2 == k and int(c_dims[0]) == m and int(c_dims[1]) == n:
                            if _elem(a_name) == "f32" and _elem(b_name) == "f32" and _elem(str(out_name)) == "f32":
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "relu": True,
                                    "bias": None,
                                    "row_mask": None,
                                    "col_mask": None,
                                }

            if matmul_v1 is None and intent_name == "matmul_bias_relu2d" and op_names == [
                "matmul",
                "broadcast_in_dim",
                "add",
                "relu",
            ]:
                op0, op1, op2, op3 = ops_list
                ins0 = _op_inputs(op0)
                out0 = _op_out(op0)
                ins1 = _op_inputs(op1)
                out1 = _op_out(op1)
                ins2 = _op_inputs(op2)
                out2 = _op_out(op2)
                ins3 = _op_inputs(op3)
                out3 = _op_out(op3)
                attrs1 = dict(getattr(op1, "attrs", {}) or {})
                if (
                    len(ins0) == 2
                    and len(ins1) == 1
                    and len(ins2) == 2
                    and len(ins3) == 1
                    and out3 == str(out_name)
                    and str(ins3[0]) == str(out2)
                    and str(ins1[0]) == "Bias"
                    and list(attrs1.get("broadcast_dims") or []) == [1]
                    and list(attrs1.get("out_shape") or []) == ["M", "N"]
                ):
                    # add(matmul_out, bias_broadcast) or add(bias_broadcast, matmul_out)
                    if set(map(str, ins2)) == {str(out0), str(out1)}:
                        a_name, b_name = str(ins0[0]), str(ins0[1])
                        bias_name = str(ins1[0])
                        a_dims = _dims(a_name)
                        b_dims = _dims(b_name)
                        c_dims = _dims(str(out_name))
                        bias_dims = _dims(bias_name)
                        if (
                            a_dims
                            and b_dims
                            and c_dims
                            and bias_dims
                            and len(a_dims) == 2
                            and len(b_dims) == 2
                            and len(c_dims) == 2
                            and len(bias_dims) == 1
                        ):
                            m, k = int(a_dims[0]), int(a_dims[1])
                            k2, n = int(b_dims[0]), int(b_dims[1])
                            if k2 == k and int(c_dims[0]) == m and int(c_dims[1]) == n and int(bias_dims[0]) == n:
                                if (
                                    _elem(a_name) == "f32"
                                    and _elem(b_name) == "f32"
                                    and _elem(bias_name) == "f32"
                                    and _elem(str(out_name)) == "f32"
                                    ):
                                        matmul_v1 = {
                                            "A": a_name,
                                            "B": b_name,
                                        "out": str(out_name),
                                        "M": int(m),
                                        "N": int(n),
                                        "K": int(k),
                                        "BM": 32,
                                        "BN": 32,
                                        "BK": 16,
                                        "relu": True,
                                        "bias": bias_name,
                                        "row_mask": None,
                                        "col_mask": None,
                                    }

            # NOTE: backend_legalize may simplify bias broadcast into `add(matmul_out, Bias)` directly.
            if matmul_v1 is None and intent_name == "matmul_bias_relu2d" and op_names == ["matmul", "add", "relu"]:
                op0, op1, op2 = ops_list
                ins0 = _op_inputs(op0)
                out0 = _op_out(op0)
                ins1 = _op_inputs(op1)
                out1 = _op_out(op1)
                ins2 = _op_inputs(op2)
                out2 = _op_out(op2)
                if (
                    len(ins0) == 2
                    and len(ins1) == 2
                    and len(ins2) == 1
                    and out2 == str(out_name)
                    and str(ins2[0]) == str(out1)
                    and set(map(str, ins1)) == {str(out0), "Bias"}
                ):
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    bias_name = "Bias"
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    bias_dims = _dims(bias_name)
                    if (
                        a_dims
                        and b_dims
                        and c_dims
                        and bias_dims
                        and len(a_dims) == 2
                        and len(b_dims) == 2
                        and len(c_dims) == 2
                        and len(bias_dims) == 1
                    ):
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if k2 == k and int(c_dims[0]) == m and int(c_dims[1]) == n and int(bias_dims[0]) == n:
                            if (
                                _elem(a_name) == "f32"
                                and _elem(b_name) == "f32"
                                and _elem(bias_name) == "f32"
                                and _elem(str(out_name)) == "f32"
                            ):
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "relu": True,
                                    "bias": bias_name,
                                    "row_mask": None,
                                    "col_mask": None,
                                }

            if matmul_v1 is None and intent_name == "matmul_fused_epilogue2d" and op_names == [
                "matmul",
                "broadcast_in_dim",
                "add",
                "broadcast_in_dim",
                "broadcast_in_dim",
                "and",
                "const",
                "where",
            ]:
                op0, op1, op2, op3, op4, op5, op6, op7 = ops_list
                ins0 = _op_inputs(op0)
                out0 = _op_out(op0)
                ins1 = _op_inputs(op1)
                out1 = _op_out(op1)
                ins2 = _op_inputs(op2)
                out2 = _op_out(op2)
                ins3 = _op_inputs(op3)
                out3 = _op_out(op3)
                ins4 = _op_inputs(op4)
                out4 = _op_out(op4)
                ins5 = _op_inputs(op5)
                out5 = _op_out(op5)
                out6 = _op_out(op6)
                ins7 = _op_inputs(op7)
                out7 = _op_out(op7)
                attrs1 = dict(getattr(op1, "attrs", {}) or {})
                attrs3 = dict(getattr(op3, "attrs", {}) or {})
                attrs4 = dict(getattr(op4, "attrs", {}) or {})
                if (
                    len(ins0) == 2
                    and len(ins1) == 1
                    and len(ins2) == 2
                    and len(ins3) == 1
                    and len(ins4) == 1
                    and len(ins5) == 2
                    and len(ins7) == 3
                    and out7 == str(out_name)
                    and list(attrs1.get("broadcast_dims") or []) == [1]
                    and list(attrs1.get("out_shape") or []) == ["M", "N"]
                    and list(attrs3.get("broadcast_dims") or []) == [0]
                    and list(attrs3.get("out_shape") or []) == ["M", "N"]
                    and list(attrs4.get("broadcast_dims") or []) == [1]
                    and list(attrs4.get("out_shape") or []) == ["M", "N"]
                    and str(ins1[0]) == "Bias"
                    and str(ins3[0]) == "RowMask"
                    and str(ins4[0]) == "ColMask"
                    and set(map(str, ins2)) == {str(out0), str(out1)}
                    and set(map(str, ins5)) == {str(out3), str(out4)}
                    and str(ins7[0]) == str(out5)
                    and str(ins7[1]) == str(out2)
                    and str(ins7[2]) == str(out6)
                ):
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    bias_name = str(ins1[0])
                    row_mask_name = str(ins3[0])
                    col_mask_name = str(ins4[0])
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    bias_dims = _dims(bias_name)
                    rm_dims = _dims(row_mask_name)
                    cm_dims = _dims(col_mask_name)
                    if (
                        a_dims
                        and b_dims
                        and c_dims
                        and bias_dims
                        and rm_dims
                        and cm_dims
                        and len(a_dims) == 2
                        and len(b_dims) == 2
                        and len(c_dims) == 2
                        and len(bias_dims) == 1
                        and len(rm_dims) == 1
                        and len(cm_dims) == 1
                    ):
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if (
                            k2 == k
                            and int(c_dims[0]) == m
                            and int(c_dims[1]) == n
                            and int(bias_dims[0]) == n
                            and int(rm_dims[0]) == m
                            and int(cm_dims[0]) == n
                        ):
                            if (
                                _elem(a_name) == "f32"
                                and _elem(b_name) == "f32"
                                and _elem(bias_name) == "f32"
                                and _elem(row_mask_name) in {"i1", "i8"}
                                and _elem(col_mask_name) in {"i1", "i8"}
                                and _elem(str(out_name)) == "f32"
                            ):
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "relu": False,
                                    "bias": bias_name,
                                    "row_mask": row_mask_name,
                                    "col_mask": col_mask_name,
                                }

            # NOTE: backend_legalize may simplify bias broadcast and col-mask broadcast for this fused epilogue:
            #   add_bias = add(matmul_out, Bias)
            #   row_mask_bcast = broadcast_in_dim(RowMask, out_shape=[M,N], dims=[0])
            #   mask = and(row_mask_bcast, ColMask)   # ColMask is rank-1 and broadcasted implicitly
            #   C = where(mask, add_bias, 0.0)
            if matmul_v1 is None and intent_name == "matmul_fused_epilogue2d" and op_names == [
                "matmul",
                "add",
                "broadcast_in_dim",
                "and",
                "const",
                "where",
            ]:
                op0, op1, op2, op3, op4, op5 = ops_list
                ins0 = _op_inputs(op0)
                out0 = _op_out(op0)
                ins1 = _op_inputs(op1)
                out1 = _op_out(op1)
                ins2 = _op_inputs(op2)
                out2 = _op_out(op2)
                ins3 = _op_inputs(op3)
                out3 = _op_out(op3)
                out4 = _op_out(op4)
                ins5 = _op_inputs(op5)
                out5 = _op_out(op5)
                attrs2 = dict(getattr(op2, "attrs", {}) or {})
                attrs4 = dict(getattr(op4, "attrs", {}) or {})
                if (
                    len(ins0) == 2
                    and len(ins1) == 2
                    and len(ins2) == 1
                    and len(ins3) == 2
                    and not _op_inputs(op4)
                    and len(ins5) == 3
                    and out5 == str(out_name)
                    and set(map(str, ins1)) == {str(out0), "Bias"}
                    and str(ins2[0]) == "RowMask"
                    and list(attrs2.get("broadcast_dims") or []) == [0]
                    and list(attrs2.get("out_shape") or []) == ["M", "N"]
                    and set(map(str, ins3)) == {str(out2), "ColMask"}
                    and float(attrs4.get("value") or 0.0) == 0.0
                    and str(attrs4.get("dtype") or "").strip() in {"", "f32"}
                    and str(ins5[0]) == str(out3)
                    and str(ins5[1]) == str(out1)
                    and str(ins5[2]) == str(out4)
                ):
                    a_name, b_name = str(ins0[0]), str(ins0[1])
                    bias_name = "Bias"
                    row_mask_name = "RowMask"
                    col_mask_name = "ColMask"
                    a_dims = _dims(a_name)
                    b_dims = _dims(b_name)
                    c_dims = _dims(str(out_name))
                    bias_dims = _dims(bias_name)
                    rm_dims = _dims(row_mask_name)
                    cm_dims = _dims(col_mask_name)
                    if (
                        a_dims
                        and b_dims
                        and c_dims
                        and bias_dims
                        and rm_dims
                        and cm_dims
                        and len(a_dims) == 2
                        and len(b_dims) == 2
                        and len(c_dims) == 2
                        and len(bias_dims) == 1
                        and len(rm_dims) == 1
                        and len(cm_dims) == 1
                    ):
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k2, n = int(b_dims[0]), int(b_dims[1])
                        if (
                            k2 == k
                            and int(c_dims[0]) == m
                            and int(c_dims[1]) == n
                            and int(bias_dims[0]) == n
                            and int(rm_dims[0]) == m
                            and int(cm_dims[0]) == n
                        ):
                            if (
                                _elem(a_name) == "f32"
                                and _elem(b_name) == "f32"
                                and _elem(bias_name) == "f32"
                                and _elem(row_mask_name) in {"i1", "i8"}
                                and _elem(col_mask_name) in {"i1", "i8"}
                                and _elem(str(out_name)) == "f32"
                            ):
                                matmul_v1 = {
                                    "A": a_name,
                                    "B": b_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "relu": False,
                                    "bias": bias_name,
                                    "row_mask": row_mask_name,
                                    "col_mask": col_mask_name,
                                }

        # Pattern: mlp2d (fused 2x matmul + bias + relu + bias).
        #
        # Expected op graph:
        #   aw1 = matmul(A, W1)
        #   aw1_b1 = add(aw1, broadcast(b1))
        #   hidden = max(aw1_b1, 0)
        #   hw2 = matmul(hidden, W2)
        #   C = add(hw2, broadcast(b2))
        if mlp2d_v1 is None and intent_name == "mlp2d":
            op_names = [str(getattr(op, "op", "")).strip() for op in ops_list]

            def _op_inputs(op: Any) -> list[str]:
                return [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]

            def _op_out(op: Any) -> str:
                return str(getattr(op, "output", "") or "").strip()

            def _dims(name: str) -> list[int] | None:
                spec = arg_specs.get(str(name))
                if not isinstance(spec, dict):
                    return None
                dims = list(spec.get("dims") or [])
                return [int(x) for x in dims] if dims else []

            def _elem(name: str) -> str:
                return str((arg_specs.get(str(name)) or {}).get("memref_elem_ty") or "").strip()

            # NOTE: backend_legalize may simplify broadcast+add+max(.,0) into add+relu.
            if op_names == ["matmul", "add", "relu", "matmul", "add"]:
                op0, op1, op2, op3, op4 = ops_list
                ins0, out0 = _op_inputs(op0), _op_out(op0)
                ins1, out1 = _op_inputs(op1), _op_out(op1)
                ins2, out2 = _op_inputs(op2), _op_out(op2)
                ins3, out3 = _op_inputs(op3), _op_out(op3)
                ins4, out4 = _op_inputs(op4), _op_out(op4)
                if (
                    len(ins0) == 2
                    and len(ins1) == 2
                    and len(ins2) == 1
                    and len(ins3) == 2
                    and len(ins4) == 2
                    and out4 == str(out_name)
                    and set(map(str, ins1)) == {str(out0), "b1"}
                    and str(ins2[0]) == str(out1)
                    and str(ins3[0]) == str(out2)
                    and str(ins3[1]) == "W2"
                    and set(map(str, ins4)) == {str(out3), "b2"}
                ):
                    a_name, w1_name = str(ins0[0]), str(ins0[1])
                    b1_name = "b1"
                    w2_name = "W2"
                    b2_name = "b2"
                    a_dims = _dims(a_name)
                    w1_dims = _dims(w1_name)
                    w2_dims = _dims(w2_name)
                    c_dims = _dims(str(out_name))
                    b1_dims = _dims(b1_name)
                    b2_dims = _dims(b2_name)
                    if (
                        a_dims
                        and w1_dims
                        and w2_dims
                        and c_dims
                        and b1_dims
                        and b2_dims
                        and len(a_dims) == 2
                        and len(w1_dims) == 2
                        and len(w2_dims) == 2
                        and len(c_dims) == 2
                        and len(b1_dims) == 1
                        and len(b2_dims) == 1
                    ):
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k1, h = int(w1_dims[0]), int(w1_dims[1])
                        h2, n = int(w2_dims[0]), int(w2_dims[1])
                        if (
                            k1 == k
                            and h2 == h
                            and int(c_dims[0]) == m
                            and int(c_dims[1]) == n
                            and int(b1_dims[0]) == h
                            and int(b2_dims[0]) == n
                        ):
                            if (
                                _elem(a_name) == "f32"
                                and _elem(w1_name) == "f32"
                                and _elem(b1_name) == "f32"
                                and _elem(w2_name) == "f32"
                                and _elem(b2_name) == "f32"
                                and _elem(str(out_name)) == "f32"
                            ):
                                mlp2d_v1 = {
                                    "A": a_name,
                                    "W1": w1_name,
                                    "b1": b1_name,
                                    "W2": w2_name,
                                    "b2": b2_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "H": int(h),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "BH": 16,
                                }

            if op_names == ["matmul", "broadcast_in_dim", "add", "const", "max", "matmul", "broadcast_in_dim", "add"]:
                op0, op1, op2, op3, op4, op5, op6, op7 = ops_list
                ins0, out0 = _op_inputs(op0), _op_out(op0)
                ins1, out1 = _op_inputs(op1), _op_out(op1)
                ins2, out2 = _op_inputs(op2), _op_out(op2)
                out3 = _op_out(op3)
                ins4, out4 = _op_inputs(op4), _op_out(op4)
                ins5, out5 = _op_inputs(op5), _op_out(op5)
                ins6, out6 = _op_inputs(op6), _op_out(op6)
                ins7, out7 = _op_inputs(op7), _op_out(op7)
                attrs1 = dict(getattr(op1, "attrs", {}) or {})
                attrs6 = dict(getattr(op6, "attrs", {}) or {})
                attrs3 = dict(getattr(op3, "attrs", {}) or {})
                if (
                    len(ins0) == 2
                    and len(ins1) == 1
                    and len(ins2) == 2
                    and not _op_inputs(op3)
                    and len(ins4) == 2
                    and len(ins5) == 2
                    and len(ins6) == 1
                    and len(ins7) == 2
                    and out7 == str(out_name)
                    and list(attrs1.get("broadcast_dims") or []) == [1]
                    and list(attrs1.get("out_shape") or []) == ["M", "H"]
                    and list(attrs6.get("broadcast_dims") or []) == [1]
                    and list(attrs6.get("out_shape") or []) == ["M", "N"]
                    and str(attrs3.get("dtype") or "").strip() == "f32"
                    and float(attrs3.get("value") or 0.0) == 0.0
                    and set(map(str, ins2)) == {str(out0), str(out1)}
                    and set(map(str, ins4)) == {str(out2), str(out3)}
                    and str(ins5[0]) == str(out4)
                    and str(ins5[1]) == "W2"
                    and set(map(str, ins7)) == {str(out5), str(out6)}
                    and str(ins1[0]) == "b1"
                    and str(ins6[0]) == "b2"
                ):
                    a_name, w1_name = str(ins0[0]), str(ins0[1])
                    b1_name = str(ins1[0])
                    w2_name = str(ins5[1])
                    b2_name = str(ins6[0])
                    a_dims = _dims(a_name)
                    w1_dims = _dims(w1_name)
                    w2_dims = _dims(w2_name)
                    c_dims = _dims(str(out_name))
                    b1_dims = _dims(b1_name)
                    b2_dims = _dims(b2_name)
                    if (
                        a_dims
                        and w1_dims
                        and w2_dims
                        and c_dims
                        and b1_dims
                        and b2_dims
                        and len(a_dims) == 2
                        and len(w1_dims) == 2
                        and len(w2_dims) == 2
                        and len(c_dims) == 2
                        and len(b1_dims) == 1
                        and len(b2_dims) == 1
                    ):
                        m, k = int(a_dims[0]), int(a_dims[1])
                        k1, h = int(w1_dims[0]), int(w1_dims[1])
                        h2, n = int(w2_dims[0]), int(w2_dims[1])
                        if k1 == k and h2 == h and int(c_dims[0]) == m and int(c_dims[1]) == n and int(b1_dims[0]) == h and int(b2_dims[0]) == n:
                            if (
                                _elem(a_name) == "f32"
                                and _elem(w1_name) == "f32"
                                and _elem(b1_name) == "f32"
                                and _elem(w2_name) == "f32"
                                and _elem(b2_name) == "f32"
                                and _elem(str(out_name)) == "f32"
                            ):
                                mlp2d_v1 = {
                                    "A": a_name,
                                    "W1": w1_name,
                                    "b1": b1_name,
                                    "W2": w2_name,
                                    "b2": b2_name,
                                    "out": str(out_name),
                                    "M": int(m),
                                    "N": int(n),
                                    "K": int(k),
                                    "H": int(h),
                                    "BM": 32,
                                    "BN": 32,
                                    "BK": 16,
                                    "BH": 16,
                                    }

        # Pattern: ai_bench_rope (const + rope) and ai_bench_warp (single warp op).
        # These ops are non-elementwise (inputs may have different numel) and need
        # explicit index math, so we lower them as dedicated kernels.
        last = ops_list[-1]
        last_name = str(getattr(last, "op", "")).strip()
        if last_name == "rope" and len(getattr(last, "inputs", []) or []) == 3:
            inp_name = str((getattr(last, "inputs", []) or [])[0])
            cos_name = str((getattr(last, "inputs", []) or [])[1])
            sin_name = str((getattr(last, "inputs", []) or [])[2])
            out_name2 = str(getattr(last, "output", "")).strip()
            inp_tt = (intent.tensors or {}).get(inp_name)
            cos_tt = (intent.tensors or {}).get(cos_name)
            sin_tt = (intent.tensors or {}).get(sin_name)
            out_tt2 = (intent.tensors or {}).get(out_name2)
            if inp_tt is not None and cos_tt is not None and sin_tt is not None and out_tt2 is not None:
                in_shape = list(getattr(inp_tt, "shape", []) or [])
                cos_shape = list(getattr(cos_tt, "shape", []) or [])
                sin_shape = list(getattr(sin_tt, "shape", []) or [])
                out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                if len(in_shape) == 4 and len(out_shape2) == 4 and len(cos_shape) == 2 and len(sin_shape) == 2:
                    seq_len = _resolve_dim_int(in_shape[0], bindings)
                    batch_num = _resolve_dim_int(in_shape[1], bindings)
                    head_num = _resolve_dim_int(in_shape[2], bindings)
                    head_dim = _resolve_dim_int(in_shape[3], bindings)
                    half = _resolve_dim_int(cos_shape[1], bindings)
                    if (
                        seq_len is not None
                        and batch_num is not None
                        and head_num is not None
                        and head_dim is not None
                        and half is not None
                        and head_dim > 0
                        and half > 0
                        and head_dim == 2 * half
                    ):
                        rope_v1 = {
                            "inp": str(inp_name),
                            "cos": str(cos_name),
                            "sin": str(sin_name),
                            "out": str(out_name2),
                            "SEQ_LEN": int(seq_len),
                            "BATCH_NUM": int(batch_num),
                            "HEAD_NUM": int(head_num),
                            "HEAD_DIM": int(head_dim),
                            "HALF": int(half),
                        }
        if last_name == "warp" and len(ops_list) == 1 and len(getattr(last, "inputs", []) or []) == 2:
            src_name = str((getattr(last, "inputs", []) or [])[0])
            off_name = str((getattr(last, "inputs", []) or [])[1])
            out_name2 = str(getattr(last, "output", "")).strip()
            src_tt = (intent.tensors or {}).get(src_name)
            off_tt = (intent.tensors or {}).get(off_name)
            out_tt2 = (intent.tensors or {}).get(out_name2)
            if src_tt is not None and off_tt is not None and out_tt2 is not None:
                src_shape = list(getattr(src_tt, "shape", []) or [])
                off_shape = list(getattr(off_tt, "shape", []) or [])
                out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                if len(src_shape) == 3 and len(off_shape) == 2 and len(out_shape2) == 3:
                    c_dim = _resolve_dim_int(src_shape[0], bindings)
                    h_dim = _resolve_dim_int(src_shape[1], bindings)
                    w_dim = _resolve_dim_int(src_shape[2], bindings)
                    if c_dim is not None and h_dim is not None and w_dim is not None and c_dim > 0 and h_dim > 0 and w_dim > 0:
                        warp_v1 = {
                            "src": str(src_name),
                            "offset": str(off_name),
                            "out": str(out_name2),
                            "C": int(c_dim),
                            "H": int(h_dim),
                            "W": int(w_dim),
                        }

        # Wave12: low-risk scalar/reduction kernels from coverage_batches denominator.
        if intent_name == "min_dim2d" and row_reduce_min_argmin_axis1 is None:
            required = {"inp", "out_value", "indices"}
            if required.issubset(set(arg_specs.keys())):
                inp_dims = list(arg_specs["inp"].get("dims") or [])
                val_dims = list(arg_specs["out_value"].get("dims") or [])
                idx_dims = list(arg_specs["indices"].get("dims") or [])
                if len(inp_dims) == 2 and len(val_dims) == 1 and len(idx_dims) == 1:
                    m0, n0 = int(inp_dims[0]), int(inp_dims[1])
                    if int(val_dims[0]) == m0 and int(idx_dims[0]) == m0 and int(out_total) == m0:
                        ok = (
                            str(arg_specs["inp"].get("memref_elem_ty") or "") == "f32"
                            and str(arg_specs["out_value"].get("memref_elem_ty") or "") == "f32"
                            and str(arg_specs["indices"].get("memref_elem_ty") or "") == "i32"
                        )
                        if ok:
                            row_reduce_min_argmin_axis1 = {
                                "inp": "inp",
                                "out_value": "out_value",
                                "out_index": "indices",
                                "M": int(m0),
                                "N": int(n0),
                            }

        if intent_name == "trace2d" and trace2d_v1 is None:
            out_key = str(out_name)
            if out_key in arg_specs:
                # FlagGems trace2d commonly uses input tensor name "input", but accept any
                # single rank-2 f32 argument as the input (excluding the scalar output).
                candidates = []
                for k, spec in (arg_specs or {}).items():
                    if str(k) == out_key:
                        continue
                    dims = list((spec or {}).get("dims") or [])
                    if len(dims) == 2 and str((spec or {}).get("memref_elem_ty") or "") == "f32":
                        candidates.append(str(k))
                if len(candidates) == 1:
                    inp_key = str(candidates[0])
                    inp_dims = list(arg_specs[inp_key].get("dims") or [])
                    out_dims2 = list(arg_specs[out_key].get("dims") or [])
                    if len(inp_dims) == 2 and len(out_dims2) == 0:
                        m0, n0 = int(inp_dims[0]), int(inp_dims[1])
                        ok = str(arg_specs[out_key].get("memref_elem_ty") or "") == "f32" and int(out_total) == 1
                        if ok and m0 > 0 and n0 > 0:
                            trace2d_v1 = {"inp": inp_key, "out": out_key, "M": int(m0), "N": int(n0)}

        if intent_name == "count_nonzero2d" and count_nonzero2d_v1 is None:
            out_key = str(out_name)
            if out_key in arg_specs:
                # FlagGems count_nonzero2d commonly uses input tensor name "x", but accept any
                # single rank-2 f32 argument as the input (excluding the scalar output).
                candidates = []
                for k, spec in (arg_specs or {}).items():
                    if str(k) == out_key:
                        continue
                    dims = list((spec or {}).get("dims") or [])
                    if len(dims) == 2 and str((spec or {}).get("memref_elem_ty") or "") == "f32":
                        candidates.append(str(k))
                if len(candidates) == 1:
                    inp_key = str(candidates[0])
                    inp_dims = list(arg_specs[inp_key].get("dims") or [])
                    out_dims2 = list(arg_specs[out_key].get("dims") or [])
                    if len(inp_dims) == 2 and len(out_dims2) == 0:
                        m0, n0 = int(inp_dims[0]), int(inp_dims[1])
                        ok = str(arg_specs[out_key].get("memref_elem_ty") or "") == "i64" and int(out_total) == 1
                        if ok and m0 > 0 and n0 > 0:
                            count_nonzero2d_v1 = {"inp": inp_key, "out": out_key, "M": int(m0), "N": int(n0)}

        if intent_name == "allclose2d" and allclose2d_v1 is None:
            required = {"A", "B", "rtol", "atol", str(out_name)}
            if required.issubset(set(arg_specs.keys())):
                a_dims = list(arg_specs["A"].get("dims") or [])
                b_dims = list(arg_specs["B"].get("dims") or [])
                out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                if len(a_dims) == 2 and a_dims == b_dims and len(out_dims2) == 0:
                    m0, n0 = int(a_dims[0]), int(a_dims[1])
                    ok = (
                        str(arg_specs["A"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["B"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["rtol"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs["atol"].get("memref_elem_ty") or "") == "f32"
                        and str(arg_specs[str(out_name)].get("memref_elem_ty") or "") == "i8"
                        and int(out_total) == 1
                    )
                    if ok and m0 > 0 and n0 > 0:
                        allclose2d_v1 = {"A": "A", "B": "B", "rtol": "rtol", "atol": "atol", "out": str(out_name), "M": int(m0), "N": int(n0)}
    if out_rank in {0, 1, 2}:
        red_sum_ops: list[tuple[int, Any]] = []
        red_max_ops: list[tuple[int, Any]] = []
        red_any_ops: list[tuple[int, Any]] = []
        red_min_ops: list[tuple[int, Any]] = []
        red_prod_ops: list[tuple[int, Any]] = []
        argmax_ops: list[tuple[int, Any]] = []
        argmin_ops: list[tuple[int, Any]] = []
        for i, op in enumerate(list(intent.ops or [])):
            name = str(getattr(op, "op", "")).strip()
            if name == "reduce_sum":
                red_sum_ops.append((int(i), op))
            elif name == "reduce_max":
                red_max_ops.append((int(i), op))
            elif name == "reduce_any":
                red_any_ops.append((int(i), op))
            elif name == "reduce_min":
                red_min_ops.append((int(i), op))
            elif name == "reduce_prod":
                red_prod_ops.append((int(i), op))
            elif name == "argmax":
                argmax_ops.append((int(i), op))
            elif name == "argmin":
                argmin_ops.append((int(i), op))

        # Wave12: scalar/row reductions for missing coverage_batches kernels.
        if out_rank == 0 and len(red_min_ops) == 1 and reduce_min_all_v1 is None:
            red_op_idx, red_op = red_min_ops[0]
            red_out = str(getattr(red_op, "output", "")).strip()
            red_inputs = [str(x) for x in list(getattr(red_op, "inputs", []) or []) if str(x).strip()]
            red_attrs = dict(getattr(red_op, "attrs", {}) or {})
            dims = red_attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list == [0, 1] and red_out and len(red_inputs) == 1:
                red_in = str(red_inputs[0])
                red_in_tt = (intent.tensors or {}).get(red_in)
                red_out_tt = (intent.tensors or {}).get(red_out)
                if red_in_tt is not None and red_out_tt is not None:
                    red_in_shape = list(getattr(red_in_tt, "shape", []) or [])
                    red_out_shape = list(getattr(red_out_tt, "shape", []) or [])
                    if len(red_in_shape) == 2 and len(red_out_shape) == 0:
                        m0 = _resolve_dim_int(red_in_shape[0], bindings)
                        n0 = _resolve_dim_int(red_in_shape[1], bindings)
                        if m0 is not None and n0 is not None and int(m0) > 0 and int(n0) > 0 and int(out_total) == 1:
                            red_in_ty = _dtype_to_mlir(str(getattr(red_in_tt, "dtype", "f32")))
                            red_out_ty = _dtype_to_mlir(str(getattr(red_out_tt, "dtype", "f32")))
                            if red_in_ty == "f32" and red_out_ty == "f32":
                                reduce_min_all_v1 = {
                                    "op_index": int(red_op_idx),
                                    "inp": str(red_in),
                                    "out": str(red_out),
                                    "M": int(m0),
                                    "N": int(n0),
                                }

        # Wave13: scalar reduce_prod (prod2d) for coverage_batches backfill.
        if out_rank == 0 and len(red_prod_ops) == 1 and reduce_prod_all_v1 is None:
            red_op_idx, red_op = red_prod_ops[0]
            red_out = str(getattr(red_op, "output", "")).strip()
            red_inputs = [str(x) for x in list(getattr(red_op, "inputs", []) or []) if str(x).strip()]
            red_attrs = dict(getattr(red_op, "attrs", {}) or {})
            dims = red_attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list == [0, 1] and red_out and len(red_inputs) == 1:
                red_in = str(red_inputs[0])
                red_in_tt = (intent.tensors or {}).get(red_in)
                red_out_tt = (intent.tensors or {}).get(red_out)
                if red_in_tt is not None and red_out_tt is not None:
                    red_in_shape = list(getattr(red_in_tt, "shape", []) or [])
                    red_out_shape = list(getattr(red_out_tt, "shape", []) or [])
                    if len(red_in_shape) == 2 and len(red_out_shape) == 0:
                        m0 = _resolve_dim_int(red_in_shape[0], bindings)
                        n0 = _resolve_dim_int(red_in_shape[1], bindings)
                        if m0 is not None and n0 is not None and int(m0) > 0 and int(n0) > 0 and int(out_total) == 1:
                            red_in_ty = _dtype_to_mlir(str(getattr(red_in_tt, "dtype", "f32")))
                            red_out_ty = _dtype_to_mlir(str(getattr(red_out_tt, "dtype", "f32")))
                            if red_in_ty == "f32" and red_out_ty == "f32":
                                reduce_prod_all_v1 = {
                                    "op_index": int(red_op_idx),
                                    "inp": str(red_in),
                                    "out": str(red_out),
                                    "M": int(m0),
                                    "N": int(n0),
                                }

        if out_rank == 1 and len(red_prod_ops) == 1 and row_reduce_prod_axis1 is None:
            red_op_idx, red_op = red_prod_ops[0]
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
                            red_in_ty = _dtype_to_mlir(str(getattr(red_in_tt, "dtype", "f32")))
                            if red_out_ty == "f32" and red_in_ty == "f32":
                                row_reduce_prod_axis1 = {
                                    "op_index": int(red_op_idx),
                                    "red_in": str(red_in),
                                    "red_out": str(red_out),
                                    "reduce_n": int(n0),
                                }

        if out_rank == 1 and len(argmax_ops) == 1 and row_argmax_axis1 is None:
            op_idx, op = argmax_ops[0]
            out2 = str(getattr(op, "output", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            attrs = dict(getattr(op, "attrs", {}) or {})
            axis = attrs.get("axis")
            try:
                axis_i = int(axis) if axis is not None else None
            except Exception:
                axis_i = None
            if axis_i == 1 and out2 and len(inputs) == 1 and out2 == str(out_name):
                in_name = str(inputs[0])
                in_tt = (intent.tensors or {}).get(in_name)
                out_tt2 = (intent.tensors or {}).get(out2)
                if in_tt is not None and out_tt2 is not None:
                    in_shape = list(getattr(in_tt, "shape", []) or [])
                    out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                    if len(in_shape) == 2 and len(out_shape2) == 1:
                        m0 = _resolve_dim_int(in_shape[0], bindings)
                        n0 = _resolve_dim_int(in_shape[1], bindings)
                        if m0 == int(out_total) and n0 is not None and int(n0) > 0:
                            in_ty = _dtype_to_mlir(str(getattr(in_tt, "dtype", "f32")))
                            out_ty = _dtype_to_mlir(str(getattr(out_tt2, "dtype", "i32")))
                            if in_ty == "f32" and out_ty == "i32":
                                row_argmax_axis1 = {
                                    "op_index": int(op_idx),
                                    "inp": str(in_name),
                                    "out": str(out2),
                                    "reduce_n": int(n0),
                                }

        if out_rank == 1 and len(argmin_ops) == 1 and row_argmin_axis1 is None:
            op_idx, op = argmin_ops[0]
            out2 = str(getattr(op, "output", "")).strip()
            inputs = [str(x) for x in list(getattr(op, "inputs", []) or []) if str(x).strip()]
            attrs = dict(getattr(op, "attrs", {}) or {})
            axis = attrs.get("axis")
            try:
                axis_i = int(axis) if axis is not None else None
            except Exception:
                axis_i = None
            if axis_i == 1 and out2 and len(inputs) == 1 and out2 == str(out_name):
                in_name = str(inputs[0])
                in_tt = (intent.tensors or {}).get(in_name)
                out_tt2 = (intent.tensors or {}).get(out2)
                if in_tt is not None and out_tt2 is not None:
                    in_shape = list(getattr(in_tt, "shape", []) or [])
                    out_shape2 = list(getattr(out_tt2, "shape", []) or [])
                    if len(in_shape) == 2 and len(out_shape2) == 1:
                        m0 = _resolve_dim_int(in_shape[0], bindings)
                        n0 = _resolve_dim_int(in_shape[1], bindings)
                        if m0 == int(out_total) and n0 is not None and int(n0) > 0:
                            in_ty = _dtype_to_mlir(str(getattr(in_tt, "dtype", "f32")))
                            out_ty = _dtype_to_mlir(str(getattr(out_tt2, "dtype", "i32")))
                            if in_ty == "f32" and out_ty == "i32":
                                row_argmin_axis1 = {
                                    "op_index": int(op_idx),
                                    "inp": str(in_name),
                                    "out": str(out2),
                                    "reduce_n": int(n0),
                                }

        if out_rank == 0 and len(red_sum_ops) == 1:
            red_op_idx, red_op = red_sum_ops[0]
            red_out = str(getattr(red_op, "output", "")).strip()
            red_inputs = [str(x) for x in list(getattr(red_op, "inputs", []) or []) if str(x).strip()]
            red_attrs = dict(getattr(red_op, "attrs", {}) or {})
            dims = red_attrs.get("dims")
            dims_list = list(dims) if isinstance(dims, list) else []
            if dims_list == [0] and red_out and len(red_inputs) == 1:
                red_in = str(red_inputs[0])
                red_in_tt = (intent.tensors or {}).get(red_in)
                red_out_tt = (intent.tensors or {}).get(red_out)
                if red_in_tt is not None and red_out_tt is not None:
                    red_in_shape = list(getattr(red_in_tt, "shape", []) or [])
                    red_out_shape = list(getattr(red_out_tt, "shape", []) or [])
                    if len(red_in_shape) == 1 and len(red_out_shape) == 0:
                        n0 = _resolve_dim_int(red_in_shape[0], bindings)
                        if n0 is not None and int(n0) > 0 and int(out_total) == 1:
                            red_out_ty = _dtype_to_mlir(str(getattr(red_out_tt, "dtype", "f32")))
                            if red_out_ty == "f32":
                                # Treat scalar reduce_sum as a degenerate row reduction with M=1.
                                row_reduce_sum_axis1 = {
                                    "op_index": int(red_op_idx),
                                    "red_in": str(red_in),
                                    "red_out": str(red_out),
                                    "reduce_n": int(n0),
                                }

        if out_rank == 1 and len(red_sum_ops) == 1:
            red_op_idx, red_op = red_sum_ops[0]
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

        if out_rank == 1 and len(red_max_ops) == 1:
            red_op_idx, red_op = red_max_ops[0]
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
                                row_reduce_max_axis1 = {
                                    "op_index": int(red_op_idx),
                                    "red_in": str(red_in),
                                    "red_out": str(red_out),
                                    "reduce_n": int(n0),
                                }

        if len(red_any_ops) == 1:
            red_op_idx, red_op = red_any_ops[0]
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
                    # Support both [M] and keepdims [M,1] outputs.
                    out_ok = len(red_out_shape) == 1 or (len(red_out_shape) == 2 and _resolve_dim_int(red_out_shape[1], bindings) == 1)
                    if len(red_in_shape) == 2 and out_ok:
                        m0 = _resolve_dim_int(red_in_shape[0], bindings)
                        n0 = _resolve_dim_int(red_in_shape[1], bindings)
                        if m0 == int(out_total) and n0 is not None and int(n0) > 0:
                            red_out_ty = _dtype_to_mlir(str(getattr(red_out_tt, "dtype", "bool")))
                            red_in_ty = _dtype_to_mlir(str(getattr(red_in_tt, "dtype", "bool")))
                            if red_out_ty == "i1" and red_in_ty == "i1":
                                row_reduce_any_axis1 = {
                                    "op_index": int(red_op_idx),
                                    "red_in": str(red_in),
                                    "red_out": str(red_out),
                                    "reduce_n": int(n0),
                                }

    # Pattern: row-wise softmax (axis=1 keepdims) -> output shape matches input.
    if out_rank == 2 and out_m is not None and out_n is not None:
        ops = list(intent.ops or [])
        if len(ops) == 5:
            op0, op1, op2, op3, op4 = ops
            n0 = str(getattr(op0, "op", "")).strip()
            n1 = str(getattr(op1, "op", "")).strip()
            n2 = str(getattr(op2, "op", "")).strip()
            n3 = str(getattr(op3, "op", "")).strip()
            n4 = str(getattr(op4, "op", "")).strip()
            if (n0, n1, n2, n3, n4) == ("reduce_max", "sub", "exp", "reduce_sum", "div"):
                max_out = str(getattr(op0, "output", "")).strip()
                centered = str(getattr(op1, "output", "")).strip()
                exp_vals = str(getattr(op2, "output", "")).strip()
                sum_out = str(getattr(op3, "output", "")).strip()
                div_out = str(getattr(op4, "output", "")).strip()
                if div_out == str(out_name):
                    op0_in = [str(x) for x in list(getattr(op0, "inputs", []) or []) if str(x).strip()]
                    op1_in = [str(x) for x in list(getattr(op1, "inputs", []) or []) if str(x).strip()]
                    op2_in = [str(x) for x in list(getattr(op2, "inputs", []) or []) if str(x).strip()]
                    op3_in = [str(x) for x in list(getattr(op3, "inputs", []) or []) if str(x).strip()]
                    op4_in = [str(x) for x in list(getattr(op4, "inputs", []) or []) if str(x).strip()]

                    op0_attrs = dict(getattr(op0, "attrs", {}) or {})
                    op3_attrs = dict(getattr(op3, "attrs", {}) or {})
                    dims0 = list(op0_attrs.get("dims") or []) if isinstance(op0_attrs.get("dims"), list) else []
                    dims3 = list(op3_attrs.get("dims") or []) if isinstance(op3_attrs.get("dims"), list) else []
                    keep0 = bool(op0_attrs.get("keepdims"))
                    keep3 = bool(op3_attrs.get("keepdims"))

                    if (
                        len(op0_in) == 1
                        and len(op1_in) == 2
                        and len(op2_in) == 1
                        and len(op3_in) == 1
                        and len(op4_in) == 2
                        and dims0 == [1]
                        and dims3 == [1]
                        and (keep0 == keep3)
                    ):
                        in_name = str(op0_in[0])
                        if (
                            op1_in[0] == in_name
                            and op1_in[1] == max_out
                            and op2_in[0] == centered
                            and op3_in[0] == exp_vals
                            and op4_in[0] == exp_vals
                            and op4_in[1] == sum_out
                            and in_name in arg_specs
                            and str(out_name) in arg_specs
                        ):
                            in_tt = (intent.tensors or {}).get(in_name)
                            out_tt = (intent.tensors or {}).get(str(out_name))
                            max_tt = (intent.tensors or {}).get(max_out)
                            sum_tt = (intent.tensors or {}).get(sum_out)
                            if in_tt is not None and out_tt is not None:
                                in_shape = list(getattr(in_tt, "shape", []) or [])
                                out_shape = list(getattr(out_tt, "shape", []) or [])
                                max_ok = True
                                sum_ok = True
                                if max_tt is not None:
                                    max_shape = list(getattr(max_tt, "shape", []) or [])
                                    if len(max_shape) == 2:
                                        m0 = _resolve_dim_int(max_shape[0], bindings)
                                        n0 = _resolve_dim_int(max_shape[1], bindings)
                                        if m0 is not None and n0 is not None:
                                            max_ok = (int(m0) == int(out_m)) and (int(n0) == 1)
                                if sum_tt is not None:
                                    sum_shape = list(getattr(sum_tt, "shape", []) or [])
                                    if len(sum_shape) == 2:
                                        m0 = _resolve_dim_int(sum_shape[0], bindings)
                                        n0 = _resolve_dim_int(sum_shape[1], bindings)
                                        if m0 is not None and n0 is not None:
                                            sum_ok = (int(m0) == int(out_m)) and (int(n0) == 1)
                                if (
                                    len(in_shape) == 2
                                    and len(out_shape) == 2
                                    and _resolve_dim_int(in_shape[0], bindings) == int(out_m)
                                    and _resolve_dim_int(in_shape[1], bindings) == int(out_n)
                                    and _resolve_dim_int(out_shape[0], bindings) == int(out_m)
                                    and _resolve_dim_int(out_shape[1], bindings) == int(out_n)
                                    and max_ok
                                    and sum_ok
                                ):
                                    in_ty = _dtype_to_mlir(str(getattr(in_tt, "dtype", "f32")))
                                    out_ty = _dtype_to_mlir(str(getattr(out_tt, "dtype", "f32")))
                                    if in_ty == "f32" and out_ty == "f32":
                                        row_softmax_axis1 = {
                                            "inp": str(in_name),
                                            "out": str(out_name),
                                            "reduce_n": int(out_n),
                                        }

    # Pattern: masked_softmax2d (triton-native) -> fuse mask+softmax in one kernel.
    #
    # Intent graph typically materializes:
    #   cast(mask)->broadcast->where(mask, inp, -1e9)->softmax(axis=1)
    #
    # We avoid depending on the exact SSA names and just require:
    # - kernel name matches
    # - external inputs include `inp` (f32 [M,N]) and `mask` ([N] int/bool)
    # - the final op is a softmax(axis=1) producing the output
    if str(kernel_name) == "masked_softmax2d" and out_rank == 2 and out_m is not None and out_n is not None:
        ops = list(intent.ops or [])
        # Backend legalize currently decomposes softmax into reduce_max/reduce_sum,
        # so do not rely on the presence of an explicit `softmax` op.
        seen_max = False
        seen_sum = False
        for op in ops:
            name = str(getattr(op, "op", "")).strip()
            if name not in {"reduce_max", "reduce_sum"}:
                continue
            attrs = dict(getattr(op, "attrs", {}) or {})
            dims = list(attrs.get("dims") or []) if isinstance(attrs.get("dims"), list) else []
            if dims != [1]:
                continue
            if name == "reduce_max":
                seen_max = True
            elif name == "reduce_sum":
                seen_sum = True
        if seen_max and seen_sum and "inp" in arg_specs and "mask" in arg_specs and str(out_name) in arg_specs:
            in_tt = (intent.tensors or {}).get("inp")
            out_tt = (intent.tensors or {}).get(str(out_name))
            mask_tt = (intent.tensors or {}).get("mask")
            if in_tt is not None and out_tt is not None and mask_tt is not None:
                in_shape = list(getattr(in_tt, "shape", []) or [])
                out_shape = list(getattr(out_tt, "shape", []) or [])
                mask_shape = list(getattr(mask_tt, "shape", []) or [])
                if (
                    len(in_shape) == 2
                    and len(out_shape) == 2
                    and len(mask_shape) == 1
                    and _resolve_dim_int(in_shape[0], bindings) == int(out_m)
                    and _resolve_dim_int(in_shape[1], bindings) == int(out_n)
                    and _resolve_dim_int(out_shape[0], bindings) == int(out_m)
                    and _resolve_dim_int(out_shape[1], bindings) == int(out_n)
                    and _resolve_dim_int(mask_shape[0], bindings) == int(out_n)
                ):
                    in_ty = _dtype_to_mlir(str(getattr(in_tt, "dtype", "f32")))
                    out_ty = _dtype_to_mlir(str(getattr(out_tt, "dtype", "f32")))
                    mask_elem_ty = str(arg_specs["mask"].get("memref_elem_ty") or "")
                    if in_ty == "f32" and out_ty == "f32" and mask_elem_ty in {"i8", "i32", "i64"}:
                        row_masked_softmax_axis1 = {
                            "inp": "inp",
                            "mask": "mask",
                            "out": str(out_name),
                            "reduce_n": int(out_n),
                        }

    # Grouped row sum (reshape+reduce): out[m, g] = sum_k inp[m, g*group_size + k]
    #
    # Intent pattern (Triton native):
    #   reshape inp[M,N] -> inp_reshaped[M,G,GROUP_SIZE]
    #   reduce_sum(axis=2) -> out[M,G]
    if intent_name == "grouped_row_sum2d" and out_rank == 2 and out_m is not None and out_n is not None:
        ops = list(intent.ops or [])
        if len(ops) == 2:
            op0, op1 = ops[0], ops[1]
            if str(getattr(op0, "op", "")).strip() == "reshape" and str(getattr(op1, "op", "")).strip() == "reduce_sum":
                red_attrs = dict(getattr(op1, "attrs", {}) or {})
                dims_raw = red_attrs.get("dims")
                dims_list = list(dims_raw) if isinstance(dims_raw, list) else []
                axis_raw = red_attrs.get("axis")
                axis_i = None
                try:
                    if axis_raw is not None:
                        axis_i = int(axis_raw)
                except Exception:
                    axis_i = None
                if dims_list == [2] or axis_i == 2:
                    if "inp" in arg_specs and str(out_name) in arg_specs:
                        in_dims = list(arg_specs["inp"].get("dims") or [])
                        out_dims2 = list(arg_specs[str(out_name)].get("dims") or [])
                        if len(in_dims) == 2 and len(out_dims2) == 2:
                            in_m, in_n = int(in_dims[0]), int(in_dims[1])
                            out_m2, out_g = int(out_dims2[0]), int(out_dims2[1])
                            if in_m != int(out_m) or out_m2 != int(out_m) or out_g != int(out_n):
                                raise RuntimeError(
                                    "grouped_row_sum2d shape mismatch: "
                                    f"inp_dims={in_dims} out_dims={out_dims2} expected_out=[{out_m},{out_n}]"
                                )
                            if str(arg_specs["inp"].get("memref_elem_ty")) != "f32" or str(
                                arg_specs[str(out_name)].get("memref_elem_ty")
                            ) != "f32":
                                raise RuntimeError("grouped_row_sum2d expects f32 inp/out tensors")
                            if int(out_g) <= 0 or int(in_n) <= 0:
                                raise RuntimeError(f"grouped_row_sum2d expects positive dims (G={out_g}, N={in_n})")
                            if int(in_n) % int(out_g) != 0:
                                raise RuntimeError(
                                    "grouped_row_sum2d expects N divisible by G "
                                    f"(N={in_n}, G={out_g}, N%G={int(in_n)%int(out_g)})"
                                )
                            group_size = int(int(in_n) // int(out_g))
                            gs_bind = bindings.get("GROUP_SIZE")
                            if gs_bind is not None:
                                try:
                                    if int(gs_bind) != int(group_size):
                                        raise RuntimeError(
                                            "grouped_row_sum2d GROUP_SIZE binding mismatch: "
                                            f"binding={int(gs_bind)} inferred={int(group_size)}"
                                        )
                                except Exception:
                                    pass
                            row_grouped_row_sum2d = {
                                "inp": "inp",
                                "out": str(out_name),
                                "M": int(out_m),
                                "G": int(out_g),
                                "N": int(in_n),
                                "GROUP_SIZE": int(group_size),
                            }

    # Row-wise norms (multi-output): implement as dedicated kernels instead of
    # trying to interpret the full op graph.
    if out_rank == 2 and out_m is not None and out_n is not None:
        if intent_name == "layer_norm_persistent":
            required = {"in_ptr", "weight_ptr", "bias_ptr", "out_ptr", "out_mean_ptr", "out_rstd_ptr"}
            if required.issubset(set(arg_specs.keys())):
                ok = (
                    str(arg_specs["in_ptr"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["weight_ptr"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["bias_ptr"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["out_ptr"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["out_mean_ptr"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["out_rstd_ptr"].get("memref_elem_ty")) == "f32"
                    and list(arg_specs["in_ptr"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["out_ptr"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["weight_ptr"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["bias_ptr"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["out_mean_ptr"].get("dims") or []) == [int(out_m)]
                    and list(arg_specs["out_rstd_ptr"].get("dims") or []) == [int(out_m)]
                )
                if ok:
                    eps_const = _extract_f32_const("eps")
                    if eps_const is None:
                        raise RuntimeError("layer_norm_persistent requires f32 scalar const 'eps'")
                    row_layer_norm_persistent = {"M": int(out_m), "N": int(out_n), "eps_const": float(eps_const)}

        elif intent_name == "layer_norm_residual2d":
            required = {"inp", "residual", "weight", "bias", "out", "mean", "rstd"}
            if required.issubset(set(arg_specs.keys())):
                ok = (
                    str(arg_specs["inp"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["residual"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["weight"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["bias"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["out"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["mean"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["rstd"].get("memref_elem_ty")) == "f32"
                    and list(arg_specs["inp"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["residual"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["out"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["weight"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["bias"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["mean"].get("dims") or []) == [int(out_m)]
                    and list(arg_specs["rstd"].get("dims") or []) == [int(out_m)]
                )
                if ok:
                    eps_const = _extract_f32_const("eps")
                    if eps_const is None:
                        raise RuntimeError("layer_norm_residual2d requires f32 scalar const 'eps'")
                    row_layer_norm_residual2d = {"M": int(out_m), "N": int(out_n), "eps_const": float(eps_const)}

        elif intent_name == "ai_bench_layernorm":
            required = {"X", "W", "B", "Y", "Mean", "Rstd"}
            if required.issubset(set(arg_specs.keys())):
                ok = (
                    str(arg_specs["X"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["W"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["B"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["Y"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["Mean"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["Rstd"].get("memref_elem_ty")) == "f32"
                    and list(arg_specs["X"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["Y"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["W"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["B"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["Mean"].get("dims") or []) == [int(out_m)]
                    and list(arg_specs["Rstd"].get("dims") or []) == [int(out_m)]
                )
                if ok:
                    eps_const = _extract_f32_const("eps")
                    if eps_const is None:
                        raise RuntimeError("ai_bench_layernorm requires f32 scalar const 'eps'")
                    # Reuse the axis-1 layer-norm emitter with different arg names.
                    row_layer_norm_persistent = {
                        "M": int(out_m),
                        "N": int(out_n),
                        "eps_const": float(eps_const),
                        "inp": "X",
                        "weight": "W",
                        "bias": "B",
                        "out": "Y",
                        "mean": "Mean",
                        "rstd": "Rstd",
                    }

        elif intent_name == "rms_norm2d":
            if out_m is None or out_n is None:
                raise RuntimeError(f"rms_norm2d expects rank-2 output, got out_rank={out_rank}")

            def _first_present(*names: str) -> str | None:
                for cand in names:
                    key = str(cand).strip()
                    if key and key in arg_specs:
                        return key
                return None

            inp_name = _first_present("inp", "input", "X")
            w_name = _first_present("weight", "W")
            out_name2 = str(out_name)
            rstd_name = _first_present("rstd", "INV_RMS", "InvRms")
            eps_name = _first_present("eps")

            if inp_name is not None and w_name is not None and rstd_name is not None and out_name2 in arg_specs:
                ok = (
                    str(arg_specs[inp_name].get("memref_elem_ty")) == "f32"
                    and str(arg_specs[w_name].get("memref_elem_ty")) == "f32"
                    and str(arg_specs[out_name2].get("memref_elem_ty")) == "f32"
                    and str(arg_specs[rstd_name].get("memref_elem_ty")) == "f32"
                    and list(arg_specs[inp_name].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs[out_name2].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs[w_name].get("dims") or []) == [int(out_n)]
                    and list(arg_specs[rstd_name].get("dims") or []) == [int(out_m)]
                )
                if ok:
                    eps_const = _extract_f32_const("eps")
                    if eps_const is not None:
                        row_rms_norm2d = {
                            "M": int(out_m),
                            "N": int(out_n),
                            "eps_const": float(eps_const),
                            "inp_name": str(inp_name),
                            "weight_name": str(w_name),
                            "out_name": str(out_name2),
                            "rstd_name": str(rstd_name),
                        }
                    else:
                        if eps_name is None:
                            raise RuntimeError("rms_norm2d requires f32 scalar eps (const op or scalar ABI tensor 'eps')")
                        eps_spec = dict(arg_specs.get(eps_name) or {})
                        eps_ok = bool(eps_spec.get("scalar")) and str(eps_spec.get("memref_elem_ty") or "") == "f32"
                        if not eps_ok:
                            raise RuntimeError("rms_norm2d requires f32 scalar eps (const op or scalar ABI tensor)")
                        row_rms_norm2d = {
                            "M": int(out_m),
                            "N": int(out_n),
                            "eps_tensor_name": str(eps_name),
                            "inp_name": str(inp_name),
                            "weight_name": str(w_name),
                            "out_name": str(out_name2),
                            "rstd_name": str(rstd_name),
                        }

        elif intent_name == "rms_norm_residual2d":
            required = {"inp", "residual", "weight", "bias", "out", "rstd"}
            if required.issubset(set(arg_specs.keys())):
                ok = (
                    str(arg_specs["inp"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["residual"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["weight"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["bias"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["out"].get("memref_elem_ty")) == "f32"
                    and str(arg_specs["rstd"].get("memref_elem_ty")) == "f32"
                    and list(arg_specs["inp"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["residual"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["out"].get("dims") or []) == [int(out_m), int(out_n)]
                    and list(arg_specs["weight"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["bias"].get("dims") or []) == [int(out_n)]
                    and list(arg_specs["rstd"].get("dims") or []) == [int(out_m)]
                )
                if ok:
                    eps_const = _extract_f32_const("eps")
                    if eps_const is None:
                        raise RuntimeError("rms_norm_residual2d requires f32 scalar const 'eps'")
                    row_rms_norm_residual2d = {"M": int(out_m), "N": int(out_n), "eps_const": float(eps_const)}

    if intent_name == "group_norm_kernel":
        required = {"X", "Y", "W", "B", "Mean", "Rstd"}
        if required.issubset(set(arg_specs.keys())):
            if out_rank != 3:
                raise RuntimeError(f"group_norm_kernel expects rank-3 output, got out_rank={out_rank}")
            n_dim = int(out_dims[0])
            c_dim = int(out_dims[1])
            hw_dim = int(out_dims[2])
            if n_dim <= 0 or c_dim <= 0 or hw_dim <= 0:
                raise RuntimeError(f"group_norm_kernel invalid output dims: out_dims={out_dims}")

            # `num_groups` is not present in the rank-3 output shape; take it from bindings.
            if "num_groups" not in bindings:
                raise RuntimeError("group_norm_kernel requires shape_bindings['num_groups']")
            num_groups = int(bindings["num_groups"])
            if num_groups <= 0:
                raise RuntimeError(f"group_norm_kernel invalid num_groups={num_groups}")
            if c_dim % num_groups != 0:
                raise RuntimeError(
                    "group_norm_kernel requires C divisible by num_groups, got "
                    f"C={c_dim} num_groups={num_groups}"
                )
            group_size = int(c_dim // num_groups)
            if group_size <= 0:
                raise RuntimeError(f"group_norm_kernel invalid group_size={group_size}")

            ok = (
                str(arg_specs["X"].get("memref_elem_ty")) == "f32"
                and str(arg_specs["Y"].get("memref_elem_ty")) == "f32"
                and str(arg_specs["W"].get("memref_elem_ty")) == "f32"
                and str(arg_specs["B"].get("memref_elem_ty")) == "f32"
                and str(arg_specs["Mean"].get("memref_elem_ty")) == "f32"
                and str(arg_specs["Rstd"].get("memref_elem_ty")) == "f32"
                and list(arg_specs["X"].get("dims") or []) == [n_dim, c_dim, hw_dim]
                and list(arg_specs["Y"].get("dims") or []) == [n_dim, c_dim, hw_dim]
                and list(arg_specs["W"].get("dims") or []) == [c_dim]
                and list(arg_specs["B"].get("dims") or []) == [c_dim]
                and list(arg_specs["Mean"].get("dims") or []) == [n_dim, num_groups]
                and list(arg_specs["Rstd"].get("dims") or []) == [n_dim, num_groups]
            )
            if ok:
                eps_const = _extract_f32_const("eps")
                if eps_const is None:
                    raise RuntimeError("group_norm_kernel requires f32 scalar const 'eps'")
                group_norm_kernel_v1 = {
                    "N": int(n_dim),
                    "C": int(c_dim),
                    "HW": int(hw_dim),
                    "num_groups": int(num_groups),
                    "group_size": int(group_size),
                    "eps_const": float(eps_const),
                }

    if intent_name == "per_token_group_quant_fp8_2d":
        required = {"y", "eps", "fp8_min", "fp8_max", "y_q", "y_s"}
        if not required.issubset(set(arg_specs.keys())):
            missing = sorted([x for x in required if x not in arg_specs])
            raise RuntimeError(f"per_token_group_quant_fp8_2d missing ABI tensors: {missing}")
        if out_rank != 2:
            raise RuntimeError(f"per_token_group_quant_fp8_2d expects rank-2 output, got out_rank={out_rank}")
        m_dim = int(out_dims[0])
        n_dim = int(out_dims[1])
        y_dims = list(arg_specs["y"].get("dims") or [])
        yq_dims = list(arg_specs["y_q"].get("dims") or [])
        ys_dims = list(arg_specs["y_s"].get("dims") or [])
        if y_dims != [m_dim, n_dim] or yq_dims != [m_dim, n_dim] or len(ys_dims) != 2 or int(ys_dims[0]) != m_dim:
            raise RuntimeError(
                "per_token_group_quant_fp8_2d shape mismatch: "
                f"y_dims={y_dims} y_q_dims={yq_dims} y_s_dims={ys_dims} expected=[{m_dim},{n_dim}]"
            )
        g_dim = int(ys_dims[1])
        if g_dim <= 0:
            raise RuntimeError(f"per_token_group_quant_fp8_2d invalid G={g_dim}")
        if n_dim % g_dim != 0:
            raise RuntimeError(f"per_token_group_quant_fp8_2d expects N divisible by G, got N={n_dim} G={g_dim}")
        group_size = int(n_dim // g_dim)
        gs_bind = bindings.get("GROUP_SIZE")
        if gs_bind is not None:
            try:
                if int(gs_bind) != int(group_size):
                    raise RuntimeError(
                        "per_token_group_quant_fp8_2d GROUP_SIZE binding mismatch: "
                        f"GROUP_SIZE={int(gs_bind)} expected={int(group_size)}"
                    )
            except Exception:
                raise RuntimeError("per_token_group_quant_fp8_2d invalid GROUP_SIZE binding") from None

        ok = (
            str(arg_specs["y"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["y_q"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["y_s"].get("memref_elem_ty")) == "f32"
            and bool(arg_specs["eps"].get("scalar"))
            and bool(arg_specs["fp8_min"].get("scalar"))
            and bool(arg_specs["fp8_max"].get("scalar"))
            and str(arg_specs["eps"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["fp8_min"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["fp8_max"].get("memref_elem_ty")) == "f32"
        )
        if not ok:
            raise RuntimeError("per_token_group_quant_fp8_2d expects f32 tensors and f32 scalar eps/fp8_min/fp8_max")
        per_token_group_quant_fp8_2d_v1 = {
            "M": int(m_dim),
            "N": int(n_dim),
            "G": int(g_dim),
            "MG": int(m_dim * g_dim),
            "GROUP_SIZE": int(group_size),
        }

    if intent_name == "batch_norm2d":
        required = {
            "input",
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "eps",
            "momentum",
            "n_elements",
            "n_minus_1",
            "output_1",
            "mean",
            "inv_std",
            "running_mean_out",
            "running_var_out",
        }
        if not required.issubset(set(arg_specs.keys())):
            missing = sorted([x for x in required if x not in arg_specs])
            raise RuntimeError(f"batch_norm2d missing ABI tensors: {missing}")
        if out_rank != 3:
            raise RuntimeError(f"batch_norm2d expects rank-3 output, got out_rank={out_rank}")
        n_dim = int(out_dims[0])
        c_dim = int(out_dims[1])
        hw_dim = int(out_dims[2])
        if n_dim <= 0 or c_dim <= 0 or hw_dim <= 0:
            raise RuntimeError(f"batch_norm2d invalid output dims: out_dims={out_dims}")
        ok = (
            str(arg_specs["input"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["weight"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["bias"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["running_mean"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["running_var"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["output_1"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["mean"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["inv_std"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["running_mean_out"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["running_var_out"].get("memref_elem_ty")) == "f32"
            and bool(arg_specs["eps"].get("scalar"))
            and bool(arg_specs["momentum"].get("scalar"))
            and bool(arg_specs["n_elements"].get("scalar"))
            and bool(arg_specs["n_minus_1"].get("scalar"))
            and str(arg_specs["eps"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["momentum"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["n_elements"].get("memref_elem_ty")) == "f32"
            and str(arg_specs["n_minus_1"].get("memref_elem_ty")) == "f32"
            and list(arg_specs["input"].get("dims") or []) == [n_dim, c_dim, hw_dim]
            and list(arg_specs["output_1"].get("dims") or []) == [n_dim, c_dim, hw_dim]
            and list(arg_specs["weight"].get("dims") or []) == [c_dim]
            and list(arg_specs["bias"].get("dims") or []) == [c_dim]
            and list(arg_specs["running_mean"].get("dims") or []) == [c_dim]
            and list(arg_specs["running_var"].get("dims") or []) == [c_dim]
            and list(arg_specs["mean"].get("dims") or []) == [c_dim]
            and list(arg_specs["inv_std"].get("dims") or []) == [c_dim]
            and list(arg_specs["running_mean_out"].get("dims") or []) == [c_dim]
            and list(arg_specs["running_var_out"].get("dims") or []) == [c_dim]
        )
        if not ok:
            raise RuntimeError("batch_norm2d expects f32 tensors, scalar f32 params, and canonical shapes")
        batch_norm2d_v1 = {"N": int(n_dim), "C": int(c_dim), "HW": int(hw_dim)}

    # Assemble module text.
    launch_override: dict[str, Any] | None = None
    kernel_kind = "elementwise_v1"
    shared_global_sym: str | None = None
    shared_global_memref_ty: str | None = None
    cuda_real_mlir_attention_cfg: dict[str, Any] | None = None
    cuda_real_mlir_matmul_cfg: dict[str, Any] | None = None
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
    gpu_module_sym_insert = len(lines)
    lines.append(f"    gpu.func @{kernel_name}({arg_sig}) kernel {{")
    if intent_name == "logspace1d":
        # Perf-first: `logspace1d` in our allowlist benchmarks is a tiny (N=64) elementwise
        # kernel where launch overhead dominates. Avoid launching mostly-idle warps.
        block_threads = 64 if int(out_total) <= 64 else 256
        blocks = (int(out_total) + int(block_threads) - 1) // int(block_threads)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(blocks), 1, 1]}
    if dropout_v1 is not None:
        kernel_kind = "dropout_philox_v1"
        x_name = str(dropout_v1["x"])
        p_name = str(dropout_v1["p"])
        seed_name = str(dropout_v1["seed"])
        out_name2 = str(dropout_v1["out"])
        n_total = int(dropout_v1["N"])
        n_rounds = int(dropout_v1.get("n_rounds") or 10)

        if str(arg_specs[x_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_dropout expects f32 input tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_dropout expects f32 output tensor")
        if str(arg_specs[p_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_dropout expects f32 scalar p")
        if str(arg_specs[seed_name].get("memref_elem_ty")) != "i32":
            raise RuntimeError("ai_bench_dropout expects i32 scalar seed")

        x_memref = str(arg_specs[x_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        p_memref = str(arg_specs[p_name]["memref"])
        seed_memref = str(arg_specs[seed_name]["memref"])

        block_threads = 256
        blocks = (int(n_total) + int(block_threads) - 1) // int(block_threads)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(blocks), 1, 1]}

        # Philox parameters (match Triton tl.rand default path for 32-bit offsets).
        key_a = -1640531527  # 0x9E3779B9
        key_b = -1150833019  # 0xBB67AE85
        round_a = -766435501  # 0xD2511F53
        round_b = -845247145  # 0xCD9E8D57
        scale_f32 = _as_f32_const(4.6566127342e-10)

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cN = arith.constant {int(n_total)} : index")
        lines.append(f"      %cBlock = arith.constant {int(block_threads)} : index")
        lines.append("      %base = arith.muli %bid, %cBlock : index")
        lines.append("      %idx = arith.addi %base, %tid : index")
        lines.append("      %pred = arith.cmpi ult, %idx, %cN : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %xv = memref.load {arg_ssa[x_name]}[%idx] : {x_memref}")
        lines.append(f"        %p_v = memref.load {arg_ssa[p_name]}[%c0] : {p_memref}")
        lines.append(f"        %seed_i32 = memref.load {arg_ssa[seed_name]}[%c0] : {seed_memref}")
        lines.append("        %off_i32 = arith.index_cast %idx : index to i32")

        # Philox state: c0=offset, c1=c2=c3=0, k0=seed_lo, k1=0.
        c0_rng = "%off_i32"
        c1_rng = _fresh("c1")
        c2_rng = _fresh("c2")
        c3_rng = _fresh("c3")
        k0_rng = "%seed_i32"
        k1_rng = _fresh("k1")
        lines.append(f"        {c1_rng} = arith.constant 0 : i32")
        lines.append(f"        {c2_rng} = arith.constant 0 : i32")
        lines.append(f"        {c3_rng} = arith.constant 0 : i32")
        lines.append(f"        {k1_rng} = arith.constant 0 : i32")

        c32_i64 = _fresh("c32_i64")
        lines.append(f"        {c32_i64} = arith.constant 32 : i64")
        c_round_a = _fresh("round_a")
        c_round_b = _fresh("round_b")
        c_key_a = _fresh("key_a")
        c_key_b = _fresh("key_b")
        lines.append(f"        {c_round_a} = arith.constant {int(round_a)} : i32")
        lines.append(f"        {c_round_b} = arith.constant {int(round_b)} : i32")
        lines.append(f"        {c_key_a} = arith.constant {int(key_a)} : i32")
        lines.append(f"        {c_key_b} = arith.constant {int(key_b)} : i32")

        def _umulhi_i32(a: str, b: str, *, tag: str) -> str:
            a64 = _fresh(f"{tag}_a64")
            b64 = _fresh(f"{tag}_b64")
            prod64 = _fresh(f"{tag}_prod64")
            hi64 = _fresh(f"{tag}_hi64")
            hi32 = _fresh(f"{tag}_hi32")
            lines.append(f"        {a64} = arith.extui {a} : i32 to i64")
            lines.append(f"        {b64} = arith.extui {b} : i32 to i64")
            lines.append(f"        {prod64} = arith.muli {a64}, {b64} : i64")
            lines.append(f"        {hi64} = arith.shrui {prod64}, {c32_i64} : i64")
            lines.append(f"        {hi32} = arith.trunci {hi64} : i64 to i32")
            return hi32

        for r in range(int(n_rounds)):
            t_c0 = c0_rng
            t_c2 = c2_rng
            umulhi_b_c2 = _umulhi_i32(c_round_b, t_c2, tag=f"r{r}_b")
            umulhi_a_c0 = _umulhi_i32(c_round_a, t_c0, tag=f"r{r}_a")

            c0_x1 = _fresh(f"r{r}_c0_x1")
            c0_new = _fresh(f"r{r}_c0")
            c2_x1 = _fresh(f"r{r}_c2_x1")
            c2_new = _fresh(f"r{r}_c2")
            c1_new = _fresh(f"r{r}_c1")
            c3_new = _fresh(f"r{r}_c3")
            k0_new = _fresh(f"r{r}_k0")
            k1_new = _fresh(f"r{r}_k1")

            # c0 = umulhi(B, c2) ^ c1 ^ k0
            lines.append(f"        {c0_x1} = arith.xori {umulhi_b_c2}, {c1_rng} : i32")
            lines.append(f"        {c0_new} = arith.xori {c0_x1}, {k0_rng} : i32")
            # c2 = umulhi(A, c0_old) ^ c3 ^ k1
            lines.append(f"        {c2_x1} = arith.xori {umulhi_a_c0}, {c3_rng} : i32")
            lines.append(f"        {c2_new} = arith.xori {c2_x1}, {k1_rng} : i32")
            # c1 = B * c2_old
            lines.append(f"        {c1_new} = arith.muli {c_round_b}, {t_c2} : i32")
            # c3 = A * c0_old
            lines.append(f"        {c3_new} = arith.muli {c_round_a}, {t_c0} : i32")
            # raise key
            lines.append(f"        {k0_new} = arith.addi {k0_rng}, {c_key_a} : i32")
            lines.append(f"        {k1_new} = arith.addi {k1_rng}, {c_key_b} : i32")

            c0_rng = c0_new
            c1_rng = c1_new
            c2_rng = c2_new
            c3_rng = c3_new
            k0_rng = k0_new
            k1_rng = k1_new

        # uint_to_uniform_float (uint32 -> float32 in [0,1))
        z_i32 = _fresh("z")
        o_i32 = _fresh("o")
        neg_p = _fresh("negp")
        neg0 = _fresh("neg0")
        neg1 = _fresh("neg1")
        abs_i32 = _fresh("abs")
        abs_f = _fresh("absf")
        scale = _fresh("scale")
        rnd = _fresh("rnd")
        lines.append(f"        {z_i32} = arith.constant 0 : i32")
        lines.append(f"        {o_i32} = arith.constant 1 : i32")
        lines.append(f"        {neg_p} = arith.cmpi slt, {c0_rng}, {z_i32} : i32")
        lines.append(f"        {neg0} = arith.subi {z_i32}, {c0_rng} : i32")
        lines.append(f"        {neg1} = arith.subi {neg0}, {o_i32} : i32")
        lines.append(f"        {abs_i32} = arith.select {neg_p}, {neg1}, {c0_rng} : i32")
        lines.append(f"        {abs_f} = arith.sitofp {abs_i32} : i32 to f32")
        lines.append(f"        {scale} = arith.constant {scale_f32} : f32")
        lines.append(f"        {rnd} = arith.mulf {abs_f}, {scale}{fm} : f32")

        keep = _fresh("keep")
        inv = _fresh("inv")
        xs = _fresh("xs")
        z_f32 = _fresh("zf")
        outv = _fresh("outv")
        lines.append(f"        {keep} = arith.cmpf ogt, {rnd}, %p_v{fm} : f32")
        lines.append("        %c1f = arith.constant 1.0 : f32")
        lines.append(f"        {inv} = arith.subf %c1f, %p_v{fm} : f32")
        lines.append(f"        {xs} = arith.divf %xv, {inv}{fm} : f32")
        lines.append(f"        {z_f32} = arith.constant 0.0 : f32")
        lines.append(f"        {outv} = arith.select {keep}, {xs}, {z_f32} : f32")
        lines.append(f"        memref.store {outv}, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("      }")
    elif correlation_v1 is not None:
        kernel_kind = "correlation_v1"
        src0_name = str(correlation_v1["src0"])
        src1_name = str(correlation_v1["src1"])
        shift_name = str(correlation_v1["out_shift"])
        out_name2 = str(correlation_v1["out"])
        ic_dim = int(correlation_v1["IC"])
        oc_dim = int(correlation_v1["OC"])
        h_dim = int(correlation_v1["H"])
        w_dim = int(correlation_v1["W"])

        if str(arg_specs[src0_name].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_correlation expects i8 src0 tensor")
        if str(arg_specs[src1_name].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_correlation expects i8 src1 tensor")
        if str(arg_specs[shift_name].get("memref_elem_ty")) != "i32":
            raise RuntimeError("ai_bench_correlation expects i32 out_shift scalar")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_correlation expects i8 output tensor")

        src0_memref = str(arg_specs[src0_name]["memref"])
        src1_memref = str(arg_specs[src1_name]["memref"])
        shift_memref = str(arg_specs[shift_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_w = 128
        blocks_w = (int(w_dim) + int(block_w) - 1) // int(block_w)
        # Match the Triton kernel's axis semantics: (width_block, height, out_channel).
        launch_override = {"block": [int(block_w), 1, 1], "grid": [int(blocks_w), int(h_dim), int(oc_dim)]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_x = gpu.block_id x")
        lines.append("      %bid_h = gpu.block_id y")
        lines.append("      %bid_oc = gpu.block_id z")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cOC = arith.constant {int(oc_dim)} : index")
        lines.append(f"      %cIC = arith.constant {int(ic_dim)} : index")
        lines.append(f"      %cBlockW = arith.constant {int(block_w)} : index")
        lines.append(f"      %cHW = arith.constant {int(h_dim * w_dim)} : index")

        lines.append("      %pred_h = arith.cmpi ult, %bid_h, %cH : index")
        lines.append("      %pred_oc = arith.cmpi ult, %bid_oc, %cOC : index")
        lines.append("      %pred_ho = arith.andi %pred_h, %pred_oc : i1")
        lines.append("      scf.if %pred_ho {")
        lines.append("        %w_base = arith.muli %bid_x, %cBlockW : index")
        lines.append("        %w = arith.addi %w_base, %tid : index")
        lines.append("        %pred_w = arith.cmpi ult, %w, %cW : index")
        lines.append("        scf.if %pred_w {")
        lines.append("          %row = arith.muli %bid_h, %cW : index")
        lines.append("          %base0 = arith.addi %row, %w : index")
        lines.append("          %out_off = arith.muli %bid_oc, %cHW : index")
        lines.append("          %out_idx = arith.addi %out_off, %base0 : index")
        lines.append("          %pred_ge = arith.cmpi uge, %w, %bid_oc : index")
        lines.append("          scf.if %pred_ge {")
        lines.append("            %w_shift = arith.subi %w, %bid_oc : index")
        lines.append("            %base1 = arith.addi %row, %w_shift : index")

        lines.append("            %acc0 = arith.constant 0 : i32")
        lines.append("            %acc = scf.for %ic = %c0 to %cIC step %c1 iter_args(%a = %acc0) -> (i32) {")
        lines.append("              %ic_hw = arith.muli %ic, %cHW : index")
        lines.append("              %idx0 = arith.addi %ic_hw, %base0 : index")
        lines.append("              %idx1 = arith.addi %ic_hw, %base1 : index")
        lines.append(f"              %v0_i8 = memref.load {arg_ssa[src0_name]}[%idx0] : {src0_memref}")
        lines.append(f"              %v1_i8 = memref.load {arg_ssa[src1_name]}[%idx1] : {src1_memref}")
        lines.append("              %v0_i32 = arith.extsi %v0_i8 : i8 to i32")
        lines.append("              %v1_i32 = arith.extsi %v1_i8 : i8 to i32")
        lines.append("              %prod = arith.muli %v0_i32, %v1_i32 : i32")
        lines.append("              %a_next = arith.addi %a, %prod : i32")
        lines.append("              scf.yield %a_next : i32")
        lines.append("            }")

        lines.append(f"            %shift_v = memref.load {arg_ssa[shift_name]}[%c0] : {shift_memref}")
        lines.append("            %acc_shift = arith.shrsi %acc, %shift_v : i32")
        lines.append("            %out_i8 = arith.trunci %acc_shift : i32 to i8")
        lines.append(f"            memref.store %out_i8, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("          } else {")
        lines.append("            %z_i8 = arith.constant 0 : i8")
        lines.append(f"            memref.store %z_i8, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif resize_v1 is not None:
        kernel_kind = "resize_bilinear2x_hwfl7_v1"
        src_name = str(resize_v1["src"])
        out_name2 = str(resize_v1["out"])
        c_dim = int(resize_v1["C"])
        h_dim = int(resize_v1["H"])
        w_dim = int(resize_v1["W"])
        oh_dim = int(resize_v1["OH"])
        ow_dim = int(resize_v1["OW"])

        if str(arg_specs[src_name].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_resize expects i8 src tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_resize expects i8 output tensor")

        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_w = 128
        blocks_w = (int(ow_dim) + int(block_w) - 1) // int(block_w)
        launch_override = {"block": [int(block_w), 1, 1], "grid": [int(oh_dim), int(c_dim), int(blocks_w)]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_h = gpu.block_id x")
        lines.append("      %bid_c = gpu.block_id y")
        lines.append("      %bid_w = gpu.block_id z")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"      %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"      %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"      %cBlockW = arith.constant {int(block_w)} : index")
        lines.append(f"      %cHm1 = arith.constant {int(h_dim - 1)} : index")
        lines.append(f"      %cWm1 = arith.constant {int(w_dim - 1)} : index")
        lines.append(f"      %cHW = arith.constant {int(h_dim * w_dim)} : index")
        lines.append(f"      %cOHW = arith.constant {int(oh_dim * ow_dim)} : index")

        lines.append("      %pred_h = arith.cmpi ult, %bid_h, %cOH : index")
        lines.append("      %pred_c = arith.cmpi ult, %bid_c, %cC : index")
        lines.append("      %pred_hc = arith.andi %pred_h, %pred_c : i1")
        lines.append("      scf.if %pred_hc {")
        lines.append("        %w_base = arith.muli %bid_w, %cBlockW : index")
        lines.append("        %w = arith.addi %w_base, %tid : index")
        lines.append("        %pred_w = arith.cmpi ult, %w, %cOW : index")
        lines.append("        scf.if %pred_w {")
        # y0/y1 and h weights
        lines.append("          %y0 = arith.divui %bid_h, %c2 : index")
        lines.append("          %y1_tmp = arith.addi %y0, %c1 : index")
        lines.append("          %pred_y1 = arith.cmpi ult, %y1_tmp, %cH : index")
        lines.append("          %y1 = arith.select %pred_y1, %y1_tmp, %cHm1 : index")
        lines.append("          %hbit = arith.remui %bid_h, %c2 : index")
        lines.append("          %hbit_i32 = arith.index_cast %hbit : index to i32")
        lines.append("          %c64_i32 = arith.constant 64 : i32")
        lines.append("          %c128_i32 = arith.constant 128 : i32")
        lines.append("          %h1 = arith.muli %hbit_i32, %c64_i32 : i32")
        lines.append("          %h0 = arith.subi %c128_i32, %h1 : i32")

        # x0/x1 and w weights
        lines.append("          %x0 = arith.divui %w, %c2 : index")
        lines.append("          %x1_tmp = arith.addi %x0, %c1 : index")
        lines.append("          %pred_x1 = arith.cmpi ult, %x1_tmp, %cW : index")
        lines.append("          %x1 = arith.select %pred_x1, %x1_tmp, %cWm1 : index")
        lines.append("          %wbit = arith.remui %w, %c2 : index")
        lines.append("          %wbit_i32 = arith.index_cast %wbit : index to i32")
        lines.append("          %w1 = arith.muli %wbit_i32, %c64_i32 : i32")
        lines.append("          %w0 = arith.subi %c128_i32, %w1 : i32")

        # src base = bid_c * H * W
        lines.append("          %src_off = arith.muli %bid_c, %cHW : index")
        lines.append("          %y0w = arith.muli %y0, %cW : index")
        lines.append("          %y1w = arith.muli %y1, %cW : index")
        lines.append("          %row0 = arith.addi %src_off, %y0w : index")
        lines.append("          %row1 = arith.addi %src_off, %y1w : index")
        lines.append("          %idx_y0x0 = arith.addi %row0, %x0 : index")
        lines.append("          %idx_y0x1 = arith.addi %row0, %x1 : index")
        lines.append("          %idx_y1x0 = arith.addi %row1, %x0 : index")
        lines.append("          %idx_y1x1 = arith.addi %row1, %x1 : index")
        lines.append(f"          %y0x0_i8 = memref.load {arg_ssa[src_name]}[%idx_y0x0] : {src_memref}")
        lines.append(f"          %y0x1_i8 = memref.load {arg_ssa[src_name]}[%idx_y0x1] : {src_memref}")
        lines.append(f"          %y1x0_i8 = memref.load {arg_ssa[src_name]}[%idx_y1x0] : {src_memref}")
        lines.append(f"          %y1x1_i8 = memref.load {arg_ssa[src_name]}[%idx_y1x1] : {src_memref}")
        lines.append("          %y0x0_i32 = arith.extsi %y0x0_i8 : i8 to i32")
        lines.append("          %y0x1_i32 = arith.extsi %y0x1_i8 : i8 to i32")
        lines.append("          %y1x0_i32 = arith.extsi %y1x0_i8 : i8 to i32")
        lines.append("          %y1x1_i32 = arith.extsi %y1x1_i8 : i8 to i32")

        lines.append("          %p00 = arith.muli %y0x0_i32, %w0 : i32")
        lines.append("          %p01 = arith.muli %y0x1_i32, %w1 : i32")
        lines.append("          %s0 = arith.addi %p00, %p01 : i32")
        lines.append("          %p10 = arith.muli %y1x0_i32, %w0 : i32")
        lines.append("          %p11 = arith.muli %y1x1_i32, %w1 : i32")
        lines.append("          %s1 = arith.addi %p10, %p11 : i32")
        lines.append("          %c7_i32 = arith.constant 7 : i32")
        lines.append("          %sum1 = arith.shrsi %s0, %c7_i32 : i32")
        lines.append("          %sum2 = arith.shrsi %s1, %c7_i32 : i32")
        lines.append("          %q0 = arith.muli %sum1, %h0 : i32")
        lines.append("          %q1 = arith.muli %sum2, %h1 : i32")
        lines.append("          %q = arith.addi %q0, %q1 : i32")
        lines.append("          %out_i32 = arith.shrsi %q, %c7_i32 : i32")
        lines.append("          %out_i8 = arith.trunci %out_i32 : i32 to i8")

        # out_idx = bid_c * (OH*OW) + bid_h * OW + w
        lines.append("          %out_off = arith.muli %bid_c, %cOHW : index")
        lines.append("          %out_row = arith.muli %bid_h, %cOW : index")
        lines.append("          %out_base = arith.addi %out_off, %out_row : index")
        lines.append("          %out_idx = arith.addi %out_base, %w : index")
        lines.append(f"          memref.store %out_i8, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif rope_v1 is not None:
        kernel_kind = "rope_v1"
        inp_name = str(rope_v1["inp"])
        cos_name = str(rope_v1["cos"])
        sin_name = str(rope_v1["sin"])
        out_name2 = str(rope_v1["out"])
        seq_len = int(rope_v1["SEQ_LEN"])
        batch_num = int(rope_v1["BATCH_NUM"])
        head_num = int(rope_v1["HEAD_NUM"])
        head_dim = int(rope_v1["HEAD_DIM"])
        half = int(rope_v1["HALF"])

        if str(arg_specs[inp_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_rope expects f32 input tensor")
        if str(arg_specs[cos_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_rope expects f32 cos tensor")
        if str(arg_specs[sin_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_rope expects f32 sin tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "f32":
            raise RuntimeError("ai_bench_rope expects f32 output tensor")

        in_memref = str(arg_specs[inp_name]["memref"])
        cos_memref = str(arg_specs[cos_name]["memref"])
        sin_memref = str(arg_specs[sin_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        # Match the Triton kernel's grid mapping: (head, batch, seq).
        launch_override = {"block": [256, 1, 1], "grid": [int(head_num), int(batch_num), int(seq_len)]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_head = gpu.block_id x")
        lines.append("      %bid_batch = gpu.block_id y")
        lines.append("      %bid_seq = gpu.block_id z")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cSeq = arith.constant {int(seq_len)} : index")
        lines.append(f"      %cBatch = arith.constant {int(batch_num)} : index")
        lines.append(f"      %cHead = arith.constant {int(head_num)} : index")
        lines.append(f"      %cHeadDim = arith.constant {int(head_dim)} : index")
        lines.append(f"      %cHalf = arith.constant {int(half)} : index")
        lines.append("      %pred_head = arith.cmpi ult, %bid_head, %cHead : index")
        lines.append("      %pred_batch = arith.cmpi ult, %bid_batch, %cBatch : index")
        lines.append("      %pred_seq = arith.cmpi ult, %bid_seq, %cSeq : index")
        lines.append("      %pred_hb = arith.andi %pred_head, %pred_batch : i1")
        lines.append("      %pred_all = arith.andi %pred_hb, %pred_seq : i1")
        lines.append("      scf.if %pred_all {")
        lines.append("        %t0 = arith.muli %bid_seq, %cBatch : index")
        lines.append("        %t1 = arith.addi %t0, %bid_batch : index")
        lines.append("        %t2 = arith.muli %t1, %cHead : index")
        lines.append("        %t3 = arith.addi %t2, %bid_head : index")
        lines.append("        %base = arith.muli %t3, %cHeadDim : index")
        lines.append("        %cos_base = arith.muli %bid_seq, %cHalf : index")
        lines.append("        scf.for %j = %tid to %cHalf step %bdim {")
        lines.append("          %idx1 = arith.addi %base, %j : index")
        lines.append("          %j2 = arith.addi %j, %cHalf : index")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append("          %cidx = arith.addi %cos_base, %j : index")
        lines.append(f"          %x1 = memref.load {arg_ssa[inp_name]}[%idx1] : {in_memref}")
        lines.append(f"          %x2 = memref.load {arg_ssa[inp_name]}[%idx2] : {in_memref}")
        # Avoid SSA name collisions with memref arguments (e.g. `%cos`/`%sin`).
        lines.append(f"          %cos_v = memref.load {arg_ssa[cos_name]}[%cidx] : {cos_memref}")
        lines.append(f"          %sin_v = memref.load {arg_ssa[sin_name]}[%cidx] : {sin_memref}")
        lines.append(f"          %x1c = arith.mulf %x1, %cos_v{fm} : f32")
        lines.append(f"          %x2s = arith.mulf %x2, %sin_v{fm} : f32")
        lines.append(f"          %y1 = arith.subf %x1c, %x2s{fm} : f32")
        lines.append(f"          %x1s = arith.mulf %x1, %sin_v{fm} : f32")
        lines.append(f"          %x2c = arith.mulf %x2, %cos_v{fm} : f32")
        lines.append(f"          %y2 = arith.addf %x1s, %x2c{fm} : f32")
        lines.append(f"          memref.store %y1, {arg_ssa[out_name2]}[%idx1] : {out_memref}")
        lines.append(f"          memref.store %y2, {arg_ssa[out_name2]}[%idx2] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif warp_v1 is not None:
        kernel_kind = "warp_v1"
        src_name = str(warp_v1["src"])
        offset_name = str(warp_v1["offset"])
        out_name2 = str(warp_v1["out"])
        c_dim = int(warp_v1["C"])
        h_dim = int(warp_v1["H"])
        w_dim = int(warp_v1["W"])

        if str(arg_specs[src_name].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_warp expects i8 src tensor")
        if str(arg_specs[offset_name].get("memref_elem_ty")) != "i16":
            raise RuntimeError("ai_bench_warp expects i16 offset tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "i8":
            raise RuntimeError("ai_bench_warp expects i8 output tensor")

        src_memref = str(arg_specs[src_name]["memref"])
        offset_memref = str(arg_specs[offset_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_w = 128
        blocks_w = (int(w_dim) + int(block_w) - 1) // int(block_w)
        launch_override = {"block": [int(block_w), 1, 1], "grid": [int(h_dim), int(c_dim), int(blocks_w)]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_h = gpu.block_id x")
        lines.append("      %bid_c = gpu.block_id y")
        lines.append("      %bid_w = gpu.block_id z")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"      %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"      %cBlockW = arith.constant {int(block_w)} : index")
        lines.append("      %pred_h = arith.cmpi ult, %bid_h, %cH : index")
        lines.append("      %pred_c = arith.cmpi ult, %bid_c, %cC : index")
        lines.append("      %pred_hc = arith.andi %pred_h, %pred_c : i1")
        lines.append("      scf.if %pred_hc {")
        lines.append("        %w_base = arith.muli %bid_w, %cBlockW : index")
        lines.append("        %w = arith.addi %w_base, %tid : index")
        lines.append("        %pred_w = arith.cmpi ult, %w, %cW : index")
        lines.append("        scf.if %pred_w {")
        lines.append("          %off_base = arith.muli %bid_h, %cW : index")
        lines.append("          %off_idx = arith.addi %off_base, %w : index")
        lines.append(f"          %off_val = memref.load {arg_ssa[offset_name]}[%off_idx] : {offset_memref}")
        lines.append("          %c8_i16 = arith.constant 8 : i16")
        lines.append("          %off_hi = arith.shrsi %off_val, %c8_i16 : i16")
        lines.append("          %off_int = arith.trunci %off_hi : i16 to i8")
        lines.append("          %off_lsh = arith.shli %off_val, %c8_i16 : i16")
        lines.append("          %off_rsh = arith.shrsi %off_lsh, %c8_i16 : i16")
        lines.append("          %off_frac = arith.trunci %off_rsh : i16 to i8")
        lines.append("          %w_i32 = arith.index_cast %w : index to i32")
        lines.append("          %w_i8 = arith.trunci %w_i32 : i32 to i8")
        lines.append("          %right_i8 = arith.subi %w_i8, %off_int : i8")
        lines.append("          %c1_i8 = arith.constant 1 : i8")
        lines.append("          %left_i8 = arith.subi %right_i8, %c1_i8 : i8")
        lines.append("          %c0_i8 = arith.constant 0 : i8")
        lines.append("          %pred_right = arith.cmpi sge, %right_i8, %c0_i8 : i8")
        lines.append("          %pred_left = arith.cmpi sge, %left_i8, %c0_i8 : i8")
        lines.append("          %mask_right = arith.andi %pred_w, %pred_right : i1")
        lines.append("          %mask_left = arith.andi %pred_w, %pred_left : i1")
        lines.append("          %t0 = arith.muli %bid_c, %cH : index")
        lines.append("          %t1 = arith.addi %t0, %bid_h : index")
        lines.append("          %src_base = arith.muli %t1, %cW : index")
        lines.append("          %z_i8 = arith.constant 0 : i8")
        lines.append("          %right_val = scf.if %mask_right -> (i8) {")
        lines.append("            %ri32 = arith.extsi %right_i8 : i8 to i32")
        lines.append("            %ridx = arith.index_cast %ri32 : i32 to index")
        lines.append("            %src_idx = arith.addi %src_base, %ridx : index")
        lines.append(f"            %rv = memref.load {arg_ssa[src_name]}[%src_idx] : {src_memref}")
        lines.append("            scf.yield %rv : i8")
        lines.append("          } else {")
        lines.append("            scf.yield %z_i8 : i8")
        lines.append("          }")
        lines.append("          %left_val = scf.if %mask_left -> (i8) {")
        lines.append("            %li32 = arith.extsi %left_i8 : i8 to i32")
        lines.append("            %lidx = arith.index_cast %li32 : i32 to index")
        lines.append("            %src_idx2 = arith.addi %src_base, %lidx : index")
        lines.append(f"            %lv = memref.load {arg_ssa[src_name]}[%src_idx2] : {src_memref}")
        lines.append("            scf.yield %lv : i8")
        lines.append("          } else {")
        lines.append("            scf.yield %z_i8 : i8")
        lines.append("          }")
        lines.append("          %rv16 = arith.extsi %right_val : i8 to i16")
        lines.append("          %lv16 = arith.extsi %left_val : i8 to i16")
        lines.append("          %rv16_sh = arith.shli %rv16, %c8_i16 : i16")
        lines.append("          %diff_i8 = arith.subi %left_val, %right_val : i8")
        lines.append("          %diff16 = arith.extsi %diff_i8 : i8 to i16")
        lines.append("          %frac16 = arith.extsi %off_frac : i8 to i16")
        lines.append("          %prod = arith.muli %diff16, %frac16 : i16")
        lines.append("          %acc = arith.addi %rv16_sh, %prod : i16")
        lines.append("          %out16 = arith.shrsi %acc, %c8_i16 : i16")
        lines.append("          %out8 = arith.trunci %out16 : i16 to i8")
        lines.append("          %out_idx = arith.addi %src_base, %w : index")
        lines.append(f"          memref.store %out8, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif matvec_v1 is not None:
        kernel_kind = "matvec_v1"
        a_name = str(matvec_v1["A"])
        b_name = str(matvec_v1["B"])
        mv_out_name = str(matvec_v1["mv_out"])
        out_name2 = str(matvec_v1["out"])
        n_dim = int(matvec_v1["N"])
        k_dim_red = int(matvec_v1["M"])
        if n_dim <= 0 or k_dim_red <= 0:
            raise RuntimeError(f"invalid matvec dims: N={n_dim} K={k_dim_red}")
        if str(arg_specs[a_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matvec expects f32 A tensor")
        if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matvec expects f32 B tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matvec expects f32 output tensor")

        a_memref = str(arg_specs[a_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(n_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"
        cuda_real_mlir_matmul_cfg = {"kind": "matvec_v1", "N": int(n_dim), "K": int(k_dim_red), "block_x": 256}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cK_red = arith.constant {int(k_dim_red)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cN : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cK_red : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial = scf.for %j = %tid to %cK_red step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %a = memref.load {arg_ssa[a_name]}[%idx] : {a_memref}")
        lines.append(f"          %b = memref.load {arg_ssa[b_name]}[%j] : {b_memref}")
        lines.append(f"          %p = arith.mulf %a, %b{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %p{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
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
            precomputed={mv_out_name: ("%sum0", "f32")},
            skip_rank2_outputs=True,
        )
        for l in body_lines:
            lines.append("  " + l)
        lines.append("        }")
        lines.append("      }")
    elif (
        matmul_v1 is not None
        and int(matmul_v1.get("BN") or 0) % 4 == 0
        and int(matmul_v1.get("BM") or 0) * (int(matmul_v1.get("BN") or 0) // 4) == 256
    ):
        kernel_kind = "matmul_tile_v2"
        a_name = str(matmul_v1["A"])
        b_name = str(matmul_v1["B"])
        out_name2 = str(matmul_v1["out"])
        bias_name = matmul_v1.get("bias")
        add_inp_name = matmul_v1.get("add_inp")
        alpha_name = matmul_v1.get("alpha")
        beta_name = matmul_v1.get("beta")
        row_mask_name = matmul_v1.get("row_mask")
        col_mask_name = matmul_v1.get("col_mask")
        relu = bool(matmul_v1.get("relu") or False)

        batch_dim = int(matmul_v1.get("BATCH") or 1)
        if int(batch_dim) > 1 and str(intent_name) == "bmm3d":
            kernel_kind = "bmm_tile_v2"
        elif int(batch_dim) > 1 and str(intent_name) == "baddbmm3d":
            kernel_kind = "baddbmm_tile_v2"

        m_dim = int(matmul_v1["M"])
        n_dim = int(matmul_v1["N"])
        k_dim = int(matmul_v1["K"])
        bm = int(matmul_v1["BM"])
        bn = int(matmul_v1["BN"])
        bk = int(matmul_v1["BK"])
        if bm <= 0 or bn <= 0 or bk <= 0:
            raise RuntimeError(f"invalid matmul tile: BM={bm} BN={bn} BK={bk}")

        if str(arg_specs[a_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 A tensor")
        if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 B tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 output tensor")
        if bias_name is not None and str(arg_specs[str(bias_name)].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul bias expects f32 tensor")
        if add_inp_name is not None:
            if str(arg_specs[str(add_inp_name)].get("memref_elem_ty")) != "f32":
                raise RuntimeError("matmul epilogue expects f32 input tensor")
            add_inp_dims = list(arg_specs[str(add_inp_name)].get("dims") or [])
            if int(batch_dim) > 1:
                if add_inp_dims != [int(batch_dim), int(m_dim), int(n_dim)]:
                    raise RuntimeError("baddbmm3d expects input dims [BATCH,M,N]")
            else:
                if add_inp_dims != [int(m_dim), int(n_dim)]:
                    raise RuntimeError("addmm2d expects input dims [M,N]")
            if alpha_name is None or beta_name is None:
                raise RuntimeError("addmm2d requires alpha/beta scalars")
            if not bool(arg_specs.get(str(alpha_name), {}).get("scalar")) or str(
                arg_specs.get(str(alpha_name), {}).get("memref_elem_ty") or ""
            ) != "f32":
                raise RuntimeError("addmm2d expects f32 scalar alpha")
            if not bool(arg_specs.get(str(beta_name), {}).get("scalar")) or str(
                arg_specs.get(str(beta_name), {}).get("memref_elem_ty") or ""
            ) != "f32":
                raise RuntimeError("addmm2d expects f32 scalar beta")
        if row_mask_name is not None and str(arg_specs[str(row_mask_name)].get("memref_elem_ty")) != "i1":
            if str(arg_specs[str(row_mask_name)].get("memref_elem_ty")) != "i8":
                raise RuntimeError("matmul row_mask expects bool-like tensor (i8 ABI) for cuda real-mlir wave")
        if col_mask_name is not None and str(arg_specs[str(col_mask_name)].get("memref_elem_ty")) != "i1":
            if str(arg_specs[str(col_mask_name)].get("memref_elem_ty")) != "i8":
                raise RuntimeError("matmul col_mask expects bool-like tensor (i8 ABI) for cuda real-mlir wave")

        a_memref = str(arg_specs[a_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        bias_memref = str(arg_specs[str(bias_name)]["memref"]) if bias_name is not None else ""
        add_inp_memref = str(arg_specs[str(add_inp_name)]["memref"]) if add_inp_name is not None else ""
        alpha_memref = str(arg_specs[str(alpha_name)]["memref"]) if alpha_name is not None else ""
        beta_memref = str(arg_specs[str(beta_name)]["memref"]) if beta_name is not None else ""
        row_mask_memref = str(arg_specs[str(row_mask_name)]["memref"]) if row_mask_name is not None else ""
        col_mask_memref = str(arg_specs[str(col_mask_name)]["memref"]) if col_mask_name is not None else ""

        cuda_real_mlir_matmul_cfg = {"BM": int(bm), "BN": int(bn), "BK": int(bk)}
        if int(batch_dim) > 1:
            cuda_real_mlir_matmul_cfg["BATCH"] = int(batch_dim)

        threads = 256
        grid_x = (int(n_dim) + int(bn) - 1) // int(bn)
        grid_y = (int(m_dim) + int(bm) - 1) // int(bm)
        launch_override = {"block": [int(threads), 1, 1], "grid": [int(grid_x), int(grid_y), int(batch_dim)]}

        tile_a_elems = int(bm) * int(bk)
        tile_b_elems = int(bk) * int(bn)
        sh_elems = int(tile_a_elems + tile_b_elems)
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"
        offset_b = int(tile_a_elems)

        col_groups = int(bn) // 4
        vec4_ty = "vector<4xf32>"
        fast_no_bounds = (
            int(m_dim) % int(bm) == 0
            and int(n_dim) % int(bn) == 0
            and int(k_dim) % int(bk) == 0
            and int(tile_a_elems) % int(threads) == 0
            and int(tile_b_elems) % int(threads) == 0
        )

        # Kernel mapping: one CTA per (BM x BN) output tile, 256 threads.
        # Each thread computes 1 row x 4 contiguous columns (vector<4xf32> accumulator).
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_n = gpu.block_id x")
        lines.append("      %bid_m = gpu.block_id y")
        lines.append("      %bid_b = gpu.block_id z")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append("      %c4 = arith.constant 4 : index")
        lines.append("      %c3 = arith.constant 3 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cK = arith.constant {int(k_dim)} : index")
        lines.append(f"      %cBM = arith.constant {int(bm)} : index")
        lines.append(f"      %cBN = arith.constant {int(bn)} : index")
        lines.append(f"      %cBK = arith.constant {int(bk)} : index")
        lines.append(f"      %cColGroups = arith.constant {int(col_groups)} : index")
        lines.append(f"      %cTileA = arith.constant {int(tile_a_elems)} : index")
        lines.append(f"      %cTileB = arith.constant {int(tile_b_elems)} : index")
        lines.append(f"      %cOffsetB = arith.constant {int(offset_b)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %base_m = arith.muli %bid_m, %cBM : index")
        lines.append("      %base_n = arith.muli %bid_n, %cBN : index")
        lines.append("      %batch_m = arith.muli %bid_b, %cM : index")
        lines.append("      %batch_k = arith.muli %bid_b, %cK : index")
        lines.append(f"      %sh0 = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"      %sh = memref.assume_alignment %sh0, 16 : {shared_global_memref_ty}")

        # Thread mapping within the tile.
        lines.append("      %row = arith.divui %tid, %cColGroups : index")
        lines.append("      %colg = arith.remui %tid, %cColGroups : index")
        lines.append("      %col_base = arith.muli %colg, %c4 : index")
        lines.append("      %gm = arith.addi %base_m, %row : index")
        lines.append("      %gn0 = arith.addi %base_n, %col_base : index")

        acc_init = _fresh("acc_init")
        lines.append(f"      {acc_init} = vector.splat %c0f : {vec4_ty}")
        acc_out = _fresh("acc_vec")
        lines.append(
            f"      {acc_out} = scf.for %k0 = %c0 to %cK step %cBK iter_args(%acc = {acc_init}) -> ({vec4_ty}) {{"
        )

        # Cooperative A tile load into shared[0:tile_a_elems).
        loads_a = (int(tile_a_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_a)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_a")
            idx = _fresh("idx_a")
            lines.append(f"        {c_off} = arith.constant {int(off)} : index")
            lines.append(f"        {idx} = arith.addi %tid, {c_off} : index")
            if fast_no_bounds:
                row = _fresh("a_row")
                kk = _fresh("a_k")
                g_row = _fresh("a_gm")
                g_k = _fresh("a_gk")
                g_row_b = _fresh("a_gm_b")
                mul = _fresh("a_mul")
                g_idx = _fresh("a_idx")
                v = _fresh("a_load")
                lines.append(f"        {row} = arith.divui {idx}, %cBK : index")
                lines.append(f"        {kk} = arith.remui {idx}, %cBK : index")
                lines.append(f"        {g_row} = arith.addi %base_m, {row} : index")
                lines.append(f"        {g_k} = arith.addi %k0, {kk} : index")
                lines.append(f"        {g_row_b} = arith.addi %batch_m, {g_row} : index")
                lines.append(f"        {mul} = arith.muli {g_row_b}, %cK : index")
                lines.append(f"        {g_idx} = arith.addi {mul}, {g_k} : index")
                lines.append(f"        {v} = memref.load {arg_ssa[a_name]}[{g_idx}] : {a_memref}")
                lines.append(f"        memref.store {v}, %sh[{idx}] : {shared_global_memref_ty}")
            else:
                pred = _fresh("pred_a")
                lines.append(f"        {pred} = arith.cmpi ult, {idx}, %cTileA : index")
                lines.append(f"        scf.if {pred} {{")
                row = _fresh("a_row")
                kk = _fresh("a_k")
                g_row = _fresh("a_gm")
                g_k = _fresh("a_gk")
                p_row = _fresh("a_pr")
                p_k = _fresh("a_pk")
                p_ok = _fresh("a_p")
                val = _fresh("a_val")
                lines.append(f"          {row} = arith.divui {idx}, %cBK : index")
                lines.append(f"          {kk} = arith.remui {idx}, %cBK : index")
                lines.append(f"          {g_row} = arith.addi %base_m, {row} : index")
                lines.append(f"          {g_k} = arith.addi %k0, {kk} : index")
                lines.append(f"          {p_row} = arith.cmpi ult, {g_row}, %cM : index")
                lines.append(f"          {p_k} = arith.cmpi ult, {g_k}, %cK : index")
                lines.append(f"          {p_ok} = arith.andi {p_row}, {p_k} : i1")
                lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
                mul = _fresh("a_mul")
                g_idx = _fresh("a_idx")
                g_row_b = _fresh("a_gm_b")
                lines.append(f"            {g_row_b} = arith.addi %batch_m, {g_row} : index")
                lines.append(f"            {mul} = arith.muli {g_row_b}, %cK : index")
                lines.append(f"            {g_idx} = arith.addi {mul}, {g_k} : index")
                v = _fresh("a_load")
                lines.append(f"            {v} = memref.load {arg_ssa[a_name]}[{g_idx}] : {a_memref}")
                lines.append(f"            scf.yield {v} : f32")
                lines.append("          } else {")
                lines.append("            scf.yield %c0f : f32")
                lines.append("          }")
                lines.append(f"          memref.store {val}, %sh[{idx}] : {shared_global_memref_ty}")
                lines.append("        }")

        # Cooperative B tile load into shared[offset_b:).
        loads_b = (int(tile_b_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_b)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_b")
            idx = _fresh("idx_b")
            lines.append(f"        {c_off} = arith.constant {int(off)} : index")
            lines.append(f"        {idx} = arith.addi %tid, {c_off} : index")
            if fast_no_bounds:
                kk = _fresh("b_k")
                col = _fresh("b_col")
                g_k = _fresh("b_gk")
                g_col = _fresh("b_gn")
                g_k_b = _fresh("b_gk_b")
                mul = _fresh("b_mul")
                g_idx = _fresh("b_idx")
                v = _fresh("b_load")
                sh_idx = _fresh("b_sh")
                lines.append(f"        {kk} = arith.divui {idx}, %cBN : index")
                lines.append(f"        {col} = arith.remui {idx}, %cBN : index")
                lines.append(f"        {g_k} = arith.addi %k0, {kk} : index")
                lines.append(f"        {g_col} = arith.addi %base_n, {col} : index")
                lines.append(f"        {g_k_b} = arith.addi %batch_k, {g_k} : index")
                lines.append(f"        {mul} = arith.muli {g_k_b}, %cN : index")
                lines.append(f"        {g_idx} = arith.addi {mul}, {g_col} : index")
                lines.append(f"        {v} = memref.load {arg_ssa[b_name]}[{g_idx}] : {b_memref}")
                lines.append(f"        {sh_idx} = arith.addi {idx}, %cOffsetB : index")
                lines.append(f"        memref.store {v}, %sh[{sh_idx}] : {shared_global_memref_ty}")
            else:
                pred = _fresh("pred_b")
                lines.append(f"        {pred} = arith.cmpi ult, {idx}, %cTileB : index")
                lines.append(f"        scf.if {pred} {{")
                kk = _fresh("b_k")
                col = _fresh("b_col")
                g_k = _fresh("b_gk")
                g_col = _fresh("b_gn")
                p_k = _fresh("b_pk")
                p_col = _fresh("b_pn")
                p_ok = _fresh("b_p")
                val = _fresh("b_val")
                sh_idx = _fresh("b_sh")
                lines.append(f"          {kk} = arith.divui {idx}, %cBN : index")
                lines.append(f"          {col} = arith.remui {idx}, %cBN : index")
                lines.append(f"          {g_k} = arith.addi %k0, {kk} : index")
                lines.append(f"          {g_col} = arith.addi %base_n, {col} : index")
                lines.append(f"          {p_k} = arith.cmpi ult, {g_k}, %cK : index")
                lines.append(f"          {p_col} = arith.cmpi ult, {g_col}, %cN : index")
                lines.append(f"          {p_ok} = arith.andi {p_k}, {p_col} : i1")
                lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
                mul = _fresh("b_mul")
                g_idx = _fresh("b_idx")
                g_k_b = _fresh("b_gk_b")
                lines.append(f"            {g_k_b} = arith.addi %batch_k, {g_k} : index")
                lines.append(f"            {mul} = arith.muli {g_k_b}, %cN : index")
                lines.append(f"            {g_idx} = arith.addi {mul}, {g_col} : index")
                v = _fresh("b_load")
                lines.append(f"            {v} = memref.load {arg_ssa[b_name]}[{g_idx}] : {b_memref}")
                lines.append(f"            scf.yield {v} : f32")
                lines.append("          } else {")
                lines.append("            scf.yield %c0f : f32")
                lines.append("          }")
                lines.append(f"          {sh_idx} = arith.addi {idx}, %cOffsetB : index")
                lines.append(f"          memref.store {val}, %sh[{sh_idx}] : {shared_global_memref_ty}")
                lines.append("        }")

        lines.append("        gpu.barrier")

        # Compute this K tile.
        acc_k = _fresh("acc_k")
        lines.append(
            f"        {acc_k} = scf.for %ki = %c0 to %cBK step %c1 iter_args(%acc_inner = %acc) -> ({vec4_ty}) {{"
        )

        a_mul = _fresh("a_mul")
        a_idx = _fresh("a_idx")
        a_val = _fresh("a")
        a_splat = _fresh("a_splat")
        lines.append(f"          {a_mul} = arith.muli %row, %cBK : index")
        lines.append(f"          {a_idx} = arith.addi {a_mul}, %ki : index")
        lines.append(f"          {a_val} = memref.load %sh[{a_idx}] : {shared_global_memref_ty}")
        lines.append(f"          {a_splat} = vector.splat {a_val} : {vec4_ty}")

        b_mul = _fresh("b_mul")
        b_idx0 = _fresh("b_idx0")
        b_idx = _fresh("b_idx")
        b_vec = _fresh("b_vec")
        lines.append(f"          {b_mul} = arith.muli %ki, %cBN : index")
        lines.append(f"          {b_idx0} = arith.addi {b_mul}, %col_base : index")
        lines.append(f"          {b_idx} = arith.addi {b_idx0}, %cOffsetB : index")
        lines.append(f"          {b_vec} = vector.load %sh[{b_idx}] : {shared_global_memref_ty}, {vec4_ty}")

        acc_new = _fresh("acc")
        # Force a real PTX `fma.rn.f32` (avoid libdevice calls like `__nv_fmaf`).
        lines.append(
            f"          {acc_new} = llvm.intr.fma({a_splat}, {b_vec}, %acc_inner) "
            f": ({vec4_ty}, {vec4_ty}, {vec4_ty}) -> {vec4_ty}"
        )
        lines.append(f"          scf.yield {acc_new} : {vec4_ty}")
        lines.append("        }")

        lines.append("        gpu.barrier")
        lines.append(f"        scf.yield {acc_k} : {vec4_ty}")
        lines.append("      }")

        # Store outputs (fused epilogue).
        if fast_no_bounds:
            val_vec = str(acc_out)
            gm_b = _fresh("gm_b")
            mul_o = _fresh("mul_o")
            idx_o = _fresh("idx_o")
            lines.append(f"      {gm_b} = arith.addi %batch_m, %gm : index")
            lines.append(f"      {mul_o} = arith.muli {gm_b}, %cN : index")
            lines.append(f"      {idx_o} = arith.addi {mul_o}, %gn0 : index")
            if alpha_name is not None:
                alpha_s = _fresh("alpha")
                alpha_v = _fresh("alpha_v")
                val2 = _fresh("val_alpha")
                lines.append(f"      {alpha_s} = memref.load {arg_ssa[str(alpha_name)]}[%c0] : {alpha_memref}")
                lines.append(f"      {alpha_v} = vector.splat {alpha_s} : {vec4_ty}")
                lines.append(f"      {val2} = arith.mulf {val_vec}, {alpha_v}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if add_inp_name is not None:
                beta_s = _fresh("beta")
                beta_v = _fresh("beta_v")
                inp_vec = _fresh("inp_vec")
                inp_scaled = _fresh("inp_scaled")
                val2 = _fresh("val_addmm")
                lines.append(f"      {beta_s} = memref.load {arg_ssa[str(beta_name)]}[%c0] : {beta_memref}")
                lines.append(f"      {beta_v} = vector.splat {beta_s} : {vec4_ty}")
                lines.append(f"      {inp_vec} = vector.load {arg_ssa[str(add_inp_name)]}[{idx_o}] : {add_inp_memref}, {vec4_ty}")
                lines.append(f"      {inp_scaled} = arith.mulf {inp_vec}, {beta_v}{fm} : {vec4_ty}")
                lines.append(f"      {val2} = arith.addf {val_vec}, {inp_scaled}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if bias_name is not None:
                bias_vec = _fresh("bias_vec")
                val2 = _fresh("val_bias")
                lines.append(f"      {bias_vec} = vector.load {arg_ssa[str(bias_name)]}[%gn0] : {bias_memref}, {vec4_ty}")
                lines.append(f"      {val2} = arith.addf {val_vec}, {bias_vec}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if relu:
                zvec = _fresh("zvec")
                relu_v = _fresh("relu")
                lines.append(f"      {zvec} = vector.splat %c0f : {vec4_ty}")
                lines.append(f"      {relu_v} = arith.maximumf {val_vec}, {zvec}{fm} : {vec4_ty}")
                val_vec = str(relu_v)
            if row_mask_name is not None and col_mask_name is not None:
                row_mask_elem_ty = str(arg_specs[str(row_mask_name)].get("memref_elem_ty") or "i8")
                col_mask_elem_ty = str(arg_specs[str(col_mask_name)].get("memref_elem_ty") or "i8")
                rm = _fresh("rm")
                rm0 = _fresh("rm0")
                rm1 = _fresh("rm1")
                rmv = _fresh("rmv")
                cm = _fresh("cm")
                cm0 = _fresh("cm0")
                cm0v = _fresh("cm0v")
                cm1 = _fresh("cm1")
                cond = _fresh("cond")
                zvec = _fresh("zvec_m")
                sel = _fresh("sel")
                lines.append(f"      {rm} = memref.load {arg_ssa[str(row_mask_name)]}[%gm] : {row_mask_memref}")
                lines.append(f"      {rm0} = arith.constant 0 : {row_mask_elem_ty}")
                lines.append(f"      {rm1} = arith.cmpi ne, {rm}, {rm0} : {row_mask_elem_ty}")
                lines.append(f"      {rmv} = vector.splat {rm1} : vector<4xi1>")
                lines.append(f"      {cm} = vector.load {arg_ssa[str(col_mask_name)]}[%gn0] : {col_mask_memref}, vector<4x{col_mask_elem_ty}>")
                lines.append(f"      {cm0} = arith.constant 0 : {col_mask_elem_ty}")
                lines.append(f"      {cm0v} = vector.splat {cm0} : vector<4x{col_mask_elem_ty}>")
                lines.append(f"      {cm1} = arith.cmpi ne, {cm}, {cm0v} : vector<4x{col_mask_elem_ty}>")
                lines.append(f"      {cond} = arith.andi {rmv}, {cm1} : vector<4xi1>")
                lines.append(f"      {zvec} = vector.splat %c0f : {vec4_ty}")
                cur_vec = str(zvec)
                for lane in range(4):
                    cond_lane = _fresh(f"cond{lane}")
                    val_lane = _fresh(f"val{lane}")
                    sel_lane = _fresh(f"sel{lane}")
                    next_vec = _fresh(f"v{lane}")
                    lines.append(f"      {cond_lane} = vector.extract {cond}[{lane}] : i1 from vector<4xi1>")
                    lines.append(f"      {val_lane} = vector.extract {val_vec}[{lane}] : f32 from {vec4_ty}")
                    lines.append(f"      {sel_lane} = arith.select {cond_lane}, {val_lane}, %c0f : f32")
                    lines.append(f"      {next_vec} = vector.insert {sel_lane}, {cur_vec}[{lane}] : f32 into {vec4_ty}")
                    cur_vec = str(next_vec)
                val_vec = str(cur_vec)

            lines.append(f"      vector.store {val_vec}, {arg_ssa[out_name2]}[{idx_o}] : {out_memref}, {vec4_ty}")
        else:
            lines.append("      %p_row = arith.cmpi ult, %gm, %cM : index")
            lines.append("      %gn3 = arith.addi %gn0, %c3 : index")
            lines.append("      %p_col_vec = arith.cmpi ult, %gn3, %cN : index")
            lines.append("      %p_ok_vec = arith.andi %p_row, %p_col_vec : i1")
            lines.append("      scf.if %p_ok_vec {")

            gm_b = _fresh("gm_b")
            mul_o = _fresh("mul_o")
            idx_o = _fresh("idx_o")
            lines.append(f"        {gm_b} = arith.addi %batch_m, %gm : index")
            lines.append(f"        {mul_o} = arith.muli {gm_b}, %cN : index")
            lines.append(f"        {idx_o} = arith.addi {mul_o}, %gn0 : index")
            val_vec = str(acc_out)
            if alpha_name is not None:
                alpha_s = _fresh("alpha")
                alpha_v = _fresh("alpha_v")
                val2 = _fresh("val_alpha")
                lines.append(f"        {alpha_s} = memref.load {arg_ssa[str(alpha_name)]}[%c0] : {alpha_memref}")
                lines.append(f"        {alpha_v} = vector.splat {alpha_s} : {vec4_ty}")
                lines.append(f"        {val2} = arith.mulf {val_vec}, {alpha_v}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if add_inp_name is not None:
                beta_s = _fresh("beta")
                beta_v = _fresh("beta_v")
                inp_vec = _fresh("inp_vec")
                inp_scaled = _fresh("inp_scaled")
                val2 = _fresh("val_addmm")
                lines.append(f"        {beta_s} = memref.load {arg_ssa[str(beta_name)]}[%c0] : {beta_memref}")
                lines.append(f"        {beta_v} = vector.splat {beta_s} : {vec4_ty}")
                lines.append(f"        {inp_vec} = vector.load {arg_ssa[str(add_inp_name)]}[{idx_o}] : {add_inp_memref}, {vec4_ty}")
                lines.append(f"        {inp_scaled} = arith.mulf {inp_vec}, {beta_v}{fm} : {vec4_ty}")
                lines.append(f"        {val2} = arith.addf {val_vec}, {inp_scaled}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if bias_name is not None:
                bias_vec = _fresh("bias_vec")
                val2 = _fresh("val_bias")
                lines.append(f"        {bias_vec} = vector.load {arg_ssa[str(bias_name)]}[%gn0] : {bias_memref}, {vec4_ty}")
                lines.append(f"        {val2} = arith.addf {val_vec}, {bias_vec}{fm} : {vec4_ty}")
                val_vec = str(val2)
            if relu:
                zvec = _fresh("zvec")
                relu_v = _fresh("relu")
                lines.append(f"        {zvec} = vector.splat %c0f : {vec4_ty}")
                lines.append(f"        {relu_v} = arith.maximumf {val_vec}, {zvec}{fm} : {vec4_ty}")
                val_vec = str(relu_v)
            if row_mask_name is not None and col_mask_name is not None:
                row_mask_elem_ty = str(arg_specs[str(row_mask_name)].get("memref_elem_ty") or "i8")
                col_mask_elem_ty = str(arg_specs[str(col_mask_name)].get("memref_elem_ty") or "i8")
                rm = _fresh("rm")
                rm0 = _fresh("rm0")
                rm1 = _fresh("rm1")
                rmv = _fresh("rmv")
                cm = _fresh("cm")
                cm0 = _fresh("cm0")
                cm0v = _fresh("cm0v")
                cm1 = _fresh("cm1")
                cond = _fresh("cond")
                zvec = _fresh("zvec_m")
                sel = _fresh("sel")
                lines.append(f"        {rm} = memref.load {arg_ssa[str(row_mask_name)]}[%gm] : {row_mask_memref}")
                lines.append(f"        {rm0} = arith.constant 0 : {row_mask_elem_ty}")
                lines.append(f"        {rm1} = arith.cmpi ne, {rm}, {rm0} : {row_mask_elem_ty}")
                lines.append(f"        {rmv} = vector.splat {rm1} : vector<4xi1>")
                lines.append(f"        {cm} = vector.load {arg_ssa[str(col_mask_name)]}[%gn0] : {col_mask_memref}, vector<4x{col_mask_elem_ty}>")
                lines.append(f"        {cm0} = arith.constant 0 : {col_mask_elem_ty}")
                lines.append(f"        {cm0v} = vector.splat {cm0} : vector<4x{col_mask_elem_ty}>")
                lines.append(f"        {cm1} = arith.cmpi ne, {cm}, {cm0v} : vector<4x{col_mask_elem_ty}>")
                lines.append(f"        {cond} = arith.andi {rmv}, {cm1} : vector<4xi1>")
                lines.append(f"        {zvec} = vector.splat %c0f : {vec4_ty}")
                cur_vec = str(zvec)
                for lane in range(4):
                    cond_lane = _fresh(f"cond{lane}")
                    val_lane = _fresh(f"val{lane}")
                    sel_lane = _fresh(f"sel{lane}")
                    next_vec = _fresh(f"v{lane}")
                    lines.append(f"        {cond_lane} = vector.extract {cond}[{lane}] : i1 from vector<4xi1>")
                    lines.append(f"        {val_lane} = vector.extract {val_vec}[{lane}] : f32 from {vec4_ty}")
                    lines.append(f"        {sel_lane} = arith.select {cond_lane}, {val_lane}, %c0f : f32")
                    lines.append(f"        {next_vec} = vector.insert {sel_lane}, {cur_vec}[{lane}] : f32 into {vec4_ty}")
                    cur_vec = str(next_vec)
                val_vec = str(cur_vec)

            lines.append(f"        vector.store {val_vec}, {arg_ssa[out_name2]}[{idx_o}] : {out_memref}, {vec4_ty}")
            lines.append("      } else {")

            # Scalar fallback for boundary tiles (gm>=M or gn0+3>=N).
            z = _fresh("z")
            lines.append(f"        {z} = arith.constant 0.0 : f32")
            gm_b = _fresh("gm_b")
            lines.append(f"        {gm_b} = arith.addi %batch_m, %gm : index")
            for lane in range(4):
                gn = _fresh(f"gn{lane}")
                p_col = _fresh(f"p_col{lane}")
                p_ok = _fresh(f"p_ok{lane}")
                val_lane = _fresh(f"val{lane}")
                lines.append(f"        {gn} = arith.addi %gn0, %c{lane} : index")
                lines.append(f"        {p_col} = arith.cmpi ult, {gn}, %cN : index")
                lines.append(f"        {p_ok} = arith.andi %p_row, {p_col} : i1")
                lines.append(f"        scf.if {p_ok} {{")
                mul = _fresh("mul_c")
                idx = _fresh("idx_c")
                lines.append(f"          {mul} = arith.muli {gm_b}, %cN : index")
                lines.append(f"          {idx} = arith.addi {mul}, {gn} : index")
                lines.append(f"          {val_lane} = vector.extract {acc_out}[{lane}] : f32 from {vec4_ty}")
                val_scalar = str(val_lane)
                if alpha_name is not None:
                    alpha_s = _fresh("alpha")
                    val2 = _fresh("val_alpha")
                    lines.append(f"          {alpha_s} = memref.load {arg_ssa[str(alpha_name)]}[%c0] : {alpha_memref}")
                    lines.append(f"          {val2} = arith.mulf {val_scalar}, {alpha_s}{fm} : f32")
                    val_scalar = str(val2)
                if add_inp_name is not None:
                    beta_s = _fresh("beta")
                    inp_v = _fresh("inp")
                    inp_scaled = _fresh("inp_scaled")
                    val2 = _fresh("val_addmm")
                    lines.append(f"          {beta_s} = memref.load {arg_ssa[str(beta_name)]}[%c0] : {beta_memref}")
                    lines.append(f"          {inp_v} = memref.load {arg_ssa[str(add_inp_name)]}[{idx}] : {add_inp_memref}")
                    lines.append(f"          {inp_scaled} = arith.mulf {inp_v}, {beta_s}{fm} : f32")
                    lines.append(f"          {val2} = arith.addf {val_scalar}, {inp_scaled}{fm} : f32")
                    val_scalar = str(val2)
                if bias_name is not None:
                    bias_v = _fresh("bias")
                    val2 = _fresh("val_bias")
                    lines.append(f"          {bias_v} = memref.load {arg_ssa[str(bias_name)]}[{gn}] : {bias_memref}")
                    lines.append(f"          {val2} = arith.addf {val_scalar}, {bias_v}{fm} : f32")
                    val_scalar = str(val2)
                if relu:
                    relu_v = _fresh("relu")
                    lines.append(f"          {relu_v} = arith.maximumf {val_scalar}, {z}{fm} : f32")
                    val_scalar = str(relu_v)
                if row_mask_name is not None and col_mask_name is not None:
                    rm = _fresh("rm")
                    cm = _fresh("cm")
                    rm0 = _fresh("rm0")
                    cm0 = _fresh("cm0")
                    rm1 = _fresh("rm1")
                    cm1 = _fresh("cm1")
                    cond = _fresh("cond")
                    sel = _fresh("sel")
                    row_mask_elem_ty = str(arg_specs[str(row_mask_name)].get("memref_elem_ty") or "i8")
                    col_mask_elem_ty = str(arg_specs[str(col_mask_name)].get("memref_elem_ty") or "i8")
                    lines.append(f"          {rm} = memref.load {arg_ssa[str(row_mask_name)]}[%gm] : {row_mask_memref}")
                    lines.append(f"          {cm} = memref.load {arg_ssa[str(col_mask_name)]}[{gn}] : {col_mask_memref}")
                    lines.append(f"          {rm0} = arith.constant 0 : {row_mask_elem_ty}")
                    lines.append(f"          {cm0} = arith.constant 0 : {col_mask_elem_ty}")
                    lines.append(f"          {rm1} = arith.cmpi ne, {rm}, {rm0} : {row_mask_elem_ty}")
                    lines.append(f"          {cm1} = arith.cmpi ne, {cm}, {cm0} : {col_mask_elem_ty}")
                    lines.append(f"          {cond} = arith.andi {rm1}, {cm1} : i1")
                    lines.append(f"          {sel} = arith.select {cond}, {val_scalar}, {z} : f32")
                    val_scalar = str(sel)
                lines.append(f"          memref.store {val_scalar}, {arg_ssa[out_name2]}[{idx}] : {out_memref}")
                lines.append("        }")

            lines.append("      }")
    elif matmul_v1 is not None:
        kernel_kind = "matmul_tile_v1"
        a_name = str(matmul_v1["A"])
        b_name = str(matmul_v1["B"])
        out_name2 = str(matmul_v1["out"])
        bias_name = matmul_v1.get("bias")
        row_mask_name = matmul_v1.get("row_mask")
        col_mask_name = matmul_v1.get("col_mask")
        relu = bool(matmul_v1.get("relu") or False)

        m_dim = int(matmul_v1["M"])
        n_dim = int(matmul_v1["N"])
        k_dim = int(matmul_v1["K"])
        bm = int(matmul_v1["BM"])
        bn = int(matmul_v1["BN"])
        bk = int(matmul_v1["BK"])
        if bm <= 0 or bn <= 0 or bk <= 0:
            raise RuntimeError(f"invalid matmul tile: BM={bm} BN={bn} BK={bk}")

        if str(arg_specs[a_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 A tensor")
        if str(arg_specs[b_name].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 B tensor")
        if str(arg_specs[out_name2].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul expects f32 output tensor")
        if bias_name is not None and str(arg_specs[str(bias_name)].get("memref_elem_ty")) != "f32":
            raise RuntimeError("matmul bias expects f32 tensor")
        if row_mask_name is not None and str(arg_specs[str(row_mask_name)].get("memref_elem_ty")) != "i1":
            if str(arg_specs[str(row_mask_name)].get("memref_elem_ty")) != "i8":
                raise RuntimeError("matmul row_mask expects bool-like tensor (i8 ABI) for cuda real-mlir wave")
        if col_mask_name is not None and str(arg_specs[str(col_mask_name)].get("memref_elem_ty")) != "i1":
            if str(arg_specs[str(col_mask_name)].get("memref_elem_ty")) != "i8":
                raise RuntimeError("matmul col_mask expects bool-like tensor (i8 ABI) for cuda real-mlir wave")

        a_memref = str(arg_specs[a_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        bias_memref = str(arg_specs[str(bias_name)]["memref"]) if bias_name is not None else ""
        row_mask_memref = str(arg_specs[str(row_mask_name)]["memref"]) if row_mask_name is not None else ""
        col_mask_memref = str(arg_specs[str(col_mask_name)]["memref"]) if col_mask_name is not None else ""

        threads = 256
        grid_x = (int(n_dim) + int(bn) - 1) // int(bn)
        grid_y = (int(m_dim) + int(bm) - 1) // int(bm)
        launch_override = {"block": [int(threads), 1, 1], "grid": [int(grid_x), int(grid_y), 1]}

        tile_a_elems = int(bm) * int(bk)
        tile_b_elems = int(bk) * int(bn)
        sh_elems = int(tile_a_elems + tile_b_elems)
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"
        offset_b = int(tile_a_elems)

        # Kernel mapping: one CTA per (BM x BN) output tile, 256 threads. Each thread
        # computes 4 output elements (tile has 1024 values for current tuned shapes).
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_n = gpu.block_id x")
        lines.append("      %bid_m = gpu.block_id y")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cK = arith.constant {int(k_dim)} : index")
        lines.append(f"      %cBM = arith.constant {int(bm)} : index")
        lines.append(f"      %cBN = arith.constant {int(bn)} : index")
        lines.append(f"      %cBK = arith.constant {int(bk)} : index")
        lines.append(f"      %cTileA = arith.constant {int(tile_a_elems)} : index")
        lines.append(f"      %cTileB = arith.constant {int(tile_b_elems)} : index")
        lines.append(f"      %cOffsetB = arith.constant {int(offset_b)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %base_m = arith.muli %bid_m, %cBM : index")
        lines.append("      %base_n = arith.muli %bid_n, %cBN : index")
        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")

        # out indices within the tile: tid + {0,256,512,768}
        lines.append("      %c256 = arith.constant 256 : index")
        lines.append("      %c512 = arith.constant 512 : index")
        lines.append("      %c768 = arith.constant 768 : index")
        lines.append("      %out0 = arith.addi %tid, %c0 : index")
        lines.append("      %out1 = arith.addi %tid, %c256 : index")
        lines.append("      %out2 = arith.addi %tid, %c512 : index")
        lines.append("      %out3 = arith.addi %tid, %c768 : index")
        lines.append("      %row0 = arith.divui %out0, %cBN : index")
        lines.append("      %col0 = arith.remui %out0, %cBN : index")
        lines.append("      %row1 = arith.divui %out1, %cBN : index")
        lines.append("      %row2 = arith.divui %out2, %cBN : index")
        lines.append("      %row3 = arith.divui %out3, %cBN : index")
        lines.append("      %gm0 = arith.addi %base_m, %row0 : index")
        lines.append("      %gm1 = arith.addi %base_m, %row1 : index")
        lines.append("      %gm2 = arith.addi %base_m, %row2 : index")
        lines.append("      %gm3 = arith.addi %base_m, %row3 : index")
        lines.append("      %gn = arith.addi %base_n, %col0 : index")

        # Outer K tiling.
        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        acc2_out = _fresh("acc2")
        acc3_out = _fresh("acc3")
        lines.append(
            f"      {acc0_out}, {acc1_out}, {acc2_out}, {acc3_out} = scf.for %k0 = %c0 to %cK step %cBK "
            "iter_args(%acc0 = %c0f, %acc1 = %c0f, %acc2 = %c0f, %acc3 = %c0f) -> (f32, f32, f32, f32) {"
        )

        # Cooperative A tile load into shared[0:tile_a_elems).
        loads_a = (int(tile_a_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_a)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_a")
            idx = _fresh("idx_a")
            pred = _fresh("pred_a")
            lines.append(f"        {c_off} = arith.constant {int(off)} : index")
            lines.append(f"        {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"        {pred} = arith.cmpi ult, {idx}, %cTileA : index")
            lines.append(f"        scf.if {pred} {{")
            row = _fresh("a_row")
            kk = _fresh("a_k")
            g_row = _fresh("a_gm")
            g_k = _fresh("a_gk")
            p_row = _fresh("a_pr")
            p_k = _fresh("a_pk")
            p_ok = _fresh("a_p")
            val = _fresh("a_val")
            lines.append(f"          {row} = arith.divui {idx}, %cBK : index")
            lines.append(f"          {kk} = arith.remui {idx}, %cBK : index")
            lines.append(f"          {g_row} = arith.addi %base_m, {row} : index")
            lines.append(f"          {g_k} = arith.addi %k0, {kk} : index")
            lines.append(f"          {p_row} = arith.cmpi ult, {g_row}, %cM : index")
            lines.append(f"          {p_k} = arith.cmpi ult, {g_k}, %cK : index")
            lines.append(f"          {p_ok} = arith.andi {p_row}, {p_k} : i1")
            lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
            mul = _fresh("a_mul")
            g_idx = _fresh("a_idx")
            lines.append(f"            {mul} = arith.muli {g_row}, %cK : index")
            lines.append(f"            {g_idx} = arith.addi {mul}, {g_k} : index")
            v = _fresh("a_load")
            lines.append(f"            {v} = memref.load {arg_ssa[a_name]}[{g_idx}] : {a_memref}")
            lines.append(f"            scf.yield {v} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          memref.store {val}, %sh[{idx}] : {shared_global_memref_ty}")
            lines.append("        }")

        # Cooperative B tile load into shared[offset_b:).
        loads_b = (int(tile_b_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_b)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_b")
            idx = _fresh("idx_b")
            pred = _fresh("pred_b")
            lines.append(f"        {c_off} = arith.constant {int(off)} : index")
            lines.append(f"        {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"        {pred} = arith.cmpi ult, {idx}, %cTileB : index")
            lines.append(f"        scf.if {pred} {{")
            kk = _fresh("b_k")
            col = _fresh("b_col")
            g_k = _fresh("b_gk")
            g_col = _fresh("b_gn")
            p_k = _fresh("b_pk")
            p_col = _fresh("b_pn")
            p_ok = _fresh("b_p")
            val = _fresh("b_val")
            sh_idx = _fresh("b_sh")
            lines.append(f"          {kk} = arith.divui {idx}, %cBN : index")
            lines.append(f"          {col} = arith.remui {idx}, %cBN : index")
            lines.append(f"          {g_k} = arith.addi %k0, {kk} : index")
            lines.append(f"          {g_col} = arith.addi %base_n, {col} : index")
            lines.append(f"          {p_k} = arith.cmpi ult, {g_k}, %cK : index")
            lines.append(f"          {p_col} = arith.cmpi ult, {g_col}, %cN : index")
            lines.append(f"          {p_ok} = arith.andi {p_k}, {p_col} : i1")
            lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
            mul = _fresh("b_mul")
            g_idx = _fresh("b_idx")
            lines.append(f"            {mul} = arith.muli {g_k}, %cN : index")
            lines.append(f"            {g_idx} = arith.addi {mul}, {g_col} : index")
            v = _fresh("b_load")
            lines.append(f"            {v} = memref.load {arg_ssa[b_name]}[{g_idx}] : {b_memref}")
            lines.append(f"            scf.yield {v} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          {sh_idx} = arith.addi {idx}, %cOffsetB : index")
            lines.append(f"          memref.store {val}, %sh[{sh_idx}] : {shared_global_memref_ty}")
            lines.append("        }")

        lines.append("        gpu.barrier")

        # Compute this K tile (unrolled over BK).
        acc0_cur = "%acc0"
        acc1_cur = "%acc1"
        acc2_cur = "%acc2"
        acc3_cur = "%acc3"
        for ki in range(int(bk)):
            cki = _fresh("cki")
            b_mul = _fresh("b_mul")
            b_idx0 = _fresh("b_idx")
            b_idx = _fresh("b_sh")
            b_val = _fresh("b")
            lines.append(f"        {cki} = arith.constant {int(ki)} : index")
            lines.append(f"        {b_mul} = arith.muli {cki}, %cBN : index")
            lines.append(f"        {b_idx0} = arith.addi {b_mul}, %col0 : index")
            lines.append(f"        {b_idx} = arith.addi {b_idx0}, %cOffsetB : index")
            lines.append(f"        {b_val} = memref.load %sh[{b_idx}] : {shared_global_memref_ty}")

            def _acc_update(row_ssa: str, acc_ssa: str) -> str:
                a_mul = _fresh("a_mul")
                a_idx = _fresh("a_idx")
                a_val = _fresh("a")
                prod = _fresh("prod")
                acc_new = _fresh("acc")
                lines.append(f"        {a_mul} = arith.muli {row_ssa}, %cBK : index")
                lines.append(f"        {a_idx} = arith.addi {a_mul}, {cki} : index")
                lines.append(f"        {a_val} = memref.load %sh[{a_idx}] : {shared_global_memref_ty}")
                lines.append(f"        {prod} = arith.mulf {a_val}, {b_val}{fm} : f32")
                lines.append(f"        {acc_new} = arith.addf {acc_ssa}, {prod}{fm} : f32")
                return str(acc_new)

            acc0_cur = _acc_update("%row0", acc0_cur)
            acc1_cur = _acc_update("%row1", acc1_cur)
            acc2_cur = _acc_update("%row2", acc2_cur)
            acc3_cur = _acc_update("%row3", acc3_cur)

        lines.append("        gpu.barrier")
        lines.append(f"        scf.yield {acc0_cur}, {acc1_cur}, {acc2_cur}, {acc3_cur} : f32, f32, f32, f32")
        lines.append("      }")

        # Store outputs (fused epilogue).
        z = _fresh("z")
        lines.append(f"      {z} = arith.constant 0.0 : f32")

        def _store_one(gm: str, acc: str) -> None:
            p_row = _fresh("p_row")
            p_col = _fresh("p_col")
            p_ok = _fresh("p_ok")
            lines.append(f"      {p_row} = arith.cmpi ult, {gm}, %cM : index")
            lines.append(f"      {p_col} = arith.cmpi ult, %gn, %cN : index")
            lines.append(f"      {p_ok} = arith.andi {p_row}, {p_col} : i1")
            lines.append(f"      scf.if {p_ok} {{")
            val = acc
            if bias_name is not None:
                bias_v = _fresh("bias")
                lines.append(f"        {bias_v} = memref.load {arg_ssa[str(bias_name)]}[%gn] : {bias_memref}")
                val2 = _fresh("val_bias")
                lines.append(f"        {val2} = arith.addf {val}, {bias_v}{fm} : f32")
                val = str(val2)
            if relu:
                relu_v = _fresh("relu")
                lines.append(f"        {relu_v} = arith.maximumf {val}, {z}{fm} : f32")
                val = str(relu_v)
            if row_mask_name is not None and col_mask_name is not None:
                rm = _fresh("rm")
                cm = _fresh("cm")
                rm0 = _fresh("rm0")
                cm0 = _fresh("cm0")
                rm1 = _fresh("rm1")
                cm1 = _fresh("cm1")
                cond = _fresh("cond")
                sel = _fresh("sel")
                row_mask_elem_ty = str(arg_specs[str(row_mask_name)].get("memref_elem_ty") or "i8")
                col_mask_elem_ty = str(arg_specs[str(col_mask_name)].get("memref_elem_ty") or "i8")
                lines.append(f"        {rm} = memref.load {arg_ssa[str(row_mask_name)]}[{gm}] : {row_mask_memref}")
                lines.append(f"        {cm} = memref.load {arg_ssa[str(col_mask_name)]}[%gn] : {col_mask_memref}")
                lines.append(f"        {rm0} = arith.constant 0 : {row_mask_elem_ty}")
                lines.append(f"        {cm0} = arith.constant 0 : {col_mask_elem_ty}")
                lines.append(f"        {rm1} = arith.cmpi ne, {rm}, {rm0} : {row_mask_elem_ty}")
                lines.append(f"        {cm1} = arith.cmpi ne, {cm}, {cm0} : {col_mask_elem_ty}")
                lines.append(f"        {cond} = arith.andi {rm1}, {cm1} : i1")
                lines.append(f"        {sel} = arith.select {cond}, {val}, {z} : f32")
                val = str(sel)
            mul = _fresh("mul_c")
            idx = _fresh("idx_c")
            lines.append(f"        {mul} = arith.muli {gm}, %cN : index")
            lines.append(f"        {idx} = arith.addi {mul}, %gn : index")
            lines.append(f"        memref.store {val}, {arg_ssa[out_name2]}[{idx}] : {out_memref}")
            lines.append("      }")

        _store_one("%gm0", acc0_out)
        _store_one("%gm1", acc1_out)
        _store_one("%gm2", acc2_out)
        _store_one("%gm3", acc3_out)
    elif mlp2d_v1 is not None:
        kernel_kind = "mlp2d_fused_v1"
        a_name = str(mlp2d_v1["A"])
        w1_name = str(mlp2d_v1["W1"])
        b1_name = str(mlp2d_v1["b1"])
        w2_name = str(mlp2d_v1["W2"])
        b2_name = str(mlp2d_v1["b2"])
        out_name2 = str(mlp2d_v1["out"])

        m_dim = int(mlp2d_v1["M"])
        n_dim = int(mlp2d_v1["N"])
        k_dim = int(mlp2d_v1["K"])
        h_dim = int(mlp2d_v1["H"])
        bm = int(mlp2d_v1["BM"])
        bn = int(mlp2d_v1["BN"])
        bk = int(mlp2d_v1["BK"])
        bh = int(mlp2d_v1["BH"])
        if bm <= 0 or bn <= 0 or bk <= 0 or bh <= 0:
            raise RuntimeError(f"invalid mlp2d tile: BM={bm} BN={bn} BK={bk} BH={bh}")

        for nm in (a_name, w1_name, b1_name, w2_name, b2_name, out_name2):
            if nm not in arg_specs:
                raise RuntimeError(f"mlp2d missing tensor in arg_specs: {nm}")
        for nm in (a_name, w1_name, b1_name, w2_name, b2_name, out_name2):
            if str(arg_specs[str(nm)].get("memref_elem_ty")) != "f32":
                raise RuntimeError(f"mlp2d expects f32 tensor for {nm}")

        a_memref = str(arg_specs[a_name]["memref"])
        w1_memref = str(arg_specs[w1_name]["memref"])
        b1_memref = str(arg_specs[b1_name]["memref"])
        w2_memref = str(arg_specs[w2_name]["memref"])
        b2_memref = str(arg_specs[b2_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        threads = 256
        grid_x = (int(n_dim) + int(bn) - 1) // int(bn)
        grid_y = (int(m_dim) + int(bm) - 1) // int(bm)
        launch_override = {"block": [int(threads), 1, 1], "grid": [int(grid_x), int(grid_y), 1]}

        tile_a_elems = int(bm) * int(bk)
        tile_w1_elems = int(bk) * int(bh)
        tile_acc1_elems = int(bm) * int(bh)
        tile_w2_elems = int(bh) * int(bn)
        sh_elems = int(tile_a_elems + tile_w1_elems + tile_acc1_elems + tile_w2_elems)
        offset_w1 = int(tile_a_elems)
        offset_acc1 = int(offset_w1 + tile_w1_elems)
        offset_w2 = int(offset_acc1 + tile_acc1_elems)

        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_n = gpu.block_id x")
        lines.append("      %bid_m = gpu.block_id y")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cK = arith.constant {int(k_dim)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cBM = arith.constant {int(bm)} : index")
        lines.append(f"      %cBN = arith.constant {int(bn)} : index")
        lines.append(f"      %cBK = arith.constant {int(bk)} : index")
        lines.append(f"      %cBH = arith.constant {int(bh)} : index")
        lines.append(f"      %cTileA = arith.constant {int(tile_a_elems)} : index")
        lines.append(f"      %cTileW1 = arith.constant {int(tile_w1_elems)} : index")
        lines.append(f"      %cTileAcc1 = arith.constant {int(tile_acc1_elems)} : index")
        lines.append(f"      %cTileW2 = arith.constant {int(tile_w2_elems)} : index")
        lines.append(f"      %cOffsetW1 = arith.constant {int(offset_w1)} : index")
        lines.append(f"      %cOffsetAcc1 = arith.constant {int(offset_acc1)} : index")
        lines.append(f"      %cOffsetW2 = arith.constant {int(offset_w2)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %base_m = arith.muli %bid_m, %cBM : index")
        lines.append("      %base_n = arith.muli %bid_n, %cBN : index")
        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")

        # out indices within the tile: tid + {0,256,512,768}
        lines.append("      %c256 = arith.constant 256 : index")
        lines.append("      %c512 = arith.constant 512 : index")
        lines.append("      %c768 = arith.constant 768 : index")
        lines.append("      %out0 = arith.addi %tid, %c0 : index")
        lines.append("      %out1 = arith.addi %tid, %c256 : index")
        lines.append("      %out2 = arith.addi %tid, %c512 : index")
        lines.append("      %out3 = arith.addi %tid, %c768 : index")
        lines.append("      %row0 = arith.divui %out0, %cBN : index")
        lines.append("      %col0 = arith.remui %out0, %cBN : index")
        lines.append("      %row1 = arith.divui %out1, %cBN : index")
        lines.append("      %row2 = arith.divui %out2, %cBN : index")
        lines.append("      %row3 = arith.divui %out3, %cBN : index")
        lines.append("      %gm0 = arith.addi %base_m, %row0 : index")
        lines.append("      %gm1 = arith.addi %base_m, %row1 : index")
        lines.append("      %gm2 = arith.addi %base_m, %row2 : index")
        lines.append("      %gm3 = arith.addi %base_m, %row3 : index")
        lines.append("      %gn = arith.addi %base_n, %col0 : index")

        # Precompute row offsets into acc1 tile (row * BH).
        lines.append("      %row0_off = arith.muli %row0, %cBH : index")
        lines.append("      %row1_off = arith.muli %row1, %cBH : index")
        lines.append("      %row2_off = arith.muli %row2, %cBH : index")
        lines.append("      %row3_off = arith.muli %row3, %cBH : index")

        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        acc2_out = _fresh("acc2")
        acc3_out = _fresh("acc3")
        lines.append(
            f"      {acc0_out}, {acc1_out}, {acc2_out}, {acc3_out} = scf.for %h0 = %c0 to %cH step %cBH "
            "iter_args(%acc0 = %c0f, %acc1 = %c0f, %acc2 = %c0f, %acc3 = %c0f) -> (f32, f32, f32, f32) {"
        )

        # Each thread computes up to 2 acc1 elements (idx0=tid, idx1=tid+256) for this h0 segment.
        idx0 = "%tid"
        idx1 = _fresh("acc1_idx1")
        lines.append(f"        {idx1} = arith.addi %tid, %c256 : index")

        # K loop to compute acc1 values for idx0/idx1.
        a0_out = _fresh("a0")
        a1_out = _fresh("a1")
        lines.append(
            f"        {a0_out}, {a1_out} = scf.for %k0 = %c0 to %cK step %cBK "
            "iter_args(%a0 = %c0f, %a1 = %c0f) -> (f32, f32) {"
        )

        # Cooperative A tile load into shared[0:tile_a_elems).
        loads_a = (int(tile_a_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_a)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_a")
            idx = _fresh("idx_a")
            pred = _fresh("pred_a")
            lines.append(f"          {c_off} = arith.constant {int(off)} : index")
            lines.append(f"          {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"          {pred} = arith.cmpi ult, {idx}, %cTileA : index")
            lines.append(f"          scf.if {pred} {{")
            row = _fresh("a_row")
            kk = _fresh("a_k")
            g_row = _fresh("a_gm")
            g_k = _fresh("a_gk")
            p_row = _fresh("a_pr")
            p_k = _fresh("a_pk")
            p_ok = _fresh("a_p")
            val = _fresh("a_val")
            lines.append(f"            {row} = arith.divui {idx}, %cBK : index")
            lines.append(f"            {kk} = arith.remui {idx}, %cBK : index")
            lines.append(f"            {g_row} = arith.addi %base_m, {row} : index")
            lines.append(f"            {g_k} = arith.addi %k0, {kk} : index")
            lines.append(f"            {p_row} = arith.cmpi ult, {g_row}, %cM : index")
            lines.append(f"            {p_k} = arith.cmpi ult, {g_k}, %cK : index")
            lines.append(f"            {p_ok} = arith.andi {p_row}, {p_k} : i1")
            lines.append(f"            {val} = scf.if {p_ok} -> (f32) {{")
            mul = _fresh("a_mul")
            g_idx = _fresh("a_idx")
            lines.append(f"              {mul} = arith.muli {g_row}, %cK : index")
            lines.append(f"              {g_idx} = arith.addi {mul}, {g_k} : index")
            v = _fresh("a_load")
            lines.append(f"              {v} = memref.load {arg_ssa[a_name]}[{g_idx}] : {a_memref}")
            lines.append(f"              scf.yield {v} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            lines.append(f"            memref.store {val}, %sh[{idx}] : {shared_global_memref_ty}")
            lines.append("          }")

        # Cooperative W1 tile load into shared[offset_w1:).
        loads_w1 = (int(tile_w1_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_w1)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_w1")
            idx = _fresh("idx_w1")
            pred = _fresh("pred_w1")
            lines.append(f"          {c_off} = arith.constant {int(off)} : index")
            lines.append(f"          {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"          {pred} = arith.cmpi ult, {idx}, %cTileW1 : index")
            lines.append(f"          scf.if {pred} {{")
            kk = _fresh("w1_k")
            hh = _fresh("w1_h")
            g_k = _fresh("w1_gk")
            g_h = _fresh("w1_gh")
            p_k = _fresh("w1_pk")
            p_h = _fresh("w1_ph")
            p_ok = _fresh("w1_p")
            val = _fresh("w1_val")
            sh_idx = _fresh("w1_sh")
            lines.append(f"            {kk} = arith.divui {idx}, %cBH : index")
            lines.append(f"            {hh} = arith.remui {idx}, %cBH : index")
            lines.append(f"            {g_k} = arith.addi %k0, {kk} : index")
            lines.append(f"            {g_h} = arith.addi %h0, {hh} : index")
            lines.append(f"            {p_k} = arith.cmpi ult, {g_k}, %cK : index")
            lines.append(f"            {p_h} = arith.cmpi ult, {g_h}, %cH : index")
            lines.append(f"            {p_ok} = arith.andi {p_k}, {p_h} : i1")
            lines.append(f"            {val} = scf.if {p_ok} -> (f32) {{")
            mul = _fresh("w1_mul")
            g_idx = _fresh("w1_idx")
            lines.append(f"              {mul} = arith.muli {g_k}, %cH : index")
            lines.append(f"              {g_idx} = arith.addi {mul}, {g_h} : index")
            v = _fresh("w1_load")
            lines.append(f"              {v} = memref.load {arg_ssa[w1_name]}[{g_idx}] : {w1_memref}")
            lines.append(f"              scf.yield {v} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            lines.append(f"            {sh_idx} = arith.addi {idx}, %cOffsetW1 : index")
            lines.append(f"            memref.store {val}, %sh[{sh_idx}] : {shared_global_memref_ty}")
            lines.append("          }")

        lines.append("          gpu.barrier")

        # Accumulate acc1 for idx0 and idx1 (two elements per thread).
        def _acc1_update(idx_ssa: str, acc_ssa: str) -> str:
            row = _fresh("acc1_row")
            hh = _fresh("acc1_h")
            acc_cur = str(acc_ssa)
            lines.append(f"          {row} = arith.divui {idx_ssa}, %cBH : index")
            lines.append(f"          {hh} = arith.remui {idx_ssa}, %cBH : index")
            for ki in range(int(bk)):
                cki = _fresh("cki")
                lines.append(f"          {cki} = arith.constant {int(ki)} : index")
                # A[row, ki]
                a_mul = _fresh("acc1_amul")
                a_idx = _fresh("acc1_aidx")
                lines.append(f"          {a_mul} = arith.muli {row}, %cBK : index")
                lines.append(f"          {a_idx} = arith.addi {a_mul}, {cki} : index")
                a_val = _fresh("a")
                lines.append(f"          {a_val} = memref.load %sh[{a_idx}] : {shared_global_memref_ty}")
                # W1[ki, hh]
                w_mul = _fresh("acc1_wmul")
                w_idx0 = _fresh("acc1_widx0")
                w_idx = _fresh("acc1_widx")
                lines.append(f"          {w_mul} = arith.muli {cki}, %cBH : index")
                lines.append(f"          {w_idx0} = arith.addi {w_mul}, {hh} : index")
                lines.append(f"          {w_idx} = arith.addi {w_idx0}, %cOffsetW1 : index")
                w_val = _fresh("w1")
                lines.append(f"          {w_val} = memref.load %sh[{w_idx}] : {shared_global_memref_ty}")
                prod = _fresh("prod")
                acc_new = _fresh("acc")
                lines.append(f"          {prod} = arith.mulf {a_val}, {w_val}{fm} : f32")
                lines.append(f"          {acc_new} = arith.addf {acc_cur}, {prod}{fm} : f32")
                acc_cur = str(acc_new)
            return acc_cur

        acc0_cur = _acc1_update(idx0, "%a0")
        acc1_cur = _acc1_update(idx1, "%a1")

        lines.append("          gpu.barrier")
        lines.append(f"          scf.yield {acc0_cur}, {acc1_cur} : f32, f32")
        lines.append("        }")

        # Apply bias+relu and store acc1 tile into shared[offset_acc1:).
        def _store_acc1(idx_ssa: str, acc_ssa: str) -> None:
            pred = _fresh("p_acc1")
            lines.append(f"        {pred} = arith.cmpi ult, {idx_ssa}, %cTileAcc1 : index")
            lines.append(f"        scf.if {pred} {{")
            row = _fresh("row")
            hh = _fresh("hh")
            gm = _fresh("gm")
            gh = _fresh("gh")
            p_row = _fresh("p_row")
            p_h = _fresh("p_h")
            p_ok = _fresh("p_ok")
            val = _fresh("val")
            sh_idx0 = _fresh("sh_idx0")
            sh_idx = _fresh("sh_idx")
            lines.append(f"          {row} = arith.divui {idx_ssa}, %cBH : index")
            lines.append(f"          {hh} = arith.remui {idx_ssa}, %cBH : index")
            lines.append(f"          {gm} = arith.addi %base_m, {row} : index")
            lines.append(f"          {gh} = arith.addi %h0, {hh} : index")
            lines.append(f"          {p_row} = arith.cmpi ult, {gm}, %cM : index")
            lines.append(f"          {p_h} = arith.cmpi ult, {gh}, %cH : index")
            lines.append(f"          {p_ok} = arith.andi {p_row}, {p_h} : i1")
            lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
            bias = _fresh("b1")
            v2 = _fresh("v2")
            relu = _fresh("relu")
            lines.append(f"            {bias} = memref.load {arg_ssa[b1_name]}[{gh}] : {b1_memref}")
            lines.append(f"            {v2} = arith.addf {acc_ssa}, {bias}{fm} : f32")
            lines.append(f"            {relu} = arith.maximumf {v2}, %c0f{fm} : f32")
            lines.append(f"            scf.yield {relu} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          {sh_idx0} = arith.addi {idx_ssa}, %cOffsetAcc1 : index")
            lines.append(f"          {sh_idx} = arith.addi {sh_idx0}, %c0 : index")
            lines.append(f"          memref.store {val}, %sh[{sh_idx}] : {shared_global_memref_ty}")
            lines.append("        }")

        _store_acc1(idx0, a0_out)
        _store_acc1(idx1, a1_out)

        lines.append("        gpu.barrier")

        # Cooperative W2 tile load into shared[offset_w2:).
        loads_w2 = (int(tile_w2_elems) + int(threads) - 1) // int(threads)
        for i in range(int(loads_w2)):
            off = int(i) * int(threads)
            c_off = _fresh("c_off_w2")
            idx = _fresh("idx_w2")
            pred = _fresh("pred_w2")
            lines.append(f"        {c_off} = arith.constant {int(off)} : index")
            lines.append(f"        {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"        {pred} = arith.cmpi ult, {idx}, %cTileW2 : index")
            lines.append(f"        scf.if {pred} {{")
            hh = _fresh("w2_h")
            col = _fresh("w2_col")
            gh = _fresh("w2_gh")
            gn = _fresh("w2_gn")
            p_h = _fresh("w2_ph")
            p_col = _fresh("w2_pn")
            p_ok = _fresh("w2_p")
            val = _fresh("w2_val")
            sh_idx = _fresh("w2_sh")
            lines.append(f"          {hh} = arith.divui {idx}, %cBN : index")
            lines.append(f"          {col} = arith.remui {idx}, %cBN : index")
            lines.append(f"          {gh} = arith.addi %h0, {hh} : index")
            lines.append(f"          {gn} = arith.addi %base_n, {col} : index")
            lines.append(f"          {p_h} = arith.cmpi ult, {gh}, %cH : index")
            lines.append(f"          {p_col} = arith.cmpi ult, {gn}, %cN : index")
            lines.append(f"          {p_ok} = arith.andi {p_h}, {p_col} : i1")
            lines.append(f"          {val} = scf.if {p_ok} -> (f32) {{")
            mul = _fresh("w2_mul")
            g_idx = _fresh("w2_idx")
            lines.append(f"            {mul} = arith.muli {gh}, %cN : index")
            lines.append(f"            {g_idx} = arith.addi {mul}, {gn} : index")
            v = _fresh("w2_load")
            lines.append(f"            {v} = memref.load {arg_ssa[w2_name]}[{g_idx}] : {w2_memref}")
            lines.append(f"            scf.yield {v} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          {sh_idx} = arith.addi {idx}, %cOffsetW2 : index")
            lines.append(f"          memref.store {val}, %sh[{sh_idx}] : {shared_global_memref_ty}")
            lines.append("        }")

        lines.append("        gpu.barrier")

        # Compute acc2 update for this H segment.
        seg_acc0 = "%acc0"
        seg_acc1 = "%acc1"
        seg_acc2 = "%acc2"
        seg_acc3 = "%acc3"
        for hi in range(int(bh)):
            chi = _fresh("chi")
            w2_mul = _fresh("w2_mul")
            w2_idx0 = _fresh("w2_idx0")
            w2_idx = _fresh("w2_idx")
            w2_val = _fresh("w2v")
            lines.append(f"        {chi} = arith.constant {int(hi)} : index")
            lines.append(f"        {w2_mul} = arith.muli {chi}, %cBN : index")
            lines.append(f"        {w2_idx0} = arith.addi {w2_mul}, %col0 : index")
            lines.append(f"        {w2_idx} = arith.addi {w2_idx0}, %cOffsetW2 : index")
            lines.append(f"        {w2_val} = memref.load %sh[{w2_idx}] : {shared_global_memref_ty}")

            def _acc2_update(row_off: str, acc_ssa: str) -> str:
                a_idx0 = _fresh("a1_idx0")
                a_idx1 = _fresh("a1_idx1")
                a_idx = _fresh("a1_idx")
                a_val = _fresh("a1v")
                prod = _fresh("prod")
                acc_new = _fresh("acc")
                lines.append(f"        {a_idx0} = arith.addi {row_off}, {chi} : index")
                lines.append(f"        {a_idx1} = arith.addi {a_idx0}, %cOffsetAcc1 : index")
                lines.append(f"        {a_idx} = arith.addi {a_idx1}, %c0 : index")
                lines.append(f"        {a_val} = memref.load %sh[{a_idx}] : {shared_global_memref_ty}")
                lines.append(f"        {prod} = arith.mulf {a_val}, {w2_val}{fm} : f32")
                lines.append(f"        {acc_new} = arith.addf {acc_ssa}, {prod}{fm} : f32")
                return str(acc_new)

            seg_acc0 = _acc2_update("%row0_off", seg_acc0)
            seg_acc1 = _acc2_update("%row1_off", seg_acc1)
            seg_acc2 = _acc2_update("%row2_off", seg_acc2)
            seg_acc3 = _acc2_update("%row3_off", seg_acc3)

        lines.append("        gpu.barrier")
        lines.append(f"        scf.yield {seg_acc0}, {seg_acc1}, {seg_acc2}, {seg_acc3} : f32, f32, f32, f32")
        lines.append("      }")

        # Store outputs with bias b2.
        def _store_one(gm: str, acc: str) -> None:
            p_row = _fresh("p_row")
            p_col = _fresh("p_col")
            p_ok = _fresh("p_ok")
            lines.append(f"      {p_row} = arith.cmpi ult, {gm}, %cM : index")
            lines.append(f"      {p_col} = arith.cmpi ult, %gn, %cN : index")
            lines.append(f"      {p_ok} = arith.andi {p_row}, {p_col} : i1")
            lines.append(f"      scf.if {p_ok} {{")
            bias = _fresh("b2")
            v2 = _fresh("v2")
            mul = _fresh("mul_c")
            idx = _fresh("idx_c")
            lines.append(f"        {bias} = memref.load {arg_ssa[b2_name]}[%gn] : {b2_memref}")
            lines.append(f"        {v2} = arith.addf {acc}, {bias}{fm} : f32")
            lines.append(f"        {mul} = arith.muli {gm}, %cN : index")
            lines.append(f"        {idx} = arith.addi {mul}, %gn : index")
            lines.append(f"        memref.store {v2}, {arg_ssa[out_name2]}[{idx}] : {out_memref}")
            lines.append("      }")

        _store_one("%gm0", acc0_out)
        _store_one("%gm1", acc1_out)
        _store_one("%gm2", acc2_out)
        _store_one("%gm3", acc3_out)
    elif row_masked_softmax_axis1 is not None:
        kernel_kind = "row_masked_softmax_axis1_v1"
        assert out_m is not None
        assert out_n is not None
        inp_name = str(row_masked_softmax_axis1["inp"])
        mask_name = str(row_masked_softmax_axis1["mask"])
        out_name2 = str(row_masked_softmax_axis1["out"])
        red_n = int(row_masked_softmax_axis1["reduce_n"])
        in_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        mask_memref = str(arg_specs[mask_name]["memref"])
        mask_elem_ty = str(arg_specs[mask_name].get("memref_elem_ty") or "")

        # One block per output row; compute masked max/sum reductions in shared memory.
        launch_override = {"block": [256, 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")

        # Mask constants: treat any non-zero value as true.
        if mask_elem_ty not in {"i8", "i32", "i64"}:
            raise RuntimeError(f"masked_softmax2d unsupported mask elem type: {mask_elem_ty}")
        lines.append(f"        %mask0 = arith.constant 0 : {mask_elem_ty}")
        lines.append("        %neg = arith.constant -1.0e9 : f32")

        # partial max
        lines.append("        %init_max = arith.constant -3.402823466e+38 : f32")
        lines.append("        %partial_max = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %init_max) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %mraw = memref.load {arg_ssa[mask_name]}[%j] : {mask_memref}")
        lines.append(f"          %m = arith.cmpi ne, %mraw, %mask0 : {mask_elem_ty}")
        lines.append("          %mx = arith.select %m, %x, %neg : f32")
        lines.append(f"          %acc_next = arith.maximumf %acc, %mx{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_max, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_max_{stride}"
            pS = f"%pS_max_{stride}"
            tid2 = f"%tid_max_{stride}"
            a = f"%a_max_{stride}"
            b = f"%b_max_{stride}"
            s = f"%s_max_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"          {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"          {s} = arith.maximumf {a}, {b}{fm} : f32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")
        lines.append("        %maxv = memref.load %sh[%c0] : memref<256xf32, 3>")

        # partial sum(exp(masked_x - max))
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %mraw = memref.load {arg_ssa[mask_name]}[%j] : {mask_memref}")
        lines.append(f"          %m = arith.cmpi ne, %mraw, %mask0 : {mask_elem_ty}")
        lines.append("          %mx = arith.select %m, %x, %neg : f32")
        lines.append(f"          %xc = arith.subf %mx, %maxv{fm} : f32")
        lines.append(f"          %e = math.exp %xc{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %e{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")

        # final output
        lines.append("        scf.for %j = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %mraw = memref.load {arg_ssa[mask_name]}[%j] : {mask_memref}")
        lines.append(f"          %m = arith.cmpi ne, %mraw, %mask0 : {mask_elem_ty}")
        lines.append("          %mx = arith.select %m, %x, %neg : f32")
        lines.append(f"          %xc = arith.subf %mx, %maxv{fm} : f32")
        lines.append(f"          %e = math.exp %xc{fm} : f32")
        lines.append(f"          %o = arith.divf %e, %sumv{fm} : f32")
        lines.append(f"          memref.store %o, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_log_softmax_axis1_v1 is not None:
        kernel_kind = "row_log_softmax_axis1_v1"
        assert out_m is not None
        assert out_n is not None
        inp_name = str(row_log_softmax_axis1_v1["inp"])
        out_name2 = str(row_log_softmax_axis1_v1["out"])
        red_n = int(row_log_softmax_axis1_v1["reduce_n"])
        in_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        if red_n <= 0:
            raise RuntimeError("log_softmax2d reduce_n must be > 0")

        launch_override = {"block": [256, 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("        %partial_max = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %neg_inf) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %acc_next = arith.maximumf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_max, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_lsm_max_{stride}"
            pS = f"%pS_lsm_max_{stride}"
            tid2 = f"%tid_lsm_max_{stride}"
            a = f"%a_lsm_max_{stride}"
            b = f"%b_lsm_max_{stride}"
            s = f"%s_lsm_max_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"          {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"          {s} = arith.maximumf {a}, {b}{fm} : f32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")
        lines.append("        %maxv = memref.load %sh[%c0] : memref<256xf32, 3>")

        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %xc = arith.subf %x, %maxv{fm} : f32")
        lines.append(f"          %e = math.exp %xc{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %e{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_lsm_sum_{stride}"
            pS = f"%pS_lsm_sum_{stride}"
            tid2 = f"%tid_lsm_sum_{stride}"
            a = f"%a_lsm_sum_{stride}"
            b = f"%b_lsm_sum_{stride}"
            s = f"%s_lsm_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %logsum = math.log %sumv{fm} : f32")

        lines.append("        scf.for %j = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %xc = arith.subf %x, %maxv{fm} : f32")
        lines.append(f"          %o = arith.subf %xc, %logsum{fm} : f32")
        lines.append(f"          memref.store %o, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_softmax_axis1 is not None:
        kernel_kind = "row_softmax_axis1_v1"
        assert out_m is not None
        assert out_n is not None
        inp_name = str(row_softmax_axis1["inp"])
        out_name2 = str(row_softmax_axis1["out"])
        red_n = int(row_softmax_axis1["reduce_n"])
        in_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        if red_n <= 0:
            raise RuntimeError("softmax2d reduce_n must be > 0")

        # One block per output row.
        #
        # For softmax over <=1024 elements, use a "one element per lane" tile
        # shape (power-of-two block size) and do reductions via subgroup shuffle
        # + 2 barriers (warp->block). This avoids the O(logN) shared-memory tree.
        if red_n <= 1024:
            # Empirically, Triton-native softmax uses far fewer than 1024 threads
            # (e.g. 4 warps). Keep thread count moderate and let each lane handle
            # multiple elements via the `%j = %tid .. step %bdim` loop.
            block_threads = 256
        else:
            block_threads = 256
        if (int(block_threads) % 32) != 0:
            raise RuntimeError(f"softmax2d block_threads must be multiple of 32, got {block_threads}")
        warps = int(block_threads // 32)

        ept = int((int(red_n) + int(block_threads) - 1) // int(block_threads))
        unroll_tile = bool(int(red_n) <= 2048 and int(ept) <= 8)
        # Cache exp(x-max) in shared memory only for the generic (non-unrolled)
        # path; the unrolled path keeps EPT values in registers.
        use_exp_shmem_cache = bool((not unroll_tile) and int(red_n) <= 2048)
        if use_exp_shmem_cache:
            sh_elems = int(block_threads + red_n)
        else:
            sh_elems = int(warps)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c0_i32 = arith.constant 0 : i32")
        lines.append("      %c32 = arith.constant 32 : i32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %lane = arith.remui %tid, %c32_idx : index")
        lines.append("      %warp = arith.divui %tid, %c32_idx : index")
        lines.append("      %c16 = arith.constant 16 : i32")
        lines.append("      %c8 = arith.constant 8 : i32")
        lines.append("      %c4 = arith.constant 4 : i32")
        lines.append("      %c2 = arith.constant 2 : i32")
        lines.append("      %c1 = arith.constant 1 : i32")
        lines.append(f"      %cWarps = arith.constant {int(warps)} : index")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        assert shared_global_sym is not None
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        if use_exp_shmem_cache:
            lines.append(f"        %cShOff = arith.constant {int(block_threads)} : index")

        # Compute per-lane partial max.
        tile_entries: list[tuple[str, str, str]] = []
        tile_exps: list[str] = []
        max_seed = "%partial_max"
        if unroll_tile:
            # EPT register tile: v[i] per lane, no exp shmem cache required.
            max_cur_local = "%neg_inf"
            for i in range(int(ept)):
                off = int(i) * int(block_threads)
                if int(i) == 0:
                    ci = "%tid"
                else:
                    c_off = f"%cOff_t{i}"
                    ci = f"%ci_t{i}"
                    lines.append(f"        {c_off} = arith.constant {int(off)} : index")
                    lines.append(f"        {ci} = arith.addi %tid, {c_off} : index")
                pred = f"%pred_t{i}"
                idx = f"%idx_t{i}"
                v = f"%v_t{i}"
                xv = f"%xv_t{i}"
                tmax = f"%tmax_t{i}"
                lines.append(f"        {pred} = arith.cmpi ult, {ci}, %cN : index")
                lines.append(f"        {idx} = arith.addi %base, {ci} : index")
                lines.append(f"        {v} = scf.if {pred} -> (f32) {{")
                lines.append(f"          {xv} = memref.load {arg_ssa[inp_name]}[{idx}] : {in_memref}")
                lines.append(f"          scf.yield {xv} : f32")
                lines.append("        } else {")
                lines.append("          scf.yield %neg_inf : f32")
                lines.append("        }")
                lines.append(f"        {tmax} = arith.maximumf {max_cur_local}, {v}{fm} : f32")
                max_cur_local = str(tmax)
                tile_entries.append((str(pred), str(idx), str(v)))
            max_seed = str(max_cur_local)
        else:
            lines.append("        %init_max = arith.constant -3.402823466e+38 : f32")
            lines.append("        %partial_max = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %init_max) -> (f32) {")
            lines.append("          %idx = arith.addi %base, %j : index")
            lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
            lines.append(f"          %acc_next = arith.maximumf %acc, %x{fm} : f32")
            lines.append("          scf.yield %acc_next : f32")
            lines.append("        }")

        # block_allreduce_max (warp shuffle + warp scratch).
        max_cur = str(max_seed)
        for off in (16, 8, 4, 2, 1):
            o = int(off)
            sh = f"%wm_sh_{o}"
            ok = f"%wm_ok_{o}"
            sel = f"%wm_sel_{o}"
            nxt = f"%wm_{o}"
            lines.append(f"        {sh}, {ok} = gpu.shuffle down {max_cur}, %c{o}, %c32 : f32")
            lines.append(f"        {sel} = arith.select {ok}, {sh}, {max_cur} : f32")
            lines.append(f"        {nxt} = arith.maximumf {max_cur}, {sel}{fm} : f32")
            max_cur = nxt
        wm_final = str(max_cur)
        lines.append("        %is_lane0 = arith.cmpi eq, %lane, %c0 : index")
        lines.append("        scf.if %is_lane0 {")
        lines.append(f"          memref.store {wm_final}, %sh[%warp] : {shared_global_memref_ty}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %is_warp0 = arith.cmpi eq, %warp, %c0 : index")
        lines.append("        scf.if %is_warp0 {")
        lines.append("          %lane_in = arith.cmpi ult, %lane, %cWarps : index")
        lines.append("          %wv = scf.if %lane_in -> (f32) {")
        lines.append(f"            %t = memref.load %sh[%lane] : {shared_global_memref_ty}")
        lines.append("            scf.yield %t : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %neg_inf : f32")
        lines.append("          }")
        bm_cur = "%wv"
        for off in (16, 8, 4, 2, 1):
            o = int(off)
            sh = f"%bm_sh_{o}"
            ok = f"%bm_ok_{o}"
            sel = f"%bm_sel_{o}"
            nxt = f"%bm_{o}"
            lines.append(f"          {sh}, {ok} = gpu.shuffle down {bm_cur}, %c{o}, %c32 : f32")
            lines.append(f"          {sel} = arith.select {ok}, {sh}, {bm_cur} : f32")
            lines.append(f"          {nxt} = arith.maximumf {bm_cur}, {sel}{fm} : f32")
            bm_cur = nxt
        bm_final = str(bm_cur)
        lines.append("          scf.if %is_lane0 {")
        lines.append(f"            memref.store {bm_final}, %sh[%c0] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append(f"        %maxv = memref.load %sh[%c0] : {shared_global_memref_ty}")

        # Compute per-lane partial sum.
        sum_seed = "%partial_sum"
        if unroll_tile:
            sum_cur_local = "%c0f"
            for i, (pred, _idx, v) in enumerate(tile_entries):
                e = f"%e_t{i}"
                xc = f"%xc_t{i}"
                ev = f"%ev_t{i}"
                tsum = f"%tsum_t{i}"
                lines.append(f"        {e} = scf.if {pred} -> (f32) {{")
                lines.append(f"          {xc} = arith.subf {v}, %maxv{fm} : f32")
                lines.append(f"          {ev} = math.exp {xc}{fm} : f32")
                lines.append(f"          scf.yield {ev} : f32")
                lines.append("        } else {")
                lines.append("          scf.yield %c0f : f32")
                lines.append("        }")
                lines.append(f"        {tsum} = arith.addf {sum_cur_local}, {e}{fm} : f32")
                sum_cur_local = str(tsum)
                tile_exps.append(str(e))
            sum_seed = str(sum_cur_local)
        else:
            lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
            lines.append("          %idx = arith.addi %base, %j : index")
            lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
            lines.append(f"          %xc = arith.subf %x, %maxv{fm} : f32")
            lines.append(f"          %e = math.exp %xc{fm} : f32")
            if use_exp_shmem_cache:
                lines.append("          %sh_i = arith.addi %cShOff, %j : index")
                lines.append(f"          memref.store %e, %sh[%sh_i] : {shared_global_memref_ty}")
            lines.append(f"          %acc_next = arith.addf %acc, %e{fm} : f32")
            lines.append("          scf.yield %acc_next : f32")
            lines.append("        }")

        # block_allreduce_sum (warp shuffle + warp scratch).
        sum_cur = str(sum_seed)
        for off in (16, 8, 4, 2, 1):
            o = int(off)
            sh = f"%ws_sh_{o}"
            ok = f"%ws_ok_{o}"
            sel = f"%ws_sel_{o}"
            nxt = f"%ws_{o}"
            lines.append(f"        {sh}, {ok} = gpu.shuffle down {sum_cur}, %c{o}, %c32 : f32")
            lines.append(f"        {sel} = arith.select {ok}, {sh}, %c0f : f32")
            lines.append(f"        {nxt} = arith.addf {sum_cur}, {sel}{fm} : f32")
            sum_cur = nxt
        ws_final = str(sum_cur)
        lines.append("        scf.if %is_lane0 {")
        lines.append(f"          memref.store {ws_final}, %sh[%warp] : {shared_global_memref_ty}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        scf.if %is_warp0 {")
        lines.append("          %lane_in2 = arith.cmpi ult, %lane, %cWarps : index")
        lines.append("          %sv = scf.if %lane_in2 -> (f32) {")
        lines.append(f"            %t2 = memref.load %sh[%lane] : {shared_global_memref_ty}")
        lines.append("            scf.yield %t2 : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        bs_cur = "%sv"
        for off in (16, 8, 4, 2, 1):
            o = int(off)
            sh = f"%bs_sh_{o}"
            ok = f"%bs_ok_{o}"
            sel = f"%bs_sel_{o}"
            nxt = f"%bs_{o}"
            lines.append(f"          {sh}, {ok} = gpu.shuffle down {bs_cur}, %c{o}, %c32 : f32")
            lines.append(f"          {sel} = arith.select {ok}, {sh}, %c0f : f32")
            lines.append(f"          {nxt} = arith.addf {bs_cur}, {sel}{fm} : f32")
            bs_cur = nxt
        bs_final = str(bs_cur)
        lines.append("          scf.if %is_lane0 {")
        lines.append(f"            memref.store {bs_final}, %sh[%c0] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append(f"        %sumv = memref.load %sh[%c0] : {shared_global_memref_ty}")
        lines.append(f"        %inv = arith.divf %c1f, %sumv{fm} : f32")

        # final output
        if unroll_tile:
            for i, (pred, idx, _v) in enumerate(tile_entries):
                e = tile_exps[int(i)]
                lines.append(f"        scf.if {pred} {{")
                lines.append(f"          %o_t{i} = arith.mulf {e}, %inv{fm} : f32")
                lines.append(f"          memref.store %o_t{i}, {arg_ssa[out_name2]}[{idx}] : {out_memref}")
                lines.append("        }")
        else:
            lines.append("        scf.for %j = %tid to %cN step %bdim {")
            lines.append("          %idx = arith.addi %base, %j : index")
            if use_exp_shmem_cache:
                lines.append("          %sh_i = arith.addi %cShOff, %j : index")
                lines.append(f"          %e = memref.load %sh[%sh_i] : {shared_global_memref_ty}")
            else:
                lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
                lines.append(f"          %xc = arith.subf %x, %maxv{fm} : f32")
                lines.append(f"          %e = math.exp %xc{fm} : f32")
            lines.append(f"          %o = arith.mulf %e, %inv{fm} : f32")
            lines.append(f"          memref.store %o, {arg_ssa[out_name2]}[%idx] : {out_memref}")
            lines.append("        }")
        lines.append("      }")
    elif row_grouped_row_sum2d is not None:
        kernel_kind = "grouped_row_sum_axis2_v1"
        assert out_m is not None
        assert out_n is not None

        inp_name = str(row_grouped_row_sum2d.get("inp") or "inp")
        out_name2 = str(row_grouped_row_sum2d.get("out") or out_name)
        in_n = int(row_grouped_row_sum2d.get("N") or 0)
        group_size = int(row_grouped_row_sum2d.get("GROUP_SIZE") or 0)
        out_g = int(row_grouped_row_sum2d.get("G") or int(out_n))
        total_out = int(out_total)
        if in_n <= 0 or group_size <= 0 or out_g <= 0:
            raise RuntimeError(
                "grouped_row_sum2d invalid dims: "
                f"N={in_n} GROUP_SIZE={group_size} G={out_g} out_total={total_out}"
            )

        in_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        # One block per (row, group) output.
        launch_override = {"block": [256, 1, 1], "grid": [int(total_out), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cTotal = arith.constant {int(total_out)} : index")
        lines.append(f"      %cG = arith.constant {int(out_g)} : index")
        lines.append(f"      %cN = arith.constant {int(in_n)} : index")
        lines.append(f"      %cGS = arith.constant {int(group_size)} : index")
        lines.append("      %pred = arith.cmpi ult, %bid, %cTotal : index")
        lines.append("      scf.if %pred {")
        lines.append("        %row = arith.divui %bid, %cG : index")
        lines.append("        %grp = arith.remui %bid, %cG : index")
        lines.append("        %row_off = arith.muli %row, %cN : index")
        lines.append("        %grp_off = arith.muli %grp, %cGS : index")
        lines.append("        %base = arith.addi %row_off, %grp_off : index")

        # partial sum
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %k = %tid to %cGS step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %k : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {in_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %sumv, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif group_norm_kernel_v1 is not None:
        kernel_kind = "group_norm_v1"
        n_dim = int(group_norm_kernel_v1["N"])
        c_dim = int(group_norm_kernel_v1["C"])
        hw_dim = int(group_norm_kernel_v1["HW"])
        num_groups = int(group_norm_kernel_v1["num_groups"])
        group_size = int(group_norm_kernel_v1["group_size"])
        eps_const = float(group_norm_kernel_v1["eps_const"])

        x_memref = str(arg_specs["X"]["memref"])
        y_memref = str(arg_specs["Y"]["memref"])
        w_memref = str(arg_specs["W"]["memref"])
        b_memref = str(arg_specs["B"]["memref"])
        mean_memref = str(arg_specs["Mean"]["memref"])
        rstd_memref = str(arg_specs["Rstd"]["memref"])

        blocks = int(n_dim * num_groups)
        elems = int(group_size * hw_dim)
        launch_override = {"block": [256, 1, 1], "grid": [int(blocks), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cBlocks = arith.constant {int(blocks)} : index")
        lines.append(f"      %cG = arith.constant {int(num_groups)} : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"      %cGS = arith.constant {int(group_size)} : index")
        lines.append(f"      %cHW = arith.constant {int(hw_dim)} : index")
        lines.append(f"      %cElems = arith.constant {int(elems)} : index")
        lines.append("      %pred = arith.cmpi ult, %bid, %cBlocks : index")
        lines.append("      scf.if %pred {")
        lines.append("        %n = arith.divui %bid, %cG : index")
        lines.append("        %grp = arith.remui %bid, %cG : index")
        lines.append("        %row_c = arith.muli %n, %cC : index")
        lines.append("        %grp_c = arith.muli %grp, %cGS : index")
        lines.append("        %c_base = arith.addi %row_c, %grp_c : index")
        lines.append("        %base = arith.muli %c_base, %cHW : index")

        # mean = sum(x) / elems
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %j = %tid to %cElems step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa['X']}[%idx] : {x_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %n_f = arith.constant {_as_f32_const(int(elems))} : f32")
        lines.append(f"        %mean_v = arith.divf %sumv, %n_f{fm} : f32")

        # var = sum((x-mean)^2) / elems
        lines.append("        %partial_var = scf.for %j2 = %tid to %cElems step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa['X']}[%idx2] : {x_memref}")
        lines.append(f"          %dx = arith.subf %x2, %mean_v{fm} : f32")
        lines.append(f"          %dx2 = arith.mulf %dx, %dx{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %dx2{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")
        lines.append("        memref.store %partial_var, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_var_{stride}"
            pS = f"%pS_var_{stride}"
            tid2 = f"%tid_var_{stride}"
            a = f"%a_var_{stride}"
            b = f"%b_var_{stride}"
            s = f"%s_var_{stride}"
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
        lines.append("        %var_sum = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %var = arith.divf %var_sum, %n_f{fm} : f32")
        lines.append(f"        %eps = arith.constant {_as_f32_const(eps_const)} : f32")
        lines.append(f"        %var_eps = arith.addf %var, %eps{fm} : f32")
        lines.append(f"        %rstd_v = math.rsqrt %var_eps{fm} : f32")

        # Store mean/rstd (one scalar each per (n,group))
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %mean_v, {arg_ssa['Mean']}[%bid] : {mean_memref}")
        lines.append(f"          memref.store %rstd_v, {arg_ssa['Rstd']}[%bid] : {rstd_memref}")
        lines.append("        }")

        # Output y: (x-mean)*rstd*weight + bias.
        lines.append("        scf.for %j3 = %tid to %cElems step %bdim {")
        lines.append("          %idx3 = arith.addi %base, %j3 : index")
        lines.append(f"          %x3 = memref.load {arg_ssa['X']}[%idx3] : {x_memref}")
        lines.append(f"          %dx3 = arith.subf %x3, %mean_v{fm} : f32")
        lines.append(f"          %xn = arith.mulf %dx3, %rstd_v{fm} : f32")
        lines.append("          %c_in_grp = arith.divui %j3, %cHW : index")
        lines.append("          %w_off = arith.muli %grp, %cGS : index")
        lines.append("          %w_idx = arith.addi %w_off, %c_in_grp : index")
        lines.append(f"          %w = memref.load {arg_ssa['W']}[%w_idx] : {w_memref}")
        lines.append(f"          %b = memref.load {arg_ssa['B']}[%w_idx] : {b_memref}")
        lines.append(f"          %xw = arith.mulf %xn, %w{fm} : f32")
        lines.append(f"          %y = arith.addf %xw, %b{fm} : f32")
        lines.append(f"          memref.store %y, {arg_ssa['Y']}[%idx3] : {y_memref}")
        lines.append("        }")
        lines.append("      }")
    elif per_token_group_quant_fp8_2d_v1 is not None:
        kernel_kind = "per_token_group_quant_fp8_2d_v1"
        m_dim = int(per_token_group_quant_fp8_2d_v1["M"])
        n_dim = int(per_token_group_quant_fp8_2d_v1["N"])
        g_dim = int(per_token_group_quant_fp8_2d_v1["G"])
        mg_dim = int(per_token_group_quant_fp8_2d_v1["MG"])
        group_size = int(per_token_group_quant_fp8_2d_v1["GROUP_SIZE"])
        if group_size <= 0 or group_size > 32:
            raise RuntimeError(f"per_token_group_quant_fp8_2d_v1 currently requires 1<=GROUP_SIZE<=32, got {group_size}")

        y_memref = str(arg_specs["y"]["memref"])
        eps_memref = str(arg_specs["eps"]["memref"])
        fp8_min_memref = str(arg_specs["fp8_min"]["memref"])
        fp8_max_memref = str(arg_specs["fp8_max"]["memref"])
        yq_memref = str(arg_specs["y_q"]["memref"])
        ys_memref = str(arg_specs["y_s"]["memref"])

        launch_override = {"block": [32, 1, 1], "grid": [int(mg_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cMG = arith.constant {int(mg_dim)} : index")
        lines.append(f"      %cG = arith.constant {int(g_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cGS = arith.constant {int(group_size)} : index")
        lines.append("      %pred = arith.cmpi ult, %bid, %cMG : index")
        lines.append("      scf.if %pred {")
        lines.append("        %row = arith.divui %bid, %cG : index")
        lines.append("        %grp = arith.remui %bid, %cG : index")
        lines.append("        %row_off = arith.muli %row, %cN : index")
        lines.append("        %grp_off = arith.muli %grp, %cGS : index")
        lines.append("        %base = arith.addi %row_off, %grp_off : index")
        lines.append("        %in_grp = arith.cmpi ult, %tid, %cGS : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %abs = scf.if %in_grp -> (f32) {")
        lines.append("          %idx = arith.addi %base, %tid : index")
        lines.append(f"          %x = memref.load {arg_ssa['y']}[%idx] : {y_memref}")
        lines.append(f"          %a = math.absf %x{fm} : f32")
        lines.append("          scf.yield %a : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")

        lines.append("        %c32 = arith.constant 32 : i32")
        lines.append("        %c16 = arith.constant 16 : i32")
        lines.append("        %c8 = arith.constant 8 : i32")
        lines.append("        %c4 = arith.constant 4 : i32")
        lines.append("        %c2 = arith.constant 2 : i32")
        lines.append("        %c1 = arith.constant 1 : i32")
        cur = "%abs"
        for off in ("%c16", "%c8", "%c4", "%c2", "%c1"):
            sh = _fresh("sh")
            ok = _fresh("ok")
            sel = _fresh("sel")
            nxt = _fresh("mx")
            lines.append(f"        {sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32 : f32")
            lines.append(f"        {sel} = arith.select {ok}, {sh}, %c0f : f32")
            lines.append(f"        {nxt} = arith.maximumf {cur}, {sel}{fm} : f32")
            cur = str(nxt)

        lines.append(f"        %eps_v = memref.load {arg_ssa['eps']}[%c0] : {eps_memref}")
        lines.append(f"        %fp8_min_v = memref.load {arg_ssa['fp8_min']}[%c0] : {fp8_min_memref}")
        lines.append(f"        %fp8_max_v = memref.load {arg_ssa['fp8_max']}[%c0] : {fp8_max_memref}")
        lines.append(f"        %absmax = arith.maximumf {cur}, %eps_v{fm} : f32")
        lines.append(f"        %scale = arith.divf %absmax, %fp8_max_v{fm} : f32")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %scale, {arg_ssa['y_s']}[%bid] : {ys_memref}")
        lines.append("        }")

        lines.append("        scf.if %in_grp {")
        lines.append("          %idx2 = arith.addi %base, %tid : index")
        lines.append(f"          %x2 = memref.load {arg_ssa['y']}[%idx2] : {y_memref}")
        lines.append(f"          %scaled = arith.divf %x2, %scale{fm} : f32")
        lines.append(f"          %cl0 = arith.maximumf %scaled, %fp8_min_v{fm} : f32")
        lines.append(f"          %cl = arith.minimumf %cl0, %fp8_max_v{fm} : f32")
        lines.append(f"          memref.store %cl, {arg_ssa['y_q']}[%idx2] : {yq_memref}")
        lines.append("        }")
        lines.append("      }")
    elif batch_norm2d_v1 is not None:
        kernel_kind = "batch_norm2d_v1"
        n_dim = int(batch_norm2d_v1["N"])
        c_dim = int(batch_norm2d_v1["C"])
        hw_dim = int(batch_norm2d_v1["HW"])
        if n_dim <= 0 or c_dim <= 0 or hw_dim <= 0:
            raise RuntimeError(f"batch_norm2d_v1 invalid dims: N={n_dim} C={c_dim} HW={hw_dim}")

        x_memref = str(arg_specs["input"]["memref"])
        w_memref = str(arg_specs["weight"]["memref"])
        b_memref = str(arg_specs["bias"]["memref"])
        rm_memref = str(arg_specs["running_mean"]["memref"])
        rv_memref = str(arg_specs["running_var"]["memref"])
        eps_memref = str(arg_specs["eps"]["memref"])
        mom_memref = str(arg_specs["momentum"]["memref"])
        ne_memref = str(arg_specs["n_elements"]["memref"])
        nm1_memref = str(arg_specs["n_minus_1"]["memref"])

        y_memref = str(arg_specs["output_1"]["memref"])
        mean_memref = str(arg_specs["mean"]["memref"])
        inv_memref = str(arg_specs["inv_std"]["memref"])
        rm_out_memref = str(arg_specs["running_mean_out"]["memref"])
        rv_out_memref = str(arg_specs["running_var_out"]["memref"])

        ch_stride = int(c_dim * hw_dim)
        elems = int(n_dim * hw_dim)
        launch_override = {"block": [256, 1, 1], "grid": [int(c_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"      %cHW = arith.constant {int(hw_dim)} : index")
        lines.append(f"      %cCHW = arith.constant {int(ch_stride)} : index")
        lines.append(f"      %cElems = arith.constant {int(elems)} : index")
        lines.append("      %pred_c = arith.cmpi ult, %bid, %cC : index")
        lines.append("      scf.if %pred_c {")
        lines.append(f"        %eps_v = memref.load {arg_ssa['eps']}[%c0] : {eps_memref}")
        lines.append(f"        %momentum_v = memref.load {arg_ssa['momentum']}[%c0] : {mom_memref}")
        lines.append(f"        %n_elements_v = memref.load {arg_ssa['n_elements']}[%c0] : {ne_memref}")
        lines.append(f"        %n_minus_1_v = memref.load {arg_ssa['n_minus_1']}[%c0] : {nm1_memref}")

        # mean sum across (n,hw)
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %i = %tid to %cElems step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %n = arith.divui %i, %cHW : index")
        lines.append("          %hw = arith.remui %i, %cHW : index")
        lines.append("          %n_off = arith.muli %n, %cCHW : index")
        lines.append("          %c_off = arith.muli %bid, %cHW : index")
        lines.append("          %base = arith.addi %n_off, %c_off : index")
        lines.append("          %idx = arith.addi %base, %hw : index")
        lines.append(f"          %x = memref.load {arg_ssa['input']}[%idx] : {x_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %mean_v = arith.divf %sumv, %n_elements_v{fm} : f32")

        # var sum across (n,hw)
        lines.append("        %partial_var = scf.for %i2 = %tid to %cElems step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %n2 = arith.divui %i2, %cHW : index")
        lines.append("          %hw2 = arith.remui %i2, %cHW : index")
        lines.append("          %n2_off = arith.muli %n2, %cCHW : index")
        lines.append("          %c2_off = arith.muli %bid, %cHW : index")
        lines.append("          %base2 = arith.addi %n2_off, %c2_off : index")
        lines.append("          %idx2 = arith.addi %base2, %hw2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa['input']}[%idx2] : {x_memref}")
        lines.append(f"          %dx = arith.subf %x2, %mean_v{fm} : f32")
        lines.append(f"          %dx2 = arith.mulf %dx, %dx{fm} : f32")
        lines.append(f"          %acc_next2 = arith.addf %acc, %dx2{fm} : f32")
        lines.append("          scf.yield %acc_next2 : f32")
        lines.append("        }")
        lines.append("        memref.store %partial_var, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_var_{stride}"
            pS = f"%pS_var_{stride}"
            tid2 = f"%tid_var_{stride}"
            a = f"%a_var_{stride}"
            b = f"%b_var_{stride}"
            s = f"%s_var_{stride}"
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
        lines.append("        %var_sum = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %var = arith.divf %var_sum, %n_elements_v{fm} : f32")
        lines.append(f"        %var_eps = arith.addf %var, %eps_v{fm} : f32")
        lines.append(f"        %inv_std_v = math.rsqrt %var_eps{fm} : f32")

        # store mean/inv_std + running stats (one lane)
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %mean_v, {arg_ssa['mean']}[%bid] : {mean_memref}")
        lines.append(f"          memref.store %inv_std_v, {arg_ssa['inv_std']}[%bid] : {inv_memref}")
        lines.append(f"          %rm_in = memref.load {arg_ssa['running_mean']}[%bid] : {rm_memref}")
        lines.append(f"          %rv_in = memref.load {arg_ssa['running_var']}[%bid] : {rv_memref}")
        lines.append("          %c1f = arith.constant 1.0 : f32")
        lines.append(f"          %one_minus = arith.subf %c1f, %momentum_v{fm} : f32")
        lines.append(f"          %rm_keep = arith.mulf %one_minus, %rm_in{fm} : f32")
        lines.append(f"          %rm_delta = arith.mulf %momentum_v, %mean_v{fm} : f32")
        lines.append(f"          %rm_out = arith.addf %rm_keep, %rm_delta{fm} : f32")
        lines.append(f"          memref.store %rm_out, {arg_ssa['running_mean_out']}[%bid] : {rm_out_memref}")
        lines.append(f"          %bessel = arith.divf %n_elements_v, %n_minus_1_v{fm} : f32")
        lines.append(f"          %unbiased = arith.mulf %var, %bessel{fm} : f32")
        lines.append(f"          %rv_keep = arith.mulf %one_minus, %rv_in{fm} : f32")
        lines.append(f"          %rv_delta = arith.mulf %momentum_v, %unbiased{fm} : f32")
        lines.append(f"          %rv_out = arith.addf %rv_keep, %rv_delta{fm} : f32")
        lines.append(f"          memref.store %rv_out, {arg_ssa['running_var_out']}[%bid] : {rv_out_memref}")
        lines.append("        }")

        # output_1
        lines.append(f"        %w = memref.load {arg_ssa['weight']}[%bid] : {w_memref}")
        lines.append(f"        %b = memref.load {arg_ssa['bias']}[%bid] : {b_memref}")
        lines.append("        scf.for %i3 = %tid to %cElems step %bdim {")
        lines.append("          %n3 = arith.divui %i3, %cHW : index")
        lines.append("          %hw3 = arith.remui %i3, %cHW : index")
        lines.append("          %n3_off = arith.muli %n3, %cCHW : index")
        lines.append("          %c3_off = arith.muli %bid, %cHW : index")
        lines.append("          %base3 = arith.addi %n3_off, %c3_off : index")
        lines.append("          %idx3 = arith.addi %base3, %hw3 : index")
        lines.append(f"          %x3 = memref.load {arg_ssa['input']}[%idx3] : {x_memref}")
        lines.append(f"          %dx3 = arith.subf %x3, %mean_v{fm} : f32")
        lines.append(f"          %xn = arith.mulf %dx3, %inv_std_v{fm} : f32")
        lines.append(f"          %xw = arith.mulf %xn, %w{fm} : f32")
        lines.append(f"          %y = arith.addf %xw, %b{fm} : f32")
        lines.append(f"          memref.store %y, {arg_ssa['output_1']}[%idx3] : {y_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_layer_norm_persistent is not None:
        kernel_kind = "layer_norm_axis1_v1"
        assert out_m is not None
        assert out_n is not None

        in_name = str(row_layer_norm_persistent.get("inp") or "in_ptr")
        w_name = str(row_layer_norm_persistent.get("weight") or "weight_ptr")
        b_name = str(row_layer_norm_persistent.get("bias") or "bias_ptr")
        out_name2 = str(row_layer_norm_persistent.get("out") or "out_ptr")
        mean_name = str(row_layer_norm_persistent.get("mean") or "out_mean_ptr")
        rstd_name = str(row_layer_norm_persistent.get("rstd") or "out_rstd_ptr")
        eps_const = float(row_layer_norm_persistent.get("eps_const") or 0.0)

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        mean_memref = str(arg_specs[mean_name]["memref"])
        rstd_memref = str(arg_specs[rstd_name]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(out_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append(f"        %eps = arith.constant {_as_f32_const(eps_const)} : f32")

        # sum
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %n_f = arith.constant {_as_f32_const(int(out_n))} : f32")
        lines.append(f"        %mean_v = arith.divf %sumv, %n_f{fm} : f32")

        # var = mean((x-mean)^2)
        lines.append("        %partial_var = scf.for %j2 = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
        lines.append(f"          %dx = arith.subf %x2, %mean_v{fm} : f32")
        lines.append(f"          %dx2 = arith.mulf %dx, %dx{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %dx2{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")
        lines.append("        memref.store %partial_var, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_var_{stride}"
            pS = f"%pS_var_{stride}"
            tid2 = f"%tid_var_{stride}"
            a = f"%a_var_{stride}"
            b = f"%b_var_{stride}"
            s = f"%s_var_{stride}"
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
        lines.append("        %var_sum = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %var = arith.divf %var_sum, %n_f{fm} : f32")
        lines.append(f"        %var_eps = arith.addf %var, %eps{fm} : f32")
        lines.append(f"        %rstd_v = math.rsqrt %var_eps{fm} : f32")

        # store mean/rstd (one lane)
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %mean_v, {arg_ssa[mean_name]}[%bid] : {mean_memref}")
        lines.append(f"          memref.store %rstd_v, {arg_ssa[rstd_name]}[%bid] : {rstd_memref}")
        lines.append("        }")

        # output
        lines.append("        scf.for %j3 = %tid to %cN step %bdim {")
        lines.append("          %idx3 = arith.addi %base, %j3 : index")
        lines.append(f"          %x3 = memref.load {arg_ssa[in_name]}[%idx3] : {in_memref}")
        lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%j3] : {w_memref}")
        lines.append(f"          %b = memref.load {arg_ssa[b_name]}[%j3] : {b_memref}")
        lines.append(f"          %dx3 = arith.subf %x3, %mean_v{fm} : f32")
        lines.append(f"          %xn = arith.mulf %dx3, %rstd_v{fm} : f32")
        lines.append(f"          %xw = arith.mulf %xn, %w{fm} : f32")
        lines.append(f"          %y = arith.addf %xw, %b{fm} : f32")
        lines.append(f"          memref.store %y, {arg_ssa[out_name2]}[%idx3] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_layer_norm_residual2d is not None:
        kernel_kind = "layer_norm_residual_axis1_v1"
        assert out_m is not None
        assert out_n is not None

        in_name = "inp"
        res_name = "residual"
        w_name = "weight"
        b_name = "bias"
        out_name2 = "out"
        mean_name = "mean"
        rstd_name = "rstd"
        eps_const = float(row_layer_norm_residual2d.get("eps_const") or 0.0)

        in_memref = str(arg_specs[in_name]["memref"])
        res_memref = str(arg_specs[res_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        mean_memref = str(arg_specs[mean_name]["memref"])
        rstd_memref = str(arg_specs[rstd_name]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(out_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append(f"        %eps = arith.constant {_as_f32_const(eps_const)} : f32")

        # sum(z)
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"          %r = memref.load {arg_ssa[res_name]}[%idx] : {res_memref}")
        lines.append(f"          %z = arith.addf %x, %r{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %z{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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
        lines.append("        %sumv = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %n_f = arith.constant {_as_f32_const(int(out_n))} : f32")
        lines.append(f"        %mean_v = arith.divf %sumv, %n_f{fm} : f32")

        # var = mean((z-mean)^2)
        lines.append("        %partial_var = scf.for %j2 = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
        lines.append(f"          %r2 = memref.load {arg_ssa[res_name]}[%idx2] : {res_memref}")
        lines.append(f"          %z2 = arith.addf %x2, %r2{fm} : f32")
        lines.append(f"          %dz = arith.subf %z2, %mean_v{fm} : f32")
        lines.append(f"          %dz2 = arith.mulf %dz, %dz{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %dz2{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")
        lines.append("        memref.store %partial_var, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_var_{stride}"
            pS = f"%pS_var_{stride}"
            tid2 = f"%tid_var_{stride}"
            a = f"%a_var_{stride}"
            b = f"%b_var_{stride}"
            s = f"%s_var_{stride}"
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
        lines.append("        %var_sum = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %var = arith.divf %var_sum, %n_f{fm} : f32")
        lines.append(f"        %var_eps = arith.addf %var, %eps{fm} : f32")
        lines.append(f"        %rstd_v = math.rsqrt %var_eps{fm} : f32")

        # store mean/rstd
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %mean_v, {arg_ssa[mean_name]}[%bid] : {mean_memref}")
        lines.append(f"          memref.store %rstd_v, {arg_ssa[rstd_name]}[%bid] : {rstd_memref}")
        lines.append("        }")

        # output
        lines.append("        scf.for %j3 = %tid to %cN step %bdim {")
        lines.append("          %idx3 = arith.addi %base, %j3 : index")
        lines.append(f"          %x3 = memref.load {arg_ssa[in_name]}[%idx3] : {in_memref}")
        lines.append(f"          %r3 = memref.load {arg_ssa[res_name]}[%idx3] : {res_memref}")
        lines.append(f"          %z3 = arith.addf %x3, %r3{fm} : f32")
        lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%j3] : {w_memref}")
        lines.append(f"          %b = memref.load {arg_ssa[b_name]}[%j3] : {b_memref}")
        lines.append(f"          %dz3 = arith.subf %z3, %mean_v{fm} : f32")
        lines.append(f"          %zn = arith.mulf %dz3, %rstd_v{fm} : f32")
        lines.append(f"          %zw = arith.mulf %zn, %w{fm} : f32")
        lines.append(f"          %y = arith.addf %zw, %b{fm} : f32")
        lines.append(f"          memref.store %y, {arg_ssa[out_name2]}[%idx3] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_rms_norm2d is not None:
        kernel_kind = "rms_norm_axis1_v2"
        assert out_m is not None
        assert out_n is not None

        in_name = str(row_rms_norm2d.get("inp_name") or "inp")
        w_name = str(row_rms_norm2d.get("weight_name") or "weight")
        out_name2 = str(row_rms_norm2d.get("out_name") or out_name)
        rstd_name = str(row_rms_norm2d.get("rstd_name") or "rstd")
        eps_tensor_name = str(row_rms_norm2d.get("eps_tensor_name") or "").strip()
        eps_const = float(row_rms_norm2d.get("eps_const") or 0.0)

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        rstd_memref = str(arg_specs[rstd_name]["memref"])
        eps_memref = str(arg_specs[eps_tensor_name]["memref"]) if eps_tensor_name else ""

        if int(out_n) == 64:
            kernel_kind = "rms_norm_axis1_v3"
            rows_per_block = max(1, min(4, int(out_m)))
            block_x = int(32 * int(rows_per_block))
            grid_x = int((int(out_m) + int(rows_per_block) - 1) // int(rows_per_block))
            launch_override = {"block": [int(block_x), 1, 1], "grid": [int(grid_x), 1, 1]}

            lines.append("      %tid = gpu.thread_id x")
            lines.append("      %bid_x = gpu.block_id x")
            lines.append("      %c0 = arith.constant 0 : index")
            lines.append("      %c1 = arith.constant 1 : index")
            lines.append("      %c2 = arith.constant 2 : index")
            lines.append("      %c32_idx = arith.constant 32 : index")
            lines.append(f"      %cRows = arith.constant {int(rows_per_block)} : index")
            lines.append(f"      %cGridX = arith.constant {int(grid_x)} : index")
            lines.append(f"      %cM = arith.constant {int(out_m)} : index")
            lines.append("      %cN = arith.constant 64 : index")
            lines.append("      %c32_i32 = arith.constant 32 : i32")
            lines.append("      %c16_i32 = arith.constant 16 : i32")
            lines.append("      %c8_i32 = arith.constant 8 : i32")
            lines.append("      %c4_i32 = arith.constant 4 : i32")
            lines.append("      %c2_i32 = arith.constant 2 : i32")
            lines.append("      %c1_i32 = arith.constant 1 : i32")
            lines.append("      %pred_block = arith.cmpi ult, %bid_x, %cGridX : index")
            lines.append("      scf.if %pred_block {")
            lines.append("        %lane = arith.remui %tid, %c32_idx : index")
            lines.append("        %warp = arith.divui %tid, %c32_idx : index")
            lines.append("        %row_base = arith.muli %bid_x, %cRows : index")
            lines.append("        %row = arith.addi %row_base, %warp : index")
            lines.append("        %pred_row = arith.cmpi ult, %row, %cM : index")
            lines.append("        scf.if %pred_row {")
            lines.append("          %base = arith.muli %row, %cN : index")
            eps_ssa = "%eps"
            if eps_tensor_name:
                eps_ssa = _fresh("eps")
                lines.append(f"          {eps_ssa} = memref.load {arg_ssa[eps_tensor_name]}[%c0] : {eps_memref}")
            else:
                lines.append(f"          %eps = arith.constant {_as_f32_const(eps_const)} : f32")
            lines.append("          %j0 = arith.muli %lane, %c2 : index")
            lines.append("          %idx0 = arith.addi %base, %j0 : index")
            lines.append("          %idx1 = arith.addi %idx0, %c1 : index")
            lines.append(f"          %x0 = memref.load {arg_ssa[in_name]}[%idx0] : {in_memref}")
            lines.append(f"          %x1 = memref.load {arg_ssa[in_name]}[%idx1] : {in_memref}")
            lines.append(f"          %x0_sq = arith.mulf %x0, %x0{fm} : f32")
            lines.append(f"          %x1_sq = arith.mulf %x1, %x1{fm} : f32")
            lines.append(f"          %sumsq0 = arith.addf %x0_sq, %x1_sq{fm} : f32")
            cur = "%sumsq0"
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                nxt = _fresh("sum")
                lines.append(f"          {sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"          {nxt} = arith.addf {cur}, {sh}{fm} : f32")
                cur = str(nxt)
            lines.append(f"          %n_f = arith.constant {_as_f32_const(64)} : f32")
            lines.append(f"          %mean_sq = arith.divf {cur}, %n_f{fm} : f32")
            lines.append(f"          %mean_eps = arith.addf %mean_sq, {eps_ssa}{fm} : f32")
            lines.append(f"          %rstd_v = math.rsqrt %mean_eps{fm} : f32")
            lines.append("          %is_lane0 = arith.cmpi eq, %lane, %c0 : index")
            lines.append("          scf.if %is_lane0 {")
            lines.append(f"            memref.store %rstd_v, {arg_ssa[rstd_name]}[%row] : {rstd_memref}")
            lines.append("          }")
            lines.append(f"          %w0 = memref.load {arg_ssa[w_name]}[%j0] : {w_memref}")
            lines.append("          %j1 = arith.addi %j0, %c1 : index")
            lines.append(f"          %w1 = memref.load {arg_ssa[w_name]}[%j1] : {w_memref}")
            lines.append(f"          %x0n = arith.mulf %x0, %rstd_v{fm} : f32")
            lines.append(f"          %x1n = arith.mulf %x1, %rstd_v{fm} : f32")
            lines.append(f"          %y0 = arith.mulf %x0n, %w0{fm} : f32")
            lines.append(f"          %y1 = arith.mulf %x1n, %w1{fm} : f32")
            lines.append(f"          memref.store %y0, {arg_ssa[out_name2]}[%idx0] : {out_memref}")
            lines.append(f"          memref.store %y1, {arg_ssa[out_name2]}[%idx1] : {out_memref}")
            lines.append("        }")
            lines.append("      }")
        else:
            kernel_kind = "rms_norm_axis1_v2"
            launch_override = {"block": [32, 1, 1], "grid": [int(out_m), 1, 1]}

            lines.append("      %tid = gpu.thread_id x")
            lines.append("      %bid = gpu.block_id x")
            lines.append("      %c0 = arith.constant 0 : index")
            lines.append(f"      %cM = arith.constant {int(out_m)} : index")
            lines.append(f"      %cN = arith.constant {int(out_n)} : index")
            lines.append("      %c32_idx = arith.constant 32 : index")
            lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
            lines.append("      scf.if %pred_row {")
            lines.append("        %base = arith.muli %bid, %cN : index")
            eps_ssa = "%eps"
            if eps_tensor_name:
                eps_ssa = _fresh("eps")
                lines.append(f"        {eps_ssa} = memref.load {arg_ssa[eps_tensor_name]}[%c0] : {eps_memref}")
            else:
                lines.append(f"        %eps = arith.constant {_as_f32_const(eps_const)} : f32")

            lines.append("        %c0f = arith.constant 0.0 : f32")
            lines.append("        %partial_sumsq = scf.for %j = %tid to %cN step %c32_idx iter_args(%acc = %c0f) -> (f32) {")
            lines.append("          %idx = arith.addi %base, %j : index")
            lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
            lines.append(f"          %xx = arith.mulf %x, %x{fm} : f32")
            lines.append(f"          %acc_next = arith.addf %acc, %xx{fm} : f32")
            lines.append("          scf.yield %acc_next : f32")
            lines.append("        }")

            lines.append("        %c32_i32 = arith.constant 32 : i32")
            lines.append("        %c16_i32 = arith.constant 16 : i32")
            lines.append("        %c8_i32 = arith.constant 8 : i32")
            lines.append("        %c4_i32 = arith.constant 4 : i32")
            lines.append("        %c2_i32 = arith.constant 2 : i32")
            lines.append("        %c1_i32 = arith.constant 1 : i32")
            cur = "%partial_sumsq"
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                nxt = _fresh("sum")
                lines.append(f"        {sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"        {nxt} = arith.addf {cur}, {sh}{fm} : f32")
                cur = str(nxt)

            lines.append(f"        %n_f = arith.constant {_as_f32_const(int(out_n))} : f32")
            lines.append(f"        %mean_sq = arith.divf {cur}, %n_f{fm} : f32")
            lines.append(f"        %mean_eps = arith.addf %mean_sq, {eps_ssa}{fm} : f32")
            lines.append(f"        %rstd_v = math.rsqrt %mean_eps{fm} : f32")

            lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
            lines.append("        scf.if %is0 {")
            lines.append(f"          memref.store %rstd_v, {arg_ssa[rstd_name]}[%bid] : {rstd_memref}")
            lines.append("        }")

            lines.append("        scf.for %j2 = %tid to %cN step %c32_idx {")
            lines.append("          %idx2 = arith.addi %base, %j2 : index")
            lines.append(f"          %x2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
            lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%j2] : {w_memref}")
            lines.append(f"          %xn = arith.mulf %x2, %rstd_v{fm} : f32")
            lines.append(f"          %y = arith.mulf %xn, %w{fm} : f32")
            lines.append(f"          memref.store %y, {arg_ssa[out_name2]}[%idx2] : {out_memref}")
            lines.append("        }")
            lines.append("      }")
    elif row_rms_norm_residual2d is not None:
        kernel_kind = "rms_norm_residual_axis1_v1"
        assert out_m is not None
        assert out_n is not None

        in_name = "inp"
        res_name = "residual"
        w_name = "weight"
        b_name = "bias"
        out_name2 = "out"
        rstd_name = "rstd"
        eps_const = float(row_rms_norm_residual2d.get("eps_const") or 0.0)

        in_memref = str(arg_specs[in_name]["memref"])
        res_memref = str(arg_specs[res_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        rstd_memref = str(arg_specs[rstd_name]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(out_m), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_m)} : index")
        lines.append(f"      %cN = arith.constant {int(out_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append(f"        %eps = arith.constant {_as_f32_const(eps_const)} : f32")

        # sumsq(z) where z = x + r + bias[j]
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %partial_sumsq = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"          %r = memref.load {arg_ssa[res_name]}[%idx] : {res_memref}")
        lines.append(f"          %b = memref.load {arg_ssa[b_name]}[%j] : {b_memref}")
        lines.append(f"          %xr = arith.addf %x, %r{fm} : f32")
        lines.append(f"          %z = arith.addf %xr, %b{fm} : f32")
        lines.append(f"          %zz = arith.mulf %z, %z{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %zz{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sumsq, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sumsq_{stride}"
            pS = f"%pS_sumsq_{stride}"
            tid2 = f"%tid_sumsq_{stride}"
            a = f"%a_sumsq_{stride}"
            b = f"%b_sumsq_{stride}"
            s = f"%s_sumsq_{stride}"
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
        lines.append("        %sumsq = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %n_f = arith.constant {_as_f32_const(int(out_n))} : f32")
        lines.append(f"        %mean_sq = arith.divf %sumsq, %n_f{fm} : f32")
        lines.append(f"        %mean_eps = arith.addf %mean_sq, %eps{fm} : f32")
        lines.append(f"        %rstd_v = math.rsqrt %mean_eps{fm} : f32")

        # store rstd
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store %rstd_v, {arg_ssa[rstd_name]}[%bid] : {rstd_memref}")
        lines.append("        }")

        # output
        lines.append("        scf.for %j2 = %tid to %cN step %bdim {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
        lines.append(f"          %r2 = memref.load {arg_ssa[res_name]}[%idx2] : {res_memref}")
        lines.append(f"          %b2 = memref.load {arg_ssa[b_name]}[%j2] : {b_memref}")
        lines.append(f"          %xr2 = arith.addf %x2, %r2{fm} : f32")
        lines.append(f"          %z2 = arith.addf %xr2, %b2{fm} : f32")
        lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%j2] : {w_memref}")
        lines.append(f"          %zn = arith.mulf %z2, %rstd_v{fm} : f32")
        lines.append(f"          %y = arith.mulf %zn, %w{fm} : f32")
        lines.append(f"          memref.store %y, {arg_ssa[out_name2]}[%idx2] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_reduce_min_argmin_axis1 is not None:
        kernel_kind = "row_reduce_min_argmin_axis1_v1"
        m_dim = int(row_reduce_min_argmin_axis1["M"])
        n_dim = int(row_reduce_min_argmin_axis1["N"])
        if m_dim != int(out_total):
            raise RuntimeError(f"min_dim2d expects out_total==M, got M={m_dim} out_total={int(out_total)}")
        if n_dim <= 0:
            raise RuntimeError(f"min_dim2d expects positive N, got N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}
        # Store (value, index) pairs as two consecutive f32s per thread.
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32x2"
        shared_global_memref_ty = "memref<512xf32, 3>"

        in_name = str(row_reduce_min_argmin_axis1["inp"])
        out_val_name = str(row_reduce_min_argmin_axis1["out_value"])
        out_idx_name = str(row_reduce_min_argmin_axis1["out_index"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_val_memref = str(arg_specs[out_val_name]["memref"])
        out_idx_memref = str(arg_specs[out_idx_name]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN_red = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %init = arith.constant 0x7F800000 : f32")
        lines.append("        %init_i32 = arith.constant 0 : i32")
        lines.append(
            "        %partial_v, %partial_i32 = scf.for %j = %tid to %cN_red step %bdim "
            "iter_args(%best = %init, %best_i = %init_i32) -> (f32, i32) {"
        )
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          %j_i32 = arith.index_cast %j : index to i32")
        lines.append("          %lt = arith.cmpf olt, %x, %best : f32")
        lines.append("          %eq = arith.cmpf oeq, %x, %best : f32")
        lines.append("          %lt_idx = arith.cmpi slt, %j_i32, %best_i : i32")
        lines.append("          %eq_and_lt = arith.andi %eq, %lt_idx : i1")
        lines.append("          %take = arith.ori %lt, %eq_and_lt : i1")
        lines.append(f"          %best_next = arith.select %take, %x, %best : f32")
        lines.append(f"          %best_i_next = arith.select %take, %j_i32, %best_i : i32")
        lines.append("          scf.yield %best_next, %best_i_next : f32, i32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %tid2 = arith.muli %tid, %c2 : index")
        lines.append(f"        memref.store %partial_v, %sh[%tid2] : {shared_global_memref_ty}")
        lines.append("        %tid2p1 = arith.addi %tid2, %c1 : index")
        lines.append("        %partial_i_f = arith.sitofp %partial_i32 : i32 to f32")
        lines.append(f"        memref.store %partial_i_f, %sh[%tid2p1] : {shared_global_memref_ty}")
        lines.append("        gpu.barrier")

        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_min_{stride}"
            pS = f"%pS_min_{stride}"
            tid2 = f"%tid_min_{stride}"
            base_a = f"%base_a_{stride}"
            base_b = f"%base_b_{stride}"
            a_val = f"%a_val_{stride}"
            a_idx_f = f"%a_idx_f_{stride}"
            b_val = f"%b_val_{stride}"
            b_idx_f = f"%b_idx_f_{stride}"
            a_idx_i32 = f"%a_idx_i32_{stride}"
            b_idx_i32 = f"%b_idx_i32_{stride}"
            lt = f"%lt_{stride}"
            eq = f"%eq_{stride}"
            lt_idx = f"%lt_idx_{stride}"
            eq_and_lt = f"%eq_and_lt_{stride}"
            take_b = f"%take_b_{stride}"
            new_val = f"%new_val_{stride}"
            new_idx = f"%new_idx_{stride}"
            new_idx_f = f"%new_idx_f_{stride}"
            base_a1 = f"%base_a1_{stride}"
            base_b1 = f"%base_b1_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {base_a} = arith.muli %tid, %c2 : index")
            lines.append(f"          {base_b} = arith.muli {tid2}, %c2 : index")
            lines.append(f"          {base_a1} = arith.addi {base_a}, %c1 : index")
            lines.append(f"          {base_b1} = arith.addi {base_b}, %c1 : index")
            lines.append(f"          {a_val} = memref.load %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_f} = memref.load %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append(f"          {b_val} = memref.load %sh[{base_b}] : {shared_global_memref_ty}")
            lines.append(f"          {b_idx_f} = memref.load %sh[{base_b1}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_i32} = arith.fptosi {a_idx_f} : f32 to i32")
            lines.append(f"          {b_idx_i32} = arith.fptosi {b_idx_f} : f32 to i32")
            lines.append(f"          {lt} = arith.cmpf olt, {b_val}, {a_val} : f32")
            lines.append(f"          {eq} = arith.cmpf oeq, {b_val}, {a_val} : f32")
            lines.append(f"          {lt_idx} = arith.cmpi slt, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          {eq_and_lt} = arith.andi {eq}, {lt_idx} : i1")
            lines.append(f"          {take_b} = arith.ori {lt}, {eq_and_lt} : i1")
            lines.append(f"          {new_val} = arith.select {take_b}, {b_val}, {a_val} : f32")
            lines.append(f"          {new_idx} = arith.select {take_b}, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          memref.store {new_val}, %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {new_idx_f} = arith.sitofp {new_idx} : i32 to f32")
            lines.append(f"          memref.store {new_idx_f}, %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          %minv = memref.load %sh[%c0] : {shared_global_memref_ty}")
        lines.append(f"          %idx_f = memref.load %sh[%c1] : {shared_global_memref_ty}")
        lines.append("          %idx_i32 = arith.fptosi %idx_f : f32 to i32")
        lines.append(f"          memref.store %minv, {arg_ssa[out_val_name]}[%bid] : {out_val_memref}")
        lines.append(f"          memref.store %idx_i32, {arg_ssa[out_idx_name]}[%bid] : {out_idx_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_argmax_axis1 is not None:
        kernel_kind = "row_argmax_axis1_v1"
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32x2"
        shared_global_memref_ty = "memref<512xf32, 3>"

        in_name = str(row_argmax_axis1["inp"])
        out_name2 = str(row_argmax_axis1["out"])
        red_n = int(row_argmax_axis1["reduce_n"])
        if red_n <= 0:
            raise RuntimeError(f"argmax2d expects positive N, got N={red_n}")
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %init = arith.constant 0xFF800000 : f32")
        lines.append("        %init_i32 = arith.constant 0 : i32")
        lines.append(
            "        %partial_v, %partial_i32 = scf.for %j = %tid to %cN_red step %bdim "
            "iter_args(%best = %init, %best_i = %init_i32) -> (f32, i32) {"
        )
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          %j_i32 = arith.index_cast %j : index to i32")
        lines.append("          %gt = arith.cmpf ogt, %x, %best : f32")
        lines.append("          %eq = arith.cmpf oeq, %x, %best : f32")
        lines.append("          %lt_idx = arith.cmpi slt, %j_i32, %best_i : i32")
        lines.append("          %eq_and_lt = arith.andi %eq, %lt_idx : i1")
        lines.append("          %take = arith.ori %gt, %eq_and_lt : i1")
        lines.append(f"          %best_next = arith.select %take, %x, %best : f32")
        lines.append(f"          %best_i_next = arith.select %take, %j_i32, %best_i : i32")
        lines.append("          scf.yield %best_next, %best_i_next : f32, i32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %tid2 = arith.muli %tid, %c2 : index")
        lines.append(f"        memref.store %partial_v, %sh[%tid2] : {shared_global_memref_ty}")
        lines.append("        %tid2p1 = arith.addi %tid2, %c1 : index")
        lines.append("        %partial_i_f = arith.sitofp %partial_i32 : i32 to f32")
        lines.append(f"        memref.store %partial_i_f, %sh[%tid2p1] : {shared_global_memref_ty}")
        lines.append("        gpu.barrier")

        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_max_{stride}"
            pS = f"%pS_max_{stride}"
            tid2 = f"%tid_max_{stride}"
            base_a = f"%base_a_{stride}"
            base_b = f"%base_b_{stride}"
            a_val = f"%a_val_{stride}"
            a_idx_f = f"%a_idx_f_{stride}"
            b_val = f"%b_val_{stride}"
            b_idx_f = f"%b_idx_f_{stride}"
            a_idx_i32 = f"%a_idx_i32_{stride}"
            b_idx_i32 = f"%b_idx_i32_{stride}"
            gt = f"%gt_{stride}"
            eq = f"%eq_{stride}"
            lt_idx = f"%lt_idx_{stride}"
            eq_and_lt = f"%eq_and_lt_{stride}"
            take_b = f"%take_b_{stride}"
            new_val = f"%new_val_{stride}"
            new_idx = f"%new_idx_{stride}"
            new_idx_f = f"%new_idx_f_{stride}"
            base_a1 = f"%base_a1_{stride}"
            base_b1 = f"%base_b1_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {base_a} = arith.muli %tid, %c2 : index")
            lines.append(f"          {base_b} = arith.muli {tid2}, %c2 : index")
            lines.append(f"          {base_a1} = arith.addi {base_a}, %c1 : index")
            lines.append(f"          {base_b1} = arith.addi {base_b}, %c1 : index")
            lines.append(f"          {a_val} = memref.load %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_f} = memref.load %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append(f"          {b_val} = memref.load %sh[{base_b}] : {shared_global_memref_ty}")
            lines.append(f"          {b_idx_f} = memref.load %sh[{base_b1}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_i32} = arith.fptosi {a_idx_f} : f32 to i32")
            lines.append(f"          {b_idx_i32} = arith.fptosi {b_idx_f} : f32 to i32")
            lines.append(f"          {gt} = arith.cmpf ogt, {b_val}, {a_val} : f32")
            lines.append(f"          {eq} = arith.cmpf oeq, {b_val}, {a_val} : f32")
            lines.append(f"          {lt_idx} = arith.cmpi slt, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          {eq_and_lt} = arith.andi {eq}, {lt_idx} : i1")
            lines.append(f"          {take_b} = arith.ori {gt}, {eq_and_lt} : i1")
            lines.append(f"          {new_val} = arith.select {take_b}, {b_val}, {a_val} : f32")
            lines.append(f"          {new_idx} = arith.select {take_b}, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          memref.store {new_val}, %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {new_idx_f} = arith.sitofp {new_idx} : i32 to f32")
            lines.append(f"          memref.store {new_idx_f}, %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          %idx_f = memref.load %sh[%c1] : {shared_global_memref_ty}")
        lines.append("          %idx_i32 = arith.fptosi %idx_f : f32 to i32")
        lines.append(f"          memref.store %idx_i32, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_argmin_axis1 is not None:
        kernel_kind = "row_argmin_axis1_v1"
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32x2"
        shared_global_memref_ty = "memref<512xf32, 3>"

        in_name = str(row_argmin_axis1["inp"])
        out_name2 = str(row_argmin_axis1["out"])
        red_n = int(row_argmin_axis1["reduce_n"])
        if red_n <= 0:
            raise RuntimeError(f"argmin2d expects positive N, got N={red_n}")
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %init = arith.constant 0x7F800000 : f32")
        lines.append("        %init_i32 = arith.constant 0 : i32")
        lines.append(
            "        %partial_v, %partial_i32 = scf.for %j = %tid to %cN_red step %bdim "
            "iter_args(%best = %init, %best_i = %init_i32) -> (f32, i32) {"
        )
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          %j_i32 = arith.index_cast %j : index to i32")
        lines.append("          %lt = arith.cmpf olt, %x, %best : f32")
        lines.append("          %eq = arith.cmpf oeq, %x, %best : f32")
        lines.append("          %lt_idx = arith.cmpi slt, %j_i32, %best_i : i32")
        lines.append("          %eq_and_lt = arith.andi %eq, %lt_idx : i1")
        lines.append("          %take = arith.ori %lt, %eq_and_lt : i1")
        lines.append(f"          %best_next = arith.select %take, %x, %best : f32")
        lines.append(f"          %best_i_next = arith.select %take, %j_i32, %best_i : i32")
        lines.append("          scf.yield %best_next, %best_i_next : f32, i32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %tid2 = arith.muli %tid, %c2 : index")
        lines.append(f"        memref.store %partial_v, %sh[%tid2] : {shared_global_memref_ty}")
        lines.append("        %tid2p1 = arith.addi %tid2, %c1 : index")
        lines.append("        %partial_i_f = arith.sitofp %partial_i32 : i32 to f32")
        lines.append(f"        memref.store %partial_i_f, %sh[%tid2p1] : {shared_global_memref_ty}")
        lines.append("        gpu.barrier")

        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_min_{stride}"
            pS = f"%pS_min_{stride}"
            tid2 = f"%tid_min_{stride}"
            base_a = f"%base_a_{stride}"
            base_b = f"%base_b_{stride}"
            a_val = f"%a_val_{stride}"
            a_idx_f = f"%a_idx_f_{stride}"
            b_val = f"%b_val_{stride}"
            b_idx_f = f"%b_idx_f_{stride}"
            a_idx_i32 = f"%a_idx_i32_{stride}"
            b_idx_i32 = f"%b_idx_i32_{stride}"
            lt = f"%lt_{stride}"
            eq = f"%eq_{stride}"
            lt_idx = f"%lt_idx_{stride}"
            eq_and_lt = f"%eq_and_lt_{stride}"
            take_b = f"%take_b_{stride}"
            new_val = f"%new_val_{stride}"
            new_idx = f"%new_idx_{stride}"
            new_idx_f = f"%new_idx_f_{stride}"
            base_a1 = f"%base_a1_{stride}"
            base_b1 = f"%base_b1_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {base_a} = arith.muli %tid, %c2 : index")
            lines.append(f"          {base_b} = arith.muli {tid2}, %c2 : index")
            lines.append(f"          {base_a1} = arith.addi {base_a}, %c1 : index")
            lines.append(f"          {base_b1} = arith.addi {base_b}, %c1 : index")
            lines.append(f"          {a_val} = memref.load %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_f} = memref.load %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append(f"          {b_val} = memref.load %sh[{base_b}] : {shared_global_memref_ty}")
            lines.append(f"          {b_idx_f} = memref.load %sh[{base_b1}] : {shared_global_memref_ty}")
            lines.append(f"          {a_idx_i32} = arith.fptosi {a_idx_f} : f32 to i32")
            lines.append(f"          {b_idx_i32} = arith.fptosi {b_idx_f} : f32 to i32")
            lines.append(f"          {lt} = arith.cmpf olt, {b_val}, {a_val} : f32")
            lines.append(f"          {eq} = arith.cmpf oeq, {b_val}, {a_val} : f32")
            lines.append(f"          {lt_idx} = arith.cmpi slt, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          {eq_and_lt} = arith.andi {eq}, {lt_idx} : i1")
            lines.append(f"          {take_b} = arith.ori {lt}, {eq_and_lt} : i1")
            lines.append(f"          {new_val} = arith.select {take_b}, {b_val}, {a_val} : f32")
            lines.append(f"          {new_idx} = arith.select {take_b}, {b_idx_i32}, {a_idx_i32} : i32")
            lines.append(f"          memref.store {new_val}, %sh[{base_a}] : {shared_global_memref_ty}")
            lines.append(f"          {new_idx_f} = arith.sitofp {new_idx} : i32 to f32")
            lines.append(f"          memref.store {new_idx_f}, %sh[{base_a1}] : {shared_global_memref_ty}")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          %idx_f = memref.load %sh[%c1] : {shared_global_memref_ty}")
        lines.append("          %idx_i32 = arith.fptosi %idx_f : f32 to i32")
        lines.append(f"          memref.store %idx_i32, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_reduce_prod_axis1 is not None:
        kernel_kind = "row_reduce_prod_axis1_v1"
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        in_name = str(row_reduce_prod_axis1["red_in"])
        out_name2 = str(row_reduce_prod_axis1["red_out"])
        red_n = int(row_reduce_prod_axis1["reduce_n"])
        if red_n <= 0:
            raise RuntimeError(f"prod_dim2d expects positive N, got N={red_n}")
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %c1f = arith.constant 1.0 : f32")
        lines.append("        %partial = scf.for %j = %tid to %cN_red step %bdim iter_args(%acc = %c1f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"          %acc_next = arith.mulf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_prod_{stride}"
            pS = f"%pS_prod_{stride}"
            tid2 = f"%tid_prod_{stride}"
            a = f"%a_prod_{stride}"
            b = f"%b_prod_{stride}"
            s = f"%s_prod_{stride}"
            lines.append(f"        {cS} = arith.constant {int(stride)} : index")
            lines.append(f"        {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"        scf.if {pS} {{")
            lines.append(f"          {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"          {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"          {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"          {s} = arith.mulf {a}, {b}{fm} : f32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %prod0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"          memref.store %prod0, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif reduce_min_all_v1 is not None:
        kernel_kind = "reduce_min_all_v1"
        m_dim = int(reduce_min_all_v1["M"])
        n_dim = int(reduce_min_all_v1["N"])
        total = int(m_dim * n_dim)
        if total <= 0:
            raise RuntimeError(f"min2d expects positive numel, got M={m_dim} N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        in_name = str(reduce_min_all_v1["inp"])
        out_name2 = str(reduce_min_all_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(total)} : index")
        lines.append("      %init = arith.constant 0x7F800000 : f32")
        lines.append("      %partial = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %init) -> (f32) {")
        lines.append(f"        %x = memref.load {arg_ssa[in_name]}[%j] : {in_memref}")
        lines.append(f"        %acc_next = arith.minimumf %acc, %x{fm} : f32")
        lines.append("        scf.yield %acc_next : f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_minall_{stride}"
            pS = f"%pS_minall_{stride}"
            tid2 = f"%tid_minall_{stride}"
            a = f"%a_minall_{stride}"
            b = f"%b_minall_{stride}"
            s = f"%s_minall_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.minimumf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %min0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        memref.store %min0, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif reduce_prod_all_v1 is not None:
        kernel_kind = "reduce_prod_all_v1"
        m_dim = int(reduce_prod_all_v1["M"])
        n_dim = int(reduce_prod_all_v1["N"])
        total = int(m_dim * n_dim)
        if total <= 0:
            raise RuntimeError(f"prod2d expects positive numel, got M={m_dim} N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        in_name = str(reduce_prod_all_v1["inp"])
        out_name2 = str(reduce_prod_all_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(total)} : index")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %partial = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c1f) -> (f32) {")
        lines.append(f"        %x = memref.load {arg_ssa[in_name]}[%j] : {in_memref}")
        lines.append(f"        %acc_next = arith.mulf %acc, %x{fm} : f32")
        lines.append("        scf.yield %acc_next : f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_prodall_{stride}"
            pS = f"%pS_prodall_{stride}"
            tid2 = f"%tid_prodall_{stride}"
            a = f"%a_prodall_{stride}"
            b = f"%b_prodall_{stride}"
            s = f"%s_prodall_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.mulf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %prod0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        memref.store %prod0, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif trace2d_v1 is not None:
        kernel_kind = "trace2d_v1"
        m_dim = int(trace2d_v1["M"])
        n_dim = int(trace2d_v1["N"])
        diag = int(min(m_dim, n_dim))
        if diag <= 0:
            raise RuntimeError(f"trace2d expects positive diag, got M={m_dim} N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        in_name = str(trace2d_v1["inp"])
        out_name2 = str(trace2d_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cDiag = arith.constant {int(diag)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %partial = scf.for %i = %tid to %cDiag step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("        %row_off = arith.muli %i, %cN : index")
        lines.append("        %idx = arith.addi %row_off, %i : index")
        lines.append(f"        %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"        %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("        scf.yield %acc_next : f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_tr_{stride}"
            pS = f"%pS_tr_{stride}"
            tid2 = f"%tid_tr_{stride}"
            a = f"%a_tr_{stride}"
            b = f"%b_tr_{stride}"
            s = f"%s_tr_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %sum0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        memref.store %sum0, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif count_nonzero2d_v1 is not None:
        kernel_kind = "count_nonzero2d_v1"
        m_dim = int(count_nonzero2d_v1["M"])
        n_dim = int(count_nonzero2d_v1["N"])
        total = int(m_dim * n_dim)
        if total <= 0:
            raise RuntimeError(f"count_nonzero2d expects positive numel, got M={m_dim} N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_i64"
        shared_global_memref_ty = "memref<256xi64, 3>"

        in_name = str(count_nonzero2d_v1["inp"])
        out_name2 = str(count_nonzero2d_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(total)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c0i64 = arith.constant 0 : i64")
        lines.append("      %c1i64 = arith.constant 1 : i64")
        lines.append("      %partial = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0i64) -> (i64) {")
        lines.append(f"        %x_val = memref.load {arg_ssa[in_name]}[%j] : {in_memref}")
        lines.append("        %nz = arith.cmpf one, %x_val, %c0f : f32")
        lines.append("        %inc = arith.select %nz, %c1i64, %c0i64 : i64")
        lines.append("        %acc_next = arith.addi %acc, %inc : i64")
        lines.append("        scf.yield %acc_next : i64")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      memref.store %partial, %sh[%tid] : memref<256xi64, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_cn_{stride}"
            pS = f"%pS_cn_{stride}"
            tid2 = f"%tid_cn_{stride}"
            a = f"%a_cn_{stride}"
            b = f"%b_cn_{stride}"
            s = f"%s_cn_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xi64, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xi64, 3>")
            lines.append(f"        {s} = arith.addi {a}, {b} : i64")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xi64, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %sum0 = memref.load %sh[%c0] : memref<256xi64, 3>")
        lines.append(f"        memref.store %sum0, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif allclose2d_v1 is not None:
        kernel_kind = "allclose2d_v1"
        m_dim = int(allclose2d_v1["M"])
        n_dim = int(allclose2d_v1["N"])
        total = int(m_dim * n_dim)
        if total <= 0:
            raise RuntimeError(f"allclose2d expects positive numel, got M={m_dim} N={n_dim}")

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_i32"
        shared_global_memref_ty = "memref<256xi32, 3>"

        a_name = str(allclose2d_v1["A"])
        b_name = str(allclose2d_v1["B"])
        rtol_name = str(allclose2d_v1["rtol"])
        atol_name = str(allclose2d_v1["atol"])
        out_name2 = str(allclose2d_v1["out"])
        a_memref = str(arg_specs[a_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        rtol_memref = str(arg_specs[rtol_name]["memref"])
        atol_memref = str(arg_specs[atol_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(total)} : index")
        lines.append("      %c1_i1 = arith.constant 1 : i1")
        lines.append("      %false = arith.constant 0 : i1")
        lines.append(f"      %rtol_val = memref.load {arg_ssa[rtol_name]}[%c0] : {rtol_memref}")
        lines.append(f"      %atol_val = memref.load {arg_ssa[atol_name]}[%c0] : {atol_memref}")
        lines.append("      %any = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %false) -> (i1) {")
        lines.append(f"        %a = memref.load {arg_ssa[a_name]}[%j] : {a_memref}")
        lines.append(f"        %b = memref.load {arg_ssa[b_name]}[%j] : {b_memref}")
        lines.append(f"        %diff = arith.subf %a, %b{fm} : f32")
        lines.append(f"        %abs_diff = math.absf %diff{fm} : f32")
        lines.append(f"        %abs_b = math.absf %b{fm} : f32")
        lines.append(f"        %rtol_term = arith.mulf %rtol_val, %abs_b{fm} : f32")
        lines.append(f"        %tol = arith.addf %atol_val, %rtol_term{fm} : f32")
        lines.append("        %close = arith.cmpf ole, %abs_diff, %tol : f32")
        lines.append("        %not_close = arith.xori %close, %c1_i1 : i1")
        lines.append("        %acc_next = arith.ori %acc, %not_close : i1")
        lines.append("        scf.yield %acc_next : i1")
        lines.append("      }")

        # Reduce any-not-close across the block in shared memory.
        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      %any_i32 = arith.extui %any : i1 to i32")
        lines.append("      memref.store %any_i32, %sh[%tid] : memref<256xi32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_ac_{stride}"
            pS = f"%pS_ac_{stride}"
            tid2 = f"%tid_ac_{stride}"
            a = f"%a_ac_{stride}"
            b = f"%b_ac_{stride}"
            s = f"%s_ac_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xi32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xi32, 3>")
            lines.append(f"        {s} = arith.ori {a}, {b} : i32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xi32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %any0_i32 = memref.load %sh[%c0] : memref<256xi32, 3>")
        lines.append("        %c0i32 = arith.constant 0 : i32")
        lines.append("        %any0 = arith.cmpi ne, %any0_i32, %c0i32 : i32")
        lines.append("        %out_i1 = arith.xori %any0, %c1_i1 : i1")
        lines.append("        %out_i8 = arith.extui %out_i1 : i1 to i8")
        lines.append(f"        memref.store %out_i8, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif row_sort_axis1_bitonic_v1 is not None:
        stable = bool(row_sort_axis1_bitonic_v1.get("stable"))
        descending = bool(row_sort_axis1_bitonic_v1.get("descending"))
        if stable:
            if descending:
                raise RuntimeError("sort_stable2d descending unsupported in row_sort_axis1_bitonic_v1")
            kernel_kind = "row_sort_axis1_bitonic_stable_v1"
        else:
            kernel_kind = "row_sort_axis1_bitonic_v1"
        m_dim = int(row_sort_axis1_bitonic_v1["M"])
        n_dim = int(row_sort_axis1_bitonic_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim}")
        sort_len = 1 << (int(n_dim - 1).bit_length())
        if sort_len > 256:
            raise RuntimeError(f"{kernel_kind} currently requires N<=256, got N={n_dim}")

        in_name = str(row_sort_axis1_bitonic_v1["inp"])
        out_name2 = str(row_sort_axis1_bitonic_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_threads = int(sort_len)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(m_dim), 1, 1]}
        if stable:
            shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32x2"
            shared_global_memref_ty = "memref<512xf32, 3>"
        else:
            shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
            shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(sort_len)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %in_bounds = arith.cmpi ult, %tid, %cN : index")
        lines.append("        %pos_inf = arith.constant 0x7F800000 : f32")
        lines.append("        %val = scf.if %in_bounds -> (f32) {")
        lines.append("          %idx = arith.addi %base, %tid : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          scf.yield %x : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %pos_inf : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %tid_i32 = arith.index_cast %tid : index to i32")
        if stable:
            lines.append("        %base2 = arith.muli %tid, %c2 : index")
            lines.append("        memref.store %val, %sh[%base2] : memref<512xf32, 3>")
            lines.append("        %base2p1 = arith.addi %base2, %c1 : index")
            lines.append("        %tid_f = arith.sitofp %tid_i32 : i32 to f32")
            lines.append("        memref.store %tid_f, %sh[%base2p1] : memref<512xf32, 3>")
        else:
            lines.append("        memref.store %val, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")

        lines.append("        %c0_i32 = arith.constant 0 : i32")
        for k in [2**x for x in range(1, int(sort_len).bit_length() + 1) if 2**x <= int(sort_len)]:
            lines.append(f"        %cK_{k}_i32 = arith.constant {int(k)} : i32")
            lines.append(f"        %and_k_{k} = arith.andi %tid_i32, %cK_{k}_i32 : i32")
            lines.append(f"        %dir_k_{k} = arith.cmpi eq, %and_k_{k}, %c0_i32 : i32")
            j = k // 2
            while j >= 1:
                lines.append(f"        %cJ_{k}_{j}_i32 = arith.constant {int(j)} : i32")
                lines.append(f"        %ixj_{k}_{j}_i32 = arith.xori %tid_i32, %cJ_{k}_{j}_i32 : i32")
                lines.append(f"        %do_{k}_{j} = arith.cmpi ugt, %ixj_{k}_{j}_i32, %tid_i32 : i32")
                lines.append(f"        scf.if %do_{k}_{j} {{")
                lines.append(f"          %ixj_{k}_{j} = arith.index_cast %ixj_{k}_{j}_i32 : i32 to index")
                if stable:
                    lines.append(f"          %base_a_{k}_{j} = arith.muli %tid, %c2 : index")
                    lines.append(f"          %base_b_{k}_{j} = arith.muli %ixj_{k}_{j}, %c2 : index")
                    lines.append(f"          %base_a1_{k}_{j} = arith.addi %base_a_{k}_{j}, %c1 : index")
                    lines.append(f"          %base_b1_{k}_{j} = arith.addi %base_b_{k}_{j}, %c1 : index")
                    lines.append(
                        f"          %a_val_{k}_{j} = memref.load %sh[%base_a_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(
                        f"          %b_val_{k}_{j} = memref.load %sh[%base_b_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(
                        f"          %a_idx_f_{k}_{j} = memref.load %sh[%base_a1_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(
                        f"          %b_idx_f_{k}_{j} = memref.load %sh[%base_b1_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(f"          %a_idx_{k}_{j} = arith.fptosi %a_idx_f_{k}_{j} : f32 to i32")
                    lines.append(f"          %b_idx_{k}_{j} = arith.fptosi %b_idx_f_{k}_{j} : f32 to i32")
                    lines.append(
                        f"          %val_gt_{k}_{j} = arith.cmpf ogt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                    )
                    lines.append(
                        f"          %val_lt_{k}_{j} = arith.cmpf olt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                    )
                    lines.append(
                        f"          %val_eq_{k}_{j} = arith.cmpf oeq, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                    )
                    lines.append(f"          %idx_gt_{k}_{j} = arith.cmpi sgt, %a_idx_{k}_{j}, %b_idx_{k}_{j} : i32")
                    lines.append(f"          %idx_lt_{k}_{j} = arith.cmpi slt, %a_idx_{k}_{j}, %b_idx_{k}_{j} : i32")
                    lines.append(f"          %eq_and_gt_{k}_{j} = arith.andi %val_eq_{k}_{j}, %idx_gt_{k}_{j} : i1")
                    lines.append(f"          %eq_and_lt_{k}_{j} = arith.andi %val_eq_{k}_{j}, %idx_lt_{k}_{j} : i1")
                    lines.append(f"          %key_gt_{k}_{j} = arith.ori %val_gt_{k}_{j}, %eq_and_gt_{k}_{j} : i1")
                    lines.append(f"          %key_lt_{k}_{j} = arith.ori %val_lt_{k}_{j}, %eq_and_lt_{k}_{j} : i1")
                    lines.append(f"          %swap_{k}_{j} = arith.select %dir_k_{k}, %key_gt_{k}_{j}, %key_lt_{k}_{j} : i1")
                    lines.append(
                        f"          %new_a_val_{k}_{j} = arith.select %swap_{k}_{j}, %b_val_{k}_{j}, %a_val_{k}_{j} : f32"
                    )
                    lines.append(
                        f"          %new_b_val_{k}_{j} = arith.select %swap_{k}_{j}, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                    )
                    lines.append(
                        f"          %new_a_idx_{k}_{j} = arith.select %swap_{k}_{j}, %b_idx_{k}_{j}, %a_idx_{k}_{j} : i32"
                    )
                    lines.append(
                        f"          %new_b_idx_{k}_{j} = arith.select %swap_{k}_{j}, %a_idx_{k}_{j}, %b_idx_{k}_{j} : i32"
                    )
                    lines.append(
                        f"          memref.store %new_a_val_{k}_{j}, %sh[%base_a_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(
                        f"          memref.store %new_b_val_{k}_{j}, %sh[%base_b_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(f"          %new_a_idx_f_{k}_{j} = arith.sitofp %new_a_idx_{k}_{j} : i32 to f32")
                    lines.append(f"          %new_b_idx_f_{k}_{j} = arith.sitofp %new_b_idx_{k}_{j} : i32 to f32")
                    lines.append(
                        f"          memref.store %new_a_idx_f_{k}_{j}, %sh[%base_a1_{k}_{j}] : {shared_global_memref_ty}"
                    )
                    lines.append(
                        f"          memref.store %new_b_idx_f_{k}_{j}, %sh[%base_b1_{k}_{j}] : {shared_global_memref_ty}"
                    )
                else:
                    lines.append(f"          %a_val_{k}_{j} = memref.load %sh[%tid] : {shared_global_memref_ty}")
                    lines.append(f"          %b_val_{k}_{j} = memref.load %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}")
                    lines.append(f"          %val_gt_{k}_{j} = arith.cmpf ogt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                    lines.append(f"          %val_lt_{k}_{j} = arith.cmpf olt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                    lines.append(f"          %swap_{k}_{j} = arith.select %dir_k_{k}, %val_gt_{k}_{j}, %val_lt_{k}_{j} : i1")
                    lines.append(
                        f"          %new_a_val_{k}_{j} = arith.select %swap_{k}_{j}, %b_val_{k}_{j}, %a_val_{k}_{j} : f32"
                    )
                    lines.append(
                        f"          %new_b_val_{k}_{j} = arith.select %swap_{k}_{j}, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                    )
                    lines.append(f"          memref.store %new_a_val_{k}_{j}, %sh[%tid] : {shared_global_memref_ty}")
                    lines.append(
                        f"          memref.store %new_b_val_{k}_{j}, %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}"
                    )
                lines.append("        }")
                lines.append("        gpu.barrier")
                j //= 2

        # Write out sorted results.
        lines.append("        %out_pred = arith.cmpi ult, %tid, %cN : index")
        lines.append("        scf.if %out_pred {")
        lines.append("          %out_idx = arith.addi %base, %tid : index")
        if stable:
            lines.append("          %base2_out = arith.muli %tid, %c2 : index")
            lines.append("          %v = memref.load %sh[%base2_out] : memref<512xf32, 3>")
            lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        else:
            if descending:
                lines.append(f"          %cN1 = arith.constant {int(n_dim - 1)} : index")
                lines.append("          %src = arith.subi %cN1, %tid : index")
                lines.append("          %v = memref.load %sh[%src] : memref<256xf32, 3>")
            else:
                lines.append("          %v = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_topk_axis1_bitonic_v1 is not None:
        kernel_kind = "row_topk_axis1_bitonic_v1"
        m_dim = int(row_topk_axis1_bitonic_v1["M"])
        n_dim = int(row_topk_axis1_bitonic_v1["N"])
        k_dim = int(row_topk_axis1_bitonic_v1["K"])
        if m_dim <= 0 or n_dim <= 0 or k_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} K={k_dim}")
        if k_dim > n_dim:
            raise RuntimeError(f"{kernel_kind} requires K<=N, got K={k_dim} N={n_dim}")
        sort_len = 1 << (int(n_dim - 1).bit_length())
        if sort_len > 256:
            raise RuntimeError(f"{kernel_kind} currently requires N<=256, got N={n_dim}")

        in_name = str(row_topk_axis1_bitonic_v1["inp"])
        out_name2 = str(row_topk_axis1_bitonic_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_threads = int(sort_len)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cK = arith.constant {int(k_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(sort_len)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %in_bounds = arith.cmpi ult, %tid, %cN : index")
        lines.append("        %neg_inf = arith.constant 0xFF800000 : f32")
        lines.append("        %val = scf.if %in_bounds -> (f32) {")
        lines.append("          %idx = arith.addi %base, %tid : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          scf.yield %x : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %neg_inf : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %val, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")

        lines.append("        %tid_i32 = arith.index_cast %tid : index to i32")
        lines.append("        %c0_i32 = arith.constant 0 : i32")
        for k in [2**x for x in range(1, int(sort_len).bit_length() + 1) if 2**x <= int(sort_len)]:
            lines.append(f"        %cK_{k}_i32 = arith.constant {int(k)} : i32")
            lines.append(f"        %and_k_{k} = arith.andi %tid_i32, %cK_{k}_i32 : i32")
            lines.append(f"        %dir_k_{k} = arith.cmpi eq, %and_k_{k}, %c0_i32 : i32")
            j = k // 2
            while j >= 1:
                lines.append(f"        %cJ_{k}_{j}_i32 = arith.constant {int(j)} : i32")
                lines.append(f"        %ixj_{k}_{j}_i32 = arith.xori %tid_i32, %cJ_{k}_{j}_i32 : i32")
                lines.append(f"        %do_{k}_{j} = arith.cmpi ugt, %ixj_{k}_{j}_i32, %tid_i32 : i32")
                lines.append(f"        scf.if %do_{k}_{j} {{")
                lines.append(f"          %ixj_{k}_{j} = arith.index_cast %ixj_{k}_{j}_i32 : i32 to index")
                lines.append(f"          %a_val_{k}_{j} = memref.load %sh[%tid] : {shared_global_memref_ty}")
                lines.append(f"          %b_val_{k}_{j} = memref.load %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}")
                lines.append(f"          %val_gt_{k}_{j} = arith.cmpf ogt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                lines.append(f"          %val_lt_{k}_{j} = arith.cmpf olt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                lines.append(f"          %swap_{k}_{j} = arith.select %dir_k_{k}, %val_gt_{k}_{j}, %val_lt_{k}_{j} : i1")
                lines.append(
                    f"          %new_a_val_{k}_{j} = arith.select %swap_{k}_{j}, %b_val_{k}_{j}, %a_val_{k}_{j} : f32"
                )
                lines.append(
                    f"          %new_b_val_{k}_{j} = arith.select %swap_{k}_{j}, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                )
                lines.append(f"          memref.store %new_a_val_{k}_{j}, %sh[%tid] : {shared_global_memref_ty}")
                lines.append(
                    f"          memref.store %new_b_val_{k}_{j}, %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}"
                )
                lines.append("        }")
                lines.append("        gpu.barrier")
                j //= 2

        lines.append("        %out_pred = arith.cmpi ult, %tid, %cK : index")
        lines.append("        scf.if %out_pred {")
        lines.append("          %out_base = arith.muli %bid, %cK : index")
        lines.append("          %out_idx = arith.addi %out_base, %tid : index")
        lines.append(f"          %cL1 = arith.constant {int(sort_len - 1)} : index")
        lines.append("          %src = arith.subi %cL1, %tid : index")
        lines.append("          %v = memref.load %sh[%src] : memref<256xf32, 3>")
        lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_quantile_axis1_sort_v1 is not None:
        kernel_kind = "row_quantile_axis1_sort_v1"
        m_dim = int(row_quantile_axis1_sort_v1["M"])
        n_dim = int(row_quantile_axis1_sort_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim}")
        sort_len = 1 << (int(n_dim - 1).bit_length())
        if sort_len > 256:
            raise RuntimeError(f"{kernel_kind} currently requires N<=256, got N={n_dim}")

        in_name = str(row_quantile_axis1_sort_v1["inp"])
        q_name = str(row_quantile_axis1_sort_v1["q"])
        out_name2 = str(row_quantile_axis1_sort_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        q_memref = str(arg_specs[q_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        block_threads = int(sort_len)
        launch_override = {"block": [int(block_threads), 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(sort_len)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %in_bounds = arith.cmpi ult, %tid, %cN : index")
        lines.append("        %pos_inf = arith.constant 0x7F800000 : f32")
        lines.append("        %val = scf.if %in_bounds -> (f32) {")
        lines.append("          %idx = arith.addi %base, %tid : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("          scf.yield %x : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %pos_inf : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %val, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")

        lines.append("        %tid_i32 = arith.index_cast %tid : index to i32")
        lines.append("        %c0_i32 = arith.constant 0 : i32")
        for k in [2**x for x in range(1, int(sort_len).bit_length() + 1) if 2**x <= int(sort_len)]:
            lines.append(f"        %cK_{k}_i32 = arith.constant {int(k)} : i32")
            lines.append(f"        %and_k_{k} = arith.andi %tid_i32, %cK_{k}_i32 : i32")
            lines.append(f"        %dir_k_{k} = arith.cmpi eq, %and_k_{k}, %c0_i32 : i32")
            j = k // 2
            while j >= 1:
                lines.append(f"        %cJ_{k}_{j}_i32 = arith.constant {int(j)} : i32")
                lines.append(f"        %ixj_{k}_{j}_i32 = arith.xori %tid_i32, %cJ_{k}_{j}_i32 : i32")
                lines.append(f"        %do_{k}_{j} = arith.cmpi ugt, %ixj_{k}_{j}_i32, %tid_i32 : i32")
                lines.append(f"        scf.if %do_{k}_{j} {{")
                lines.append(f"          %ixj_{k}_{j} = arith.index_cast %ixj_{k}_{j}_i32 : i32 to index")
                lines.append(f"          %a_val_{k}_{j} = memref.load %sh[%tid] : {shared_global_memref_ty}")
                lines.append(f"          %b_val_{k}_{j} = memref.load %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}")
                lines.append(f"          %val_gt_{k}_{j} = arith.cmpf ogt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                lines.append(f"          %val_lt_{k}_{j} = arith.cmpf olt, %a_val_{k}_{j}, %b_val_{k}_{j} : f32")
                lines.append(f"          %swap_{k}_{j} = arith.select %dir_k_{k}, %val_gt_{k}_{j}, %val_lt_{k}_{j} : i1")
                lines.append(
                    f"          %new_a_val_{k}_{j} = arith.select %swap_{k}_{j}, %b_val_{k}_{j}, %a_val_{k}_{j} : f32"
                )
                lines.append(
                    f"          %new_b_val_{k}_{j} = arith.select %swap_{k}_{j}, %a_val_{k}_{j}, %b_val_{k}_{j} : f32"
                )
                lines.append(f"          memref.store %new_a_val_{k}_{j}, %sh[%tid] : {shared_global_memref_ty}")
                lines.append(
                    f"          memref.store %new_b_val_{k}_{j}, %sh[%ixj_{k}_{j}] : {shared_global_memref_ty}"
                )
                lines.append("        }")
                lines.append("        gpu.barrier")
                j //= 2

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %c0f = arith.constant 0.0 : f32")
        lines.append("          %c1f = arith.constant 1.0 : f32")
        lines.append(f"          %q_raw = memref.load {arg_ssa[q_name]}[%c0] : {q_memref}")
        lines.append(f"          %q0 = arith.maximumf %q_raw, %c0f{fm} : f32")
        lines.append(f"          %q_clamped = arith.minimumf %q0, %c1f{fm} : f32")
        lines.append(f"          %n1_f = arith.constant {_as_f32_const(int(n_dim - 1))} : f32")
        lines.append(f"          %pos = arith.mulf %q_clamped, %n1_f{fm} : f32")
        lines.append("          %lower_i32 = arith.fptosi %pos : f32 to i32")
        lines.append("          %lower_f = arith.sitofp %lower_i32 : i32 to f32")
        lines.append(f"          %frac = arith.subf %pos, %lower_f{fm} : f32")
        lines.append("          %c1_i32 = arith.constant 1 : i32")
        lines.append(f"          %cN1_i32 = arith.constant {int(n_dim - 1)} : i32")
        lines.append("          %upper_raw = arith.addi %lower_i32, %c1_i32 : i32")
        lines.append("          %upper_gt = arith.cmpi sgt, %upper_raw, %cN1_i32 : i32")
        lines.append("          %upper_i32 = arith.select %upper_gt, %cN1_i32, %upper_raw : i32")
        lines.append("          %lower_idx = arith.index_cast %lower_i32 : i32 to index")
        lines.append("          %upper_idx = arith.index_cast %upper_i32 : i32 to index")
        lines.append(f"          %v0 = memref.load %sh[%lower_idx] : {shared_global_memref_ty}")
        lines.append(f"          %v1 = memref.load %sh[%upper_idx] : {shared_global_memref_ty}")
        lines.append(f"          %diff = arith.subf %v1, %v0{fm} : f32")
        lines.append(f"          %q_diff = arith.mulf %frac, %diff{fm} : f32")
        lines.append(f"          %outv = arith.addf %v0, %q_diff{fm} : f32")
        lines.append(f"          memref.store %outv, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif index_add2d_axis0_v1 is not None:
        kernel_kind = "index_add2d_axis0_v1"
        m_dim = int(index_add2d_axis0_v1["M"])
        n_dim = int(index_add2d_axis0_v1["N"])
        l_dim = int(index_add2d_axis0_v1["L"])
        alpha_f = float(index_add2d_axis0_v1.get("alpha", 1.0))
        if m_dim <= 0 or n_dim <= 0 or l_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} L={l_dim}")

        base_name = str(index_add2d_axis0_v1["base"])
        index_name = str(index_add2d_axis0_v1["index"])
        src_name = str(index_add2d_axis0_v1["src"])
        out_name2 = str(index_add2d_axis0_v1["out"])
        base_memref = str(arg_specs[base_name]["memref"])
        index_memref = str(arg_specs[index_name]["memref"])
        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(l_dim)} : index")
        lines.append(f"      %alpha = arith.constant {_as_f32_const(alpha_f)} : f32")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %row_off = arith.muli %bid, %cN : index")
        lines.append("        %bid_i32 = arith.index_cast %bid : index to i32")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %base_v = memref.load {arg_ssa[base_name]}[%idx] : {base_memref}")
        lines.append("          %acc = scf.for %ii = %c0 to %cL step %c1 iter_args(%a = %base_v) -> (f32) {")
        lines.append(f"            %row_i32 = memref.load {arg_ssa[index_name]}[%ii] : {index_memref}")
        lines.append("            %is_row = arith.cmpi eq, %row_i32, %bid_i32 : i32")
        lines.append("            %a_next = scf.if %is_row -> (f32) {")
        lines.append("              %src_row_off = arith.muli %ii, %cN : index")
        lines.append("              %src_idx = arith.addi %src_row_off, %jj : index")
        lines.append(f"              %sv = memref.load {arg_ssa[src_name]}[%src_idx] : {src_memref}")
        lines.append(f"              %scaled = arith.mulf %sv, %alpha{fm} : f32")
        lines.append(f"              %sum = arith.addf %a, %scaled{fm} : f32")
        lines.append("              scf.yield %sum : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %a : f32")
        lines.append("            }")
        lines.append("            scf.yield %a_next : f32")
        lines.append("          }")
        lines.append(f"          memref.store %acc, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif index_put2d_v1 is not None:
        kernel_kind = "index_put2d_v1"
        m_dim = int(index_put2d_v1["M"])
        n_dim = int(index_put2d_v1["N"])
        l_dim = int(index_put2d_v1["L"])
        if m_dim <= 0 or n_dim <= 0 or l_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} L={l_dim}")

        base_name = str(index_put2d_v1["base"])
        row_name = str(index_put2d_v1["row_idx"])
        col_name = str(index_put2d_v1["col_idx"])
        val_name = str(index_put2d_v1["values"])
        out_name2 = str(index_put2d_v1["out"])
        base_memref = str(arg_specs[base_name]["memref"])
        row_memref = str(arg_specs[row_name]["memref"])
        col_memref = str(arg_specs[col_name]["memref"])
        val_memref = str(arg_specs[val_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(l_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %row_off = arith.muli %bid, %cN : index")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %v = memref.load {arg_ssa[base_name]}[%idx] : {base_memref}")
        lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %bid_i32 = arith.index_cast %bid : index to i32")
        lines.append(f"          %c0_i32 = arith.constant 0 : i32")
        lines.append(f"          %cN_i32 = arith.constant {int(n_dim)} : i32")
        lines.append("          scf.for %ii = %c0 to %cL step %c1 {")
        lines.append(f"            %r_i32 = memref.load {arg_ssa[row_name]}[%ii] : {row_memref}")
        lines.append("            %match_row = arith.cmpi eq, %r_i32, %bid_i32 : i32")
        lines.append("            scf.if %match_row {")
        lines.append(f"              %c_i32 = memref.load {arg_ssa[col_name]}[%ii] : {col_memref}")
        lines.append("              %ge0 = arith.cmpi sge, %c_i32, %c0_i32 : i32")
        lines.append("              %ltN = arith.cmpi slt, %c_i32, %cN_i32 : i32")
        lines.append("              %inb = arith.andi %ge0, %ltN : i1")
        lines.append("              scf.if %inb {")
        lines.append("                %cc = arith.index_cast %c_i32 : i32 to index")
        lines.append("                %dst = arith.addi %row_off, %cc : index")
        lines.append(f"                %vv = memref.load {arg_ssa[val_name]}[%ii] : {val_memref}")
        lines.append(f"                memref.store %vv, {arg_ssa[out_name2]}[%dst] : {out_memref}")
        lines.append("              }")
        lines.append("            }")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif scatter2d_dim1_v1 is not None:
        kernel_kind = "scatter2d_dim1_v1"
        m_dim = int(scatter2d_dim1_v1["M"])
        n_dim = int(scatter2d_dim1_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim}")

        inp_name = str(scatter2d_dim1_v1["inp"])
        idx_name = str(scatter2d_dim1_v1["index"])
        src_name = str(scatter2d_dim1_v1["src"])
        out_name2 = str(scatter2d_dim1_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        idx_memref = str(arg_specs[idx_name]["memref"])
        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %row_off = arith.muli %bid, %cN : index")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %v = memref.load {arg_ssa[inp_name]}[%idx] : {inp_memref}")
        lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %c0_i32 = arith.constant 0 : i32")
        lines.append(f"        %cN_i32 = arith.constant {int(n_dim)} : i32")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %src_idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %dst_i32 = memref.load {arg_ssa[idx_name]}[%src_idx] : {idx_memref}")
        lines.append("          %ge0 = arith.cmpi sge, %dst_i32, %c0_i32 : i32")
        lines.append("          %ltN = arith.cmpi slt, %dst_i32, %cN_i32 : i32")
        lines.append("          %inb = arith.andi %ge0, %ltN : i1")
        lines.append("          scf.if %inb {")
        lines.append("            %dst = arith.index_cast %dst_i32 : i32 to index")
        lines.append("            %out_idx = arith.addi %row_off, %dst : index")
        lines.append(f"            %vv = memref.load {arg_ssa[src_name]}[%src_idx] : {src_memref}")
        lines.append(f"            memref.store %vv, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif select_scatter2d_dim1_v1 is not None:
        kernel_kind = "select_scatter2d_dim1_v1"
        m_dim = int(select_scatter2d_dim1_v1["M"])
        n_dim = int(select_scatter2d_dim1_v1["N"])
        col_i = int(select_scatter2d_dim1_v1["index"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim}")
        if col_i < 0 or col_i >= n_dim:
            raise RuntimeError(f"{kernel_kind} requires 0<=index<N, got index={col_i} N={n_dim}")

        inp_name = str(select_scatter2d_dim1_v1["inp"])
        src_name = str(select_scatter2d_dim1_v1["src"])
        out_name2 = str(select_scatter2d_dim1_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cCol = arith.constant {int(col_i)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %row_off = arith.muli %bid, %cN : index")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %v = memref.load {arg_ssa[inp_name]}[%idx] : {inp_memref}")
        lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %dst = arith.addi %row_off, %cCol : index")
        lines.append(f"          %sv = memref.load {arg_ssa[src_name]}[%bid] : {src_memref}")
        lines.append(f"          memref.store %sv, {arg_ssa[out_name2]}[%dst] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif slice_scatter2d_dim1_v1 is not None:
        kernel_kind = "slice_scatter2d_dim1_v1"
        m_dim = int(slice_scatter2d_dim1_v1["M"])
        n_dim = int(slice_scatter2d_dim1_v1["N"])
        start_i = int(slice_scatter2d_dim1_v1["start"])
        end_i = int(slice_scatter2d_dim1_v1["end"])
        step_i = int(slice_scatter2d_dim1_v1["step"])
        l_dim = int(slice_scatter2d_dim1_v1["L"])
        if m_dim <= 0 or n_dim <= 0 or l_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} L={l_dim}")
        if step_i <= 0:
            raise RuntimeError(f"{kernel_kind} requires step>0, got step={step_i}")
        if not (0 <= start_i <= end_i <= n_dim):
            raise RuntimeError(f"{kernel_kind} requires 0<=start<=end<=N, got start={start_i} end={end_i} N={n_dim}")

        inp_name = str(slice_scatter2d_dim1_v1["inp"])
        src_name = str(slice_scatter2d_dim1_v1["src"])
        out_name2 = str(slice_scatter2d_dim1_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(l_dim)} : index")
        lines.append(f"      %cStart = arith.constant {int(start_i)} : index")
        lines.append(f"      %cStep = arith.constant {int(step_i)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %row_off = arith.muli %bid, %cN : index")
        lines.append("        scf.for %jj = %tid to %cN step %bdim {")
        lines.append("          %idx = arith.addi %row_off, %jj : index")
        lines.append(f"          %v = memref.load {arg_ssa[inp_name]}[%idx] : {inp_memref}")
        lines.append(f"          memref.store %v, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %src_row_off = arith.muli %bid, %cL : index")
        lines.append("        scf.for %jj = %tid to %cL step %bdim {")
        lines.append("          %dst_mul = arith.muli %jj, %cStep : index")
        lines.append("          %dst_col = arith.addi %cStart, %dst_mul : index")
        lines.append("          %out_idx = arith.addi %row_off, %dst_col : index")
        lines.append("          %src_idx = arith.addi %src_row_off, %jj : index")
        lines.append(f"          %vv = memref.load {arg_ssa[src_name]}[%src_idx] : {src_memref}")
        lines.append(f"          memref.store %vv, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif masked_select2d_v1 is not None:
        kernel_kind = "masked_select2d_prefixsum_v1"
        m_dim = int(masked_select2d_v1["M"])
        n_dim = int(masked_select2d_v1["N"])
        l_dim = int(masked_select2d_v1["L"])
        t_dim = int(masked_select2d_v1["T"])
        block_threads = int(masked_select2d_v1["block_threads"])
        if m_dim <= 0 or n_dim <= 0 or l_dim <= 0 or t_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} L={l_dim} T={t_dim}")
        if l_dim > t_dim:
            raise RuntimeError(f"{kernel_kind} requires L<=T, got L={l_dim} T={t_dim}")
        if t_dim > 1024 or block_threads > 1024:
            raise RuntimeError(f"{kernel_kind} currently requires T<=1024, got T={t_dim} block={block_threads}")

        inp_name = str(masked_select2d_v1["inp"])
        mask_name = str(masked_select2d_v1["mask"])
        out_name2 = str(masked_select2d_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        mask_memref = str(arg_specs[mask_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        mask_elem_ty = str(arg_specs[mask_name].get("memref_elem_ty") or "")

        launch_override = {"block": [int(block_threads), 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_scan_i32"
        shared_global_memref_ty = f"memref<{int(block_threads)}xi32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cT = arith.constant {int(t_dim)} : index")
        lines.append(f"      %cL = arith.constant {int(l_dim)} : index")
        lines.append("      %pred = arith.cmpi ult, %tid, %cT : index")
        lines.append("      %m = scf.if %pred -> (i1) {")
        if mask_elem_ty == "i1":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        scf.yield %raw : i1")
        elif mask_elem_ty == "i8":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        %c0_i8 = arith.constant 0 : i8")
            lines.append("        %p = arith.cmpi ne, %raw, %c0_i8 : i8")
            lines.append("        scf.yield %p : i1")
        elif mask_elem_ty == "i32":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        %c0_i32 = arith.constant 0 : i32")
            lines.append("        %p = arith.cmpi ne, %raw, %c0_i32 : i32")
            lines.append("        scf.yield %p : i1")
        else:
            raise RuntimeError(f"{kernel_kind} unsupported mask memref_elem_ty: {mask_elem_ty}")
        lines.append("      } else {")
        lines.append("        %f = arith.constant 0 : i1")
        lines.append("        scf.yield %f : i1")
        lines.append("      }")
        lines.append("      %m_i32 = arith.extui %m : i1 to i32")
        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"      memref.store %m_i32, %sh[%tid] : {shared_global_memref_ty}")
        lines.append("      gpu.barrier")

        # Inclusive scan over i32 flags in shared memory.
        for off in [2**x for x in range(0, int(block_threads).bit_length()) if 2**x < int(block_threads)]:
            if off <= 0:
                continue
            lines.append(f"      %cOff_{off} = arith.constant {int(off)} : index")
            lines.append(f"      %ge_{off} = arith.cmpi uge, %tid, %cOff_{off} : index")
            lines.append(f"      %val_{off} = scf.if %ge_{off} -> (i32) {{")
            lines.append(f"        %a_{off} = memref.load %sh[%tid] : {shared_global_memref_ty}")
            lines.append(f"        %tid_off_{off} = arith.subi %tid, %cOff_{off} : index")
            lines.append(f"        %b_{off} = memref.load %sh[%tid_off_{off}] : {shared_global_memref_ty}")
            lines.append(f"        %sum_{off} = arith.addi %a_{off}, %b_{off} : i32")
            lines.append(f"        scf.yield %sum_{off} : i32")
            lines.append("      } else {")
            lines.append(f"        %a0_{off} = memref.load %sh[%tid] : {shared_global_memref_ty}")
            lines.append(f"        scf.yield %a0_{off} : i32")
            lines.append("      }")
            lines.append("      gpu.barrier")
            lines.append(f"      memref.store %val_{off}, %sh[%tid] : {shared_global_memref_ty}")
            lines.append("      gpu.barrier")

        lines.append("      scf.if %pred {")
        lines.append("        scf.if %m {")
        lines.append(f"          %prefix = memref.load %sh[%tid] : {shared_global_memref_ty}")
        lines.append("          %c1_i32 = arith.constant 1 : i32")
        lines.append("          %pos_i32 = arith.subi %prefix, %c1_i32 : i32")
        lines.append(f"          %cL_i32 = arith.constant {int(l_dim)} : i32")
        lines.append("          %lt = arith.cmpi slt, %pos_i32, %cL_i32 : i32")
        lines.append("          scf.if %lt {")
        lines.append("            %pos = arith.index_cast %pos_i32 : i32 to index")
        lines.append(f"            %x = memref.load {arg_ssa[inp_name]}[%tid] : {inp_memref}")
        lines.append(f"            memref.store %x, {arg_ssa[out_name2]}[%pos] : {out_memref}")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif masked_scatter2d_v1 is not None:
        kernel_kind = "masked_scatter2d_prefixsum_v1"
        m_dim = int(masked_scatter2d_v1["M"])
        n_dim = int(masked_scatter2d_v1["N"])
        l_dim = int(masked_scatter2d_v1["L"])
        t_dim = int(masked_scatter2d_v1["T"])
        block_threads = int(masked_scatter2d_v1["block_threads"])
        if m_dim <= 0 or n_dim <= 0 or l_dim <= 0 or t_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} L={l_dim} T={t_dim}")
        if l_dim > t_dim:
            raise RuntimeError(f"{kernel_kind} requires L<=T, got L={l_dim} T={t_dim}")
        if t_dim > 1024 or block_threads > 1024:
            raise RuntimeError(f"{kernel_kind} currently requires T<=1024, got T={t_dim} block={block_threads}")

        inp_name = str(masked_scatter2d_v1["inp"])
        mask_name = str(masked_scatter2d_v1["mask"])
        src_name = str(masked_scatter2d_v1["src"])
        out_name2 = str(masked_scatter2d_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        mask_memref = str(arg_specs[mask_name]["memref"])
        src_memref = str(arg_specs[src_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        mask_elem_ty = str(arg_specs[mask_name].get("memref_elem_ty") or "")

        launch_override = {"block": [int(block_threads), 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_scan_i32"
        shared_global_memref_ty = f"memref<{int(block_threads)}xi32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cT = arith.constant {int(t_dim)} : index")
        lines.append("      %pred = arith.cmpi ult, %tid, %cT : index")
        lines.append("      %m = scf.if %pred -> (i1) {")
        if mask_elem_ty == "i1":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        scf.yield %raw : i1")
        elif mask_elem_ty == "i8":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        %c0_i8 = arith.constant 0 : i8")
            lines.append("        %p = arith.cmpi ne, %raw, %c0_i8 : i8")
            lines.append("        scf.yield %p : i1")
        elif mask_elem_ty == "i32":
            lines.append(f"        %raw = memref.load {arg_ssa[mask_name]}[%tid] : {mask_memref}")
            lines.append("        %c0_i32 = arith.constant 0 : i32")
            lines.append("        %p = arith.cmpi ne, %raw, %c0_i32 : i32")
            lines.append("        scf.yield %p : i1")
        else:
            raise RuntimeError(f"{kernel_kind} unsupported mask memref_elem_ty: {mask_elem_ty}")
        lines.append("      } else {")
        lines.append("        %f = arith.constant 0 : i1")
        lines.append("        scf.yield %f : i1")
        lines.append("      }")
        lines.append("      %m_i32 = arith.extui %m : i1 to i32")
        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"      memref.store %m_i32, %sh[%tid] : {shared_global_memref_ty}")
        lines.append("      gpu.barrier")

        for off in [2**x for x in range(0, int(block_threads).bit_length()) if 2**x < int(block_threads)]:
            if off <= 0:
                continue
            lines.append(f"      %cOff_{off} = arith.constant {int(off)} : index")
            lines.append(f"      %ge_{off} = arith.cmpi uge, %tid, %cOff_{off} : index")
            lines.append(f"      %val_{off} = scf.if %ge_{off} -> (i32) {{")
            lines.append(f"        %a_{off} = memref.load %sh[%tid] : {shared_global_memref_ty}")
            lines.append(f"        %tid_off_{off} = arith.subi %tid, %cOff_{off} : index")
            lines.append(f"        %b_{off} = memref.load %sh[%tid_off_{off}] : {shared_global_memref_ty}")
            lines.append(f"        %sum_{off} = arith.addi %a_{off}, %b_{off} : i32")
            lines.append(f"        scf.yield %sum_{off} : i32")
            lines.append("      } else {")
            lines.append(f"        %a0_{off} = memref.load %sh[%tid] : {shared_global_memref_ty}")
            lines.append(f"        scf.yield %a0_{off} : i32")
            lines.append("      }")
            lines.append("      gpu.barrier")
            lines.append(f"      memref.store %val_{off}, %sh[%tid] : {shared_global_memref_ty}")
            lines.append("      gpu.barrier")

        lines.append("      scf.if %pred {")
        lines.append(f"        %x = memref.load {arg_ssa[inp_name]}[%tid] : {inp_memref}")
        lines.append("        %outv = scf.if %m -> (f32) {")
        lines.append(f"          %prefix = memref.load %sh[%tid] : {shared_global_memref_ty}")
        lines.append("          %c1_i32 = arith.constant 1 : i32")
        lines.append("          %pos_i32 = arith.subi %prefix, %c1_i32 : i32")
        lines.append(f"          %cL_i32 = arith.constant {int(l_dim)} : i32")
        lines.append("          %lt = arith.cmpi slt, %pos_i32, %cL_i32 : i32")
        lines.append("          %v = scf.if %lt -> (f32) {")
        lines.append("            %pos = arith.index_cast %pos_i32 : i32 to index")
        lines.append(f"            %sv = memref.load {arg_ssa[src_name]}[%pos] : {src_memref}")
        lines.append("            scf.yield %sv : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %x : f32")
        lines.append("          }")
        lines.append("          scf.yield %v : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %x : f32")
        lines.append("        }")
        lines.append(f"        memref.store %outv, {arg_ssa[out_name2]}[%tid] : {out_memref}")
        lines.append("      }")
    elif avg_pool2d_nchw_v1 is not None:
        kernel_kind = "avg_pool2d_nchw_v1"
        n_dim = int(avg_pool2d_nchw_v1["N"])
        c_dim = int(avg_pool2d_nchw_v1["C"])
        h_dim = int(avg_pool2d_nchw_v1["H"])
        w_dim = int(avg_pool2d_nchw_v1["W"])
        oh_dim = int(avg_pool2d_nchw_v1["OH"])
        ow_dim = int(avg_pool2d_nchw_v1["OW"])
        if n_dim <= 0 or c_dim <= 0 or h_dim <= 0 or w_dim <= 0 or oh_dim <= 0 or ow_dim <= 0:
            raise RuntimeError(
                "avg_pool2d_nchw expects positive dims, got "
                f"N={n_dim} C={c_dim} H={h_dim} W={w_dim} OH={oh_dim} OW={ow_dim}"
            )

        in_name = str(avg_pool2d_nchw_v1["inp"])
        out_name2 = str(avg_pool2d_nchw_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"        %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cHW = arith.constant {int(h_dim * w_dim)} : index")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %cc = arith.remui %t2, %cC : index")
        lines.append("        %nn = arith.divui %t2, %cC : index")
        lines.append("        %c2 = arith.constant 2 : index")
        lines.append("        %ih0 = arith.muli %oh, %c2 : index")
        lines.append("        %iw0 = arith.muli %ow, %c2 : index")
        lines.append("        %nc = arith.muli %nn, %cC : index")
        lines.append("        %ncc = arith.addi %nc, %cc : index")
        lines.append("        %base_nc = arith.muli %ncc, %cHW : index")
        lines.append("        %ih_mul = arith.muli %ih0, %cW : index")
        lines.append("        %base_h = arith.addi %base_nc, %ih_mul : index")
        lines.append("        %idx0 = arith.addi %base_h, %iw0 : index")
        lines.append(f"        %x0 = memref.load {arg_ssa[in_name]}[%idx0] : {in_memref}")
        lines.append("        %idx1 = arith.addi %idx0, %c1 : index")
        lines.append(f"        %x1 = memref.load {arg_ssa[in_name]}[%idx1] : {in_memref}")
        lines.append("        %idx2 = arith.addi %idx0, %cW : index")
        lines.append(f"        %x2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
        lines.append("        %idx3 = arith.addi %idx2, %c1 : index")
        lines.append(f"        %x3 = memref.load {arg_ssa[in_name]}[%idx3] : {in_memref}")
        lines.append(f"        %s01 = arith.addf %x0, %x1{fm} : f32")
        lines.append(f"        %s23 = arith.addf %x2, %x3{fm} : f32")
        lines.append(f"        %sum = arith.addf %s01, %s23{fm} : f32")
        lines.append("        %c025 = arith.constant 0.25 : f32")
        lines.append(f"        %avg = arith.mulf %sum, %c025{fm} : f32")
        lines.append(f"        memref.store %avg, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif max_pool2d_with_indices_nchw_v1 is not None:
        kernel_kind = "max_pool2d_with_indices_nchw_v1"
        n_dim = int(max_pool2d_with_indices_nchw_v1["N"])
        c_dim = int(max_pool2d_with_indices_nchw_v1["C"])
        h_dim = int(max_pool2d_with_indices_nchw_v1["H"])
        w_dim = int(max_pool2d_with_indices_nchw_v1["W"])
        oh_dim = int(max_pool2d_with_indices_nchw_v1["OH"])
        ow_dim = int(max_pool2d_with_indices_nchw_v1["OW"])
        if n_dim <= 0 or c_dim <= 0 or h_dim <= 0 or w_dim <= 0 or oh_dim <= 0 or ow_dim <= 0:
            raise RuntimeError(
                "max_pool2d_with_indices_nchw expects positive dims, got "
                f"N={n_dim} C={c_dim} H={h_dim} W={w_dim} OH={oh_dim} OW={ow_dim}"
            )

        in_name = str(max_pool2d_with_indices_nchw_v1["inp"])
        out_name2 = str(max_pool2d_with_indices_nchw_v1["out"])
        indices_name = str(max_pool2d_with_indices_nchw_v1["indices"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        indices_memref = str(arg_specs[indices_name]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"        %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cHW = arith.constant {int(h_dim * w_dim)} : index")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %cc = arith.remui %t2, %cC : index")
        lines.append("        %nn = arith.divui %t2, %cC : index")
        lines.append("        %c2 = arith.constant 2 : index")
        lines.append("        %ih0 = arith.muli %oh, %c2 : index")
        lines.append("        %iw0 = arith.muli %ow, %c2 : index")
        lines.append("        %nc = arith.muli %nn, %cC : index")
        lines.append("        %ncc = arith.addi %nc, %cc : index")
        lines.append("        %base_nc = arith.muli %ncc, %cHW : index")
        lines.append("        %ih_mul = arith.muli %ih0, %cW : index")
        lines.append("        %base_h = arith.addi %base_nc, %ih_mul : index")
        lines.append("        %idx0 = arith.addi %base_h, %iw0 : index")
        lines.append("        %p0 = arith.addi %ih_mul, %iw0 : index")
        lines.append(f"        %v0 = memref.load {arg_ssa[in_name]}[%idx0] : {in_memref}")
        lines.append("        %idx1 = arith.addi %idx0, %c1 : index")
        lines.append("        %p1 = arith.addi %p0, %c1 : index")
        lines.append(f"        %v1 = memref.load {arg_ssa[in_name]}[%idx1] : {in_memref}")
        lines.append("        %idx2 = arith.addi %idx0, %cW : index")
        lines.append("        %p2 = arith.addi %p0, %cW : index")
        lines.append(f"        %v2 = memref.load {arg_ssa[in_name]}[%idx2] : {in_memref}")
        lines.append("        %idx3 = arith.addi %idx2, %c1 : index")
        lines.append("        %p3 = arith.addi %p2, %c1 : index")
        lines.append(f"        %v3 = memref.load {arg_ssa[in_name]}[%idx3] : {in_memref}")

        lines.append("        %gt1 = arith.cmpf ogt, %v1, %v0 : f32")
        lines.append("        %eq1 = arith.cmpf oeq, %v1, %v0 : f32")
        lines.append("        %idx_lt1 = arith.cmpi ult, %p1, %p0 : index")
        lines.append("        %eq_lt1 = arith.andi %eq1, %idx_lt1 : i1")
        lines.append("        %better1 = arith.ori %gt1, %eq_lt1 : i1")
        lines.append("        %best_v1 = arith.select %better1, %v1, %v0 : f32")
        lines.append("        %best_i1 = arith.select %better1, %p1, %p0 : index")

        lines.append("        %gt2 = arith.cmpf ogt, %v2, %best_v1 : f32")
        lines.append("        %eq2 = arith.cmpf oeq, %v2, %best_v1 : f32")
        lines.append("        %idx_lt2 = arith.cmpi ult, %p2, %best_i1 : index")
        lines.append("        %eq_lt2 = arith.andi %eq2, %idx_lt2 : i1")
        lines.append("        %better2 = arith.ori %gt2, %eq_lt2 : i1")
        lines.append("        %best_v2 = arith.select %better2, %v2, %best_v1 : f32")
        lines.append("        %best_i2 = arith.select %better2, %p2, %best_i1 : index")

        lines.append("        %gt3 = arith.cmpf ogt, %v3, %best_v2 : f32")
        lines.append("        %eq3 = arith.cmpf oeq, %v3, %best_v2 : f32")
        lines.append("        %idx_lt3 = arith.cmpi ult, %p3, %best_i2 : index")
        lines.append("        %eq_lt3 = arith.andi %eq3, %idx_lt3 : i1")
        lines.append("        %better3 = arith.ori %gt3, %eq_lt3 : i1")
        lines.append("        %best_v3 = arith.select %better3, %v3, %best_v2 : f32")
        lines.append("        %best_i3 = arith.select %better3, %p3, %best_i2 : index")

        lines.append(f"        memref.store %best_v3, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("        %best_i3_i64 = arith.index_cast %best_i3 : index to i64")
        lines.append(f"        memref.store %best_i3_i64, {arg_ssa[indices_name]}[%lin] : {indices_memref}")
        lines.append("      }")
    elif stack2d_v1 is not None:
        kernel_kind = "stack2d_v1"
        m_dim = int(stack2d_v1["M"])
        n_dim = int(stack2d_v1["N"])
        plane = int(m_dim * n_dim)
        if plane <= 0:
            raise RuntimeError(f"stack2d expects positive plane, got M={m_dim} N={n_dim}")

        in0_name = str(stack2d_v1["inp0"])
        in1_name = str(stack2d_v1["inp1"])
        out_name2 = str(stack2d_v1["out"])
        in0_memref = str(arg_specs[in0_name]["memref"])
        in1_memref = str(arg_specs[in1_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cPlane = arith.constant {int(plane)} : index")
        lines.append("        %which = arith.divui %lin, %cPlane : index")
        lines.append("        %idx_in = arith.remui %lin, %cPlane : index")
        lines.append("        %is0 = arith.cmpi eq, %which, %c0 : index")
        lines.append("        %v = scf.if %is0 -> (f32) {")
        lines.append(f"          %a = memref.load {arg_ssa[in0_name]}[%idx_in] : {in0_memref}")
        lines.append("          scf.yield %a : f32")
        lines.append("        } else {")
        lines.append(f"          %b = memref.load {arg_ssa[in1_name]}[%idx_in] : {in1_memref}")
        lines.append("          scf.yield %b : f32")
        lines.append("        }")
        lines.append(f"        memref.store %v, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif polar2d_v1 is not None:
        kernel_kind = "polar2d_v1"
        abs_name = str(polar2d_v1["abs"])
        ang_name = str(polar2d_v1["angle"])
        out_name2 = str(polar2d_v1["out"])
        abs_memref = str(arg_specs[abs_name]["memref"])
        ang_memref = str(arg_specs[ang_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c2 = arith.constant 2 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append("        %comp = arith.remui %lin, %c2 : index")
        lines.append("        %base = arith.divui %lin, %c2 : index")
        lines.append(f"        %abs_v = memref.load {arg_ssa[abs_name]}[%base] : {abs_memref}")
        lines.append(f"        %ang_v = memref.load {arg_ssa[ang_name]}[%base] : {ang_memref}")
        lines.append("        %is0 = arith.cmpi eq, %comp, %c0 : index")
        lines.append("        %v = scf.if %is0 -> (f32) {")
        lines.append(f"          %c = math.cos %ang_v{fm} : f32")
        lines.append(f"          %r = arith.mulf %abs_v, %c{fm} : f32")
        lines.append("          scf.yield %r : f32")
        lines.append("        } else {")
        lines.append(f"          %s = math.sin %ang_v{fm} : f32")
        lines.append(f"          %i = arith.mulf %abs_v, %s{fm} : f32")
        lines.append("          scf.yield %i : f32")
        lines.append("        }")
        lines.append(f"        memref.store %v, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif diag_embed2d_v1 is not None:
        kernel_kind = "diag_embed2d_v1"
        b_dim = int(diag_embed2d_v1["B"])
        n_dim = int(diag_embed2d_v1["N"])
        nn = int(n_dim * n_dim)
        if b_dim <= 0 or n_dim <= 0 or nn <= 0:
            raise RuntimeError(f"diag_embed2d expects positive dims, got B={b_dim} N={n_dim}")

        x_name = str(diag_embed2d_v1["x"])
        out_name2 = str(diag_embed2d_v1["out"])
        x_memref = str(arg_specs[x_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cNN = arith.constant {int(nn)} : index")
        lines.append("        %rem = arith.remui %lin, %cNN : index")
        lines.append("        %bb = arith.divui %lin, %cNN : index")
        lines.append("        %jj = arith.remui %rem, %cN : index")
        lines.append("        %ii = arith.divui %rem, %cN : index")
        lines.append("        %is_diag = arith.cmpi eq, %ii, %jj : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %v = scf.if %is_diag -> (f32) {")
        lines.append("          %row_off = arith.muli %bb, %cN : index")
        lines.append("          %idx_x = arith.addi %row_off, %ii : index")
        lines.append(f"          %x_v = memref.load {arg_ssa[x_name]}[%idx_x] : {x_memref}")
        lines.append("          scf.yield %x_v : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")
        lines.append(f"        memref.store %v, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif kron2d_v1 is not None:
        kernel_kind = "kron2d_v1"
        m_dim = int(kron2d_v1["M"])
        n_dim = int(kron2d_v1["N"])
        p_dim = int(kron2d_v1["P"])
        q_dim = int(kron2d_v1["Q"])
        mp_dim = int(kron2d_v1["MP"])
        nq_dim = int(kron2d_v1["NQ"])
        if m_dim <= 0 or n_dim <= 0 or p_dim <= 0 or q_dim <= 0:
            raise RuntimeError(f"kron2d expects positive dims, got M={m_dim} N={n_dim} P={p_dim} Q={q_dim}")
        if mp_dim != m_dim * p_dim or nq_dim != n_dim * q_dim:
            raise RuntimeError(
                "kron2d output dims mismatch: "
                f"M={m_dim} N={n_dim} P={p_dim} Q={q_dim} MP={mp_dim} NQ={nq_dim}"
            )

        a_name = str(kron2d_v1["A"])
        b_name = str(kron2d_v1["B"])
        out_name2 = str(kron2d_v1["out"])
        a_memref = str(arg_specs[a_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cP = arith.constant {int(p_dim)} : index")
        lines.append(f"        %cQ = arith.constant {int(q_dim)} : index")
        lines.append(f"        %cNQ = arith.constant {int(nq_dim)} : index")
        lines.append("        %out_col = arith.remui %lin, %cNQ : index")
        lines.append("        %out_row = arith.divui %lin, %cNQ : index")
        lines.append("        %p = arith.remui %out_row, %cP : index")
        lines.append("        %i = arith.divui %out_row, %cP : index")
        lines.append("        %q = arith.remui %out_col, %cQ : index")
        lines.append("        %j = arith.divui %out_col, %cQ : index")
        lines.append("        %iN = arith.muli %i, %cN : index")
        lines.append("        %a_idx = arith.addi %iN, %j : index")
        lines.append("        %pQ = arith.muli %p, %cQ : index")
        lines.append("        %b_idx = arith.addi %pQ, %q : index")
        lines.append(f"        %a = memref.load {arg_ssa[a_name]}[%a_idx] : {a_memref}")
        lines.append(f"        %b = memref.load {arg_ssa[b_name]}[%b_idx] : {b_memref}")
        lines.append(f"        %o = arith.mulf %a, %b{fm} : f32")
        lines.append(f"        memref.store %o, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif cumsum2d_v1 is not None:
        kernel_kind = "cumsum2d_axis1_v1"
        m_dim = int(cumsum2d_v1["M"])
        n_dim = int(cumsum2d_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"cumsum2d expects positive dims, got M={m_dim} N={n_dim}")
        if int(m_dim * n_dim) != int(out_total):
            raise RuntimeError(f"cumsum2d expects out_total==M*N, got out_total={int(out_total)} M={m_dim} N={n_dim}")

        in_name = str(cumsum2d_v1["inp"])
        out_name2 = str(cumsum2d_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        # One row per block, warp-prefix scan per 256-column tile.
        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c31 = arith.constant 31 : index")
        lines.append("      %c255 = arith.constant 255 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %lane = arith.remui %tid, %c32_idx : index")
        lines.append("      %warp = arith.divui %tid, %c32_idx : index")
        lines.append("      %cWarps = arith.constant 8 : index")
        lines.append("      %cCarry = arith.constant 16 : index")
        lines.append("      %c256 = arith.constant 256 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %is_tid0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is_tid0 {")
        lines.append("          memref.store %c0f, %sh[%cCarry] : memref<256xf32, 3>")
        lines.append("        }")
        lines.append("        gpu.barrier")

        lines.append("        scf.for %t = %c0 to %cN step %c256 {")
        lines.append("          %col = arith.addi %t, %tid : index")
        lines.append("          %in_range = arith.cmpi ult, %col, %cN : index")
        lines.append("          %idx = arith.addi %base, %col : index")
        lines.append("          %x = scf.if %in_range -> (f32) {")
        lines.append(f"            %xv = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("            scf.yield %xv : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        lines.append("          %carry0 = memref.load %sh[%cCarry] : memref<256xf32, 3>")

        # Warp inclusive scan (shuffle up).
        lines.append("          %s1, %ok1 = gpu.shuffle up %x, %c1_i32, %c32_i32 : f32")
        lines.append(f"          %a1 = arith.addf %x, %s1{fm} : f32")
        lines.append("          %v1 = arith.select %ok1, %a1, %x : f32")
        lines.append("          %s2, %ok2 = gpu.shuffle up %v1, %c2_i32, %c32_i32 : f32")
        lines.append(f"          %a2 = arith.addf %v1, %s2{fm} : f32")
        lines.append("          %v2 = arith.select %ok2, %a2, %v1 : f32")
        lines.append("          %s4, %ok4 = gpu.shuffle up %v2, %c4_i32, %c32_i32 : f32")
        lines.append(f"          %a4 = arith.addf %v2, %s4{fm} : f32")
        lines.append("          %v4 = arith.select %ok4, %a4, %v2 : f32")
        lines.append("          %s8, %ok8 = gpu.shuffle up %v4, %c8_i32, %c32_i32 : f32")
        lines.append(f"          %a8 = arith.addf %v4, %s8{fm} : f32")
        lines.append("          %v8 = arith.select %ok8, %a8, %v4 : f32")
        lines.append("          %s16, %ok16 = gpu.shuffle up %v8, %c16_i32, %c32_i32 : f32")
        lines.append(f"          %a16 = arith.addf %v8, %s16{fm} : f32")
        lines.append("          %v16 = arith.select %ok16, %a16, %v8 : f32")

        # Store warp totals (lane 31) into shared[warp].
        lines.append("          %is_lane31 = arith.cmpi eq, %lane, %c31 : index")
        lines.append("          scf.if %is_lane31 {")
        lines.append("            memref.store %v16, %sh[%warp] : memref<256xf32, 3>")
        lines.append("          }")
        lines.append("          gpu.barrier")

        # Warp0: scan warp totals and store warp offsets into shared[cWarps + lane].
        lines.append("          %is_warp0 = arith.cmpi eq, %warp, %c0 : index")
        lines.append("          scf.if %is_warp0 {")
        lines.append("            %lane_in = arith.cmpi ult, %lane, %cWarps : index")
        lines.append("            %wv0 = scf.if %lane_in -> (f32) {")
        lines.append("              %wt = memref.load %sh[%lane] : memref<256xf32, 3>")
        lines.append("              scf.yield %wt : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        lines.append("            %ws1, %wok1 = gpu.shuffle up %wv0, %c1_i32, %c32_i32 : f32")
        lines.append(f"            %wa1 = arith.addf %wv0, %ws1{fm} : f32")
        lines.append("            %wv1 = arith.select %wok1, %wa1, %wv0 : f32")
        lines.append("            %ws2, %wok2 = gpu.shuffle up %wv1, %c2_i32, %c32_i32 : f32")
        lines.append(f"            %wa2 = arith.addf %wv1, %ws2{fm} : f32")
        lines.append("            %wv2 = arith.select %wok2, %wa2, %wv1 : f32")
        lines.append("            %ws4, %wok4 = gpu.shuffle up %wv2, %c4_i32, %c32_i32 : f32")
        lines.append(f"            %wa4 = arith.addf %wv2, %ws4{fm} : f32")
        lines.append("            %wv4 = arith.select %wok4, %wa4, %wv2 : f32")
        lines.append("            %ws8, %wok8 = gpu.shuffle up %wv4, %c8_i32, %c32_i32 : f32")
        lines.append(f"            %wa8 = arith.addf %wv4, %ws8{fm} : f32")
        lines.append("            %wv8 = arith.select %wok8, %wa8, %wv4 : f32")
        lines.append("            %ws16, %wok16 = gpu.shuffle up %wv8, %c16_i32, %c32_i32 : f32")
        lines.append(f"            %wa16 = arith.addf %wv8, %ws16{fm} : f32")
        lines.append("            %wv16 = arith.select %wok16, %wa16, %wv8 : f32")
        lines.append("            %wprev, %wok_prev = gpu.shuffle up %wv16, %c1_i32, %c32_i32 : f32")
        lines.append("            %woff = arith.select %wok_prev, %wprev, %c0f : f32")
        lines.append("            scf.if %lane_in {")
        lines.append("              %off_idx = arith.addi %cWarps, %lane : index")
        lines.append("              memref.store %woff, %sh[%off_idx] : memref<256xf32, 3>")
        lines.append("            }")
        lines.append("          }")
        lines.append("          gpu.barrier")

        lines.append("          %off_idx2 = arith.addi %cWarps, %warp : index")
        lines.append("          %woff2 = memref.load %sh[%off_idx2] : memref<256xf32, 3>")
        lines.append(f"          %block_scan = arith.addf %v16, %woff2{fm} : f32")
        lines.append(f"          %full_scan = arith.addf %block_scan, %carry0{fm} : f32")
        lines.append("          scf.if %in_range {")
        lines.append(f"            memref.store %full_scan, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("          }")
        lines.append("          %is_last = arith.cmpi eq, %tid, %c255 : index")
        lines.append("          scf.if %is_last {")
        lines.append("            memref.store %full_scan, %sh[%cCarry] : memref<256xf32, 3>")
        lines.append("          }")
        lines.append("          gpu.barrier")
        lines.append("        }")
        lines.append("      }")
    elif normed_cumsum2d_v1 is not None:
        kernel_kind = "normed_cumsum2d_axis1_v1"
        m_dim = int(normed_cumsum2d_v1["M"])
        n_dim = int(normed_cumsum2d_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"normed_cumsum2d expects positive dims, got M={m_dim} N={n_dim}")
        if int(m_dim * n_dim) != int(out_total):
            raise RuntimeError(
                f"normed_cumsum2d expects out_total==M*N, got out_total={int(out_total)} M={m_dim} N={n_dim}"
            )

        in_name = str(normed_cumsum2d_v1["inp"])
        eps_name = str(normed_cumsum2d_v1["eps"])
        out_name2 = str(normed_cumsum2d_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        eps_memref = str(arg_specs[eps_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        # One row per block, (prefix-sum)/(row-sum+eps).
        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c31 = arith.constant 31 : index")
        lines.append("      %c255 = arith.constant 255 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %lane = arith.remui %tid, %c32_idx : index")
        lines.append("      %warp = arith.divui %tid, %c32_idx : index")
        lines.append("      %cWarps = arith.constant 8 : index")
        lines.append("      %cCarry = arith.constant 16 : index")
        lines.append("      %cDenom = arith.constant 17 : index")
        lines.append("      %c256 = arith.constant 256 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %eps_v = memref.load {0}[%c0] : {1}".format(arg_ssa[eps_name], eps_memref))
        # row sum (denom) reduction
        lines.append("        %partial = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")
        lines.append("        memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_nc_{stride}"
            pS = f"%pS_nc_{stride}"
            tid2 = f"%tid_nc_{stride}"
            a = f"%a_nc_{stride}"
            b = f"%b_nc_{stride}"
            s = f"%s_nc_{stride}"
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
        lines.append("        %sum0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %denom0 = arith.addf %sum0, %eps_v{fm} : f32")
        lines.append("        %is_tid0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is_tid0 {")
        lines.append("          memref.store %denom0, %sh[%cDenom] : memref<256xf32, 3>")
        lines.append("          memref.store %c0f, %sh[%cCarry] : memref<256xf32, 3>")
        lines.append("        }")
        lines.append("        gpu.barrier")
        lines.append("        %denom = memref.load %sh[%cDenom] : memref<256xf32, 3>")

        # prefix scan and store normalized output
        lines.append("        scf.for %t = %c0 to %cN step %c256 {")
        lines.append("          %col = arith.addi %t, %tid : index")
        lines.append("          %in_range = arith.cmpi ult, %col, %cN : index")
        lines.append("          %idx = arith.addi %base, %col : index")
        lines.append("          %x = scf.if %in_range -> (f32) {")
        lines.append(f"            %xv = memref.load {arg_ssa[in_name]}[%idx] : {in_memref}")
        lines.append("            scf.yield %xv : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        lines.append("          %carry0 = memref.load %sh[%cCarry] : memref<256xf32, 3>")

        lines.append("          %s1, %ok1 = gpu.shuffle up %x, %c1_i32, %c32_i32 : f32")
        lines.append(f"          %a1 = arith.addf %x, %s1{fm} : f32")
        lines.append("          %v1 = arith.select %ok1, %a1, %x : f32")
        lines.append("          %s2, %ok2 = gpu.shuffle up %v1, %c2_i32, %c32_i32 : f32")
        lines.append(f"          %a2 = arith.addf %v1, %s2{fm} : f32")
        lines.append("          %v2 = arith.select %ok2, %a2, %v1 : f32")
        lines.append("          %s4, %ok4 = gpu.shuffle up %v2, %c4_i32, %c32_i32 : f32")
        lines.append(f"          %a4 = arith.addf %v2, %s4{fm} : f32")
        lines.append("          %v4 = arith.select %ok4, %a4, %v2 : f32")
        lines.append("          %s8, %ok8 = gpu.shuffle up %v4, %c8_i32, %c32_i32 : f32")
        lines.append(f"          %a8 = arith.addf %v4, %s8{fm} : f32")
        lines.append("          %v8 = arith.select %ok8, %a8, %v4 : f32")
        lines.append("          %s16, %ok16 = gpu.shuffle up %v8, %c16_i32, %c32_i32 : f32")
        lines.append(f"          %a16 = arith.addf %v8, %s16{fm} : f32")
        lines.append("          %v16 = arith.select %ok16, %a16, %v8 : f32")

        lines.append("          %is_lane31 = arith.cmpi eq, %lane, %c31 : index")
        lines.append("          scf.if %is_lane31 {")
        lines.append("            memref.store %v16, %sh[%warp] : memref<256xf32, 3>")
        lines.append("          }")
        lines.append("          gpu.barrier")
        lines.append("          %is_warp0 = arith.cmpi eq, %warp, %c0 : index")
        lines.append("          scf.if %is_warp0 {")
        lines.append("            %lane_in = arith.cmpi ult, %lane, %cWarps : index")
        lines.append("            %wv0 = scf.if %lane_in -> (f32) {")
        lines.append("              %wt = memref.load %sh[%lane] : memref<256xf32, 3>")
        lines.append("              scf.yield %wt : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        lines.append("            %ws1, %wok1 = gpu.shuffle up %wv0, %c1_i32, %c32_i32 : f32")
        lines.append(f"            %wa1 = arith.addf %wv0, %ws1{fm} : f32")
        lines.append("            %wv1 = arith.select %wok1, %wa1, %wv0 : f32")
        lines.append("            %ws2, %wok2 = gpu.shuffle up %wv1, %c2_i32, %c32_i32 : f32")
        lines.append(f"            %wa2 = arith.addf %wv1, %ws2{fm} : f32")
        lines.append("            %wv2 = arith.select %wok2, %wa2, %wv1 : f32")
        lines.append("            %ws4, %wok4 = gpu.shuffle up %wv2, %c4_i32, %c32_i32 : f32")
        lines.append(f"            %wa4 = arith.addf %wv2, %ws4{fm} : f32")
        lines.append("            %wv4 = arith.select %wok4, %wa4, %wv2 : f32")
        lines.append("            %ws8, %wok8 = gpu.shuffle up %wv4, %c8_i32, %c32_i32 : f32")
        lines.append(f"            %wa8 = arith.addf %wv4, %ws8{fm} : f32")
        lines.append("            %wv8 = arith.select %wok8, %wa8, %wv4 : f32")
        lines.append("            %ws16, %wok16 = gpu.shuffle up %wv8, %c16_i32, %c32_i32 : f32")
        lines.append(f"            %wa16 = arith.addf %wv8, %ws16{fm} : f32")
        lines.append("            %wv16 = arith.select %wok16, %wa16, %wv8 : f32")
        lines.append("            %wprev, %wok_prev = gpu.shuffle up %wv16, %c1_i32, %c32_i32 : f32")
        lines.append("            %woff = arith.select %wok_prev, %wprev, %c0f : f32")
        lines.append("            scf.if %lane_in {")
        lines.append("              %off_idx = arith.addi %cWarps, %lane : index")
        lines.append("              memref.store %woff, %sh[%off_idx] : memref<256xf32, 3>")
        lines.append("            }")
        lines.append("          }")
        lines.append("          gpu.barrier")

        lines.append("          %off_idx2 = arith.addi %cWarps, %warp : index")
        lines.append("          %woff2 = memref.load %sh[%off_idx2] : memref<256xf32, 3>")
        lines.append(f"          %block_scan = arith.addf %v16, %woff2{fm} : f32")
        lines.append(f"          %full_scan = arith.addf %block_scan, %carry0{fm} : f32")
        lines.append(f"          %y = arith.divf %full_scan, %denom{fm} : f32")
        lines.append("          scf.if %in_range {")
        lines.append(f"            memref.store %y, {arg_ssa[out_name2]}[%idx] : {out_memref}")
        lines.append("          }")
        lines.append("          %is_last = arith.cmpi eq, %tid, %c255 : index")
        lines.append("          scf.if %is_last {")
        lines.append("            memref.store %full_scan, %sh[%cCarry] : memref<256xf32, 3>")
        lines.append("          }")
        lines.append("          gpu.barrier")
        lines.append("        }")
        lines.append("      }")
    elif cummax1d_v1 is not None:
        kernel_kind = "cummax1d_axis0_v1"
        n_dim = int(cummax1d_v1["N"])
        if n_dim <= 0 or int(out_total) != int(n_dim):
            raise RuntimeError(f"cummax1d expects out_total==N, got out_total={int(out_total)} N={n_dim}")

        x_name = str(cummax1d_v1["x"])
        out_name2 = str(cummax1d_v1["out"])
        x_memref = str(arg_specs[x_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append("        %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("        %final = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %neg_inf) -> (f32) {")
        lines.append(f"          %xv = memref.load {arg_ssa[x_name]}[%i] : {x_memref}")
        lines.append(f"          %next = arith.maximumf %acc, %xv{fm} : f32")
        lines.append(f"          memref.store %next, {arg_ssa[out_name2]}[%i] : {out_memref}")
        lines.append("          scf.yield %next : f32")
        lines.append("        }")
        lines.append("      }")
    elif cummin1d_v1 is not None:
        kernel_kind = "cummin1d_axis0_v1"
        n_dim = int(cummin1d_v1["N"])
        if n_dim <= 0 or int(out_total) != int(n_dim):
            raise RuntimeError(f"cummin1d expects out_total==N, got out_total={int(out_total)} N={n_dim}")

        x_name = str(cummin1d_v1["x"])
        out_name2 = str(cummin1d_v1["out"])
        x_memref = str(arg_specs[x_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append("        %pos_inf = arith.constant 0x7F800000 : f32")
        lines.append("        %final = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %pos_inf) -> (f32) {")
        lines.append(f"          %xv = memref.load {arg_ssa[x_name]}[%i] : {x_memref}")
        lines.append(f"          %next = arith.minimumf %acc, %xv{fm} : f32")
        lines.append(f"          memref.store %next, {arg_ssa[out_name2]}[%i] : {out_memref}")
        lines.append("          scf.yield %next : f32")
        lines.append("        }")
        lines.append("      }")
    elif upsample_nearest1d_ncl_v1 is not None:
        kernel_kind = "upsample_nearest1d_ncl_v1"
        n_dim = int(upsample_nearest1d_ncl_v1["N"])
        c_dim = int(upsample_nearest1d_ncl_v1["C"])
        il_dim = int(upsample_nearest1d_ncl_v1["IL"])
        ol_dim = int(upsample_nearest1d_ncl_v1["OL"])
        if n_dim <= 0 or c_dim <= 0 or il_dim <= 0 or ol_dim <= 0:
            raise RuntimeError(f"upsample_nearest1d_ncl expects positive dims, got N={n_dim} C={c_dim} IL={il_dim} OL={ol_dim}")

        in_name = str(upsample_nearest1d_ncl_v1["inp"])
        out_name2 = str(upsample_nearest1d_ncl_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"        %cIL = arith.constant {int(il_dim)} : index")
        lines.append(f"        %cOL = arith.constant {int(ol_dim)} : index")
        lines.append("        %ol = arith.remui %lin, %cOL : index")
        lines.append("        %t1 = arith.divui %lin, %cOL : index")
        lines.append("        %cc = arith.remui %t1, %cC : index")
        lines.append("        %nn = arith.divui %t1, %cC : index")
        lines.append("        %ol_mul = arith.muli %ol, %cIL : index")
        lines.append("        %il = arith.divui %ol_mul, %cOL : index")
        lines.append("        %nc = arith.muli %nn, %cC : index")
        lines.append("        %ncc = arith.addi %nc, %cc : index")
        lines.append("        %base_in = arith.muli %ncc, %cIL : index")
        lines.append("        %idx_in = arith.addi %base_in, %il : index")
        lines.append(f"        %x = memref.load {arg_ssa[in_name]}[%idx_in] : {in_memref}")
        lines.append(f"        memref.store %x, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif upsample_nearest2d_nchw_v1 is not None:
        kernel_kind = "upsample_nearest2d_nchw_v1"
        n_dim = int(upsample_nearest2d_nchw_v1["N"])
        c_dim = int(upsample_nearest2d_nchw_v1["C"])
        ih_dim = int(upsample_nearest2d_nchw_v1["IH"])
        iw_dim = int(upsample_nearest2d_nchw_v1["IW"])
        oh_dim = int(upsample_nearest2d_nchw_v1["OH"])
        ow_dim = int(upsample_nearest2d_nchw_v1["OW"])
        if n_dim <= 0 or c_dim <= 0 or ih_dim <= 0 or iw_dim <= 0 or oh_dim <= 0 or ow_dim <= 0:
            raise RuntimeError(
                "upsample_nearest2d_nchw expects positive dims, got "
                f"N={n_dim} C={c_dim} IH={ih_dim} IW={iw_dim} OH={oh_dim} OW={ow_dim}"
            )

        in_name = str(upsample_nearest2d_nchw_v1["inp"])
        out_name2 = str(upsample_nearest2d_nchw_v1["out"])
        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"        %cIH = arith.constant {int(ih_dim)} : index")
        lines.append(f"        %cIW = arith.constant {int(iw_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cIHW = arith.constant {int(ih_dim * iw_dim)} : index")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %cc = arith.remui %t2, %cC : index")
        lines.append("        %nn = arith.divui %t2, %cC : index")
        lines.append("        %oh_mul = arith.muli %oh, %cIH : index")
        lines.append("        %ih = arith.divui %oh_mul, %cOH : index")
        lines.append("        %ow_mul = arith.muli %ow, %cIW : index")
        lines.append("        %iw = arith.divui %ow_mul, %cOW : index")
        lines.append("        %nc = arith.muli %nn, %cC : index")
        lines.append("        %ncc = arith.addi %nc, %cc : index")
        lines.append("        %base_nc = arith.muli %ncc, %cIHW : index")
        lines.append("        %ih_mul2 = arith.muli %ih, %cIW : index")
        lines.append("        %base_h = arith.addi %base_nc, %ih_mul2 : index")
        lines.append("        %idx_in = arith.addi %base_h, %iw : index")
        lines.append(f"        %x = memref.load {arg_ssa[in_name]}[%idx_in] : {in_memref}")
        lines.append(f"        memref.store %x, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif glu2d_v1 is not None:
        kernel_kind = "glu2d_v1"
        n_dim = int(glu2d_v1["N"])
        nh_dim = int(glu2d_v1["N_HALF"])
        if n_dim <= 0 or nh_dim <= 0 or n_dim != 2 * nh_dim:
            raise RuntimeError(f"glu2d expects N == 2*N_HALF, got N={n_dim} N_HALF={nh_dim}")

        x_name = str(glu2d_v1["x"])
        out_name2 = str(glu2d_v1["out"])
        x_memref = str(arg_specs[x_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cNH = arith.constant {int(nh_dim)} : index")
        lines.append("        %col = arith.remui %lin, %cNH : index")
        lines.append("        %row = arith.divui %lin, %cNH : index")
        lines.append("        %row_off = arith.muli %row, %cN : index")
        lines.append("        %idx_a = arith.addi %row_off, %col : index")
        lines.append("        %idx_b = arith.addi %idx_a, %cNH : index")
        lines.append(f"        %a = memref.load {arg_ssa[x_name]}[%idx_a] : {x_memref}")
        lines.append(f"        %b = memref.load {arg_ssa[x_name]}[%idx_b] : {x_memref}")
        lines.append(f"        %neg_b = arith.negf %b{fm} : f32")
        lines.append(f"        %e = math.exp %neg_b{fm} : f32")
        lines.append("        %c1f = arith.constant 1.0 : f32")
        lines.append(f"        %den = arith.addf %c1f, %e{fm} : f32")
        lines.append(f"        %sig = arith.divf %c1f, %den{fm} : f32")
        lines.append(f"        %o = arith.mulf %a, %sig{fm} : f32")
        lines.append(f"        memref.store %o, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif weight_norm2d_v1 is not None:
        kernel_kind = "weight_norm2d_v1"
        m_dim = int(weight_norm2d_v1["M"])
        n_dim = int(weight_norm2d_v1["N"])
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"weight_norm2d expects positive dims, got M={m_dim} N={n_dim}")

        v_name = str(weight_norm2d_v1["v"])
        g_name = str(weight_norm2d_v1["g"])
        out_name2 = str(weight_norm2d_v1["out"])
        v_memref = str(arg_specs[v_name]["memref"])
        g_memref = str(arg_specs[g_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        # sumsq(v) across columns
        lines.append("        %partial = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[v_name]}[%idx] : {v_memref}")
        lines.append(f"          %xx = arith.mulf %x, %x{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc, %xx{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        # Shared-memory reduce across threads in the block (assume block.x==256).
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
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

        lines.append("        %sumsq = memref.load %sh[%c0] : memref<256xf32, 3>")
        g_v = _fresh("g")
        lines.append(f"        {g_v} = memref.load {arg_ssa[g_name]}[%bid] : {g_memref}")
        lines.append(f"        %norm = math.sqrt %sumsq{fm} : f32")
        lines.append(f"        %scale = arith.divf {g_v}, %norm{fm} : f32")

        # out = v * (g / norm)
        lines.append("        scf.for %j2 = %tid to %cN step %bdim {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa[v_name]}[%idx2] : {v_memref}")
        lines.append(f"          %o2 = arith.mulf %x2, %scale{fm} : f32")
        lines.append(f"          memref.store %o2, {arg_ssa[out_name2]}[%idx2] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif mse_loss2d_v1 is not None:
        kernel_kind = "mse_loss2d_v1"
        m_dim = int(mse_loss2d_v1["M"])
        n_dim = int(mse_loss2d_v1["N"])
        total = int(m_dim * n_dim)
        if total <= 0:
            raise RuntimeError(f"mse_loss2d expects positive dims, got M={m_dim} N={n_dim}")

        inp_name = str(mse_loss2d_v1["inp"])
        tgt_name = str(mse_loss2d_v1["target"])
        out_name2 = str(mse_loss2d_v1["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        tgt_memref = str(arg_specs[tgt_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(total)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %partial = scf.for %i = %tid to %c_total step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append(f"        %x = memref.load {arg_ssa[inp_name]}[%i] : {inp_memref}")
        lines.append(f"        %y = memref.load {arg_ssa[tgt_name]}[%i] : {tgt_memref}")
        lines.append(f"        %d = arith.subf %x, %y{fm} : f32")
        lines.append(f"        %sq = arith.mulf %d, %d{fm} : f32")
        lines.append(f"        %acc_next = arith.addf %acc, %sq{fm} : f32")
        lines.append("        scf.yield %acc_next : f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("      memref.store %partial, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_{stride}"
            pS = f"%pS_{stride}"
            tid2 = f"%tid_{stride}"
            a = f"%a_{stride}"
            b = f"%b_{stride}"
            s = f"%s_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %sum0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %cTotF = arith.constant {_as_f32_const(total)} : f32")
        lines.append(f"        %mean = arith.divf %sum0, %cTotF{fm} : f32")
        lines.append(f"        memref.store %mean, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif nll_loss_forward_v1 is not None:
        kernel_kind = "nll_loss_forward_v1"
        n_dim = int(nll_loss_forward_v1["N"])
        c_dim = int(nll_loss_forward_v1["C"])
        ignore_i = int(nll_loss_forward_v1.get("ignore_index") or -100)
        if n_dim <= 0 or c_dim <= 1:
            raise RuntimeError(f"{kernel_kind} expects N>0 and C>1, got N={n_dim} C={c_dim}")

        logits_name = str(nll_loss_forward_v1["self"])
        tgt_name = str(nll_loss_forward_v1["target"])
        w_name = str(nll_loss_forward_v1["weight"])
        out_name2 = str(nll_loss_forward_v1["out"])
        logits_memref = str(arg_specs[logits_name]["memref"])
        tgt_memref = str(arg_specs[tgt_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append(
            "      %partial_loss, %partial_w = scf.for %i = %tid to %cN step %bdim "
            "iter_args(%acc_loss = %c0f, %acc_w = %c0f) -> (f32, f32) {"
        )
        lines.append(f"        %t = memref.load {arg_ssa[tgt_name]}[%i] : {tgt_memref}")
        lines.append(f"        %cIgnore = arith.constant {int(ignore_i)} : i64")
        lines.append("        %valid = arith.cmpi ne, %t, %cIgnore : i64")
        lines.append("        %loss_i, %w_i = scf.if %valid -> (f32, f32) {")
        lines.append("          %c0_i64 = arith.constant 0 : i64")
        lines.append(f"          %cMax_i64 = arith.constant {int(c_dim - 1)} : i64")
        lines.append("          %t0 = arith.maxsi %t, %c0_i64 : i64")
        lines.append("          %t1 = arith.minsi %t0, %cMax_i64 : i64")
        lines.append("          %t_idx = arith.index_cast %t1 : i64 to index")
        lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%t_idx] : {w_memref}")
        lines.append("          %base = arith.muli %i, %cC : index")
        lines.append("          %log_idx = arith.addi %base, %t_idx : index")
        lines.append(f"          %log = memref.load {arg_ssa[logits_name]}[%log_idx] : {logits_memref}")
        lines.append(f"          %neg = arith.negf %log{fm} : f32")
        lines.append(f"          %lw = arith.mulf %neg, %w{fm} : f32")
        lines.append("          scf.yield %lw, %w : f32, f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f, %c0f : f32, f32")
        lines.append("        }")
        lines.append(f"        %acc_loss_next = arith.addf %acc_loss, %loss_i{fm} : f32")
        lines.append(f"        %acc_w_next = arith.addf %acc_w, %w_i{fm} : f32")
        lines.append("        scf.yield %acc_loss_next, %acc_w_next : f32, f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        # Reduce loss sum.
        lines.append("      memref.store %partial_loss, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_loss_{stride}"
            pS = f"%pS_loss_{stride}"
            tid2 = f"%tid_loss_{stride}"
            a = f"%a_loss_{stride}"
            b = f"%b_loss_{stride}"
            s = f"%s_loss_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %sum_loss = memref.load %sh[%c0] : memref<256xf32, 3>")

        # Reduce weight sum (reuse shared buffer).
        lines.append("      memref.store %partial_w, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_w_{stride}"
            pS = f"%pS_w_{stride}"
            tid2 = f"%tid_w_{stride}"
            a = f"%a_w_{stride}"
            b = f"%b_w_{stride}"
            s = f"%s_w_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %sum_w = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append(f"        %cEps = arith.constant {_as_f32_const(1.0e-12)} : f32")
        lines.append(f"        %den = arith.maximumf %sum_w, %cEps{fm} : f32")
        lines.append(f"        %outv = arith.divf %sum_loss, %den{fm} : f32")
        lines.append(f"        memref.store %outv, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif nll_loss2d_forward_v1 is not None:
        kernel_kind = "nll_loss2d_forward_v1"
        n_dim = int(nll_loss2d_forward_v1["N"])
        c_dim = int(nll_loss2d_forward_v1["C"])
        h_dim = int(nll_loss2d_forward_v1["H"])
        w_dim = int(nll_loss2d_forward_v1["W"])
        ignore_i = int(nll_loss2d_forward_v1.get("ignore_index") or -100)
        if n_dim <= 0 or c_dim <= 1 or h_dim <= 0 or w_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects N>0 C>1 H>0 W>0, got N={n_dim} C={c_dim} H={h_dim} W={w_dim}")

        logits_name = str(nll_loss2d_forward_v1["self"])
        tgt_name = str(nll_loss2d_forward_v1["target"])
        w_name = str(nll_loss2d_forward_v1["weight"])
        out_name2 = str(nll_loss2d_forward_v1["out"])
        logits_memref = str(arg_specs[logits_name]["memref"])
        tgt_memref = str(arg_specs[tgt_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        total = int(n_dim * h_dim * w_dim)
        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(total)} : index")
        lines.append(f"      %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"      %cHW = arith.constant {int(h_dim * w_dim)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append(
            "      %partial_loss, %partial_w = scf.for %lin = %tid to %c_total step %bdim "
            "iter_args(%acc_loss = %c0f, %acc_w = %c0f) -> (f32, f32) {"
        )
        lines.append(f"        %t = memref.load {arg_ssa[tgt_name]}[%lin] : {tgt_memref}")
        lines.append(f"        %cIgnore = arith.constant {int(ignore_i)} : i64")
        lines.append("        %valid = arith.cmpi ne, %t, %cIgnore : i64")
        lines.append("        %loss_i, %w_i = scf.if %valid -> (f32, f32) {")
        lines.append("          %c0_i64 = arith.constant 0 : i64")
        lines.append(f"          %cMax_i64 = arith.constant {int(c_dim - 1)} : i64")
        lines.append("          %t0 = arith.maxsi %t, %c0_i64 : i64")
        lines.append("          %t1 = arith.minsi %t0, %cMax_i64 : i64")
        lines.append("          %t_idx = arith.index_cast %t1 : i64 to index")
        lines.append(f"          %w = memref.load {arg_ssa[w_name]}[%t_idx] : {w_memref}")
        lines.append("          %n = arith.divui %lin, %cHW : index")
        lines.append("          %rem = arith.remui %lin, %cHW : index")
        lines.append("          %nC = arith.muli %n, %cC : index")
        lines.append("          %nc = arith.addi %nC, %t_idx : index")
        lines.append("          %base = arith.muli %nc, %cHW : index")
        lines.append("          %log_idx = arith.addi %base, %rem : index")
        lines.append(f"          %log = memref.load {arg_ssa[logits_name]}[%log_idx] : {logits_memref}")
        lines.append(f"          %neg = arith.negf %log{fm} : f32")
        lines.append(f"          %lw = arith.mulf %neg, %w{fm} : f32")
        lines.append("          scf.yield %lw, %w : f32, f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f, %c0f : f32, f32")
        lines.append("        }")
        lines.append(f"        %acc_loss_next = arith.addf %acc_loss, %loss_i{fm} : f32")
        lines.append(f"        %acc_w_next = arith.addf %acc_w, %w_i{fm} : f32")
        lines.append("        scf.yield %acc_loss_next, %acc_w_next : f32, f32")
        lines.append("      }")

        lines.append(f"      %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        # Reduce loss sum.
        lines.append("      memref.store %partial_loss, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_loss_{stride}"
            pS = f"%pS_loss_{stride}"
            tid2 = f"%tid_loss_{stride}"
            a = f"%a_loss_{stride}"
            b = f"%b_loss_{stride}"
            s = f"%s_loss_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %sum_loss = memref.load %sh[%c0] : memref<256xf32, 3>")

        # Reduce weight sum (reuse shared buffer).
        lines.append("      memref.store %partial_w, %sh[%tid] : memref<256xf32, 3>")
        lines.append("      gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_w_{stride}"
            pS = f"%pS_w_{stride}"
            tid2 = f"%tid_w_{stride}"
            a = f"%a_w_{stride}"
            b = f"%b_w_{stride}"
            s = f"%s_w_{stride}"
            lines.append(f"      {cS} = arith.constant {int(stride)} : index")
            lines.append(f"      {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"      scf.if {pS} {{")
            lines.append(f"        {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"        {a} = memref.load %sh[%tid] : memref<256xf32, 3>")
            lines.append(f"        {b} = memref.load %sh[{tid2}] : memref<256xf32, 3>")
            lines.append(f"        {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"        memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("      }")
            lines.append("      gpu.barrier")

        lines.append("      %sum_w = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append(f"        %cEps = arith.constant {_as_f32_const(1.0e-12)} : f32")
        lines.append(f"        %den = arith.maximumf %sum_w, %cEps{fm} : f32")
        lines.append(f"        %outv = arith.divf %sum_loss, %den{fm} : f32")
        lines.append(f"        memref.store %outv, {arg_ssa[out_name2]}[%c0] : {out_memref}")
        lines.append("      }")
    elif row_std_axis1_v1 is not None or row_var_axis1_v1 is not None:
        is_std = row_std_axis1_v1 is not None
        cfg = row_std_axis1_v1 if row_std_axis1_v1 is not None else row_var_axis1_v1
        assert cfg is not None
        kernel_kind = "row_std_axis1_v1" if is_std else "row_var_axis1_v1"
        m_dim = int(cfg["M"])
        n_dim = int(cfg["N"])
        ddof = int(cfg.get("ddof") or 0)
        if m_dim <= 0 or n_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim}")
        if ddof != 1:
            raise RuntimeError(f"{kernel_kind} currently requires ddof=1, got ddof={ddof}")
        if n_dim - ddof <= 0:
            raise RuntimeError(f"{kernel_kind} invalid denom: N={n_dim} ddof={ddof}")

        inp_name = str(cfg["inp"])
        out_name2 = str(cfg["out"])
        inp_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [int(m_dim), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(m_dim)} : index")
        lines.append(f"      %cN = arith.constant {int(n_dim)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")

        # sum
        lines.append("        %partial_sum = scf.for %j = %tid to %cN step %bdim iter_args(%acc = %c0f) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")
        lines.append(f"          %x = memref.load {arg_ssa[inp_name]}[%idx] : {inp_memref}")
        lines.append(f"          %acc_next = arith.addf %acc, %x{fm} : f32")
        lines.append("          scf.yield %acc_next : f32")
        lines.append("        }")

        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        memref.store %partial_sum, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sum_{stride}"
            pS = f"%pS_sum_{stride}"
            tid2 = f"%tid_sum_{stride}"
            a = f"%a_sum_{stride}"
            b = f"%b_sum_{stride}"
            s = f"%s_sum_{stride}"
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

        lines.append("        %sum0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %cN_f = arith.constant {_as_f32_const(n_dim)} : f32")
        lines.append(f"        %mean = arith.divf %sum0, %cN_f{fm} : f32")

        # sumsq (x - mean)^2
        lines.append("        %partial_sqs = scf.for %j2 = %tid to %cN step %bdim iter_args(%acc2 = %c0f) -> (f32) {")
        lines.append("          %idx2 = arith.addi %base, %j2 : index")
        lines.append(f"          %x2 = memref.load {arg_ssa[inp_name]}[%idx2] : {inp_memref}")
        lines.append(f"          %d2 = arith.subf %x2, %mean{fm} : f32")
        lines.append(f"          %sq2 = arith.mulf %d2, %d2{fm} : f32")
        lines.append(f"          %acc2_next = arith.addf %acc2, %sq2{fm} : f32")
        lines.append("          scf.yield %acc2_next : f32")
        lines.append("        }")

        lines.append("        memref.store %partial_sqs, %sh[%tid] : memref<256xf32, 3>")
        lines.append("        gpu.barrier")
        for stride in (128, 64, 32, 16, 8, 4, 2, 1):
            cS = f"%cS_sqs_{stride}"
            pS = f"%pS_sqs_{stride}"
            tid2 = f"%tid_sqs_{stride}"
            a = f"%a_sqs_{stride}"
            b = f"%b_sqs_{stride}"
            s = f"%s_sqs_{stride}"
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

        lines.append("        %sumsq0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        lines.append(f"        %cDen_f = arith.constant {_as_f32_const(n_dim - ddof)} : f32")
        lines.append(f"        %var = arith.divf %sumsq0, %cDen_f{fm} : f32")
        out_ssa = "%var"
        if is_std:
            lines.append(f"        %outv = math.sqrt %var{fm} : f32")
            out_ssa = "%outv"

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append(f"          memref.store {out_ssa}, {arg_ssa[out_name2]}[%bid] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif row_reduce_sum_axis1 is not None:
        kernel_kind = "row_reduce_sum_axis1_v1"
        # One block per output row; let each block reduce across the N dimension.
        # This keeps global loads coalesced (threads iterate columns).
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

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
        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
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
        if out_rank == 0 and red_out == out_name:
            # Scalar reductions (e.g. dot/vdot): avoid re-evaluating per-element ops
            # with a 1D row index. Just store the reduced value.
            lines.append(f"          memref.store %sum0, {arg_ssa[str(out_name)]}[%c0] : {out_memref}")
        else:
            body_lines = _emit_elementwise_for_index(
                "%bid",
                precomputed={red_out: ("%sum0", "f32")},
                skip_rank2_outputs=True,
            )
            for l in body_lines:
                lines.append("  " + l)
        lines.append("        }")
        lines.append("      }")
    elif row_reduce_max_axis1 is not None:
        kernel_kind = "row_reduce_max_axis1_v1"
        # One block per output row; let each block reduce across the N dimension.
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = "memref<256xf32, 3>"

        red_in = str(row_reduce_max_axis1["red_in"])
        red_out = str(row_reduce_max_axis1["red_out"])
        red_op_index = int(row_reduce_max_axis1["op_index"])
        red_n = int(row_reduce_max_axis1["reduce_n"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        # f32 min as init to avoid relying on -inf parsing.
        lines.append("        %init = arith.constant -3.402823466e+38 : f32")
        # Per-thread partial max.
        lines.append("        %partial = scf.for %j = %tid to %cN_red step %bdim iter_args(%acc = %init) -> (f32) {")
        lines.append("          %idx = arith.addi %base, %j : index")

        # Pre-reduction elementwise evaluation up to the reduce_max op (limited subset).
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

        # Evaluate ops up to reduce_max to materialize `red_in` at this element.
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
        elem_lines.append(f"          %acc_next = arith.maximumf %acc, {val_ssa}{fm} : f32")
        elem_lines.append("          scf.yield %acc_next : f32")

        lines.extend(elem_lines)
        lines.append("        }")

        # Shared-memory reduce across threads in the block (assume block.x==256).
        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xf32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
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
            lines.append(f"          {s} = arith.maximumf {a}, {b}{fm} : f32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xf32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %max0 = memref.load %sh[%c0] : memref<256xf32, 3>")
        body_lines = _emit_elementwise_for_index(
            "%bid",
            precomputed={red_out: ("%max0", "f32")},
            skip_rank2_outputs=True,
        )
        for l in body_lines:
            lines.append("  " + l)
        lines.append("        }")
        lines.append("      }")
    elif row_reduce_any_axis1 is not None:
        kernel_kind = "row_reduce_any_axis1_v1"
        # One block per output row; reduce across the N dimension to a single bool.
        launch_override = {"block": [256, 1, 1], "grid": [int(out_total), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_i32"
        shared_global_memref_ty = "memref<256xi32, 3>"

        red_in = str(row_reduce_any_axis1["red_in"])
        red_out = str(row_reduce_any_axis1["red_out"])
        red_op_index = int(row_reduce_any_axis1["op_index"])
        red_n = int(row_reduce_any_axis1["reduce_n"])

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cM = arith.constant {int(out_total)} : index")
        lines.append(f"      %cN_red = arith.constant {int(red_n)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cM : index")
        lines.append("      scf.if %pred_row {")
        lines.append("        %base = arith.muli %bid, %cN_red : index")
        lines.append("        %false = arith.constant 0 : i1")
        # Per-thread partial any.
        lines.append("        %partial = scf.for %j = %tid to %cN_red step %bdim iter_args(%acc = %false) -> (i1) {")
        lines.append("          %idx = arith.addi %base, %j : index")

        # Pre-reduction elementwise evaluation up to the reduce_any op (limited subset).
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

        # Evaluate ops up to reduce_any to materialize `red_in` at this element.
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
            if op_name == "abs":
                if len(inputs) != 1:
                    raise RuntimeError("abs expects 1 input")
                src_v, src_ty = _value_elem(inputs[0])
                if src_ty != "f32":
                    raise RuntimeError(f"abs row-reduce pre-eval supports f32 only (src_ty={src_ty})")
                dst = _fresh("abs")
                elem_lines.append(f"          {dst} = math.absf {src_v}{fm} : f32")
                computed[outv] = dst
                computed_ty[outv] = "f32"
                continue
            if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
                if len(inputs) != 2:
                    raise RuntimeError(f"{op_name} expects 2 inputs")
                a_v, a_ty = _value_elem(inputs[0])
                b_v, b_ty = _value_elem(inputs[1])
                dst = _fresh(op_name)
                float_tys = {"f16", "bf16", "f32"}
                int_tys = {"i1", "i32", "i64"}
                if a_ty in float_tys or b_ty in float_tys:
                    lhs = _coerce_elem(a_v, a_ty, "f32")
                    rhs = _coerce_elem(b_v, b_ty, "f32")
                    pred = {
                        "eq": "oeq",
                        "ne": "one",
                        "lt": "olt",
                        "le": "ole",
                        "gt": "ogt",
                        "ge": "oge",
                    }[op_name]
                    elem_lines.append(f"          {dst} = arith.cmpf {pred}, {lhs}, {rhs} : f32")
                elif a_ty in int_tys and b_ty in int_tys:
                    cmp_ty = (
                        "i64"
                        if ("i64" in {a_ty, b_ty})
                        else ("i32" if ("i32" in {a_ty, b_ty}) else "i1")
                    )
                    lhs = _coerce_elem(a_v, a_ty, cmp_ty)
                    rhs = _coerce_elem(b_v, b_ty, cmp_ty)
                    pred = {
                        "eq": "eq",
                        "ne": "ne",
                        "lt": "slt",
                        "le": "sle",
                        "gt": "sgt",
                        "ge": "sge",
                    }[op_name]
                    elem_lines.append(f"          {dst} = arith.cmpi {pred}, {lhs}, {rhs} : {cmp_ty}")
                else:
                    raise RuntimeError(f"{op_name} unsupported input dtypes in row-reduce: {a_ty} vs {b_ty}")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue
            if op_name == "and":
                if len(inputs) != 2:
                    raise RuntimeError("and expects 2 inputs")
                a_v, a_ty = _value_elem(inputs[0])
                b_v, b_ty = _value_elem(inputs[1])
                if a_ty != "i1" or b_ty != "i1":
                    raise RuntimeError(f"and row-reduce pre-eval supports i1 only (got {a_ty},{b_ty})")
                dst = _fresh("and")
                elem_lines.append(f"          {dst} = arith.andi {a_v}, {b_v} : i1")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue
            if op_name == "not":
                if len(inputs) != 1:
                    raise RuntimeError("not expects 1 input")
                a_v, a_ty = _value_elem(inputs[0])
                if a_ty != "i1":
                    raise RuntimeError(f"not row-reduce pre-eval supports i1 only (got {a_ty})")
                c1 = _fresh("c1")
                dst = _fresh("not")
                elem_lines.append(f"          {c1} = arith.constant 1 : i1")
                elem_lines.append(f"          {dst} = arith.xori {a_v}, {c1} : i1")
                computed[outv] = dst
                computed_ty[outv] = "i1"
                continue
            raise RuntimeError(f"unsupported op in row-reduce pre-eval: {op_name}")

        val_ssa, val_ty = _value_elem(red_in)
        if val_ty != "i1":
            raise RuntimeError(f"row-reduce expects i1 reduction input, got {val_ty} for {red_in}")
        elem_lines.append(f"          %acc_next = arith.ori %acc, {val_ssa} : i1")
        elem_lines.append("          scf.yield %acc_next : i1")

        lines.extend(elem_lines)
        lines.append("        }")

        # Shared-memory reduce across threads in the block (assume block.x==256).
        #
        # Note: keep the shared-memory element type >= i32 for NVPTX/llc
        # robustness (smaller integer element types have triggered verifier
        # issues in practice).
        assert shared_global_sym is not None
        assert shared_global_memref_ty == "memref<256xi32, 3>"
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append("        %partial_i32 = arith.extui %partial : i1 to i32")
        lines.append("        memref.store %partial_i32, %sh[%tid] : memref<256xi32, 3>")
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
            lines.append(f"          {a} = memref.load %sh[%tid] : memref<256xi32, 3>")
            lines.append(f"          {b} = memref.load %sh[{tid2}] : memref<256xi32, 3>")
            lines.append(f"          {s} = arith.ori {a}, {b} : i32")
            lines.append(f"          memref.store {s}, %sh[%tid] : memref<256xi32, 3>")
            lines.append("        }")
            lines.append("        gpu.barrier")

        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %any0_i32 = memref.load %sh[%c0] : memref<256xi32, 3>")
        lines.append("          %c0i32 = arith.constant 0 : i32")
        lines.append("          %any0 = arith.cmpi ne, %any0_i32, %c0i32 : i32")
        body_lines = _emit_elementwise_for_index(
            "%bid",
            precomputed={red_out: ("%any0", "i1")},
            skip_rank2_outputs=True,
        )
        for l in body_lines:
            lines.append("  " + l)
        lines.append("        }")
        lines.append("      }")
    elif (
            attn2d_v1 is not None
            and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", default=False)
            and intent_name == "flash_attention2d"
            and int(attn2d_v1.get("HEAD_DIM") or 0) == 64
            and not _env_flag("INTENTIR_CUDA_REAL_MLIR_FLASH_ATTN_V2", default=False)
            and not _env_flag("INTENTIR_CUDA_REAL_MLIR_FLASH_ATTN_V3", default=False)
        ):
            # Perf-first: flash_attention2d is a min-ratio offender in the Triton-native
            # perf lane. Use a multi-query per CTA implementation that reuses K/V tiles
            # via shared memory and uses sub-warp (width=16) shuffles for reductions.
            #
            # Mapping (HEAD_DIM==64 only):
            # - block.x = 256 (8 warps)
            # - each warp computes 2 queries (2x half-warps), each lane computes 4 contiguous cols
            kernel_kind = "attn2d_causal_softmax_v5"
            q_name = str(attn2d_v1["Q"])
            k_name = str(attn2d_v1["K"])
            v_name = str(attn2d_v1["V"])
            out_name2 = str(attn2d_v1["out"])
            sm_scale_name = str(attn2d_v1["sm_scale"])
            q_ctx = int(attn2d_v1["Q_CTX"])
            kv_ctx = int(attn2d_v1["KV_CTX"])
            hd = int(attn2d_v1["HEAD_DIM"])
    
            q_memref = str(arg_specs[q_name]["memref"])
            k_memref = str(arg_specs[k_name]["memref"])
            v_memref = str(arg_specs[v_name]["memref"])
            out_memref = str(arg_specs[out_name2]["memref"])
            sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])
    
            block_x = 256
            block_m = 16
            req_block_kv = 32
            block_kv = int(min(int(req_block_kv), int(kv_ctx))) if int(kv_ctx) > 0 else int(req_block_kv)
            grid_x = (int(q_ctx) + int(block_m) - 1) // int(block_m)
            launch_override = {"block": [int(block_x), 1, 1], "grid": [int(grid_x), 1, 1]}
            cuda_real_mlir_attention_cfg = {
                "block_x": int(block_x),
                "block_m": int(block_m),
                "block_kv": int(block_kv),
                "subwarp_width": 16,
                "vec": 4,
                "shared_kv": True,
                "barrier": "nvvm.barrier0",
            }
    
            vec4_ty = "vector<4xf32>"
    
            # Shared layout: [K_tile (block_kv*hd), V_tile (block_kv*hd)].
            tile_elems = int(block_kv) * int(hd)
            offset_v = int(tile_elems)
            sh_elems = int(tile_elems) * 2
            shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
            shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"
    
            def _subwarp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
                cur = str(val_ssa)
                for off in ("%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                    sh = _fresh("sh")
                    nxt = _fresh("sum")
                    ok = _fresh("ok")
                    # Use width=32 (full-warp) so the generated `shfl.sync` mask includes all lanes.
                    # Offsets {8,4,2,1} ensure we reduce within each half-warp independently.
                    lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                    lines.append(f"{indent}{nxt} = arith.addf {cur}, {sh}{fm} : f32")
                    cur = str(nxt)
                return cur
    
            lines.append("      %tid = gpu.thread_id x")
            lines.append("      %bid_x = gpu.block_id x")
            lines.append("      %c0 = arith.constant 0 : index")
            lines.append("      %c1 = arith.constant 1 : index")
            lines.append("      %c2 = arith.constant 2 : index")
            lines.append("      %c4 = arith.constant 4 : index")
            lines.append("      %c0f = arith.constant 0.0 : f32")
            lines.append("      %c1f = arith.constant 1.0 : f32")
            lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
            lines.append("      %c16_idx = arith.constant 16 : index")
            lines.append("      %c32_idx = arith.constant 32 : index")
            lines.append("      %c32_i32 = arith.constant 32 : i32")
            lines.append("      %c8_i32 = arith.constant 8 : i32")
            lines.append("      %c4_i32 = arith.constant 4 : i32")
            lines.append("      %c2_i32 = arith.constant 2 : i32")
            lines.append("      %c1_i32 = arith.constant 1 : i32")
            lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
            lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
            lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
            lines.append(f"      %cHD = arith.constant {int(hd)} : index")
            lines.append(f"      %cBlockM = arith.constant {int(block_m)} : index")
            lines.append(f"      %cBlockKV = arith.constant {int(block_kv)} : index")
            lines.append(f"      %cGridX = arith.constant {int(grid_x)} : index")
            lines.append(f"      %cOffsetV = arith.constant {int(offset_v)} : index")
            lines.append("      %pred_block = arith.cmpi ult, %bid_x, %cGridX : index")
            lines.append("      scf.if %pred_block {")
            lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
            lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
            lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")
    
            # Warp/lane decomposition (block.x == 256 => 8 warps). Two queries per warp (half-warps).
            lines.append("        %lane = arith.remui %tid, %c32_idx : index")
            lines.append("        %warp = arith.divui %tid, %c32_idx : index")
            lines.append("        %lane16 = arith.remui %lane, %c16_idx : index")
            lines.append("        %half = arith.divui %lane, %c16_idx : index")
            lines.append("        %d_base = arith.muli %lane16, %c4 : index")
    
            # Map CTA -> q_base, and (warp,half) -> q within the CTA.
            lines.append("        %q_base = arith.muli %bid_x, %cBlockM : index")
            lines.append("        %warp2 = arith.muli %warp, %c2 : index")
            lines.append("        %q0 = arith.addi %q_base, %warp2 : index")
            lines.append("        %q = arith.addi %q0, %half : index")
            lines.append("        %pred_q = arith.cmpi ult, %q, %cQ : index")
            lines.append("        %base_q = arith.muli %q, %cHD : index")
    
            # Load Q vector (or zero) for this (q, lane16*4).
            lines.append(f"        %zvec = vector.splat %c0f : {vec4_ty}")
            q_vec = _fresh("q_vec")
            lines.append(f"        {q_vec} = scf.if %pred_q -> ({vec4_ty}) {{")
            q_idx = _fresh("q_idx")
            q_load = _fresh("q_load")
            lines.append(f"          {q_idx} = arith.addi %base_q, %d_base : index")
            lines.append(f"          {q_load} = vector.load {arg_ssa[q_name]}[{q_idx}] : {q_memref}, {vec4_ty}")
            lines.append(f"          scf.yield {q_load} : {vec4_ty}")
            lines.append("        } else {")
            lines.append(f"          scf.yield %zvec : {vec4_ty}")
            lines.append("        }")
    
            # Skip KV tiles that are always masked by causality for this CTA.
            lines.append("        %q_last0 = arith.addi %q_base, %cBlockM : index")
            lines.append("        %q_last = arith.subi %q_last0, %c1 : index")
            lines.append("        %q_lim = arith.subi %cQ, %c1 : index")
            lines.append("        %p_q_last = arith.cmpi ule, %q_last, %q_lim : index")
            lines.append("        %q_max = arith.select %p_q_last, %q_last, %q_lim : index")
            lines.append("        %kv_end0 = arith.addi %q_max, %c1 : index")
            lines.append("        %p_kv_end = arith.cmpi ule, %kv_end0, %cKV : index")
            lines.append("        %kv_end_max = arith.select %p_kv_end, %kv_end0, %cKV : index")
    
            m_out = _fresh("m")
            l_out = _fresh("l")
            acc_out = _fresh("acc")
            lines.append(
                f"        {m_out}, {l_out}, {acc_out} = scf.for %tile0 = %c0 to %kv_end_max step %cBlockKV "
                f"iter_args(%m_i = %neg_inf, %l_i = %c0f, %a = %zvec) -> (f32, f32, {vec4_ty}) {{"
            )
    
            tile_end0 = _fresh("tile_end0")
            tile_end_pred = _fresh("tile_end_pred")
            tile_end = _fresh("tile_end")
            tile_len = _fresh("tile_len")
            tile_elems_dyn = _fresh("tile_elems")
            lines.append(f"          {tile_end0} = arith.addi %tile0, %cBlockKV : index")
            lines.append(f"          {tile_end_pred} = arith.cmpi ule, {tile_end0}, %kv_end_max : index")
            lines.append(f"          {tile_end} = arith.select {tile_end_pred}, {tile_end0}, %kv_end_max : index")
            lines.append(f"          {tile_len} = arith.subi {tile_end}, %tile0 : index")
            lines.append(f"          {tile_elems_dyn} = arith.muli {tile_len}, %cHD : index")
    
            # Cooperative load K/V tile into shared.
            base_tile = _fresh("base_tile")
            lines.append(f"          {base_tile} = arith.muli %tile0, %cHD : index")
            loads = (int(tile_elems) + int(block_x) - 1) // int(block_x)
            for i in range(int(loads)):
                off = int(i) * int(block_x)
                c_off = _fresh("c_off")
                idx = _fresh("idx")
                pred = _fresh("pred")
                lines.append(f"          {c_off} = arith.constant {int(off)} : index")
                lines.append(f"          {idx} = arith.addi %tid, {c_off} : index")
                lines.append(f"          {pred} = arith.cmpi ult, {idx}, {tile_elems_dyn} : index")
                lines.append(f"          scf.if {pred} {{")
                k_val = _fresh("kval")
                v_val = _fresh("vval")
                idx_g = _fresh("idx_g")
                lines.append(f"            {idx_g} = arith.addi {base_tile}, {idx} : index")
                lines.append(f"            {k_val} = memref.load {arg_ssa[k_name]}[{idx_g}] : {k_memref}")
                lines.append(f"            {v_val} = memref.load {arg_ssa[v_name]}[{idx_g}] : {v_memref}")
                idx_vsh = _fresh("idx_vsh")
                lines.append(f"            memref.store {k_val}, %sh[{idx}] : {shared_global_memref_ty}")
                lines.append(f"            {idx_vsh} = arith.addi {idx}, %cOffsetV : index")
                lines.append(f"            memref.store {v_val}, %sh[{idx_vsh}] : {shared_global_memref_ty}")
                lines.append("          }")
    
            lines.append("          nvvm.barrier0")
    
            # Update per-query state (skip compute for out-of-range warps, but still participate in barriers).
            m_next = _fresh("m_next")
            l_next = _fresh("l_next")
            a_next = _fresh("a_next")
            lines.append(f"          {m_next}, {l_next}, {a_next} = scf.if %pred_q -> (f32, f32, {vec4_ty}) {{")
    
            m_tile = _fresh("m_tile")
            l_tile = _fresh("l_tile")
            a_tile = _fresh("a_tile")
            lines.append(
                f"            {m_tile}, {l_tile}, {a_tile} = scf.for %t2 = %c0 to {tile_len} step %c1 "
                f"iter_args(%m = %m_i, %l = %l_i, %acc = %a) -> (f32, f32, {vec4_ty}) {{"
            )
            kv_ssa = _fresh("kv")
            pred_attend = _fresh("pred_attend")
            lines.append(f"              {kv_ssa} = arith.addi %tile0, %t2 : index")
            lines.append(f"              {pred_attend} = arith.cmpi ule, {kv_ssa}, %q : index")

            base_row = _fresh("base")
            lines.append(f"              {base_row} = arith.muli %t2, %cHD : index")

            # Compute dot partial for attending lanes (others use 0.0), then reduce with
            # warp shuffles outside the predicated region to avoid `shfl.sync` divergence.
            partial = _fresh("partial")
            lines.append(f"              {partial} = scf.if {pred_attend} -> (f32) {{")
            idx_k = _fresh("idx_k")
            k_vec = _fresh("k_vec")
            prod = _fresh("prod")
            lines.append(f"                {idx_k} = arith.addi {base_row}, %d_base : index")
            lines.append(f"                {k_vec} = vector.load %sh[{idx_k}] : {shared_global_memref_ty}, {vec4_ty}")
            lines.append(f"                {prod} = arith.mulf {q_vec}, {k_vec}{fm} : {vec4_ty}")
            p0 = _fresh("p0")
            p1 = _fresh("p1")
            p2 = _fresh("p2")
            p3 = _fresh("p3")
            s01 = _fresh("s01")
            s23 = _fresh("s23")
            partial_in = _fresh("partial_in")
            lines.append(f"                {p0} = vector.extract {prod}[0] : f32 from {vec4_ty}")
            lines.append(f"                {p1} = vector.extract {prod}[1] : f32 from {vec4_ty}")
            lines.append(f"                {p2} = vector.extract {prod}[2] : f32 from {vec4_ty}")
            lines.append(f"                {p3} = vector.extract {prod}[3] : f32 from {vec4_ty}")
            lines.append(f"                {s01} = arith.addf {p0}, {p1}{fm} : f32")
            lines.append(f"                {s23} = arith.addf {p2}, {p3}{fm} : f32")
            lines.append(f"                {partial_in} = arith.addf {s01}, {s23}{fm} : f32")
            lines.append(f"                scf.yield {partial_in} : f32")
            lines.append("              } else {")
            lines.append("                scf.yield %c0f : f32")
            lines.append("              }")
            dot = _subwarp_allreduce_sum_xor(partial, indent="              ")
            score = _fresh("score")
            lines.append(f"              {score} = arith.mulf {dot}, %sm2{fm} : f32")

            m_kv = _fresh("m_kv")
            l_kv = _fresh("l_kv")
            a_kv = _fresh("a_kv")
            lines.append(f"              {m_kv}, {l_kv}, {a_kv} = scf.if {pred_attend} -> (f32, f32, {vec4_ty}) {{")
    
            pred_new = _fresh("pred_new")
            m_new = _fresh("m_new")
            alpha = _fresh("alpha")
            p = _fresh("p")
            lines.append(f"                {pred_new} = arith.cmpf ogt, {score}, %m : f32")
            lines.append(f"                {m_new} = arith.maximumf %m, {score}{fm} : f32")
            delta = _fresh("delta")
            exp_delta = _fresh("exp_delta")
            lines.append(f"                {delta} = arith.subf {score}, %m{fm} : f32")
            lines.append(f"                {exp_delta} = math.exp2 {delta} : f32")
            lines.append(f"                {alpha} = scf.if {pred_new} -> (f32) {{")
            inv = _fresh("inv")
            lines.append(f"                  {inv} = arith.divf %c1f, {exp_delta}{fm} : f32")
            lines.append(f"                  scf.yield {inv} : f32")
            lines.append("                } else {")
            lines.append("                  scf.yield %c1f : f32")
            lines.append("                }")
            lines.append(f"                {p} = arith.select {pred_new}, %c1f, {exp_delta} : f32")
    
            alpha_v = _fresh("alpha_v")
            p_v = _fresh("p_v")
            acc_scaled = _fresh("acc_scaled")
            lines.append(f"                {alpha_v} = vector.splat {alpha} : {vec4_ty}")
            lines.append(f"                {p_v} = vector.splat {p} : {vec4_ty}")
            lines.append(f"                {acc_scaled} = arith.mulf %acc, {alpha_v}{fm} : {vec4_ty}")
    
            base_v = _fresh("base_v")
            idx_v = _fresh("idx_v")
            v_vec = _fresh("v_vec")
            lines.append(f"                {base_v} = arith.addi {base_row}, %cOffsetV : index")
            lines.append(f"                {idx_v} = arith.addi {base_v}, %d_base : index")
            lines.append(f"                {v_vec} = vector.load %sh[{idx_v}] : {shared_global_memref_ty}, {vec4_ty}")
    
            acc_next2 = _fresh("acc_next")
            l_new = _fresh("l_new")
            lines.append(
                f"                {acc_next2} = llvm.intr.fma({p_v}, {v_vec}, {acc_scaled}) "
                f": ({vec4_ty}, {vec4_ty}, {vec4_ty}) -> {vec4_ty}"
            )
            lines.append(f"                {l_new} = llvm.intr.fma(%l, {alpha}, {p}) : (f32, f32, f32) -> f32")
            lines.append(f"                scf.yield {m_new}, {l_new}, {acc_next2} : f32, f32, {vec4_ty}")
            lines.append("              } else {")
            lines.append(f"                scf.yield %m, %l, %acc : f32, f32, {vec4_ty}")
            lines.append("              }")
    
            lines.append(f"              scf.yield {m_kv}, {l_kv}, {a_kv} : f32, f32, {vec4_ty}")
            lines.append("            }")
    
            lines.append(f"            scf.yield {m_tile}, {l_tile}, {a_tile} : f32, f32, {vec4_ty}")
            lines.append("          } else {")
            lines.append(f"            scf.yield %m_i, %l_i, %a : f32, f32, {vec4_ty}")
            lines.append("          }")
    
            lines.append("          nvvm.barrier0")
            lines.append(f"          scf.yield {m_next}, {l_next}, {a_next} : f32, f32, {vec4_ty}")
            lines.append("        }")
    
            sum_nz = _fresh("sum_nz")
            l_safe = _fresh("l_safe")
            l_vec = _fresh("l_vec")
            out_vec = _fresh("out_vec")
            lines.append(f"        {sum_nz} = arith.cmpf one, {l_out}, %c0f : f32")
            lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_out}, %c1f : f32")
            lines.append(f"        {l_vec} = vector.splat {l_safe} : {vec4_ty}")
            lines.append(f"        {out_vec} = arith.divf {acc_out}, {l_vec}{fm} : {vec4_ty}")
            lines.append("        scf.if %pred_q {")
            idx_o = _fresh("idx_o")
            lines.append(f"          {idx_o} = arith.addi %base_q, %d_base : index")
            lines.append(f"          vector.store {out_vec}, {arg_ssa[out_name2]}[{idx_o}] : {out_memref}, {vec4_ty}")
            lines.append("        }")
            lines.append("      }")
    elif (
        attn2d_v1 is not None
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", default=False)
        and intent_name == "flash_attention2d"
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_FLASH_ATTN_V2", default=False)
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_FLASH_ATTN_V3", default=False)
    ):
        # Perf-first: flash_attention2d is a min-ratio offender in the Triton-native
        # perf lane. Use a multi-query per CTA implementation (8 warps / CTA) that
        # reuses K/V tiles via shared memory and uses warp shuffles for reductions.
        kernel_kind = "attn2d_causal_softmax_v4"
        q_name = str(attn2d_v1["Q"])
        k_name = str(attn2d_v1["K"])
        v_name = str(attn2d_v1["V"])
        out_name2 = str(attn2d_v1["out"])
        sm_scale_name = str(attn2d_v1["sm_scale"])
        q_ctx = int(attn2d_v1["Q_CTX"])
        kv_ctx = int(attn2d_v1["KV_CTX"])
        hd = int(attn2d_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        block_x = 256
        block_m = 8
        req_block_kv = 32
        block_kv = int(min(int(req_block_kv), int(kv_ctx))) if int(kv_ctx) > 0 else int(req_block_kv)
        grid_x = (int(q_ctx) + int(block_m) - 1) // int(block_m)
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(grid_x), 1, 1]}
        cuda_real_mlir_attention_cfg = {
            "block_x": int(block_x),
            "block_m": int(block_m),
            "block_kv": int(block_kv),
            "shared_kv": True,
            "barrier": "nvvm.barrier0",
        }

        # Shared layout: [K_tile (block_kv*hd), V_tile (block_kv*hd)].
        tile_elems = int(block_kv) * int(hd)
        offset_v = int(tile_elems)
        sh_elems = int(tile_elems) * 2
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        fast_hd64 = int(hd) == 64

        def _warp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
            cur = str(val_ssa)
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                nxt = _fresh("sum")
                ok = _fresh("ok")
                # `gpu.shuffle` returns (value, valid). For a full warp shuffle with width=32,
                # `valid` is always true; skipping the `select` avoids extra control/dataflow
                # in the inner dot loop and improves PTX quality for attention.
                lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"{indent}{nxt} = arith.addf {cur}, {sh}{fm} : f32")
                cur = str(nxt)
            return cur

        def _emit_dot_score_shared(*, t_ssa: str, indent: str) -> tuple[str, str]:
            kv = _fresh("kv")
            pred_attend = _fresh("pred_attend")
            score = _fresh("score")
            lines.append(f"{indent}{kv} = arith.addi %tile0, {t_ssa} : index")
            lines.append(f"{indent}{pred_attend} = arith.cmpi ule, {kv}, %q : index")
            lines.append(f"{indent}{score} = scf.if {pred_attend} -> (f32) {{")
            base = _fresh("base")
            lines.append(f"{indent}  {base} = arith.muli {t_ssa}, %cHD : index")
            if fast_hd64:
                idx0 = _fresh("idx_k0")
                idx1 = _fresh("idx_k1")
                k0 = _fresh("k0")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {idx0} = arith.addi {base}, %lane : index")
                lines.append(f"{indent}  {idx1} = arith.addi {base}, %d1 : index")
                lines.append(f"{indent}  {k0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
                lines.append(f"{indent}  {k1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
            else:
                k0 = _fresh("k0")
                lines.append(f"{indent}  {k0} = scf.if %pred_d0 -> (f32) {{")
                idx0 = _fresh("idx_k0")
                kv0 = _fresh("kv0")
                lines.append(f"{indent}    {idx0} = arith.addi {base}, %lane : index")
                lines.append(f"{indent}    {kv0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
                lines.append(f"{indent}    scf.yield {kv0} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {k1} = scf.if %pred_d1 -> (f32) {{")
                idx1 = _fresh("idx_k1")
                kv1 = _fresh("kv1")
                lines.append(f"{indent}    {idx1} = arith.addi {base}, %d1 : index")
                lines.append(f"{indent}    {kv1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
                lines.append(f"{indent}    scf.yield {kv1} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
            tmp0 = _fresh("tmp0")
            partial = _fresh("partial")
            lines.append(f"{indent}  {tmp0} = llvm.intr.fma({q0}, {k0}, %c0f) : (f32, f32, f32) -> f32")
            lines.append(f"{indent}  {partial} = llvm.intr.fma({q1}, {k1}, {tmp0}) : (f32, f32, f32) -> f32")
            dot = _warp_allreduce_sum_xor(partial, indent=f"{indent}  ")
            scaled = _fresh("scaled")
            lines.append(f"{indent}  {scaled} = arith.mulf {dot}, %sm2{fm} : f32")
            lines.append(f"{indent}  scf.yield {scaled} : f32")
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}  scf.yield %neg_inf : f32")
            lines.append(f"{indent}}}")
            return pred_attend, score

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_x = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c0_i32 = arith.constant 0 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBlockM = arith.constant {int(block_m)} : index")
        lines.append(f"      %cBlockKV = arith.constant {int(block_kv)} : index")
        lines.append(f"      %cGridX = arith.constant {int(grid_x)} : index")
        lines.append(f"      %cTileElems = arith.constant {int(tile_elems)} : index")
        lines.append(f"      %cOffsetV = arith.constant {int(offset_v)} : index")
        lines.append("      %pred_block = arith.cmpi ult, %bid_x, %cGridX : index")
        lines.append("      scf.if %pred_block {")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
        lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")

        # Warp/lane decomposition (block.x == 256 => 8 warps).
        lines.append("        %lane = arith.remui %tid, %c32_idx : index")
        lines.append("        %warp = arith.divui %tid, %c32_idx : index")
        lines.append("        %d1 = arith.addi %lane, %c32_idx : index")
        lines.append("        %is0 = arith.cmpi eq, %lane, %c0 : index")
        if fast_hd64:
            lines.append("        %pred_d0 = arith.constant 1 : i1")
            lines.append("        %pred_d1 = arith.constant 1 : i1")
        else:
            lines.append("        %pred_d0 = arith.cmpi ult, %lane, %cHD : index")
            lines.append("        %pred_d1 = arith.cmpi ult, %d1, %cHD : index")

        # Map CTA -> q_base, and warp -> q within the CTA.
        lines.append("        %q_base = arith.muli %bid_x, %cBlockM : index")
        lines.append("        %q = arith.addi %q_base, %warp : index")
        lines.append("        %pred_q = arith.cmpi ult, %q, %cQ : index")
        lines.append("        %base_q = arith.muli %q, %cHD : index")

        q0 = _fresh("q0")
        q1 = _fresh("q1")
        if fast_hd64:
            lines.append(f"        {q0} = scf.if %pred_q -> (f32) {{")
            idx0 = _fresh("idx_q0")
            v0 = _fresh("q0_load")
            lines.append(f"          {idx0} = arith.addi %base_q, %lane : index")
            lines.append(f"          {v0} = memref.load {arg_ssa[q_name]}[{idx0}] : {q_memref}")
            lines.append(f"          scf.yield {v0} : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")
            lines.append(f"        {q1} = scf.if %pred_q -> (f32) {{")
            idx1 = _fresh("idx_q1")
            v1 = _fresh("q1_load")
            lines.append(f"          {idx1} = arith.addi %base_q, %d1 : index")
            lines.append(f"          {v1} = memref.load {arg_ssa[q_name]}[{idx1}] : {q_memref}")
            lines.append(f"          scf.yield {v1} : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")
        else:
            lines.append(f"        {q0} = scf.if %pred_q -> (f32) {{")
            q0_in = _fresh("q0v")
            lines.append(f"          {q0_in} = scf.if %pred_d0 -> (f32) {{")
            idx0 = _fresh("idx_q0")
            v0 = _fresh("q0_load")
            lines.append(f"            {idx0} = arith.addi %base_q, %lane : index")
            lines.append(f"            {v0} = memref.load {arg_ssa[q_name]}[{idx0}] : {q_memref}")
            lines.append(f"            scf.yield {v0} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          scf.yield {q0_in} : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")
            lines.append(f"        {q1} = scf.if %pred_q -> (f32) {{")
            q1_in = _fresh("q1v")
            lines.append(f"          {q1_in} = scf.if %pred_d1 -> (f32) {{")
            idx1 = _fresh("idx_q1")
            v1 = _fresh("q1_load")
            lines.append(f"            {idx1} = arith.addi %base_q, %d1 : index")
            lines.append(f"            {v1} = memref.load {arg_ssa[q_name]}[{idx1}] : {q_memref}")
            lines.append(f"            scf.yield {v1} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lines.append(f"          scf.yield {q1_in} : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")

        # Skip KV tiles that are always masked by causality for this CTA.
        lines.append("        %q_last0 = arith.addi %q_base, %cBlockM : index")
        lines.append("        %q_last = arith.subi %q_last0, %c1 : index")
        lines.append("        %q_lim = arith.subi %cQ, %c1 : index")
        lines.append("        %p_q_last = arith.cmpi ule, %q_last, %q_lim : index")
        lines.append("        %q_max = arith.select %p_q_last, %q_last, %q_lim : index")
        lines.append("        %kv_end0 = arith.addi %q_max, %c1 : index")
        lines.append("        %p_kv_end = arith.cmpi ule, %kv_end0, %cKV : index")
        lines.append("        %kv_end_max = arith.select %p_kv_end, %kv_end0, %cKV : index")

        m_out = _fresh("m")
        l_out = _fresh("l")
        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        lines.append(
            f"        {m_out}, {l_out}, {acc0_out}, {acc1_out} = scf.for %tile0 = %c0 to %kv_end_max step %cBlockKV "
            "iter_args(%m_i = %neg_inf, %l_i = %c0f, %a0 = %c0f, %a1 = %c0f) -> (f32, f32, f32, f32) {"
        )

        tile_end0 = _fresh("tile_end0")
        tile_end_pred = _fresh("tile_end_pred")
        tile_end = _fresh("tile_end")
        tile_len = _fresh("tile_len")
        tile_elems_dyn = _fresh("tile_elems")
        lines.append(f"          {tile_end0} = arith.addi %tile0, %cBlockKV : index")
        lines.append(f"          {tile_end_pred} = arith.cmpi ule, {tile_end0}, %kv_end_max : index")
        lines.append(f"          {tile_end} = arith.select {tile_end_pred}, {tile_end0}, %kv_end_max : index")
        lines.append(f"          {tile_len} = arith.subi {tile_end}, %tile0 : index")
        lines.append(f"          {tile_elems_dyn} = arith.muli {tile_len}, %cHD : index")

        # Cooperative load K/V tile into shared.
        base_tile = _fresh("base_tile")
        lines.append(f"          {base_tile} = arith.muli %tile0, %cHD : index")
        loads = (int(tile_elems) + int(block_x) - 1) // int(block_x)
        for i in range(int(loads)):
            off = int(i) * int(block_x)
            c_off = _fresh("c_off")
            idx = _fresh("idx")
            pred = _fresh("pred")
            lines.append(f"          {c_off} = arith.constant {int(off)} : index")
            lines.append(f"          {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"          {pred} = arith.cmpi ult, {idx}, {tile_elems_dyn} : index")
            lines.append(f"          scf.if {pred} {{")
            k_val = _fresh("kval")
            v_val = _fresh("vval")
            idx_g = _fresh("idx_g")
            lines.append(f"            {idx_g} = arith.addi {base_tile}, {idx} : index")
            lines.append(f"            {k_val} = memref.load {arg_ssa[k_name]}[{idx_g}] : {k_memref}")
            lines.append(f"            {v_val} = memref.load {arg_ssa[v_name]}[{idx_g}] : {v_memref}")
            idx_vsh = _fresh("idx_vsh")
            lines.append(f"            memref.store {k_val}, %sh[{idx}] : {shared_global_memref_ty}")
            lines.append(f"            {idx_vsh} = arith.addi {idx}, %cOffsetV : index")
            lines.append(f"            memref.store {v_val}, %sh[{idx_vsh}] : {shared_global_memref_ty}")
            lines.append("          }")

        lines.append("          nvvm.barrier0")

        # Update per-query state (skip compute for out-of-range warps, but still
        # participate in barriers).
        m_next = _fresh("m_next")
        l_next = _fresh("l_next")
        a0_next = _fresh("a0_next")
        a1_next = _fresh("a1_next")
        lines.append(f"          {m_next}, {l_next}, {a0_next}, {a1_next} = scf.if %pred_q -> (f32, f32, f32, f32) {{")

        # Online softmax (single-pass) to avoid recomputing dot/score twice.
        m_tile = _fresh("m_tile")
        l_tile = _fresh("l_tile")
        acc0_tile = _fresh("acc0_tile")
        acc1_tile = _fresh("acc1_tile")
        lines.append(
            f"            {m_tile}, {l_tile}, {acc0_tile}, {acc1_tile} = scf.for %t2 = %c0 to {tile_len} step %c1 "
            "iter_args(%m = %m_i, %l = %l_i, %acc0 = %a0, %acc1 = %a1) -> (f32, f32, f32, f32) {"
        )
        kv_ssa = _fresh("kv")
        pred_attend = _fresh("pred_attend")
        lines.append(f"              {kv_ssa} = arith.addi %tile0, %t2 : index")
        lines.append(f"              {pred_attend} = arith.cmpi ule, {kv_ssa}, %q : index")

        m_kv = _fresh("m_kv")
        l_kv = _fresh("l_kv")
        acc0_kv = _fresh("acc0_kv")
        acc1_kv = _fresh("acc1_kv")
        lines.append(f"              {m_kv}, {l_kv}, {acc0_kv}, {acc1_kv} = scf.if {pred_attend} -> (f32, f32, f32, f32) {{")

        base = _fresh("base")
        lines.append(f"                {base} = arith.muli %t2, %cHD : index")
        if fast_hd64:
            idx_k0 = _fresh("idx_k0")
            idx_k1 = _fresh("idx_k1")
            k0 = _fresh("k0")
            k1 = _fresh("k1")
            lines.append(f"                {idx_k0} = arith.addi {base}, %lane : index")
            lines.append(f"                {idx_k1} = arith.addi {base}, %d1 : index")
            lines.append(f"                {k0} = memref.load %sh[{idx_k0}] : {shared_global_memref_ty}")
            lines.append(f"                {k1} = memref.load %sh[{idx_k1}] : {shared_global_memref_ty}")
        else:
            k0 = _fresh("k0")
            lines.append(f"                {k0} = scf.if %pred_d0 -> (f32) {{")
            idx_k0 = _fresh("idx_k0")
            kv0 = _fresh("kv0")
            lines.append(f"                  {idx_k0} = arith.addi {base}, %lane : index")
            lines.append(f"                  {kv0} = memref.load %sh[{idx_k0}] : {shared_global_memref_ty}")
            lines.append(f"                  scf.yield {kv0} : f32")
            lines.append("                } else {")
            lines.append("                  scf.yield %c0f : f32")
            lines.append("                }")
            k1 = _fresh("k1")
            lines.append(f"                {k1} = scf.if %pred_d1 -> (f32) {{")
            idx_k1 = _fresh("idx_k1")
            kv1 = _fresh("kv1")
            lines.append(f"                  {idx_k1} = arith.addi {base}, %d1 : index")
            lines.append(f"                  {kv1} = memref.load %sh[{idx_k1}] : {shared_global_memref_ty}")
            lines.append(f"                  scf.yield {kv1} : f32")
            lines.append("                } else {")
            lines.append("                  scf.yield %c0f : f32")
            lines.append("                }")

        tmp0 = _fresh("tmp0")
        partial = _fresh("partial")
        lines.append(f"                {tmp0} = llvm.intr.fma({q0}, {k0}, %c0f) : (f32, f32, f32) -> f32")
        lines.append(f"                {partial} = llvm.intr.fma({q1}, {k1}, {tmp0}) : (f32, f32, f32) -> f32")
        dot = _warp_allreduce_sum_xor(partial, indent="                ")
        score2 = _fresh("score")
        lines.append(f"                {score2} = arith.mulf {dot}, %sm2{fm} : f32")

        pred_new = _fresh("pred_new")
        m_new = _fresh("m_new")
        alpha = _fresh("alpha")
        p = _fresh("p")
        lines.append(f"                {pred_new} = arith.cmpf ogt, {score2}, %m : f32")
        lines.append(f"                {m_new} = arith.maximumf %m, {score2}{fm} : f32")
        delta = _fresh("delta")
        exp_delta = _fresh("exp_delta")
        lines.append(f"                {delta} = arith.subf {score2}, %m{fm} : f32")
        lines.append(f"                {exp_delta} = math.exp2 {delta} : f32")
        lines.append(f"                {alpha} = scf.if {pred_new} -> (f32) {{")
        inv = _fresh("inv")
        lines.append(f"                  {inv} = arith.divf %c1f, {exp_delta}{fm} : f32")
        lines.append(f"                  scf.yield {inv} : f32")
        lines.append("                } else {")
        lines.append("                  scf.yield %c1f : f32")
        lines.append("                }")
        lines.append(f"                {p} = arith.select {pred_new}, %c1f, {exp_delta} : f32")

        acc0_scaled = _fresh("acc0_scaled")
        acc1_scaled = _fresh("acc1_scaled")
        lines.append(f"                {acc0_scaled} = arith.mulf %acc0, {alpha}{fm} : f32")
        lines.append(f"                {acc1_scaled} = arith.mulf %acc1, {alpha}{fm} : f32")

        base_v = _fresh("base_v")
        lines.append(f"                {base_v} = arith.addi {base}, %cOffsetV : index")
        if fast_hd64:
            idx0 = _fresh("idx_v0")
            idx1 = _fresh("idx_v1")
            vv0 = _fresh("vv0")
            vv1 = _fresh("vv1")
            lines.append(f"                {idx0} = arith.addi {base_v}, %lane : index")
            lines.append(f"                {idx1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"                {vv0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
            lines.append(f"                {vv1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
        else:
            vv0 = _fresh("vv0")
            vv1 = _fresh("vv1")
            lines.append(f"                {vv0} = scf.if %pred_d0 -> (f32) {{")
            idx0 = _fresh("idx_v0")
            v0 = _fresh("v0")
            lines.append(f"                  {idx0} = arith.addi {base_v}, %lane : index")
            lines.append(f"                  {v0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
            lines.append(f"                  scf.yield {v0} : f32")
            lines.append("                } else {")
            lines.append("                  scf.yield %c0f : f32")
            lines.append("                }")
            lines.append(f"                {vv1} = scf.if %pred_d1 -> (f32) {{")
            idx1 = _fresh("idx_v1")
            v1 = _fresh("v1")
            lines.append(f"                  {idx1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"                  {v1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
            lines.append(f"                  scf.yield {v1} : f32")
            lines.append("                } else {")
            lines.append("                  scf.yield %c0f : f32")
            lines.append("                }")

        b0_next = _fresh("b0_next")
        b1_next = _fresh("b1_next")
        l_new = _fresh("l_new")
        lines.append(f"                {b0_next} = llvm.intr.fma({p}, {vv0}, {acc0_scaled}) : (f32, f32, f32) -> f32")
        lines.append(f"                {b1_next} = llvm.intr.fma({p}, {vv1}, {acc1_scaled}) : (f32, f32, f32) -> f32")
        lines.append(f"                {l_new} = llvm.intr.fma(%l, {alpha}, {p}) : (f32, f32, f32) -> f32")
        lines.append(f"                scf.yield {m_new}, {l_new}, {b0_next}, {b1_next} : f32, f32, f32, f32")
        lines.append("              } else {")
        lines.append("                scf.yield %m, %l, %acc0, %acc1 : f32, f32, f32, f32")
        lines.append("              }")

        lines.append(f"              scf.yield {m_kv}, {l_kv}, {acc0_kv}, {acc1_kv} : f32, f32, f32, f32")
        lines.append("            }")

        lines.append(f"            scf.yield {m_tile}, {l_tile}, {acc0_tile}, {acc1_tile} : f32, f32, f32, f32")
        lines.append("          } else {")
        lines.append("            scf.yield %m_i, %l_i, %a0, %a1 : f32, f32, f32, f32")
        lines.append("          }")

        lines.append("          nvvm.barrier0")
        lines.append(f"          scf.yield {m_next}, {l_next}, {a0_next}, {a1_next} : f32, f32, f32, f32")
        lines.append("        }")

        sum_nz = _fresh("sum_nz")
        l_safe = _fresh("l_safe")
        out0 = _fresh("out0")
        out1 = _fresh("out1")
        lines.append(f"        {sum_nz} = arith.cmpf one, {l_out}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_out}, %c1f : f32")
        lines.append(f"        {out0} = arith.divf {acc0_out}, {l_safe}{fm} : f32")
        lines.append(f"        {out1} = arith.divf {acc1_out}, {l_safe}{fm} : f32")
        lines.append("        scf.if %pred_q {")
        if fast_hd64:
            idx_o0 = _fresh("idx_o0")
            idx_o1 = _fresh("idx_o1")
            lines.append(f"          {idx_o0} = arith.addi %base_q, %lane : index")
            lines.append(f"          {idx_o1} = arith.addi %base_q, %d1 : index")
            lines.append(f"          memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
            lines.append(f"          memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
        else:
            lines.append("          scf.if %pred_d0 {")
            idx_o0 = _fresh("idx_o0")
            lines.append(f"            {idx_o0} = arith.addi %base_q, %lane : index")
            lines.append(f"            memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
            lines.append("          }")
            lines.append("          scf.if %pred_d1 {")
            idx_o1 = _fresh("idx_o1")
            lines.append(f"            {idx_o1} = arith.addi %base_q, %d1 : index")
            lines.append(f"            memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
            lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif (
        attn2d_v1 is not None
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", default=False)
        and intent_name == "flash_attention2d"
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_FLASH_ATTN_V2", default=False)
    ):
        # Perf-first: flash_attention2d is a min-ratio offender in the Triton-native
        # perf lane. Avoid redundant scalar softmax math across lanes by computing
        # (m/l/alpha/p) on lane0 and broadcasting alpha/p via warp shuffles.
        kernel_kind = "attn2d_causal_softmax_v3"
        q_name = str(attn2d_v1["Q"])
        k_name = str(attn2d_v1["K"])
        v_name = str(attn2d_v1["V"])
        out_name2 = str(attn2d_v1["out"])
        sm_scale_name = str(attn2d_v1["sm_scale"])
        q_ctx = int(attn2d_v1["Q_CTX"])
        kv_ctx = int(attn2d_v1["KV_CTX"])
        hd = int(attn2d_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        block_x = 32
        req_block_kv = 32
        # Clamp KV tiling for meta/auditing; this implementation iterates KV linearly.
        block_kv = int(min(int(req_block_kv), int(kv_ctx))) if int(kv_ctx) > 0 else int(req_block_kv)
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(q_ctx), 1, 1]}
        cuda_real_mlir_attention_cfg = {
            "block_x": int(block_x),
            "block_kv": int(block_kv),
            "softmax_scalar_lane0": True,
        }
        fast_hd64 = int(hd) == 64

        def _warp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
            cur = str(val_ssa)
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                sel = _fresh("sel")
                nxt = _fresh("sum")
                lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"{indent}{sel} = arith.select {ok}, {sh}, %c0f : f32")
                lines.append(f"{indent}{nxt} = arith.addf {cur}, {sel}{fm} : f32")
                cur = str(nxt)
            return cur

        def _emit_dot_score(*, kv_ssa: str, indent: str) -> tuple[str, str]:
            pred_kv = _fresh("pred_kv")
            pred_causal = _fresh("pred_causal")
            pred_attend = _fresh("pred_attend")
            score = _fresh("score")
            lines.append(f"{indent}{pred_kv} = arith.cmpi ult, {kv_ssa}, %cKV : index")
            lines.append(f"{indent}{pred_causal} = arith.cmpi ule, {kv_ssa}, %bid : index")
            lines.append(f"{indent}{pred_attend} = arith.andi {pred_kv}, {pred_causal} : i1")
            lines.append(f"{indent}{score} = scf.if {pred_attend} -> (f32) {{")
            base_k = _fresh("base_k")
            lines.append(f"{indent}  {base_k} = arith.muli {kv_ssa}, %cHD : index")
            if fast_hd64:
                idx_k0 = _fresh("idx_k0")
                idx_k1 = _fresh("idx_k1")
                k0 = _fresh("k0")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {idx_k0} = arith.addi {base_k}, %tid : index")
                lines.append(f"{indent}  {idx_k1} = arith.addi {base_k}, %d1 : index")
                lines.append(f"{indent}  {k0} = memref.load {arg_ssa[k_name]}[{idx_k0}] : {k_memref}")
                lines.append(f"{indent}  {k1} = memref.load {arg_ssa[k_name]}[{idx_k1}] : {k_memref}")
            else:
                k0 = _fresh("k0")
                lines.append(f"{indent}  {k0} = scf.if %pred_d0 -> (f32) {{")
                idx_k0 = _fresh("idx_k0")
                kv0 = _fresh("kv0")
                lines.append(f"{indent}    {idx_k0} = arith.addi {base_k}, %tid : index")
                lines.append(f"{indent}    {kv0} = memref.load {arg_ssa[k_name]}[{idx_k0}] : {k_memref}")
                lines.append(f"{indent}    scf.yield {kv0} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {k1} = scf.if %pred_d1 -> (f32) {{")
                idx_k1 = _fresh("idx_k1")
                kv1 = _fresh("kv1")
                lines.append(f"{indent}    {idx_k1} = arith.addi {base_k}, %d1 : index")
                lines.append(f"{indent}    {kv1} = memref.load {arg_ssa[k_name]}[{idx_k1}] : {k_memref}")
                lines.append(f"{indent}    scf.yield {kv1} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
            partial = _fresh("partial")
            tmp0 = _fresh("tmp0")
            lines.append(f"{indent}  {tmp0} = llvm.intr.fma(%q0, {k0}, %c0f) : (f32, f32, f32) -> f32")
            lines.append(f"{indent}  {partial} = llvm.intr.fma(%q1, {k1}, {tmp0}) : (f32, f32, f32) -> f32")
            dot = _warp_allreduce_sum_xor(partial, indent=f"{indent}  ")
            scaled = _fresh("scaled")
            lines.append(f"{indent}  {scaled} = arith.mulf {dot}, %sm2{fm} : f32")
            lines.append(f"{indent}  scf.yield {scaled} : f32")
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}  scf.yield %neg_inf : f32")
            lines.append(f"{indent}}}")
            return pred_attend, score

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %c0_i32 = arith.constant 0 : i32")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cQ : index")
        lines.append("      scf.if %pred_row {")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
        lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")
        lines.append("        %base_q = arith.muli %bid, %cHD : index")
        lines.append("        %d1 = arith.addi %tid, %c32_idx : index")
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        if fast_hd64:
            lines.append("        %pred_d0 = arith.constant 1 : i1")
            lines.append("        %pred_d1 = arith.constant 1 : i1")
            lines.append("        %idx_q0 = arith.addi %base_q, %tid : index")
            lines.append("        %idx_q1 = arith.addi %base_q, %d1 : index")
            lines.append(f"        %q0 = memref.load {arg_ssa[q_name]}[%idx_q0] : {q_memref}")
            lines.append(f"        %q1 = memref.load {arg_ssa[q_name]}[%idx_q1] : {q_memref}")
        else:
            lines.append("        %pred_d0 = arith.cmpi ult, %tid, %cHD : index")
            lines.append("        %pred_d1 = arith.cmpi ult, %d1, %cHD : index")
            lines.append("        %q0 = scf.if %pred_d0 -> (f32) {")
            lines.append("          %idx_q0 = arith.addi %base_q, %tid : index")
            lines.append(f"          %q0v = memref.load {arg_ssa[q_name]}[%idx_q0] : {q_memref}")
            lines.append("          scf.yield %q0v : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")
            lines.append("        %q1 = scf.if %pred_d1 -> (f32) {")
            lines.append("          %idx_q1 = arith.addi %base_q, %d1 : index")
            lines.append(f"          %q1v = memref.load {arg_ssa[q_name]}[%idx_q1] : {q_memref}")
            lines.append("          scf.yield %q1v : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")

        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        m_out = _fresh("m")
        l_out = _fresh("l")
        lines.append(
            f"        {acc0_out}, {acc1_out}, {m_out}, {l_out} = scf.for %kv = %c0 to %cKV step %c1 "
            "iter_args(%a0 = %c0f, %a1 = %c0f, %m = %neg_inf, %l = %c0f) -> (f32, f32, f32, f32) {"
        )
        pred_attend, score = _emit_dot_score(kv_ssa="%kv", indent="          ")

        m_lane0 = _fresh("m_lane0")
        alpha_lane0 = _fresh("alpha_lane0")
        p_lane0 = _fresh("p_lane0")
        l_lane0 = _fresh("l_lane0")
        lines.append(
            f"          {m_lane0}, {alpha_lane0}, {p_lane0}, {l_lane0} = scf.if %is0 -> (f32, f32, f32, f32) {{"
        )
        m_new0 = _fresh("m_new")
        delta_m = _fresh("delta_m")
        alpha0 = _fresh("alpha")
        delta_s = _fresh("delta_s")
        p0 = _fresh("p")
        l_scaled0 = _fresh("l_scaled")
        l_new0 = _fresh("l_new")
        lines.append(f"            {m_new0} = arith.maximumf %m, {score}{fm} : f32")
        lines.append(f"            {delta_m} = arith.subf %m, {m_new0}{fm} : f32")
        lines.append(f"            {alpha0} = math.exp2 {delta_m} : f32")
        lines.append(f"            {delta_s} = arith.subf {score}, {m_new0}{fm} : f32")
        lines.append(f"            {p0} = math.exp2 {delta_s} : f32")
        lines.append(f"            {l_scaled0} = arith.mulf %l, {alpha0}{fm} : f32")
        lines.append(f"            {l_new0} = arith.addf {l_scaled0}, {p0}{fm} : f32")
        lines.append(f"            scf.yield {m_new0}, {alpha0}, {p0}, {l_new0} : f32, f32, f32, f32")
        lines.append("          } else {")
        lines.append("            scf.yield %m, %c0f, %c0f, %l : f32, f32, f32, f32")
        lines.append("          }")

        alpha_sh = _fresh("alpha_sh")
        alpha_ok = _fresh("alpha_ok")
        alpha = _fresh("alpha")
        lines.append(f"          {alpha_sh}, {alpha_ok} = gpu.shuffle idx {alpha_lane0}, %c0_i32, %c32_i32 : f32")
        lines.append(f"          {alpha} = arith.select {alpha_ok}, {alpha_sh}, %c1f : f32")
        p_sh = _fresh("p_sh")
        p_ok = _fresh("p_ok")
        p = _fresh("p")
        lines.append(f"          {p_sh}, {p_ok} = gpu.shuffle idx {p_lane0}, %c0_i32, %c32_i32 : f32")
        lines.append(f"          {p} = arith.select {p_ok}, {p_sh}, %c0f : f32")

        a0_scaled = _fresh("a0_scaled")
        a1_scaled = _fresh("a1_scaled")
        lines.append(f"          {a0_scaled} = arith.mulf %a0, {alpha}{fm} : f32")
        lines.append(f"          {a1_scaled} = arith.mulf %a1, {alpha}{fm} : f32")

        pred_v0 = _fresh("pred_v0")
        pred_v1 = _fresh("pred_v1")
        lines.append(f"          {pred_v0} = arith.andi {pred_attend}, %pred_d0 : i1")
        lines.append(f"          {pred_v1} = arith.andi {pred_attend}, %pred_d1 : i1")
        base_v = _fresh("base_v")
        lines.append(f"          {base_v} = arith.muli %kv, %cHD : index")

        if fast_hd64:
            vv0 = _fresh("vv0")
            vv1 = _fresh("vv1")
            lines.append(f"          {vv0}, {vv1} = scf.if {pred_attend} -> (f32, f32) {{")
            idx_v0 = _fresh("idx_v0")
            idx_v1 = _fresh("idx_v1")
            v0 = _fresh("v0")
            v1 = _fresh("v1")
            lines.append(f"            {idx_v0} = arith.addi {base_v}, %tid : index")
            lines.append(f"            {idx_v1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"            {v0} = memref.load {arg_ssa[v_name]}[{idx_v0}] : {v_memref}")
            lines.append(f"            {v1} = memref.load {arg_ssa[v_name]}[{idx_v1}] : {v_memref}")
            lines.append(f"            scf.yield {v0}, {v1} : f32, f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f, %c0f : f32, f32")
            lines.append("          }")
        else:
            vv0 = _fresh("vv0")
            lines.append(f"          {vv0} = scf.if {pred_v0} -> (f32) {{")
            idx_v0 = _fresh("idx_v0")
            v0 = _fresh("v0")
            lines.append(f"            {idx_v0} = arith.addi {base_v}, %tid : index")
            lines.append(f"            {v0} = memref.load {arg_ssa[v_name]}[{idx_v0}] : {v_memref}")
            lines.append(f"            scf.yield {v0} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            vv1 = _fresh("vv1")
            lines.append(f"          {vv1} = scf.if {pred_v1} -> (f32) {{")
            idx_v1 = _fresh("idx_v1")
            v1 = _fresh("v1")
            lines.append(f"            {idx_v1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"            {v1} = memref.load {arg_ssa[v_name]}[{idx_v1}] : {v_memref}")
            lines.append(f"            scf.yield {v1} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")

        b0_next = _fresh("b0_next")
        b1_next = _fresh("b1_next")
        lines.append(f"          {b0_next} = llvm.intr.fma({p}, {vv0}, {a0_scaled}) : (f32, f32, f32) -> f32")
        lines.append(f"          {b1_next} = llvm.intr.fma({p}, {vv1}, {a1_scaled}) : (f32, f32, f32) -> f32")
        lines.append(f"          scf.yield {b0_next}, {b1_next}, {m_lane0}, {l_lane0} : f32, f32, f32, f32")
        lines.append("        }")

        l_sh = _fresh("l_sh")
        l_ok = _fresh("l_ok")
        l_all = _fresh("l_all")
        lines.append(f"        {l_sh}, {l_ok} = gpu.shuffle idx {l_out}, %c0_i32, %c32_i32 : f32")
        lines.append(f"        {l_all} = arith.select {l_ok}, {l_sh}, %c1f : f32")

        sum_nz = _fresh("sum_nz")
        l_safe = _fresh("l_safe")
        out0 = _fresh("out0")
        out1 = _fresh("out1")
        lines.append(f"        {sum_nz} = arith.cmpf one, {l_all}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_all}, %c1f : f32")
        lines.append(f"        {out0} = arith.divf {acc0_out}, {l_safe}{fm} : f32")
        lines.append(f"        {out1} = arith.divf {acc1_out}, {l_safe}{fm} : f32")
        lines.append("        scf.if %pred_d0 {")
        idx_o0 = _fresh("idx_o0")
        lines.append(f"          {idx_o0} = arith.addi %base_q, %tid : index")
        lines.append(f"          memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
        lines.append("        }")
        lines.append("        scf.if %pred_d1 {")
        idx_o1 = _fresh("idx_o1")
        lines.append(f"          {idx_o1} = arith.addi %base_q, %d1 : index")
        lines.append(f"          memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif sdpa_bhsd_v1 is not None:
        kernel_kind = "sdpa_bhsd_v2"
        q_name = str(sdpa_bhsd_v1["query"])
        k_name = str(sdpa_bhsd_v1["key"])
        v_name = str(sdpa_bhsd_v1["value"])
        out_name2 = str(sdpa_bhsd_v1["out"])
        b_dim = int(sdpa_bhsd_v1["B"])
        h_dim = int(sdpa_bhsd_v1["H"])
        q_len = int(sdpa_bhsd_v1["Q"])
        k_len = int(sdpa_bhsd_v1["K"])
        d_dim = int(sdpa_bhsd_v1["D"])
        is_causal = bool(sdpa_bhsd_v1.get("is_causal"))
        if b_dim <= 0 or h_dim <= 0 or q_len <= 0 or k_len <= 0 or d_dim <= 0:
            raise RuntimeError(
                f"sdpa_bhsd_v2 expects positive dims, got B={b_dim} H={h_dim} Q={q_len} K={k_len} D={d_dim}"
            )
        if int(d_dim) > 256:
            raise RuntimeError(f"sdpa_bhsd_v2 currently requires D<=256, got D={d_dim}")

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        blocks = int(int(b_dim) * int(h_dim) * int(q_len))
        launch_override = {"block": [32, 1, 1], "grid": [int(blocks), 1, 1]}

        scale_const = _as_f32_const(1.0 / math.sqrt(float(d_dim)))

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cBlocks = arith.constant {int(blocks)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cQ = arith.constant {int(q_len)} : index")
        lines.append(f"      %cK = arith.constant {int(k_len)} : index")
        lines.append(f"      %cD = arith.constant {int(d_dim)} : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append(f"      %scale = arith.constant {scale_const} : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %pred = arith.cmpi ult, %bid, %cBlocks : index")
        lines.append("      scf.if %pred {")

        # Decode block id -> (b,h,q). Flattened row-major: (b*H + h)*Q + q.
        lines.append("        %bh = arith.divui %bid, %cQ : index")
        lines.append("        %q = arith.remui %bid, %cQ : index")
        lines.append("        %b = arith.divui %bh, %cH : index")
        lines.append("        %h = arith.remui %bh, %cH : index")
        lines.append("        %bh0 = arith.muli %b, %cH : index")
        lines.append("        %bh2 = arith.addi %bh0, %h : index")
        lines.append("        %q_base0 = arith.muli %bh2, %cQ : index")
        lines.append("        %q_base = arith.addi %q_base0, %q : index")
        lines.append("        %q_off = arith.muli %q_base, %cD : index")
        lines.append("        %k_base0 = arith.muli %bh2, %cK : index")

        lines.append("        %pred_d = arith.cmpi ult, %tid, %cD : index")
        lines.append("        %q_lane = scf.if %pred_d -> (f32) {")
        lines.append("          %q_idx = arith.addi %q_off, %tid : index")
        lines.append(f"          %qv = memref.load {arg_ssa[q_name]}[%q_idx] : {q_memref}")
        lines.append("          scf.yield %qv : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")

        acc_out = _fresh("acc_sdpa")
        m_out = _fresh("m_sdpa")
        l_out = _fresh("l_sdpa")
        lines.append(
            f"        {acc_out}, {m_out}, {l_out} = scf.for %kk = %c0 to %cK step %c1 "
            "iter_args(%acc = %c0f, %m = %neg_inf, %l = %c0f) -> (f32, f32, f32) {"
        )
        lines.append("          %k_base = arith.addi %k_base0, %kk : index")
        lines.append("          %k_off = arith.muli %k_base, %cD : index")
        lines.append("          %partial = scf.if %pred_d -> (f32) {")
        lines.append("            %k_idx = arith.addi %k_off, %tid : index")
        lines.append(f"            %kv = memref.load {arg_ssa[k_name]}[%k_idx] : {k_memref}")
        lines.append(f"            %prod = arith.mulf %q_lane, %kv{fm} : f32")
        lines.append("            scf.yield %prod : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")

        cur = "%partial"
        for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
            sh = _fresh("sh")
            ok = _fresh("ok")
            nxt = _fresh("sum")
            lines.append(f"          {sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
            lines.append(f"          {nxt} = arith.addf {cur}, {sh}{fm} : f32")
            cur = str(nxt)

        lines.append(f"          %score0 = arith.mulf {cur}, %scale{fm} : f32")
        score_ssa = "%score0"
        if is_causal:
            lines.append("          %mask = arith.cmpi ugt, %kk, %q : index")
            lines.append("          %score = arith.select %mask, %neg_inf, %score0 : f32")
            score_ssa = "%score"

        lines.append(f"          %m_new = arith.maximumf %m, {score_ssa}{fm} : f32")
        lines.append(f"          %delta = arith.subf %m, %m_new{fm} : f32")
        lines.append(f"          %alpha = math.exp %delta{fm} : f32")
        lines.append(f"          %l_scaled = arith.mulf %l, %alpha{fm} : f32")
        lines.append(f"          %s_delta = arith.subf {score_ssa}, %m_new{fm} : f32")
        lines.append(f"          %p = math.exp %s_delta{fm} : f32")
        lines.append(f"          %l_next = arith.addf %l_scaled, %p{fm} : f32")
        lines.append(f"          %acc_scaled = arith.mulf %acc, %alpha{fm} : f32")
        lines.append("          %v_lane = scf.if %pred_d -> (f32) {")
        lines.append("            %v_idx = arith.addi %k_off, %tid : index")
        lines.append(f"            %vv = memref.load {arg_ssa[v_name]}[%v_idx] : {v_memref}")
        lines.append("            scf.yield %vv : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        lines.append(f"          %pv = arith.mulf %p, %v_lane{fm} : f32")
        lines.append(f"          %acc_next = arith.addf %acc_scaled, %pv{fm} : f32")
        lines.append(f"          scf.yield %acc_next, %m_new, %l_next : f32, f32, f32")
        lines.append("        }")

        l_ok = _fresh("l_ok")
        l_safe = _fresh("l_safe")
        inv = _fresh("inv")
        outv = _fresh("outv")
        lines.append(f"        {l_ok} = arith.cmpf one, {l_out}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {l_ok}, {l_out}, %c1f : f32")
        lines.append(f"        {inv} = arith.divf %c1f, {l_safe}{fm} : f32")
        lines.append(f"        {outv} = arith.mulf {acc_out}, {inv}{fm} : f32")
        lines.append("        scf.if %pred_d {")
        lines.append("          %out_idx = arith.addi %q_off, %tid : index")
        lines.append(f"          memref.store {outv}, {arg_ssa[out_name2]}[%out_idx] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif conv1d_ncl_v1 is not None:
        kernel_kind = "conv1d_ncl_v1"
        in_name = str(conv1d_ncl_v1["input"])
        w_name = str(conv1d_ncl_v1["weight"])
        b_name = str(conv1d_ncl_v1["bias"])
        out_name2 = str(conv1d_ncl_v1["out"])
        n_dim = int(conv1d_ncl_v1["N"])
        c_in = int(conv1d_ncl_v1["C_IN"])
        c_out = int(conv1d_ncl_v1["C_OUT"])
        l_dim = int(conv1d_ncl_v1["L"])
        k_dim = int(conv1d_ncl_v1["K"])
        stride = int(conv1d_ncl_v1["STRIDE"])
        padding = int(conv1d_ncl_v1["PADDING"])
        dilation = int(conv1d_ncl_v1["DILATION"])
        groups = int(conv1d_ncl_v1["GROUPS"])
        c_per_g = int(conv1d_ncl_v1["C_PER_G"])
        ol_dim = int(conv1d_ncl_v1["OL"])
        if (
            n_dim <= 0
            or c_in <= 0
            or c_out <= 0
            or l_dim <= 0
            or k_dim <= 0
            or ol_dim <= 0
            or stride <= 0
            or dilation <= 0
            or padding < 0
            or groups <= 0
        ):
            raise RuntimeError(
                "conv1d_ncl_v1 expects positive dims/stride/dilation/groups, got "
                f"N={n_dim} C_IN={c_in} C_OUT={c_out} L={l_dim} K={k_dim} OL={ol_dim} "
                f"stride={stride} padding={padding} dilation={dilation} groups={groups}"
            )
        if c_in % groups != 0 or c_out % groups != 0:
            raise RuntimeError(f"conv1d_ncl_v1 expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
        if c_per_g != (c_in // groups):
            raise RuntimeError(f"conv1d_ncl_v1 expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
        oc_per_g = int(c_out // groups)

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cOL = arith.constant {int(ol_dim)} : index")
        lines.append(f"        %cC_OUT = arith.constant {int(c_out)} : index")
        lines.append(f"        %cL = arith.constant {int(l_dim)} : index")
        lines.append(f"        %cC_IN = arith.constant {int(c_in)} : index")
        lines.append(f"        %cC_PER_G = arith.constant {int(c_per_g)} : index")
        lines.append(f"        %cK = arith.constant {int(k_dim)} : index")
        lines.append(f"        %cStride = arith.constant {int(stride)} : index")
        lines.append(f"        %cPad = arith.constant {int(padding)} : index")
        lines.append(f"        %cDil = arith.constant {int(dilation)} : index")
        lines.append(f"        %cOCPerG = arith.constant {int(oc_per_g)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %ol = arith.remui %lin, %cOL : index")
        lines.append("        %t1 = arith.divui %lin, %cOL : index")
        lines.append("        %oc = arith.remui %t1, %cC_OUT : index")
        lines.append("        %nn = arith.divui %t1, %cC_OUT : index")
        lines.append("        %g = arith.divui %oc, %cOCPerG : index")
        lines.append("        %in_c_base = arith.muli %g, %cC_PER_G : index")
        lines.append("        %nc0 = arith.muli %nn, %cC_IN : index")
        lines.append("        %oc_base = arith.muli %oc, %cC_PER_G : index")
        lines.append("        %ol_stride = arith.muli %ol, %cStride : index")
        lines.append(f"        %bias_v = memref.load {arg_ssa[b_name]}[%oc] : {b_memref}")
        lines.append("        %acc = scf.for %ic = %c0 to %cC_PER_G step %c1 iter_args(%a0 = %bias_v) -> (f32) {")
        lines.append("          %in_c = arith.addi %in_c_base, %ic : index")
        lines.append("          %nci = arith.addi %nc0, %in_c : index")
        lines.append("          %base_in = arith.muli %nci, %cL : index")
        lines.append("          %oc_ic = arith.addi %oc_base, %ic : index")
        lines.append("          %w_base = arith.muli %oc_ic, %cK : index")
        lines.append("          %acc2 = scf.for %kk = %c0 to %cK step %c1 iter_args(%a1 = %a0) -> (f32) {")
        lines.append("            %kk_dil = arith.muli %kk, %cDil : index")
        lines.append("            %tmp_l = arith.addi %ol_stride, %kk_dil : index")
        lines.append("            %in_l = arith.subi %tmp_l, %cPad : index")
        lines.append("            %ge0 = arith.cmpi sge, %in_l, %c0 : index")
        lines.append("            %ltL = arith.cmpi slt, %in_l, %cL : index")
        lines.append("            %in_ok = arith.andi %ge0, %ltL : i1")
        lines.append("            %x = scf.if %in_ok -> (f32) {")
        lines.append("              %in_idx = arith.addi %base_in, %in_l : index")
        lines.append(f"              %v = memref.load {arg_ssa[in_name]}[%in_idx] : {in_memref}")
        lines.append("              scf.yield %v : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        lines.append("            %w_idx = arith.addi %w_base, %kk : index")
        lines.append(f"            %w = memref.load {arg_ssa[w_name]}[%w_idx] : {w_memref}")
        lines.append(f"            %prod = arith.mulf %x, %w{fm} : f32")
        lines.append(f"            %a_next = arith.addf %a1, %prod{fm} : f32")
        lines.append("            scf.yield %a_next : f32")
        lines.append("          }")
        lines.append("          scf.yield %acc2 : f32")
        lines.append("        }")
        lines.append(f"        memref.store %acc, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif conv2d_nchw_v1 is not None:
        kernel_kind = "conv2d_nchw_v1"
        in_name = str(conv2d_nchw_v1["input"])
        w_name = str(conv2d_nchw_v1["weight"])
        b_name = str(conv2d_nchw_v1["bias"])
        out_name2 = str(conv2d_nchw_v1["out"])
        n_dim = int(conv2d_nchw_v1["N"])
        c_in = int(conv2d_nchw_v1["C_IN"])
        c_out = int(conv2d_nchw_v1["C_OUT"])
        h_dim = int(conv2d_nchw_v1["H"])
        w_dim = int(conv2d_nchw_v1["W"])
        kh = int(conv2d_nchw_v1["KH"])
        kw = int(conv2d_nchw_v1["KW"])
        sh = int(conv2d_nchw_v1["SH"])
        sw = int(conv2d_nchw_v1["SW"])
        ph = int(conv2d_nchw_v1["PH"])
        pw = int(conv2d_nchw_v1["PW"])
        dh = int(conv2d_nchw_v1["DH"])
        dw = int(conv2d_nchw_v1["DW"])
        groups = int(conv2d_nchw_v1["GROUPS"])
        c_per_g = int(conv2d_nchw_v1["C_PER_G"])
        oh_dim = int(conv2d_nchw_v1["OH"])
        ow_dim = int(conv2d_nchw_v1["OW"])
        if (
            n_dim <= 0
            or c_in <= 0
            or c_out <= 0
            or h_dim <= 0
            or w_dim <= 0
            or kh <= 0
            or kw <= 0
            or oh_dim <= 0
            or ow_dim <= 0
            or sh <= 0
            or sw <= 0
            or dh <= 0
            or dw <= 0
            or ph < 0
            or pw < 0
            or groups <= 0
        ):
            raise RuntimeError(
                "conv2d_nchw_v1 expects positive dims/stride/dilation/groups, got "
                f"N={n_dim} C_IN={c_in} C_OUT={c_out} H={h_dim} W={w_dim} KH={kh} KW={kw} "
                f"OH={oh_dim} OW={ow_dim} sh={sh} sw={sw} ph={ph} pw={pw} dh={dh} dw={dw} groups={groups}"
            )
        if c_in % groups != 0 or c_out % groups != 0:
            raise RuntimeError(f"conv2d_nchw_v1 expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
        if c_per_g != (c_in // groups):
            raise RuntimeError(f"conv2d_nchw_v1 expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
        oc_per_g = int(c_out // groups)

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        hw = int(h_dim * w_dim)
        khkw = int(kh * kw)

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC_OUT = arith.constant {int(c_out)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cC_IN = arith.constant {int(c_in)} : index")
        lines.append(f"        %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"        %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"        %cHW = arith.constant {int(hw)} : index")
        lines.append(f"        %cKH = arith.constant {int(kh)} : index")
        lines.append(f"        %cKW = arith.constant {int(kw)} : index")
        lines.append(f"        %cKHKW = arith.constant {int(khkw)} : index")
        lines.append(f"        %cSH = arith.constant {int(sh)} : index")
        lines.append(f"        %cSW = arith.constant {int(sw)} : index")
        lines.append(f"        %cPH = arith.constant {int(ph)} : index")
        lines.append(f"        %cPW = arith.constant {int(pw)} : index")
        lines.append(f"        %cDH = arith.constant {int(dh)} : index")
        lines.append(f"        %cDW = arith.constant {int(dw)} : index")
        lines.append(f"        %cC_PER_G = arith.constant {int(c_per_g)} : index")
        lines.append(f"        %cOCPerG = arith.constant {int(oc_per_g)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %oc = arith.remui %t2, %cC_OUT : index")
        lines.append("        %nn = arith.divui %t2, %cC_OUT : index")
        lines.append("        %g = arith.divui %oc, %cOCPerG : index")
        lines.append("        %in_c_base = arith.muli %g, %cC_PER_G : index")
        lines.append("        %nc0 = arith.muli %nn, %cC_IN : index")
        lines.append("        %oc_base = arith.muli %oc, %cC_PER_G : index")
        lines.append("        %oh_sh = arith.muli %oh, %cSH : index")
        lines.append("        %ow_sw = arith.muli %ow, %cSW : index")
        lines.append(f"        %bias_v = memref.load {arg_ssa[b_name]}[%oc] : {b_memref}")
        lines.append("        %acc = scf.for %ic = %c0 to %cC_PER_G step %c1 iter_args(%a0 = %bias_v) -> (f32) {")
        lines.append("          %in_c = arith.addi %in_c_base, %ic : index")
        lines.append("          %nci = arith.addi %nc0, %in_c : index")
        lines.append("          %base_in = arith.muli %nci, %cHW : index")
        lines.append("          %oc_ic = arith.addi %oc_base, %ic : index")
        lines.append("          %w_base_ic = arith.muli %oc_ic, %cKHKW : index")
        lines.append("          %acc_kh = scf.for %kh_i = %c0 to %cKH step %c1 iter_args(%a1 = %a0) -> (f32) {")
        lines.append("            %kh_dh = arith.muli %kh_i, %cDH : index")
        lines.append("            %tmp_h = arith.addi %oh_sh, %kh_dh : index")
        lines.append("            %in_h = arith.subi %tmp_h, %cPH : index")
        lines.append("            %h_ge0 = arith.cmpi sge, %in_h, %c0 : index")
        lines.append("            %h_lt = arith.cmpi slt, %in_h, %cH : index")
        lines.append("            %h_ok = arith.andi %h_ge0, %h_lt : i1")
        lines.append("            %w_kh = arith.muli %kh_i, %cKW : index")
        lines.append("            %acc_kw = scf.for %kw_i = %c0 to %cKW step %c1 iter_args(%a2 = %a1) -> (f32) {")
        lines.append("              %kw_dw = arith.muli %kw_i, %cDW : index")
        lines.append("              %tmp_w = arith.addi %ow_sw, %kw_dw : index")
        lines.append("              %in_w = arith.subi %tmp_w, %cPW : index")
        lines.append("              %w_ge0 = arith.cmpi sge, %in_w, %c0 : index")
        lines.append("              %w_lt = arith.cmpi slt, %in_w, %cW : index")
        lines.append("              %w_ok = arith.andi %w_ge0, %w_lt : i1")
        lines.append("              %in_ok = arith.andi %h_ok, %w_ok : i1")
        lines.append("              %x = scf.if %in_ok -> (f32) {")
        lines.append("                %h_mul = arith.muli %in_h, %cW : index")
        lines.append("                %hw_off = arith.addi %h_mul, %in_w : index")
        lines.append("                %in_idx = arith.addi %base_in, %hw_off : index")
        lines.append(f"                %v = memref.load {arg_ssa[in_name]}[%in_idx] : {in_memref}")
        lines.append("                scf.yield %v : f32")
        lines.append("              } else {")
        lines.append("                scf.yield %c0f : f32")
        lines.append("              }")
        lines.append("              %w_off = arith.addi %w_kh, %kw_i : index")
        lines.append("              %w_idx = arith.addi %w_base_ic, %w_off : index")
        lines.append(f"              %w = memref.load {arg_ssa[w_name]}[%w_idx] : {w_memref}")
        lines.append(f"              %prod = arith.mulf %x, %w{fm} : f32")
        lines.append(f"              %a_next = arith.addf %a2, %prod{fm} : f32")
        lines.append("              scf.yield %a_next : f32")
        lines.append("            }")
        lines.append("            scf.yield %acc_kw : f32")
        lines.append("          }")
        lines.append("          scf.yield %acc_kh : f32")
        lines.append("        }")
        lines.append(f"        memref.store %acc, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif conv3d_ncdhw_v1 is not None:
        kernel_kind = "conv3d_ncdhw_v1"
        in_name = str(conv3d_ncdhw_v1["input"])
        w_name = str(conv3d_ncdhw_v1["weight"])
        b_name = str(conv3d_ncdhw_v1["bias"])
        out_name2 = str(conv3d_ncdhw_v1["out"])
        n_dim = int(conv3d_ncdhw_v1["N"])
        c_in = int(conv3d_ncdhw_v1["C_IN"])
        c_out = int(conv3d_ncdhw_v1["C_OUT"])
        d_dim = int(conv3d_ncdhw_v1["D"])
        h_dim = int(conv3d_ncdhw_v1["H"])
        w_dim = int(conv3d_ncdhw_v1["W"])
        kd = int(conv3d_ncdhw_v1["KD"])
        kh = int(conv3d_ncdhw_v1["KH"])
        kw = int(conv3d_ncdhw_v1["KW"])
        sd = int(conv3d_ncdhw_v1["SD"])
        sh = int(conv3d_ncdhw_v1["SH"])
        sw = int(conv3d_ncdhw_v1["SW"])
        pd = int(conv3d_ncdhw_v1["PD"])
        ph = int(conv3d_ncdhw_v1["PH"])
        pw = int(conv3d_ncdhw_v1["PW"])
        dd = int(conv3d_ncdhw_v1["DD"])
        dh = int(conv3d_ncdhw_v1["DH"])
        dw = int(conv3d_ncdhw_v1["DW"])
        groups = int(conv3d_ncdhw_v1["GROUPS"])
        c_per_g = int(conv3d_ncdhw_v1["C_PER_G"])
        od_dim = int(conv3d_ncdhw_v1["OD"])
        oh_dim = int(conv3d_ncdhw_v1["OH"])
        ow_dim = int(conv3d_ncdhw_v1["OW"])
        if (
            n_dim <= 0
            or c_in <= 0
            or c_out <= 0
            or d_dim <= 0
            or h_dim <= 0
            or w_dim <= 0
            or kd <= 0
            or kh <= 0
            or kw <= 0
            or od_dim <= 0
            or oh_dim <= 0
            or ow_dim <= 0
            or sd <= 0
            or sh <= 0
            or sw <= 0
            or dd <= 0
            or dh <= 0
            or dw <= 0
            or pd < 0
            or ph < 0
            or pw < 0
            or groups <= 0
        ):
            raise RuntimeError(
                "conv3d_ncdhw_v1 expects positive dims/stride/dilation/groups, got "
                f"N={n_dim} C_IN={c_in} C_OUT={c_out} D={d_dim} H={h_dim} W={w_dim} "
                f"KD={kd} KH={kh} KW={kw} OD={od_dim} OH={oh_dim} OW={ow_dim} "
                f"sd={sd} sh={sh} sw={sw} pd={pd} ph={ph} pw={pw} dd={dd} dh={dh} dw={dw} groups={groups}"
            )
        if c_in % groups != 0 or c_out % groups != 0:
            raise RuntimeError(f"conv3d_ncdhw_v1 expects channels divisible by groups, got C_IN={c_in} C_OUT={c_out} groups={groups}")
        if c_per_g != (c_in // groups):
            raise RuntimeError(f"conv3d_ncdhw_v1 expects C_PER_G=C_IN/groups, got C_PER_G={c_per_g} C_IN={c_in} groups={groups}")
        oc_per_g = int(c_out // groups)

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        hw = int(h_dim * w_dim)
        dhw = int(d_dim * h_dim * w_dim)
        khkw = int(kh * kw)
        kdhw = int(kd * kh * kw)

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC_OUT = arith.constant {int(c_out)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cOD = arith.constant {int(od_dim)} : index")
        lines.append(f"        %cC_IN = arith.constant {int(c_in)} : index")
        lines.append(f"        %cD = arith.constant {int(d_dim)} : index")
        lines.append(f"        %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"        %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"        %cHW = arith.constant {int(hw)} : index")
        lines.append(f"        %cDHW = arith.constant {int(dhw)} : index")
        lines.append(f"        %cKD = arith.constant {int(kd)} : index")
        lines.append(f"        %cKH = arith.constant {int(kh)} : index")
        lines.append(f"        %cKW = arith.constant {int(kw)} : index")
        lines.append(f"        %cKHKW = arith.constant {int(khkw)} : index")
        lines.append(f"        %cKDKHKW = arith.constant {int(kdhw)} : index")
        lines.append(f"        %cSD = arith.constant {int(sd)} : index")
        lines.append(f"        %cSH = arith.constant {int(sh)} : index")
        lines.append(f"        %cSW = arith.constant {int(sw)} : index")
        lines.append(f"        %cPD = arith.constant {int(pd)} : index")
        lines.append(f"        %cPH = arith.constant {int(ph)} : index")
        lines.append(f"        %cPW = arith.constant {int(pw)} : index")
        lines.append(f"        %cDD = arith.constant {int(dd)} : index")
        lines.append(f"        %cDH = arith.constant {int(dh)} : index")
        lines.append(f"        %cDW = arith.constant {int(dw)} : index")
        lines.append(f"        %cC_PER_G = arith.constant {int(c_per_g)} : index")
        lines.append(f"        %cOCPerG = arith.constant {int(oc_per_g)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %od = arith.remui %t2, %cOD : index")
        lines.append("        %t3 = arith.divui %t2, %cOD : index")
        lines.append("        %oc = arith.remui %t3, %cC_OUT : index")
        lines.append("        %nn = arith.divui %t3, %cC_OUT : index")
        lines.append("        %g = arith.divui %oc, %cOCPerG : index")
        lines.append("        %in_c_base = arith.muli %g, %cC_PER_G : index")
        lines.append("        %nc0 = arith.muli %nn, %cC_IN : index")
        lines.append("        %oc_base = arith.muli %oc, %cC_PER_G : index")
        lines.append("        %od_sd = arith.muli %od, %cSD : index")
        lines.append("        %oh_sh = arith.muli %oh, %cSH : index")
        lines.append("        %ow_sw = arith.muli %ow, %cSW : index")
        lines.append(f"        %bias_v = memref.load {arg_ssa[b_name]}[%oc] : {b_memref}")
        lines.append("        %acc = scf.for %ic = %c0 to %cC_PER_G step %c1 iter_args(%a0 = %bias_v) -> (f32) {")
        lines.append("          %in_c = arith.addi %in_c_base, %ic : index")
        lines.append("          %nci = arith.addi %nc0, %in_c : index")
        lines.append("          %base_in = arith.muli %nci, %cDHW : index")
        lines.append("          %oc_ic = arith.addi %oc_base, %ic : index")
        lines.append("          %w_base_ic = arith.muli %oc_ic, %cKDKHKW : index")
        lines.append("          %acc_kd = scf.for %kd_i = %c0 to %cKD step %c1 iter_args(%a1 = %a0) -> (f32) {")
        lines.append("            %kd_dd = arith.muli %kd_i, %cDD : index")
        lines.append("            %tmp_d = arith.addi %od_sd, %kd_dd : index")
        lines.append("            %in_d = arith.subi %tmp_d, %cPD : index")
        lines.append("            %d_ge0 = arith.cmpi sge, %in_d, %c0 : index")
        lines.append("            %d_lt = arith.cmpi slt, %in_d, %cD : index")
        lines.append("            %d_ok = arith.andi %d_ge0, %d_lt : i1")
        lines.append("            %w_kd = arith.muli %kd_i, %cKHKW : index")
        lines.append("            %acc_kh = scf.for %kh_i = %c0 to %cKH step %c1 iter_args(%a2 = %a1) -> (f32) {")
        lines.append("              %kh_dh = arith.muli %kh_i, %cDH : index")
        lines.append("              %tmp_h = arith.addi %oh_sh, %kh_dh : index")
        lines.append("              %in_h = arith.subi %tmp_h, %cPH : index")
        lines.append("              %h_ge0 = arith.cmpi sge, %in_h, %c0 : index")
        lines.append("              %h_lt = arith.cmpi slt, %in_h, %cH : index")
        lines.append("              %h_ok = arith.andi %h_ge0, %h_lt : i1")
        lines.append("              %w_kh = arith.muli %kh_i, %cKW : index")
        lines.append("              %w_kd_kh = arith.addi %w_kd, %w_kh : index")
        lines.append("              %acc_kw = scf.for %kw_i = %c0 to %cKW step %c1 iter_args(%a3 = %a2) -> (f32) {")
        lines.append("                %kw_dw = arith.muli %kw_i, %cDW : index")
        lines.append("                %tmp_w = arith.addi %ow_sw, %kw_dw : index")
        lines.append("                %in_w = arith.subi %tmp_w, %cPW : index")
        lines.append("                %w_ge0 = arith.cmpi sge, %in_w, %c0 : index")
        lines.append("                %w_lt = arith.cmpi slt, %in_w, %cW : index")
        lines.append("                %w_ok = arith.andi %w_ge0, %w_lt : i1")
        lines.append("                %dh_ok = arith.andi %d_ok, %h_ok : i1")
        lines.append("                %in_ok = arith.andi %dh_ok, %w_ok : i1")
        lines.append("                %x = scf.if %in_ok -> (f32) {")
        lines.append("                  %d_off = arith.muli %in_d, %cHW : index")
        lines.append("                  %h_off = arith.muli %in_h, %cW : index")
        lines.append("                  %dh_off = arith.addi %d_off, %h_off : index")
        lines.append("                  %dhw_off = arith.addi %dh_off, %in_w : index")
        lines.append("                  %in_idx = arith.addi %base_in, %dhw_off : index")
        lines.append(f"                  %v = memref.load {arg_ssa[in_name]}[%in_idx] : {in_memref}")
        lines.append("                  scf.yield %v : f32")
        lines.append("                } else {")
        lines.append("                  scf.yield %c0f : f32")
        lines.append("                }")
        lines.append("                %w_off = arith.addi %w_kd_kh, %kw_i : index")
        lines.append("                %w_idx = arith.addi %w_base_ic, %w_off : index")
        lines.append(f"                %w = memref.load {arg_ssa[w_name]}[%w_idx] : {w_memref}")
        lines.append(f"                %prod = arith.mulf %x, %w{fm} : f32")
        lines.append(f"                %a_next = arith.addf %a3, %prod{fm} : f32")
        lines.append("                scf.yield %a_next : f32")
        lines.append("              }")
        lines.append("              scf.yield %acc_kw : f32")
        lines.append("            }")
        lines.append("            scf.yield %acc_kh : f32")
        lines.append("          }")
        lines.append("          scf.yield %acc_kd : f32")
        lines.append("        }")
        lines.append(f"        memref.store %acc, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif conv_depthwise2d_nchw_v1 is not None:
        kernel_kind = "conv_depthwise2d_nchw_v1"
        in_name = str(conv_depthwise2d_nchw_v1["input"])
        w_name = str(conv_depthwise2d_nchw_v1["weight"])
        b_name = str(conv_depthwise2d_nchw_v1["bias"])
        out_name2 = str(conv_depthwise2d_nchw_v1["out"])
        n_dim = int(conv_depthwise2d_nchw_v1["N"])
        c_in = int(conv_depthwise2d_nchw_v1["C_IN"])
        c_out = int(conv_depthwise2d_nchw_v1["C_OUT"])
        h_dim = int(conv_depthwise2d_nchw_v1["H"])
        w_dim = int(conv_depthwise2d_nchw_v1["W"])
        kh = int(conv_depthwise2d_nchw_v1["KH"])
        kw = int(conv_depthwise2d_nchw_v1["KW"])
        sh = int(conv_depthwise2d_nchw_v1["SH"])
        sw = int(conv_depthwise2d_nchw_v1["SW"])
        ph = int(conv_depthwise2d_nchw_v1["PH"])
        pw = int(conv_depthwise2d_nchw_v1["PW"])
        dh = int(conv_depthwise2d_nchw_v1["DH"])
        dw = int(conv_depthwise2d_nchw_v1["DW"])
        mult = int(conv_depthwise2d_nchw_v1["MULT"])
        oh_dim = int(conv_depthwise2d_nchw_v1["OH"])
        ow_dim = int(conv_depthwise2d_nchw_v1["OW"])
        if (
            n_dim <= 0
            or c_in <= 0
            or c_out <= 0
            or h_dim <= 0
            or w_dim <= 0
            or kh <= 0
            or kw <= 0
            or oh_dim <= 0
            or ow_dim <= 0
            or sh <= 0
            or sw <= 0
            or dh <= 0
            or dw <= 0
            or ph < 0
            or pw < 0
            or mult <= 0
        ):
            raise RuntimeError(
                "conv_depthwise2d_nchw_v1 expects positive dims/stride/dilation/mult, got "
                f"N={n_dim} C_IN={c_in} C_OUT={c_out} H={h_dim} W={w_dim} KH={kh} KW={kw} "
                f"OH={oh_dim} OW={ow_dim} sh={sh} sw={sw} ph={ph} pw={pw} dh={dh} dw={dw} mult={mult}"
            )
        if c_out != (c_in * mult):
            raise RuntimeError(f"conv_depthwise2d_nchw_v1 expects C_OUT=C_IN*MULT, got C_OUT={c_out} C_IN={c_in} MULT={mult}")

        in_memref = str(arg_specs[in_name]["memref"])
        w_memref = str(arg_specs[w_name]["memref"])
        b_memref = str(arg_specs[b_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        hw = int(h_dim * w_dim)
        khkw = int(kh * kw)

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append(f"        %cC_OUT = arith.constant {int(c_out)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cC_IN = arith.constant {int(c_in)} : index")
        lines.append(f"        %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"        %cW = arith.constant {int(w_dim)} : index")
        lines.append(f"        %cHW = arith.constant {int(hw)} : index")
        lines.append(f"        %cKH = arith.constant {int(kh)} : index")
        lines.append(f"        %cKW = arith.constant {int(kw)} : index")
        lines.append(f"        %cKHKW = arith.constant {int(khkw)} : index")
        lines.append(f"        %cSH = arith.constant {int(sh)} : index")
        lines.append(f"        %cSW = arith.constant {int(sw)} : index")
        lines.append(f"        %cPH = arith.constant {int(ph)} : index")
        lines.append(f"        %cPW = arith.constant {int(pw)} : index")
        lines.append(f"        %cDH = arith.constant {int(dh)} : index")
        lines.append(f"        %cDW = arith.constant {int(dw)} : index")
        lines.append(f"        %cMULT = arith.constant {int(mult)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %oc = arith.remui %t2, %cC_OUT : index")
        lines.append("        %nn = arith.divui %t2, %cC_OUT : index")
        lines.append("        %ic = arith.divui %oc, %cMULT : index")
        lines.append("        %nc0 = arith.muli %nn, %cC_IN : index")
        lines.append("        %nci = arith.addi %nc0, %ic : index")
        lines.append("        %base_in = arith.muli %nci, %cHW : index")
        lines.append("        %w_base = arith.muli %oc, %cKHKW : index")
        lines.append("        %oh_sh = arith.muli %oh, %cSH : index")
        lines.append("        %ow_sw = arith.muli %ow, %cSW : index")
        lines.append(f"        %bias_v = memref.load {arg_ssa[b_name]}[%oc] : {b_memref}")
        lines.append("        %acc = scf.for %kh_i = %c0 to %cKH step %c1 iter_args(%a0 = %bias_v) -> (f32) {")
        lines.append("          %kh_dh = arith.muli %kh_i, %cDH : index")
        lines.append("          %tmp_h = arith.addi %oh_sh, %kh_dh : index")
        lines.append("          %in_h = arith.subi %tmp_h, %cPH : index")
        lines.append("          %h_ge0 = arith.cmpi sge, %in_h, %c0 : index")
        lines.append("          %h_lt = arith.cmpi slt, %in_h, %cH : index")
        lines.append("          %h_ok = arith.andi %h_ge0, %h_lt : i1")
        lines.append("          %w_kh = arith.muli %kh_i, %cKW : index")
        lines.append("          %acc_kw = scf.for %kw_i = %c0 to %cKW step %c1 iter_args(%a1 = %a0) -> (f32) {")
        lines.append("            %kw_dw = arith.muli %kw_i, %cDW : index")
        lines.append("            %tmp_w = arith.addi %ow_sw, %kw_dw : index")
        lines.append("            %in_w = arith.subi %tmp_w, %cPW : index")
        lines.append("            %w_ge0 = arith.cmpi sge, %in_w, %c0 : index")
        lines.append("            %w_lt = arith.cmpi slt, %in_w, %cW : index")
        lines.append("            %w_ok = arith.andi %w_ge0, %w_lt : i1")
        lines.append("            %in_ok = arith.andi %h_ok, %w_ok : i1")
        lines.append("            %x = scf.if %in_ok -> (f32) {")
        lines.append("              %h_mul = arith.muli %in_h, %cW : index")
        lines.append("              %hw_off = arith.addi %h_mul, %in_w : index")
        lines.append("              %in_idx = arith.addi %base_in, %hw_off : index")
        lines.append(f"              %v = memref.load {arg_ssa[in_name]}[%in_idx] : {in_memref}")
        lines.append("              scf.yield %v : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        lines.append("            %w_off = arith.addi %w_kh, %kw_i : index")
        lines.append("            %w_idx = arith.addi %w_base, %w_off : index")
        lines.append(f"            %w = memref.load {arg_ssa[w_name]}[%w_idx] : {w_memref}")
        lines.append(f"            %prod = arith.mulf %x, %w{fm} : f32")
        lines.append(f"            %a_next = arith.addf %a1, %prod{fm} : f32")
        lines.append("            scf.yield %a_next : f32")
        lines.append("          }")
        lines.append("          scf.yield %acc_kw : f32")
        lines.append("        }")
        lines.append(f"        memref.store %acc, {arg_ssa[out_name2]}[%lin] : {out_memref}")
        lines.append("      }")
    elif unique2d_v1 is not None:
        kernel_kind = "unique2d_v1"
        inp_name = str(unique2d_v1["inp"])
        out_name2 = str(unique2d_v1["out"])
        n_dim = int(unique2d_v1["N"])
        u_dim = int(unique2d_v1["U"])
        if n_dim <= 0 or u_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got N={n_dim} U={u_dim}")

        inp_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %c1 = arith.constant 1 : index")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cU = arith.constant {int(u_dim)} : index")
        lines.append("        %c0_i32 = arith.constant 0 : i32")
        lines.append("        %false = arith.constant 0 : i1")
        lines.append("        scf.for %ii = %c0 to %cU step %c1 {")
        lines.append(f"          memref.store %c0_i32, {arg_ssa[out_name2]}[%ii] : {out_memref}")
        lines.append("        }")

        lines.append("        %_pos = scf.for %ii2 = %c0 to %cN step %c1 iter_args(%pos = %c0) -> (index) {")
        lines.append(f"          %xv = memref.load {arg_ssa[inp_name]}[%ii2] : {inp_memref}")
        lines.append("          %dup = scf.for %jj = %c0 to %pos step %c1 iter_args(%d = %false) -> (i1) {")
        lines.append(f"            %prev = memref.load {arg_ssa[out_name2]}[%jj] : {out_memref}")
        lines.append("            %eq = arith.cmpi eq, %xv, %prev : i32")
        lines.append("            %d2 = arith.ori %d, %eq : i1")
        lines.append("            scf.yield %d2 : i1")
        lines.append("          }")
        lines.append("          %pos_next = scf.if %dup -> (index) {")
        lines.append("            scf.yield %pos : index")
        lines.append("          } else {")
        lines.append("            %pos_ok = arith.cmpi ult, %pos, %cU : index")
        lines.append("            %pos2 = scf.if %pos_ok -> (index) {")
        lines.append(f"              memref.store %xv, {arg_ssa[out_name2]}[%pos] : {out_memref}")
        lines.append("              %p_inc = arith.addi %pos, %c1 : index")
        lines.append("              scf.yield %p_inc : index")
        lines.append("            } else {")
        lines.append("              scf.yield %pos : index")
        lines.append("            }")
        lines.append("            scf.yield %pos2 : index")
        lines.append("          }")
        lines.append("          scf.yield %pos_next : index")
        lines.append("        }")

        lines.append("        scf.for %si = %c0 to %cU step %c1 {")
        lines.append("          %sj0 = arith.addi %si, %c1 : index")
        lines.append("          scf.for %sj = %sj0 to %cU step %c1 {")
        lines.append(f"            %a = memref.load {arg_ssa[out_name2]}[%si] : {out_memref}")
        lines.append(f"            %b = memref.load {arg_ssa[out_name2]}[%sj] : {out_memref}")
        lines.append("            %gt = arith.cmpi sgt, %a, %b : i32")
        lines.append("            scf.if %gt {")
        lines.append(f"              memref.store %b, {arg_ssa[out_name2]}[%si] : {out_memref}")
        lines.append(f"              memref.store %a, {arg_ssa[out_name2]}[%sj] : {out_memref}")
        lines.append("            }")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif nonzero2d_v1 is not None:
        kernel_kind = "nonzero2d_v1"
        inp_name = str(nonzero2d_v1["inp"])
        out_name2 = str(nonzero2d_v1["out"])
        m_dim = int(nonzero2d_v1["M"])
        n_dim = int(nonzero2d_v1["N"])
        nnz_dim = int(nonzero2d_v1["NNZ"])
        if m_dim <= 0 or n_dim <= 0 or nnz_dim <= 0:
            raise RuntimeError(f"{kernel_kind} expects positive dims, got M={m_dim} N={n_dim} NNZ={nnz_dim}")

        inp_memref = str(arg_specs[inp_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])

        launch_override = {"block": [256, 1, 1], "grid": [1, 1, 1]}

        total = int(int(m_dim) * int(n_dim))

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("      scf.if %is0 {")
        lines.append("        %c1 = arith.constant 1 : index")
        lines.append("        %c2 = arith.constant 2 : index")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cT = arith.constant {int(total)} : index")
        lines.append(f"        %cNNZ = arith.constant {int(nnz_dim)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %_pos = scf.for %ii = %c0 to %cT step %c1 iter_args(%pos = %c0) -> (index) {")
        lines.append(f"          %xv = memref.load {arg_ssa[inp_name]}[%ii] : {inp_memref}")
        lines.append("          %nz = arith.cmpf one, %xv, %c0f : f32")
        lines.append("          %pos_next = scf.if %nz -> (index) {")
        lines.append("            %pos_ok = arith.cmpi ult, %pos, %cNNZ : index")
        lines.append("            %pos2 = scf.if %pos_ok -> (index) {")
        lines.append("              %row = arith.divui %ii, %cN : index")
        lines.append("              %col = arith.remui %ii, %cN : index")
        lines.append("              %row_i64 = arith.index_cast %row : index to i64")
        lines.append("              %col_i64 = arith.index_cast %col : index to i64")
        lines.append("              %out_base = arith.muli %pos, %c2 : index")
        lines.append(f"              memref.store %row_i64, {arg_ssa[out_name2]}[%out_base] : {out_memref}")
        lines.append("              %out_base1 = arith.addi %out_base, %c1 : index")
        lines.append(f"              memref.store %col_i64, {arg_ssa[out_name2]}[%out_base1] : {out_memref}")
        lines.append("              %pos_inc = arith.addi %pos, %c1 : index")
        lines.append("              scf.yield %pos_inc : index")
        lines.append("            } else {")
        lines.append("              scf.yield %pos : index")
        lines.append("            }")
        lines.append("            scf.yield %pos2 : index")
        lines.append("          } else {")
        lines.append("            scf.yield %pos : index")
        lines.append("          }")
        lines.append("          scf.yield %pos_next : index")
        lines.append("        }")
        lines.append("      }")
    elif attn2d_v1 is not None and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", default=False):
        kernel_kind = "attn2d_causal_softmax_v2"
        q_name = str(attn2d_v1["Q"])
        k_name = str(attn2d_v1["K"])
        v_name = str(attn2d_v1["V"])
        out_name2 = str(attn2d_v1["out"])
        sm_scale_name = str(attn2d_v1["sm_scale"])
        q_ctx = int(attn2d_v1["Q_CTX"])
        kv_ctx = int(attn2d_v1["KV_CTX"])
        hd = int(attn2d_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        block_x = 32
        req_block_kv = 64 if intent_name == "masked_attention2d" else 32
        # Real-MLIR is shape-specialized: clamp KV tiling to the concrete KV_CTX
        # to avoid burning cycles iterating over out-of-range KV slots on small
        # perf shapes (e.g. KV_CTX=16).
        block_kv = int(min(int(req_block_kv), int(kv_ctx))) if int(kv_ctx) > 0 else int(req_block_kv)
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(q_ctx), 1, 1]}
        cuda_real_mlir_attention_cfg = {"block_x": int(block_x), "block_kv": int(block_kv)}
        fast_hd64 = int(hd) == 64

        def _warp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
            cur = str(val_ssa)
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                sel = _fresh("sel")
                nxt = _fresh("sum")
                lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"{indent}{sel} = arith.select {ok}, {sh}, %c0f : f32")
                lines.append(f"{indent}{nxt} = arith.addf {cur}, {sel}{fm} : f32")
                cur = str(nxt)
            return cur

        def _emit_dot_score(*, kv_ssa: str, indent: str) -> tuple[str, str]:
            pred_kv = _fresh("pred_kv")
            pred_causal = _fresh("pred_causal")
            pred_attend = _fresh("pred_attend")
            score = _fresh("score")
            lines.append(f"{indent}{pred_kv} = arith.cmpi ult, {kv_ssa}, %cKV : index")
            lines.append(f"{indent}{pred_causal} = arith.cmpi ule, {kv_ssa}, %bid : index")
            lines.append(f"{indent}{pred_attend} = arith.andi {pred_kv}, {pred_causal} : i1")
            lines.append(f"{indent}{score} = scf.if {pred_attend} -> (f32) {{")
            base_k = _fresh("base_k")
            lines.append(f"{indent}  {base_k} = arith.muli {kv_ssa}, %cHD : index")
            if fast_hd64:
                idx_k0 = _fresh("idx_k0")
                idx_k1 = _fresh("idx_k1")
                k0 = _fresh("k0")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {idx_k0} = arith.addi {base_k}, %tid : index")
                lines.append(f"{indent}  {idx_k1} = arith.addi {base_k}, %d1 : index")
                lines.append(f"{indent}  {k0} = memref.load {arg_ssa[k_name]}[{idx_k0}] : {k_memref}")
                lines.append(f"{indent}  {k1} = memref.load {arg_ssa[k_name]}[{idx_k1}] : {k_memref}")
            else:
                k0 = _fresh("k0")
                lines.append(f"{indent}  {k0} = scf.if %pred_d0 -> (f32) {{")
                idx_k0 = _fresh("idx_k0")
                kv0 = _fresh("kv0")
                lines.append(f"{indent}    {idx_k0} = arith.addi {base_k}, %tid : index")
                lines.append(f"{indent}    {kv0} = memref.load {arg_ssa[k_name]}[{idx_k0}] : {k_memref}")
                lines.append(f"{indent}    scf.yield {kv0} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
                k1 = _fresh("k1")
                lines.append(f"{indent}  {k1} = scf.if %pred_d1 -> (f32) {{")
                idx_k1 = _fresh("idx_k1")
                kv1 = _fresh("kv1")
                lines.append(f"{indent}    {idx_k1} = arith.addi {base_k}, %d1 : index")
                lines.append(f"{indent}    {kv1} = memref.load {arg_ssa[k_name]}[{idx_k1}] : {k_memref}")
                lines.append(f"{indent}    scf.yield {kv1} : f32")
                lines.append(f"{indent}  }} else {{")
                lines.append(f"{indent}    scf.yield %c0f : f32")
                lines.append(f"{indent}  }}")
            partial = _fresh("partial")
            tmp0 = _fresh("tmp0")
            # Force a real PTX `fma.rn.f32` (avoid libdevice calls like `__nv_fmaf`).
            lines.append(f"{indent}  {tmp0} = llvm.intr.fma(%q0, {k0}, %c0f) : (f32, f32, f32) -> f32")
            lines.append(f"{indent}  {partial} = llvm.intr.fma(%q1, {k1}, {tmp0}) : (f32, f32, f32) -> f32")
            dot = _warp_allreduce_sum_xor(partial, indent=f"{indent}  ")
            scaled = _fresh("scaled")
            lines.append(f"{indent}  {scaled} = arith.mulf {dot}, %sm2{fm} : f32")
            lines.append(f"{indent}  scf.yield {scaled} : f32")
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}  scf.yield %neg_inf : f32")
            lines.append(f"{indent}}}")
            return pred_attend, score

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBlockKV = arith.constant {int(block_kv)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cQ : index")
        lines.append("      scf.if %pred_row {")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
        lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")
        lines.append("        %base_q = arith.muli %bid, %cHD : index")
        lines.append("        %d1 = arith.addi %tid, %c32_idx : index")
        if fast_hd64:
            lines.append("        %pred_d0 = arith.constant 1 : i1")
            lines.append("        %pred_d1 = arith.constant 1 : i1")
            lines.append("        %idx_q0 = arith.addi %base_q, %tid : index")
            lines.append("        %idx_q1 = arith.addi %base_q, %d1 : index")
            lines.append(f"        %q0 = memref.load {arg_ssa[q_name]}[%idx_q0] : {q_memref}")
            lines.append(f"        %q1 = memref.load {arg_ssa[q_name]}[%idx_q1] : {q_memref}")
        else:
            lines.append("        %pred_d0 = arith.cmpi ult, %tid, %cHD : index")
            lines.append("        %pred_d1 = arith.cmpi ult, %d1, %cHD : index")
            lines.append("        %q0 = scf.if %pred_d0 -> (f32) {")
            lines.append("          %idx_q0 = arith.addi %base_q, %tid : index")
            lines.append(f"          %q0v = memref.load {arg_ssa[q_name]}[%idx_q0] : {q_memref}")
            lines.append("          scf.yield %q0v : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")
            lines.append("        %q1 = scf.if %pred_d1 -> (f32) {")
            lines.append("          %idx_q1 = arith.addi %base_q, %d1 : index")
            lines.append(f"          %q1v = memref.load {arg_ssa[q_name]}[%idx_q1] : {q_memref}")
            lines.append("          scf.yield %q1v : f32")
            lines.append("        } else {")
            lines.append("          scf.yield %c0f : f32")
            lines.append("        }")

        m_out = _fresh("m")
        l_out = _fresh("l")
        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        lines.append(
            f"        {m_out}, {l_out}, {acc0_out}, {acc1_out} = scf.for %tile0 = %c0 to %cKV step %cBlockKV "
            "iter_args(%m_i = %neg_inf, %l_i = %c0f, %a0 = %c0f, %a1 = %c0f) -> (f32, f32, f32, f32) {"
        )

        # Reduce register pressure by keeping per-tile KV loops in MLIR (avoid
        # Python unrolling + keeping every `score` live across pass1/pass2).
        tile_end = _fresh("tile_end")
        max_tile = _fresh("max_tile")
        lines.append(f"          {tile_end} = arith.addi %tile0, %cBlockKV : index")
        lines.append(
            f"          {max_tile} = scf.for %kv = %tile0 to {tile_end} step %c1 "
            "iter_args(%acc = %neg_inf) -> (f32) {"
        )
        _pred0, score0 = _emit_dot_score(kv_ssa="%kv", indent="            ")
        acc_next = _fresh("acc_next")
        lines.append(f"            {acc_next} = arith.maximumf %acc, {score0}{fm} : f32")
        lines.append(f"            scf.yield {acc_next} : f32")
        lines.append("          }")

        m_new = _fresh("m_new")
        lines.append(f"          {m_new} = arith.maximumf %m_i, {max_tile}{fm} : f32")
        delta = _fresh("delta")
        alpha = _fresh("alpha")
        lines.append(f"          {delta} = arith.subf %m_i, {m_new}{fm} : f32")
        lines.append(f"          {alpha} = math.exp2 {delta} : f32")
        l_scaled = _fresh("l_scaled")
        a0_scaled = _fresh("a0_scaled")
        a1_scaled = _fresh("a1_scaled")
        lines.append(f"          {l_scaled} = arith.mulf %l_i, {alpha}{fm} : f32")
        lines.append(f"          {a0_scaled} = arith.mulf %a0, {alpha}{fm} : f32")
        lines.append(f"          {a1_scaled} = arith.mulf %a1, {alpha}{fm} : f32")

        sum_out = _fresh("sum")
        acc0_sum = _fresh("acc0_sum")
        acc1_sum = _fresh("acc1_sum")
        lines.append(
            f"          {sum_out}, {acc0_sum}, {acc1_sum} = scf.for %kv2 = %tile0 to {tile_end} step %c1 "
            f"iter_args(%s = %c0f, %b0 = {a0_scaled}, %b1 = {a1_scaled}) -> (f32, f32, f32) {{"
        )
        pred2, score2 = _emit_dot_score(kv_ssa="%kv2", indent="            ")
        shift = _fresh("shift")
        p = _fresh("p")
        lines.append(f"            {shift} = arith.subf {score2}, {m_new}{fm} : f32")
        lines.append(f"            {p} = scf.if {pred2} -> (f32) {{")
        pv = _fresh("pv")
        lines.append(f"              {pv} = math.exp2 {shift} : f32")
        lines.append(f"              scf.yield {pv} : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        s_next = _fresh("s_next")
        lines.append(f"            {s_next} = arith.addf %s, {p}{fm} : f32")

        base_v = _fresh("base_v")
        lines.append(f"            {base_v} = arith.muli %kv2, %cHD : index")

        if fast_hd64:
            vv0 = _fresh("vv0")
            vv1 = _fresh("vv1")
            lines.append(f"            {vv0}, {vv1} = scf.if {pred2} -> (f32, f32) {{")
            idx_v0 = _fresh("idx_v0")
            idx_v1 = _fresh("idx_v1")
            v0 = _fresh("v0")
            v1 = _fresh("v1")
            lines.append(f"              {idx_v0} = arith.addi {base_v}, %tid : index")
            lines.append(f"              {idx_v1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"              {v0} = memref.load {arg_ssa[v_name]}[{idx_v0}] : {v_memref}")
            lines.append(f"              {v1} = memref.load {arg_ssa[v_name]}[{idx_v1}] : {v_memref}")
            lines.append(f"              scf.yield {v0}, {v1} : f32, f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f, %c0f : f32, f32")
            lines.append("            }")
        else:
            pred_v0 = _fresh("pred_v0")
            pred_v1 = _fresh("pred_v1")
            lines.append(f"            {pred_v0} = arith.andi {pred2}, %pred_d0 : i1")
            lines.append(f"            {pred_v1} = arith.andi {pred2}, %pred_d1 : i1")
            vv0 = _fresh("vv0")
            lines.append(f"            {vv0} = scf.if {pred_v0} -> (f32) {{")
            idx_v0 = _fresh("idx_v0")
            v0 = _fresh("v0")
            lines.append(f"              {idx_v0} = arith.addi {base_v}, %tid : index")
            lines.append(f"              {v0} = memref.load {arg_ssa[v_name]}[{idx_v0}] : {v_memref}")
            lines.append(f"              scf.yield {v0} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            vv1 = _fresh("vv1")
            lines.append(f"            {vv1} = scf.if {pred_v1} -> (f32) {{")
            idx_v1 = _fresh("idx_v1")
            v1 = _fresh("v1")
            lines.append(f"              {idx_v1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"              {v1} = memref.load {arg_ssa[v_name]}[{idx_v1}] : {v_memref}")
            lines.append(f"              scf.yield {v1} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")

        b0_next = _fresh("b0_next")
        b1_next = _fresh("b1_next")
        lines.append(f"            {b0_next} = llvm.intr.fma({p}, {vv0}, %b0) : (f32, f32, f32) -> f32")
        lines.append(f"            {b1_next} = llvm.intr.fma({p}, {vv1}, %b1) : (f32, f32, f32) -> f32")
        lines.append(f"            scf.yield {s_next}, {b0_next}, {b1_next} : f32, f32, f32")
        lines.append("          }")

        l_new = _fresh("l_new")
        lines.append(f"          {l_new} = arith.addf {l_scaled}, {sum_out}{fm} : f32")
        lines.append(f"          scf.yield {m_new}, {l_new}, {acc0_sum}, {acc1_sum} : f32, f32, f32, f32")
        lines.append("        }")

        sum_nz = _fresh("sum_nz")
        l_safe = _fresh("l_safe")
        out0 = _fresh("out0")
        out1 = _fresh("out1")
        lines.append(f"        {sum_nz} = arith.cmpf one, {l_out}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_out}, %c1f : f32")
        lines.append(f"        {out0} = arith.divf {acc0_out}, {l_safe}{fm} : f32")
        lines.append(f"        {out1} = arith.divf {acc1_out}, {l_safe}{fm} : f32")
        lines.append("        scf.if %pred_d0 {")
        idx_o0 = _fresh("idx_o0")
        lines.append(f"          {idx_o0} = arith.addi %base_q, %tid : index")
        lines.append(f"          memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
        lines.append("        }")
        lines.append("        scf.if %pred_d1 {")
        idx_o1 = _fresh("idx_o1")
        lines.append(f"          {idx_o1} = arith.addi %base_q, %d1 : index")
        lines.append(f"          memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_V1", default=False) and attn2d_v1 is not None:
        kernel_kind = "attn2d_causal_softmax_v1"
        q_name = str(attn2d_v1["Q"])
        k_name = str(attn2d_v1["K"])
        v_name = str(attn2d_v1["V"])
        out_name2 = str(attn2d_v1["out"])
        sm_scale_name = str(attn2d_v1["sm_scale"])
        q_ctx = int(attn2d_v1["Q_CTX"])
        kv_ctx = int(attn2d_v1["KV_CTX"])
        hd = int(attn2d_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        # Performance note: older versions used block.x=256 and reduced a HD-length dot
        # with many idle threads participating in gpu.barrier. For the small HEAD_DIM
        # values in Triton-native coverage (16/32/64), a single warp is enough.
        block_x = 32
        lanes_per_thread = int((int(hd) + int(block_x) - 1) // int(block_x))
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(q_ctx), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        scratch_off = int(hd)
        scores_off = int(scratch_off + int(block_x))
        weights_off = int(scores_off + kv_ctx)
        sh_elems = int(weights_off + kv_ctx)
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBX = arith.constant {int(block_x)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cQ : index")
        lines.append("      scf.if %pred_row {")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")

        # Load q row into shared[0:HD].
        lines.append("        %mul_q = arith.muli %bid, %cHD : index")
        for lane in range(lanes_per_thread):
            if lane == 0:
                gd = "%tid"
            else:
                c_lane = _fresh(f"c_lane_{lane}")
                gd = _fresh(f"gd_{lane}")
                lines.append(f"        {c_lane} = arith.constant {int(lane * int(block_x))} : index")
                lines.append(f"        {gd} = arith.addi %tid, {c_lane} : index")
            pred = _fresh(f"pred_qd_{lane}")
            idx = _fresh(f"idx_q_{lane}")
            qv = _fresh(f"qv_{lane}")
            lines.append(f"        {pred} = arith.cmpi ult, {gd}, %cHD : index")
            lines.append(f"        scf.if {pred} {{")
            lines.append(f"          {idx} = arith.addi %mul_q, {gd} : index")
            lines.append(f"          {qv} = memref.load {arg_ssa[q_name]}[{idx}] : {q_memref}")
            lines.append(f"          memref.store {qv}, %sh[{gd}] : {shared_global_memref_ty}")
            lines.append("        }")
        lines.append("        gpu.barrier")

        lines.append(f"        %cScratch = arith.constant {int(scratch_off)} : index")
        lines.append(f"        %cScores = arith.constant {int(scores_off)} : index")
        lines.append(f"        %cWeights = arith.constant {int(weights_off)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %neg = arith.constant -1.0e9 : f32")

        # Compute scores[kv] = dot(Q[q,:], K[kv,:]) * sm_scale, with causal mask kv>q.
        lines.append("        scf.for %kv = %c0 to %cKV step %c1 {")
        lines.append("          %mul_k = arith.muli %kv, %cHD : index")
        lane_ps: list[str] = []
        for lane in range(lanes_per_thread):
            if lane == 0:
                gd = "%tid"
            else:
                c_lane = _fresh(f"c_lane_k_{lane}")
                gd = _fresh(f"gd_k_{lane}")
                lines.append(f"          {c_lane} = arith.constant {int(lane * int(block_x))} : index")
                lines.append(f"          {gd} = arith.addi %tid, {c_lane} : index")
            pred_d = _fresh(f"pred_d_{lane}")
            prod_lane = _fresh(f"prod_{lane}")
            idx_k = _fresh(f"idx_k_{lane}")
            kvv = _fresh(f"kvv_{lane}")
            qvv = _fresh(f"qvv_{lane}")
            p = _fresh(f"p_{lane}")
            lines.append(f"          {pred_d} = arith.cmpi ult, {gd}, %cHD : index")
            lines.append(f"          {prod_lane} = scf.if {pred_d} -> (f32) {{")
            lines.append(f"            {idx_k} = arith.addi %mul_k, {gd} : index")
            lines.append(f"            {kvv} = memref.load {arg_ssa[k_name]}[{idx_k}] : {k_memref}")
            lines.append(f"            {qvv} = memref.load %sh[{gd}] : {shared_global_memref_ty}")
            lines.append(f"            {p} = arith.mulf {kvv}, {qvv}{fm} : f32")
            lines.append(f"            scf.yield {p} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lane_ps.append(str(prod_lane))
        if len(lane_ps) == 1:
            # MLIR has no SSA "alias" assignment; materialize a cheap identity op.
            lines.append(f"          %prod = arith.addf {lane_ps[0]}, %c0f{fm} : f32")
        else:
            acc = lane_ps[0]
            for i, p_lane in enumerate(lane_ps[1:], start=1):
                s = _fresh(f"sum_lane_{i}")
                lines.append(f"          {s} = arith.addf {acc}, {p_lane}{fm} : f32")
                acc = str(s)
            lines.append(f"          %prod = arith.addf {acc}, %c0f{fm} : f32")
        lines.append("          %sidx = arith.addi %cScratch, %tid : index")
        lines.append(f"          memref.store %prod, %sh[%sidx] : {shared_global_memref_ty}")
        lines.append("          gpu.barrier")
        dot_strides: list[int] = []
        s = int(block_x) // 2
        while s >= 1:
            dot_strides.append(int(s))
            s //= 2
        for stride in dot_strides:
            cS = f"%cS_dot_{stride}"
            pS = f"%pS_dot_{stride}"
            tid2 = f"%tid_dot_{stride}"
            a = f"%a_dot_{stride}"
            b = f"%b_dot_{stride}"
            s = f"%s_dot_{stride}"
            idx_a = f"%idx_a_dot_{stride}"
            idx_b = f"%idx_b_dot_{stride}"
            lines.append(f"          {cS} = arith.constant {int(stride)} : index")
            lines.append(f"          {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"          scf.if {pS} {{")
            lines.append(f"            {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"            {idx_a} = arith.addi %cScratch, %tid : index")
            lines.append(f"            {idx_b} = arith.addi %cScratch, {tid2} : index")
            lines.append(f"            {a} = memref.load %sh[{idx_a}] : {shared_global_memref_ty}")
            lines.append(f"            {b} = memref.load %sh[{idx_b}] : {shared_global_memref_ty}")
            lines.append(f"            {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"            memref.store {s}, %sh[{idx_a}] : {shared_global_memref_ty}")
            lines.append("          }")
            lines.append("          gpu.barrier")
        lines.append("          %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("          scf.if %is0 {")
        lines.append(f"            %sum = memref.load %sh[%cScratch] : {shared_global_memref_ty}")
        lines.append(f"            %scaled = arith.mulf %sum, %sm{fm} : f32")
        lines.append("            %kv_gt_q = arith.cmpi ugt, %kv, %bid : index")
        lines.append("            %score = arith.select %kv_gt_q, %neg, %scaled : f32")
        lines.append("            %off = arith.addi %cScores, %kv : index")
        lines.append(f"            memref.store %score, %sh[%off] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("          gpu.barrier")
        lines.append("        }")

        # Softmax normalization on scores -> weights.
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %init_max = arith.constant -3.402823466e+38 : f32")
        lines.append("          %maxv = scf.for %k = %c0 to %cKV step %c1 iter_args(%acc = %init_max) -> (f32) {")
        lines.append("            %off = arith.addi %cScores, %k : index")
        lines.append(f"            %x = memref.load %sh[%off] : {shared_global_memref_ty}")
        lines.append(f"            %acc_next = arith.maximumf %acc, %x{fm} : f32")
        lines.append("            scf.yield %acc_next : f32")
        lines.append("          }")
        lines.append("          %sumv = scf.for %k2 = %c0 to %cKV step %c1 iter_args(%acc = %c0f) -> (f32) {")
        lines.append("            %off2 = arith.addi %cScores, %k2 : index")
        lines.append(f"            %x2 = memref.load %sh[%off2] : {shared_global_memref_ty}")
        lines.append(f"            %xc = arith.subf %x2, %maxv{fm} : f32")
        lines.append(f"            %e = math.exp %xc{fm} : f32")
        lines.append("            %wo = arith.addi %cWeights, %k2 : index")
        lines.append(f"            memref.store %e, %sh[%wo] : {shared_global_memref_ty}")
        lines.append(f"            %acc_next2 = arith.addf %acc, %e{fm} : f32")
        lines.append("            scf.yield %acc_next2 : f32")
        lines.append("          }")
        lines.append("          %c1f = arith.constant 1.0 : f32")
        lines.append("          %sum_nz = arith.cmpf one, %sumv, %c0f : f32")
        lines.append("          %sum_safe = arith.select %sum_nz, %sumv, %c1f : f32")
        lines.append("          scf.for %k3 = %c0 to %cKV step %c1 {")
        lines.append("            %wo3 = arith.addi %cWeights, %k3 : index")
        lines.append(f"            %e3 = memref.load %sh[%wo3] : {shared_global_memref_ty}")
        lines.append(f"            %p3 = arith.divf %e3, %sum_safe{fm} : f32")
        lines.append(f"            memref.store %p3, %sh[%wo3] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("        }")
        lines.append("        gpu.barrier")

        # Weighted sum: Out[q, d] = sum_k weights[k] * V[k, d]
        lines.append("        %pred_od = arith.cmpi ult, %tid, %cHD : index")
        lines.append("        scf.if %pred_od {")
        lines.append("          %acc = scf.for %k4 = %c0 to %cKV step %c1 iter_args(%a = %c0f) -> (f32) {")
        lines.append("            %wo4 = arith.addi %cWeights, %k4 : index")
        lines.append(f"            %w = memref.load %sh[%wo4] : {shared_global_memref_ty}")
        lines.append("            %mul_v = arith.muli %k4, %cHD : index")
        lines.append("            %idx_v = arith.addi %mul_v, %tid : index")
        lines.append(f"            %vv = memref.load {arg_ssa[v_name]}[%idx_v] : {v_memref}")
        lines.append(f"            %p = arith.mulf %w, %vv{fm} : f32")
        lines.append(f"            %a2 = arith.addf %a, %p{fm} : f32")
        lines.append("            scf.yield %a2 : f32")
        lines.append("          }")
        lines.append("          %mul_o = arith.muli %bid, %cHD : index")
        lines.append("          %idx_o = arith.addi %mul_o, %tid : index")
        lines.append(f"          memref.store %acc, {arg_ssa[out_name2]}[%idx_o] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif (
        attn_fwd_v1 is not None
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V1", default=False)
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V2", default=False)
    ):
        # Perf-first: `_attn_fwd` is the min-ratio offender in the Triton-native lane.
        #
        # Implement a multi-query per CTA version that reuses K/V tiles via shared
        # memory and uses warp shuffles for reductions. Keep v2 (warp-only) as a
        # debug toggle via `INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V2=1`.
        kernel_kind = "attn_fwd_tiled_v3"
        q_name = str(attn_fwd_v1["Q"])
        k_name = str(attn_fwd_v1["K"])
        v_name = str(attn_fwd_v1["V"])
        out_name2 = str(attn_fwd_v1["out"])
        sm_scale_name = str(attn_fwd_v1["sm_scale"])
        z_dim = int(attn_fwd_v1["Z"])
        h_dim = int(attn_fwd_v1["num_head"])
        q_ctx = int(attn_fwd_v1["Q_CTX"])
        kv_ctx = int(attn_fwd_v1["KV_CTX"])
        hd = int(attn_fwd_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        block_x = 256
        block_m = 8
        block_kv = 16
        grid_x = (int(q_ctx) + int(block_m) - 1) // int(block_m)
        grid_y = int(z_dim) * int(h_dim)
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(grid_x), int(grid_y), 1]}
        cuda_real_mlir_attention_cfg = {"block_x": int(block_x), "block_m": int(block_m), "block_kv": int(block_kv)}

        # Shared layout: [K_tile (block_kv*hd), V_tile (block_kv*hd)].
        tile_elems = int(block_kv) * int(hd)
        offset_v = int(tile_elems)
        sh_elems = int(tile_elems) * 2
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        def _warp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
            cur = str(val_ssa)
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                sel = _fresh("sel")
                nxt = _fresh("sum")
                lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"{indent}{sel} = arith.select {ok}, {sh}, %c0f : f32")
                lines.append(f"{indent}{nxt} = arith.addf {cur}, {sel}{fm} : f32")
                cur = str(nxt)
            return cur

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid_x = gpu.block_id x")
        lines.append("      %bid_zh = gpu.block_id y")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
        lines.append(f"      %cZ = arith.constant {int(z_dim)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBlockM = arith.constant {int(block_m)} : index")
        lines.append(f"      %cBlockKV = arith.constant {int(block_kv)} : index")
        lines.append(f"      %cGridX = arith.constant {int(grid_x)} : index")
        lines.append(f"      %cGridY = arith.constant {int(grid_y)} : index")
        lines.append(f"      %cTileElems = arith.constant {int(tile_elems)} : index")
        lines.append(f"      %cOffsetV = arith.constant {int(offset_v)} : index")
        lines.append("      %pred_x = arith.cmpi ult, %bid_x, %cGridX : index")
        lines.append("      %pred_y = arith.cmpi ult, %bid_zh, %cGridY : index")
        lines.append("      %pred_block = arith.andi %pred_x, %pred_y : i1")
        lines.append("      scf.if %pred_block {")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
        lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")

        # Warp/lane decomposition (block.x == 256 => 8 warps).
        lines.append("        %lane = arith.remui %tid, %c32_idx : index")
        lines.append("        %warp = arith.divui %tid, %c32_idx : index")
        lines.append("        %d1 = arith.addi %lane, %c32_idx : index")
        lines.append("        %pred_d0 = arith.cmpi ult, %lane, %cHD : index")
        lines.append("        %pred_d1 = arith.cmpi ult, %d1, %cHD : index")

        # Map CTA -> (zh, q_base), and warp -> q within the CTA.
        lines.append("        %q_base = arith.muli %bid_x, %cBlockM : index")
        lines.append("        %q = arith.addi %q_base, %warp : index")
        lines.append("        %pred_q = arith.cmpi ult, %q, %cQ : index")

        # Base indices for Q/Out row (flattened) for this warp.
        lines.append("        %mul_qrow = arith.muli %bid_zh, %cQ : index")
        lines.append("        %qrow = arith.addi %mul_qrow, %q : index")
        lines.append("        %base_q = arith.muli %qrow, %cHD : index")

        q0 = _fresh("q0")
        q1 = _fresh("q1")
        lines.append(f"        {q0} = scf.if %pred_q -> (f32) {{")
        q0_in = _fresh("q0v")
        lines.append(f"          {q0_in} = scf.if %pred_d0 -> (f32) {{")
        idx_q0 = _fresh("idx_q0")
        v_q0 = _fresh("q0_load")
        lines.append(f"            {idx_q0} = arith.addi %base_q, %lane : index")
        lines.append(f"            {v_q0} = memref.load {arg_ssa[q_name]}[{idx_q0}] : {q_memref}")
        lines.append(f"            scf.yield {v_q0} : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        lines.append(f"          scf.yield {q0_in} : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")
        lines.append(f"        {q1} = scf.if %pred_q -> (f32) {{")
        q1_in = _fresh("q1v")
        lines.append(f"          {q1_in} = scf.if %pred_d1 -> (f32) {{")
        idx_q1 = _fresh("idx_q1")
        v_q1 = _fresh("q1_load")
        lines.append(f"            {idx_q1} = arith.addi %base_q, %d1 : index")
        lines.append(f"            {v_q1} = memref.load {arg_ssa[q_name]}[{idx_q1}] : {q_memref}")
        lines.append(f"            scf.yield {v_q1} : f32")
        lines.append("          } else {")
        lines.append("            scf.yield %c0f : f32")
        lines.append("          }")
        lines.append(f"          scf.yield {q1_in} : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")

        m_out = _fresh("m")
        l_out = _fresh("l")
        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        lines.append(
            f"        {m_out}, {l_out}, {acc0_out}, {acc1_out} = scf.for %tile0 = %c0 to %cKV step %cBlockKV "
            "iter_args(%m_i = %neg_inf, %l_i = %c0f, %a0 = %c0f, %a1 = %c0f) -> (f32, f32, f32, f32) {"
        )

        # Cooperative load K/V tile into shared.
        loads = (int(tile_elems) + int(block_x) - 1) // int(block_x)
        for i in range(int(loads)):
            off = int(i) * int(block_x)
            c_off = _fresh("c_off")
            idx = _fresh("idx")
            pred = _fresh("pred")
            lines.append(f"          {c_off} = arith.constant {int(off)} : index")
            lines.append(f"          {idx} = arith.addi %tid, {c_off} : index")
            lines.append(f"          {pred} = arith.cmpi ult, {idx}, %cTileElems : index")
            lines.append(f"          scf.if {pred} {{")
            row = _fresh("row")
            d = _fresh("d")
            kv = _fresh("kv")
            pred_kv = _fresh("pred_kv")
            k_val = _fresh("kval")
            v_val = _fresh("vval")
            lines.append(f"            {row} = arith.divui {idx}, %cHD : index")
            lines.append(f"            {d} = arith.remui {idx}, %cHD : index")
            lines.append(f"            {kv} = arith.addi %tile0, {row} : index")
            lines.append(f"            {pred_kv} = arith.cmpi ult, {kv}, %cKV : index")
            lines.append(f"            {k_val} = scf.if {pred_kv} -> (f32) {{")
            mul_zh = _fresh("mul_zh")
            row_k = _fresh("row_k")
            base_k = _fresh("base_k")
            idx_k = _fresh("idx_k")
            k_load = _fresh("k_load")
            lines.append(f"              {mul_zh} = arith.muli %bid_zh, %cKV : index")
            lines.append(f"              {row_k} = arith.addi {mul_zh}, {kv} : index")
            lines.append(f"              {base_k} = arith.muli {row_k}, %cHD : index")
            lines.append(f"              {idx_k} = arith.addi {base_k}, {d} : index")
            lines.append(f"              {k_load} = memref.load {arg_ssa[k_name]}[{idx_k}] : {k_memref}")
            lines.append(f"              scf.yield {k_load} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            lines.append(f"            {v_val} = scf.if {pred_kv} -> (f32) {{")
            mul_zh2 = _fresh("mul_zh")
            row_v = _fresh("row_v")
            base_v = _fresh("base_v")
            idx_v = _fresh("idx_v")
            v_load = _fresh("v_load")
            lines.append(f"              {mul_zh2} = arith.muli %bid_zh, %cKV : index")
            lines.append(f"              {row_v} = arith.addi {mul_zh2}, {kv} : index")
            lines.append(f"              {base_v} = arith.muli {row_v}, %cHD : index")
            lines.append(f"              {idx_v} = arith.addi {base_v}, {d} : index")
            lines.append(f"              {v_load} = memref.load {arg_ssa[v_name]}[{idx_v}] : {v_memref}")
            lines.append(f"              scf.yield {v_load} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            idx_vsh = _fresh("idx_vsh")
            lines.append(f"            memref.store {k_val}, %sh[{idx}] : {shared_global_memref_ty}")
            lines.append(f"            {idx_vsh} = arith.addi {idx}, %cOffsetV : index")
            lines.append(f"            memref.store {v_val}, %sh[{idx_vsh}] : {shared_global_memref_ty}")
            lines.append("          }")

        lines.append("          gpu.barrier")

        # Update per-query state (skip compute for out-of-range warps, but still
        # participate in barriers).
        m_next = _fresh("m_next")
        l_next = _fresh("l_next")
        a0_next = _fresh("a0_next")
        a1_next = _fresh("a1_next")
        lines.append(f"          {m_next}, {l_next}, {a0_next}, {a1_next} = scf.if %pred_q -> (f32, f32, f32, f32) {{")

        def _emit_dot_score_shared(*, t_ssa: str, indent: str) -> tuple[str, str]:
            kv = _fresh("kv")
            pred_kv = _fresh("pred_kv")
            score = _fresh("score")
            lines.append(f"{indent}{kv} = arith.addi %tile0, {t_ssa} : index")
            lines.append(f"{indent}{pred_kv} = arith.cmpi ult, {kv}, %cKV : index")
            lines.append(f"{indent}{score} = scf.if {pred_kv} -> (f32) {{")
            base = _fresh("base")
            lines.append(f"{indent}  {base} = arith.muli {t_ssa}, %cHD : index")
            k0 = _fresh("k0")
            k1 = _fresh("k1")
            lines.append(f"{indent}  {k0} = scf.if %pred_d0 -> (f32) {{")
            idx0 = _fresh("idx_k0")
            v0 = _fresh("kv0")
            lines.append(f"{indent}    {idx0} = arith.addi {base}, %lane : index")
            lines.append(f"{indent}    {v0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
            lines.append(f"{indent}    scf.yield {v0} : f32")
            lines.append(f"{indent}  }} else {{")
            lines.append(f"{indent}    scf.yield %c0f : f32")
            lines.append(f"{indent}  }}")
            lines.append(f"{indent}  {k1} = scf.if %pred_d1 -> (f32) {{")
            idx1 = _fresh("idx_k1")
            v1 = _fresh("kv1")
            lines.append(f"{indent}    {idx1} = arith.addi {base}, %d1 : index")
            lines.append(f"{indent}    {v1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
            lines.append(f"{indent}    scf.yield {v1} : f32")
            lines.append(f"{indent}  }} else {{")
            lines.append(f"{indent}    scf.yield %c0f : f32")
            lines.append(f"{indent}  }}")
            partial = _fresh("partial")
            tmp0 = _fresh("tmp0")
            lines.append(f"{indent}  {tmp0} = llvm.intr.fma({q0}, {k0}, %c0f) : (f32, f32, f32) -> f32")
            lines.append(f"{indent}  {partial} = llvm.intr.fma({q1}, {k1}, {tmp0}) : (f32, f32, f32) -> f32")
            dot = _warp_allreduce_sum_xor(partial, indent=f"{indent}  ")
            scaled = _fresh("scaled")
            lines.append(f"{indent}  {scaled} = arith.mulf {dot}, %sm2{fm} : f32")
            lines.append(f"{indent}  scf.yield {scaled} : f32")
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}  scf.yield %neg_inf : f32")
            lines.append(f"{indent}}}")
            return pred_kv, score

        # Compute per-KV scores once (avoid pass2 dot recompute).
        tile_scores: list[tuple[str, str, str]] = []
        for t in range(int(block_kv)):
            ct = _fresh("ct")
            lines.append(f"            {ct} = arith.constant {int(t)} : index")
            pred_t, score_t = _emit_dot_score_shared(t_ssa=str(ct), indent="            ")
            tile_scores.append((str(ct), str(pred_t), str(score_t)))

        max_tile_ssa = "%neg_inf"
        for _ct, _pred, score_t in tile_scores:
            mx_next = _fresh("mx_next")
            lines.append(f"            {mx_next} = arith.maximumf {max_tile_ssa}, {score_t}{fm} : f32")
            max_tile_ssa = str(mx_next)

        m_new = _fresh("m_new")
        delta = _fresh("delta")
        alpha = _fresh("alpha")
        lines.append(f"            {m_new} = arith.maximumf %m_i, {max_tile_ssa}{fm} : f32")
        lines.append(f"            {delta} = arith.subf %m_i, {m_new}{fm} : f32")
        lines.append(f"            {alpha} = math.exp2 {delta} : f32")
        l_scaled = _fresh("l_scaled")
        a0_scaled = _fresh("a0_scaled")
        a1_scaled = _fresh("a1_scaled")
        lines.append(f"            {l_scaled} = arith.mulf %l_i, {alpha}{fm} : f32")
        lines.append(f"            {a0_scaled} = arith.mulf %a0, {alpha}{fm} : f32")
        lines.append(f"            {a1_scaled} = arith.mulf %a1, {alpha}{fm} : f32")

        sum_cur = "%c0f"
        acc0_cur = str(a0_scaled)
        acc1_cur = str(a1_scaled)
        for t_ssa, pred_kv2, score_t in tile_scores:
            shift = _fresh("shift")
            p = _fresh("p")
            lines.append(f"            {shift} = arith.subf {score_t}, {m_new}{fm} : f32")
            lines.append(f"            {p} = scf.if {pred_kv2} -> (f32) {{")
            pv = _fresh("pv")
            lines.append(f"              {pv} = math.exp2 {shift} : f32")
            lines.append(f"              scf.yield {pv} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            s_next = _fresh("s_next")
            lines.append(f"            {s_next} = arith.addf {sum_cur}, {p}{fm} : f32")
            sum_cur = str(s_next)

            base = _fresh("base")
            base_v = _fresh("base_v")
            lines.append(f"            {base} = arith.muli {t_ssa}, %cHD : index")
            lines.append(f"            {base_v} = arith.addi {base}, %cOffsetV : index")
            v0 = _fresh("v0")
            v1 = _fresh("v1")
            lines.append(f"            {v0} = scf.if %pred_d0 -> (f32) {{")
            idx0 = _fresh("idx_v0")
            vv0 = _fresh("vv0")
            lines.append(f"              {idx0} = arith.addi {base_v}, %lane : index")
            lines.append(f"              {vv0} = memref.load %sh[{idx0}] : {shared_global_memref_ty}")
            lines.append(f"              scf.yield {vv0} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            lines.append(f"            {v1} = scf.if %pred_d1 -> (f32) {{")
            idx1 = _fresh("idx_v1")
            vv1 = _fresh("vv1")
            lines.append(f"              {idx1} = arith.addi {base_v}, %d1 : index")
            lines.append(f"              {vv1} = memref.load %sh[{idx1}] : {shared_global_memref_ty}")
            lines.append(f"              scf.yield {vv1} : f32")
            lines.append("            } else {")
            lines.append("              scf.yield %c0f : f32")
            lines.append("            }")
            b0_next = _fresh("b0_next")
            b1_next = _fresh("b1_next")
            lines.append(
                f"            {b0_next} = llvm.intr.fma({p}, {v0}, {acc0_cur}) : (f32, f32, f32) -> f32"
            )
            lines.append(
                f"            {b1_next} = llvm.intr.fma({p}, {v1}, {acc1_cur}) : (f32, f32, f32) -> f32"
            )
            acc0_cur = str(b0_next)
            acc1_cur = str(b1_next)

        l_new = _fresh("l_new")
        lines.append(f"            {l_new} = arith.addf {l_scaled}, {sum_cur}{fm} : f32")
        lines.append(f"            scf.yield {m_new}, {l_new}, {acc0_cur}, {acc1_cur} : f32, f32, f32, f32")
        lines.append("          } else {")
        lines.append("            scf.yield %m_i, %l_i, %a0, %a1 : f32, f32, f32, f32")
        lines.append("          }")

        lines.append("          gpu.barrier")
        lines.append(f"          scf.yield {m_next}, {l_next}, {a0_next}, {a1_next} : f32, f32, f32, f32")
        lines.append("        }")

        sum_nz = _fresh("sum_nz")
        l_safe = _fresh("l_safe")
        out0 = _fresh("out0")
        out1 = _fresh("out1")
        lines.append(f"        {sum_nz} = arith.cmpf one, {l_out}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_out}, %c1f : f32")
        lines.append(f"        {out0} = arith.divf {acc0_out}, {l_safe}{fm} : f32")
        lines.append(f"        {out1} = arith.divf {acc1_out}, {l_safe}{fm} : f32")
        lines.append("        scf.if %pred_q {")
        lines.append("          scf.if %pred_d0 {")
        idx_o0 = _fresh("idx_o0")
        lines.append(f"            {idx_o0} = arith.addi %base_q, %lane : index")
        lines.append(f"            memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
        lines.append("          }")
        lines.append("          scf.if %pred_d1 {")
        idx_o1 = _fresh("idx_o1")
        lines.append(f"            {idx_o1} = arith.addi %base_q, %d1 : index")
        lines.append(f"            memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
        lines.append("          }")
        lines.append("        }")
        lines.append("      }")
    elif (
        attn_fwd_v1 is not None
        and _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V2", default=False)
        and not _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V1", default=False)
    ):
        kernel_kind = "attn_fwd_softmax_v2"
        q_name = str(attn_fwd_v1["Q"])
        k_name = str(attn_fwd_v1["K"])
        v_name = str(attn_fwd_v1["V"])
        out_name2 = str(attn_fwd_v1["out"])
        sm_scale_name = str(attn_fwd_v1["sm_scale"])
        z_dim = int(attn_fwd_v1["Z"])
        h_dim = int(attn_fwd_v1["num_head"])
        q_ctx = int(attn_fwd_v1["Q_CTX"])
        kv_ctx = int(attn_fwd_v1["KV_CTX"])
        hd = int(attn_fwd_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        blocks = int(z_dim) * int(h_dim) * int(q_ctx)
        block_x = 32
        block_kv = 16
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(blocks), 1, 1]}
        cuda_real_mlir_attention_cfg = {"block_x": int(block_x), "block_kv": int(block_kv)}

        # For `_attn_fwd`, match the Triton baseline's exp2-based softmax by working
        # in log2 space: score = dot(Q,K) * sm_scale * LOG2E, then p = exp2(score - m).
        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append("      %c0f = arith.constant 0.0 : f32")
        lines.append("      %c1f = arith.constant 1.0 : f32")
        lines.append("      %neg_inf = arith.constant -3.402823466e+38 : f32")
        lines.append("      %c32_idx = arith.constant 32 : index")
        lines.append("      %c16_i32 = arith.constant 16 : i32")
        lines.append("      %c8_i32 = arith.constant 8 : i32")
        lines.append("      %c4_i32 = arith.constant 4 : i32")
        lines.append("      %c2_i32 = arith.constant 2 : i32")
        lines.append("      %c1_i32 = arith.constant 1 : i32")
        lines.append("      %c32_i32 = arith.constant 32 : i32")
        lines.append("      %cLOG2E = arith.constant 1.44269504 : f32")
        lines.append(f"      %cZ = arith.constant {int(z_dim)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBlockKV = arith.constant {int(block_kv)} : index")
        lines.append(f"      %cBlocks = arith.constant {int(blocks)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cBlocks : index")
        lines.append("      scf.if %pred_row {")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")
        lines.append(f"        %sm2 = arith.mulf %sm, %cLOG2E{fm} : f32")

        # Decode bid -> (z, head, q).
        lines.append("        %q = arith.remui %bid, %cQ : index")
        lines.append("        %tmp0 = arith.divui %bid, %cQ : index")
        lines.append("        %head = arith.remui %tmp0, %cH : index")
        lines.append("        %z = arith.divui %tmp0, %cH : index")
        lines.append("        %mul_zh = arith.muli %z, %cH : index")
        lines.append("        %zh = arith.addi %mul_zh, %head : index")

        # Base indices for Q/Out row (flattened).
        lines.append("        %mul_qrow = arith.muli %zh, %cQ : index")
        lines.append("        %qrow = arith.addi %mul_qrow, %q : index")
        lines.append("        %base_q = arith.muli %qrow, %cHD : index")
        lines.append("        %d1 = arith.addi %tid, %c32_idx : index")
        lines.append("        %pred_d0 = arith.cmpi ult, %tid, %cHD : index")
        lines.append("        %pred_d1 = arith.cmpi ult, %d1, %cHD : index")
        lines.append("        %q0 = scf.if %pred_d0 -> (f32) {")
        lines.append("          %idx_q0 = arith.addi %base_q, %tid : index")
        lines.append(f"          %q0v = memref.load {arg_ssa[q_name]}[%idx_q0] : {q_memref}")
        lines.append("          scf.yield %q0v : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")
        lines.append("        %q1 = scf.if %pred_d1 -> (f32) {")
        lines.append("          %idx_q1 = arith.addi %base_q, %d1 : index")
        lines.append(f"          %q1v = memref.load {arg_ssa[q_name]}[%idx_q1] : {q_memref}")
        lines.append("          scf.yield %q1v : f32")
        lines.append("        } else {")
        lines.append("          scf.yield %c0f : f32")
        lines.append("        }")

        def _warp_allreduce_sum_xor(val_ssa: str, *, indent: str) -> str:
            cur = str(val_ssa)
            for off in ("%c16_i32", "%c8_i32", "%c4_i32", "%c2_i32", "%c1_i32"):
                sh = _fresh("sh")
                ok = _fresh("ok")
                sel = _fresh("sel")
                nxt = _fresh("sum")
                lines.append(f"{indent}{sh}, {ok} = gpu.shuffle xor {cur}, {off}, %c32_i32 : f32")
                lines.append(f"{indent}{sel} = arith.select {ok}, {sh}, %c0f : f32")
                lines.append(f"{indent}{nxt} = arith.addf {cur}, {sel}{fm} : f32")
                cur = str(nxt)
            return cur

        def _emit_dot_score(*, kv_ssa: str, indent: str) -> tuple[str, str]:
            pred_kv = _fresh("pred_kv")
            score = _fresh("score")
            lines.append(f"{indent}{pred_kv} = arith.cmpi ult, {kv_ssa}, %cKV : index")
            lines.append(f"{indent}{score} = scf.if {pred_kv} -> (f32) {{")
            base_krow0 = _fresh("base_krow0")
            base_krow = _fresh("base_krow")
            base_k = _fresh("base_k")
            lines.append(f"{indent}  {base_krow0} = arith.muli %zh, %cKV : index")
            lines.append(f"{indent}  {base_krow} = arith.addi {base_krow0}, {kv_ssa} : index")
            lines.append(f"{indent}  {base_k} = arith.muli {base_krow}, %cHD : index")
            k0 = _fresh("k0")
            lines.append(f"{indent}  {k0} = scf.if %pred_d0 -> (f32) {{")
            idx_k0 = _fresh("idx_k0")
            kv0 = _fresh("kv0")
            lines.append(f"{indent}    {idx_k0} = arith.addi {base_k}, %tid : index")
            lines.append(f"{indent}    {kv0} = memref.load {arg_ssa[k_name]}[{idx_k0}] : {k_memref}")
            lines.append(f"{indent}    scf.yield {kv0} : f32")
            lines.append(f"{indent}  }} else {{")
            lines.append(f"{indent}    scf.yield %c0f : f32")
            lines.append(f"{indent}  }}")
            k1 = _fresh("k1")
            lines.append(f"{indent}  {k1} = scf.if %pred_d1 -> (f32) {{")
            idx_k1 = _fresh("idx_k1")
            kv1 = _fresh("kv1")
            lines.append(f"{indent}    {idx_k1} = arith.addi {base_k}, %d1 : index")
            lines.append(f"{indent}    {kv1} = memref.load {arg_ssa[k_name]}[{idx_k1}] : {k_memref}")
            lines.append(f"{indent}    scf.yield {kv1} : f32")
            lines.append(f"{indent}  }} else {{")
            lines.append(f"{indent}    scf.yield %c0f : f32")
            lines.append(f"{indent}  }}")
            prod0 = _fresh("prod0")
            prod1 = _fresh("prod1")
            partial = _fresh("partial")
            lines.append(f"{indent}  {prod0} = arith.mulf %q0, {k0}{fm} : f32")
            lines.append(f"{indent}  {prod1} = arith.mulf %q1, {k1}{fm} : f32")
            lines.append(f"{indent}  {partial} = arith.addf {prod0}, {prod1}{fm} : f32")
            dot = _warp_allreduce_sum_xor(partial, indent=f"{indent}  ")
            scaled = _fresh("scaled")
            lines.append(f"{indent}  {scaled} = arith.mulf {dot}, %sm2{fm} : f32")
            lines.append(f"{indent}  scf.yield {scaled} : f32")
            lines.append(f"{indent}}} else {{")
            lines.append(f"{indent}  scf.yield %neg_inf : f32")
            lines.append(f"{indent}}}")
            return pred_kv, score

        m_out = _fresh("m")
        l_out = _fresh("l")
        acc0_out = _fresh("acc0")
        acc1_out = _fresh("acc1")
        lines.append(
            f"        {m_out}, {l_out}, {acc0_out}, {acc1_out} = scf.for %tile0 = %c0 to %cKV step %cBlockKV "
            "iter_args(%m_i = %neg_inf, %l_i = %c0f, %a0 = %c0f, %a1 = %c0f) -> (f32, f32, f32, f32) {"
        )

        max_tile = _fresh("max_tile")
        lines.append(
            f"          {max_tile} = scf.for %t = %c0 to %cBlockKV step %c1 iter_args(%mx = %neg_inf) -> (f32) {{"
        )
        kv_ssa = _fresh("kv")
        lines.append(f"            {kv_ssa} = arith.addi %tile0, %t : index")
        _pred_kv, score = _emit_dot_score(kv_ssa=kv_ssa, indent="            ")
        mx_next = _fresh("mx_next")
        lines.append(f"            {mx_next} = arith.maximumf %mx, {score}{fm} : f32")
        lines.append(f"            scf.yield {mx_next} : f32")
        lines.append("          }")

        m_new = _fresh("m_new")
        lines.append(f"          {m_new} = arith.maximumf %m_i, {max_tile}{fm} : f32")
        delta = _fresh("delta")
        alpha = _fresh("alpha")
        lines.append(f"          {delta} = arith.subf %m_i, {m_new}{fm} : f32")
        lines.append(f"          {alpha} = math.exp2 {delta} : f32")
        l_scaled = _fresh("l_scaled")
        a0_scaled = _fresh("a0_scaled")
        a1_scaled = _fresh("a1_scaled")
        lines.append(f"          {l_scaled} = arith.mulf %l_i, {alpha}{fm} : f32")
        lines.append(f"          {a0_scaled} = arith.mulf %a0, {alpha}{fm} : f32")
        lines.append(f"          {a1_scaled} = arith.mulf %a1, {alpha}{fm} : f32")

        sum_out = _fresh("sum")
        a0_out = _fresh("a0")
        a1_out = _fresh("a1")
        lines.append(
            f"          {sum_out}, {a0_out}, {a1_out} = scf.for %t2 = %c0 to %cBlockKV step %c1 "
            f"iter_args(%s = %c0f, %b0 = {a0_scaled}, %b1 = {a1_scaled}) -> (f32, f32, f32) {{"
        )
        kv2 = _fresh("kv")
        lines.append(f"            {kv2} = arith.addi %tile0, %t2 : index")
        pred_kv2, score2 = _emit_dot_score(kv_ssa=kv2, indent="            ")
        shift = _fresh("shift")
        p = _fresh("p")
        lines.append(f"            {shift} = arith.subf {score2}, {m_new}{fm} : f32")
        lines.append(f"            {p} = math.exp2 {shift} : f32")
        s_next = _fresh("s_next")
        lines.append(f"            {s_next} = arith.addf %s, {p}{fm} : f32")

        pred_v0 = _fresh("pred_v0")
        pred_v1 = _fresh("pred_v1")
        lines.append(f"            {pred_v0} = arith.andi {pred_kv2}, %pred_d0 : i1")
        lines.append(f"            {pred_v1} = arith.andi {pred_kv2}, %pred_d1 : i1")
        base_vrow0 = _fresh("base_vrow0")
        base_vrow = _fresh("base_vrow")
        base_v = _fresh("base_v")
        lines.append(f"            {base_vrow0} = arith.muli %zh, %cKV : index")
        lines.append(f"            {base_vrow} = arith.addi {base_vrow0}, {kv2} : index")
        lines.append(f"            {base_v} = arith.muli {base_vrow}, %cHD : index")

        vv0 = _fresh("vv0")
        lines.append(f"            {vv0} = scf.if {pred_v0} -> (f32) {{")
        idx_v0 = _fresh("idx_v0")
        v0 = _fresh("v0")
        lines.append(f"              {idx_v0} = arith.addi {base_v}, %tid : index")
        lines.append(f"              {v0} = memref.load {arg_ssa[v_name]}[{idx_v0}] : {v_memref}")
        lines.append(f"              scf.yield {v0} : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")
        vv1 = _fresh("vv1")
        lines.append(f"            {vv1} = scf.if {pred_v1} -> (f32) {{")
        idx_v1 = _fresh("idx_v1")
        v1 = _fresh("v1")
        lines.append(f"              {idx_v1} = arith.addi {base_v}, %d1 : index")
        lines.append(f"              {v1} = memref.load {arg_ssa[v_name]}[{idx_v1}] : {v_memref}")
        lines.append(f"              scf.yield {v1} : f32")
        lines.append("            } else {")
        lines.append("              scf.yield %c0f : f32")
        lines.append("            }")

        pv0 = _fresh("pv0")
        pv1 = _fresh("pv1")
        b0_next = _fresh("b0_next")
        b1_next = _fresh("b1_next")
        lines.append(f"            {pv0} = arith.mulf {p}, {vv0}{fm} : f32")
        lines.append(f"            {pv1} = arith.mulf {p}, {vv1}{fm} : f32")
        lines.append(f"            {b0_next} = arith.addf %b0, {pv0}{fm} : f32")
        lines.append(f"            {b1_next} = arith.addf %b1, {pv1}{fm} : f32")
        lines.append(f"            scf.yield {s_next}, {b0_next}, {b1_next} : f32, f32, f32")
        lines.append("          }")

        l_new = _fresh("l_new")
        lines.append(f"          {l_new} = arith.addf {l_scaled}, {sum_out}{fm} : f32")
        lines.append(f"          scf.yield {m_new}, {l_new}, {a0_out}, {a1_out} : f32, f32, f32, f32")
        lines.append("        }")

        sum_nz = _fresh("sum_nz")
        l_safe = _fresh("l_safe")
        out0 = _fresh("out0")
        out1 = _fresh("out1")
        lines.append(f"        {sum_nz} = arith.cmpf one, {l_out}, %c0f : f32")
        lines.append(f"        {l_safe} = arith.select {sum_nz}, {l_out}, %c1f : f32")
        lines.append(f"        {out0} = arith.divf {acc0_out}, {l_safe}{fm} : f32")
        lines.append(f"        {out1} = arith.divf {acc1_out}, {l_safe}{fm} : f32")
        lines.append("        scf.if %pred_d0 {")
        idx_o0 = _fresh("idx_o0")
        lines.append(f"          {idx_o0} = arith.addi %base_q, %tid : index")
        lines.append(f"          memref.store {out0}, {arg_ssa[out_name2]}[{idx_o0}] : {out_memref}")
        lines.append("        }")
        lines.append("        scf.if %pred_d1 {")
        idx_o1 = _fresh("idx_o1")
        lines.append(f"          {idx_o1} = arith.addi %base_q, %d1 : index")
        lines.append(f"          memref.store {out1}, {arg_ssa[out_name2]}[{idx_o1}] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif _env_flag("INTENTIR_CUDA_REAL_MLIR_ATTN_FWD_V1", default=False) and attn_fwd_v1 is not None:
        kernel_kind = "attn_fwd_softmax_v1"
        q_name = str(attn_fwd_v1["Q"])
        k_name = str(attn_fwd_v1["K"])
        v_name = str(attn_fwd_v1["V"])
        out_name2 = str(attn_fwd_v1["out"])
        sm_scale_name = str(attn_fwd_v1["sm_scale"])
        z_dim = int(attn_fwd_v1["Z"])
        h_dim = int(attn_fwd_v1["num_head"])
        q_ctx = int(attn_fwd_v1["Q_CTX"])
        kv_ctx = int(attn_fwd_v1["KV_CTX"])
        hd = int(attn_fwd_v1["HEAD_DIM"])

        q_memref = str(arg_specs[q_name]["memref"])
        k_memref = str(arg_specs[k_name]["memref"])
        v_memref = str(arg_specs[v_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        sm_scale_memref = str(arg_specs[sm_scale_name]["memref"])

        blocks = int(z_dim) * int(h_dim) * int(q_ctx)
        # Same rationale as attn2d_v1: keep the dot-product reduction warp-sized
        # to avoid large-idle-thread gpu.barrier overhead.
        block_x = 32
        lanes_per_thread = int((int(hd) + int(block_x) - 1) // int(block_x))
        launch_override = {"block": [int(block_x), 1, 1], "grid": [int(blocks), 1, 1]}
        shared_global_sym = f"__intentir_sh_{_mlir_ident(kernel_name)}_f32"
        scratch_off = int(hd)
        scores_off = int(scratch_off + int(block_x))
        weights_off = int(scores_off + kv_ctx)
        sh_elems = int(weights_off + kv_ctx)
        shared_global_memref_ty = f"memref<{int(sh_elems)}xf32, 3>"

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append("      %c1 = arith.constant 1 : index")
        lines.append(f"      %cZ = arith.constant {int(z_dim)} : index")
        lines.append(f"      %cH = arith.constant {int(h_dim)} : index")
        lines.append(f"      %cQ = arith.constant {int(q_ctx)} : index")
        lines.append(f"      %cKV = arith.constant {int(kv_ctx)} : index")
        lines.append(f"      %cHD = arith.constant {int(hd)} : index")
        lines.append(f"      %cBX = arith.constant {int(block_x)} : index")
        lines.append(f"      %cBlocks = arith.constant {int(blocks)} : index")
        lines.append("      %pred_row = arith.cmpi ult, %bid, %cBlocks : index")
        lines.append("      scf.if %pred_row {")
        lines.append(f"        %sh = memref.get_global @{shared_global_sym} : {shared_global_memref_ty}")
        lines.append(f"        %sm = memref.load {arg_ssa[sm_scale_name]}[%c0] : {sm_scale_memref}")

        # Decode bid -> (z, head, q).
        lines.append("        %q = arith.remui %bid, %cQ : index")
        lines.append("        %tmp0 = arith.divui %bid, %cQ : index")
        lines.append("        %head = arith.remui %tmp0, %cH : index")
        lines.append("        %z = arith.divui %tmp0, %cH : index")

        # Load q vector into shared[0:HD].
        # Use distinct SSA names in the outer scope to avoid collisions with
        # later (nested) regions that also compute z/head helpers.
        lines.append("        %mul_zh_q = arith.muli %z, %cH : index")
        lines.append("        %zh_q = arith.addi %mul_zh_q, %head : index")
        lines.append("        %mul_qrow_q = arith.muli %zh_q, %cQ : index")
        lines.append("        %qrow_q = arith.addi %mul_qrow_q, %q : index")
        lines.append("        %base_q_q = arith.muli %qrow_q, %cHD : index")
        for lane in range(lanes_per_thread):
            if lane == 0:
                gd = "%tid"
            else:
                c_lane = _fresh(f"c_lane_q_{lane}")
                gd = _fresh(f"gd_q_{lane}")
                lines.append(f"        {c_lane} = arith.constant {int(lane * int(block_x))} : index")
                lines.append(f"        {gd} = arith.addi %tid, {c_lane} : index")
            pred = _fresh(f"pred_qd2_{lane}")
            idx = _fresh(f"idx_q2_{lane}")
            qv = _fresh(f"qv2_{lane}")
            lines.append(f"        {pred} = arith.cmpi ult, {gd}, %cHD : index")
            lines.append(f"        scf.if {pred} {{")
            lines.append(f"          {idx} = arith.addi %base_q_q, {gd} : index")
            lines.append(f"          {qv} = memref.load {arg_ssa[q_name]}[{idx}] : {q_memref}")
            lines.append(f"          memref.store {qv}, %sh[{gd}] : {shared_global_memref_ty}")
            lines.append("        }")
        lines.append("        gpu.barrier")

        lines.append(f"        %cScratch = arith.constant {int(scratch_off)} : index")
        lines.append(f"        %cScores = arith.constant {int(scores_off)} : index")
        lines.append(f"        %cWeights = arith.constant {int(weights_off)} : index")
        lines.append("        %c0f = arith.constant 0.0 : f32")

        # Compute scores[kv] = dot(Q[q,:], K[kv,:]) * sm_scale (no causal mask).
        lines.append("        scf.for %kv = %c0 to %cKV step %c1 {")
        lines.append("          %mul_krow = arith.muli %zh_q, %cKV : index")
        lines.append("          %krow = arith.addi %mul_krow, %kv : index")
        lines.append("          %base_k = arith.muli %krow, %cHD : index")
        lane_ps: list[str] = []
        for lane in range(lanes_per_thread):
            if lane == 0:
                gd = "%tid"
            else:
                c_lane = _fresh(f"c_lane_k2_{lane}")
                gd = _fresh(f"gd_k2_{lane}")
                lines.append(f"          {c_lane} = arith.constant {int(lane * int(block_x))} : index")
                lines.append(f"          {gd} = arith.addi %tid, {c_lane} : index")
            pred_d = _fresh(f"pred_d2_{lane}")
            prod_lane = _fresh(f"prod2_{lane}")
            idx_k = _fresh(f"idx_k2_{lane}")
            kvv = _fresh(f"kvv2_{lane}")
            qvv = _fresh(f"qvv2_{lane}")
            p = _fresh(f"p2_{lane}")
            lines.append(f"          {pred_d} = arith.cmpi ult, {gd}, %cHD : index")
            lines.append(f"          {prod_lane} = scf.if {pred_d} -> (f32) {{")
            lines.append(f"            {idx_k} = arith.addi %base_k, {gd} : index")
            lines.append(f"            {kvv} = memref.load {arg_ssa[k_name]}[{idx_k}] : {k_memref}")
            lines.append(f"            {qvv} = memref.load %sh[{gd}] : {shared_global_memref_ty}")
            lines.append(f"            {p} = arith.mulf {kvv}, {qvv}{fm} : f32")
            lines.append(f"            scf.yield {p} : f32")
            lines.append("          } else {")
            lines.append("            scf.yield %c0f : f32")
            lines.append("          }")
            lane_ps.append(str(prod_lane))
        if len(lane_ps) == 1:
            lines.append(f"          %prod = arith.addf {lane_ps[0]}, %c0f{fm} : f32")
        else:
            acc = lane_ps[0]
            for i, p_lane in enumerate(lane_ps[1:], start=1):
                s = _fresh(f"sum_lane2_{i}")
                lines.append(f"          {s} = arith.addf {acc}, {p_lane}{fm} : f32")
                acc = str(s)
            lines.append(f"          %prod = arith.addf {acc}, %c0f{fm} : f32")
        lines.append("          %sidx = arith.addi %cScratch, %tid : index")
        lines.append(f"          memref.store %prod, %sh[%sidx] : {shared_global_memref_ty}")
        lines.append("          gpu.barrier")
        dot_strides: list[int] = []
        s = int(block_x) // 2
        while s >= 1:
            dot_strides.append(int(s))
            s //= 2
        for stride in dot_strides:
            cS = f"%cS_dot2_{stride}"
            pS = f"%pS_dot2_{stride}"
            tid2 = f"%tid_dot2_{stride}"
            a = f"%a_dot2_{stride}"
            b = f"%b_dot2_{stride}"
            s = f"%s_dot2_{stride}"
            idx_a = f"%idx_a_dot2_{stride}"
            idx_b = f"%idx_b_dot2_{stride}"
            lines.append(f"          {cS} = arith.constant {int(stride)} : index")
            lines.append(f"          {pS} = arith.cmpi ult, %tid, {cS} : index")
            lines.append(f"          scf.if {pS} {{")
            lines.append(f"            {tid2} = arith.addi %tid, {cS} : index")
            lines.append(f"            {idx_a} = arith.addi %cScratch, %tid : index")
            lines.append(f"            {idx_b} = arith.addi %cScratch, {tid2} : index")
            lines.append(f"            {a} = memref.load %sh[{idx_a}] : {shared_global_memref_ty}")
            lines.append(f"            {b} = memref.load %sh[{idx_b}] : {shared_global_memref_ty}")
            lines.append(f"            {s} = arith.addf {a}, {b}{fm} : f32")
            lines.append(f"            memref.store {s}, %sh[{idx_a}] : {shared_global_memref_ty}")
            lines.append("          }")
            lines.append("          gpu.barrier")
        lines.append("          %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("          scf.if %is0 {")
        lines.append(f"            %sum = memref.load %sh[%cScratch] : {shared_global_memref_ty}")
        lines.append(f"            %scaled = arith.mulf %sum, %sm{fm} : f32")
        lines.append("            %off = arith.addi %cScores, %kv : index")
        lines.append(f"            memref.store %scaled, %sh[%off] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("          gpu.barrier")
        lines.append("        }")

        # Softmax normalization on scores -> weights.
        lines.append("        %is0 = arith.cmpi eq, %tid, %c0 : index")
        lines.append("        scf.if %is0 {")
        lines.append("          %init_max = arith.constant -3.402823466e+38 : f32")
        lines.append("          %maxv = scf.for %k = %c0 to %cKV step %c1 iter_args(%acc = %init_max) -> (f32) {")
        lines.append("            %off = arith.addi %cScores, %k : index")
        lines.append(f"            %x = memref.load %sh[%off] : {shared_global_memref_ty}")
        lines.append(f"            %acc_next = arith.maximumf %acc, %x{fm} : f32")
        lines.append("            scf.yield %acc_next : f32")
        lines.append("          }")
        lines.append("          %sumv = scf.for %k2 = %c0 to %cKV step %c1 iter_args(%acc = %c0f) -> (f32) {")
        lines.append("            %off2 = arith.addi %cScores, %k2 : index")
        lines.append(f"            %x2 = memref.load %sh[%off2] : {shared_global_memref_ty}")
        lines.append(f"            %xc = arith.subf %x2, %maxv{fm} : f32")
        lines.append(f"            %e = math.exp %xc{fm} : f32")
        lines.append("            %wo = arith.addi %cWeights, %k2 : index")
        lines.append(f"            memref.store %e, %sh[%wo] : {shared_global_memref_ty}")
        lines.append(f"            %acc_next2 = arith.addf %acc, %e{fm} : f32")
        lines.append("            scf.yield %acc_next2 : f32")
        lines.append("          }")
        lines.append("          %c1f = arith.constant 1.0 : f32")
        lines.append("          %sum_nz = arith.cmpf one, %sumv, %c0f : f32")
        lines.append("          %sum_safe = arith.select %sum_nz, %sumv, %c1f : f32")
        lines.append("          scf.for %k3 = %c0 to %cKV step %c1 {")
        lines.append("            %wo3 = arith.addi %cWeights, %k3 : index")
        lines.append(f"            %e3 = memref.load %sh[%wo3] : {shared_global_memref_ty}")
        lines.append(f"            %p3 = arith.divf %e3, %sum_safe{fm} : f32")
        lines.append(f"            memref.store %p3, %sh[%wo3] : {shared_global_memref_ty}")
        lines.append("          }")
        lines.append("        }")
        lines.append("        gpu.barrier")

        # Weighted sum: Out[z,head,q,d] = sum_k weights[k] * V[z,head,k,d]
        lines.append("        %pred_od = arith.cmpi ult, %tid, %cHD : index")
        lines.append("        scf.if %pred_od {")
        lines.append("          %acc = scf.for %k4 = %c0 to %cKV step %c1 iter_args(%a = %c0f) -> (f32) {")
        lines.append("            %wo4 = arith.addi %cWeights, %k4 : index")
        lines.append(f"            %w = memref.load %sh[%wo4] : {shared_global_memref_ty}")
        lines.append("            %mul_zh = arith.muli %z, %cH : index")
        lines.append("            %zh = arith.addi %mul_zh, %head : index")
        lines.append("            %mul_vrow = arith.muli %zh, %cKV : index")
        lines.append("            %vrow = arith.addi %mul_vrow, %k4 : index")
        lines.append("            %base_v = arith.muli %vrow, %cHD : index")
        lines.append("            %idx_v = arith.addi %base_v, %tid : index")
        lines.append(f"            %vv = memref.load {arg_ssa[v_name]}[%idx_v] : {v_memref}")
        lines.append(f"            %p = arith.mulf %w, %vv{fm} : f32")
        lines.append(f"            %a2 = arith.addf %a, %p{fm} : f32")
        lines.append("            scf.yield %a2 : f32")
        lines.append("          }")
        lines.append("          %mul_zh = arith.muli %z, %cH : index")
        lines.append("          %zh = arith.addi %mul_zh, %head : index")
        lines.append("          %mul_qrow = arith.muli %zh, %cQ : index")
        lines.append("          %qrow = arith.addi %mul_qrow, %q : index")
        lines.append("          %base_o = arith.muli %qrow, %cHD : index")
        lines.append("          %idx_o = arith.addi %base_o, %tid : index")
        lines.append(f"          memref.store %acc, {arg_ssa[out_name2]}[%idx_o] : {out_memref}")
        lines.append("        }")
        lines.append("      }")
    elif upsample_bicubic2d_aa_v1 is not None:
        kernel_kind = "upsample_bicubic2d_aa_v1"
        in_name = str(upsample_bicubic2d_aa_v1["inp"])
        out_name2 = str(upsample_bicubic2d_aa_v1["out"])
        rs_h_name = str(upsample_bicubic2d_aa_v1["reciprocal_scale_h"])
        rs_w_name = str(upsample_bicubic2d_aa_v1["reciprocal_scale_w"])
        n_dim = int(upsample_bicubic2d_aa_v1["N"])
        c_dim = int(upsample_bicubic2d_aa_v1["C"])
        ih_dim = int(upsample_bicubic2d_aa_v1["IH"])
        iw_dim = int(upsample_bicubic2d_aa_v1["IW"])
        oh_dim = int(upsample_bicubic2d_aa_v1["OH"])
        ow_dim = int(upsample_bicubic2d_aa_v1["OW"])
        a_param = float(upsample_bicubic2d_aa_v1.get("a") or -0.5)
        support = float(upsample_bicubic2d_aa_v1.get("support") or 2.0)
        invscale = float(upsample_bicubic2d_aa_v1.get("invscale") or 1.0)

        in_memref = str(arg_specs[in_name]["memref"])
        out_memref = str(arg_specs[out_name2]["memref"])
        rs_h_memref = str(arg_specs[rs_h_name]["memref"])
        rs_w_memref = str(arg_specs[rs_w_name]["memref"])

        grid_x = int((int(out_total) + 255) // 256)
        launch_override = {"block": [256, 1, 1], "grid": [int(grid_x), 1, 1]}

        lines.append("      %tid = gpu.thread_id x")
        lines.append("      %bid = gpu.block_id x")
        lines.append("      %bdim = gpu.block_dim x")
        lines.append("      %tmp = arith.muli %bid, %bdim : index")
        lines.append("      %lin = arith.addi %tmp, %tid : index")
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %c_total = arith.constant {int(out_total)} : index")
        lines.append("      %pred = arith.cmpi ult, %lin, %c_total : index")
        lines.append("      scf.if %pred {")
        lines.append("        %c1 = arith.constant 1 : index")
        lines.append(f"        %cN = arith.constant {int(n_dim)} : index")
        lines.append(f"        %cC = arith.constant {int(c_dim)} : index")
        lines.append(f"        %cIH = arith.constant {int(ih_dim)} : index")
        lines.append(f"        %cIW = arith.constant {int(iw_dim)} : index")
        lines.append(f"        %cOH = arith.constant {int(oh_dim)} : index")
        lines.append(f"        %cOW = arith.constant {int(ow_dim)} : index")
        lines.append(f"        %rs_h = memref.load {arg_ssa[rs_h_name]}[%c0] : {rs_h_memref}")
        lines.append(f"        %rs_w = memref.load {arg_ssa[rs_w_name]}[%c0] : {rs_w_memref}")
        lines.append("        %c0f = arith.constant 0.0 : f32")
        lines.append("        %c1f = arith.constant 1.0 : f32")
        lines.append(f"        %cHalf = arith.constant {_as_f32_const(0.5)} : f32")
        lines.append(f"        %cSupport = arith.constant {_as_f32_const(support)} : f32")
        lines.append(f"        %cInvscale = arith.constant {_as_f32_const(invscale)} : f32")
        lines.append(f"        %cA = arith.constant {_as_f32_const(a_param)} : f32")

        # Decode linear idx -> (n,c,oh,ow) for NCHW.
        lines.append("        %ow = arith.remui %lin, %cOW : index")
        lines.append("        %t1 = arith.divui %lin, %cOW : index")
        lines.append("        %oh = arith.remui %t1, %cOH : index")
        lines.append("        %t2 = arith.divui %t1, %cOH : index")
        lines.append("        %cc = arith.remui %t2, %cC : index")
        lines.append("        %nn = arith.divui %t2, %cC : index")

        # float(oh/ow)
        lines.append("        %ow_i32 = arith.index_cast %ow : index to i32")
        lines.append("        %oh_i32 = arith.index_cast %oh : index to i32")
        lines.append("        %ow_f = arith.sitofp %ow_i32 : i32 to f32")
        lines.append("        %oh_f = arith.sitofp %oh_i32 : i32 to f32")

        # center = (o + 0.5) * reciprocal_scale
        lines.append(f"        %ow_p = arith.addf %ow_f, %cHalf{fm} : f32")
        lines.append(f"        %oh_p = arith.addf %oh_f, %cHalf{fm} : f32")
        lines.append(f"        %center_w = arith.mulf %ow_p, %rs_w{fm} : f32")
        lines.append(f"        %center_h = arith.mulf %oh_p, %rs_h{fm} : f32")

        # span_start = max(center - support + 0.5, 0).to i32
        lines.append(f"        %tmp_w0 = arith.subf %center_w, %cSupport{fm} : f32")
        lines.append(f"        %tmp_w1 = arith.addf %tmp_w0, %cHalf{fm} : f32")
        lines.append("        %c0f2 = arith.constant 0.0 : f32")
        lines.append(f"        %tmp_w2 = arith.maximumf %tmp_w1, %c0f2{fm} : f32")
        lines.append("        %span_start_w = arith.fptosi %tmp_w2 : f32 to i32")

        lines.append(f"        %tmp_h0 = arith.subf %center_h, %cSupport{fm} : f32")
        lines.append(f"        %tmp_h1 = arith.addf %tmp_h0, %cHalf{fm} : f32")
        lines.append(f"        %tmp_h2 = arith.maximumf %tmp_h1, %c0f2{fm} : f32")
        lines.append("        %span_start_h = arith.fptosi %tmp_h2 : f32 to i32")

        # span_size = (min(center + support + 0.5, I) - span_start).to i32
        lines.append("        %iw_i32 = arith.constant " + str(int(iw_dim)) + " : i32")
        lines.append("        %ih_i32 = arith.constant " + str(int(ih_dim)) + " : i32")
        lines.append("        %iw_f = arith.sitofp %iw_i32 : i32 to f32")
        lines.append("        %ih_f = arith.sitofp %ih_i32 : i32 to f32")
        lines.append(f"        %tmp_w3 = arith.addf %center_w, %cSupport{fm} : f32")
        lines.append(f"        %tmp_w4 = arith.addf %tmp_w3, %cHalf{fm} : f32")
        lines.append(f"        %tmp_w5 = arith.minimumf %tmp_w4, %iw_f{fm} : f32")
        lines.append("        %ssw_f = arith.sitofp %span_start_w : i32 to f32")
        lines.append(f"        %span_w_f = arith.subf %tmp_w5, %ssw_f{fm} : f32")
        lines.append("        %span_size_w = arith.fptosi %span_w_f : f32 to i32")

        lines.append(f"        %tmp_h3 = arith.addf %center_h, %cSupport{fm} : f32")
        lines.append(f"        %tmp_h4 = arith.addf %tmp_h3, %cHalf{fm} : f32")
        lines.append(f"        %tmp_h5 = arith.minimumf %tmp_h4, %ih_f{fm} : f32")
        lines.append("        %ssh_f = arith.sitofp %span_start_h : i32 to f32")
        lines.append(f"        %span_h_f = arith.subf %tmp_h5, %ssh_f{fm} : f32")
        lines.append("        %span_size_h = arith.fptosi %span_h_f : f32 to i32")

        # start_minus_center = span_start - center
        lines.append(f"        %start_minus_center_w = arith.subf %ssw_f, %center_w{fm} : f32")
        lines.append(f"        %start_minus_center_h = arith.subf %ssh_f, %center_h{fm} : f32")

        # Keys cubic (a=-0.5) piecewise poly:
        #   t < 1: ((a+2)*t - (a+3))*t*t + 1
        #   1 <= t < 2: a * (t^3 - 5*t^2 + 8*t - 4)
        cA2 = _fresh("cA2")
        cA3 = _fresh("cA3")
        c2f = _fresh("c2f")
        c5f = _fresh("c5f")
        c8f = _fresh("c8f")
        c4f = _fresh("c4f")
        lines.append(f"        {cA2} = arith.constant {_as_f32_const(a_param + 2.0)} : f32")
        lines.append(f"        {cA3} = arith.constant {_as_f32_const(a_param + 3.0)} : f32")
        lines.append(f"        {c2f} = arith.constant 2.0 : f32")
        lines.append(f"        {c5f} = arith.constant 5.0 : f32")
        lines.append(f"        {c8f} = arith.constant 8.0 : f32")
        lines.append(f"        {c4f} = arith.constant 4.0 : f32")

        def _emit_keys_cubic_weight(tag: str, t_ssa: str) -> str:
            t2 = _fresh(f"{tag}_t2")
            t3 = _fresh(f"{tag}_t3")
            tmp0 = _fresh(f"{tag}_tmp0")
            tmp1 = _fresh(f"{tag}_tmp1")
            tmp2 = _fresh(f"{tag}_tmp2")
            poly1 = _fresh(f"{tag}_poly1")
            tmp3 = _fresh(f"{tag}_tmp3")
            tmp4 = _fresh(f"{tag}_tmp4")
            tmp5 = _fresh(f"{tag}_tmp5")
            tmp6 = _fresh(f"{tag}_tmp6")
            tmp7 = _fresh(f"{tag}_tmp7")
            poly2 = _fresh(f"{tag}_poly2")
            lt1 = _fresh(f"{tag}_lt1")
            lt2 = _fresh(f"{tag}_lt2")
            sel2 = _fresh(f"{tag}_sel2")
            out = _fresh(f"{tag}_w")
            lines.append(f"        {t2} = arith.mulf {t_ssa}, {t_ssa}{fm} : f32")
            lines.append(f"        {t3} = arith.mulf {t2}, {t_ssa}{fm} : f32")
            # poly1 = ((a+2)*t - (a+3))*t*t + 1
            lines.append(f"        {tmp0} = arith.mulf {cA2}, {t_ssa}{fm} : f32")
            lines.append(f"        {tmp1} = arith.subf {tmp0}, {cA3}{fm} : f32")
            lines.append(f"        {tmp2} = arith.mulf {tmp1}, {t2}{fm} : f32")
            lines.append(f"        {poly1} = arith.addf {tmp2}, %c1f{fm} : f32")
            # poly2 = a*(t^3 - 5*t^2 + 8*t - 4)
            lines.append(f"        {tmp3} = arith.subf {t_ssa}, {c5f}{fm} : f32")
            lines.append(f"        {tmp4} = arith.mulf {tmp3}, {t_ssa}{fm} : f32")
            lines.append(f"        {tmp5} = arith.addf {tmp4}, {c8f}{fm} : f32")
            lines.append(f"        {tmp6} = arith.mulf {tmp5}, {t_ssa}{fm} : f32")
            lines.append(f"        {tmp7} = arith.subf {tmp6}, {c4f}{fm} : f32")
            lines.append(f"        {poly2} = arith.mulf %cA, {tmp7}{fm} : f32")
            # piecewise
            lines.append(f"        {lt1} = arith.cmpf olt, {t_ssa}, %c1f : f32")
            lines.append(f"        {lt2} = arith.cmpf olt, {t_ssa}, {c2f} : f32")
            lines.append(f"        {sel2} = arith.select {lt2}, {poly2}, %c0f : f32")
            lines.append(f"        {out} = arith.select {lt1}, {poly1}, {sel2} : f32")
            return out

        # Weights X/Y (5 taps) with gating by k<span_size.
        wx: list[str] = []
        wy: list[str] = []
        mx: list[str] = []
        my: list[str] = []
        ix: list[str] = []
        iy: list[str] = []
        for axis, span_start, span_size, smc, dim_i32, wlist, mlist, ilist in [
            ("x", "%span_start_w", "%span_size_w", "%start_minus_center_w", "%iw_i32", wx, mx, ix),
            ("y", "%span_start_h", "%span_size_h", "%start_minus_center_h", "%ih_i32", wy, my, iy),
        ]:
            for k in range(5):
                k_i32 = _fresh(f"k{axis}{k}_i32")
                k_f = _fresh(f"k{axis}{k}_f")
                kt = _fresh(f"t{axis}{k}_0")
                kt2 = _fresh(f"t{axis}{k}_1")
                kt3 = _fresh(f"t{axis}{k}_2")
                kt_abs = _fresh(f"t{axis}{k}_abs")
                k_lt = _fresh(f"klt_{axis}{k}")
                w_gated = _fresh(f"w_{axis}{k}_g")
                idx_i32 = _fresh(f"idx_{axis}{k}_i32")
                mask_i1 = _fresh(f"mask_{axis}{k}")
                idx_idx = _fresh(f"idx_{axis}{k}")
                lines.append(f"        {k_i32} = arith.constant {int(k)} : i32")
                lines.append(f"        {k_lt} = arith.cmpi slt, {k_i32}, {span_size} : i32")
                lines.append(f"        {k_f} = arith.sitofp {k_i32} : i32 to f32")
                lines.append(f"        {kt} = arith.addf {k_f}, {smc}{fm} : f32")
                lines.append(f"        {kt2} = arith.addf {kt}, %cHalf{fm} : f32")
                lines.append(f"        {kt3} = arith.mulf {kt2}, %cInvscale{fm} : f32")
                lines.append(f"        {kt_abs} = math.absf {kt3}{fm} : f32")
                w_raw = _emit_keys_cubic_weight(f"w{axis}{k}", kt_abs)
                lines.append(f"        {w_gated} = arith.select {k_lt}, {w_raw}, %c0f : f32")
                lines.append(f"        {idx_i32} = arith.addi {span_start}, {k_i32} : i32")
                lines.append(f"        {mask_i1} = arith.cmpi slt, {idx_i32}, {dim_i32} : i32")
                lines.append(f"        {idx_idx} = arith.index_cast {idx_i32} : i32 to index")
                wlist.append(w_gated)
                mlist.append(mask_i1)
                ilist.append(idx_idx)

        # Normalize weights (avoid div-by-zero).
        def _norm_weights(tag: str, w: list[str]) -> list[str]:
            if len(w) != 5:
                return w
            total = _fresh(f"{tag}_tot")
            tmp = _fresh(f"{tag}_t1")
            tmp2 = _fresh(f"{tag}_t2")
            tmp3 = _fresh(f"{tag}_t3")
            lines.append(f"        {tmp} = arith.addf {w[0]}, {w[1]}{fm} : f32")
            lines.append(f"        {tmp2} = arith.addf {tmp}, {w[2]}{fm} : f32")
            lines.append(f"        {tmp3} = arith.addf {tmp2}, {w[3]}{fm} : f32")
            lines.append(f"        {total} = arith.addf {tmp3}, {w[4]}{fm} : f32")
            nz = _fresh(f"{tag}_nz")
            safe = _fresh(f"{tag}_safe")
            lines.append(f"        {nz} = arith.cmpf one, {total}, %c0f : f32")
            lines.append(f"        {safe} = arith.select {nz}, {total}, %c1f : f32")
            out_w: list[str] = []
            for i in range(5):
                wi = _fresh(f"{tag}_w{i}")
                lines.append(f"        {wi} = arith.divf {w[i]}, {safe}{fm} : f32")
                out_w.append(wi)
            return out_w

        wxn = _norm_weights("wx", wx)
        wyn = _norm_weights("wy", wy)

        # Compute base pointer offset for this (n,c): base_nc0 = (n*C + c)*IH
        lines.append("        %nc0 = arith.muli %nn, %cC : index")
        lines.append("        %nc = arith.addi %nc0, %cc : index")
        lines.append("        %base_nc0 = arith.muli %nc, %cIH : index")

        # Accumulate 5x5 separable bicubic.
        lines.append("        %acc0 = arith.constant 0.0 : f32")
        acc = "%acc0"
        for ky in range(5):
            row_acc = _fresh(f"row{ky}")
            lines.append(f"        {row_acc} = arith.constant 0.0 : f32")
            for kx in range(5):
                m2 = _fresh(f"m2_{ky}_{kx}")
                lines.append(f"        {m2} = arith.andi {my[ky]}, {mx[kx]} : i1")
                val = _fresh(f"val_{ky}_{kx}")
                lines.append(f"        {val} = scf.if {m2} -> (f32) {{")
                lines.append(f"          %row = arith.addi %base_nc0, {iy[ky]} : index")
                lines.append("          %row_base = arith.muli %row, %cIW : index")
                lines.append(f"          %idx_in = arith.addi %row_base, {ix[kx]} : index")
                lines.append(f"          %v = memref.load {arg_ssa[in_name]}[%idx_in] : {in_memref}")
                lines.append("          scf.yield %v : f32")
                lines.append("        } else {")
                lines.append("          scf.yield %c0f : f32")
                lines.append("        }")
                contrib = _fresh(f"cx_{ky}_{kx}")
                lines.append(f"        {contrib} = arith.mulf {val}, {wxn[kx]}{fm} : f32")
                row_next = _fresh(f"row{ky}_{kx}")
                lines.append(f"        {row_next} = arith.addf {row_acc}, {contrib}{fm} : f32")
                row_acc = row_next
            rowy = _fresh(f"rowy{ky}")
            acc_next = _fresh(f"acc{ky}")
            lines.append(f"        {rowy} = arith.mulf {row_acc}, {wyn[ky]}{fm} : f32")
            lines.append(f"        {acc_next} = arith.addf {acc}, {rowy}{fm} : f32")
            acc = str(acc_next)

        lines.append(f"        memref.store {acc}, {arg_ssa[out_name2]}[%lin] : {out_memref}")
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
            precomputed: dict[str, tuple[str, str]] | None = None
            if intent_name == "logspace1d":
                # `logspace1d` has multiple scalar inputs; loading them per-element is
                # expensive for tiny N (e.g. N=64). Broadcast lane0 scalar loads across
                # each warp via `gpu.shuffle idx`.
                precomputed = {}
                c32_idx = _fresh("c32_idx")
                lane = _fresh("lane")
                is_lane0 = _fresh("is_lane0")
                c0_i32 = _fresh("c0_i32")
                c32_i32 = _fresh("c32_i32")
                c0f = _fresh("c0f")
                lines.append(f"      {c32_idx} = arith.constant 32 : index")
                lines.append(f"      {lane} = arith.remui %tid, {c32_idx} : index")
                lines.append(f"      {is_lane0} = arith.cmpi eq, {lane}, %c0 : index")
                lines.append(f"      {c0_i32} = arith.constant 0 : i32")
                lines.append(f"      {c32_i32} = arith.constant 32 : i32")
                lines.append(f"      {c0f} = arith.constant 0.0 : f32")
                for scalar_name in ("start", "end", "denom", "log_base"):
                    if scalar_name not in arg_specs:
                        continue
                    if str(arg_specs[scalar_name].get("broadcast") or "") != "scalar":
                        continue
                    scalar_memref = str(arg_specs[scalar_name]["memref"])
                    lane0 = _fresh(f"{scalar_name}_lane0")
                    loaded = _fresh(f"{scalar_name}_loaded")
                    sh = _fresh(f"{scalar_name}_sh")
                    ok = _fresh(f"{scalar_name}_ok")
                    lines.append(f"      {lane0} = scf.if {is_lane0} -> (f32) {{")
                    lines.append(f"        {loaded} = memref.load {arg_ssa[scalar_name]}[%c0] : {scalar_memref}")
                    lines.append(f"        scf.yield {loaded} : f32")
                    lines.append("      } else {")
                    lines.append(f"        scf.yield {c0f} : f32")
                    lines.append("      }")
                    lines.append(f"      {sh}, {ok} = gpu.shuffle idx {lane0}, {c0_i32}, {c32_i32} : f32")
                    precomputed[str(scalar_name)] = (str(sh), "f32")
            lines.append(f"      %pred = arith.cmpi ult, {idx_ssa}, %c_total : index")
            lines.append("      scf.if %pred {")
            lines.extend(_emit_elementwise_for_index(idx_ssa, precomputed=precomputed))
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
    if shared_global_sym and shared_global_memref_ty:
        lines.insert(
            gpu_module_sym_insert,
            f'    memref.global "private" @{shared_global_sym} : {shared_global_memref_ty} {{alignment = 16}}',
        )
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
    if cuda_real_mlir_attention_cfg:
        out.meta["cuda_real_mlir_attention_cfg"] = dict(cuda_real_mlir_attention_cfg)
    if cuda_real_mlir_matmul_cfg:
        out.meta["cuda_real_mlir_matmul_cfg"] = dict(cuda_real_mlir_matmul_cfg)
    if launch_override:
        out.meta["cuda_real_mlir_launch_override"] = dict(launch_override)
    if repair_actions:
        out.meta["cuda_real_mlir_intent_repair_actions"] = list(repair_actions)
    return out


__all__ = ["lower_intent_to_cuda_gpu_kernel"]
