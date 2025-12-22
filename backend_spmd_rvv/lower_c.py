"""
Generic lowering: IntentIR ops -> plain C (compiled with -march=rv64gcv).

This is not a decompiler from TTIR. It assumes we already have a Verified IntentIR
and concrete shape bindings (from the real Triton launch artifacts).

Runtime protocol (remote runner writes these files):
  - External inputs:   <name>.bin
  - Reference outputs: <name>_ref.bin

Binary format:
  - float tensors:  raw float32, C-order
  - bool/i1 tensors: raw uint8 (0/1), C-order
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from intent_ir.ir_types import IntentFunction, IntentIRValidationError, Op


def lower_intent_to_c_with_files(
    intent: IntentFunction,
    *,
    shape_bindings: Dict[str, int],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    use_rvv: bool = True,
) -> str:
    produced = {op.output for op in intent.ops if op.output}
    used: set[str] = set()
    for op in intent.ops:
        for n in op.inputs:
            used.add(n)
    external_inputs = sorted([n for n in used if n in intent.tensors and n not in produced])
    outputs = list(intent.outputs)

    # Resolve declared tensor shapes to concrete ints.
    shape_env: Dict[str, Tuple[int, ...]] = {}
    dtype_env: Dict[str, str] = {}
    for name, t in intent.tensors.items():
        shape_env[name] = _resolve_shape_tuple(t.shape, shape_bindings)
        dtype_env[name] = t.dtype

    # Infer missing intermediate tensors (LLM often omits them from `tensors`).
    for op in intent.ops:
        if not op.output:
            raise IntentIRValidationError(f"lowering: op missing output: {op}")
        if op.op == "const":
            shape_env.setdefault(op.output, ())
            dtype_env.setdefault(op.output, "f32")
            continue
        if op.output in shape_env:
            # Enforce semantic dtypes for certain ops even if declared wrongly.
            if op.op in {"ne", "reduce_any"}:
                dtype_env[op.output] = "bool"
            if op.op in {"reduce_sum", "reduce_max", "softmax", "matmul", "add", "sub", "mul", "div", "max", "min", "rsqrt"}:
                dtype_env.setdefault(op.output, "f32")
            continue
        out_shape, out_dtype = _infer_output_shape_dtype(op, shape_env, dtype_env, shape_bindings)
        shape_env[op.output] = out_shape
        dtype_env[op.output] = out_dtype

    # Validate we can bind all external tensors.
    for name in external_inputs + outputs:
        if name not in shape_env:
            raise IntentIRValidationError(f"lowering: missing tensor declaration for {name}")
        _ = _numel(shape_env[name])

    # Const values resolved in Python (so we don't need expr parsing in C).
    const_values: Dict[str, float] = {}
    for op in intent.ops:
        if op.op == "const":
            v = op.attrs.get("value")
            const_values[op.output] = float(_resolve_const_value(v, shape_bindings))

    # Track C variable names and ownership.
    cvar: Dict[str, str] = {}
    ctype: Dict[str, str] = {}

    def _ctype_for(name: str) -> str:
        dt = dtype_env.get(name, "f32")
        if dt in {"bool", "i1"}:
            return "uint8_t"
        return "float"

    def _decl_ptr(name: str) -> str:
        return f"{_ctype_for(name)}* {name}"

    # Helpers to decide scalar broadcasting.
    def _is_scalar(name: str) -> bool:
        return _numel(shape_env.get(name, ())) == 1

    lines: List[str] = []
    lines += [
        "#include <math.h>",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#if defined(__riscv_vector) || defined(__riscv_v)\n#include <riscv_vector.h>\n#endif",
        "",
        "static inline float cubic_weight(float t, float a) {",
        "  float x = fabsf(t);",
        "  if (x < 1.0f) {",
        "    return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;",
        "  }",
        "  if (x < 2.0f) {",
        "    return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;",
        "  }",
        "  return 0.0f;",
        "}",
        "",
        "static int read_bytes(const char* path, void* dst, size_t bytes) {",
        "  FILE* f = fopen(path, \"rb\");",
        "  if (!f) { perror(path); return 0; }",
        "  size_t got = fread(dst, 1, bytes, f);",
        "  fclose(f);",
        "  return got == bytes;",
        "}",
        "",
        "static int compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol) {",
        "  double max_abs = 0.0, max_rel = 0.0; size_t worst = 0;",
        "  for (size_t i = 0; i < n; ++i) {",
        "    double a = (double)got[i];",
        "    double b = (double)ref[i];",
        "    double abs_e = fabs(a - b);",
        "    double rel_e = abs_e / (fabs(b) + 1e-8);",
        "    if (abs_e > max_abs) { max_abs = abs_e; max_rel = rel_e; worst = i; }",
        "  }",
        "  int ok = (max_abs <= (double)atol) || (max_rel <= (double)rtol);",
        "  printf(\"%s: ok=%d max_abs=%g max_rel=%g worst_i=%zu got=%g ref=%g\\n\",",
        "         name, ok, max_abs, max_rel, worst, (double)got[worst], (double)ref[worst]);",
        "  return ok;",
        "}",
        "",
        "static int compare_u8(const char* name, const uint8_t* got, const uint8_t* ref, size_t n) {",
        "  for (size_t i = 0; i < n; ++i) {",
        "    if (got[i] != ref[i]) {",
        "      fprintf(stderr, \"%s mismatch at %zu: got=%u ref=%u\\n\", name, i, (unsigned)got[i], (unsigned)ref[i]);",
        "      return 0;",
        "    }",
        "  }",
        "  printf(\"%s: ok=1 (exact)\\n\", name);",
        "  return 1;",
        "}",
        "",
        "static inline size_t idx2(int i, int j, int D1) { return (size_t)i * (size_t)D1 + (size_t)j; }",
        "static inline size_t idx3(int i, int j, int k, int D1, int D2) { return ((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k; }",
        "static inline size_t idx4(int i, int j, int k, int l, int D1, int D2, int D3) { return (((size_t)i * (size_t)D1 + (size_t)j) * (size_t)D2 + (size_t)k) * (size_t)D3 + (size_t)l; }",
        "",
        "int main() {",
        "  // Make remote-captured output deterministic (avoid stdio buffering surprises over SSH pipes).",
        "  setvbuf(stdout, NULL, _IONBF, 0);",
        "  setvbuf(stderr, NULL, _IONBF, 0);",
        f"  const float ATOL = {_c_float(float(atol))};",
        f"  const float RTOL = {_c_float(float(rtol))};",
        "",
    ]

    # 1) Read external inputs
    for name in external_inputs:
        shp = shape_env[name]
        n = _numel(shp)
        t = _ctype_for(name)
        lines.append(f"  // input {name}: shape={list(shp)} dtype={dtype_env.get(name)}")
        lines.append(f"  {t}* {name} = ({t}*)malloc(sizeof({t}) * (size_t){n});")
        lines.append(f"  if (!{name}) {{ fprintf(stderr, \"alloc failed: {name}\\n\"); return 2; }}")
        lines.append(f"  if (!read_bytes(\"{name}.bin\", {name}, sizeof({t}) * (size_t){n})) return 2;")
        cvar[name] = name
        ctype[name] = t
        lines.append("")

    # 2) Materialize consts as 1-element arrays (float only)
    for name, val in const_values.items():
        lines.append(f"  float* {name} = (float*)malloc(sizeof(float));")
        lines.append(f"  if (!{name}) {{ fprintf(stderr, \"alloc failed: {name}\\n\"); return 2; }}")
        lines.append(f"  {name}[0] = {_c_float(float(val))};")
        cvar[name] = name
        ctype[name] = "float"
        lines.append("")

    # 3) Lower ops in order.
    for op in intent.ops:
        out = op.output
        if op.op == "const":
            continue
        if not out:
            raise IntentIRValidationError(f"lowering: op missing output: {op}")

        if op.op in {"reshape", "identity", "layout_cast"}:
            inp = op.inputs[0]
            t = _ctype_for(out)
            cvar[out] = out
            ctype[out] = t
            lines.append(f"  // {op.op} {inp} -> {out} (alias)")
            lines.append(f"  {t}* {out} = ({t}*){cvar[inp]};")
            lines.append("")
            continue

        # Ensure op-defined semantic dtypes.
        if op.op in {"ne", "reduce_any"}:
            dtype_env[out] = "bool"
        out_shape = shape_env[out]
        out_n = _numel(out_shape)
        t = _ctype_for(out)
        lines.append(f"  // op {op.op} -> {out}: shape={list(out_shape)} dtype={dtype_env.get(out)}")
        lines.append(f"  {t}* {out} = ({t}*)malloc(sizeof({t}) * (size_t){out_n});")
        lines.append(f"  if (!{out}) {{ fprintf(stderr, \"alloc failed: {out}\\n\"); return 2; }}")
        cvar[out] = out
        ctype[out] = t

        if op.op == "transpose":
            inp = op.inputs[0]
            perm = op.attrs.get("perm")
            if not isinstance(perm, list):
                raise IntentIRValidationError("transpose requires attrs.perm list[int]")
            in_shape = shape_env[inp]
            lines += _emit_transpose(out, inp, in_shape, tuple(out_shape), tuple(int(p) for p in perm))
        elif op.op == "broadcast_in_dim":
            inp = op.inputs[0]
            bcast_dims = op.attrs.get("broadcast_dims")
            out_shape_attr = op.attrs.get("out_shape")
            if not isinstance(bcast_dims, list) or not isinstance(out_shape_attr, list):
                raise IntentIRValidationError("broadcast_in_dim requires attrs.broadcast_dims and attrs.out_shape")
            in_shape = shape_env[inp]
            lines += _emit_broadcast_in_dim(out, inp, in_shape, tuple(out_shape), tuple(int(d) for d in bcast_dims))
        elif op.op in {"add", "sub", "mul", "div", "max", "min"}:
            a, b = op.inputs[0], op.inputs[1]
            lines += _emit_elemwise_bin(
                op.op,
                out,
                a,
                b,
                shape_env[a],
                shape_env[b],
                out_shape,
                use_rvv=use_rvv,
            )
        elif op.op == "ne":
            a, b = op.inputs[0], op.inputs[1]
            lines += _emit_ne(out, a, b, shape_env[a], shape_env[b], out_shape)
        elif op.op == "rsqrt":
            a = op.inputs[0]
            lines += _emit_rsqrt(out, a, out_shape, scalar_a=_numel(shape_env[a]) == 1)
        elif op.op == "reduce_sum":
            a = op.inputs[0]
            dims = op.attrs.get("dims", op.attrs.get("axis"))
            if not isinstance(dims, list):
                raise IntentIRValidationError("reduce_sum requires attrs.dims list[int]")
            keepdims = bool(op.attrs.get("keepdims", False))
            scale = op.attrs.get("scale")
            scale_val = None
            if scale is not None:
                scale_val = float(_resolve_const_value(scale, shape_bindings))
            lines += _emit_reduce_sum(out, a, shape_env[a], out_shape, tuple(int(d) for d in dims), keepdims, scale_val)
        elif op.op == "reduce_max":
            a = op.inputs[0]
            dims = op.attrs.get("dims", op.attrs.get("axis"))
            if not isinstance(dims, list):
                raise IntentIRValidationError("reduce_max requires attrs.dims list[int]")
            keepdims = bool(op.attrs.get("keepdims", False))
            lines += _emit_reduce_max(out, a, shape_env[a], out_shape, tuple(int(d) for d in dims), keepdims)
        elif op.op == "reduce_any":
            a = op.inputs[0]
            dims = op.attrs.get("dims", op.attrs.get("axis"))
            if not isinstance(dims, list):
                raise IntentIRValidationError("reduce_any requires attrs.dims list[int]")
            keepdims = bool(op.attrs.get("keepdims", False))
            lines += _emit_reduce_any(out, a, shape_env[a], out_shape, tuple(int(d) for d in dims), keepdims)
        elif op.op == "exp":
            a = op.inputs[0]
            lines += _emit_unary_exp(out, a, out_shape, scalar_a=_is_scalar(a))
        elif op.op == "relu":
            a = op.inputs[0]
            lines += _emit_unary_relu(out, a, out_shape, scalar_a=_is_scalar(a))
        elif op.op == "softmax":
            a = op.inputs[0]
            axis = op.attrs.get("axis")
            if axis is None:
                raise IntentIRValidationError("softmax requires attrs.axis")
            lines += _emit_softmax(out, a, shape_env[a], int(axis))
        elif op.op == "matmul":
            a, b = op.inputs[0], op.inputs[1]
            lines += _emit_matmul(out, a, b, shape_env[a], shape_env[b], out_shape)
        elif op.op == "custom_call":
            callee = op.attrs.get("callee")
            if callee == "upsample_bicubic2d_aa":
                x = op.inputs[0]
                rs_h_name = None
                rs_w_name = None
                for nm in op.inputs[1:]:
                    s = str(nm)
                    if "reciprocal_scale_h" in s:
                        rs_h_name = nm
                    if "reciprocal_scale_w" in s:
                        rs_w_name = nm
                lines += _emit_upsample_bicubic2d_aa(
                    out,
                    x,
                    rs_h_name,
                    rs_w_name,
                    in_shape=shape_env[x],
                    out_shape=out_shape,
                    a=float(op.attrs.get("a", -0.5)),
                )
            else:
                raise IntentIRValidationError(f"custom_call lowering unsupported callee: {callee}")
        else:
            raise IntentIRValidationError(f"lowering: unsupported op: {op.op}")

        lines.append("")

    # 4) Compare outputs to reference outputs.
    ok_exprs: List[str] = []
    for name in outputs:
        shp = shape_env[name]
        n = _numel(shp)
        dt = dtype_env.get(name, "f32")
        if dt in {"bool", "i1"}:
            lines.append(f"  uint8_t* {name}_ref = (uint8_t*)malloc(sizeof(uint8_t) * (size_t){n});")
            lines.append(f"  if (!{name}_ref) {{ fprintf(stderr, \"alloc failed: {name}_ref\\n\"); return 2; }}")
            lines.append(f"  if (!read_bytes(\"{name}_ref.bin\", {name}_ref, sizeof(uint8_t) * (size_t){n})) return 2;")
            lines.append(f"  int ok_{name} = compare_u8(\"{name}\", (const uint8_t*){name}, (const uint8_t*){name}_ref, (size_t){n});")
        else:
            lines.append(f"  float* {name}_ref = (float*)malloc(sizeof(float) * (size_t){n});")
            lines.append(f"  if (!{name}_ref) {{ fprintf(stderr, \"alloc failed: {name}_ref\\n\"); return 2; }}")
            lines.append(f"  if (!read_bytes(\"{name}_ref.bin\", {name}_ref, sizeof(float) * (size_t){n})) return 2;")
            lines.append(f"  int ok_{name} = compare_f32(\"{name}\", (const float*){name}, (const float*){name}_ref, (size_t){n}, ATOL, RTOL);")
        ok_exprs.append(f"ok_{name}")
        lines.append("")

    if ok_exprs:
        lines.append(f"  int ok = {' && '.join(ok_exprs)};")
    else:
        lines.append("  int ok = 1;")
    lines.append("  printf(ok ? \"PASS lowered\\n\" : \"FAIL lowered\\n\");")
    lines.append("  return ok ? 0 : 1;")
    lines.append("}")

    return "\n".join(lines)


def _resolve_shape_tuple(shape, bindings: Dict[str, int]) -> Tuple[int, ...]:
    out: List[int] = []
    for d in shape:
        if hasattr(d, "kind") and getattr(d, "kind") == "sym":
            v = bindings.get(d.value)
            if v is None:
                raise IntentIRValidationError(f"unbound symbol in shape: {d.value}")
            out.append(int(v))
        elif hasattr(d, "kind") and getattr(d, "kind") == "const":
            out.append(int(d.value))
        elif isinstance(d, str):
            if d.isdigit():
                out.append(int(d))
            elif d in bindings:
                out.append(int(bindings[d]))
            else:
                raise IntentIRValidationError(f"unbound symbol in shape: {d}")
        else:
            out.append(int(d))
    return tuple(out)


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _resolve_const_value(val: Any, bindings: Dict[str, int]) -> float:
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        if val == "eps":
            return 1e-5
        if val in bindings:
            return float(bindings[val])
        # allow simple expressions like "group_size * HW"
        try:
            return float(eval(val, {}, dict(bindings)))
        except Exception as e:
            raise IntentIRValidationError(f"unresolved const value: {val}") from e
    raise IntentIRValidationError(f"unsupported const value type: {type(val)}")


def _c_float(x: float) -> str:
    """
    Emit a C float literal that always parses as floating-point (avoid `0f`).
    """
    if math.isnan(x):
        return "NAN"
    if math.isinf(x):
        return "INFINITY" if x > 0 else "(-INFINITY)"
    s = f"{x:.10g}"
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def _infer_output_shape_dtype(
    op: Op,
    shape_env: Dict[str, Tuple[int, ...]],
    dtype_env: Dict[str, str],
    shape_bindings: Dict[str, int],
) -> Tuple[Tuple[int, ...], str]:
    """
    Infer shape/dtype for intermediate outputs that are not explicitly declared in intent.tensors.
    Only supports the op set used by our three real kernels.
    """
    kind = op.op
    if kind in {"reshape"}:
        shp = op.attrs.get("shape")
        if not isinstance(shp, list):
            raise IntentIRValidationError("reshape requires attrs.shape")
        return _resolve_shape_attr(shp, shape_bindings), dtype_env.get(op.inputs[0], "f32")
    if kind in {"identity", "layout_cast"}:
        inp = op.inputs[0]
        return shape_env[inp], dtype_env.get(inp, "f32")
    if kind == "transpose":
        inp = op.inputs[0]
        perm = op.attrs.get("perm")
        if not isinstance(perm, list):
            raise IntentIRValidationError("transpose requires attrs.perm")
        in_shape = shape_env[inp]
        if len(perm) != len(in_shape):
            raise IntentIRValidationError("transpose perm rank mismatch")
        out_shape = tuple(in_shape[int(p)] for p in perm)
        return out_shape, dtype_env.get(inp, "f32")
    if kind == "broadcast_in_dim":
        out_shape = op.attrs.get("out_shape")
        if not isinstance(out_shape, list):
            raise IntentIRValidationError("broadcast_in_dim requires attrs.out_shape")
        # Output dtype is same as input.
        inp = op.inputs[0]
        return _resolve_shape_attr(out_shape, shape_bindings), dtype_env.get(inp, "f32")
    if kind in {"add", "sub", "mul", "div", "max", "min"}:
        a, b = op.inputs[0], op.inputs[1]
        sa, sb = shape_env[a], shape_env[b]
        return _broadcast_shape(sa, sb), "f32"
    if kind == "ne":
        a, b = op.inputs[0], op.inputs[1]
        sa, sb = shape_env[a], shape_env[b]
        return _broadcast_shape(sa, sb), "bool"
    if kind == "rsqrt":
        a = op.inputs[0]
        return shape_env[a], "f32"
    if kind in {"reduce_sum", "reduce_any", "reduce_max"}:
        a = op.inputs[0]
        in_shape = shape_env[a]
        dims = op.attrs.get("dims", op.attrs.get("axis"))
        if not isinstance(dims, list):
            raise IntentIRValidationError(f"{kind} requires attrs.dims/axis list[int]")
        dims_set = set(int(d) for d in dims)
        keepdims = bool(op.attrs.get("keepdims", False))
        out: List[int] = []
        for i, sz in enumerate(in_shape):
            if i in dims_set:
                if keepdims:
                    out.append(1)
            else:
                out.append(int(sz))
        return tuple(out), ("bool" if kind == "reduce_any" else "f32")
    if kind == "softmax":
        a = op.inputs[0]
        return shape_env[a], "f32"
    if kind in {"exp", "relu"}:
        a = op.inputs[0]
        return shape_env[a], "f32"
    if kind == "matmul":
        a, b = op.inputs[0], op.inputs[1]
        sa, sb = shape_env[a], shape_env[b]
        if len(sa) == 2 and len(sb) == 2:
            return (sa[0], sb[1]), "f32"
        if len(sa) == 4 and len(sb) == 4:
            return (sa[0], sa[1], sa[2], sb[3]), "f32"
        raise IntentIRValidationError("matmul infer supports only rank-2 or rank-4")
    raise IntentIRValidationError(f"cannot infer output for op {kind}")


def _broadcast_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Numpy-style broadcasting: align trailing dims and allow 1 expansion.
    """
    ra, rb = len(a), len(b)
    r = max(ra, rb)
    aa = (1,) * (r - ra) + tuple(int(x) for x in a)
    bb = (1,) * (r - rb) + tuple(int(x) for x in b)
    out: List[int] = []
    for da, db in zip(aa, bb):
        if da == db:
            out.append(int(da))
        elif da == 1:
            out.append(int(db))
        elif db == 1:
            out.append(int(da))
        else:
            raise IntentIRValidationError(f"broadcast shape mismatch: {a} vs {b}")
    return tuple(out)


def _resolve_shape_attr(shape_attr: List[Any], bindings: Dict[str, int]) -> Tuple[int, ...]:
    out: List[int] = []
    for s in shape_attr:
        if isinstance(s, (int, np.integer)):
            out.append(int(s))
        elif isinstance(s, str):
            if s.isdigit():
                out.append(int(s))
            elif s in bindings:
                out.append(int(bindings[s]))
            else:
                raise IntentIRValidationError(f"unbound shape symbol in attrs: {s}")
        elif hasattr(s, "kind") and getattr(s, "kind") == "sym":
            v = bindings.get(s.value)
            if v is None:
                raise IntentIRValidationError(f"unbound shape symbol in attrs: {s.value}")
            out.append(int(v))
        elif hasattr(s, "kind") and getattr(s, "kind") == "const":
            out.append(int(s.value))
        else:
            raise IntentIRValidationError(f"unsupported shape attr element: {s}")
    return tuple(out)

def _emit_transpose(out: str, inp: str, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], perm: Tuple[int, ...]) -> List[str]:
    if len(in_shape) != len(out_shape) or len(perm) != len(in_shape):
        raise IntentIRValidationError("transpose rank mismatch")
    r = len(in_shape)
    if r == 2:
        I0, I1 = in_shape
        O0, O1 = out_shape
        return [
            f"  for (int i = 0; i < {O0}; ++i) {{",
            f"    for (int j = 0; j < {O1}; ++j) {{",
            f"      // out[i,j] = in[perm^-1(i,j)]",
            f"      int a0 = (perm0={perm[0]}, perm1={perm[1]}); (void)a0; // keep generator simple",
            f"    }}",
            f"  }}",
        ]
    # For now, only implement the 4D transpose used by attention: perm=[0,1,3,2]
    if r != 4:
        raise IntentIRValidationError("transpose lowering supports only rank-4 (and limited patterns)")
    if perm != (0, 1, 3, 2):
        raise IntentIRValidationError(f"transpose lowering supports only perm [0,1,3,2], got {list(perm)}")
    B, H, K, D = in_shape[0], in_shape[1], in_shape[2], in_shape[3]
    # out shape should be [B,H,D,K]
    return [
        f"  for (int b = 0; b < {B}; ++b) {{",
        f"    for (int h = 0; h < {H}; ++h) {{",
        f"      for (int k = 0; k < {K}; ++k) {{",
        f"        for (int d = 0; d < {D}; ++d) {{",
        f"          {out}[idx4(b,h,d,k,{H},{out_shape[2]},{out_shape[3]})] = {inp}[idx4(b,h,k,d,{H},{K},{D})];",
        f"        }}",
        f"      }}",
        f"    }}",
        f"  }}",
    ]


def _emit_broadcast_in_dim(out: str, inp: str, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], bcast_dims: Tuple[int, ...]) -> List[str]:
    r_in = len(in_shape)
    r_out = len(out_shape)
    if r_in != len(bcast_dims):
        raise IntentIRValidationError("broadcast_in_dim broadcast_dims length must equal input rank")
    if r_out > 4 or r_in > 4:
        raise IntentIRValidationError("broadcast_in_dim lowering supports rank<=4")

    # Build input strides (row-major).
    in_strides = []
    s = 1
    for d in reversed(in_shape):
        in_strides.append(s)
        s *= int(d)
    in_strides = list(reversed(in_strides))

    def in_index_expr(out_idxs: List[str]) -> str:
        terms: List[str] = []
        for in_dim in range(r_in):
            od = bcast_dims[in_dim]
            terms.append(f"((size_t){out_idxs[od]} * (size_t){in_strides[in_dim]})")
        if not terms:
            return "0"
        return " + ".join(terms)

    # Emit loops for r_out up to 4.
    idx_names = ["i0", "i1", "i2", "i3"][:r_out]
    loops: List[str] = []
    for i, d in enumerate(out_shape):
        loops.append(f"  for (int {idx_names[i]} = 0; {idx_names[i]} < {int(d)}; ++{idx_names[i]}) {{")
    # output flat index
    if r_out == 1:
        out_idx = f"(size_t){idx_names[0]}"
    elif r_out == 2:
        out_idx = f"idx2({idx_names[0]},{idx_names[1]},{int(out_shape[1])})"
    elif r_out == 3:
        out_idx = f"idx3({idx_names[0]},{idx_names[1]},{idx_names[2]},{int(out_shape[1])},{int(out_shape[2])})"
    else:
        out_idx = f"idx4({idx_names[0]},{idx_names[1]},{idx_names[2]},{idx_names[3]},{int(out_shape[1])},{int(out_shape[2])},{int(out_shape[3])})"
    in_idx = in_index_expr(idx_names)
    loops.append(f"    {out}[{out_idx}] = {inp}[{in_idx}];")
    for _ in out_shape:
        loops.append("  }")
    return loops


def _emit_elemwise_bin(
    op: str,
    out: str,
    a: str,
    b: str,
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    use_rvv: bool,
) -> List[str]:
    n = _numel(out_shape)
    scalar_a = _numel(a_shape) == 1
    scalar_b = _numel(b_shape) == 1
    return (
        _emit_elemwise_bin_rvv(op, out, a, b, a_shape, b_shape, out_shape, n, scalar_a=scalar_a, scalar_b=scalar_b)
        if use_rvv
        else _emit_elemwise_bin_scalar(op, out, a, b, a_shape, b_shape, out_shape, n, scalar_a=scalar_a, scalar_b=scalar_b)
    )


def _emit_elemwise_bin_scalar(
    op: str,
    out: str,
    a: str,
    b: str,
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    n: int,
    *,
    scalar_a: bool,
    scalar_b: bool,
) -> List[str]:
    c_op = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "max": "fmaxf",
        "min": "fminf",
    }[op]
    # Fast path: scalar and exact-shape elementwise (plus legacy prefix-broadcast).
    if scalar_a or scalar_b or (tuple(a_shape) == tuple(out_shape) and tuple(b_shape) == tuple(out_shape)):
        lines: List[str] = [f"  for (size_t i = 0; i < (size_t){n}; ++i) {{"]
        a_expr = _elem_index_expr(a, a_shape, out_shape, scalar=scalar_a)
        b_expr = _elem_index_expr(b, b_shape, out_shape, scalar=scalar_b)
        if op in {"max", "min"}:
            lines.append(f"    {out}[i] = {c_op}({a_expr}, {b_expr});")
        else:
            lines.append(f"    {out}[i] = {a_expr} {c_op} {b_expr};")
        lines.append("  }")
        return lines

    # General numpy-style broadcasting (align trailing dims), rank<=4.
    r = len(out_shape)
    if r > 4:
        raise IntentIRValidationError("elemwise broadcast supports rank<=4")
    pa = (1,) * (r - len(a_shape)) + tuple(int(x) for x in a_shape)
    pb = (1,) * (r - len(b_shape)) + tuple(int(x) for x in b_shape)
    po = tuple(int(x) for x in out_shape)
    for d_in, d_out in zip(pa, po):
        if d_in not in (1, d_out):
            raise IntentIRValidationError(f"elemwise broadcast mismatch: a_shape={a_shape} out_shape={out_shape}")
    for d_in, d_out in zip(pb, po):
        if d_in not in (1, d_out):
            raise IntentIRValidationError(f"elemwise broadcast mismatch: b_shape={b_shape} out_shape={out_shape}")

    idx_names = ["i0", "i1", "i2", "i3"][:r]
    loops: List[str] = []
    for i, d in enumerate(po):
        loops.append(f"  for (int {idx_names[i]} = 0; {idx_names[i]} < {int(d)}; ++{idx_names[i]}) {{")
    out_idx = _flat_idx_expr(idx_names, po)

    def idx_expr(padded_shape: Tuple[int, ...]) -> str:
        in_idxs = ["0" if int(dim) == 1 else idx_names[i] for i, dim in enumerate(padded_shape)]
        return _flat_idx_expr(in_idxs, padded_shape)

    a_idx = idx_expr(pa)
    b_idx = idx_expr(pb)
    if op in {"max", "min"}:
        expr = f"{c_op}({a}[{a_idx}], {b}[{b_idx}])"
    else:
        expr = f"({a}[{a_idx}] {c_op} {b}[{b_idx}])"
    lines = [f"  // elemwise {op} broadcast"] + loops + [f"    {out}[{out_idx}] = {expr};"] + ["  }"] * r
    return lines


def _emit_elemwise_bin_rvv(
    op: str,
    out: str,
    a: str,
    b: str,
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    n: int,
    *,
    scalar_a: bool,
    scalar_b: bool,
) -> List[str]:
    """
    RVV vectorized elementwise for float32 arrays. Falls back to scalar when RVV isn't enabled by compiler.
    """
    # RVV path only supports dense elementwise (no prefix-broadcast) for now.
    if (not scalar_a and a_shape != out_shape) or (not scalar_b and b_shape != out_shape):
        return _emit_elemwise_bin_scalar(op, out, a, b, a_shape, b_shape, out_shape, n, scalar_a=scalar_a, scalar_b=scalar_b)
    lines: List[str] = []
    lines.append("#if defined(__riscv_vector) || defined(__riscv_v)")
    lines.append("  {")
    lines.append(f"    size_t i = 0;")
    lines.append(f"    while (i < (size_t){n}) {{")
    lines.append("      size_t vl = __riscv_vsetvl_e32m1((size_t){N} - i);".format(N=n))
    if scalar_a:
        lines.append(f"      float a0 = {a}[0];")
    else:
        lines.append(f"      vfloat32m1_t va = __riscv_vle32_v_f32m1({a} + i, vl);")
    if scalar_b:
        lines.append(f"      float b0 = {b}[0];")
    else:
        lines.append(f"      vfloat32m1_t vb = __riscv_vle32_v_f32m1({b} + i, vl);")

    def vv(expr: str) -> str:
        return expr

    if op == "add":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(a0 + b0, vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t vy = __riscv_vfadd_vf_f32m1(vb, a0, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfadd_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfadd_vv_f32m1(va, vb, vl);")
    elif op == "sub":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(a0 - b0, vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t va0 = __riscv_vfmv_v_f_f32m1(a0, vl);")
            lines.append("      vfloat32m1_t vy = __riscv_vfsub_vv_f32m1(va0, vb, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfsub_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfsub_vv_f32m1(va, vb, vl);")
    elif op == "mul":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(a0 * b0, vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(vb, a0, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmul_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfmul_vv_f32m1(va, vb, vl);")
    elif op == "div":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(a0 / b0, vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t va0 = __riscv_vfmv_v_f_f32m1(a0, vl);")
            lines.append("      vfloat32m1_t vy = __riscv_vfdiv_vv_f32m1(va0, vb, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfdiv_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfdiv_vv_f32m1(va, vb, vl);")
    elif op == "max":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(fmaxf(a0, b0), vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t vy = __riscv_vfmax_vf_f32m1(vb, a0, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmax_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfmax_vv_f32m1(va, vb, vl);")
    elif op == "min":
        if scalar_a and scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmv_v_f_f32m1(fminf(a0, b0), vl);")
        elif scalar_a:
            lines.append("      vfloat32m1_t vy = __riscv_vfmin_vf_f32m1(vb, a0, vl);")
        elif scalar_b:
            lines.append("      vfloat32m1_t vy = __riscv_vfmin_vf_f32m1(va, b0, vl);")
        else:
            lines.append("      vfloat32m1_t vy = __riscv_vfmin_vv_f32m1(va, vb, vl);")
    else:
        # fallback
        return _emit_elemwise_bin_scalar(op, out, a, b, a_shape, b_shape, out_shape, n, scalar_a=scalar_a, scalar_b=scalar_b)

    lines.append(f"      __riscv_vse32_v_f32m1({out} + i, vy, vl);")
    lines.append("      i += vl;")
    lines.append("    }")
    lines.append("  }")
    lines.append("#else")
    lines.extend(_emit_elemwise_bin_scalar(op, out, a, b, a_shape, b_shape, out_shape, n, scalar_a=scalar_a, scalar_b=scalar_b))
    lines.append("#endif")
    return lines


def _emit_ne(out: str, a: str, b: str, a_shape: Tuple[int, ...], b_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> List[str]:
    n = _numel(out_shape)
    scalar_a = _numel(a_shape) == 1
    scalar_b = _numel(b_shape) == 1
    if scalar_a or scalar_b or (tuple(a_shape) == tuple(out_shape) and tuple(b_shape) == tuple(out_shape)):
        lines: List[str] = [f"  for (size_t i = 0; i < (size_t){n}; ++i) {{"]
        a_expr = _elem_index_expr(a, a_shape, out_shape, scalar=scalar_a)
        b_expr = _elem_index_expr(b, b_shape, out_shape, scalar=scalar_b)
        lines.append(f"    {out}[i] = ({a_expr} != {b_expr}) ? 1 : 0;")
        lines.append("  }")
        return lines

    r = len(out_shape)
    if r > 4:
        raise IntentIRValidationError("ne broadcast supports rank<=4")
    pa = (1,) * (r - len(a_shape)) + tuple(int(x) for x in a_shape)
    pb = (1,) * (r - len(b_shape)) + tuple(int(x) for x in b_shape)
    po = tuple(int(x) for x in out_shape)
    for d_in, d_out in zip(pa, po):
        if d_in not in (1, d_out):
            raise IntentIRValidationError(f"ne broadcast mismatch: a_shape={a_shape} out_shape={out_shape}")
    for d_in, d_out in zip(pb, po):
        if d_in not in (1, d_out):
            raise IntentIRValidationError(f"ne broadcast mismatch: b_shape={b_shape} out_shape={out_shape}")

    idx_names = ["i0", "i1", "i2", "i3"][:r]
    loops: List[str] = []
    for i, d in enumerate(po):
        loops.append(f"  for (int {idx_names[i]} = 0; {idx_names[i]} < {int(d)}; ++{idx_names[i]}) {{")
    out_idx = _flat_idx_expr(idx_names, po)

    def idx_expr(padded_shape: Tuple[int, ...]) -> str:
        in_idxs = ["0" if int(dim) == 1 else idx_names[i] for i, dim in enumerate(padded_shape)]
        return _flat_idx_expr(in_idxs, padded_shape)

    a_idx = idx_expr(pa)
    b_idx = idx_expr(pb)
    lines = [f"  // ne broadcast"] + loops + [f"    {out}[{out_idx}] = ({a}[{a_idx}] != {b}[{b_idx}]) ? 1 : 0;"] + ["  }"] * r
    return lines


def _elem_index_expr(name: str, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], *, scalar: bool) -> str:
    """
    Produce an indexing expression into `name` based on flat output index `i`.
    Supports:
      - scalar broadcast
      - same-shape elementwise
      - legacy prefix-broadcast: in_shape matches out_shape[:r_in], index = i / prod(trailing)
    For general numpy-style broadcasting (e.g., [N] -> [M,N]), the emitter uses
    explicit nested loops instead of this helper.
    """
    if scalar or _numel(in_shape) == 1:
        return f"{name}[0]"
    if tuple(in_shape) == tuple(out_shape):
        return f"{name}[i]"
    r_in = len(in_shape)
    r_out = len(out_shape)
    if r_in >= 1 and r_in < r_out and tuple(in_shape) == tuple(out_shape[:r_in]):
        trailing = 1
        for d in out_shape[r_in:]:
            trailing *= int(d)
        if trailing <= 0:
            raise IntentIRValidationError("invalid trailing product for broadcast")
        if trailing == 1:
            return f"{name}[i]"
        return f"{name}[(size_t)(i / (size_t){trailing})]"
    raise IntentIRValidationError(f"unsupported elementwise broadcast for {name}: in_shape={in_shape} out_shape={out_shape}")


def _emit_rsqrt(out: str, a: str, out_shape: Tuple[int, ...], *, scalar_a: bool) -> List[str]:
    n = _numel(out_shape)
    a_expr = f"{a}[0]" if scalar_a else f"{a}[i]"
    return [
        f"  for (size_t i = 0; i < (size_t){n}; ++i) {{",
        f"    {out}[i] = 1.0f / sqrtf({a_expr});",
        f"  }}",
    ]


def _emit_reduce_sum(
    out: str,
    a: str,
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    dims: Tuple[int, ...],
    keepdims: bool,
    scale: float | None,
) -> List[str]:
    # Implement only rank<=4 and keepdims=False/True with contiguous row-major.
    r = len(in_shape)
    if r > 4:
        raise IntentIRValidationError("reduce_sum lowering supports rank<=4")
    dims_set = set(int(d) for d in dims)
    if any(d < 0 or d >= r for d in dims_set):
        raise IntentIRValidationError("reduce_sum dims out of range")

    # Determine mapping from output indices to input indices.
    out_rank = len(out_shape)
    # For our IR, output tensor is already declared with the intended rank, so we only support
    # the straightforward mapping where keepdims=False drops the reduced dims.
    if keepdims:
        if out_rank != r:
            raise IntentIRValidationError("reduce_sum keepdims expects output rank == input rank")
    else:
        if out_rank != r - len(dims_set):
            raise IntentIRValidationError("reduce_sum output rank mismatch for keepdims=False")

    # Emit nested loops (up to 4D) for output, inner loops for reduced dims.
    # Build lists of dim sizes.
    D = list(int(x) for x in in_shape)
    # Output dims sizes
    OD = list(int(x) for x in out_shape)

    # Index variables
    out_vars = [f"o{i}" for i in range(out_rank)]
    lines: List[str] = []
    for i, sz in enumerate(OD):
        lines.append(f"  for (int {out_vars[i]} = 0; {out_vars[i]} < {sz}; ++{out_vars[i]}) {{")

    # Initialize accumulator
    out_idx_expr = _flat_idx_expr(out_vars, out_shape)
    lines.append("    double acc = 0.0;")

    # Build input index vars: either taken from output or looped over reduced dims.
    in_vars = [None] * r  # type: ignore[list-item]
    if keepdims:
        # Output vars align 1:1 with input vars
        out_it = iter(out_vars)
        for di in range(r):
            if di in dims_set:
                in_vars[di] = f"r{di}"
            else:
                in_vars[di] = next(out_it)
    else:
        out_it = iter(out_vars)
        for di in range(r):
            if di in dims_set:
                in_vars[di] = f"r{di}"
            else:
                in_vars[di] = next(out_it)

    # Reduced loops
    for di in sorted(dims_set):
        lines.append(f"    for (int r{di} = 0; r{di} < {D[di]}; ++r{di}) {{")

    in_idx_expr = _flat_idx_expr([str(v) for v in in_vars], in_shape)
    lines.append(f"      acc += (double){a}[{in_idx_expr}];")

    for _ in sorted(dims_set):
        lines.append("    }")

    if scale is not None:
        lines.append(f"    acc *= (double){_c_float(float(scale))};")
    lines.append(f"    {out}[{out_idx_expr}] = (float)acc;")

    for _ in OD:
        lines.append("  }")
    return lines


def _emit_reduce_max(out: str, a: str, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], dims: Tuple[int, ...], keepdims: bool) -> List[str]:
    r = len(in_shape)
    if r > 4:
        raise IntentIRValidationError("reduce_max lowering supports rank<=4")
    dims_set = set(int(d) for d in dims)
    if any(d < 0 or d >= r for d in dims_set):
        raise IntentIRValidationError("reduce_max dims out of range")
    out_rank = len(out_shape)
    if keepdims:
        if out_rank != r:
            raise IntentIRValidationError("reduce_max keepdims expects output rank == input rank")
    else:
        if out_rank != r - len(dims_set):
            raise IntentIRValidationError("reduce_max output rank mismatch for keepdims=False")

    D = list(int(x) for x in in_shape)
    OD = list(int(x) for x in out_shape)
    out_vars = [f"o{i}" for i in range(out_rank)]
    lines: List[str] = []
    for i, sz in enumerate(OD):
        lines.append(f"  for (int {out_vars[i]} = 0; {out_vars[i]} < {sz}; ++{out_vars[i]}) {{")
    out_idx_expr = _flat_idx_expr(out_vars, out_shape)
    lines.append("    float m = -INFINITY;")

    in_vars = [None] * r  # type: ignore[list-item]
    out_it = iter(out_vars)
    for di in range(r):
        if di in dims_set:
            in_vars[di] = f"r{di}"
        else:
            in_vars[di] = next(out_it)

    for di in sorted(dims_set):
        lines.append(f"    for (int r{di} = 0; r{di} < {D[di]}; ++r{di}) {{")
    in_idx_expr = _flat_idx_expr([str(v) for v in in_vars], in_shape)
    lines.append(f"      float v = {a}[{in_idx_expr}];")
    lines.append("      m = fmaxf(m, v);")
    for _ in sorted(dims_set):
        lines.append("    }")
    lines.append(f"    {out}[{out_idx_expr}] = m;")
    for _ in OD:
        lines.append("  }")
    return lines


def _emit_reduce_any(out: str, a: str, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...], dims: Tuple[int, ...], keepdims: bool) -> List[str]:
    r = len(in_shape)
    if r > 4:
        raise IntentIRValidationError("reduce_any lowering supports rank<=4")
    dims_set = set(int(d) for d in dims)
    if keepdims:
        if len(out_shape) != r:
            raise IntentIRValidationError("reduce_any keepdims expects output rank == input rank")
    else:
        if len(out_shape) != r - len(dims_set):
            raise IntentIRValidationError("reduce_any output rank mismatch for keepdims=False")

    D = list(int(x) for x in in_shape)
    OD = list(int(x) for x in out_shape)
    out_vars = [f"o{i}" for i in range(len(OD))]
    lines: List[str] = []
    for i, sz in enumerate(OD):
        lines.append(f"  for (int {out_vars[i]} = 0; {out_vars[i]} < {sz}; ++{out_vars[i]}) {{")
    out_idx_expr = _flat_idx_expr(out_vars, out_shape)
    lines.append("    uint8_t acc = 0;")

    in_vars = [None] * r  # type: ignore[list-item]
    out_it = iter(out_vars)
    for di in range(r):
        if di in dims_set:
            in_vars[di] = f"r{di}"
        else:
            in_vars[di] = next(out_it)

    for di in sorted(dims_set):
        lines.append(f"    for (int r{di} = 0; r{di} < {D[di]}; ++r{di}) {{")
    in_idx_expr = _flat_idx_expr([str(v) for v in in_vars], in_shape)
    lines.append(f"      acc = acc || ({a}[{in_idx_expr}] != 0);")
    for _ in sorted(dims_set):
        lines.append("    }")
    lines.append(f"    {out}[{out_idx_expr}] = acc;")
    for _ in OD:
        lines.append("  }")
    return lines


def _emit_unary_exp(out: str, a: str, out_shape: Tuple[int, ...], *, scalar_a: bool) -> List[str]:
    n = _numel(out_shape)
    a_expr = f"{a}[0]" if scalar_a else f"{a}[i]"
    return [
        f"  for (size_t i = 0; i < (size_t){n}; ++i) {{",
        f"    {out}[i] = expf({a_expr});",
        f"  }}",
    ]


def _emit_unary_relu(out: str, a: str, out_shape: Tuple[int, ...], *, scalar_a: bool) -> List[str]:
    n = _numel(out_shape)
    a_expr = f"{a}[0]" if scalar_a else f"{a}[i]"
    return [
        f"  for (size_t i = 0; i < (size_t){n}; ++i) {{",
        f"    float v = {a_expr};",
        f"    {out}[i] = v > 0.0f ? v : 0.0f;",
        f"  }}",
    ]


def _emit_softmax(out: str, a: str, in_shape: Tuple[int, ...], axis: int) -> List[str]:
    """
    Lower softmax along the last axis for rank 1..4 tensors.
    """
    r = len(in_shape)
    if r == 0:
        raise IntentIRValidationError("softmax on scalar is unsupported")
    ax = int(axis)
    if ax < 0:
        ax += r
    if ax != r - 1:
        raise IntentIRValidationError("softmax lowering currently supports axis == last dimension only")
    D = [int(x) for x in in_shape]
    if r == 1:
        K = D[0]
        return [
            "  // softmax rank-1",
            "  {",
            "    double mx = -1e30;",
            f"    for (int k = 0; k < {K}; ++k) {{ double v = (double){a}[(size_t)k]; if (v > mx) mx = v; }}",
            "    double sum = 0.0;",
            f"    for (int k = 0; k < {K}; ++k) {{ double e = exp((double){a}[(size_t)k] - mx); {out}[(size_t)k] = (float)e; sum += e; }}",
            "    double inv = 1.0 / sum;",
            f"    for (int k = 0; k < {K}; ++k) {{ {out}[(size_t)k] = (float)((double){out}[(size_t)k] * inv); }}",
            "  }",
        ]
    if r == 2:
        M, K = D[0], D[1]
        return [
            "  // softmax rank-2 axis=1",
            f"  for (int m = 0; m < {M}; ++m) {{",
            "    double mx = -1e30;",
            f"    for (int k = 0; k < {K}; ++k) {{ double v = (double){a}[idx2(m,k,{K})]; if (v > mx) mx = v; }}",
            "    double sum = 0.0;",
            f"    for (int k = 0; k < {K}; ++k) {{ double e = exp((double){a}[idx2(m,k,{K})] - mx); {out}[idx2(m,k,{K})] = (float)e; sum += e; }}",
            "    double inv = 1.0 / sum;",
            f"    for (int k = 0; k < {K}; ++k) {{ {out}[idx2(m,k,{K})] = (float)((double){out}[idx2(m,k,{K})] * inv); }}",
            "  }",
        ]
    if r == 3:
        A0, A1, K = D[0], D[1], D[2]
        return [
            "  // softmax rank-3 axis=2",
            f"  for (int i0 = 0; i0 < {A0}; ++i0) {{",
            f"    for (int i1 = 0; i1 < {A1}; ++i1) {{",
            "      double mx = -1e30;",
            f"      for (int k = 0; k < {K}; ++k) {{ double v = (double){a}[idx3(i0,i1,k,{A1},{K})]; if (v > mx) mx = v; }}",
            "      double sum = 0.0;",
            f"      for (int k = 0; k < {K}; ++k) {{ double e = exp((double){a}[idx3(i0,i1,k,{A1},{K})] - mx); {out}[idx3(i0,i1,k,{A1},{K})] = (float)e; sum += e; }}",
            "      double inv = 1.0 / sum;",
            f"      for (int k = 0; k < {K}; ++k) {{ {out}[idx3(i0,i1,k,{A1},{K})] = (float)((double){out}[idx3(i0,i1,k,{A1},{K})] * inv); }}",
            "    }",
            "  }",
        ]
    if r == 4:
        B, H, Q, K = D
        return [
            "  // softmax rank-4 axis=3",
            f"  for (int b = 0; b < {B}; ++b) {{",
            f"    for (int h = 0; h < {H}; ++h) {{",
            f"      for (int q = 0; q < {Q}; ++q) {{",
            "        double mx = -1e30;",
            f"        for (int k = 0; k < {K}; ++k) {{",
            f"          double v = (double){a}[idx4(b,h,q,k,{H},{Q},{K})];",
            "          if (v > mx) mx = v;",
            "        }",
            "        double sum = 0.0;",
            f"        for (int k = 0; k < {K}; ++k) {{",
            f"          double e = exp((double){a}[idx4(b,h,q,k,{H},{Q},{K})] - mx);",
            f"          {out}[idx4(b,h,q,k,{H},{Q},{K})] = (float)e;",
            "          sum += e;",
            "        }",
            "        double inv = 1.0 / sum;",
            f"        for (int k = 0; k < {K}; ++k) {{",
            f"          {out}[idx4(b,h,q,k,{H},{Q},{K})] = (float)((double){out}[idx4(b,h,q,k,{H},{Q},{K})] * inv);",
            "        }",
            "      }",
            "    }",
            "  }",
        ]
    raise IntentIRValidationError("softmax lowering supports rank<=4 only")


def _emit_matmul(
    out: str,
    a: str,
    b: str,
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
) -> List[str]:
    # Support 2D and the 4D batched attention case.
    if len(a_shape) == 2 and len(b_shape) == 2:
        M, K = (int(x) for x in a_shape)
        K2, N = (int(x) for x in b_shape)
        if K2 != K:
            raise IntentIRValidationError("matmul shape mismatch (2D)")
        return [
            "#if defined(__riscv_vector) || defined(__riscv_v)",
            f"  for (int m = 0; m < {M}; ++m) {{",
            f"    for (int n = 0; n < {N}; ++n) {{",
            "      double acc = 0.0;",
            f"      const float* arow = (const float*){a} + (size_t)m * (size_t){K};",
            "      int k0 = 0;",
            f"      const ptrdiff_t bstride = (ptrdiff_t)((size_t){N} * sizeof(float));",
            f"      while (k0 < {K}) {{",
            f"        size_t vl = __riscv_vsetvl_e32m1((size_t){K} - (size_t)k0);",
            "        vfloat32m1_t va = __riscv_vle32_v_f32m1(arow + k0, vl);",
            f"        const float* bbase = (const float*){b} + (size_t)k0 * (size_t){N} + (size_t)n;",
            "        vfloat32m1_t vb = __riscv_vlse32_v_f32m1(bbase, bstride, vl);",
            "        vfloat32m1_t vmul = __riscv_vfmul_vv_f32m1(va, vb, vl);",
            "        vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);",
            "        vfloat32m1_t vs = __riscv_vfredusum_vs_f32m1_f32m1(vmul, v0, vl);",
            "        float s = __riscv_vfmv_f_s_f32m1_f32(vs);",
            "        acc += (double)s;",
            "        k0 += (int)vl;",
            "      }",
            f"      {out}[idx2(m,n,{N})] = (float)acc;",
            "    }",
            "  }",
            "#else",
            f"  for (int m = 0; m < {M}; ++m) {{",
            f"    for (int n = 0; n < {N}; ++n) {{",
            "      double acc = 0.0;",
            f"      for (int k = 0; k < {K}; ++k) {{",
            f"        acc += (double){a}[idx2(m,k,{K})] * (double){b}[idx2(k,n,{N})];",
            "      }",
            f"      {out}[idx2(m,n,{N})] = (float)acc;",
            "    }",
            "  }",
            "#endif",
        ]
    if len(a_shape) == 4 and len(b_shape) == 4 and len(out_shape) == 4:
        B, H, M, K = (int(x) for x in a_shape)
        B2, H2, K2, N = (int(x) for x in b_shape)
        if (B2, H2, K2) != (B, H, K):
            raise IntentIRValidationError("matmul shape mismatch (4D)")
        return [
            "#if defined(__riscv_vector) || defined(__riscv_v)",
            f"  for (int b0 = 0; b0 < {B}; ++b0) {{",
            f"    for (int h0 = 0; h0 < {H}; ++h0) {{",
            f"      for (int m0 = 0; m0 < {M}; ++m0) {{",
            f"        for (int n0 = 0; n0 < {N}; ++n0) {{",
            "          double acc = 0.0;",
            f"          const float* arow = (const float*){a} + idx4(b0,h0,m0,0,{H},{M},{K});",
            f"          const float* bcol0 = (const float*){b} + idx4(b0,h0,0,n0,{H},{K},{N});",
            f"          const ptrdiff_t bstride = (ptrdiff_t)((size_t){N} * sizeof(float));",
            "          int k0 = 0;",
            f"          while (k0 < {K}) {{",
            f"            size_t vl = __riscv_vsetvl_e32m1((size_t){K} - (size_t)k0);",
            "            vfloat32m1_t va = __riscv_vle32_v_f32m1(arow + k0, vl);",
            f"            vfloat32m1_t vb = __riscv_vlse32_v_f32m1(bcol0 + (size_t)k0 * (size_t){N}, bstride, vl);",
            "            vfloat32m1_t vmul = __riscv_vfmul_vv_f32m1(va, vb, vl);",
            "            vfloat32m1_t v0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);",
            "            vfloat32m1_t vs = __riscv_vfredusum_vs_f32m1_f32m1(vmul, v0, vl);",
            "            float s = __riscv_vfmv_f_s_f32m1_f32(vs);",
            "            acc += (double)s;",
            "            k0 += (int)vl;",
            "          }",
            f"          {out}[idx4(b0,h0,m0,n0,{H},{M},{N})] = (float)acc;",
            "        }",
            "      }",
            "    }",
            "  }",
            "#else",
            f"  for (int b0 = 0; b0 < {B}; ++b0) {{",
            f"    for (int h0 = 0; h0 < {H}; ++h0) {{",
            f"      for (int m0 = 0; m0 < {M}; ++m0) {{",
            f"        for (int n0 = 0; n0 < {N}; ++n0) {{",
            "          double acc = 0.0;",
            f"          for (int k0 = 0; k0 < {K}; ++k0) {{",
            f"            acc += (double){a}[idx4(b0,h0,m0,k0,{H},{M},{K})] * (double){b}[idx4(b0,h0,k0,n0,{H},{K},{N})];",
            "          }",
            f"          {out}[idx4(b0,h0,m0,n0,{H},{M},{N})] = (float)acc;",
            "        }",
            "      }",
            "    }",
            "  }",
            "#endif",
        ]
    raise IntentIRValidationError("matmul lowering supports only rank-2 or rank-4")


def _emit_upsample_bicubic2d_aa(
    out: str,
    x: str,
    rs_h_name: str | None,
    rs_w_name: str | None,
    *,
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...],
    a: float,
) -> List[str]:
    """
    Emit scalar C for Triton upsample_bicubic2d_aa_kernel semantics.
    Input:  [N,C,IH,IW]
    Output: [N,C,OH,OW]
    Uses support=2.0, invscale=1.0, 5-tap weights like the kernel.
    """
    if len(in_shape) != 4 or len(out_shape) != 4:
        raise IntentIRValidationError("upsample_bicubic2d_aa expects rank-4 input/output")
    N, C, IH, IW = (int(v) for v in in_shape)
    N2, C2, OH, OW = (int(v) for v in out_shape)
    if (N2, C2) != (N, C):
        raise IntentIRValidationError("upsample_bicubic2d_aa N/C mismatch")
    # Resolve reciprocal scales: read from scalar tensor if provided, else default IH/OH, IW/OW.
    rs_h_expr = f"({rs_h_name}[0])" if rs_h_name else f"((float){IH} / (float){OH})"
    rs_w_expr = f"({rs_w_name}[0])" if rs_w_name else f"((float){IW} / (float){OW})"
    af = _c_float(float(a))
    lines: List[str] = []
    lines += [
        "  // upsample_bicubic2d_aa (scalar)",
        "  {",
        f"    const float a = {af};",
        f"    const float rs_h = {rs_h_expr};",
        f"    const float rs_w = {rs_w_expr};",
        "    const float support_h = 2.0f;",
        "    const float support_w = 2.0f;",
        "    const float invscale_h = 1.0f;",
        "    const float invscale_w = 1.0f;",
        f"    for (int n = 0; n < {N}; ++n) {{",
        f"      for (int c = 0; c < {C}; ++c) {{",
        f"        for (int oh = 0; oh < {OH}; ++oh) {{",
        "          float center_h = ((float)oh + 0.5f) * rs_h;",
        "          int span_start_h = (int)floorf(fmaxf(center_h - support_h + 0.5f, 0.0f));",
        f"          int span_size_h = (int)floorf(fminf(center_h + support_h + 0.5f, (float){IH}) - (float)span_start_h);",
        "          float start_minus_center_h = (float)span_start_h - center_h;",
        "          float wy[5];",
        "          for (int i = 0; i < 5; ++i) { wy[i] = 0.0f; }",
        "          float wy_sum = 0.0f;",
        "          for (int y = 0; y < 5; ++y) {",
        "            if (y < span_size_h) {",
        "              float t = ((float)y + start_minus_center_h + 0.5f) * invscale_h;",
        "              float w = cubic_weight(t, a);",
        "              wy[y] = w; wy_sum += w;",
        "            }",
        "          }",
        "          if (wy_sum == 0.0f) wy_sum = 1.0f;",
        "          for (int y = 0; y < 5; ++y) wy[y] /= wy_sum;",
        f"          for (int ow = 0; ow < {OW}; ++ow) {{",
        "            float center_w = ((float)ow + 0.5f) * rs_w;",
        "            int span_start_w = (int)floorf(fmaxf(center_w - support_w + 0.5f, 0.0f));",
        f"            int span_size_w = (int)floorf(fminf(center_w + support_w + 0.5f, (float){IW}) - (float)span_start_w);",
        "            float start_minus_center_w = (float)span_start_w - center_w;",
        "            float wx[5];",
        "            for (int i = 0; i < 5; ++i) { wx[i] = 0.0f; }",
        "            float wx_sum = 0.0f;",
        "            for (int xk = 0; xk < 5; ++xk) {",
        "              if (xk < span_size_w) {",
        "                float t = ((float)xk + start_minus_center_w + 0.5f) * invscale_w;",
        "                float w = cubic_weight(t, a);",
        "                wx[xk] = w; wx_sum += w;",
        "              }",
        "            }",
        "            if (wx_sum == 0.0f) wx_sum = 1.0f;",
        "            for (int xk = 0; xk < 5; ++xk) wx[xk] /= wx_sum;",
        "            double acc = 0.0;",
        "            for (int y = 0; y < 5; ++y) {",
        "              int iy = span_start_h + y;",
        f"              if (iy < 0 || iy >= {IH}) continue;",
        "              float wyv = wy[y];",
        "              if (wyv == 0.0f) continue;",
        "              for (int xk = 0; xk < 5; ++xk) {",
        "                int ix = span_start_w + xk;",
        f"                if (ix < 0 || ix >= {IW}) continue;",
        "                float wxv = wx[xk];",
        "                if (wxv == 0.0f) continue;",
        f"                size_t idx = (((size_t)n * (size_t){C} + (size_t)c) * (size_t){IH} + (size_t)iy) * (size_t){IW} + (size_t)ix;",
        f"                acc += (double){x}[idx] * (double)wyv * (double)wxv;",
        "              }",
        "            }",
        f"            size_t oidx = (((size_t)n * (size_t){C} + (size_t)c) * (size_t){OH} + (size_t)oh) * (size_t){OW} + (size_t)ow;",
        f"            {out}[oidx] = (float)acc;",
        "          }",
        "        }",
        "      }",
        "    }",
        "  }",
    ]
    return lines


def _flat_idx_expr(vars_: List[str], shape: Tuple[int, ...]) -> str:
    r = len(shape)
    if r == 0:
        return "0"
    if r == 1:
        return f"(size_t){vars_[0]}"
    if r == 2:
        return f"idx2({vars_[0]},{vars_[1]},{int(shape[1])})"
    if r == 3:
        return f"idx3({vars_[0]},{vars_[1]},{vars_[2]},{int(shape[1])},{int(shape[2])})"
    if r == 4:
        return f"idx4({vars_[0]},{vars_[1]},{vars_[2]},{vars_[3]},{int(shape[1])},{int(shape[2])},{int(shape[3])})"
    raise IntentIRValidationError("indexing supports rank<=4")
