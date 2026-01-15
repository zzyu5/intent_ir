"""
Task2: Parse LLM-produced JSON into Intent-IR IntentFunction (v1.1 aware).

Supports merged JSON or split tensorization/symbolization, infers outputs when
missing, normalizes layouts, and reports user-friendly errors via
LLMJsonParseError.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..ir.ir_types import IntentFunction, IntentIRValidationError, parse_layout

AXIS_ROLE_VALUES = {"spatial", "reduction", "batch", "channel"}
AXIS_ROLE_ALIASES = {
    # Attention-style dims.
    "seq": "spatial",
    "sequence": "spatial",
    "token": "spatial",
    "tokens": "spatial",
    "ctx": "spatial",
    "context": "spatial",
    # Common informal shorthands.
    "reduce": "reduction",
    "red": "reduction",
    "chan": "channel",
}


class LLMJsonParseError(Exception):
    def __init__(self, message: str, path: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.path = path
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.path:
            parts.append(f"path={self.path}")
        if self.hint:
            parts.append(f"hint={self.hint}")
        return " | ".join(parts)


@dataclass
class CandidateIntent:
    intent: IntentFunction
    problem_params: Dict[str, Any] = field(default_factory=dict)
    schedule_params: Dict[str, Any] = field(default_factory=dict)
    raw_json: Dict[str, Any] = field(default_factory=dict)
    llm_trace: Dict[str, Any] = field(default_factory=dict)


def merge_tensor_and_symbol_json(tensor_d: Dict[str, Any], symbol_d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = copy.deepcopy(tensor_d) if tensor_d else {}
    if symbol_d:
        merged.setdefault("problem_params", symbol_d.get("problem_params"))
        merged.setdefault("schedule_params", symbol_d.get("schedule_params"))
    return merged


def normalize_candidate_json(d: Dict[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(d)

    # tensors: allow dict or list-of-objects with name
    tensors_raw = data.get("tensors") or {}
    if isinstance(tensors_raw, list):
        tensors: Dict[str, Dict[str, Any]] = {}
        for idx, t in enumerate(tensors_raw):
            if not isinstance(t, dict) or "name" not in t:
                raise LLMJsonParseError("tensor entry must be object with name", path=f"tensors[{idx}]")
            name = t["name"]
            entry = {k: v for k, v in t.items() if k != "name"}
            tensors[name] = entry
        tensors_raw = tensors
    if not isinstance(tensors_raw, dict):
        raise LLMJsonParseError("tensors must be object", path="tensors")
    replacements = {"BLOCK_M": "M", "BLOCK_N": "N", "BLOCK_K": "K"}
    for name, t in tensors_raw.items():
        if not isinstance(t, dict):
            raise LLMJsonParseError("tensor must be object", path=f"tensors.{name}")
        if "layout" not in t or t["layout"] is None:
            t["layout"] = "row_major"
        else:
            try:
                parse_layout(t["layout"])
            except IntentIRValidationError as e:
                # Layout is often incidental in LLM outputs (e.g., "NCH", "scalar", "contiguous").
                # Our current core IR only models row/col major; fall back to row_major
                # instead of failing the whole parse.
                t["layout"] = "row_major"
        # normalize dtype if using long form
        dtype = t.get("dtype")
        if isinstance(dtype, str):
            dt = dtype.lower()
            if dt in {"float16", "fp16"}:
                t["dtype"] = "f16"
            elif dt in {"float32", "fp32", "float"}:
                t["dtype"] = "f32"
            elif dt in {"bfloat16", "bf16"}:
                t["dtype"] = "bf16"
            elif dt in {"int1", "i1"}:
                t["dtype"] = "i1"
            elif dt in {"int8", "i8"}:
                t["dtype"] = "i8"
            elif dt in {"uint8", "u8"}:
                t["dtype"] = "u8"
            elif dt in {"int32", "i32"}:
                t["dtype"] = "i32"
            elif dt in {"int64", "i64"}:
                t["dtype"] = "i64"
            elif dt in {"bool", "boolean"}:
                t["dtype"] = "bool"
        if "shape" not in t:
            t["shape"] = []
        # replace tile symbols with axis symbols for downstream bindings
        t["shape"] = [replacements.get(d, d) for d in t["shape"]]
    tensors = tensors_raw
    data["tensors"] = tensors

    def _canonical_tensor_ref(name: Any) -> Any:
        """
        Canonicalize common LLM-emitted value names to match declared tensor names.

        This is strictly a *naming* repair (no semantic reshaping): for example,
        some providers name the output of a store as `store_C` while the declared
        tensor is `C`. When the base tensor exists, we rewrite references so IR
        validation can proceed and downstream stages see stable names.
        """
        if isinstance(name, dict):
            # Some providers wrap tensor refs as objects, e.g. {"tensor":"X"} or {"name":"X"}.
            for k in ("tensor", "name", "value", "id", "ref", "var"):
                v = name.get(k)
                if isinstance(v, str) and v:
                    name = v
                    break
            else:
                # If the dict is a single-key wrapper, unwrap it.
                if len(name) == 1:
                    v = next(iter(name.values()))
                    if isinstance(v, str) and v:
                        name = v
        if not isinstance(name, str):
            return name
        if name.startswith("store_"):
            base = name[len("store_") :]
            if base in tensors and name not in tensors:
                return base
        return name

    # Collect shape symbols (strings) appearing in tensor shapes and shape attrs.
    shape_symbols: set[str] = set()
    for t in tensors.values():
        if isinstance(t, dict):
            for dim in t.get("shape", []):
                if isinstance(dim, str) and dim:
                    shape_symbols.add(dim)

    # Known arg tensors to enforce original shapes (no invented grouped shapes)
    arg_tensor_names = set(tensors.keys())
    ops_raw = data.get("ops") or []
    if not isinstance(ops_raw, list):
        raise LLMJsonParseError("ops must be list", path="ops")
    ops: List[Dict[str, Any]] = []
    produced_outputs: List[str] = []
    used_names = set(tensors.keys())
    current_ssa: Dict[str, str] = {}
    seen_op_outputs: set[str] = set()

    def _fresh_name(base: str) -> str:
        base = str(base)
        cand = base
        i = 0
        while cand in used_names:
            i += 1
            cand = f"{base}_{i}"
        used_names.add(cand)
        return cand

    def _infer_scalar_dtype(op_dict: Dict[str, Any]) -> str:
        # Prefer declared tensor dtype for output/input when available.
        out_name = op_dict.get("output")
        if out_name in tensors and isinstance(tensors[out_name], dict):
            dt = tensors[out_name].get("dtype")
            if isinstance(dt, str):
                return dt
        inps = op_dict.get("inputs") or []
        if inps and inps[0] in tensors and isinstance(tensors[inps[0]], dict):
            dt = tensors[inps[0]].get("dtype")
            if isinstance(dt, str):
                return dt
        return "f32"

    def _normalize_dtype_str(dt: Any, *, fallback: str) -> str:
        if not isinstance(dt, str):
            return fallback
        dd = dt.lower()
        if dd in {"float16", "fp16"}:
            return "f16"
        if dd in {"float32", "fp32", "float"}:
            return "f32"
        if dd in {"float64", "fp64"}:
            return "f64"
        if dd in {"bfloat16", "bf16"}:
            return "bf16"
        if dd in {"int1", "i1"}:
            return "i1"
        if dd in {"int8", "i8"}:
            return "i8"
        if dd in {"uint8", "u8"}:
            return "u8"
        if dd in {"int32", "i32"}:
            return "i32"
        if dd in {"int64", "i64"}:
            return "i64"
        if dd in {"bool", "boolean"}:
            return "bool"
        return fallback

    def _resolve_input_name(x: Any) -> Any:
        x = _canonical_tensor_ref(x)
        if isinstance(x, str) and x in current_ssa:
            return current_ssa[x]
        return x
    for idx, op in enumerate(ops_raw):
        if not isinstance(op, dict):
            raise LLMJsonParseError("op must be object", path=f"ops[{idx}]")
        if "op" not in op and "type" in op:
            op["op"] = op.pop("type")
        if "op" not in op and "op_type" in op:
            op["op"] = op.pop("op_type")
        if isinstance(op.get("op"), str) and op["op"].startswith("intent."):
            op["op"] = op["op"].split(".", 1)[1]
        # Normalize common op name variants
        if op.get("op") == "reduce":
            op["op"] = "reduce_sum"
        if op.get("op") == "reduce_any":
            op["op"] = "reduce_any"
        if isinstance(op.get("op"), str) and op["op"].lower() == "neq":
            op["op"] = "ne"
        if op.get("op") in {"compare_ne", "compare_not_equal"}:
            op["op"] = "ne"
        if op.get("op") == "neq":
            op["op"] = "ne"
        if op.get("op") == "compare_not_equal":
            op["op"] = "ne"
        if op.get("op") == "reduce_or":
            op["op"] = "reduce_max"
        if op.get("op") == "elemwise":
            inner = op["attrs"].get("op") or op.get("name")
            if inner:
                op["op"] = inner
                op["attrs"].pop("op", None)
        if op.get("op") == "or":
            op["op"] = "max"
        if "output" not in op and "outputs" in op and isinstance(op["outputs"], list) and op["outputs"]:
            op["output"] = op["outputs"][0]
        # Some providers emit `name` as the produced value identifier (instead of `output`).
        # Accept it only when `output` is absent to preserve canonical schema.
        if "output" not in op and "name" in op and isinstance(op["name"], str) and op["name"]:
            op["output"] = op["name"]
        # Additional common output field variants.
        if "output" not in op and "out" in op and isinstance(op.get("out"), str) and op["out"]:
            op["output"] = op.pop("out")
        if "output" not in op and "dst" in op and isinstance(op.get("dst"), str) and op["dst"]:
            op["output"] = op.pop("dst")
        if "output" not in op and "result" in op and isinstance(op.get("result"), str) and op["result"]:
            op["output"] = op.pop("result")
        if "attrs" not in op or op["attrs"] is None:
            op["attrs"] = {}
        # Merge provider-style `attributes` into `attrs` (non-semantic schema normalization).
        if "attributes" in op and isinstance(op.get("attributes"), dict):
            for k, v in dict(op["attributes"]).items():
                op["attrs"].setdefault(k, v)
            op.pop("attributes", None)
        # Some providers put common attrs at the top-level.
        for k in ("dims", "axes", "axis", "dimension", "dimensions", "keepdims", "init", "shape", "out_shape"):
            if k in op and k not in op["attrs"]:
                op["attrs"][k] = op.pop(k)

        # Input shorthands: allow `x` / `a,b` / `lhs,rhs` when `inputs` is absent.
        if op.get("inputs") is None:
            if "x" in op:
                op["inputs"] = [op.pop("x")]
            elif "a" in op and "b" in op:
                op["inputs"] = [op.pop("a"), op.pop("b")]
            elif "lhs" in op and "rhs" in op:
                op["inputs"] = [op.pop("lhs"), op.pop("rhs")]

        # Normalize inputs to list[str] and canonicalize common naming variants.
        inps = op.get("inputs")
        if inps is None:
            op["inputs"] = []
        elif isinstance(inps, str):
            op["inputs"] = [inps]
        elif not isinstance(inps, list):
            raise LLMJsonParseError("op.inputs must be list", path=f"ops[{idx}].inputs")
        # Apply naming canonicalization + SSA-resolution after list normalization.
        op["inputs"] = [_resolve_input_name(x) for x in (op.get("inputs") or [])]
        # Inline const objects inside inputs (common provider shorthand):
        #   inputs: ["x", {"op":"const","value":0.0,"dtype":"f32"}]
        inps2 = list(op.get("inputs") or [])
        for j, x in enumerate(list(inps2)):
            if not isinstance(x, dict):
                continue
            op_name = x.get("op") or x.get("type")
            if isinstance(op_name, str) and op_name.lower() != "const":
                continue
            # Accept {"op":"const","value":...,"dtype":...} and similar.
            val = x.get("value")
            if val is None and isinstance(x.get("attrs"), dict):
                val = x["attrs"].get("value")
            if val is None and "const" in x:
                val = x.get("const")
            if val is None:
                raise LLMJsonParseError("inline const missing value", path=f"ops[{idx}].inputs[{j}].value")
            const_dtype = _normalize_dtype_str(x.get("dtype") or (x.get("attrs") or {}).get("dtype"), fallback=_infer_scalar_dtype(op))
            const_out = _fresh_name(f"{op.get('output') or f'op{idx}'}__const{j}")
            ops.append({"op": "const", "inputs": [], "output": const_out, "attrs": {"value": val, "dtype": const_dtype}})
            produced_outputs.append(const_out)
            inps2[j] = const_out
        op["inputs"] = inps2

        # Comparator shorthand: allow 1 input + rhs_const attr.
        if op.get("op") in {"ne", "lt", "le", "gt", "ge"}:
            inps3 = list(op.get("inputs") or [])
            if len(inps3) == 1:
                rhs_val = None
                for k in ("rhs_const", "rhs", "other", "scalar", "value", "const"):
                    if k in op["attrs"]:
                        rhs_val = op["attrs"].pop(k)
                        break
                if rhs_val is not None:
                    const_dtype = _infer_scalar_dtype(op)
                    const_out = _fresh_name(f"{op.get('output') or f'op{idx}'}__rhs")
                    ops.append({"op": "const", "inputs": [], "output": const_out, "attrs": {"value": rhs_val, "dtype": const_dtype}})
                    produced_outputs.append(const_out)
                    op["inputs"] = [inps3[0], const_out]

        # Enforce `list[str]` after normalization (avoid late unhashable dict crashes).
        for j, x in enumerate(list(op.get("inputs") or [])):
            if not isinstance(x, str):
                raise LLMJsonParseError(
                    f"op.inputs[{j}] must be string after normalization, got {type(x).__name__}",
                    path=f"ops[{idx}].inputs[{j}]",
                )

        # Canonicalize common attrs key variants (non-semantic).
        if op.get("op") == "iota":
            if "axis" not in op["attrs"] and "dimension" in op["attrs"]:
                op["attrs"]["axis"] = op["attrs"].pop("dimension")
            dt = op["attrs"].get("dtype")
            if isinstance(dt, str):
                dd = dt.lower()
                if dd in {"float16", "fp16"}:
                    op["attrs"]["dtype"] = "f16"
                elif dd in {"float32", "fp32", "float"}:
                    op["attrs"]["dtype"] = "f32"
                elif dd in {"bfloat16", "bf16"}:
                    op["attrs"]["dtype"] = "bf16"
                elif dd in {"int1", "i1"}:
                    op["attrs"]["dtype"] = "i1"
                elif dd in {"int8", "i8"}:
                    op["attrs"]["dtype"] = "i8"
                elif dd in {"uint8", "u8"}:
                    op["attrs"]["dtype"] = "u8"
                elif dd in {"int32", "i32"}:
                    op["attrs"]["dtype"] = "i32"
                elif dd in {"int64", "i64"}:
                    op["attrs"]["dtype"] = "i64"
                elif dd in {"bool", "boolean"}:
                    op["attrs"]["dtype"] = "bool"
        if op.get("op") == "cast":
            if "to" not in op["attrs"] and "dtype" in op["attrs"]:
                op["attrs"]["to"] = op["attrs"].pop("dtype")
            # Some providers omit `to` entirely; infer from the declared output dtype.
            if "to" not in op["attrs"]:
                out_name = _canonical_tensor_ref(op.get("output"))
                if isinstance(out_name, str) and isinstance(tensors.get(out_name), dict):
                    dt = tensors[out_name].get("dtype")
                    if isinstance(dt, str) and dt:
                        op["attrs"]["to"] = dt
            if "to" not in op["attrs"]:
                # Fallback: treat cast as a no-op cast when output dtype is unavailable.
                inps = op.get("inputs") or []
                if isinstance(inps, list) and inps and isinstance(inps[0], str) and isinstance(tensors.get(inps[0]), dict):
                    dt = tensors[inps[0]].get("dtype")
                    if isinstance(dt, str) and dt:
                        op["attrs"]["to"] = dt
            # Normalize dtype spellings commonly emitted by providers.
            to = op["attrs"].get("to")
            if isinstance(to, str):
                dt = to.lower()
                if dt in {"float16", "fp16"}:
                    op["attrs"]["to"] = "f16"
                elif dt in {"float32", "fp32", "float"}:
                    op["attrs"]["to"] = "f32"
                elif dt in {"bfloat16", "bf16"}:
                    op["attrs"]["to"] = "bf16"
                elif dt in {"int1", "i1"}:
                    op["attrs"]["to"] = "i1"
                elif dt in {"int8", "i8"}:
                    op["attrs"]["to"] = "i8"
                elif dt in {"uint8", "u8"}:
                    op["attrs"]["to"] = "u8"
                elif dt in {"int32", "i32"}:
                    op["attrs"]["to"] = "i32"
                elif dt in {"int64", "i64"}:
                    op["attrs"]["to"] = "i64"
                elif dt in {"bool", "boolean"}:
                    op["attrs"]["to"] = "bool"
        if op.get("op") == "matmul":
            attrs = op["attrs"]
            # Canonicalize common transpose flag spellings.
            if "transpose_rhs" in attrs and "transpose_b" not in attrs:
                attrs["transpose_b"] = attrs.pop("transpose_rhs")
            if "transpose_lhs" in attrs and "transpose_a" not in attrs:
                attrs["transpose_a"] = attrs.pop("transpose_lhs")
            for k in ("transpose_a", "transpose_b"):
                v = attrs.get(k)
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in {"1", "true", "yes", "y"}:
                        attrs[k] = True
                    elif s in {"0", "false", "no", "n"}:
                        attrs[k] = False
        if op.get("op") == "transpose":
            attrs = op["attrs"]
            if "perm" not in attrs and "axes" in attrs:
                attrs["perm"] = attrs.pop("axes")
            if "perm" not in attrs and "order" in attrs:
                attrs["perm"] = attrs.pop("order")
            perm = attrs.get("perm")
            if isinstance(perm, tuple):
                perm = list(perm)
            if isinstance(perm, str):
                parts = [p.strip() for p in perm.replace(" ", ",").split(",") if p.strip()]
                try:
                    perm = [int(p) for p in parts]
                except Exception:
                    perm = None
            if isinstance(perm, dict):
                # Common encodings:
                # - {"0":1,"1":0}
                # - {"from0":1,"from1":0}
                items = []
                for k, v in perm.items():
                    kk = str(k).strip()
                    if kk.isdigit():
                        items.append((int(kk), v))
                    elif kk.lower().startswith("from") and kk[4:].isdigit():
                        items.append((int(kk[4:]), v))
                if items:
                    try:
                        items.sort(key=lambda kv: kv[0])
                        perm = [int(v) for _k, v in items]
                    except Exception:
                        perm = None
            if isinstance(perm, list):
                # Accept list[str] digits or list[str] dim-names (map via input tensor shape).
                out_perm: List[int] = []
                all_str = all(isinstance(x, str) for x in perm)
                if all_str:
                    # First try numeric strings.
                    try:
                        out_perm = [int(str(x).strip()) for x in perm]
                    except Exception:
                        out_perm = []
                    if not out_perm:
                        inps = op.get("inputs") or []
                        inp0 = inps[0] if isinstance(inps, list) and inps else None
                        in_shape = None
                        if isinstance(inp0, str) and isinstance(tensors.get(inp0), dict):
                            sh = tensors[inp0].get("shape")
                            if isinstance(sh, list):
                                in_shape = list(sh)
                        if in_shape is not None:
                            try:
                                out_perm = [int(in_shape.index(str(d))) for d in perm]
                            except Exception:
                                out_perm = []
                else:
                    for x in perm:
                        if isinstance(x, int):
                            out_perm.append(int(x))
                        elif isinstance(x, str) and x.strip().isdigit():
                            out_perm.append(int(x.strip()))
                if out_perm:
                    attrs["perm"] = out_perm
            # Final fallback: infer a simple 2D transpose when possible.
            if "perm" not in attrs:
                inps = op.get("inputs") or []
                inp0 = inps[0] if isinstance(inps, list) and inps else None
                if isinstance(inp0, str) and isinstance(tensors.get(inp0), dict):
                    sh = tensors[inp0].get("shape")
                    if isinstance(sh, list) and len(sh) == 2:
                        attrs["perm"] = [1, 0]
        if op.get("op") == "broadcast_in_dim":
            attrs = op["attrs"]
            if "out_shape" not in attrs and "shape" in attrs:
                attrs["out_shape"] = attrs.pop("shape")
            if "broadcast_dims" not in attrs and "dims" in attrs:
                attrs["broadcast_dims"] = attrs.pop("dims")
            # Some providers forget `out_shape`; when the output tensor has a declared
            # shape, we can safely derive it (needed to infer broadcast_dims below).
            out_shape = attrs.get("out_shape")
            if not (isinstance(out_shape, list) and out_shape):
                out_name = _canonical_tensor_ref(op.get("output"))
                t = tensors.get(out_name) if isinstance(out_name, str) else None
                if isinstance(t, dict) and isinstance(t.get("shape"), list) and t["shape"]:
                    attrs["out_shape"] = list(t["shape"])
            if "broadcast_dims" not in attrs and "broadcast_dimensions" in attrs:
                attrs["broadcast_dims"] = attrs.pop("broadcast_dimensions")
            if "broadcast_dims" not in attrs and "broadcast_dim" in attrs:
                attrs["broadcast_dims"] = [attrs.pop("broadcast_dim")]
            bd = attrs.get("broadcast_dims")
            if isinstance(bd, int):
                attrs["broadcast_dims"] = [bd]
            elif isinstance(bd, str):
                # Allow "0" or "0,1"
                parts = [p.strip() for p in bd.split(",") if p.strip()]
                if parts:
                    try:
                        attrs["broadcast_dims"] = [int(p) for p in parts]
                    except Exception:
                        pass
            elif isinstance(bd, tuple):
                attrs["broadcast_dims"] = list(bd)
            elif isinstance(bd, dict):
                picked = bd.get("dims") or bd.get("broadcast_dims") or bd.get("axes")
                if isinstance(picked, list):
                    attrs["broadcast_dims"] = picked
                else:
                    # Common dict encodings:
                    # - {"0":0,"1":1} (input-axis-index -> out-axis-index)
                    # - {"N":0,"G":1} (input-axis-symbol -> out-axis-index)
                    idx_items = []
                    all_digit_keys = True
                    for k, v in bd.items():
                        if not (isinstance(k, str) and k.isdigit()):
                            all_digit_keys = False
                            break
                        idx_items.append((int(k), v))
                    if all_digit_keys and idx_items:
                        try:
                            idx_items.sort(key=lambda kv: kv[0])
                            attrs["broadcast_dims"] = [int(v) for _k, v in idx_items]
                        except Exception:
                            pass
                    else:
                        inp0 = None
                        inps = op.get("inputs") or []
                        if isinstance(inps, list) and inps and isinstance(inps[0], str):
                            inp0 = inps[0]
                        in_shape = None
                        if isinstance(inp0, str) and isinstance(tensors.get(inp0), dict):
                            sh = tensors[inp0].get("shape")
                            if isinstance(sh, list):
                                in_shape = sh
                        if in_shape is not None:
                            try:
                                mapped = []
                                for d in in_shape:
                                    if not isinstance(d, str) or d not in bd:
                                        mapped = []
                                        break
                                    mapped.append(int(bd[d]))
                                if mapped:
                                    attrs["broadcast_dims"] = mapped
                            except Exception:
                                pass
            if isinstance(attrs.get("broadcast_dims"), list):
                out_bds: list[int] = []
                for x in attrs["broadcast_dims"]:
                    try:
                        out_bds.append(int(x))
                    except Exception:
                        continue
                attrs["broadcast_dims"] = out_bds
            # Final fallback: infer broadcast dims from input/output shapes when possible.
            if not isinstance(attrs.get("broadcast_dims"), list):
                inp0 = None
                inps = op.get("inputs") or []
                if isinstance(inps, list) and inps and isinstance(inps[0], str):
                    inp0 = inps[0]
                out_shape = attrs.get("out_shape")
                in_shape = None
                if isinstance(inp0, str) and isinstance(tensors.get(inp0), dict):
                    sh = tensors[inp0].get("shape")
                    if isinstance(sh, list):
                        in_shape = sh
                if isinstance(out_shape, list) and in_shape is not None:
                    def _norm_dim(d: Any) -> Any:
                        if isinstance(d, str) and d.isdigit():
                            try:
                                return int(d)
                            except Exception:
                                return d
                        return d
                    # scalar broadcast
                    if len(in_shape) == 0:
                        attrs["broadcast_dims"] = []
                    else:
                        out_shape_norm = [_norm_dim(d) for d in out_shape]
                        idxs: list[int] = []
                        cursor = 0
                        for d in in_shape:
                            d_norm = _norm_dim(d)
                            try:
                                j = out_shape_norm.index(d_norm, cursor)
                            except Exception:
                                j = -1
                            if j < 0:
                                idxs = []
                                break
                            idxs.append(int(j))
                            cursor = j + 1
                        if idxs:
                            attrs["broadcast_dims"] = idxs
        if op.get("op") == "const":
            # Some providers place const fields at top-level instead of attrs.
            if "value" not in op["attrs"]:
                for k in ("value", "val", "const"):
                    if k in op:
                        op["attrs"]["value"] = op.pop(k)
                        break
            if "dtype" not in op["attrs"] and "dtype" in op:
                op["attrs"]["dtype"] = op.pop("dtype")
            dt = op["attrs"].get("dtype")
            if isinstance(dt, str):
                dd = dt.lower()
                if dd in {"float16", "fp16"}:
                    op["attrs"]["dtype"] = "f16"
                elif dd in {"float32", "fp32", "float"}:
                    op["attrs"]["dtype"] = "f32"
                elif dd in {"bfloat16", "bf16"}:
                    op["attrs"]["dtype"] = "bf16"
                elif dd in {"int1", "i1"}:
                    op["attrs"]["dtype"] = "i1"
                elif dd in {"int8", "i8"}:
                    op["attrs"]["dtype"] = "i8"
                elif dd in {"uint8", "u8"}:
                    op["attrs"]["dtype"] = "u8"
                elif dd in {"int32", "i32"}:
                    op["attrs"]["dtype"] = "i32"
                elif dd in {"int64", "i64"}:
                    op["attrs"]["dtype"] = "i64"
                elif dd in {"bool", "boolean"}:
                    op["attrs"]["dtype"] = "bool"

        # Canonicalize scalar shorthand for numeric binary ops into explicit const + binary op.
        # This keeps downstream stages simple/strict (interpreter, C lowering, RVV lowering).
        if op.get("op") in {"add", "sub", "mul", "div", "max", "min"}:
            inps = op.get("inputs") or []
            if isinstance(inps, list) and len(inps) == 1:
                attrs = op["attrs"]
                scalar_val = None
                for k in ("scalar", "addend", "subtract", "mul_factor", "divisor", "other", "const", "rhs_const", "rhs"):
                    if k in attrs:
                        scalar_val = attrs.pop(k)
                        break
                if scalar_val is not None:
                    const_out = _fresh_name(f"{op.get('output') or f'const_{idx}'}__const")
                    const_dtype = _infer_scalar_dtype(op)
                    ops.append({"op": "const", "inputs": [], "output": const_out, "attrs": {"value": scalar_val, "dtype": const_dtype}})
                    produced_outputs.append(const_out)
                    op["inputs"] = [inps[0], const_out]
        # Fill reshape shape from output tensor shape if missing
        if op.get("op") == "reshape":
            shape = op["attrs"].get("shape")
            if not shape:
                out_name = op.get("output")
                if out_name and out_name in tensors and "shape" in tensors[out_name]:
                    op["attrs"]["shape"] = tensors[out_name]["shape"]
        # For reduce_sum allow "axes" to map to dims
        if op.get("op") in {"reduce_sum", "reduce_max", "reduce_any"}:
            attrs = op["attrs"]
            if "keepdims" not in attrs and "keep_dims" in attrs:
                attrs["keepdims"] = attrs.pop("keep_dims")
            if "dims" not in attrs and "dimensions" in attrs:
                attrs["dims"] = attrs.get("dimensions")
            if "dims" not in attrs and "axes" in attrs:
                attrs["dims"] = attrs.get("axes")
            if "dims" not in attrs and "axis" in attrs:
                attrs["dims"] = [attrs["axis"]] if not isinstance(attrs["axis"], list) else attrs["axis"]
            if "dims" not in attrs:
                attrs["dims"] = [0]
        # Canonicalize + SSA-rename output names (e.g., store_C -> C, and ensure unique outputs).
        if "output" in op and op["output"]:
            base_out = _canonical_tensor_ref(op["output"])
            out_name = base_out
            if isinstance(out_name, str) and out_name:
                # Enforce unique op outputs by SSA-renaming duplicates (common LLM mistake).
                if out_name in seen_op_outputs:
                    new_name = _fresh_name(out_name)
                    # Preserve typing info when available (helps interpreter/backends).
                    if out_name in tensors and new_name not in tensors and isinstance(tensors.get(out_name), dict):
                        tensors[new_name] = dict(tensors[out_name])
                    out_name = new_name
                op["output"] = out_name
                seen_op_outputs.add(out_name)
                current_ssa[str(base_out)] = str(out_name)
            else:
                op["output"] = out_name
            produced_outputs.append(op["output"])
            used_names.add(op["output"])
        ops.append(op)
    data["ops"] = ops

    # Add additional shape symbols that appear in op attrs (reshape/broadcast/iota).
    for op in ops:
        attrs = op.get("attrs") or {}
        for key in ("shape", "out_shape"):
            v = attrs.get(key)
            if isinstance(v, list):
                for dim in v:
                    if isinstance(dim, str) and dim:
                        shape_symbols.add(dim)
        if op.get("op") == "iota":
            v = attrs.get("shape")
            if isinstance(v, list):
                for dim in v:
                    if isinstance(dim, str) and dim:
                        shape_symbols.add(dim)

    # Materialize symbolic scalar inputs as const ops derived from shape bindings.
    # This keeps "derived symbols" (e.g., group_size/HW/C/num_groups) inside IR rather than
    # relying on diff_runner to inject missing scalar arrays.
    produced_set = set(produced_outputs)
    used_set: set[str] = set()
    for op in ops:
        for inp in op.get("inputs") or []:
            if isinstance(inp, str):
                used_set.add(inp)
    prefix_ops: List[Dict[str, Any]] = []
    for name, t in list(tensors.items()):
        if name in produced_set:
            continue
        if name not in used_set:
            continue
        if not isinstance(t, dict):
            continue
        if t.get("shape") != []:
            continue
        # eps is a true numeric constant in our reference runners (unless explicitly varied elsewhere)
        if name == "eps":
            prefix_ops.append({"op": "const", "inputs": [], "output": name, "attrs": {"value": 1e-5, "dtype": t.get("dtype", "f32")}})
            produced_set.add(name)
            continue
        # Only lift scalars that are clearly "shape symbols" (dims/derived dims).
        if name in shape_symbols:
            prefix_ops.append({"op": "const", "inputs": [], "output": name, "attrs": {"value": name, "dtype": t.get("dtype", "i32")}})
            produced_set.add(name)
            continue
    if prefix_ops:
        data["ops"] = prefix_ops + data["ops"]

    # Fix up a few common semantic dtype issues from LLM outputs.
    # - reduce_any must produce a boolean tensor (i1/bool), not i32.
    # - ne produces boolean values; if its output is declared as a tensor, keep it boolean.
    # - identity should preserve dtype (especially for bool pipelines like reduce_any -> out).
    try:
        for op in ops:
            op_type = op.get("op")
            out_name = op.get("output")
            if not out_name or out_name not in tensors:
                continue
            if op_type in {"reduce_any", "ne"}:
                dtype = tensors[out_name].get("dtype")
                if dtype in {"i32", "i64", "f16", "bf16", "f32", "f64", "i8", "u8"}:
                    tensors[out_name]["dtype"] = "bool"
            if op_type == "identity":
                inps = op.get("inputs") or []
                if inps and inps[0] in tensors:
                    src_dt = tensors[inps[0]].get("dtype")
                    if isinstance(src_dt, str) and src_dt in {"bool", "i1"}:
                        tensors[out_name]["dtype"] = "bool"
    except Exception:
        # Best-effort normalization; validation will catch anything inconsistent.
        pass

    # normalize parallel_axes possibly given as list of objects
    pa_raw = data.get("parallel_axes") or []
    # Some providers emit parallel_axes as an object (axis -> role/meta). Use keys.
    if isinstance(pa_raw, dict):
        pa_raw = [k for k in pa_raw.keys() if isinstance(k, str)]
    if isinstance(pa_raw, list) and pa_raw and isinstance(pa_raw[0], dict):
        pa_raw = [p.get("name") for p in pa_raw if isinstance(p, dict) and "name" in p]
    # collect known symbolic axes from raw tensor dicts
    known_axes = set()
    for t in tensors.values():
        if isinstance(t, dict):
            for d in t.get("shape", []):
                if isinstance(d, str):
                    known_axes.add(d)
    if isinstance(pa_raw, list):
        pa_raw = [ax for ax in pa_raw if ax in known_axes]
    data["parallel_axes"] = pa_raw

    # normalize axis_roles values to strings
    ar_raw = data.get("axis_roles") or {}
    if isinstance(ar_raw, dict):
        # Handle inverted form: {role: [axes]} or {role: axis}
        if ar_raw and all(k in AXIS_ROLE_VALUES for k in ar_raw.keys()):
            inverted = {}
            for role, axes in ar_raw.items():
                if isinstance(axes, list):
                    for ax in axes:
                        inverted[ax] = role
                else:
                    inverted[axes] = role
            ar_raw = inverted
        norm_ar: Dict[str, str] = {}
        for ax, val in ar_raw.items():
            if isinstance(val, str):
                role = val.strip()
                role_key = role.lower()
                role_norm = AXIS_ROLE_ALIASES.get(role_key, role_key)
                # Keep only supported roles; axis_roles is metadata and should not
                # hard-fail parsing for new frontends/kernels.
                if role_norm in AXIS_ROLE_VALUES:
                    norm_ar[ax] = role_norm
            elif isinstance(val, list) and val:
                first = val[0]
                role = first if isinstance(first, str) else str(first)
                role_key = role.strip().lower()
                role_norm = AXIS_ROLE_ALIASES.get(role_key, role_key)
                if role_norm in AXIS_ROLE_VALUES:
                    norm_ar[ax] = role_norm
            elif isinstance(val, dict):
                picked = val.get("role") or val.get("type") or val.get("name")
                if picked is not None:
                    role = str(picked)
                    role_key = role.strip().lower()
                    role_norm = AXIS_ROLE_ALIASES.get(role_key, role_key)
                    if role_norm in AXIS_ROLE_VALUES:
                        norm_ar[ax] = role_norm
            else:
                role = str(val)
                role_key = role.strip().lower()
                role_norm = AXIS_ROLE_ALIASES.get(role_key, role_key)
                if role_norm in AXIS_ROLE_VALUES:
                    norm_ar[ax] = role_norm
        # Keep derived/implicit axes (e.g., group_size/num_groups) even if they are not
        # explicit tensor shape symbols; downstream stages may still use this metadata.
        data["axis_roles"] = norm_ar


    outputs = data.get("outputs")
    produced_outputs = [op.get("output") for op in ops if op.get("output")]
    if outputs is None:
        if ops:
            last_out = produced_outputs[-1] if produced_outputs else None
            if last_out and last_out in tensors:
                data["outputs"] = [last_out]
            else:
                raise LLMJsonParseError(
                    "outputs missing and cannot infer; ensure last op output is declared in tensors",
                    path="outputs",
                    hint="Add outputs field or align last op.output with a tensor name",
                )
        else:
            raise LLMJsonParseError("outputs missing and no ops to infer", path="outputs")
    else:
        # Allow common LLM variants: string -> singleton list.
        if isinstance(outputs, str):
            outputs = [outputs]
        # Some providers emit outputs as an object; accept common fields or keys.
        if isinstance(outputs, dict):
            picked = outputs.get("name") or outputs.get("output") or outputs.get("tensor") or outputs.get("id")
            if isinstance(picked, str) and picked:
                outputs = [picked]
            else:
                outputs = [k for k in outputs.keys() if isinstance(k, str)]
        # Some providers emit outputs as list of objects; accept ["name"] when present.
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
            picked: List[str] = []
            for i, o in enumerate(outputs):
                if not isinstance(o, dict):
                    continue
                name = o.get("name") or o.get("output") or o.get("tensor") or o.get("id")
                if isinstance(name, str) and name:
                    picked.append(name)
            if picked:
                outputs = picked
        # If a provider emitted a scalar (bool/number), treat it as "missing" and infer.
        if isinstance(outputs, (bool, int, float)):
            outputs = None
            if ops:
                last_out = produced_outputs[-1] if produced_outputs else None
                if last_out and last_out in tensors:
                    data["outputs"] = [last_out]
                else:
                    raise LLMJsonParseError(
                        "outputs missing and cannot infer; ensure last op output is declared in tensors",
                        path="outputs",
                        hint="Add outputs field as a list of output tensor names.",
                    )
            else:
                raise LLMJsonParseError("outputs missing and no ops to infer", path="outputs")
            outputs = data["outputs"]
        if not isinstance(outputs, list):
            raise LLMJsonParseError("outputs must be a list if provided", path="outputs")
        outputs = [_canonical_tensor_ref(o) for o in outputs]
        # Resolve SSA-renamed outputs: if the LLM reuses a name, we SSA-rename the
        # op.output; update `outputs` to match the final produced name.
        resolved: List[Any] = []
        for o in outputs:
            if isinstance(o, str) and o in current_ssa:
                resolved.append(str(current_ssa[o]))
            elif isinstance(o, str):
                resolved.append(o)
            else:
                resolved.append(o)
        outputs = resolved
        kept = [o for o in outputs if o in produced_outputs]
        missing = [o for o in outputs if o not in produced_outputs]
        if missing:
            raise LLMJsonParseError(
                f"outputs must be produced by ops; missing produced outputs: {missing}",
                path="outputs",
                hint="Add the missing ops that produce these outputs (do not use placeholder const).",
            )
        if not kept:
            raise LLMJsonParseError(
                "outputs list does not reference any produced op outputs",
                path="outputs",
                hint="Ensure each output name matches an op.output",
            )
        data["outputs"] = kept

    schedule = data.get("schedule")
    if schedule is not None and not isinstance(schedule, dict):
        raise LLMJsonParseError("schedule must be object", path="schedule")
    if isinstance(schedule, dict):
        # normalize common variants
        if "tile_size" in schedule and "tile_m" not in schedule and "tile_n" not in schedule:
            schedule["tile_n"] = schedule.pop("tile_size")
        # normalize 2D-tile naming variants (common from some providers)
        if "tile_y" in schedule and "tile_m" not in schedule:
            schedule["tile_m"] = schedule.pop("tile_y")
        if "tile_x" in schedule and "tile_n" not in schedule:
            schedule["tile_n"] = schedule.pop("tile_x")
        # Infer tile fields from memory_hint keys when present (keeps schedule visible even
        # if the provider reports tiles only via memory_hint).
        mh = schedule.get("memory_hint")
        if isinstance(mh, dict):
            keys = {str(k).upper() for k in mh.keys() if isinstance(k, str)}
            if "BLOCK_X" in keys and "BLOCK_Y" in keys:
                schedule.setdefault("tile_m", "BLOCK_Y")
                schedule.setdefault("tile_n", "BLOCK_X")
            if "BLOCK_M" in keys and "BLOCK_N" in keys:
                schedule.setdefault("tile_m", "BLOCK_M")
                schedule.setdefault("tile_n", "BLOCK_N")
            if "TILE_N" in keys:
                schedule.setdefault("tile_n", "TILE_N")
        if "tile_sizes" in schedule and isinstance(schedule["tile_sizes"], dict):
            ts = schedule.pop("tile_sizes")
            if "M" in ts and "tile_m" not in schedule:
                schedule["tile_m"] = ts["M"]
            if "N" in ts and "tile_n" not in schedule:
                schedule["tile_n"] = ts["N"]
            if "K" in ts and "tile_k" not in schedule:
                schedule["tile_k"] = ts["K"]
        if "axis_bindings" in schedule and isinstance(schedule["axis_bindings"], dict):
            allowed = {"tile_m", "tile_n", "tile_k", "vec_width"}
            cleaned = {k: v for k, v in schedule["axis_bindings"].items() if k in allowed}
            schedule["axis_bindings"] = cleaned
        data["schedule"] = schedule

    return data


def parse_candidate_json(d: Dict[str, Any]) -> CandidateIntent:
    normalized = normalize_candidate_json(d)
    raw = copy.deepcopy(normalized)
    problem_params = normalized.get("problem_params") or {}
    schedule_params = normalized.get("schedule_params") or {}
    try:
        intent = IntentFunction.from_json_dict(normalized)
    except IntentIRValidationError as e:
        raise LLMJsonParseError(str(e))
    return CandidateIntent(
        intent=intent,
        problem_params=problem_params,
        schedule_params=schedule_params,
        raw_json=raw,
    )


__all__ = [
    "LLMJsonParseError",
    "CandidateIntent",
    "merge_tensor_and_symbol_json",
    "normalize_candidate_json",
    "parse_candidate_json",
]
