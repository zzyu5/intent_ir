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

from .ir_types import IntentFunction, IntentIRValidationError, parse_layout

AXIS_ROLE_VALUES = {"spatial", "reduction", "batch", "channel"}


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
                raise LLMJsonParseError(str(e), path=f"tensors.{name}.layout")
        # normalize dtype if using long form
        dtype = t.get("dtype")
        if dtype in {"float16", "fp16"}:
            t["dtype"] = "f16"
        elif dtype in {"float32", "fp32", "float"}:
            t["dtype"] = "f32"
        if "shape" not in t:
            t["shape"] = []
        # replace tile symbols with axis symbols for downstream bindings
        t["shape"] = [replacements.get(d, d) for d in t["shape"]]
    tensors = tensors_raw
    data["tensors"] = tensors

    # Known arg tensors to enforce original shapes (no invented grouped shapes)
    arg_tensor_names = set(tensors.keys())
    ops_raw = data.get("ops") or []
    if not isinstance(ops_raw, list):
        raise LLMJsonParseError("ops must be list", path="ops")
    ops: List[Dict[str, Any]] = []
    produced_outputs: List[str] = []
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
        # Backward-compat: treat a prior dedicated high-level op name as custom_call.
        if op.get("op") == "upsample_bicubic2d_aa":
            op["attrs"] = dict(op.get("attrs") or {})
            op["attrs"].setdefault("callee", "upsample_bicubic2d_aa")
            op["op"] = "custom_call"
        if op.get("op") == "elemwise":
            inner = op["attrs"].get("op") or op.get("name")
            if inner:
                op["op"] = inner
                op["attrs"].pop("op", None)
        if op.get("op") == "or":
            op["op"] = "max"
        if "output" not in op and "outputs" in op and isinstance(op["outputs"], list) and op["outputs"]:
            op["output"] = op["outputs"][0]
        if "attrs" not in op or op["attrs"] is None:
            op["attrs"] = {}
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
            if "dims" not in attrs and "axis" in attrs:
                attrs["dims"] = [attrs["axis"]] if not isinstance(attrs["axis"], list) else attrs["axis"]
            if "dims" not in attrs:
                attrs["dims"] = [0]
        if "output" in op and op["output"]:
            produced_outputs.append(op["output"])
        ops.append(op)
    data["ops"] = ops

    # Fix up a few common semantic dtype issues from LLM outputs.
    # - reduce_any must produce a boolean tensor (i1/bool), not i32.
    # - ne produces boolean values; if its output is declared as a tensor, keep it boolean.
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
    except Exception:
        # Best-effort normalization; validation will catch anything inconsistent.
        pass

    # normalize parallel_axes possibly given as list of objects
    pa_raw = data.get("parallel_axes") or []
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
                norm_ar[ax] = val
            elif isinstance(val, list) and val:
                first = val[0]
                norm_ar[ax] = first if isinstance(first, str) else str(first)
            elif isinstance(val, dict):
                picked = val.get("role") or val.get("type") or val.get("name")
                if picked is not None:
                    norm_ar[ax] = picked
                else:
                    norm_ar[ax] = str(val)
            else:
                norm_ar[ax] = str(val)
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
        if not isinstance(outputs, list):
            raise LLMJsonParseError("outputs must be a list if provided", path="outputs")
        kept = [o for o in outputs if o in produced_outputs]
        missing = [o for o in outputs if o not in produced_outputs]
        if missing:
            # add placeholder const ops for missing outputs to keep structure
            for m in missing:
                ops.append({"op": "const", "inputs": [], "output": m, "attrs": {"value": f"placeholder:{m}"}})
                if m not in tensors:
                    tensors[m] = {"dtype": "f32", "shape": [], "layout": "row_major"}
                produced_outputs.append(m)
                kept.append(m)
        if not kept and produced_outputs:
            kept = [produced_outputs[-1]]
        data["outputs"] = kept

    schedule = data.get("schedule")
    if schedule is not None and not isinstance(schedule, dict):
        raise LLMJsonParseError("schedule must be object", path="schedule")
    if isinstance(schedule, dict):
        # normalize common variants
        if "tile_size" in schedule and "tile_m" not in schedule and "tile_n" not in schedule:
            schedule["tile_n"] = schedule.pop("tile_size")
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
