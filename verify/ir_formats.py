from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from intent_ir.llm import strip_code_fence


def _balanced_delims(text: str) -> List[str]:
    """
    Lightweight delimiter balance check for MLIR-like textual IR.

    We intentionally only check (), [], {} because MLIR types use <> heavily and
    "->" confounds a naive angle-bracket matcher.
    """
    opens = {"(": ")", "[": "]", "{": "}"}
    closes = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in str(text):
        if ch in opens:
            stack.append(ch)
            continue
        if ch in closes:
            if not stack or stack[-1] != closes[ch]:
                return [f"unbalanced delimiter: unexpected '{ch}'"]
            stack.pop()
    if stack:
        return [f"unbalanced delimiter: missing '{opens[stack[-1]]}'"]
    return []


def _io_spec_tensor_ranks(io_spec: Any) -> Optional[List[int]]:
    if not isinstance(io_spec, dict):
        return None
    tensors = io_spec.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        return None
    ranks: List[int] = []
    for t in tensors.values():
        if not isinstance(t, dict):
            continue
        shape = t.get("shape")
        if not isinstance(shape, list):
            continue
        ranks.append(int(len(shape)))
    return ranks if ranks else None


def _io_spec_symbols(io_spec: Any) -> List[str]:
    if not isinstance(io_spec, dict):
        return []
    tensors = io_spec.get("tensors")
    if not isinstance(tensors, dict):
        return []
    out: List[str] = []
    for t in tensors.values():
        if not isinstance(t, dict):
            continue
        shape = t.get("shape")
        if not isinstance(shape, list):
            continue
        for d in shape:
            if isinstance(d, str) and d.strip() and d not in out:
                out.append(d)
    return out


def _extract_tensor_type_ranks_from_mlir(text: str) -> List[int]:
    """
    Extract ranks of tensor/memref types from the *function signature*.

    We use a heuristic regex and compute rank as (#segments split by 'x') - 1.
    """
    ranks: List[int] = []

    t = str(text)
    sig_start = t.find("func.func")
    if sig_start < 0:
        sig_start = t.find("func @")
    if sig_start < 0:
        return ranks
    sig_end = t.find("{", sig_start)
    if sig_end < 0:
        # Fallback: just take the first line.
        nl = t.find("\n", sig_start)
        sig_end = nl if nl >= 0 else len(t)
    sig = t[sig_start:sig_end]

    for m in re.finditer(r"(tensor|memref)<([^>]+)>", sig):
        body = m.group(2)
        parts = [p for p in body.split("x") if p]
        if len(parts) >= 2:
            ranks.append(int(len(parts) - 1))
    return ranks


def _iter_linalg_generic_windows(text: str) -> List[str]:
    """
    Return windows starting at each `linalg.generic` occurrence.

    We don't have a real MLIR parser here, so we slice text into per-op windows
    using the next occurrence as a delimiter (plus a max cap).
    """
    t = str(text)
    pos = [m.start() for m in re.finditer(r"linalg\.generic\b", t)]
    if not pos:
        return []
    out: List[str] = []
    for i, p in enumerate(pos):
        nxt = pos[i + 1] if (i + 1) < len(pos) else None
        end = int(nxt) if nxt is not None else min(len(t), p + 12000)
        end = min(len(t), max(p, end))
        out.append(t[p:end])
    return out


def _count_linalg_operands_in_sig(stmt: str) -> tuple[int, int]:
    """
    Return (n_ins, n_outs) by counting SSA values inside ins(...) and outs(...).
    """
    s = str(stmt)
    m = re.search(r"ins\((.*?)\)\s*outs\((.*?)\)", s, re.S)
    if not m:
        return (0, 0)
    ins = m.group(1)
    outs = m.group(2)
    n_ins = len(re.findall(r"%[A-Za-z_][A-Za-z0-9_]*", ins))
    n_outs = len(re.findall(r"%[A-Za-z_][A-Za-z0-9_]*", outs))
    return (int(n_ins), int(n_outs))


def _extract_inline_affine_maps(indexing_maps_blob: str) -> List[Tuple[int, int]]:
    """
    Return list of (domain_rank, uses_rank) for each inline affine_map in the blob.
    """
    out: List[Tuple[int, int]] = []
    for m in re.finditer(r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>", indexing_maps_blob):
        dom = [x.strip() for x in m.group(1).split(",") if x.strip()]
        rng = [x.strip() for x in m.group(2).split(",") if x.strip()]
        out.append((int(len(dom)), int(len(rng))))
    return out


def _extract_iterator_types(blob: str) -> List[str]:
    return [m.group(1) for m in re.finditer(r"\"(parallel|reduction|window)\"", blob)]


def _validate_linalg_generic_attrs(text: str) -> List[str]:
    """
    Lightweight structural checks that reflect *real* MLIR Linalg requirements:
    - linalg.generic should carry `indexing_maps` and `iterator_types`
    - #maps count should match ins+outs operand count
    - iterator_types length should match affine_map domain rank (when inlined)
    """
    errs: List[str] = []
    stmts = _iter_linalg_generic_windows(text)
    if not stmts:
        return ["missing any 'linalg.generic' op"]

    for idx, stmt in enumerate(stmts):
        prefix = f"linalg.generic[{idx}] "

        # Require the key attributes exist somewhere near the op.
        if "indexing_maps" not in stmt:
            errs.append(prefix + "missing indexing_maps attribute")
            continue
        if "iterator_types" not in stmt:
            errs.append(prefix + "missing iterator_types attribute")
            continue

        n_ins, n_outs = _count_linalg_operands_in_sig(stmt)
        n_operands = int(n_ins + n_outs)
        if n_operands <= 0:
            errs.append(prefix + "cannot determine ins/outs operand count")
            continue

        m_maps = re.search(r"indexing_maps\s*=\s*\[(.*?)\]", stmt, re.S)
        if not m_maps:
            errs.append(prefix + "cannot parse indexing_maps=[...] list")
            continue
        maps_blob = m_maps.group(1)
        maps = _extract_inline_affine_maps(maps_blob)
        if not maps:
            errs.append(prefix + "indexing_maps has no inline affine_map<...> entries (please inline maps)")
            continue
        if len(maps) != n_operands:
            errs.append(prefix + f"indexing_maps count mismatch: got={len(maps)} expected={n_operands} (ins={n_ins}, outs={n_outs})")

        m_it = re.search(r"iterator_types\s*=\s*\[(.*?)\]", stmt, re.S)
        if not m_it:
            errs.append(prefix + "cannot parse iterator_types=[...] list")
            continue
        iters = _extract_iterator_types(m_it.group(1))
        if not iters:
            errs.append(prefix + "iterator_types list empty or missing known iterator strings")
            continue

        dom_ranks = {int(d) for (d, _u) in maps}
        if len(dom_ranks) != 1:
            errs.append(prefix + f"affine_map domain ranks disagree: {sorted(dom_ranks)}")
        else:
            dom_rank = next(iter(dom_ranks))
            if int(dom_rank) != int(len(iters)):
                errs.append(prefix + f"iterator_types length mismatch: got={len(iters)} expected={dom_rank} (from affine_map domain)")

        # linalg.generic must have a region that yields results.
        if "linalg.yield" not in stmt:
            errs.append(prefix + "missing linalg.yield")

    return errs


def validate_mlir_linalg_text(text: str, *, io_spec: Any | None = None) -> List[str]:
    """
    Validate that the output looks like *syntactically plausible* MLIR (Linalg).

    This is intentionally NOT a full MLIR parser; E6 only needs a stable,
    deterministic "legal-looking IR" gate for comparing LLM output formats.
    """
    t = strip_code_fence(str(text or "")).strip()
    if not t:
        return ["empty output"]

    errs: List[str] = []
    errs += _balanced_delims(t)

    # Basic structure anchors.
    if "module" not in t:
        errs.append("missing 'module'")
    if "func.func" not in t and "func @" not in t:
        errs.append("missing 'func.func' or 'func @'")
    if "linalg." not in t:
        errs.append("missing any 'linalg.' op")

    # Linalg structural requirements (lightweight).
    errs += _validate_linalg_generic_attrs(t)

    # Common non-IR artifacts / placeholders.
    bad_markers = ["<TODO", "TODO", "…", "...", "PLACEHOLDER", "fill_me", "TBD"]
    if any(x in t for x in bad_markers):
        errs.append("contains placeholder markers")
    if "```" in t:
        errs.append("contains markdown code fences")

    # Optional structure completeness check against io_spec:
    # - total tensor/memref type count and rank multiset must match.
    exp_ranks = _io_spec_tensor_ranks(io_spec)
    if exp_ranks:
        got_ranks = _extract_tensor_type_ranks_from_mlir(t)
        # Accept duplicates (e.g., output both as argument and return); require
        # that the signature covers at least all expected tensor ranks.
        if len(got_ranks) < len(exp_ranks):
            errs.append(f"tensor type count mismatch: got={len(got_ranks)} expected_at_least={len(exp_ranks)}")
        else:
            exp_counts: Dict[int, int] = {}
            got_counts: Dict[int, int] = {}
            for r in exp_ranks:
                exp_counts[int(r)] = int(exp_counts.get(int(r), 0)) + 1
            for r in got_ranks:
                got_counts[int(r)] = int(got_counts.get(int(r), 0)) + 1
            missing: List[Tuple[int, int, int]] = []
            for rk, need in exp_counts.items():
                have = int(got_counts.get(int(rk), 0))
                if have < int(need):
                    missing.append((int(rk), int(have), int(need)))
            if missing:
                errs.append(f"tensor rank coverage mismatch: missing={missing} got={sorted(got_ranks)} expected={sorted(exp_ranks)}")

    return errs


def validate_mlir_linalg_text_lenient(text: str) -> List[str]:
    """
    Lenient MLIR(Linalg) plausibility checks for experiments.

    Rationale:
      - For E6.2 we want a *fair* comparison about "IR+contract honesty" under
        missing evidence, not about whether the LLM can remember every required
        `linalg.generic` attribute (indexing_maps/iterator_types).
      - This validator keeps stable structural anchors but does not enforce the
        full `linalg.generic` attribute schema.
    """
    t = strip_code_fence(str(text or "")).strip()
    if not t:
        return ["empty output"]

    errs: List[str] = []
    errs += _balanced_delims(t)
    if "module" not in t:
        errs.append("missing 'module'")
    if "func.func" not in t and "func @" not in t:
        errs.append("missing 'func.func' or 'func @'")
    if "linalg." not in t:
        errs.append("missing any 'linalg.' op")

    bad_markers = ["<TODO", "TODO", "…", "...", "PLACEHOLDER", "fill_me", "TBD"]
    if any(x in t for x in bad_markers):
        errs.append("contains placeholder markers")
    if "```" in t:
        errs.append("contains markdown code fences")
    return errs


@dataclass(frozen=True)
class TileDslValidation:
    ok: bool
    errors: List[str]


def validate_tile_dsl_json(obj: Any, *, io_spec: Any | None = None) -> List[str]:
    """
    Validate a minimal tile-centric schedule JSON.

    Expected shape (loose):
      {
        "schema_version": "tile_dsl_v0",
        "kernel": "...",
        "schedule": {
          "tile": {"M": 128, "N": 128, ...},
          "vec_width": 8,
          "num_threads": 1
        }
      }
    """
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    d: Dict[str, Any] = obj
    errs: List[str] = []

    sv = d.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        errs.append("missing schema_version")
    kernel = d.get("kernel")
    if not isinstance(kernel, str) or not kernel.strip():
        errs.append("missing kernel name")
    sched = d.get("schedule")
    if not isinstance(sched, dict):
        errs.append("missing schedule dict")
        return errs

    tile = sched.get("tile")
    if not isinstance(tile, dict) or not tile:
        errs.append("missing schedule.tile (non-empty dict)")
    else:
        for k, v in tile.items():
            if not isinstance(k, str) or not k.strip():
                errs.append("schedule.tile has non-string axis key")
                break
            if not isinstance(v, int) or int(v) <= 0:
                errs.append(f"schedule.tile.{k} must be a positive int")
                break

    vw = sched.get("vec_width")
    if vw is not None and (not isinstance(vw, int) or int(vw) <= 0):
        errs.append("schedule.vec_width must be a positive int if present")
    nt = sched.get("num_threads")
    if nt is not None and (not isinstance(nt, int) or int(nt) <= 0):
        errs.append("schedule.num_threads must be a positive int if present")

    # Optional structure completeness check: schedule axes should be drawn from
    # known symbolic dimensions in io_spec (when available).
    if not errs and io_spec is not None:
        syms = set(_io_spec_symbols(io_spec))
        if syms:
            bad = [str(k) for k in (tile or {}).keys() if str(k) not in syms]
            if bad:
                errs.append(f"schedule.tile has unknown axes: {bad} (known={sorted(syms)})")

    return errs


__all__ = ["validate_mlir_linalg_text", "validate_tile_dsl_json"]
