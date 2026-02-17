"""
E6.2: Evidence ablation + contract calibration / honesty.

Motivation (paper-facing):
  - MLIR Linalg is a strong general-purpose IR, but it is not designed as an
    "LLM output carrier" that must also declare per-kernel uncertainty boundaries
    (contract) and support verification-oriented workflows.
  - IntentIR's advantage is not "more expressive", but "better suited for kernel
    lifting": partial outputs, explicit contracts, and consistent structure.

Why this E6.2 is fairer than E6.1:
  - We do NOT score IntentIR using certificate-only checks that Linalg lacks.
  - Instead, BOTH representations must output "IR + contract", and we evaluate:
      (1) Overclaim under missing evidence (FULL when evidence is insufficient)
      (2) Abstention quality (PARTIAL/OOS with explicit assumptions)
      (3) IR/contract consistency (contract claims match IR structure)

Method:
  - Start from the same KernelDescriptor (source + evidence).
  - Apply ablations to the EVIDENCE provided to the LLM (drop anchors, drop mask
    details, etc.).
  - Ask the LLM to emit:
      - IntentIR JSON with embedded contract at top-level key `contract`
      - Linalg MLIR text + external sidecar contract JSON (in the same response)

This experiment is lightweight: it does not run dynamic diff/remote execution.
It is meant to support the narrative that IntentIR is a better "semantic recovery
artifact" under uncertainty.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.ir.ir_types import IntentFunction  # noqa: E402
from intent_ir.llm import DEFAULT_MODEL, LLMClientError, chat_completion, parse_json_block, strip_code_fence  # noqa: E402
from intent_ir.parser import LLMJsonParseError, parse_candidate_json  # noqa: E402
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor  # noqa: E402
from verify.ir_formats import validate_mlir_linalg_text_contract_grade, validate_mlir_linalg_text_lenient  # noqa: E402


LEVEL_ORDER: dict[str, int] = {"OUT_OF_SCOPE": 0, "PARTIAL": 1, "FULL": 2}
LEVELS = set(LEVEL_ORDER.keys())


def _kernels_from_pipeline(frontend: str, suite: str) -> List[str]:
    if suite not in {"smoke", "coverage"}:
        raise ValueError(f"unknown suite: {suite}")
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

        specs_fn = default_kernel_specs if suite == "smoke" else coverage_kernel_specs
        return [s.name for s in specs_fn()]
    raise ValueError(f"unknown frontend: {frontend}")


def _spec_from_pipeline(frontend: str, kernel: str) -> Any:
    if frontend == "triton":
        from pipeline.triton.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    if frontend == "tilelang":
        from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    if frontend == "cuda":
        from pipeline.cuda.core import coverage_kernel_specs  # noqa: PLC0415

        for s in coverage_kernel_specs():
            if getattr(s, "name", None) == kernel:
                return s
        raise KeyError(kernel)
    raise ValueError(f"unknown frontend: {frontend}")


def _descriptor_from_json(d: Dict[str, Any]) -> KernelDescriptor:
    art = d.get("artifacts") if isinstance(d.get("artifacts"), dict) else {}
    artifacts = KernelArtifactBundle(
        ttir_text=art.get("ttir_text"),
        ttir_path=art.get("ttir_path"),
        llvm_ir_text=art.get("llvm_ir_text"),
        ptx_text=art.get("ptx_text"),
        extra=(art.get("extra") if isinstance(art.get("extra"), dict) else {}),
    )
    return KernelDescriptor(
        schema_version=str(d.get("schema_version") or "kernel_desc_v1.0"),
        name=str(d.get("name") or "kernel"),
        frontend=str(d.get("frontend") or "triton"),  # type: ignore[arg-type]
        source_kind=str(d.get("source_kind") or "source"),  # type: ignore[arg-type]
        source_text=str(d.get("source_text") or ""),
        launch=(d.get("launch") if isinstance(d.get("launch"), dict) else {}),
        io_spec=(d.get("io_spec") if isinstance(d.get("io_spec"), dict) else {}),
        artifacts=artifacts,
        frontend_facts=(d.get("frontend_facts") if isinstance(d.get("frontend_facts"), dict) else {}),
        frontend_constraints=(d.get("frontend_constraints") if isinstance(d.get("frontend_constraints"), dict) else {}),
        meta=(d.get("meta") if isinstance(d.get("meta"), dict) else {}),
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_descriptor(*, frontend: str, kernel: str, artifact_dir: Path) -> KernelDescriptor:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec_from_pipeline(frontend, kernel)
    from pipeline import registry as pipeline_registry  # noqa: PLC0415

    adapter = pipeline_registry.get(str(frontend))
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(artifact_dir)
    if str(frontend) == "cuda":
        ptx_entry = getattr(spec, "ptx_entry", None)
        if isinstance(ptx_entry, str) and ptx_entry.strip():
            desc.meta["ptx_entry"] = str(ptx_entry)
    desc = adapter.ensure_artifacts(desc, spec)
    facts = adapter.extract_facts(desc)
    _ = adapter.extract_constraints(desc, facts)
    (artifact_dir / f"{kernel}.descriptor.json").write_text(json.dumps(desc.to_json_dict(), indent=2), encoding="utf-8")
    return desc


def _load_or_prepare_descriptor(*, frontend: str, kernel: str, artifact_dir: Path, refresh: bool) -> KernelDescriptor:
    p = artifact_dir / f"{kernel}.descriptor.json"
    if p.exists() and not refresh:
        return _descriptor_from_json(_load_json(p))
    return _prepare_descriptor(frontend=frontend, kernel=kernel, artifact_dir=artifact_dir)


def _anchors_from_descriptor(desc: KernelDescriptor) -> Dict[str, Any]:
    ff = desc.frontend_facts or {}
    if isinstance(ff.get("anchors"), dict):
        return dict(ff["anchors"])
    # Triton historically used flattened keys; keep a fallback.
    return {k: ff.get(k) for k in ("has_dot", "has_reduce", "has_atomic", "has_barrier", "has_async", "has_copy")}


def _evidence_view(desc: KernelDescriptor, *, ablation: str) -> Dict[str, Any]:
    """
    Build the (possibly ablated) evidence payload provided to the LLM.

    IMPORTANT: This is the experimental knob. The descriptor itself stays intact.
    """
    fc = desc.frontend_constraints or {}
    meta = fc.get("meta") if isinstance(fc.get("meta"), dict) else {}
    ev: Dict[str, Any] = {
        "kernel": str(desc.name),
        "frontend": str(desc.frontend),
        "io_spec": desc.io_spec,
        "launch": desc.launch,
        "anchors": _anchors_from_descriptor(desc),
        "constraints": {
            "needs_mask": fc.get("needs_mask"),
            "symbol_ranges": meta.get("symbol_ranges"),
            "predicate_clauses": meta.get("predicate_clauses"),
        },
        "ablation": str(ablation),
    }

    a = str(ablation)
    if a in {"no_launch", "no_launch_no_mask", "no_launch_no_anchors", "no_all"}:
        ev.pop("launch", None)
    if a in {"no_mask", "no_launch_no_mask", "no_all"}:
        c = ev.get("constraints")
        if isinstance(c, dict):
            c.pop("needs_mask", None)
            c.pop("predicate_clauses", None)
    if a in {"no_anchors", "no_launch_no_anchors", "no_all"}:
        ev.pop("anchors", None)
    return ev


def _oracle_from_evidence(ev: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Define the *contract oracle* purely from the provided evidence.

    This is the key fairness point: we do not use extra verifier-only signals for
    IntentIR. Both reps are judged against the same "evidence sufficiency" oracle.
    """
    gaps: List[str] = []
    anchors = ev.get("anchors")
    if not isinstance(anchors, dict):
        gaps.append("anchors")
        return "PARTIAL", gaps

    if bool(anchors.get("has_atomic")) or bool(anchors.get("has_barrier")):
        return "OUT_OF_SCOPE", []

    c = ev.get("constraints")
    if not isinstance(c, dict):
        gaps.append("constraints")
        return "PARTIAL", gaps

    nm = c.get("needs_mask")
    if not isinstance(nm, bool):
        gaps.append("needs_mask")
        return "PARTIAL", gaps

    if nm:
        pc = c.get("predicate_clauses")
        if not isinstance(pc, list) or not pc:
            gaps.append("mask_details")
            return "PARTIAL", gaps

    return "FULL", []


def _validate_contract_obj(obj: Any) -> List[str]:
    if not isinstance(obj, dict):
        return ["contract must be an object"]
    sv = obj.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        return ["contract.schema_version must be a non-empty string"]
    lvl = obj.get("level")
    if lvl not in LEVELS:
        return [f"contract.level must be one of {sorted(LEVELS)}"]
    assumptions = obj.get("assumptions")
    if assumptions is None:
        return ["contract.assumptions missing"]
    if not isinstance(assumptions, list) or not all(isinstance(x, str) for x in assumptions):
        return ["contract.assumptions must be a list of strings"]
    if lvl == "FULL" and assumptions:
        return ["contract.level=FULL requires empty assumptions"]
    if lvl == "PARTIAL" and not assumptions:
        return ["contract.level=PARTIAL requires non-empty assumptions"]
    if lvl == "OUT_OF_SCOPE" and not assumptions:
        return ["contract.level=OUT_OF_SCOPE requires non-empty assumptions"]
    gaps = obj.get("evidence_gaps")
    if gaps is None:
        return ["contract.evidence_gaps missing"]
    if not isinstance(gaps, list) or not all(isinstance(x, str) for x in gaps):
        return ["contract.evidence_gaps must be a list of strings"]
    claims = obj.get("claims")
    if claims is None:
        return ["contract.claims missing"]
    if not isinstance(claims, dict):
        return ["contract.claims must be an object"]
    return []


def _assumption_precision(contract: Dict[str, Any], oracle_gaps: List[str]) -> Tuple[bool, List[str]]:
    """
    Assumption / evidence-gap precision.

    For PARTIAL contracts induced by missing evidence, the contract should
    explicitly mention the missing categories (at least in `evidence_gaps`).
    """
    if contract.get("level") != "PARTIAL":
        return True, []
    if not oracle_gaps:
        return True, []
    eg = contract.get("evidence_gaps")
    egaps = [str(x).strip().lower() for x in (eg if isinstance(eg, list) else []) if str(x).strip()]
    errors: List[str] = []
    for g in oracle_gaps:
        key = str(g).strip().lower()
        if not key:
            continue
        if not any((key == x) or (key in x) or (x in key) for x in egaps):
            errors.append(f"evidence_gaps missing '{g}'")
    return not errors, errors


def _anchor_errors_intentir(intent: IntentFunction, claimed_level: str, anchors: Dict[str, Any]) -> List[str]:
    """
    IR/contract consistency checks for IntentIR under FULL claims.

    We key these checks off the provided evidence anchors (fairness): if the
    evidence says "has_dot", the IR should contain a matmul-like anchor; if it
    says "has_reduce", the IR should contain reduce_* (or softmax).
    """
    if str(claimed_level) != "FULL":
        return []
    ops = [str(op.op) for op in (intent.ops or [])]
    errs: List[str] = []
    if bool(anchors.get("has_dot")) and "matmul" not in ops:
        errs.append("missing matmul anchor for evidence.has_dot")
    if bool(anchors.get("has_reduce")):
        # Treat matmul as a reduction only when has_dot is also true (GEMM-like).
        has_reduce_like = any(o.startswith("reduce_") for o in ops) or ("softmax" in ops)
        if bool(anchors.get("has_dot")):
            has_reduce_like = bool(has_reduce_like or ("matmul" in ops))
        if not has_reduce_like:
            errs.append("missing reduce_* (or softmax) for evidence.has_reduce")
    # If evidence provides neither dot nor reduce, still require non-empty semantics.
    if not bool(anchors.get("has_dot")) and not bool(anchors.get("has_reduce")):
        if not ops:
            errs.append("no ops emitted under FULL claim")
    return errs


def _anchor_errors_linalg(mlir_text: str, claimed_level: str, anchors: Dict[str, Any]) -> List[str]:
    if str(claimed_level) != "FULL":
        return []
    t = str(mlir_text)
    errs: List[str] = []

    def _has_reduction_generic(*, min_ins: int) -> bool:
        for m in re.finditer(r"linalg\.generic\b", t):
            win = t[m.start() : min(len(t), m.start() + 6000)]
            if '"reduction"' not in win:
                continue
            mm = re.search(r"ins\((.*?)\)\s*outs\(", win, re.S)
            if not mm:
                continue
            n_ins = len(re.findall(r"%[A-Za-z_][A-Za-z0-9_]*", mm.group(1)))
            if n_ins >= int(min_ins):
                return True
        return False

    if bool(anchors.get("has_dot")):
        has_matmul = "linalg.matmul" in t
        has_matmul_like_generic = _has_reduction_generic(min_ins=2)
        if not (has_matmul or has_matmul_like_generic):
            errs.append("missing matmul anchor for evidence.has_dot (need linalg.matmul or reduction linalg.generic with >=2 inputs)")
    if bool(anchors.get("has_reduce")):
        # Accept matmul as a reduction anchor (GEMM-like kernels).
        has_reduce = _has_reduction_generic(min_ins=1) or ("linalg.reduce" in t) or ("linalg.matmul" in t)
        if not has_reduce:
            errs.append('missing reduction anchor for evidence.has_reduce (need linalg.generic with "reduction" iterator_types)')
    if not bool(anchors.get("has_dot")) and not bool(anchors.get("has_reduce")):
        if "linalg." not in t:
            errs.append("missing any linalg.* op under FULL claim")
    return errs


def _binding_errors_intentir(intent: IntentFunction, claimed_level: str, anchors: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
    """
    Contract↔IR binding checks (representation-sensitive) for FULL claims.

    The goal is to stress the *coordination burden*:
      - IntentIR embeds the contract inside the same structured object.
      - Linalg uses a sidecar contract; binding is easier to drift.
    """
    if str(claimed_level) != "FULL":
        return []
    claims = contract.get("claims")
    if not isinstance(claims, dict):
        return ["contract.claims missing or not an object under FULL claim"]

    ops = list(intent.ops or [])
    errs: List[str] = []
    if bool(anchors.get("has_dot")):
        w = claims.get("dot_witness")
        if not isinstance(w, str) or not w.strip():
            errs.append("missing contract.claims.dot_witness for evidence.has_dot")
        else:
            if not any(str(op.op) == "matmul" and str(op.output) == str(w) for op in ops):
                errs.append("contract.claims.dot_witness not bound to any matmul op output")

    if bool(anchors.get("has_reduce")):
        w = claims.get("reduce_witness")
        if not isinstance(w, str) or not w.strip():
            errs.append("missing contract.claims.reduce_witness for evidence.has_reduce")
        else:
            allow_reduce_ops = ("reduce_", "softmax")
            ok = any(
                (str(op.op).startswith(allow_reduce_ops[0]) or str(op.op) == allow_reduce_ops[1]) and str(op.output) == str(w)
                for op in ops
            )
            # GEMM-like: allow reduce_witness to bind to matmul output (matmul is a reduction over K).
            if not ok and bool(anchors.get("has_dot")):
                ok = any(str(op.op) == "matmul" and str(op.output) == str(w) for op in ops)
            if not ok:
                errs.append("contract.claims.reduce_witness not bound to any reduce_* (or softmax) op output")
    return errs


def _binding_errors_linalg(mlir_text: str, claimed_level: str, anchors: Dict[str, Any], contract: Dict[str, Any]) -> List[str]:
    if str(claimed_level) != "FULL":
        return []
    claims = contract.get("claims")
    if not isinstance(claims, dict):
        return ["contract.claims missing or not an object under FULL claim"]

    t = str(mlir_text)
    errs: List[str] = []

    def _check_ssa_witness(key: str, allow_ops: tuple[str, ...]) -> None:
        w = claims.get(key)
        if not isinstance(w, str) or not w.strip():
            errs.append(f"missing contract.claims.{key} under FULL claim")
            return
        w2 = str(w).strip()
        if not re.fullmatch(r"%[A-Za-z_][A-Za-z0-9_]*", w2):
            errs.append(f"contract.claims.{key} must be an SSA name like %0/%x (got={w2!r})")
            return
        allow_pat = "|".join([re.escape(x) for x in allow_ops])
        if not re.search(rf"(^|\n)\s*{re.escape(w2)}\s*=\s*linalg\.({allow_pat})\b", t, re.M):
            errs.append(f"contract.claims.{key} not bound to any linalg.{allow_pat} result SSA")

    if bool(anchors.get("has_dot")):
        _check_ssa_witness("dot_witness", ("matmul", "generic"))
    if bool(anchors.get("has_reduce")):
        # Allow reduce witness to bind to matmul result for GEMM-like kernels.
        allow = ("generic", "matmul") if bool(anchors.get("has_dot")) else ("generic",)
        _check_ssa_witness("reduce_witness", allow)
    return errs


def _messages_for_intentir(*, desc: KernelDescriptor, evidence: Dict[str, Any], feedback: List[str]) -> List[Dict[str, str]]:
    fb = "\n".join([f"- {x}" for x in feedback]) if feedback else ""
    sys_prompt = "\n".join(
        [
            "You are a compiler engineer.",
            "Task: produce IntentIR JSON (v1.1) PLUS an embedded contract.",
            "Output rules:",
            "- Output STRICT JSON only (no markdown fences, no prose).",
            "- The JSON MUST be a valid IntentIR function object with keys: name, tensors, ops, outputs, contract.",
            "- ops MUST be a list of objects; each op MUST have: op (string), inputs (list[str]), output (string), optional attrs (object).",
            "- Use high-level IntentIR ops (matmul, reduce_*, softmax, exp, add/sub/mul/div, ne/lt/le/gt/ge/and/or, rsqrt, where, broadcast_in_dim, gather, reshape, const).",
            "- Macro ops are allowed when appropriate (e.g., upsample_bicubic2d_aa) and will be expanded later.",
            "- Do NOT emit low-level pseudo-ops like load/store/ptr/arithmetic SSA.",
            "- Do NOT use op name 'compare' (use ne/lt/le/gt/ge instead).",
            "- Use `const` to model scalar literals (e.g., 0.0) as SSA tensors when needed for comparisons.",
            "- Prefer reshape/transpose/broadcast_in_dim for layout changes.",
            "- IMPORTANT: Do NOT use gather unless you also create explicit index tensors (e.g., via iota). If you need upsample/resize semantics, prefer the macro op upsample_bicubic2d_aa instead of ad-hoc gather chains.",
            "- Embed contract at top-level key `contract` (an object). Do NOT put contract under meta.*.",
            "",
            "Contract schema (contract):",
            "- Required keys: schema_version (string), level (FULL/PARTIAL/OUT_OF_SCOPE), assumptions (list of strings), evidence_gaps (list of strings), claims (object)",
            "- level=FULL => assumptions MUST be empty.",
            "- level=PARTIAL => assumptions MUST be non-empty.",
            "- level=OUT_OF_SCOPE => include a short assumption explaining why.",
            "",
            "Contract↔IR binding (IMPORTANT for level=FULL):",
            "- If EVIDENCE.anchors.has_dot=true: set contract.claims.dot_witness to the output name of the matmul op.",
            "- If EVIDENCE.anchors.has_reduce=true: set contract.claims.reduce_witness to the output name of the reduce_* (or softmax) op.",
            "- If BOTH has_dot=true and has_reduce=true (GEMM-like): you MAY set reduce_witness == dot_witness (matmul output).",
            "",
            "Minimal example (shape symbols are allowed):",
            "{",
            '  "name": "kernel",',
            '  "tensors": {"X": {"dtype": "f32", "shape": ["M","N"], "layout": "row_major"}, "O": {"dtype": "f32", "shape": ["M"], "layout": "row_major"}},',
            '  "ops": [{"op": "reduce_sum", "inputs": ["X"], "output": "O", "attrs": {"reduce_dims": [1], "keepdims": false}}],',
            '  "outputs": ["O"],',
            '  "contract": {"schema_version": "intentir_contract_v1", "level": "FULL", "assumptions": [], "evidence_gaps": [], "claims": {}}',
            "}",
            "",
            "Honesty rule (IMPORTANT):",
            "- Contract level must be derived ONLY from the provided EVIDENCE object below.",
            "- If anchors/mask details are missing in EVIDENCE, you MUST NOT claim FULL; use PARTIAL and list the missing evidence in evidence_gaps.",
            "",
            "evidence_gaps vocabulary (use these exact tokens, no paraphrases):",
            '- "anchors" (EVIDENCE has no anchors field)',
            '- "needs_mask" (EVIDENCE.constraints.needs_mask is missing)',
            '- "mask_details" (needs_mask=true but EVIDENCE.constraints.predicate_clauses missing/empty)',
            '- "constraints" (EVIDENCE has no constraints field)',
            '- "launch" (EVIDENCE has no launch field)',
            "If level=PARTIAL, evidence_gaps MUST include all missing categories that justify PARTIAL.",
            'Example (PARTIAL due to missing needs_mask): {"schema_version":"intentir_contract_v1","level":"PARTIAL","assumptions":["mask info missing"],"evidence_gaps":["needs_mask"],"claims":{}}',
            "",
            "Do not guess missing evidence from the source code.",
        ]
    )
    user = "\n".join(
        [
            f"Kernel: {desc.name}",
            f"Frontend: {desc.frontend}",
            "",
            "SOURCE:",
            str(desc.source_text[:4000]),
            "",
            "EVIDENCE (JSON, possibly ablated):",
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            "",
            ("Feedback:\n" + fb) if fb else "",
        ]
    ).strip()
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def _messages_for_linalg(*, desc: KernelDescriptor, evidence: Dict[str, Any], feedback: List[str]) -> List[Dict[str, str]]:
    fb = "\n".join([f"- {x}" for x in feedback]) if feedback else ""
    sys_prompt = "\n".join(
        [
            "You are a compiler engineer.",
            "Task: output MLIR (Linalg dialect) AND an external sidecar contract.",
            "Output rules:",
            "- Output plain text only (no markdown fences).",
            "- First output the MLIR module text.",
            "- Then output a marker line: CONTRACT_JSON:",
            "- After CONTRACT_JSON:, output STRICT JSON for the contract object.",
            "- The MLIR MUST contain at least one linalg.* op (prefer linalg.generic or linalg.matmul).",
            "",
            "Contract JSON schema:",
            "- Required keys: schema_version (string), level (FULL/PARTIAL/OUT_OF_SCOPE), assumptions (list of strings), evidence_gaps (list of strings), claims (object)",
            "- level=FULL => assumptions MUST be empty.",
            "- level=PARTIAL => assumptions MUST be non-empty.",
            "",
            "Contract↔IR binding (IMPORTANT for level=FULL):",
            "- If EVIDENCE.anchors.has_dot=true: set contract.claims.dot_witness to the SSA name (e.g., %0) of the linalg.matmul (or matmul-like linalg.generic) result.",
            "- If EVIDENCE.anchors.has_reduce=true: set contract.claims.reduce_witness to the SSA name (e.g., %1) of a reduction linalg.generic result.",
            "- If BOTH has_dot=true and has_reduce=true (GEMM-like): you MAY set reduce_witness == dot_witness (matmul result SSA).",
            "",
            "Minimal MLIR skeleton (you may simplify types/dims):",
            "#map0 = affine_map<(i,j) -> (i,j)>",
            "module {",
            "  func.func @kernel(%arg0: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {",
            '    %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel","parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%out : tensor<?x?xf32>) {',
            '      ^bb0(%a: f32, %b: f32):',
            "        linalg.yield %a : f32",
            "    } -> tensor<?x?xf32>",
            "    return %0 : tensor<?x?xf32>",
            "  }",
            "}",
            "",
            "Honesty rule (IMPORTANT):",
            "- Contract level must be derived ONLY from the provided EVIDENCE object below.",
            "- If anchors/mask details are missing in EVIDENCE, you MUST NOT claim FULL; use PARTIAL and list the missing evidence in evidence_gaps.",
            "",
            "evidence_gaps vocabulary (use these exact tokens, no paraphrases):",
            '- "anchors" (EVIDENCE has no anchors field)',
            '- "needs_mask" (EVIDENCE.constraints.needs_mask is missing)',
            '- "mask_details" (needs_mask=true but EVIDENCE.constraints.predicate_clauses missing/empty)',
            '- "constraints" (EVIDENCE has no constraints field)',
            '- "launch" (EVIDENCE has no launch field)',
            "If level=PARTIAL, evidence_gaps MUST include all missing categories that justify PARTIAL.",
            'Example (PARTIAL due to missing needs_mask): {"schema_version":"intentir_contract_v1","level":"PARTIAL","assumptions":["mask info missing"],"evidence_gaps":["needs_mask"],"claims":{}}',
            "",
            "Do not guess missing evidence from the source code.",
        ]
    )
    user = "\n".join(
        [
            f"Kernel: {desc.name}",
            f"Frontend: {desc.frontend}",
            "",
            "SOURCE:",
            str(desc.source_text[:4000]),
            "",
            "EVIDENCE (JSON, possibly ablated):",
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            "",
            ("Feedback:\n" + fb) if fb else "",
        ]
    ).strip()
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def _split_linalg_and_contract(raw: str) -> Tuple[str, Optional[Dict[str, Any]], List[str]]:
    txt = strip_code_fence(raw)
    m = re.search(r"\nCONTRACT_JSON\s*:\s*\n", txt)
    if not m:
        # Allow a less strict marker (single-line).
        m2 = re.search(r"CONTRACT_JSON\s*:\s*", txt)
        if not m2:
            return txt.strip(), None, ["missing CONTRACT_JSON marker"]
        mlir = txt[: m2.start()].strip()
        contract_text = txt[m2.end() :].strip()
    else:
        mlir = txt[: m.start()].strip()
        contract_text = txt[m.end() :].strip()
    if not contract_text:
        return mlir, None, ["empty contract JSON"]
    try:
        obj = parse_json_block(contract_text)
        return mlir, obj, []
    except Exception as e:
        return mlir, None, [f"contract JSON parse failed: {type(e).__name__}: {e}"]


@dataclass(frozen=True)
class SampleResult:
    frontend: str
    kernel: str
    rep: str
    ablation: str
    ok: bool
    contract_level: Optional[str]
    oracle_level: str
    oracle_gaps: List[str]
    overclaim: Optional[bool]
    underclaim: Optional[bool]
    ir_ok: bool
    contract_ok: bool
    consistency_ok: bool
    binding_ok: bool
    abstention_ok: bool
    assumption_precision_ok: bool
    reasons: List[str]
    llm: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "frontend": self.frontend,
            "kernel": self.kernel,
            "rep": self.rep,
            "ablation": self.ablation,
            "ok": self.ok,
            "contract_level": self.contract_level,
            "oracle_level": self.oracle_level,
            "oracle_gaps": list(self.oracle_gaps),
            "overclaim": self.overclaim,
            "underclaim": self.underclaim,
            "ir_ok": self.ir_ok,
            "contract_ok": self.contract_ok,
            "consistency_ok": self.consistency_ok,
            "binding_ok": self.binding_ok,
            "abstention_ok": self.abstention_ok,
            "assumption_precision_ok": self.assumption_precision_ok,
            "reasons": list(self.reasons),
            "llm": dict(self.llm),
        }


def _maybe_rate_limit(*, last_api_s: Optional[float], rpm: int) -> Optional[float]:
    if rpm <= 0:
        return last_api_s
    gap = 60.0 / float(rpm)
    now = time.time()
    if last_api_s is None:
        return now
    dt = now - last_api_s
    if dt < gap:
        time.sleep(gap - dt)
    return time.time()


def _run_intentir_once(
    *,
    desc: KernelDescriptor,
    evidence: Dict[str, Any],
    model: str,
    timeout_s: int,
    use_cache: bool,
    cache_dir: Optional[str],
    allow_fallback: bool,
    rpm: int,
    last_api_s: Optional[float],
    repair_rounds: int,
    save_raw: bool,
) -> Tuple[SampleResult, Optional[float]]:
    feedback: List[str] = []
    rounds: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    oracle_level, oracle_gaps = _oracle_from_evidence(evidence)

    contract_level: Optional[str] = None
    overclaim: Optional[bool] = None
    underclaim: Optional[bool] = None

    for r in range(0, max(0, int(repair_rounds)) + 1):
        msgs = _messages_for_intentir(desc=desc, evidence=evidence, feedback=feedback)
        try:
            resp = chat_completion(
                msgs,
                model=str(model),
                stream=False,
                timeout=int(timeout_s),
                max_retries=2,
                max_total_wait_s=45,
                use_cache=bool(use_cache),
                cache_dir=cache_dir,
                allow_fallback=bool(allow_fallback),
                temperature=0,
                max_tokens=2600,
            )
        except LLMClientError as e:
            rounds.append({"round": r, "ok": False, "error": str(e)})
            feedback = feedback + [f"Previous failure: {type(e).__name__}: {e}"]
            continue

        cache_hit = bool(resp.meta.get("cache_hit"))
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            last_api_s = _maybe_rate_limit(last_api_s=last_api_s, rpm=rpm)

        raw = resp.first_message()
        raw_preview = strip_code_fence(str(raw))[:800] if save_raw else None
        reasons: List[str] = []
        ir_ok = False
        contract_ok = False
        consistency_ok = False
        binding_ok = False
        abstention_ok = False
        assumption_precision_ok = False
        contract_level = None
        overclaim = None
        underclaim = None

        contract_src: Optional[Dict[str, Any]] = None
        contract_src_loc: Optional[str] = None
        intent: Optional[IntentFunction] = None

        try:
            obj = parse_json_block(raw)
        except Exception as e:
            reasons.append(f"intent JSON parse failed: {type(e).__name__}: {e}")
            obj = None

        if isinstance(obj, dict):
            # Contract may appear top-level (required) or legacy-style under meta.contract.
            if isinstance(obj.get("contract"), dict):
                contract_src = obj.get("contract")  # type: ignore[assignment]
                contract_src_loc = "contract"
            else:
                meta0 = obj.get("meta") if isinstance(obj.get("meta"), dict) else None
                if isinstance(meta0, dict) and isinstance(meta0.get("contract"), dict):
                    contract_src = meta0.get("contract")  # type: ignore[assignment]
                    contract_src_loc = "meta.contract"

            # Parse IntentIR IR (ignore contract for parsing so we can score IR vs contract separately).
            obj_ir = dict(obj)
            obj_ir.pop("contract", None)
            meta2 = obj_ir.get("meta") if isinstance(obj_ir.get("meta"), dict) else None
            if isinstance(meta2, dict):
                meta2 = dict(meta2)
                meta2.pop("contract", None)
                if meta2:
                    obj_ir["meta"] = meta2
                else:
                    obj_ir.pop("meta", None)
            try:
                # Use the production LLM JSON normalizer/parser for fairness:
                # this is deterministic schema correction (no extra evidence) that
                # reflects how IntentIR is consumed in the real pipeline.
                cand = parse_candidate_json(obj_ir)
                intent = cand.intent
                ir_ok = True
            except Exception as e_ir:
                reasons.append(f"intent IR parse failed: {type(e_ir).__name__}: {e_ir}")

            if contract_src is None:
                reasons.append("missing contract object")
            else:
                if contract_src_loc != "contract":
                    reasons.append("contract must be top-level key `contract` (not meta.contract)")
                contract_level = str(contract_src.get("level")) if isinstance(contract_src.get("level"), str) else None
                cerrs = _validate_contract_obj(contract_src)
                contract_ok = not cerrs
                reasons.extend(cerrs)

            if contract_ok and contract_level is not None and contract_src is not None:
                lv = str(contract_level)
                overclaim = LEVEL_ORDER[lv] > LEVEL_ORDER[oracle_level]
                underclaim = LEVEL_ORDER[lv] < LEVEL_ORDER[oracle_level]
                if bool(overclaim):
                    reasons.append(f"overclaim: contract.level={lv} > oracle={oracle_level}")
                if bool(underclaim):
                    reasons.append(f"underclaim: contract.level={lv} < oracle={oracle_level}")
                anchors = evidence.get("anchors") if isinstance(evidence.get("anchors"), dict) else {}
                if lv == "FULL":
                    if ir_ok and intent is not None:
                        anchor_errs = _anchor_errors_intentir(intent, lv, anchors)
                        bind_errs = _binding_errors_intentir(intent, lv, anchors, contract_src)
                        reasons.extend(anchor_errs)
                        reasons.extend(bind_errs)
                        consistency_ok = not anchor_errs
                        binding_ok = not bind_errs
                    else:
                        consistency_ok = False
                        binding_ok = False
                        reasons.append("FULL contract but IntentIR failed to validate")
                else:
                    consistency_ok = True
                    binding_ok = True
                abstention_ok = (oracle_level == "FULL") or (lv != "FULL")
                if not abstention_ok:
                    reasons.append(f"abstention_fail: oracle={oracle_level} but contract.level={lv}")
                assumption_precision_ok, ap_errs = _assumption_precision(contract_src, oracle_gaps)
                reasons.extend(ap_errs)

        needs_ir = bool(contract_level == "FULL")
        ok = bool(
            contract_ok
            and abstention_ok
            and (not bool(overclaim))
            and consistency_ok
            and (binding_ok if needs_ir else True)
            and assumption_precision_ok
            and ((ir_ok) if needs_ir else True)
        )
        rounds.append(
            {
                "round": r,
                "ok": ok,
                "ir_ok": ir_ok,
                "contract_ok": contract_ok,
                "consistency_ok": consistency_ok,
                "binding_ok": binding_ok,
                "abstention_ok": abstention_ok,
                "assumption_precision_ok": assumption_precision_ok,
                "contract_level": contract_level,
                "oracle_level": oracle_level,
                "oracle_gaps": list(oracle_gaps),
                "overclaim": overclaim,
                "underclaim": underclaim,
                "reasons": list(reasons[:12]),
                "cache_hit": cache_hit,
                **({"raw_preview": raw_preview} if save_raw and raw_preview else {}),
            }
        )
        if ok:
            return (
                SampleResult(
                    frontend=str(desc.frontend),
                    kernel=str(desc.name),
                    rep="intentir",
                    ablation=str(evidence.get("ablation") or "full"),
                    ok=True,
                    contract_level=contract_level,
                    oracle_level=oracle_level,
                    oracle_gaps=list(oracle_gaps),
                    overclaim=overclaim,
                    underclaim=underclaim,
                    ir_ok=ir_ok,
                    contract_ok=contract_ok,
                    consistency_ok=consistency_ok,
                    binding_ok=binding_ok,
                    abstention_ok=abstention_ok,
                    assumption_precision_ok=assumption_precision_ok,
                    reasons=[],
                    llm={
                        "stats": {
                            "llm_calls": int(cache_hits + cache_misses),
                            "cache_hits": int(cache_hits),
                            "cache_misses": int(cache_misses),
                            "api_calls": int(cache_misses),
                        },
                        "chosen": {
                            "model": resp.meta.get("response_model") or resp.meta.get("model") or model,
                            "base_url": resp.meta.get("base_url"),
                            "cache_hit": cache_hit,
                        },
                        "rounds": list(rounds),
                    },
                ),
                last_api_s,
            )

        feedback = feedback + ["Validator errors: " + ", ".join([str(x) for x in reasons[:6] if str(x).strip()])]

    last = rounds[-1] if rounds else {}
    return (
        SampleResult(
            frontend=str(desc.frontend),
            kernel=str(desc.name),
            rep="intentir",
            ablation=str(evidence.get("ablation") or "full"),
            ok=False,
            contract_level=(last.get("contract_level") if isinstance(last.get("contract_level"), str) else contract_level),
            oracle_level=str(oracle_level),
            oracle_gaps=list(oracle_gaps),
            overclaim=overclaim,
            underclaim=underclaim,
            ir_ok=bool(last.get("ir_ok")) if isinstance(last, dict) else False,
            contract_ok=bool(last.get("contract_ok")) if isinstance(last, dict) else False,
            consistency_ok=bool(last.get("consistency_ok")) if isinstance(last, dict) else False,
            binding_ok=bool(last.get("binding_ok")) if isinstance(last, dict) else False,
            abstention_ok=bool(last.get("abstention_ok")) if isinstance(last, dict) else False,
            assumption_precision_ok=bool(last.get("assumption_precision_ok")) if isinstance(last, dict) else False,
            reasons=list(last.get("reasons") or []) if isinstance(last, dict) else ["no attempts"],
            llm={
                "stats": {
                    "llm_calls": int(cache_hits + cache_misses),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "api_calls": int(cache_misses),
                },
                "rounds": rounds,
            },
        ),
        last_api_s,
    )


def _run_linalg_once(
    *,
    desc: KernelDescriptor,
    evidence: Dict[str, Any],
    model: str,
    timeout_s: int,
    use_cache: bool,
    cache_dir: Optional[str],
    allow_fallback: bool,
    rpm: int,
    last_api_s: Optional[float],
    repair_rounds: int,
    save_raw: bool,
) -> Tuple[SampleResult, Optional[float]]:
    feedback: List[str] = []
    rounds: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    oracle_level, oracle_gaps = _oracle_from_evidence(evidence)

    contract_level: Optional[str] = None
    overclaim: Optional[bool] = None
    underclaim: Optional[bool] = None

    for r in range(0, max(0, int(repair_rounds)) + 1):
        msgs = _messages_for_linalg(desc=desc, evidence=evidence, feedback=feedback)
        try:
            resp = chat_completion(
                msgs,
                model=str(model),
                stream=False,
                timeout=int(timeout_s),
                max_retries=2,
                max_total_wait_s=45,
                use_cache=bool(use_cache),
                cache_dir=cache_dir,
                allow_fallback=bool(allow_fallback),
                temperature=0,
                max_tokens=2600,
            )
        except LLMClientError as e:
            rounds.append({"round": r, "ok": False, "error": str(e)})
            feedback = feedback + [f"Previous failure: {type(e).__name__}: {e}"]
            continue

        cache_hit = bool(resp.meta.get("cache_hit"))
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
            last_api_s = _maybe_rate_limit(last_api_s=last_api_s, rpm=rpm)

        raw = resp.first_message()
        raw_preview = strip_code_fence(str(raw))[:800] if save_raw else None
        reasons: List[str] = []
        ir_ok = False
        contract_ok = False
        consistency_ok = False
        binding_ok = False
        abstention_ok = False
        assumption_precision_ok = False
        contract_level = None
        overclaim = None
        underclaim = None

        mlir, contract, split_errs = _split_linalg_and_contract(raw)
        if split_errs:
            reasons.extend(split_errs)

        if isinstance(contract, dict):
            contract_level = str(contract.get("level")) if isinstance(contract.get("level"), str) else None
            cerrs = _validate_contract_obj(contract)
            contract_ok = not cerrs
            reasons.extend(cerrs)
            if contract_ok and contract_level is not None:
                lv = str(contract_level)
                overclaim = LEVEL_ORDER[lv] > LEVEL_ORDER[oracle_level]
                underclaim = LEVEL_ORDER[lv] < LEVEL_ORDER[oracle_level]
                if bool(overclaim):
                    reasons.append(f"overclaim: contract.level={lv} > oracle={oracle_level}")
                if bool(underclaim):
                    reasons.append(f"underclaim: contract.level={lv} < oracle={oracle_level}")
                anchors = evidence.get("anchors") if isinstance(evidence.get("anchors"), dict) else {}
                anchor_errs = _anchor_errors_linalg(mlir, lv, anchors)
                bind_errs = _binding_errors_linalg(mlir, lv, anchors, contract)
                reasons.extend(anchor_errs)
                reasons.extend(bind_errs)
                consistency_ok = (not anchor_errs) if lv == "FULL" else True
                binding_ok = (not bind_errs) if lv == "FULL" else True
                abstention_ok = (oracle_level == "FULL") or (lv != "FULL")
                if not abstention_ok:
                    reasons.append(f"abstention_fail: oracle={oracle_level} but contract.level={lv}")
                assumption_precision_ok, ap_errs = _assumption_precision(contract, oracle_gaps)
                reasons.extend(ap_errs)
        else:
            reasons.append("missing contract object")

        if mlir.strip():
            if str(contract_level) == "FULL":
                mlir_errs = validate_mlir_linalg_text_contract_grade(mlir, io_spec=desc.io_spec)
            else:
                mlir_errs = validate_mlir_linalg_text_lenient(mlir)
            if mlir_errs:
                reasons.extend(list(mlir_errs[:6]))
            else:
                ir_ok = True
        else:
            reasons.append("empty MLIR text")

        needs_ir = bool(contract_level == "FULL")
        ok = bool(
            contract_ok
            and abstention_ok
            and (not bool(overclaim))
            and consistency_ok
            and (binding_ok if needs_ir else True)
            and assumption_precision_ok
            and ((ir_ok) if needs_ir else True)
        )
        rounds.append(
            {
                "round": r,
                "ok": ok,
                "ir_ok": ir_ok,
                "contract_ok": contract_ok,
                "consistency_ok": consistency_ok,
                "binding_ok": binding_ok,
                "abstention_ok": abstention_ok,
                "assumption_precision_ok": assumption_precision_ok,
                "contract_level": contract_level,
                "oracle_level": oracle_level,
                "oracle_gaps": list(oracle_gaps),
                "overclaim": overclaim,
                "underclaim": underclaim,
                "reasons": list(reasons[:12]),
                "cache_hit": cache_hit,
                **({"raw_preview": raw_preview} if save_raw and raw_preview else {}),
            }
        )
        if ok:
            return (
                SampleResult(
                    frontend=str(desc.frontend),
                    kernel=str(desc.name),
                    rep="linalg",
                    ablation=str(evidence.get("ablation") or "full"),
                    ok=True,
                    contract_level=contract_level,
                    oracle_level=oracle_level,
                    oracle_gaps=list(oracle_gaps),
                    overclaim=overclaim,
                    underclaim=underclaim,
                    ir_ok=ir_ok,
                    contract_ok=contract_ok,
                    consistency_ok=consistency_ok,
                    binding_ok=binding_ok,
                    abstention_ok=abstention_ok,
                    assumption_precision_ok=assumption_precision_ok,
                    reasons=[],
                    llm={
                        "stats": {
                            "llm_calls": int(cache_hits + cache_misses),
                            "cache_hits": int(cache_hits),
                            "cache_misses": int(cache_misses),
                            "api_calls": int(cache_misses),
                        },
                        "chosen": {
                            "model": resp.meta.get("response_model") or resp.meta.get("model") or model,
                            "base_url": resp.meta.get("base_url"),
                            "cache_hit": cache_hit,
                        },
                        "rounds": list(rounds),
                    },
                ),
                last_api_s,
            )

        feedback = feedback + ["Validator errors: " + ", ".join([str(x) for x in reasons[:6] if str(x).strip()])]

    last = rounds[-1] if rounds else {}
    return (
        SampleResult(
            frontend=str(desc.frontend),
            kernel=str(desc.name),
            rep="linalg",
            ablation=str(evidence.get("ablation") or "full"),
            ok=False,
            contract_level=(last.get("contract_level") if isinstance(last.get("contract_level"), str) else contract_level),
            oracle_level=str(oracle_level),
            oracle_gaps=list(oracle_gaps),
            overclaim=overclaim,
            underclaim=underclaim,
            ir_ok=bool(last.get("ir_ok")) if isinstance(last, dict) else False,
            contract_ok=bool(last.get("contract_ok")) if isinstance(last, dict) else False,
            consistency_ok=bool(last.get("consistency_ok")) if isinstance(last, dict) else False,
            binding_ok=bool(last.get("binding_ok")) if isinstance(last, dict) else False,
            abstention_ok=bool(last.get("abstention_ok")) if isinstance(last, dict) else False,
            assumption_precision_ok=bool(last.get("assumption_precision_ok")) if isinstance(last, dict) else False,
            reasons=list(last.get("reasons") or []) if isinstance(last, dict) else ["no attempts"],
            llm={
                "stats": {
                    "llm_calls": int(cache_hits + cache_misses),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "api_calls": int(cache_misses),
                },
                "rounds": rounds,
            },
        ),
        last_api_s,
    )


def _summarize(results: List[SampleResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_rep": {}}
    by_rep: Dict[str, List[SampleResult]] = {}
    for r in results:
        by_rep.setdefault(str(r.rep), []).append(r)
    for rep, items in by_rep.items():
        n = len(items)
        ok = sum(1 for x in items if x.ok)
        over = sum(1 for x in items if x.overclaim is True)
        under = sum(1 for x in items if x.underclaim is True)
        bind_ok = sum(1 for x in items if x.binding_ok)
        full = sum(1 for x in items if x.contract_level == "FULL")
        full_false = sum(
            1
            for x in items
            if x.contract_level == "FULL"
            and (x.overclaim is True or (not x.consistency_ok) or (not x.binding_ok) or (not x.ir_ok) or (not x.contract_ok))
        )
        partial = sum(1 for x in items if x.contract_level == "PARTIAL")
        abst_ok = sum(1 for x in items if x.abstention_ok)
        ap_ok = sum(1 for x in items if x.assumption_precision_ok)
        out["by_rep"][rep] = {
            "n": int(n),
            "ok": int(ok),
            "ok_rate": (float(ok) / float(n)) if n else None,
            "overclaim": int(over),
            "overclaim_rate": (float(over) / float(n)) if n else None,
            "underclaim": int(under),
            "underclaim_rate": (float(under) / float(n)) if n else None,
            "binding_ok": int(bind_ok),
            "binding_ok_rate": (float(bind_ok) / float(n)) if n else None,
            "full_claims": int(full),
            "full_false_accept": int(full_false),
            "full_false_accept_rate": (float(full_false) / float(full)) if full else None,
            "partial_claims": int(partial),
            "abstention_ok": int(abst_ok),
            "abstention_ok_rate": (float(abst_ok) / float(n)) if n else None,
            "assumption_precision_ok": int(ap_ok),
            "assumption_precision_ok_rate": (float(ap_ok) / float(n)) if n else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda", "all"], default="cuda")
    ap.add_argument("--suite", choices=["smoke", "coverage"], default="smoke")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable: restrict to kernel name(s)")
    ap.add_argument("--rep", choices=["intentir", "linalg", "both"], default="both")
    ap.add_argument(
        "--ablation",
        action="append",
        default=[],
        help="repeatable; allowed: full,no_launch,no_mask,no_anchors,no_launch_no_mask,no_launch_no_anchors,no_all",
    )

    ap.add_argument("--triton-dir", default=str(ROOT / "artifacts" / "full_pipeline_verify"))
    ap.add_argument("--tilelang-dir", default=str(ROOT / "artifacts" / "tilelang_full_pipeline"))
    ap.add_argument("--cuda-dir", default=str(ROOT / "artifacts" / "cuda_full_pipeline"))

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--repair-rounds", type=int, default=1)
    ap.add_argument("--cache", choices=["on", "off"], default="off")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--rpm", type=int, default=0, help="rate limit for cache-miss calls; 0 disables")
    ap.add_argument("--refresh-artifacts", action="store_true")
    ap.add_argument("--save-raw", action="store_true", help="include a small raw response preview per attempt (debug)")
    ap.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "experiments" / "E6" / "e6_2_contract_calibration_latest.json"),
    )
    ap.add_argument("--resume", action="store_true", help="resume from --out if it already exists")
    args = ap.parse_args()

    frontends = ["triton", "tilelang", "cuda"] if str(args.frontend) == "all" else [str(args.frontend)]
    wanted = {str(x) for x in (args.kernel or []) if str(x).strip()}
    reps = ["intentir", "linalg"] if str(args.rep) == "both" else [str(args.rep)]
    ablations = [str(x) for x in (args.ablation or []) if str(x).strip()] or ["full", "no_mask", "no_anchors", "no_all"]

    use_cache = str(args.cache) == "on"
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    allow_fallback = not bool(args.no_fallback)

    triton_dir = Path(str(args.triton_dir))
    tilelang_dir = Path(str(args.tilelang_dir))
    cuda_dir = Path(str(args.cuda_dir))
    out_path = Path(str(args.out))

    results: List[SampleResult] = []
    done: set[tuple[str, str, str, str, str]] = set()
    last_api_s: Optional[float] = None

    if bool(args.resume) and out_path.exists():
        try:
            prev = _load_json(out_path)
            for it in list(prev.get("results") or []):
                if not isinstance(it, dict):
                    continue
                fe = str(it.get("frontend") or "")
                k = str(it.get("kernel") or "")
                ab = str(it.get("ablation") or "")
                rep = str(it.get("rep") or "")
                if not (fe and k and ab and rep):
                    continue
                done.add((fe, k, ab, rep, str(prev.get("suite") or "")))
                og = it.get("oracle_gaps")
                oracle_gaps = [str(x) for x in og if str(x).strip()] if isinstance(og, list) else []
                results.append(
                    SampleResult(
                        frontend=fe,
                        kernel=k,
                        rep=rep,
                        ablation=ab,
                        ok=bool(it.get("ok")),
                        contract_level=(None if it.get("contract_level") is None else str(it.get("contract_level"))),
                        oracle_level=str(it.get("oracle_level") or "PARTIAL"),
                        oracle_gaps=oracle_gaps,
                        overclaim=(None if it.get("overclaim") is None else bool(it.get("overclaim"))),
                        underclaim=(None if it.get("underclaim") is None else bool(it.get("underclaim"))),
                        ir_ok=bool(it.get("ir_ok")),
                        contract_ok=bool(it.get("contract_ok")),
                        consistency_ok=bool(it.get("consistency_ok")),
                        binding_ok=bool(it.get("binding_ok")) if it.get("binding_ok") is not None else False,
                        abstention_ok=bool(it.get("abstention_ok")),
                        assumption_precision_ok=bool(it.get("assumption_precision_ok")),
                        reasons=list(it.get("reasons") or []),
                        llm=(it.get("llm") if isinstance(it.get("llm"), dict) else {}),
                    )
                )
            if results:
                print(f"[E6.2] resume: loaded {len(results)} prior results from {out_path}", flush=True)
        except Exception as e:
            print(f"[E6.2] resume ignored (failed to load {out_path}): {type(e).__name__}: {e}", flush=True)
            results = []
            done = set()

    for fe in frontends:
        ks = _kernels_from_pipeline(str(fe), str(args.suite))
        if wanted:
            ks = [k for k in ks if k in wanted]
        art_dir = triton_dir if fe == "triton" else (tilelang_dir if fe == "tilelang" else cuda_dir)
        for k in ks:
            desc = _load_or_prepare_descriptor(frontend=str(fe), kernel=str(k), artifact_dir=art_dir, refresh=bool(args.refresh_artifacts))
            for ab in ablations:
                ev = _evidence_view(desc, ablation=str(ab))
                # Keep a stable copy so repair feedback doesn't mutate evidence.
                ev0 = copy.deepcopy(ev)
                for rep in reps:
                    if (str(fe), str(k), str(ab), str(rep), str(args.suite)) in done:
                        continue
                    if rep == "intentir":
                        rr, last_api_s = _run_intentir_once(
                            desc=desc,
                            evidence=ev0,
                            model=str(args.model),
                            timeout_s=int(args.timeout),
                            use_cache=bool(use_cache),
                            cache_dir=cache_dir,
                            allow_fallback=bool(allow_fallback),
                            rpm=int(args.rpm),
                            last_api_s=last_api_s,
                            repair_rounds=int(args.repair_rounds),
                            save_raw=bool(args.save_raw),
                        )
                    elif rep == "linalg":
                        rr, last_api_s = _run_linalg_once(
                            desc=desc,
                            evidence=ev0,
                            model=str(args.model),
                            timeout_s=int(args.timeout),
                            use_cache=bool(use_cache),
                            cache_dir=cache_dir,
                            allow_fallback=bool(allow_fallback),
                            rpm=int(args.rpm),
                            last_api_s=last_api_s,
                            repair_rounds=int(args.repair_rounds),
                            save_raw=bool(args.save_raw),
                        )
                    else:
                        raise SystemExit(f"unknown rep: {rep}")
                    results.append(rr)
                    status = "OK" if rr.ok else "FAIL"
                    print(f"[{fe}:{k}:{ab}:{rep}] {status}", flush=True)

                    payload = {
                        "experiment": "E6_2_contract_calibration",
                        "suite": str(args.suite),
                        "frontends": list(frontends),
                        "reps": list(reps),
                        "ablations": list(ablations),
                        "model": str(args.model),
                        "cache": ("on" if use_cache else "off"),
                        "repair_rounds": int(args.repair_rounds),
                        "results": [r.to_json_dict() for r in results],
                        "summary": _summarize(results),
                    }
                    _write_json(out_path, payload)

    print(str(out_path))


if __name__ == "__main__":
    main()
