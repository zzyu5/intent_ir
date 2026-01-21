"""
E6: IR usability / LLM-generation robustness (lightweight).

This experiment compares LLM output validity across three IR formats:
  1) IntentIR JSON (our structured IR)
  2) MLIR Linalg (text IR; lightweight syntax validator only)
  3) Tile-centric schedule JSON (strict-ish schema)

We measure:
  - one-shot valid rate (first attempt is valid)
  - final valid rate under a repair budget (feedback rounds)
  - repair rounds distribution

This intentionally does NOT run dynamic diff or any backend; the goal is to
quantify "LLM output carrier usability", not performance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.llm import DEFAULT_MODEL, LLMClientError, chat_completion, parse_json_block, strip_code_fence  # noqa: E402
from intent_ir.llm.llm_hub import LLMIntentHub, _maybe_truncate_source  # noqa: E402
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor  # noqa: E402
from verify.ir_formats import validate_mlir_linalg_text, validate_tile_dsl_json  # noqa: E402


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


def _maybe_rate_limit(*, last_api_s: Optional[float], rpm: int) -> Optional[float]:
    if rpm <= 0:
        return last_api_s
    gap = 60.0 / float(rpm)
    now = time.time()
    if last_api_s is None:
        return now
    wait = gap - (now - last_api_s)
    if wait > 0:
        time.sleep(wait)
        return time.time()
    return now


def _clear_cache_dir(cache_dir: Optional[str]) -> None:
    cd = Path(cache_dir) if cache_dir is not None else (Path.home() / ".cache" / "intentir" / "llm")
    home = Path.home().resolve()
    cd_res = cd.resolve()
    if home not in cd_res.parents and cd_res != home:
        raise SystemExit(f"--clear-cache refused: cache dir is outside HOME: {cd_res}")
    shutil.rmtree(cd_res, ignore_errors=True)
    cd_res.mkdir(parents=True, exist_ok=True)


def _prepare_descriptor(*, frontend: str, kernel: str, artifact_dir: Path) -> KernelDescriptor:
    """
    Build a fresh descriptor + frontend evidence for this kernel.

    Used when running E6 on a new suite without existing `*.descriptor.json`.
    """
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


def _messages_for_linalg(*, desc: KernelDescriptor, feedback: List[str]) -> List[Dict[str, str]]:
    src = _maybe_truncate_source(desc.source_text)
    fb = "\n".join([f"- {x}" for x in feedback]) if feedback else ""
    sys_prompt = "\n".join(
        [
            "You are a compiler engineer.",
            "Task: Output syntactically plausible MLIR (text form) using the Linalg dialect.",
            "Output rules:",
            "- Output plain text only (no markdown fences, no prose).",
            "- Must contain: 'module { ... }' and a function ('func.func' preferred).",
            "- Must contain at least one 'linalg.' op (e.g., linalg.generic / linalg.matmul).",
            "- The function signature must cover ALL tensor arguments (by count and rank) from io_spec.tensors.",
            "- You may use 'tensor<?x...xf32>' or 'memref<?x...xf32>' as long as ranks match io_spec.",
            "- If unsure about exact dynamic sizes, use '?' dimensions but keep ranks consistent.",
        ]
    )
    user = "\n".join(
        [
            f"Kernel: {desc.name}",
            f"Frontend: {desc.frontend}",
            "",
            "SOURCE:",
            str(src),
            "",
            "EVIDENCE (JSON, may be partial):",
            json.dumps(
                {
                    "io_spec": desc.io_spec,
                    "launch": desc.launch,
                    "frontend_facts": desc.frontend_facts,
                    "frontend_constraints": desc.frontend_constraints,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "",
            ("Feedback (fix the MLIR to satisfy the validator):\n" + fb if fb else "").strip(),
        ]
    ).strip()
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def _messages_for_tile(*, desc: KernelDescriptor, feedback: List[str]) -> List[Dict[str, str]]:
    src = _maybe_truncate_source(desc.source_text)
    fb = "\n".join([f"- {x}" for x in feedback]) if feedback else ""
    sys_prompt = "\n".join(
        [
            "You are a compiler schedule/tuning assistant.",
            "Task: Output STRICT JSON for a tile-centric schedule description.",
            "Output rules:",
            "- Output STRICT JSON only (no markdown fences, no prose).",
            "- Required top-level keys: schema_version (string), kernel (string), schedule (object).",
            "- schedule.tile must be a non-empty object mapping axis names to positive integers.",
            "- schedule.vec_width and schedule.num_threads are optional positive integers.",
            "- schedule.tile axis names must be chosen from the symbolic dims appearing in io_spec.tensors[*].shape.",
        ]
    )
    user = "\n".join(
        [
            f"Kernel: {desc.name}",
            f"Frontend: {desc.frontend}",
            "",
            "SOURCE:",
            str(src),
            "",
            "EVIDENCE (JSON, may be partial):",
            json.dumps(
                {
                    "io_spec": desc.io_spec,
                    "launch": desc.launch,
                    "frontend_facts": desc.frontend_facts,
                    "frontend_constraints": desc.frontend_constraints,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "",
            ("Feedback (fix the JSON to satisfy the validator):\n" + fb if fb else "").strip(),
        ]
    ).strip()
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class RepResult:
    frontend: str
    kernel: str
    rep: str
    ok: bool
    rounds_used: int
    one_shot_ok: bool
    category: str
    reasons: List[str]
    llm: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "frontend": self.frontend,
            "kernel": self.kernel,
            "rep": self.rep,
            "ok": self.ok,
            "rounds_used": self.rounds_used,
            "one_shot_ok": self.one_shot_ok,
            "category": self.category,
            "reasons": list(self.reasons),
            "llm": dict(self.llm),
        }


def _run_rep_intentir(
    *,
    hub: LLMIntentHub,
    desc: KernelDescriptor,
    model: Optional[str],
    repair_rounds: int,
    rpm: int,
    last_api_s: Optional[float],
) -> Tuple[RepResult, Optional[float]]:
    feedback: List[str] = []
    rounds: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    for r in range(0, max(0, int(repair_rounds)) + 1):
        try:
            cand = hub.lift(desc, feedback=feedback, model=model)
            trace = cand.llm_trace.get("extract_trace") if isinstance(cand.llm_trace, dict) else None
            chosen = (trace or {}).get("chosen") if isinstance(trace, dict) else None
            cache_hit = bool((chosen or {}).get("cache_hit")) if isinstance(chosen, dict) else False
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
                last_api_s = _maybe_rate_limit(last_api_s=last_api_s, rpm=rpm)
            rounds.append({"round": r, "ok": True, "cache_hit": cache_hit, "chosen": chosen})
            return (
                RepResult(
                    frontend=str(desc.frontend),
                    kernel=str(desc.name),
                    rep="intentir",
                    ok=True,
                    rounds_used=r + 1,
                    one_shot_ok=(r == 0),
                    category="ok",
                    reasons=[],
                    llm={
                        "stats": {
                            "llm_calls": int(cache_hits + cache_misses),
                            "cache_hits": int(cache_hits),
                            "cache_misses": int(cache_misses),
                            "api_calls": int(cache_misses),
                        },
                        "rounds": list(rounds),
                    },
                ),
                last_api_s,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            # IMPORTANT: provide a compact, actionable feedback string.
            feedback = feedback + [f"Previous failure: {msg}"]
            rounds.append({"round": r, "ok": False, "error": msg})
            continue

    return (
        RepResult(
            frontend=str(desc.frontend),
            kernel=str(desc.name),
            rep="intentir",
            ok=False,
            rounds_used=int(len(rounds)),
            one_shot_ok=False,
            category="invalid",
            reasons=[str(rounds[-1].get("error"))] if rounds else ["no attempts"],
            llm={"stats": {"llm_calls": int(cache_hits + cache_misses), "cache_hits": cache_hits, "cache_misses": cache_misses, "api_calls": cache_misses}, "rounds": rounds},
        ),
        last_api_s,
    )


def _run_rep_linalg(
    *,
    desc: KernelDescriptor,
    model: str,
    timeout_s: int,
    parse_retries: int,
    repair_rounds: int,
    use_cache: bool,
    cache_dir: Optional[str],
    no_fallback: bool,
    rpm: int,
    last_api_s: Optional[float],
) -> Tuple[RepResult, Optional[float]]:
    feedback: List[str] = []
    rounds: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    last_errs: List[str] = []

    for r in range(0, max(0, int(repair_rounds)) + 1):
        messages = _messages_for_linalg(desc=desc, feedback=feedback)
        try:
            resp = chat_completion(
                messages,
                model=str(model),
                stream=False,
                timeout=int(timeout_s),
                max_retries=max(1, int(parse_retries)),
                max_total_wait_s=45,
                use_cache=bool(use_cache),
                cache_dir=cache_dir,
                allow_fallback=(not bool(no_fallback)),
                temperature=0,
                max_tokens=2048,
            )
            raw = resp.first_message()
            cache_hit = bool(resp.meta.get("cache_hit"))
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
                last_api_s = _maybe_rate_limit(last_api_s=last_api_s, rpm=rpm)
            txt = strip_code_fence(raw).strip()
            errs = validate_mlir_linalg_text(txt, io_spec=desc.io_spec)
            ok = not errs
            rounds.append(
                {
                    "round": r,
                    "ok": ok,
                    "cache_hit": cache_hit,
                    "chosen": {"model": resp.meta.get("response_model") or resp.meta.get("model") or model, "base_url": resp.meta.get("base_url")},
                    "errors": list(errs),
                }
            )
            if ok:
                return (
                    RepResult(
                        frontend=str(desc.frontend),
                        kernel=str(desc.name),
                        rep="linalg",
                        ok=True,
                        rounds_used=r + 1,
                        one_shot_ok=(r == 0),
                        category="ok",
                        reasons=[],
                        llm={
                            "stats": {"llm_calls": int(cache_hits + cache_misses), "cache_hits": cache_hits, "cache_misses": cache_misses, "api_calls": cache_misses},
                            "rounds": list(rounds),
                        },
                    ),
                    last_api_s,
                )
            last_errs = list(errs)
            feedback = feedback + [f"Validator errors: {', '.join(errs[:6])}"]
        except LLMClientError as e:
            msg = f"{type(e).__name__}: {e}"
            rounds.append({"round": r, "ok": False, "error": msg})
            last_errs = [msg]
            feedback = feedback + [f"Previous failure: {msg}"]
            continue

    return (
        RepResult(
            frontend=str(desc.frontend),
            kernel=str(desc.name),
            rep="linalg",
            ok=False,
            rounds_used=int(len(rounds)),
            one_shot_ok=False,
            category="invalid",
            reasons=list(last_errs) if last_errs else ["no attempts"],
            llm={"stats": {"llm_calls": int(cache_hits + cache_misses), "cache_hits": cache_hits, "cache_misses": cache_misses, "api_calls": cache_misses}, "rounds": rounds},
        ),
        last_api_s,
    )


def _run_rep_tile(
    *,
    desc: KernelDescriptor,
    model: str,
    timeout_s: int,
    parse_retries: int,
    repair_rounds: int,
    use_cache: bool,
    cache_dir: Optional[str],
    no_fallback: bool,
    rpm: int,
    last_api_s: Optional[float],
) -> Tuple[RepResult, Optional[float]]:
    feedback: List[str] = []
    rounds: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    last_errs: List[str] = []

    for r in range(0, max(0, int(repair_rounds)) + 1):
        messages = _messages_for_tile(desc=desc, feedback=feedback)
        try:
            resp = chat_completion(
                messages,
                model=str(model),
                stream=False,
                timeout=int(timeout_s),
                max_retries=max(1, int(parse_retries)),
                max_total_wait_s=45,
                use_cache=bool(use_cache),
                cache_dir=cache_dir,
                allow_fallback=(not bool(no_fallback)),
                temperature=0,
                max_tokens=1200,
            )
            raw = resp.first_message()
            obj = parse_json_block(raw)
            chosen = {
                "model": resp.meta.get("response_model") or resp.meta.get("model") or model,
                "base_url": resp.meta.get("base_url"),
                "cache_hit": bool(resp.meta.get("cache_hit")),
            }
            cache_hit = bool(chosen.get("cache_hit"))
            if cache_hit:
                cache_hits += 1
            else:
                cache_misses += 1
                last_api_s = _maybe_rate_limit(last_api_s=last_api_s, rpm=rpm)
            errs = validate_tile_dsl_json(obj, io_spec=desc.io_spec)
            ok = not errs
            rounds.append({"round": r, "ok": ok, "cache_hit": cache_hit, "chosen": chosen, "errors": list(errs)})
            if ok:
                return (
                    RepResult(
                        frontend=str(desc.frontend),
                        kernel=str(desc.name),
                        rep="tile",
                        ok=True,
                        rounds_used=r + 1,
                        one_shot_ok=(r == 0),
                        category="ok",
                        reasons=[],
                        llm={
                            "stats": {"llm_calls": int(cache_hits + cache_misses), "cache_hits": cache_hits, "cache_misses": cache_misses, "api_calls": cache_misses},
                            "rounds": list(rounds),
                        },
                    ),
                    last_api_s,
                )
            last_errs = list(errs)
            feedback = feedback + [f"Validator errors: {', '.join(errs[:6])}"]
        except LLMClientError as e:
            msg = f"{type(e).__name__}: {e}"
            rounds.append({"round": r, "ok": False, "error": msg})
            last_errs = [msg]
            feedback = feedback + [f"Previous failure: {msg}"]
            continue

    return (
        RepResult(
            frontend=str(desc.frontend),
            kernel=str(desc.name),
            rep="tile",
            ok=False,
            rounds_used=int(len(rounds)),
            one_shot_ok=False,
            category="invalid",
            reasons=list(last_errs) if last_errs else ["no attempts"],
            llm={"stats": {"llm_calls": int(cache_hits + cache_misses), "cache_hits": cache_hits, "cache_misses": cache_misses, "api_calls": cache_misses}, "rounds": rounds},
        ),
        last_api_s,
    )


def _summarize(results: List[RepResult]) -> Dict[str, Any]:
    by_rep: Dict[str, List[RepResult]] = {}
    for r in results:
        by_rep.setdefault(str(r.rep), []).append(r)

    out: Dict[str, Any] = {"by_rep": {}}
    for rep, items in by_rep.items():
        n = len(items)
        ok = sum(1 for x in items if x.ok)
        one_shot_ok = sum(1 for x in items if x.one_shot_ok)
        rounds_ok = [int(x.rounds_used) for x in items if x.ok and int(x.rounds_used) > 0]
        out["by_rep"][rep] = {
            "n": int(n),
            "ok": int(ok),
            "ok_rate": (float(ok) / float(n)) if n else None,
            "one_shot_ok": int(one_shot_ok),
            "one_shot_ok_rate": (float(one_shot_ok) / float(n)) if n else None,
            "rounds_used_avg_if_ok": (sum(rounds_ok) / float(len(rounds_ok))) if rounds_ok else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["triton", "tilelang", "cuda", "all"], default="all")
    ap.add_argument("--suite", choices=["smoke", "coverage"], default="coverage")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable: restrict to kernel name(s)")
    ap.add_argument("--rep", choices=["intentir", "linalg", "tile", "all"], default="all")

    ap.add_argument("--triton-dir", default=str(ROOT / "artifacts" / "full_pipeline_verify"))
    ap.add_argument("--tilelang-dir", default=str(ROOT / "artifacts" / "tilelang_full_pipeline"))
    ap.add_argument("--cuda-dir", default=str(ROOT / "artifacts" / "cuda_full_pipeline"))

    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--parse-retries", type=int, default=2)
    ap.add_argument("--repair-rounds", type=int, default=1)
    ap.add_argument("--cache", choices=["on", "off"], default="off")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--refresh-artifacts", action="store_true")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--rpm", type=int, default=5, help="rate limit for cache-miss calls; 0 disables")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "E6" / "e6_ir_usability_latest.json"))
    ap.add_argument("--resume", action="store_true", help="resume from --out if it already exists")
    args = ap.parse_args()

    frontends = ["triton", "tilelang", "cuda"] if str(args.frontend) == "all" else [str(args.frontend)]
    wanted = {str(x) for x in (args.kernel or []) if str(x).strip()}
    reps = ["intentir", "linalg", "tile"] if str(args.rep) == "all" else [str(args.rep)]

    use_cache = str(args.cache) == "on"
    cache_dir = str(args.cache_dir) if args.cache_dir else os.getenv("INTENTIR_LLM_CACHE_DIR")
    if bool(args.clear_cache):
        _clear_cache_dir(cache_dir)

    hub = LLMIntentHub(timeout_s=int(args.timeout), max_attempts=2, max_parse_retries=2, extra_chat_kwargs={"use_cache": bool(use_cache), **({"cache_dir": cache_dir} if cache_dir else {})})

    triton_dir = Path(str(args.triton_dir))
    tilelang_dir = Path(str(args.tilelang_dir))
    cuda_dir = Path(str(args.cuda_dir))
    out_path = Path(str(args.out))

    results: List[RepResult] = []
    done: set[tuple[str, str, str]] = set()
    if bool(args.resume) and out_path.exists():
        try:
            prev = _load_json(out_path)
            for it in list(prev.get("results") or []):
                if not isinstance(it, dict):
                    continue
                fe = str(it.get("frontend") or "")
                k = str(it.get("kernel") or "")
                rep = str(it.get("rep") or "")
                if not fe or not k or not rep:
                    continue
                done.add((fe, k, rep))
                results.append(
                    RepResult(
                        frontend=fe,
                        kernel=k,
                        rep=rep,
                        ok=bool(it.get("ok")),
                        rounds_used=int(it.get("rounds_used") or 0),
                        one_shot_ok=bool(it.get("one_shot_ok")),
                        category=str(it.get("category") or ""),
                        reasons=list(it.get("reasons") or []),
                        llm=(it.get("llm") if isinstance(it.get("llm"), dict) else {}),
                    )
                )
            if results:
                print(f"[E6] resume: loaded {len(results)} prior results from {out_path}", flush=True)
        except Exception as e:
            print(f"[E6] resume ignored (failed to load {out_path}): {type(e).__name__}: {e}", flush=True)
            results = []
            done = set()
    last_api_s: Optional[float] = None

    for fe in frontends:
        ks = _kernels_from_pipeline(str(fe), str(args.suite))
        if wanted:
            ks = [k for k in ks if k in wanted]
        art_dir = triton_dir if fe == "triton" else (tilelang_dir if fe == "tilelang" else cuda_dir)
        for k in ks:
            desc = _load_or_prepare_descriptor(frontend=str(fe), kernel=str(k), artifact_dir=art_dir, refresh=bool(args.refresh_artifacts))
            for rep in reps:
                if (str(fe), str(k), str(rep)) in done:
                    continue
                if rep == "intentir":
                    rr, last_api_s = _run_rep_intentir(
                        hub=hub,
                        desc=desc,
                        model=str(args.model) if args.model else None,
                        repair_rounds=int(args.repair_rounds),
                        rpm=int(args.rpm),
                        last_api_s=last_api_s,
                    )
                elif rep == "linalg":
                    rr, last_api_s = _run_rep_linalg(
                        desc=desc,
                        model=str(args.model),
                        timeout_s=int(args.timeout),
                        parse_retries=int(args.parse_retries),
                        repair_rounds=int(args.repair_rounds),
                        use_cache=bool(use_cache),
                        cache_dir=cache_dir,
                        no_fallback=bool(args.no_fallback),
                        rpm=int(args.rpm),
                        last_api_s=last_api_s,
                    )
                elif rep == "tile":
                    rr, last_api_s = _run_rep_tile(
                        desc=desc,
                        model=str(args.model),
                        timeout_s=int(args.timeout),
                        parse_retries=int(args.parse_retries),
                        repair_rounds=int(args.repair_rounds),
                        use_cache=bool(use_cache),
                        cache_dir=cache_dir,
                        no_fallback=bool(args.no_fallback),
                        rpm=int(args.rpm),
                        last_api_s=last_api_s,
                    )
                else:
                    raise SystemExit(f"unknown rep: {rep}")
                results.append(rr)
                done.add((str(rr.frontend), str(rr.kernel), str(rr.rep)))
                status = "OK" if rr.ok else "FAIL"
                print(f"[{fe}:{k}:{rep}] {status} rounds={rr.rounds_used}", flush=True)

                # Checkpoint after each result to avoid losing long runs.
                payload = {
                    "experiment": "E6_ir_usability",
                    "suite": str(args.suite),
                    "frontends": list(frontends),
                    "reps": list(reps),
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
