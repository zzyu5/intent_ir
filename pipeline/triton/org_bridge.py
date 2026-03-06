from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from intent_ir.ir import IntentFunction

from pipeline.triton.core import (
    _candidate_line,
    _compiler_stack_name,
    _detect_cuda_arch,
    _normalize_cuda_arch_key,
    _org_budget,
    _org_candidates_jsonl_path,
    _org_candidates_txt_path,
    _org_doc_path,
    _org_mode,
    _org_model,
    _org_plan_path,
    _org_seed_path,
    _org_seed_policy,
)


ROOT = Path(__file__).resolve().parents[2]
ORG_RUNTIME_ROOT = ROOT / "ORG-Migrate"


def _ensure_org_runtime_on_path() -> None:
    sp = str(ORG_RUNTIME_ROOT)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def load_org_module(module_name: str):
    _ensure_org_runtime_on_path()
    return importlib.import_module(str(module_name))


def load_org_attr(module_name: str, attr_name: str):
    mod = load_org_module(module_name)
    return getattr(mod, str(attr_name))


def _load_text_artifact(text_value: object, path_value: object, meta_path: object) -> tuple[str, str]:
    text = str(text_value or "")
    if text.strip():
        return text, ""
    for raw in (path_value, meta_path):
        if raw is None:
            continue
        p = Path(str(raw))
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8"), str(p)
            except Exception:
                continue
    return "", ""


def _build_intent_summary(intent: IntentFunction) -> dict[str, object]:
    return {
        "name": str(intent.name or ""),
        "op_names": [str(op.op) for op in list(intent.ops or []) if str(getattr(op, "op", "")).strip()],
        "outputs": [str(x) for x in list(intent.outputs or []) if str(x).strip()],
        "parallel_axes": [str(x) for x in list(intent.parallel_axes or []) if str(x).strip()],
        "axis_roles": dict(getattr(intent, "axis_roles", {}) or {}),
        "schedule": (intent.schedule.__dict__ if getattr(intent, "schedule", None) is not None else None),
    }


def _resolve_source_oracle_facts(*, spec_name: str, shape_bindings: Mapping[str, int]) -> dict[str, Any]:
    source_arch = _normalize_cuda_arch_key(os.getenv("INTENTIR_ORG_SOURCE_ARCH", ""))
    source_stack_env = str(os.getenv("INTENTIR_ORG_SOURCE_COMPILER_STACK", "") or "").strip().lower()
    source_stack = source_stack_env or _compiler_stack_name()
    source_db_env = str(os.getenv("INTENTIR_ORG_SOURCE_TUNING_DB", "") or "").strip()
    extract_source_oracle_facts = load_org_attr("org.facts.source_oracle", "extract_source_oracle_facts")
    return extract_source_oracle_facts(
        kernel=str(spec_name),
        source_arch=str(source_arch),
        shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        compiler_stack=str(source_stack),
        db_path=(str(source_db_env) if source_db_env else None),
    )


def run_org_sidecar(
    *,
    spec_name: str,
    out_dir: Path,
    desc,
    intent: IntentFunction,
    report: dict[str, object],
    shape_bindings: dict[str, int],
    triton_provider: str,
    backend_target: str | None,
) -> None:
    mode = _org_mode()
    if mode == "off":
        return

    org_report: dict[str, object] = {
        "enabled": True,
        "mode": str(mode),
        "seed_policy": str(_org_seed_policy()),
        "model": (str(_org_model()) if _org_model() else None),
        "budget": int(_org_budget()),
        "runtime_root": str(ORG_RUNTIME_ROOT),
    }
    report["org"] = org_report

    diff_ok = bool((report.get("diff") or {}).get("ok"))
    static_ok = False
    if isinstance(report.get("static_validation"), dict):
        static_ok = bool((report.get("static_validation") or {}).get("ok"))
    if (not diff_ok) or (not static_ok):
        reason = f"skip_org: diff_ok={diff_ok} static_ok={static_ok}"
        org_report["skipped"] = True
        org_report["reason"] = reason
        if mode == "strict":
            raise RuntimeError(reason)
        return

    seed_policy = _org_seed_policy()
    if seed_policy not in {"auto", "force_llm", "force_cache"}:
        raise ValueError(f"unsupported INTENTIR_ORG_SEED_POLICY={seed_policy!r}")

    seed_path = _org_seed_path(out_dir, spec_name)
    org_path = _org_doc_path(out_dir, spec_name)
    plan_path = _org_plan_path(out_dir, spec_name)
    cand_jsonl_path = _org_candidates_jsonl_path(out_dir, spec_name)
    cand_txt_path = _org_candidates_txt_path(out_dir, spec_name)
    org_report["seed_path"] = str(seed_path)
    org_report["org_path"] = str(org_path)

    intent_summary = _build_intent_summary(intent)
    target_arch = _detect_cuda_arch() or _normalize_cuda_arch_key(str(backend_target or "")) or ""
    source_oracle_facts = _resolve_source_oracle_facts(spec_name=spec_name, shape_bindings=shape_bindings)
    source_oracle = dict(source_oracle_facts.get("oracle") or {})

    extra_evidence = {
        "shape_bindings": {str(k): int(v) for k, v in dict(shape_bindings or {}).items() if str(k).strip()},
        "backend_target": (str(backend_target) if backend_target is not None else None),
        "triton_provider": str(triton_provider),
        "contract_level": str((report.get("contract") or {}).get("level") or ""),
        "source_arch": str(source_oracle.get("arch") or ""),
        "target_arch": str(target_arch),
        "source_compiler_stack": str(source_oracle.get("compiler_stack") or ""),
        "source_oracle_facts": dict(source_oracle_facts),
    }
    quality = {
        "diff_ok": bool(diff_ok),
        "static_ok": bool(static_ok),
        "contract_level": str((report.get("contract") or {}).get("level") or ""),
    }
    llm_fallback_used = bool((report.get("llm_fallback") or {}).get("used"))

    ttgir_facts: dict[str, Any] | None = None
    ptx_facts: dict[str, Any] | None = None
    ttir_summary: dict[str, Any] | None = None
    if desc is not None:
        build_ttir_summary = load_org_attr("org.facts.ttir", "build_ttir_summary")
        extract_ttgir_mechanism_facts = load_org_attr("org.facts.ttgir", "extract_ttgir_mechanism_facts")
        extract_ptx_mechanism_facts = load_org_attr("org.facts.ptx", "extract_ptx_mechanism_facts")

        ttir_summary = build_ttir_summary(desc)
        ttgir_text, ttgir_path = _load_text_artifact(
            getattr(getattr(desc, "artifacts", None), "ttgir_text", None),
            getattr(getattr(desc, "artifacts", None), "ttgir_path", None),
            (getattr(desc, "meta", {}) or {}).get("ttgir_original_path"),
        )
        ptx_text, ptx_path = _load_text_artifact(
            getattr(getattr(desc, "artifacts", None), "ptx_text", None),
            (getattr(getattr(desc, "artifacts", None), "extra", {}) or {}).get("ptx_path"),
            (getattr(desc, "meta", {}) or {}).get("ptx_original_path"),
        )
        if ttgir_text.strip():
            ttgir_facts = extract_ttgir_mechanism_facts(ttgir_text, kernel_name=str(spec_name), artifact_path=(ttgir_path or None))
            extra_evidence["ttgir_facts"] = dict(ttgir_facts)
        ptx_facts = extract_ptx_mechanism_facts(ptx_text, kernel_name=str(spec_name), artifact_path=(ptx_path or None))
        extra_evidence["ptx_facts"] = dict(ptx_facts)
        extra_evidence["ttir_summary"] = dict(ttir_summary)
        org_report["evidence_source"] = {
            "primary": ("ttgir" if ttgir_facts is not None else "ttir"),
            "ttgir_available": bool(ttgir_facts is not None),
            "ttgir_path": (ttgir_path or None),
            "ptx_available": bool((ptx_facts or {}).get("artifacts", {}).get("ptx_available")),
            "ttir_available": bool((ttir_summary or {}).get("available")),
        }

    if mode in {"apply", "strict"} and str(spec_name) in {"flash_attention2d", "matmul_fused_epilogue2d"} and ttgir_facts is None:
        org_report["ok"] = False
        org_report["error"] = "ttgir_missing"
        if mode == "strict":
            raise RuntimeError("ttgir_missing")
        return

    org_doc = None
    org_raw_json: dict[str, Any] | None = None
    org_trace: dict[str, Any] = {}
    cache_used = False

    try:
        load_org_seed = load_org_attr("org.io", "load_org_seed")
        save_org_seed = load_org_attr("org.io", "save_org_seed")
        is_seed_trusted_for_auto = load_org_attr("org.io", "is_seed_trusted_for_auto")
        LLMOrgHub = load_org_attr("org.llm_hub", "LLMOrgHub")

        should_try_cache = bool(seed_policy in {"auto", "force_cache"})
        should_try_llm = bool(seed_policy in {"auto", "force_llm"})
        if should_try_cache and seed_path.is_file():
            cache_allowed = True
            cache_reason = "trusted"
            if seed_policy == "auto":
                try:
                    seed = load_org_seed(seed_path)
                    cache_allowed, cache_reason = is_seed_trusted_for_auto(seed)
                except Exception as exc:  # noqa: BLE001
                    cache_allowed = False
                    cache_reason = f"invalid_seed:{type(exc).__name__}"
            if cache_allowed:
                seed = load_org_seed(seed_path)
                org_doc = seed.org
                org_raw_json = dict(seed.raw_json or {}) if seed.raw_json else None
                org_trace = dict(seed.llm_trace or {})
                cache_used = True
                org_report["cache_used"] = True
                org_report["cache_reason"] = str(cache_reason)
            else:
                org_report["cache_used"] = False
                org_report["cache_reason"] = str(cache_reason)
        elif seed_policy == "force_cache":
            raise RuntimeError(f"no cached org seed for {spec_name}: {seed_path}")

        if org_doc is None and should_try_llm:
            hub = LLMOrgHub()
            candidate = hub.lift(
                desc,
                intent_summary=intent_summary,
                extra_evidence=extra_evidence,
                model=(_org_model() or None),
            )
            org_doc = candidate.org
            org_raw_json = dict(candidate.raw_json)
            org_trace = dict(candidate.llm_trace)
            org_report["llm_trace"] = dict(org_trace)
            org_report["prompt_hash"] = str(candidate.prompt_hash)
            if (not llm_fallback_used) and diff_ok and static_ok and seed_policy in {"auto", "force_llm"}:
                save_org_seed(
                    path=seed_path,
                    kernel=spec_name,
                    triton_provider=str(triton_provider),
                    backend_target=backend_target,
                    org=org_doc,
                    raw_json=org_raw_json,
                    llm_trace=org_trace,
                    quality=quality,
                )
                org_report["seed_saved"] = True
    except Exception as exc:  # noqa: BLE001
        org_report["ok"] = False
        org_report["error"] = f"{type(exc).__name__}: {exc}"
        if mode == "strict":
            raise
        return

    if org_doc is None:
        org_report["ok"] = False
        org_report["error"] = "org_doc_missing"
        if mode == "strict":
            raise RuntimeError("org_doc_missing")
        return

    org_path.write_text(json.dumps(org_doc.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    org_report["ok"] = True
    org_report["cache_used"] = bool(cache_used)

    if mode not in {"apply", "strict"}:
        return
    backend_norm = str(backend_target or "").strip().lower()
    if backend_norm and not backend_norm.startswith("cuda"):
        org_report["apply_skipped"] = True
        org_report["apply_reason"] = "backend_target_not_cuda"
        return

    build_hardware_model = load_org_attr("org.mapping.hardware_model", "build_hardware_model")
    hardware_model = build_hardware_model(target=str(backend_target or ""), arch=str(target_arch))

    try:
        budget = int(_org_budget())
        if str(spec_name) == "flash_attention2d":
            plan_flash_attention2d = load_org_attr("org.mapping.cuda.flash_attention2d", "plan_flash_attention2d")
            plan = plan_flash_attention2d(
                org_doc,
                shape_bindings=dict(shape_bindings),
                source_oracle=dict(source_oracle),
                hardware_model=hardware_model,
                budget=int(budget),
            )
        elif str(spec_name) == "matmul_fused_epilogue2d":
            plan_matmul_fused_epilogue2d = load_org_attr(
                "org.mapping.cuda.matmul_fused_epilogue2d", "plan_matmul_fused_epilogue2d"
            )
            plan = plan_matmul_fused_epilogue2d(
                org_doc,
                shape_bindings=dict(shape_bindings),
                source_oracle=dict(source_oracle),
                hardware_model=hardware_model,
                budget=int(budget),
            )
        else:
            org_report["apply_skipped"] = True
            org_report["apply_reason"] = "org_kernel_deferred"
            return

        plan_path.write_text(json.dumps(plan.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        org_report["plan_path"] = str(plan_path)
        org_report["arch"] = str(target_arch)
        org_report["source_oracle"] = dict(source_oracle)

        lines_jsonl: list[str] = []
        lines_txt: list[str] = [
            f"# kernel={spec_name} target={backend_target} budget={budget} compiler_stack={_compiler_stack_name()} arch={target_arch}",
            "# candidate syntax: <kernel_kind>:K=V,A=B",
            "# tune supports: python scripts/intentir.py tune --candidate-file <this_file> ...",
        ]
        for candidate in list(plan.candidates or []):
            lines_txt.append(_candidate_line(candidate.kernel_kind, candidate.bindings))
            lines_jsonl.append(
                json.dumps(
                    {
                        "schema_version": "intentir_tuning_db_entry_v1",
                        "backend": "cuda",
                        "compiler_stack": str(_compiler_stack_name()),
                        "kernel": str(spec_name),
                        "arch": str(target_arch),
                        "bindings": {str(k): int(v) for k, v in dict(candidate.bindings or {}).items()},
                        "kernel_kind": str(candidate.kernel_kind),
                        "note": "org_candidate",
                    },
                    ensure_ascii=False,
                )
            )
        cand_jsonl_path.write_text("\n".join(lines_jsonl) + ("\n" if lines_jsonl else ""), encoding="utf-8")
        cand_txt_path.write_text("\n".join(lines_txt) + "\n", encoding="utf-8")
        org_report["candidates_path"] = str(cand_jsonl_path)
        org_report["candidates_txt_path"] = str(cand_txt_path)
        org_report["candidates_count"] = int(len(list(plan.candidates or [])))
    except Exception as exc:  # noqa: BLE001
        org_report["apply_ok"] = False
        org_report["apply_error"] = f"{type(exc).__name__}: {exc}"
        if mode == "strict":
            raise


__all__ = ["load_org_attr", "load_org_module", "run_org_sidecar"]
