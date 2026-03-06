from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from intent_ir.ir import IntentFunction

from pipeline.triton.core import (
    _apply_source_oracle_ordering,
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


def _build_intent_summary(intent: IntentFunction) -> dict[str, object]:
    return {
        "name": str(intent.name or ""),
        "op_names": [str(op.op) for op in list(intent.ops or []) if str(getattr(op, "op", "")).strip()],
        "outputs": [str(x) for x in list(intent.outputs or []) if str(x).strip()],
        "parallel_axes": [str(x) for x in list(intent.parallel_axes or []) if str(x).strip()],
        "axis_roles": dict(getattr(intent, "axis_roles", {}) or {}),
        "schedule": (intent.schedule.__dict__ if getattr(intent, "schedule", None) is not None else None),
    }


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
    extra_evidence = {
        "shape_bindings": {str(k): int(v) for k, v in dict(shape_bindings or {}).items() if str(k).strip()},
        "backend_target": (str(backend_target) if backend_target is not None else None),
        "triton_provider": str(triton_provider),
        "contract_level": str((report.get("contract") or {}).get("level") or ""),
    }
    quality = {
        "diff_ok": bool(diff_ok),
        "static_ok": bool(static_ok),
        "contract_level": str((report.get("contract") or {}).get("level") or ""),
    }
    llm_fallback_used = bool((report.get("llm_fallback") or {}).get("used"))

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
                except Exception as e:  # noqa: BLE001
                    cache_allowed = False
                    cache_reason = f"invalid_seed:{type(e).__name__}"
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
                try:
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
                except Exception as e:  # noqa: BLE001
                    org_report["seed_saved"] = False
                    org_report["seed_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        org_report["ok"] = False
        org_report["error"] = f"{type(e).__name__}: {e}"
        if mode == "strict":
            raise
        return

    if org_doc is None:
        org_report["ok"] = False
        org_report["error"] = "org_doc_missing"
        if mode == "strict":
            raise RuntimeError("org_doc_missing")
        return

    try:
        org_path.write_text(json.dumps(org_doc.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        org_report["ok"] = True
        org_report["cache_used"] = bool(cache_used)
    except Exception as e:  # noqa: BLE001
        org_report["ok"] = False
        org_report["error"] = f"write_org_error:{type(e).__name__}: {e}"
        if mode == "strict":
            raise
        return

    if mode not in {"apply", "strict"}:
        return

    backend_norm = str(backend_target or "").strip().lower()
    if backend_norm and not backend_norm.startswith("cuda"):
        org_report["apply_skipped"] = True
        org_report["apply_reason"] = "backend_target_not_cuda"
        return

    source_oracle: dict[str, object] | None = None
    source_arch = _normalize_cuda_arch_key(os.getenv("INTENTIR_ORG_SOURCE_ARCH", ""))
    if source_arch:
        source_stack_env = str(os.getenv("INTENTIR_ORG_SOURCE_COMPILER_STACK", "") or "").strip().lower()
        source_stack = source_stack_env or _compiler_stack_name()
        source_db_env = str(os.getenv("INTENTIR_ORG_SOURCE_TUNING_DB", "") or "").strip()
        try:
            from pipeline.common.tuning_db import load_tuning_db_jsonl, resolve_tuning_db_path, resolve_tuning_entries

            db_path = resolve_tuning_db_path(
                path=(Path(source_db_env) if source_db_env else None),
                backend="cuda",
            )
            if db_path is not None and Path(db_path).is_file():
                db = load_tuning_db_jsonl(path=Path(db_path), backend="cuda")
                entries = db.get((str(spec_name), str(source_arch))) or []
                merged, kk = resolve_tuning_entries(
                    entries,
                    shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
                    compiler_stack=str(source_stack),
                )
                kk = str(kk or "").strip()
                merged = {str(k): int(v) for k, v in dict(merged).items() if str(k).strip()}
                if kk or merged:
                    source_oracle = {
                        "arch": str(source_arch),
                        "compiler_stack": str(source_stack),
                        "db_path": str(db_path),
                        "kernel_kind": str(kk),
                        "bindings": dict(merged),
                    }
        except Exception as e:  # noqa: BLE001
            org_report["source_oracle_error"] = f"{type(e).__name__}: {e}"

    try:
        stack = _compiler_stack_name()
        budget = int(_org_budget())
        enum_budget = max(int(budget), 32)
        if source_oracle is not None:
            enum_budget = max(enum_budget, 128)

        if str(spec_name) == "flash_attention2d":
            plan_flash_attention2d = load_org_attr("org.mapping.cuda.flash_attention2d", "plan_flash_attention2d")
            enable_cpp_extras = stack in {"cpp", "cpp_plugin", "c++"}
            plan = plan_flash_attention2d(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
                enable_cpp_extras=bool(enable_cpp_extras),
            )
        elif str(spec_name) == "masked_attention2d":
            plan_masked_attention2d = load_org_attr("org.mapping.cuda.masked_attention2d", "plan_masked_attention2d")
            plan = plan_masked_attention2d(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
                compiler_stack=str(stack),
            )
        elif str(spec_name) == "_attn_fwd":
            plan_attn_fwd = load_org_attr("org.mapping.cuda.attn_fwd", "plan_attn_fwd")
            plan = plan_attn_fwd(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
                compiler_stack=str(stack),
            )
        elif str(spec_name) == "matmul_fused_epilogue2d":
            plan_matmul_fused_epilogue2d = load_org_attr(
                "org.mapping.cuda.matmul_fused_epilogue2d", "plan_matmul_fused_epilogue2d"
            )
            plan = plan_matmul_fused_epilogue2d(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
            )
        elif str(spec_name) == "ai_bench_softmax":
            plan_ai_bench_softmax = load_org_attr("org.mapping.cuda.ai_bench_softmax", "plan_ai_bench_softmax")
            plan = plan_ai_bench_softmax(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
            )
        elif str(spec_name) == "ai_bench_matmul":
            plan_ai_bench_matmul = load_org_attr("org.mapping.cuda.ai_bench_matmul", "plan_ai_bench_matmul")
            plan = plan_ai_bench_matmul(
                org_doc,
                shape_bindings=dict(shape_bindings),
                target=str(backend_target or "cuda"),
                budget=enum_budget,
            )
        else:
            org_report["apply_skipped"] = True
            org_report["apply_reason"] = "kernel_not_supported"
            return

        if source_oracle is not None:
            org_report["source_oracle"] = dict(source_oracle)
            _apply_source_oracle_ordering(plan, oracle=source_oracle, budget=int(budget))
        else:
            if int(budget) > 0:
                plan.candidates = list(plan.candidates or [])[: int(budget)]

        plan_path.write_text(json.dumps(plan.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        org_report["plan_path"] = str(plan_path)

        arch = _detect_cuda_arch() or ""
        org_report["arch"] = str(arch)

        head_dim = int(shape_bindings.get("HEAD_DIM") or 0)
        when = {"HEAD_DIM": int(head_dim)} if head_dim > 0 else {}
        lines_jsonl: list[str] = []
        lines_txt: list[str] = []
        lines_txt.append(f"# kernel={spec_name} target={backend_target} budget={budget} compiler_stack={stack} arch={arch}")
        if source_oracle is not None:
            so_kind = str(source_oracle.get("kernel_kind") or "").strip()
            so_arch = str(source_oracle.get("arch") or "").strip()
            so_stack = str(source_oracle.get("compiler_stack") or "").strip()
            so_bind = source_oracle.get("bindings")
            flat = ""
            if isinstance(so_bind, Mapping):
                flat = ",".join(f"{k}={int(v)}" for k, v in sorted({str(k): int(v) for k, v in dict(so_bind).items()}.items()))
            lines_txt.append(f"# source_oracle arch={so_arch} compiler_stack={so_stack} kernel_kind={so_kind} bindings={flat}")
        lines_txt.append("# candidate syntax: <kernel_kind>:K=V,A=B")
        lines_txt.append("# tune supports: python scripts/intentir.py tune --candidate-file <this_file> ...")
        stack_norm = "cpp_plugin" if stack in {"cpp", "c++"} else str(stack)
        for c in list(plan.candidates or []):
            lines_txt.append(_candidate_line(c.kernel_kind, c.bindings))
            entry = {
                "schema_version": "intentir_tuning_db_entry_v1",
                "backend": "cuda",
                "compiler_stack": str(stack_norm),
                "kernel": str(spec_name),
                "arch": str(arch),
                "when": dict(when),
                "bindings": {str(k): int(v) for k, v in dict(c.bindings or {}).items()},
                "kernel_kind": str(c.kernel_kind),
                "note": "org_candidate",
            }
            lines_jsonl.append(json.dumps(entry, ensure_ascii=False))
        cand_jsonl_path.write_text("\n".join(lines_jsonl) + ("\n" if lines_jsonl else ""), encoding="utf-8")
        cand_txt_path.write_text("\n".join(lines_txt) + "\n", encoding="utf-8")
        org_report["candidates_path"] = str(cand_jsonl_path)
        org_report["candidates_txt_path"] = str(cand_txt_path)
        org_report["candidates_count"] = int(len(list(plan.candidates or [])))
    except Exception as e:  # noqa: BLE001
        org_report["apply_ok"] = False
        org_report["apply_error"] = f"{type(e).__name__}: {e}"
        if mode == "strict":
            raise


__all__ = ["load_org_attr", "load_org_module", "run_org_sidecar"]
