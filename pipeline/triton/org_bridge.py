from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain

from pipeline.triton.core import (
    _candidate_line,
    _compiler_stack_name,
    _compiler_cpp_wave_name,
    _detect_cuda_arch,
    _emit_mlir_shadow_artifacts,
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
from pipeline.triton.remote_source_oracle import apply_remote_source_oracle, remote_source_enabled


ROOT = Path(__file__).resolve().parents[2]
ORG_RUNTIME_ROOT = ROOT / "ORG-Migrate"


def _org_blindfold_enabled() -> bool:
    raw = str(os.getenv("INTENTIR_ORG_BLINDFOLD", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _org_blindfold_label() -> str:
    raw = str(os.getenv("INTENTIR_ORG_BLINDFOLD_LABEL", "") or "").strip()
    return raw or "target_kernel_func"


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


def _resolve_existing_path(*values: object) -> str:
    for raw in values:
        if raw is None:
            continue
        p = Path(str(raw))
        if p.is_file():
            return str(p)
    return ""


def _build_intent_summary(intent: IntentFunction) -> dict[str, object]:
    return {
        "name": str(intent.name or ""),
        "op_names": [str(op.op) for op in list(intent.ops or []) if str(getattr(op, "op", "")).strip()],
        "outputs": [str(x) for x in list(intent.outputs or []) if str(x).strip()],
        "parallel_axes": [str(x) for x in list(intent.parallel_axes or []) if str(x).strip()],
        "axis_roles": dict(getattr(intent, "axis_roles", {}) or {}),
        "regions": [
            {
                "id": str(getattr(region, "id", "") or ""),
                "kind": str(getattr(region, "kind", "") or ""),
                "predicate": str(getattr(region, "predicate", "") or ""),
                "path_id": str(getattr(region, "path_id", "") or ""),
                "inputs": [str(x) for x in list(getattr(region, "inputs", []) or []) if str(x).strip()],
                "outputs": [str(x) for x in list(getattr(region, "outputs", []) or []) if str(x).strip()],
                "meta": dict(getattr(region, "meta", {}) or {}),
            }
            for region in list(getattr(intent, "regions", []) or [])
        ],
        "schedule": (intent.schedule.__dict__ if getattr(intent, "schedule", None) is not None else None),
    }


def _resolve_source_oracle_facts(*, spec_name: str, shape_bindings: Mapping[str, int]) -> dict[str, Any]:
    source_arch = _normalize_cuda_arch_key(os.getenv("INTENTIR_ORG_SOURCE_ARCH", ""))
    target_arch = _detect_cuda_arch() or _normalize_cuda_arch_key(str(os.getenv("INTENTIR_CUDA_SM", "") or ""))
    source_stack_env = str(os.getenv("INTENTIR_ORG_SOURCE_COMPILER_STACK", "") or "").strip().lower()
    source_stack = source_stack_env or _compiler_stack_name()
    source_db_env = str(os.getenv("INTENTIR_ORG_SOURCE_TUNING_DB", "") or "").strip()
    extract_source_oracle_facts = load_org_attr("org.facts.source_oracle", "extract_source_oracle_facts")
    return extract_source_oracle_facts(
        kernel=str(spec_name),
        source_arch=str(source_arch),
        target_arch=str(target_arch),
        shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        compiler_stack=str(source_stack),
        db_path=(str(source_db_env) if source_db_env else None),
    )


def _org_compile_topk() -> int:
    raw = str(os.getenv("INTENTIR_ORG_COMPILE_TOPK", "4") or "").strip()
    try:
        return max(0, int(raw))
    except Exception:
        return 4


def _org_ignore_diff_gate() -> bool:
    raw = str(os.getenv("INTENTIR_ORG_IGNORE_DIFF_GATE", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _compile_check_id(candidate_line: str, *, idx: int) -> str:
    h = hashlib.sha256(str(candidate_line).encode("utf-8")).hexdigest()[:10]
    return f"{idx:02d}_{h}"


def _toolchain_env_overrides(toolchain_model: Mapping[str, Any] | None) -> dict[str, str]:
    model = dict(toolchain_model or {})
    env: dict[str, str] = {}
    if bool(model.get("requires_real_mlir")):
        env["INTENTIR_REAL_MLIR"] = "1"
        env["INTENTIR_CUDA_REAL_MLIR_ALLOW_UNKNOWN"] = "1"
    cuda_wave = str(model.get("cuda_real_mlir_wave") or "").strip().lower()
    if cuda_wave:
        env["INTENTIR_CUDA_REAL_MLIR_WAVE"] = cuda_wave
    rvv_wave = str(model.get("rvv_real_mlir_wave") or "").strip().lower()
    if rvv_wave:
        env["INTENTIR_RVV_REAL_MLIR_WAVE"] = rvv_wave
    return env


def _report_contract_exec_meta(report: Mapping[str, Any]) -> tuple[str, str, str, str, str, bool | None, str]:
    mlir = dict(report.get("mlir") or {})
    exec_meta = dict(
        (
            mlir.get("downstream_cuda_std_llvm_contract_exec_meta")
            or mlir.get("downstream_cuda_contract_exec_meta")
            or mlir.get("downstream_cuda_llvm_contract_exec_meta")
            or {}
        )
    )
    contract_path = str(
        mlir.get("downstream_cuda_std_llvm_contract_path")
        or mlir.get("downstream_cuda_contract_path")
        or mlir.get("downstream_cuda_llvm_contract_path")
        or mlir.get("downstream_contract_path")
        or ""
    )
    ptx_path = str(exec_meta.get("cuda_ptx_path") or "")
    entry = str(((exec_meta.get("cuda_ptx_entries") or [None]) or [None])[0] or "")
    requested_sm = str(exec_meta.get("cuda_requested_sm") or "")
    effective_sm = str(exec_meta.get("cuda_effective_sm") or "")
    downleveled = exec_meta.get("cuda_target_downleveled")
    error = str(mlir.get("error") or "")
    return contract_path, ptx_path, entry, requested_sm, effective_sm, downleveled, error


def _run_inline_compile_check(
    *,
    spec_name: str,
    cand_dir: Path,
    backend_target: str | None,
    intent: IntentFunction,
    shape_bindings: Mapping[str, int],
    kernel_kind: str = "",
    candidate_bindings: Mapping[str, int] | None = None,
    env_updates: Mapping[str, str],
) -> tuple[bool, dict[str, Any], str]:
    report: dict[str, Any] = {"kernel": str(spec_name), "mlir": {}}
    report_path = cand_dir / f"{spec_name}.json"
    saved_env = {str(k): os.environ.get(str(k)) for k in dict(env_updates or {}).keys()}
    try:
        for key, value in dict(env_updates or {}).items():
            os.environ[str(key)] = str(value)
        intent_copy = IntentFunction.from_json_dict(intent.to_json_dict())
        intent_copy.meta = dict(getattr(intent_copy, "meta", {}) or {})
        intent_copy.meta.setdefault("kernel", str(spec_name))
        intent_copy.meta.setdefault("spec_name", str(spec_name))
        merged_shape_bindings = {
            str(k): int(v)
            for k, v in dict(shape_bindings or {}).items()
            if str(k).strip()
        }
        for key, value in dict(candidate_bindings or {}).items():
            key_s = str(key).strip()
            if not key_s:
                continue
            try:
                merged_shape_bindings[key_s] = int(value)
            except Exception:
                continue
        if merged_shape_bindings:
            intent_copy.meta["shape_bindings"] = dict(merged_shape_bindings)
        kernel_kind_s = str(kernel_kind or "").strip()
        if kernel_kind_s and kernel_kind_s != "generic_fallback_v1":
            intent_copy.meta["intentir_kernel_kind_override"] = kernel_kind_s
            intent_copy.meta["intentir_org_compile_check_candidate"] = {
                "kernel_kind": kernel_kind_s,
                "bindings": {
                    str(k): int(v)
                    for k, v in dict(candidate_bindings or {}).items()
                    if str(k).strip()
                },
            }
        _emit_mlir_shadow_artifacts(
            spec_name=str(spec_name),
            out_dir=Path(cand_dir),
            intent=intent_copy,
            report=report,
            backend_target=backend_target,
            shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        )
        ok = True
        error = ""
    except Exception as exc:  # noqa: BLE001
        report["mlir"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for key, old in saved_env.items():
            if old is None:
                os.environ.pop(str(key), None)
            else:
                os.environ[str(key)] = str(old)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return ok, report, error


def _run_compile_check_candidates(
    *,
    spec_name: str,
    out_dir: Path,
    backend_target: str | None,
    target_arch: str,
    candidates: list[object],
    intent: IntentFunction,
    shape_bindings: Mapping[str, int],
    toolchain_model: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    limit = int(_org_compile_topk())
    if limit <= 0:
        return []
    CompileCheck = load_org_attr("org.backend_model", "CompileCheck")
    checks: list[dict[str, Any]] = []
    compile_root = Path(out_dir) / "org_compile_checks"
    compile_root.mkdir(parents=True, exist_ok=True)
    compiler_stack = str(_compiler_stack_name())
    compiler_cpp_wave = str(_compiler_cpp_wave_name()) if compiler_stack in {"cpp", "cpp_plugin", "c++"} else ""
    def _append_compile_check(
        *,
        idx: int,
        candidate_line: str,
        kernel_kind: str,
        bindings: Mapping[str, int],
        env_updates: Mapping[str, str],
    ) -> None:
        cand_dir = compile_root / _compile_check_id(candidate_line, idx=idx)
        cand_dir.mkdir(parents=True, exist_ok=True)
        report_path = cand_dir / f"{spec_name}.json"
        contract_path = ""
        ptx_path = ""
        entry = ""
        requested_sm = ""
        effective_sm = ""
        downleveled: bool | None = None
        error = ""
        ok, report, run_error = _run_inline_compile_check(
            spec_name=str(spec_name),
            cand_dir=Path(cand_dir),
            backend_target=backend_target,
            intent=intent,
            shape_bindings=shape_bindings,
            kernel_kind=str(kernel_kind),
            candidate_bindings=dict(bindings or {}),
            env_updates=env_updates,
        )
        contract_path, ptx_path, entry, requested_sm, effective_sm, downleveled, error = _report_contract_exec_meta(report)
        if not error:
            error = str(run_error or "")
        ok = bool(ok and contract_path and ptx_path and entry)
        check = CompileCheck(
            candidate=str(candidate_line),
            kernel_kind=str(kernel_kind),
            bindings={str(k): int(v) for k, v in dict(bindings or {}).items()},
            report_path=str(report_path),
            contract_path=str(contract_path),
            ptx_path=str(ptx_path),
            entry=str(entry),
            requested_sm=str(requested_sm),
            effective_sm=str(effective_sm),
            downleveled=downleveled,
            ok=bool(ok),
            error=str(error),
        )
        checks.append(check.to_json_dict())

    for idx, candidate in enumerate(list(candidates or [])[: int(limit)]):
        cand_line = _candidate_line(getattr(candidate, "kernel_kind"), getattr(candidate, "bindings"))
        cand_dir = compile_root / _compile_check_id(cand_line, idx=idx)
        tuning_path = cand_dir / "tuning.jsonl"
        cand_dir.mkdir(parents=True, exist_ok=True)
        tuning_path.write_text(
            json.dumps(
                {
                    "schema_version": "intentir_tuning_db_entry_v1",
                    "backend": "cuda",
                    "compiler_stack": compiler_stack,
                    "kernel": str(spec_name),
                    "arch": str(target_arch),
                    "bindings": {str(k): int(v) for k, v in dict(getattr(candidate, "bindings", {}) or {}).items()},
                    "kernel_kind": str(getattr(candidate, "kernel_kind")),
                    "note": "org_compile_check",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        env_updates = {
            "INTENTIR_ORG_MODE": "off",
            "INTENTIR_CUDA_TUNING_DB": str(tuning_path),
            "INTENTIR_COMPILER_STACK": compiler_stack,
        }
        if compiler_cpp_wave:
            env_updates["INTENTIR_COMPILER_CPP_WAVE"] = compiler_cpp_wave
        env_updates.update(_toolchain_env_overrides(toolchain_model))
        _append_compile_check(
            idx=idx,
            candidate_line=str(cand_line),
            kernel_kind=str(getattr(candidate, "kernel_kind")),
            bindings={str(k): int(v) for k, v in dict(getattr(candidate, "bindings", {}) or {}).items()},
            env_updates=env_updates,
        )
    if not any(bool(dict(x).get("ok")) for x in list(checks or [])):
        fallback_env = {
            "INTENTIR_ORG_MODE": "off",
            "INTENTIR_COMPILER_STACK": compiler_stack,
        }
        if compiler_cpp_wave:
            fallback_env["INTENTIR_COMPILER_CPP_WAVE"] = compiler_cpp_wave
        fallback_env.update(_toolchain_env_overrides(toolchain_model))
        _append_compile_check(
            idx=int(len(checks)),
            candidate_line="generic_fallback_v1",
            kernel_kind="generic_fallback_v1",
            bindings={},
            env_updates=fallback_env,
        )
    return checks


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
        "compiler_stack": str(_compiler_stack_name()),
        "compiler_cpp_wave": (str(_compiler_cpp_wave_name()) if str(_compiler_stack_name()) in {"cpp", "cpp_plugin", "c++"} else ""),
        "runtime_root": str(ORG_RUNTIME_ROOT),
    }
    report["org"] = org_report

    diff_ok = bool((report.get("diff") or {}).get("ok"))
    static_ok = False
    if isinstance(report.get("static_validation"), dict):
        static_ok = bool((report.get("static_validation") or {}).get("ok"))
    elif diff_ok:
        static_ok = True
        org_report["static_validation_assumed"] = True
    if ((not diff_ok) or (not static_ok)) and (not _org_ignore_diff_gate()):
        reason = f"skip_org: diff_ok={diff_ok} static_ok={static_ok}"
        org_report["skipped"] = True
        org_report["reason"] = reason
        if mode == "strict":
            raise RuntimeError(reason)
        return
    if (not diff_ok) or (not static_ok):
        org_report["diff_gate_overridden"] = True
        org_report["diff_gate_status"] = {"diff_ok": diff_ok, "static_ok": static_ok}

    seed_policy = _org_seed_policy()
    if seed_policy not in {"auto", "force_llm", "force_cache"}:
        raise ValueError(f"unsupported INTENTIR_ORG_SEED_POLICY={seed_policy!r}")

    seed_path = _org_seed_path(out_dir, spec_name)
    org_path = _org_doc_path(out_dir, spec_name)
    plan_path = _org_plan_path(out_dir, spec_name)
    cand_jsonl_path = _org_candidates_jsonl_path(out_dir, spec_name)
    cand_txt_path = _org_candidates_txt_path(out_dir, spec_name)
    ttgir_facts_path = Path(out_dir) / f"{str(spec_name)}.org_ttgir_facts.json"
    ptx_facts_path = Path(out_dir) / f"{str(spec_name)}.org_ptx_facts.json"
    source_oracle_facts_path = Path(out_dir) / f"{str(spec_name)}.org_source_oracle_facts.json"
    hardware_model_path = Path(out_dir) / f"{str(spec_name)}.org_hardware_model.json"
    org_report["seed_path"] = str(seed_path)
    org_report["org_path"] = str(org_path)

    intent_summary = _build_intent_summary(intent)
    target_arch = _detect_cuda_arch() or _normalize_cuda_arch_key(str(backend_target or "")) or ""
    source_oracle_facts = _resolve_source_oracle_facts(spec_name=spec_name, shape_bindings=shape_bindings)
    source_oracle = dict(source_oracle_facts.get("oracle") or {})
    source_oracle_facts_path.write_text(json.dumps(source_oracle_facts, indent=2, ensure_ascii=False), encoding="utf-8")
    org_report["source_oracle_facts_path"] = str(source_oracle_facts_path)

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
        if remote_source_enabled():
            remote_source = apply_remote_source_oracle(
                spec_name=str(spec_name),
                out_dir=Path(out_dir),
                desc=desc,
                shape_bindings=shape_bindings,
            )
            if isinstance(remote_source, dict):
                org_report["remote_source"] = dict(remote_source)
                remote_source_arch = str(remote_source.get("source_arch") or "").strip()
                if remote_source_arch:
                    extra_evidence["source_arch"] = remote_source_arch
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
        llvm_ir_path = _resolve_existing_path(
            (getattr(getattr(desc, "artifacts", None), "extra", {}) or {}).get("llvm_ir_path"),
            (getattr(desc, "meta", {}) or {}).get("llvm_ir_original_path"),
        )
        cubin_path = _resolve_existing_path(
            (getattr(getattr(desc, "artifacts", None), "extra", {}) or {}).get("cubin_path"),
            (getattr(desc, "meta", {}) or {}).get("cubin_original_path"),
        )
        facts_kernel_name = _org_blindfold_label() if _org_blindfold_enabled() else str(spec_name)
        if ttgir_text.strip():
            ttgir_facts = extract_ttgir_mechanism_facts(ttgir_text, kernel_name=facts_kernel_name, artifact_path=(ttgir_path or None))
            extra_evidence["ttgir_facts"] = dict(ttgir_facts)
            ttgir_facts_path.write_text(json.dumps(ttgir_facts, indent=2, ensure_ascii=False), encoding="utf-8")
            org_report["ttgir_facts_path"] = str(ttgir_facts_path)
        ptx_facts = extract_ptx_mechanism_facts(ptx_text, kernel_name=facts_kernel_name, artifact_path=(ptx_path or None))
        extra_evidence["ptx_facts"] = dict(ptx_facts)
        ptx_facts_path.write_text(json.dumps(ptx_facts, indent=2, ensure_ascii=False), encoding="utf-8")
        org_report["ptx_facts_path"] = str(ptx_facts_path)
        extra_evidence["ttir_summary"] = dict(ttir_summary)
        org_report["evidence_source"] = {
            "primary": ("ttgir" if ttgir_facts is not None else "ttir"),
            "ttgir_available": bool(ttgir_facts is not None),
            "ttgir_path": (ttgir_path or None),
            "ptx_available": bool((ptx_facts or {}).get("artifacts", {}).get("ptx_available")),
            "ptx_path": (ptx_path or None),
            "llvm_ir_path": (llvm_ir_path or None),
            "cubin_path": (cubin_path or None),
            "ttir_available": bool((ttir_summary or {}).get("available")),
        }

    if mode in {"apply", "strict"} and ttgir_facts is None and not bool((ptx_facts or {}).get("mechanisms")):
        org_report["ok"] = False
        org_report["error"] = "insufficient_schedule_evidence"
        if mode == "strict":
            raise RuntimeError("insufficient_schedule_evidence")
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
    build_toolchain_model = load_org_attr("org.backend_model", "build_toolchain_model")
    hardware_model = build_hardware_model(target=str(backend_target or ""), arch=str(target_arch))
    org_report["hardware_model"] = hardware_model.to_json_dict()
    org_report["hardware_cluster"] = str(hardware_model.arch_cluster)
    org_report["source_oracle_available"] = bool(source_oracle_facts.get("available"))
    hardware_model_path.write_text(json.dumps(hardware_model.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    org_report["hardware_model_path"] = str(hardware_model_path)
    mlir_report = dict(report.get("mlir") or {}) if isinstance(report.get("mlir"), dict) else {}
    toolchain_model = build_toolchain_model(
        toolchain_report=(
            dict(mlir_report.get("toolchain") or {})
            if isinstance(mlir_report.get("toolchain"), Mapping)
            else detect_mlir_toolchain()
        ),
        contract_exec_meta=(
            dict(mlir_report.get("downstream_cuda_std_llvm_contract_exec_meta") or mlir_report.get("downstream_cuda_contract_exec_meta") or {})
            if (
                isinstance(mlir_report.get("downstream_cuda_std_llvm_contract_exec_meta"), Mapping)
                or isinstance(mlir_report.get("downstream_cuda_contract_exec_meta"), Mapping)
            )
            else {}
        ),
        compiler_stack=str(_compiler_stack_name()),
        requested_sm=str(target_arch),
        execution_ir=str(mlir_report.get("execution_ir") or ""),
        llvm_pipeline=str(mlir_report.get("llvm_pipeline") or ""),
        cuda_real_mlir_wave=str(mlir_report.get("cuda_real_mlir_wave") or ""),
        rvv_real_mlir_wave=str(mlir_report.get("rvv_real_mlir_wave") or ""),
    )

    try:
        budget = int(_org_budget())
        plan_cuda_kernel = load_org_attr("org.mapping.cuda.universal_planner", "plan_cuda_kernel")
        try:
            plan = plan_cuda_kernel(
                str(spec_name),
                org_doc,
                shape_bindings=dict(shape_bindings),
                source_oracle=dict(source_oracle),
                hardware_model=hardware_model,
                ttgir_facts=dict(ttgir_facts or {}),
                ptx_facts=dict(ptx_facts or {}),
                toolchain_model=toolchain_model.to_json_dict(),
                budget=int(budget),
            )
        except ValueError as exc:
            compile_checks = _run_compile_check_candidates(
                spec_name=str(spec_name),
                out_dir=Path(out_dir),
                backend_target=backend_target,
                target_arch=str(target_arch),
                candidates=[],
                intent=intent,
                shape_bindings=dict(shape_bindings),
                toolchain_model=toolchain_model.to_json_dict(),
            )
            org_report["apply_skipped"] = True
            org_report["apply_reason"] = "org_kernel_deferred"
            org_report["apply_error"] = f"{type(exc).__name__}: {exc}"
            org_report["compile_checks"] = list(compile_checks or [])
            org_report["compile_checks_count"] = int(len(list(compile_checks or [])))
            org_report["realizations"] = [dict(x) for x in list(compile_checks or []) if bool(dict(x).get("ok"))]
            return

        plan.toolchain_model = dict(toolchain_model.to_json_dict())
        plan.effective_target = {
            "backend_target": str(backend_target or ""),
            "requested_sm": str((plan.toolchain_model or {}).get("requested_sm") or ""),
            "effective_sm": str((plan.toolchain_model or {}).get("effective_sm") or ""),
            "downleveled": (plan.toolchain_model or {}).get("downleveled"),
        }
        compile_checks = _run_compile_check_candidates(
            spec_name=str(spec_name),
            out_dir=Path(out_dir),
            backend_target=backend_target,
            target_arch=str(target_arch),
            candidates=list(plan.candidates or []),
            intent=intent,
            shape_bindings=dict(shape_bindings),
            toolchain_model=toolchain_model.to_json_dict(),
        )
        plan.compile_checks = list(compile_checks)
        plan.realizations = [dict(x) for x in list(compile_checks or []) if bool(dict(x).get("ok"))]
        if (not str((plan.toolchain_model or {}).get("requested_sm") or "")) and plan.realizations:
            first_realization = dict(plan.realizations[0] or {})
            plan.toolchain_model["requested_sm"] = str(first_realization.get("requested_sm") or "")
            plan.toolchain_model["effective_sm"] = str(first_realization.get("effective_sm") or "")
            plan.toolchain_model["downleveled"] = first_realization.get("downleveled")
            plan.effective_target = {
                "backend_target": str(backend_target or ""),
                "requested_sm": str(first_realization.get("requested_sm") or ""),
                "effective_sm": str(first_realization.get("effective_sm") or ""),
                "downleveled": first_realization.get("downleveled"),
            }

        plan_path.write_text(json.dumps(plan.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        org_report["plan_path"] = str(plan_path)
        org_report["arch"] = str(target_arch)
        org_report["source_oracle"] = dict(source_oracle)
        org_report["toolchain_model"] = dict(plan.toolchain_model or {})
        org_report["effective_target"] = dict(plan.effective_target or {})
        org_report["compile_checks_count"] = int(len(list(plan.compile_checks or [])))

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
