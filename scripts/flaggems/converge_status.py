"""
Converge FlagGems registry status using execution artifacts.

Inputs are optional and can come from:
- Triton provider reports (`artifacts/flaggems_triton_full_pipeline/*.json`)
- RVV smoke summary (`scripts/backend_codegen_smoke.py --json --out ...`)
- CUDA smoke summary (`scripts/cuda_backend_smoke.py --json --out ...`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _parse_scope_values(raw_values: list[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for raw in list(raw_values or []):
        for token in str(raw).split(","):
            v = str(token).strip()
            if v:
                out.append(v)
    return tuple(sorted(set(out)))


def _load_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_result_map(summary: dict | None) -> tuple[dict[str, bool], dict[str, dict[str, Any]], bool, str | None]:
    if not isinstance(summary, dict):
        return {}, {}, False, None
    skipped = bool(summary.get("skipped"))
    skip_reason = str(summary.get("skip_reason")) if summary.get("skip_reason") is not None else None
    results = summary.get("results")
    if not isinstance(results, list):
        return {}, {}, skipped, skip_reason
    out: dict[str, bool] = {}
    details: dict[str, dict[str, Any]] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        k = r.get("kernel")
        if not isinstance(k, str) or not k:
            continue
        out[k] = bool(r.get("ok"))
        details[k] = dict(r)
    return out, details, skipped, skip_reason


def _provider_report_state(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return {"exists": False, "diff_ok": False}
    try:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {"exists": True, "diff_ok": False, "parse_error": True}
    diff_ok = bool(((rep.get("diff") or {}).get("ok")))
    return {"exists": True, "diff_ok": diff_ok}


def _derive_status(
    *,
    intent_ops: list[str],
    has_e2e_spec: bool,
    provider_ok: bool,
    rvv: str,
    cuda: str,
) -> tuple[str, str]:
    if not intent_ops:
        return "blocked_ir", "no_intentir_mapping"
    if not has_e2e_spec:
        return "blocked_backend", "missing_e2e_spec"
    if not provider_ok:
        return "blocked_backend", "pipeline_diff_failed_or_missing"

    rvv_pass = rvv == "pass"
    cuda_pass = cuda == "pass"
    if rvv_pass and cuda_pass:
        return "dual_pass", "runtime_dual_backend_pass"
    if rvv_pass and cuda in {"fail", "unknown", "skip"}:
        if cuda == "fail":
            return "rvv_only", "runtime_cuda_fail"
        if cuda == "skip":
            return "rvv_only", "runtime_cuda_skip"
        return "rvv_only", "runtime_cuda_unknown"
    if cuda_pass and rvv in {"fail", "unknown", "skip"}:
        if rvv == "fail":
            return "cuda_only", "runtime_rvv_fail"
        if rvv == "skip":
            return "cuda_only", "runtime_rvv_skip"
        return "cuda_only", "runtime_rvv_unknown"
    return "blocked_backend", "runtime_backend_fail"


def _classify_runtime_reason(state: str, detail: dict[str, Any], *, skip_reason: str | None) -> tuple[str, str]:
    s = str(state)
    if s == "pass":
        return "ok", "backend runtime passed"
    if s == "skip":
        reason = str(skip_reason or "env_unavailable")
        return "env_unavailable", f"backend skipped: {reason}"
    if s == "unknown":
        return "runtime_unknown", "missing result for kernel in backend summary"

    detail_reason = str(detail.get("reason_code") or "").strip().lower()
    if detail_reason in {"compile_timeout", "launch_timeout", "runtime_timeout"}:
        return "runtime_timeout", f"backend runtime timeout ({detail_reason})"
    if detail_reason == "env_unavailable":
        return "env_unavailable", "backend environment unavailable"
    if detail_reason == "lowering_missing_op":
        return "lowering_missing_op", "backend lowering does not support one or more ops"
    if detail_reason == "diff_fail":
        return "diff_fail", "numerical diff failed"
    if detail_reason in {"runtime_fail", "artifact_missing"}:
        return "runtime_fail", ("backend runtime failed" if not detail_reason else f"backend runtime failed ({detail_reason})")

    err = detail.get("error")
    msg_parts: list[str] = []
    if isinstance(err, dict):
        et = err.get("type")
        em = err.get("message")
        if et is not None:
            msg_parts.append(str(et))
        if em is not None:
            msg_parts.append(str(em))
    for key in ("stderr", "stdout"):
        val = detail.get(key)
        if isinstance(val, str) and val.strip():
            msg_parts.append(val.strip())
    msg = " | ".join(msg_parts).lower()
    if "timeout" in msg:
        return "runtime_timeout", "backend runtime timeout"
    if "unsupported op" in msg or "unsupported" in msg or "lowering" in msg:
        return "lowering_missing_op", "backend lowering does not support one or more ops"
    if "diff" in msg or "mismatch" in msg:
        return "diff_fail", "numerical diff failed"
    return "runtime_fail", ("backend runtime failed" if not msg_parts else " | ".join(msg_parts))


def _derive_reason_code(
    *,
    status_reason: str,
    provider_state: dict[str, Any],
    rvv_reason_code: str,
    cuda_reason_code: str,
) -> str:
    reason = str(status_reason)
    if reason == "pipeline_diff_failed_or_missing":
        if not bool(provider_state.get("exists")):
            return "provider_report_missing"
        if bool(provider_state.get("parse_error")):
            return "provider_report_parse_error"
        return "diff_fail"
    if reason in {
        "runtime_cuda_unknown",
        "runtime_rvv_unknown",
        "runtime_backend_fail",
        "runtime_cuda_fail",
        "runtime_rvv_fail",
    }:
        if rvv_reason_code == "runtime_timeout" or cuda_reason_code == "runtime_timeout":
            return "runtime_timeout"
        if rvv_reason_code == "env_unavailable" or cuda_reason_code == "env_unavailable":
            return "env_unavailable"
        if rvv_reason_code == "lowering_missing_op" or cuda_reason_code == "lowering_missing_op":
            return "lowering_missing_op"
        if rvv_reason_code == "diff_fail" or cuda_reason_code == "diff_fail":
            return "diff_fail"
    if reason == "runtime_cuda_skip" or reason == "runtime_rvv_skip":
        return "env_unavailable"
    return reason


def _entry_evidence_hash(*, kernel: str | None, provider_state: dict[str, Any], rvv_state: str, cuda_state: str, reason_code: str) -> str:
    payload = {
        "kernel": (kernel or ""),
        "provider_exists": bool(provider_state.get("exists")),
        "provider_diff_ok": bool(provider_state.get("diff_ok")),
        "rvv_state": str(rvv_state),
        "cuda_state": str(cuda_state),
        "reason_code": str(reason_code),
    }
    txt = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=(ROOT / "pipeline" / "triton" / "flaggems_registry.json"))
    ap.add_argument("--provider-report-dir", type=Path, default=(ROOT / "artifacts" / "flaggems_triton_full_pipeline"))
    ap.add_argument("--rvv-json", type=Path, default=None, help="RVV smoke JSON summary path")
    ap.add_argument("--cuda-json", type=Path, default=None, help="CUDA smoke JSON summary path")
    ap.add_argument(
        "--scope-kernels",
        action="append",
        default=[],
        help="Kernel name(s) to scope convergence on. Accepts repeated args or comma-separated values.",
    )
    ap.add_argument(
        "--scope-semantic-ops",
        action="append",
        default=[],
        help="Semantic op name(s) to scope convergence on. Accepts repeated args or comma-separated values.",
    )
    ap.add_argument(
        "--scope-mode",
        choices=["active_only", "kernel_alias", "both"],
        default="active_only",
        help=(
            "How scoped entries are selected: "
            "active_only=only --scope-semantic-ops, "
            "kernel_alias=kernel OR semantic-op union, "
            "both=emit both sets while legacy scoped fields follow active_only."
        ),
    )
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_coverage" / "status_converged.json"))
    ap.add_argument("--write-registry", action="store_true", help="overwrite registry with converged statuses")
    args = ap.parse_args()

    if not args.registry.is_file():
        raise FileNotFoundError(f"registry not found: {args.registry}")
    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = [e for e in (reg.get("entries") or []) if isinstance(e, dict)]

    rvv_summary = _load_json(args.rvv_json)
    cuda_summary = _load_json(args.cuda_json)
    scope_kernels = _parse_scope_values(args.scope_kernels)
    scope_semantic_ops = _parse_scope_values(args.scope_semantic_ops)
    scope_mode = str(args.scope_mode)
    scope_kernels_set = set(scope_kernels)
    scope_semantic_ops_set = set(scope_semantic_ops)
    scope_enabled = bool(scope_kernels_set or scope_semantic_ops_set)
    rvv_map, rvv_detail_map, rvv_skipped, rvv_skip_reason = _load_result_map(rvv_summary)
    cuda_map, cuda_detail_map, cuda_skipped, cuda_skip_reason = _load_result_map(cuda_summary)

    converged_entries: list[dict[str, Any]] = []
    counts_global: dict[str, int] = {}
    counts_scoped: dict[str, int] = {}
    counts_scoped_active: dict[str, int] = {}
    counts_scoped_kernel_alias: dict[str, int] = {}
    for e in entries:
        out = dict(e)
        spec = out.get("e2e_spec")
        kernel = str(spec) if isinstance(spec, str) and spec else None
        semantic_op = str(out.get("semantic_op") or "")

        provider_state = {"exists": False, "diff_ok": False}
        if kernel is not None:
            provider_state = _provider_report_state(args.provider_report_dir / f"{kernel}.json")

        def _runtime_state(result_map: dict[str, bool], *, skipped: bool) -> str:
            if kernel is None:
                return "unknown"
            if skipped:
                return "skip"
            if kernel not in result_map:
                return "unknown"
            return "pass" if bool(result_map[kernel]) else "fail"

        rvv_state = _runtime_state(rvv_map, skipped=rvv_skipped)
        cuda_state = _runtime_state(cuda_map, skipped=cuda_skipped)

        intent_ops = [str(x) for x in (out.get("intent_ops") or []) if isinstance(x, str)]
        status, reason = _derive_status(
            intent_ops=intent_ops,
            has_e2e_spec=(kernel is not None),
            provider_ok=bool(provider_state.get("diff_ok")),
            rvv=rvv_state,
            cuda=cuda_state,
        )
        out["status"] = status
        out["status_reason"] = reason
        rvv_detail = dict(rvv_detail_map.get(kernel, {})) if kernel is not None else {}
        cuda_detail = dict(cuda_detail_map.get(kernel, {})) if kernel is not None else {}
        rvv_reason_code, rvv_reason_detail = _classify_runtime_reason(
            rvv_state,
            rvv_detail,
            skip_reason=rvv_skip_reason,
        )
        cuda_reason_code, cuda_reason_detail = _classify_runtime_reason(
            cuda_state,
            cuda_detail,
            skip_reason=cuda_skip_reason,
        )
        reason_code = _derive_reason_code(
            status_reason=reason,
            provider_state=provider_state,
            rvv_reason_code=rvv_reason_code,
            cuda_reason_code=cuda_reason_code,
        )
        out["reason_code"] = reason_code
        out["runtime_detail"] = {
            "rvv": {"reason_code": rvv_reason_code, "reason_detail": rvv_reason_detail},
            "cuda": {"reason_code": cuda_reason_code, "reason_detail": cuda_reason_detail},
        }
        out["runtime"] = {
            "provider": provider_state,
            "rvv": rvv_state,
            "cuda": cuda_state,
        }
        provider_exists = bool(provider_state.get("exists"))
        rvv_has_result = rvv_state != "unknown"
        cuda_has_result = cuda_state != "unknown"
        artifact_complete = bool(provider_exists and (kernel is None or (rvv_has_result and cuda_has_result)))
        determinability = bool(str(reason_code).strip() not in {"", "unknown"})
        out["compiler_stage"] = {
            "provider_report": "present" if provider_exists else "missing",
            "rvv_result": ("present" if rvv_has_result else "missing"),
            "cuda_result": ("present" if cuda_has_result else "missing"),
        }
        out["artifact_complete"] = bool(artifact_complete)
        out["determinability"] = bool(determinability)
        out["evidence_hash"] = _entry_evidence_hash(
            kernel=kernel,
            provider_state=provider_state,
            rvv_state=rvv_state,
            cuda_state=cuda_state,
            reason_code=reason_code,
        )
        if not scope_enabled:
            in_scope_active = True
            in_scope_kernel_alias = True
        else:
            in_scope_active = bool(semantic_op and semantic_op in scope_semantic_ops_set)
            in_scope_kernel_alias = bool(
                (kernel is not None and kernel in scope_kernels_set)
                or (semantic_op and semantic_op in scope_semantic_ops_set)
            )
        if scope_mode == "kernel_alias":
            in_scope = in_scope_kernel_alias
        else:
            # active_only and both share active-only legacy scoped fields.
            in_scope = in_scope_active
        out["in_scope"] = bool(in_scope)
        out["in_scope_active"] = bool(in_scope_active)
        out["in_scope_kernel_alias"] = bool(in_scope_kernel_alias)
        converged_entries.append(out)
        counts_global[status] = counts_global.get(status, 0) + 1
        if in_scope:
            counts_scoped[status] = counts_scoped.get(status, 0) + 1
        if in_scope_active:
            counts_scoped_active[status] = counts_scoped_active.get(status, 0) + 1
        if in_scope_kernel_alias:
            counts_scoped_kernel_alias[status] = counts_scoped_kernel_alias.get(status, 0) + 1

    scoped_entries = [e for e in converged_entries if bool(e.get("in_scope"))]
    scoped_entries_active = [e for e in converged_entries if bool(e.get("in_scope_active"))]
    scoped_entries_kernel_alias = [e for e in converged_entries if bool(e.get("in_scope_kernel_alias"))]
    determinability_ok_count = sum(1 for e in converged_entries if bool(e.get("determinability")))
    artifact_complete_count = sum(1 for e in converged_entries if bool(e.get("artifact_complete")))

    result = {
        "schema_version": "flaggems_registry_converged_v3",
        "registry_path": str(args.registry),
        "provider_report_dir": str(args.provider_report_dir),
        "rvv_json": (str(args.rvv_json) if args.rvv_json else None),
        "cuda_json": (str(args.cuda_json) if args.cuda_json else None),
        # Backward-compat: counts_by_status remains global status count.
        "counts_by_status": dict(sorted(counts_global.items(), key=lambda kv: kv[0])),
        "counts_global": dict(sorted(counts_global.items(), key=lambda kv: kv[0])),
        "counts_scoped": dict(sorted(counts_scoped.items(), key=lambda kv: kv[0])),
        "counts_scoped_active": dict(sorted(counts_scoped_active.items(), key=lambda kv: kv[0])),
        "counts_scoped_kernel_alias": dict(sorted(counts_scoped_kernel_alias.items(), key=lambda kv: kv[0])),
        "global_entries_count": len(converged_entries),
        "scoped_entries_count": len(scoped_entries),
        "scoped_entries_active_count": len(scoped_entries_active),
        "scoped_entries_kernel_alias_count": len(scoped_entries_kernel_alias),
        "scope_enabled": bool(scope_enabled),
        "scope_mode": scope_mode,
        "scope_kernels": list(scope_kernels),
        "scope_semantic_ops": list(scope_semantic_ops),
        "determinability_ok_count": int(determinability_ok_count),
        "artifact_complete_count": int(artifact_complete_count),
        "determinability_ratio": (float(determinability_ok_count) / float(len(converged_entries)) if converged_entries else 0.0),
        "artifact_complete_ratio": (float(artifact_complete_count) / float(len(converged_entries)) if converged_entries else 0.0),
        "entries": converged_entries,
        "scoped_entries": scoped_entries,
        "scoped_entries_active": scoped_entries_active,
        "scoped_entries_kernel_alias": scoped_entries_kernel_alias,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Converged status report written: {args.out}")

    if args.write_registry:
        reg2 = dict(reg)
        if scope_enabled:
            # Scoped convergence must not overwrite out-of-scope registry entries.
            merged_entries: list[dict[str, Any]] = []
            for idx, conv in enumerate(converged_entries):
                if bool(conv.get("in_scope")):
                    merged_entries.append(conv)
                else:
                    if idx < len(entries):
                        merged_entries.append(dict(entries[idx]))
                    else:
                        merged_entries.append(conv)
        else:
            merged_entries = converged_entries

        merged_counts: dict[str, int] = {}
        for row in merged_entries:
            status = str(row.get("status") or "").strip()
            if not status:
                continue
            merged_counts[status] = merged_counts.get(status, 0) + 1

        reg2["entries"] = merged_entries
        reg2["counts"] = dict(reg2.get("counts") or {})
        reg2["counts"]["by_status"] = dict(sorted(merged_counts.items(), key=lambda kv: kv[0]))
        args.registry.write_text(json.dumps(reg2, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Registry updated: {args.registry}")


if __name__ == "__main__":
    main()
