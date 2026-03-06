from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pipeline.common.tuning_db import load_tuning_db_jsonl, resolve_tuning_db_path, resolve_tuning_entries


def _normalize_oracle_bindings(*, kernel: str, kernel_kind: str, bindings: Mapping[str, int]) -> dict[str, int]:
    out = {str(k): int(v) for k, v in dict(bindings or {}).items() if str(k).strip()}
    kernel_s = str(kernel or "").strip()
    kind_s = str(kernel_kind or "").strip()
    if kernel_s == "flash_attention2d" and kind_s == "attn2d_causal_softmax_v6":
        out.setdefault("ATTN_SCORE_WARPS", 6)
    if kernel_s == "matmul_fused_epilogue2d" and kind_s == "matmul_mma_tf32_v1":
        out.setdefault("MMA_BM", 32)
        out.setdefault("MMA_BN", 32)
        out.setdefault("MMA_BK", 32)
    return out


def _infer_source_arch(
    *,
    db: Mapping[tuple[str, str], object],
    kernel: str,
    compiler_stack: str,
    target_arch: str,
    shape_bindings: Mapping[str, int],
) -> str:
    def _arch_rank(value: str) -> tuple[int, str]:
        arch_s = str(value or "").strip().lower()
        if arch_s.startswith("sm"):
            try:
                return (int(arch_s[2:]), arch_s)
            except Exception:
                return (-1, arch_s)
        return (-1, arch_s)

    kernel_s = str(kernel or "").strip()
    compiler_stack_s = str(compiler_stack or "").strip().lower()
    target_arch_s = str(target_arch or "").strip()
    candidates: list[str] = []
    for (entry_kernel, entry_arch), entries in dict(db or {}).items():
        if str(entry_kernel) != kernel_s:
            continue
        resolved, kernel_kind = resolve_tuning_entries(
            list(entries or []),
            shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
            compiler_stack=str(compiler_stack_s),
        )
        if not str(kernel_kind or "").strip() and not dict(resolved or {}):
            continue
        arch_s = str(entry_arch or "").strip()
        if arch_s:
            candidates.append(arch_s)
    uniq = sorted(set(candidates))
    if target_arch_s:
        non_target = [arch for arch in uniq if arch != target_arch_s]
        if len(non_target) == 1:
            return str(non_target[0])
        if non_target:
            return str(sorted(non_target, key=_arch_rank, reverse=True)[0])
    if len(uniq) == 1:
        return str(uniq[0])
    return ""


def extract_source_oracle_facts(
    *,
    kernel: str,
    source_arch: str,
    target_arch: str = "",
    shape_bindings: Mapping[str, int],
    compiler_stack: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    source_arch_s = str(source_arch or "").strip()
    target_arch_s = str(target_arch or "").strip()
    compiler_stack_s = str(compiler_stack or "").strip().lower()
    db_file = resolve_tuning_db_path(path=(Path(db_path) if db_path else None), backend="cuda")
    evidence: list[dict[str, Any]] = []
    oracle = {
        "kernel_kind": "",
        "bindings": {},
        "arch": source_arch_s,
        "compiler_stack": compiler_stack_s,
        "evidence_refs": [],
    }

    if db_file is None or not Path(db_file).is_file():
        return {
            "schema_version": "org_source_oracle_facts_v1",
            "available": False,
            "source": {
                "kernel": str(kernel),
                "arch": source_arch_s,
                "compiler_stack": compiler_stack_s,
                "db_path": (str(db_file) if db_file is not None else None),
            },
            "oracle": oracle,
            "evidence": evidence,
        }

    db = load_tuning_db_jsonl(path=Path(db_file), backend="cuda")
    if not source_arch_s:
        source_arch_s = _infer_source_arch(
            db=db,
            kernel=str(kernel),
            compiler_stack=str(compiler_stack_s),
            target_arch=str(target_arch_s),
            shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        )
    if not source_arch_s:
        return {
            "schema_version": "org_source_oracle_facts_v1",
            "available": False,
            "source": {
                "kernel": str(kernel),
                "arch": "",
                "compiler_stack": compiler_stack_s,
                "db_path": str(db_file),
            },
            "oracle": {
                "kernel_kind": "",
                "bindings": {},
                "arch": "",
                "compiler_stack": compiler_stack_s,
                "evidence_refs": [],
            },
            "evidence": evidence,
        }
    entries = db.get((str(kernel), str(source_arch_s))) or []
    merged, kernel_kind = resolve_tuning_entries(
        entries,
        shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        compiler_stack=str(compiler_stack_s),
    )
    merged_bindings = _normalize_oracle_bindings(
        kernel=str(kernel),
        kernel_kind=str(kernel_kind or ""),
        bindings={str(k): int(v) for k, v in dict(merged or {}).items() if str(k).strip()},
    )
    kernel_kind_s = str(kernel_kind or "").strip()
    if kernel_kind_s or merged_bindings:
        evidence.append(
            {
                "id": "source_oracle_db",
                "kind": "tuning_db",
                "path": str(db_file),
                "summary": f"source oracle for {kernel}@{source_arch_s}",
            }
        )
        oracle = {
            "kernel_kind": kernel_kind_s,
            "bindings": merged_bindings,
            "arch": source_arch_s,
            "compiler_stack": compiler_stack_s,
            "evidence_refs": ["source_oracle_db"],
        }

    return {
        "schema_version": "org_source_oracle_facts_v1",
        "available": bool(kernel_kind_s or merged_bindings),
        "source": {
            "kernel": str(kernel),
            "arch": source_arch_s,
            "compiler_stack": compiler_stack_s,
            "db_path": str(db_file),
        },
        "oracle": oracle,
        "evidence": evidence,
    }


__all__ = ["extract_source_oracle_facts"]
