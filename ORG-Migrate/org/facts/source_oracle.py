from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pipeline.common.tuning_db import load_tuning_db_jsonl, resolve_tuning_db_path, resolve_tuning_entries


def extract_source_oracle_facts(
    *,
    kernel: str,
    source_arch: str,
    shape_bindings: Mapping[str, int],
    compiler_stack: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    source_arch_s = str(source_arch or "").strip()
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

    if db_file is None or not Path(db_file).is_file() or not source_arch_s:
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
    entries = db.get((str(kernel), str(source_arch_s))) or []
    merged, kernel_kind = resolve_tuning_entries(
        entries,
        shape_bindings={str(k): int(v) for k, v in dict(shape_bindings or {}).items()},
        compiler_stack=str(compiler_stack_s),
    )
    merged_bindings = {str(k): int(v) for k, v in dict(merged or {}).items() if str(k).strip()}
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
