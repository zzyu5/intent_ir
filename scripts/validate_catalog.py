"""
Validate scripts catalog governance for FlagGems long-task workflow.

Rules enforced:
1) every catalog entry has required fields and points to an existing path.
2) active/deprecated status is explicit; deprecated entries require replacement.
3) key workflow/docs references only point to active catalog entries.
4) active operational scripts under scripts/flaggems + scripts/intentir are cataloged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SCRIPT_REF_RE = re.compile(r"(scripts/[A-Za-z0-9_./-]+\.(?:py|sh))")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_script_refs(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    refs = set(m.group(1) for m in SCRIPT_REF_RE.finditer(text))
    return set(sorted(refs))


def _append_err(errors: list[str], msg: str) -> None:
    errors.append(str(msg))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, default=(ROOT / "scripts" / "CATALOG.json"))
    ap.add_argument("--out", type=Path, default=(ROOT / "artifacts" / "flaggems_matrix" / "catalog_validation.json"))
    args = ap.parse_args()

    if not args.catalog.is_file():
        raise FileNotFoundError(f"catalog not found: {args.catalog}")

    payload = _load_json(args.catalog)
    entries = [e for e in list(payload.get("entries") or []) if isinstance(e, dict)]
    errors: list[str] = []
    warnings: list[str] = []
    required_fields = {
        "path",
        "owner_lane",
        "purpose",
        "inputs",
        "outputs",
        "status",
        "replacement",
        "tests",
    }
    by_path: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(entries):
        missing = [k for k in sorted(required_fields) if k not in row]
        if missing:
            _append_err(errors, f"catalog entry[{idx}] missing fields: {missing}")
            continue
        path = str(row.get("path") or "").strip()
        if not path:
            _append_err(errors, f"catalog entry[{idx}] has empty path")
            continue
        if path in by_path:
            _append_err(errors, f"duplicate catalog path: {path}")
            continue
        status = str(row.get("status") or "").strip()
        if status not in {"active", "deprecated"}:
            _append_err(errors, f"{path}: invalid status={status}")
        repl = str(row.get("replacement") or "").strip()
        if status == "deprecated" and not repl:
            _append_err(errors, f"{path}: deprecated entry must set replacement")
        full = ROOT / path
        if not full.exists():
            _append_err(errors, f"{path}: path does not exist")
        by_path[path] = row

    # Ensure operational script directories are cataloged to avoid duplicate entrypoints.
    must_catalog_globs = [
        "scripts/*.py",
        "scripts/flaggems/*.py",
        "scripts/intentir/*.py",
        "scripts/triton/*.py",
    ]
    for pattern in must_catalog_globs:
        for full in sorted(ROOT.glob(pattern)):
            rel = _repo_rel(full)
            if rel not in by_path:
                _append_err(errors, f"missing catalog entry for operational script: {rel}")

    # Key workflow/docs may only reference active scripts.
    referenced_from = [
        ROOT / "workflow" / "flaggems" / "README.md",
        ROOT / "workflow" / "flaggems" / "init.sh",
        ROOT / "workflow" / "flaggems" / "nightly.sh",
        ROOT / "scripts" / "README.md",
    ]
    refs: dict[str, list[str]] = {}
    for source in referenced_from:
        for ref in sorted(_collect_script_refs(source)):
            refs.setdefault(ref, []).append(_repo_rel(source))

    for ref, sources in sorted(refs.items(), key=lambda kv: kv[0]):
        row = by_path.get(ref)
        if row is None:
            _append_err(errors, f"{ref}: referenced by {sources} but missing in catalog")
            continue
        if str(row.get("status") or "") != "active":
            _append_err(errors, f"{ref}: referenced by {sources} but catalog status is not active")

    # Ensure one active entry per (owner_lane, purpose) to prevent duplicate script creation.
    active_by_purpose: dict[tuple[str, str], str] = {}
    for path, row in by_path.items():
        if str(row.get("status") or "") != "active":
            continue
        key = (str(row.get("owner_lane") or "").strip(), str(row.get("purpose") or "").strip())
        if key in active_by_purpose:
            prev = active_by_purpose[key]
            _append_err(
                errors,
                f"duplicate active purpose for lane {key[0]!r}: {prev} and {path} share purpose {key[1]!r}",
            )
        else:
            active_by_purpose[key] = path

    if not entries:
        warnings.append("catalog has no entries")

    ok = len(errors) == 0
    report = {
        "ok": bool(ok),
        "catalog": _repo_rel(args.catalog),
        "entry_count": len(entries),
        "active_count": sum(1 for e in entries if str(e.get("status") or "") == "active"),
        "deprecated_count": sum(1 for e in entries if str(e.get("status") or "") == "deprecated"),
        "errors": errors,
        "warnings": warnings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Catalog validation report written: {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
