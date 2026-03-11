from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIGER_OPS_ROOT = ROOT.parent / "Liger-Kernel" / "src" / "liger_kernel" / "ops"
DEFAULT_OUT = ROOT / "artifacts" / "liger_true_migration" / "all_liger_sweep"
TARGET_LABEL = "5090D(sm120)"
SOURCE_LABEL = "H100(sm90)"
SUCCESS_TOL = 1.0e-5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.triton.providers.liger.specs import liger_kernel_specs  # noqa: E402


def _discover_liger_ops() -> list[str]:
    stems: list[str] = []
    for path in sorted(LIGER_OPS_ROOT.glob("*.py")):
        stem = path.stem.strip()
        if not stem or stem in {"__init__", "utils"}:
            continue
        stems.append(stem)
    return stems


def _round_like_base(value: int, base: int) -> int:
    base_i = max(1, int(base))
    value_i = max(1, int(value))
    for align in (256, 128, 64, 32, 16, 8, 4, 2):
        if base_i % align == 0:
            return max(align, int(math.ceil(value_i / align) * align))
    return value_i


def _round_tiny_like_base(value: int, base: int) -> int:
    base_i = max(1, int(base))
    value_i = max(1, int(value))
    candidates = [align for align in (256, 128, 64, 32, 16, 8, 4, 2) if base_i % align == 0 and align <= value_i]
    if candidates:
        align = max(candidates)
        return max(align, int(value_i // align) * align)
    return value_i


def _tiny_dim_value(key: str, base: int) -> int:
    if key in {"M", "N", "BT", "V", "S", "Q_CTX", "KV_CTX", "HW"}:
        target = min(int(base), 64 if int(base) >= 64 else int(base))
        return _round_tiny_like_base(max(1, target), int(base))
    if key in {"C", "K"}:
        target = min(int(base), 64 if int(base) >= 64 else int(base))
        return _round_tiny_like_base(max(1, target), int(base))
    if key in {"QH", "KH", "HEAD_NUM", "BATCH_NUM", "num_groups"}:
        return _round_tiny_like_base(max(1, min(int(base), 4)), int(base))
    if key in {"HD", "HEAD_DIM"}:
        target = min(int(base), 64 if int(base) >= 64 else int(base))
        return _round_tiny_like_base(max(1, target), int(base))
    if key in {"B", "N"}:
        return max(1, min(int(base), 2))
    return max(1, min(int(base), 8))


def _massive_dim_value(key: str, base: int) -> int:
    base_i = int(base)
    if key == "V":
        target = min(max(base_i * 512, 4096), 2097152)
        return _round_like_base(target, base_i)
    if key == "N":
        target = min(max(base_i * 4, base_i), 65536)
        return _round_like_base(target, base_i)
    if key in {"M", "BT"}:
        target = min(max(base_i * 4, base_i), 2097152)
        return _round_like_base(target, base_i)
    if key in {"S", "Q_CTX", "KV_CTX", "HW"}:
        target = min(max(base_i * 4, base_i), 4096)
        return _round_like_base(target, base_i)
    if key in {"C", "K"}:
        target = min(max(base_i * 4, base_i), 4096)
        return _round_like_base(target, base_i)
    if key in {"QH", "KH", "HEAD_NUM", "BATCH_NUM", "num_groups"}:
        target = min(max(base_i * 2, base_i), 64)
        return _round_like_base(target, base_i)
    if key in {"HD", "HEAD_DIM"}:
        target = min(max(base_i * 2, base_i), 256)
        return _round_like_base(target, base_i)
    return base_i


def _shape_variant_bindings(spec, label: str) -> dict[str, int]:
    base = {str(k): int(v) for k, v in dict(spec.canonical_shapes).items()}
    mutate_keys = [str(k) for k in list(spec.vary_axes or []) if str(k) not in set(spec.exclude_axes or [])]
    out = dict(base)
    if label == "Normal":
        return spec.normalize_shapes(out) if spec.normalize_shapes else out
    dominant_key = max(mutate_keys, key=lambda k: int(base.get(k, 1))) if mutate_keys else ""
    for key in mutate_keys:
        base_v = int(base.get(key, 1))
        if label == "Tiny":
            out[key] = int(_tiny_dim_value(key, base_v))
        elif label == "Massive":
            if key == dominant_key:
                out[key] = int(_massive_dim_value(key, base_v))
            elif key == mutate_keys[0]:
                out[key] = int(_tiny_dim_value(key, base_v))
            else:
                out[key] = int(base_v)
    return spec.normalize_shapes(out) if spec.normalize_shapes else out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shape_key(kernel: str, bindings: dict[str, int]) -> tuple[str, tuple[tuple[str, int], ...]]:
    return (str(kernel), tuple(sorted((str(k), int(v)) for k, v in dict(bindings).items())))


def _format_correctness(max_abs: dict[str, Any]) -> str:
    if not max_abs:
        return ""
    parts = [f"{k}={float(v):.3e}" for k, v in sorted(max_abs.items())]
    return ", ".join(parts)


def _run_one_case(
    *,
    kernel: str,
    label: str,
    bindings: dict[str, int],
    out_dir: Path,
    warmup: int,
    iters: int,
    repeats: int,
    timeout_s: int,
    remote_ssh: str,
) -> dict[str, Any]:
    run_dir = out_dir / kernel / label.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "liger_true_migration.py"),
        "--kernels",
        kernel,
        "--out",
        str(run_dir),
        "--warmup",
        str(int(warmup)),
        "--iters",
        str(int(iters)),
        "--repeats",
        str(int(repeats)),
        "--bindings-json",
        json.dumps(bindings, ensure_ascii=False),
    ]
    env = dict(os.environ)
    env.setdefault("INTENTIR_ORG_REMOTE_SOURCE_ENABLE", "1")
    env.setdefault("INTENTIR_ORG_MODE", "apply")
    env.setdefault("INTENTIR_REAL_MLIR", "1")
    env.setdefault("INTENTIR_CUDA_REAL_MLIR_ALLOW_UNKNOWN", "1")
    env.setdefault("INTENTIR_ORG_SEED_POLICY", "force_llm")
    env["INTENTIR_ORG_REMOTE_SOURCE_SSH"] = str(remote_ssh)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "kernel": kernel,
            "shape_label": label,
            "shape_bindings": dict(bindings),
            "source": SOURCE_LABEL,
            "target": TARGET_LABEL,
            "native_qps": 0.0,
            "guided_qps": 0.0,
            "ratio": 0.0,
            "correctness": "",
            "failure_reason": f"LLM_Timeout: {exc}",
            "report_path": "",
            "stdout_path": "",
            "stderr_path": "",
        }
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    stdout_path.write_text(str(proc.stdout or ""), encoding="utf-8")
    stderr_path.write_text(str(proc.stderr or ""), encoding="utf-8")
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        reason = f"Pipeline_Fail(returncode={proc.returncode})"
        if str(proc.stderr or "").strip():
            reason = f"{reason}: {str(proc.stderr).strip().splitlines()[-1]}"
        return {
            "kernel": kernel,
            "shape_label": label,
            "shape_bindings": dict(bindings),
            "source": SOURCE_LABEL,
            "target": TARGET_LABEL,
            "native_qps": 0.0,
            "guided_qps": 0.0,
            "ratio": 0.0,
            "correctness": "",
            "failure_reason": reason,
            "report_path": "",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    row = dict((_load_json(summary_path).get("rows") or [{}])[0])
    guided_status = row.get("guided_status") if isinstance(row.get("guided_status"), dict) else {}
    max_abs = row.get("max_abs") if isinstance(row.get("max_abs"), dict) else {}
    ok = bool(guided_status.get("ok")) and bool(max_abs) and max(float(v) for v in max_abs.values()) <= SUCCESS_TOL
    failure_reason = ""
    if not ok:
        failure_reason = str(guided_status.get("error") or row.get("error") or "Guided_Invalid")
    remote = row.get("remote_source") if isinstance(row.get("remote_source"), dict) else {}
    source_arch = str(remote.get("source_arch") or remote.get("arch") or SOURCE_LABEL).strip() or SOURCE_LABEL
    return {
        "kernel": kernel,
        "shape_label": label,
        "shape_bindings": dict(bindings),
        "source": source_arch,
        "target": TARGET_LABEL,
        "native_qps": float(row.get("native_qps") or 0.0),
        "guided_qps": float(row.get("guided_qps") or 0.0),
        "ratio": float(row.get("ratio") or 0.0),
        "correctness": _format_correctness(max_abs) if ok else "",
        "failure_reason": failure_reason,
        "report_path": str(row.get("report_path") or ""),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "guided_candidate": dict((row.get("guided_candidate") or {})),
    }


def _cached_row_to_master(row: dict[str, Any], *, label: str) -> dict[str, Any]:
    guided_status = row.get("guided_status") if isinstance(row.get("guided_status"), dict) else {}
    max_abs = row.get("max_abs") if isinstance(row.get("max_abs"), dict) else {}
    ok = bool(guided_status.get("ok")) and bool(max_abs) and max(float(v) for v in max_abs.values()) <= SUCCESS_TOL
    failure_reason = ""
    if not ok:
        failure_reason = str(guided_status.get("error") or row.get("error") or "Guided_Invalid")
    remote = row.get("remote_source") if isinstance(row.get("remote_source"), dict) else {}
    source_arch = str(remote.get("source_arch") or remote.get("arch") or SOURCE_LABEL).strip() or SOURCE_LABEL
    return {
        "kernel": str(row.get("kernel") or ""),
        "shape_label": label,
        "shape_bindings": dict(row.get("shapes") or {}),
        "source": source_arch,
        "target": TARGET_LABEL,
        "native_qps": float(row.get("native_qps") or 0.0),
        "guided_qps": float(row.get("guided_qps") or 0.0),
        "ratio": float(row.get("ratio") or 0.0),
        "correctness": _format_correctness(max_abs) if ok else "",
        "failure_reason": failure_reason,
        "report_path": str(row.get("report_path") or ""),
        "stdout_path": "",
        "stderr_path": "",
        "guided_candidate": dict((row.get("guided_candidate") or {})),
        "cache_hit": True,
    }


def _build_existing_result_index(*, search_root: Path, exclude_root: Path) -> dict[tuple[str, tuple[tuple[str, int], ...]], dict[str, Any]]:
    index: dict[tuple[str, tuple[tuple[str, int], ...]], tuple[float, dict[str, Any]]] = {}
    for path in search_root.rglob("summary.json"):
        try:
            if exclude_root in path.parents:
                continue
            payload = _load_json(path)
            rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                kernel = str(row.get("kernel") or "").strip()
                shapes = row.get("shapes") if isinstance(row.get("shapes"), dict) else {}
                if not kernel or not shapes:
                    continue
                key = _shape_key(kernel, {str(k): int(v) for k, v in shapes.items()})
                score = float(path.stat().st_mtime)
                prev = index.get(key)
                if prev is None or score >= float(prev[0]):
                    index[key] = (score, dict(row))
        except Exception:
            continue
    return {k: v for k, (_score, v) in index.items()}


def _unsupported_rows(kernel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in ("Tiny", "Normal", "Massive"):
        rows.append(
            {
                "kernel": kernel,
                "shape_label": label,
                "shape_bindings": {},
                "source": "N/A",
                "target": TARGET_LABEL,
                "native_qps": 0.0,
                "guided_qps": 0.0,
                "ratio": 0.0,
                "correctness": "",
                "failure_reason": "UNSUPPORTED_PROVIDER_SPEC",
                "report_path": "",
                "stdout_path": "",
                "stderr_path": "",
            }
        )
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Kernel Name | Shape | H100 Source | Target (5090D) | Guided QPS | Native QPS | Ratio | Correctness / Failure Reason |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        correctness = str(row.get("correctness") or "").strip()
        failure = str(row.get("failure_reason") or "").strip()
        status = correctness if correctness else failure
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("kernel") or ""),
                    str(row.get("shape_label") or ""),
                    str(row.get("source") or ""),
                    str(row.get("target") or ""),
                    f"{float(row.get('guided_qps') or 0.0):.4f}",
                    f"{float(row.get('native_qps') or 0.0):.4f}",
                    f"{float(row.get('ratio') or 0.0):.5f}x",
                    status.replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _load_existing_rows(out_root: Path) -> list[dict[str, Any]]:
    summary_path = out_root / "summary.json"
    if not summary_path.is_file():
        return []
    try:
        payload = _load_json(summary_path)
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        return [dict(r) for r in rows if isinstance(r, dict)]
    except Exception:
        return []


def _write_summary(out_root: Path, *, discovered_stems: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_rows = [r for r in rows if str(r.get("correctness") or "").strip()]
    parity_rows = [r for r in success_rows if float(r.get("ratio") or 0.0) >= 0.95]
    unique_success = sorted({str(r.get("kernel") or "") for r in success_rows})
    unique_parity = sorted({str(r.get("kernel") or "") for r in parity_rows})
    summary = {
        "discovered_ops": len(discovered_stems),
        "discovered_kernel_names": [f"liger_{x}" for x in discovered_stems],
        "successful_rows": len(success_rows),
        "successful_unique_kernels": len(unique_success),
        "parity_rows_ge_0_95x": len(parity_rows),
        "parity_unique_kernels_ge_0_95x": len(unique_parity),
        "rows": rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    matrix_md = _markdown_table(rows)
    (out_root / "ultimate_all_liger_matrix.md").write_text(matrix_md + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--remote-ssh", default=os.getenv("INTENTIR_ORG_REMOTE_SOURCE_SSH", "h100"))
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    spec_map = {str(spec.name): spec for spec in liger_kernel_specs()}
    discovered_stems = _discover_liger_ops()
    cached_index = _build_existing_result_index(search_root=ROOT / "artifacts" / "liger_true_migration", exclude_root=out_root)
    rows: list[dict[str, Any]] = _load_existing_rows(out_root)
    completed = {
        (str(r.get("kernel") or ""), str(r.get("shape_label") or ""))
        for r in rows
        if str(r.get("kernel") or "").strip() and str(r.get("shape_label") or "").strip()
    }

    for stem in discovered_stems:
        kernel = f"liger_{stem}"
        spec = spec_map.get(kernel)
        if spec is None:
            for row in _unsupported_rows(kernel):
                if (str(row.get("kernel") or ""), str(row.get("shape_label") or "")) in completed:
                    continue
                rows.append(row)
                completed.add((str(row.get("kernel") or ""), str(row.get("shape_label") or "")))
                print(json.dumps(row, ensure_ascii=False), flush=True)
            _write_summary(out_root, discovered_stems=discovered_stems, rows=rows)
            continue
        for label in ("Tiny", "Normal", "Massive"):
            if (kernel, label) in completed:
                continue
            bindings = _shape_variant_bindings(spec, label)
            cached = cached_index.get(_shape_key(kernel, bindings))
            if cached is not None:
                row = _cached_row_to_master(cached, label=label)
                rows.append(row)
                completed.add((kernel, label))
                print(json.dumps(row, ensure_ascii=False), flush=True)
                _write_summary(out_root, discovered_stems=discovered_stems, rows=rows)
                continue
            row = _run_one_case(
                kernel=kernel,
                label=label,
                bindings=bindings,
                out_dir=out_root,
                warmup=int(args.warmup),
                iters=int(args.iters),
                repeats=int(args.repeats),
                timeout_s=int(args.timeout),
                remote_ssh=str(args.remote_ssh),
            )
            rows.append(row)
            completed.add((kernel, label))
            print(json.dumps(row, ensure_ascii=False), flush=True)
            _write_summary(out_root, discovered_stems=discovered_stems, rows=rows)

    summary = _write_summary(out_root, discovered_stems=discovered_stems, rows=rows)
    matrix_md = _markdown_table(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    print(matrix_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
