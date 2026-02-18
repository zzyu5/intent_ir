from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .module import IntentMLIRModule
from .passes import PASS_REGISTRY
from .toolchain import detect_mlir_toolchain
from .convert_to_intent import to_intent

ROOT = Path(__file__).resolve().parents[2]
PIPELINES_DIR = Path(__file__).resolve().parent / "pipelines"


@dataclass
class PassRecord:
    name: str
    kind: str
    ok: bool
    ms: float
    detail: str
    before_path: str = ""
    after_path: str = ""
    before_stats: dict[str, Any] | None = None
    after_stats: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": bool(self.ok),
            "ms": float(self.ms),
            "detail": str(self.detail),
            "before_path": str(self.before_path),
            "after_path": str(self.after_path),
            "before_stats": dict(self.before_stats or {}),
            "after_stats": dict(self.after_stats or {}),
        }


def run_pipeline(
    module: IntentMLIRModule,
    pipeline_name: str,
    *,
    backend: str | None = None,
    out_dir: Path | None = None,
    fail_on_error: bool = False,
) -> tuple[IntentMLIRModule, dict[str, Any]]:
    pipeline_spec = _load_pipeline_spec(str(pipeline_name), backend=backend)
    trace: list[PassRecord] = []
    current = module
    toolchain = detect_mlir_toolchain()

    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    for idx, pass_name in enumerate(list(pipeline_spec.get("passes") or [])):
        t0 = time.perf_counter()
        before_path = ""
        after_path = ""
        before_stats = _module_stats(current)
        try:
            if out is not None:
                before_path = str(out / f"pass_{idx:03d}_{_safe(pass_name)}.before.mlir")
                Path(before_path).write_text(current.module_text, encoding="utf-8")
            current = _run_one_pass(current, pass_name, backend=backend, toolchain=toolchain)
            after_stats = _module_stats(current)
            if out is not None:
                after_path = str(out / f"pass_{idx:03d}_{_safe(pass_name)}.after.mlir")
                Path(after_path).write_text(current.module_text, encoding="utf-8")
            dt = float((time.perf_counter() - t0) * 1000.0)
            trace.append(
                PassRecord(
                    name=str(pass_name),
                    kind=("mlir-opt" if str(pass_name).startswith("mlir-opt:") else "python"),
                    ok=True,
                    ms=dt,
                    detail="ok",
                    before_path=before_path,
                    after_path=after_path,
                    before_stats=before_stats,
                    after_stats=after_stats,
                )
            )
        except Exception as e:
            dt = float((time.perf_counter() - t0) * 1000.0)
            after_stats = _module_stats(current)
            trace.append(
                PassRecord(
                    name=str(pass_name),
                    kind=("mlir-opt" if str(pass_name).startswith("mlir-opt:") else "python"),
                    ok=False,
                    ms=dt,
                    detail=f"{type(e).__name__}: {e}",
                    before_path=before_path,
                    after_path=after_path,
                    before_stats=before_stats,
                    after_stats=after_stats,
                )
            )
            if fail_on_error:
                raise
            break

    payload = {
        "schema_version": "intent_mlir_pass_trace_v1",
        "pipeline": str(pipeline_name),
        "backend": (str(backend) if backend else None),
        "toolchain": toolchain,
        "passes": [x.to_json_dict() for x in trace],
        "ok": bool(all(x.ok for x in trace)),
        "input_stats": _module_stats(module),
        "output_stats": _module_stats(current),
    }
    if out is not None:
        trace_path = out / "pass_trace.json"
        trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["pass_trace_path"] = str(trace_path)
    return current, payload


def _run_one_pass(
    module: IntentMLIRModule,
    pass_name: str,
    *,
    backend: str | None,
    toolchain: dict[str, Any],
) -> IntentMLIRModule:
    name = str(pass_name).strip()
    if not name:
        return module
    if name.startswith("python:"):
        key = name.split(":", 1)[1].strip()
        fn = PASS_REGISTRY.get(key)
        if fn is None:
            raise ValueError(f"unknown python pass: {key}")
        return fn(module, backend=backend)
    if name.startswith("mlir-opt:"):
        pass_arg = name.split(":", 1)[1].strip()
        return _run_mlir_opt_pass(module, pass_arg=pass_arg, toolchain=toolchain)
    raise ValueError(f"unsupported pass selector: {name}")


def _run_mlir_opt_pass(module: IntentMLIRModule, *, pass_arg: str, toolchain: dict[str, Any]) -> IntentMLIRModule:
    tool = (((toolchain.get("tools") or {}).get("mlir-opt") or {}).get("path") or "").strip()
    if not tool:
        raise RuntimeError("mlir-opt unavailable")
    p = subprocess.run(
        [tool, f"--{pass_arg}"],
        input=str(module.module_text),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"mlir-opt failed: {p.stderr or p.stdout}")
    out = IntentMLIRModule(
        module_text=str(p.stdout or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["last_mlir_opt_pass"] = str(pass_arg)
    return out


def _load_pipeline_spec(name: str, *, backend: str | None = None) -> dict[str, Any]:
    base = str(name or "").strip()
    if not base:
        raise ValueError("pipeline name is empty")
    filename = base if base.endswith(".yaml") else f"{base}.yaml"
    path = PIPELINES_DIR / filename
    if backend and base.startswith("downstream") and not path.exists():
        path = PIPELINES_DIR / f"downstream_{backend}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"pipeline spec not found: {path}")
    raw = path.read_text(encoding="utf-8")
    passes = _parse_yaml_pass_list(raw)
    return {"name": base, "path": str(path), "passes": passes}


def _parse_yaml_pass_list(text: str) -> list[str]:
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if isinstance(payload, dict):
            rows = payload.get("passes")
            if isinstance(rows, list):
                return [str(x).strip() for x in rows if str(x).strip()]
    except Exception:
        pass
    out: list[str] = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("-"):
            item = s[1:].strip()
            if item:
                out.append(item)
    return out


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _module_stats(module: IntentMLIRModule) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "symbols": int(len(list(module.symbols or []))),
        "dialect_version": str(module.dialect_version),
    }
    try:
        intent = to_intent(module)
        stats.update(
            {
                "intent_name": str(intent.name),
                "ops": int(len(list(intent.ops or []))),
                "tensors": int(len(dict(intent.tensors or {}))),
                "outputs": int(len(list(intent.outputs or []))),
            }
        )
    except Exception as e:
        stats["intent_decode_error"] = f"{type(e).__name__}: {e}"
    return stats
