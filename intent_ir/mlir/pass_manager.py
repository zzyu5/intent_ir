from __future__ import annotations

import json
import shlex
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


@dataclass
class PassExecutionResult:
    module: IntentMLIRModule
    kind: str
    detail: str


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
            exec_result = _run_one_pass(current, pass_name, backend=backend, toolchain=toolchain)
            current = exec_result.module
            after_stats = _module_stats(current)
            if out is not None:
                after_path = str(out / f"pass_{idx:03d}_{_safe(pass_name)}.after.mlir")
                Path(after_path).write_text(current.module_text, encoding="utf-8")
            dt = float((time.perf_counter() - t0) * 1000.0)
            trace.append(
                PassRecord(
                    name=str(pass_name),
                    kind=str(exec_result.kind),
                    ok=True,
                    ms=dt,
                    detail=str(exec_result.detail),
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
                    kind=_pass_kind(str(pass_name)),
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
) -> PassExecutionResult:
    name = str(pass_name).strip()
    if not name:
        return PassExecutionResult(module=module, kind="python", detail="skipped_empty")
    if name.startswith("python:"):
        key = name.split(":", 1)[1].strip()
        fn = PASS_REGISTRY.get(key)
        if fn is None:
            raise ValueError(f"unknown python pass: {key}")
        return PassExecutionResult(module=fn(module, backend=backend), kind="python", detail="ok")
    if name.startswith("mlir-opt?:"):
        pass_arg = name.split(":", 1)[1].strip()
        tool = _tool_path(toolchain, "mlir-opt")
        if not tool:
            return PassExecutionResult(
                module=module,
                kind="mlir-opt",
                detail="skipped_optional_tool_unavailable:mlir-opt",
            )
        try:
            return PassExecutionResult(
                module=_run_mlir_opt_pass(module, pass_arg=pass_arg, tool=tool),
                kind="mlir-opt",
                detail="ok",
            )
        except Exception as e:
            return PassExecutionResult(
                module=module,
                kind="mlir-opt",
                detail=f"skipped_optional_pass_failed:mlir-opt:{type(e).__name__}:{e}",
            )
    if name.startswith("mlir-opt:"):
        pass_arg = name.split(":", 1)[1].strip()
        tool = _tool_path(toolchain, "mlir-opt")
        if not tool:
            raise RuntimeError("mlir-opt unavailable")
        return PassExecutionResult(
            module=_run_mlir_opt_pass(module, pass_arg=pass_arg, tool=tool),
            kind="mlir-opt",
            detail="ok",
        )
    if name.startswith("mlir-translate?:"):
        pass_arg = name.split(":", 1)[1].strip()
        tool = _tool_path(toolchain, "mlir-translate")
        if not tool:
            return PassExecutionResult(
                module=module,
                kind="mlir-translate",
                detail="skipped_optional_tool_unavailable:mlir-translate",
            )
        try:
            return PassExecutionResult(
                module=_run_mlir_translate_pass(module, pass_arg=pass_arg, tool=tool),
                kind="mlir-translate",
                detail="ok",
            )
        except Exception as e:
            return PassExecutionResult(
                module=module,
                kind="mlir-translate",
                detail=f"skipped_optional_pass_failed:mlir-translate:{type(e).__name__}:{e}",
            )
    if name.startswith("mlir-translate:"):
        pass_arg = name.split(":", 1)[1].strip()
        tool = _tool_path(toolchain, "mlir-translate")
        if not tool:
            raise RuntimeError("mlir-translate unavailable")
        return PassExecutionResult(
            module=_run_mlir_translate_pass(module, pass_arg=pass_arg, tool=tool),
            kind="mlir-translate",
            detail="ok",
        )
    raise ValueError(f"unsupported pass selector: {name}")


def _tool_path(toolchain: dict[str, Any], name: str) -> str:
    return str((((toolchain.get("tools") or {}).get(name) or {}).get("path") or "")).strip()


def _run_mlir_opt_pass(module: IntentMLIRModule, *, pass_arg: str, tool: str) -> IntentMLIRModule:
    arg_tokens = [x for x in shlex.split(str(pass_arg or "").strip()) if str(x).strip()]
    if not arg_tokens:
        raise RuntimeError("mlir-opt pass selector missing pass argument")
    cli_args = [x if str(x).startswith("-") else f"--{x}" for x in arg_tokens]
    p = subprocess.run(
        [tool, *cli_args],
        input=str(module.module_text),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"mlir-opt failed ({' '.join(cli_args)}): {p.stderr or p.stdout}")
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


def _run_mlir_translate_pass(module: IntentMLIRModule, *, pass_arg: str, tool: str) -> IntentMLIRModule:
    arg_tokens = [x for x in shlex.split(str(pass_arg or "").strip()) if str(x).strip()]
    if not arg_tokens:
        raise RuntimeError("mlir-translate selector missing argument")
    cli_args = [x if str(x).startswith("-") else f"--{x}" for x in arg_tokens]
    p = subprocess.run(
        [tool, *cli_args],
        input=str(module.module_text),
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"mlir-translate failed ({' '.join(cli_args)}): {p.stderr or p.stdout}")
    out = IntentMLIRModule(
        module_text=str(p.stdout or ""),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["last_mlir_translate"] = str(pass_arg)
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


def _pass_kind(pass_name: str) -> str:
    name = str(pass_name).strip()
    if name.startswith("mlir-opt"):
        return "mlir-opt"
    if name.startswith("mlir-translate"):
        return "mlir-translate"
    return "python"
