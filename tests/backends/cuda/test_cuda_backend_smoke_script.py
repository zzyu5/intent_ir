from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "cuda_backend_smoke.py"
    spec = importlib.util.spec_from_file_location("cuda_backend_smoke_script", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_one_with_stage_timeouts_compile_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()

    def _fake_worker(*args, **kwargs):
        _ = args, kwargs
        return {"ok": False, "timed_out": True}

    monkeypatch.setattr(mod, "_run_worker_with_timeout", _fake_worker)
    res = mod._run_one_with_stage_timeouts(
        "k",
        frontend="triton",
        triton_provider="flaggems",
        artifact_dir=None,
        compile_timeout_sec=3,
        launch_timeout_sec=5,
        runtime_backend="nvcc",
    )
    assert res["ok"] is False
    assert res["reason_code"] == "compile_timeout"
    assert "compile stage exceeded timeout_sec=3" in str((res.get("error") or {}).get("message"))


def test_run_one_with_stage_timeouts_launch_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    payloads = [
        {
            "ok": True,
            "timed_out": False,
            "result": {"kernel": "k", "lower_ms": 1.25, "compile_ms": 2.5, "launch_ms": 0.0, "total_ms": 3.75},
        },
        {"ok": False, "timed_out": True},
    ]

    def _fake_worker(*args, **kwargs):
        _ = args, kwargs
        return payloads.pop(0)

    monkeypatch.setattr(mod, "_run_worker_with_timeout", _fake_worker)
    res = mod._run_one_with_stage_timeouts(
        "k",
        frontend="triton",
        triton_provider="flaggems",
        artifact_dir=None,
        compile_timeout_sec=7,
        launch_timeout_sec=9,
        runtime_backend="nvcc",
    )
    assert res["ok"] is False
    assert res["reason_code"] == "launch_timeout"
    assert res["lower_ms"] == pytest.approx(1.25)
    assert res["compile_ms"] == pytest.approx(2.5)
    assert res["launch_ms"] == pytest.approx(0.0)


def test_main_uses_stage_specific_timeouts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "cuda.json"
    calls: list[dict[str, int]] = []

    monkeypatch.setattr(mod, "_cuda_env_ready", lambda: (True, "ok"))

    def _fake_run_one_with_stage_timeouts(kernel: str, **kwargs):
        calls.append(
            {
                "compile_timeout_sec": int(kwargs["compile_timeout_sec"]),
                "launch_timeout_sec": int(kwargs["launch_timeout_sec"]),
                "runtime_backend": str(kwargs["runtime_backend"]),
            }
        )
        return {
            "kernel": str(kernel),
            "ok": True,
            "reason_code": "ok",
            "lower_ms": 1.0,
            "compile_ms": 2.0,
            "launch_ms": 3.0,
            "total_ms": 6.0,
        }

    monkeypatch.setattr(mod, "_run_one_with_stage_timeouts", _fake_run_one_with_stage_timeouts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_backend_smoke.py",
            "--kernel",
            "diag2d",
            "--timeout-sec",
            "99",
            "--compile-timeout-sec",
            "7",
            "--launch-timeout-sec",
            "11",
            "--json",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 0
    assert calls == [
        {
            "compile_timeout_sec": 7,
            "launch_timeout_sec": 11,
            "runtime_backend": "nvcc",
        }
    ]
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["timeout_sec"] == 99
    assert summary["compile_timeout_sec"] == 7
    assert summary["launch_timeout_sec"] == 11
    assert summary["runtime_backend"] == "nvcc"
    assert "codegen_mode" not in summary
    assert "effective_codegen_mode" not in summary


def test_main_timeout_probe_only_updates_runtime_detail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "cuda.json"

    monkeypatch.setattr(mod, "_cuda_env_ready", lambda: (True, "ok"))
    monkeypatch.setattr(
        mod,
        "_run_one_with_stage_timeouts",
        lambda kernel, **kwargs: {
            "kernel": str(kernel),
            "ok": False,
            "reason_code": "launch_timeout",
            "error": {"type": "TimeoutError", "message": "launch timeout"},
            "lower_ms": 1.0,
            "compile_ms": 2.0,
            "launch_ms": 0.0,
            "total_ms": 3.0,
        },
    )
    monkeypatch.setattr(
        mod,
        "_probe_timeout_runtime_detail",
        lambda **kwargs: {
            "ok": False,
            "reason_code": "compile_timeout",
            "error": {"type": "TimeoutError", "message": "probe timeout"},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_backend_smoke.py",
            "--kernel",
            "diag2d",
            "--refine-timeout-reason",
            "--json",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 1
    summary = json.loads(out.read_text(encoding="utf-8"))
    result = summary["results"][0]
    assert result["reason_code"] == "launch_timeout"
    assert "timeout_probe" in (result.get("runtime_detail") or {})


def test_main_respects_runtime_backend_nvrtc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "cuda.json"
    seen: list[str] = []

    monkeypatch.setattr(mod, "_cuda_env_ready", lambda: (True, "ok"))

    def _fake_run_one_with_stage_timeouts(kernel: str, **kwargs):
        _ = kernel
        seen.append(str(kwargs["runtime_backend"]))
        return {
            "kernel": "k",
            "ok": True,
            "reason_code": "ok",
            "lower_ms": 0.0,
            "compile_ms": 0.0,
            "launch_ms": 0.0,
            "total_ms": 0.0,
        }

    monkeypatch.setattr(mod, "_run_one_with_stage_timeouts", _fake_run_one_with_stage_timeouts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_backend_smoke.py",
            "--kernel",
            "diag2d",
            "--runtime-backend",
            "nvrtc",
            "--json",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 0
    assert seen == ["nvrtc"]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["runtime_backend"] == "nvrtc"
    assert "codegen_mode" not in payload
    assert "effective_codegen_mode" not in payload


def test_run_one_with_stage_timeouts_propagates_compile_cache_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    payloads = [
        {
            "ok": True,
            "timed_out": False,
            "result": {
                "kernel": "k",
                "ok": True,
                "reason_code": "ok",
                "lower_ms": 1.0,
                "compile_ms": 2.0,
                "launch_ms": 0.0,
                "total_ms": 3.0,
                "compile_cache_hit": True,
                "compile_module_name": "intentir_cuda_k_demo",
                "compile_build_dir": "/tmp/intentir_cuda_k_demo",
            },
        },
        {
            "ok": True,
            "timed_out": False,
            "result": {
                "kernel": "k",
                "ok": True,
                "reason_code": "ok",
                "launch_ms": 3.5,
            },
        },
    ]

    def _fake_worker(*args, **kwargs):
        _ = args, kwargs
        return payloads.pop(0)

    monkeypatch.setattr(mod, "_run_worker_with_timeout", _fake_worker)
    res = mod._run_one_with_stage_timeouts(
        "k",
        frontend="triton",
        triton_provider="flaggems",
        artifact_dir=None,
        compile_timeout_sec=5,
        launch_timeout_sec=7,
        runtime_backend="nvcc",
    )
    assert res["ok"] is True
    assert res["compile_cache_hit"] is True
    assert res["compile_module_name"] == "intentir_cuda_k_demo"
    assert res["compile_build_dir"] == "/tmp/intentir_cuda_k_demo"


def test_main_progress_prints_per_kernel_in_json_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = _load_module()
    out = tmp_path / "cuda.json"

    monkeypatch.setattr(mod, "_cuda_env_ready", lambda: (True, "ok"))

    def _fake_run_one_with_stage_timeouts(kernel: str, **kwargs):
        _ = kwargs
        return {
            "kernel": str(kernel),
            "ok": True,
            "reason_code": "ok",
            "lower_ms": 1.0,
            "compile_ms": 2.0,
            "launch_ms": 3.0,
            "total_ms": 6.0,
        }

    monkeypatch.setattr(mod, "_run_one_with_stage_timeouts", _fake_run_one_with_stage_timeouts)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuda_backend_smoke.py",
            "--kernel",
            "diag2d",
            "--json",
            "--progress",
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 0
    stdout = capsys.readouterr().out
    assert "[cuda][1/1] START kernel=diag2d" in stdout
    assert "[cuda][1/1] DONE kernel=diag2d ok=True reason=ok" in stdout
