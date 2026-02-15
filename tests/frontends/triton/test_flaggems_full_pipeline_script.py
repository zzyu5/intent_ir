from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "triton" / "flaggems_full_pipeline_verify.py"
    spec = importlib.util.spec_from_file_location("flaggems_full_pipeline_verify", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_pipeline_writes_failure_report_on_kernel_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "resolve_flaggems_execution",
        lambda **kwargs: SimpleNamespace(
            use_intent_ir=False,
            intentir_mode="auto",
            execution_policy=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        mod,
        "default_flaggems_kernel_specs",
        lambda **kwargs: [SimpleNamespace(name="tile2d")],
    )
    monkeypatch.setattr(mod, "coverage_flaggems_kernel_specs", lambda **kwargs: [SimpleNamespace(name="tile2d")])
    monkeypatch.setattr(
        mod,
        "run_pipeline_for_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flaggems_full_pipeline_verify.py",
            "--suite",
            "smoke",
            "--kernel",
            "tile2d",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    mod.main()

    report = json.loads((tmp_path / "out" / "tile2d.json").read_text(encoding="utf-8"))
    assert report["reason_code"] == "pipeline_exception"
    assert report["diff"]["ok"] is False
    assert "boom" in str(report["error"]["message"])


def test_full_pipeline_strict_kernel_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "resolve_flaggems_execution",
        lambda **kwargs: SimpleNamespace(
            use_intent_ir=False,
            intentir_mode="auto",
            execution_policy=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        mod,
        "default_flaggems_kernel_specs",
        lambda **kwargs: [SimpleNamespace(name="tile2d")],
    )
    monkeypatch.setattr(mod, "coverage_flaggems_kernel_specs", lambda **kwargs: [SimpleNamespace(name="tile2d")])
    monkeypatch.setattr(
        mod,
        "run_pipeline_for_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flaggems_full_pipeline_verify.py",
            "--suite",
            "smoke",
            "--kernel",
            "tile2d",
            "--strict-kernel-failure",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 1
    report = json.loads((tmp_path / "out" / "tile2d.json").read_text(encoding="utf-8"))
    assert report["reason_code"] == "pipeline_exception"
