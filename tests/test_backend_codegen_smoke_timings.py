from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts" / "backend_codegen_smoke.py"
    spec = importlib.util.spec_from_file_location("backend_codegen_smoke", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_one_compile_failure_contains_timing_fields(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()

    # Minimal fake report/baseline artifacts.
    artifact = tmp_path / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "k.json").write_text(
        """
        {
          "intent": {
            "name": "f",
            "tensors": {
              "x": {"dtype": "f32", "shape": [2]},
              "y": {"dtype": "f32", "shape": [2]}
            },
            "ops": [{"op":"identity","inputs":["x"],"attrs":{},"output":"y"}],
            "outputs": ["y"]
          },
          "baseline": {"shapes": {"M": 2}}
        }
        """.strip(),
        encoding="utf-8",
    )
    np.savez(artifact / "k.baseline.npz", x=np.ones((2,), dtype=np.float32), y=np.ones((2,), dtype=np.float32))

    def _fake_lower(*args, **kwargs):
        _ = args, kwargs
        return "int main(){return 0;}"

    class _P:
        def __init__(self, rc: int, out: str = "", err: str = ""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def _fake_run(*args, **kwargs):
        _ = args, kwargs
        return _P(1, "", "compile error")

    monkeypatch.setattr(mod, "lower_intent_to_c_with_files", _fake_lower)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    result = mod.run_one("k", frontend="triton", triton_provider="native", artifact_dir=str(artifact), keep_tmp=False)
    for key in ("lower_ms", "compile_ms", "launch_ms", "total_ms"):
        assert key in result
