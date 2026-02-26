from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts" / "backend_codegen_smoke.py"
    spec = importlib.util.spec_from_file_location("backend_codegen_smoke", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lower_intent_to_c_with_files_requires_explicit_compat_for_non_contract(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_ALLOW_COMPAT_C_SRC", raising=False)
    try:
        _ = mod.lower_intent_to_c_with_files(
            {"name": "k", "tensors": {}, "ops": [], "outputs": []},
            shape_bindings={},
        )
    except RuntimeError as e:
        msg = str(e)
        assert "strict hard-cut" in msg
        assert "INTENTIR_RVV_ALLOW_COMPAT_C_SRC=1" in msg
    else:
        raise AssertionError("expected explicit compat requirement")


def test_lower_intent_to_c_with_files_accepts_contract_without_compat(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("INTENTIR_RVV_ALLOW_COMPAT_C_SRC", raising=False)
    monkeypatch.setattr(mod, "lower_rvv_contract_to_c_src", lambda *_a, **_k: "ok")
    out = mod.lower_intent_to_c_with_files(
        {
            "schema_version": "intent_mlir_backend_contract_v2",
            "backend": "rvv",
            "kernel_name": "k",
            "executable": {"format": "rvv_elf", "path": "x", "entry": "k", "target": "rvv"},
            "artifacts": {},
        },
        shape_bindings={},
    )
    assert out == "ok"


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


def test_run_one_tuning_uses_json_first_selector(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()

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

    class _Schedule:
        def to_json_dict(self):
            return {"tile_n": 8}

    class _Tuned:
        schedule = _Schedule()

    seen: dict[str, object] = {}

    def _fake_select(intent_json, **kwargs):
        seen["intent_json"] = dict(intent_json)
        seen["kwargs"] = dict(kwargs)
        return _Tuned()

    monkeypatch.setattr(mod, "lower_intent_to_c_with_files", _fake_lower)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod, "select_schedule_from_intent_json", _fake_select)
    monkeypatch.setattr(mod, "load_profile", lambda *_a, **_k: object())

    _ = mod.run_one(
        "k",
        frontend="triton",
        triton_provider="native",
        artifact_dir=str(artifact),
        keep_tmp=False,
        tune_request=mod.TuningRequest(mode="auto", budget=1),
    )

    assert isinstance(seen.get("intent_json"), dict)
    assert (seen.get("intent_json") or {}).get("ops")


def test_write_bin_respects_declared_dtype_over_bool_array(tmp_path: Path) -> None:
    mod = _load_module()
    out_f32 = tmp_path / "f32.bin"
    out_i32 = tmp_path / "i32.bin"
    arr_bool = np.array([[True, False], [False, True]], dtype=np.bool_)

    mod._write_bin(out_f32, arr_bool, "f32")
    mod._write_bin(out_i32, arr_bool, "i32")

    # 4 elements -> 16 bytes for f32 / i32 payloads.
    assert out_f32.stat().st_size == 16
    assert out_i32.stat().st_size == 16


def test_extract_buffer_declared_dtypes_parses_generated_c() -> None:
    mod = _load_module()
    c_src = """
IntentirBufferDesc inputs[] = {
  {"A", (void**)&t_A, (size_t)(sizeof(float) * (size_t)8), INTENTIR_DTYPE_F32},
};
IntentirBufferDesc outputs[] = {
  {"out", (void**)&t_out, (size_t)(sizeof(uint8_t) * (size_t)8), INTENTIR_DTYPE_U8},
};
""".strip()
    got = mod._extract_buffer_declared_dtypes(c_src)
    assert got["A"] == "f32"
    assert got["out"] == "u8"


def test_extract_buffer_declared_dtypes_parses_integer_tokens() -> None:
    mod = _load_module()
    c_src = """
IntentirBufferDesc inputs[] = {
  {"row_idx", (void**)&t_row_idx, (size_t)(sizeof(int32_t) * (size_t)16), INTENTIR_DTYPE_I32},
};
IntentirBufferDesc outputs[] = {
  {"count", (void**)&t_count, (size_t)(sizeof(int64_t) * (size_t)1), INTENTIR_DTYPE_I64},
};
""".strip()
    got = mod._extract_buffer_declared_dtypes(c_src)
    assert got["row_idx"] == "i32"
    assert got["count"] == "i64"


def test_run_one_keep_tmp_keeps_directory(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()

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

    result = mod.run_one("k", frontend="triton", triton_provider="native", artifact_dir=str(artifact), keep_tmp=True)
    tmpdir = Path(str(result.get("tmpdir")))
    assert tmpdir.exists()
    assert (tmpdir / "main.c").exists()


def test_run_one_prefers_declared_buffer_dtype_when_writing_bins(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()

    artifact = tmp_path / "artifacts"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "k.json").write_text(
        """
        {
          "intent": {
            "name": "f",
            "tensors": {
              "x": {"dtype": "f32", "shape": [4]},
              "y": {"dtype": "i32", "shape": [4]}
            },
            "ops": [{"op":"identity","inputs":["x"],"attrs":{},"output":"y"}],
            "outputs": ["y"]
          },
          "baseline": {"shapes": {"N": 4}}
        }
        """.strip(),
        encoding="utf-8",
    )
    np.savez(
        artifact / "k.baseline.npz",
        x=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        y=np.array([True, False, True, False], dtype=np.bool_),
    )

    c_src = """
IntentirBufferDesc inputs[] = {
  {"x", (void**)&t_x, (size_t)(sizeof(float) * (size_t)4), INTENTIR_DTYPE_F32},
};
IntentirBufferDesc outputs[] = {
  {"y", (void**)&t_y, (size_t)(sizeof(uint8_t) * (size_t)4), INTENTIR_DTYPE_U8},
};
int main(){return 0;}
""".strip()

    def _fake_lower(*args, **kwargs):
        _ = args, kwargs
        return c_src

    class _P:
        def __init__(self, rc: int, out: str = "", err: str = ""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    def _fake_run(*args, **kwargs):
        _ = args, kwargs
        return _P(1, "", "compile error")

    seen: dict[str, str] = {}
    real_write = mod._write_bin

    def _spy_write(path, arr, dtype):
        seen[Path(path).name] = str(dtype)
        return real_write(path, arr, dtype)

    monkeypatch.setattr(mod, "lower_intent_to_c_with_files", _fake_lower)
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setattr(mod, "_write_bin", _spy_write)

    _ = mod.run_one("k", frontend="triton", triton_provider="native", artifact_dir=str(artifact), keep_tmp=False)
    assert seen.get("x.bin") == "f32"
    assert seen.get("y_ref.bin") == "u8"


def test_main_progress_prints_per_kernel_in_json_mode(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _load_module()
    out = tmp_path / "rvv.json"

    def _fake_run_one(kernel: str, **kwargs):
        _ = kwargs
        return {
            "kernel": str(kernel),
            "ok": True,
            "rc": 0,
            "reason_code": "ok",
            "lower_ms": 1.0,
            "compile_ms": 2.0,
            "launch_ms": 3.0,
            "total_ms": 6.0,
        }

    monkeypatch.setattr(mod, "run_one", _fake_run_one)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backend_codegen_smoke.py",
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
    assert "[rvv][1/1] START kernel=diag2d" in stdout
    assert "[rvv][1/1] DONE kernel=diag2d ok=True reason=ok" in stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
