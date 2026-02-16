from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "rvv_remote_run.py"
    spec = importlib.util.spec_from_file_location("rvv_remote_run", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_to_raw_bytes_respects_declared_dtype_over_bool_array() -> None:
    mod = _load_module()
    arr_bool = np.array([[True, False], [False, True]], dtype=np.bool_)

    raw_f32 = mod._to_raw_bytes(arr_bool, "f32")
    raw_i32 = mod._to_raw_bytes(arr_bool, "i32")
    raw_bool = mod._to_raw_bytes(arr_bool, "bool")

    # 4 elements -> 16 bytes for f32/i32, 4 bytes for bool(u8) payload.
    assert len(raw_f32) == 16
    assert len(raw_i32) == 16
    assert len(raw_bool) == 4
