from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"


def _load_perf_runner_module():
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DummySpec:
    def __init__(self, *, canonical_shapes: dict[str, int] | None = None, constexpr: dict[str, int] | None = None) -> None:
        self.canonical_shapes = dict(canonical_shapes or {})
        self.constexpr = dict(constexpr or {})


def test_gpu_perf_kernel_allowlist_parses_list_and_dict(tmp_path: Path) -> None:
    mod = _load_perf_runner_module()

    p_list = tmp_path / "allow_list.json"
    p_list.write_text(json.dumps(["a", "b", " ", ""], indent=2), encoding="utf-8")
    allow, payload, loaded = mod._load_kernel_allowlist(p_list)
    assert loaded is True
    assert allow == {"a", "b"}
    assert isinstance(payload, dict)

    p_dict = tmp_path / "allow_dict.json"
    p_dict.write_text(json.dumps({"schema_version": "x", "kernels": ["k1", "k2"]}, indent=2), encoding="utf-8")
    allow2, payload2, loaded2 = mod._load_kernel_allowlist(p_dict)
    assert loaded2 is True
    assert allow2 == {"k1", "k2"}
    assert payload2.get("schema_version") == "x"


def test_gpu_perf_kernel_allowlist_rejects_invalid_payload(tmp_path: Path) -> None:
    mod = _load_perf_runner_module()
    p_bad = tmp_path / "bad.json"
    p_bad.write_text(json.dumps({"nope": []}, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit):
        _ = mod._load_kernel_allowlist(p_bad)


def test_gpu_perf_allowlist_excluded_row_sets_skip_reason_and_shape() -> None:
    mod = _load_perf_runner_module()
    row = mod._skipped_row_allowlist_excluded(
        kernel="ai_bench_matmul",
        family="matmul",
        chunk_name="chunk_001",
        bench_mode="graph",
        spec_entry={"spec": _DummySpec(canonical_shapes={"M": 256}, constexpr={}), "source": "triton_native"},
    )
    assert row["skip_reason"] == "allowlist_excluded"
    assert row["reason_code"] == "allowlist_excluded"
    assert row["count_in_denominator"] is False
    assert row["shape"]["M"] == 256
    assert row["shape"]["BLOCK_M"] == 64
    assert row["shape"]["BLOCK_N"] == 16
    assert row["shape"]["BLOCK_K"] == 16
