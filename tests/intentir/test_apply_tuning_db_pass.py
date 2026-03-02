from __future__ import annotations

import json

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import to_mlir
from intent_ir.mlir.passes.apply_tuning_db import apply_tuning_db


def test_apply_tuning_db_injects_bindings_and_kernel_kind(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "cuda.jsonl"
    db.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "intentir_tuning_db_entry_v1",
                        "backend": "cuda",
                        "kernel": "ai_bench_matmul",
                        "arch": "sm89",
                        "bindings": {"tile_n": 384},
                        "kernel_kind": "matmul_tile_v1",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INTENTIR_CUDA_TUNING_DB", str(db))
    monkeypatch.setenv("INTENTIR_CUDA_SM", "sm_89")

    intent = IntentFunction.from_json_dict(
        {
            "name": "ai_bench_matmul",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )
    mod = to_mlir(intent)
    mod.meta["spec_name"] = "ai_bench_matmul"
    mod.meta["shape_bindings"] = {"M": 64, "N": 16, "K": 16}

    out = apply_tuning_db(mod, backend="cuda")
    assert out.meta.get("intentir_tuning_source") == "tuning_db"
    assert dict(out.meta.get("intentir_tuning_applied") or {}) == {"tile_n": 384}
    assert out.meta.get("intentir_kernel_kind_override") == "matmul_tile_v1"
    assert dict(out.meta.get("shape_bindings") or {}).get("tile_n") == 384


def test_apply_tuning_db_no_match_records_none(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db = tmp_path / "cuda.jsonl"
    db.write_text(json.dumps({"backend": "cuda", "kernel": "k", "arch": "sm89", "bindings": {"tile_n": 1}}) + "\n", encoding="utf-8")
    monkeypatch.setenv("INTENTIR_CUDA_TUNING_DB", str(db))
    monkeypatch.setenv("INTENTIR_CUDA_SM", "sm89")

    intent = IntentFunction.from_json_dict(
        {
            "name": "ai_bench_matmul",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )
    mod = to_mlir(intent)
    mod.meta["spec_name"] = "ai_bench_matmul"
    mod.meta["shape_bindings"] = {"M": 64, "N": 16, "K": 16}

    out = apply_tuning_db(mod, backend="cuda")
    assert out.meta.get("intentir_tuning_source") == "none"
    assert dict(out.meta.get("intentir_tuning_applied") or {}) == {}
