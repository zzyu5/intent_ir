from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_gpu_perf_shape_telemetry_merges_bindings_and_constexpr_and_overrides() -> None:
    mod = _load_perf_runner_module()

    shape = mod._shape_telemetry_for_kernel(
        kernel="flash_attention2d",
        ctx_bindings={"Q_CTX": "64", "KV_CTX": 64, "HEAD_DIM": 64, "bad": "x"},
        spec_entry={"spec": _DummySpec(canonical_shapes={"Q_CTX": 1, "KV_CTX": 1}, constexpr={"BLOCK_KV": 999}), "source": "triton_native"},
    )
    assert shape["Q_CTX"] == 64
    assert shape["KV_CTX"] == 64
    assert shape["HEAD_DIM"] == 64
    assert shape["BLOCK_KV"] == 32
    assert "bad" not in shape


def test_gpu_perf_shape_telemetry_records_attn_overrides_only_for_triton_native() -> None:
    mod = _load_perf_runner_module()

    attn = mod._shape_telemetry_for_kernel(
        kernel="_attn_fwd",
        ctx_bindings={"Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        spec_entry={"spec": _DummySpec(), "source": "triton_native"},
    )
    assert attn["BLOCK_M"] == 16
    assert attn["BLOCK_N"] == 16
    assert attn["STAGE"] == 1
    assert attn["HAS_ATTN_MASK"] == 0
    assert attn["PRE_LOAD_V"] == 0

    non_native = mod._shape_telemetry_for_kernel(
        kernel="ai_bench_matmul",
        ctx_bindings={"M": 256},
        spec_entry={"spec": _DummySpec(constexpr={"BLOCK_M": 7, "BLOCK_N": 7, "BLOCK_K": 7}), "source": "flaggems_native"},
    )
    assert non_native["BLOCK_M"] == 7
    assert non_native["BLOCK_N"] == 7
    assert non_native["BLOCK_K"] == 7

