from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeTensor:
    def __init__(self, arr):
        a = np.asarray(arr)
        self._arr = a
        self.shape = tuple(a.shape)
        self.dtype = a.dtype

    def contiguous(self):
        return self

    def to(self, dtype=None):  # noqa: ARG002
        return self


def _install_fake_torch(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod.torch, "as_tensor", lambda arr, device=None: _FakeTensor(arr))


def _fake_spec_map(module_name: str) -> dict[str, dict]:
    return {"k": {"spec": SimpleNamespace(module=module_name), "source": "triton_native"}}


@pytest.mark.parametrize(
    ("kernel", "inputs_np", "bindings", "callable_name", "expected_launch_source"),
    [
        (
            "add2d",
            {"x": np.ones((4, 64), dtype=np.float32), "y": np.ones((4, 64), dtype=np.float32), "alpha": np.array(1.0, dtype=np.float32)},
            {"M": 4, "N": 64, "alpha": 1},
            "add2d",
            "kernel_adapter:add2d",
        ),
        (
            "clamp2d",
            {"x": np.ones((4, 64), dtype=np.float32), "mini": np.array(-1.0, dtype=np.float32), "maxi": np.array(1.0, dtype=np.float32)},
            {"M": 4, "N": 64, "mini": -1.0, "maxi": 1.0},
            "clamp2d",
            "kernel_adapter:clamp2d",
        ),
        (
            "softmax_inner",
            {"input": np.ones((4, 64), dtype=np.float32)},
            {"M": 4, "N": 64},
            "softmax",
            "kernel_adapter:softmax_inner",
        ),
        (
            "group_norm_kernel",
            {
                "X": np.ones((2, 4, 4), dtype=np.float32),
                "W": np.ones((4,), dtype=np.float32),
                "B": np.zeros((4,), dtype=np.float32),
            },
            {"N": 2, "C": 4, "HW": 4, "num_groups": 2, "eps": 1e-5},
            "group_norm",
            "kernel_adapter:group_norm_kernel",
        ),
        (
            "layer_norm_persistent",
            {
                "in_ptr": np.ones((4, 64), dtype=np.float32),
                "weight_ptr": np.ones((64,), dtype=np.float32),
                "bias_ptr": np.zeros((64,), dtype=np.float32),
            },
            {"M": 4, "N": 64, "eps": 1e-5},
            "layer_norm",
            "kernel_adapter:layer_norm_persistent",
        ),
        (
            "rms_norm2d",
            {"input": np.ones((4, 64), dtype=np.float32), "weight": np.ones((64,), dtype=np.float32)},
            {"M": 4, "N": 64, "eps": 1e-5},
            "rms_norm2d",
            "kernel_adapter:rms_norm2d",
        ),
        (
            "where2d",
            {
                "self": np.ones((4, 64), dtype=np.float32),
                "other": np.zeros((4, 64), dtype=np.float32),
                "condition": np.ones((4, 64), dtype=np.bool_),
            },
            {"M": 4, "N": 64},
            "where2d",
            "kernel_adapter:where2d",
        ),
        (
            "upsample_bicubic2d_aa",
            {
                "ptr_i": np.ones((1, 1, 4, 4), dtype=np.float32),
                "reciprocal_scale_h": np.array(1.0, dtype=np.float32),
                "reciprocal_scale_w": np.array(1.0, dtype=np.float32),
            },
            {"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 4, "OW": 4},
            "_upsample_bicubic2d_aa",
            "kernel_adapter:upsample_bicubic2d_aa",
        ),
    ],
)
def test_build_native_launch_fn_uses_kernel_adapter_for_known_blocked_kernels(
    monkeypatch: pytest.MonkeyPatch,
    kernel: str,
    inputs_np: dict,
    bindings: dict,
    callable_name: str,
    expected_launch_source: str,
) -> None:
    mod = _load_module()
    _install_fake_torch(mod, monkeypatch)

    calls: list[tuple[str, int]] = []

    def _record(*args, **kwargs):  # noqa: ARG001
        calls.append((callable_name, len(args)))
        return None

    fake_module = SimpleNamespace(__name__="fake.module")
    setattr(fake_module, callable_name, _record)
    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: fake_module)

    spec_map = {str(kernel): {"spec": SimpleNamespace(module="fake.module"), "source": "triton_native"}}
    run_fn, module_name, meta = mod._build_native_launch_fn(
        kernel=str(kernel),
        inputs_np=dict(inputs_np),
        bindings=dict(bindings),
        spec_map=spec_map,
        device="cuda",
    )
    assert module_name == "fake.module"
    assert str(meta.get("launch_source") or "") == expected_launch_source

    run_fn()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("kernel", "inputs_np", "bindings", "expected_launch_source", "expected_scope_token", "expected_call"),
    [
        (
            "scaled_dot_product_attention_bhsd",
            {
                "query": np.ones((1, 2, 8, 16), dtype=np.float32),
                "key": np.ones((1, 2, 8, 16), dtype=np.float32),
                "value": np.ones((1, 2, 8, 16), dtype=np.float32),
                "scale": np.array(0.25, dtype=np.float32),
                "is_causal": np.array(0, dtype=np.int32),
            },
            {"B": 1, "H": 2, "Q": 8, "K": 8, "D": 16},
            "kernel_adapter:scaled_dot_product_attention_bhsd",
            "scaled_dot_product_attention",
            "sdpa",
        ),
        (
            "flash_attn_varlen_func_bhsd",
            {
                "query": np.ones((1, 2, 8, 16), dtype=np.float32),
                "key": np.ones((1, 2, 8, 16), dtype=np.float32),
                "value": np.ones((1, 2, 8, 16), dtype=np.float32),
                "scale": np.array(0.25, dtype=np.float32),
                "is_causal": np.array(0, dtype=np.int32),
            },
            {"B": 1, "H": 2, "Q": 8, "K": 8, "D": 16},
            "kernel_adapter:scaled_dot_product_attention_bhsd",
            "scaled_dot_product_attention",
            "sdpa",
        ),
        (
            "unique2d",
            {"inp": np.array([1, 2, 1, 3], dtype=np.int32)},
            {"N": 4},
            "kernel_adapter:unique2d",
            "",
            "unique",
        ),
    ],
)
def test_build_native_launch_fn_uses_attention_unique_adapters(
    monkeypatch: pytest.MonkeyPatch,
    kernel: str,
    inputs_np: dict,
    bindings: dict,
    expected_launch_source: str,
    expected_scope_token: str,
    expected_call: str,
) -> None:
    mod = _load_module()
    _install_fake_torch(mod, monkeypatch)

    scopes: list[list[str]] = []
    calls: list[str] = []

    class _Scope:
        def __init__(self, include):
            self.include = list(include or [])

        def __enter__(self):
            scopes.append(list(self.include))
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ARG002
            return False

    def _use_gems(*, include=None):
        return _Scope(include)

    def _scaled_dot_product_attention(query, key, value, **kwargs):  # noqa: ARG001
        calls.append("sdpa")
        return query

    def _unique2(inp, *, sorted=False, return_inverse=False, return_counts=False):
        calls.append("unique")
        assert bool(sorted) is False
        assert bool(return_inverse) is False
        assert bool(return_counts) is False
        return inp, None, None

    fake_module = SimpleNamespace(
        __name__="fake.module",
        flag_gems=SimpleNamespace(use_gems=_use_gems),
        flag_gems_ops=SimpleNamespace(
            scaled_dot_product_attention=_scaled_dot_product_attention,
            _unique2=_unique2,
        ),
        default_flaggems_kernel_specs=lambda: {"noop": True},
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: fake_module)

    spec_map = {str(kernel): {"spec": SimpleNamespace(module="fake.module"), "source": "flaggems_native"}}
    run_fn, module_name, meta = mod._build_native_launch_fn(
        kernel=str(kernel),
        inputs_np=dict(inputs_np),
        bindings=dict(bindings),
        spec_map=spec_map,
        device="cuda",
    )
    assert module_name == "fake.module"
    assert str(meta.get("launch_source") or "") == expected_launch_source

    run_fn()
    if expected_call:
        assert calls == [expected_call]
    else:
        assert calls == []
    if expected_scope_token:
        assert any(expected_scope_token in scope for scope in scopes)


def test_build_native_launch_fn_skips_zero_arg_helper_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    _install_fake_torch(mod, monkeypatch)

    calls: list[str] = []

    def _helper():
        return {"noop": True}

    def _foo(x):  # noqa: ARG001
        calls.append("foo")
        return None

    fake_module = SimpleNamespace(
        __name__="fake.module",
        default_flaggems_kernel_specs=_helper,
        foo=_foo,
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: fake_module)

    run_fn, _module_name, meta = mod._build_native_launch_fn(
        kernel="foo2d",
        inputs_np={"x": np.ones((4, 4), dtype=np.float32)},
        bindings={"M": 4, "N": 4},
        spec_map={"foo2d": {"spec": SimpleNamespace(module="fake.module"), "source": "flaggems_native"}},
        device="cuda",
    )
    assert str(meta.get("launch_source") or "") == "heuristic_signature"
    assert int(meta.get("arg_count") or 0) >= 1
    run_fn()
    assert calls == ["foo"]


def test_build_native_launch_fn_flaggems_native_prefers_ops_callable_over_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    _install_fake_torch(mod, monkeypatch)

    helper_calls: list[str] = []
    native_calls: list[tuple[str, int]] = []

    def _helper():
        helper_calls.append("helper")
        return {"noop": True}

    def _normed_cumsum(inp, dim=-1):  # noqa: ARG001
        native_calls.append(("normed_cumsum", int(dim)))
        return inp

    fake_module = SimpleNamespace(
        __name__="fake.module",
        default_flaggems_kernel_specs=_helper,
        flag_gems_ops=SimpleNamespace(normed_cumsum=_normed_cumsum),
    )
    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: fake_module)

    run_fn, _module_name, meta = mod._build_native_launch_fn(
        kernel="normed_cumsum2d",
        inputs_np={
            "inp": np.ones((4, 64), dtype=np.float32),
            "axis": np.array(1, dtype=np.int32),
            "eps": np.array(1.0e-6, dtype=np.float32),
        },
        bindings={"M": 4, "N": 64, "AXIS": 1, "EPS": 1.0e-6},
        spec_map={"normed_cumsum2d": {"spec": SimpleNamespace(module="fake.module"), "source": "flaggems_native"}},
        device="cuda",
    )
    assert str(meta.get("launch_source") or "") == "kernel_adapter:normed_cumsum2d"
    run_fn()
    assert len(native_calls) == 1
    assert native_calls[0][0] == "normed_cumsum"
    assert helper_calls == []


def test_to_status_entries_emits_runtime_fallback_and_native_launch_fields() -> None:
    mod = _load_module()
    rows = [
        {
            "kernel": "k0",
            "semantic_op": "k0",
            "family": "f0",
            "reason_code": "ok",
            "reason_detail": "ok",
            "count_in_denominator": True,
            "ok": True,
            "ratio": 0.95,
            "qps_native": 1000.0,
            "qps_intentir": 950.0,
            "latency_native_ms": 0.1,
            "latency_intentir_ms": 0.11,
            "capture_ms_native": 1.0,
            "capture_ms_intentir": 1.1,
            "replay_ms_native": 0.1,
            "replay_ms_intentir": 0.11,
            "runtime_fallback": True,
            "runtime_fallback_detail": "cuda_ptx_origin=nvcc_dlto_fallback_from_llvm",
            "cuda_ptx_origin": "nvcc_dlto_fallback_from_llvm",
            "native_launch_source": "kernel_adapter:add2d",
            "native_launch_error": "",
        }
    ]
    entries = mod._to_status_entries(rows, threshold=0.80)
    assert len(entries) == 1
    row = entries[0]
    assert bool(row.get("runtime_fallback")) is True
    assert "cuda_ptx_origin=nvcc_dlto_fallback_from_llvm" in str(row.get("runtime_fallback_detail") or "")
    native_detail = dict((row.get("runtime_detail") or {}).get("native") or {})
    assert str(native_detail.get("launch_source") or "") == "kernel_adapter:add2d"


def test_build_native_launch_fn_prefers_scalar_inputs_over_integerized_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    _install_fake_torch(mod, monkeypatch)

    seen: list[tuple[float, float]] = []

    def _clamp2d(_x, lo, hi):
        seen.append((float(lo), float(hi)))
        return None

    fake_module = SimpleNamespace(__name__="fake.module")
    setattr(fake_module, "clamp2d", _clamp2d)
    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: fake_module)

    run_fn, _module_name, meta = mod._build_native_launch_fn(
        kernel="clamp2d",
        inputs_np={
            "x": np.ones((4, 64), dtype=np.float32),
            "mini": np.array(-0.5, dtype=np.float32),
            "maxi": np.array(0.5, dtype=np.float32),
        },
        bindings={"M": 4, "N": 64, "mini": 0, "maxi": 0},
        spec_map={"clamp2d": {"spec": SimpleNamespace(module="fake.module"), "source": "triton_native"}},
        device="cuda",
    )
    assert str(meta.get("launch_source") or "") == "kernel_adapter:clamp2d"
    run_fn()
    assert seen == [(-0.5, 0.5)]


def test_stabilize_near_threshold_ratio_applies_order_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    # Order: intent_native (intent then native), native_intent (native then intent).
    bench_seq = [
        {"qps": 84.0, "latency_ms": 0.0142, "capture_ms": 1.2, "replay_ms": 0.0142},
        {"qps": 100.0, "latency_ms": 0.0119, "capture_ms": 1.1, "replay_ms": 0.0119},
        {"qps": 101.0, "latency_ms": 0.0118, "capture_ms": 1.0, "replay_ms": 0.0118},
        {"qps": 84.0, "latency_ms": 0.0141, "capture_ms": 1.3, "replay_ms": 0.0141},
    ]

    def _fake_bench_graph(_fn, *, warmup, iters, repeats):  # noqa: ARG001
        assert warmup >= 1
        assert iters >= 2000
        assert repeats >= 7
        return bench_seq.pop(0)

    monkeypatch.setattr(mod, "_bench_graph", _fake_bench_graph)

    native_bench = {"qps": 100.0, "latency_ms": 0.0119, "capture_ms": 1.0, "replay_ms": 0.0119}
    intent_bench = {"qps": 79.0, "latency_ms": 0.0148, "capture_ms": 1.4, "replay_ms": 0.0148}
    native_out, intent_out, ratio_out, meta = mod._stabilize_near_threshold_ratio(
        ratio=0.79,
        threshold=0.80,
        native_latency_ms=0.0119,
        intent_latency_ms=0.0148,
        native_fn=lambda: None,
        intent_fn=lambda: None,
        warmup=20,
        iters=200,
        repeats=5,
        native_bench=native_bench,
        intent_bench=intent_bench,
    )

    assert bool(meta.get("applied")) is True
    assert float(meta.get("ratio_initial")) == pytest.approx(0.79)
    assert ratio_out >= 0.80
    assert float(native_out["qps"]) > 0.0
    assert float(intent_out["qps"]) > 0.0
    assert bench_seq == []


def test_stabilize_near_threshold_ratio_skips_for_large_latency() -> None:
    mod = _load_module()
    native_bench = {"qps": 100.0, "latency_ms": 0.20, "capture_ms": 1.0, "replay_ms": 0.20}
    intent_bench = {"qps": 70.0, "latency_ms": 0.28, "capture_ms": 1.2, "replay_ms": 0.28}
    native_out, intent_out, ratio_out, meta = mod._stabilize_near_threshold_ratio(
        ratio=0.70,
        threshold=0.80,
        native_latency_ms=0.20,
        intent_latency_ms=0.28,
        native_fn=lambda: None,
        intent_fn=lambda: None,
        warmup=20,
        iters=200,
        repeats=5,
        native_bench=native_bench,
        intent_bench=intent_bench,
    )
    assert bool(meta.get("applied")) is False
    assert float(ratio_out) == pytest.approx(0.70)
    assert native_out == native_bench
    assert intent_out == intent_bench


def test_apply_intentir_perf_binding_overrides_adds_kernel_specific_defaults() -> None:
    mod = _load_module()
    merged, applied = mod._apply_intentir_perf_binding_overrides(
        kernel="conv3d_ncdhw",
        bindings={"M": 16, "N": 32},
    )
    assert int(merged.get("tile_n")) == 192
    assert int(applied.get("tile_n")) == 192


def test_apply_intentir_perf_binding_overrides_respects_existing_and_disable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    merged, applied = mod._apply_intentir_perf_binding_overrides(
        kernel="sort_stable2d",
        bindings={"tile_n": 256},
    )
    assert int(merged.get("tile_n")) == 256
    assert applied == {}

    monkeypatch.setenv("INTENTIR_GPU_PERF_DISABLE_KERNEL_TUNING", "1")
    merged2, applied2 = mod._apply_intentir_perf_binding_overrides(
        kernel="sort_stable2d",
        bindings={"M": 4, "N": 64},
    )
    assert "tile_n" not in merged2
    assert applied2 == {}


def test_perf_rebuild_kernel_set_disabled_by_default_in_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "strict")
    monkeypatch.delenv("INTENTIR_GPU_PERF_DISABLE_CONTRACT_REBUILD", raising=False)
    monkeypatch.delenv("INTENTIR_GPU_PERF_REBUILD_KERNELS", raising=False)
    assert mod._perf_rebuild_kernel_set() == set()


def test_perf_rebuild_kernel_set_respects_explicit_list_in_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_FALLBACK_POLICY", "strict")
    monkeypatch.delenv("INTENTIR_GPU_PERF_DISABLE_CONTRACT_REBUILD", raising=False)
    monkeypatch.setenv("INTENTIR_GPU_PERF_REBUILD_KERNELS", "batch_norm2d, mm2d")
    assert mod._perf_rebuild_kernel_set() == {"batch_norm2d", "mm2d"}


def test_maybe_rewrite_contract_for_perf_rebuild_uses_intent_seed_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_GPU_PERF_REBUILD_KERNELS", "batch_norm2d")
    monkeypatch.delenv("INTENTIR_GPU_PERF_DISABLE_CONTRACT_REBUILD", raising=False)

    mlir_path = tmp_path / "batch_norm2d.intentir.intentdialect.downstream_cuda_llvm.module.mlir"
    mlir_path.write_text("module {}", encoding="utf-8")
    seed_path = tmp_path / "batch_norm2d.intent_seed.json"
    seed_path.write_text(
        '{"intent_expanded": {"name": "batch_norm2d", "ops": [], "tensors": {}, "outputs": []}}',
        encoding="utf-8",
    )
    payload = {
        "kernel_name": "batch_norm2d",
        "artifacts": {"mlir_module_path": str(mlir_path)},
        "executable": {"format": "cuda_ptx", "path": "k.ptx", "target": "cuda", "entry": "batch_norm2d"},
        "reason_context": {},
    }
    out, meta = mod._maybe_rewrite_contract_for_perf_rebuild(kernel="batch_norm2d", contract_payload=dict(payload))
    assert bool(meta.get("enabled")) is True
    assert bool(meta.get("applied")) is True
    exe = dict(out.get("executable") or {})
    assert str(exe.get("format") or "") == "cuda_mlir_module"
    assert str(exe.get("path") or "") == str(mlir_path)
    assert isinstance(dict(out.get("reason_context") or {}).get("fallback_intent_json"), dict)


def test_maybe_rewrite_contract_for_perf_rebuild_no_seed_keeps_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_GPU_PERF_REBUILD_KERNELS", "batch_norm2d")
    monkeypatch.delenv("INTENTIR_GPU_PERF_DISABLE_CONTRACT_REBUILD", raising=False)

    mlir_path = tmp_path / "batch_norm2d.intentir.intentdialect.downstream_cuda_llvm.module.mlir"
    mlir_path.write_text("module {}", encoding="utf-8")
    payload = {
        "kernel_name": "batch_norm2d",
        "artifacts": {"mlir_module_path": str(mlir_path)},
        "executable": {"format": "cuda_ptx", "path": "k.ptx", "target": "cuda", "entry": "batch_norm2d"},
        "reason_context": {},
    }
    out, meta = mod._maybe_rewrite_contract_for_perf_rebuild(kernel="batch_norm2d", contract_payload=dict(payload))
    assert bool(meta.get("enabled")) is True
    assert bool(meta.get("applied")) is False
    exe = dict(out.get("executable") or {})
    assert str(exe.get("format") or "") == "cuda_ptx"


def test_maybe_rewrite_contract_for_perf_rebuild_prefers_seed_over_existing_reason_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setenv("INTENTIR_GPU_PERF_REBUILD_KERNELS", "batch_norm2d")
    monkeypatch.delenv("INTENTIR_GPU_PERF_DISABLE_CONTRACT_REBUILD", raising=False)

    mlir_path = tmp_path / "batch_norm2d.intentir.intentdialect.downstream_cuda_llvm.module.mlir"
    mlir_path.write_text("module {}", encoding="utf-8")
    seed_path = tmp_path / "batch_norm2d.intent_seed.json"
    seed_path.write_text(
        '{"intent_expanded": {"name": "from_seed", "ops": [], "tensors": {}, "outputs": []}}',
        encoding="utf-8",
    )
    payload = {
        "kernel_name": "batch_norm2d",
        "artifacts": {"mlir_module_path": str(mlir_path)},
        "executable": {"format": "cuda_ptx", "path": "k.ptx", "target": "cuda", "entry": "batch_norm2d"},
        "reason_context": {"fallback_intent_json": {"name": "from_reason_ctx"}},
    }
    out, meta = mod._maybe_rewrite_contract_for_perf_rebuild(kernel="batch_norm2d", contract_payload=dict(payload))
    assert bool(meta.get("applied")) is True
    selected = dict(dict(out.get("reason_context") or {}).get("fallback_intent_json") or {})
    assert str(selected.get("name") or "") == "from_seed"
