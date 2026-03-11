from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"


def _load_perf_runner_module():
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", _SCRIPT)
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


def test_build_intentir_launch_fn_aliases_aligned_tensor_args(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_perf_runner_module()

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod.torch, "as_tensor", lambda arr, device=None: _FakeTensor(arr))
    monkeypatch.setattr(mod.torch, "empty", lambda shape, device=None, dtype=None: _FakeTensor(np.empty(shape, dtype=np.float32)))
    monkeypatch.setattr(mod.torch, "tensor", lambda v, device=None, dtype=None: _FakeTensor(np.asarray(v)))

    monkeypatch.setattr(
        mod,
        "_prepare_kernel_context",
        lambda *args, **kwargs: {
            "bindings": {"M": 4},
            "mlir_contract": {"schema_version": "intent_mlir_backend_contract_v2"},
            "tensor_specs": {},
            "baseline": {},
            "external_inputs": {},
            "outputs": ["Out"],
            "intent_json": {},
        },
    )
    monkeypatch.setattr(mod, "_apply_intentir_perf_binding_overrides", lambda **kwargs: (dict(kwargs["bindings"]), {}, "none"))
    monkeypatch.setattr(mod, "_maybe_rewrite_contract_for_perf_rebuild", lambda **kwargs: (dict(kwargs["contract_payload"]), {}))
    monkeypatch.setattr(mod, "_build_inputs_np", lambda **kwargs: {"K": np.ones((4,), dtype=np.float32)})

    lowered = {
        "kernel_name": "k",
        "io_spec": {
            "arg_names": ["K", "K__aligned", "Out", "Out__aligned"],
            "tensors": {
                "K": {"dtype": "f32", "shape": [4]},
                "K__aligned": {"dtype": "f32", "shape": [4]},
                "Out": {"dtype": "f32", "shape": [4]},
                "Out__aligned": {"dtype": "f32", "shape": [4]},
            },
            "outputs": ["Out"],
            "scalars": {},
        },
        "launch": {"grid": [1, 1, 1], "block": [1, 1, 1], "shared_mem": 0},
        "output_names": ["Out"],
        "bindings": {},
        "cuda_ptx": b"// fake ptx",
        "executable_format": "cuda_ptx",
        "contract_schema_version": "intent_mlir_backend_contract_v2",
        "cuda_ptx_origin": "llvm_llc",
    }
    monkeypatch.setattr(mod, "lower_cuda_contract_to_kernel", lambda *args, **kwargs: dict(lowered))

    calls: list[tuple] = []

    class _FakeCudaModule:
        def launch(self, *args):
            calls.append(args)

    monkeypatch.setattr(mod, "load_cuda_ptx_module", lambda **kwargs: _FakeCudaModule())

    run_fn, _meta = mod._build_intentir_launch_fn(kernel="flash_attention2d", artifact_dir=None, device="cuda")
    run_fn()

    assert len(calls) == 1
    args = calls[0]
    # K__aligned and Out__aligned are memref ABI slots that should alias the base tensor inputs/outputs.
    assert args[0] is args[1]
    assert args[2] is args[3]
