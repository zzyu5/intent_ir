from __future__ import annotations

import shutil

import numpy as np
import pytest

try:
    import torch
except Exception:
    torch = None

from intent_ir.ir import IntentFunction


def _cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_matmul_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "mm_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "matmul", "inputs": ["A", "B"], "output": "C", "attrs": {}}],
            "outputs": ["C"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 8, "tile_n": 16, "tile_k": 16, "parallel_axes": ["M", "N"]},
        }
    )

    M, N, K = 32, 48, 16
    bindings = {"M": M, "N": N, "K": K}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    ref = A @ B

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A, "B": B},
        output_names=lowered.output_names,
    )
    got = out["C"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_fused_elementwise_broadcast_relu_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "ew_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "T": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "add", "inputs": ["A", "B"], "output": "T", "attrs": {}},
                {"op": "relu", "inputs": ["T"], "output": "Out", "attrs": {}},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M", "N"]},
        }
    )

    M, N = 8, 16
    bindings = {"M": M, "N": N}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=np.float32)
    B = rng.standard_normal((N,), dtype=np.float32)
    ref = np.maximum(A + B[None, :], 0.0).astype(np.float32)

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A, "B": B},
        output_names=lowered.output_names,
    )
    got = out["Out"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_transpose2d_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "transpose_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["N", "M"], "layout": "row_major"},
            },
            "ops": [{"op": "transpose", "inputs": ["A"], "output": "B", "attrs": {"perm": [1, 0]}}],
            "outputs": ["B"],
            "parallel_axes": ["M", "N"],
            "schedule": {"tile_m": 16, "tile_n": 16, "parallel_axes": ["M", "N"]},
        }
    )

    M, N = 16, 8
    bindings = {"M": M, "N": N}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=np.float32)
    ref = A.T

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A},
        output_names=lowered.output_names,
    )
    got = out["B"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_reduce_sum_row_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "row_sum_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "S": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_sum", "inputs": ["A"], "output": "S", "attrs": {"dims": [1]}}],
            "outputs": ["S"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M"]},
        }
    )

    M, N = 8, 32
    bindings = {"M": M, "N": N}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=np.float32)
    ref = A.sum(axis=1).astype(np.float32)

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A},
        output_names=lowered.output_names,
    )
    got = out["S"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_reduce_max_row_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "row_max_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "R": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [{"op": "reduce_max", "inputs": ["A"], "output": "R", "attrs": {"dims": [1]}}],
            "outputs": ["R"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M"]},
        }
    )

    M, N = 8, 32
    bindings = {"M": M, "N": N}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=np.float32)
    ref = A.max(axis=1).astype(np.float32)

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A},
        output_names=lowered.output_names,
    )
    got = out["R"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_any_dim_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "any_dim_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "T": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "Out": {"dtype": "i1", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "z", "attrs": {"value": 0.0}},
                {"op": "ne", "inputs": ["A", "z"], "output": "T", "attrs": {}},
                {"op": "reduce_any", "inputs": ["T"], "output": "Out", "attrs": {"dims": [1]}},
            ],
            "outputs": ["Out"],
            "parallel_axes": ["M"],
            "schedule": {"tile_n": 256, "parallel_axes": ["M"]},
        }
    )

    M, N = 8, 8
    bindings = {"M": M, "N": N}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=(M, N), dtype=np.int32).astype(np.float32)
    ref = np.any(A != 0.0, axis=1)

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A},
        output_names=lowered.output_names,
    )
    got = out["Out"]
    assert got.shape == ref.shape
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, ref)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available (torch extension build)")
def test_cuda_backend_gather2d_matches_numpy():
    from backends.cuda.codegen.intentir_to_cuda import lower_intent_to_cuda_kernel
    from backends.cuda.runtime import run_cuda_kernel

    intent = IntentFunction.from_json_dict(
        {
            "name": "gather2d_cuda_smoke",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "row": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "col": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            },
            "ops": [{"op": "gather", "inputs": ["A", "row", "col"], "output": "Out", "attrs": {"other_value": -1.0}}],
            "outputs": ["Out"],
            "parallel_axes": ["L"],
            "schedule": {"tile_n": 256, "parallel_axes": ["L"]},
        }
    )

    M, N, L = 7, 9, 32
    bindings = {"M": M, "N": N, "L": L}
    lowered = lower_intent_to_cuda_kernel(intent, shape_bindings=bindings)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, N), dtype=np.float32)
    # Mix in-bounds and out-of-bounds indices.
    row = rng.integers(-2, M + 2, size=(L,), dtype=np.int32)
    col = rng.integers(-2, N + 2, size=(L,), dtype=np.int32)
    ref = np.full((L,), -1.0, dtype=np.float32)
    for i in range(L):
        r = int(row[i])
        c = int(col[i])
        if 0 <= r < M and 0 <= c < N:
            ref[i] = A[r, c]

    out = run_cuda_kernel(
        kernel_name=lowered.kernel_name,
        cuda_src=lowered.cuda_src,
        io_spec=lowered.io_spec,
        launch=lowered.launch,
        bindings=lowered.bindings,
        inputs_np={"A": A, "row": row, "col": col},
        output_names=lowered.output_names,
    )
    got = out["Out"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)
