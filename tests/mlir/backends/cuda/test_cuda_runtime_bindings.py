from __future__ import annotations

from frontends.cuda.runtime import _augment_scalar_bindings_from_io_spec


def test_runtime_binding_augment_fills_concat_aliases() -> None:
    io_spec = {
        "arg_names": ["A", "B", "out", "M0", "N0", "M1", "N1", "M_OUT", "N_OUT"],
        "tensors": {
            "A": {"dtype": "f32", "shape": ["M", "N"]},
            "B": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"]},
        },
        "scalars": {
            "M0": "i32",
            "N0": "i32",
            "M1": "i32",
            "N1": "i32",
            "M_OUT": "i32",
            "N_OUT": "i32",
        },
        "outputs": ["out"],
    }
    merged = _augment_scalar_bindings_from_io_spec(
        bindings={"M": 4, "N": 8, "M_OUT": 4, "N_OUT": 16},
        io_spec=io_spec,
    )
    assert int(merged.get("M0")) == 4
    assert int(merged.get("M1")) == 4
    assert int(merged.get("N0")) == 8
    assert int(merged.get("N1")) == 8
    assert int(merged.get("M_OUT")) == 4
    assert int(merged.get("N_OUT")) == 16


def test_runtime_binding_augment_overwrites_none_placeholders() -> None:
    io_spec = {
        "arg_names": ["A", "B", "out", "M0", "N0", "M1", "N1", "M_OUT", "N_OUT", "T"],
        "tensors": {
            "A": {"dtype": "f32", "shape": ["M", "N"]},
            "B": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"]},
        },
        "scalars": {
            "M0": "i32",
            "N0": "i32",
            "M1": "i32",
            "N1": "i32",
            "M_OUT": "i32",
            "N_OUT": "i32",
            "T": "i32",
        },
        "outputs": ["out"],
    }
    merged = _augment_scalar_bindings_from_io_spec(
        bindings={"M": 4, "N": 8, "M_OUT": 4, "N_OUT": 16, "M0": None, "N0": None, "M1": None, "N1": None, "T": None},
        io_spec=io_spec,
    )
    assert int(merged.get("M0")) == 4
    assert int(merged.get("N0")) == 8
    assert int(merged.get("M1")) == 4
    assert int(merged.get("N1")) == 8
    assert int(merged.get("M_OUT")) == 4
    assert int(merged.get("N_OUT")) == 16
    assert int(merged.get("T")) == 64


def test_runtime_binding_augment_derives_t() -> None:
    io_spec = {
        "arg_names": ["x", "out", "M_OUT", "N_OUT", "T"],
        "tensors": {
            "x": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "f32", "shape": ["M_OUT", "N_OUT"]},
        },
        "scalars": {"M_OUT": "i32", "N_OUT": "i32", "T": "i32"},
        "outputs": ["out"],
    }
    merged = _augment_scalar_bindings_from_io_spec(
        bindings={"M": 4, "N": 8, "M_OUT": 6, "N_OUT": 10},
        io_spec=io_spec,
    )
    assert int(merged.get("T")) == 60


def test_runtime_binding_augment_prefers_symbolic_out_expr_over_plain_mn() -> None:
    io_spec = {
        "arg_names": ["x", "out", "M", "N", "M_OUT", "N_OUT"],
        "tensors": {
            "x": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "f32", "shape": ["M + 1", "N + 3"]},
        },
        "scalars": {"M": "i32", "N": "i32", "M_OUT": "i32", "N_OUT": "i32"},
        "outputs": ["out"],
    }
    merged = _augment_scalar_bindings_from_io_spec(
        bindings={"M": 4, "N": 64, "M + 1": 5, "N + 3": 67},
        io_spec=io_spec,
    )
    assert int(merged.get("M_OUT")) == 5
    assert int(merged.get("N_OUT")) == 67
