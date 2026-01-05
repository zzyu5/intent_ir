import pytest

from intent_ir.ir import (
    IntentFunction,
    IntentIRValidationError,
)


def _minimal_intent_json():
    return {
        "name": "gemm_bias_relu",
        "tensors": {
            "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
            "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
            "bias": {"dtype": "f16", "shape": ["N"], "layout": "row_major"},
            "C": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "matmul", "inputs": ["A", "B"], "output": "Y", "attrs": {"accum_dtype": "f32"}},
            {
                "op": "broadcast_in_dim",
                "inputs": ["bias"],
                "output": "bias2d",
                "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
            },
            {"op": "add", "inputs": ["Y", "bias2d"], "output": "Z"},
            {"op": "relu", "inputs": ["Z"], "output": "C"},
        ],
        "outputs": ["C"],
        "parallel_axes": ["M", "N"],
        "axis_roles": {"M": "spatial", "N": "spatial", "K": "reduction"},
        "schedule": {
            "tile_m": "BLOCK_M",
            "tile_n": "BLOCK_N",
            "tile_k": "BLOCK_K",
            "vec_width": "VEC",
            "axis_bindings": {"tile_m": "M", "tile_n": "N", "tile_k": "K"},
            "vec_axis": "N",
            "parallel_axes": ["M", "N"],
            "memory_hint": {"residency": {"A": "local", "B": "local", "C": "local"}},
        },
    }


def test_roundtrip_minimal_gemm():
    src = _minimal_intent_json()
    intent = IntentFunction.from_json_dict(src)
    out = intent.to_json_dict()
    # Key structural fields should survive round-trip
    assert out["tensors"] == src["tensors"]
    assert out["outputs"] == src["outputs"]
    assert out["parallel_axes"] == src["parallel_axes"]
    assert out["axis_roles"] == src["axis_roles"]
    assert out["schedule"]["axis_bindings"] == src["schedule"]["axis_bindings"]


def test_invalid_dtype_rejected():
    bad = _minimal_intent_json()
    bad["tensors"]["A"]["dtype"] = "f8"
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_op_input_undefined_rejected():
    bad = _minimal_intent_json()
    # Undefined input with a close match and a common broadcast-intent suffix.
    bad["ops"][2]["inputs"] = ["Y", "bias_2d"]
    with pytest.raises(IntentIRValidationError) as exc:
        IntentFunction.from_json_dict(bad)
    msg = str(exc.value)
    assert "references undefined tensor" in msg
    assert "Did you mean" in msg
    assert "Available tensors at this point" in msg
    assert "broadcast_in_dim" in msg


def test_output_not_in_tensors_rejected():
    bad = _minimal_intent_json()
    bad["outputs"] = ["Y"]
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_schedule_field_types():
    bad = _minimal_intent_json()
    bad["schedule"]["tile_m"] = [16]
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_layout_blocked_params():
    bad = _minimal_intent_json()
    bad["tensors"]["A"]["layout"] = {"kind": "blocked"}
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)
