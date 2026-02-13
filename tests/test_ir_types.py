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


def test_op_meta_roundtrip_and_validation():
    src = _minimal_intent_json()
    src["ops"][0]["meta"] = {
        "provider": "flaggems",
        "source_op": "add",
        "capability_state": "dual_pass",
    }
    intent = IntentFunction.from_json_dict(src)
    out = intent.to_json_dict()
    assert out["ops"][0]["meta"]["provider"] == "flaggems"
    assert out["ops"][0]["meta"]["source_op"] == "add"


def test_function_meta_flaggems_requires_source_and_state():
    bad = _minimal_intent_json()
    bad["meta"] = {"provider": "flaggems"}
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_new_structure_ops_validate_success() -> None:
    src = {
        "name": "structure_ok",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M2", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "concat", "inputs": ["X", "Y"], "output": "Out", "attrs": {"axis": 0}},
            {"op": "tile", "inputs": ["X"], "output": "T", "attrs": {"repeats": [2, 1]}},
            {"op": "repeat_interleave", "inputs": ["X"], "output": "R", "attrs": {"repeats": 2, "axis": 1}},
            {"op": "pad", "inputs": ["X"], "output": "P", "attrs": {"pad_width": {"pairs": [[1, 0], [0, 1]]}}},
            {"op": "sort", "inputs": ["X"], "output": "S", "attrs": {"axis": 1, "descending": True, "stable": True}},
            {"op": "topk", "inputs": ["X"], "output": "K", "attrs": {"k": 1, "axis": 1}},
            {"op": "unique", "inputs": ["K"], "output": "U", "attrs": {"sorted": False}},
            {"op": "nonzero", "inputs": ["X"], "output": "NZ"},
        ],
        "outputs": ["Out"],
    }
    intent = IntentFunction.from_json_dict(src)
    assert intent.name == "structure_ok"


def test_topk_requires_non_negative_k() -> None:
    bad = {
        "name": "topk_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
        },
        "ops": [{"op": "topk", "inputs": ["X"], "output": "Out", "attrs": {"k": -1, "axis": 1}}],
        "outputs": ["Out"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_pad_requires_valid_pad_width_pairs() -> None:
    bad = {
        "name": "pad_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M2", "N2"], "layout": "row_major"},
        },
        "ops": [{"op": "pad", "inputs": ["X"], "output": "Out", "attrs": {"pad_width": {"pairs": [[1], [0, 1]]}}}],
        "outputs": ["Out"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)
