from intent_ir.ir import IntentFunction
from intent_ir.ir.repair import materialize_missing_op_output_tensors


def _shape_vals(intent: IntentFunction, name: str) -> list[object]:
    return [d.value for d in intent.tensors[name].shape]


def test_materialize_missing_tensor_for_elementwise_chain() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "add2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "alpha": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "mul", "inputs": ["y", "alpha"], "output": "y_scaled"},
                {"op": "add", "inputs": ["x", "y_scaled"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    actions = materialize_missing_op_output_tensors(intent)
    assert actions
    assert "y_scaled" in intent.tensors
    assert intent.tensors["y_scaled"].dtype == "f32"
    assert _shape_vals(intent, "y_scaled") == ["M", "N"]


def test_materialize_missing_tensor_for_reduce_shape() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "reduce_sum2d",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_sum", "inputs": ["x"], "output": "row_sum", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "identity", "inputs": ["row_sum"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    actions = materialize_missing_op_output_tensors(intent)
    assert actions
    assert "row_sum" in intent.tensors
    assert intent.tensors["row_sum"].dtype == "f32"
    assert _shape_vals(intent, "row_sum") == ["M"]


def test_materialize_missing_tensor_for_cast_dtype() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "cast2d",
            "tensors": {
                "x": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                {"op": "identity", "inputs": ["x_f32"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    actions = materialize_missing_op_output_tensors(intent)
    assert actions
    assert "x_f32" in intent.tensors
    assert intent.tensors["x_f32"].dtype == "f32"
    assert _shape_vals(intent, "x_f32") == ["M", "N"]


def test_materialize_missing_tensor_for_matmul_shape_and_dtype() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "mm2d",
            "tensors": {
                "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
                "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
                "C": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "matmul",
                    "inputs": ["A", "B"],
                    "output": "acc_main",
                    "attrs": {"transpose_a": False, "transpose_b": False, "accumulator_dtype": "f32"},
                },
                {"op": "cast", "inputs": ["acc_main"], "output": "C", "attrs": {"to": "f32"}},
            ],
            "outputs": ["C"],
        }
    )
    actions = materialize_missing_op_output_tensors(intent)
    assert actions
    assert "acc_main" in intent.tensors
    assert intent.tensors["acc_main"].dtype == "f32"
    assert _shape_vals(intent, "acc_main") == ["M", "N"]


def test_materialize_missing_tensor_for_conv2d_shape_uses_c_out() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "conv2d_nchw",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN_TOTAL", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_IN", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv2d",
                    "inputs": ["input", "weight"],
                    "output": "conv_out",
                    "attrs": {"stride": ["SH", "SW"], "padding": ["PH", "PW"], "dilation": ["DH", "DW"], "groups": "GROUPS"},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["bias"],
                    "output": "bias_bcast",
                    "attrs": {"out_shape": ["N", "C_OUT", "OH", "OW"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["conv_out", "bias_bcast"], "output": "output"},
            ],
            "outputs": ["output"],
        }
    )
    actions = materialize_missing_op_output_tensors(intent)
    assert actions
    assert "conv_out" in intent.tensors
    assert "bias_bcast" in intent.tensors
    assert _shape_vals(intent, "conv_out") == ["N", "C_OUT", "OH", "OW"]
    assert _shape_vals(intent, "bias_bcast") == ["N", "C_OUT", "OH", "OW"]
