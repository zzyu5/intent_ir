from intent_ir.ir import IntentFunction, canonicalize_for_consistency


def test_canonicalize_relu_max_zero_bcast():
    intent = IntentFunction.from_json_dict(
        {
            "name": "relu_like",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "zero": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "zero2d": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["zero"],
                    "output": "zero2d",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": []},
                },
                {"op": "max", "inputs": ["inp", "zero2d"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    canon = canonicalize_for_consistency(intent)
    ops = [o.op for o in canon.ops]
    assert ops == ["relu"]
    assert canon.ops[0].inputs == ["inp"]


def test_canonicalize_mask_mul_to_where():
    intent = IntentFunction.from_json_dict(
        {
            "name": "masked_mul",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "mask_f": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "cast", "inputs": ["mask"], "output": "mask_f", "attrs": {"to": "f32"}},
                {"op": "mul", "inputs": ["x", "mask_f"], "output": "y"},
            ],
            "outputs": ["y"],
        }
    )
    canon = canonicalize_for_consistency(intent)
    assert [o.op for o in canon.ops] == ["const", "where"]
    where = canon.ops[1]
    assert where.op == "where"
    assert where.inputs[0] == "mask"
    assert where.inputs[1] == "x"


def test_canonicalize_folds_trailing_broadcast():
    intent = IntentFunction.from_json_dict(
        {
            "name": "add_bias",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                "bias2d": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["bias"],
                    "output": "bias2d",
                    "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["inp", "bias2d"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    canon = canonicalize_for_consistency(intent)
    assert [o.op for o in canon.ops] == ["add"]
    assert canon.ops[0].inputs == ["inp", "bias"]


def test_canonicalize_expands_softmax_op():
    intent = IntentFunction.from_json_dict(
        {
            "name": "softmax_macro",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "softmax", "inputs": ["x"], "output": "y", "attrs": {"axis": 1}},
            ],
            "outputs": ["y"],
        }
    )
    canon = canonicalize_for_consistency(intent)
    assert [o.op for o in canon.ops] == ["reduce_max", "sub", "exp", "reduce_sum", "div"]
    assert canon.ops[-1].op == "div"
    assert canon.ops[-1].output == "y"


def test_canonicalize_reduce_max_ne_zero_to_reduce_any():
    intent = IntentFunction.from_json_dict(
        {
            "name": "any_like",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mx": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "z": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "out": {"dtype": "bool", "shape": ["M"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_max", "inputs": ["x"], "output": "mx", "attrs": {"dims": [1], "keepdims": False}},
                {"op": "const", "inputs": [], "output": "z", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "ne", "inputs": ["mx", "z"], "output": "out"},
            ],
            "outputs": ["out"],
        }
    )
    canon = canonicalize_for_consistency(intent)
    assert [o.op for o in canon.ops] == ["const", "ne", "reduce_any"]
    assert canon.ops[-1].op == "reduce_any"
    assert canon.ops[-1].output == "out"
